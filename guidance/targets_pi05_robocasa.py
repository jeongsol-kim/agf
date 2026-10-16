from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from tasks.datasets import RoboCasaCriticDataTaskDataset


@dataclass(frozen=True)
class ActionSpec:
    horizon: int
    action_dim: int


@dataclass(frozen=True)
class AGFDiscreteOptions:
    num_steps: int | None = None

    # Subsampling approximates only the backward pass; the forward grid is unchanged.
    time_subsample: int = 1

    randomize_checkpoint_phase: bool = False

    critic_image_size: int = 64


def _selected_action_region(tensor: torch.Tensor, spec: ActionSpec) -> torch.Tensor:
    return tensor[:, : spec.horizon, : spec.action_dim]


def _linear_particle_sigma(t: float, sigma_min: float, sigma_max: float) -> float:

    return float(sigma_min + (sigma_max - sigma_min) * t)


def _linear_particle_count(t: float, count_min: int, count_max: int) -> int:

    raw = count_min + (count_max - count_min) * t
    return max(1, round(raw))


def _freeze_for_target_extraction(module: Any, name: str) -> None:
    if module is None or not hasattr(module, "parameters"):
        raise ValueError(f"{name} must be a torch module-like object with parameters()")
    for param in module.parameters():
        param.requires_grad_(False)
        param.grad = None


def _resize_critic_image(x: torch.Tensor, image_size: int) -> torch.Tensor:

    x = x.float()
    if x.max() > 2.0:
        x = x / 255.0
    if tuple(x.shape[-2:]) != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x


def _critic_images(
    batch: dict[str, Any], image_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = RoboCasaCriticDataTaskDataset.CAMERA_KEYS
    return tuple(_resize_critic_image(batch[k], image_size).to(device) for k in keys)


def _velocity_vjp(
    policy_model: Any,
    pre_pad: torch.Tensor,
    past_kv: Any,
    x_t: torch.Tensor,
    t: float,
    bsize: int,
    device: torch.device,
    a_tilde: torch.Tensor,
) -> torch.Tensor:

    x_req = x_t.detach().requires_grad_(True)
    t_tensor = torch.full((bsize,), t, device=device)
    v_psi = policy_model.denoise_step(pre_pad, copy.deepcopy(past_kv), x_req, t_tensor)
    vjp_v = torch.autograd.grad(v_psi, x_req, grad_outputs=a_tilde.detach())[0]
    return vjp_v.detach()


def compute_agf_adjoint_targets(
    policy: Any,
    critic: Any,
    guidance_model: Any,
    batch: dict[str, Any],
    score_action_fn: Callable[[Any, tuple[torch.Tensor, ...], torch.Tensor, int], torch.Tensor],
    make_att_2d_masks_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    spec: ActionSpec,
    options: AGFDiscreteOptions,
    guidance_weight: float = 1.0,
    use_guidance_rollout: bool = True,
    use_sharpness_aware: bool = False,
    num_perturbation: int = 8,
    sharpness_sigma: float = 0.02,
    use_particle_adjoint: bool = False,
    particle_sigma_fn: Callable[[float], float] | None = None,
    particle_count_fn: Callable[[float], int] | None = None,
) -> dict[str, torch.Tensor]:

    if options.time_subsample <= 0:
        raise ValueError(f"time_subsample must be positive, got {options.time_subsample}")

    _freeze_for_target_extraction(policy, "policy")
    _freeze_for_target_extraction(critic, "critic")
    if use_guidance_rollout:
        _freeze_for_target_extraction(guidance_model, "guidance_model")

    m = policy.model
    device = next(policy.parameters()).device
    num_steps = int(options.num_steps or getattr(m.config, "num_inference_steps"))
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]
    bsize = int(tokens.shape[0])

    with torch.no_grad():
        pre_emb, pre_pad, pre_att = m.embed_prefix(images, img_masks, tokens, masks)

        crit_image, crit_image2, crit_image3 = _critic_images(batch, options.critic_image_size, device)
        crit_state = batch["observation.state"].to(device).float()

        att_2d = make_att_2d_masks_fn(pre_pad, pre_att)
        att_2d = m._prepare_attention_masks_4d(att_2d)
        pos_ids = torch.cumsum(pre_pad, dim=1) - 1

        try:
            m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        except AttributeError:
            pass

        _, past_kv = m.paligemma_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[pre_emb, None],
            use_cache=True,
        )

        x = m.sample_noise((bsize, m.config.chunk_size, m.config.max_action_dim), device)

    dt = -1.0 / num_steps

    xs: list[torch.Tensor] = [x.detach().clone()]
    t_values: list[float] = [1.0]

    with torch.no_grad():
        for step in range(num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.full((bsize,), t, device=device)
            v_psi = m.denoise_step(pre_pad, copy.deepcopy(past_kv), x, t_tensor)

            drift = v_psi
            if use_guidance_rollout:
                if guidance_model is None:
                    raise ValueError("guidance_model is required when use_guidance_rollout=True")

                g = guidance_model(crit_image, crit_image2, crit_image3, crit_state, x, t_tensor).detach()

                g_region = _selected_action_region(g, spec)
                b_region = _selected_action_region(v_psi, spec)
                g_norm = g_region.flatten(start_dim=1).norm(dim=-1).clamp_min(1e-8)
                b_norm = b_region.flatten(start_dim=1).norm(dim=-1)
                clip_scale = (b_norm / g_norm).clamp(max=1.0)
                g = g * clip_scale.view(-1, 1, 1)
                drift = drift - float(guidance_weight) * g

            x = x + dt * drift

            t_next = max(1.0 + (step + 1) * dt, 0.0)
            xs.append(x.detach().clone())
            t_values.append(float(t_next))

    x_clean = xs[-1].detach()

    a_req = x_clean[:, : spec.horizon, : spec.action_dim].detach().requires_grad_(True)
    crit_obs = (crit_image.detach(), crit_image2.detach(), crit_image3.detach(), crit_state.detach())

    if use_sharpness_aware:
        q_per = []

        for _ in range(num_perturbation):
            per_a_req = a_req + sharpness_sigma * torch.randn_like(a_req)
            q_per.append(score_action_fn(critic, crit_obs, per_a_req, spec.horizon))
        q = torch.stack(q_per, dim=0).mean(dim=0)
    else:
        q = score_action_fn(critic, crit_obs, a_req, spec.horizon)

    seed_selected = torch.autograd.grad(q.sum(), a_req)[0].detach()

    lambda_t = torch.zeros_like(x_clean)
    lambda_t[:, : spec.horizon, : spec.action_dim] = seed_selected

    phase = 0
    if options.randomize_checkpoint_phase and options.time_subsample > 1:
        phase = int(torch.randint(0, options.time_subsample, (1,)).item())

    selected_indices = list(range(phase, num_steps + 1, options.time_subsample))
    if not selected_indices or selected_indices[0] != 0:
        selected_indices.insert(0, 0)
    if selected_indices[-1] != num_steps:
        selected_indices.append(num_steps)
    t_selected = [t_values[i] for i in selected_indices]

    lambdas: list[torch.Tensor | None] = [None] * (num_steps + 1)
    lambdas[-1] = lambda_t.detach().clone()

    checkpoints = sorted((i for i in selected_indices if i < num_steps), reverse=True)
    prev_idx = num_steps
    for idx in checkpoints:
        dt_eff = (prev_idx - idx) * dt
        t_eval = 1.0 + idx * dt
        x_k = xs[idx].detach()

        if use_particle_adjoint:
            step_sigma = float(particle_sigma_fn(t_eval)) if particle_sigma_fn is not None else sharpness_sigma
            step_count = int(particle_count_fn(t_eval)) if particle_count_fn is not None else num_perturbation
            vjp_per = []
            for _ in range(step_count):
                x_k_pert = x_k + step_sigma * torch.randn_like(x_k)
                vjp_per.append(
                    _velocity_vjp(
                        policy_model=m,
                        pre_pad=pre_pad,
                        past_kv=past_kv,
                        x_t=x_k_pert,
                        t=t_eval,
                        bsize=bsize,
                        device=device,
                        a_tilde=lambda_t,
                    )
                )
            vjp_drift = torch.stack(vjp_per, dim=0).mean(dim=0)
        else:
            vjp_drift = _velocity_vjp(
                policy_model=m,
                pre_pad=pre_pad,
                past_kv=past_kv,
                x_t=x_k,
                t=t_eval,
                bsize=bsize,
                device=device,
                a_tilde=lambda_t,
            )

        lambda_t = lambda_t + dt_eff * vjp_drift
        lambdas[idx] = lambda_t.detach().clone()
        prev_idx = idx

    x_selected = [xs[i].detach().cpu() for i in selected_indices]
    lambda_selected = [lambdas[i].detach().cpu() for i in selected_indices]  # type: ignore[union-attr]

    x_samples = torch.cat(x_selected, dim=0)
    raw_adjoints = torch.cat(lambda_selected, dim=0)

    t_samples = torch.cat([torch.full((bsize,), float(t), dtype=torch.float32) for t in t_selected], dim=0)

    n_sel = len(selected_indices)
    image_samples = crit_image.detach().cpu().repeat(n_sel, 1, 1, 1)
    image2_samples = crit_image2.detach().cpu().repeat(n_sel, 1, 1, 1)
    image3_samples = crit_image3.detach().cpu().repeat(n_sel, 1, 1, 1)
    state_samples = crit_state.detach().cpu().repeat(n_sel, 1)

    q_samples = q.detach().cpu().repeat(n_sel)

    adjoint_norm = _selected_action_region(raw_adjoints, spec).norm(dim=(1, 2))
    seed_norm = seed_selected.norm(dim=(1, 2))

    return {
        "image": image_samples,
        "image2": image2_samples,
        "image3": image3_samples,
        "state": state_samples,
        "x_t": x_samples,
        "t": t_samples,
        "adjoint_target": raw_adjoints,
        "seed_norm": seed_norm.detach().cpu(),
        "adjoint_norm": adjoint_norm.detach().cpu(),
        "q_value": q_samples,
    }
