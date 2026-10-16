from __future__ import annotations

import copy
from typing import Any, Callable

import torch
import torch.nn.functional as F

from guidance.targets_pi05_robocasa import (
    ActionSpec,
    _freeze_for_target_extraction,
    _selected_action_region,
    _velocity_vjp,
)


def _resize_critic_image(x: torch.Tensor, image_size: int) -> torch.Tensor:

    x = x.float()
    if x.max() > 2.0:
        x = x / 255.0
    if tuple(x.shape[-2:]) != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x


def _critic_images(batch: dict[str, Any], image_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image = _resize_critic_image(batch["observation.images.image"], image_size).to(device)
    image2 = _resize_critic_image(batch["observation.images.image2"], image_size).to(device)
    return image, image2


def _prepare_attention_masks_4d_compat(m: Any, att_2d_masks: torch.Tensor) -> torch.Tensor:
    from lerobot.policies.pi05 import modeling_pi05

    public_fn = getattr(modeling_pi05, "prepare_attention_masks_4d", None)
    if public_fn is not None:
        return public_fn(att_2d_masks)

    private_fn = getattr(m, "_prepare_attention_masks_4d", None)
    if private_fn is not None:
        return private_fn(att_2d_masks)

    raise AttributeError(
        "Could not find a pi0.5 attention-mask 4D conversion function. "
        "Neither modeling_pi05.prepare_attention_masks_4d nor "
        "PI05Pytorch._prepare_attention_masks_4d exists. "
        f"Loaded modeling_pi05 from: {getattr(modeling_pi05, '__file__', None)}"
    )


def compute_agf_adjoint_targets_libero(
    policy: Any,
    critic: Any,
    guidance_model: Any,
    batch: dict[str, Any],
    score_action_fn: Callable[[Any, tuple[torch.Tensor, ...], torch.Tensor, int], torch.Tensor],
    make_att_2d_masks_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    spec: ActionSpec,
    critic_image_size: int = 64,
    num_steps: int | None = None,
    guidance_weight: float = 1.0,
    use_guidance_rollout: bool = True,
) -> dict[str, torch.Tensor]:

    _freeze_for_target_extraction(policy, "policy")
    _freeze_for_target_extraction(critic, "critic")
    if use_guidance_rollout:
        _freeze_for_target_extraction(guidance_model, "guidance_model")

    m = policy.model
    device = next(policy.parameters()).device
    num_steps = int(num_steps or getattr(m.config, "num_inference_steps"))
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]
    bsize = int(tokens.shape[0])

    with torch.no_grad():
        pre_emb, pre_pad, pre_att = m.embed_prefix(images, img_masks, tokens, masks)

        crit_image, crit_image2 = _critic_images(batch, critic_image_size, device)

        att_2d = make_att_2d_masks_fn(pre_pad, pre_att)
        att_2d = _prepare_attention_masks_4d_compat(m, att_2d)
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

    with torch.no_grad():
        for step in range(num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.full((bsize,), t, device=device)
            v_psi = m.denoise_step(pre_pad, copy.deepcopy(past_kv), x, t_tensor)

            drift = v_psi
            if use_guidance_rollout:
                if guidance_model is None:
                    raise ValueError("guidance_model is required when use_guidance_rollout=True")
                g = guidance_model(crit_image, crit_image2, x, t_tensor).detach()

                g_region = _selected_action_region(g, spec)
                v_region = _selected_action_region(v_psi, spec)
                g_norm = g_region.flatten(start_dim=1).norm(dim=-1).clamp_min(1e-8)
                v_norm = v_region.flatten(start_dim=1).norm(dim=-1)
                clip_scale = (v_norm / g_norm).clamp(max=1.0)
                g = g * clip_scale.view(-1, 1, 1)

                drift = drift - float(guidance_weight) * g

            x = x + dt * drift
            xs.append(x.detach().clone())

    x_clean = xs[-1].detach()

    a_req = _selected_action_region(x_clean, spec).detach().requires_grad_(True)
    crit_obs = (crit_image.detach(), crit_image2.detach())
    q = score_action_fn(critic, crit_obs, a_req, spec.horizon)
    seed_selected = torch.autograd.grad(q.sum(), a_req)[0].detach()

    lambda_t = torch.zeros_like(x_clean)
    lambda_t[:, : spec.horizon, : spec.action_dim] = seed_selected

    lambdas: list[torch.Tensor | None] = [None] * (num_steps + 1)
    lambdas[num_steps] = lambda_t.detach().clone()

    for idx in range(num_steps - 1, -1, -1):
        t = 1.0 + idx * dt
        x_k = xs[idx].detach()
        vjp_v = _velocity_vjp(
            policy_model=m, pre_pad=pre_pad, past_kv=past_kv, x_t=x_k, t=t, bsize=bsize, device=device, a_tilde=lambda_t
        )
        lambda_t = lambda_t + dt * vjp_v
        lambdas[idx] = lambda_t.detach().clone()

    x_samples = torch.cat([xs[i].detach().cpu() for i in range(num_steps + 1)], dim=0)
    raw_adjoints = torch.cat([lambdas[i].detach().cpu() for i in range(num_steps + 1)], dim=0)  # type: ignore[union-attr]
    t_samples = torch.cat(
        [torch.full((bsize,), max(1.0 + i * dt, 0.0), dtype=torch.float32) for i in range(num_steps + 1)], dim=0
    )

    n_grid = num_steps + 1
    image_samples = crit_image.detach().cpu().repeat(n_grid, 1, 1, 1)
    image2_samples = crit_image2.detach().cpu().repeat(n_grid, 1, 1, 1)
    q_samples = q.detach().cpu().repeat(n_grid)

    adjoint_norm = _selected_action_region(raw_adjoints, spec).norm(dim=(1, 2))
    seed_norm = seed_selected.norm(dim=(1, 2))

    return {
        "image": image_samples,
        "image2": image2_samples,
        "x_t": x_samples,
        "t": t_samples,
        "adjoint_target": raw_adjoints,
        "seed_norm": seed_norm.detach().cpu(),
        "adjoint_norm": adjoint_norm.detach().cpu(),
        "q_value": q_samples,
    }
