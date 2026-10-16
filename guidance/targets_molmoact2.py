"""With positive dt and v_guided = v - w*g, the AGF target is the negative adjoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from guidance.sampling_molmoact2 import molmoact2_prefix_context, molmoact2_qpilots_images, critic_q


@dataclass
class ActionSpec:
    horizon: int
    action_dim: int


def _selected_action_region(x: torch.Tensor, spec: ActionSpec) -> torch.Tensor:
    return x[:, : spec.horizon, : spec.action_dim]


def _linear_particle_count(t: float, count_min: int, count_max: int) -> int:

    raw = count_min + (count_max - count_min) * t
    return max(1, round(raw))


def _molmoact2_velocity_vjp(
    action_expert, action_context, modulation, x_t: torch.Tensor, a_tilde: torch.Tensor
) -> torch.Tensor:

    x_req = x_t.detach().requires_grad_(True)
    v_psi = action_expert.forward_with_context(
        x_req, modulation.conditioning, context=action_context, modulation=modulation
    )
    vjp_v = torch.autograd.grad(v_psi, x_req, grad_outputs=a_tilde.detach())[0]
    return vjp_v.detach()


def compute_molmoact2_agf_adjoint_targets(
    policy: Any,
    critic: Any,
    guidance_model: Any,
    batch: dict[str, Any],
    args,
    guidance_weight: float = 1.0,
    use_guidance_rollout: bool = True,
) -> dict[str, torch.Tensor]:

    spec = ActionSpec(args.critic_horizon, args.action_dim)
    device = next(policy.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    ctx = molmoact2_prefix_context(policy, batch, generator)
    backbone = ctx["backbone"]
    action_expert = ctx["action_expert"]
    action_context = ctx["action_context"]
    action_dim_is_pad = ctx["action_dim_is_pad"]
    mask_enabled = policy.config.mask_action_dim_padding

    num_steps = int(
        args.agf_num_steps
        or getattr(policy.config, "num_inference_steps", None)
        or backbone.config.flow_matching_num_steps
    )
    dt = 1.0 / num_steps
    bsize = ctx["batch_size"]

    flow_timesteps = [
        torch.full((bsize,), idx / num_steps, device=device, dtype=torch.float32) for idx in range(num_steps)
    ]
    modulation_cache = action_expert.get_or_prepare_modulation_cache(
        flow_timesteps, cache_key=(num_steps, bsize, device, ctx["trajectory"].dtype)
    )

    state = molmoact2_qpilots_images(batch, image_size=args.critic_image_size)

    xs: list[torch.Tensor] = []
    x = ctx["trajectory"]
    with torch.no_grad():
        xs.append(x.detach().clone())
        for step in range(num_steps):
            modulation = modulation_cache[step]
            v_psi = action_expert.forward_with_context(
                x, modulation.conditioning, context=action_context, modulation=modulation
            )
            v_psi = backbone._mask_action_dim_tensor(v_psi, action_dim_is_pad=action_dim_is_pad, enabled=mask_enabled)

            drift = v_psi
            if use_guidance_rollout:
                if guidance_model is None:
                    raise ValueError("guidance_model is required when use_guidance_rollout=True")
                t_tensor = flow_timesteps[step]
                g = guidance_model(state[0], state[1], x, t_tensor).detach()
                g_region = _selected_action_region(g, spec)
                v_region = _selected_action_region(v_psi, spec)
                g_norm = g_region.flatten(start_dim=1).norm(dim=-1).clamp_min(1e-8)
                v_norm = v_region.flatten(start_dim=1).norm(dim=-1)
                clip_scale = (v_norm / g_norm).clamp(max=1.0)
                g = g * clip_scale.view(-1, 1, 1)

                drift = drift - float(guidance_weight) * g

            x = x + dt * drift
            x = backbone._mask_action_dim_tensor(x, action_dim_is_pad=action_dim_is_pad, enabled=mask_enabled)
            xs.append(x.detach().clone())

    x_clean = xs[-1].detach()

    a_req = _selected_action_region(x_clean, spec).detach().requires_grad_(True)
    q = critic_q(critic, state, a_req, spec.horizon).mean()
    seed_selected = torch.autograd.grad(q, a_req)[0].detach()

    lambda_t = torch.zeros_like(x_clean)
    lambda_t[:, : spec.horizon, : spec.action_dim] = seed_selected

    particle_count_fn = None
    if args.particle_count_schedule == "linear":
        # Reverse endpoints: MolmoAct2 starts at noise, where particle count is highest.
        particle_count_fn = lambda t: _linear_particle_count(  # noqa: E731
            t, args.particle_count_max, args.particle_count_min
        )

    lambdas: list[torch.Tensor | None] = [None] * (num_steps + 1)
    lambdas[num_steps] = lambda_t.detach().clone()

    for idx in range(num_steps - 1, -1, -1):
        modulation = modulation_cache[idx]
        x_k = xs[idx].detach()
        if args.use_particle_adjoint:
            t_eval = idx / num_steps
            step_count = particle_count_fn(t_eval) if particle_count_fn is not None else args.num_perturbation
            vjp_per = []
            for _ in range(step_count):
                x_k_pert = x_k + args.sharpness_sigma * torch.randn_like(x_k)
                vjp_per.append(_molmoact2_velocity_vjp(action_expert, action_context, modulation, x_k_pert, lambda_t))
            vjp_v = torch.stack(vjp_per, dim=0).mean(dim=0)
        else:
            vjp_v = _molmoact2_velocity_vjp(action_expert, action_context, modulation, x_k, lambda_t)
        lambda_t = lambda_t + dt * vjp_v
        lambdas[idx] = lambda_t.detach().clone()

    x_samples = torch.cat([xs[i].detach().cpu() for i in range(num_steps + 1)], dim=0)
    raw_adjoints = torch.cat([lambdas[i].detach().cpu() for i in range(num_steps + 1)], dim=0)  # type: ignore[union-attr]
    t_samples = torch.cat(
        [torch.full((bsize,), min(i / num_steps, 1.0), dtype=torch.float32) for i in range(num_steps + 1)], dim=0
    )

    adjoint_target = -raw_adjoints

    n_grid = num_steps + 1
    image_samples = state[0].detach().cpu().repeat(n_grid, 1, 1, 1)
    image2_samples = state[1].detach().cpu().repeat(n_grid, 1, 1, 1)
    q_samples = q.detach().cpu().repeat(n_grid)

    adjoint_norm = _selected_action_region(raw_adjoints, spec).norm(dim=(1, 2))
    seed_norm = seed_selected.norm(dim=(1, 2))

    return {
        "image": image_samples,
        "image2": image2_samples,
        "x_t": x_samples,
        "t": t_samples,
        "adjoint_target": adjoint_target,
        "seed_norm": seed_norm.detach().cpu(),
        "adjoint_norm": adjoint_norm.detach().cpu(),
        "q_value": q_samples,
    }
