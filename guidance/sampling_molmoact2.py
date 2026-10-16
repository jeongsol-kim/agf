"""MolmoAct2 flows from t=0 (noise) to t=1 (clean), with positive dt."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def molmoact2_qpilots_images(
    batch,
    image_size: int = 64,
    image_key: str = "observation.images.image",
    image2_key: str = "observation.images.wrist_image",
):

    def _resize(x):
        x = x.float()
        if x.max() > 2.0:
            x = x / 255.0
        if tuple(x.shape[-2:]) != (image_size, image_size):
            x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return x

    image = _resize(batch[image_key])
    image2 = _resize(batch[image2_key])
    return image, image2


def critic_q(critic, state, action_chunk, horizon: int):
    image, image2 = state
    action = action_chunk[:, :horizon]
    return critic.reward(image, image2, action)


def _tile_batch(batch, k: int):

    out = {}
    for key, val in batch.items():
        out[key] = val.repeat_interleave(k, dim=0) if torch.is_tensor(val) else val
    return out


def molmoact2_qgf_velocity(
    critic, state, x_t, v_t, t: float, horizon: int, action_dim: int, weight: float, max_norm: float | None
):
    """QGF detaches velocity and adds the Q gradient for the positive-dt convention."""
    if critic is None or weight == 0.0:
        return v_t, None

    x = x_t.detach().clone().requires_grad_(True)
    x1_hat = x + (1.0 - t) * v_t.detach()
    action = x1_hat[:, :, :action_dim]

    q = critic_q(critic, state, action, horizon).mean()
    grad = torch.autograd.grad(q, x)[0]

    g = torch.zeros_like(grad)
    g[:, :horizon, :action_dim] = grad[:, :horizon, :action_dim]

    if max_norm is not None:
        region = g[:, :horizon, :action_dim]
        norm = region.norm(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        scale = (max_norm / norm).clamp(max=1.0)
        g[:, :horizon, :action_dim] = region * scale

    applied = weight * g[:, :horizon, :action_dim]
    v_region = v_t[:, :horizon, :action_dim]
    info = {
        "q": float(q.detach().cpu()),
        "qgf_norm": float(g[:, :horizon, :action_dim].norm(dim=(1, 2)).mean().detach().cpu()),
        "raw_grad_norm": float(grad[:, :horizon, :action_dim].norm(dim=(1, 2)).mean().detach().cpu()),
        "applied_norm": float(applied.norm(dim=(1, 2)).mean().detach().cpu()),
        "v_t_norm": float(v_region.norm(dim=(1, 2)).mean().detach().cpu()),
    }

    return v_t + weight * g.detach(), info


def molmoact2_agf_velocity(guidance_model, state, x_t, t_tensor, v_t, horizon: int, action_dim: int, weight: float):
    """AGF subtracts guidance; its training target already accounts for positive dt."""
    image, image2 = state
    g = guidance_model(image, image2, x_t, t_tensor).detach()

    g_region = g[:, :horizon, :action_dim]
    v_region = v_t[:, :horizon, :action_dim]
    g_norm = g_region.flatten(1).norm(dim=-1).clamp_min(1e-8)
    v_norm = v_region.flatten(1).norm(dim=-1)
    g = g * (v_norm / g_norm).clamp(max=1.0).view(-1, 1, 1)

    info = {"agf_norm": float(g[:, :horizon, :action_dim].norm(dim=(1, 2)).mean().cpu())}
    return v_t - weight * g, info


def molmoact2_qdps_velocity(
    critic,
    state,
    action_expert,
    backbone,
    action_context,
    modulation,
    x_t,
    action_dim_is_pad,
    mask_enabled: bool,
    t: float,
    horizon: int,
    action_dim: int,
    weight: float,
    max_norm: float | None,
):
    """DPS differentiates through the denoiser; end-to-end MolmoAct2 gradient flow remains unverified."""
    if critic is None or weight == 0.0:
        with torch.no_grad():
            v_t = action_expert.forward_with_context(
                x_t, modulation.conditioning, context=action_context, modulation=modulation
            )
            v_t = backbone._mask_action_dim_tensor(v_t, action_dim_is_pad=action_dim_is_pad, enabled=mask_enabled)
        return v_t, None

    x = x_t.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        v_t = action_expert.forward_with_context(
            x, modulation.conditioning, context=action_context, modulation=modulation
        )
        v_t = backbone._mask_action_dim_tensor(v_t, action_dim_is_pad=action_dim_is_pad, enabled=mask_enabled)
        x1_hat = x + (1.0 - t) * v_t
        action = x1_hat[:, :, :action_dim]
        q = critic_q(critic, state, action, horizon).mean()
        grad = torch.autograd.grad(q, x)[0]

    g = torch.zeros_like(grad)
    g[:, :horizon, :action_dim] = grad[:, :horizon, :action_dim]

    if max_norm is not None:
        region = g[:, :horizon, :action_dim]
        norm = region.norm(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        scale = (max_norm / norm).clamp(max=1.0)
        g[:, :horizon, :action_dim] = region * scale

    info = {
        "q": float(q.detach().cpu()),
        "qdps_norm": float(g[:, :horizon, :action_dim].norm(dim=(1, 2)).mean().detach().cpu()),
    }
    return v_t.detach() + weight * g.detach(), info


def _patch_time_conditioning_dtype(action_expert):
    """Match time embeddings to the layer weight dtype, including LoRA-wrapped layers."""
    import types

    def _time_conditioning(self, timesteps):
        conditioning = self.time_embed[0](timesteps)
        first = self.time_embed[1]
        weight = getattr(first, "weight", None)
        if weight is None and hasattr(first, "base_layer"):
            weight = first.base_layer.weight
        if weight is not None:
            conditioning = conditioning.to(dtype=weight.dtype)
        for module in list(self.time_embed.children())[1:]:
            conditioning = module(conditioning)
        return conditioning

    action_expert._time_conditioning = types.MethodType(_time_conditioning, action_expert)


def molmoact2_prefix_context(policy, batch, generator):

    backbone = policy._backbone()
    action_expert = backbone._require_action_expert()
    model_inputs = policy._model_inputs(batch)

    with torch.no_grad():
        outputs = backbone(**model_inputs, use_cache=True, output_attentions=False, output_hidden_states=False)
    encoder_kv_states = backbone._extract_kv_states(outputs.past_key_values)
    encoder_attention_mask = backbone._get_encoder_attention_mask(
        model_inputs.get("input_ids"), model_inputs.get("attention_mask")
    )

    depth_gate, depth_mask = backbone._depth_gate_from_condition(
        input_ids=model_inputs.get("input_ids"),
        encoder_attention_mask=encoder_attention_mask,
        layer_kv_states=encoder_kv_states,
    )
    encoder_kv_states = backbone._apply_depth_gate_to_layer_kv_states(encoder_kv_states, depth_mask, depth_gate)

    action_dim_is_pad = batch.get("action_dim_is_pad")
    action_horizon = policy._generation_action_horizon()
    source_tensor = encoder_kv_states[0][0]
    batch_size = int(source_tensor.shape[0])
    device = source_tensor.device
    trajectory_dtype = action_expert.action_embed.weight.dtype

    trajectory = torch.randn(
        (batch_size, action_horizon, backbone.config.max_action_dim),
        device=device,
        dtype=trajectory_dtype,
        generator=generator,
    )
    trajectory = backbone._mask_action_dim_tensor(
        trajectory, action_dim_is_pad=action_dim_is_pad, enabled=policy.config.mask_action_dim_padding
    )

    action_context = action_expert.prepare_context(
        encoder_kv_states=encoder_kv_states,
        encoder_attention_mask=encoder_attention_mask,
        state_embeddings=None,
        batch_size=batch_size,
        seq_len=trajectory.shape[1],
        device=device,
        dtype=trajectory.dtype,
    )

    return {
        "backbone": backbone,
        "action_expert": action_expert,
        "action_context": action_context,
        "trajectory": trajectory,
        "action_dim_is_pad": action_dim_is_pad,
        "batch_size": batch_size,
        "device": device,
    }


def sample_action_chunk_molmoact2(policy, critic, guidance_model, batch, args, generator):

    backbone = policy._backbone()
    action_expert = backbone._require_action_expert()

    ctx = molmoact2_prefix_context(policy, batch, generator)
    trajectory = ctx["trajectory"]
    action_dim_is_pad = ctx["action_dim_is_pad"]
    batch_size = ctx["batch_size"]
    device = ctx["device"]
    mask_enabled = policy.config.mask_action_dim_padding

    num_steps = int(getattr(policy.config, "num_inference_steps", None) or backbone.config.flow_matching_num_steps)
    dt = 1.0 / num_steps

    action_dim = policy._output_action_dim(batch)
    state = molmoact2_qpilots_images(batch, image_size=args.critic_image_size)

    flow_timesteps = [
        torch.full((batch_size,), idx / num_steps, device=device, dtype=torch.float32) for idx in range(num_steps)
    ]
    modulation_cache = action_expert.get_or_prepare_modulation_cache(
        flow_timesteps, cache_key=(num_steps, batch_size, device, trajectory.dtype)
    )

    q_values, qgf_norms, raw_grad_norms, applied_norms, v_t_norms = [], [], [], [], []
    agf_norms = []
    qdps_norms = []

    for step in range(num_steps):
        t = step / num_steps
        modulation = modulation_cache[step]

        if args.mode == "qdps":

            v_t, info = molmoact2_qdps_velocity(
                critic=critic,
                state=state,
                action_expert=action_expert,
                backbone=backbone,
                action_context=ctx["action_context"],
                modulation=modulation,
                x_t=trajectory,
                action_dim_is_pad=action_dim_is_pad,
                mask_enabled=mask_enabled,
                t=t,
                horizon=args.critic_horizon,
                action_dim=action_dim,
                weight=args.qdps_weight,
                max_norm=args.qdps_max_norm,
            )
            if info is not None:
                q_values.append(info["q"])
                qdps_norms.append(info["qdps_norm"])
            trajectory = trajectory + dt * v_t.detach()
            trajectory = backbone._mask_action_dim_tensor(
                trajectory, action_dim_is_pad=action_dim_is_pad, enabled=mask_enabled
            )
            continue

        with torch.no_grad():
            v_t = action_expert.forward_with_context(
                trajectory, modulation.conditioning, context=ctx["action_context"], modulation=modulation
            )
            v_t = backbone._mask_action_dim_tensor(v_t, action_dim_is_pad=action_dim_is_pad, enabled=mask_enabled)

        if args.mode == "qgf":
            v_t, info = molmoact2_qgf_velocity(
                critic=critic,
                state=state,
                x_t=trajectory,
                v_t=v_t,
                t=t,
                horizon=args.critic_horizon,
                action_dim=action_dim,
                weight=args.qgf_weight,
                max_norm=args.qgf_max_norm,
            )
            if info is not None:
                q_values.append(info["q"])
                qgf_norms.append(info["qgf_norm"])
                raw_grad_norms.append(info["raw_grad_norm"])
                applied_norms.append(info["applied_norm"])
                v_t_norms.append(info["v_t_norm"])
        elif args.mode == "agf":
            v_t, info = molmoact2_agf_velocity(
                guidance_model=guidance_model,
                state=state,
                x_t=trajectory,
                t_tensor=flow_timesteps[step],
                v_t=v_t,
                horizon=args.critic_horizon,
                action_dim=action_dim,
                weight=args.agf_weight,
            )
            agf_norms.append(info["agf_norm"])

        trajectory = trajectory + dt * v_t.detach()
        trajectory = backbone._mask_action_dim_tensor(
            trajectory, action_dim_is_pad=action_dim_is_pad, enabled=mask_enabled
        )

    raw_action = trajectory[:, :, :action_dim].detach()

    info = {}
    if args.mode == "qdps" and q_values:
        info = {
            "q_mean": sum(q_values) / len(q_values),
            "q_last": q_values[-1],
            "qdps_norm_mean": sum(qdps_norms) / len(qdps_norms),
        }
    elif q_values:
        info = {
            "q_mean": sum(q_values) / len(q_values),
            "q_last": q_values[-1],
            "qgf_norm_mean": sum(qgf_norms) / len(qgf_norms),
            "raw_grad_norm_mean": sum(raw_grad_norms) / len(raw_grad_norms),
            "applied_norm_mean": sum(applied_norms) / len(applied_norms),
            "v_t_norm_mean": sum(v_t_norms) / len(v_t_norms),
        }
    elif agf_norms:
        info = {"agf_norm_mean": sum(agf_norms) / len(agf_norms)}
    return raw_action, info


def sample_action_chunk_molmoact2_qrerank(policy, critic, batch, args, generator):

    k = args.qrerank_k
    backbone = policy._backbone()
    action_expert = backbone._require_action_expert()
    action_dim = policy._output_action_dim(batch)

    batch_k = _tile_batch(batch, k)
    with torch.no_grad():
        ctx = molmoact2_prefix_context(policy, batch_k, generator)
        trajectory = ctx["trajectory"]
        action_dim_is_pad = ctx["action_dim_is_pad"]
        batch_size_k = ctx["batch_size"]
        mask_enabled = policy.config.mask_action_dim_padding

        num_steps = int(getattr(policy.config, "num_inference_steps", None) or backbone.config.flow_matching_num_steps)
        dt = 1.0 / num_steps

        flow_timesteps = [
            torch.full((batch_size_k,), idx / num_steps, device=trajectory.device, dtype=torch.float32)
            for idx in range(num_steps)
        ]
        modulation_cache = action_expert.get_or_prepare_modulation_cache(
            flow_timesteps, cache_key=(num_steps, batch_size_k, trajectory.device, trajectory.dtype)
        )

        for step in range(num_steps):
            modulation = modulation_cache[step]
            v_t = action_expert.forward_with_context(
                trajectory, modulation.conditioning, context=ctx["action_context"], modulation=modulation
            )
            v_t = backbone._mask_action_dim_tensor(v_t, action_dim_is_pad=action_dim_is_pad, enabled=mask_enabled)
            trajectory = trajectory + dt * v_t
            trajectory = backbone._mask_action_dim_tensor(
                trajectory, action_dim_is_pad=action_dim_is_pad, enabled=mask_enabled
            )

        raw_action = trajectory[:, :, :action_dim]
        state_k = molmoact2_qpilots_images(batch_k, image_size=args.critic_image_size)
        q = critic_q(critic, state_k, raw_action, args.critic_horizon)
        best_idx = int(q.argmax().item())
        chosen = raw_action[best_idx : best_idx + 1].detach()

        info = {
            "q_values": q.detach().cpu().tolist(),
            "q_best": float(q[best_idx].detach().cpu()),
            "q_mean": float(q.mean().detach().cpu()),
            "q_std": float(q.std().detach().cpu()) if k > 1 else 0.0,
            "chosen_idx": best_idx,
        }
    return chosen, info
