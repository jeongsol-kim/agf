from __future__ import annotations

import os


os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import copy

import torch


from guidance.targets_pi05_robocasa import _resize_critic_image
from tasks.datasets import RoboCasaCriticDataTaskDataset


def get_action_dim(policy):
    if hasattr(policy.config, "action_feature"):
        return policy.config.action_feature.shape[0]
    return policy.config.output_features["action"].shape[0]


def get_num_steps(model):
    if hasattr(model.config, "num_steps"):
        return model.config.num_steps
    return model.config.num_inference_steps


def qpilots_images(batch, camera_map, image_size: int = 64):

    native_names = [key.removeprefix("observation.images.") for key in RoboCasaCriticDataTaskDataset.CAMERA_KEYS]
    images = tuple(
        _resize_critic_image(batch[f"observation.images.{camera_map.get(name, name)}"], image_size)
        for name in native_names
    )
    state = batch["observation.state"].float()
    return images + (state,)


def critic_q(critic, state, action_chunk, horizon: int):

    action = action_chunk[:, :horizon].reshape(action_chunk.shape[0], -1)
    image, image2, image3, agent_pos = state
    action_dim = action.shape[-1] // horizon
    action = action.view(action_chunk.shape[0], horizon, action_dim)
    return critic.reward((image, image2, image3), agent_pos, action)


def qgf_velocity(
    critic, state, x_t, v_t, t: float, horizon: int, action_dim: int, weight: float, max_norm: float | None
):
    """Detach the denoiser velocity when differentiating Q with respect to the noisy action."""
    if critic is None or weight == 0.0:
        return v_t, None

    x = x_t.detach().clone().requires_grad_(True)
    x0_hat = x - t * v_t.detach()
    action = x0_hat[:, :, :action_dim]

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
        "qgf_norm": float(g[:, :horizon, :action_dim].norm(dim=(1, 2)).mean().detach().cpu()),
    }

    return v_t - weight * g.detach(), info


def qdps_velocity(
    critic,
    state,
    m,
    pre_pad,
    past_kv,
    x_t,
    t_tensor,
    t: float,
    horizon: int,
    action_dim: int,
    weight: float,
    max_norm: float | None,
):
    """Differentiate Q through the denoiser, including its action Jacobian."""
    if critic is None or weight == 0.0:
        with torch.no_grad():
            v_t = m.denoise_step(pre_pad, copy.deepcopy(past_kv), x_t, t_tensor)
        return v_t, None

    x = x_t.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        v_t = m.denoise_step(pre_pad, copy.deepcopy(past_kv), x, t_tensor)
    x0_hat = x - t * v_t
    action = x0_hat[:, :, :action_dim]

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
        "qgf_norm": float(g[:, :horizon, :action_dim].norm(dim=(1, 2)).mean().detach().cpu()),
    }

    return v_t.detach() - weight * g.detach(), info


def qrerank_sample(
    m,
    critic,
    pre_pad,
    past_kv,
    state,
    batch_size: int,
    action_dim: int,
    num_steps: int,
    dt: float,
    device,
    chunk_size: int,
    max_action_dim: int,
    horizon: int,
    k: int,
):

    from transformers.cache_utils import DynamicCache

    pre_pad_k = pre_pad.repeat_interleave(k, dim=0)
    past_kv_k = DynamicCache(
        tuple(
            (keys.repeat_interleave(k, dim=0), values.repeat_interleave(k, dim=0), sliding_window)
            for keys, values, sliding_window in past_kv
        )
    )
    state_k = tuple(s.repeat_interleave(k, dim=0) for s in state)

    x_t = m.sample_noise((batch_size * k, chunk_size, max_action_dim), device)

    with torch.no_grad():
        for step in range(num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.full((batch_size * k,), t, device=device)
            v_t = m.denoise_step(pre_pad_k, copy.deepcopy(past_kv_k), x_t, t_tensor)
            x_t = x_t + dt * v_t

        raw_action = x_t[:, :, :action_dim]
        q = critic_q(critic, state_k, raw_action, horizon).view(batch_size, k)

    local_idx = q.argmax(dim=1)
    idx = torch.arange(batch_size, device=device) * k + local_idx
    chosen = raw_action[idx]

    info = {
        "q_values": q.detach().cpu().tolist(),
        "q_best": q.gather(1, local_idx[:, None]).squeeze(1).detach().cpu().tolist(),
        "q_mean": q.mean(dim=1).detach().cpu().tolist(),
        "q_std": q.std(dim=1).detach().cpu().tolist(),
        "chosen_idx": local_idx.detach().cpu().tolist(),
    }
    return chosen, info


def agf_velocity(
    guidance_model, image, image2, image3, state, x, t_tensor, v_psi, horizon: int, action_dim: int, weight: float
):

    b_ml = v_psi
    g = guidance_model(image, image2, image3, state, x, t_tensor).detach()

    g_region = g[:, :horizon, :action_dim]
    b_region = b_ml[:, :horizon, :action_dim]
    g_norm = g_region.flatten(1).norm(dim=-1).clamp_min(1e-8)
    b_norm = b_region.flatten(1).norm(dim=-1)
    g = g * (b_norm / g_norm).clamp(max=1.0).view(-1, 1, 1)

    info = {"agf_norm": float(g[:, :horizon, :action_dim].norm(dim=(1, 2)).mean().cpu())}
    return b_ml - weight * g, info


@torch.no_grad()
def build_policy_prefix(policy, batch):
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    m = policy.model

    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]

    pre_emb, pre_pad, pre_att = m.embed_prefix(images, img_masks, tokens, masks)

    att_2d = make_att_2d_masks(pre_pad, pre_att)
    att_2d = m._prepare_attention_masks_4d(att_2d)
    pos_ids = torch.cumsum(pre_pad, dim=1) - 1

    try:
        m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    except AttributeError:
        pass

    _, past_kv = m.paligemma_with_expert.forward(
        attention_mask=att_2d, position_ids=pos_ids, past_key_values=None, inputs_embeds=[pre_emb, None], use_cache=True
    )

    return pre_pad, past_kv, tokens.shape[0]


def sample_action_chunk(policy, critic, guidance_model, batch, camera_map, args):

    m = policy.model
    device = next(policy.parameters()).device

    action_dim = get_action_dim(policy)
    num_steps = get_num_steps(m)

    pre_pad, past_kv, batch_size = build_policy_prefix(policy, batch)

    dt = -1.0 / num_steps
    state = qpilots_images(batch, camera_map, image_size=args.qpilots_image_size)

    if args.mode == "qrerank":
        return qrerank_sample(
            m=m,
            critic=critic,
            pre_pad=pre_pad,
            past_kv=past_kv,
            state=state,
            batch_size=batch_size,
            action_dim=action_dim,
            num_steps=num_steps,
            dt=dt,
            device=device,
            chunk_size=m.config.chunk_size,
            max_action_dim=m.config.max_action_dim,
            horizon=args.critic_horizon,
            k=args.qrerank_k,
        )

    x_t = m.sample_noise((batch_size, m.config.chunk_size, m.config.max_action_dim), device)

    agf_norms = []
    q_values = []
    qgf_norms = []

    for step in range(num_steps):
        t = 1.0 + step * dt
        t_tensor = torch.full((batch_size,), t, device=device)

        if args.mode != "qdps":

            with torch.no_grad():
                v_t = m.denoise_step(pre_pad, copy.deepcopy(past_kv), x_t, t_tensor)

        if args.mode == "agf":
            v_t, info = agf_velocity(
                guidance_model=guidance_model,
                image=state[0],
                image2=state[1],
                image3=state[2],
                state=state[3],
                x=x_t,
                t_tensor=t_tensor,
                v_psi=v_t,
                horizon=args.critic_horizon,
                action_dim=action_dim,
                weight=args.agf_weight,
            )
            agf_norms.append(info["agf_norm"])

        elif args.mode == "qgf":
            v_t, info = qgf_velocity(
                critic=critic,
                state=state,
                x_t=x_t,
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

        elif args.mode == "qdps":
            v_t, info = qdps_velocity(
                critic=critic,
                state=state,
                m=m,
                pre_pad=pre_pad,
                past_kv=past_kv,
                x_t=x_t,
                t_tensor=t_tensor,
                t=t,
                horizon=args.critic_horizon,
                action_dim=action_dim,
                weight=args.qgf_weight,
                max_norm=args.qgf_max_norm,
            )

            if info is not None:
                q_values.append(info["q"])
                qgf_norms.append(info["qgf_norm"])

        x_t = x_t + dt * v_t.detach()

    raw_action = x_t[:, :, :action_dim].detach()

    info = {}
    if agf_norms:
        info = {"agf_norm_mean": sum(agf_norms) / len(agf_norms)}
    elif q_values:
        info = {
            "q_mean": sum(q_values) / len(q_values),
            "q_last": q_values[-1],
            "qgf_norm_mean": sum(qgf_norms) / len(qgf_norms),
        }

    return raw_action, info
