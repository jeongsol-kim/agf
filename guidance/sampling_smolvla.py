from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

import logging

_orig_FileHandler = logging.FileHandler


def _patched_FileHandler(filename, *a, **kw):
    if filename == "/tmp/robosuite.log":
        filename = "/tmp/robosuite_{}.log".format(__import__("os").getuid())
    return _orig_FileHandler(filename, *a, **kw)


logging.FileHandler = _patched_FileHandler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_seed(base_seed: int, task_id: int, episode_id: int) -> int:
    return base_seed + task_id * 10000 + episode_id


def add_batch_dim(x):
    if isinstance(x, dict):
        return {k: add_batch_dim(v) for k, v in x.items()}
    return x[None]


def get_action_dim(policy):
    if hasattr(policy.config, "action_feature"):
        return policy.config.action_feature.shape[0]
    return policy.config.output_features["action"].shape[0]


def get_num_steps(model):
    if hasattr(model.config, "num_steps"):
        return model.config.num_steps
    return model.config.num_inference_steps


def obs_to_batch(obs, task_desc, env_preprocessor, preprocessor):
    from lerobot.envs.utils import preprocess_observation

    raw = {"pixels": obs["pixels"], "robot_state": add_batch_dim(obs["robot_state"])}

    batch = preprocess_observation(raw)
    batch["task"] = [task_desc]
    batch = env_preprocessor(batch)
    batch = preprocessor(batch)
    return batch


def qpilots_images(batch, image_size: int = 64):

    def _resize(x):
        x = x.float()
        if x.max() > 2.0:
            x = x / 255.0
        if tuple(x.shape[-2:]) != (image_size, image_size):
            x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return x

    image = _resize(batch["observation.images.image"])
    image2 = _resize(batch["observation.images.image2"])
    return image, image2


def critic_q(critic, state, action_chunk, horizon: int):

    image, image2 = state
    return critic.reward(image, image2, action_chunk[:, :horizon])


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

    pre_pad_k = pre_pad.repeat_interleave(k, dim=0)
    past_kv_k = {
        idx: {name: t.repeat_interleave(k, dim=0) for name, t in layer.items()} for idx, layer in past_kv.items()
    }
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


def agf_scheduled_weight(base_weight: float, t: float, schedule: str, max_scale: float = 8.0) -> float:

    if schedule == "constant":
        return base_weight
    if schedule == "sigma_half":
        if t >= 1.0 - 1e-6:
            return 0.0
        sigma = (2.0 * max(t, 0.0) / max(1.0 - t, 1e-4)) ** 0.5
        return base_weight * min(sigma / 2.0, max_scale)
    raise ValueError(f"Unknown agf weight schedule: {schedule!r}")


def guidance_net_velocity(
    guidance_model, image, image2, x, t_tensor, v_psi, horizon: int, action_dim: int, weight: float
):

    b_ml = v_psi

    g = guidance_model(image, image2, x, t_tensor).detach()

    g_region = g[:, :horizon, :action_dim]
    b_region = b_ml[:, :horizon, :action_dim]
    g_norm = g_region.flatten(1).norm(dim=-1).clamp_min(1e-8)
    b_norm = b_region.flatten(1).norm(dim=-1)
    g = g * (b_norm / g_norm).clamp(max=1.0).view(-1, 1, 1)

    info = {"agf_norm": float(g[:, :horizon, :action_dim].norm(dim=(1, 2)).mean().cpu())}
    return b_ml - weight * g, info


@torch.no_grad()
def build_policy_prefix(policy, batch):
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    m = policy.model

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]

    pre_emb, pre_pad, pre_att = m.embed_prefix(images, img_masks, tokens, masks, state=state)

    att_2d = make_att_2d_masks(pre_pad, pre_att)
    pos_ids = torch.cumsum(pre_pad, dim=1) - 1

    _, past_kv = m.vlm_with_expert.forward(
        attention_mask=att_2d,
        position_ids=pos_ids,
        past_key_values=None,
        inputs_embeds=[pre_emb, None],
        use_cache=m.config.use_cache,
        fill_kv_cache=True,
    )

    return pre_pad, past_kv, tokens.shape[0]


def sample_action_chunk(policy, critic, guidance_model, batch, args):

    m = policy.model
    device = next(policy.parameters()).device

    action_dim = get_action_dim(policy)
    num_steps = get_num_steps(m)

    pre_pad, past_kv, batch_size = build_policy_prefix(policy, batch)

    dt = -1.0 / num_steps
    state = qpilots_images(batch, image_size=args.qpilots_image_size)

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

    guidance_norms = []
    q_values = []
    qgf_norms = []

    for step in range(num_steps):
        t = 1.0 + step * dt
        t_tensor = torch.full((batch_size,), t, device=device)

        if args.mode != "qdps":

            with torch.no_grad():
                v_t = m.denoise_step(pre_pad, copy.deepcopy(past_kv), x_t, t_tensor)

        if args.mode == "agf":
            weight = agf_scheduled_weight(
                args.agf_weight, t, args.agf_weight_schedule, max_scale=args.agf_weight_schedule_max
            )
            v_t, info = guidance_net_velocity(
                guidance_model=guidance_model,
                image=state[0],
                image2=state[1],
                x=x_t,
                t_tensor=t_tensor,
                v_psi=v_t,
                horizon=args.critic_horizon,
                action_dim=action_dim,
                weight=weight,
            )
            guidance_norms.append(info["agf_norm"])

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
    if guidance_norms:
        info = {"agf_norm_mean": sum(guidance_norms) / len(guidance_norms)}
    elif q_values:
        info = {
            "q_mean": sum(q_values) / len(q_values),
            "q_last": q_values[-1],
            "qgf_norm_mean": sum(qgf_norms) / len(qgf_norms),
        }

    return raw_action, info
