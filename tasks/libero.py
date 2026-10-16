import copy

import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def add_batch(x):
    if isinstance(x, dict):
        return {k: add_batch(v) for k, v in x.items()}
    return x[None]


def obs_to_batch(obs, task, env_preprocessor, preprocessor):
    raw = {"pixels": obs["pixels"], "robot_state": add_batch(obs["robot_state"])}
    batch = preprocess_observation(raw)
    batch["task"] = [task]
    batch = env_preprocessor(batch)
    batch = preprocessor(batch)
    return batch


def get_action_dim(policy):
    if hasattr(policy.config, "action_feature"):
        return policy.config.action_feature.shape[0]
    return policy.config.output_features["action"].shape[0]


def get_num_steps(model):
    if hasattr(model.config, "num_steps"):
        return model.config.num_steps
    if hasattr(model.config, "num_inference_steps"):
        return model.config.num_inference_steps
    raise AttributeError("Cannot find num_steps or num_inference_steps in model.config")


@torch.inference_mode()
def flow_action(policy, batch, solver="heun"):
    m = policy.model
    device = next(policy.parameters()).device
    B = batch["observation.language.tokens"].shape[0]
    T = get_num_steps(m)

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

    x = m.sample_noise((B, m.config.chunk_size, m.config.max_action_dim), device)
    dt = -1.0 / T

    def denoise(x, t):
        return m.denoise_step(pre_pad, copy.deepcopy(past_kv), x, torch.full((B,), t, device=device))

    for i in range(T):
        t = 1.0 + i * dt

        if solver == "heun":
            v1 = denoise(x, t)
            v2 = denoise(x + dt * v1, max(t + dt, 0.0))
            v = 0.5 * (v1 + v2)
        else:
            v = denoise(x, t)

        x = x + dt * v

    return x[:, :, : get_action_dim(policy)]
