import os


os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import copy
import random
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.io_utils import write_video


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_seed(base_seed: int, task_id: int, episode_id: int) -> int:
    return base_seed + task_id * 10000 + episode_id


def obs_to_model_input(obs, task_desc, camera_map, preprocessor):
    observation = {
        "pixels": {camera_map.get(name, name): image for name, image in obs["pixels"].items()},
        "agent_pos": lerobot_to_openpi_state(obs["agent_pos"]),
    }
    observation = preprocess_observation(observation)
    observation["task"] = [task_desc]
    return preprocessor(observation)


def lerobot_to_openpi_state(s):
    return np.concatenate([s[..., 7:10], s[..., 10:14], s[..., 0:3], s[..., 3:7], s[..., 14:16]], axis=-1)


def openpi_to_lerobot_action(a):
    return np.concatenate([a[..., 7:11], a[..., 11:12], a[..., 0:3], a[..., 3:6], a[..., 6:7]], axis=-1)


def _get_num_inference_steps(model):
    if hasattr(model.config, "num_inference_steps"):
        return model.config.num_inference_steps
    if hasattr(model.config, "num_steps"):
        return model.config.num_steps
    raise AttributeError("Cannot find num_inference_steps or num_steps in model.config")


def _get_action_dim(policy):
    if hasattr(policy.config, "action_feature"):
        return policy.config.action_feature.shape[0]
    if hasattr(policy.config, "output_features"):
        return policy.config.output_features["action"].shape[0]
    raise AttributeError("Cannot find action dimension from policy.config")


def flow_trajectory(policy, batch, num_steps=None, solver="euler"):

    m = policy.model
    device = next(policy.parameters()).device
    num_steps = num_steps or _get_num_inference_steps(m)

    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]

    with torch.inference_mode():
        pre_emb, pre_pad, pre_att = m.embed_prefix(images, img_masks, tokens, masks)

        att_2d = make_att_2d_masks(pre_pad, pre_att)
        att_2d = m._prepare_attention_masks_4d(att_2d)
        pos_ids = torch.cumsum(pre_pad, dim=1) - 1

        try:
            (m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation) = "eager"
        except AttributeError:
            pass

        _, past_kv = m.paligemma_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[pre_emb, None],
            use_cache=True,
        )

        batch_size = tokens.shape[0]
        x_t = m.sample_noise((batch_size, m.config.chunk_size, m.config.max_action_dim), device)
        dt = -1.0 / num_steps

        def step_fn(x, t):
            return m.denoise_step(pre_pad, copy.deepcopy(past_kv), x, torch.full((batch_size,), t, device=device))

        for step in range(num_steps):
            t = 1.0 + step * dt
            if solver == "heun":
                v1 = step_fn(x_t, t)
                v2 = step_fn(x_t + dt * v1, max(t + dt, 0.0))
                v_t = 0.5 * (v1 + v2)
            else:
                v_t = step_fn(x_t, t)
            x_t = x_t + dt * v_t

        action = x_t[:, :, : _get_action_dim(policy)]

    return action


VIDEO_CAMERA = "robot0_agentview_left"


def to_uint8_rgb(frame):
    frame = np.asarray(frame)

    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    if np.issubdtype(frame.dtype, np.floating):
        if float(frame.min()) >= 0.0 and float(frame.max()) <= 1.0 + 1e-6:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(frame)


def get_video_frame(obs, camera_name=VIDEO_CAMERA):
    return to_uint8_rgb(obs["pixels"][camera_name]).copy()


def save_video(frames, path, fps=20):
    if not frames:
        print(f"[warn] no frames to save: {path}")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(video_path=path, stacked_frames=frames, fps=fps)


def build_camera_map(policy):
    native_cameras = ["robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"]
    camera_aliases = ["camera1", "camera2", "camera3"]
    input_features = getattr(policy.config, "input_features", {})

    camera_map = {}
    for native, alias in zip(native_cameras, camera_aliases, strict=True):
        native_key = f"observation.images.{native}"
        alias_key = f"observation.images.{alias}"
        if alias_key in input_features and native_key not in input_features:
            camera_map[native] = alias
        else:
            camera_map[native] = native

    print("RoboCasa camera mapping:")
    for source, target in camera_map.items():
        print(f"  {source} -> {target}")

    return camera_map
