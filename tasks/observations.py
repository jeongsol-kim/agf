from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", os.path.join(os.environ.get("LIBERO_PRO_ROOT", "LIBERO-PRO"), ".libero_runtime")
)

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from libero.libero import benchmark, get_libero_path

from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_pre_post_processors
from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline


from tasks.libero_pro import (
    CONDITION_INDEX,
    TASK_MAX_STEPS,
    extract_language_from_text,
    load_perturbation_config,
    load_perturbation_module,
    make_env_from_bddl,
    make_episode_seed,
    perturb_bddl_content,
    reset_env,
    set_seed,
    sha1_text,
    to_cpu,
    write_episode_bddl,
)


EPISODE_CONDITION_STRIDE = 100_000


def add_batch_dim(x: Any):
    if isinstance(x, dict):
        return {k: add_batch_dim(v) for k, v in x.items()}

    return x[None]


def move_tensors_to_device(x: Any, device: str):
    if torch.is_tensor(x):
        return x.to(device)

    if isinstance(x, dict):
        return {k: move_tensors_to_device(v, device) for k, v in x.items()}

    if isinstance(x, list):
        return [move_tensors_to_device(v, device) for v in x]

    if isinstance(x, tuple):
        return tuple(move_tensors_to_device(v, device) for v in x)

    return x


def _controller_ee_ori_mat(env) -> np.ndarray:

    if hasattr(env, "robots"):
        return np.asarray(env.robots[0].controller.ee_ori_mat)

    if hasattr(env, "env") and hasattr(env.env, "robots"):
        return np.asarray(env.env.robots[0].controller.ee_ori_mat)

    raise AttributeError("Could not locate LIBERO robot controller for ee_ori_mat")


def obs_to_lerobot_batch(env, obs, task_description: str, env_preprocessor, preprocessor, device: str):

    raw = {
        "pixels": {
            "image": np.ascontiguousarray(np.asarray(obs["agentview_image"])),
            "image2": np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"])),
        },
        "robot_state": add_batch_dim(
            {
                "eef": {
                    "pos": np.asarray(obs["robot0_eef_pos"]),
                    "quat": np.asarray(obs["robot0_eef_quat"]),
                    "mat": (_controller_ee_ori_mat(env)),
                },
                "gripper": {
                    "qpos": np.asarray(obs["robot0_gripper_qpos"]),
                    "qvel": np.asarray(obs["robot0_gripper_qvel"]),
                },
                "joints": {"pos": np.asarray(obs["robot0_joint_pos"]), "vel": np.asarray(obs["robot0_joint_vel"])},
            }
        ),
    }

    batch = preprocess_observation(raw)

    batch["task"] = [task_description]

    batch = env_preprocessor(batch)

    batch = preprocessor(batch)

    return move_tensors_to_device(batch, device)


def load_policy_and_processors(policy_type: str, pretrained_path: str, device: str):
    print()
    print("=" * 80)
    print(f"Loading {policy_type}: " f"{pretrained_path}")
    print("=" * 80)

    if policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(pretrained_path)

    elif policy_type == "pi05":
        from lerobot.policies.pi05 import PI05Policy

        policy = PI05Policy.from_pretrained(pretrained_path)

    else:
        raise ValueError(policy_type)

    policy = policy.to(device).eval()

    for p in policy.parameters():
        p.requires_grad_(False)

    (preprocessor, postprocessor) = make_pre_post_processors(policy.config, pretrained_path=(pretrained_path))

    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    print("chunk_size:", getattr(policy.config, "chunk_size", None))

    print("n_action_steps(config):", getattr(policy.config, "n_action_steps", None))

    return (policy, env_preprocessor, preprocessor, postprocessor)


@torch.inference_mode()
def predict_action_chunk(policy, batch, postprocessor):

    a0_raw = policy.predict_action_chunk(batch)

    a0_env = postprocessor(a0_raw)

    if not torch.is_tensor(a0_raw):
        a0_raw = torch.as_tensor(a0_raw)

    if not torch.is_tensor(a0_env):
        a0_env = torch.as_tensor(a0_env)

    if a0_raw.ndim != 3:
        raise RuntimeError("a0_raw should be [B,T,A], " f"got {tuple(a0_raw.shape)}")

    if a0_env.ndim != 3:
        raise RuntimeError("a0_env should be [B,T,A], " f"got {tuple(a0_env.shape)}")

    return (a0_raw, a0_env)


def run_episode_collect(
    env,
    policy_type: str,
    policy,
    env_preprocessor,
    preprocessor,
    postprocessor,
    task_description: str,
    max_steps: int,
    n_action_steps: int,
    gamma: float,
    policy_seed: int,
    reset_seed: int,
    num_steps_wait: int,
    device: str,
):
    set_seed(policy_seed)

    policy.eval()
    policy.reset()

    obs = reset_env(env=env, seed=reset_seed, num_steps_wait=(num_steps_wait))

    queue = []
    samples = []

    success = False
    success_step = None

    terminal_reason = "max_steps"

    step = 0

    for step in range(max_steps):
        if not queue:

            model_batch = obs_to_lerobot_batch(
                env=env,
                obs=obs,
                task_description=(task_description),
                env_preprocessor=(env_preprocessor),
                preprocessor=(preprocessor),
                device=device,
            )

            (a0_raw, a0_env) = predict_action_chunk(policy=policy, batch=model_batch, postprocessor=(postprocessor))

            chunk_len = int(a0_env.shape[1])

            if n_action_steps > chunk_len:
                raise RuntimeError(f"n_action_steps=" f"{n_action_steps}, " f"chunk_len=" f"{chunk_len}")

            samples.append(
                {
                    "batch": (to_cpu(model_batch)),
                    "a0_raw": (a0_raw.detach().cpu()),
                    "a0_env": (a0_env.detach().cpu()),
                    "step": int(step),
                    "task": (task_description),
                    "chunk_id": int(len(samples)),
                }
            )

            queue = list(a0_env[0, :n_action_steps].detach().float().cpu().numpy())

        action = queue.pop(0)

        (obs, _, done, _) = env.step(action)

        success = bool(env.check_success())

        if success:
            success_step = int(step)
            terminal_reason = "success"
            break

        if done:
            terminal_reason = "done"
            break

    episode_steps = int(step + 1)

    for idx, sample in enumerate(samples):
        next_step = samples[idx + 1]["step"] if idx + 1 < len(samples) else episode_steps

        sample["success"] = float(success)

        sample["episode_steps"] = episode_steps

        sample["success_step"] = success_step

        sample["terminal_reason"] = terminal_reason

        sample["executed_steps"] = int(max(0, int(next_step) - int(sample["step"])))

        if success:

            delta = max(success_step - int(sample["step"]), 0)

            sample["target"] = float(gamma**delta)

        else:

            sample["target"] = 0.0

    return (samples, success, episode_steps)


def collect_rollouts(
    policy_type: str,
    pretrained_path: str,
    base_suite: str,
    conditions: list[str],
    episodes_per_condition: int,
    output_root: str,
    generated_bddl_root: str,
    libero_pro_root: str,
    task_ids: list[int] | None,
    n_action_steps: int,
    device: str,
    base_seed: int,
    gamma: float,
    image_size: int,
    num_steps_wait: int,
    resume: bool = False,
):
    set_seed(base_seed)

    libero_pro_root = Path(libero_pro_root).resolve()

    output_dir = Path(output_root) / base_suite

    output_dir.mkdir(parents=True, exist_ok=True)

    generated_bddl_root = Path(generated_bddl_root) / policy_type

    generated_bddl_root.mkdir(parents=True, exist_ok=True)

    perturbation = load_perturbation_module(libero_pro_root)

    perturbation_configs = load_perturbation_config(libero_pro_root)

    print("\nPerturbation configs:")

    for key, value in perturbation_configs.items():
        print(f"  {key:12s}: " f"{value}")

    benchmark_dict = benchmark.get_benchmark_dict()

    if base_suite not in benchmark_dict:
        raise ValueError(f"Unknown suite: " f"{base_suite}")

    task_suite = benchmark_dict[base_suite]()

    if task_ids is None:
        task_ids = list(range(task_suite.n_tasks))

    (policy, env_preprocessor, preprocessor, postprocessor) = load_policy_and_processors(
        policy_type=policy_type, pretrained_path=(pretrained_path), device=device
    )

    print("collector n_action_steps:", n_action_steps)

    max_steps = TASK_MAX_STEPS[base_suite]

    summary_path = output_dir / "summary.json"

    summary = {}

    if resume and summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        print(f"[resume] loaded " f"{summary_path} " f"({len(summary)} task(s))")

    for task_id in task_ids:

        task = task_suite.get_task(task_id)

        base_bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        if not base_bddl_path.exists():
            raise FileNotFoundError(base_bddl_path)

        with open(base_bddl_path, "r", encoding="utf-8") as f:
            base_content = f.read()

        task_name = base_bddl_path.stem

        task_path = output_dir / f"task{task_id:02d}.pt"

        if resume:

            existing = summary.get(str(task_id), {}).get("conditions", {})

            already_done = task_path.exists() and all(
                existing.get(c, {}).get("n_episodes") == episodes_per_condition for c in conditions
            )

            if already_done:
                print(f"[resume] task " f"{task_id:02d} " "already complete, " "skipping.")
                continue

            if task_path.exists():
                print(
                    f"[resume] task "
                    f"{task_id:02d} "
                    "is partial; restarting "
                    "this task from scratch "
                    "to avoid duplicate / "
                    "broken episode links."
                )

        original_language = extract_language_from_text(base_content)

        print()
        print("#" * 80)

        print(f"{policy_type} | " f"TASK {task_id:02d}")

        print(original_language)

        print("#" * 80)

        task_samples = []

        condition_summary = {}

        realization_hashes = {condition: set() for condition in conditions}

        for condition in conditions:

            print()
            print("=" * 80)

            print(f"condition: " f"{condition}")

            print("=" * 80)

            successes = []

            for ep in range(episodes_per_condition):

                episode_seed = make_episode_seed(
                    base_seed=(base_seed), task_id=(task_id), condition=(condition), episode_id=ep
                )

                perturbation_seed = episode_seed

                reset_seed = episode_seed + 100_000_000

                policy_seed = episode_seed + 200_000_000

                (episode_content, perturbation_info) = perturb_bddl_content(
                    perturbation=(perturbation),
                    base_content=(base_content),
                    condition=(condition),
                    base_suite=(base_suite),
                    task_name=(task_name),
                    perturbation_seed=(perturbation_seed),
                    perturbation_configs=(perturbation_configs),
                )

                task_description = extract_language_from_text(episode_content)

                realization_hash = sha1_text(episode_content)

                realization_hashes[condition].add(realization_hash)

                episode_bddl_path = write_episode_bddl(
                    generated_bddl_root=(generated_bddl_root),
                    base_suite=(base_suite),
                    task_id=(task_id),
                    task_name=(task_name),
                    condition=(condition),
                    episode_id=ep,
                    perturbation_seed=(perturbation_seed),
                    content=(episode_content),
                )

                env = None

                try:

                    env = make_env_from_bddl(bddl_path=(episode_bddl_path), image_size=(image_size), seed=(reset_seed))

                    (samples, success, steps) = run_episode_collect(
                        env=env,
                        policy_type=(policy_type),
                        policy=policy,
                        env_preprocessor=(env_preprocessor),
                        preprocessor=(preprocessor),
                        postprocessor=(postprocessor),
                        task_description=(task_description),
                        max_steps=(max_steps),
                        n_action_steps=(n_action_steps),
                        gamma=gamma,
                        policy_seed=(policy_seed),
                        reset_seed=(reset_seed),
                        num_steps_wait=(num_steps_wait),
                        device=device,
                    )

                except Exception as exc:

                    print(
                        f"[task "
                        f"{task_id:02d} | "
                        f"{condition:11s} | "
                        f"ep {ep:02d}] "
                        "ERROR during rollout, "
                        "skipping episode: "
                        f"{exc!r} "
                        f"pert_seed="
                        f"{perturbation_seed} "
                        f"bddl="
                        f"{episode_bddl_path}"
                    )

                    samples = []
                    success = False
                    steps = 0

                finally:

                    if env is not None:
                        env.close()

                global_episode_id = CONDITION_INDEX[condition] * EPISODE_CONDITION_STRIDE + ep

                for sample in samples:

                    sample["policy_type"] = policy_type

                    sample["pretrained_path"] = pretrained_path

                    sample["suite"] = base_suite

                    sample["task_id"] = int(task_id)

                    sample["episode_id"] = int(global_episode_id)

                    sample["condition_episode_id"] = int(ep)

                    sample["perturbation"] = condition

                    sample["seed"] = int(episode_seed)

                    sample["perturbation_seed"] = int(perturbation_seed)

                    sample["reset_seed"] = int(reset_seed)

                    sample["policy_seed"] = int(policy_seed)

                    sample["perturbation_hash"] = realization_hash

                    sample["base_task"] = original_language

                    sample["task"] = task_description

                    sample["generated_bddl"] = str(episode_bddl_path)

                    for key, value in perturbation_info.items():
                        sample[key] = value

                task_samples.extend(samples)

                successes.append(bool(success))

                sr = float(np.mean(successes))

                print(
                    f"[task "
                    f"{task_id:02d} | "
                    f"{condition:11s} | "
                    f"ep {ep:02d}] "
                    f"episode_id="
                    f"{global_episode_id} "
                    f"pert_seed="
                    f"{perturbation_seed} "
                    f"hash="
                    f"{realization_hash[:8]} "
                    f"success="
                    f"{success} "
                    f"steps="
                    f"{steps} "
                    f"chunks="
                    f"{len(samples)} "
                    f"SR="
                    f"{sr:.3f}"
                )

                torch.save(task_samples, task_path)

            condition_summary[condition] = {
                "n_episodes": int(len(successes)),
                "n_success": int(sum(successes)),
                "success_rate": float(np.mean(successes)),
                "n_unique_perturbations": int(len(realization_hashes[condition])),
            }

            summary[str(task_id)] = {
                "policy_type": (policy_type),
                "pretrained_path": (pretrained_path),
                "task": (original_language),
                "n_samples": int(len(task_samples)),
                "conditions": (condition_summary),
                "path": str(task_path),
            }

            with open(summary_path, "w", encoding="utf-8") as f:

                json.dump(summary, f, indent=2)

        n_total_episodes = episodes_per_condition * len(conditions)

        n_total_success = sum(x["n_success"] for x in condition_summary.values())

        summary[str(task_id)]["n_episodes_total"] = int(n_total_episodes)

        summary[str(task_id)]["n_success_total"] = int(n_total_success)

        summary[str(task_id)]["success_rate_total"] = float(n_total_success / max(n_total_episodes, 1))

        with open(summary_path, "w", encoding="utf-8") as f:

            json.dump(summary, f, indent=2)

        print()

        print(f"Saved task " f"{task_id}: " f"{task_path}")

        print(f"Samples: " f"{len(task_samples)}")

        for condition in conditions:

            print(f"  {condition:11s}: " f"{len(realization_hashes[condition])} " "unique BDDL realization(s)")

    return summary
