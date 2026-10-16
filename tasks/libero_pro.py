from __future__ import annotations

import os


os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


os.environ.setdefault(
    "LIBERO_CONFIG_PATH", os.path.join(os.environ.get("LIBERO_PRO_ROOT", "LIBERO-PRO"), ".libero_runtime")
)


import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.policies import make_pre_post_processors


DEFAULT_LIBERO_PRO_ROOT = Path(os.environ.get("LIBERO_PRO_ROOT", "LIBERO-PRO"))

TASK_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}

DEFAULT_CONDITIONS = ("clean", "position", "object", "semantic", "environment")

CONDITION_INDEX = {"clean": 0, "position": 1, "object": 2, "semantic": 3, "environment": 4}


def set_seed(seed: int):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_episode_seed(base_seed: int, task_id: int, condition: str, episode_id: int):

    condition_id = CONDITION_INDEX[condition]

    return int(base_seed + task_id * 1_000_000 + condition_id * 10_000 + episode_id)


def to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()

    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}

    if isinstance(x, list):
        return [to_cpu(v) for v in x]

    if isinstance(x, tuple):
        return tuple(to_cpu(v) for v in x)

    return x


def parse_csv(value: str):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_task_ids(value: str | None):
    if value is None:
        return None

    return [int(x.strip()) for x in value.split(",") if x.strip()]


def sha1_text(text: str):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()

    quat[3] = np.clip(quat[3], -1.0, 1.0)

    den = np.sqrt(max(1.0 - quat[3] ** 2, 0.0))

    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)

    return (quat[:3] * 2.0 * np.arccos(quat[3]) / den).astype(np.float32)


def load_perturbation_module(libero_pro_root: Path):

    root_str = str(libero_pro_root.resolve())

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    import perturbation

    return perturbation


def resolve_config_path(value: str, libero_pro_root: Path):

    if value is None:
        return None

    path = Path(value)

    if path.is_absolute():
        return str(path)

    raw = str(value)

    candidates = [Path.cwd() / raw, libero_pro_root / raw]

    cleaned = raw.lstrip("./")

    if cleaned.startswith("LIBERO-PRO/"):
        cleaned = cleaned[len("LIBERO-PRO/") :]

        candidates.append(libero_pro_root / cleaned)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"Could not resolve config path: {value}\n" f"Candidates:\n" + "\n".join(str(x) for x in candidates)
    )


def load_perturbation_config(libero_pro_root: Path):
    cfg_path = libero_pro_root / "evaluation_config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ood_cfg = dict(cfg.get("ood_task_configs", {}))

    resolved = {}

    for key, path in ood_cfg.items():
        resolved[key] = resolve_config_path(path, libero_pro_root)

    required = {"environment", "swap", "object", "language"}

    missing = required - set(resolved)

    if missing:
        raise RuntimeError("Missing LIBERO-Pro OOD configs: " f"{sorted(missing)}")

    return resolved


def extract_language_from_text(text: str):

    match = re.search(r"\(:language\s*(.*?)\)", text, flags=re.S)

    if not match:
        raise RuntimeError("Could not find (:language ...) " "in BDDL.")

    return match.group(1).strip()


def random_environment_perturb(
    perturbation, content: str, config_path: str, base_suite: str, task_name: str, seed: int
):

    parser = perturbation.BDDLParser(content)

    p = perturbation.EnvironmentReplacePerturbator(parser, config_path)

    current_env = p._extract_current_env(base_suite, task_name)

    if not current_env:
        raise RuntimeError(f"No environment config for " f"{base_suite}/{task_name}")

    if current_env not in p.ALLOWED_ENVS:
        raise RuntimeError(f"Current environment " f"{current_env!r} is not in " f"{p.ALLOWED_ENVS}")

    candidates = sorted(p.ALLOWED_ENVS - {current_env})

    if not candidates:
        raise RuntimeError("No alternative environment.")

    rng = random.Random(int(seed))

    new_env = rng.choice(candidates)

    new_content = content.replace(current_env, new_env)

    new_content = p._rewrite_problem_env_token(new_content, new_env)

    new_fix_type = p.ENV_FIXTYPE.get(new_env, None)

    if new_fix_type:
        new_content = p._rewrite_fixtures_type(new_content, new_env, new_fix_type)

    print(f"[dynamic environment] " f"{task_name}: " f"{current_env} -> {new_env} " f"(seed={seed})")

    return (new_content, {"environment_from": (current_env), "environment_to": (new_env)})


def perturb_bddl_content(
    perturbation,
    base_content: str,
    condition: str,
    base_suite: str,
    task_name: str,
    perturbation_seed: int,
    perturbation_configs: dict,
):

    if condition == "clean":
        return (base_content, {})

    random.seed(int(perturbation_seed))

    np.random.seed(int(perturbation_seed) % (2**32 - 1))

    if condition == "environment":

        return random_environment_perturb(
            perturbation=(perturbation),
            content=(base_content),
            config_path=(perturbation_configs["environment"]),
            base_suite=(base_suite),
            task_name=(task_name),
            seed=(perturbation_seed),
        )

    flags = perturbation.PerturbFlags(
        use_environment=False,
        use_swap=(condition == "position"),
        use_object=(condition == "object"),
        use_language=(condition == "semantic"),
        use_task=False,
    )

    pipeline = perturbation.BDDLCombinedPerturbator(configs=(perturbation_configs))

    new_content = pipeline.perturb_content(
        content=base_content,
        task_suite_name=(base_suite),
        task_name=(task_name),
        flags=flags,
        seed=int(perturbation_seed),
    )

    return (new_content, {})


def write_episode_bddl(
    generated_bddl_root: Path,
    base_suite: str,
    task_id: int,
    task_name: str,
    condition: str,
    episode_id: int,
    perturbation_seed: int,
    content: str,
):

    condition_dir = generated_bddl_root / base_suite / f"task{task_id:02d}" / condition

    condition_dir.mkdir(parents=True, exist_ok=True)

    path = condition_dir / (f"ep{episode_id:03d}" f"_seed{perturbation_seed}" f"_{task_name}.bddl")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path


def make_env_from_bddl(bddl_path: Path, image_size: int, seed: int):
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=(image_size), camera_widths=(image_size))

    env.seed(int(seed))

    return env


def reset_env(env, seed: int, num_steps_wait: int):

    env.seed(int(seed))

    obs = env.reset()

    dummy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(dummy_action)

    return obs


def obs_to_batch(obs, task_description: str, preprocessor):
    image = np.asarray(obs["agentview_image"])

    wrist = np.asarray(obs["robot0_eye_in_hand_image"])

    image = np.ascontiguousarray(image[::-1, ::-1])

    wrist = np.ascontiguousarray(wrist[::-1, ::-1])

    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)

    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)

    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)

    axis_angle = quat2axisangle(eef_quat)

    state = np.concatenate([eef_pos, axis_angle, gripper_qpos]).astype(np.float32)

    if state.shape != (8,):
        raise RuntimeError(f"Unexpected state shape: " f"{state.shape}")

    sample = {
        "observation.images.image": (torch.from_numpy(image).permute(2, 0, 1).contiguous()),
        "observation.images.wrist_image": (torch.from_numpy(wrist).permute(2, 0, 1).contiguous()),
        "observation.state": (torch.from_numpy(state)),
        "task": (task_description),
    }

    return preprocessor(sample)


@torch.inference_mode()
def predict_action_chunk(policy, batch, postprocessor):
    a0_raw = policy.predict_action_chunk(batch)

    a0_env = postprocessor(a0_raw)

    if not torch.is_tensor(a0_env):
        a0_env = torch.as_tensor(a0_env)

    if a0_raw.ndim != 3:
        raise RuntimeError(f"a0_raw should be [B,T,A], " f"got {tuple(a0_raw.shape)}")

    if a0_env.ndim != 3:
        raise RuntimeError(f"a0_env should be [B,T,A], " f"got {tuple(a0_env.shape)}")

    return (a0_raw, a0_env)


def run_episode_collect(
    env,
    policy,
    preprocessor,
    postprocessor,
    task_description: str,
    max_steps: int,
    n_action_steps: int,
    gamma: float,
    policy_seed: int,
    reset_seed: int,
    num_steps_wait: int,
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

            batch = obs_to_batch(obs=obs, task_description=(task_description), preprocessor=(preprocessor))

            (a0_raw, a0_env) = predict_action_chunk(policy=policy, batch=batch, postprocessor=(postprocessor))

            chunk_len = int(a0_env.shape[1])

            if n_action_steps > chunk_len:
                raise RuntimeError(f"n_action_steps=" f"{n_action_steps}, " f"chunk_len=" f"{chunk_len}")

            sample = {
                "batch": (to_cpu(batch)),
                "a0_raw": (a0_raw.detach().cpu()),
                "a0_env": (a0_env.detach().cpu()),
                "step": int(step),
                "task": (task_description),
                "chunk_id": int(len(samples)),
            }

            samples.append(sample)

            queue = list(a0_env[0, :n_action_steps].detach().float().cpu().numpy())

        action = queue.pop(0)

        obs, _, done, _ = env.step(action)

        success = bool(env.check_success())

        if success:
            success_step = step

            terminal_reason = "success"

            break

        if done:
            terminal_reason = "done"

            break

    episode_steps = step + 1

    for idx, sample in enumerate(samples):

        if idx + 1 < len(samples):
            next_step = samples[idx + 1]["step"]
        else:
            next_step = episode_steps

        sample["success"] = float(success)

        sample["episode_steps"] = int(episode_steps)

        sample["success_step"] = None if success_step is None else int(success_step)

        sample["terminal_reason"] = terminal_reason

        sample["executed_steps"] = int(max(0, next_step - sample["step"]))

        if success:
            delta = max(int(success_step) - int(sample["step"]), 0)

            sample["target"] = float(gamma**delta)

        else:
            sample["target"] = 0.0

    return (samples, success, episode_steps)


def collect_rollouts(
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

    generated_bddl_root = Path(generated_bddl_root)

    output_dir = Path(output_root) / base_suite

    output_dir.mkdir(parents=True, exist_ok=True)

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

    n_tasks = task_suite.n_tasks

    if task_ids is None:
        task_ids = list(range(n_tasks))

    print()
    print("=" * 80)
    print("Loading MolmoAct2")
    print("=" * 80)

    from lerobot.policies.molmoact2 import MolmoAct2Policy

    policy = MolmoAct2Policy.from_pretrained(pretrained_path).to(device).eval()

    policy.config.inference_action_mode = "continuous"

    for p in policy.parameters():
        p.requires_grad_(False)

    (preprocessor, postprocessor) = make_pre_post_processors(
        policy.config,
        pretrained_path=(pretrained_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    print("chunk_size:", getattr(policy.config, "chunk_size", None))

    print("n_action_steps(config):", getattr(policy.config, "n_action_steps", None))

    print("collector n_action_steps:", n_action_steps)

    max_steps = TASK_MAX_STEPS[base_suite]

    summary = {}
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        print(f"[summary] loaded existing summary: " f"{summary_path} " f"({len(summary)} task(s) recorded)")

    for task_id in task_ids:

        task = task_suite.get_task(task_id)

        base_bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        if not (base_bddl_path.exists()):
            raise FileNotFoundError(base_bddl_path)

        with open(base_bddl_path, "r", encoding="utf-8") as f:
            base_content = f.read()

        task_name = base_bddl_path.stem

        if resume:
            existing = summary.get(str(task_id), {}).get("conditions", {})

            already_done = (output_dir / f"task{task_id:02d}.pt").exists() and all(
                existing.get(c, {}).get("n_episodes") == episodes_per_condition for c in conditions
            )

            if already_done:
                print(f"[resume] task {task_id:02d} " f"already complete, skipping.")
                continue

        original_language = extract_language_from_text(base_content)

        print()
        print("#" * 80)
        print(f"TASK {task_id:02d}")
        print(original_language)
        print("#" * 80)

        task_path = output_dir / f"task{task_id:02d}.pt"

        if task_path.exists():
            existing_task_samples = torch.load(task_path, map_location="cpu")
            task_samples = [r for r in existing_task_samples if r.get("perturbation") not in conditions]
            if len(task_samples) != len(existing_task_samples):
                print(
                    f"[backfill] task {task_id:02d}: "
                    f"kept {len(task_samples)}/"
                    f"{len(existing_task_samples)} "
                    f"existing samples from "
                    f"conditions not being "
                    f"re-collected this run "
                    f"({conditions})"
                )
        else:
            task_samples = []

        condition_summary = {}

        realization_hashes = {condition: set() for condition in conditions}

        for condition in conditions:

            print()
            print("=" * 80)
            print(f"condition: {condition}")
            print("=" * 80)

            successes = []

            for ep in range(episodes_per_condition):

                episode_seed = make_episode_seed(
                    base_seed=(base_seed), task_id=(task_id), condition=(condition), episode_id=(ep)
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

                realization_hash = sha1_text(episode_content)

                realization_hashes[condition].add(realization_hash)

                task_description = extract_language_from_text(episode_content)

                episode_bddl_path = write_episode_bddl(
                    generated_bddl_root=(generated_bddl_root),
                    base_suite=(base_suite),
                    task_id=(task_id),
                    task_name=(task_name),
                    condition=(condition),
                    episode_id=(ep),
                    perturbation_seed=(perturbation_seed),
                    content=(episode_content),
                )

                env = None

                try:

                    if condition != "clean" and episode_content == base_content:

                        raise RuntimeError(
                            "Perturbation was a no-op "
                            "(no registered candidate objects available "
                            "for this task/condition -- check "
                            "ood_object.yaml). "
                            f"suite={base_suite} task={task_name} "
                            f"condition={condition} "
                            f"seed={perturbation_seed}"
                        )

                    env = make_env_from_bddl(bddl_path=(episode_bddl_path), image_size=(image_size), seed=(reset_seed))

                    (samples, success, steps) = run_episode_collect(
                        env=env,
                        policy=policy,
                        preprocessor=(preprocessor),
                        postprocessor=(postprocessor),
                        task_description=(task_description),
                        max_steps=(max_steps),
                        n_action_steps=(n_action_steps),
                        gamma=(gamma),
                        policy_seed=(policy_seed),
                        reset_seed=(reset_seed),
                        num_steps_wait=(num_steps_wait),
                    )

                except Exception as exc:

                    print(
                        f"[task {task_id:02d}"
                        f" | {condition:11s}"
                        f" | ep {ep:02d}] "
                        f"ERROR during rollout, "
                        f"skipping episode: "
                        f"{exc!r} "
                        f"pert_seed={perturbation_seed} "
                        f"bddl={episode_bddl_path}"
                    )

                    samples = []
                    success = False
                    steps = 0

                finally:
                    if env is not None:
                        env.close()

                for sample in samples:

                    sample["suite"] = base_suite

                    sample["task_id"] = int(task_id)

                    sample["episode_id"] = int(ep)

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

                successes.append(success)

                sr = float(np.mean(successes))

                print(
                    f"[task {task_id:02d}"
                    f" | {condition:11s}"
                    f" | ep {ep:02d}] "
                    f"pert_seed="
                    f"{perturbation_seed} "
                    f"hash="
                    f"{realization_hash[:8]} "
                    f"success={success} "
                    f"steps={steps} "
                    f"chunks={len(samples)} "
                    f"SR={sr:.3f}"
                )

                task_path = output_dir / f"task{task_id:02d}.pt"

                torch.save(task_samples, task_path)

            condition_summary[condition] = {
                "n_episodes": int(len(successes)),
                "n_success": int(sum(successes)),
                "success_rate": float(np.mean(successes)),
                "n_unique_perturbations": int(len(realization_hashes[condition])),
            }

            summary[str(task_id)] = {
                "task": (original_language),
                "n_samples": int(len(task_samples)),
                "conditions": (condition_summary),
                "path": str(output_dir / f"task{task_id:02d}.pt"),
            }

            with open(output_dir / "summary.json", "w", encoding="utf-8") as f:

                json.dump(summary, f, indent=2)

        n_total_episodes = episodes_per_condition * len(conditions)

        n_total_success = sum(x["n_success"] for x in condition_summary.values())

        summary[str(task_id)]["n_episodes_total"] = int(n_total_episodes)

        summary[str(task_id)]["n_success_total"] = int(n_total_success)

        summary[str(task_id)]["success_rate_total"] = float(n_total_success / max(n_total_episodes, 1))

        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:

            json.dump(summary, f, indent=2)

        print()
        print(f"Saved task {task_id}: " f"{output_dir / f'task{task_id:02d}.pt'}")

        print(f"Samples: " f"{len(task_samples)}")

        for condition in conditions:

            n_unique = len(realization_hashes[condition])

            print(f"  {condition:11s}: " f"{n_unique} unique " f"BDDL realization(s)")

    return summary
