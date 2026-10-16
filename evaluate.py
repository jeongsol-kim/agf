from __future__ import annotations

import importlib


def run_episode_smolvla(
    env,
    policy,
    critic,
    guidance_model,
    env_preprocessor,
    preprocessor,
    postprocessor,
    args,
    seed: int,
    task_id: int = 0,
):
    from guidance.sampling_smolvla import obs_to_batch, sample_action_chunk, set_seed
    import os

    set_seed(seed)

    obs, _ = env.reset(seed=seed)
    queue = []
    success = False
    episode_return = 0.0
    q_logs = []
    frames = [obs["pixels"]["image"]] if args.save_video else None

    for step in range(args.max_steps):
        if not queue:
            batch = obs_to_batch(obs, env.task_description, env_preprocessor, preprocessor)

            raw_chunk, q_info = sample_action_chunk(policy, critic, guidance_model, batch, args)

            if q_info:
                q_info["env_step"] = step
                q_logs.append(q_info)

            action_chunk = postprocessor(raw_chunk)
            queue = list(action_chunk[0, : args.n_action_steps].cpu().numpy())

        action = queue.pop(0)

        obs, reward, terminated, _, info = env.step(action)
        episode_return += float(reward)
        if frames is not None:
            frames.append(obs["pixels"]["image"])

        success = bool(info.get("is_success", False))
        if success or terminated:
            break

    if frames is not None:
        from lerobot.utils.io_utils import write_video

        video_path = os.path.join(
            args.output_dir, "videos", f"task{task_id:02d}_seed{seed}_{'success' if success else 'fail'}.mp4"
        )
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        write_video(video_path, frames, fps=20)

    return {"success": success, "steps": step + 1, "return": episode_return, "q_logs": q_logs}


def evaluate_tasks_smolvla(policy, critic, guidance_model, env_preprocessor, preprocessor, postprocessor, args):
    from guidance.sampling_smolvla import episode_seed
    import json
    import os

    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(args.suite)
    task_ids = args.task_ids if args.task_ids is not None else list(range(len(suite.tasks)))

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    for task_id in task_ids:
        env = LiberoEnv(suite, task_id, args.suite, obs_type="pixels_agent_pos", episode_length=args.max_steps)

        episodes = []

        try:
            for ep in range(args.n_episodes):
                seed = episode_seed(args.seed, task_id, ep)

                result = run_episode_smolvla(
                    env=env,
                    policy=policy,
                    critic=critic,
                    guidance_model=guidance_model,
                    env_preprocessor=env_preprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    args=args,
                    seed=seed,
                    task_id=task_id,
                )

                result["seed"] = seed
                episodes.append(result)

                print(
                    f"[{args.suite} | task {task_id:02d} | ep {ep:02d}] "
                    f"success={result['success']} steps={result['steps']}"
                )

        finally:
            env.close()

        sr = sum(e["success"] for e in episodes) / len(episodes)

        results[str(task_id)] = {"task": suite.get_task(task_id).language, "success_rate": sr, "episodes": episodes}

        print(f"[task {task_id:02d}] SR={sr:.1%} | {suite.get_task(task_id).language}")

    overall = sum(r["success_rate"] for r in results.values()) / len(results)

    summary = {
        "suite": args.suite,
        "pretrained_path": args.pretrained_path,
        "mode": args.mode,
        "overall_sr": overall,
        "n_episodes": args.n_episodes,
        "max_steps": args.max_steps,
        "n_action_steps": args.n_action_steps,
        "critic_ckpt": args.critic_ckpt,
        "critic_horizon": args.critic_horizon,
        "qgf_weight": args.qgf_weight,
        "qgf_max_norm": args.qgf_max_norm,
        "qrerank_k": args.qrerank_k,
        "agf_weight": args.agf_weight,
        "agf_weight_schedule": args.agf_weight_schedule,
        "agf_weight_schedule_max": args.agf_weight_schedule_max,
        "gradnet_ckpt": args.gradnet_ckpt,
        "seed": args.seed,
        "tasks": results,
    }

    path = os.path.join(args.output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOverall SR: {overall:.1%}")
    print(f"Saved to {path}")

    return summary


def evaluate_smolvla(args):
    from critics.libero import load_critic
    from guidance.networks_libero import image_encoders_from_critic, load_guidance_checkpoint
    from guidance.sampling_smolvla import get_action_dim, set_seed

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline

    set_seed(args.seed)

    if args.mode in ("qgf", "qdps", "qrerank") and args.critic_ckpt is None:
        raise ValueError(f"--critic_ckpt is required when --mode {args.mode}")
    if args.mode == "agf" and args.gradnet_ckpt is None:
        raise ValueError(f"--gradnet_ckpt is required when --mode {args.mode}")
    if args.mode == "agf" and args.critic_ckpt is None:
        raise ValueError(f"--critic_ckpt is required when --mode {args.mode} (needed for image encoders)")

    policy = SmolVLAPolicy.from_pretrained(args.pretrained_path).to(args.device)
    policy = policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.pretrained_path)

    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    critic = None
    guidance_model = None

    if args.mode in ("qgf", "qdps", "qrerank"):
        critic = load_critic(args.critic_ckpt, device=args.device)
        critic.eval()

        for p in critic.parameters():
            p.requires_grad_(False)

        expected_dim = args.critic_horizon * get_action_dim(policy)
        if expected_dim != critic.cfg.action_dim * args.critic_horizon:
            raise ValueError(
                f"critic.cfg.action_dim={critic.cfg.action_dim}, "
                f"but critic_horizon * action_dim = {expected_dim}. "
                f"Check --critic_horizon."
            )

    elif args.mode == "agf":
        critic = load_critic(args.critic_ckpt, device=args.device)
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)
        image_encoders = image_encoders_from_critic(critic, ensemble_size=1)

        guidance_model, _, _ = load_guidance_checkpoint(args.gradnet_ckpt, image_encoders, device=args.device)
        guidance_model.eval()
        for p in guidance_model.parameters():
            p.requires_grad_(False)

    evaluate_tasks_smolvla(
        policy=policy,
        critic=critic,
        guidance_model=guidance_model,
        env_preprocessor=env_preprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        args=args,
    )


def run_episode_guided_smolvla_pro(
    env,
    policy,
    critic,
    guidance_model,
    env_preprocessor,
    preprocessor,
    postprocessor,
    task_description: str,
    max_steps: int,
    n_action_steps: int,
    policy_seed: int,
    reset_seed: int,
    num_steps_wait: int,
    device: str,
    args,
):

    from tasks.libero_pro import reset_env, set_seed
    from tasks.observations import obs_to_lerobot_batch
    from guidance.sampling_smolvla import sample_action_chunk

    set_seed(policy_seed)
    policy.eval()
    policy.reset()

    obs = reset_env(env, seed=reset_seed, num_steps_wait=num_steps_wait)

    queue = []
    success = False
    q_logs = []
    step = 0

    for step in range(max_steps):
        if not queue:
            batch = obs_to_lerobot_batch(
                env=env,
                obs=obs,
                task_description=task_description,
                env_preprocessor=env_preprocessor,
                preprocessor=preprocessor,
                device=device,
            )

            raw_chunk, q_info = sample_action_chunk(policy, critic, guidance_model, batch, args)

            if q_info:
                q_info["env_step"] = step
                q_logs.append(q_info)

            action_chunk = postprocessor(raw_chunk)
            queue = list(action_chunk[0, :n_action_steps].detach().float().cpu().numpy())

        action = queue.pop(0)
        obs, _, done, _ = env.step(action)
        success = bool(env.check_success())

        if success or done:
            break

    return {"success": success, "steps": int(step) + 1, "q_logs": q_logs}


def evaluate_libero_pro_smolvla_pro(policy, env_preprocessor, preprocessor, postprocessor, args):
    from tasks.libero_pro import (
        TASK_MAX_STEPS,
        extract_language_from_text,
        load_perturbation_config,
        load_perturbation_module,
        make_env_from_bddl,
        make_episode_seed,
        perturb_bddl_content,
        write_episode_bddl,
    )
    from libero.libero import benchmark, get_libero_path
    from critics.libero import load_critic
    from guidance.networks_libero import image_encoders_from_critic, load_guidance_checkpoint
    from pathlib import Path
    import json
    import os

    libero_pro_root = Path(args.libero_pro_root).resolve()
    generated_bddl_root = Path(args.generated_bddl_root) / "smolvla_eval"
    generated_bddl_root.mkdir(parents=True, exist_ok=True)

    perturbation = load_perturbation_module(libero_pro_root)
    perturbation_configs = load_perturbation_config(libero_pro_root)

    task_suite = benchmark.get_benchmark_dict()[args.base_suite]()
    task_ids = args.task_ids if args.task_ids is not None else list(range(task_suite.n_tasks))
    max_steps = args.max_steps or TASK_MAX_STEPS[args.base_suite]

    os.makedirs(args.output_dir, exist_ok=True)

    condition_episodes = {c: [] for c in args.conditions}

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        base_bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not base_bddl_path.exists():
            raise FileNotFoundError(base_bddl_path)
        with open(base_bddl_path, "r", encoding="utf-8") as f:
            base_content = f.read()
        task_name = base_bddl_path.stem

        this_critic = None
        this_guidance_model = None
        if args.mode in ("qgf", "qdps", "qrerank", "agf"):
            critic_ckpt = args.critic_ckpt_template.format(suite=args.base_suite, task=task_id)
            this_critic = load_critic(critic_ckpt, device=args.device)
            this_critic.eval()
            for p in this_critic.parameters():
                p.requires_grad_(False)
            if args.mode == "agf":
                gradnet_ckpt = args.gradnet_ckpt_template.format(suite=args.base_suite, task=task_id)
                image_encoders = image_encoders_from_critic(this_critic, ensemble_size=1)
                this_guidance_model, _, _ = load_guidance_checkpoint(gradnet_ckpt, image_encoders, device=args.device)
                this_guidance_model.eval()
                for p in this_guidance_model.parameters():
                    p.requires_grad_(False)

        for condition in args.conditions:
            for ep in range(args.episodes_per_condition):
                episode_seed_ = make_episode_seed(
                    base_seed=args.seed, task_id=task_id, condition=condition, episode_id=ep
                )
                perturbation_seed = episode_seed_
                reset_seed = episode_seed_ + 100_000_000
                policy_seed = episode_seed_ + 200_000_000

                episode_content, _perturbation_info = perturb_bddl_content(
                    perturbation=perturbation,
                    base_content=base_content,
                    condition=condition,
                    base_suite=args.base_suite,
                    task_name=task_name,
                    perturbation_seed=perturbation_seed,
                    perturbation_configs=perturbation_configs,
                )
                task_description = extract_language_from_text(episode_content)

                episode_bddl_path = write_episode_bddl(
                    generated_bddl_root=generated_bddl_root,
                    base_suite=args.base_suite,
                    task_id=task_id,
                    task_name=task_name,
                    condition=condition,
                    episode_id=ep,
                    perturbation_seed=perturbation_seed,
                    content=episode_content,
                )

                env = None
                try:
                    env = make_env_from_bddl(bddl_path=episode_bddl_path, image_size=args.image_size, seed=reset_seed)
                    result = run_episode_guided_smolvla_pro(
                        env=env,
                        policy=policy,
                        critic=this_critic,
                        guidance_model=this_guidance_model,
                        env_preprocessor=env_preprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        task_description=task_description,
                        max_steps=max_steps,
                        n_action_steps=args.n_action_steps,
                        policy_seed=policy_seed,
                        reset_seed=reset_seed,
                        num_steps_wait=args.num_steps_wait,
                        device=args.device,
                        args=args,
                    )
                except Exception as exc:
                    print(
                        f"[task {task_id:02d} | {condition:11s} | ep {ep:02d}] "
                        f"ERROR during rollout, counting as failure: {exc!r} "
                        f"pert_seed={perturbation_seed} bddl={episode_bddl_path}"
                    )
                    result = {"success": False, "steps": 0, "q_logs": []}
                finally:
                    if env is not None:
                        env.close()

                result["task_id"] = int(task_id)
                result["condition"] = condition
                result["episode"] = int(ep)
                result["seed"] = int(episode_seed_)
                condition_episodes[condition].append(result)

                print(
                    f"[{args.base_suite} | task {task_id:02d} | {condition:11s} | ep {ep:02d}] "
                    f"success={result['success']} steps={result['steps']}"
                )

    per_condition_sr = {
        c: (sum(r["success"] for r in eps) / len(eps) if eps else float("nan")) for c, eps in condition_episodes.items()
    }
    n_total = sum(len(eps) for eps in condition_episodes.values())
    overall_sr = (
        sum(sum(r["success"] for r in eps) for eps in condition_episodes.values()) / n_total
        if n_total
        else float("nan")
    )

    summary = {
        "base_suite": args.base_suite,
        "conditions": list(args.conditions),
        "mode": args.mode,
        "overall_sr": overall_sr,
        "per_condition_sr": per_condition_sr,
        "episodes_per_condition": args.episodes_per_condition,
        "max_steps": max_steps,
        "n_action_steps": args.n_action_steps,
        "critic_ckpt_template": args.critic_ckpt_template,
        "critic_horizon": args.critic_horizon,
        "qgf_weight": args.qgf_weight,
        "qgf_max_norm": args.qgf_max_norm,
        "qrerank_k": args.qrerank_k,
        "agf_weight": args.agf_weight,
        "agf_weight_schedule": args.agf_weight_schedule,
        "agf_weight_schedule_max": args.agf_weight_schedule_max,
        "gradnet_ckpt_template": args.gradnet_ckpt_template,
        "seed": args.seed,
        "task_ids": list(task_ids),
        "condition_episodes": condition_episodes,
    }

    path = os.path.join(args.output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nOverall SR: {overall_sr:.1%}")
    for c, sr in per_condition_sr.items():
        print(f"  {c:12s}: {sr:.1%}")
    print(f"Saved to {path}")

    return summary


def evaluate_smolvla_pro(args):
    # Load LIBERO before tasks.libero_pro so the user's LIBERO config is used;
    # tasks.libero_pro otherwise defaults LIBERO_CONFIG_PATH to the LIBERO-PRO runtime dir.
    importlib.import_module("libero.libero")
    from tasks.libero_pro import parse_csv, parse_task_ids, set_seed
    from tasks.observations import load_policy_and_processors

    args.conditions = parse_csv(args.conditions)
    args.task_ids = parse_task_ids(args.task_ids)
    set_seed(args.seed)

    if args.mode in ("qgf", "qdps", "qrerank", "agf") and args.critic_ckpt_template is None:
        raise ValueError(f"--critic_ckpt_template is required when --mode {args.mode}")
    if args.mode == "agf" and args.gradnet_ckpt_template is None:
        raise ValueError("--gradnet_ckpt_template is required when --mode agf")

    policy, env_preprocessor, preprocessor, postprocessor = load_policy_and_processors(
        policy_type="smolvla", pretrained_path=args.pretrained_path, device=args.device
    )

    evaluate_libero_pro_smolvla_pro(
        policy=policy,
        env_preprocessor=env_preprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        args=args,
    )


def run_episode_guided_pi05_pro(
    env,
    policy,
    critic,
    guidance_model,
    env_preprocessor,
    preprocessor,
    postprocessor,
    task_description: str,
    max_steps: int,
    n_action_steps: int,
    policy_seed: int,
    reset_seed: int,
    num_steps_wait: int,
    device: str,
    args,
):
    from tasks.libero_pro import reset_env, set_seed
    from tasks.observations import obs_to_lerobot_batch
    from guidance.sampling_pi05_libero import sample_action_chunk

    set_seed(policy_seed)

    policy.eval()
    policy.reset()

    obs = reset_env(env, seed=reset_seed, num_steps_wait=num_steps_wait)

    queue = []
    success = False
    q_logs = []

    step = 0

    for step in range(max_steps):
        if not queue:
            batch = obs_to_lerobot_batch(
                env=env,
                obs=obs,
                task_description=task_description,
                env_preprocessor=env_preprocessor,
                preprocessor=preprocessor,
                device=device,
            )

            raw_chunk, q_info = sample_action_chunk(policy, critic, guidance_model, batch, args)

            if q_info:
                q_info["env_step"] = step
                q_logs.append(q_info)

            action_chunk = postprocessor(raw_chunk)

            queue = list(action_chunk[0, :n_action_steps].detach().float().cpu().numpy())

        action = queue.pop(0)

        obs, _, done, _ = env.step(action)

        success = bool(env.check_success())

        if success or done:
            break

    return {"success": success, "steps": int(step) + 1, "q_logs": q_logs}


def evaluate_libero_pro_pi05_pro(policy, env_preprocessor, preprocessor, postprocessor, args):
    from tasks.libero_pro import (
        TASK_MAX_STEPS,
        extract_language_from_text,
        load_perturbation_config,
        load_perturbation_module,
        make_env_from_bddl,
        make_episode_seed,
        perturb_bddl_content,
        write_episode_bddl,
    )
    from libero.libero import benchmark, get_libero_path
    from critics.libero import load_critic
    from guidance.networks_libero import image_encoders_from_critic, load_guidance_checkpoint
    from pathlib import Path
    import json
    import os

    libero_pro_root = Path(args.libero_pro_root).resolve()

    generated_bddl_root = Path(args.generated_bddl_root) / "pi05_eval"

    generated_bddl_root.mkdir(parents=True, exist_ok=True)

    perturbation = load_perturbation_module(libero_pro_root)

    perturbation_configs = load_perturbation_config(libero_pro_root)

    task_suite = benchmark.get_benchmark_dict()[args.base_suite]()

    task_ids = args.task_ids if args.task_ids is not None else list(range(task_suite.n_tasks))

    max_steps = args.max_steps or TASK_MAX_STEPS[args.base_suite]

    os.makedirs(args.output_dir, exist_ok=True)

    condition_episodes = {c: [] for c in args.conditions}

    for task_id in task_ids:
        task = task_suite.get_task(task_id)

        base_bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        if not base_bddl_path.exists():
            raise FileNotFoundError(base_bddl_path)

        with open(base_bddl_path, "r", encoding="utf-8") as f:
            base_content = f.read()

        task_name = base_bddl_path.stem

        this_critic = None
        this_guidance_model = None

        if args.mode in ("qgf", "qdps", "qrerank", "agf"):
            critic_ckpt = args.critic_ckpt_template.format(suite=args.base_suite, task=task_id)

            print(f"\n[task {task_id:02d}] " f"loading critic: {critic_ckpt}")

            this_critic = load_critic(critic_ckpt, device=args.device)

            this_critic.eval()

            for p in this_critic.parameters():
                p.requires_grad_(False)

            if args.mode == "agf":
                gradnet_ckpt = args.gradnet_ckpt_template.format(suite=args.base_suite, task=task_id)

                print(f"[task {task_id:02d}] " f"loading GuidanceNet: {gradnet_ckpt}")

                image_encoders = image_encoders_from_critic(this_critic, ensemble_size=1)

                (this_guidance_model, _, _) = load_guidance_checkpoint(gradnet_ckpt, image_encoders, device=args.device)

                this_guidance_model.eval()

                for p in this_guidance_model.parameters():
                    p.requires_grad_(False)

        for condition in args.conditions:
            for ep in range(args.episodes_per_condition):
                episode_seed_ = make_episode_seed(
                    base_seed=args.seed, task_id=task_id, condition=condition, episode_id=ep
                )

                perturbation_seed = episode_seed_

                reset_seed = episode_seed_ + 100_000_000

                policy_seed = episode_seed_ + 200_000_000

                episode_content, perturbation_info = perturb_bddl_content(
                    perturbation=perturbation,
                    base_content=base_content,
                    condition=condition,
                    base_suite=args.base_suite,
                    task_name=task_name,
                    perturbation_seed=perturbation_seed,
                    perturbation_configs=perturbation_configs,
                )

                task_description = extract_language_from_text(episode_content)

                episode_bddl_path = write_episode_bddl(
                    generated_bddl_root=generated_bddl_root,
                    base_suite=args.base_suite,
                    task_id=task_id,
                    task_name=task_name,
                    condition=condition,
                    episode_id=ep,
                    perturbation_seed=perturbation_seed,
                    content=episode_content,
                )

                env = None

                try:
                    env = make_env_from_bddl(bddl_path=episode_bddl_path, image_size=args.image_size, seed=reset_seed)

                    result = run_episode_guided_pi05_pro(
                        env=env,
                        policy=policy,
                        critic=this_critic,
                        guidance_model=this_guidance_model,
                        env_preprocessor=env_preprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        task_description=task_description,
                        max_steps=max_steps,
                        n_action_steps=args.n_action_steps,
                        policy_seed=policy_seed,
                        reset_seed=reset_seed,
                        num_steps_wait=args.num_steps_wait,
                        device=args.device,
                        args=args,
                    )

                except Exception as exc:
                    print(
                        f"[task {task_id:02d} | "
                        f"{condition:11s} | "
                        f"ep {ep:02d}] "
                        f"ERROR during rollout, "
                        f"counting as failure: "
                        f"{exc!r} "
                        f"pert_seed="
                        f"{perturbation_seed} "
                        f"bddl="
                        f"{episode_bddl_path}"
                    )

                    result = {"success": False, "steps": 0, "q_logs": []}

                finally:
                    if env is not None:
                        env.close()

                result["task_id"] = int(task_id)

                result["condition"] = condition

                result["episode"] = int(ep)

                result["seed"] = int(episode_seed_)

                result["perturbation_info"] = perturbation_info

                condition_episodes[condition].append(result)

                print(
                    f"[{args.base_suite} | "
                    f"task {task_id:02d} | "
                    f"{condition:11s} | "
                    f"ep {ep:02d}] "
                    f"success="
                    f"{result['success']} "
                    f"steps="
                    f"{result['steps']}"
                )

    per_condition_sr = {
        c: (sum(r["success"] for r in eps) / len(eps) if eps else float("nan")) for c, eps in condition_episodes.items()
    }

    n_total = sum(len(eps) for eps in condition_episodes.values())

    if n_total:
        overall_sr = sum(sum(r["success"] for r in eps) for eps in condition_episodes.values()) / n_total
    else:
        overall_sr = float("nan")

    summary = {
        "base_suite": args.base_suite,
        "conditions": list(args.conditions),
        "mode": args.mode,
        "overall_sr": overall_sr,
        "per_condition_sr": per_condition_sr,
        "episodes_per_condition": args.episodes_per_condition,
        "max_steps": max_steps,
        "n_action_steps": args.n_action_steps,
        "critic_ckpt_template": args.critic_ckpt_template,
        "critic_horizon": args.critic_horizon,
        "qgf_weight": args.qgf_weight,
        "qgf_max_norm": args.qgf_max_norm,
        "qrerank_k": args.qrerank_k,
        "agf_weight": args.agf_weight,
        "gradnet_ckpt_template": args.gradnet_ckpt_template,
        "seed": args.seed,
        "task_ids": list(task_ids),
        "condition_episodes": condition_episodes,
    }

    path = os.path.join(args.output_dir, "results.json")

    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nOverall SR: " f"{overall_sr:.1%}")

    for c, sr in per_condition_sr.items():
        print(f"  {c:12s}: " f"{sr:.1%}")

    print(f"Saved to {path}")

    return summary


def evaluate_pi05_pro(args):
    # Load LIBERO before tasks.libero_pro so the user's LIBERO config is used;
    # tasks.libero_pro otherwise defaults LIBERO_CONFIG_PATH to the LIBERO-PRO runtime dir.
    importlib.import_module("libero.libero")
    from tasks.libero_pro import parse_csv, parse_task_ids, set_seed
    from tasks.observations import load_policy_and_processors

    args.conditions = parse_csv(args.conditions)

    args.task_ids = parse_task_ids(args.task_ids)

    set_seed(args.seed)

    if args.mode in ("qgf", "qdps", "qrerank", "agf") and args.critic_ckpt_template is None:
        raise ValueError("--critic_ckpt_template is required " f"when --mode {args.mode}")

    if args.mode == "agf" and args.gradnet_ckpt_template is None:
        raise ValueError("--gradnet_ckpt_template is required " "when --mode agf")

    policy, env_preprocessor, preprocessor, postprocessor = load_policy_and_processors(
        policy_type="pi05", pretrained_path=args.pretrained_path, device=args.device
    )

    evaluate_libero_pro_pi05_pro(
        policy=policy,
        env_preprocessor=env_preprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        args=args,
    )


def run_episode_pi05_robocasa(
    env,
    policy,
    critic,
    guidance_model,
    camera_map,
    preprocessor,
    postprocessor,
    args,
    max_steps: int,
    seed: int,
    save_path=None,
):
    from tasks.robocasa import get_video_frame, obs_to_model_input, openpi_to_lerobot_action, save_video, set_seed
    from guidance.sampling_pi05_robocasa import sample_action_chunk

    set_seed(seed)
    policy.eval()
    policy.reset()

    obs, _ = env.reset(seed=seed)

    queue = []
    success = False
    q_logs = []
    frames = []

    if save_path:
        frames.append(get_video_frame(obs))

    step = 0
    for step in range(max_steps):
        if not queue:
            batch = obs_to_model_input(obs, env.task_description, camera_map, preprocessor)

            raw_chunk, q_info = sample_action_chunk(policy, critic, guidance_model, batch, camera_map, args)

            if q_info:
                q_info["env_step"] = step
                q_logs.append(q_info)

            action_chunk = postprocessor(raw_chunk)
            action_chunk_np = action_chunk[0, : args.n_action_steps].detach().float().cpu().numpy()
            action_chunk_np = openpi_to_lerobot_action(action_chunk_np)
            queue = list(action_chunk_np)

        action_np = queue.pop(0)

        if step == 0:
            env.reset(seed=seed)
        obs, _, terminated, truncated, info = env.step(action_np)
        success = bool(info.get("is_success", False))

        if save_path:
            frames.append(get_video_frame(obs))

        if terminated or truncated or success:
            break

    if save_path and frames:
        save_video(frames, save_path)

    return {"success": success, "steps": step + 1, "q_logs": q_logs}


def build_robocasa_env_pi05_robocasa(task_name: str, max_steps: int, args):

    from lerobot.envs.robocasa import RoboCasaEnv

    env = RoboCasaEnv(
        task=task_name,
        camera_name="robot0_agentview_left,robot0_eye_in_hand,robot0_agentview_right",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=args.observation_size,
        observation_height=args.observation_size,
        visualization_width=args.observation_size,
        visualization_height=args.observation_size,
        split=args.split,
        episode_length=max_steps,
        obj_registries=args.obj_registries,
    )
    env.reset(seed=args.seed)
    return env


def run_episode_with_retries_pi05_robocasa(
    env,
    policy,
    critic,
    guidance_model,
    camera_map,
    preprocessor,
    postprocessor,
    args,
    max_steps: int,
    seed: int,
    task_name: str,
    save_path=None,
    max_retries: int = 3,
):
    for attempt in range(max_retries):
        try:
            if env is None:
                env = build_robocasa_env_pi05_robocasa(task_name, max_steps, args)
            result = run_episode_pi05_robocasa(
                env=env,
                policy=policy,
                critic=critic,
                guidance_model=guidance_model,
                camera_map=camera_map,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                args=args,
                max_steps=max_steps,
                seed=seed,
                save_path=save_path,
            )
            return result, env
        except Exception as e:
            verb = "rebuilding env and retrying" if attempt + 1 < max_retries else "giving up"
            print(
                f"[{task_name} | seed={seed}] episode crashed on attempt "
                f"{attempt + 1}/{max_retries} ({type(e).__name__}: {e}); {verb}."
            )
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

            env = None

    return {"success": False, "steps": 0, "q_logs": [], "error": "env_reset_failed"}, env


def evaluate_pi05_robocasa(args):
    from tasks.robocasa import build_camera_map, episode_seed, set_seed
    from critics.robocasa import load_critic
    from guidance.networks_robocasa import image_encoders_from_critic, load_guidance_checkpoint
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    import json
    import os

    set_seed(args.seed)

    if args.mode in ("qgf", "qdps", "qrerank") and args.critic_ckpt_template is None:
        raise ValueError(f"--critic_ckpt_template is required when --mode {args.mode}")
    if args.mode == "agf" and args.gradnet_ckpt_template is None:
        raise ValueError("--gradnet_ckpt_template is required when --mode agf")
    if args.mode == "agf" and args.critic_ckpt_template is None:
        raise ValueError("--critic_ckpt_template is required when --mode agf (needed for image encoders)")

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    policy = preprocessor = postprocessor = camera_map = None

    for task_index, task_name in enumerate(args.task_names):
        print("\n" + "=" * 80)
        print(f"Evaluating RoboCasa task: {task_name}")
        print("=" * 80)

        max_steps = int(get_task_horizon(task_name))

        env = build_robocasa_env_pi05_robocasa(task_name, max_steps, args)
        if policy is None:
            policy = PI05Policy.from_pretrained(args.pretrained_path).to(args.device)
            policy = policy.eval()
            for p in policy.parameters():
                p.requires_grad_(False)
            preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.pretrained_path)
            camera_map = build_camera_map(policy)

        critic = guidance_model = None
        if args.mode in ("qgf", "qdps", "qrerank", "agf"):
            critic = load_critic(args.critic_ckpt_template.format(task=task_name), device=args.device)
            critic.eval()
            for p in critic.parameters():
                p.requires_grad_(False)

        if args.mode == "agf":
            image_encoders = image_encoders_from_critic(critic, ensemble_size=1)
            guidance_model, _, _ = load_guidance_checkpoint(
                args.gradnet_ckpt_template.format(task=task_name), image_encoders, device=args.device
            )
            guidance_model.eval()
            for p in guidance_model.parameters():
                p.requires_grad_(False)

        save_dir = None
        if args.save_videos:
            save_dir = os.path.join(args.output_dir, task_name, "videos")
            os.makedirs(save_dir, exist_ok=True)

        episodes = []
        try:
            for ep in range(args.n_episodes):
                seed = episode_seed(args.seed, task_index, ep)
                save_path = os.path.join(save_dir, f"ep{ep:02d}.mp4") if save_dir else None

                result, env = run_episode_with_retries_pi05_robocasa(
                    env=env,
                    policy=policy,
                    critic=critic,
                    guidance_model=guidance_model,
                    camera_map=camera_map,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    args=args,
                    max_steps=max_steps,
                    seed=seed,
                    task_name=task_name,
                    save_path=save_path,
                )
                result["seed"] = seed
                episodes.append(result)

                status = "success" if result["success"] else result.get("error", "fail")
                print(
                    f"[{task_name} | ep {ep:02d}] "
                    f"seed={seed} success={result['success']} steps={result['steps']} ({status})"
                )
        finally:
            if env is not None:
                env.close()

        sr = sum(e["success"] for e in episodes) / len(episodes)
        results[task_name] = {"success_rate": sr, "episodes": episodes}
        print(f"[{task_name}] SR={sr:.1%}")

    overall = sum(r["success_rate"] for r in results.values()) / len(results)

    summary = {
        "mode": args.mode,
        "pretrained_path": args.pretrained_path,
        "overall_sr": overall,
        "n_episodes": args.n_episodes,
        "n_action_steps": args.n_action_steps,
        "critic_ckpt_template": args.critic_ckpt_template,
        "critic_horizon": args.critic_horizon,
        "qgf_weight": args.qgf_weight,
        "qgf_max_norm": args.qgf_max_norm,
        "qrerank_k": args.qrerank_k,
        "agf_weight": args.agf_weight,
        "gradnet_ckpt_template": args.gradnet_ckpt_template,
        "seed": args.seed,
        "tasks": results,
    }

    path = os.path.join(args.output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOverall SR: {overall:.1%}")
    print(f"Saved to {path}")

    return summary


def run_episode_molmoact2(
    env, obs, task_description, policy, critic, guidance_model, preprocessor, postprocessor, args, policy_seed
):
    from guidance.sampling_molmoact2 import (
        sample_action_chunk_molmoact2,
        sample_action_chunk_molmoact2_qrerank,
        set_seed,
    )
    from tasks.libero_pro import obs_to_batch
    import numpy as np
    import torch

    set_seed(policy_seed)
    generator = torch.Generator(device=next(policy.parameters()).device).manual_seed(policy_seed)

    queue = []
    success = False
    q_logs = []

    for step in range(args.max_steps):
        if not queue:
            batch = obs_to_batch(obs=obs, task_description=task_description, preprocessor=preprocessor)

            if args.mode == "qrerank":
                raw_chunk, q_info = sample_action_chunk_molmoact2_qrerank(policy, critic, batch, args, generator)
            else:
                raw_chunk, q_info = sample_action_chunk_molmoact2(
                    policy, critic, guidance_model, batch, args, generator
                )
            if q_info:
                q_info["env_step"] = step
                q_logs.append(q_info)

            action_chunk = postprocessor(raw_chunk)
            n = min(args.n_action_steps, action_chunk.shape[1])
            queue = list(action_chunk[0, :n].detach().float().cpu().numpy())

        action = np.asarray(queue.pop(0), dtype=np.float32).reshape(-1)
        if action.shape != (7,):
            raise RuntimeError(f"Unexpected action shape: {action.shape}")
        if not np.all(np.isfinite(action)):
            raise RuntimeError(f"Non-finite action: {action}")

        obs, reward, done, info = env.step(action)
        success = bool(env.check_success())
        if success or done:
            break

    return {"success": success, "steps": step + 1, "q_logs": q_logs}


def evaluate_tasks_molmoact2(policy, critic, guidance_model, preprocessor, postprocessor, args):
    from tasks.libero_pro import (
        load_perturbation_config,
        load_perturbation_module,
        make_env_from_bddl,
        make_episode_seed,
        perturb_bddl_content,
        reset_env,
        write_episode_bddl,
    )
    from libero.libero import benchmark, get_libero_path
    from pathlib import Path
    import json
    import os

    libero_pro_root = Path(args.libero_pro_root).resolve()
    perturbation = load_perturbation_module(libero_pro_root)
    perturbation_configs = load_perturbation_config(libero_pro_root)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = args.task_ids if args.task_ids is not None else list(range(task_suite.n_tasks))

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        base_bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        base_content = base_bddl_path.read_text(encoding="utf-8")
        task_name = base_bddl_path.stem

        condition_results = {}

        for condition in conditions:
            episodes = []

            for ep in range(args.n_episodes):
                episode_seed_val = make_episode_seed(
                    base_seed=args.seed, task_id=task_id, condition=condition, episode_id=ep
                )
                perturbation_seed = episode_seed_val
                reset_seed = episode_seed_val + 100_000_000
                policy_seed = episode_seed_val + 200_000_000

                env = None
                try:
                    episode_content, _ = perturb_bddl_content(
                        perturbation=perturbation,
                        base_content=base_content,
                        condition=condition,
                        base_suite=args.suite,
                        task_name=task_name,
                        perturbation_seed=perturbation_seed,
                        perturbation_configs=perturbation_configs,
                    )
                    episode_bddl_path = write_episode_bddl(
                        generated_bddl_root=Path(args.generated_bddl_root),
                        base_suite=args.suite,
                        task_id=task_id,
                        task_name=task_name,
                        condition=condition,
                        episode_id=ep,
                        perturbation_seed=perturbation_seed,
                        content=episode_content,
                    )
                    env = make_env_from_bddl(bddl_path=episode_bddl_path, image_size=args.image_size, seed=reset_seed)
                    obs = reset_env(env=env, seed=reset_seed, num_steps_wait=args.num_steps_wait)

                    result = run_episode_molmoact2(
                        env=env,
                        obs=obs,
                        task_description=task.language,
                        policy=policy,
                        critic=critic,
                        guidance_model=guidance_model,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        args=args,
                        policy_seed=policy_seed,
                    )
                except Exception as exc:

                    print(
                        f"[{args.suite} | task {task_id:02d} | {condition:11s} | ep {ep:02d}] "
                        f"ERROR during rollout, skipping episode: {exc!r} pert_seed={perturbation_seed}"
                    )
                    result = {"success": False, "steps": 0, "q_logs": [], "error": repr(exc)}
                finally:
                    if env is not None:
                        env.close()

                result["seed"] = reset_seed
                episodes.append(result)
                print(
                    f"[{args.suite} | task {task_id:02d} | {condition:11s} | ep {ep:02d}] "
                    f"success={result['success']} steps={result['steps']}"
                )

            n_success = sum(e["success"] for e in episodes)
            sr = n_success / len(episodes) if episodes else 0.0
            condition_results[condition] = {
                "n_episodes": len(episodes),
                "n_success": n_success,
                "success_rate": sr,
                "episodes": episodes,
            }
            print(f"[task {task_id:02d} | {condition}] SR={sr:.1%}")

        total_ep = sum(c["n_episodes"] for c in condition_results.values())
        total_succ = sum(c["n_success"] for c in condition_results.values())
        results[str(task_id)] = {
            "task": task.language,
            "conditions": condition_results,
            "success_rate_total": total_succ / total_ep if total_ep else 0.0,
        }
        print(f"[task {task_id:02d}] overall SR={results[str(task_id)]['success_rate_total']:.1%} | {task.language}")

    overall = sum(r["success_rate_total"] for r in results.values()) / len(results) if results else 0.0
    summary = {
        "suite": args.suite,
        "conditions": conditions,
        "mode": args.mode,
        "overall_sr": overall,
        "n_episodes_per_condition": args.n_episodes,
        "critic_ckpt": args.critic_ckpt,
        "critic_horizon": args.critic_horizon,
        "qgf_weight": args.qgf_weight,
        "qgf_max_norm": args.qgf_max_norm,
        "gradnet_ckpt": args.gradnet_ckpt,
        "agf_weight": args.agf_weight,
        "qrerank_k": args.qrerank_k,
        "qdps_weight": args.qdps_weight,
        "qdps_max_norm": args.qdps_max_norm,
        "seed": args.seed,
        "tasks": results,
    }
    path = os.path.join(args.output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOverall SR: {overall:.1%}\nSaved to {path}")
    return summary


def evaluate_molmoact2(args):
    from tasks.libero_pro import DEFAULT_LIBERO_PRO_ROOT, TASK_MAX_STEPS
    from critics.libero import load_critic
    from guidance.networks_libero import image_encoders_from_critic, load_guidance_checkpoint
    from guidance.sampling_molmoact2 import _patch_time_conditioning_dtype
    import torch

    if args.mode == "qgf" and args.critic_ckpt is None:
        raise ValueError("--mode qgf requires --critic_ckpt")
    if args.mode == "qrerank" and args.critic_ckpt is None:
        raise ValueError("--mode qrerank requires --critic_ckpt")
    if args.mode == "qdps" and args.critic_ckpt is None:
        raise ValueError("--mode qdps requires --critic_ckpt")
    if args.mode == "agf" and args.gradnet_ckpt is None:
        raise ValueError("--mode agf requires --gradnet_ckpt")
    if args.mode == "agf" and args.critic_ckpt is None:
        raise ValueError("--mode agf requires --critic_ckpt (needed for GuidanceNet's reused image encoders)")
    if args.max_steps is None:
        args.max_steps = TASK_MAX_STEPS[args.suite]
    if args.libero_pro_root is None:
        args.libero_pro_root = str(DEFAULT_LIBERO_PRO_ROOT)

    from lerobot.policies.molmoact2 import MolmoAct2Policy
    from lerobot.policies import make_pre_post_processors

    policy = MolmoAct2Policy.from_pretrained(args.pretrained_path).to(args.device).eval()
    policy.config.inference_action_mode = "continuous"
    for p in policy.parameters():
        p.requires_grad_(False)
    if getattr(policy.config, "enable_lora_action_expert", False):

        _patch_time_conditioning_dtype(policy._backbone()._require_action_expert())

        model_dtype = getattr(torch, policy.config.model_dtype)
        policy = policy.to(dtype=model_dtype)

    if args.n_action_steps is None:
        args.n_action_steps = policy.config.n_action_steps

    critic = None
    if args.critic_ckpt is not None:
        critic = load_critic(args.critic_ckpt, device=args.device).eval()
        for p in critic.parameters():
            p.requires_grad_(False)

    guidance_model = None
    if args.mode == "agf":
        image_encoders = image_encoders_from_critic(critic, ensemble_size=1)
        guidance_model, _, _ = load_guidance_checkpoint(args.gradnet_ckpt, image_encoders, device=args.device)
        guidance_model.eval()
        for p in guidance_model.parameters():
            p.requires_grad_(False)

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    evaluate_tasks_molmoact2(policy, critic, guidance_model, preprocessor, postprocessor, args)


def main(argv=None):
    import argparse
    import sys
    import options

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description="Evaluate a VLA. Select --vla and --env; add --help for evaluation options.",
    )
    parser.add_argument("--vla", choices=("smolvla", "pi05", "molmoact2"), required=True)
    parser.add_argument("--env", choices=("libero", "libero_pro", "robocasa"), required=True)
    if not argv or (
        any(x in argv for x in ("-h", "--help"))
        and not all(any(x == flag or x.startswith(flag + "=") for x in argv) for flag in ("--vla", "--env"))
    ):
        parser.print_help()
        print("\nSmolVLA/MolmoAct2: libero, libero_pro. pi05: libero_pro, robocasa.")
        return
    selection, remaining = parser.parse_known_args(argv)
    if selection.env == "robocasa" and selection.vla != "pi05":
        parser.error("RoboCasa evaluation is supported only for pi05")
    if selection.vla == "pi05" and selection.env == "libero":
        parser.error("For pi05, use --env libero_pro --conditions clean for clean LIBERO evaluation")

    if selection.vla == "molmoact2":
        args = options.eval_molmoact2(remaining)
        return evaluate_molmoact2(args)
    if selection.vla == "smolvla":
        if selection.env == "libero":
            args = options.eval_smolvla(remaining)
            return evaluate_smolvla(args)
        args = options.eval_smolvla_pro(remaining)
        return evaluate_smolvla_pro(args)
    if selection.env == "robocasa":
        args = options.eval_pi05_robocasa(remaining)
        return evaluate_pi05_robocasa(args)
    args = options.eval_pi05_pro(remaining)
    return evaluate_pi05_pro(args)


if __name__ == "__main__":
    main()
