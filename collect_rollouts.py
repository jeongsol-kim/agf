from __future__ import annotations


def _to_cpu(x):
    import torch

    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu(v) for v in x]
    return x


def run_episode_collect_smolvla(
    env,
    policy,
    env_preprocessor,
    preprocessor,
    postprocessor,
    max_steps=300,
    n_action_steps=10,
    solver="heun",
    gamma=0.99,
):
    from tasks.libero import flow_action, obs_to_batch

    obs, _ = env.reset()
    queue = []
    samples = []
    success = False
    success_step = None
    terminal_reason = "max_steps"

    for step in range(max_steps):
        if not queue:
            batch = obs_to_batch(obs, env.task_description, env_preprocessor, preprocessor)

            a0_raw = flow_action(policy, batch, solver=solver)
            a0_env = postprocessor(a0_raw)

            samples.append(
                {
                    "batch": _to_cpu(batch),
                    "a0_raw": a0_raw.cpu(),
                    "a0_env": a0_env.cpu(),
                    "step": step,
                    "task": env.task_description,
                    "chunk_id": len(samples),
                }
            )

            queue = list(a0_env[0, :n_action_steps].cpu().numpy())

        action = queue.pop(0)
        obs, _, terminated, _, info = env.step(action)

        success = bool(info.get("is_success", False))
        if success:
            success_step = step
            terminal_reason = "success"
            break
        if terminated:
            terminal_reason = "terminated"
            break

    episode_steps = step + 1
    for idx, s in enumerate(samples):
        next_step = samples[idx + 1]["step"] if idx + 1 < len(samples) else episode_steps
        s["success"] = float(success)
        s["episode_steps"] = episode_steps
        s["success_step"] = success_step
        s["terminal_reason"] = terminal_reason
        s["executed_steps"] = max(0, next_step - s["step"])
        s["target"] = float(gamma ** max(success_step - s["step"], 0)) if success else 0.0

    return samples, success, episode_steps


def collect_suite_smolvla(
    suite_name,
    policy,
    env_preprocessor,
    preprocessor,
    postprocessor,
    out_dir,
    n_episodes=10,
    max_steps=300,
    n_action_steps=10,
    solver="heun",
):
    from lerobot.envs.libero import LiberoEnv, _get_suite
    import json
    import os
    import torch

    suite = _get_suite(suite_name)
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    for tid in range(len(suite.tasks)):
        env = LiberoEnv(suite, tid, suite_name, obs_type="pixels_agent_pos", episode_length=max_steps)

        task_samples = []
        successes = []

        try:
            for ep in range(n_episodes):
                samples, success, steps = run_episode_collect_smolvla(
                    env,
                    policy,
                    env_preprocessor,
                    preprocessor,
                    postprocessor,
                    max_steps=max_steps,
                    n_action_steps=n_action_steps,
                    solver=solver,
                )

                for s in samples:
                    s["suite"] = suite_name
                    s["task_id"] = tid
                    s["episode_id"] = ep

                task_samples.extend(samples)
                successes.append(success)

                print(
                    f"[{suite_name} task {tid:02d} ep {ep:02d}] "
                    f"success={success} steps={steps} samples={len(samples)}"
                )
        finally:
            env.close()

        path = f"{out_dir}/task{tid:02d}.pt"
        torch.save(task_samples, path)

        summary[str(tid)] = {
            "task": suite.get_task(tid).language,
            "path": path,
            "n_samples": len(task_samples),
            "success_rate": sum(successes) / len(successes),
        }

    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def collect_smolvla_libero(args):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline

    policy = SmolVLAPolicy.from_pretrained(args.pretrained_path).to(args.device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.pretrained_path)

    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    for suite_name in [x.strip() for x in args.suites.split(",") if x.strip()]:
        collect_suite_smolvla(
            suite_name=suite_name,
            policy=policy,
            env_preprocessor=env_preprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            out_dir=f"{args.output_root}/{suite_name}",
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            n_action_steps=args.n_action_steps,
            solver=args.solver,
        )


def collect_vla_libero_pro(policy_type, args):
    from tasks.libero_pro import DEFAULT_CONDITIONS, parse_csv, parse_task_ids
    from tasks.observations import collect_rollouts

    conditions = parse_csv(args.conditions)
    unknown = set(conditions) - set(DEFAULT_CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")

    task_ids = parse_task_ids(args.task_ids)

    return collect_rollouts(
        policy_type=policy_type,
        pretrained_path=args.pretrained_path,
        base_suite=args.base_suite,
        conditions=conditions,
        episodes_per_condition=args.episodes_per_condition,
        output_root=args.output_root,
        generated_bddl_root=args.generated_bddl_root,
        libero_pro_root=args.libero_pro_root,
        task_ids=task_ids,
        n_action_steps=args.n_action_steps,
        device=args.device,
        base_seed=args.seed,
        gamma=args.gamma,
        image_size=args.image_size,
        num_steps_wait=args.num_steps_wait,
        resume=args.resume,
    )


def collect_molmoact2_libero_pro(args):
    from tasks.libero_pro import DEFAULT_CONDITIONS, collect_rollouts, parse_csv, parse_task_ids

    conditions = parse_csv(args.conditions)
    unknown = set(conditions) - set(DEFAULT_CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")

    task_ids = parse_task_ids(args.task_ids)

    return collect_rollouts(
        pretrained_path=args.pretrained_path,
        base_suite=args.base_suite,
        conditions=conditions,
        episodes_per_condition=args.episodes_per_condition,
        output_root=args.output_root,
        generated_bddl_root=args.generated_bddl_root,
        libero_pro_root=args.libero_pro_root,
        task_ids=task_ids,
        n_action_steps=args.n_action_steps,
        device=args.device,
        base_seed=args.seed,
        gamma=args.gamma,
        image_size=args.image_size,
        num_steps_wait=args.num_steps_wait,
        resume=args.resume,
    )


def run_episode_collect_pi05_robocasa(
    env, policy, camera_map, preprocessor, postprocessor, max_steps, n_action_steps, solver, gamma, seed
):
    from tasks.robocasa import flow_trajectory, obs_to_model_input, openpi_to_lerobot_action, set_seed
    import torch

    set_seed(seed)
    policy.eval()
    policy.reset()

    obs, _ = env.reset(seed=seed)

    queue = []
    samples = []
    success = False
    success_step = None
    terminal_reason = "max_steps"

    step = 0
    for step in range(max_steps):
        if not queue:
            batch = obs_to_model_input(obs, env.task_description, camera_map, preprocessor)
            a0_raw = flow_trajectory(policy, batch, solver=solver)
            action_chunk_np = postprocessor(a0_raw)[0].detach().float().cpu().numpy()
            a0_env_np = openpi_to_lerobot_action(action_chunk_np)

            samples.append(
                {
                    "batch": _to_cpu(batch),
                    "a0_raw": a0_raw.detach().cpu(),
                    "a0_env": torch.from_numpy(a0_env_np)[None],
                    "step": step,
                    "task": env.task_description,
                    "chunk_id": len(samples),
                }
            )

            queue = list(a0_env_np[:n_action_steps])

        action_np = queue.pop(0)

        if step == 0:
            env.reset(seed=seed)
        obs, _, terminated, truncated, info = env.step(action_np)
        success = bool(info.get("is_success", False))

        if success:
            success_step = step
            terminal_reason = "success"
            break
        if terminated:
            terminal_reason = "terminated"
            break
        if truncated:
            terminal_reason = "truncated"
            break

    episode_steps = step + 1
    for idx, s in enumerate(samples):
        next_step = samples[idx + 1]["step"] if idx + 1 < len(samples) else episode_steps
        s["success"] = float(success)
        s["episode_steps"] = episode_steps
        s["success_step"] = success_step
        s["terminal_reason"] = terminal_reason
        s["executed_steps"] = max(0, next_step - s["step"])
        s["target"] = float(gamma ** max(success_step - s["step"], 0)) if success else 0.0

    return samples, success, episode_steps


def collect_pi05_robocasa(args):
    from lerobot.envs.robocasa import RoboCasaEnv
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from options import ATOMIC_TASKS
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from tasks.robocasa import build_camera_map, episode_seed, set_seed
    import json
    import os
    import torch

    seed = args.seed
    obj_registries = tuple(x for x in args.obj_registries.split(",") if x)

    set_seed(seed)
    group_dir = f"{args.output_root}/{args.group_name}"
    os.makedirs(group_dir, exist_ok=True)

    policy = preprocessor = postprocessor = camera_map = None
    # Keep entries of tasks collected by earlier (subset) runs.
    summary_path = f"{group_dir}/summary.json"
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    for task_name in args.task_names:
        # Index in the full atomic list, so file names and episode seeds do not
        # depend on which subset is collected.
        task_index = ATOMIC_TASKS.index(task_name)

        print("\n" + "=" * 80)
        print(f"Collecting RoboCasa task: {task_name}")
        print("=" * 80)

        max_steps = int(get_task_horizon(task_name))

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
            obj_registries=obj_registries,
        )

        env.reset(seed=seed)
        if policy is None:
            policy = PI05Policy.from_pretrained(args.pretrained_path)
            policy = policy.to(args.device).eval()

            preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.pretrained_path)
            camera_map = build_camera_map(policy)

        task_samples = []
        successes = []

        try:
            for ep in range(args.n_episodes):
                seed_i = episode_seed(seed, task_index, ep)
                samples, success, steps = run_episode_collect_pi05_robocasa(
                    env=env,
                    policy=policy,
                    camera_map=camera_map,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    max_steps=max_steps,
                    n_action_steps=args.n_action_steps,
                    solver=args.solver,
                    gamma=args.gamma,
                    seed=seed_i,
                )

                for s in samples:
                    s["suite"] = args.group_name
                    s["task_id"] = task_index
                    s["episode_id"] = ep

                task_samples.extend(samples)
                successes.append(success)

                print(
                    f"[{task_name} | ep {ep:02d}] "
                    f"seed={seed_i} success={success} steps={steps} samples={len(samples)}"
                )
        finally:
            env.close()

        path = f"{group_dir}/task{task_index:02d}.pt"
        torch.save(task_samples, path)

        summary[str(task_index)] = {
            "task": task_name,
            "path": path,
            "n_samples": len(task_samples),
            "success_rate": sum(successes) / len(successes),
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def main(argv=None):
    import argparse
    import sys
    import options

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description="Collect VLA rollouts for Q/AGF training. Select --vla and --env; add --help for options.",
    )
    parser.add_argument("--vla", choices=("smolvla", "pi05", "molmoact2"), required=True)
    parser.add_argument("--env", choices=("libero", "libero_pro", "robocasa"), required=True)
    if not argv or (
        any(x in argv for x in ("-h", "--help"))
        and not all(any(x == flag or x.startswith(flag + "=") for x in argv) for flag in ("--vla", "--env"))
    ):
        parser.print_help()
        print("\nsmolvla: libero, libero_pro. pi05: libero_pro, robocasa. molmoact2: libero_pro.")
        return
    selection, remaining = parser.parse_known_args(argv)
    if selection.env == "robocasa" and selection.vla != "pi05":
        parser.error("RoboCasa collection is supported only for pi05")
    if selection.env == "libero" and selection.vla != "smolvla":
        parser.error(f"For {selection.vla}, use --env libero_pro --conditions clean for clean LIBERO rollouts")

    if selection.env == "libero":
        args = options.collect_smolvla_libero(remaining)
        return collect_smolvla_libero(args)
    if selection.env == "robocasa":
        args = options.collect_pi05_robocasa(remaining)
        return collect_pi05_robocasa(args)
    args = options.collect_libero_pro(selection.vla, remaining)
    if selection.vla == "molmoact2":
        return collect_molmoact2_libero_pro(args)
    return collect_vla_libero_pro(selection.vla, args)


if __name__ == "__main__":
    main()
