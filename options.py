import argparse

import os


LIBERO_SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]
PRO_CONDITIONS = "clean,position,object,semantic,environment"

ATOMIC_TASKS = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]

IMAGE2_KEY_HELP = (
    "wrist-camera key in the saved batch: 'observation.images.image2' (smolvla_clean_action) "
    "or 'observation.images.wrist_image' (molmoact2_libero_pro_dynamic)"
)


# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------


def _critic_training(p, critic_data_root, output_dir):
    p.add_argument("--critic_data_root", default=critic_data_root)
    p.add_argument("--output_dir", default=output_dir)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)


def _guidance_training(
    p, *, pretrained_path, output_dir, n_steps, batch_size, critic_horizon, action_dim, batch_help=None
):
    p.add_argument("--pretrained_path", default=pretrained_path)
    p.add_argument("--critic_ckpt", required=True)
    p.add_argument("--output_dir", default=output_dir)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=n_steps)
    p.add_argument("--batch_size", type=int, default=batch_size, help=batch_help)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--ensemble_size", type=int, default=1)
    p.add_argument("--image_feature_dim", type=int, default=256)
    p.add_argument("--hidden_dims", default="128,128,128", help="comma-separated MLP hidden dims")
    p.add_argument("--no_layer_norm", action="store_true")
    p.add_argument("--time_embed_dim", type=int, default=64)
    p.add_argument("--target_scale", type=float, default=1.0)
    p.add_argument("--critic_horizon", type=int, default=critic_horizon)
    p.add_argument("--action_dim", type=int, default=action_dim)
    p.add_argument("--critic_image_size", type=int, default=64)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--agf_num_steps", type=int, default=None, help="default: the policy's inference step count")
    p.add_argument("--overfit_first_batch", action="store_true")
    p.add_argument("--resume_from")


def _particle_adjoint(p, count_help):
    p.add_argument(
        "--use_particle_adjoint",
        action="store_true",
        help="average each backward-step VJP over --num_perturbation perturbed copies of x_k",
    )
    p.add_argument("--num_perturbation", type=int, default=4)
    p.add_argument("--sharpness_sigma", type=float, default=0.02, help="perturbation stddev on x_k")
    p.add_argument("--particle_count_schedule", choices=("constant", "linear"), default="constant", help=count_help)
    p.add_argument("--particle_count_min", type=int, default=2)
    p.add_argument("--particle_count_max", type=int, default=6)


def _adjoint_options(p):
    p.add_argument(
        "--agf_time_subsample",
        type=int,
        default=1,
        help="stride between exact backward-adjoint steps; >1 approximates the skipped steps",
    )
    p.add_argument(
        "--agf_randomize_time_subsample", action="store_true", help="redraw the --agf_time_subsample phase every batch"
    )
    p.add_argument("--use_sharpness_aware", action="store_true")
    _particle_adjoint(p, "linear: count(t) = min + (max - min) * t")
    p.add_argument(
        "--particle_sigma_schedule",
        choices=("constant", "linear"),
        default="constant",
        help="linear: sigma(t) = min + (max - min) * t; constant uses --sharpness_sigma",
    )
    p.add_argument("--particle_sigma_min", type=float, default=0.0)
    p.add_argument("--particle_sigma_max", type=float, default=0.02)


def _libero_pro(p, *, episodes_per_condition, generated_bddl_root):
    p.add_argument("--base_suite", required=True, choices=LIBERO_SUITES)
    p.add_argument("--conditions", default=PRO_CONDITIONS)
    p.add_argument("--task_ids", default=None, help="comma-separated; default = all tasks in --base_suite")
    p.add_argument("--episodes_per_condition", type=int, default=episodes_per_condition)
    p.add_argument("--libero_pro_root", default=os.environ.get("LIBERO_PRO_ROOT", "LIBERO-PRO"))
    p.add_argument("--generated_bddl_root", default=generated_bddl_root)
    p.add_argument("--image_size", type=int, default=256, help="environment render resolution")
    p.add_argument("--num_steps_wait", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1_000_000)


def _guided_eval(p, *, modes, critic_horizon, critic_image_flag="--qpilots_image_size"):
    p.add_argument("--mode", choices=modes, default="none")
    p.add_argument("--critic_horizon", type=int, default=critic_horizon)
    p.add_argument(critic_image_flag, type=int, default=64)
    p.add_argument("--qgf_weight", type=float, default=0.05)
    p.add_argument("--qgf_max_norm", type=float, default=1.0)
    p.add_argument("--qrerank_k", type=int, default=8)
    p.add_argument("--agf_weight", type=float, default=1.0)


def _agf_weight_schedule(p):
    p.add_argument(
        "--agf_weight_schedule",
        choices=["constant", "sigma_half"],
        default="constant",
        help="sigma_half: w(t) = agf_weight * sigma(t) / 2",
    )
    p.add_argument("--agf_weight_schedule_max", type=float, default=8.0, help="ceiling on sigma(t) / 2")


def _template_ckpts(p, critic_example, gradnet_example):
    p.add_argument("--critic_ckpt_template", default=None, help=f"e.g. '{critic_example}'")
    p.add_argument("--gradnet_ckpt_template", default=None, help=f"e.g. '{gradnet_example}'")


# ---------------------------------------------------------------------------
# Q training
# ---------------------------------------------------------------------------


def critic_libero(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="HuggingFaceVLA/libero")
    p.add_argument(
        "--type",
        dest="data_type",
        choices=("offpolicy", "onpolicy"),
        default="onpolicy",
        help="offpolicy uses LeRobot --data_path; onpolicy uses saved critic_data",
    )
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--image2_key", default="observation.images.image2", help=IMAGE2_KEY_HELP)
    p.add_argument("--task_id", type=int, default=None, help="train on this task's episodes only")
    _critic_training(p, "critic_data/smolvla_clean_action", "critics/qpilots_libero")
    return p.parse_args(argv)


def critic_robocasa(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--group_name", default="atomic")
    p.add_argument("--task_name", required=True, help="RoboCasa task, e.g. CloseFridge (one critic per task)")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128, help="width of the 3-layer Q-head MLP")
    _critic_training(p, "critic_data/pi05_robocasa_clean_action", "critics/pi05_sarsa_robocasa")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# AGF training
# ---------------------------------------------------------------------------


def agf_smolvla(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="HuggingFaceVLA/libero")
    p.add_argument("--data_type", choices=("demo", "rollout"), default="rollout")
    p.add_argument("--critic_data_root", default="critic_data/smolvla_clean_action")
    p.add_argument("--task", action="append", help="exact task language string; repeatable")
    p.add_argument("--libero_suite", help="e.g. libero_goal")
    p.add_argument("--task_id", type=int, default=None, help="0-indexed task id within --libero_suite")
    _guidance_training(
        p,
        pretrained_path="HuggingFaceVLA/smolvla_libero",
        output_dir="guidance/compact",
        n_steps=5000,
        batch_size=128,
        critic_horizon=10,
        action_dim=7,
    )
    _adjoint_options(p)
    p.add_argument(
        "--use_true_scale_target",
        action="store_true",
        help="scale targets to ||a_real - x0|| instead of sigma(t); requires --data_type rollout",
    )
    return p.parse_args(argv)


def agf_pi05_robocasa(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--critic_data_root", default="critic_data/pi05_robocasa_clean_action")
    p.add_argument("--group_name", default="atomic")
    p.add_argument("--task_name", required=True, help="RoboCasa task, matching the per-task critic")
    _guidance_training(
        p,
        pretrained_path="lerobot/pi05_robocasa",
        output_dir="guidance/pi05_robocasa_compact",
        n_steps=5000,
        batch_size=32,
        critic_horizon=50,
        action_dim=12,
    )
    _adjoint_options(p)
    return p.parse_args(argv)


def agf_pi05_libero(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_type", choices=("rollout",), default="rollout")
    p.add_argument("--critic_data_root", default="critic_data/pi05_libero_pro_dynamic")
    p.add_argument("--libero_suite", required=True, help="e.g. libero_goal")
    p.add_argument("--task_id", type=int, required=True, help="0-indexed task id within --libero_suite")
    _guidance_training(
        p,
        pretrained_path="lerobot/pi05_libero_finetuned",
        output_dir="adjoint/pi05_libero_pro",
        n_steps=5000,
        batch_size=32,
        critic_horizon=10,
        action_dim=7,
    )
    # The shared rollout dataloader reads this SmolVLA-only flag; pi0.5 targets use sigma(t) scaling.
    p.set_defaults(use_true_scale_target=False)
    return p.parse_args(argv)


def agf_molmoact2(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--critic_data_root", default="critic_data/molmoact2_libero_pro_dynamic")
    p.add_argument("--libero_suite", required=True, help="e.g. libero_goal")
    p.add_argument("--task_id", type=int, required=True, help="0-indexed task id within --libero_suite")
    _guidance_training(
        p,
        pretrained_path="allenai/MolmoAct2-LIBERO-LeRobot",
        output_dir="adjoint/molmoact2_libero_pro",
        n_steps=1000,
        batch_size=32,
        critic_horizon=10,
        action_dim=7,
        batch_help="real parallel batch (ragged image crops are concatenated)",
    )
    p.set_defaults(save_every=100)
    p.add_argument(
        "--grad_accum_steps", type=int, default=1, help="micro-batches of --batch_size accumulated per optimizer step"
    )
    _particle_adjoint(p, "linear: more particles near noise (t=0 for MolmoAct2), fewer near clean")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_smolvla(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--task_ids", type=int, nargs="*", default=None)
    p.add_argument("--pretrained_path", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--critic_ckpt", default=None)
    p.add_argument("--gradnet_ckpt", default=None, help="GuidanceNet checkpoint, required for --mode agf")
    p.add_argument("--output_dir", default="results/smolvla_minimal_qgf")
    p.add_argument("--device", default="cuda")
    _guided_eval(p, modes=["none", "qgf", "qdps", "qrerank", "agf"], critic_horizon=10)
    _agf_weight_schedule(p)
    p.add_argument("--save_video", action="store_true", help="save an agentview mp4 per episode")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def eval_smolvla_pro(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", default="HuggingFaceVLA/smolvla_libero")
    _libero_pro(p, episodes_per_condition=20, generated_bddl_root="generated_bddl/smolvla_libero_pro_eval")
    _template_ckpts(
        p,
        "critics/smolvla_libero_pro/{suite}_{task}/qpilots_libero_q_step010000.pt",
        "adjoint/smolvla_libero_pro/{suite}_{task}/nomemory_rollout/agf_discrete_guidance.pt",
    )
    _guided_eval(p, modes=["none", "qgf", "qdps", "qrerank", "agf"], critic_horizon=10)
    _agf_weight_schedule(p)
    p.add_argument("--max_steps", type=int, default=None, help="default: TASK_MAX_STEPS[--base_suite]")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--output_dir", default="results/smolvla_libero_pro")
    return p.parse_args(argv)


def eval_pi05_pro(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", default="lerobot/pi05_libero_finetuned")
    _libero_pro(p, episodes_per_condition=20, generated_bddl_root="generated_bddl/pi05_libero_pro_eval")
    _template_ckpts(
        p,
        "critics/pi05_libero_pro/{suite}_{task}/qpilots_libero_q_step010000.pt",
        "adjoint/pi05_libero_pro/{suite}_{task}/nomemory_rollout/agf_discrete_guidance.pt",
    )
    _guided_eval(p, modes=["none", "qgf", "qdps", "qrerank", "agf"], critic_horizon=10)
    p.add_argument("--max_steps", type=int, default=None, help="default: TASK_MAX_STEPS[--base_suite]")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--output_dir", default="results/pi05_libero_pro")
    return p.parse_args(argv)


def eval_pi05_robocasa(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task_names", nargs="*", default=ATOMIC_TASKS)
    p.add_argument("--pretrained_path", default="lerobot/pi05_robocasa")
    _template_ckpts(
        p,
        "critics/pi05_sarsa_robocasa/{task}/qpilots_robocasa_q_step010000.pt",
        "adjoint/pi05_sarsa_robocasa/{task}/nomemory_rollout/agf_discrete_guidance.pt",
    )
    p.add_argument("--output_dir", default="results/pi05_robocasa_guidance")
    p.add_argument("--device", default="cuda")
    _guided_eval(p, modes=["none", "qgf", "qdps", "qrerank", "agf"], critic_horizon=50)
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--n_action_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="pretrain")
    p.add_argument("--obj_registries", nargs="*", default=["lightwheel", "aigen_objs", "objaverse"])
    p.add_argument("--observation_size", type=int, default=256)
    p.add_argument("--save_videos", action="store_true")
    return p.parse_args(argv)


def eval_molmoact2(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, help="e.g. libero_goal")
    p.add_argument("--task_ids", type=int, nargs="*", default=None)
    p.add_argument("--n_episodes", type=int, default=10, help="episodes per perturbation condition")
    p.add_argument("--conditions", default=PRO_CONDITIONS)
    p.add_argument("--libero_pro_root", default=None, help="default: $LIBERO_PRO_ROOT or LIBERO-PRO")
    p.add_argument("--generated_bddl_root", default="generated_bddl/molmoact2_qgf_eval")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--critic_ckpt", default=None)
    p.add_argument("--gradnet_ckpt", default=None, help="GuidanceNet checkpoint, required for --mode agf")
    _guided_eval(
        p, modes=["none", "qgf", "agf", "qrerank", "qdps"], critic_horizon=10, critic_image_flag="--critic_image_size"
    )
    p.add_argument("--qdps_weight", type=float, default=0.05)
    p.add_argument("--qdps_max_norm", type=float, default=1.0)
    p.add_argument("--pretrained_path", default="allenai/MolmoAct2-LIBERO-LeRobot")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_steps_wait", type=int, default=50)
    p.add_argument("--n_action_steps", type=int, default=None, help="default: policy.config.n_action_steps")
    p.add_argument("--max_steps", type=int, default=None, help="default: TASK_MAX_STEPS[suite]")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def eval_critic(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--critic_data_root", default="critic_data/smolvla_clean_action")
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--image2_key", default="observation.images.image2", help=IMAGE2_KEY_HELP)
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--use_target", action="store_true")
    p.add_argument("--output_path", default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def collect_smolvla_libero(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument(
        "--suites",
        default="libero_spatial,libero_object,libero_goal,libero_10",
        help="comma-separated; every task of each suite is collected",
    )
    p.add_argument("--output_root", default="critic_data/smolvla_clean_action")
    p.add_argument("--n_episodes", type=int, default=100, help="episodes per task")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--solver", choices=("euler", "heun"), default="euler")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


LIBERO_PRO_COLLECT_DEFAULTS = {
    # vla: (pretrained_path, generated_bddl_root)
    "smolvla": ("HuggingFaceVLA/smolvla_libero", "generated_bddl/vla_libero_pro_dynamic"),
    "pi05": ("lerobot/pi05_libero_finetuned", "generated_bddl/vla_libero_pro_dynamic"),
    "molmoact2": ("allenai/MolmoAct2-LIBERO-LeRobot", "generated_bddl/molmoact2_libero_pro_dynamic"),
}


def collect_libero_pro(vla, argv=None):
    pretrained_path, generated_bddl_root = LIBERO_PRO_COLLECT_DEFAULTS[vla]
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", default=pretrained_path)
    _libero_pro(p, episodes_per_condition=20, generated_bddl_root=generated_bddl_root)
    p.add_argument("--output_root", default=f"critic_data/{vla}_libero_pro_dynamic")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--resume", action="store_true", help="skip tasks whose data is already complete")
    return p.parse_args(argv)


def collect_pi05_robocasa(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", default="lerobot/pi05_robocasa")
    p.add_argument(
        "--task_names",
        nargs="+",
        choices=ATOMIC_TASKS,
        default=ATOMIC_TASKS,
        metavar="TASK",
        help="subset of the atomic tasks; file indices and seeds stay those of the full list",
    )
    p.add_argument("--output_root", default="critic_data/pi05_robocasa_clean_action")
    p.add_argument("--group_name", default="atomic")
    p.add_argument("--n_episodes", type=int, default=100, help="episodes per task")
    p.add_argument("--n_action_steps", type=int, default=50)
    p.add_argument("--solver", choices=("euler", "heun"), default="euler")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--obj_registries", default="lightwheel,aigen_objs,objaverse", help="comma-separated")
    p.add_argument("--observation_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    return p.parse_args(argv)
