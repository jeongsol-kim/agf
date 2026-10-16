from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tasks.datasets import RoboCasaCriticDataAGFDataset


from guidance.networks_robocasa import save_guidance_checkpoint
from guidance.targets_pi05_robocasa import (
    AGFDiscreteOptions,
    ActionSpec,
    _linear_particle_count,
    _linear_particle_sigma,
    _selected_action_region,
    compute_agf_adjoint_targets,
)


def score_action(critic, state, action, horizon):

    image, image2, image3, agent_pos = state
    return critic.reward((image, image2, image3), agent_pos, action[:, :horizon])


def region_mse(pred, target, spec):
    return F.mse_loss(_selected_action_region(pred, spec), _selected_action_region(target, spec))


def metric_row(step, loss, pred, target, spec, extra=None):
    p = _selected_action_region(pred.detach(), spec).reshape(pred.shape[0], -1)
    y = _selected_action_region(target.detach(), spec).reshape(target.shape[0], -1)
    row = {
        "step": step,
        "loss": float(loss.detach().cpu()),
        "cosine": float(F.cosine_similarity(p, y, dim=-1, eps=1e-8).mean().cpu()),
        "pred_norm": float(p.norm(dim=-1).mean().cpu()),
        "target_norm": float(y.norm(dim=-1).mean().cpu()),
        "norm_ratio": float((p.norm(dim=-1).mean() / (y.norm(dim=-1).mean() + 1e-8)).cpu()),
    }
    if extra:
        row.update(extra)
    print(" | ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    return row


def agf_collate_fn(rows):
    keys = rows[0].keys()
    batch = {}
    for key in keys:
        values = [row[key] for row in rows]
        if torch.is_tensor(values[0]):
            if values[0].ndim > 0 and values[0].shape[0] == 1:
                batch[key] = torch.cat(values, dim=0)
            else:
                batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values
    return batch


def move_batch_to_device(batch, device: str):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_agf_dataloader(args):
    dataset = RoboCasaCriticDataAGFDataset(
        critic_data_root=args.critic_data_root,
        group_name=args.group_name,
        task_name=args.task_name,
        action_horizon=args.critic_horizon,
        action_dim=args.action_dim,
        image_size=args.critic_image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=agf_collate_fn,
    )
    return loader


def next_agf_batch(iterator, loader, device: str):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    batch = move_batch_to_device(batch, device)
    batch["observation.language.attention_mask"] = batch["observation.language.attention_mask"].bool()
    return batch, iterator


def agf_targets(policy, critic, guidance_model, batch, args, first_batch: bool = False):
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    spec = ActionSpec(args.critic_horizon, args.action_dim)
    opts = AGFDiscreteOptions(
        num_steps=args.agf_num_steps,
        time_subsample=args.agf_time_subsample,
        randomize_checkpoint_phase=args.agf_randomize_time_subsample,
        solver="euler",
        max_batch_size=args.batch_size,
        critic_image_size=args.critic_image_size,
    )
    particle_sigma_fn = None
    if args.particle_sigma_schedule == "linear":
        particle_sigma_fn = lambda t: _linear_particle_sigma(  # noqa: E731
            t, args.particle_sigma_min, args.particle_sigma_max
        )
    particle_count_fn = None
    if args.particle_count_schedule == "linear":
        particle_count_fn = lambda t: _linear_particle_count(  # noqa: E731
            t, args.particle_count_min, args.particle_count_max
        )

    return compute_agf_adjoint_targets(
        policy=policy,
        critic=critic,
        guidance_model=guidance_model,
        batch=batch,
        score_action_fn=score_action,
        make_att_2d_masks_fn=make_att_2d_masks,
        spec=spec,
        options=opts,
        guidance_weight=1.0,
        use_guidance_rollout=(guidance_model is not None and not first_batch),
        use_sharpness_aware=args.use_sharpness_aware,
        num_perturbation=args.num_perturbation,
        sharpness_sigma=args.sharpness_sigma,
        use_particle_adjoint=args.use_particle_adjoint,
        particle_sigma_fn=particle_sigma_fn,
        particle_count_fn=particle_count_fn,
    )


def set_trainable(module, trainable: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(trainable)


def save_log_and_plot(log, output_dir):
    with open(os.path.join(output_dir, "guidance_train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    if not log:
        return

    keys = [
        k
        for k in ["loss", "cosine", "norm_ratio", "seed_norm", "adjoint_norm", "adjoint_norm_s0", "grad_norm"]
        if k in log[0]
    ]
    ncols = 3
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, k in zip(axes, keys):
        values = [max(float(row[k]), 1e-12) for row in log]
        ax.plot(values)
        if k in {"loss", "norm_ratio"}:
            ax.set_yscale("log")
            ax.set_title(f"{k} (log scale)")
        else:
            ax.set_title(k)
        ax.set_xlabel("logged step")
        ax.grid(alpha=0.3)
    for ax in axes[len(keys) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "guidance_train_log.png"), dpi=120)
    plt.close(fig)


def save_training_outputs(args, model, cfg, optim, log, default_name):
    save_log_and_plot(log, args.output_dir)

    save_path = os.path.join(args.output_dir, default_name)
    step = log[-1]["step"] if log else 0
    metrics = log[-1] if log else {}
    save_guidance_checkpoint(save_path, model, cfg, optimizer=optim, step=step, metrics=metrics)
    print(f"saved {save_path}")
