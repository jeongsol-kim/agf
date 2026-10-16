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

from tasks.datasets import LiberoCriticDataAGFDataset


from guidance.networks_libero import save_guidance_checkpoint
from guidance.targets_molmoact2 import _selected_action_region


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


MOLMOACT2_PAD_TOKEN_ID = 151643


def _pad_right(x: torch.Tensor, target_len: int, value) -> torch.Tensor:
    pad = target_len - x.shape[1]
    if pad <= 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[1] = pad
    return torch.cat([x, x.new_full(pad_shape, value)], dim=1)


def agf_collate_fn(rows):

    if len(rows) == 1:
        return rows[0]

    max_len = max(r["input_ids"].shape[1] for r in rows)
    out = {
        "input_ids": torch.cat([_pad_right(r["input_ids"], max_len, MOLMOACT2_PAD_TOKEN_ID) for r in rows], dim=0),
        "attention_mask": torch.cat([_pad_right(r["attention_mask"], max_len, 0) for r in rows], dim=0),
        "token_type_ids": torch.cat([_pad_right(r["token_type_ids"], max_len, False) for r in rows], dim=0),
    }
    for key in ("pixel_values", "image_token_pooling", "image_grids", "image_num_crops"):
        out[key] = torch.cat([r[key] for r in rows], dim=0)
    for key in ("action_dim_is_pad", "observation.images.image", "observation.images.wrist_image", "observation.state"):
        out[key] = torch.cat([r[key] for r in rows], dim=0)
    out["task"] = [t for r in rows for t in r["task"]]
    return out


def move_batch_to_device(batch, device: str):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_agf_dataloader(args):
    dataset = LiberoCriticDataAGFDataset(
        critic_data_root=args.critic_data_root,
        suite=args.libero_suite,
        task_id=args.task_id,
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
    return move_batch_to_device(batch, device), iterator


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
