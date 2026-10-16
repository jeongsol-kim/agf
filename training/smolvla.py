from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tasks.datasets import LiberoCriticDataAGFDataset


from guidance.networks_libero import save_guidance_checkpoint
from guidance.targets_smolvla import (
    AGFDiscreteOptions,
    ActionSpec,
    _linear_particle_count,
    _linear_particle_sigma,
    _selected_action_region,
    compute_agf_adjoint_targets,
)


def score_action(critic, state, action, horizon):

    image, image2 = state
    return critic.reward(image, image2, action[:, :horizon])


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


def libero_suite_task_strings(suite: str, task_ids: list[int]) -> list[str]:
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[suite]()
    return [task_suite.get_task(i - 1).language for i in task_ids]


def resolve_tasks(args) -> list[str]:
    tasks = list(args.task or [])
    if args.libero_suite:
        if args.task_id is not None:
            tasks += libero_suite_task_strings(args.libero_suite, [args.task_id + 1])
    elif args.task_id is not None:
        raise ValueError("--task_id requires --libero_suite")
    return tasks


def _valid_state_indices(start: int, end: int, action_horizon: int) -> list[int]:

    if end - start < action_horizon:
        return []
    return list(range(start, end - action_horizon + 1))


class LiberoAGFObservationDataset(Dataset):

    def __init__(self, repo_id: str, tasks: Sequence[str], action_horizon: int):
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.action_horizon = action_horizon
        self.meta = LeRobotDatasetMetadata(repo_id)
        self.dataset = LeRobotDataset(repo_id)

        task_set = set(tasks)
        if not task_set:
            print(f"no --task/--libero_suite given; loading all valid states of {repo_id}")

        self.indices: list[int] = []
        self.task_texts: list[str] = []

        for episode in self.meta.episodes:
            ep_tasks = list(episode.get("tasks") or [])
            if task_set and not task_set.intersection(ep_tasks):
                continue
            if not ep_tasks:
                raise ValueError(f"episode has no task language in metadata: {episode}")

            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            for idx in _valid_state_indices(start, end, action_horizon):
                self.indices.append(idx)
                self.task_texts.append(ep_tasks[0])

        if not self.indices:
            raise ValueError(f"no valid states found for tasks={list(tasks)!r} in {repo_id}")

        print(f"loaded {len(self.indices)} valid AGF states from {len(self.meta.episodes)} episodes " f"of {repo_id}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        sample = self.dataset[self.indices[i]]
        out = {
            k: sample[k]
            for k in sample.keys()
            if k.startswith("observation.") and not k.startswith("observation.language.")
        }
        out["task"] = self.task_texts[i]
        return out


def _pad_right(x: torch.Tensor, target_len: int, value) -> torch.Tensor:
    pad = target_len - x.shape[1]
    if pad <= 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[1] = pad
    return torch.cat([x, x.new_full(pad_shape, value)], dim=1)


_LANGUAGE_KEYS_TO_PAD = {"observation.language.tokens": 0, "observation.language.attention_mask": 0}


class LiberoAGFTrueScaleDataset(LiberoCriticDataAGFDataset):

    def __getitem__(self, i):
        from lerobot.utils.constants import ACTION

        record = self.records[i]
        out = dict(record["batch"])

        action = torch.as_tensor(record[self.action_key]).float()
        if action.ndim == 3:
            if action.shape[0] != 1:
                raise ValueError(f"Expected action batch dimension 1, got {tuple(action.shape)}")
            action = action[0]
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action shape (T,{self.action_dim}), got {tuple(action.shape)}")
        out[ACTION] = action
        return out


def agf_collate_fn(rows):
    keys = rows[0].keys()
    batch = {}
    for key in keys:
        values = [row[key] for row in rows]
        if torch.is_tensor(values[0]):
            if values[0].ndim > 0 and values[0].shape[0] == 1:
                if key in _LANGUAGE_KEYS_TO_PAD and len({v.shape[1] for v in values}) > 1:
                    max_len = max(v.shape[1] for v in values)
                    values = [_pad_right(v, max_len, _LANGUAGE_KEYS_TO_PAD[key]) for v in values]
                try:
                    batch[key] = torch.cat(values, dim=0)
                except RuntimeError:
                    shapes = [tuple(v.shape) for v in values]
                    raise RuntimeError(f"agf_collate_fn: shape mismatch for key {key!r}, per-sample shapes: {shapes}")
            else:
                batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values
    return batch


def add_language_tokens(batch, policy, device: str):
    if "observation.language.tokens" in batch and "observation.language.attention_mask" in batch:
        batch["observation.language.tokens"] = batch["observation.language.tokens"].to(device)
        batch["observation.language.attention_mask"] = batch["observation.language.attention_mask"].to(device).bool()
        return batch

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    padding = "max_length" if policy.config.pad_language_to == "max_length" else True
    encoded = tokenizer(
        batch["task"],
        padding=padding,
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    batch["observation.language.tokens"] = encoded["input_ids"].to(device)
    batch["observation.language.attention_mask"] = encoded["attention_mask"].to(device).bool()
    return batch


def move_batch_to_device(batch, device: str):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_agf_dataloader(args):
    if args.data_type == "demo":
        dataset = LiberoAGFObservationDataset(
            repo_id=args.repo_id, tasks=resolve_tasks(args), action_horizon=args.critic_horizon
        )
    elif args.data_type == "rollout":
        if args.libero_suite is None or args.task_id is None:
            raise ValueError("--data_type rollout requires --libero_suite and --task_id")
        dataset_cls = LiberoAGFTrueScaleDataset if args.use_true_scale_target else LiberoCriticDataAGFDataset
        dataset_kwargs = dict(
            critic_data_root=args.critic_data_root,
            suite=args.libero_suite,
            task_id=args.task_id,
            action_horizon=args.critic_horizon,
            action_dim=args.action_dim,
            image_size=args.critic_image_size,
        )
        if args.use_true_scale_target:

            dataset_kwargs["action_key"] = "a0_raw"
        dataset = dataset_cls(**dataset_kwargs)
    else:
        raise ValueError(f"unknown data_type {args.data_type!r}")

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


def next_agf_batch(iterator, loader, policy, device: str):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    batch = move_batch_to_device(batch, device)
    batch = add_language_tokens(batch, policy, device)
    return batch, iterator


def agf_targets(policy, critic, guidance_model, batch, args, first_batch: bool = False):
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

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

    real_action = None
    if args.use_true_scale_target:
        from lerobot.utils.constants import ACTION

        if ACTION not in batch:
            raise ValueError(
                "--use_true_scale_target requires a real action in the batch "
                "(--data_type rollout with a dataset that includes it, e.g. "
                "LiberoAGFTrueScaleDataset)"
            )
        real_action = batch[ACTION]

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
        real_action=real_action,
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
