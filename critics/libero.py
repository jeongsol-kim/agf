import os
import json
import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Config:

    action_horizon: int = 10
    action_dim: int = 7

    ensemble_size: int = 10
    cnn_channels: tuple = (32, 64, 128, 256)
    feature_dim: int = 256
    hidden_dims: tuple = (128, 128, 128)
    layer_norm: bool = True

    gamma: float = 0.99
    rho: float = 0.5
    tau: float = 0.005

    batch_size: int = 256
    lr: float = 3e-4
    warmup_steps: int = 500
    decay_steps: int = 300_000
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0

    n_steps: int = 500_000
    log_every: int = 10
    save_every: int = 1000
    num_workers: int = 4


def make_mlp(in_dim, hidden_dims, out_dim, layer_norm=True):
    layers = []
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.SiLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class ImageEncoder(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        c1, c2, c3, c4 = cfg.cnn_channels

        self.net = nn.Sequential(
            nn.Conv2d(6, c1, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim) if cfg.layer_norm else nn.Identity(),
            nn.SiLU(),
        )

    def forward(self, image, image2):
        x = torch.cat([image, image2], dim=1)
        return self.net(x)


class SingleQ(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ImageEncoder(cfg)
        action_flat_dim = cfg.action_horizon * cfg.action_dim
        self.q_head = make_mlp(
            in_dim=cfg.feature_dim + action_flat_dim, hidden_dims=cfg.hidden_dims, out_dim=1, layer_norm=cfg.layer_norm
        )

    def forward(self, image, image2, action):
        z = self.encoder(image, image2)
        a = action.flatten(start_dim=1)
        q = self.q_head(torch.cat([z, a], dim=-1))
        return q.squeeze(-1)


class QEnsemble(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.qs = nn.ModuleList([SingleQ(cfg) for _ in range(cfg.ensemble_size)])

    def forward(self, image, image2, action):
        return torch.stack([q(image, image2, action) for q in self.qs], dim=-1)

    def pessimistic(self, image, image2, action):
        q = self.forward(image, image2, action)
        return q.mean(dim=-1) - self.cfg.rho * q.std(dim=-1, unbiased=False)


class QPILOTSLiberoCritic(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.q = QEnsemble(cfg)
        self.q_target = copy.deepcopy(self.q)

        for p in self.q_target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self):
        for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
            pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def loss(self, batch):

        g = batch["image"]
        w = batch["image2"]
        a = batch["action"]
        r = batch["reward"]

        g2 = batch["next_image"]
        w2 = batch["next_image2"]
        a2 = batch["next_action"]

        valid = batch["valid"]

        with torch.no_grad():
            q_next = self.q_target.pessimistic(g2, w2, a2)
            discount = self.cfg.gamma ** batch["executed_steps"]
            target = r + discount * valid * q_next

        q_pred = self.q(g, w, a)
        loss = F.mse_loss(q_pred, target.unsqueeze(-1).expand_as(q_pred))

        with torch.no_grad():
            q_mean = q_pred.mean(dim=-1)

            terminal_mask = valid < 0.5
            nonterminal_mask = valid > 0.5

            q_terminal_mean = (
                q_mean[terminal_mask].mean()
                if terminal_mask.any()
                else torch.tensor(float("nan"), device=q_mean.device)
            )
            q_nonterminal_mean = (
                q_mean[nonterminal_mask].mean()
                if nonterminal_mask.any()
                else torch.tensor(float("nan"), device=q_mean.device)
            )

            info = {
                "q_loss": float(loss.item()),
                "td_abs": float((q_mean - target).abs().mean().item()),
                "reward_mean": float(r.mean().item()),
                "reward_max": float(r.max().item()),
                "done_mean": float((1.0 - valid).mean().item()),
                "q_mean": float(q_mean.mean().item()),
                "target_mean": float(target.mean().item()),
                "q_terminal_mean": float(q_terminal_mean.item()),
                "q_nonterminal_mean": float(q_nonterminal_mean.item()),
            }

        return loss, info

    def reward(self, image, image2, action, use_target=True):

        net = self.q_target if use_target else self.q
        return net.pessimistic(image, image2, action)


def set_lr(optimizer, step, cfg: Config):

    if step < cfg.warmup_steps:
        scale = (step + 1) / cfg.warmup_steps
    else:
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.decay_steps - cfg.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    lr = cfg.lr * scale
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


_LOG_KEYS = [
    "q_loss",
    "td_abs",
    "reward_mean",
    "reward_max",
    "done_mean",
    "q_mean",
    "target_mean",
    "q_terminal_mean",
    "q_nonterminal_mean",
    "grad_norm",
    "lr",
]


def save_log(log: dict, output_dir: str):

    with open(os.path.join(output_dir, "train_log.json"), "w") as f:
        json.dump(log, f)


def plot_log(log: dict, output_dir: str):

    keys = [k for k in _LOG_KEYS if len(log[k]) > 0]
    if not keys:
        return

    ncols = 3
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, k in zip(axes, keys):
        ax.plot(log[k])
        ax.set_title(k)
        ax.set_xlabel("logged step")
        ax.grid(alpha=0.3)

    for ax in axes[len(keys) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "train_log.png"), dpi=120)
    plt.close(fig)


def load_critic(ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["cfg"])

    critic = QPILOTSLiberoCritic(cfg).to(device)
    critic.q.load_state_dict(ckpt["q"])
    critic.q_target.load_state_dict(ckpt["q_target"])
    critic.eval()
    return critic
