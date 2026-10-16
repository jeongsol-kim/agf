from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class GuidanceConfig:
    guidance_type: str
    action_dim: int
    critic_horizon: int
    chunk_size: int
    max_action_dim: int
    state_dim: int
    image_feature_dim: int = 256
    ensemble_size: int = 1
    hidden_dims: tuple = (128, 128, 128)
    layer_norm: bool = True
    time_embed_dim: int = 64
    target_scale: float = 1.0
    time_convention: str = "pi05_t_1_to_0_dt_negative"

    def __post_init__(self) -> None:
        if self.guidance_type not in {"qgf_distill", "agf_discrete"}:
            raise ValueError(f"guidance_type must be 'qgf_distill' or 'agf_discrete', got {self.guidance_type!r}")
        for name in (
            "image_feature_dim",
            "ensemble_size",
            "action_dim",
            "critic_horizon",
            "chunk_size",
            "max_action_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.critic_horizon > self.chunk_size:
            raise ValueError(f"critic_horizon={self.critic_horizon} exceeds chunk_size={self.chunk_size}")
        if self.action_dim > self.max_action_dim:
            raise ValueError(f"action_dim={self.action_dim} exceeds max_action_dim={self.max_action_dim}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuidanceConfig":
        required = {"guidance_type", "action_dim", "critic_horizon", "chunk_size", "max_action_dim", "state_dim"}
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"Guidance checkpoint config missing required fields: {missing}")
        fields = cls.__dataclass_fields__
        data = dict(data)
        if "hidden_dims" in data:
            data["hidden_dims"] = tuple(data["hidden_dims"])
        return cls(**{key: data[key] for key in fields if key in data})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _FiLMMLP(nn.Module):

    def __init__(
        self, input_dim: int, cond_dim: int, hidden_dims: tuple, output_dim: int, layer_norm: bool = True
    ) -> None:
        super().__init__()
        self.layer_norm = layer_norm
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.film_generators = nn.ModuleList()
        self.act = nn.SiLU()

        dim = input_dim
        for h in hidden_dims:
            self.linears.append(nn.Linear(dim, h))
            self.norms.append(nn.LayerNorm(h) if layer_norm else nn.Identity())
            self.film_generators.append(nn.Linear(cond_dim, 2 * h))
            dim = h
        self.out = nn.Linear(dim, output_dim)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x
        for linear, norm, film_gen in zip(self.linears, self.norms, self.film_generators):
            h = linear(h)
            h = norm(h)
            gamma, beta = film_gen(cond).chunk(2, dim=-1)
            h = gamma * h + beta
            h = self.act(h)
        return self.out(h)


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"time embedding dimension must be positive, got {dim}")
    half = dim // 2
    if half == 0:
        return t[:, None]
    freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1)))
    emb = torch.cat([torch.sin(t[:, None] * freq[None]), torch.cos(t[:, None] * freq[None])], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, emb.new_zeros(emb.shape[0], 1)], dim=-1)
    return emb


class GuidanceNet(nn.Module):
    def __init__(self, cfg: GuidanceConfig, image_encoders: list[nn.Module]):
        super().__init__()
        if len(image_encoders) < cfg.ensemble_size:
            raise ValueError(
                f"cfg.ensemble_size={cfg.ensemble_size} but only {len(image_encoders)} " f"image_encoders were provided"
            )
        self.cfg = cfg

        self.image_encoders = nn.ModuleList(image_encoders[: cfg.ensemble_size])
        for encoder in self.image_encoders:
            for p in encoder.parameters():
                p.requires_grad_(False)

        region_dim = cfg.critic_horizon * cfg.action_dim
        cond_dim = cfg.image_feature_dim + cfg.state_dim + cfg.time_embed_dim
        self.net = _FiLMMLP(region_dim, cond_dim, cfg.hidden_dims, region_dim, cfg.layer_norm)

    def _format_t(self, t: torch.Tensor | float | int, batch_size: int, device, dtype) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=dtype)
        else:
            t = t.to(device=device, dtype=dtype)

        if t.ndim == 0:
            return t.expand(batch_size)
        if t.ndim == 1 and t.shape[0] == 1:
            return t.expand(batch_size)
        if t.ndim == 1 and t.shape[0] == batch_size:
            return t
        raise ValueError(f"t must be a scalar or shape [B] with B={batch_size}, got {tuple(t.shape)}")

    def _encode_images(self, image: torch.Tensor, image2: torch.Tensor, image3: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = torch.stack([enc((image, image2, image3)) for enc in self.image_encoders], dim=0)
        return feats.mean(dim=0)

    def forward(
        self,
        image: torch.Tensor,
        image2: torch.Tensor,
        image3: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor | float | int,
    ) -> torch.Tensor:

        batch_size = image.shape[0]
        if not (x_t.shape[:1] == image.shape[:1] == image2.shape[:1] == image3.shape[:1] == state.shape[:1]):
            raise ValueError(
                f"image/image2/image3/state/x_t batch sizes differ: "
                f"{image.shape[0]}, {image2.shape[0]}, {image3.shape[0]}, {state.shape[0]}, {x_t.shape[0]}"
            )

        image_feat = self._encode_images(image, image2, image3)
        t_vec = self._format_t(t, batch_size, x_t.device, x_t.dtype)
        action_region = x_t[:, : self.cfg.critic_horizon, : self.cfg.action_dim].reshape(batch_size, -1)
        time_emb = _sinusoidal_time_embedding(t_vec, self.cfg.time_embed_dim)
        cond = torch.cat([image_feat, state.to(image_feat.dtype), time_emb], dim=-1)

        pred_region = self.net(action_region, cond)
        pred_region = pred_region.reshape(batch_size, self.cfg.critic_horizon, self.cfg.action_dim)
        if self.cfg.target_scale != 1.0:
            pred_region = pred_region * float(self.cfg.target_scale)

        out = x_t.new_zeros(x_t.shape)
        out[:, : self.cfg.critic_horizon, : self.cfg.action_dim] = pred_region
        return out


def image_encoders_from_critic(critic: Any, ensemble_size: int = 1, use_target: bool = True) -> list[nn.Module]:

    net = critic.q_target if use_target else critic.q
    if ensemble_size > len(net.qs):
        raise ValueError(f"ensemble_size={ensemble_size} exceeds critic ensemble size {len(net.qs)}")
    return [net.qs[i].encoder for i in range(ensemble_size)]


def save_guidance_checkpoint(
    path: str,
    model: GuidanceNet,
    cfg: GuidanceConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    step: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    cfg = cfg or model.cfg
    if cfg.to_dict() != model.cfg.to_dict():
        raise ValueError("Checkpoint cfg does not match model.cfg")
    payload: dict[str, Any] = {
        "cfg": cfg.to_dict(),
        "model_state": model.state_dict(),
        "step": step,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)


def load_guidance_checkpoint(
    path: str, image_encoders: list[nn.Module], device: str | torch.device = "cpu", load_optimizer: bool = False
) -> tuple[GuidanceNet, GuidanceConfig, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Guidance checkpoint must be a dict, got {type(ckpt).__name__}")
    if "cfg" not in ckpt:
        raise ValueError("Guidance checkpoint missing required field 'cfg'")
    if "model_state" not in ckpt:
        raise ValueError("Guidance checkpoint missing required field 'model_state'")

    cfg = GuidanceConfig.from_dict(ckpt["cfg"])
    model = GuidanceNet(cfg, image_encoders).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    extra = {"step": ckpt.get("step"), "metrics": ckpt.get("metrics", {})}
    if load_optimizer:
        extra["optimizer_state"] = ckpt.get("optimizer_state")
    return model, cfg, extra
