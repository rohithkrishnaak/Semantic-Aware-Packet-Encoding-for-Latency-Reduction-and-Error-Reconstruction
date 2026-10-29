"""Receiver task model (cost linear in accepted tiles) and the int8 linear admission scorer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class TileReceiver(nn.Module):
    """Encodes each accepted tile independently, then averages per-tile class logits.

    Because image logits are the mean of per-tile logits, compute is exactly
    proportional to the number of accepted tiles, and each tile's contribution
    to the decision is additive (which gives a clean per-tile utility target).
    """

    def __init__(self, n_tiles: int = 16, tile: int = 8, n_classes: int = 100, width: int = 96):
        super().__init__()
        self.tile = tile
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.pos = nn.Embedding(n_tiles, 2 * width)
        self.head = nn.Sequential(nn.Linear(2 * width, 256), nn.ReLU(), nn.Linear(256, n_classes))

    def tile_logits(self, tiles: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """tiles: (M,t,t,3) uint8 HWC, pos: (M,) -> (M, n_classes)."""
        x = tiles.permute(0, 3, 1, 2).float().div(255.0).sub(0.5).div(0.25)
        return self.head(self.encoder(x) + self.pos(pos))

    def forward(self, tiles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tiles: (B,P,t,t,3), mask: (B,P) bool. Only masked-in tiles are encoded."""
        b, p = mask.shape
        pos = torch.arange(p, device=tiles.device).expand(b, p)
        out = torch.zeros(b, p, self.head[-1].out_features, device=tiles.device)
        out[mask] = self.tile_logits(tiles[mask], pos[mask])
        return out.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


@dataclass
class TagScheme:
    """Projection DINOv2 tile token -> int8 tag, with a global fixed scale."""

    scheme_id: int
    mean: np.ndarray  # (D,)
    proj: np.ndarray  # (D, tag_dim)
    scale: float

    def tags(self, tok: np.ndarray) -> np.ndarray:
        z = (tok.astype(np.float32) - self.mean) @ self.proj / self.scale
        return np.clip(np.rint(z), -127, 127).astype(np.int8)


@dataclass
class IntScorer:
    """score = sum(w_q * tag) + w_pos[pos] + b, all integer: implementable in XDP/eBPF."""

    w: np.ndarray      # (tag_dim,) int8
    w_pos: np.ndarray  # (P,) int32 (zeros for the tag-only variant)
    b: int

    def score(self, tag: np.ndarray, pos: np.ndarray | int) -> np.ndarray:
        return tag.astype(np.int32) @ self.w.astype(np.int32) + self.w_pos[pos] + self.b

    @classmethod
    def fit(cls, tags: np.ndarray, pos: np.ndarray, util: np.ndarray, n_pos: int,
            use_pos: bool, l2: float = 1.0) -> "IntScorer":
        """Ridge regression of tile utility on tag (+ position one-hot), then int quantize."""
        x = tags.reshape(-1, tags.shape[-1]).astype(np.float64)
        if use_pos:
            x = np.hstack([x, np.eye(n_pos)[pos.reshape(-1)]])
        x = np.hstack([x, np.ones((len(x), 1))])
        y = util.reshape(-1).astype(np.float64)
        coef = np.linalg.solve(x.T @ x + l2 * np.eye(x.shape[1]), x.T @ y)
        d = tags.shape[-1]
        q = 127.0 / max(np.abs(coef[:d]).max(), 1e-12)
        w = np.rint(coef[:d] * q).astype(np.int8)
        w_pos = np.rint(coef[d:d + n_pos] * q).astype(np.int32) if use_pos else np.zeros(n_pos, np.int32)
        return cls(w, w_pos, int(np.rint(coef[-1] * q)))


def load_admission(path) -> tuple[TagScheme, dict, np.ndarray, int]:
    """Load (tag scheme, {"tag","tagpos": IntScorer, "ctx": ContextScorer}, position prior, grid)."""
    a = np.load(path)
    scheme = TagScheme(int(a["scheme_id"]), a["tag_mean"], a["tag_proj"], float(a["tag_scale"]))
    scorers = {k: IntScorer(a[f"{k}_w"], a[f"{k}_w_pos"], int(a[f"{k}_b"])) for k in ("tag", "tagpos")}
    scorers["ctx"] = ContextScorer(a["ctx_w1"], a["ctx_b1"], a["ctx_w2"], float(a["ctx_b2"]))
    return scheme, scorers, a["pos_prior"], int(a["grid"])


def load_receiver(path, grid: int, device="cpu") -> TileReceiver:
    model = TileReceiver(n_tiles=grid * grid, tile=32 // grid)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@dataclass
class ContextScorer:
    """User-space rescoring stage: tiny MLP over [tag, tag - mean tag of the frame's arrived packets].

    Needs the whole (arrived part of the) frame, so it runs after buffering, not in XDP.
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    in_scale: float = 30.0

    @staticmethod
    def features(tags: np.ndarray, available: np.ndarray | None = None) -> np.ndarray:
        t = tags.astype(np.float32)
        m = np.ones(t.shape[:-1], bool) if available is None else available
        mean = (t * m[..., None]).sum(-2, keepdims=True) / np.maximum(m.sum(-1)[..., None, None], 1)
        return np.concatenate([t, t - mean], -1)

    def score(self, tags: np.ndarray, available: np.ndarray | None = None) -> np.ndarray:
        x = self.features(tags, available) / self.in_scale
        return np.maximum(x @ self.w1 + self.b1, 0) @ self.w2 + self.b2

    @classmethod
    def fit(cls, tags: np.ndarray, util: np.ndarray, epochs: int = 15, hidden: int = 128,
            device="cpu", seed: int = 0) -> "ContextScorer":
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        d = tags.shape[-1]
        net = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(device)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        y = torch.tensor(util, device=device, dtype=torch.float32)
        for _ in range(epochs):
            # random packet loss during training so the frame-context features stay valid under loss
            avail = rng.random(util.shape) >= rng.uniform(0, 0.6, (len(util), 1))
            avail[np.arange(len(util)), rng.integers(0, util.shape[1], len(util))] = True
            x = torch.tensor(cls.features(tags, avail) / cls.in_scale, device=device, dtype=torch.float32)
            mask = torch.tensor(avail, device=device)
            xs, ys = x[mask], y[mask] - _masked_mean(y, mask)[mask]
            perm = torch.randperm(len(xs), device=device)
            for i in range(0, len(xs), 4096):
                j = perm[i:i + 4096]
                loss = ((net(xs[j])[:, 0] - ys[j]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        l1, l2 = net[0], net[2]
        return cls(l1.weight.detach().cpu().numpy().T, l1.bias.detach().cpu().numpy(),
                   l2.weight.detach().cpu().numpy()[0], float(l2.bias.item()))


def _masked_mean(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-frame mean of y over available tiles, broadcast back to (B,P)."""
    return ((y * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)).expand_as(y)
