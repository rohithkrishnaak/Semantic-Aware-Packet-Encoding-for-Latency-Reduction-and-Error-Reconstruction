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


def grid_adjacency(grid: int) -> np.ndarray:
    """(P,P) 0/1 matrix of 4-neighbours on the tile grid."""
    p = grid * grid
    a = np.zeros((p, p), np.float32)
    for i in range(p):
        r, c = divmod(i, grid)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= r + dr < grid and 0 <= c + dc < grid:
                a[i, (r + dr) * grid + c + dc] = 1
    return a


def select_diverse(score: np.ndarray, tags: np.ndarray, available: np.ndarray, k: int, grid: int,
                   lam_adj: float, lam_sim: float) -> np.ndarray:
    """Greedy top-k with a diversity penalty (maximal marginal relevance).

    Each next pick maximises score - lam_adj * (#already-picked 4-neighbours)
    - lam_sim * (max cosine similarity of its tag to an already-picked tag).
    Returns (N, k) tile indices in pick order, -1 where fewer than k tiles were available.
    """
    n, p = score.shape
    adj = grid_adjacency(grid)
    t = tags.astype(np.float32)
    t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-6)
    sim = t @ t.transpose(0, 2, 1)  # (N,P,P)
    picked = np.zeros((n, p), bool)
    near = np.zeros((n, p), np.float32)
    maxsim = np.full((n, p), -1.0, np.float32)
    order = np.full((n, k), -1, np.int64)
    rows = np.arange(n)
    for j in range(k):
        pen = score - lam_adj * near - lam_sim * np.maximum(maxsim, 0)
        pen = np.where(available & ~picked, pen, -np.inf)
        best = pen.argmax(1)
        ok = np.isfinite(pen[rows, best])
        order[ok, j] = best[ok]
        picked[rows[ok], best[ok]] = True
        near[ok] += adj[best[ok]]
        maxsim[ok] = np.maximum(maxsim[ok], sim[rows[ok], best[ok]])
    return order


def order_to_mask(order: np.ndarray, p: int) -> np.ndarray:
    mask = np.zeros((len(order), p), bool)
    rows, cols = np.nonzero(order >= 0)
    mask[rows, order[rows, cols]] = True
    return mask


def exit_confidence(mean_logits: np.ndarray) -> np.ndarray:
    """Softmax margin between the top two classes; the early-exit stopping signal."""
    m = mean_logits - mean_logits.max(-1, keepdims=True)
    pr = np.exp(m) / np.exp(m).sum(-1, keepdims=True)
    top2 = np.partition(pr, -2, axis=-1)[..., -2:]
    return top2[..., 1] - top2[..., 0]


def exit_bounds(budget: int, kmax_mult: int, p: int) -> tuple[int, int]:
    """Early exit always processes budget-1 tiles and may extend to kmax_mult * budget."""
    return max(1, budget - 1), min(p, kmax_mult * budget)


def adaptive_exit(z: np.ndarray, order: np.ndarray, theta: float, kmin: int, kmax: int) -> np.ndarray:
    """Early exit: process tiles in `order`, stop once the softmax margin (top1 - top2) >= theta.

    The receiver's image logits are the running mean of per-tile logits, so each extra tile
    costs one tile encode and no recomputation. Returns the number of tiles used per frame
    (0 if nothing arrived). z: (N,P,C) per-tile logits, order: (N,>=kmax) indices, -1 = none.
    """
    n = len(z)
    used = np.zeros(n, np.int64)
    done = np.zeros(n, bool)
    acc = np.zeros((n, z.shape[-1]), np.float64)
    rows = np.arange(n)
    for j in range(kmax):
        has = order[:, j] >= 0
        live = has & ~done
        acc[live] += z[rows[live], order[live, j]]
        used[live] = j + 1
        if j + 1 >= kmin:
            conf = exit_confidence(acc[live] / (j + 1))
            done[np.flatnonzero(live)[conf >= theta]] = True
        done |= ~has
    return used


def load_policy_params(path) -> dict:
    """Optional v3 selection parameters (diversity weights, early-exit thresholds per budget)."""
    a = np.load(path)
    out = {}
    if "div_lam_adj" in a.files:
        out["lam_adj"], out["lam_sim"] = float(a["div_lam_adj"]), float(a["div_lam_sim"])
    if "exit_budgets" in a.files:
        out["exit"] = {int(b): float(t) for b, t in zip(a["exit_budgets"], a["exit_theta"])}
        out["exit_kmax_mult"] = int(a["exit_kmax_mult"])
    return out


def _masked_mean(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-frame mean of y over available tiles, broadcast back to (B,P)."""
    return ((y * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)).expand_as(y)
