"""Phase 2: tag projection, receiver task model, per-tile utility, int8 admission scorers.

Everything here is trained on the CIFAR-100 training file only:
  - receiver model + tag projection: train split (45k)
  - admission scorers: fit on train split, target = per-tile utility centered per frame
    (only the ranking inside a frame matters); evaluated on val
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saep.data import ARTIFACTS, CIFAR_DIR, load_cifar, splits  # noqa: E402
from saep.models import ContextScorer, IntScorer, TagScheme, TileReceiver, load_receiver  # noqa: E402


def to_tiles(images: np.ndarray, grid: int) -> np.ndarray:
    """(N,3072) channel-planar -> (N, grid*grid, t, t, 3) HWC tiles, raster order."""
    t = 32 // grid
    hwc = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(
        hwc.reshape(-1, grid, t, grid, t, 3).transpose(0, 1, 3, 2, 4, 5).reshape(-1, grid * grid, t, t, 3))


def fit_tag_scheme(tok: np.ndarray, tag_dim: int, scheme_id: int, seed: int = 0) -> TagScheme:
    flat = tok.reshape(-1, tok.shape[-1]).astype(np.float32)
    sample = flat[np.random.default_rng(seed).choice(len(flat), min(200_000, len(flat)), replace=False)]
    mean = sample.mean(0)
    _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
    proj = vt[:tag_dim].T.astype(np.float32)
    scale = float(np.percentile(np.abs((sample - mean) @ proj), 99.9) / 127.0)
    return TagScheme(scheme_id, mean, proj, scale)


def random_masks(b: int, p: int, device) -> torch.Tensor:
    k = torch.randint(1, p + 1, (b, 1), device=device)
    rank = torch.rand(b, p, device=device).argsort(1).argsort(1)
    return rank < k


def train_receiver(tiles: np.ndarray, labels: np.ndarray, grid: int, epochs: int, device) -> TileReceiver:
    p, t = grid * grid, 32 // grid
    model = TileReceiver(n_tiles=p, tile=t).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    x_all = torch.from_numpy(tiles).to(device)
    y_all = torch.from_numpy(labels).to(device)
    steps = epochs * ((len(x_all) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_all), device=device)
        total, correct, n = 0.0, 0, 0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            x, y = x_all[idx], y_all[idx]
            flip = torch.rand(len(x), device=device) < 0.5  # horizontal flip of the whole image
            xf = x.view(-1, grid, grid, t, t, 3).flip(dims=(2, 4)).view_as(x)
            x = torch.where(flip.view(-1, 1, 1, 1, 1), xf, x)
            logits = model(x, random_masks(len(x), p, device))
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(x)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(x)
        print(f"epoch {epoch + 1}/{epochs} loss {total / n:.3f} acc(random masks) {correct / n:.3f}")
    return model.eval()


@torch.inference_mode()
def tile_logits(model: TileReceiver, tiles: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    """All per-tile logits: (N, P, C)."""
    out = []
    p = tiles.shape[1]
    for i in range(0, len(tiles), batch):
        x = torch.from_numpy(tiles[i:i + batch]).to(device)
        b = len(x)
        pos = torch.arange(p, device=device).repeat(b)
        out.append(model.tile_logits(x.reshape(b * p, *x.shape[2:]), pos).view(b, p, -1).cpu().numpy())
    return np.concatenate(out)


def tile_utility(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-tile log-probability of the true class: the tile's additive evidence."""
    logp = z - np.logaddexp.reduce(z, axis=-1, keepdims=True)
    return np.take_along_axis(logp, labels[:, None, None], axis=-1)[..., 0]


def linear_probe(x_tr, y_tr, x_te, y_te, device, epochs: int = 100) -> float:
    xt, yt = torch.tensor(x_tr, device=device).float(), torch.tensor(y_tr, device=device)
    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    lin = torch.nn.Linear(xt.shape[1], 100).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-5)
    for _ in range(epochs):
        for i in range(0, len(xt), 4096):
            loss = F.cross_entropy(lin((xt[i:i + 4096] - mu) / sd), yt[i:i + 4096])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = lin((torch.tensor(x_te, device=device).float() - mu) / sd).argmax(1).cpu().numpy()
    return float((pred == y_te).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--scheme-id", type=int, default=1)
    ap.add_argument("--reuse-receiver", action="store_true", help="load artifacts/receiver.pt instead of training")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    feats = np.load(ARTIFACTS / "features_train.npz")
    grid = int(feats["grid"])
    n = len(feats["labels"])
    sp = splits()
    tr = np.array([i for i in sp["train"] if i < n])
    va = np.array([i for i in sp["val"] if i < n])
    images, labels = load_cifar(CIFAR_DIR / "train")
    tiles = to_tiles(images[:n], grid)
    tok = feats["tile_tok"]

    print("fitting tag projection ...")
    scheme = fit_tag_scheme(tok[tr], args.tag_dim, args.scheme_id)
    tags = scheme.tags(tok)

    if args.reuse_receiver:
        model = load_receiver(ARTIFACTS / "receiver.pt", grid, device)
    else:
        print("training receiver model ...")
        t0 = time.time()
        model = train_receiver(tiles[tr], labels[tr], grid, args.epochs, device)
        print(f"  done in {time.time() - t0:.0f}s")

    z_va = tile_logits(model, tiles[va], device)
    util_va = tile_utility(z_va, labels[va])
    print(f"val acc (all tiles): {(z_va.mean(1).argmax(1) == labels[va]).mean():.3f}")

    p = grid * grid
    util_tr = tile_utility(tile_logits(model, tiles[tr], device), labels[tr])
    util_tr_c = util_tr - util_tr.mean(1, keepdims=True)
    pos_tr = np.broadcast_to(np.arange(p), (len(tr), p))
    pos_va = np.broadcast_to(np.arange(p), (len(va), p))
    print("fitting admission scorers ...")
    scorers = {
        "tag": IntScorer.fit(tags[tr], pos_tr, util_tr_c, p, use_pos=False),
        "tagpos": IntScorer.fit(tags[tr], pos_tr, util_tr_c, p, use_pos=True),
    }
    ctx = ContextScorer.fit(tags[tr], util_tr, device=device)
    pos_prior = util_tr_c.mean(0)
    util_va_c = util_va - util_va.mean(1, keepdims=True)
    for name, sc in [*scorers.items(), ("ctx", ctx)]:
        pred = sc.score(tags[va]) if name == "ctx" else sc.score(tags[va], pos_va)
        r = np.corrcoef(pred.ravel(), util_va_c.ravel())[0, 1]
        print(f"scorer {name}: corr(score, centered utility) on val = {r:.3f}")

    print("control baselines (linear probes, train -> val) ...")
    tag_only = linear_probe(tags[tr].astype(np.float32).mean(1), labels[tr],
                            tags[va].astype(np.float32).mean(1), labels[va], device)
    cls_only = linear_probe(feats["cls"][tr], labels[tr], feats["cls"][va], labels[va], device)
    print(f"  tag-only (mean of 16 int8 tags): {tag_only:.3f}   full-image CLS (v1 header): {cls_only:.3f}")

    if not args.reuse_receiver:
        torch.save(model.state_dict(), ARTIFACTS / "receiver.pt")
    np.savez(ARTIFACTS / "admission.npz", grid=grid, scheme_id=scheme.scheme_id, tag_mean=scheme.mean,
             tag_proj=scheme.proj, tag_scale=scheme.scale, pos_prior=pos_prior,
             tag_only_val=tag_only, cls_only_val=cls_only,
             ctx_w1=ctx.w1, ctx_b1=ctx.b1, ctx_w2=ctx.w2, ctx_b2=ctx.b2,
             **{f"{k}_{f}": getattr(s, f) for k, s in scorers.items() for f in ("w", "w_pos", "b")})
    print(f"saved to {ARTIFACTS}")


if __name__ == "__main__":
    main()
