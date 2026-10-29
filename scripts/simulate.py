"""Phase 3: offline simulator on the CIFAR-100 test set.

Question: under a receiver budget (k tiles/frame), uncontrolled network loss, or a
bottleneck link, does a learned tag-only admission policy beat random / FIFO?

Accuracy for any accepted set is exact, not approximate: the receiver's image
logits are the mean of per-tile logits, so we compute per-tile logits once.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saep.data import ARTIFACTS, CIFAR_DIR, load_cifar  # noqa: E402
from saep.models import load_admission, load_receiver  # noqa: E402
from scripts.train import tile_logits, tile_utility, to_tiles  # noqa: E402


def accuracy(z: np.ndarray, labels: np.ndarray, accepted: np.ndarray) -> float:
    s = (z * accepted[..., None]).sum(1)
    ok = (s.argmax(1) == labels) & accepted.any(1)  # empty frame = wrong
    return float(ok.mean())


def select_topk(score: np.ndarray, available: np.ndarray, k: int) -> np.ndarray:
    """Admit the k highest-scoring available tiles of each frame (ties broken by index).

    score may be a callable(available) -> scores, for context-dependent scorers.
    """
    if callable(score):
        score = score(available)
    s = np.where(available, score.astype(np.float64), -np.inf)
    order = np.argsort(-s, axis=1, kind="stable")[:, :k]
    acc = np.zeros_like(available)
    np.put_along_axis(acc, order, True, axis=1)
    return acc & available


def cpu_cost_curve(model, tiles: np.ndarray, reps: int = 200) -> list[tuple[int, float]]:
    """Receiver inference time vs number of accepted tiles (single frame, 1 CPU thread)."""
    torch.set_num_threads(1)
    model = model.cpu()
    x = torch.from_numpy(tiles[:1])
    out = []
    with torch.inference_mode():
        for k in (1, 2, 4, 8, 12, 16):
            mask = torch.zeros(1, tiles.shape[1], dtype=torch.bool)
            mask[0, :k] = True
            for _ in range(10):
                model(x, mask)
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x, mask)
            out.append((k, (time.perf_counter() - t0) / reps * 1e3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scheme, scorers, pos_prior, grid = load_admission(ARTIFACTS / "admission.npz")
    model = load_receiver(ARTIFACTS / "receiver.pt", grid, device)
    feats = np.load(ARTIFACTS / "features_test.npz")
    n = len(feats["labels"])
    images, labels = load_cifar(CIFAR_DIR / "test")
    images, labels = images[:n], labels[:n]
    tiles = to_tiles(images, grid)
    p = grid * grid
    tags = scheme.tags(feats["tile_tok"])
    pos = np.broadcast_to(np.arange(p), (n, p))

    z = tile_logits(model, tiles, device)
    util = tile_utility(z, labels)
    policies = {
        "fifo": np.broadcast_to(-np.arange(p, dtype=np.float64), (n, p)),   # raster arrival order
        "random": rng.random((n, p)),
        "position_prior": np.broadcast_to(pos_prior, (n, p)),               # learned, no tag
        "tag": scorers["tag"].score(tags, pos),                             # learned, tag only
        "tag+pos": scorers["tagpos"].score(tags, pos),
        "tag+ctx_mlp": None,                                                # learned, frame context
        "oracle": util,                                                     # uses the label: upper bound
    }
    policies["tag+ctx_mlp"] = lambda avail: scorers["ctx"].score(tags, avail)
    rows = []
    all_avail = np.ones((n, p), bool)

    print(f"test images: {n}, packets/frame: {p}, tag: {tags.shape[-1]} B int8")
    print(f"full-frame accuracy (no drops): {accuracy(z, labels, all_avail):.4f}")

    print("\n[A] receiver budget k tiles/frame, no network loss")
    ks = list(range(1, p + 1))
    print("k   " + " ".join(f"{name:>14}" for name in policies))
    for k in ks:
        accs = {name: accuracy(z, labels, select_topk(s, all_avail, k)) for name, s in policies.items()}
        rows += [("budget", k, 0.0, name, a) for name, a in accs.items()]
        print(f"{k:<3} " + " ".join(f"{a:>14.4f}" for a in accs.values()))

    print("\n[B] uncontrolled iid network loss p, then receiver budget k")
    for loss in (0.1, 0.3, 0.5):
        avail = rng.random((n, p)) >= loss
        for k in (2, 4, 8):
            accs = {name: accuracy(z, labels, select_topk(s, avail, k)) for name, s in policies.items()}
            rows += [("loss", k, loss, name, a) for name, a in accs.items()]
            print(f"p={loss:.1f} k={k:<2} " + " ".join(f"{nm}={a:.4f}" for nm, a in accs.items()))

    print("\n[C] bottleneck link: only c packets/frame get through (tail drop)")
    print("    sender order: raster (no feedback) vs scorer order (with SAEC feedback)")
    for c in (2, 4, 8):
        raster = select_topk(policies["fifo"], all_avail, c)
        fb = select_topk(policies["tag+ctx_mlp"], all_avail, c)
        a_r, a_f = accuracy(z, labels, raster), accuracy(z, labels, fb)
        rows += [("bottleneck", c, 0.0, "raster_sender", a_r), ("bottleneck", c, 0.0, "feedback_sender", a_f)]
        print(f"c={c:<2} raster={a_r:.4f}  feedback={a_f:.4f}")

    print("\n[D] receiver CPU cost vs accepted tiles (1 thread)")
    for k, ms in cpu_cost_curve(model, tiles):
        print(f"  k={k:<2} {ms:.3f} ms/frame")
        rows.append(("cpu_ms", k, 0.0, "receiver", ms))

    out = ARTIFACTS / "sim_results.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "k", "loss", "policy", "value"])
        w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name in policies:
            ax.plot(ks, [r[4] for r in rows if r[0] == "budget" and r[3] == name], marker="o", ms=3, label=name)
        ax.set_xlabel("admitted tiles per frame (k)")
        ax.set_ylabel("top-1 accuracy (CIFAR-100 test)")
        ax.set_title("Admission policy vs receiver budget")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(ARTIFACTS / "sim_budget.png", dpi=130)
    except ImportError:
        pass
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
