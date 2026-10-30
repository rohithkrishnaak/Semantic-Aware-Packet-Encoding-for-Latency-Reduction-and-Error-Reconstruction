"""Test-set ablation for SAEP v3 (run after scripts/train_v3.py).

Every row adds one change on top of the previous one, so each improvement is measured
separately. All tuning happened on the validation split; the 10k test images are used once here.

  v2           PCA tags, MSE context MLP, v2 receiver, top-k               (the current project)
  +tags+rank   learned 32 B int8 projection + listwise ranking loss       (items 5 + 2)
  +receiver    receiver fine-tuned on scorer-admitted tiles                (item 1)
  +diversity   greedy selection penalising adjacent / similar tiles       (item 3)
  +early exit  per-frame k, same *average* tiles/frame as the budget      (item 4)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saep.data import ARTIFACTS, CIFAR_DIR, load_cifar  # noqa: E402
from saep.models import (ContextScorer, TagScheme, adaptive_exit, exit_bounds, load_admission,  # noqa: E402
                         load_policy_params, load_receiver, order_to_mask, select_diverse)
from scripts.simulate import accuracy, select_topk  # noqa: E402
from scripts.train import tile_logits, tile_utility, to_tiles  # noqa: E402

V3 = ARTIFACTS / "v3"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    feats = np.load(ARTIFACTS / "features_test.npz")
    n = len(feats["labels"])
    images, labels = load_cifar(CIFAR_DIR / "test")
    images, labels = images[:n], labels[:n]
    scheme2, sc2, _, grid = load_admission(ARTIFACTS / "admission.npz")
    scheme3, sc3, _, _ = load_admission(V3 / "admission.npz")
    pol = load_policy_params(V3 / "admission.npz")
    a = np.load(V3 / "admission_stageA.npz")
    scheme_a = TagScheme(int(a["scheme_id"]), a["tag_mean"], a["tag_proj"], float(a["tag_scale"]))
    ctx_a = ContextScorer(a["ctx_w1"], a["ctx_b1"], a["ctx_w2"], float(a["ctx_b2"]))
    p = grid * grid
    tiles = to_tiles(images, grid)
    tok = feats["tile_tok"]
    z2 = tile_logits(load_receiver(ARTIFACTS / "receiver.pt", grid, device), tiles, device)
    z3 = tile_logits(load_receiver(V3 / "receiver.pt", grid, device), tiles, device)
    t2, ta, t3 = scheme2.tags(tok), scheme_a.tags(tok), scheme3.tags(tok)
    lam = (pol["lam_adj"], pol["lam_sim"])
    rows = []

    def ladder(avail: np.ndarray, budgets, tag: str, loss: float):
        out = {}
        s2 = sc2["ctx"].score(t2, avail); sa = ctx_a.score(ta, avail); s3 = sc3["ctx"].score(t3, avail)
        full = select_diverse(s3, t3, avail, p, grid, *lam)
        for b in budgets:
            r = {
                "v2": accuracy(z2, labels, select_topk(s2, avail, b)),
                "+tags+rank": accuracy(z2, labels, select_topk(sa, avail, b)),
                "+receiver": accuracy(z3, labels, select_topk(s3, avail, b)),
                "+diversity": accuracy(z3, labels, order_to_mask(full[:, :b], p)),
            }
            if b in pol["exit"]:
                used = adaptive_exit(z3, full, pol["exit"][b], *exit_bounds(b, pol["exit_kmax_mult"], p))
                mask = order_to_mask(np.where(np.arange(p)[None] < used[:, None], full, -1), p)
                r["+early exit"] = accuracy(z3, labels, mask)
                r["avg tiles"] = float(used[avail.any(1)].mean())
            out[b] = r
            rows.extend((tag, b, loss, k, v) for k, v in r.items())
        return out

    all_avail = np.ones((n, p), bool)
    print(f"test images: {n}.  All 16 tiles: v2 receiver {accuracy(z2, labels, all_avail):.4f}, "
          f"v3 receiver {accuracy(z3, labels, all_avail):.4f}")
    print(f"tag-only leakage probe (val): v2 {float(np.load(ARTIFACTS / 'admission.npz')['tag_only_val']):.3f}, "
          f"v3 {float(np.load(V3 / 'admission.npz')['tag_only_val']):.3f}")
    print(f"diversity: lam_adj={lam[0]} lam_sim={lam[1]}; early exit: at least k-1 tiles, at most {pol['exit_kmax_mult']}x k, stop on softmax margin\n")

    cols = ["v2", "+tags+rank", "+receiver", "+diversity", "+early exit"]
    print("[A] receiver budget k (no loss)")
    print("k   " + "".join(f"{c:>13}" for c in cols) + "   avg tiles")
    res = ladder(all_avail, range(1, 9), "budget", 0.0)
    for k, r in res.items():
        print(f"{k:<3} " + "".join(f"{r[c]:>13.4f}" if c in r else f"{'':>13}" for c in cols)
              + (f"   {r['avg tiles']:.2f}" if "avg tiles" in r else ""))

    print("\n[B] iid network loss p, then budget k=4")
    for loss in (0.1, 0.3, 0.5):
        avail = rng.random((n, p)) >= loss
        r = ladder(avail, [4], "loss", loss)[4]
        print(f"p={loss:.1f} " + "  ".join(f"{c}={r[c]:.4f}" for c in cols) + f"  avg tiles={r['avg tiles']:.2f}")

    print("\n[C] int8 XDP-stage scorer alone, k=4 (exact top-k)")
    pos = np.broadcast_to(np.arange(p), (n, p))
    print(f"  v2 tags: {accuracy(z2, labels, select_topk(sc2['tag'].score(t2, pos), all_avail, 4)):.4f}   "
          f"v3 tags (+v3 receiver): {accuracy(z3, labels, select_topk(sc3['tag'].score(t3, pos), all_avail, 4)):.4f}")
    oracle = tile_utility(z3, labels)
    print(f"  oracle (uses the label), v3 receiver, k=4: {accuracy(z3, labels, select_topk(oracle, all_avail, 4)):.4f}")

    with (V3 / "sim_results_v3.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "k", "loss", "variant", "value"])
        w.writerows(rows)
    print(f"\nwrote {V3 / 'sim_results_v3.csv'}")


if __name__ == "__main__":
    main()
