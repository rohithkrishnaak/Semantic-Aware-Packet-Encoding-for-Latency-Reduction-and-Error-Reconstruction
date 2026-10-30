"""Export real artifacts for the browser simulator (viz/).

Writes viz/data.js with the same test frames the UDP demo replays: pixels, int8 tags,
per-tile receiver logits (image logits are their mean, so any admitted subset is exact),
the trained admission scorers, measured results, and Python reference values that
viz/parity_test.js checks the JavaScript engine against.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saep.data import ARTIFACTS, CIFAR_DIR, ROOT, load_cifar  # noqa: E402
from saep.models import (adaptive_exit, exit_bounds, load_admission, load_policy_params,  # noqa: E402
                         load_receiver, order_to_mask, select_diverse)
from saep.protocol import Packet, encode_packet  # noqa: E402
from scripts.simulate import accuracy, select_topk  # noqa: E402
from scripts.train import tile_logits, to_tiles  # noqa: E402

VIZ = ROOT / "viz"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def build_standalone(data_name: str = "data.js", out_name: str = "SAEP_Simulator.html") -> None:
    """viz/simulator.html + engine.js + data file -> one offline file (fonts fall back if offline)."""
    page = (VIZ / "simulator.html").read_text(encoding="utf-8")
    for tag, name in (("data.js", data_name), ("engine.js", "engine.js")):
        page = page.replace(f'<script src="{tag}"></script>',
                            "<script>\n" + (VIZ / name).read_text(encoding="utf-8") + "\n</script>")
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n")
    (VIZ / out_name).write_text(head + page + "\n</body>\n</html>\n", encoding="utf-8")
    print(f"wrote {VIZ / out_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--standalone-only", action="store_true", help="only rebuild the offline HTML file")
    ap.add_argument("--artifacts", type=Path, default=ARTIFACTS,
                    help="model dir; artifacts/v3 writes viz/data_v3.js and viz/SAEP_Simulator_v3.html")
    args = ap.parse_args()
    v3 = args.artifacts.resolve() != ARTIFACTS.resolve()
    data_name, out_name = ("data_v3.js", "SAEP_Simulator_v3.html") if v3 else ("data.js", "SAEP_Simulator.html")
    if args.standalone_only:
        return build_standalone(data_name, out_name)

    scheme, scorers, _, grid = load_admission(args.artifacts / "admission.npz")
    model = load_receiver(args.artifacts / "receiver.pt", grid)
    policy = load_policy_params(args.artifacts / "admission.npz")
    feats = np.load(ARTIFACTS / "features_test.npz")
    sl = slice(args.offset, args.offset + args.frames)
    images, labels = load_cifar(CIFAR_DIR / "test")
    images, labels = images[sl], labels[sl]
    tiles = to_tiles(images, grid)
    tags = scheme.tags(feats["tile_tok"][sl])
    n, p = tiles.shape[:2]
    z = tile_logits(model, tiles, "cpu").astype(np.float32)
    with (CIFAR_DIR / "meta").open("rb") as f:
        names = [s.decode() for s in pickle.load(f, encoding="bytes")[b"fine_label_names"]]

    # Python reference values for the parity test (all packets arrive, offline top-k)
    pos = np.broadcast_to(np.arange(p), (n, p))
    all_avail = np.ones((n, p), bool)
    ctx_full = scorers["ctx"].score(tags)
    ref = {"acc_all16": accuracy(z, labels, all_avail), "curves": {}}
    pol = {"fifo": np.broadcast_to(-np.arange(p, dtype=np.float64), (n, p)),
           "tag": scorers["tag"].score(tags, pos), "ctx": ctx_full}
    for name, s in pol.items():
        ref["curves"][name] = [accuracy(z, labels, select_topk(s, all_avail, k)) for k in range(1, p + 1)]
    drop = np.ones((n, p), bool)
    drop[:, ::3] = False  # deterministic loss pattern: context features over arrived tiles only
    ref["ctx_scores_partial_f0"] = scorers["ctx"].score(tags[:1], drop[:1])[0].tolist()
    ref["ctx_scores_f0"] = ctx_full[0].tolist()
    ref["tag_scores_f0"] = scorers["tag"].score(tags[0], np.arange(p)).tolist()
    if policy:  # v3 selection rules, for the parity test
        full = select_diverse(ctx_full, tags, all_avail, p, grid, policy["lam_adj"], policy["lam_sim"])
        ref["div_order_f0"] = full[0].tolist()
        ref["div_acc_k4"] = accuracy(z, labels, order_to_mask(full[:, :4], p))
        used = adaptive_exit(z, full, policy["exit"][4], *exit_bounds(4, policy["exit_kmax_mult"], p))
        ref["exit_used_k4"] = used.tolist()
        ref["exit_acc_k4"] = accuracy(z, labels, order_to_mask(np.where(np.arange(p)[None] < used[:, None], full, -1), p))
    ref["packet_f3_i5"] = encode_packet(Packet(3, 5, p, scheme.scheme_id, tags[3, 5],
                                               tiles[3, 5].tobytes())).hex()

    measured_budget, cpu = {}, []
    with (ARTIFACTS / "sim_results.csv").open() as f:
        for r in csv.DictReader(f):
            if r["scenario"] == "budget":
                measured_budget.setdefault(r["policy"], []).append(float(r["value"]))
            elif r["scenario"] == "cpu_ms":
                cpu.append([int(r["k"]), float(r["value"])])
    v3_csv = args.artifacts / "sim_results_v3.csv"
    if v3 and v3_csv.exists():  # v3 curve (diversity, k = 1..8) on the same 10k test images
        with v3_csv.open() as f:
            measured_budget["v3"] = [float(r["value"]) for r in csv.DictReader(f)
                                     if r["scenario"] == "budget" and r["variant"] == "+diversity"]

    data = {
        "model": "v3" if v3 else "v2",
        "policy": {"lam_adj": policy["lam_adj"], "lam_sim": policy["lam_sim"],
                   "exit": {str(b): t for b, t in policy["exit"].items()},
                   "kmax_mult": policy["exit_kmax_mult"]} if policy else None,
        "n": n, "grid": grid, "tile": 32 // grid, "tag_dim": tags.shape[-1], "classes": len(names),
        "scheme_id": scheme.scheme_id, "offset": args.offset,
        "class_names": names, "labels": labels.tolist(),
        "images": b64(images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)),  # HWC uint8
        "tags": b64(tags), "logits": b64(z),
        "tag_scorer": {"w": scorers["tag"].w.tolist(), "w_pos": scorers["tag"].w_pos.tolist(),
                       "b": int(scorers["tag"].b)},
        "ctx": {"w1": b64(scorers["ctx"].w1.astype(np.float32)), "b1": b64(scorers["ctx"].b1.astype(np.float32)),
                "w2": b64(scorers["ctx"].w2.astype(np.float32)), "b2": float(scorers["ctx"].b2),
                "in_scale": float(scorers["ctx"].in_scale), "hidden": int(scorers["ctx"].b1.shape[0])},
        "cpu_ms": cpu,
        "measured": {"udp": json.loads((args.artifacts / "demo_results.json").read_text()),
                     "offline_budget": measured_budget, "offline_n": int(len(feats["labels"]))},
        "ref": ref,
    }
    VIZ.mkdir(exist_ok=True)
    (VIZ / data_name).write_text("window.SAEP_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {VIZ / data_name}: {n} frames, all-tiles acc {ref['acc_all16']:.3f}, "
          f"k=4 fifo {ref['curves']['fifo'][3]:.3f} tag {ref['curves']['tag'][3]:.3f} "
          f"ctx {ref['curves']['ctx'][3]:.3f}")
    build_standalone(data_name, out_name)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
