"""Phase 1: run DINOv2 once on the GPU and store per-tile pooled patch tokens.

Outputs artifacts/features_{train,test}.npz with
  tile_tok (N, grid*grid, D) fp16, cls (N, D) fp16, labels (N,)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saep.data import ARTIFACTS, CIFAR_DIR, load_cifar  # noqa: E402

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


@torch.inference_mode()
def extract(model, images: np.ndarray, grid: int, batch: int, device) -> tuple[np.ndarray, np.ndarray]:
    patches = 16  # 224 / 14
    if patches % grid:
        raise ValueError("grid must divide 16")
    p = patches // grid
    toks, clss = [], []
    for start in range(0, len(images), batch):
        x = torch.from_numpy(images[start:start + batch]).view(-1, 3, 32, 32).float().div_(255)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = ((x - MEAN) / STD).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model.forward_features(x)
        pt = out["x_norm_patchtokens"].float()
        b, _, d = pt.shape
        tile = pt.reshape(b, grid, p, grid, p, d).mean(dim=(2, 4)).reshape(b, grid * grid, d)
        toks.append(tile.half().cpu().numpy())
        clss.append(out["x_norm_clstoken"].half().cpu().numpy())
        print(f"\r{min(start + batch, len(images))}/{len(images)}", end="", flush=True)
    print()
    return np.concatenate(toks), np.concatenate(clss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", args.model).to(device).eval()
    ARTIFACTS.mkdir(exist_ok=True)
    for split in ("train", "test"):
        images, labels = load_cifar(CIFAR_DIR / split)
        images, labels = images[:args.max_images], labels[:args.max_images]
        tok, cls = extract(model, images, args.grid, args.batch, device)
        np.savez(ARTIFACTS / f"features_{split}.npz", tile_tok=tok, cls=cls, labels=labels,
                 grid=args.grid)
        print(f"{split}: tile_tok {tok.shape} cls {cls.shape}")


if __name__ == "__main__":
    main()
