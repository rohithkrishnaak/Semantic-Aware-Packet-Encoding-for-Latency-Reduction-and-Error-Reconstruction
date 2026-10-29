"""CIFAR-100 loading (Toronto python format or the Kaggle `train` pickle) and fixed splits."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
CIFAR_DIR = DATA / "cifar-100-python"


def load_cifar(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,3072) uint8 channel-planar images and (N,) fine labels."""
    with Path(path).open("rb") as f:
        record = pickle.load(f, encoding="bytes")
    images = np.asarray(record.get(b"data", record.get("data")), np.uint8)
    labels = np.asarray(record.get(b"fine_labels", record.get("fine_labels")), np.int64)
    if images.ndim != 2 or images.shape[1] != 3072 or len(labels) != len(images):
        raise ValueError(f"unexpected CIFAR layout in {path}")
    return images, labels


def splits(n_val: int = 5000, seed: int = 0) -> dict[str, list[int]]:
    """Stratified train/val split of the training file, cached in data/splits.json."""
    path = DATA / "splits.json"
    if path.exists():
        return json.loads(path.read_text())
    _, labels = load_cifar(CIFAR_DIR / "train")
    rng = np.random.default_rng(seed)
    per_class = n_val // 100
    val = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in range(100)])
    train = np.setdiff1d(np.arange(len(labels)), val)
    out = {"seed": seed, "train": sorted(train.tolist()), "val": sorted(val.tolist())}
    path.write_text(json.dumps(out))
    return out
