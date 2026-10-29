"""Broadcast CIFAR-100 images with DINOv2 vectors in a custom UDP header.

LEGACY (SAEP v1). Kept for reference; superseded by saep/protocol.py (v2).
See README "Why v2" for the problems with this design.

The receiver must understand the packet format defined by ``PACKET_HEADER``.
Each UDP datagram contains one complete DINOv2 vector in its header and one
fragment of the original 32x32 RGB CIFAR image as its payload.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn.functional as functional


MAGIC = b"SAEP"
PROTOCOL_VERSION = 1
EMBEDDING_FLOAT16 = 1
FLAG_FIRST_FRAGMENT = 1
FLAG_LAST_FRAGMENT = 2

# magic, version, flags, header length, frame id, fragment index,
# fragment count, image length, embedding dimension, embedding dtype,
# payload length, payload CRC32, CIFAR fine label, reserved
PACKET_HEADER = struct.Struct("!4sBBHIHHIHBHIBB")
BASE_HEADER_LENGTH = PACKET_HEADER.size
RGB_IMAGE_LENGTH = 3 * 32 * 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_cifar_training_file(path: Path) -> tuple[torch.Tensor, Sequence[int]]:
    """Load and validate the CIFAR-100 Python-format training file."""
    if not path.is_file():
        raise FileNotFoundError(f"CIFAR training file does not exist: {path}")

    with path.open("rb") as training_file:
        record = pickle.load(training_file, encoding="bytes")

    if not isinstance(record, dict):
        raise ValueError("CIFAR training file did not contain a dictionary")

    raw_images = record.get(b"data", record.get("data"))
    raw_labels = record.get(b"fine_labels", record.get("fine_labels"))
    if raw_images is None or raw_labels is None:
        raise ValueError("CIFAR training file is missing data or fine_labels")

    images = torch.as_tensor(raw_images, dtype=torch.uint8)
    if images.ndim != 2 or images.shape[1] != RGB_IMAGE_LENGTH:
        raise ValueError(
            f"Expected CIFAR data with shape (N, {RGB_IMAGE_LENGTH}), "
            f"got {tuple(images.shape)}"
        )
    labels = [int(label) for label in raw_labels]
    if len(labels) != images.shape[0] or any(label < 0 or label > 99 for label in labels):
        raise ValueError("CIFAR fine_labels are invalid or do not match image count")
    return images, labels


def image_batches(
    images: torch.Tensor, labels: Sequence[int], batch_size: int
) -> Iterator[tuple[torch.Tensor, Sequence[int]]]:
    """Convert CIFAR's channel-first byte layout to normalized DINO input."""
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size]
        batch = batch.reshape(-1, 3, 32, 32).float().div_(255.0)
        batch = functional.interpolate(
            batch, size=(224, 224), mode="bicubic", align_corners=False
        )
        mean = torch.tensor(IMAGENET_MEAN, dtype=batch.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=batch.dtype).view(1, 3, 1, 1)
        yield (batch - mean) / std, labels[start : start + batch.shape[0]]


def load_dinov2(model_name: str, device: torch.device):
    """Load the requested official DINOv2 model; never substitute an encoder."""
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()
    return model


def encode_batch(model, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        features = model(batch.to(device))
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise RuntimeError("DINOv2 returned an unexpected feature tensor")
    return functional.normalize(features, p=2, dim=1).to(dtype=torch.float16).cpu()


def send_image(
    sock: socket.socket,
    destination: tuple[str, int],
    frame_id: int,
    label: int,
    image: torch.Tensor,
    embedding: torch.Tensor,
    mtu: int,
    interval: float,
) -> int:
    embedding_bytes = embedding.contiguous().numpy().tobytes()
    embedding_dimension = embedding.numel()
    fragment_payload_size = mtu - BASE_HEADER_LENGTH - len(embedding_bytes)
    if fragment_payload_size <= 0:
        raise ValueError(
            f"MTU {mtu} is too small for the {embedding_dimension}-value "
            "DINOv2 header vector"
        )

    image_bytes = image.contiguous().numpy().tobytes()
    if len(image_bytes) != RGB_IMAGE_LENGTH:
        raise ValueError(f"Unexpected image size: {len(image_bytes)}")
    fragment_count = (len(image_bytes) + fragment_payload_size - 1) // fragment_payload_size
    for fragment_index in range(fragment_count):
        start = fragment_index * fragment_payload_size
        payload = image_bytes[start : start + fragment_payload_size]
        flags = 0
        if fragment_index == 0:
            flags |= FLAG_FIRST_FRAGMENT
        if fragment_index == fragment_count - 1:
            flags |= FLAG_LAST_FRAGMENT
        header = PACKET_HEADER.pack(
            MAGIC,
            PROTOCOL_VERSION,
            flags,
            BASE_HEADER_LENGTH + len(embedding_bytes),
            frame_id,
            fragment_index,
            fragment_count,
            len(image_bytes),
            embedding_dimension,
            EMBEDDING_FLOAT16,
            len(payload),
            zlib.crc32(payload),
            label,
            0,
        )
        datagram = header + embedding_bytes + payload
        if len(datagram) > mtu:
            raise RuntimeError("Constructed datagram exceeds the configured MTU")
        sock.sendto(datagram, destination)
        if interval:
            time.sleep(interval)
    return fragment_count


def broadcast(args: argparse.Namespace) -> None:
    images, labels = load_cifar_training_file(args.train_file)
    if args.max_images is not None:
        images = images[: args.max_images]
        labels = labels[: args.max_images]

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = load_dinov2(args.model, device)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        destination = (args.broadcast, args.port)
        sent_images = 0
        sent_fragments = 0
        for batch, batch_labels in image_batches(images, labels, args.batch_size):
            embeddings = encode_batch(model, batch, device)
            for embedding, label in zip(embeddings, batch_labels):
                sent_fragments += send_image(
                    sock,
                    destination,
                    sent_images,
                    label,
                    images[sent_images],
                    embedding,
                    args.mtu,
                    args.interval,
                )
                sent_images += 1
                print(
                    f"broadcast image={sent_images}/{len(images)} "
                    f"fragments={sent_fragments} destination={args.broadcast}:{args.port}",
                    flush=True,
                )
    finally:
        sock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Broadcast CIFAR-100 images with DINOv2 semantic packet headers."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path(__file__).parent / "cifar-100-image-classification-challenge" / "train",
        help="CIFAR-100 Python-format training file",
    )
    parser.add_argument("--broadcast", default="255.255.255.255", help="IPv4 broadcast address")
    parser.add_argument("--port", type=int, default=5000, help="UDP destination port")
    parser.add_argument(
        "--model",
        default="dinov2_vits14",
        help="DINOv2 torch hub model name, for example dinov2_vits14 or dinov2_vitb14",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--mtu", type=int, default=1400, help="Maximum UDP datagram size")
    parser.add_argument("--interval", type=float, default=0.0, help="Seconds between fragments")
    parser.add_argument("--device", default="auto", help="Torch device, or auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.mtu <= BASE_HEADER_LENGTH or args.interval < 0:
        raise ValueError("batch-size must be positive, MTU must exceed the header, and interval non-negative")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("max-images must be positive")
    broadcast(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
