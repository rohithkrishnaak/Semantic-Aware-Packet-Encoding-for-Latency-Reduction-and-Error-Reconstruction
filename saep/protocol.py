"""SAEP v2 wire format: one spatial tile per packet plus a small int8 semantic tag.

Data packet layout::

    SAEP_V2 header (24 B) | tag (tag_dim B, int8) | payload (payload_len B, HWC uint8 RGB)

Control packet (receiver -> sender)::

    SAEC header (13 B) | optional int8 query vector (query_dim B)
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

MAGIC = b"SAEP"
CONTROL_MAGIC = b"SAEC"
PROTOCOL_VERSION = 2
TAG_INT8 = 1
FLAG_FIRST = 1
FLAG_LAST = 2

# magic, version, flags, header_len, frame_id, pkt_index, pkt_count,
# tag_dim, tag_dtype, tag_scheme_id, payload_len, payload_crc32
SAEP_V2 = struct.Struct("!4sBBHIBBBBHHI")
# magic, version, flags, budget_per_frame, threshold, query_dim
SAEC = struct.Struct("!4sBBHiB")

IMAGE_SIDE = 32


@dataclass(frozen=True)
class Packet:
    frame_id: int
    pkt_index: int
    pkt_count: int
    tag_scheme_id: int
    tag: np.ndarray  # int8, shape (tag_dim,)
    payload: bytes


@dataclass(frozen=True)
class Control:
    budget_per_frame: int
    threshold: int
    query: np.ndarray | None = None


def encode_packet(pkt: Packet) -> bytes:
    tag = np.ascontiguousarray(pkt.tag, dtype=np.int8)
    flags = (FLAG_FIRST if pkt.pkt_index == 0 else 0) | (
        FLAG_LAST if pkt.pkt_index == pkt.pkt_count - 1 else 0
    )
    header = SAEP_V2.pack(
        MAGIC, PROTOCOL_VERSION, flags, SAEP_V2.size + tag.size,
        pkt.frame_id, pkt.pkt_index, pkt.pkt_count,
        tag.size, TAG_INT8, pkt.tag_scheme_id,
        len(pkt.payload), zlib.crc32(pkt.payload),
    )
    return header + tag.tobytes() + pkt.payload


def decode_packet(datagram: bytes, check_crc: bool = True) -> Packet:
    """Parse and validate a data packet; raises ValueError on any malformation."""
    if len(datagram) < SAEP_V2.size:
        raise ValueError("datagram shorter than SAEP v2 header")
    (magic, version, _flags, header_len, frame_id, pkt_index, pkt_count,
     tag_dim, tag_dtype, scheme, payload_len, crc) = SAEP_V2.unpack_from(datagram)
    if magic != MAGIC or version != PROTOCOL_VERSION:
        raise ValueError("bad magic or version")
    if tag_dtype != TAG_INT8 or header_len != SAEP_V2.size + tag_dim:
        raise ValueError("bad tag descriptor")
    if len(datagram) != header_len + payload_len:
        raise ValueError("length mismatch")
    if pkt_count == 0 or pkt_index >= pkt_count:
        raise ValueError("bad packet index")
    payload = bytes(datagram[header_len:])
    if check_crc and zlib.crc32(payload) != crc:
        raise ValueError("payload CRC mismatch")
    tag = np.frombuffer(datagram, dtype=np.int8, count=tag_dim, offset=SAEP_V2.size)
    return Packet(frame_id, pkt_index, pkt_count, scheme, tag, payload)


def peek_tag(datagram: bytes) -> tuple[int, int, np.ndarray]:
    """Cheap early-path view (what XDP would see): frame id, index, tag. No CRC."""
    _, _, _, _, frame_id, pkt_index, _, tag_dim, *_ = SAEP_V2.unpack_from(datagram)
    return frame_id, pkt_index, np.frombuffer(datagram, np.int8, tag_dim, SAEP_V2.size)


def encode_control(ctrl: Control) -> bytes:
    query = b"" if ctrl.query is None else np.asarray(ctrl.query, np.int8).tobytes()
    return SAEC.pack(CONTROL_MAGIC, PROTOCOL_VERSION, 0, ctrl.budget_per_frame,
                     ctrl.threshold, len(query)) + query


def decode_control(datagram: bytes) -> Control:
    if len(datagram) < SAEC.size:
        raise ValueError("datagram shorter than SAEC header")
    magic, version, _flags, budget, threshold, qdim = SAEC.unpack_from(datagram)
    if magic != CONTROL_MAGIC or version != PROTOCOL_VERSION:
        raise ValueError("bad control magic or version")
    if len(datagram) != SAEC.size + qdim:
        raise ValueError("control length mismatch")
    query = np.frombuffer(datagram, np.int8, qdim, SAEC.size) if qdim else None
    return Control(budget, threshold, query)


def image_to_tiles(chw_flat: np.ndarray, grid: int) -> np.ndarray:
    """CIFAR channel-planar (3072,) uint8 -> (grid*grid, t, t, 3) HWC tiles, raster order."""
    t = IMAGE_SIDE // grid
    hwc = np.asarray(chw_flat, np.uint8).reshape(3, IMAGE_SIDE, IMAGE_SIDE).transpose(1, 2, 0)
    return hwc.reshape(grid, t, grid, t, 3).transpose(0, 2, 1, 3, 4).reshape(grid * grid, t, t, 3)


def tiles_to_image(tiles: dict[int, bytes], grid: int) -> np.ndarray:
    """Reassemble received tiles into a (32,32,3) HWC image; missing tiles are zero."""
    t = IMAGE_SIDE // grid
    out = np.zeros((IMAGE_SIDE, IMAGE_SIDE, 3), np.uint8)
    for idx, payload in tiles.items():
        r, c = divmod(idx, grid)
        out[r * t:(r + 1) * t, c * t:(c + 1) * t] = np.frombuffer(payload, np.uint8).reshape(t, t, 3)
    return out
