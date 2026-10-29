"""SAEP v2 sender: replays precomputed tiles + int8 tags over UDP at a fixed frame rate.

Labels are never put on the wire; they go to a sidecar CSV keyed by frame_id.
If --feedback is on, the sender listens for SAEC control packets and then sends
each frame's packets in descending tag-score order, optionally only the top `budget`.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saep.data import ARTIFACTS, CIFAR_DIR, load_cifar  # noqa: E402
from saep.models import load_admission  # noqa: E402
from saep.protocol import Packet, decode_control, encode_packet  # noqa: E402
from scripts.train import to_tiles  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="127.0.0.1:6000", help="receiver (or link emulator) host:port")
    ap.add_argument("--control-port", type=int, default=6002, help="local UDP port for SAEC feedback")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0, help="first test image index")
    ap.add_argument("--feedback", action="store_true", help="reorder/filter using receiver feedback")
    ap.add_argument("--sender-filter", action="store_true", help="with feedback: send only top `budget`")
    args = ap.parse_args()

    scheme, scorers, _, grid = load_admission(ARTIFACTS / "admission.npz")
    feats = np.load(ARTIFACTS / "features_test.npz")
    sl = slice(args.offset, args.offset + args.frames)
    images, labels = load_cifar(CIFAR_DIR / "test")
    tiles = to_tiles(images[sl], grid)
    tags = scheme.tags(feats["tile_tok"][sl])
    p = grid * grid
    scores = scorers["ctx"].score(tags)  # sender sees whole frame: use the context scorer
    n = len(tiles)

    (ARTIFACTS / "sidecar_labels.csv").write_text(
        "frame_id,label\n" + "".join(f"{i},{labels[sl][i]}\n" for i in range(n)))

    host, port = args.dest.rsplit(":", 1)
    dest = (host, int(port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl.bind(("127.0.0.1", args.control_port))
    ctrl.setblocking(False)

    budget = p
    period = 1.0 / args.fps
    t_next = time.perf_counter()
    sent = 0
    for f in range(n):
        while True:  # drain feedback
            try:
                budget = decode_control(ctrl.recv(2048)).budget_per_frame
            except (BlockingIOError, ConnectionResetError):
                break
            except ValueError:
                continue
        order = np.arange(p)
        if args.feedback:
            order = np.argsort(-scores[f], kind="stable")
            if args.sender_filter:
                order = order[:budget]
        gap = period / len(order)  # spread this frame's packets over the frame period
        for idx in order:
            pkt = Packet(f, int(idx), p, scheme.scheme_id, tags[f, idx], tiles[f, idx].tobytes())
            sock.sendto(encode_packet(pkt), dest)
            sent += 1
            t_next += gap
            while (d := t_next - time.perf_counter()) > 0:
                if d > 0.002:
                    time.sleep(d - 0.001)  # then spin for the last ms
    print(f"sender: frames={n} packets={sent} feedback={args.feedback} last_budget={budget}", flush=True)


if __name__ == "__main__":
    main()
