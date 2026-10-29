"""SAEP v2 receiver with learned admission.

Per packet (the part that would live in XDP on the Pi):
    peek tag -> integer score -> admit if score >= tau and frame quota < k, else drop.
tau is tracked by a load controller so that, on average, k of the packets that
actually arrive per frame are admitted. Admitted payloads go to the receiver
model, whose cost is linear in admitted tiles. Feedback (SAEC) is sent to the sender.
"""

from __future__ import annotations

import argparse
import collections
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saep.data import ARTIFACTS  # noqa: E402
from saep.models import load_admission, load_receiver  # noqa: E402
from saep.protocol import Control, decode_packet, encode_control, peek_tag  # noqa: E402


class ThresholdController:
    """tau = (1 - k/arrivals)-quantile of recent scores, so ~k arrivals/frame pass."""

    def __init__(self, k: int, window: int = 2048):
        self.k = k
        self.scores: collections.deque[int] = collections.deque(maxlen=window)
        self.arrivals: collections.deque[int] = collections.deque(maxlen=64)
        self.tau = -(2**31)

    def observe_score(self, s: int) -> None:
        self.scores.append(s)

    def end_frame(self, arrived: int) -> None:
        self.arrivals.append(arrived)
        mean_arr = float(np.mean(self.arrivals))
        if len(self.scores) < 64 or mean_arr <= self.k:
            self.tau = -(2**31)
        else:
            self.tau = int(np.quantile(np.fromiter(self.scores, np.int64), 1 - self.k / mean_arr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6001)
    ap.add_argument("--feedback", default="127.0.0.1:6002", help="sender control address")
    ap.add_argument("--budget", type=int, default=16, help="max admitted tiles per frame (k)")
    ap.add_argument("--policy", choices=("ctx", "tag", "tagpos", "fifo", "random"), default="ctx",
                    help="ctx = buffer frame, rescore with context MLP (user space); others are per-packet")
    ap.add_argument("--frame-timeout", type=float, default=0.15, help="s after first packet")
    ap.add_argument("--idle-exit", type=float, default=3.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    scheme, scorers, _, grid = load_admission(ARTIFACTS / "admission.npz")
    model = load_receiver(ARTIFACTS / "receiver.pt", grid)
    p, t = grid * grid, 32 // grid
    sidecar = {}
    rng = np.random.default_rng(0)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    sock.bind(("127.0.0.1", args.port))
    sock.settimeout(0.05)
    fh, fp = args.feedback.rsplit(":", 1)
    fb_addr = (fh, int(fp))

    ctl = ThresholdController(args.budget)
    frames: dict[int, dict] = {}
    st = collections.Counter()
    correct = 0
    proc_ms: list[float] = []
    last_rx = last_fb = time.perf_counter()

    def finish(fid: int) -> None:
        nonlocal correct
        fr = frames.pop(fid)
        ctl.end_frame(fr["arrived"])
        st["frames"] += 1
        if not sidecar:
            sidecar.update(_load_sidecar())
        if not fr["tiles"]:
            st["empty_frames"] += 1
            return
        t0 = time.perf_counter()
        idx = np.array(list(fr["tiles"]))
        if args.policy == "ctx":  # user-space rescoring stage over the buffered frame
            keep = np.argsort(-scorers["ctx"].score(np.stack([fr["tags"][i] for i in idx])),
                              kind="stable")[:args.budget]
            st["controlled_drop"] += len(idx) - len(keep)
            st["admitted"] += len(keep)
            idx = idx[keep]
        x = torch.from_numpy(np.stack([np.frombuffer(fr["tiles"][i], np.uint8).reshape(t, t, 3) for i in idx]))
        with torch.inference_mode():
            logits = model.tile_logits(x, torch.from_numpy(idx)).mean(0)
        proc_ms.append((time.perf_counter() - t0) * 1e3)
        correct += int(int(logits.argmax()) == sidecar.get(fid, -1))

    while True:
        now = time.perf_counter()
        if now - last_fb > 0.1:  # SAEC at 10 Hz
            sock.sendto(encode_control(Control(args.budget, max(ctl.tau, -(2**31)))), fb_addr)
            last_fb = now
        for fid in [f for f, fr in frames.items() if now - fr["t0"] > args.frame_timeout]:
            finish(fid)
        try:
            data = sock.recv(65535)
        except (socket.timeout, ConnectionResetError):
            if now - last_rx > args.idle_exit and st["rx"]:
                break
            continue
        last_rx = time.perf_counter()
        st["rx"] += 1
        # early path: header + tag only
        fid, pidx, tag = peek_tag(data)
        if args.policy in ("tag", "tagpos"):
            score = int(scorers[args.policy].score(tag, pidx))
        elif args.policy in ("fifo", "ctx"):
            score = 0
        else:
            score = int(rng.integers(0, 1 << 16))
        fr = frames.setdefault(fid, {"t0": last_rx, "arrived": 0, "tiles": {}, "tags": {}})
        fr["arrived"] += 1
        if args.policy == "ctx":
            try:
                pkt = decode_packet(data)
            except ValueError:
                st["malformed"] += 1
                continue
            fr["tiles"][pkt.pkt_index] = pkt.payload
            fr["tags"][pkt.pkt_index] = pkt.tag
            if len(fr["tiles"]) == pkt.pkt_count:
                finish(fid)
            continue
        if args.policy != "fifo":
            ctl.observe_score(score)
        quota_left = args.budget - len(fr["tiles"])
        pkts_left = data[13] - pidx  # header byte 13 = pkt_count; count includes this packet
        must_fill = pkts_left <= quota_left  # assumes in-order arrival; under loss it only under-fills
        if quota_left <= 0 or (args.policy != "fifo" and score < ctl.tau and not must_fill):
            st["controlled_drop"] += 1
            continue
        try:
            pkt = decode_packet(data)
        except ValueError:
            st["malformed"] += 1
            continue
        fr["tiles"][pkt.pkt_index] = pkt.payload
        st["admitted"] += 1
    for fid in list(frames):
        finish(fid)

    n_sent = len(sidecar) * p
    out = {
        "policy": args.policy, "budget": args.budget, "frames_seen": st["frames"],
        "accuracy": correct / max(len(sidecar), 1),
        "packets_expected": n_sent, "received": st["rx"],
        "uncontrolled_loss": 1 - st["rx"] / max(n_sent, 1),
        "controlled_drops": st["controlled_drop"], "admitted": st["admitted"],
        "admitted_per_frame": st["admitted"] / max(st["frames"], 1),
        "empty_frames": st["empty_frames"], "malformed": st["malformed"],
        "proc_ms_mean": float(np.mean(proc_ms)) if proc_ms else 0.0,
    }
    print(json.dumps(out) if args.json else "\n".join(f"{k}: {v}" for k, v in out.items()), flush=True)


def _load_sidecar() -> dict[int, int]:
    path = ARTIFACTS / "sidecar_labels.csv"
    rows = path.read_text().splitlines()[1:]
    return {int(a): int(b) for a, b in (r.split(",") for r in rows)}


if __name__ == "__main__":
    main()
