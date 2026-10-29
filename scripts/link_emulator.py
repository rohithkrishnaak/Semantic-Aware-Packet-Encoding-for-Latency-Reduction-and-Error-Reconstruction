"""Userspace stand-in for `tc netem`/`tbf` (Windows has neither): a UDP relay with
random loss, a rate limit and a drop-tail queue. Uncontrolled drops happen here.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import select
import socket
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=6000)
    ap.add_argument("--forward", default="127.0.0.1:6001")
    ap.add_argument("--loss", type=float, default=0.0, help="iid loss probability")
    ap.add_argument("--rate", type=float, default=0.0, help="packets/s, 0 = unlimited")
    ap.add_argument("--queue", type=int, default=16, help="drop-tail queue limit (packets)")
    ap.add_argument("--idle-exit", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    host, port = args.forward.rsplit(":", 1)
    fwd = (host, int(port))
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    rx.bind(("127.0.0.1", args.listen))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    q: collections.deque[bytes] = collections.deque()
    stats = collections.Counter()
    next_tx = time.perf_counter()
    last_rx = None
    print("link: ready", flush=True)
    while True:
        now = time.perf_counter()
        while q and (args.rate <= 0 or now >= next_tx):
            tx.sendto(q.popleft(), fwd)
            stats["forwarded"] += 1
            next_tx = max(next_tx, now) + (1.0 / args.rate if args.rate > 0 else 0)
        timeout = max(0.0, next_tx - now) if q else 0.05
        if select.select([rx], [], [], timeout)[0]:
            data = rx.recv(65535)
            last_rx = time.perf_counter()
            stats["in"] += 1
            if rnd.random() < args.loss:
                stats["random_loss"] += 1
            elif len(q) >= args.queue:
                stats["queue_drop"] += 1
            else:
                q.append(data)
        elif last_rx and not q and time.perf_counter() - last_rx > args.idle_exit:
            break
    print("link: " + json.dumps(dict(stats)), flush=True)


if __name__ == "__main__":
    main()
