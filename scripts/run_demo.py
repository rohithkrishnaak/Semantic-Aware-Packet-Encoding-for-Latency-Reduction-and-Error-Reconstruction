"""End-to-end localhost demo: sender -> link emulator -> receiver, several configurations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(name: str, frames: int, fps: float, budget: int, policy: str, loss: float, rate: float,
        feedback: bool, artifacts: Path, recv_extra: tuple[str, ...] = ()) -> dict:
    link = subprocess.Popen([PY, "scripts/link_emulator.py", "--loss", str(loss), "--rate", str(rate),
                             "--queue", "8"], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    assert link.stdout.readline().startswith("link: ready")
    recv = subprocess.Popen([PY, "scripts/receiver.py", "--budget", str(budget), "--policy", policy, "--json",
                             "--artifacts", str(artifacts), *recv_extra], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    time.sleep(3.0)  # let the receiver load torch + model
    send_cmd = [PY, "scripts/sender.py", "--frames", str(frames), "--fps", str(fps), "--artifacts", str(artifacts)]
    if feedback:
        send_cmd += ["--feedback", "--sender-filter"]
    subprocess.run(send_cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    result = json.loads(recv.communicate(timeout=120)[0].strip().splitlines()[-1])
    link_stats = link.communicate(timeout=30)[0].strip().splitlines()[-1]
    result["link"] = json.loads(link_stats.removeprefix("link: "))
    result["name"] = name
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--v3", action="store_true", help="use artifacts/v3 and add diversity / early-exit rows")
    args = ap.parse_args()
    artifacts = ROOT / "artifacts" / ("v3" if args.v3 else "")
    ctx_extra = ("--diverse",) if args.v3 else ()
    f, fps = args.frames, args.fps
    pkt_rate = 16 * fps
    configs = [
        ("clean link, no admission", 16, "fifo", 0.0, 0, False),
        ("CPU budget k=4, FIFO", 4, "fifo", 0.0, 0, False),
        ("CPU budget k=4, random", 4, "random", 0.0, 0, False),
        ("CPU budget k=4, int8 tag (XDP-able)", 4, "tag", 0.0, 0, False),
        ("CPU budget k=4, tag + context MLP", 4, "ctx", 0.0, 0, False, ctx_extra),
        ("20% loss + k=4, FIFO", 4, "fifo", 0.2, 0, False),
        ("20% loss + k=4, tag + context MLP", 4, "ctx", 0.2, 0, False, ctx_extra),
        ("bottleneck 30% rate, raster sender", 16, "fifo", 0.0, pkt_rate * 0.30, False),
        ("bottleneck 30% rate, feedback sender (k=4)", 4, "fifo", 0.0, pkt_rate * 0.30, True),
    ]
    if args.v3:
        configs += [
            ("CPU budget avg 4, context MLP + early exit", 4, "ctx", 0.0, 0, False, ("--early-exit",)),
            ("20% loss + avg 4, context MLP + early exit", 4, "ctx", 0.2, 0, False, ("--early-exit",)),
        ]
    results = []
    for name, k, pol, loss, rate, fb, *extra in configs:
        r = run(name, f, fps, k, pol, loss, rate, fb, artifacts, *extra)
        results.append(r)
        print(f"{name:<40} acc={r['accuracy']:.3f}  uncontrolled_loss={r['uncontrolled_loss']:.2f}  "
              f"admitted/frame={r['admitted_per_frame']:.1f}  proc={r['proc_ms_mean']:.2f}ms  link={r['link']}",
              flush=True)
    (artifacts / "demo_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
