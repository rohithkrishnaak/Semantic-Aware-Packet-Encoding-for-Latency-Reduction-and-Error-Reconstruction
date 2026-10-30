# Semantic-Aware Packet Encoding (SAEP v2) — MVP

A resource-constrained receiver gets a stream of image packets. Each packet carries a small
**learned semantic tag** (32 B int8 derived from DINOv2). When the receiver is CPU-limited or the
link is congested, it decides **from the tags alone** which packets to process, aiming to keep
task accuracy high while processing few packets.

This MVP covers the offline simulator (milestones M0–M3 of `plan v1.txt`) plus a working UDP
protocol stack on localhost with a userspace link emulator. It has **no Raspberry Pi
dependency**. Pi, netns and XDP work comes next (see *Roadmap*).

## Results (CIFAR-100 test set, 10k images, 4×4 tiles = 16 packets/frame)

Receiver task model: per-tile CNN with mean-pooled logits, trained on the pixels of admitted
tiles only. It reaches **50.0%** top-1 with all 16 tiles. Its CPU cost is linear in admitted
tiles: 0.50 ms for 1 tile and 2.90 ms for 16 tiles on one core.

**Offline simulator, receiver budget k tiles/frame** (`scripts/simulate.py`, accuracy):

| k | FIFO | random | position prior | int8 tag (linear) | tag + context MLP | oracle* |
|---|---|---|---|---|---|---|
| 1 | 11.0 | 16.8 | 16.8 | 22.2 | **27.7** | 68.8 |
| 2 | 18.8 | 25.3 | 22.0 | 28.9 | **34.5** | 77.3 |
| 4 | 22.6 | 35.3 | 28.4 | 37.3 | **41.3** | 78.6 |
| 8 | 38.1 | 43.4 | 38.1 | 44.2 | **46.8** | 73.6 |
| 16 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |

\*The oracle uses the true label. It is an upper bound, not a policy.

- At k=4 the learned policy keeps 83% of full-frame accuracy using 25% of the packets and about
  36% of the compute. Random keeps 71%.
- With uncontrolled iid loss of 10–50% before admission, the ranking holds. For example, at
  p=0.3, k=4: FIFO 28.6, random 34.7, context MLP 40.8.
- In a bottleneck where only c packets/frame get through, a sender that orders by score using
  feedback beats a raster sender: 41.3 vs 22.6 at c=4.

**End-to-end UDP on localhost** (`scripts/run_demo.py`, 300 frames at 25 fps,
sender → link emulator → receiver):

| configuration | accuracy | notes |
|---|---|---|
| clean link, all 16 tiles | 54.7 | 3.5 ms/frame receiver compute |
| k=4, FIFO | 25.7 | |
| k=4, random | 36.7 | |
| k=4, int8 tag, streaming threshold (XDP-style) | 33.0 | below random; see findings |
| k=4, tag + context MLP | **40.7** | 1.6–1.8 ms/frame |
| 20% link loss + k=4, FIFO | 28.0 | 949 uncontrolled drops |
| 20% link loss + k=4, context MLP | **40.7** | |
| 30% rate bottleneck, raster sender | 34.7–36.3 | ~3570 queue drops; varies run to run |
| 30% rate bottleneck, feedback sender (SAEC, k=4) | **40.7–41.0** | 5–17 queue drops, sender transmits 75% fewer packets |

The test images differ between the two tables (300 vs 10k), so compare within a table. The two
bottleneck rows depend on OS timing and move by a point or two between runs.

Controls (linear probes on val): **tag-only** classification from the mean of the 16 int8 tags
reaches 71.3%, and the **v1 full-image CLS header** reaches 83.4%. Both beat the
pixel receiver, confirming the problem v1 had: the header alone would answer the task. In v2 the
tags are used *only* for admission, and the receiver classifies from pixels.

## Findings that changed the plan

1. **The int8 linear scorer (XDP stage) is weak.** Its correlation with tile utility is ≈0.1,
   and it beats random only for k ≤ 4 with exact top-k. As a streaming per-packet threshold it
   is slightly *below* random, because random selection spreads tiles across the image while a
   noisy threshold doesn't. Utility depends mostly on *the rest of the frame*, which a
   per-packet linear function can't see.
2. **Frame context fixes it.** A 64→128→1 MLP over `[tag, tag − mean(tag of arrived packets)]`
   (8.4k params, microseconds per frame) gives +6 points over random at k=4. It needs the
   frame buffered, so it lives in user space, not XDP.
3. **Train on per-frame-centered utility.** Only the ranking inside a frame matters. Using
   centered targets and fitting on the 45k train split (not 5k val) turned the linear scorer
   from below random into above random.
4. **Reordering alone doesn't help a paced sender.** When packets are spread across the frame
   period, a rate-limited queue drops them in an interleaved pattern no matter the order. The
   feedback loop must make the sender **filter** to the advertised budget (plan §9). The budget
   also needs headroom below link capacity: at exactly 100% load, jitter still caused 362 drops.

### Revised architecture (replaces plan §3 receiver box)

```
[early stage, XDP-able]  int8 linear score -> cheap pre-filter only (drop clear losers, protect
                         the queue under overload). Optional on the Pi.
[user space]             buffer frame until last packet / timeout -> context-MLP rescoring
                         -> top-k by CPU budget -> receiver model (cost ∝ k)
[feedback, SAEC 10 Hz]   budget k (≤ link capacity with headroom) -> sender sends only
                         its top-k by the same context scorer. Under congestion this is the
                         mechanism that matters, since the receiver can't choose packets
                         the network already dropped.
```

## Live simulator (browser)

Open `viz/SAEP_Simulator.html` (after `python scripts/export_viz.py`). It is a single offline file; no
server is needed. It replays the same 300 test frames as the UDP demo through a JavaScript port of
`sender.py`, `link_emulator.py` and `receiver.py`, using the real pixels, int8 tags, per-tile CNN
outputs and trained scorer weights. You can pick the nine `run_demo.py` presets or change policy,
budget k, loss, bottleneck rate and SAEC feedback, step frame by frame, click a tile to see its real
248-byte SAEP packet, and compare all five policies on the same arrivals.

Validation (`node viz/parity_test.js`): packet bytes, int8 scores and context-MLP scores are
identical to Python, offline accuracy for every k matches exactly, and the five deterministic UDP
scenarios match the real socket run frame for frame. Scenarios with random loss or a rate limit
match within a few points. The page shows this comparison in its validation table.

Finding from building it: on Windows, `select()` in `link_emulator.py` wakes on the 15.6 ms timer
tick or on the next arrival, so a 120 pkt/s limit actually forwards about 101 pkt/s. The surviving
packets then lock onto one column of the 4x4 grid, which is why the raster-sender bottleneck result
is lower than an ideal token-bucket link would give. The simulator's "Link timer" option switches
between the two.

## Layout

```
saep/protocol.py        SAEP v2 data packet (24 B header + 32 B tag + 192 B tile) and SAEC control
saep/models.py          TileReceiver, TagScheme (PCA→int8), IntScorer (XDP-able), ContextScorer
saep/data.py            CIFAR-100 loading, stratified 45k/5k split (data/splits.json)
scripts/extract_features.py  Phase 1: DINOv2 ViT-S/14 once on GPU → per-tile pooled patch tokens
scripts/train.py        Phase 2: tag projection, receiver, utilities, scorers, control probes
scripts/simulate.py     Phase 3: offline policy comparison (+ CSV and plot in artifacts/)
scripts/sender.py / receiver.py / link_emulator.py / run_demo.py   UDP end-to-end
scripts/export_viz.py   real artifacts -> viz/data.js (+ offline viz/SAEP_Simulator.html)
viz/engine.js           JS port of sender, link emulator and receiver (no DOM)
viz/simulator.html      live browser simulator (source page)
viz/parity_test.js      checks the JS engine against Python (16 checks)
tests/test_protocol.py  wire format, corruption rejection, tiling, int scorer
legacy/broadcaster_v1.py  original v1 broadcaster (reference only)
```

## Reproduce

```bash
pip install -r requirements.txt
mkdir -p data && curl -L https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz | tar xz -C data
python -m unittest discover -s tests
python scripts/extract_features.py      # ~5 min on an RTX 5060 laptop GPU
python scripts/train.py                 # ~7 min (receiver 30 epochs); --reuse-receiver to refit scorers only
python scripts/simulate.py
python scripts/run_demo.py --frames 300
python scripts/export_viz.py            # builds viz/data.js and viz/SAEP_Simulator.html
node viz/parity_test.js                 # JS engine vs Python (needs Node.js)
```

The Kaggle `train` pickle has the same format, so `saep.data.load_cifar` reads it too.

## Simplifications vs. the plan

- Tag projection is PCA (384→32) with a global int8 scale. It isn't a trained projection yet.
- `tc netem`/`tbf` and network namespaces are replaced by `scripts/link_emulator.py` (iid loss,
  rate limit, drop-tail queue), because the dev machine runs Windows.
- No XDP program yet. `saep.protocol.peek_tag` plus `IntScorer.score` is exactly the integer
  arithmetic the eBPF program would do.
- The receiver runs PyTorch on CPU. ONNX export for the Pi is still to do.

## Roadmap (with the Pi)

1. Export the receiver and context scorer to ONNX and measure ms/frame vs k on the Pi
   (`taskset`, log `vcgencmd` temperature and throttling).
2. Linux netns + `tc netem`/`tbf` instead of the userspace emulator. Keep the same scripts.
3. XDP pre-filter (int8 score vs τ in a BPF map) and a load controller that sets k from CPU
   quota and queue depth.
4. Train the projection end-to-end for the admission objective (rank loss on centered utility)
   instead of using PCA. Try tag sizes of 16, 32 and 64 B.
