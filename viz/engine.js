/* SAEP v2 simulation engine: a JavaScript port of scripts/sender.py, link_emulator.py and
 * receiver.py, driven by the real artifacts exported in data.js (scripts/export_viz.py).
 * Pure logic, no DOM: viz/parity_test.js runs it under Node against Python reference values. */
(function (root) {
  "use strict";

  // Default Windows timer resolution (1/64 s): select() timeouts in link_emulator.py round up to it,
  // so the link mostly transmits when the next packet wakes it. 0 = ideal token-bucket link.
  const WINDOWS_TIMER_TICK = 1 / 64;
  const HEADER = 24, SAEC_LEN = 13, FLAG_FIRST = 1, FLAG_LAST = 2, MAGIC = [0x53, 0x41, 0x45, 0x50];

  function bytes(b64) {
    const s = root.atob(b64), out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }

  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();

  function crc32(buf) {  // same polynomial as zlib.crc32
    let c = 0xffffffff;
    for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // numpy.quantile with the default linear method, then int() truncation like receiver.py
  function quantileTrunc(values, q) {
    const s = Float64Array.from(values).sort();
    const h = (s.length - 1) * q, lo = Math.floor(h), hi = Math.min(lo + 1, s.length - 1);
    return Math.trunc(s[lo] + (h - lo) * (s[hi] - s[lo]));
  }

  class Dataset {
    constructor(d) {
      this.d = d;
      this.n = d.n; this.grid = d.grid; this.P = d.grid * d.grid; this.T = d.tile;
      this.D = d.tag_dim; this.C = d.classes;
      this.images = bytes(d.images);
      this.tags = new Int8Array(bytes(d.tags).buffer);
      this.logits = new Float32Array(bytes(d.logits).buffer);
      this.w1 = new Float32Array(bytes(d.ctx.w1).buffer);
      this.b1 = new Float32Array(bytes(d.ctx.b1).buffer);
      this.w2 = new Float32Array(bytes(d.ctx.w2).buffer);
      this.H = d.ctx.hidden;
      this.labels = d.labels; this.names = d.class_names;
      this.cpu = d.cpu_ms;
      this.policy = d.policy || null;  // v3: diversity weights and early-exit thresholds
      this.adj = Array.from({ length: this.P }, (_, i) => {
        const r = Math.floor(i / this.grid), c = i % this.grid, out = [];
        if (r > 0) out.push(i - this.grid); if (r < this.grid - 1) out.push(i + this.grid);
        if (c > 0) out.push(i - 1); if (c < this.grid - 1) out.push(i + 1);
        return out;
      });
    }

    /* saep.models.select_diverse for one frame: greedy pick maximising
     * score - lam_adj * (#picked 4-neighbours) - lam_sim * max(cos sim to a picked tag, 0) */
    selectDiverse(f, idx, s, k) {
      const P = this.P, D = this.D, { lam_adj, lam_sim } = this.policy;
      const score = new Float64Array(P).fill(-Infinity), unit = [];
      idx.forEach((i, j) => (score[i] = s[j]));
      for (let i = 0; i < P; i++) {
        const t = this.tag(f, i);
        let n = 0;
        for (let j = 0; j < D; j++) n += t[j] * t[j];
        n = Math.max(Math.sqrt(n), 1e-6);
        unit.push(Float64Array.from(t, v => v / n));
      }
      const near = new Float64Array(P), maxsim = new Float64Array(P).fill(-1), picked = new Uint8Array(P), order = [];
      for (let j = 0; j < k; j++) {
        let best = -1, bv = -Infinity;
        for (let i = 0; i < P; i++) {
          if (picked[i] || score[i] === -Infinity) continue;
          const v = score[i] - lam_adj * near[i] - lam_sim * Math.max(maxsim[i], 0);
          if (v > bv) { bv = v; best = i; }
        }
        if (best < 0) break;
        order.push(best); picked[best] = 1;
        for (const nb of this.adj[best]) near[nb] += 1;
        for (let i = 0; i < P; i++) {
          let dot = 0;
          for (let j = 0; j < D; j++) dot += unit[best][j] * unit[i][j];
          if (dot > maxsim[i]) maxsim[i] = dot;
        }
      }
      return order;
    }

    /* saep.models.exit_confidence: softmax margin between the top two classes */
    exitMargin(sumLogits, n) {
      const C = this.C;
      let m = -Infinity;
      for (let c = 0; c < C; c++) m = Math.max(m, sumLogits[c] / n);
      let sum = 0, a = 0, b = 0;
      for (let c = 0; c < C; c++) {
        const e = Math.exp(sumLogits[c] / n - m);
        sum += e;
        if (e > a) { b = a; a = e; } else if (e > b) b = e;
      }
      return (a - b) / sum;
    }

    /* Early exit over a ranked tile list: k-1 tiles first, then one at a time until confident */
    earlyExit(f, ranked, budget) {
      const theta = this.policy.exit[String(budget)];
      const kmin = Math.max(1, budget - 1), kmax = Math.min(this.P, this.policy.kmax_mult * budget);
      const list = ranked.slice(0, kmax), C = this.C, acc = new Float64Array(C);
      const add = i => { const o = (f * this.P + i) * C; for (let c = 0; c < C; c++) acc[c] += this.logits[o + c]; };
      let used = Math.min(kmin, list.length);
      for (let j = 0; j < used; j++) add(list[j]);
      while (used < list.length && this.exitMargin(acc, used) < theta) add(list[used++]);
      return list.slice(0, used);
    }
    canExit(budget) { return !!(this.policy && this.policy.exit[String(budget)] != null); }
    tag(f, i) { const o = (f * this.P + i) * this.D; return this.tags.subarray(o, o + this.D); }

    /* HWC uint8 bytes of tile i, exactly the 192-byte packet payload */
    tilePayload(f, i) {
      const T = this.T, g = this.grid, r = Math.floor(i / g), c = i % g, out = new Uint8Array(T * T * 3);
      const base = f * 32 * 32 * 3;
      for (let y = 0; y < T; y++)
        for (let x = 0; x < T; x++)
          for (let ch = 0; ch < 3; ch++)
            out[(y * T + x) * 3 + ch] = this.images[base + ((r * T + y) * 32 + (c * T + x)) * 3 + ch];
      return out;
    }

    /* IntScorer.score: integer dot product, what the XDP stage computes */
    tagScore(f, i) {
      const t = this.tag(f, i), s = this.d.tag_scorer;
      let acc = 0;
      for (let j = 0; j < this.D; j++) acc += t[j] * s.w[j];
      return acc + s.w_pos[i] + s.b;
    }

    /* ContextScorer.score over the tiles in `idx` (the arrived part of the frame) */
    ctxScores(f, idx) {
      const D = this.D, H = this.H, sc = this.d.ctx.in_scale, mean = new Float64Array(D);
      for (const i of idx) { const t = this.tag(f, i); for (let j = 0; j < D; j++) mean[j] += t[j]; }
      for (let j = 0; j < D; j++) mean[j] /= Math.max(idx.length, 1);
      const x = new Float64Array(2 * D), out = [];
      for (const i of idx) {
        const t = this.tag(f, i);
        for (let j = 0; j < D; j++) { x[j] = t[j] / sc; x[D + j] = (t[j] - mean[j]) / sc; }
        let s = this.d.ctx.b2;
        for (let h = 0; h < H; h++) {
          let a = this.b1[h];
          for (let j = 0; j < 2 * D; j++) a += x[j] * this.w1[j * H + h];
          if (a > 0) s += a * this.w2[h];
        }
        out.push(s);
      }
      return out;
    }

    /* Receiver decision: image logits = mean of admitted per-tile logits */
    classify(f, admitted) {
      const C = this.C, z = new Float64Array(C);
      for (const i of admitted) {
        const o = (f * this.P + i) * C;
        for (let c = 0; c < C; c++) z[c] += this.logits[o + c];
      }
      for (let c = 0; c < C; c++) z[c] /= Math.max(admitted.length, 1);
      let m = -Infinity;
      for (let c = 0; c < C; c++) m = Math.max(m, z[c]);
      let sum = 0;
      const pr = new Float64Array(C);
      for (let c = 0; c < C; c++) { pr[c] = Math.exp(z[c] - m); sum += pr[c]; }
      const top = Array.from(pr.keys()).sort((a, b) => pr[b] - pr[a]).slice(0, 3)
        .map(c => ({ c, name: this.names[c], p: pr[c] / sum }));
      const pred = admitted.length ? top[0].c : -1;
      return { pred, top, correct: pred === this.labels[f] };
    }

    /* Tile utility (log-probability of the true class): what the oracle ranks by */
    utility(f, i) {
      const C = this.C, o = (f * this.P + i) * C;
      let m = -Infinity;
      for (let c = 0; c < C; c++) m = Math.max(m, this.logits[o + c]);
      let s = 0;
      for (let c = 0; c < C; c++) s += Math.exp(this.logits[o + c] - m);
      return this.logits[o + this.labels[f]] - m - Math.log(s);
    }

    /* Receiver CPU time for k tiles, interpolated from the measured curve (simulate.py [D]) */
    cpuMs(k) {
      if (k <= 0) return 0;
      const c = this.cpu;
      if (k <= c[0][0]) return c[0][1] * k / c[0][0];
      for (let j = 1; j < c.length; j++)
        if (k <= c[j][0]) return c[j - 1][1] + (c[j][1] - c[j - 1][1]) * (k - c[j - 1][0]) / (c[j][0] - c[j - 1][0]);
      return c[c.length - 1][1];
    }

    /* saep.protocol.encode_packet: 24 B header (!4sBBHIBBBBHHI) + int8 tag + payload */
    encodePacket(f, i) {
      const payload = this.tilePayload(f, i), tag = this.tag(f, i);
      const buf = new Uint8Array(HEADER + this.D + payload.length), v = new DataView(buf.buffer);
      buf.set(MAGIC, 0);
      v.setUint8(4, 2);
      v.setUint8(5, (i === 0 ? FLAG_FIRST : 0) | (i === this.P - 1 ? FLAG_LAST : 0));
      v.setUint16(6, HEADER + this.D);
      v.setUint32(8, f);
      v.setUint8(12, i); v.setUint8(13, this.P); v.setUint8(14, this.D);
      v.setUint8(15, 1); v.setUint16(16, this.d.scheme_id);
      v.setUint16(18, payload.length);
      v.setUint32(20, crc32(payload));
      buf.set(new Uint8Array(tag.buffer, tag.byteOffset, tag.length), HEADER);
      buf.set(payload, HEADER + this.D);
      return buf;
    }
  }

  /* saep.protocol.encode_control: !4sBBHiB */
  function encodeControl(budget, threshold) {
    const buf = new Uint8Array(SAEC_LEN), v = new DataView(buf.buffer);
    buf.set([0x53, 0x41, 0x45, 0x43], 0);
    v.setUint8(4, 2); v.setUint8(5, 0); v.setUint16(6, budget);
    v.setInt32(8, Math.max(threshold, -(2 ** 31))); v.setUint8(12, 0);
    return buf;
  }

  /* receiver.py ThresholdController */
  class ThresholdController {
    constructor(k) { this.k = k; this.scores = []; this.arrivals = []; this.tau = -(2 ** 31); }
    observe(s) { this.scores.push(s); if (this.scores.length > 2048) this.scores.shift(); }
    endFrame(arrived) {
      this.arrivals.push(arrived);
      if (this.arrivals.length > 64) this.arrivals.shift();
      const mean = this.arrivals.reduce((a, b) => a + b, 0) / this.arrivals.length;
      this.tau = this.scores.length < 64 || mean <= this.k ? -(2 ** 31) : quantileTrunc(this.scores, 1 - this.k / mean);
    }
  }

  /* run_demo.py presets: name, k, policy, loss, rate (fraction of 16*fps), sender feedback */
  const SCENARIOS = [
    ["clean link, no admission", 16, "fifo", 0.0, 0, false],
    ["CPU budget k=4, FIFO", 4, "fifo", 0.0, 0, false],
    ["CPU budget k=4, random", 4, "random", 0.0, 0, false],
    ["CPU budget k=4, int8 tag (XDP-able)", 4, "tag", 0.0, 0, false],
    ["CPU budget k=4, tag + context MLP", 4, "ctx", 0.0, 0, false],
    ["20% loss + k=4, FIFO", 4, "fifo", 0.2, 0, false],
    ["20% loss + k=4, tag + context MLP", 4, "ctx", 0.2, 0, false],
    ["bottleneck 30% rate, raster sender", 16, "fifo", 0.0, 0.3, false],
    ["bottleneck 30% rate, feedback sender (k=4)", 4, "fifo", 0.0, 0.3, true],
  ].map(([name, k, policy, loss, rate, feedback]) => ({ name, k, policy, loss, rate, feedback }));

  /* One run of sender -> link emulator -> receiver over the dataset's frames.
   * Frames are produced one at a time (next()), so the UI can animate them as they come. */
  class Simulation {
    constructor(ds, cfg) {
      this.ds = ds;
      this.cfg = Object.assign({ k: 4, policy: "ctx", loss: 0, rate: 0, feedback: false,
        fps: 25, queue: 8, seed: 1, prop: 0, saecHz: 10, frameTimeout: 0.15, earlyExit: false,
        linkTick: WINDOWS_TIMER_TICK, jitter: 0.0015 }, cfg);
      this.jitterRng = mulberry32(this.cfg.seed + 13);
      this.period = 1 / this.cfg.fps;
      this.pktRate = this.cfg.rate > 0 ? this.cfg.rate * ds.P * this.cfg.fps : 0;  // packets/s
      this.lossRng = mulberry32(this.cfg.seed);
      this.policyRng = mulberry32(this.cfg.seed + 7);
      this.ctl = new ThresholdController(this.cfg.k);
      this.pending = [];       // frames waiting for the receiver's timeout to close
      this.lastRecv = 0;
      this.f = 0;
      this.stats = { frames: 0, sent: 0, random_loss: 0, queue_drop: 0, received: 0, controlled_drop: 0,
        admitted: 0, correct: 0, empty: 0, cpu: 0 };
      this.transmit();
    }
    done() { return this.f >= this.ds.n; }

    /* sender.py + link_emulator.py for the whole run. Neither depends on what the receiver
     * admits (the SAEC budget is a fixed k), so the wire timeline is computed up front. */
    transmit() {
      const ds = this.ds, cfg = this.cfg, P = ds.P, prop = cfg.prop, rate = this.pktRate;
      const tick = cfg.linkTick, q = [];
      let tNext = 0, nextTx = 0, lastWake = 0, budget = P, saecNext = 0.02;
      // link_emulator.py main loop: transmit while now >= next_tx, then select() until data or timeout
      const serve = now => {
        while (q.length && (rate <= 0 || now >= nextTx)) {
          const p = q.shift();
          p.tDepart = now; p.tRecv = now + prop;
          nextTx = Math.max(nextTx, now) + (rate > 0 ? 1 / rate : 0);
        }
      };
      const timerWake = now => {  // select() timeout fires on the OS timer tick, not exactly at next_tx
        const dt = Math.max(0, nextTx - now);
        return now + (tick > 0 ? Math.max(1, Math.ceil(dt / tick - 1e-9)) * tick : Math.max(dt, 1e-9));
      };
      this.wire = [];
      for (let f = 0; f < ds.n; f++) {
        const tFrame = tNext;
        if (cfg.feedback)  // SAEC at 10 Hz; the sender drains its control socket at each frame start
          for (; saecNext <= tFrame; saecNext += 1 / cfg.saecHz) budget = cfg.k;
        let order = Array.from({ length: P }, (_, i) => i), senderScores = null;
        if (cfg.feedback) {  // sender sees the whole frame and sends only its top `budget`
          senderScores = ds.ctxScores(f, order);
          order = (ds.policy ? ds.selectDiverse(f, order, senderScores, P)
            : order.slice().sort((a, b) => senderScores[b] - senderScores[a] || a - b)).slice(0, budget);
        }
        const gap = this.period / order.length, pkts = [];
        for (const idx of order) {
          const p = { f, idx, tSend: tNext, tLink: tNext + prop };
          tNext += gap;
          while (q.length) {
            const w = timerWake(lastWake);
            if (w >= p.tLink) break;
            serve(w); lastWake = w;
          }
          p.qLen = q.length;
          if (this.lossRng() < cfg.loss) p.fate = "loss";
          else if (q.length >= cfg.queue) p.fate = "qdrop";
          else { p.fate = "ok"; q.push(p); }
          // OS scheduling delay before the loop runs; it only matters when the rate limiter holds packets
          lastWake = p.tLink + (rate > 0 ? this.jitterRng() * cfg.jitter : 0);
          serve(lastWake);
          pkts.push(p);
        }
        this.wire.push({ tFrame, order, pkts, senderScores });
      }
      while (q.length) { const w = timerWake(lastWake); serve(w); lastWake = w; }
    }

    next() {
      const ds = this.ds, cfg = this.cfg, f = this.f++;
      const { tFrame, order, pkts, senderScores } = this.wire[f];
      for (const p of pkts) {
        this.stats.sent++;
        if (p.fate === "loss") this.stats.random_loss++;
        else if (p.fate === "qdrop") this.stats.queue_drop++;
      }
      const arrived = pkts.filter(p => p.fate === "ok").sort((a, b) => a.tRecv - b.tRecv);
      this.stats.received += arrived.length;
      const rec = { f, label: ds.labels[f], tFrame, order, pkts, arrived: arrived.map(p => p.idx),
        senderScores, budget: cfg.k, senderBudget: order.length };
      this.receive(rec, arrived);
      rec.tClose = Math.max(tFrame + this.period, ...pkts.map(p => p.tRecv ?? p.tLink)) + 0.004;
      return rec;
    }

    receive(rec, arrived) {
      const ds = this.ds, cfg = this.cfg, f = rec.f, P = ds.P, k = cfg.k;
      let admitted = [];
      rec.scores = {};
      if (cfg.policy === "ctx") {  // buffer the frame, rescore in user space, keep top-k
        const idx = arrived.map(p => p.idx), s = ds.ctxScores(f, idx);
        idx.forEach((i, j) => (rec.scores[i] = s[j]));
        if (ds.policy) {  // v3: diversity-aware order, optionally early exit (receiver.py --diverse/--early-exit)
          const ranked = ds.selectDiverse(f, idx, s, P);
          admitted = cfg.earlyExit && ds.canExit(k) ? ds.earlyExit(f, ranked, k) : ranked.slice(0, k);
        } else {
          const keep = idx.map((_, j) => j).sort((a, b) => s[b] - s[a] || a - b).slice(0, k);
          admitted = keep.map(j => idx[j]);
        }
        for (const p of arrived) p.decision = admitted.includes(p.idx) ? "admit" : "cdrop";
      } else {  // per-packet early path: peek tag, score, threshold + quota
        for (const p of arrived) {
          // receiver.py closes these frames only on its 150 ms timeout, checked once per loop
          // iteration before recv(), so tau updates lag a few frames behind the arrivals
          while (this.pending.length && this.pending[0].tFirst + this.cfg.frameTimeout < this.lastRecv)
            this.ctl.endFrame(this.pending.shift().arrived);
          this.lastRecv = p.tRecv;
          let score = 0;
          if (cfg.policy === "tag") score = ds.tagScore(f, p.idx);
          else if (cfg.policy === "random") score = Math.floor(this.policyRng() * 65536);
          rec.scores[p.idx] = score;
          if (cfg.policy !== "fifo") this.ctl.observe(score);
          const quotaLeft = k - admitted.length, pktsLeft = P - p.idx, mustFill = pktsLeft <= quotaLeft;
          if (quotaLeft <= 0 || (cfg.policy !== "fifo" && score < this.ctl.tau && !mustFill)) p.decision = "cdrop";
          else { p.decision = "admit"; admitted.push(p.idx); }
        }
      }
      if (cfg.policy === "ctx") this.ctl.endFrame(arrived.length);
      else if (arrived.length) this.pending.push({ tFirst: arrived[0].tRecv, arrived: arrived.length });
      rec.tau = this.ctl.tau;
      const cdrop = arrived.length - admitted.length;
      this.stats.controlled_drop += cdrop;
      this.stats.admitted += admitted.length;
      this.stats.frames++;
      rec.admitted = admitted;
      Object.assign(rec, ds.classify(f, admitted));
      rec.cpuMs = ds.cpuMs(admitted.length);
      this.stats.cpu += rec.cpuMs;
      if (!admitted.length) this.stats.empty++;
      if (rec.correct) this.stats.correct++;
    }

    runAll() { while (!this.done()) this.next(); return this.summary(); }

    summary() {
      const s = this.stats, n = Math.max(s.frames, 1), expected = s.frames * this.ds.P;
      return { accuracy: s.correct / n, sent: s.sent, received: s.received,
        uncontrolled_loss: 1 - s.received / Math.max(expected, 1), random_loss: s.random_loss,
        queue_drop: s.queue_drop, controlled_drops: s.controlled_drop, admitted_per_frame: s.admitted / n,
        cpu_ms: s.cpu / n, frames: s.frames, empty: s.empty };
    }
  }

  /* Same arrivals, every policy's top-k pick (simulate.py select_topk semantics) */
  function comparePolicies(ds, f, arrivedIdx, k, rng) {
    const pickBy = s => arrivedIdx.map((i, j) => [i, s[j], j]).sort((a, b) => b[1] - a[1] || a[2] - b[2])
      .slice(0, k).map(x => x[0]);
    const ctx = ds.ctxScores(f, arrivedIdx);
    const picks = {
      fifo: arrivedIdx.slice(0, k),
      random: pickBy(arrivedIdx.map(() => rng())),
      tag: pickBy(arrivedIdx.map(i => ds.tagScore(f, i))),
      ctx: ds.policy ? ds.selectDiverse(f, arrivedIdx, ctx, k) : pickBy(ctx),
      oracle: pickBy(arrivedIdx.map(i => ds.utility(f, i))),
    };
    const out = {};
    for (const [name, adm] of Object.entries(picks)) out[name] = { admitted: adm, ...ds.classify(f, adm) };
    return out;
  }

  /* run_demo.py --v3 adds two early-exit rows */
  const EXIT_SCENARIOS = [
    { name: "CPU budget avg 4, context MLP + early exit", k: 4, policy: "ctx", loss: 0.0, rate: 0, feedback: false, earlyExit: true },
    { name: "20% loss + avg 4, context MLP + early exit", k: 4, policy: "ctx", loss: 0.2, rate: 0, feedback: false, earlyExit: true },
  ];
  const scenarios = ds => ds.policy ? [...SCENARIOS, ...EXIT_SCENARIOS] : SCENARIOS;

  root.SAEPEngine = { Dataset, Simulation, ThresholdController, SCENARIOS, scenarios, comparePolicies, crc32,
    encodeControl, mulberry32, HEADER, SAEC_LEN, WINDOWS_TIMER_TICK };
})(typeof globalThis !== "undefined" ? globalThis : window);
