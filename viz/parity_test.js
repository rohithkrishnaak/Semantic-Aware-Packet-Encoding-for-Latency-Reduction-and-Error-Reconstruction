/* Checks the browser engine against Python: node viz/parity_test.js [data_v3.js]
 * Run python scripts/export_viz.py [--artifacts artifacts/v3] first. Exits non-zero on any mismatch. */
"use strict";
const path = require("path");
global.window = global;
require(path.join(__dirname, process.argv[2] || "data.js"));
require(path.join(__dirname, "engine.js"));

const { Dataset, Simulation, scenarios } = SAEPEngine;
const D = window.SAEP_DATA, ds = new Dataset(D), ref = D.ref;
let failed = 0;
function check(name, ok, detail) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failed++;
}
const maxDiff = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));

// 1. wire format: byte-identical to saep.protocol.encode_packet
const hex = Buffer.from(ds.encodePacket(3, 5)).toString("hex");
check("SAEP v2 packet bytes (frame 3, tile 5, 248 B, CRC32)", hex === ref.packet_f3_i5, `${hex.length / 2} B`);

// 2. scorers
const all = [...Array(ds.P).keys()];
const tagS = all.map(i => ds.tagScore(0, i));
check("int8 tag scorer (XDP arithmetic), frame 0", maxDiff(tagS, ref.tag_scores_f0) === 0);
const d1 = maxDiff(ds.ctxScores(0, all), ref.ctx_scores_f0);
check("context MLP scores, full frame 0", d1 < 1e-4, `max |diff| ${d1.toExponential(1)}`);
const part = all.filter(i => i % 3 !== 0);
const d2 = maxDiff(ds.ctxScores(0, part), part.map(i => ref.ctx_scores_partial_f0[i]));
check("context MLP scores, frame 0 with tiles 0,3,6,9,12,15 lost", d2 < 1e-4, `max |diff| ${d2.toExponential(1)}`);

// 3. offline accuracy curves (all packets arrive, top-k) on the same 300 frames
for (const pol of ["fifo", "tag", "ctx"]) {
  const js = [];
  for (let k = 1; k <= ds.P; k++) {
    let ok = 0;
    for (let f = 0; f < ds.n; f++) {
      const s = pol === "fifo" ? all.map(i => -i) : pol === "tag" ? all.map(i => ds.tagScore(f, i)) : ds.ctxScores(f, all);
      const adm = all.slice().sort((a, b) => s[b] - s[a] || a - b).slice(0, k);
      ok += ds.classify(f, adm).correct;
    }
    js.push(ok / ds.n);
  }
  check(`offline accuracy vs k, ${pol}`, maxDiff(js, ref.curves[pol]) < 1e-9,
    `k=4: js ${js[3].toFixed(3)} py ${ref.curves[pol][3].toFixed(3)}`);
}

// 3b. v3 selection rules: diversity order and early exit, vs saep.models
if (ds.policy) {
  const all16 = [...Array(ds.P).keys()];
  check("diversity order, frame 0 (select_diverse)", JSON.stringify(ds.selectDiverse(0, all16, ds.ctxScores(0, all16), ds.P)) === JSON.stringify(ref.div_order_f0));
  let okd = 0, oke = 0, sameUsed = 0;
  for (let f = 0; f < ds.n; f++) {
    const ranked = ds.selectDiverse(f, all16, ds.ctxScores(f, all16), ds.P);
    okd += ds.classify(f, ranked.slice(0, 4)).correct;
    const ex = ds.earlyExit(f, ranked, 4);
    oke += ds.classify(f, ex).correct;
    sameUsed += ex.length === ref.exit_used_k4[f];
  }
  check("diversity top-4 accuracy", Math.abs(okd / ds.n - ref.div_acc_k4) < 1e-9, `js ${(okd / ds.n).toFixed(3)} py ${ref.div_acc_k4.toFixed(3)}`);
  check("early exit: tiles used per frame", sameUsed === ds.n, `${sameUsed}/${ds.n} frames identical`);
  check("early exit accuracy (avg budget 4)", Math.abs(oke / ds.n - ref.exit_acc_k4) < 1e-9, `js ${(oke / ds.n).toFixed(3)} py ${ref.exit_acc_k4.toFixed(3)}`);
}

// 4. full pipeline vs the measured UDP run (demo_results.json)
console.log("\nscenario                                     UDP (real sockets)   JS engine");
for (const sc of scenarios(ds)) {
  const js = new Simulation(ds, sc).runAll(), udp = D.measured.udp.find(r => r.name === sc.name);
  const deterministic = sc.loss === 0 && sc.rate === 0 && sc.policy !== "random";
  const diff = Math.abs(js.accuracy - udp.accuracy);
  const ok = deterministic ? diff < 1e-9 : diff < 0.05;
  console.log(`${ok ? "PASS" : "FAIL"}  ${sc.name.padEnd(44)} ${(udp.accuracy * 100).toFixed(1).padStart(5)}%` +
    `   ${(js.accuracy * 100).toFixed(1).padStart(5)}%  ${deterministic ? "(must match exactly)" : "(stochastic, within 5 pts)"}`);
  if (!ok) failed++;
}
console.log(failed ? `\n${failed} check(s) failed` : "\nall checks passed");
process.exit(failed ? 1 : 0);
