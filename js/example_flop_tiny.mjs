import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { PloPostflopWasm } from "./plo_postflop.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const wasmPath = path.resolve(
  __dirname,
  "../target/wasm32-unknown-unknown/release/plo_wasm.wasm"
);

const wasmBytes = fs.readFileSync(wasmPath);
const solver = await PloPostflopWasm.fromBytes(wasmBytes);

// Part A1/A3: Same fixed iteration count as `crates/plo_cfr/examples/bench_flop_tiny.rs`!
const FIXED_FLOP_TINY_ITERS = 100;

// Measured native 100-iteration baseline from `bench_flop_tiny`:
const NATIVE_100_EV_OOP = 48.780203512;
const NATIVE_100_EV_IP = 51.219796488;
const NATIVE_100_EXPLOIT = 0.006743357;

console.log(
  `=== WASM Flop Tiny Microbenchmark (N=${FIXED_FLOP_TINY_ITERS}, 20 vs 20, Two-Tone Ks9s4d, 2 Sizes, iso=true) ===`
);

solver.reset();
const boardLen = solver.setBoard("Ks9s4d");
assert.equal(boardLen, 3);

// Exact 20 vs 20 suit-symmetric live PLO-4 combos on `Ks9s4d` (`{h, c}` isomorphic -> 34,880 decision nodes vs 57,240 without iso)
const oopStr = "AAQQ$ds, JJTT$ds, 8877$ds, AsJsQhTh, AsJsQcTc";
const ipStr = "AAJJ$ds, QQTT$ds, 7766$ds, AsQsJhTh, AsQsJcTc";
const nOop = solver.setRange(0, oopStr);
const nIp = solver.setRange(1, ipStr);
assert.equal(nOop, 20);
assert.equal(nIp, 20);

// Enforce strict 64 MiB memory guardrail (`21.02 MiB` actual < `64 MiB`)
solver.setMemoryBudgetMb(64);
solver.setTreeConfig({
  startingPot: 100,
  effectiveStack: 50,
  maxBetsPerStreet: 4,
  oopBet: "25%, 50%",
  oopRaise: "100%",
  ipBet: "25%, 50%",
  ipRaise: "100%",
});

const tBuild = performance.now();
const memReport = solver.buildGame();
const buildMs = performance.now() - tBuild;

console.log("Memory Report:", memReport);
assert.equal(memReport.decisionNodes, 34880);
assert(
  memReport.totalEstimatedBytes < 64 * 1024 * 1024,
  `Expected < 64 MiB, got ${memReport.totalEstimatedBytes}`
);

const expl0 = solver.exploitability();
console.log(
  `Built in ${buildMs.toFixed(1)} ms | Estimated=${(
    memReport.totalEstimatedBytes /
    (1024 * 1024)
  ).toFixed(2)} MiB | Initial exploitability=${expl0.toExponential(6)} chips`
);

const tSolve = performance.now();
const iters = solver.stepIterations(FIXED_FLOP_TINY_ITERS);
const solveSecs = (performance.now() - tSolve) / 1000.0;
assert.equal(iters, FIXED_FLOP_TINY_ITERS);

const expl100 = solver.exploitability();
const root = solver.queryNode();
const wasmLinearMib = solver.wasmMemoryBytes() / (1024 * 1024);

const deltaEvOop = Math.abs(root.rootEv[0] - NATIVE_100_EV_OOP);
const deltaEvIp = Math.abs(root.rootEv[1] - NATIVE_100_EV_IP);
const deltaExpl = Math.abs(expl100 - NATIVE_100_EXPLOIT);

console.log(
  `After ${FIXED_FLOP_TINY_ITERS} WASM iters: ${solveSecs.toFixed(2)} s (${(
    FIXED_FLOP_TINY_ITERS / solveSecs
  ).toFixed(2)} iters/s)`
);
console.log(
  `  WASM   (N=100) : exploitability=${expl100.toExponential(6)} chips | root EV=[${root.rootEv[0].toFixed(
    9
  )}, ${root.rootEv[1].toFixed(9)}] | Peak WASM Linear Memory=${wasmLinearMib.toFixed(2)} MiB`
);
console.log(
  `  Native (N=100) : exploitability=${NATIVE_100_EXPLOIT.toExponential(
    6
  )} chips | root EV=[${NATIVE_100_EV_OOP.toFixed(9)}, ${NATIVE_100_EV_IP.toFixed(9)}]`
);
console.log(
  `  Same-N Deltas  : |ΔEV_oop|=${deltaEvOop.toExponential(3)} chips, |ΔEV_ip|=${deltaEvIp.toExponential(
    3
  )} chips, |Δexploit|=${deltaExpl.toExponential(3)} chips`
);

// Part A3 assertions:
assert(deltaEvOop < 0.05, `|EV_oop_wasm - EV_oop_native_100iters| = ${deltaEvOop} >= 0.05`);
assert(deltaEvIp < 0.05, `|EV_ip_wasm - EV_ip_native_100iters| = ${deltaEvIp} >= 0.05`);
assert(deltaExpl < 0.02, `|exploit_wasm - exploit_native| = ${deltaExpl} >= 0.02`);
assert(expl100 < expl0, `Expected exploitability to drop (${expl0} -> ${expl100})`);
assert(Math.abs(root.rootEv[0] + root.rootEv[1] - 100.0) <= 1e-9);
assert.deepEqual(root.actions, ["Check:0", "Bet:25", "Allin:50"]);

console.log("WASM FLOP TINY SAME-N (N=100) CONTRACT PASSED WITHOUT OOM!");
