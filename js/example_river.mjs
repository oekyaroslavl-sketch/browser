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

console.log("=== 1. Exact PLO Equity Check (Flop Set vs Monster Wrap+FD) ===");
const flopEq = solver.computeEquity("AhAd7c2s", "QhJhTd9s", "AsTc8h");
console.log("AhAd7c2s vs QhJhTd9s on AsTc8h:", flopEq);
assert(flopEq.validWeightSum === 820, "C(41,2) = 820 runouts expected");
assert(Math.abs(flopEq.equityP0 + flopEq.equityP1 - 1.0) < 1e-6);

console.log("\n=== 2. Realistic PLO River Solve in WASM ===");
// Board: Ks Qs Td 7h 2c (Broadway straight AJ/J9, sets KK/QQ/TT/77, Kings-up, missed flush draws)
// OOP: Nut straights (AJ98), sets of Tens, two-pair bluff-catchers, and missed spade/heart draws
// IP: Nut straights (AJT9), top set Kings (KKJJ$ds), middle sets Queens (QQ99$ds), and missed draws
solver.reset();
const boardLen = solver.setBoard("KsQsTd7h2c");
assert.equal(boardLen, 5);

const nOop = solver.setRange(0, "(AJ)(98), TT99$ds, AAJJ$ds, AhJh8d6d, 9s8s5d4d");
const nIp = solver.setRange(1, "(AJ)(65), KKJJ$ds, QQ99$ds, AdQd8c6c, 8s6s5h4h");
console.log(`Active unblocked combos -> OOP: ${nOop}, IP: ${nIp}`);

solver.setTreeConfig({
  startingPot: 100,
  effectiveStack: 200, // 2.0 SPR Pot-Limit Omaha river!
  maxBetsPerStreet: 4,
  oopBet: "50%, 100%",
  oopRaise: "100%",
  ipBet: "50%, 100%",
  ipRaise: "100%",
});

const memReport = solver.buildGame();
console.log("Game tree memory report:", memReport);

const t0 = performance.now();
const expl = solver.solve(400, 0.05);
const dt = performance.now() - t0;
console.log(
  `Solved in ${dt.toFixed(1)} ms | Exploitability: ${expl.toFixed(4)} chips (${expl.toFixed(2)}% of 100 pot)`
);
console.log(
  `Peak WASM linear memory: ${(solver.wasmMemoryBytes() / (1024 * 1024)).toFixed(2)} MB`
);

assert(expl < 0.25, `Expected low exploitability, got ${expl}`);

const rootNode = solver.queryNode();
console.log("Root actions (Pot-Limit capped at 100, never 200 all-in!):", rootNode.actions);
// Verify Pot-Limit invariant: effective stack is 200, pot is 100 -> max opening bet MUST be Bet:100!
assert.deepEqual(rootNode.actions, ["Check:0", "Bet:50", "Bet:100"]);
assert(Math.abs(rootNode.rootEv[0] + rootNode.rootEv[1] - 100.0) < 0.05);

// Verify every hand's action probabilities sum to 1.0
const numActions = rootNode.actions.length;
const numHands = rootNode.oopHands.length;
for (let h = 0; h < numHands; h++) {
  let sum = 0;
  for (let a = 0; a < numActions; a++) {
    sum += rootNode.strategy[a * numHands + h];
  }
  assert(Math.abs(sum - 1.0) < 1e-4, `Strategy sum mismatch for hand ${h}: ${sum}`);
}

console.log("\nSample OOP Root Strategies:");
for (let h = 0; h < numHands; h += Math.max(1, Math.floor(numHands / 8))) {
  const handInfo = rootNode.oopHands[h];
  const probs = rootNode.actions
    .map((act, a) => `${act}=${(100 * rootNode.strategy[a * numHands + h]).toFixed(1)}%`)
    .join(", ");
  console.log(
    `  ${handInfo.hand} | eq=${(100 * rootNode.equityOop[h]).toFixed(1)}% | EV=${rootNode.evOop[h].toFixed(2)} | ${probs}`
  );
}

// Navigate OOP Bet:100 -> inspect IP's Pot-Limit response (`Fold:0`, `Call:0`, `Allin:200` since pot raise is 100 + 300 = 400 >= 200 stack)
solver.playAction(2);
const ipVsPot = solver.queryNode();
console.log("\nIP actions facing OOP Pot Bet (100):", ipVsPot.actions);
assert.deepEqual(ipVsPot.actions, ["Fold:0", "Call:0", "Allin:200"]);

console.log("\n=== 3. Turn Solve with Chance Card Navigation in WASM ===");
solver.reset();
solver.setBoard("AsKs7c2c");
solver.setRange(0, "AAQQ$ds, (QJ)(T9)");
solver.setRange(1, "KKJJ$ds, (QT)(98)");
solver.setTreeConfig({
  startingPot: 100,
  effectiveStack: 100,
  maxBetsPerStreet: 3,
  oopBet: "100%",
  oopRaise: "100%",
  ipBet: "100%",
  ipRaise: "100%",
});
const turnMem = solver.buildGame();
const turnExpl = solver.solve(100, 0.25);
console.log(
  `Turn solve (${turnMem.decisionNodes} decision nodes) -> Exploitability: ${turnExpl.toFixed(3)} chips`
);
solver.playAction(0); // OOP Check
solver.playAction(0); // IP Check -> River Chance Node!
const chanceNode = solver.queryNode();
console.log(
  `Chance node type=${chanceNode.nodeType}, representative river deals=${chanceNode.chanceCards.length} (suit-folded from 48)`
);
assert.equal(chanceNode.nodeType, "chance");
solver.playChance(chanceNode.chanceCards[0]);
const riverAfterDeal = solver.queryNode();
console.log(`Dealt ${chanceNode.chanceCards[0]} -> board=${riverAfterDeal.board}, actions=${riverAfterDeal.actions.join("/")}`);
assert.equal(riverAfterDeal.nodeType, "oop");
assert.equal(riverAfterDeal.street, "river");
assert.equal(riverAfterDeal.player, 0);

// Also deal a FOLDED heart card (`7h`) to prove WASM `playChance` + `queryNode` supports all 48 real cards!
solver.backToRoot();
solver.playAction(0);
solver.playAction(0);
assert.equal(chanceNode.allChanceCards.length, 48);
solver.playChance("7h");
const riverAfterFoldedHeart = solver.queryNode();
assert.equal(riverAfterFoldedHeart.board, "AsKs7c2c7h");
assert.equal(riverAfterFoldedHeart.street, "river");

console.log("\n=== 4. Frozen JS API Contract: Exact Polar River Equilibrium & WASM Determinism ===");
// Frozen contract iteration count: `FIXED_POLAR_ITERATIONS = 1000` pure DCFR steps (`stepIterations(1000)`, zero early-exit target).
// Analytical Nash equilibrium on `KsQsJs7d2c` (`Pot = 100`, `Stack = 100`, `100%` Pot bet):
// - OOP `{AsTs4h3h (Nuts), Ah9s6d5d (PLO bare-9s Air)}` vs IP `{KdQd4c3c (Two-Pair Bluff-Catcher)}`:
//   * Root EV: `EV_OOP = 75.0`, `EV_IP = 25.0` (verified within `<= 0.05` chips)
//   * OOP Nut bet frequency: `1.00`, OOP Air bluff frequency: `0.50` (verified within `<= 0.02`)
//   * IP Bluff-Catcher call frequency vs Allin:100: `0.50` (verified within `<= 0.02`)
//   * Repeated WASM runs at `FIXED_POLAR_ITERATIONS = 1000`: EV & strategy agree within `<= 1e-9`!
const FIXED_POLAR_ITERATIONS = 1000;

function runPolarRiverSession() {
  solver.reset();
  solver.setBoard("KsQsJs7d2c");
  solver.setRange(0, "AsTs4h3h, Ah9s6d5d");
  solver.setRange(1, "KdQd4c3c");
  solver.setTreeConfig({
    startingPot: 100,
    effectiveStack: 100,
    maxBetsPerStreet: 4,
    oopBet: "100%",
    oopRaise: "100%",
    ipBet: "100%",
    ipRaise: "100%",
  });
  solver.buildGame();
  const iters = solver.stepIterations(FIXED_POLAR_ITERATIONS);
  assert.equal(iters, FIXED_POLAR_ITERATIONS);
  const expl = solver.exploitability();
  const root = solver.queryNode();
  solver.playAction(1); // OOP Allin:100
  const ipNode = solver.queryNode();
  return { expl, root, ipNode };
}

const pRun1 = runPolarRiverSession();
const pRun2 = runPolarRiverSession();

// Verify frozen `queryNode()` schema (`street`, `player`, `pot`, `actions`, `strategy`, `ev`, `rootEv`, `iterations`)
assert.equal(pRun1.root.iterations, FIXED_POLAR_ITERATIONS);
assert.equal(pRun1.root.street, "river");
assert.equal(pRun1.root.player, 0);
assert.equal(pRun1.root.pot, 100);
assert.deepEqual(pRun1.root.actions, ["Check:0", "Allin:100"]);
assert.equal(pRun1.ipNode.street, "river");
assert.equal(pRun1.ipNode.player, 1);
assert.equal(pRun1.ipNode.pot, 200);
assert.deepEqual(pRun1.ipNode.actions, ["Fold:0", "Call:0"]);

// Closed-form Nash assertions (`75.0 / 25.0` EV within `0.05`, Air bluffs `0.50` within `0.02`, Catcher calls `0.50` within `0.02`)
assert(pRun1.expl < 0.05, `Expected polar exploitability < 0.05, got ${pRun1.expl}`);
assert(Math.abs(pRun1.root.rootEv[0] - 75.0) <= 0.05, `OOP EV=${pRun1.root.rootEv[0]} outside 75.0 ± 0.05`);
assert(Math.abs(pRun1.root.rootEv[1] - 25.0) <= 0.05, `IP EV=${pRun1.root.rootEv[1]} outside 25.0 ± 0.05`);

const nutIdx = pRun1.root.oopHands.findIndex((h) => h.hand === "AsTs4h3h");
const airIdx = pRun1.root.oopHands.findIndex((h) => h.hand === "Ah9s6d5d");
const nutBetFreq = pRun1.root.strategy[2 + nutIdx];
const airBluffFreq = pRun1.root.strategy[2 + airIdx];
const catcherCallFreq = pRun1.ipNode.strategy[1];

assert(Math.abs(nutBetFreq - 1.0) <= 0.01, `Nut bet freq=${nutBetFreq}`);
assert(Math.abs(airBluffFreq - 0.5) <= 0.02, `Air bluff freq=${airBluffFreq} outside 0.50 ± 0.02`);
assert(Math.abs(catcherCallFreq - 0.5) <= 0.02, `Catcher call freq=${catcherCallFreq} outside 0.50 ± 0.02`);
assert(Math.abs(pRun1.root.ev[nutIdx] - 150.0) <= 0.05, `Nut EV=${pRun1.root.ev[nutIdx]}`);
assert(Math.abs(pRun1.root.ev[airIdx] - 0.0) <= 0.05, `Air EV=${pRun1.root.ev[airIdx]}`);

// WASM Determinism after fixed `FIXED_POLAR_ITERATIONS = 1000`: EV and strategy agree within `<= 1e-9`!
assert(Math.abs(pRun1.expl - pRun2.expl) <= 1e-9);
assert(Math.abs(pRun1.root.rootEv[0] - pRun2.root.rootEv[0]) <= 1e-9);
assert(Math.abs(pRun1.root.rootEv[1] - pRun2.root.rootEv[1]) <= 1e-9);
for (let i = 0; i < pRun1.root.ev.length; i++) {
  assert(Math.abs(pRun1.root.ev[i] - pRun2.root.ev[i]) <= 1e-9);
}
for (let i = 0; i < pRun1.root.strategy.length; i++) {
  assert(Math.abs(pRun1.root.strategy[i] - pRun2.root.strategy[i]) <= 1e-9);
}
assert.deepEqual(pRun1.ipNode.strategy, pRun2.ipNode.strategy);
console.log(
  `Polar River (${FIXED_POLAR_ITERATIONS} fixed iters) verified: EV=[${pRun1.root.rootEv[0].toFixed(4)}, ${pRun1.root.rootEv[1].toFixed(4)}], AirBluff=${(100 * airBluffFreq).toFixed(2)}%, CatcherCall=${(100 * catcherCallFreq).toFixed(2)}%, Determinism<=1e-9`
);

// Step 4 WASM check: Repeated 2D-bucketed (`setAbstractionBuckets(8, 8, 8)`) solves in WASM agree within `<= 1e-9` on EV & strategy!
function runBucketed2dWasmSession(oopRangeStr, ipRangeStr) {
  solver.reset();
  solver.setBoard("KsQs4d2c");
  solver.setRange(0, oopRangeStr);
  solver.setRange(1, ipRangeStr);
  solver.setAbstractionBuckets(8, 8, 8);
  solver.setTreeConfig({
    startingPot: 100,
    effectiveStack: 100,
    maxBetsPerStreet: 3,
    oopBet: "50%, 100%",
    oopRaise: "100%",
    ipBet: "50%, 100%",
    ipRaise: "100%",
  });
  solver.buildGame();
  solver.stepIterations(60);
  return solver.queryNode();
}

const bWasm1 = runBucketed2dWasmSession(
  "AA99$ds, (QJ)(98), As9h8d7c, Ah9c8d7h, 6655$ds",
  "AAKK$ds, (JT)(98), KdQd6c5c, AsTs6h5h"
);
const bWasm2 = runBucketed2dWasmSession(
  "6655$ds, Ah9c8d7h, As9h8d7c, (QJ)(98), AA99$ds",
  "AsTs6h5h, KdQd6c5c, (JT)(98), AAKK$ds"
);
assert(Math.abs(bWasm1.rootEv[0] - bWasm2.rootEv[0]) <= 1e-9);
assert(Math.abs(bWasm1.rootEv[1] - bWasm2.rootEv[1]) <= 1e-9);
for (let i = 0; i < bWasm1.ev.length; i++) {
  assert(Math.abs(bWasm1.ev[i] - bWasm2.ev[i]) <= 1e-9);
}
for (let i = 0; i < bWasm1.strategy.length; i++) {
  assert(Math.abs(bWasm1.strategy[i] - bWasm2.strategy[i]) <= 1e-9);
}
console.log(
  `2D-Bucketed WASM determinism verified: EV=[${bWasm1.rootEv[0].toFixed(4)}, ${bWasm1.rootEv[1].toFixed(4)}] (<= 1e-9 across reordered inputs)`
);

console.log("\nALL WASM END-TO-END CHECKS PASSED!");
