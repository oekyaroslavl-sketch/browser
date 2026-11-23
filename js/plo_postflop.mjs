// Ergonomic JavaScript / WebAssembly wrapper for `plo_wasm.wasm` (`wasm32-unknown-unknown`).
// Works identically in Browser WebWorkers (`fetch` / `ArrayBuffer`) and Node.js (`fs.readFileSync`).

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8");

export class PloPostflopWasm {
  /**
   * @param {WebAssembly.Instance} instance
   */
  constructor(instance) {
    this.instance = instance;
    this.exports = instance.exports;
  }

  /**
   * Instantiates `plo_wasm.wasm` from a Uint8Array / ArrayBuffer / Buffer.
   * @param {BufferSource} wasmBytes
   * @returns {Promise<PloPostflopWasm>}
   */
  static async fromBytes(wasmBytes) {
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    return new PloPostflopWasm(instance);
  }

  _memU8() {
    return new Uint8Array(this.exports.memory.buffer);
  }

  _withUtf8(str, fn) {
    const bytes = encoder.encode(str);
    if (bytes.length === 0) {
      return fn(0, 0);
    }
    const ptr = this.exports.plo_alloc(bytes.length);
    try {
      this._memU8().set(bytes, ptr);
      return fn(ptr, bytes.length);
    } finally {
      this.exports.plo_dealloc(ptr, bytes.length);
    }
  }

  _withMultiUtf8(strings, fn) {
    const allocs = [];
    try {
      for (const s of strings) {
        const bytes = encoder.encode(s);
        if (bytes.length === 0) {
          allocs.push([0, 0]);
        } else {
          const ptr = this.exports.plo_alloc(bytes.length);
          this._memU8().set(bytes, ptr);
          allocs.push([ptr, bytes.length]);
        }
      }
      return fn(allocs);
    } finally {
      for (const [ptr, len] of allocs) {
        if (len > 0) {
          this.exports.plo_dealloc(ptr, len);
        }
      }
    }
  }

  _readOutputString() {
    const ptr = this.exports.plo_output_ptr();
    const len = this.exports.plo_output_len();
    return decoder.decode(this._memU8().subarray(ptr, ptr + len));
  }

  _readErrorString() {
    const ptr = this.exports.plo_error_ptr();
    const len = this.exports.plo_error_len();
    return decoder.decode(this._memU8().subarray(ptr, ptr + len));
  }

  _checkCode(code) {
    if (code < 0) {
      throw new Error(this._readErrorString() || `WASM error (${code})`);
    }
    return code;
  }

  reset() {
    this.exports.plo_session_reset();
  }

  /**
   * Evaluates a 4-card PLO hand (`[c0, c1, c2, c3]` card ids in `0..52`) on a board (`3..5` card ids).
   * Returns exact PLO 2+3 equivalence class in `1..=7462` (higher = stronger).
   */
  evalHandRaw(hole4Ids, boardIds) {
    if (hole4Ids.length !== 4) {
      throw new Error("PLO requires exactly 4 hole cards");
    }
    const ptr = this.exports.plo_alloc(boardIds.length);
    try {
      this._memU8().set(boardIds, ptr);
      const rank = this.exports.plo_eval_hand_raw(
        hole4Ids[0],
        hole4Ids[1],
        hole4Ids[2],
        hole4Ids[3],
        ptr,
        boardIds.length
      );
      if (rank < 0) {
        throw new Error("Invalid or overlapping PLO cards");
      }
      return rank;
    } finally {
      this.exports.plo_dealloc(ptr, boardIds.length);
    }
  }

  /**
   * Computes exact blocker-aware PLO equity between two 4-card hands or two PLO range strings on `board`.
   */
  computeEquity(rangeOrHand0, rangeOrHand1, boardStr) {
    return this._withMultiUtf8([rangeOrHand0, rangeOrHand1, boardStr], ([[p0, l0], [p1, l1], [pb, lb]]) => {
      this._checkCode(this.exports.plo_compute_equity_json(p0, l0, p1, l1, pb, lb));
      return JSON.parse(this._readOutputString());
    });
  }

  setBoard(boardStr) {
    return this._withUtf8(boardStr, (ptr, len) => this._checkCode(this.exports.plo_set_board(ptr, len)));
  }

  setRange(player, rangeStr) {
    return this._withUtf8(rangeStr, (ptr, len) =>
      this._checkCode(this.exports.plo_set_range(player, ptr, len))
    );
  }

  setTreeConfig({
    startingPot = 100,
    effectiveStack = 100,
    maxBetsPerStreet = 4,
    oopBet = "50%, 100%",
    oopRaise = "100%",
    ipBet = "50%, 100%",
    ipRaise = "100%",
  } = {}) {
    return this._withMultiUtf8([oopBet, oopRaise, ipBet, ipRaise], ([[obP, obL], [orP, orL], [ibP, ibL], [irP, irL]]) => {
      this._checkCode(
        this.exports.plo_set_tree_config(
          startingPot,
          effectiveStack,
          maxBetsPerStreet,
          obP,
          obL,
          orP,
          orL,
          ibP,
          ibL,
          irP,
          irL
        )
      );
    });
  }

  setAbstractionBuckets(flopBuckets = 0, turnBuckets = 0, riverBuckets = 0) {
    this.exports.plo_set_abstraction_buckets(flopBuckets, turnBuckets, riverBuckets);
  }

  setMemoryBudgetMb(maxMb) {
    this.exports.plo_set_memory_budget_mb(maxMb);
  }

  buildGame() {
    this._checkCode(this.exports.plo_build_game());
    return JSON.parse(this._readOutputString());
  }

  stepIterations(numSteps = 25) {
    return this._checkCode(this.exports.plo_step_iterations(numSteps));
  }

  solve(maxIterations = 300, targetExploitability = 0.25) {
    const expl = this.exports.plo_solve(maxIterations, targetExploitability);
    if (Number.isNaN(expl)) {
      throw new Error(this._readErrorString());
    }
    return expl;
  }

  exploitability() {
    const expl = this.exports.plo_exploitability();
    if (Number.isNaN(expl)) {
      throw new Error(this._readErrorString());
    }
    return expl;
  }

  backToRoot() {
    this._checkCode(this.exports.plo_back_to_root());
  }

  playAction(actionIdx) {
    this._checkCode(this.exports.plo_play_action(actionIdx));
  }

  playChance(cardStr) {
    this._withUtf8(cardStr, (ptr, len) => this._checkCode(this.exports.plo_play_chance(ptr, len)));
  }

  queryNode() {
    this._checkCode(this.exports.plo_query_node_json());
    return JSON.parse(this._readOutputString());
  }

  wasmMemoryBytes() {
    return this.exports.memory.buffer.byteLength;
  }
}
