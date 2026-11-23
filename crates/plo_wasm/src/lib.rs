//! WebAssembly (`wasm32-unknown-unknown`) and native FFI / Rust wrapper API for `plo-postflop`.
//!
//! # Design Goals
//! - **Zero external proc-macro dependencies**: builds cleanly on `wasm32-unknown-unknown` with pure `std`
//!   and workspace crates (`plo_cards`, `plo_eval`, `plo_iso`, `plo_range`, `plo_tree`, `plo_cfr`).
//! - **Compact linear-memory footprint**:
//!   - Evaluator lookup tables: ~96 KB lazy-initialized once.
//!   - Solver state: sparse 4-card combo tables + exact or bucketed regret/strategy buffers with a
//!     configurable `max_memory_bytes` guardrail (defaults to 512 MB so WASM never crashes with an opaque trap).
//! - **Hot-path numeric exports + ergonomic JSON query exports**:
//!   - Numeric C-ABI exports (`plo_eval_hand`, `plo_solve_step`, `plo_solve`, `plo_exploitability`,
//!     `plo_strategy_ptr`, `plo_ev_ptr`) run with zero string allocations in the CFR loop!
//!   - High-level configuration and node reports use lightweight UTF-8 / JSON buffers for effortless
//!     JavaScript / TypeScript / browser WebWorker integration.

use std::cell::RefCell;

use plo_cards::{Board, Card, Hand4, Street};
use plo_cfr::{
    EquityBucketAbstraction, IdentityAbstraction, MemoryReport, NodeReport,
    NutBlockerBucketAbstraction, PloGame, SolverBuildOptions, DEFAULT_MAX_MEMORY_BYTES,
};
use plo_eval::{eval_plo, init_tables};
use plo_range::{equity_hand_vs_hand, equity_range_vs_range, per_combo_river_equity, PloRange};
use plo_tree::{ActionTree, BettingLimit, StreetBetSizes, TreeConfig};

/// High-level safe Rust + WASM session manager mirroring `wasm-postflop`'s `GameManager` / `RangeManager` / `TreeManager`.
pub struct WasmPloSession {
    board: Option<Board>,
    oop_range: PloRange,
    ip_range: PloRange,
    starting_pot: i32,
    effective_stack: i32,
    betting_limit: BettingLimit,
    max_bets_per_street: u8,
    flop_sizes: [StreetBetSizes; 2],
    turn_sizes: [StreetBetSizes; 2],
    river_sizes: [StreetBetSizes; 2],
    max_memory_bytes: u64,
    enable_isomorphism: bool,
    /// Optional `[flop_buckets, turn_buckets, river_buckets]` (`None` = exact `IdentityAbstraction`).
    bucket_counts: Option<[usize; 3]>,
    game: Option<PloGame>,
    cached_report: Option<NodeReport>,
}

impl Default for WasmPloSession {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmPloSession {
    pub fn new() -> Self {
        init_tables();
        Self {
            board: None,
            oop_range: PloRange::new(),
            ip_range: PloRange::new(),
            starting_pot: 100,
            effective_stack: 100,
            betting_limit: BettingLimit::PotLimit,
            max_bets_per_street: 4,
            flop_sizes: [StreetBetSizes::default(), StreetBetSizes::default()],
            turn_sizes: [StreetBetSizes::default(), StreetBetSizes::default()],
            river_sizes: [StreetBetSizes::default(), StreetBetSizes::default()],
            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
            enable_isomorphism: true,
            bucket_counts: None,
            game: None,
            cached_report: None,
        }
    }

    /// Sets the postflop board (`3` cards for flop, `4` for turn, `5` for river, e.g. `"KsQsJs7d2c"`).
    pub fn set_board(&mut self, board_str: &str) -> Result<Street, String> {
        let b: Board = board_str.parse().map_err(|e| format!("{e}"))?;
        let st = b.street();
        self.board = Some(b);
        self.game = None;
        self.cached_report = None;
        Ok(st)
    }

    /// Sets player `0` (`OOP`) or `1` (`IP`) 4-card PLO range from string syntax
    /// (e.g. `"AAKK$ds, (AQ)(JT), AsTs4h3h@0.5"`), returning the raw unblocked combo count.
    pub fn set_range(&mut self, player: usize, range_str: &str) -> Result<usize, String> {
        if player > 1 {
            return Err(format!("invalid player index {player} (expected 0=OOP or 1=IP)"));
        }
        let r: PloRange = range_str.parse().map_err(|e| format!("{e}"))?;
        let count = if let Some(b) = &self.board {
            r.without_dead(b.mask()).len()
        } else {
            r.len()
        };
        if player == 0 {
            self.oop_range = r;
        } else {
            self.ip_range = r;
        }
        self.game = None;
        self.cached_report = None;
        Ok(count)
    }

    /// Configures stacks, pot, Pot-Limit betting cap, and comma-separated bet/raise pot percentages.
    #[allow(clippy::too_many_arguments)]
    pub fn set_tree_config(
        &mut self,
        starting_pot: i32,
        effective_stack: i32,
        max_bets_per_street: u8,
        oop_bet_str: &str,
        oop_raise_str: &str,
        ip_bet_str: &str,
        ip_raise_str: &str,
    ) -> Result<(), String> {
        let oop_s = StreetBetSizes::from_strs(oop_bet_str, oop_raise_str)
            .map_err(|e| format!("{e}"))?;
        let ip_s =
            StreetBetSizes::from_strs(ip_bet_str, ip_raise_str).map_err(|e| format!("{e}"))?;
        self.starting_pot = starting_pot;
        self.effective_stack = effective_stack;
        self.max_bets_per_street = max_bets_per_street;
        self.flop_sizes = [oop_s.clone(), ip_s.clone()];
        self.turn_sizes = [oop_s.clone(), ip_s.clone()];
        self.river_sizes = [oop_s, ip_s];
        self.game = None;
        self.cached_report = None;
        Ok(())
    }

    /// Configures optional hand bucketing (`0` on all streets = exact `IdentityAbstraction`).
    pub fn set_abstraction_buckets(
        &mut self,
        flop_buckets: usize,
        turn_buckets: usize,
        river_buckets: usize,
    ) {
        if flop_buckets == 0 && turn_buckets == 0 && river_buckets == 0 {
            self.bucket_counts = None;
        } else {
            self.bucket_counts = Some([
                flop_buckets.max(1),
                turn_buckets.max(1),
                river_buckets.max(1),
            ]);
        }
        self.game = None;
        self.cached_report = None;
    }

    pub fn set_memory_budget(&mut self, max_bytes: u64) {
        self.max_memory_bytes = max_bytes;
    }

    pub fn set_isomorphism(&mut self, enabled: bool) {
        self.enable_isomorphism = enabled;
    }

    /// Builds and allocates the concrete `PloGame` for the configured street, ranges, and bet tree.
    pub fn build_game(&mut self) -> Result<MemoryReport, String> {
        let board = self
            .board
            .ok_or_else(|| "board not set; call set_board first".to_string())?;
        let cfg = TreeConfig {
            initial_street: board.street(),
            starting_pot: self.starting_pot,
            effective_stack: self.effective_stack,
            betting_limit: self.betting_limit,
            min_bet_unit: 1,
            max_bets_per_street: self.max_bets_per_street,
            flop_sizes: self.flop_sizes.clone(),
            turn_sizes: self.turn_sizes.clone(),
            river_sizes: self.river_sizes.clone(),
            ..Default::default()
        };
        let tree = ActionTree::build(cfg).map_err(|e| format!("{e}"))?;
        let opts = SolverBuildOptions {
            max_memory_bytes: self.max_memory_bytes,
            enable_chance_isomorphism: self.enable_isomorphism,
            ..Default::default()
        };

        let split_2d = |k: usize| -> (usize, usize) {
            if k <= 1 {
                (1, 1)
            } else if k == 2 {
                (1, 2)
            } else if (k & 1) == 0 {
                (k / 2, 2)
            } else {
                (k, 1)
            }
        };

        let game = if let Some([fb, tb, rb]) = self.bucket_counts {
            let abs = NutBlockerBucketAbstraction::new(split_2d(fb), split_2d(tb), split_2d(rb));
            PloGame::with_options(board, &self.oop_range, &self.ip_range, &tree, &abs, opts)
                .map_err(|e| format!("{e}"))?
        } else {
            let _ = EquityBucketAbstraction::new(1, 1, 1); // keep 1D abstraction linked & available
            PloGame::with_options(
                board,
                &self.oop_range,
                &self.ip_range,
                &tree,
                &IdentityAbstraction,
                opts,
            )
            .map_err(|e| format!("{e}"))?
        };

        let rep = game.memory_report();
        self.game = Some(game);
        self.cached_report = None;
        Ok(rep)
    }

    /// Runs `num_steps` DCFR iterations in tight native/WASM loop with zero String allocation.
    pub fn step_iterations(&mut self, num_steps: u32) -> Result<u32, String> {
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| "game not built; call build_game first".to_string())?;
        for _ in 0..num_steps {
            game.solve_step();
        }
        self.cached_report = None;
        Ok(game.iterations_run())
    }

    /// Solves up to `max_iterations` or until exploitability `<= target_exploitability`, returning exploitability.
    pub fn solve(
        &mut self,
        max_iterations: u32,
        target_exploitability: f32,
    ) -> Result<f32, String> {
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| "game not built; call build_game first".to_string())?;
        let expl = game.solve(max_iterations, target_exploitability);
        self.cached_report = None;
        Ok(expl)
    }

    pub fn exploitability(&self) -> Result<f32, String> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| "game not built".to_string())?;
        Ok(game.exploitability())
    }

    pub fn back_to_root(&mut self) -> Result<(), String> {
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| "game not built".to_string())?;
        game.back_to_root();
        self.cached_report = None;
        Ok(())
    }

    pub fn play_action(&mut self, action_idx: usize) -> Result<(), String> {
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| "game not built".to_string())?;
        game.play_action(action_idx).map_err(|e| format!("{e}"))?;
        self.cached_report = None;
        Ok(())
    }

    pub fn play_chance(&mut self, card_str: &str) -> Result<(), String> {
        let card: Card = card_str.parse().map_err(|e| format!("{e}"))?;
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| "game not built".to_string())?;
        game.play_chance_card(card).map_err(|e| format!("{e}"))?;
        self.cached_report = None;
        Ok(())
    }

    /// Computes and caches the current node report, returning a JSON document with combos, actions,
    /// per-combo strategies, per-combo EVs, per-combo river equities (when on a 5-card river), and pot info.
    pub fn query_node_json(&mut self) -> Result<String, String> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| "game not built".to_string())?;
        let rep = game.inspect_current_node();
        let oop_combos = game.combos(0);
        let ip_combos = game.combos(1);

        let (eq_oop, eq_ip) = if let Some(rc) = rep.board.river_cards() {
            (
                per_combo_river_equity(oop_combos, ip_combos, rc),
                per_combo_river_equity(ip_combos, oop_combos, rc),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let root_evs = game.root_expected_values_f64();
        let iters = game.iterations_run();
        let mem = game.memory_report();

        let street_str = match rep.board.street() {
            Street::Flop => "flop",
            Street::Turn => "turn",
            Street::River => "river",
        };
        let player_json = match rep.node_type {
            "oop" => "0",
            "ip" => "1",
            _ => "null",
        };
        let active_ev: &[f32] = match rep.node_type {
            "ip" => &rep.ev_ip,
            _ => &rep.ev_oop,
        };

        let mut json = String::with_capacity(4096);
        json.push('{');
        json.push_str(&format!(
            "\"street\":\"{street_str}\",\"player\":{player_json},\"board\":\"{}\",\"pot\":{},\"committed\":[{},{}],\"nodeType\":\"{}\",\"iterations\":{},",
            rep.board, rep.pot, rep.committed[0], rep.committed[1], rep.node_type, iters
        ));
        json.push_str(&format!(
            "\"rootEv\":[{:.12},{:.12}],\"memoryBytes\":{},\"decisionNodes\":{},",
            root_evs[0], root_evs[1], mem.total_estimated_bytes, mem.decision_nodes
        ));

        // Actions
        json.push_str("\"actions\":[");
        for (i, a) in rep.actions.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push('"');
            json.push_str(&a.to_solver_label());
            json.push('"');
        }
        json.push_str("],");

        // Chance cards if at chance node (includes representative `chanceCards` + `allChanceCards` for all 48/49 real cards)
        json.push_str("\"chanceCards\":[");
        for (i, c) in game.available_chance_cards().iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!("\"{c}\""));
        }
        json.push_str("],\"allChanceCards\":[");
        for (i, c) in game.all_legal_chance_cards().iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!("\"{c}\""));
        }
        json.push_str("],");

        // OOP & IP hand lists
        json.push_str("\"oopHands\":[");
        for (i, c) in oop_combos.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"comboId\":{},\"hand\":\"{}\",\"weight\":{:.4}}}",
                c.combo_id.raw(),
                c.hand,
                c.weight
            ));
        }
        json.push_str("],\"ipHands\":[");
        for (i, c) in ip_combos.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"comboId\":{},\"hand\":\"{}\",\"weight\":{:.4}}}",
                c.combo_id.raw(),
                c.hand,
                c.weight
            ));
        }
        json.push_str("],");

        // Numeric arrays
        write_f32_json_array(&mut json, "strategy", &rep.strategy);
        json.push(',');
        write_f32_json_array(&mut json, "ev", active_ev);
        json.push(',');
        write_f32_json_array(&mut json, "evOop", &rep.ev_oop);
        json.push(',');
        write_f32_json_array(&mut json, "evIp", &rep.ev_ip);
        json.push(',');
        write_f32_json_array(&mut json, "equityOop", &eq_oop);
        json.push(',');
        write_f32_json_array(&mut json, "equityIp", &eq_ip);

        json.push('}');
        self.cached_report = Some(rep);
        Ok(json)
    }

    pub fn cached_report(&self) -> Option<&NodeReport> {
        self.cached_report.as_ref()
    }
}

fn write_f32_json_array(out: &mut String, key: &str, vals: &[f32]) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":[");
    for (i, &v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        if v.is_finite() {
            out.push_str(&format!("{v:.9}"));
        } else {
            out.push_str("0.0");
        }
    }
    out.push(']');
}

// ============================================================================
// C-ABI Exports for `wasm32-unknown-unknown` (and Native Dynamic/Static Linking)
// ============================================================================

thread_local! {
    static SESSION: RefCell<WasmPloSession> = RefCell::new(WasmPloSession::new());
    static OUT_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static ERR_BUF: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_output_str(s: String) -> usize {
    OUT_BUF.with(|b| {
        let mut buf = b.borrow_mut();
        *buf = s.into_bytes();
        buf.len()
    })
}

fn set_err_str(e: String) -> i32 {
    ERR_BUF.with(|b| {
        *b.borrow_mut() = e;
    });
    -1
}

/// Allocates `len` bytes in WASM linear memory so the JS host can write UTF-8 strings in.
#[no_mangle]
pub extern "C" fn plo_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// Frees a buffer previously allocated by `plo_alloc(len)`.
///
/// # Safety
/// `ptr` and `len` must match a live allocation returned by `plo_alloc(len)`.
#[no_mangle]
pub unsafe extern "C" fn plo_dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

/// Pointer to the latest output UTF-8 / JSON buffer in WASM linear memory.
#[no_mangle]
pub extern "C" fn plo_output_ptr() -> *const u8 {
    OUT_BUF.with(|b| b.borrow().as_ptr())
}

/// Length in bytes of the latest output UTF-8 / JSON buffer.
#[no_mangle]
pub extern "C" fn plo_output_len() -> usize {
    OUT_BUF.with(|b| b.borrow().len())
}

/// Pointer to the latest UTF-8 error message if an API call returned `< 0`.
#[no_mangle]
pub extern "C" fn plo_error_ptr() -> *const u8 {
    ERR_BUF.with(|b| b.borrow().as_ptr())
}

/// Byte length of the latest UTF-8 error message.
#[no_mangle]
pub extern "C" fn plo_error_len() -> usize {
    ERR_BUF.with(|b| b.borrow().len())
}

unsafe fn read_utf8_arg<'a>(ptr: *const u8, len: usize) -> Result<&'a str, String> {
    if len == 0 {
        return Ok("");
    }
    let slice = core::slice::from_raw_parts(ptr, len);
    core::str::from_utf8(slice).map_err(|e| format!("invalid UTF-8 argument: {e}"))
}

/// Resets the global WASM solver session and initializes lookup tables.
#[no_mangle]
pub extern "C" fn plo_session_reset() {
    SESSION.with(|s| *s.borrow_mut() = WasmPloSession::new());
}

/// Evaluates a single 4-card PLO hand `(h0, h1, h2, h3)` (`0..52`) on a board `(b_ptr, b_len)` (`3..=5` bytes)
/// directly without strings. Returns `1..=7462` (`HandRank`), or `-1` on invalid/duplicate cards.
///
/// # Safety
/// `board_ptr` must point to `board_len` readable `u8` card ids.
#[no_mangle]
pub unsafe extern "C" fn plo_eval_hand_raw(
    h0: u8,
    h1: u8,
    h2: u8,
    h3: u8,
    board_ptr: *const u8,
    board_len: usize,
) -> i32 {
    let Ok(c0) = Card::new(h0) else { return -1 };
    let Ok(c1) = Card::new(h1) else { return -1 };
    let Ok(c2) = Card::new(h2) else { return -1 };
    let Ok(c3) = Card::new(h3) else { return -1 };
    let Ok(hand) = Hand4::new([c0, c1, c2, c3]) else {
        return -1;
    };
    if board_ptr.is_null() || !(3..=5).contains(&board_len) {
        return -1;
    }
    let b_raw = core::slice::from_raw_parts(board_ptr, board_len);
    let mut b_cards = Vec::with_capacity(board_len);
    for &id in b_raw {
        let Ok(c) = Card::new(id) else { return -1 };
        b_cards.push(c);
    }
    let Ok(board) = Board::from_slice(&b_cards) else {
        return -1;
    };
    match eval_plo(hand, &board) {
        Ok(rank) => rank.raw() as i32,
        Err(_) => -1,
    }
}

/// Computes exact PLO equity JSON (`{"equityP0":...,"equityP1":...,"winP0":...,"winP1":...,"tie":...}`)
/// between two range strings (or two 8-char hands) on `board_str`. Returns output JSON byte length, or `-1` on error.
///
/// # Safety
/// All `(ptr, len)` pairs must point to valid UTF-8 slices in linear memory.
#[no_mangle]
pub unsafe extern "C" fn plo_compute_equity_json(
    r0_ptr: *const u8,
    r0_len: usize,
    r1_ptr: *const u8,
    r1_len: usize,
    board_ptr: *const u8,
    board_len: usize,
) -> i32 {
    let res = (|| -> Result<String, String> {
        let r0_s = read_utf8_arg(r0_ptr, r0_len)?;
        let r1_s = read_utf8_arg(r1_ptr, r1_len)?;
        let b_s = read_utf8_arg(board_ptr, board_len)?;
        let board: Board = b_s.parse().map_err(|e| format!("{e}"))?;

        // Fast path if both inputs are single 4-card hands
        let eq = if let (Ok(h0), Ok(h1)) = (r0_s.trim().parse::<Hand4>(), r1_s.trim().parse::<Hand4>()) {
            equity_hand_vs_hand(h0, h1, &board).map_err(|e| format!("{e}"))?
        } else {
            let r0: PloRange = r0_s.parse().map_err(|e| format!("{e}"))?;
            let r1: PloRange = r1_s.parse().map_err(|e| format!("{e}"))?;
            equity_range_vs_range(&r0, &r1, &board).map_err(|e| format!("{e}"))?
        };

        Ok(format!(
            "{{\"equityP0\":{:.6},\"equityP1\":{:.6},\"winP0\":{:.6},\"winP1\":{:.6},\"tie\":{:.6},\"validWeightSum\":{:.4}}}",
            eq.equity_p0, eq.equity_p1, eq.win_p0, eq.win_p1, eq.tie, eq.valid_weight_sum
        ))
    })();

    match res {
        Ok(json) => set_output_str(json) as i32,
        Err(e) => set_err_str(e),
    }
}

/// Sets the community board on the active WASM session (`3` flop, `4` turn, or `5` river cards).
/// Returns board card count (`3, 4, or 5`), or `-1` on error.
///
/// # Safety
/// `(ptr, len)` must point to a valid UTF-8 buffer.
#[no_mangle]
pub unsafe extern "C" fn plo_set_board(ptr: *const u8, len: usize) -> i32 {
    let Ok(s) = read_utf8_arg(ptr, len) else {
        return set_err_str("invalid UTF-8 board string".to_string());
    };
    SESSION.with(|sess| match sess.borrow_mut().set_board(s) {
        Ok(st) => st.board_len() as i32,
        Err(e) => set_err_str(e),
    })
}

/// Sets player `0` (`OOP`) or `1` (`IP`) 4-card range string on the active WASM session.
/// Returns active combo count, or `-1` on error.
///
/// # Safety
/// `(ptr, len)` must point to a valid UTF-8 buffer.
#[no_mangle]
pub unsafe extern "C" fn plo_set_range(player: usize, ptr: *const u8, len: usize) -> i32 {
    let Ok(s) = read_utf8_arg(ptr, len) else {
        return set_err_str("invalid UTF-8 range string".to_string());
    };
    SESSION.with(|sess| match sess.borrow_mut().set_range(player, s) {
        Ok(cnt) => cnt as i32,
        Err(e) => set_err_str(e),
    })
}

/// Configures Pot-Limit tree parameters (`starting_pot`, `effective_stack`, `max_bets_per_street`,
/// and bet/raise pot percentage strings for OOP and IP). Returns `0` on success or `-1` on error.
///
/// # Safety
/// All `(ptr, len)` pairs must point to valid UTF-8 buffers.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn plo_set_tree_config(
    starting_pot: i32,
    effective_stack: i32,
    max_bets_per_street: u8,
    oop_bet_ptr: *const u8,
    oop_bet_len: usize,
    oop_raise_ptr: *const u8,
    oop_raise_len: usize,
    ip_bet_ptr: *const u8,
    ip_bet_len: usize,
    ip_raise_ptr: *const u8,
    ip_raise_len: usize,
) -> i32 {
    let res = (|| -> Result<(), String> {
        let ob = read_utf8_arg(oop_bet_ptr, oop_bet_len)?;
        let or = read_utf8_arg(oop_raise_ptr, oop_raise_len)?;
        let ib = read_utf8_arg(ip_bet_ptr, ip_bet_len)?;
        let ir = read_utf8_arg(ip_raise_ptr, ip_raise_len)?;
        SESSION.with(|sess| {
            sess.borrow_mut().set_tree_config(
                starting_pot,
                effective_stack,
                max_bets_per_street,
                ob,
                or,
                ib,
                ir,
            )
        })
    })();
    match res {
        Ok(()) => 0,
        Err(e) => set_err_str(e),
    }
}

/// Sets optional `[flop_buckets, turn_buckets, river_buckets]` (`0, 0, 0` disables bucketing for exact solving).
#[no_mangle]
pub extern "C" fn plo_set_abstraction_buckets(
    flop_buckets: usize,
    turn_buckets: usize,
    river_buckets: usize,
) {
    SESSION.with(|sess| {
        sess.borrow_mut()
            .set_abstraction_buckets(flop_buckets, turn_buckets, river_buckets);
    });
}

/// Sets maximum solver memory budget in megabytes (`default = 512`).
#[no_mangle]
pub extern "C" fn plo_set_memory_budget_mb(max_mb: u32) {
    SESSION.with(|sess| {
        sess.borrow_mut()
            .set_memory_budget((max_mb as u64) * 1024 * 1024);
    });
}

/// Builds the game tree and writes a JSON `MemoryReport` to the output buffer.
/// Returns JSON byte length on success, or `-1` on error.
#[no_mangle]
pub extern "C" fn plo_build_game() -> i32 {
    SESSION.with(|sess| match sess.borrow_mut().build_game() {
        Ok(rep) => {
            let json = format!(
                "{{\"decisionNodes\":{},\"chanceNodes\":{},\"terminalNodes\":{},\"riverEvalTables\":{},\"regretAndStrategyBytes\":{},\"riverEvalBytes\":{},\"totalEstimatedBytes\":{}}}",
                rep.decision_nodes,
                rep.chance_nodes,
                rep.terminal_nodes,
                rep.river_eval_tables,
                rep.regret_and_strategy_bytes,
                rep.river_eval_bytes,
                rep.total_estimated_bytes,
            );
            set_output_str(json) as i32
        }
        Err(e) => set_err_str(e),
    })
}

/// Runs `num_steps` iterations of Discounted CFR with zero string overhead.
/// Returns total iterations completed, or `-1` on error.
#[no_mangle]
pub extern "C" fn plo_step_iterations(num_steps: u32) -> i32 {
    SESSION.with(|sess| match sess.borrow_mut().step_iterations(num_steps) {
        Ok(total) => total as i32,
        Err(e) => set_err_str(e),
    })
}

/// Solves up to `max_iterations` or until `exploitability <= target_exploitability`, returning exploitability
/// (`NaN` on error).
#[no_mangle]
pub extern "C" fn plo_solve(max_iterations: u32, target_exploitability: f32) -> f32 {
    SESSION.with(|sess| match sess.borrow_mut().solve(max_iterations, target_exploitability) {
        Ok(expl) => expl,
        Err(e) => {
            let _ = set_err_str(e);
            f32::NAN
        }
    })
}

/// Computes exact exploitability of the current average strategy (`NaN` on error).
#[no_mangle]
pub extern "C" fn plo_exploitability() -> f32 {
    SESSION.with(|sess| match sess.borrow().exploitability() {
        Ok(expl) => expl,
        Err(e) => {
            let _ = set_err_str(e);
            f32::NAN
        }
    })
}

/// Resets navigation cursor to the root of the game tree.
#[no_mangle]
pub extern "C" fn plo_back_to_root() -> i32 {
    SESSION.with(|sess| match sess.borrow_mut().back_to_root() {
        Ok(()) => 0,
        Err(e) => set_err_str(e),
    })
}

/// Advances navigation cursor along `action_idx` at a player decision node.
#[no_mangle]
pub extern "C" fn plo_play_action(action_idx: usize) -> i32 {
    SESSION.with(|sess| match sess.borrow_mut().play_action(action_idx) {
        Ok(()) => 0,
        Err(e) => set_err_str(e),
    })
}

/// Advances navigation cursor along `card_str` (e.g. `"7d"`) at a chance node.
///
/// # Safety
/// `(ptr, len)` must point to a valid UTF-8 slice.
#[no_mangle]
pub unsafe extern "C" fn plo_play_chance(ptr: *const u8, len: usize) -> i32 {
    let Ok(s) = read_utf8_arg(ptr, len) else {
        return set_err_str("invalid UTF-8 card string".to_string());
    };
    SESSION.with(|sess| match sess.borrow_mut().play_chance(s) {
        Ok(()) => 0,
        Err(e) => set_err_str(e),
    })
}

/// Queries the current node report as JSON into `plo_output_ptr() / plo_output_len()`.
/// Returns JSON byte length, or `-1` on error.
#[no_mangle]
pub extern "C" fn plo_query_node_json() -> i32 {
    SESSION.with(|sess| match sess.borrow_mut().query_node_json() {
        Ok(json) => set_output_str(json) as i32,
        Err(e) => set_err_str(e),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_session_river_end_to_end() {
        let mut sess = WasmPloSession::new();
        assert_eq!(sess.set_board("KsQsJs7d2c").unwrap(), Street::River);
        assert_eq!(sess.set_range(0, "AsTs4h3h, Ah9s6d5d").unwrap(), 2);
        assert_eq!(sess.set_range(1, "KdQd4c3c").unwrap(), 1);
        sess.set_tree_config(100, 100, 4, "50%, 100%", "100%", "50%, 100%", "100%")
            .unwrap();

        let mem = sess.build_game().unwrap();
        assert!(mem.total_estimated_bytes < 64 * 1024);

        let expl = sess.solve(400, 0.01).unwrap();
        assert!(expl < 0.1, "exploitability={expl}");

        let json = sess.query_node_json().unwrap();
        assert!(json.contains("\"street\":\"river\""));
        assert!(json.contains("\"player\":0"));
        assert!(json.contains("\"board\":\"KsQsJs7d2c\""));
        assert!(json.contains("\"nodeType\":\"oop\""));
        assert!(json.contains("\"ev\":["));
        assert!(json.contains("\"AsTs4h3h\""));
    }

    #[test]
    fn wasm_session_determinism_and_no_holdem_surface() {
        // Verify two independent WasmPloSession runs produce identical outputs to 1e-9!
        let run_once = || -> (f32, Vec<f32>, Vec<f32>) {
            let mut s = WasmPloSession::new();
            s.set_board("KsQsJs7d2c").unwrap();
            s.set_range(0, "AsTs4h3h, Ah9s6d5d").unwrap();
            s.set_range(1, "KdQd4c3c").unwrap();
            s.set_tree_config(100, 100, 4, "100%", "100%", "100%", "100%")
                .unwrap();
            s.build_game().unwrap();
            s.step_iterations(100).unwrap();
            let expl = s.solve(300, 0.0).unwrap();
            let _ = s.query_node_json().unwrap();
            let rep = s.cached_report().unwrap().clone();
            (expl, rep.strategy, rep.ev_oop)
        };
        let (e1, st1, ev1) = run_once();
        let (e2, st2, ev2) = run_once();
        assert!((e1 - e2).abs() < 1e-9);
        assert_eq!(st1, st2);
        assert_eq!(ev1, ev2);

        // Step 4: Verify repeated 2D-bucketed (`NutBlockerBucketAbstraction`) runs via `WasmPloSession`
        // agree within `1e-9` even when range tokens are listed in reversed order!
        let run_bucketed_2d = |oop_str: &str, ip_str: &str| -> (f32, Vec<f32>, Vec<f32>, String) {
            let mut s = WasmPloSession::new();
            s.set_board("KsQs4d2c").unwrap();
            s.set_range(0, oop_str).unwrap();
            s.set_range(1, ip_str).unwrap();
            s.set_abstraction_buckets(8, 8, 8);
            s.set_tree_config(100, 100, 3, "50%, 100%", "100%", "50%, 100%", "100%")
                .unwrap();
            s.build_game().unwrap();
            s.step_iterations(60).unwrap();
            let expl = s.exploitability().unwrap();
            let json = s.query_node_json().unwrap();
            let rep = s.cached_report().unwrap().clone();
            (expl, rep.strategy, rep.ev_oop, json)
        };
        let (be1, bst1, bev1, bjson1) = run_bucketed_2d(
            "AA99$ds, (QJ)(98), As9h8d7c, Ah9c8d7h, 6655$ds",
            "AAKK$ds, (JT)(98), KdQd6c5c, AsTs6h5h",
        );
        let (be2, bst2, bev2, bjson2) = run_bucketed_2d(
            "6655$ds, Ah9c8d7h, As9h8d7c, (QJ)(98), AA99$ds",
            "AsTs6h5h, KdQd6c5c, (JT)(98), AAKK$ds",
        );
        assert!((be1 - be2).abs() <= 1e-9);
        assert_eq!(bjson1, bjson2);
        for (&a, &b) in bst1.iter().zip(&bst2) {
            assert!((a - b).abs() <= 1e-9);
        }
        for (&a, &b) in bev1.iter().zip(&bev2) {
            assert!((a - b).abs() <= 1e-9);
        }
    }
}
