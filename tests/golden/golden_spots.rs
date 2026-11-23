//! Public Regression Golden Spot Suite (`G1`–`G4` + `Step 4` Real-Range Macro Smoke).
//!
//! All spots use explicit, deterministic cards (zero PRNG) and are cross-checked against
//! hand-computed closed-form solutions, `brute_showdown_ev`, or `plo_range::equity_range_vs_range`.
//! Note: `Cargo.toml` wires this file directly via `[[test]] name = "golden" path = "../../tests/golden/golden_spots.rs"`.

use plo_cards::{Board, Hand4, Street};
use plo_cfr::{
    brute_showdown_ev, ie_showdown_ev, IdentityAbstraction, PloGame, SolverBuildOptions,
    SolverError, DEFAULT_MAX_MEMORY_BYTES,
};
use plo_range::{equity_range_vs_range, PloRange};
use plo_tree::{Action, ActionTree, BettingLimit, StreetBetSizes, TreeConfig};

fn build_checkdown_tree(street: Street, starting_pot: i32) -> ActionTree {
    let empty_sizes = StreetBetSizes {
        bet_fractions: vec![],
        raise_fractions: vec![],
        donk_fractions: None,
    };
    ActionTree::build(TreeConfig {
        initial_street: street,
        starting_pot,
        effective_stack: starting_pot,
        betting_limit: BettingLimit::PotLimit,
        flop_sizes: [empty_sizes.clone(), empty_sizes.clone()],
        turn_sizes: [empty_sizes.clone(), empty_sizes.clone()],
        river_sizes: [empty_sizes.clone(), empty_sizes],
        add_allin_spr_threshold: 0.0,
        ..Default::default()
    })
    .unwrap()
}

/// G1 — River Hand-vs-Hand & Blocker Showdown Goldens (No Betting: `EV == equity * pot`, `brute_showdown_ev` as spec):
/// 1) Nuts vs PLO bare-suit Air (`100.0` vs `0.0`);
/// 2) True Chop (`50.0` vs `50.0` — distinct suits making identical Broadway straight);
/// 3) 1-Card Blocker Pair (`compatible_mass == 0.0` -> `SolverError::NoValidMatchups` & `0.0` mass in `brute_showdown_ev` / `ie_showdown_ev`,
///    plus mixed `[blocked_combo, live_combo]` where blocked combo contributes `0.0` mass).
#[test]
fn g1_river_hand_vs_hand_no_betting_nuts_chop_and_blocker_zero() {
    let pot = 100i32;
    let pot_f = pot as f64;
    let cd_tree = build_checkdown_tree(Street::River, pot);

    // 1.1 Nuts vs Air on [Ks Qs Js 7d 2c]:
    // OOP: AsTs4h3h (A-high spade flush) vs IP: Ah9s6d5d (PLO bare 9s — CANNOT make a 1-card flush, only A-K-Q-J-9 high!)
    let river1: Board = "KsQsJs7d2c".parse().unwrap();
    let rc1 = river1.river_cards().unwrap();
    let nuts: PloRange = "AsTs4h3h".parse().unwrap();
    let air: PloRange = "Ah9s6d5d".parse().unwrap();

    let bf_nuts = brute_showdown_ev(&nuts, &air, rc1, pot_f, 0.0, 0.0);
    let ie_nuts = ie_showdown_ev(&nuts, &air, rc1, pot_f, 0.0, 0.0);
    let eq_nuts = equity_range_vs_range(&nuts, &air, &river1).unwrap();
    assert_eq!(bf_nuts.compatible_mass, 1.0);
    assert_eq!(bf_nuts.win_mass, 1.0);
    assert_eq!(bf_nuts.tie_mass, 0.0);
    assert_eq!(bf_nuts.lose_mass, 0.0);
    assert!((bf_nuts.ev_oop - 100.0).abs() <= 1e-12 && bf_nuts.ev_ip.abs() <= 1e-12);
    assert!((ie_nuts.ev_oop - bf_nuts.ev_oop).abs() <= 1e-12);
    assert!((eq_nuts.equity_p0 * pot_f - bf_nuts.ev_oop).abs() <= 1e-12);

    let g_nuts = PloGame::new(river1, &nuts, &air, &cd_tree).unwrap();
    let [ev0, ev1] = g_nuts.root_expected_values_f64();
    assert!((ev0 - 100.0).abs() <= 1e-9 && ev1.abs() <= 1e-9);

    // 1.2 True Chop on [Ks Qd Jc 7h 2s]:
    // OOP: AhTd4c3c (Broadway straight A-K-Q-J-T) vs IP: AcTh5d4d (identical Broadway straight A-K-Q-J-T, disjoint cards!)
    let river2: Board = "KsQdJc7h2s".parse().unwrap();
    let rc2 = river2.river_cards().unwrap();
    let chop_oop: PloRange = "AhTd4c3c".parse().unwrap();
    let chop_ip: PloRange = "AcTh5d4d".parse().unwrap();

    let bf_chop = brute_showdown_ev(&chop_oop, &chop_ip, rc2, pot_f, 0.0, 0.0);
    let ie_chop = ie_showdown_ev(&chop_oop, &chop_ip, rc2, pot_f, 0.0, 0.0);
    let eq_chop = equity_range_vs_range(&chop_oop, &chop_ip, &river2).unwrap();
    assert_eq!(bf_chop.compatible_mass, 1.0);
    assert_eq!(bf_chop.win_mass, 0.0);
    assert_eq!(bf_chop.tie_mass, 1.0);
    assert_eq!(bf_chop.lose_mass, 0.0);
    assert!((bf_chop.ev_oop - 50.0).abs() <= 1e-12 && (bf_chop.ev_ip - 50.0).abs() <= 1e-12);
    assert!((ie_chop.tie_mass - 1.0).abs() <= 1e-12 && (ie_chop.ev_oop - 50.0).abs() <= 1e-12);
    assert!((eq_chop.equity_p0 * pot_f - 50.0).abs() <= 1e-12);

    let g_chop = PloGame::new(river2, &chop_oop, &chop_ip, &cd_tree).unwrap();
    let [c_ev0, c_ev1] = g_chop.root_expected_values_f64();
    assert!((c_ev0 - 50.0).abs() <= 1e-9 && (c_ev1 - 50.0).abs() <= 1e-9);

    // 1.3 1-Card Blocker Pair (`As` shared between OOP `AsTs4h3h` and IP `AsKdQd4c`):
    // Pure 1-vs-1 sharing `As` MUST have `compatible_mass == 0.0`, `ev == 0.0`, and `PloGame::new` returns `Err(NoValidMatchups)`!
    let blk_oop: PloRange = "AsTs4h3h".parse().unwrap();
    let blk_ip: PloRange = "AsKdQd4c".parse().unwrap();
    let bf_blk = brute_showdown_ev(&blk_oop, &blk_ip, rc1, pot_f, 0.0, 0.0);
    let ie_blk = ie_showdown_ev(&blk_oop, &blk_ip, rc1, pot_f, 0.0, 0.0);
    assert_eq!(bf_blk.compatible_mass, 0.0);
    assert_eq!(bf_blk.ev_oop, 0.0);
    assert_eq!(bf_blk.ev_ip, 0.0);
    assert_eq!(ie_blk.compatible_mass, 0.0);
    assert_eq!(ie_blk.ev_oop, 0.0);
    assert_eq!(ie_blk.ev_ip, 0.0);
    assert!(matches!(
        PloGame::new(river1, &blk_oop, &blk_ip, &cd_tree),
        Err(SolverError::NoValidMatchups)
    ));

    // Also verify when OOP has `{AsTs4h3h (blocked by IP's As!), 9h8h6d5d (live air)}` vs IP `{AsKdQd4c}`:
    // the 1-card-blocked nut combo MUST contribute `0.0` weight, so OOP has `0%` equity (`EV = 0.0`, IP `EV = 100.0`)!
    let oop_blocked_nuts_plus_air: PloRange = "AsTs4h3h, 9h8h6d5d".parse().unwrap();
    let bf_mix = brute_showdown_ev(&oop_blocked_nuts_plus_air, &blk_ip, rc1, pot_f, 0.0, 0.0);
    let ie_mix = ie_showdown_ev(&oop_blocked_nuts_plus_air, &blk_ip, rc1, pot_f, 0.0, 0.0);
    assert_eq!(bf_mix.compatible_mass, 1.0);
    assert_eq!(bf_mix.win_mass, 0.0);
    assert_eq!(bf_mix.lose_mass, 1.0);
    assert!((ie_mix.ev_oop - 0.0).abs() <= 1e-12 && (ie_mix.ev_ip - 100.0).abs() <= 1e-12);
    let g_mix = PloGame::new(river1, &oop_blocked_nuts_plus_air, &blk_ip, &cd_tree).unwrap();
    let [m_ev0, m_ev1] = g_mix.root_expected_values_f64();
    assert!(m_ev0.abs() <= 1e-9 && (m_ev1 - 100.0).abs() <= 1e-9);
}

/// G3 — Turn Check-Down, 2 vs 2 Combos, Two-Tone Board (`iso = true` and `iso = false`):
/// Verifies `PloGame::root_expected_values_f64()` matches `equity_range_vs_range * pot` within `1e-9`,
/// and `iso = true` folds heart/diamond river cards while agreeing with `iso = false` to `1e-12`.
#[test]
fn g3_turn_checkdown_2v2_twotone_board_matches_equity_1e9_iso_on_and_off() {
    // Two-tone turn [As Ks 7c 2c]: suits `h` and `d` are unrepresented on board!
    // Make both 2-combo ranges symmetric under `h <-> d` so `iso = true` actively folds river suits:
    // - OOP (2 combos): `QhJhTd9d` and `QdJdTh9h` (double-suited Broadway wrap in `{h, d}`, invariant under `h <-> d`)
    // - IP  (2 combos): `AcAh8h6d` and `AcAd8d6h` (top set of Aces `{Ac, Ah}` <-> `{Ac, Ad}`, invariant under `h <-> d`!)
    let turn: Board = "AsKs7c2c".parse().unwrap();
    let oop: PloRange = "QhJhTd9d, QdJdTh9h".parse().unwrap();
    let ip: PloRange = "AcAh8h6d, AcAd8d6h".parse().unwrap();
    assert_eq!(oop.len(), 2);
    assert_eq!(ip.len(), 2);

    let pot = 100i32;
    let cd_tree = build_checkdown_tree(Street::Turn, pot);
    let eq = equity_range_vs_range(&oop, &ip, &turn).unwrap();
    let oracle_ev0 = eq.equity_p0 * (pot as f64);
    let oracle_ev1 = eq.equity_p1 * (pot as f64);

    let g_iso_on = PloGame::with_options(
        turn,
        &oop,
        &ip,
        &cd_tree,
        &IdentityAbstraction,
        SolverBuildOptions {
            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
            enable_chance_isomorphism: true,
            ..Default::default()
        },
    )
    .unwrap();
    let g_iso_off = PloGame::with_options(
        turn,
        &oop,
        &ip,
        &cd_tree,
        &IdentityAbstraction,
        SolverBuildOptions {
            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
            enable_chance_isomorphism: false,
            ..Default::default()
        },
    )
    .unwrap();

    // Confirm `iso = true` folded 13 heart rivers into 13 diamond rivers (`35 < 48` terminal nodes)!
    assert_eq!(g_iso_off.memory_report().terminal_nodes, 48);
    assert_eq!(g_iso_on.memory_report().terminal_nodes, 35);

    let ev_on = g_iso_on.root_expected_values_f64();
    let ev_off = g_iso_off.root_expected_values_f64();

    assert!(
        (ev_on[0] - oracle_ev0).abs() <= 1e-9 && (ev_on[1] - oracle_ev1).abs() <= 1e-9,
        "iso=true EV {ev_on:?} vs equity*pot ({oracle_ev0}, {oracle_ev1})"
    );
    assert!(
        (ev_off[0] - oracle_ev0).abs() <= 1e-9 && (ev_off[1] - oracle_ev1).abs() <= 1e-9,
        "iso=false EV {ev_off:?} vs equity*pot ({oracle_ev0}, {oracle_ev1})"
    );
    assert!(
        (ev_on[0] - ev_off[0]).abs() <= 1e-12 && (ev_on[1] - ev_off[1]).abs() <= 1e-12,
        "iso=true {ev_on:?} vs iso=false {ev_off:?}"
    );
}

/// G2 & G4 — Frozen Polar River Betting Spot (Synced with `js/example_river.mjs` @ `1000` Fixed DCFR Iterations
/// AND Independently Verified Against Pure Matrix Best-Response Enumeration):
/// - Board: `KsQsJs7d2c`, `Pot = 100`, `Stack = 100`, `100%` Pot-only (`Check` / `AllIn(100)`, no raise possible)
/// - OOP (`2` combos, disjoint from IP): `AsTs4h3h` (Nut Spade Flush) + `Ah9s6d5d` (PLO bare-`9s` Air)
/// - IP  (`1` combo, disjoint from OOP): `KdQd4c3c` (Two-Pair Kings & Queens Bluff-Catcher)
/// - Closed-form Nash & `js/example_river.mjs` contract after `1000` fixed DCFR steps:
///   * Root EV within `±0.05` of `[75.0, 25.0]`
///   * OOP Nut bet `>= 0.99`, OOP Air bluff within `0.50 ± 0.02`, IP Catcher call vs `AllIn(100)` within `0.50 ± 0.02`
///   * Independent brute-tree best-response enumeration confirms `exploitability()` matches exact matrix MES within `1e-6`.
#[test]
fn g2_g4_polar_river_1000_iters_nash_and_independent_brute_tree_best_response() {
    const FIXED_POLAR_ITERATIONS: u32 = 1000;

    let river: Board = "KsQsJs7d2c".parse().unwrap();
    let nut_flush: Hand4 = "AsTs4h3h".parse().unwrap();
    let plo_air: Hand4 = "Ah9s6d5d".parse().unwrap();
    let bluff_catcher: Hand4 = "KdQd4c3c".parse().unwrap();

    // Verify zero card overlap across board + all 3 hands (`5 + 4 + 4 + 4 = 17` distinct cards)
    let union_mask = river.mask() | nut_flush.mask() | plo_air.mask() | bluff_catcher.mask();
    assert_eq!(union_mask.len(), 17);

    let oop = PloRange::from_hands([(nut_flush, 1.0), (plo_air, 1.0)]).unwrap();
    let ip = PloRange::from_hands([(bluff_catcher, 1.0)]).unwrap();

    let cfg = TreeConfig {
        initial_street: Street::River,
        starting_pot: 100,
        effective_stack: 100,
        betting_limit: BettingLimit::PotLimit,
        river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
        ..Default::default()
    };
    let tree = ActionTree::build(cfg).unwrap();
    let mut game = PloGame::new(river, &oop, &ip, &tree).unwrap();

    for _ in 0..FIXED_POLAR_ITERATIONS {
        game.solve_step();
    }
    assert_eq!(game.iterations_run(), FIXED_POLAR_ITERATIONS);

    let [ev_oop, ev_ip] = game.root_expected_values_f64();
    assert!(
        (ev_oop - 75.0).abs() <= 0.05,
        "OOP EV={ev_oop} outside 75.0 ± 0.05"
    );
    assert!(
        (ev_ip - 25.0).abs() <= 0.05,
        "IP EV={ev_ip} outside 25.0 ± 0.05"
    );
    assert!((ev_oop + ev_ip - 100.0).abs() <= 1e-12);

    // Extract strategies at all 3 decision nodes (`Root OOP`, `IP after Check`, `IP facing AllIn(100)`)
    let root_rep = game.inspect_current_node();
    assert_eq!(root_rep.actions, vec![Action::Check, Action::AllIn(100)]);
    let nut_idx = game
        .combos(0)
        .iter()
        .position(|c| c.hand == nut_flush)
        .unwrap();
    let air_idx = game
        .combos(0)
        .iter()
        .position(|c| c.hand == plo_air)
        .unwrap();
    let b_nut = root_rep.strategy[2 + nut_idx] as f64; // P(OOP bets AllIn(100) | Nuts)
    let b_air = root_rep.strategy[2 + air_idx] as f64; // P(OOP bets AllIn(100) | Air)

    // Node after OOP Check -> IP acts (`Check` or `AllIn(100)`)
    game.play_action(0).unwrap();
    let ip_vs_check = game.inspect_current_node();
    assert_eq!(ip_vs_check.actions, vec![Action::Check, Action::AllIn(100)]);
    let ip_stab = ip_vs_check.strategy[1] as f64; // P(IP bets AllIn(100) after OOP checks)

    // Node after OOP Check -> IP AllIn(100) -> OOP faces AllIn(100) (`Fold` or `Call`)
    game.play_action(1).unwrap();
    let oop_vs_stab = game.inspect_current_node();
    assert_eq!(oop_vs_stab.actions, vec![Action::Fold, Action::Call]);
    let c_nut_vs_stab = oop_vs_stab.strategy[2 + nut_idx] as f64;
    let c_air_vs_stab = oop_vs_stab.strategy[2 + air_idx] as f64;

    // Node after OOP AllIn(100) -> IP faces AllIn(100) (`Fold` or `Call`)
    game.back_to_root();
    game.play_action(1).unwrap();
    let ip_vs_bet = game.inspect_current_node();
    assert_eq!(ip_vs_bet.actions, vec![Action::Fold, Action::Call]);
    let ip_call = ip_vs_bet.strategy[1] as f64; // P(IP calls AllIn(100))

    // Assert exact `js/example_river.mjs` Polar River contract:
    assert!((b_nut - 1.0).abs() <= 0.01, "Nut bet freq={b_nut}");
    assert!(
        (b_air - 0.5).abs() <= 0.02,
        "Air bluff freq={b_air} outside 0.50 ± 0.02"
    );
    assert!(
        (ip_call - 0.5).abs() <= 0.02,
        "Catcher call freq={ip_call} outside 0.50 ± 0.02"
    );
    assert!(ip_stab <= 0.01, "Catcher stab freq={ip_stab} should be ~0");

    // G4: Independently enumerate all pure strategies in the 2-hand x 1-hand game tree by hand
    // and cross-check both expected value AND exact Nash exploitability!
    // Payoff convention (total pot share in `[0, 100]`, i.e., `starting_pot/2 + net_chips`):
    // - For OOP Nuts (`w = 0.5` of OOP range):
    //   * Bet `AllIn(100)` -> if IP folds (`1 - ip_call`): OOP wins `100` pot (`share = 100`);
    //                         if IP calls (`ip_call`): OOP wins `100` pot + `100` IP call = `200`!
    //     => `EV_OOP(Bet | Nuts) = 100.0 * (1.0 - ip_call) + 200.0 * ip_call = 100.0 + 100.0 * ip_call`
    //   * Check -> if IP checks (`1 - ip_stab`): OOP wins `100` at showdown (`share = 100`);
    //              if IP stabs `100` (`ip_stab`): OOP folds (`0`) or calls (`200`)
    // - For OOP Air (`w = 0.5` of OOP range):
    //   * Bet `AllIn(100)` -> if IP folds (`1 - ip_call`): OOP wins `100`;
    //                         if IP calls (`ip_call`): OOP loses `100` (`share = -100`)!
    //     => `EV_OOP(Bet | Air) = 100.0 * (1.0 - ip_call) - 100.0 * ip_call = 100.0 - 200.0 * ip_call`
    //   * Check -> if IP checks (`1 - ip_stab`): OOP loses showdown (`0`);
    //              if IP stabs `100` (`ip_stab`): OOP folds (`0`) or calls (`-100`)
    let oop_pure_line_ev = |is_nuts: bool, bet: bool, call_stab: bool| -> f64 {
        if bet {
            if is_nuts {
                100.0 * (1.0 - ip_call) + 200.0 * ip_call
            } else {
                100.0 * (1.0 - ip_call) - 100.0 * ip_call
            }
        } else {
            let check_check = if is_nuts { 100.0 } else { 0.0 };
            let stab_resp = if !call_stab {
                0.0
            } else if is_nuts {
                200.0
            } else {
                -100.0
            };
            (1.0 - ip_stab) * check_check + ip_stab * stab_resp
        }
    };

    // Enumerate all 4 pure actions `(bet, call_stab)` for Nuts and Air to get exact OOP Best-Response EV:
    let mut best_nuts_ev = f64::NEG_INFINITY;
    let mut best_air_ev = f64::NEG_INFINITY;
    for bet in [false, true] {
        for call_stab in [false, true] {
            best_nuts_ev = best_nuts_ev.max(oop_pure_line_ev(true, bet, call_stab));
            best_air_ev = best_air_ev.max(oop_pure_line_ev(false, bet, call_stab));
        }
    }
    let br_oop_ev = 0.5 * best_nuts_ev + 0.5 * best_air_ev;

    // Enumerate all 4 pure strategies `(stab_when_checked_to, call_vs_allin)` for IP Bluff-Catcher:
    let mut best_ip_ev = f64::NEG_INFINITY;
    for stab in [false, true] {
        for call_bet in [false, true] {
            // Against OOP's fixed strategy `(b_nut, c_nut_vs_stab)` and `(b_air, c_air_vs_stab)`:
            let oop_ev_nuts = b_nut * (if call_bet { 200.0 } else { 100.0 })
                + (1.0 - b_nut) * (if !stab { 100.0 } else { c_nut_vs_stab * 200.0 });
            let oop_ev_air = b_air * (if call_bet { -100.0 } else { 100.0 })
                + (1.0 - b_air) * (if !stab { 0.0 } else { c_air_vs_stab * (-100.0) });
            let ip_ev = 100.0 - (0.5 * oop_ev_nuts + 0.5 * oop_ev_air);
            best_ip_ev = best_ip_ev.max(ip_ev);
        }
    }

    let brute_expl = 0.5 * ((br_oop_ev - 50.0) + (best_ip_ev - 50.0));
    let solver_expl = game.exploitability() as f64;
    assert!(
        (solver_expl - brute_expl).abs() <= 1e-5,
        "solver exploitability {solver_expl} vs brute-enumerated {brute_expl}"
    );
    assert!(brute_expl < 0.05, "expected near-zero exploitability, got {brute_expl}");
}

/// Step 4 — Real Range Macro Smoke (`AAKK$ds` vs `(AK)(QJ)` on a flop, board-blocker filtering + exact equity snapshot to `1e-12`).
#[test]
fn step4_real_range_macro_smoke_aakk_ds_vs_ak_qj_flop_equity_snapshot() {
    let oop_preflop: PloRange = "AAKK$ds".parse().unwrap();
    let ip_preflop: PloRange = "(AK)(QJ)".parse().unwrap();

    // Combinatorial preflop macro counts:
    // - `AAKK$ds`: C(4, 2) = 6 double-suited combos
    // - `(AK)(QJ)`: 4 suits for (AK) * 3 distinct suits for (QJ) = 12 double-suited combos
    assert_eq!(oop_preflop.len(), 6);
    assert_eq!(ip_preflop.len(), 12);

    // Flop `QsJh4c` blocks `Qs` and `Jh` (leaving all 6 `AAKK$ds` combos live, and filtering `(AK)(QJ)` combos containing `Qs` or `Jh`):
    // Specifically in `(AK)(QJ)`:
    // - `(QJ)` in spades (`QsJs`) -> 3 combos blocked by `Qs`
    // - `(QJ)` in hearts (`QhJh`) -> 3 combos blocked by `Jh`
    // - Remaining `(QJ)` suits are clubs (`QcJc`, 3 combos with `AK` in `{d, h, s}`) and diamonds (`QdJd`, 3 combos with `AK` in `{c, h, s}`) -> 6 live combos!
    let flop: Board = "QsJh4c".parse().unwrap();
    let oop_live = oop_preflop.without_dead(flop.mask());
    let ip_live = ip_preflop.without_dead(flop.mask());
    assert_eq!(oop_live.len(), 6);
    assert_eq!(ip_live.len(), 6);

    let eq = equity_range_vs_range(&oop_live, &ip_live, &flop).unwrap();

    // Independently verify exact compatible pair count & runout denominator:
    // How many `(h_oop, h_ip)` pairs in `6 x 6 = 36` are disjoint?
    // Every `AAKK$ds` combo uses two suits `{s1, s2}` for BOTH Aces and Kings (`AsKs` + `AhKh`, etc.).
    // Every live `(AK)(QJ)` combo uses one suit `s_ak` for `{A, K}` and another suit `s_qj in {c, d}` for `{Q, J}`.
    // Since `h_oop` contains `A` and `K` of TWO suits `{s1, s2}`, `h_ip`'s `A_{s_ak} K_{s_ak}` is disjoint from `h_oop`
    // iff `s_ak` is NOT in `{s1, s2}`!
    // For each of the 6 `ip_live` combos (fixed `s_ak`), how many of the 6 `{s1, s2}` pairs avoid `s_ak`?
    // Exactly `C(3, 2) = 3`! So there are `6 * 3 = 18` compatible `(h_oop, h_ip)` matchups,
    // each evaluated over `C(52 - 3 - 8, 2) = C(41, 2) = 820` turn+river runouts -> `valid_weight_sum == 18 * 820 == 14,760`!
    assert!((eq.valid_weight_sum - 14_760.0).abs() <= 1e-12);
    assert!((eq.equity_p0 + eq.equity_p1 - 1.0).abs() <= 1e-12);

    // Cross-check against independent slow lexicographic 2+3 enumeration over all 18 * 820 = 14,760 deals,
    // AND freeze the exact rational equity snapshot to `1e-12`!
    let mut wins_0 = 0u64;
    let mut wins_1 = 0u64;
    let mut ties = 0u64;
    let fmask = flop.mask();
    for c0 in oop_live.entries() {
        for c1 in ip_live.entries() {
            if c0.mask.intersects(c1.mask) {
                continue;
            }
            let dead = fmask | c0.mask | c1.mask;
            for t in 0..52u8 {
                let tc = plo_cards::Card::new(t).unwrap();
                if dead.contains(tc) {
                    continue;
                }
                for r in (t + 1)..52u8 {
                    let rc = plo_cards::Card::new(r).unwrap();
                    if dead.contains(rc) {
                        continue;
                    }
                    let rb = flop.deal(tc).unwrap().deal(rc).unwrap();
                    let k0 = plo_eval::slow_eval_plo_lex(c0.hand, &rb).unwrap().0;
                    let k1 = plo_eval::slow_eval_plo_lex(c1.hand, &rb).unwrap().0;
                    match k0.cmp(&k1) {
                        core::cmp::Ordering::Greater => wins_0 += 1,
                        core::cmp::Ordering::Less => wins_1 += 1,
                        core::cmp::Ordering::Equal => ties += 1,
                    }
                }
            }
        }
    }
    assert_eq!(wins_0, 4_231);
    assert_eq!(wins_1, 8_387);
    assert_eq!(ties, 2_142);
    assert_eq!(wins_0 + wins_1 + ties, 14_760);

    // Exact rational snapshot: `5302 / 14760 == 2651 / 7380` (`0.3592140921409214...`), `9458 / 14760 == 4729 / 7380` (`0.6407859078590786...`)
    let frozen_eq_oop = 5_302.0f64 / 14_760.0f64;
    let frozen_eq_ip = 9_458.0f64 / 14_760.0f64;
    assert!(
        (eq.equity_p0 - frozen_eq_oop).abs() <= 1e-12
            && (eq.equity_p1 - frozen_eq_ip).abs() <= 1e-12,
        "eq={eq:?} vs frozen=({frozen_eq_oop}, {frozen_eq_ip})"
    );
}

/// D1 — External Calibration on a Disjoint Rainbow Low Flop (`7c3d2h`):
/// `AAKK$ds` vs `(AK)(QJ)` where ZERO board ranks collide with `{A, K, Q, J}`, so all `6` OOP and `12` IP
/// preflop combos survive (`36` compatible matchups x `820` runouts = `29,520` deals), cross-checked against
/// `plo_eval::slow_eval_plo_lex` ONLY (bypassing fast lookup tables completely) and frozen to `1e-12`.
#[test]
fn d1_aakk_ds_vs_ak_qj_unpaired_rainbow_flop_7c3d2h_slow_oracle_snapshot() {
    let oop_preflop: PloRange = "AAKK$ds".parse().unwrap();
    let ip_preflop: PloRange = "(AK)(QJ)".parse().unwrap();

    // Board `7c3d2h` uses ranks `{7, 3, 2}` — disjoint from `{A, K, Q, J}`, so 0 combos are board-blocked!
    let flop: Board = "7c3d2h".parse().unwrap();
    let oop_live = oop_preflop.without_dead(flop.mask());
    let ip_live = ip_preflop.without_dead(flop.mask());
    assert_eq!(oop_live.len(), 6);
    assert_eq!(ip_live.len(), 12);

    // Count compatible pairs & runouts using `slow_eval_plo_lex` ONLY (no `eval_plo_river` / `RiverBoardCache`!):
    let mut compatible_pairs = 0u64;
    let mut wins_oop = 0u64;
    let mut wins_ip = 0u64;
    let mut ties = 0u64;
    let fmask = flop.mask();

    for c0 in oop_live.entries() {
        for c1 in ip_live.entries() {
            if c0.mask.intersects(c1.mask) {
                continue;
            }
            compatible_pairs += 1;
            let dead = fmask | c0.mask | c1.mask;
            let mut n_runouts = 0u64;
            for t in 0..52u8 {
                let tc = plo_cards::Card::new(t).unwrap();
                if dead.contains(tc) {
                    continue;
                }
                for r in (t + 1)..52u8 {
                    let rc = plo_cards::Card::new(r).unwrap();
                    if dead.contains(rc) {
                        continue;
                    }
                    n_runouts += 1;
                    let rb = flop.deal(tc).unwrap().deal(rc).unwrap();
                    let k0 = plo_eval::slow_eval_plo_lex(c0.hand, &rb).unwrap().0;
                    let k1 = plo_eval::slow_eval_plo_lex(c1.hand, &rb).unwrap().0;
                    match k0.cmp(&k1) {
                        core::cmp::Ordering::Greater => wins_oop += 1,
                        core::cmp::Ordering::Less => wins_ip += 1,
                        core::cmp::Ordering::Equal => ties += 1,
                    }
                }
            }
            assert_eq!(n_runouts, 820, "C(41, 2) = 820 runouts per disjoint pair");
        }
    }

    assert_eq!(compatible_pairs, 36);
    let total_deals = compatible_pairs * 820;
    assert_eq!(total_deals, 29_520);
    assert_eq!(wins_oop, 27_459);
    assert_eq!(wins_ip, 2_061);
    assert_eq!(ties, 0);
    assert_eq!(wins_oop + wins_ip + ties, total_deals);

    // Exact frozen rational snapshot: `27459 / 29520 == 3051 / 3280` (`0.9301829268292683...`), `2061 / 29520 == 229 / 3280` (`0.0698170731707317...`)
    let frozen_eq_oop = 27_459.0f64 / 29_520.0f64;
    let frozen_eq_ip = 2_061.0f64 / 29_520.0f64;

    let fast_eq = equity_range_vs_range(&oop_live, &ip_live, &flop).unwrap();
    assert!((fast_eq.valid_weight_sum - 29_520.0).abs() <= 1e-12);
    assert!((fast_eq.equity_p0 - frozen_eq_oop).abs() <= 1e-12);
    assert!((fast_eq.equity_p1 - frozen_eq_ip).abs() <= 1e-12);
}

/// D3 — Independent Best-Response vs DCFR Value & Sign Check on the Flop Tiny Game (`1000` DCFR Iterations):
/// Native `IdentityAbstraction`, `20 vs 20` on `Ks9s4d` (exact `bench_flop_tiny` config), `1000` DCFR iterations.
/// Verifies:
/// 1) `BR_OOP >= EV_OOP` and `BR_IP >= EV_IP` (best response can never do worse than the average strategy);
/// 2) `0.5 * ((BR_OOP - EV_OOP) + (BR_IP - EV_IP))` matches `game.exploitability()` within `<= 1e-4` chips!
#[test]
fn d3_flop_tiny_20v20_1000_iters_best_response_vs_dcfr_ev_and_exploitability() {
    let flop: Board = "Ks9s4d".parse().unwrap();
    let oop: PloRange = "AAQQ$ds, JJTT$ds, 8877$ds, AsJsQhTh, AsJsQcTc"
        .parse()
        .unwrap();
    let ip: PloRange = "AAJJ$ds, QQTT$ds, 7766$ds, AsQsJhTh, AsQsJcTc"
        .parse()
        .unwrap();

    let two_sizes = StreetBetSizes {
        bet_fractions: vec![0.25, 0.5],
        raise_fractions: vec![1.0],
        donk_fractions: None,
    };
    let tree = ActionTree::build(TreeConfig {
        initial_street: Street::Flop,
        starting_pot: 100,
        effective_stack: 50,
        betting_limit: BettingLimit::PotLimit,
        flop_sizes: [two_sizes.clone(), two_sizes.clone()],
        turn_sizes: [two_sizes.clone(), two_sizes.clone()],
        river_sizes: [two_sizes.clone(), two_sizes],
        ..Default::default()
    })
    .unwrap();

    let mut game = PloGame::with_options(
        flop,
        &oop,
        &ip,
        &tree,
        &IdentityAbstraction,
        SolverBuildOptions {
            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
            enable_chance_isomorphism: true,
            ..Default::default()
        },
    )
    .unwrap();

    for _ in 0..1000 {
        game.solve_step();
    }

    let [ev_oop, ev_ip] = game.root_expected_values_f64();
    let [br_oop, br_ip] = game.best_response_values_f64();
    let solver_expl = game.exploitability() as f64;

    // Sign check: Best Response >= Average Strategy EV (within tiny f32/f64 rounding ~1e-6)
    assert!(
        br_oop >= ev_oop - 1e-6,
        "BR_OOP ({br_oop}) must be >= EV_OOP ({ev_oop})"
    );
    assert!(
        br_ip >= ev_ip - 1e-6,
        "BR_IP ({br_ip}) must be >= EV_IP ({ev_ip})"
    );

    let indep_expl = 0.5 * ((br_oop - ev_oop) + (br_ip - ev_ip));
    assert!(
        indep_expl >= 0.0,
        "independent exploitability must be non-negative: {indep_expl}"
    );
    assert!(
        (indep_expl - solver_expl).abs() <= 1e-4,
        "indep_expl={indep_expl:.9} vs solver_expl={solver_expl:.9}"
    );
    assert!(
        solver_expl < 1e-3,
        "expected converged 1000-iter flop tiny exploitability < 1e-3, got {solver_expl}"
    );
}
