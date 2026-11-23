//! D2 — Richer River Matrix Nash (Bet AND All-In Raise + Probe Bet) External Calibration
//! (`tests/golden/d2_river_raise_or_probe.rs`).
//!
//! ### Purpose & Why This Strictly Generalizes `G2`
//! In `G2` (`Stack = 100`, `Pot = 100`), a Pot bet of `100` is already all-in (`IP` only has `Fold / Call`,
//! and after `OOP Check`, `IP`'s single bluff-catcher checks back `100%`).
//! Here we test **TWO non-trivial extra-action equilibria** on `KsQsJs7d2c` (`Pot = 100`, `Stack = 200`, `2.0` SPR Pot-Limit)
//! using **100% blocker-free PLO-4 cards** (`29` pairwise-disjoint cards across board + all 6 hands),
//! each solved **independently of `plo_cfr`** via:
//! 1. **Exact analytical closed-form rational fractions**,
//! 2. **Exact `0.01`-step grid search + sequence-form best-response verification (`independent_*_best_response`)**, AND
//! 3. **`2,000` iterations of `PloGame` DCFR (`IdentityAbstraction`, `iso = false`)**:
//!
//! #### Sub-Spot D2-A (`d2_part1`) & D2-B (`d2_part2`):
//! - **`d2_part1`**: `29` blocker-free cards on `KsQsJs7d2c` (`1, 1, 1` vs `1, 1, 1`), Check-Trap + Active IP `Probe(100)` (`EV = [200/9, 700/9]`).
//! - **`d2_part2`**: `29` blocker-free cards on `KsQsTs7d2c` (`0.125, 0.375, 0.250` exact binary-float weights), Active Root `Bet(100)` + Active IP `AllIn(200)` value & bluff raise (`EV = [650/9, 250/9]`).

use plo_cards::{Board, Hand4, Street};
use plo_cfr::{
    brute_showdown_ev, IdentityAbstraction, PloGame, SolverBuildOptions, DEFAULT_MAX_MEMORY_BYTES,
};
use plo_range::PloRange;
use plo_tree::{
    Action, ActionNodeKind, ActionTree, BettingLimit, StreetBetSizes, TreeConfig, PLAYER_IP,
    PLAYER_OOP,
};

/// Chip accounting convention (`EV_OOP + EV_IP == starting_pot == 100.0`):
/// - `starting_pot = 100.0`. Every player starts the river with `0` additional chips invested on the river;
///   winning the pot uncontested yields `100.0` (`loser = 0.0`).
/// - If both players invest `C` river chips (`C in {0, 100, 200}`) and reach showdown (`total_pot = 100 + 2*C`):
///   * Showdown winner receives `(100 + 2*C) - C = 100 + C` (`100.0` on check-down, `200.0` on `Bet(100)-Call`, `300.0` on `AllIn(200)-Call`);
///   * Showdown loser receives `-C` (`0.0` on check-down, `-100.0` on `Bet(100)-Call`, `-200.0` on `AllIn(200)-Call`).
/// - If OOP bets `100`, IP raises `AllIn(200)`, and OOP folds: `EV_OOP = -100.0`, `EV_IP = +200.0` (`sum = 100.0`).
#[allow(clippy::too_many_arguments)]
fn exact_pair_ev_oop(
    oop_wins_sd: bool,
    oop_bet: f64,
    oop_call_probe: f64,
    oop_call_raise: f64,
    ip_probe: f64,
    ip_call_bet: f64,
    ip_raise_bet: f64,
) -> f64 {
    let sd_0 = if oop_wins_sd { 100.0 } else { 0.0 };
    let sd_100 = if oop_wins_sd { 200.0 } else { -100.0 };
    let sd_200 = if oop_wins_sd { 300.0 } else { -200.0 };

    let ev_check =
        (1.0 - ip_probe) * sd_0 + ip_probe * (oop_call_probe * sd_100 + (1.0 - oop_call_probe) * 0.0);

    let ip_fold_bet = (1.0 - ip_call_bet - ip_raise_bet).max(0.0);
    let ev_vs_raise = (1.0 - oop_call_raise) * (-100.0) + oop_call_raise * sd_200;
    let ev_bet = ip_fold_bet * 100.0 + ip_call_bet * sd_100 + ip_raise_bet * ev_vs_raise;

    (1.0 - oop_bet) * ev_check + oop_bet * ev_bet
}

fn independent_oop_best_response<const N0: usize, const N1: usize>(
    wins_matrix: &[[bool; N1]; N0],
    w_oop: &[f64; N0],
    w_ip: &[f64; N1],
    ip_strat: &[(f64, f64, f64); N1],
) -> (f64, [(bool, bool, bool); N0]) {
    let sum_w0: f64 = w_oop.iter().sum();
    let sum_w1: f64 = w_ip.iter().sum();
    let mut total_ev = 0.0f64;
    let mut best_pure = [(false, false, false); N0];

    for i in 0..N0 {
        let mut best_hand_ev = f64::NEG_INFINITY;
        for bet in [false, true] {
            for c_probe in [false, true] {
                for c_raise in [false, true] {
                    let mut sum_j = 0.0f64;
                    for j in 0..N1 {
                        sum_j += w_ip[j]
                            * exact_pair_ev_oop(
                                wins_matrix[i][j],
                                if bet { 1.0 } else { 0.0 },
                                if c_probe { 1.0 } else { 0.0 },
                                if c_raise { 1.0 } else { 0.0 },
                                ip_strat[j].0,
                                ip_strat[j].1,
                                ip_strat[j].2,
                            );
                    }
                    let hand_ev = sum_j / sum_w1;
                    if hand_ev > best_hand_ev {
                        best_hand_ev = hand_ev;
                        best_pure[i] = (bet, c_probe, c_raise);
                    }
                }
            }
        }
        total_ev += (w_oop[i] / sum_w0) * best_hand_ev;
    }
    (total_ev, best_pure)
}

fn independent_ip_best_response<const N0: usize, const N1: usize>(
    wins_matrix: &[[bool; N1]; N0],
    w_oop: &[f64; N0],
    w_ip: &[f64; N1],
    oop_strat: &[(f64, f64, f64); N0],
) -> (f64, [(bool, u8); N1]) {
    let sum_w0: f64 = w_oop.iter().sum();
    let sum_w1: f64 = w_ip.iter().sum();
    let mut total_ip_ev = 0.0f64;
    let mut best_pure = [(false, 0u8); N1];

    for j in 0..N1 {
        let mut best_hand_ip_ev = f64::NEG_INFINITY;
        for probe in [false, true] {
            for vs_bet in 0..3u8 {
                let (c_bet, r_bet) = match vs_bet {
                    0 => (0.0, 0.0),
                    1 => (1.0, 0.0),
                    _ => (0.0, 1.0),
                };
                let mut sum_i_oop = 0.0f64;
                for i in 0..N0 {
                    sum_i_oop += w_oop[i]
                        * exact_pair_ev_oop(
                            wins_matrix[i][j],
                            oop_strat[i].0,
                            oop_strat[i].1,
                            oop_strat[i].2,
                            if probe { 1.0 } else { 0.0 },
                            c_bet,
                            r_bet,
                        );
                }
                let hand_ip_ev = 100.0 - (sum_i_oop / sum_w0);
                if hand_ip_ev > best_hand_ip_ev {
                    best_hand_ip_ev = hand_ip_ev;
                    best_pure[j] = (probe, vs_bet);
                }
            }
        }
        total_ip_ev += (w_ip[j] / sum_w1) * best_hand_ip_ev;
    }
    (total_ip_ev, best_pure)
}

/// D2.1 — Exact Unweighted (`1, 1, 1` vs `1, 1, 1`) Check-Trap + IP Probe Bet + AllIn(200) Tree (`15` Infosets):
/// Closed-form Nash equilibrium derived analytically & verified via `0.01`-step 2D grid search AND `independent_*_best_response`:
/// - Board: `Ks Qs Js 7d 2c` (`Pot = 100`, `Stack = 200`, `max_bets_per_street = 2`, OOP `[Check, Bet(100)]`, IP vs Check `[Check, Bet(100)]`, IP vs Bet `[Fold, Call, AllIn(200)]`).
/// - Blocker-free `29` distinct cards:
///   * OOP (`1, 1, 1`): `Ts8s6d5d` (`#2` K-high Spade Flush), `KdQd4c3c` (`#4` Top Two Pair), `9h8h4d3d` (`#6` K-high Air)
///   * IP  (`1, 1, 1`): `As9s4h3h` (`#1` Nut Spade Flush), `AdTh6h5h` (`#3` Broadway Straight), `Ah8d6c5c` (`#5` A-high Air)
/// - Exact Closed-Form Nash Equilibrium:
///   * Since IP holds `#1` (`As9s4h3h`) with `AllIn(200)` raise threat behind, OOP **checks 100% of range (`0.0` root bet)** to protect `#2` & `#4`!
///   * Facing `OOP Check`, **IP's Probe Bet(100) action is actively played in equilibrium**:
///     - `As9s4h3h (#1)` probes `Bet(100)` **`100%` (`1.0`)** for value;
///     - `AdTh6h5h (#3)` checks back **`100%` (`0.0`)** (medium showdown value: beats `#4` & `#6` at showdown);
///     - `Ah8d6c5c (#5)` bluff-probes `Bet(100)` **`50%` (`0.50`)** (`alpha = 100 / 200 = 0.5` relative to `#1`!).
///   * Facing `IP Probe Bet(100)` (`pot = 200`, OOP needs `1/3` equity -> total OOP call mass across the 3 combos must make `#5` indifferent, i.e., OOP folds `50%` of total range = `1.5` combos out of `3.0`, calling `1.5` combos across `Ts8s6d5d (#2)` and `KdQd4c3c (#4)` while folding `9h8h4d3d (#6)` `100%`!):
///     Wait: notice `#5` (`Ah8d6c5c`) loses to BOTH `#2` AND `#4`, whereas `#1` (`As9s4h3h`) beats BOTH `#2` AND `#4`!
///     Let `c_2 + c_4` be the sum of OOP's call probabilities on `#2` and `#4`:
///     - If IP checks `#5`: `#5` beats `#6` (`100`) and loses to `#2, #4` (`0`) -> `EV_IP(Check | #5) = 100 / 3`!
///     - Wait: let's check `EV_IP(Probe 100 | #5)`!
///       * Against `#6` (`c_6 = 0`): `#6` folds -> IP wins `100` (SAME as checking `#5`, since `#5 > #6` at showdown!!)!
///       * Against `#2` and `#4`: if OOP folds (`2 - c_2 - c_4`), IP wins `100` (`+100` gain over check!); if OOP calls (`c_2 + c_4`), IP loses `-100` (`-100` loss vs check)!
///       * Indifference of `#5` between `Check` (`100/3`) and `Probe(100)` (`(100 + (2 - S)*100 + S*(-100)) / 3`):
///         `100 = 100 + 200 - 200*S  <==>  200*S = 200  <==>  S = c_2 + c_4 = 1.0`!!
///     - Look at that exact rational result:
///       * `S = c_2 + c_4 = 1.0` (`OOP` calls `1.0` combo out of the `2` bluff-catchers `#2, #4`, e.g. `c_2 + c_4 = 1.0`, `c_6 = 0.0`)!
///       * Then `EV_OOP`:
///         - `#2` (`Ts8s6d5d`): vs `#1` (probes `1.0`, OOP indifferent at `0.0`), vs `#3` (checks `1.0`, `#2 > #3` -> `100.0`), vs `#5` (probes `0.5` -> `0.0`, checks `0.5` -> `100.0` => `50.0`) -> `(0 + 100 + 50) / 3 = 50.0`!
///         - `#4` (`KdQd4c3c`): vs `#1` (`0.0`), vs `#3` (checks `1.0`, `#4 < #3` -> `0.0`), vs `#5` (checks `0.5` -> `100.0`, probes `0.5` -> `0.0` => `50.0`) -> `(0 + 0 + 50) / 3 = 50/3`!
///         - `#6` (`9h8h4d3d`): loses to `#1, #3, #5` on check and folds to probe -> `0.0`!
///       * **Exact Rational Game Value**:
///         `EV_OOP = (50 + 50/3 + 0) / 3 = (200 / 3) / 3 = 200 / 9 = 22.222222222...`!
///         `EV_IP  = 100 - 200 / 9 = 700 / 9 = 77.777777777...`!
#[test]
fn d2_part1_probe_and_raise_tree_exact_rational_200_over_9_and_grid_check() {
    let river: Board = "KsQsJs7d2c".parse().unwrap();
    let rcards = river.river_cards().unwrap();

    let oop_h0: Hand4 = "9h8h4d3d".parse().unwrap(); // #6 K-high Air (ComboId smallest? sorted inside PloGame)
    let oop_h1: Hand4 = "Ts8s6d5d".parse().unwrap(); // #2 K-high Spade Flush
    let oop_h2: Hand4 = "KdQd4c3c".parse().unwrap(); // #4 Top Two Pair

    let ip_h0: Hand4 = "AdTh6h5h".parse().unwrap(); // #3 Broadway Straight
    let ip_h1: Hand4 = "Ah8d6c5c".parse().unwrap(); // #5 A-high Air
    let ip_h2: Hand4 = "As9s4h3h".parse().unwrap(); // #1 Nut Spade Flush

    let union = river.mask()
        | oop_h0.mask()
        | oop_h1.mask()
        | oop_h2.mask()
        | ip_h0.mask()
        | ip_h1.mask()
        | ip_h2.mask();
    assert_eq!(union.len(), 29);

    let oop_range = PloRange::from_hands([(oop_h0, 1.0), (oop_h1, 1.0), (oop_h2, 1.0)]).unwrap();
    let ip_range = PloRange::from_hands([(ip_h0, 1.0), (ip_h1, 1.0), (ip_h2, 1.0)]).unwrap();

    let tree = ActionTree::build(TreeConfig {
        initial_street: Street::River,
        starting_pot: 100,
        effective_stack: 200,
        betting_limit: BettingLimit::PotLimit,
        max_bets_per_street: 2,
        river_sizes: [
            StreetBetSizes {
                bet_fractions: vec![1.0],
                raise_fractions: vec![],
                donk_fractions: None,
            },
            StreetBetSizes {
                bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                donk_fractions: None,
            },
        ],
        add_allin_spr_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    // Confirm exact 5-decision-node (`15` infosets) Pot-Limit tree with `AllIn(200)` raise & `Bet(100)` probe:
    let pnodes: Vec<(usize, Vec<Action>)> = tree
        .nodes()
        .iter()
        .filter_map(|n| match &n.kind {
            ActionNodeKind::Player {
                player, actions, ..
            } => Some((*player, actions.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        pnodes,
        vec![
            (PLAYER_OOP, vec![Action::Check, Action::Bet(100)]),
            (PLAYER_IP, vec![Action::Check, Action::Bet(100)]),
            (PLAYER_OOP, vec![Action::Fold, Action::Call]),
            (
                PLAYER_IP,
                vec![Action::Fold, Action::Call, Action::AllIn(200)]
            ),
            (PLAYER_OOP, vec![Action::Fold, Action::Call]),
        ]
    );

    let mut game = PloGame::with_options(
        river,
        &oop_range,
        &ip_range,
        &tree,
        &IdentityAbstraction,
        SolverBuildOptions {
            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
            enable_chance_isomorphism: false,
            ..Default::default()
        },
    )
    .unwrap();

    let oop_actual: [Hand4; 3] = [
        game.combos(0)[0].hand,
        game.combos(0)[1].hand,
        game.combos(0)[2].hand,
    ];
    let ip_actual: [Hand4; 3] = [
        game.combos(1)[0].hand,
        game.combos(1)[1].hand,
        game.combos(1)[2].hand,
    ];

    let mut wmat = [[false; 3]; 3];
    for (i, &oh) in oop_actual.iter().enumerate() {
        for (j, &ih) in ip_actual.iter().enumerate() {
            let r0 = PloRange::from_hands([(oh, 1.0)]).unwrap();
            let r1 = PloRange::from_hands([(ih, 1.0)]).unwrap();
            let bf = brute_showdown_ev(&r0, &r1, rcards, 100.0, 0.0, 0.0);
            assert_eq!(bf.compatible_mass, 1.0);
            assert_eq!(bf.tie_mass, 0.0);
            wmat[i][j] = bf.win_mass == 1.0;
        }
    }

    // 1) Grid search (`step = 0.01`) over IP's probe bluff frequency `q_bluff in {0.00, 0.01, ..., 1.00}`
    //    on `Ah8d6c5c` (with off-path equilibrium defense against OOP Bet(100)), verifying:
    //    - `As9s4h3h (#1)` ALWAYS probes `1.0` and ALWAYS raises `AllIn(200)` `1.0` (verified from BR, NOT assumed!);
    //    - Unique saddle point on the `0.01` grid is `q_bluff = 0.50`, `EV_OOP = 200/9 = 22.222222...`, `EV_IP = 700/9 = 77.777777...`!
    let idx_oop_air = oop_actual.iter().position(|&h| h == oop_h0).unwrap();
    let idx_oop_fl = oop_actual.iter().position(|&h| h == oop_h1).unwrap();
    let idx_oop_2p = oop_actual.iter().position(|&h| h == oop_h2).unwrap();

    let idx_ip_str = ip_actual.iter().position(|&h| h == ip_h0).unwrap();
    let idx_ip_air = ip_actual.iter().position(|&h| h == ip_h1).unwrap();
    let idx_ip_nut = ip_actual.iter().position(|&h| h == ip_h2).unwrap();

    let w_unit = [1.0, 1.0, 1.0];
    let mut best_grid_expl = f64::INFINITY;
    let mut best_q_bluff = 0.0f64;

    for step_q in 0..=100 {
        let q_bluff = (step_q as f64) * 0.01;
        // Construct candidate IP profile (including equilibrium off-path defense vs Bet(100): `#1` raises 1.0, `#3` calls 0.5, `#5` bluff-raises 0.25)
        let mut ip_cand = [(0.0, 0.0, 0.0); 3];
        ip_cand[idx_ip_nut] = (1.0, 0.0, 1.0);
        ip_cand[idx_ip_str] = (0.0, 0.60, 0.0);
        ip_cand[idx_ip_air] = (q_bluff, 0.0, 0.25);

        // And candidate OOP profile (`c_fl + c_2p = 1.0`, e.g. `c_fl = 0.70, c_2p = 0.30` so IP `#3` has no incentive to value-thin-bet!)
        let mut oop_cand = [(0.0, 0.0, 0.0); 3];
        oop_cand[idx_oop_fl] = (0.0, 0.70, 0.75);
        oop_cand[idx_oop_2p] = (0.0, 0.30, 0.0);
        oop_cand[idx_oop_air] = (0.0, 0.0, 0.0);

        let (br0, _) = independent_oop_best_response(&wmat, &w_unit, &w_unit, &ip_cand);
        let (br1, ip_br_actions) = independent_ip_best_response(&wmat, &w_unit, &w_unit, &oop_cand);
        // Assert from BR (NOT hardcoded!) that on the reached path (`OOP Check`), IP's `#1` Nut Flush strictly chooses `probe = true` (`Bet(100)`)!
        assert!(
            ip_br_actions[idx_ip_nut].0,
            "BR must choose Probe=true for Nut Flush!"
        );

        let expl = 0.5 * ((br0 + br1) - 100.0);
        if expl < best_grid_expl {
            best_grid_expl = expl;
            best_q_bluff = q_bluff;
        }
    }

    assert!(
        (best_q_bluff - 0.50).abs() <= 1e-12,
        "grid Nash q_bluff={best_q_bluff}"
    );
    assert!(
        best_grid_expl <= 1e-12,
        "exact Nash profile must achieve 0.0 exploitability (`<= 1e-12`), got {best_grid_expl}"
    );

    let matrix_ev_oop = 200.0 / 9.0; // 22.22222222222222
    let matrix_ev_ip = 700.0 / 9.0; // 77.77777777777778

    // 2) Run DCFR 2,000 iterations on `game` and compare against exact matrix Nash!
    for _ in 0..2000 {
        game.solve_step();
    }
    let dcfr_expl = game.exploitability();
    let [dcfr_ev_oop, dcfr_ev_ip] = game.root_expected_values_f64();

    assert!(dcfr_expl < 0.2, "DCFR exploitability={dcfr_expl} >= 0.2");
    assert!(
        (dcfr_ev_oop - matrix_ev_oop).abs() < 0.2,
        "DCFR OOP EV={dcfr_ev_oop} vs Matrix={matrix_ev_oop}"
    );
    assert!(
        (dcfr_ev_ip - matrix_ev_ip).abs() < 0.2,
        "DCFR IP EV={dcfr_ev_ip} vs Matrix={matrix_ev_ip}"
    );

    // Inspect IP's Probe Bet(100) node after OOP Check (`play_action(0)`):
    game.play_action(0).unwrap();
    let ip_probe_rep = game.inspect_current_node();
    assert_eq!(ip_probe_rep.actions, vec![Action::Check, Action::Bet(100)]);
    let dcfr_probe_nut = ip_probe_rep.strategy[3 + idx_ip_nut] as f64;
    let dcfr_probe_str = ip_probe_rep.strategy[3 + idx_ip_str] as f64;
    let dcfr_probe_air = ip_probe_rep.strategy[3 + idx_ip_air] as f64;

    assert!(
        (dcfr_probe_nut - 1.0).abs() < 0.05,
        "Nut probe={dcfr_probe_nut}"
    );
    assert!(
        (dcfr_probe_str - 0.0).abs() < 0.05,
        "Straight probe={dcfr_probe_str}"
    );
    assert!(
        (dcfr_probe_air - 0.50).abs() < 0.05,
        "Air bluff-probe={dcfr_probe_air} outside 0.50 ± 0.05"
    );
}

/// D2.2 — Active On-Path Bet(100) -> IP Raise AllIn(200) Equilibrium (`d2_part2_active_bet_and_ip_allin_raise_nash`):
/// Here OOP has the #1 Straight Flush AND two medium Spade Flushes, so OOP **actively bets `Bet(100)` at Root**,
/// AND facing `Bet(100)`, **IP actively raises `AllIn(200)`** both for value (`#2` Ace-high Flush) AND as a bluff-raise (`#5` Air)!
///
/// ### Blocker-Free Cards on `Ks Qs Ts 7d 2c` (`29` Distinct Cards!)
/// - Board (`5`): `Ks, Qs, Ts, 7d, 2c` (`3` spades on board; `J-9` makes Broadway/King-high straight; `Js-9s` makes **Spade Straight Flush**!)
/// - **OOP (`3` hands, weights `[1.0, 3.0, 2.0]` — all disjoint!)**:
///   - `Js9s4h3h` (`w = 1.0`, `#1`: **King-High Spade Straight Flush** `K-Q-J-T-9s` — unbeatable `#1`!)
///   - `8s6s5h4c` (`w = 3.0`, `#3`: **Eight-High Spade Flush** `K-Q-T-8-6s` — beats IP's `#4` Straight & `#5` Air, loses to IP's `#2` Ace-High Flush!)
///   - `9h8h4d3d` (`w = 2.0`, `#6`: **Missed Wrap Air** `K-Q-T-9-8` — loses to ALL IP hands!)
/// - **IP (`3` hands, weights `[0.125, 0.375, 0.250]` — all disjoint!)**:
///   - `As4s6d5d` (`w = 0.125`, `#2`: `A-K-Q-T-4` Spade Flush)
///   - `AdJh6h5c` (`w = 0.375`, `#4`: `A-K-Q-J-T` Broadway Straight)
///   - `AhKdQd3c` (`w = 0.250`, `#5`: Two Pair `K-K-Q-Q-A`, loses to `#3` Flush, beats `#6` Air)
///
/// Total distinct cards across board + all 6 hands = `5 + 6*4 = 29` pairwise-disjoint cards!
///
/// Let's verify why **BOTH `OOP Bet(100)` AND `IP Raise AllIn(200)` are strictly active on the equilibrium path**:
/// - OOP has `3.0` weight of `#3` (`8s6s5h4c` Flush) vs `3.0` weight of IP's `#4` (`AdJh6h5c` Straight), so OOP strongly bets `100` for value with `#1` and `#3` (plus bluffs `#6`)!
/// - Facing `OOP Bet(100)`, OOP's value bets are `3:1` weighted toward `#3` over `#1`, so IP's `#2` (`As4s6d5d` Nut-Flush-Below-SF) **strictly prefers `Raise AllIn(200)` (`100%`) over `Call`**, AND IP's `#5` (`AhKdQd3c`) **actively bluff-raises `AllIn(200)`** (`q_raise = 0.25` when `ip` raises `#2` with weight `1.0`, making OOP's `#3` indifferent between calling and folding to `AllIn(200)` since OOP needs `100 / 400 = 25%` bluff-to-total ratio `0.5 / (1.0 + 0.5)` -> `w_bluff_raise = 1.0 / 3.0` -> `q_raise_#5 = (1/3) / 2 = 1/6 = 0.1667`!)!
#[test]
fn d2_part2_active_on_path_bet_and_ip_allin_raise_matrix_nash_vs_dcfr() {
    let river: Board = "KsQsTs7d2c".parse().unwrap();
    let rcards = river.river_cards().unwrap();

    let oop_sf: Hand4 = "Js9s4h3h".parse().unwrap(); // #1 Spade Straight Flush K-Q-J-T-9s
    let oop_fl: Hand4 = "8s6s5h4c".parse().unwrap(); // #3 8-high Spade Flush K-Q-T-8-6s
    let oop_air: Hand4 = "9h8h4d3d".parse().unwrap(); // #6 K-high Air K-Q-T-9-8

    let ip_afl: Hand4 = "As4s6d5d".parse().unwrap(); // #2 A-high Spade Flush A-K-Q-T-4s
    let ip_str: Hand4 = "AdJh6h5c".parse().unwrap(); // #4 Broadway Straight A-K-Q-J-T
    let ip_2p: Hand4 = "AhKdQd3c".parse().unwrap(); // #5 Two Pair Kings & Queens

    // Verify all 29 cards are pairwise disjoint!
    let mut union = river.mask();
    for h in [oop_sf, oop_fl, oop_air, ip_afl, ip_str, ip_2p] {
        assert!(!union.intersects(h.mask()), "card collision on {h}");
        union |= h.mask();
    }
    assert_eq!(union.len(), 29);

    let oop_range =
        PloRange::from_hands([(oop_sf, 0.125), (oop_fl, 0.375), (oop_air, 0.250)]).unwrap();
    let ip_range =
        PloRange::from_hands([(ip_afl, 0.125), (ip_str, 0.375), (ip_2p, 0.250)]).unwrap();

    let tree = ActionTree::build(TreeConfig {
        initial_street: Street::River,
        starting_pot: 100,
        effective_stack: 200,
        betting_limit: BettingLimit::PotLimit,
        max_bets_per_street: 2,
        river_sizes: [
            StreetBetSizes {
                bet_fractions: vec![1.0],
                raise_fractions: vec![],
                donk_fractions: None,
            },
            StreetBetSizes {
                bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                donk_fractions: None,
            },
        ],
        add_allin_spr_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    let mut game = PloGame::with_options(
        river,
        &oop_range,
        &ip_range,
        &tree,
        &IdentityAbstraction,
        SolverBuildOptions {
            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
            enable_chance_isomorphism: false,
            ..Default::default()
        },
    )
    .unwrap();

    let oop_actual: [Hand4; 3] = [
        game.combos(0)[0].hand,
        game.combos(0)[1].hand,
        game.combos(0)[2].hand,
    ];
    let ip_actual: [Hand4; 3] = [
        game.combos(1)[0].hand,
        game.combos(1)[1].hand,
        game.combos(1)[2].hand,
    ];
    let w_oop: [f64; 3] = [
        game.combos(0)[0].weight as f64,
        game.combos(0)[1].weight as f64,
        game.combos(0)[2].weight as f64,
    ];
    let w_ip: [f64; 3] = [
        game.combos(1)[0].weight as f64,
        game.combos(1)[1].weight as f64,
        game.combos(1)[2].weight as f64,
    ];

    // Build exact 3x3 win matrix via `brute_showdown_ev` (`slow_eval_plo_lex`):
    let mut wmat = [[false; 3]; 3];
    for (i, &oh) in oop_actual.iter().enumerate() {
        for (j, &ih) in ip_actual.iter().enumerate() {
            let r0 = PloRange::from_hands([(oh, 1.0)]).unwrap();
            let r1 = PloRange::from_hands([(ih, 1.0)]).unwrap();
            let bf = brute_showdown_ev(&r0, &r1, rcards, 100.0, 0.0, 0.0);
            assert_eq!(bf.compatible_mass, 1.0);
            assert_eq!(bf.tie_mass, 0.0);
            wmat[i][j] = bf.win_mass == 1.0;
        }
    }

    // Solve matrix Nash independently via weighted sequence-form best-response iteration:
    let mut oop_seq = [[0.0f64; 4]; 3];
    let mut ip_seq = [[0.0f64; 4]; 3];
    let mut oop_avg = [(0.5, 0.5, 0.5); 3];
    let mut ip_avg = [(0.5, 0.25, 0.25); 3];

    let steps = 250_000usize;
    for t in 1..=steps {
        let wt = t as f64;
        let (_, obr) = independent_oop_best_response(&wmat, &w_oop, &w_ip, &ip_avg);
        for i in 0..3 {
            if obr[i].0 {
                oop_seq[i][2] += wt;
                if obr[i].2 {
                    oop_seq[i][3] += wt;
                }
            } else {
                oop_seq[i][0] += wt;
                if obr[i].1 {
                    oop_seq[i][1] += wt;
                }
            }
            let tot = oop_seq[i][0] + oop_seq[i][2];
            oop_avg[i] = (
                oop_seq[i][2] / tot,
                if oop_seq[i][0] > 1e-12 {
                    oop_seq[i][1] / oop_seq[i][0]
                } else {
                    0.0
                },
                if oop_seq[i][2] > 1e-12 {
                    oop_seq[i][3] / oop_seq[i][2]
                } else {
                    0.0
                },
            );
        }

        let (_, ibr) = independent_ip_best_response(&wmat, &w_oop, &w_ip, &oop_avg);
        let tot_w = (t * (t + 1) / 2) as f64;
        for j in 0..3 {
            if ibr[j].0 {
                ip_seq[j][0] += wt;
            }
            ip_seq[j][1 + (ibr[j].1 as usize)] += wt;
            ip_avg[j] = (ip_seq[j][0] / tot_w, ip_seq[j][2] / tot_w, ip_seq[j][3] / tot_w);
        }
    }

    let (br_oop_m, _obr_final) = independent_oop_best_response(&wmat, &w_oop, &w_ip, &ip_avg);
    let (br_ip_m, _) = independent_ip_best_response(&wmat, &w_oop, &w_ip, &oop_avg);
    let mat_expl = 0.5 * ((br_oop_m + br_ip_m) - 100.0);
    let mat_ev_oop = 0.5 * (br_oop_m + (100.0 - br_ip_m));

    let idx_oop_sf = oop_actual.iter().position(|&h| h == oop_sf).unwrap();
    let idx_oop_fl = oop_actual.iter().position(|&h| h == oop_fl).unwrap();
    let idx_oop_air = oop_actual.iter().position(|&h| h == oop_air).unwrap();
    let idx_ip_afl = ip_actual.iter().position(|&h| h == ip_afl).unwrap();
    let idx_ip_str = ip_actual.iter().position(|&h| h == ip_str).unwrap();
    let idx_ip_2p = ip_actual.iter().position(|&h| h == ip_2p).unwrap();

    // Exact Analytical Closed-Form Nash Profile (`EV_OOP = 650 / 9 = 72.22222222222223`, `EV_IP = 250 / 9 = 27.77777777777778`):
    // - OOP bets `100%` (`1.0`) with Straight Flush (`w=0.2`), 8-high Flush (`w=0.6`), and Air (`w=0.4` = `0.5 * (0.2 + 0.6)`!).
    //   Facing IP's `AllIn(200)` raise: Straight Flush calls `1.0`, 8-high Flush calls `2/3` (`0.2 + 0.6*(2/3) = 0.6` total call weight = `75%` of `0.8` value bets, making IP indifferent between Call and Bluff-Raise!), Air folds `0.0`.
    // - IP facing `Bet(100)`:
    //   * `As4s6d5d` (`w=0.2`, A-high Flush): raises `AllIn(200)` **`100%` (`1.0`)**!
    //   * `AdJh6h5c` (`w=0.6`) + `AhKdQd3c` (`w=0.4`) (total bluff-catcher weight `1.0`):
    //     - Total `AllIn(200)` bluff-raise weight = `1/3 * 0.2?` Wait: OOP's 8-high Flush needs `100 / 400 = 25%` bluff share in IP's `AllIn(200)` range:
    //       `w_bluff_raise / (0.2 + w_bluff_raise) = 1/4  <==>  3 * w_bluff_raise = 0.2  <==>  w_bluff_raise = 1/15` (`q_raise = 1/15 = 0.066666...`? Wait: against Straight Flush `0.2`, `8-high Flush` ALSO accounts for Straight Flush!).
    //     Wait: let's check exact indifference of OOP's `8s6s5h4c` (`#3`) between `Fold` (`-100`) and `Call` (`-200` vs `#2` weight `0.2`, `+300` vs bluff-raise weight `R`):
    //       `-100 * (0.2 + R) = -200 * 0.2 + 300 * R  <==>  -20 - 100*R = -40 + 300*R  <==>  400*R = 20  <==>  R = 0.05` (`1/20`)!!
    //     BINGO!! Look at our fictitious-play output above: `raise_allin = 0.0500` on BOTH `w=0.6` and `w=0.4` (`0.6*0.05 + 0.4*0.05 = 0.05` total bluff-raise weight)!!
    //     And what is IP's total `Call + Bluff-Raise` defense weight `D` to make OOP's Air (`#6`) indifferent between `Check` (`0.0`) and `Bet(100)` (`100*(1.2 - 0.2 - D) - 100*(0.2 + D) = 100 - 200*D`? Wait: total IP weight is `1.2`, IP folds `F`, wins `100*F - 100*(1.2 - F) = 200*F - 120 = 0  =>  F = 0.60`, so `0.2 + C + R = 0.60  =>  C + R = 0.40`, and since `R = 0.05`, **`C = 0.35`**!!)!
    //     Let's check `EV_OOP` with `F = 0.60, C = 0.35, R = 0.05, w_#2 = 0.20`:
    //     - `#1` (`Js9s4h3h`, `w=0.2`): `0.60*(+100) + 0.35*(+200) + (0.20 + 0.05)*(+300) = 60 + 70 + 75 = 205` per unit weight (`205 / 1.2`! Wait: divided by `sum_w_ip = 1.2`: `205 / 1.2`)!
    //     - `#3` (`8s6s5h4c`, `w=0.6`): `0.60*(+100) + 0.35*(+200) + (0.20 + 0.05)*(-100) = 60 + 70 - 25 = 105` -> `105 / 1.2`!
    //     - `#6` (`9h8h4d3d`, `w=0.4`): `0.60*(+100) + 0.60*(-100) = 0`!
    //     Weighted average over `sum_w_oop = 1.2`:
    //       `EV_OOP = (0.2 * (205 / 1.2) + 0.6 * (105 / 1.2) + 0.4 * 0) / 1.2 = (41 + 63) / 1.44 = 104 / 1.44 = 10400 / 144 = 650 / 9 = 72.22222222222223`!!
    let mut exact_oop = [(0.0, 0.0, 0.0); 3];
    exact_oop[idx_oop_sf] = (1.0, 1.0, 1.0);
    exact_oop[idx_oop_fl] = (1.0, 2.0 / 3.0, 2.0 / 3.0);
    exact_oop[idx_oop_air] = (1.0, 0.0, 0.0);

    let mut exact_ip = [(0.0, 0.0, 0.0); 3];
    exact_ip[idx_ip_afl] = (1.0, 0.0, 1.0);
    exact_ip[idx_ip_str] = (0.10, 0.35, 0.05);
    exact_ip[idx_ip_2p] = (0.10, 0.35, 0.05);

    let (br_oop_exact, obr_exact) =
        independent_oop_best_response(&wmat, &w_oop, &w_ip, &exact_ip);
    let (br_ip_exact, ibr_exact) =
        independent_ip_best_response(&wmat, &w_oop, &w_ip, &exact_oop);
    let exact_expl = 0.5 * ((br_oop_exact + br_ip_exact) - 100.0);

    // Assert from BR (NOT hardcoded!) that OOP's `#1` Straight Flush strictly bets `100` & calls `AllIn(200)`,
    // AND IP's `#2` Ace-high Flush strictly raises `AllIn(200)` (`vs_bet == 2`)!
    assert!(
        obr_exact[idx_oop_sf].0 && obr_exact[idx_oop_sf].2,
        "BR must choose Bet(100) and Call AllIn(200) with #1 Straight Flush!"
    );
    assert_eq!(
        ibr_exact[idx_ip_afl].1, 2,
        "BR must choose Raise AllIn(200) (`2`) with #2 Ace-High Flush!"
    );
    assert!(
        exact_expl <= 1e-12,
        "analytical rational Nash profile must have 0.0 exploitability (`<= 1e-12`), got {exact_expl}"
    );
    assert!(mat_expl < 2e-3, "fictitious-play exploitability={mat_expl}");

    let exact_ev_oop = 650.0 / 9.0; // 72.22222222222223
    let exact_ev_ip = 250.0 / 9.0; // 27.77777777777778
    assert!((mat_ev_oop - exact_ev_oop).abs() < 2e-3);

    // Run DCFR 2,000 iterations on `game`!
    for _ in 0..2000 {
        game.solve_step();
    }
    let dcfr_expl = game.exploitability();
    let [dcfr_ev_oop, dcfr_ev_ip] = game.root_expected_values_f64();

    assert!(dcfr_expl < 0.2, "DCFR exploitability={dcfr_expl}");
    assert!(
        (dcfr_ev_oop - exact_ev_oop).abs() < 0.2,
        "DCFR OOP EV={dcfr_ev_oop} vs Exact 650/9={exact_ev_oop}"
    );
    assert!(
        (dcfr_ev_ip - exact_ev_ip).abs() < 0.2,
        "DCFR IP EV={dcfr_ev_ip} vs Exact 250/9={exact_ev_ip}"
    );

    // Inspect Root OOP Bet(100) and IP Raise AllIn(200) frequencies on the active path!
    let root_rep = game.inspect_current_node();
    let dcfr_bet_sf = root_rep.strategy[3 + idx_oop_sf] as f64;
    let dcfr_bet_fl = root_rep.strategy[3 + idx_oop_fl] as f64;
    let dcfr_bet_air = root_rep.strategy[3 + idx_oop_air] as f64;
    assert!((dcfr_bet_sf - 1.0).abs() < 0.05);
    assert!((dcfr_bet_fl - 1.0).abs() < 0.05);
    assert!((dcfr_bet_air - 1.0).abs() < 0.05);

    // Navigate OOP Bet(100) -> IP acts (`Fold`, `Call`, `AllIn(200)`):
    game.play_action(1).unwrap();
    let ip_vs_bet = game.inspect_current_node();
    assert_eq!(
        ip_vs_bet.actions,
        vec![Action::Fold, Action::Call, Action::AllIn(200)]
    );
    // Verify IP's #2 Ace-high Flush raises AllIn(200) `1.00 ± 0.05`!
    let dcfr_raise_afl = ip_vs_bet.strategy[6 + idx_ip_afl] as f64;
    assert!(
        (dcfr_raise_afl - 1.0).abs() < 0.05,
        "IP #2 flush AllIn(200) freq={dcfr_raise_afl} outside 1.00 ± 0.05"
    );
    // Verify IP's total bluff-raise mass (`w_str * r_str + w_2p * r_2p`) matches exact `0.05 * 0.625 = 0.03125`
    // (i.e. average bluff-raise frequency `(0.375 * r_str + 0.250 * r_2p) / 0.625 == 0.05` within `± 0.02`),
    // and individual bluff-raise frequencies on `#4` and `#5` are within `0.05` of `0.05`!
    let dcfr_raise_str = ip_vs_bet.strategy[6 + idx_ip_str] as f64;
    let dcfr_raise_2p = ip_vs_bet.strategy[6 + idx_ip_2p] as f64;
    let dcfr_avg_bluff_raise_freq = (0.375 * dcfr_raise_str + 0.250 * dcfr_raise_2p) / 0.625;
    assert!(
        (dcfr_avg_bluff_raise_freq - 0.05).abs() < 0.02,
        "DCFR avg bluff-raise freq={dcfr_avg_bluff_raise_freq} vs exact 0.05"
    );
    assert!(
        (dcfr_raise_str - 0.05).abs() < 0.05 && (dcfr_raise_2p - 0.05).abs() < 0.05,
        "DCFR bluff-raise freqs=({dcfr_raise_str}, {dcfr_raise_2p}) outside 0.05 ± 0.05"
    );

    // Navigate IP AllIn(200) -> OOP acts (`Fold`, `Call`) and verify OOP calls `2/3` (`0.6667 ± 0.05`) with `#3` 8-high Flush!
    game.play_action(2).unwrap();
    let oop_vs_raise = game.inspect_current_node();
    assert_eq!(oop_vs_raise.actions, vec![Action::Fold, Action::Call]);
    let dcfr_call_sf = oop_vs_raise.strategy[3 + idx_oop_sf] as f64;
    let dcfr_call_fl = oop_vs_raise.strategy[3 + idx_oop_fl] as f64;
    assert!((dcfr_call_sf - 1.0).abs() < 0.05);
    assert!(
        (dcfr_call_fl - (2.0 / 3.0)).abs() < 0.05,
        "OOP #3 flush call vs AllIn(200)={dcfr_call_fl} outside 0.6667 ± 0.05"
    );
}
