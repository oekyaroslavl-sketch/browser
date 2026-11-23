//! Native Flop Microbenchmark (`crates/plo_cfr/examples/bench_flop_tiny.rs`)
//!
//! Runs a realistic compact PLO-4 Flop solve with `FIXED_FLOP_TINY_ITERS = 100` (matching `js/example_flop_tiny.mjs`),
//! then continues to `1000` iterations and cross-checks `best_response_values_f64()` against `exploitability_f64()`.

use std::time::Instant;

use plo_cards::{Board, Street};
use plo_cfr::{
    estimate_memory_for_config, IdentityAbstraction, PloGame, SolverBuildOptions,
    DEFAULT_MAX_MEMORY_BYTES,
};
use plo_range::PloRange;
use plo_tree::{ActionNodeKind, ActionTree, BettingLimit, StreetBetSizes, TreeConfig};

pub const FIXED_FLOP_TINY_ITERS: u32 = 100;

fn main() {
    let flop: Board = "Ks9s4d".parse().expect("valid flop");
    let starting_pot = 100;
    let effective_stack = 50;
    let spr = (effective_stack as f64) / (starting_pot as f64);

    // Suit-symmetric PLO macros yielding exact `20` live combos each on `Ks9s4d`:
    let oop_str = "AAQQ$ds, JJTT$ds, 8877$ds, AsJsQhTh, AsJsQcTc";
    let ip_str = "AAJJ$ds, QQTT$ds, 7766$ds, AsQsJhTh, AsQsJcTc";

    let oop: PloRange = oop_str.parse().expect("valid OOP range");
    let ip: PloRange = ip_str.parse().expect("valid IP range");
    let oop_live = oop.without_dead(flop.mask());
    let ip_live = ip.without_dead(flop.mask());
    assert_eq!(oop_live.len(), 20, "expected 20 live OOP combos");
    assert_eq!(ip_live.len(), 20, "expected 20 live IP combos");

    let bet_fracs = vec![0.25, 0.5];
    let raise_fracs = vec![1.0];
    let two_sizes = StreetBetSizes {
        bet_fractions: bet_fracs.clone(),
        raise_fractions: raise_fracs.clone(),
        donk_fractions: None,
    };
    let tree = ActionTree::build(TreeConfig {
        initial_street: Street::Flop,
        starting_pot,
        effective_stack,
        betting_limit: BettingLimit::PotLimit,
        flop_sizes: [two_sizes.clone(), two_sizes.clone()],
        turn_sizes: [two_sizes.clone(), two_sizes.clone()],
        river_sizes: [two_sizes.clone(), two_sizes],
        ..Default::default()
    })
    .expect("valid Pot-Limit tree");

    // A4 Proof of Exact Action-Node Topology (`34,880` decision nodes under `iso = true`, `57,240` under `iso = false`):
    // Why do `(Pot=100, Stack=50, bet=[25%, 50%], raise=[100%])` and `(Pot=100, Stack=100, bet=[50%, 100%], raise=[100%])`
    // produce the EXACT SAME abstract `ActionTree` skeleton `(8 flop, 16 turn, 24 river)`?
    // Because in both trees the opening bettor has `1` non-allin bet (`25` into `50` stack vs `50` into `100` stack)
    // AND `1` all-in/2nd bet (`AllIn(50)` vs `Bet(100)=AllIn(100)`):
    // - Facing the smaller bet (`25` or `50`), a `100%` pot raise (`25 + 150 = 175 >= 50`, or `50 + 200 = 250 >= 100`) caps at `AllIn`,
    //   ending further raises immediately; if called instead (`pot = 150` with `25` behind, or `pot = 200` with `50` behind),
    //   on the next street `25% * 150 = 38 > 25` (resp. `50% * 200 = 100 > 50`), so BOTH bet fractions collapse to a single `AllIn`!
    // - Consequently, both configs have identical per-street player-node counts `(N_flop=8, N_turn=16, N_river=24)`.
    // On two-tone `Ks9s4d` with `{h, c}`-symmetric ranges (`36` representative turns, `1,429` representative river subtrees:
    //   `10` rainbow turn cards `{3d?}` etc. have `48` unfolded rivers, `26` two-tone/paired turn cards fold `{h, c}` to `35..36` rivers -> sum = `1,429`):
    //   `decision_nodes = 8*1 + 16*36 + 24*1429 = 8 + 576 + 34,296 = 34,880` (`iso=true`),
    //   and unfolded:   `8*1 + 16*49 + 24*2352 = 8 + 784 + 56,448 = 57,240` (`iso=false`)!
    let mut skel_by_street = [0usize; 3];
    for anode in tree.nodes() {
        if let ActionNodeKind::Player { street, .. } = anode.kind {
            match street {
                Street::Flop => skel_by_street[0] += 1,
                Street::Turn => skel_by_street[1] += 1,
                Street::River => skel_by_street[2] += 1,
            }
        }
    }
    assert_eq!(skel_by_street, [8, 16, 24]);

    let est_iso_on =
        estimate_memory_for_config(flop, &oop, &ip, &tree, &IdentityAbstraction, true).unwrap();
    let est_iso_off =
        estimate_memory_for_config(flop, &oop, &ip, &tree, &IdentityAbstraction, false).unwrap();

    let t_build = Instant::now();
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
    .expect("fits well within 512 MiB budget");
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    // A2 & A4: `mem` is read directly from `THIS` built `PloGame::memory_report()`!
    let mem = game.memory_report();
    assert_eq!(mem.decision_nodes, est_iso_on.decision_nodes);
    assert_eq!(mem.decision_nodes, 8 + 16 * 36 + 24 * 1429);
    assert!(mem.decision_nodes < est_iso_off.decision_nodes);

    println!("=== PLO-4 Native Flop Tiny Microbenchmark ===");
    println!("Board            : {flop} (two-tone)");
    println!(
        "Config           : pot={starting_pot}, stack={effective_stack}, SPR={spr:.2}, bet={bet_fracs:?}, raise={raise_fracs:?}, iso={}, abstraction=IdentityAbstraction",
        game.is_chance_isomorphism_enabled()
    );
    println!("OOP Range (20)   : {oop_str}");
    println!("IP  Range (20)   : {ip_str}");
    println!(
        "Skeleton Nodes   : flop={}, turn={}, river={} -> live game.memory_report().decision_nodes = {} (iso=true) vs {} (iso=false)",
        skel_by_street[0], skel_by_street[1], skel_by_street[2], mem.decision_nodes, est_iso_off.decision_nodes
    );
    println!(
        "Chance / Term    : {} chance nodes, {} terminal nodes, {} unique river tables",
        mem.chance_nodes, mem.terminal_nodes, mem.river_eval_tables
    );
    println!(
        "Memory Bytes     : regret_and_strategy_bytes={} B ({:.2} MiB), river_eval_bytes={} B ({:.2} MiB), total_estimated_bytes={} B ({:.2} MiB)",
        mem.regret_and_strategy_bytes,
        mem.regret_and_strategy_bytes as f64 / (1024.0 * 1024.0),
        mem.river_eval_bytes,
        mem.river_eval_bytes as f64 / (1024.0 * 1024.0),
        mem.total_estimated_bytes,
        mem.total_estimated_bytes as f64 / (1024.0 * 1024.0),
    );
    println!("Tree Build Time  : {build_ms:.2} ms");

    let expl_0 = game.exploitability_f64();
    println!("Initial (0 iters): exploitability = {expl_0:.8} chips ({expl_0:.6e})");

    // Run to FIXED_FLOP_TINY_ITERS = 100 iterations
    let t_100 = Instant::now();
    for _ in 0..FIXED_FLOP_TINY_ITERS {
        game.solve_step();
    }
    let secs_100 = t_100.elapsed().as_secs_f64();
    let iters_per_sec_100 = (FIXED_FLOP_TINY_ITERS as f64) / secs_100;
    let expl_100 = game.exploitability_f64();
    let ev_100 = game.root_expected_values_f64();
    println!(
        "After {:4} iters : {:.2} s ({:.2} iters/s) | exploitability = {:.8} chips ({:.6e}) | root EV = [{:.9}, {:.9}]",
        FIXED_FLOP_TINY_ITERS, secs_100, iters_per_sec_100, expl_100, expl_100, ev_100[0], ev_100[1]
    );

    // Run additional 900 iterations -> total 1000 iterations
    let rem_iters = 1000 - FIXED_FLOP_TINY_ITERS;
    let t_900 = Instant::now();
    for _ in 0..rem_iters {
        game.solve_step();
    }
    let secs_900 = t_900.elapsed().as_secs_f64();
    let total_secs = secs_100 + secs_900;
    let iters_per_sec_1000 = 1000.0 / total_secs;
    let expl_1000 = game.exploitability_f64();
    let ev_1000 = game.root_expected_values_f64();
    let br_1000 = game.best_response_values_f64();
    println!(
        "After 1000 iters : {:.2} s ({:.2} iters/s) | exploitability = {:.8} chips ({:.6e}) | root EV = [{:.9}, {:.9}] | BR = [{:.9}, {:.9}]",
        total_secs, iters_per_sec_1000, expl_1000, expl_1000, ev_1000[0], ev_1000[1], br_1000[0], br_1000[1]
    );

    assert!(expl_100 < expl_0, "exploitability must drop by 100 iters");
    assert!(
        expl_1000 < expl_100,
        "exploitability must drop further by 1000 iters ({expl_100:.6e} -> {expl_1000:.6e})"
    );
    assert!((ev_100[0] + ev_100[1] - 100.0).abs() <= 1e-9);
    assert!((ev_1000[0] + ev_1000[1] - 100.0).abs() <= 1e-9);
}
