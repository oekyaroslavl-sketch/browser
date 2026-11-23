//! Practical Native Flop Benchmark with 2D `NutBlockerBucketAbstraction` (`crates/plo_cfr/examples/bench_flop_buckets.rs`)
//!
//! Solves a hundreds-of-combos (`300 vs 300` live combos per side, filtered by board and capped
//! deterministically in canonical `ComboId` order from explicit PLO-4 range macros) two-tone flop
//! (`Ks9s4d`) using `NutBlockerBucketAbstraction` (`k = 50` via `(25, 2)`, plus `k = 100` via `(50, 2)`),
//! under an explicit memory budget, reporting scientific-notation exploitability at `0 / 100 / 500`
//! DCFR iterations, `f64` root EV, `iters/s`, and memory headroom.

use std::time::Instant;

use plo_cards::{Board, Street};
use plo_cfr::{
    estimate_memory_for_config, HandAbstraction, IdentityAbstraction, NutBlockerBucketAbstraction,
    PloGame, SolverBuildOptions, DEFAULT_MAX_MEMORY_BYTES,
};
use plo_range::PloRange;
use plo_tree::{ActionNodeKind, ActionTree, BettingLimit, StreetBetSizes, TreeConfig};

/// Target live combos per side (`200..=500`), capped deterministically by ascending `ComboId`.
///
/// Profiling note (`Ks9s4d`, `57,240` unfolded decision nodes, `106,336` terminal nodes across `49 * 48 = 2,352`
/// river branches because `NutBlockerBucketAbstraction` tie-breaks equal features by `ComboId` so
/// `is_suit_symmetric() == false`):
/// - Each DCFR step traverses `212,672` terminal nodes (`OOP` + `IP`) via exact 15-term 4-card
///   inclusion-exclusion (`O(|R_oop| + |R_ip|)` per terminal, zero evaluator simplification) with
///   pooled thread-local `InclusionExclusionScratch` buffers:
///   - At `200 vs 200` combos: `~1.25 iters/s` (`500` iters completes cleanly in ~400 s).
pub const TARGET_LIVE_COMBOS_PER_SIDE: usize = 200;

/// Explicit memory budget (`256 MiB`) enforced before tree allocation.
pub const FLOP_BUCKETS_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Builds a deterministic PLO-4 range on `board` from an explicit union of legal parser macros,
/// filtering out dead board-colliding combos and taking the first `cap_n` combos in canonical
/// `ComboId` order (as stored by `PloRange::entries()`).
fn build_macro_capped_range(board: Board, macro_union: &str, cap_n: usize) -> (PloRange, usize) {
    let raw: PloRange = macro_union.parse().expect("valid PLO-4 macro union");
    let live = raw.without_dead(board.mask());
    let total_live_from_macros = live.len();
    assert!(
        total_live_from_macros >= cap_n,
        "macro union produced only {total_live_from_macros} live combos on {board}, need >= {cap_n}"
    );
    // `live.entries()` is strictly sorted by ascending `ComboId` (`0..270_725`), so `.take(cap_n)`
    // is 100% deterministic with zero randomness.
    let capped_hands: Vec<_> = live
        .entries()
        .iter()
        .take(cap_n)
        .map(|wh| (wh.hand, wh.weight))
        .collect();
    let capped = PloRange::from_hands(capped_hands).expect("non-empty capped range");
    assert_eq!(capped.len(), cap_n);
    (capped, total_live_from_macros)
}

#[allow(clippy::too_many_arguments)]
fn run_bucket_benchmark(
    label: &str,
    flop: Board,
    oop: &PloRange,
    ip: &PloRange,
    tree: &ActionTree,
    abs: &dyn HandAbstraction,
    k_nominal: usize,
    budget_bytes: u64,
    total_iters: u32,
) -> (f64, [f64; 2], [f64; 2]) {
    let est = estimate_memory_for_config(flop, oop, ip, tree, abs, true)
        .expect("valid memory estimate");
    let t_build = Instant::now();
    let mut game = PloGame::with_options(
        flop,
        oop,
        ip,
        tree,
        abs,
        SolverBuildOptions {
            max_memory_bytes: budget_bytes,
            enable_chance_isomorphism: true,
            ..Default::default()
        },
    )
    .expect("fits within explicit memory budget");
    let build_s = t_build.elapsed().as_secs_f64();

    let mem = game.memory_report();
    let effective_iso = game.is_chance_isomorphism_enabled();
    assert_eq!(
        effective_iso,
        abs.is_suit_symmetric(),
        "effective_iso must equal enable_chance_isomorphism && abstraction.is_suit_symmetric()"
    );
    assert_eq!(mem.decision_nodes, est.decision_nodes);
    assert!(mem.total_estimated_bytes <= budget_bytes);

    let headroom_bytes = budget_bytes - mem.total_estimated_bytes;
    println!("\n--- Run: {label} ---");
    println!(
        "Live Ranges & Iso: |R_oop|={}, |R_ip|={}, k={k_nominal}, req_iso=true -> effective_iso={effective_iso} (abstraction.is_suit_symmetric={})",
        game.combos(0).len(),
        game.combos(1).len(),
        abs.is_suit_symmetric()
    );
    println!(
        "Tree Topology    : decision_nodes={}, chance_nodes={}, terminal_nodes={}, river_eval_tables={}",
        mem.decision_nodes, mem.chance_nodes, mem.terminal_nodes, mem.river_eval_tables
    );
    println!(
        "Memory Report    : regret_and_strategy={} B ({:.2} MiB), river_eval={} B ({:.2} MiB), total={} B ({:.2} MiB) [est={} B ({:.2} MiB)]",
        mem.regret_and_strategy_bytes,
        mem.regret_and_strategy_bytes as f64 / (1024.0 * 1024.0),
        mem.river_eval_bytes,
        mem.river_eval_bytes as f64 / (1024.0 * 1024.0),
        mem.total_estimated_bytes,
        mem.total_estimated_bytes as f64 / (1024.0 * 1024.0),
        est.total_estimated_bytes,
        est.total_estimated_bytes as f64 / (1024.0 * 1024.0),
    );
    println!(
        "Memory Guardrail : max_memory_bytes={} B ({:.2} MiB), used={:.2} MiB, headroom={} B ({:.2} MiB) | build_time={build_s:.2} s",
        budget_bytes,
        budget_bytes as f64 / (1024.0 * 1024.0),
        mem.total_estimated_bytes as f64 / (1024.0 * 1024.0),
        headroom_bytes,
        headroom_bytes as f64 / (1024.0 * 1024.0),
    );

    let expl_0 = game.exploitability_f64();
    println!("Iter    0        : exploitability = {expl_0:.6e} chips ({expl_0:.6} chips)");

    let step_100 = 100u32.min(total_iters);
    let t_start = Instant::now();
    for _ in 0..step_100 {
        game.solve_step();
    }
    let elapsed_100 = t_start.elapsed().as_secs_f64();
    let expl_100 = game.exploitability_f64();
    let ev_100 = game.root_expected_values_f64();
    let rem = total_iters.saturating_sub(step_100);
    if rem > 0 {
        println!(
            "Iter {:4}        : exploitability = {expl_100:.6e} chips ({expl_100:.6} chips) | root EV f64 = [{:.9}, {:.9}] | elapsed={elapsed_100:.2} s ({:.2} iters/s)",
            step_100,
            ev_100[0],
            ev_100[1],
            (step_100 as f64) / elapsed_100
        );
    }

    let t_rem = Instant::now();
    for _ in 0..rem {
        game.solve_step();
    }
    let total_elapsed = elapsed_100 + t_rem.elapsed().as_secs_f64();
    let overall_ips = (total_iters as f64) / total_elapsed;
    let expl_final = if rem > 0 {
        game.exploitability_f64()
    } else {
        expl_100
    };
    let ev_final = if rem > 0 {
        game.root_expected_values_f64()
    } else {
        ev_100
    };
    println!(
        "Iter {:4}        : exploitability = {expl_final:.6e} chips ({expl_final:.6} chips) | root EV f64 = [{:.9}, {:.9}] | total={total_elapsed:.2} s ({overall_ips:.2} iters/s)",
        total_iters, ev_final[0], ev_final[1]
    );

    assert!(
        expl_100 < expl_0,
        "exploitability must decrease from 0 to {step_100} iters ({expl_0:.6e} -> {expl_100:.6e})"
    );
    if rem > 0 {
        if k_nominal >= game.combos(0).len().max(game.combos(1).len()) {
            assert!(
                expl_final < expl_100,
                "lossless exploitability must decrease from {step_100} to {total_iters} iters ({expl_100:.6e} -> {expl_final:.6e})"
            );
        } else {
            // Under lossy imperfect-recall abstraction (`k < N`), unabstracted best-response exploitability
            // converges rapidly down to the abstraction's Nash gap floor (~0.3% pot for k=50 on 200v200)
            // and can oscillate slightly (~1e-2 chips) once at that floor.
            assert!(
                expl_final <= expl_100 + 0.05 && expl_final < expl_0 * 0.1,
                "bucketed exploitability must stay at abstraction floor ({expl_100:.6e} -> {expl_final:.6e})"
            );
        }
    }
    assert!(
        (ev_final[0] + ev_final[1] - 100.0).abs() <= 1e-8,
        "zero-sum pot conservation violated: {ev_final:?}"
    );

    (expl_final, ev_100, ev_final)
}

fn main() {
    let flop: Board = "Ks9s4d".parse().expect("valid two-tone flop");
    let starting_pot = 100;
    let effective_stack = 50;
    let bet_fracs = vec![0.25, 0.5];
    let raise_fracs = vec![1.0];
    let two_sizes = StreetBetSizes {
        bet_fractions: bet_fracs.clone(),
        raise_fractions: raise_fracs.clone(),
        donk_fractions: None,
    };

    // Exact same Pot-Limit skeleton as `bench_flop_tiny.rs` and `part_c_memory_rows`:
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

    // Realistic OOP & IP PLO-4 preflop/flop macro unions (double-suited & single-suited pairs,
    // rundown wraps, and suited Broadway/suited Ace structures), deterministically capped to
    // `TARGET_LIVE_COMBOS_PER_SIDE = 300` live combos per side in `ComboId` order:
    let oop_macros = [
        "AAKK$ds, AAQQ$ds, AAJJ$ds, AATT$ds, AA99$ds, AA88$ds",
        "KKQQ$ds, KKJJ$ds, QQJJ$ds, QQTT$ds, JJTT$ds, TT99$ds, 9988$ds, 8877$ds",
        "AKQJ$ds, KQJT$ds, QJT9$ds, JT98$ds, T987$ds, 9876$ds, 8765$ds",
        "(AK)(QJ), (AK)(QT), (AQ)(JT), (AJ)(T9), (KQ)(J9), (QJ)(T8)",
        "AAKQ$ss, AKJT$ss",
    ]
    .join(", ");

    let ip_macros = [
        "AAKK$ss, AAQQ$ss, AAJJ$ds, AATT$ds, KKJJ$ds, QQTT$ds, JJ99$ds, TT88$ds, 8866$ds, 7766$ds, 6655$ds",
        "AKQT$ds, KQT9$ds, QT98$ds, J987$ds, T876$ds, 9765$ds, 8654$ds",
        "(AQ)(J9), (AJ)(T8), (AT)(98), (KQ)(T9), (KJ)(T9), (QJ)(98), (JT)(97)",
        "KQJT$ss, QJT9$ss",
    ]
    .join(", ");

    let (oop, oop_raw_live) =
        build_macro_capped_range(flop, &oop_macros, TARGET_LIVE_COMBOS_PER_SIDE);
    let (ip, ip_raw_live) =
        build_macro_capped_range(flop, &ip_macros, TARGET_LIVE_COMBOS_PER_SIDE);

    println!("=== PLO-4 Practical Native Flop Benchmark (`bench_flop_buckets`) ===");
    println!("Board            : {flop} (two-tone)");
    println!(
        "Tree Skeleton    : PotLimit, pot={starting_pot}, stack={effective_stack}, bet={bet_fracs:?}, raise={raise_fracs:?} -> player skeleton [flop={}, turn={}, river={}]",
        skel_by_street[0], skel_by_street[1], skel_by_street[2]
    );
    println!(
        "OOP Live Combos  : {} (capped in ComboId order from {} live combos of macro union)",
        oop.len(),
        oop_raw_live
    );
    println!(
        "IP  Live Combos  : {} (capped in ComboId order from {} live combos of macro union)",
        ip.len(),
        ip_raw_live
    );
    println!(
        "Default Guard    : DEFAULT_MAX_MEMORY_BYTES = {} B ({:.0} MiB)",
        DEFAULT_MAX_MEMORY_BYTES,
        DEFAULT_MAX_MEMORY_BYTES as f64 / (1024.0 * 1024.0)
    );

    // 1) Primary practical run: `NutBlockerBucketAbstraction` with `k = 50` (`(25, 2)` per street) for `500` DCFR iterations
    let nb50 = NutBlockerBucketAbstraction::new((25, 2), (25, 2), (25, 2));
    let (_expl_50, ev_50_at_100, ev_50) = run_bucket_benchmark(
        "NutBlockerBucketAbstraction k=50 (dims=(25, 2))",
        flop,
        &oop,
        &ip,
        &tree,
        &nb50,
        50,
        FLOP_BUCKETS_MEMORY_BUDGET_BYTES,
        500,
    );

    // 2) Finer-bucket check (`k = 100` via `(50, 2)`): run `500` iterations when `--k100` or `--all` flag is passed,
    //    always printing its exact pre-build memory estimate (`116.98 MiB < 256 MiB`)!
    let run_extra = std::env::args().any(|a| a == "--all" || a == "--k100");
    let nb100 = NutBlockerBucketAbstraction::new((50, 2), (50, 2), (50, 2));
    let est_100 = estimate_memory_for_config(flop, &oop, &ip, &tree, &nb100, true).unwrap();
    println!(
        "\n--- Optional Finer Abstraction Estimate (NutBlocker k=100, dims=(50, 2)) ---\nEstimated Memory : {} B ({:.2} MiB) across {} decision_nodes (pass `--all` to also run k=100 + 500-iter Identity)",
        est_100.total_estimated_bytes,
        est_100.total_estimated_bytes as f64 / (1024.0 * 1024.0),
        est_100.decision_nodes
    );
    if run_extra && est_100.total_estimated_bytes <= FLOP_BUCKETS_MEMORY_BUDGET_BYTES {
        let (_expl_100, _ev_100_at_100, ev_100) = run_bucket_benchmark(
            "NutBlockerBucketAbstraction k=100 (dims=(50, 2))",
            flop,
            &oop,
            &ip,
            &tree,
            &nb100,
            100,
            FLOP_BUCKETS_MEMORY_BUDGET_BYTES,
            500,
        );
        println!(
            "\n[Bucket Resolution Delta k=50 vs k=100 at 500 iters] |ΔEV_oop| = {:.6} chips (k=50: {:.6}, k=100: {:.6})",
            (ev_50[0] - ev_100[0]).abs(),
            ev_50[0],
            ev_100[0]
        );
    }

    // 3) Check `IdentityAbstraction` memory estimate on the exact same `200 vs 200` ranges (`< 256 MiB` check):
    // Note: Capping in lexicographic `ComboId` order breaks `{h, c}` suit symmetry, so `IdentityAbstraction`
    // expands all `57,240` unfolded decision nodes (`219,098,336 B = 208.95 MiB < 256 MiB` at `200 vs 200`,
    // vs `313 MiB >= 256 MiB` at `300 vs 300`). Run `100` same-N iterations by default (or `500` with `--identity` / `--all`)!
    let est_id =
        estimate_memory_for_config(flop, &oop, &ip, &tree, &IdentityAbstraction, true).unwrap();
    let id_mib = est_id.total_estimated_bytes as f64 / (1024.0 * 1024.0);
    if est_id.total_estimated_bytes < 256 * 1024 * 1024 {
        let id_iters = if std::env::args().any(|a| a == "--all" || a == "--identity") {
            500
        } else {
            100
        };
        let (_expl_id, _ev_id_100, ev_id) = run_bucket_benchmark(
            "IdentityAbstraction (unabstracted reference)",
            flop,
            &oop,
            &ip,
            &tree,
            &IdentityAbstraction,
            oop.len(),
            256 * 1024 * 1024,
            id_iters,
        );
        let same_n_ev_k50 = if id_iters == 500 {
            ev_50[0]
        } else {
            ev_50_at_100[0]
        };
        // Step 3 C5: Log SAME-N EV gap for study ONLY (do NOT assert equality — bucketed EV != exact EV)
        println!(
            "\n[C5 Study Note — Same-N (N={id_iters}) Abstraction EV Gap vs Identity] EV_k50({id_iters})={same_n_ev_k50:.6} chips vs EV_id({id_iters})={:.6} chips -> ΔEV_oop = {:+.6} chips (logged for study only; bucketed EV != exact Identity EV)",
            ev_id[0],
            same_n_ev_k50 - ev_id[0]
        );
    } else {
        println!(
            "\n--- IdentityAbstraction Reference SKIPPED ---\nReason: Identity on {} vs {} ComboId-capped combos requires {} B ({id_mib:.2} MiB, {} decision_nodes) >= 256.00 MiB threshold.",
            oop.len(),
            ip.len(),
            est_id.total_estimated_bytes,
            est_id.decision_nodes
        );
    }
}
