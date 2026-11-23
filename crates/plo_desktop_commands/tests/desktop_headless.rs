//! Headless Desktop Integration Suite (`Phase 1` & `Phase 2`):
//! 1. Fake Nandez `.xlsx` + `.txt` pack loading, canonical spot mapping, manual spot binding persistence,
//!    and exact board + Hero 4-card dead-card blocker removal.
//! 2. Strict 4-card PLO Hero validation (Hold'em 2-card `AKs`, `AA`, `AsKs` rejected).
//! 3. Polar river Hero (`KdQd4c3c` IP bluff-catcher) vs Villain (`AsTs4h3h, Ah9s6d5d` OOP polar range)
//!    on `KsQsJs7d2c` (`1000` iterations -> EV within `25.0 / 75.0 ± 0.05`, Hero call freq `~0.50`).
//! 4. Instant `compute_equity` matching `plo_range::equity_range_vs_range` within `1e-12` without building a tree.
//! 5. `MemoryBudgetExceeded` refusal on Flop `Identity` `500 vs 500` with `128 MiB` budget (zero crash/OOM),
//!    plus `Auto` flop fallback to `NutBlocker k=50`.
//! 6. `AtomicBool` cooperative `stop_solve` halting worker-thread iteration cleanly.
//! 7. Optional `#[ignore]` local real-pack smoke check when `.local/nandez_pack` exists on disk.

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use plo_cards::Board;
use plo_desktop_commands::{
    canonical_default_spot_for_seat, compute_default_desktop_budget_mib, detect_system_ram_mib,
    dispatch_command, parse_hero_hand, parse_memory_budget_input, DesktopState, Json, SpotConfig,
    CANONICAL_SPOT_RULES, FALLBACK_DESKTOP_BUDGET_MIB, MANDATORY_FOOTER_TEXT,
};
use plo_range::{equity_range_vs_range, PloRange};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn temp_settings_file(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("plo_desktop_test_{tag}_{nanos}.json"))
}

#[test]
fn t1_fake_pack_importer_spot_bindings_and_board_hero_blockers() {
    let settings_path = temp_settings_file("t1");
    let state = DesktopState::with_settings_path(settings_path.clone());

    let fake_dir = workspace_root()
        .join("tests")
        .join("fixtures")
        .join("fake_nandez");
    let spots = state.set_pack_folder(&fake_dir).expect("load fake_nandez pack");

    // Verify all 15 canonical slots are present in exact 5-group structure + 6th "Other / unmapped files"
    // (`available == true` for every spot defined in Fake_Preflop_Chart.xlsx, `available == false` for missing ones)
    let available_ids: Vec<&str> = spots
        .iter()
        .filter(|s| s.available)
        .map(|s| s.spot_id.as_str())
        .collect();
    assert!(available_ids.contains(&"RFI_BTN"), "expected RFI_BTN in {available_ids:?}");
    assert!(
        available_ids.contains(&"BB_DEF_VS_1_RAISER"),
        "expected BB_DEF_VS_1_RAISER in {available_ids:?}"
    );
    assert!(
        available_ids.contains(&"BB_DEF_MULTIWAY"),
        "expected BB_DEF_MULTIWAY in {available_ids:?}"
    );
    assert!(
        available_ids.contains(&"THREEBET_IP_VS_15_20"),
        "expected THREEBET_IP_VS_15_20 in {available_ids:?}"
    );
    assert!(
        available_ids.contains(&"THREEBET_IP_VS_25_35"),
        "expected THREEBET_IP_VS_25_35 in {available_ids:?}"
    );
    // Canonical spots absent from Fake_Preflop_Chart.xlsx (e.g. RFI_UTG, THREEBET_OOP_*, CC_IP_*)
    // MUST still appear in list_spots with `available == false` so no UI group is silently dropped!
    let missing_utg = spots.iter().find(|s| s.spot_id == "RFI_UTG").expect("RFI_UTG slot");
    assert!(!missing_utg.available, "RFI_UTG should be marked not-in-pack in fake fixture");
    assert_eq!(missing_utg.group, "RFI (open)");
    assert_eq!(missing_utg.combo_count_full, 0);

    // Unmapped .txt file appears in group "Other / unmapped files"
    let raw_txt_entry = spots
        .iter()
        .find(|s| s.source_key.ends_with("BTN_Coldcall_Synthetic.txt"))
        .expect("expected recursive .txt range source");
    assert_eq!(raw_txt_entry.group, "Other / unmapped files");
    assert!(raw_txt_entry.available);
    assert_eq!(raw_txt_entry.combo_count_full, 5);

    // Verify all 5 canonical groups + Other are present in `list_spots` JSON with `group`, `label`, `id`, `combo_count_full`
    let listed_initial = state.list_spots();
    let json_spots = listed_initial.get("spots").unwrap().as_arr().unwrap();
    let groups_seen: std::collections::BTreeSet<&str> = json_spots
        .iter()
        .map(|s| s.get("group").unwrap().as_str().unwrap())
        .collect();
    for expected_group in [
        "RFI (open)",
        "BB defense",
        "3-bet IP",
        "3-bet OOP",
        "Cold call IP",
        "Other / unmapped files",
    ] {
        assert!(
            groups_seen.contains(expected_group),
            "missing UI group {expected_group:?} in {groups_seen:?}"
        );
    }
    for s in json_spots {
        assert!(s.get("id").and_then(|v| v.as_str()).is_some());
        assert!(s.get("label").and_then(|v| v.as_str()).is_some());
        assert!(s.get("group").and_then(|v| v.as_str()).is_some());
        assert!(s.get("combo_count_full").and_then(|v| v.as_usize()).is_some());
    }

    // Verify changing spot changes Villain PloRange (different live counts on empty board):
    // RFI_BTN has 7 combos, BB_DEF_MULTIWAY has 4 combos, THREEBET_IP_VS_15_20 has 3 combos
    let (_, live_btn_empty, _) = state.villain_from_spot("RFI_BTN", "", "").unwrap();
    let (_, live_mw_empty, _) = state.villain_from_spot("BB_DEF_MULTIWAY", "", "").unwrap();
    let (_, live_3b_empty, _) = state
        .villain_from_spot("THREEBET_IP_VS_15_20", "", "")
        .unwrap();
    assert_eq!(live_btn_empty.len(), 7);
    assert_eq!(live_mw_empty.len(), 4);
    assert_eq!(live_3b_empty.len(), 3);
    assert_ne!(live_btn_empty.len(), live_mw_empty.len());
    assert_ne!(live_mw_empty.len(), live_3b_empty.len());

    // Verify settings (including last_spot_id) persisted to disk and reload cleanly
    let reloaded = DesktopState::with_settings_path(settings_path.clone());
    let listed = reloaded.list_spots();
    assert_eq!(listed.get("loaded").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(
        listed.get("last_spot_id").and_then(|v| v.as_str()),
        Some("THREEBET_IP_VS_15_20")
    );
    assert_eq!(
        listed.get("footer").and_then(|v| v.as_str()),
        Some(MANDATORY_FOOTER_TEXT)
    );

    // Check blocker removal on RFI_BTN (7 raw combos in Fake_Preflop_Chart.xlsx:
    // AsKsQhJh, AdKdQcJc, AhKhQdJd, AcKcQsJs, Ts9s8h7h, 9d8d7c6c, 6s5s4d3d)
    // Board: As 9d 2h (blocks AsKsQhJh and 9d8d7c6c -> 5 remaining)
    // Hero:  Ad Kd 5c 2c (blocks AdKdQcJc -> 4 remaining: AhKhQdJd, AcKcQsJs, Ts9s8h7h, 6s5s4d3d!)
    let (raw_r, live_r, preview) = reloaded
        .villain_from_spot("RFI_BTN", "As9d2h", "AdKd5c2c")
        .expect("villain_from_spot with blockers");
    assert_eq!(raw_r.len(), 7);
    assert_eq!(live_r.len(), 4);
    assert_eq!(
        preview.get("blocked_combos").and_then(|v| v.as_usize()),
        Some(3)
    );

    // Verify manual spot binding override (`bind_spot` persists and re-points spot ID)
    reloaded
        .bind_spot("CC_IP_BTN_VS_30", "BTN_Coldcall_Synthetic.txt")
        .expect("bind custom .txt to canonical spot");
    let reloaded2 = DesktopState::with_settings_path(settings_path.clone());
    let (raw_txt, live_txt, _) = reloaded2
        .villain_from_spot("CC_IP_BTN_VS_30", "As9d2h", "6s5s4d3d")
        .unwrap();
    assert_eq!(raw_txt.len(), 5);
    assert_eq!(live_txt.len(), 3); // AsKsQhJh and 9d8d7c6c blocked by board As9d2h

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t2_strict_hero_4card_validation_rejects_holdem() {
    assert!(parse_hero_hand("AsKs").is_err(), "2-card Hold'em suited must fail");
    assert!(parse_hero_hand("AKs").is_err(), "Hold'em AKs shorthand must fail");
    assert!(parse_hero_hand("AA").is_err(), "Hold'em AA shorthand must fail");
    assert!(parse_hero_hand("AAKK").is_err(), "4-rank macro without suits must fail");
    assert!(
        parse_hero_hand("AsAsKhQh").is_err(),
        "duplicate card in Hero hand must fail"
    );

    let (hand, range) = parse_hero_hand("As Ks Qh Jh").expect("valid 4-card PLO hand");
    assert_eq!(range.len(), 1);
    assert_eq!(range.weight_of(hand), 1.0);
}

#[test]
fn t3_polar_river_hero_vs_villain_1000_iters_matches_75_25_nash() {
    let settings_path = temp_settings_file("t3");
    let state = Arc::new(DesktopState::with_settings_path(settings_path.clone()));

    // Hero is IP holding `KdQd4c3c` (bluff-catcher) vs Villain OOP polar range `AsTs4h3h, Ah9s6d5d`
    // on river `KsQsJs7d2c`, Pot=100, Stack=100, 100% pot-only.
    let cfg = SpotConfig {
        board: "KsQsJs7d2c".to_string(),
        hero_hand: "KdQd4c3c".to_string(),
        hero_seat: "IP".to_string(),
        use_pack: false,
        villain_spot_id: String::new(),
        villain_override_range: "AsTs4h3h, Ah9s6d5d".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        bet_sizes: "100%".to_string(),
        raise_sizes: "100%".to_string(),
        abstraction_mode: "Auto".to_string(),
        max_iterations: 1000,
        memory_budget_mib: 256,
        ..SpotConfig::default()
    };

    let mut progress_events = Vec::new();
    let res = state
        .solve_blocking(&cfg, 250, |ev| progress_events.push(ev.clone()))
        .expect("polar river solve");

    assert!(progress_events.len() >= 4);
    let last = progress_events.last().unwrap();
    assert_eq!(last.status, "completed");
    assert_eq!(last.iterations, 1000);
    assert!(
        (last.ev_villain - 75.0).abs() <= 0.05,
        "Villain OOP EV={} expected ~75.0",
        last.ev_villain
    );
    assert!(
        (last.ev_hero - 25.0).abs() <= 0.05,
        "Hero IP EV={} expected ~25.0",
        last.ev_hero
    );

    // Inspect root node (Villain OOP acts: Check or AllIn(100)), then navigate to Hero IP facing AllIn(100)
    let root_node = res.get("node").unwrap();
    assert_eq!(root_node.get("node_type").and_then(|v| v.as_str()), Some("oop"));
    assert_eq!(
        root_node.get("hero_acts_here").and_then(|v| v.as_bool()),
        Some(false)
    );

    // Play Villain action 1 (`AllIn(100)`) -> Hero IP acts (`Fold` or `Call`) with ~50% call frequency!
    let after_bet = state.play_action(1).expect("play AllIn(100)");
    assert_eq!(
        after_bet.get("hero_acts_here").and_then(|v| v.as_bool()),
        Some(true)
    );
    let hero_strat = after_bet.get("hero_strategy").unwrap().as_arr().unwrap();
    assert_eq!(hero_strat.len(), 2); // Fold, Call
    let call_prob = hero_strat[1].get("probability").unwrap().as_f64().unwrap();
    assert!(
        (call_prob - 0.50).abs() <= 0.02,
        "Hero IP bluff-catcher call prob={call_prob} expected 0.50 ± 0.02"
    );

    // Check GTO+ breadcrumb history label ("OOP AllIn(100)") and rewind_to
    let hist = after_bet.get("history").unwrap().as_arr().unwrap();
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0].as_str().unwrap(), "OOP AllIn(100)");

    // Step forward to terminal via Call (index 1), then rewind_to(1) and rewind_to(0)
    let after_call = state.play_action(1).unwrap();
    assert_eq!(after_call.get("history").unwrap().as_arr().unwrap().len(), 2);
    assert_eq!(
        after_call.get("history").unwrap().as_arr().unwrap()[1]
            .as_str()
            .unwrap(),
        "IP Call"
    );

    let rewound_1 = state.rewind_to(1).unwrap();
    assert_eq!(rewound_1.get("history").unwrap().as_arr().unwrap().len(), 1);
    assert_eq!(rewound_1.get("node_type").unwrap().as_str(), Some("ip"));

    let back = state.rewind_to(0).unwrap();
    assert_eq!(back.get("history").unwrap().as_arr().unwrap().len(), 0);

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t4_instant_equity_matches_plo_range_within_1e12_without_tree() {
    let settings_path = temp_settings_file("t4");
    let state = DesktopState::with_settings_path(settings_path.clone());
    let fake_dir = workspace_root()
        .join("tests")
        .join("fixtures")
        .join("fake_nandez");
    state.set_pack_folder(&fake_dir).unwrap();

    // Turn spot: Hero OOP `AsTs4h3h` vs Fake Pack `RFI_BTN` on `KsQs7d2c`
    let cfg = SpotConfig {
        board: "KsQs7d2c".to_string(),
        hero_hand: "AsTs4h3h".to_string(),
        hero_seat: "OOP".to_string(),
        use_pack: true,
        villain_spot_id: "RFI_BTN".to_string(),
        villain_override_range: String::new(),
        ..SpotConfig::default()
    };

    let eq_json = state.compute_equity(&cfg).expect("compute_equity");
    // Verify no game tree was built in session!
    assert!(state.query_node().is_err());

    let board: Board = "KsQs7d2c".parse().unwrap();
    let (_, hero_r) = parse_hero_hand("AsTs4h3h").unwrap();
    let (_, live_v, _) = state
        .villain_from_spot("RFI_BTN", "KsQs7d2c", "AsTs4h3h")
        .unwrap();
    let direct = equity_range_vs_range(&hero_r, &live_v, &board).unwrap();

    let cmd_hero_eq = eq_json.get("equity_hero").unwrap().as_f64().unwrap();
    let cmd_vil_eq = eq_json.get("equity_villain").unwrap().as_f64().unwrap();
    assert!(
        (cmd_hero_eq - direct.equity_p0).abs() <= 1e-12,
        "cmd={cmd_hero_eq} vs direct={}",
        direct.equity_p0
    );
    assert!((cmd_vil_eq - direct.equity_p1).abs() <= 1e-12);

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t5_memory_budget_exceeded_on_identity_500v500_at_128_mib_and_auto_selects_k50() {
    let settings_path = temp_settings_file("t5");
    let state = Arc::new(DesktopState::with_settings_path(settings_path.clone()));

    let board: Board = "Ks9s4d".parse().unwrap();
    let full_macro: PloRange = "AAKK, AAQQ, AAKJ, AAKT, KKQQ, KQJT, QJT9, JT98, T987, 9876"
        .parse()
        .unwrap();
    let live = full_macro.without_dead(board.mask());
    assert!(live.len() >= 500, "expected >=500 live combos, got {}", live.len());
    let capped_500: String = live
        .entries()
        .iter()
        .take(500)
        .map(|e| e.hand.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let id_cfg = SpotConfig {
        board: "Ks9s4d".to_string(),
        abstraction_mode: "Identity".to_string(),
        starting_pot: 100,
        effective_stack: 50,
        bet_sizes: "25%, 50%".to_string(),
        raise_sizes: "100%".to_string(),
        memory_budget_mib: 128,
        lab_two_range_mode: true,
        lab_oop_range: capped_500.clone(),
        lab_ip_range: capped_500.clone(),
        ..SpotConfig::default()
    };

    let err = state
        .build_game(&id_cfg)
        .expect_err("Identity 500v500 on Flop MUST exceed 128 MiB budget cleanly");
    assert!(
        err.contains("MemoryBudgetExceeded") && err.contains("Hint:"),
        "expected MemoryBudgetExceeded + human hint, got: {err}"
    );

    // With Auto mode at 128 MiB, Auto MUST NOT choose Identity on 500v500; it selects NutBlocker k=50 (~80.5 MiB <= 128 MiB)!
    let mut auto_cfg = id_cfg.clone();
    auto_cfg.abstraction_mode = "Auto".to_string();
    let est = dispatch_command(&state, "estimate_spot", &auto_cfg.to_json()).unwrap();
    assert_eq!(
        est.get("abstraction").and_then(|v| v.as_str()),
        Some("NutBlocker k=50 (25x2)")
    );
    assert_eq!(est.get("fits_budget").and_then(|v| v.as_bool()), Some(true));

    // Now set desktop budget = 4096 MiB (4 GiB — NO 512 MiB WASM cap!):
    // Explicit Identity flop 500 vs 500 (~307–502 MiB < 4096 MiB) MUST pass estimate & build_game!
    let mut id_4096 = id_cfg.clone();
    id_4096.memory_budget_mib = 4096;
    let est_id_4096 = dispatch_command(&state, "estimate_spot", &id_4096.to_json()).unwrap();
    assert_eq!(
        est_id_4096.get("budget_mib").and_then(|v| v.as_u32()),
        Some(4096),
        "desktop must not clamp 4096 MiB back down to 512 MiB!"
    );
    assert_eq!(
        est_id_4096.get("fits_budget").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert!(
        est_id_4096
            .get("summary_text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .contains("Budget 4 GiB")
    );
    // Actually build the game at 4096 MiB to prove `PloGame::with_options` accepts it cleanly
    let built_4096 = state
        .build_game(&id_4096)
        .expect("Identity 500v500 (~502 MiB) must be allowed when desktop budget is 4096 MiB");
    assert_eq!(
        built_4096.get("fits_budget").and_then(|v| v.as_bool()),
        Some(true)
    );

    // Even when RAM budget is 16 GiB (16384 MiB), Auto mode MUST NOT blindly auto-select Identity on 500 vs 500!
    let mut auto_16g = id_cfg.clone();
    auto_16g.abstraction_mode = "Auto".to_string();
    auto_16g.memory_budget_mib = 16_384;
    let est_auto_16g = dispatch_command(&state, "estimate_spot", &auto_16g.to_json()).unwrap();
    assert_eq!(
        est_auto_16g.get("abstraction").and_then(|v| v.as_str()),
        Some("NutBlocker k=50 (25x2)"),
        "Auto must still choose NutBlocker k=50 on fat 500v500 flop even at 16 GiB budget"
    );

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t8_desktop_system_ram_budget_policy_and_settings_persistence() {
    // 1. Verify exact formula:
    //    default = min(50% of system RAM, 16 GiB), floor 4 GiB if RAM >= 8 GiB, else 50% RAM;
    //    fallback 8192 MiB if RAM unknown.
    let fallback_none = compute_default_desktop_budget_mib(None);
    assert_eq!(fallback_none, FALLBACK_DESKTOP_BUDGET_MIB);
    assert!(fallback_none >= 4096);
    assert_eq!(compute_default_desktop_budget_mib(Some(8_192)), 4_096); // 8 GiB -> 4 GiB
    assert_eq!(compute_default_desktop_budget_mib(Some(16_384)), 8_192); // 16 GiB -> 8 GiB
    assert_eq!(compute_default_desktop_budget_mib(Some(32_768)), 16_384); // 32 GiB -> 16 GiB
    assert_eq!(compute_default_desktop_budget_mib(Some(65_536)), 16_384); // 64 GiB -> capped at 16 GiB
    assert_eq!(compute_default_desktop_budget_mib(Some(4_096)), 2_048); // < 8 GiB -> 50% RAM (never 256/512!)

    // If current host has >= 8 GiB RAM (or RAM is unknown), command-layer default is >= 4096 MiB;
    // if running inside a small <8 GiB container, mock 16 GiB via `compute_default_desktop_budget_mib`
    // and verify live `detect_system_ram_mib()` matches `50%` of container RAM (> 512 MiB).
    let settings_path = temp_settings_file("t8_ram");
    let state = Arc::new(DesktopState::with_settings_path(settings_path.clone()));
    let eff = state.effective_memory_budget_mib();
    match detect_system_ram_mib() {
        Some(host_mib) if host_mib >= 8_192 => {
            assert!(
                eff >= 4_096,
                "expected desktop default >= 4096 MiB on {host_mib} MiB machine, got {eff}"
            );
        }
        None => {
            assert!(eff >= 4_096, "fallback desktop default must be >= 4096 MiB, got {eff}");
        }
        Some(small_container_mib) => {
            assert_eq!(eff, compute_default_desktop_budget_mib(Some(small_container_mib)));
            assert!(
                eff > 512,
                "even in small {small_container_mib} MiB sandbox, default {eff} must exceed old 512 MiB WASM cap"
            );
        }
    }

    // 2. Verify flexible MiB / GiB parsing & `set_memory_budget` persistence in settings.json (`2048 .. 65536`)
    assert_eq!(parse_memory_budget_input("2048").unwrap(), 2048);
    assert_eq!(parse_memory_budget_input("16 GiB").unwrap(), 16_384);
    assert_eq!(parse_memory_budget_input("64GB").unwrap(), 65_536);

    let set_res = dispatch_command(
        &state,
        "set_memory_budget",
        &Json::obj([("budget", Json::Str("24 GiB".to_string()))]),
    )
    .unwrap();
    assert_eq!(
        set_res.get("memory_budget_mib").and_then(|v| v.as_u32()),
        Some(24_576)
    );
    assert_eq!(
        set_res.get("memory_budget_human").and_then(|v| v.as_str()),
        Some("24 GiB")
    );

    let reloaded = DesktopState::with_settings_path(settings_path.clone());
    assert_eq!(reloaded.effective_memory_budget_mib(), 24_576);
    assert_eq!(
        reloaded
            .list_spots()
            .get("memory_budget_mib")
            .and_then(|v| v.as_u32()),
        Some(24_576)
    );

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t6_stop_solve_atomic_bool_cancels_worker_cleanly() {
    let settings_path = temp_settings_file("t6");
    let state = Arc::new(DesktopState::with_settings_path(settings_path.clone()));

    // Turn spot with enough work that 50,000 iterations would take several seconds unless stopped
    let cfg = SpotConfig {
        board: "Ks9s4d2c".to_string(),
        hero_hand: "AsTsQhJh".to_string(),
        hero_seat: "OOP".to_string(),
        use_pack: false,
        villain_override_range: "AAKK$ds, KKQQ$ds, QJT9$ds, JT98$ds".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        bet_sizes: "50%, 100%".to_string(),
        raise_sizes: "100%".to_string(),
        abstraction_mode: "Auto".to_string(),
        max_iterations: 50_000,
        memory_budget_mib: 256,
        ..SpotConfig::default()
    };

    let worker_state = Arc::clone(&state);
    let handle = thread::spawn(move || {
        worker_state.solve_blocking(&cfg, 5, |ev| {
            if ev.iterations >= 10 {
                worker_state.stop_solve();
            }
        })
    });

    // Also trigger from outside after a brief delay as backup
    thread::sleep(Duration::from_millis(50));
    state.stop_solve();

    let res = handle.join().expect("worker thread panicked").expect("solve returned error");
    let prog = res.get("progress").unwrap();
    assert_eq!(prog.get("status").and_then(|v| v.as_str()), Some("stopped"));
    let ran = prog.get("iterations").unwrap().as_u32().unwrap();
    assert!(ran < 50_000, "expected cooperative early stop < 50000, got {ran}");

    // Config save/load round-trip check
    let cfg_file = temp_settings_file("saved_spot");
    let orig_cfg = SpotConfig::default();
    state.save_spot_config_to_file(&orig_cfg, &cfg_file).unwrap();
    let loaded_cfg = state.load_spot_config_from_file(&cfg_file).unwrap();
    assert_eq!(orig_cfg, loaded_cfg);

    let _ = std::fs::remove_file(cfg_file);
    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t7_smart_default_spot_by_hero_seat_and_persistence() {
    let settings_path = temp_settings_file("t7_seat_default");
    let state = Arc::new(DesktopState::with_settings_path(settings_path.clone()));

    // Pure function contract:
    // Hero OOP (likely BB) -> Villain opened RFI_BTN
    // Hero IP  (likely BTN/CO) -> Villain defends BB_DEF_VS_1_RAISER
    assert_eq!(canonical_default_spot_for_seat("OOP"), "RFI_BTN");
    assert_eq!(canonical_default_spot_for_seat("IP"), "BB_DEF_VS_1_RAISER");

    let fake_dir = workspace_root()
        .join("tests")
        .join("fixtures")
        .join("fake_nandez");
    state.set_pack_folder(&fake_dir).unwrap();

    // Command-level check via `dispatch_command("default_spot_for_seat", ...)`
    let oop_def = dispatch_command(
        &state,
        "default_spot_for_seat",
        &Json::obj([("hero_seat", Json::Str("OOP".to_string()))]),
    )
    .unwrap();
    assert_eq!(
        oop_def.get("spot_id").and_then(|v| v.as_str()),
        Some("RFI_BTN")
    );

    let ip_def = dispatch_command(
        &state,
        "default_spot_for_seat",
        &Json::obj([("hero_seat", Json::Str("IP".to_string()))]),
    )
    .unwrap();
    assert_eq!(
        ip_def.get("spot_id").and_then(|v| v.as_str()),
        Some("BB_DEF_VS_1_RAISER")
    );

    // Also verify `list_spots` reports both defaults and `compute_equity` with empty `villain_spot_id`
    // uses the seat-appropriate smart default while still allowing explicit override to any spot!
    let listed = state.list_spots();
    assert_eq!(
        listed.get("default_spot_oop").and_then(|v| v.as_str()),
        Some("RFI_BTN")
    );
    assert_eq!(
        listed.get("default_spot_ip").and_then(|v| v.as_str()),
        Some("BB_DEF_VS_1_RAISER")
    );

    // Empty villain_spot_id + Hero OOP -> resolves RFI_BTN (7 raw combos in fake fixture)
    let eq_oop = state
        .compute_equity(&SpotConfig {
            board: "2s3d4h7c8s".to_string(),
            hero_hand: "AsKsQhJh".to_string(),
            hero_seat: "OOP".to_string(),
            use_pack: true,
            villain_spot_id: String::new(),
            ..SpotConfig::default()
        })
        .unwrap();
    assert_eq!(
        eq_oop.get("raw_villain_combos").and_then(|v| v.as_usize()),
        Some(7)
    );

    // Explicit user override to THREEBET_IP_VS_15_20 (3 raw combos) works from any seat and persists
    let eq_override = state
        .compute_equity(&SpotConfig {
            board: "2s3d4h7c8s".to_string(),
            hero_hand: "KdQdJcTc".to_string(),
            hero_seat: "IP".to_string(),
            use_pack: true,
            villain_spot_id: "THREEBET_IP_VS_15_20".to_string(),
            ..SpotConfig::default()
        })
        .unwrap();
    assert_eq!(
        eq_override
            .get("raw_villain_combos")
            .and_then(|v| v.as_usize()),
        Some(3)
    );

    let reloaded = DesktopState::with_settings_path(settings_path.clone());
    assert_eq!(
        reloaded
            .list_spots()
            .get("last_spot_id")
            .and_then(|v| v.as_str()),
        Some("THREEBET_IP_VS_15_20")
    );

    let _ = std::fs::remove_file(settings_path);
}

#[test]
#[ignore = "requires local gitignored .local/nandez_pack folder"]
fn local_real_nandez_pack_parses_if_present() {
    let real_dir = workspace_root().join(".local").join("nandez_pack");
    if !real_dir.exists() {
        return;
    }
    let settings_path = temp_settings_file("real_nandez");
    let state = DesktopState::with_settings_path(settings_path.clone());
    let spots = state
        .set_pack_folder(&real_dir)
        .expect("parse real JNandez Preflop Chart.xlsx");

    // Must expose all 15 canonical IDs documented in DESKTOP.md, all `available == true` with `combo_count_full > 0`
    assert_eq!(CANONICAL_SPOT_RULES.len(), 15);
    let expected_ids = [
        "RFI_UTG",
        "RFI_MP",
        "RFI_CO",
        "RFI_BTN",
        "BB_DEF_VS_1_RAISER",
        "BB_DEF_MULTIWAY",
        "THREEBET_IP_VS_15_20",
        "THREEBET_IP_VS_25_35",
        "THREEBET_OOP_BB_VS_30",
        "THREEBET_OOP_SB_VS_30",
        "THREEBET_OOP_BB_VS_60",
        "THREEBET_OOP_SB_VS_60",
        "CC_IP_MP_CO_VS_15",
        "CC_IP_BTN_VS_15",
        "CC_IP_BTN_VS_30",
    ];
    for id in expected_ids {
        let desc = spots
            .iter()
            .find(|s| s.spot_id == id)
            .unwrap_or_else(|| panic!("missing canonical spot {id} in real pack"));
        assert!(desc.available, "canonical spot {id} must be available in real pack");
        assert!(
            desc.combo_count_full > 0,
            "canonical spot {id} must have >0 combos on empty board"
        );
    }

    // Changing spot changes Villain PloRange on an empty board
    let (_, utg_empty, _) = state.villain_from_spot("RFI_UTG", "", "").unwrap();
    let (_, btn_empty, _) = state.villain_from_spot("RFI_BTN", "", "").unwrap();
    let (_, bb_empty, _) = state
        .villain_from_spot("BB_DEF_VS_1_RAISER", "", "")
        .unwrap();
    let (_, ip3b_empty, _) = state
        .villain_from_spot("THREEBET_IP_VS_15_20", "", "")
        .unwrap();
    assert!(utg_empty.len() < btn_empty.len());
    assert_ne!(btn_empty.len(), bb_empty.len());
    assert!(ip3b_empty.len() < utg_empty.len());

    let (_, live_utg, preview) = state
        .villain_from_spot("RFI_UTG", "KsQsJs7d2c", "AsTs4h3h")
        .expect("RFI_UTG with blockers");
    assert!(!live_utg.is_empty());
    assert!(live_utg.len() < utg_empty.len());
    assert_eq!(
        preview
            .get("sample_top_20")
            .unwrap()
            .as_arr()
            .unwrap()
            .len(),
        20
    );
    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t9_per_street_flop_turn_river_oop_ip_and_donk_bet_sizes_wire_through() {
    let settings_path = temp_settings_file("t9");
    let state = DesktopState::with_settings_path(settings_path.clone());

    // Turn spot (`KsQs7d2c`, pot=100, stack=300) with distinct Turn OOP (25%), Turn IP (60%),
    // River OOP Donk (30%), River OOP Check->IP Bet (80%).
    let cfg = SpotConfig {
        board: "KsQs7d2c".to_string(),
        hero_hand: "AsTs4h3h".to_string(),
        hero_seat: "OOP".to_string(),
        use_pack: false,
        villain_override_range: "AAKK$ds, KQJT$ds".to_string(),
        starting_pot: 100,
        effective_stack: 300,
        bet_sizes: "100%".to_string(),
        raise_sizes: "100%".to_string(),
        flop_oop_bets: "33%".to_string(),
        flop_ip_bets: "50%".to_string(),
        turn_oop_bets: "25%".to_string(),
        turn_oop_raises: "100%".to_string(),
        turn_ip_bets: "60%".to_string(),
        turn_ip_raises: "100%".to_string(),
        river_oop_bets: "75%".to_string(),
        river_oop_donks: "30%".to_string(),
        river_ip_bets: "80%".to_string(),
        abstraction_mode: "Identity".to_string(),
        max_iterations: 5,
        ..SpotConfig::default()
    };

    // JSON round-trip preserves all per-street fields
    let roundtrip = SpotConfig::from_json(&cfg.to_json()).expect("roundtrip SpotConfig");
    assert_eq!(roundtrip, cfg);

    let solved = state.solve_blocking(&cfg, 0, |_| {}).expect("solve turn");
    let root_node = solved.get("node").unwrap();
    let root_actions: Vec<String> = root_node
        .get("actions")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|a| a.get("label").unwrap().as_str().unwrap().to_string())
        .collect();
    // Turn root OOP should have Check + Bet(25) (25% of 100)
    assert_eq!(root_actions, vec!["Check", "Bet(25)"]);

    // OOP Check (0) -> Turn IP should have Check + Bet(60) (60% of 100)
    let turn_ip_node = state.play_action(0).unwrap();
    let turn_ip_actions: Vec<String> = turn_ip_node
        .get("actions")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|a| a.get("label").unwrap().as_str().unwrap().to_string())
        .collect();
    assert_eq!(turn_ip_actions, vec!["Check", "Bet(60)"]);

    // IP Bet(60) (index 1) -> OOP Call (index 1) -> Pot on river is 100 + 60 + 60 = 220;
    // Since OOP called IP's Turn bet, River OOP uses `river_oop_donks = "30%"` -> 30% of 220 = Bet(66)!
    let _ = state.play_action(1).unwrap(); // IP Bet(60)
    let chance_node = state.play_action(1).unwrap(); // OOP Call -> Chance node
    assert_eq!(chance_node.get("node_type").unwrap().as_str(), Some("chance"));
    let first_river_card = chance_node
        .get("legal_chance_cards")
        .unwrap()
        .as_arr()
        .unwrap()[0]
        .as_str()
        .unwrap()
        .to_string();
    let river_oop_donk_node = state.play_chance(&first_river_card).unwrap();
    let river_donk_actions: Vec<String> = river_oop_donk_node
        .get("actions")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|a| a.get("label").unwrap().as_str().unwrap().to_string())
        .collect();
    assert_eq!(river_donk_actions, vec!["Check", "Bet(66)"]);

    // Rewind to Turn IP (depth 1), check back (index 0), deal river -> OOP did NOT call turn,
    // so River OOP uses `river_oop_bets = "75%"` -> 75% of 100 = Bet(75), and IP uses `80%` -> Bet(80)!
    let _ = state.rewind_to(1).unwrap();
    let chance2 = state.play_action(0).unwrap(); // IP Check back
    let card2 = chance2
        .get("legal_chance_cards")
        .unwrap()
        .as_arr()
        .unwrap()[0]
        .as_str()
        .unwrap()
        .to_string();
    let river_oop_reg = state.play_chance(&card2).unwrap();
    let river_reg_actions: Vec<String> = river_oop_reg
        .get("actions")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|a| a.get("label").unwrap().as_str().unwrap().to_string())
        .collect();
    assert_eq!(river_reg_actions, vec!["Check", "Bet(75)"]);

    let river_ip_reg = state.play_action(0).unwrap(); // OOP Check
    let river_ip_actions: Vec<String> = river_ip_reg
        .get("actions")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|a| a.get("label").unwrap().as_str().unwrap().to_string())
        .collect();
    assert_eq!(river_ip_actions, vec!["Check", "Bet(80)"]);

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t10_solve_streets_modes_flop_only_turn_only_river_only_and_all() {
    let settings_path = temp_settings_file("t10");
    let state = DesktopState::with_settings_path(settings_path.clone());

    // 1) 5-card board `KsQsJs7d2c` with `FlopOnly`: automatically trims to 3-card flop `KsQsJs`
    //    and checks down Turn + River (decision_nodes vastly smaller than full Flop+Turn+River)!
    let base_flop_all = SpotConfig {
        board: "KsQsJs".to_string(),
        hero_hand: "AsTs4h3h".to_string(),
        hero_seat: "OOP".to_string(),
        use_pack: false,
        villain_override_range: "AAKK$ds, KQJT$ds".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        flop_oop_bets: "50%, 100%".to_string(),
        flop_ip_bets: "50%, 100%".to_string(),
        turn_oop_bets: "50%, 100%".to_string(),
        turn_ip_bets: "50%, 100%".to_string(),
        river_oop_bets: "50%, 100%".to_string(),
        river_ip_bets: "50%, 100%".to_string(),
        solve_streets: "All".to_string(),
        abstraction_mode: "Identity".to_string(),
        max_iterations: 5,
        ..SpotConfig::default()
    };
    let est_all = state.estimate_spot(&base_flop_all).expect("est All");
    let dec_all = est_all.get("decision_nodes").unwrap().as_f64().unwrap();

    let flop_only_from_5 = SpotConfig {
        board: "KsQsJs7d2c".to_string(),
        solve_streets: "FlopOnly".to_string(),
        ..base_flop_all.clone()
    };
    let est_flop_only = state
        .estimate_spot(&flop_only_from_5)
        .expect("est FlopOnly");
    let dec_flop_only = est_flop_only
        .get("decision_nodes")
        .unwrap()
        .as_f64()
        .unwrap();
    // With Turn & River locked to check-down, decision_nodes exist ONLY on the single Flop board (< 30)!
    assert!(
        dec_flop_only < 30.0 && dec_all > 10_000.0,
        "FlopOnly dec={dec_flop_only} vs All dec={dec_all}"
    );

    // Solve FlopOnly and verify that OOP Check -> IP Check -> Deal Turn -> Deal River goes straight to terminal!
    let solved_fo = state
        .solve_blocking(&flop_only_from_5, 0, |_| {})
        .expect("solve FlopOnly");
    assert_eq!(
        solved_fo
            .get("node")
            .unwrap()
            .get("board")
            .unwrap()
            .as_str(),
        Some("KsQsJs")
    );
    let _ = state.play_action(0).unwrap(); // Flop OOP Check
    let turn_chance = state.play_action(0).unwrap(); // Flop IP Check -> Turn Chance
    assert_eq!(
        turn_chance.get("node_type").unwrap().as_str(),
        Some("chance")
    );
    let t_card = turn_chance
        .get("legal_chance_cards")
        .unwrap()
        .as_arr()
        .unwrap()[0]
        .as_str()
        .unwrap()
        .to_string();
    let river_chance = state.play_chance(&t_card).unwrap();
    // No Turn betting! Directly at River chance node:
    assert_eq!(
        river_chance.get("node_type").unwrap().as_str(),
        Some("chance")
    );
    let r_card = river_chance
        .get("legal_chance_cards")
        .unwrap()
        .as_arr()
        .unwrap()[0]
        .as_str()
        .unwrap()
        .to_string();
    let showdown_node = state.play_chance(&r_card).unwrap();
    // No River betting! Directly at terminal showdown:
    assert_eq!(
        showdown_node.get("node_type").unwrap().as_str(),
        Some("terminal")
    );

    // 2) `TurnOnly` on 5-card board trims to 4-card Turn `KsQsJs7d` and checks down River;
    //    `TurnOnly` on 3-card board returns clear validation error.
    let turn_only_ok = SpotConfig {
        board: "KsQsJs7d2c".to_string(),
        solve_streets: "TurnOnly".to_string(),
        ..base_flop_all.clone()
    };
    let est_to = state.estimate_spot(&turn_only_ok).expect("est TurnOnly");
    assert_eq!(est_to.get("street").unwrap().as_str(), Some("Turn"));
    assert!(est_to.get("decision_nodes").unwrap().as_f64().unwrap() < 30.0);

    let turn_only_err = SpotConfig {
        board: "KsQsJs".to_string(),
        solve_streets: "TurnOnly".to_string(),
        ..base_flop_all.clone()
    };
    assert!(state.estimate_spot(&turn_only_err).is_err());

    // 3) `RiverOnly` requires 5 board cards
    let river_only_err = SpotConfig {
        board: "KsQsJs7d".to_string(),
        solve_streets: "RiverOnly".to_string(),
        ..base_flop_all.clone()
    };
    assert!(state.estimate_spot(&river_only_err).is_err());

    let river_only_ok = SpotConfig {
        board: "KsQsJs7d2c".to_string(),
        solve_streets: "RiverOnly".to_string(),
        ..base_flop_all
    };
    let est_ro = state.estimate_spot(&river_only_ok).expect("est RiverOnly");
    assert_eq!(est_ro.get("street").unwrap().as_str(), Some("River"));

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t11_plo_hand_classes_and_ppt_villain_filter() {
    let settings_path = temp_settings_file("t11_classes_filter");
    let state = DesktopState::with_settings_path(settings_path.clone());

    // Monotone/two-tone flop `KsQs7d` with diverse Villain hands (none overlapping Hero `Jd9h6c5c`):
    // - `AsTs2h3c` = Nut Flush Draw (`nut_fd` or `combo_draw` with Broadway gutshot!)
    // - `KcKh4h2c` = Set of Kings (`set_trips`)
    // - `AhAd8c4d` = Overpair Aces (`overpair`)
    // - `JcTc8d2d` = Straight Draw / Wrap
    let cfg = SpotConfig {
        board: "KsQs7d".to_string(),
        hero_hand: "Jd9h6c5c".to_string(),
        hero_seat: "IP".to_string(),
        use_pack: false,
        villain_override_range: "AsTs2h3c, KcKh4h2c, AhAd8c4d, JcTc8d2d".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        solve_streets: "FlopOnly".to_string(),
        max_iterations: 40,
        ..SpotConfig::default()
    };

    let res = state.solve_blocking(&cfg, 20, |_| {}).expect("solve flop");
    let node = res.get("node").unwrap();
    let classes = node.get("villain_hand_classes").unwrap().as_arr().unwrap();
    assert!(
        classes.len() >= 3,
        "expected multiple PLO-4 hand classes, got {}",
        classes.len()
    );

    let class_ids: Vec<&str> = classes
        .iter()
        .filter_map(|c| c.get("class_id").and_then(|v| v.as_str()))
        .collect();
    assert!(class_ids.contains(&"set_trips"));
    assert!(class_ids.contains(&"overpair"));

    // Filter by `AA`: only `AhAd8c9h` matches!
    let filtered_aa = state.set_villain_filter("AA").unwrap();
    assert_eq!(
        filtered_aa.get("filtered_villain_combos").unwrap().as_usize(),
        Some(1)
    );
    let rows_aa = filtered_aa
        .get("villain_detail_rows")
        .unwrap()
        .as_arr()
        .unwrap();
    assert_eq!(rows_aa.len(), 1);
    assert!(rows_aa[0]
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("Ah"));

    // Filter by `nfd`: `AsTs2h3c` holds `AsTs` on `KsQs7d` (nut flush draw)!
    let filtered_nfd = state.set_villain_filter("nfd").unwrap();
    assert_eq!(
        filtered_nfd
            .get("filtered_villain_combos")
            .unwrap()
            .as_usize(),
        Some(1)
    );

    // Clear filter restores all 4 combos
    let cleared = state.set_villain_filter("").unwrap();
    assert_eq!(
        cleared.get("filtered_villain_combos").unwrap().as_usize(),
        Some(4)
    );

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t12_node_locking_shifts_counter_exploitative_strategy_and_unlocks() {
    let settings_path = temp_settings_file("t12_node_lock");
    let state = DesktopState::with_settings_path(settings_path.clone());

    // Polar river: Villain OOP has Nuts (`AsTs4h3h`) + Air (`9c8c4d3d`) vs Hero IP Bluffcatcher (`KdQd5h6h`).
    // Pot=100, Stack=100, Bet=100%.
    // In Nash equilibrium, Villain checks ~50% / bets ~50%, and Hero calls 50% vs a pot bet.
    // If we LOCK Villain OOP at Root to Bet 100% with ENTIRE range (`[0.0, 1.0]`, i.e., 50% nuts + 50% air,
    // giving Hero 50% equity vs pot-sized bet needing 33.3%), Hero IP's counter-exploitative response MUST
    // shift to ~100% Call!
    let cfg = SpotConfig {
        board: "KsQsJs7d2c".to_string(),
        hero_hand: "KdQd5h6h".to_string(),
        hero_seat: "IP".to_string(),
        use_pack: false,
        villain_override_range: "AsTs4h3h, 9c8c4d3d".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        river_oop_bets: "100%".to_string(),
        river_oop_raises: "100%".to_string(),
        river_ip_bets: "100%".to_string(),
        river_ip_raises: "100%".to_string(),
        solve_streets: "RiverOnly".to_string(),
        max_iterations: 400,
        ..SpotConfig::default()
    };

    let _ = state.solve_blocking(&cfg, 200, |_| {}).unwrap();
    // Lock Root (OOP Villain: Action 0 = Check, Action 1 = AllIn(100)) to 0% Check / 100% Bet and re-solve 300 iters!
    let locked_root = state.lock_current_node(&[0.0, 100.0], 300).unwrap();
    assert_eq!(locked_root.get("is_locked").unwrap().as_bool(), Some(true));
    assert_eq!(
        locked_root.get("locked_node_count").unwrap().as_usize(),
        Some(1)
    );

    let v_agg = locked_root
        .get("villain_aggregate_strategy")
        .unwrap()
        .as_arr()
        .unwrap();
    assert!((v_agg[0].get("probability").unwrap().as_f64().unwrap() - 0.0).abs() < 1e-5);
    assert!((v_agg[1].get("probability").unwrap().as_f64().unwrap() - 1.0).abs() < 1e-5);

    // Step into Action 1 (Villain Bet 100) -> Hero IP faces [Fold, Call]
    let hero_node = state.play_action(1).unwrap();
    assert_eq!(
        hero_node.get("hero_acts_here").unwrap().as_bool(),
        Some(true)
    );
    let h_strat = hero_node.get("hero_strategy").unwrap().as_arr().unwrap();
    // Action 1 is Call -> must be > 95% against an overbluffing 50%-air pot bet!
    let call_prob = h_strat[1].get("probability").unwrap().as_f64().unwrap();
    assert!(
        call_prob > 0.95,
        "Hero IP must exploitatively pure-call (>95%) vs locked 100% range-bet, got {call_prob}"
    );
    // Also verify `#6` Drill Mode EV-loss grading fields on `hero_strategy`:
    // Fold EV = 0.0 (or 50 chips loss relative to Call EV = 50.0!), so Fold `ev_loss > 40.0` and Call `ev_loss == 0.0`!
    let fold_loss = h_strat[0].get("ev_loss").unwrap().as_f64().unwrap();
    let call_loss = h_strat[1].get("ev_loss").unwrap().as_f64().unwrap();
    assert!(fold_loss > 40.0, "expected large EV loss on Fold, got {fold_loss}");
    assert!(call_loss < 1e-4, "expected 0 EV loss on best action Call, got {call_loss}");

    // Unlock all nodes and re-solve 600 iters -> returns to ~50% Call equilibrium!
    let _ = state.reset_cursor().unwrap();
    let unlocked = state.unlock_node(true, 600).unwrap();
    assert_eq!(unlocked.get("is_locked").unwrap().as_bool(), Some(false));
    assert_eq!(
        unlocked.get("locked_node_count").unwrap().as_usize(),
        Some(0)
    );

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t13_chance_runout_4x13_heatmap_and_target_dev_autostop_and_rake() {
    let settings_path = temp_settings_file("t13_heatmap_rake");
    let state = DesktopState::with_settings_path(settings_path.clone());

    // 1) Turn spot -> step to River Chance node and verify `chance_runout_stats`
    // `KcKd7c2d` has Set of Kings AND `7c`: on Turn `KsQsJs7d` it already has Kings full? Wait: `KcKd` uses 2 hole cards -> `KcKd` + `Ks7d` + `7h` = Kings full of Sevens (`KsKcKd7d7h`)! And on Turn `KsQsJs7d`, `KcKd7c2d` has 10 boat/quad outs (25% equity -> still calls or let's check on River bet!). Even simpler: with `effective_stack: 50` (`pot = 100`, `call = 50` for `200` total pot = 25% pot odds!), `KcKd7c2d` has 10 outs (`1 Kh` + `3 Q` + `3 J` + `2 7` = `9/44 = 20.5%`), OR give Villain `KcKdQcJd` (`1 Kh` + `2 Q` + `2 J` + `3 7` + `3 T`? `KcKd` + `Ks7d7h` = Boat, `QcJd` + `QsJsQh` = Boat -> 10+ outs > 25% pot odds, so Villain pure-calls Turn All-In!).
    let turn_cfg = SpotConfig {
        // Rainbow Turn `KsQhJc7d` so `AsTs` doesn't already have a 5-card Royal Flush!
        board: "KsQhJc7d".to_string(),
        hero_hand: "AsTs4h3h".to_string(), // Nut Broadway straight currently (`AsTs` + `KsQhJc`)
        hero_seat: "IP".to_string(),
        use_pack: false,
        // `KK87` has Set of Kings (`1` Quad out + `2` Seven outs + `3` Queen outs + `3` Jack outs = `9 / 40 = 22.5%` equity!):
        // - On brick `2c`: `KK` + `KsQhJc` = Trip Kings -> loses to Hero's Broadway straight (`Hero EV = +115`)!
        // - On `Kd`/`Kc` (`Kh` if live): `KK` + `KsKxQh` = Quad Kings -> beats Hero's Broadway straight (`Hero EV = -15`)!
        // - On `7c`/`7s`: `KK` + `Ks7d7c` = Kings full of Sevens -> beats Hero's Broadway straight!
        villain_override_range: "KK87".to_string(),
        starting_pot: 100,
        effective_stack: 15,
        turn_oop_bets: "100%".to_string(),
        turn_ip_bets: "100%".to_string(),
        solve_streets: "TurnOnly".to_string(),
        target_dev_pct: 0.01,
        max_iterations: 2000,
        ..SpotConfig::default()
    };

    let res = state.solve_blocking(&turn_cfg, 1, |_| {}).unwrap();
    let iters_used = res
        .get("progress")
        .unwrap()
        .get("iterations")
        .unwrap()
        .as_u32()
        .unwrap();
    assert!(
        iters_used < 2000,
        "Target dEV% auto-stop should finish well before 2000 iterations, used {iters_used}"
    );

    // Step OOP Check (`play_action(0)`), then Hero IP Bet/AllIn (`play_action(1)`), then OOP Call (`play_action(1)`)
    // where BOTH players reach the River Chance node with positive reach!
    let r_root = res.get("node").unwrap();
    let n1 = state.play_action(0).unwrap(); // OOP Check
    let n2 = state.play_action(1).unwrap(); // Hero IP AllIn(15)
    // Facing Hero IP AllIn(15), OOP `KK87` has `[Fold, Call]` (`0 = Fold`, `1 = Call`) -> `play_action(1)` is Call!
    // Wait: what WERE the actions at `n2`? Let's find `"Call"` dynamically by name!
    let oop_facing_acts = n2.get("actions").unwrap().as_arr().unwrap();
    let call_idx = oop_facing_acts
        .iter()
        .position(|a| {
            a.get("label")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .starts_with("Call")
        })
        .unwrap_or_else(|| panic!("Expected Call action in OOP facing node, got {oop_facing_acts:?}"));
    let chance_node = state.play_action(call_idx).unwrap(); // OOP Call(15) -> River Chance Node!
    let _ = (&r_root, &n1, &n2);
    assert_eq!(
        chance_node.get("node_type").unwrap().as_str(),
        Some("chance")
    );
    let runout_stats = chance_node
        .get("chance_runout_stats")
        .unwrap()
        .as_arr()
        .unwrap();
    // 52 - 4 board cards = 48 legal deck cards
    assert_eq!(runout_stats.len(), 48);
    let ev_of = |card_code: &str| -> f64 {
        runout_stats
            .iter()
            .find(|r| r.get("card").and_then(|v| v.as_str()) == Some(card_code))
            .and_then(|r| r.get("hero_ev"))
            .and_then(|v| v.as_f64())
            .unwrap()
    };
    let ev_brick_2c = ev_of("2c");
    // `Kh` gives EVERY live `KdKc8x7y` combo Quad Kings (`KdKc` + `KsKhQs`) while Hero only has Broadway straight (`AsTs` + `KsQsJs`)!
    let ev_quads_kh = ev_of("Kh");
    assert!(
        ev_brick_2c > ev_quads_kh + 50.0,
        "Hero AsTs4h3h EV on brick 2c ({ev_brick_2c}) must exceed Kh where KdKc8x7y makes Quads ({ev_quads_kh})"
    );

    // 2) Showdown Rake % + Cap: verify `rake_pct = 5.0, rake_cap = 3.0` lowers `EV_OOP + EV_IP` below `starting_pot`
    // while `rake_pct = 0.0` preserves exact zero-sum `EV_OOP + EV_IP == starting_pot`!
    let unraked_cfg = SpotConfig {
        board: "KsQsJs7d2c".to_string(),
        hero_hand: "AsTs4h3h".to_string(),
        hero_seat: "IP".to_string(),
        use_pack: false,
        villain_override_range: "9c8c4d3d".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        rake_pct: 0.0,
        rake_cap: 0.0,
        solve_streets: "RiverOnly".to_string(),
        max_iterations: 50,
        ..SpotConfig::default()
    };
    let unraked = state.solve_blocking(&unraked_cfg, 25, |_| {}).unwrap();
    let un_node = unraked.get("node").unwrap();
    let sum_unraked = un_node.get("root_ev_hero").unwrap().as_f64().unwrap()
        + un_node.get("root_ev_villain").unwrap().as_f64().unwrap();
    assert!(
        (sum_unraked - 100.0).abs() < 1e-6,
        "unraked EV sum must equal 100.0, got {sum_unraked}"
    );

    let raked_cfg = SpotConfig {
        rake_pct: 5.0,
        rake_cap: 3.0,
        ..unraked_cfg
    };
    let _raked = state.solve_blocking(&raked_cfg, 25, |_| {}).unwrap();
    let _ = state.play_action(0).unwrap(); // OOP Check
    let rk_sd = state.play_action(0).unwrap(); // IP Check -> Showdown Terminal Node (`pot = 100`)!
    let sum_raked = rk_sd.get("node_ev_hero").unwrap().as_f64().unwrap()
        + rk_sd.get("node_ev_villain").unwrap().as_f64().unwrap();
    // On checkdown showdown of pot=100, 5% rake capped at 3 chips = 3.0 chips deducted from winner!
    assert!(
        (96.99..=97.01).contains(&sum_raked),
        "5% rake capped at 3 chips on 100 pot checkdown should leave 97.0 total EV, got {sum_raked}"
    );

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t14_exact_subgame_resolving_upgrades_bucketed_turn_to_identity_river_and_exits() {
    let settings_path = temp_settings_file("t14_subgame");
    let state = DesktopState::with_settings_path(settings_path.clone());

    // Solve a `TurnOnly` spot (`NutBlocker50`) where River checks down in the parent game,
    // then deal River `2c` and call `resolve_subgame` to solve the full River betting tree in `Identity`!
    let turn_cfg = SpotConfig {
        board: "KsQhJc7d".to_string(),
        hero_hand: "AsTs4h3h".to_string(),
        hero_seat: "IP".to_string(),
        use_pack: false,
        villain_override_range: "KK87,QQ98,9c8c6d5d".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        turn_oop_bets: "50%".to_string(),
        turn_ip_bets: "50%".to_string(),
        river_oop_bets: "75%".to_string(),
        river_ip_bets: "75%".to_string(),
        abstraction_mode: "NutBlocker50".to_string(),
        solve_streets: "All".to_string(),
        max_iterations: 40,
        ..SpotConfig::default()
    };

    let _ = state.solve_blocking(&turn_cfg, 20, |_| {}).unwrap();
    let _ = state.play_action(0).unwrap(); // OOP Check
    let _ = state.play_action(0).unwrap(); // IP Check -> River Chance
    let river_parent = state.play_chance("2c").unwrap();

    assert_eq!(
        river_parent.get("street").and_then(|v| v.as_str()),
        Some("River")
    );
    assert_eq!(
        river_parent.get("can_subgame_resolve").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        river_parent.get("in_subgame").and_then(|v| v.as_bool()),
        Some(false)
    );
    assert!(
        river_parent
            .get("abstraction")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .contains("NutBlocker"),
        "Parent game used NutBlocker50"
    );

    // Resolve exact Identity subgame on River `KsQhJc7d2c`!
    let sub_node = state.resolve_subgame(250, 0.1).unwrap();
    assert_eq!(
        sub_node.get("in_subgame").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        sub_node.get("abstraction").and_then(|v| v.as_str()),
        Some("Identity (exact combos)")
    );
    assert!(
        sub_node.get("exploitability").and_then(|v| v.as_f64()).unwrap() < 0.5,
        "Resolved River Identity subgame should achieve tight <0.5 chip exploitability"
    );

    // Return back to parent tree via `exit_subgame`
    let back_parent = state.exit_subgame().unwrap();
    assert_eq!(
        back_parent.get("in_subgame").and_then(|v| v.as_bool()),
        Some(false)
    );
    assert!(
        back_parent
            .get("abstraction")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .contains("NutBlocker")
    );

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t15_saved_spots_library_node_report_export_and_hand_history_parser() {
    let settings_path = temp_settings_file("t15_lib_hh");
    let state = DesktopState::with_settings_path(settings_path.clone());

    // 1) Parse PokerStars PLO-4 Hand History (#3)
    let ps_hh = "\
PokerStars Hand #249001234: Omaha Pot Limit ($1/$2 USD) - 2026/11/22 18:00:00 ET
Table 'PLO4-Arena' 6-max Seat #1 is the button
Seat 1: HeroPlayer ($200 in chips)
Seat 2: SmallB ($200 in chips)
Seat 3: BigB ($200 in chips)
SmallB: posts small blind $1
BigB: posts big blind $2
*** HOLE CARDS ***
Dealt to HeroPlayer [As Ks Qh Jh]
HeroPlayer: raises $5 to $7
SmallB: folds
BigB: calls $5
*** FLOP *** [Ts 9s 4d]
BigB: checks
HeroPlayer: bets $10
BigB: calls $10
*** TURN *** [Ts 9s 4d] [2c]
BigB: checks
";
    let parsed = state.parse_hand_history(ps_hh).unwrap();
    assert_eq!(parsed.get("room").and_then(|v| v.as_str()), Some("PokerStars"));
    assert_eq!(parsed.get("hero_hand").and_then(|v| v.as_str()), Some("AsKsQhJh"));
    assert_eq!(parsed.get("flop_board").and_then(|v| v.as_str()), Some("Ts9s4d"));
    assert_eq!(parsed.get("final_board").and_then(|v| v.as_str()), Some("Ts9s4d2c"));
    assert_eq!(parsed.get("hero_seat").and_then(|v| v.as_str()), Some("IP"));
    assert_eq!(
        parsed.get("suggested_spot_id").and_then(|v| v.as_str()),
        Some("BB_DEF_VS_1_RAISER")
    );

    // Reject 2-card Hold'em hand history!
    let nlhe_hh = ps_hh.replace("[As Ks Qh Jh]", "[As Ks]");
    assert!(state.parse_hand_history(&nlhe_hh).is_err());

    // 2) Solve a spot, bookmark it (#2), persist & reload, and export CSV/Markdown report
    let cfg = SpotConfig {
        board: "KsQhJc7d2c".to_string(),
        hero_hand: "AsTs4h3h".to_string(),
        hero_seat: "IP".to_string(),
        use_pack: false,
        villain_override_range: "KK87,9c8c5d4d".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        solve_streets: "RiverOnly".to_string(),
        max_iterations: 100,
        ..SpotConfig::default()
    };
    let _ = state.solve_blocking(&cfg, 50, |_| {}).unwrap();

    let bm_res = state
        .save_spot_bookmark("BTN vs BB River Test", "SRP Study", cfg.clone())
        .unwrap();
    let saved_arr = bm_res.get("saved_spots").unwrap().as_arr().unwrap();
    assert_eq!(saved_arr.len(), 1);
    let bm_id = saved_arr[0]
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap()
        .to_string();

    // Reload from disk to confirm persistence
    let reloaded_state = DesktopState::with_settings_path(settings_path.clone());
    assert_eq!(reloaded_state.list_saved_spots().as_arr().unwrap().len(), 1);

    // Export CSV + Markdown
    let rep = state.export_node_report().unwrap();
    let csv = rep.get("csv").and_then(|v| v.as_str()).unwrap();
    let md = rep.get("markdown").and_then(|v| v.as_str()).unwrap();
    assert!(csv.contains("Class,SharePct,Combos,Weight,EV"));
    assert!(md.contains("PLO-4 Spot Report"));

    // Delete bookmark
    let after_del = state.delete_saved_spot(&bm_id).unwrap();
    assert!(after_del.as_arr().unwrap().is_empty());

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t16_blocker_matrix_equity_quintiles_flop_subset_class_locking_and_minimap() {
    let settings_path = temp_settings_file("t16_phase2_rest");
    let state = DesktopState::with_settings_path(settings_path.clone());

    // 1) Solve River spot with Nuts (`Flush`) vs Air in Villain's range to test:
    //    - `#4` `blocker_impact` (52 cards) & `equity_buckets` (5 quintiles)
    //    - `#6` `lock_node_by_classes` (lock Villain `air` to pure Bet while keeping `flush` pure Bet or Check!)
    //    - `#7` `tree_minimap` & `jump_street_path`
    let cfg = SpotConfig {
        board: "KsQs7s4d2c".to_string(),
        hero_hand: "KdKc4h3h".to_string(), // Set of Kings bluffcatcher (no spade!)
        hero_seat: "IP".to_string(),
        use_pack: false,
        // Villain OOP has Nut Flush (`AsJs9d8d`) and pure Air (`9h8h6c5c`)
        villain_override_range: "AsJs9d8d,9h8h6c5c".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        river_oop_bets: "100%".to_string(),
        river_ip_bets: "100%".to_string(),
        solve_streets: "RiverOnly".to_string(),
        max_iterations: 400,
        ..SpotConfig::default()
    };
    let solved = state.solve_blocking(&cfg, 100, |_| {}).unwrap();
    let root = solved.get("node").unwrap();

    // `#4a` 52-card blocker impact matrix
    let blockers = root.get("blocker_impact").unwrap().as_arr().unwrap();
    assert_eq!(blockers.len(), 52);
    let as_entry = blockers
        .iter()
        .find(|b| b.get("card").and_then(|v| v.as_str()) == Some("As"))
        .unwrap();
    assert!(
        as_entry.get("blocked_value_pct").and_then(|v| v.as_f64()).unwrap() > 99.0,
        "As blocks 100% of Villain's Flush value combo AsJs9d8d"
    );

    // `#4b` 5-quintile equity/EV distribution
    let eq_bins = root.get("equity_buckets").unwrap().as_arr().unwrap();
    assert_eq!(eq_bins.len(), 5);

    // `#7` Street Tree Minimap & `jump_street_path`
    let mm = root.get("tree_minimap").unwrap().as_arr().unwrap();
    assert!(mm.len() >= 3, "River betting tree minimap should have >=3 nodes");
    // Jump directly to `[1]` (OOP Bet 100 -> Hero IP facing Bet)
    let jumped = state.jump_street_path(&[1]).unwrap();
    assert_eq!(jumped.get("hero_acts_here").and_then(|v| v.as_bool()), Some(true));

    // Rewind to root (OOP acts) and test `#6` Per-Hand-Class Node Locking:
    // Lock Villain's `flush` (`AsJs9d8d`) to 100% Check `[100, 0]` AND `air` (`9h8h6c5c`) to 100% Bet `[0, 100]`!
    // Then when Villain bets, Villain has 100% Air -> Hero IP `KdKc4h3h` MUST pure-Call (`> 99%`)!
    let _ = state.reset_cursor().unwrap();
    let mut class_locks = std::collections::BTreeMap::new();
    class_locks.insert("flush".to_string(), vec![100.0, 0.0]); // Flush checks 100%
    class_locks.insert("air".to_string(), vec![0.0, 100.0]);   // Air bluffs 100%
    let _ = state.lock_node_by_classes(&class_locks, 300).unwrap();

    let facing_bluff = state.jump_street_path(&[1]).unwrap(); // OOP Bet -> Hero IP acts
    let hero_strat = facing_bluff.get("hero_strategy").unwrap().as_arr().unwrap();
    let call_prob = hero_strat
        .iter()
        .find(|s| {
            s.get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .starts_with("Call")
        })
        .and_then(|s| s.get("probability"))
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!(
        call_prob > 0.98,
        "Against class-locked 100% Air bet / 0% Flush bet, Hero Set must pure-Call (>98%), got {call_prob}"
    );

    // 2) `#5` Multi-Flop Subset C-Bet Report
    let subset_cfg = SpotConfig {
        board: "AsKs7d".to_string(),
        hero_hand: "QdJdTc9c".to_string(),
        hero_seat: "IP".to_string(),
        use_pack: false,
        villain_override_range: "AA82,KK72".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        ..SpotConfig::default()
    };
    let subset_rep = state
        .run_flop_subset_report(
            &subset_cfg,
            &["AsKs7d".to_string(), "9h9d4s".to_string(), "QsJs2d".to_string()],
            40,
        )
        .unwrap();
    let sub_rows = subset_rep.get("rows").unwrap().as_arr().unwrap();
    assert_eq!(sub_rows.len(), 3);

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t17_phase3_full_hand_ev_audit_and_simplify_bet_sizes() {
    let settings_path = temp_settings_file("t17_phase3_audit_simplify");
    let state = std::sync::Arc::new(DesktopState::with_settings_path(settings_path.clone()));

    // `#1` Full Hand EV Audit across Flop -> Turn -> River
    let raw_hh = "\
PokerStars Hand #99001122: Omaha Pot Limit ($1/$2 USD) - 2026/11/22 20:00:00 ET
Table 'PLO4-Audit' 6-max Seat #1 is the button
Seat 1: Hero ($200 in chips)
Seat 2: SmallBlind ($200 in chips)
Seat 3: BigBlind ($200 in chips)
SmallBlind: posts small blind $1
BigBlind: posts big blind $2
*** HOLE CARDS ***
Dealt to Hero [As Ks Qh Jh]
Hero: raises $5 to $7
SmallBlind: folds
BigBlind: calls $5
*** FLOP *** [Ts 9s 4d]
BigBlind: checks
Hero: bets $10
BigBlind: calls $10
*** TURN *** [Ts 9s 4d] [2s]
BigBlind: checks
Hero: bets $25
BigBlind: calls $25
*** RIVER *** [Ts 9s 4d 2s] [7c]
BigBlind: bets $85
Hero: folds
";

    let audit = dispatch_command(
        &state,
        "audit_hand_history",
        &Json::obj([
            ("text", Json::Str(raw_hh.to_string())),
            ("villain_override_range", Json::Str("QdJd8c6c, 8h8d6h5h".to_string())),
            ("iterations_per_street", Json::Num(25.0)),
        ]),
    )
    .unwrap();

    let decisions = audit.get("decisions").unwrap().as_arr().unwrap();
    assert_eq!(
        decisions.len(),
        3,
        "Expected 3 audited Hero decisions (Flop, Turn, River), got {}",
        decisions.len()
    );
    // On River `Ts9s4d2s7c`, Hero holds the 1st Nut Flush (`AsKs` + `Ts9s2s`), so folding to Villain's river bet is a massive Blunder!
    let river_dec = &decisions[2];
    assert_eq!(river_dec.get("street").and_then(|v| v.as_str()), Some("River"));
    assert_eq!(
        river_dec.get("grade").and_then(|v| v.as_str()),
        Some("Blunder"),
        "Folding the Nut Flush on the river must be graded Blunder, got: {}",
        river_dec.to_json_string()
    );
    assert!(
        river_dec
            .get("ev_loss_chips")
            .and_then(|v| v.as_f64())
            .unwrap()
            > 50.0
    );

    // `#3` Auto-Simplify Strategy / Pure-Size Optimizer
    // Polarized vs bluff-catcher setup: Villain OOP holds `KdKc9h8h` (Set of Kings) vs Hero IP `AsJs4h3h` (Nut Broadway).
    // Wait: in Hero-one-combo vs Villain-range, if Hero holds ONLY the Nuts, Villain pure-folds to any bet unless Villain has some stronger combos!
    // Give Villain 1 stronger Full House (`KsKcQhQd`) + 3 weaker Sets (`JdJc9h8h, JdJh6c5c, JhJc6d5d`) so Villain calls!
    let simp_cfg = SpotConfig {
        board: "KsQsJh7d2c".to_string(),
        hero_hand: "AsTs4h3h".to_string(), // Broadway straight (beats Sets, loses to Boat)
        hero_seat: "IP".to_string(),
        use_pack: false,
        villain_override_range: "KdKcQdQc, JdJc9h8h, JdJh6c5c, JhJc6d5d".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        ..SpotConfig::default()
    };
    let simp = dispatch_command(
        &state,
        "compare_bet_simplifications",
        &Json::obj([
            ("config", simp_cfg.to_json()),
            (
                "candidate_sizes",
                Json::Arr(vec![
                    Json::Str("33%".to_string()),
                    Json::Str("75%".to_string()),
                    Json::Str("100%".to_string()),
                ]),
            ),
            ("iterations", Json::Num(120.0)),
        ]),
    )
    .unwrap();
    let rows = simp.get("rows").unwrap().as_arr().unwrap();
    assert_eq!(rows.len(), 4, "1 baseline + 3 candidate single sizes");
    let rec_sz = simp.get("recommended_size").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        ["33%", "75%", "100%"].contains(&rec_sz),
        "recommended_size must pick one of the evaluated pure sizes, got {rec_sz:?}"
    );
    for r in rows {
        let ret = r.get("ev_retention_pct").and_then(|v| v.as_f64()).unwrap();
        assert!(ret > 50.0 && ret <= 105.0, "valid EV retention %, got {ret}");
    }

    let _ = std::fs::remove_file(settings_path);
}

#[test]
fn t18_phase3_bot_playout_binary_snapshot_convergence_and_nut_outs() {
    let settings_path = temp_settings_file("t18_phase3_bot_snap_outs");
    let snap_path = temp_settings_file("t18_tree_snapshot").with_extension("plotree");
    let state = std::sync::Arc::new(DesktopState::with_settings_path(settings_path.clone()));

    // Turn spot where Hero `AsKsQhJh` (`Double-Suited ($ds)`) has both Nut Flush outs (`s`) and Nut Broadway outs (`T`) on `9s8sJc2d`!
    let turn_cfg = SpotConfig {
        board: "9s8sJc2d".to_string(),
        hero_hand: "AsKsQhJh".to_string(),
        hero_seat: "IP".to_string(),
        use_pack: false,
        villain_override_range: "QsTs7d6d, JdJh5c4c".to_string(),
        starting_pot: 100,
        effective_stack: 100,
        turn_oop_bets: "100%".to_string(),
        turn_ip_bets: "100%".to_string(),
        river_oop_bets: "100%".to_string(),
        river_ip_bets: "100%".to_string(),
        solve_streets: "All".to_string(),
        max_iterations: 150,
        ..SpotConfig::default()
    };

    let solved = state.solve_blocking(&turn_cfg, 30, |_| {}).unwrap();
    let root = solved.get("node").unwrap();

    // `#4` Live Convergence History (`convergence_history` array populated with monotonic iterations)
    let conv = root.get("convergence_history").unwrap().as_arr().unwrap();
    assert!(
        conv.len() >= 2,
        "Expected >=2 convergence checkpoints from solve_blocking, got {}",
        conv.len()
    );
    let last_conv_iter = conv.last().unwrap().get("iteration").unwrap().as_u32().unwrap();
    assert_eq!(last_conv_iter, 150);

    // `#6` Hero Suit & Nut-Out Matrix on Turn `9s8sJc2d` for `AsKsQhJh`
    let nut_outs = root.get("hero_nut_outs").unwrap();
    assert_eq!(
        nut_outs.get("suit_structure").and_then(|v| v.as_str()),
        Some("Double-Suited ($ds)")
    );
    let nut_cnt = nut_outs.get("nut_out_count").and_then(|v| v.as_f64()).unwrap();
    assert!(
        nut_cnt >= 7.0,
        "AsKsQhJh on 9s8sJc2d has live spade Nut Flush outs + Broadway outs, got {nut_cnt}"
    );

    // `#5` Binary Solved Tree Snapshot (`.plotree`) save & instant reload into a fresh `DesktopState`
    let orig_ev = root.get("root_ev_hero").unwrap().as_f64().unwrap();
    let snap_str = snap_path.to_string_lossy().to_string();
    let saved_snap = dispatch_command(
        &state,
        "save_tree_snapshot",
        &Json::obj([("path", Json::Str(snap_str.clone()))]),
    )
    .unwrap();
    assert!(saved_snap.get("bytes_written").unwrap().as_f64().unwrap() > 100.0);

    let fresh_settings = temp_settings_file("t18_fresh_reload");
    let fresh_state = std::sync::Arc::new(DesktopState::with_settings_path(fresh_settings.clone()));
    let loaded_snap = dispatch_command(
        &fresh_state,
        "load_tree_snapshot",
        &Json::obj([("path", Json::Str(snap_str))]),
    )
    .unwrap();
    let reloaded_node = loaded_snap.get("node").unwrap();
    let reloaded_ev = reloaded_node.get("root_ev_hero").unwrap().as_f64().unwrap();
    assert!(
        (orig_ev - reloaded_ev).abs() < 1e-4,
        "Reloaded .plotree snapshot must preserve exact solved Hero Root EV ({orig_ev} vs {reloaded_ev})"
    );
    assert_eq!(
        reloaded_node.get("iterations").unwrap().as_u32(),
        Some(150)
    );

    // `#2` Play vs Solver Bot full hand playout from Turn -> Chance River -> Showdown!
    let bot_start = dispatch_command(
        &fresh_state,
        "start_bot_hand",
        &Json::obj([("seed", Json::Num(42.0))]),
    )
    .unwrap();
    let mut cur_node = bot_start.get("node").unwrap().clone();
    assert_eq!(cur_node.get("bot_active").and_then(|v| v.as_bool()), Some(true));

    // Step Hero through any Hero decision nodes until Terminal is reached
    for step_idx in 0..8 {
        let nt = cur_node.get("node_type").and_then(|v| v.as_str()).unwrap_or("");
        if nt == "terminal" {
            break;
        }
        assert_eq!(
            cur_node.get("hero_acts_here").and_then(|v| v.as_bool()),
            Some(true),
            "Bot auto-stepper must stop only on Hero decision or Terminal (got {nt} at step {step_idx})"
        );
        // Always pick check/call (`0` or `1` if facing bet) so we reach showdown
        let acts = cur_node.get("actions").unwrap().as_arr().unwrap();
        let call_or_chk = acts
            .iter()
            .position(|a| {
                let l = a.get("label").and_then(|v| v.as_str()).unwrap_or("");
                l.starts_with("Check") || l.starts_with("Call")
            })
            .unwrap_or(0);
        let step_res = dispatch_command(
            &fresh_state,
            "bot_play_hero_action",
            &Json::obj([
                ("action_index", Json::Num(call_or_chk as f64)),
                ("seed", Json::Num((100 + step_idx) as f64)),
            ]),
        )
        .unwrap();
        cur_node = step_res.get("node").unwrap().clone();
    }

    assert_eq!(
        cur_node.get("node_type").and_then(|v| v.as_str()),
        Some("terminal"),
        "Expected Bot hand playout to reach Terminal"
    );
    let sd = cur_node.get("bot_showdown").unwrap();
    assert!(
        sd.get("bot_hand").and_then(|v| v.as_str()).is_some(),
        "Bot showdown must reveal Villain's sampled 4-card hand"
    );
    assert_eq!(sd.get("went_to_showdown").and_then(|v| v.as_bool()), Some(true));

    let stopped = dispatch_command(&fresh_state, "stop_bot_hand", &Json::obj([])).unwrap();
    assert_eq!(stopped.get("bot_active").and_then(|v| v.as_bool()), Some(false));

    let _ = std::fs::remove_file(settings_path);
    let _ = std::fs::remove_file(fresh_settings);
    let _ = std::fs::remove_file(snap_path);
}


