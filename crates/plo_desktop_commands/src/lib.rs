//! Headless orchestration layer for **`PLO-4 Postflop — Hero vs Range`**.
//!
//! # Architectural Contract
//! - Contains **NO** poker hand evaluation, blocker combinatorics, betting-tree rules, or DCFR math;
//!   all domain logic delegates strictly to `plo_cards`, `plo_range`, `plo_tree`, and `plo_cfr`.
//! - Enforces the **Hero (1 exact 4-card combo) vs Villain (`PloRange`)** product model by default,
//!   rejecting 2-card Hold'em inputs (`AKs`, `AA`, `AsKs`), filtering Villain's range by both
//!   `board.mask() | hero.mask()` dead cards, enforcing `Auto` abstraction & memory guardrails
//!   (`256 MiB` default, `512 MiB` hard cap), and driving worker-thread DCFR via `AtomicBool` stop
//!   + progress callbacks.

pub mod classifier;
pub mod formats;
pub mod hh_parser;
pub mod importer;
pub mod settings;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use plo_cards::{Board, Card, CardMask, Hand4, Street};
use plo_cfr::{
    estimate_memory_for_config, native_solver_threads, HandAbstraction, IdentityAbstraction,
    MemoryReport, NutBlockerBucketAbstraction, PloGame, SolverBuildOptions, SolverError,
};
use plo_range::{equity_range_vs_range, EquitySummary, PloRange};
use plo_tree::{
    parse_fraction_list, ActionTree, BettingLimit, StreetBetSizes, TreeConfig, PLAYER_IP,
    PLAYER_OOP,
};

pub use classifier::{
    analyze_plo_hand, matches_hand_filter, HandBoardFeatures, PloHandClass, ALL_PLO_HAND_CLASSES,
};
pub use formats::Json;
pub use hh_parser::{parse_plo4_hand_history, ParsedHandHistory};
pub use importer::{LoadedRangePack, SpotDescriptor, SpotGroup, CANONICAL_SPOT_RULES};
pub use settings::{
    compute_default_desktop_budget_mib, default_desktop_budget_mib, detect_system_ram_mib,
    format_mib_human, parse_memory_budget_input, AppSettings, SavedSpotEntry, SpotConfig,
    FALLBACK_DESKTOP_BUDGET_MIB, MAX_DESKTOP_BUDGET_MIB, MIN_DESKTOP_BUDGET_MIB,
};

/// Detailed technical & legal note surfaced in the `?` footer popover.
pub const MANDATORY_FOOTER_TEXT: &str = "Hero one combo vs villain range. Not BTN-range vs BB-range. Buckets share strategy; showdown is exact 2+3. NutBlocker disables suit iso. 512 MiB was the WASM/browser cap — Desktop uses system RAM (always estimated before build; wide flop trees are still compute-bound by the 41×40 chance tree). Purchased range pack is not included in the app.";

/// Canonical opponent-range smart default for a given Hero seat:
/// - Hero `OOP` (e.g. BB) -> Villain opened on `RFI_BTN`
/// - Hero `IP` (e.g. BTN/CO) -> Villain defends `BB_DEF_VS_1_RAISER`
pub fn canonical_default_spot_for_seat(hero_seat: &str) -> &'static str {
    if hero_seat.trim().eq_ignore_ascii_case("IP") {
        "BB_DEF_VS_1_RAISER"
    } else {
        "RFI_BTN"
    }
}

/// Parses a strict 4-card PLO Hero hand (accepting `"AsKsQhJh"` or `"As Ks Qh Jh"`),
/// explicitly rejecting 2-card Hold'em hands (`"AsKs"`, `"AKs"`, `"AA"`) and range macros (`"AAKK"`).
pub fn parse_hero_hand(input: &str) -> Result<(Hand4, PloRange), String> {
    let cleaned: String = input
        .chars()
        .filter(|c| !c.is_ascii_whitespace() && *c != ',' && *c != '-')
        .collect();
    if cleaned.is_empty() {
        return Err("Hero hand is empty; enter exact 4 PLO hole cards (e.g. AsKsQhJh)".to_string());
    }
    if cleaned.len() == 2 || cleaned.len() == 3 || cleaned.len() == 4 {
        // E.g. "AA", "AKs", "AKo", "AsKs" (Hold'em 2-card) or 4-rank macro "AAKK" without suits!
        let has_suits = cleaned
            .as_bytes()
            .chunks(2)
            .all(|ch| ch.len() == 2 && Card::parse_rank_char(ch[0]).is_some() && Card::parse_suit_char(ch[1]).is_some());
        if cleaned.len() == 4 && has_suits {
            return Err(format!(
                "Hold'em 2-card hand {input:?} rejected: PLO-4 Hero requires exact 4 hole cards (8 chars, e.g. AsKsQhJh)"
            ));
        }
        return Err(format!(
            "Invalid PLO-4 Hero hand {input:?}: enter exact 4 suited hole cards (e.g. AsKsQhJh), not Hold'em shorthand or a rank macro"
        ));
    }
    if cleaned.len() != 8 {
        return Err(format!(
            "PLO-4 Hero hand requires exact 4 cards (8 characters like AsKsQhJh), got {} characters in {input:?}",
            cleaned.len()
        ));
    }
    let hand = cleaned
        .parse::<Hand4>()
        .map_err(|e| format!("Invalid 4-card PLO Hero hand {input:?}: {e}"))?;
    let singleton = PloRange::from_hands([(hand, 1.0)]).map_err(|e| e.to_string())?;
    Ok((hand, singleton))
}

/// Parses a postflop board (`3` Flop, `4` Turn, or `5` River cards; whitespace allowed).
pub fn parse_postflop_board(input: &str) -> Result<Board, String> {
    let cleaned: String = input
        .chars()
        .filter(|c| !c.is_ascii_whitespace() && *c != ',')
        .collect();
    cleaned
        .parse::<Board>()
        .map_err(|e| format!("Invalid postflop board {input:?}: {e}"))
}

/// Concrete abstraction choice resolved after applying `Auto` rules on `(street, live_oop, live_ip, budget)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedAbstraction {
    Identity,
    NutBlocker50,
    NutBlocker100,
}

impl ResolvedAbstraction {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Identity => "Identity (exact combos)",
            Self::NutBlocker50 => "NutBlocker k=50 (25x2)",
            Self::NutBlocker100 => "NutBlocker k=100 (50x2)",
        }
    }

    pub fn boxed(self) -> Box<dyn HandAbstraction> {
        match self {
            Self::Identity => Box::new(IdentityAbstraction),
            Self::NutBlocker50 => Box::new(NutBlockerBucketAbstraction::new((25, 2), (25, 2), (25, 2))),
            Self::NutBlocker100 => Box::new(NutBlockerBucketAbstraction::new((50, 2), (50, 2), (50, 2))),
        }
    }
}

/// Progress snapshot emitted during `solve_loop` (`"solve-progress"` event payload).
#[derive(Clone, Debug)]
pub struct SolveProgressEvent {
    pub iterations: u32,
    pub max_iterations: u32,
    pub iters_per_sec: f64,
    pub exploitability: f64,
    pub exploitability_sci: String,
    pub ev_oop: f64,
    pub ev_ip: f64,
    pub ev_hero: f64,
    pub ev_villain: f64,
    pub hero_ev_pct_pot: f64,
    pub status: String, // "running" | "completed" | "stopped"
}

impl SolveProgressEvent {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("iterations", Json::Num(self.iterations as f64)),
            ("max_iterations", Json::Num(self.max_iterations as f64)),
            ("iters_per_sec", Json::Num(self.iters_per_sec)),
            ("exploitability", Json::Num(self.exploitability)),
            ("exploitability_sci", Json::Str(self.exploitability_sci.clone())),
            ("ev_oop", Json::Num(self.ev_oop)),
            ("ev_ip", Json::Num(self.ev_ip)),
            ("ev_hero", Json::Num(self.ev_hero)),
            ("ev_villain", Json::Num(self.ev_villain)),
            ("hero_ev_pct_pot", Json::Num(self.hero_ev_pct_pot)),
            ("status", Json::Str(self.status.clone())),
        ])
    }
}

/// Resolved ranges, seat assignment, tree config, and Auto warnings ready for estimation or solving.
pub struct PreparedSpot {
    pub board: Board,
    pub hero_hand: Option<Hand4>,
    pub hero_player: usize, // `PLAYER_OOP` (0) or `PLAYER_IP` (1)
    pub oop_range: PloRange,
    pub ip_range: PloRange,
    pub raw_villain_combos: usize,
    pub live_villain_combos: usize,
    pub live_oop_combos: usize,
    pub live_ip_combos: usize,
    pub action_tree: ActionTree,
    pub chosen_abstraction: ResolvedAbstraction,
    pub recommended_iterations: u32,
    pub effective_iso: bool,
    pub mem_report: MemoryReport,
    pub budget_bytes: u64,
    pub solve_streets_label: String,
    pub warnings: Vec<String>,
}

impl PreparedSpot {
    pub fn estimate_to_json(&self) -> Json {
        let mib = self.mem_report.total_estimated_bytes as f64 / (1024.0 * 1024.0);
        let budget_mib = self.budget_bytes as f64 / (1024.0 * 1024.0);
        let fits = self.mem_report.total_estimated_bytes <= self.budget_bytes;
        let est_human = format_mib_human(mib);
        let budget_human = format_mib_human(budget_mib);
        let summary_text = format!("Est. {est_human} / Budget {budget_human}");
        Json::obj([
            ("street", Json::Str(format!("{:?}", self.board.street()))),
            ("solve_streets_label", Json::Str(self.solve_streets_label.clone())),
            ("hero_seat", Json::Str(if self.hero_player == PLAYER_OOP { "OOP" } else { "IP" }.to_string())),
            ("raw_villain_combos", Json::Num(self.raw_villain_combos as f64)),
            ("live_villain_combos", Json::Num(self.live_villain_combos as f64)),
            ("live_oop_combos", Json::Num(self.live_oop_combos as f64)),
            ("live_ip_combos", Json::Num(self.live_ip_combos as f64)),
            ("abstraction", Json::Str(self.chosen_abstraction.as_label().to_string())),
            ("effective_iso", Json::Bool(self.effective_iso)),
            ("recommended_iterations", Json::Num(self.recommended_iterations as f64)),
            ("decision_nodes", Json::Num(self.mem_report.decision_nodes as f64)),
            ("chance_nodes", Json::Num(self.mem_report.chance_nodes as f64)),
            ("terminal_nodes", Json::Num(self.mem_report.terminal_nodes as f64)),
            ("river_eval_tables", Json::Num(self.mem_report.river_eval_tables as f64)),
            ("regret_and_strategy_bytes", Json::Num(self.mem_report.regret_and_strategy_bytes as f64)),
            ("river_eval_bytes", Json::Num(self.mem_report.river_eval_bytes as f64)),
            ("total_estimated_bytes", Json::Num(self.mem_report.total_estimated_bytes as f64)),
            ("total_estimated_mib", Json::Num(mib)),
            ("budget_bytes", Json::Num(self.budget_bytes as f64)),
            ("budget_mib", Json::Num(budget_mib)),
            ("estimated_human", Json::Str(est_human)),
            ("budget_human", Json::Str(budget_human)),
            ("summary_text", Json::Str(summary_text)),
            ("fits_budget", Json::Bool(fits)),
            (
                "warnings",
                Json::Arr(self.warnings.iter().cloned().map(Json::Str).collect()),
            ),
        ])
    }
}

struct ActiveGameSession {
    game: PloGame,
    spot_config: SpotConfig,
    hero_hand: Option<Hand4>,
    hero_player: usize,
    abstraction: ResolvedAbstraction,
    warnings: Vec<String>,
    history_labels: Vec<String>,
    villain_filter: String,
    convergence_log: Vec<(u32, f64, f64)>, // `(iterations, exploitability, hero_root_ev)`
    bot_hand: Option<Hand4>,
    parent_session: Option<Box<ActiveGameSession>>,
}

struct SolvingGuard<'a>(&'a AtomicBool);

impl Drop for SolvingGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Thread-safe stateful command engine shared by headless tests, Tauri 2 IPC commands,
/// and the local preview server.
pub struct DesktopState {
    settings_path: PathBuf,
    settings: Mutex<AppSettings>,
    pack: Mutex<Option<LoadedRangePack>>,
    session: Mutex<Option<ActiveGameSession>>,
    stop_flag: Arc<AtomicBool>,
    solving_flag: Arc<AtomicBool>,
    latest_progress: Mutex<Option<SolveProgressEvent>>,
}

impl Default for DesktopState {
    fn default() -> Self {
        Self::with_settings_path(AppSettings::default_config_path())
    }
}

impl DesktopState {
    pub fn with_settings_path(settings_path: PathBuf) -> Self {
        let settings = AppSettings::load_from_path(&settings_path);
        let initial_pack = settings
            .pack_folder
            .as_ref()
            .and_then(|p| LoadedRangePack::load_from_dir(Path::new(p), &settings.spot_bindings).ok());

        Self {
            settings_path,
            settings: Mutex::new(settings),
            pack: Mutex::new(initial_pack),
            session: Mutex::new(None),
            stop_flag: Arc::new(AtomicBool::new(false)),
            solving_flag: Arc::new(AtomicBool::new(false)),
            latest_progress: Mutex::new(None),
        }
    }

    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop_flag)
    }

    pub fn is_solving(&self) -> bool {
        self.solving_flag.load(Ordering::SeqCst)
    }

    fn acquire_exclusive_op(&self, busy_msg: &str) -> Result<SolvingGuard<'_>, String> {
        self.solving_flag
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map(|_| SolvingGuard(&self.solving_flag))
            .map_err(|_| busy_msg.to_string())
    }

    // ========================================================================
    // 1. Pack & Settings Commands
    // ========================================================================

    /// Sets and loads the user's range pack folder, persisting `settings.json` in OS app-data.
    pub fn set_pack_folder(&self, path: impl AsRef<Path>) -> Result<Vec<SpotDescriptor>, String> {
        let folder = path.as_ref();
        let bindings = self.settings.lock().unwrap().spot_bindings.clone();
        let loaded = LoadedRangePack::load_from_dir(folder, &bindings)?;
        let spots = loaded.spots.clone();

        {
            let mut st = self.settings.lock().unwrap();
            st.pack_folder = Some(folder.to_string_lossy().into_owned());
            let _ = st.save_to_path(&self.settings_path);
        }
        *self.pack.lock().unwrap() = Some(loaded);
        Ok(spots)
    }

    /// Updates a custom `spot_id -> source_key` binding and persists `settings.json`.
    pub fn bind_spot(&self, spot_id: &str, source_key: &str) -> Result<Vec<SpotDescriptor>, String> {
        let mut st = self.settings.lock().unwrap();
        if source_key.trim().is_empty() {
            st.spot_bindings.remove(spot_id);
        } else {
            st.spot_bindings
                .insert(spot_id.trim().to_string(), source_key.trim().to_string());
        }
        let _ = st.save_to_path(&self.settings_path);
        let bindings = st.spot_bindings.clone();
        drop(st);

        let mut pack_guard = self.pack.lock().unwrap();
        if let Some(pack) = pack_guard.as_mut() {
            pack.rebuild_spots(&bindings);
            Ok(pack.spots.clone())
        } else {
            Ok(Vec::new())
        }
    }

    /// Returns the smart default `spot_id` for `hero_seat` against the currently loaded pack:
    /// - Hero `OOP` (likely BB) -> `"RFI_BTN"` (if available in pack)
    /// - Hero `IP` (likely BTN/CO) -> `"BB_DEF_VS_1_RAISER"` (if available in pack)
    ///
    /// Falls back to the first available spot in the loaded pack if the preferred canonical spot is absent.
    pub fn default_spot_for_seat(&self, hero_seat: &str) -> Option<String> {
        let preferred = canonical_default_spot_for_seat(hero_seat);
        let pack_guard = self.pack.lock().unwrap();
        let pack = pack_guard.as_ref()?;
        if pack.has_available_spot(preferred) {
            Some(preferred.to_string())
        } else {
            pack.first_available_spot_id()
        }
    }

    /// Persists the user's last selected `spot_id` in `settings.json`.
    pub fn select_spot(&self, spot_id: &str) -> Result<(), String> {
        let trimmed = spot_id.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        let mut st = self.settings.lock().unwrap();
        if st.last_spot_id.as_deref() != Some(trimmed) {
            st.last_spot_id = Some(trimmed.to_string());
            st.save_to_path(&self.settings_path)?;
        }
        Ok(())
    }

    /// Persists the user's memory budget (in MiB) in `settings.json`.
    pub fn set_memory_budget(&self, budget_mib: u32) -> Result<u32, String> {
        let clamped = budget_mib.clamp(MIN_DESKTOP_BUDGET_MIB, MAX_DESKTOP_BUDGET_MIB);
        let mut st = self.settings.lock().unwrap();
        if st.memory_budget_mib != Some(clamped) {
            st.memory_budget_mib = Some(clamped);
            st.save_to_path(&self.settings_path)?;
        }
        Ok(clamped)
    }

    /// Returns the currently configured (or system-RAM default) desktop memory budget in MiB.
    pub fn effective_memory_budget_mib(&self) -> u32 {
        self.settings.lock().unwrap().effective_budget_mib()
    }

    /// Lists all 15 canonical spots (marked `available: true/false` with `group`, `label`, `id`, `combo_count_full`)
    /// plus any extra unmapped files/columns in `"Other / unmapped files"`, along with seat smart defaults,
    /// persisted `last_spot_id`, and desktop system-RAM `memory_budget_mib`.
    pub fn list_spots(&self) -> Json {
        let st = self.settings.lock().unwrap().clone();
        let eff_budget_mib = st.effective_budget_mib();
        let sys_ram_mib = detect_system_ram_mib();
        let pack_guard = self.pack.lock().unwrap();
        let (loaded, spots, sources, def_oop, def_ip, persisted_valid) =
            if let Some(pack) = pack_guard.as_ref() {
                let d_oop = if pack.has_available_spot("RFI_BTN") {
                    Some("RFI_BTN".to_string())
                } else {
                    pack.first_available_spot_id()
                };
                let d_ip = if pack.has_available_spot("BB_DEF_VS_1_RAISER") {
                    Some("BB_DEF_VS_1_RAISER".to_string())
                } else {
                    pack.first_available_spot_id()
                };
                let last_ok = st
                    .last_spot_id
                    .as_deref()
                    .filter(|id| pack.has_available_spot(id))
                    .map(|s| s.to_string());
                (
                    true,
                    pack.spots.iter().map(|s| s.to_json()).collect(),
                    pack.available_source_keys()
                        .into_iter()
                        .map(Json::Str)
                        .collect(),
                    d_oop,
                    d_ip,
                    last_ok,
                )
            } else {
                (false, Vec::new(), Vec::new(), None, None, st.last_spot_id.clone())
            };
        Json::obj([
            ("loaded", Json::Bool(loaded)),
            (
                "pack_folder",
                st.pack_folder.map(Json::Str).unwrap_or(Json::Null),
            ),
            (
                "last_spot_id",
                persisted_valid.map(Json::Str).unwrap_or(Json::Null),
            ),
            (
                "default_spot_oop",
                def_oop.map(Json::Str).unwrap_or(Json::Null),
            ),
            (
                "default_spot_ip",
                def_ip.map(Json::Str).unwrap_or(Json::Null),
            ),
            ("memory_budget_mib", Json::Num(eff_budget_mib as f64)),
            (
                "memory_budget_human",
                Json::Str(format_mib_human(eff_budget_mib as f64)),
            ),
            (
                "system_ram_mib",
                sys_ram_mib
                    .map(|m| Json::Num(m as f64))
                    .unwrap_or(Json::Null),
            ),
            ("spots", Json::Arr(spots)),
            ("available_sources", Json::Arr(sources)),
            (
                "saved_spots",
                Json::Arr(st.saved_spots.iter().map(|s| s.to_json()).collect()),
            ),
            ("native_threads", Json::Num(native_solver_threads() as f64)),
            ("footer", Json::Str(MANDATORY_FOOTER_TEXT.to_string())),
        ])
    }

    // ========================================================================
    //Saved Spot Library (`#2`), Hand History Parser (`#3`), & Multi-Flop Subset Report (`#5`)
    // ========================================================================

    pub fn list_saved_spots(&self) -> Json {
        let st = self.settings.lock().unwrap();
        Json::Arr(st.saved_spots.iter().map(|s| s.to_json()).collect())
    }

    pub fn save_spot_bookmark(
        &self,
        name: &str,
        folder: &str,
        config: SpotConfig,
    ) -> Result<Json, String> {
        let clean_name = if name.trim().is_empty() {
            format!(
                "{} | {} vs {}",
                config.board.trim(),
                config.hero_hand.trim(),
                if config.use_pack {
                    config.villain_spot_id.trim()
                } else {
                    "Custom"
                }
            )
        } else {
            name.trim().to_string()
        };
        let clean_folder = if folder.trim().is_empty() {
            "General".to_string()
        } else {
            folder.trim().to_string()
        };

        // Grab summary EV / exploitability if an active game session is loaded
        let (summary_hero_ev, summary_expl) = {
            let guard = self.session.lock().unwrap();
            if let Some(sess) = guard.as_ref() {
                let evs = sess.game.root_expected_values_f64();
                (Some(evs[sess.hero_player]), Some(sess.game.exploitability_f64()))
            } else {
                (None, None)
            }
        };

        let now_id = format!(
            "spot_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        );
        let entry = SavedSpotEntry {
            id: now_id,
            name: clean_name,
            folder: clean_folder,
            created_at: "2026-11-22".to_string(),
            config,
            summary_hero_ev,
            summary_expl,
        };

        let mut st = self.settings.lock().unwrap();
        // Replace existing bookmark with identical name+folder or push to front
        if let Some(pos) = st
            .saved_spots
            .iter()
            .position(|x| x.name == entry.name && x.folder == entry.folder)
        {
            st.saved_spots[pos] = entry.clone();
        } else {
            st.saved_spots.insert(0, entry.clone());
        }
        st.save_to_path(&self.settings_path)?;
        let arr: Vec<Json> = st.saved_spots.iter().map(|s| s.to_json()).collect();
        Ok(Json::obj([
            ("saved", entry.to_json()),
            ("saved_spots", Json::Arr(arr)),
        ]))
    }

    pub fn delete_saved_spot(&self, spot_id: &str) -> Result<Json, String> {
        let mut st = self.settings.lock().unwrap();
        st.saved_spots.retain(|s| s.id != spot_id);
        st.save_to_path(&self.settings_path)?;
        Ok(Json::Arr(st.saved_spots.iter().map(|s| s.to_json()).collect()))
    }

    /// Exports the currently inspected node (`Hand Classes`, `Hero/Villain Actions`, `Combos`) as CSV + Markdown summary (`#2`).
    pub fn export_node_report(&self) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        let node_json = build_node_query_json(sess);
        let board = node_json.get("board").and_then(|v| v.as_str()).unwrap_or("");
        let street = node_json.get("street").and_then(|v| v.as_str()).unwrap_or("");
        let pot = node_json.get("pot").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let hero_hand = node_json
            .get("hero_hand")
            .and_then(|v| v.as_str())
            .unwrap_or("Range");
        let hero_seat = node_json
            .get("hero_seat")
            .and_then(|v| v.as_str())
            .unwrap_or("OOP");
        let node_ev_hero = node_json
            .get("node_ev_hero")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let expl = node_json
            .get("exploitability")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let acts: Vec<String> = node_json
            .get("actions")
            .and_then(|v| v.as_arr())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.get("label").and_then(|s| s.as_str()).map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Build CSV of Villain Hand Classes + Action Frequencies
        let mut csv = String::new();
        csv.push_str("Class,SharePct,Combos,Weight,EV");
        for a in &acts {
            csv.push(',');
            csv.push_str(&a.replace(',', ";"));
        }
        csv.push('\n');

        let mut md = format!(
            "### PLO-4 Spot Report — {street} `{board}` (Pot {pot:.0})\n- **Hero ({hero_seat})**: `{hero_hand}` | Node EV: **{node_ev_hero:.2}** | dEV: `{expl:.4}`\n"
        );
        if let Some(h_arr) = node_json.get("hero_strategy").and_then(|v| v.as_arr()) {
            if !h_arr.is_empty() {
                md.push_str("- **Hero Strategy**: ");
                let parts: Vec<String> = h_arr
                    .iter()
                    .map(|item| {
                        let a = item.get("action").and_then(|v| v.as_str()).unwrap_or("?");
                        let p = item.get("probability").and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0;
                        let ev = item.get("ev").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        format!("{a}: {p:.1}% (EV {ev:.2})")
                    })
                    .collect();
                md.push_str(&parts.join(" | "));
                md.push('\n');
            }
        }
        md.push_str("\n| Hand Class | Share % | Combos | EV |\n|---|---:|---:|---:|\n");

        if let Some(classes) = node_json.get("villain_hand_classes").and_then(|v| v.as_arr()) {
            for c in classes {
                let label = c.get("label").and_then(|v| v.as_str()).unwrap_or("");
                let share = c.get("share_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let cnt = c.get("combo_count").and_then(|v| v.as_f64()).unwrap_or(0.0) as usize;
                let w = c.get("weight").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let ev = c.get("ev").and_then(|v| v.as_f64()).unwrap_or(0.0);
                csv.push_str(&format!(
                    "\"{}\",{share:.2},{cnt},{w:.4},{ev:.3}",
                    label.replace('"', "\"\"")
                ));
                if let Some(probs) = c.get("action_probs").and_then(|v| v.as_arr()) {
                    for p in probs {
                        csv.push_str(&format!(",{:.4}", p.as_f64().unwrap_or(0.0)));
                    }
                }
                csv.push('\n');
                md.push_str(&format!("| {label} | {share:.1}% | {cnt} | {ev:.2} |\n"));
            }
        }

        Ok(Json::obj([
            ("csv", Json::Str(csv)),
            ("markdown", Json::Str(md)),
            ("node", node_json),
        ]))
    }

    /// Parses a PokerStars / GGPoker / CoinPoker PLO-4 hand history (`#3`) and returns structured JSON.
    pub fn parse_hand_history(&self, hh_text: &str) -> Result<Json, String> {
        let parsed = parse_plo4_hand_history(hh_text)?;
        Ok(parsed.to_json())
    }

    /// Resolves a spot from the loaded range pack, removes combos blocked by `board` and `hero_hand`
    /// (if provided), persists `last_spot_id` in `settings.json`, and returns `(raw_range, live_range)` + top-20 preview.
    pub fn villain_from_spot(
        &self,
        spot_id: &str,
        board_str: &str,
        hero_str: &str,
    ) -> Result<(PloRange, PloRange, Json), String> {
        let raw_range = {
            let pack_guard = self.pack.lock().unwrap();
            let pack = pack_guard.as_ref().ok_or_else(|| {
                "No range pack folder loaded. Open Settings to select your range pack folder, or uncheck 'Use pack' to type a villain range.".to_string()
            })?;
            pack.range_for_spot(spot_id)?.clone()
        };
        let _ = self.select_spot(spot_id);

        let mut dead = CardMask::EMPTY;
        if !board_str.trim().is_empty() {
            let b = parse_postflop_board(board_str)?;
            dead |= b.mask();
        }
        if !hero_str.trim().is_empty() {
            let (h, _) = parse_hero_hand(hero_str)?;
            if dead.intersects(h.mask()) {
                return Err("Hero hole cards overlap with board cards".to_string());
            }
            dead |= h.mask();
        }

        let live_range = raw_range.without_dead(dead);
        let preview = build_range_preview_json(spot_id, &raw_range, &live_range);
        Ok((raw_range, live_range, preview))
    }

    /// Previews either a pack spot or a manual override range with live board+hero blocker filtering.
    pub fn preview_villain_range(
        &self,
        use_pack: bool,
        spot_id: &str,
        override_str: &str,
        board_str: &str,
        hero_str: &str,
    ) -> Result<Json, String> {
        if use_pack {
            let (_, _, j) = self.villain_from_spot(spot_id, board_str, hero_str)?;
            Ok(j)
        } else {
            let raw_range = override_str
                .parse::<PloRange>()
                .map_err(|e| format!("Invalid Villain PLO range: {e}"))?;
            if raw_range.is_empty() {
                return Err("Villain range is empty".to_string());
            }
            let mut dead = CardMask::EMPTY;
            if !board_str.trim().is_empty() {
                dead |= parse_postflop_board(board_str)?.mask();
            }
            if !hero_str.trim().is_empty() {
                let (h, _) = parse_hero_hand(hero_str)?;
                if dead.intersects(h.mask()) {
                    return Err("Hero hole cards overlap with board cards".to_string());
                }
                dead |= h.mask();
            }
            let live_range = raw_range.without_dead(dead);
            Ok(build_range_preview_json("manual_override", &raw_range, &live_range))
        }
    }

    // ========================================================================
    // 2. Instant Equity (NO Tree Construction!)
    // ========================================================================

    /// Computes exact PLO-4 `equity_range_vs_range` (Hero singleton vs Villain range, or Lab 2-range)
    /// directly on `board` without building an `ActionTree` or `PloGame`.
    pub fn compute_equity(&self, cfg: &SpotConfig) -> Result<Json, String> {
        let full_board = parse_postflop_board(&cfg.board)?;
        let (board, _, _, _) = resolve_solve_streets(&full_board, &cfg.solve_streets)?;
        let (oop_range, ip_range, hero_player, raw_v, live_v) = self.resolve_ranges(cfg, &board)?;

        let t0 = Instant::now();
        let eq: EquitySummary = equity_range_vs_range(&oop_range, &ip_range, &board)
            .map_err(|e| format!("Equity error: {e}"))?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let (hero_eq, villain_eq, hero_win, villain_win) = if hero_player == PLAYER_OOP {
            (eq.equity_p0, eq.equity_p1, eq.win_p0, eq.win_p1)
        } else {
            (eq.equity_p1, eq.equity_p0, eq.win_p1, eq.win_p0)
        };

        Ok(Json::obj([
            ("hero_seat", Json::Str(if hero_player == PLAYER_OOP { "OOP" } else { "IP" }.to_string())),
            ("equity_hero", Json::Num(hero_eq)),
            ("equity_villain", Json::Num(villain_eq)),
            ("equity_oop", Json::Num(eq.equity_p0)),
            ("equity_ip", Json::Num(eq.equity_p1)),
            ("win_hero", Json::Num(hero_win)),
            ("win_villain", Json::Num(villain_win)),
            ("tie", Json::Num(eq.tie)),
            ("valid_matchup_weight", Json::Num(eq.valid_weight_sum)),
            ("raw_villain_combos", Json::Num(raw_v as f64)),
            ("live_villain_combos", Json::Num(live_v as f64)),
            ("elapsed_ms", Json::Num(elapsed_ms)),
        ]))
    }

    // ========================================================================
    // 3. Spot Preparation, Auto Abstraction Policy & Memory Estimation
    // ========================================================================

    pub fn prepare_spot(&self, cfg: &SpotConfig) -> Result<PreparedSpot, String> {
        let full_board = parse_postflop_board(&cfg.board)?;
        let (board, active_betting_streets, solve_streets_label, street_mode_note) =
            resolve_solve_streets(&full_board, &cfg.solve_streets)?;
        if cfg.starting_pot <= 0 {
            return Err("Starting pot must be > 0 chips".to_string());
        }
        if cfg.effective_stack <= 0 {
            return Err("Effective stack must be > 0 chips".to_string());
        }

        let (oop_range, ip_range, hero_player, raw_v, live_v) = self.resolve_ranges(cfg, &board)?;
        let hero_hand = if cfg.lab_two_range_mode {
            None
        } else {
            Some(parse_hero_hand(&cfg.hero_hand)?.0)
        };

        let flop_oop = build_player_street_sizes(
            "Flop OOP",
            &cfg.flop_oop_bets,
            &cfg.flop_oop_raises,
            None,
            &cfg.bet_sizes,
            &cfg.raise_sizes,
        )?;
        let flop_ip = build_player_street_sizes(
            "Flop IP",
            &cfg.flop_ip_bets,
            &cfg.flop_ip_raises,
            None,
            &cfg.bet_sizes,
            &cfg.raise_sizes,
        )?;
        let turn_oop = build_player_street_sizes(
            "Turn OOP",
            &cfg.turn_oop_bets,
            &cfg.turn_oop_raises,
            Some(&cfg.turn_oop_donks),
            &cfg.bet_sizes,
            &cfg.raise_sizes,
        )?;
        let turn_ip = build_player_street_sizes(
            "Turn IP",
            &cfg.turn_ip_bets,
            &cfg.turn_ip_raises,
            None,
            &cfg.bet_sizes,
            &cfg.raise_sizes,
        )?;
        let river_oop = build_player_street_sizes(
            "River OOP",
            &cfg.river_oop_bets,
            &cfg.river_oop_raises,
            Some(&cfg.river_oop_donks),
            &cfg.bet_sizes,
            &cfg.raise_sizes,
        )?;
        let river_ip = build_player_street_sizes(
            "River IP",
            &cfg.river_ip_bets,
            &cfg.river_ip_raises,
            None,
            &cfg.bet_sizes,
            &cfg.raise_sizes,
        )?;

        let tree_cfg = TreeConfig {
            initial_street: board.street(),
            starting_pot: cfg.starting_pot,
            effective_stack: cfg.effective_stack,
            betting_limit: BettingLimit::PotLimit,
            flop_sizes: [flop_oop, flop_ip],
            turn_sizes: [turn_oop, turn_ip],
            river_sizes: [river_oop, river_ip],
            active_betting_streets,
            ..TreeConfig::default()
        };
        let action_tree =
            ActionTree::build(tree_cfg).map_err(|e| format!("Betting tree error: {e}"))?;

        let live_oop = oop_range.without_dead(board.mask());
        let live_ip = ip_range.without_dead(board.mask());
        if live_oop.is_empty() || live_ip.is_empty() {
            return Err("No unblocked combos remain on this board".to_string());
        }
        // Check non-overlapping matchup existence up front
        let has_matchup = live_oop
            .entries()
            .iter()
            .any(|c0| live_ip.entries().iter().any(|c1| !c0.mask.intersects(c1.mask)));
        if !has_matchup {
            return Err("No non-overlapping 4-card OOP/IP hand pairs exist on this board".to_string());
        }

        let budget_bytes = cfg.budget_bytes();
        let max_side_live = live_oop.len().max(live_ip.len());
        let mut warnings = Vec::new();
        if let Some(note) = street_mode_note {
            warnings.push(note);
        }

        let (chosen_abstraction, recommended_iterations) =
            match cfg.abstraction_mode.trim().to_ascii_lowercase().as_str() {
                "identity" => (ResolvedAbstraction::Identity, default_iters_for_street(board.street())),
                "nutblocker50" | "nutblocker k=50" | "k=50" => {
                    warnings.push("NutBlocker k=50 disables suit isomorphism (buckets share strategy; showdown is exact 2+3).".to_string());
                    (ResolvedAbstraction::NutBlocker50, default_iters_for_street(board.street()))
                }
                "nutblocker100" | "nutblocker k=100" | "k=100" => {
                    warnings.push("NutBlocker k=100 disables suit isomorphism (buckets share strategy; showdown is exact 2+3).".to_string());
                    (ResolvedAbstraction::NutBlocker100, default_iters_for_street(board.street()))
                }
                _ => {
                    // Auto (recommended) policy:
                    // - River: prefer Identity if estimate <= budget, else NutBlocker k=50
                    // - Turn:  Identity when live combos <= 150 AND estimate <= budget, else NutBlocker k=50
                    // - Flop:  NEVER auto-start Identity on fat ranges (e.g. 500 vs 500) even when RAM is 16+ GiB;
                    //          use Identity only when compact (<= 150 live combos, <= 40k pairs) AND estimate <= budget,
                    //          otherwise switch to NutBlocker k=50.
                    let id_est = estimate_memory_for_config(
                        board,
                        &live_oop,
                        &live_ip,
                        &action_tree,
                        &IdentityAbstraction,
                        true,
                    )
                    .map_err(|e| e.to_string())?;

                    match board.street() {
                        Street::River => {
                            if id_est.total_estimated_bytes <= budget_bytes {
                                (ResolvedAbstraction::Identity, 1000)
                            } else {
                                warnings.push(format!(
                                    "Auto selected NutBlocker k=50 on River (Identity est. {:.1} MiB > budget {}); iso off.",
                                    id_est.total_estimated_bytes as f64 / (1024.0 * 1024.0),
                                    format_mib_human(budget_bytes as f64 / (1024.0 * 1024.0))
                                ));
                                (ResolvedAbstraction::NutBlocker50, 1000)
                            }
                        }
                        Street::Turn => {
                            if max_side_live <= 150 && id_est.total_estimated_bytes <= budget_bytes {
                                (ResolvedAbstraction::Identity, 500)
                            } else {
                                warnings.push(format!(
                                    "Auto selected NutBlocker k=50 on Turn (Identity est. {:.1} MiB, live combos = {max_side_live}); iso off.",
                                    id_est.total_estimated_bytes as f64 / (1024.0 * 1024.0)
                                ));
                                (ResolvedAbstraction::NutBlocker50, 500)
                            }
                        }
                        Street::Flop => {
                            let is_flop_only = active_betting_streets == [true, false, false];
                            if !is_flop_only && (live_v > 300 || max_side_live > 300) {
                                warnings.push(format!(
                                    "Large Flop range ({max_side_live} live combos): raising RAM does not shrink the 41×40 chance tree; consider Solve Mode = 'Flop Only', narrowing Villain's range, or starting on Turn/River if iterations are slow."
                                ));
                            }
                            if (is_flop_only && id_est.total_estimated_bytes <= budget_bytes)
                                || (max_side_live <= 150
                                    && live_oop.len().saturating_mul(live_ip.len()) <= 40_000
                                    && id_est.total_estimated_bytes <= budget_bytes)
                            {
                                (
                                    ResolvedAbstraction::Identity,
                                    if is_flop_only { 500 } else { 250 },
                                )
                            } else {
                                warnings.push(format!(
                                    "Auto selected NutBlocker k=50 on Flop (Identity requires {:.1} MiB / {max_side_live} live combos); iso off.",
                                    id_est.total_estimated_bytes as f64 / (1024.0 * 1024.0)
                                ));
                                (ResolvedAbstraction::NutBlocker50, 200)
                            }
                        }
                    }
                }
            };

        let abs_box = chosen_abstraction.boxed();
        let effective_iso = abs_box.is_suit_symmetric();
        let mem_report = estimate_memory_for_config(
            board,
            &live_oop,
            &live_ip,
            &action_tree,
            abs_box.as_ref(),
            true,
        )
        .map_err(|e| e.to_string())?;

        Ok(PreparedSpot {
            board,
            hero_hand,
            hero_player,
            oop_range: live_oop.clone(),
            ip_range: live_ip.clone(),
            raw_villain_combos: raw_v,
            live_villain_combos: live_v,
            live_oop_combos: live_oop.len(),
            live_ip_combos: live_ip.len(),
            action_tree,
            chosen_abstraction,
            recommended_iterations,
            effective_iso,
            mem_report,
            budget_bytes,
            solve_streets_label,
            warnings,
        })
    }

    /// Estimates tree memory and returns a full JSON breakdown without allocating `PloGame`.
    pub fn estimate_spot(&self, cfg: &SpotConfig) -> Result<Json, String> {
        let prep = self.prepare_spot(cfg)?;
        Ok(prep.estimate_to_json())
    }

    // ========================================================================
    // 4. Build Game, Solve Loop (with Stop + Progress Callback), and Tree Cursor
    // ========================================================================

    /// Builds and stores a `PloGame` in session state after strictly checking `estimate_tree_memory <= budget`.
    pub fn build_game(&self, cfg: &SpotConfig) -> Result<Json, String> {
        let _exclusive =
            self.acquire_exclusive_op("A solve is currently running; click Stop first")?;
        let prep = self.prepare_spot(cfg)?;

        // Always enforce pre-allocation memory budget guardrail!
        if prep.mem_report.total_estimated_bytes > prep.budget_bytes {
            return Err(format_memory_budget_exceeded(&prep));
        }

        let abs_box = prep.chosen_abstraction.boxed();
        let build_opts = SolverBuildOptions {
            max_memory_bytes: prep.budget_bytes,
            enable_chance_isomorphism: true,
            rake_fraction: (cfg.rake_pct / 100.0).clamp(0.0, 0.5),
            rake_cap_chips: cfg.rake_cap.max(0.0),
        };
        let game = PloGame::with_options(
            prep.board,
            &prep.oop_range,
            &prep.ip_range,
            &prep.action_tree,
            abs_box.as_ref(),
            build_opts,
        )
        .map_err(|e| match e {
            SolverError::MemoryBudgetExceeded { .. } => format_memory_budget_exceeded(&prep),
            other => other.to_string(),
        })?;

        let est_json = prep.estimate_to_json();
        *self.session.lock().unwrap() = Some(ActiveGameSession {
            game,
            spot_config: cfg.clone(),
            hero_hand: prep.hero_hand,
            hero_player: prep.hero_player,
            abstraction: prep.chosen_abstraction,
            warnings: prep.warnings,
            history_labels: Vec::new(),
            villain_filter: String::new(),
            convergence_log: Vec::new(),
            bot_hand: None,
            parent_session: None,
        });
        Ok(est_json)
    }

    /// Requests cooperative cancellation of an active `solve_loop` between DCFR iterations.
    pub fn stop_solve(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }

    pub fn latest_progress(&self) -> Option<SolveProgressEvent> {
        self.latest_progress.lock().unwrap().clone()
    }

    /// Runs `max_iterations` (or builds first if needed) synchronously on the calling worker thread,
    /// streaming [`SolveProgressEvent`] updates via `on_progress` and honoring `stop_solve()`.
    ///
    /// Note: During iterations, the game is temporarily moved out of `self.session` so `stop_solve()`
    /// and `latest_progress()` never block on the solver mutex!
    pub fn solve_blocking<F>(
        &self,
        cfg: &SpotConfig,
        report_every: u32,
        mut on_progress: F,
    ) -> Result<Json, String>
    where
        F: FnMut(&SolveProgressEvent),
    {
        let exclusive = self.acquire_exclusive_op("Solver is already running")?;
        self.stop_flag.store(false, Ordering::SeqCst);

        // Build fresh game for `cfg`
        let mut active = self.prepare_and_build_unlocked(cfg)?;

        let max_iters = cfg.max_iterations.max(1);
        let pot_f64 = active.game.tree_config().starting_pot as f64;
        let target_expl_chips = if cfg.target_dev_pct > 0.0 && pot_f64 > 0.0 {
            (cfg.target_dev_pct / 100.0) * pot_f64
        } else {
            0.0
        };
        let hero_player = active.hero_player;
        let start_iters = active.game.iterations_run();
        let t0 = Instant::now();

        let stride = if report_every > 0 {
            report_every
        } else {
            match active.game.root_board().street() {
                Street::River => 25,
                Street::Turn => 10,
                Street::Flop => 1,
            }
        };

        let mut conv_points: Vec<(u32, f64, f64)> = Vec::new();
        let (_, _, stopped) = active.game.solve_with_hooks(
            max_iters,
            target_expl_chips,
            stride,
            &self.stop_flag,
            |iters_done, expl_f64, evs_f64| {
                let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
                let delta = iters_done.saturating_sub(start_iters) as f64;
                let ips = delta / elapsed;
                let ev_hero = evs_f64[hero_player];
                let ev_villain = evs_f64[hero_player ^ 1];
                conv_points.push((iters_done, expl_f64, ev_hero));
                let ev = SolveProgressEvent {
                    iterations: iters_done,
                    max_iterations: start_iters + max_iters,
                    iters_per_sec: ips,
                    exploitability: expl_f64,
                    exploitability_sci: format!("{expl_f64:.4e}"),
                    ev_oop: evs_f64[0],
                    ev_ip: evs_f64[1],
                    ev_hero,
                    ev_villain,
                    hero_ev_pct_pot: if pot_f64 > 0.0 {
                        (ev_hero / pot_f64) * 100.0
                    } else {
                        0.0
                    },
                    status: "running".to_string(),
                };
                *self.latest_progress.lock().unwrap() = Some(ev.clone());
                on_progress(&ev);
            },
        );

        let final_expl = active.game.exploitability_f64();
        let final_evs = active.game.root_expected_values_f64();
        let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
        let ips = (active.game.iterations_run().saturating_sub(start_iters) as f64) / elapsed;
        let ev_hero = final_evs[hero_player];
        let ev_villain = final_evs[hero_player ^ 1];
        let final_status = if stopped { "stopped" } else { "completed" };
        let final_iters = active.game.iterations_run();
        if conv_points.last().map(|p| p.0) != Some(final_iters) {
            conv_points.push((final_iters, final_expl, ev_hero));
        }
        active.convergence_log = conv_points;

        let final_event = SolveProgressEvent {
            iterations: active.game.iterations_run(),
            max_iterations: start_iters + max_iters,
            iters_per_sec: ips,
            exploitability: final_expl,
            exploitability_sci: format!("{final_expl:.4e}"),
            ev_oop: final_evs[0],
            ev_ip: final_evs[1],
            ev_hero,
            ev_villain,
            hero_ev_pct_pot: if pot_f64 > 0.0 {
                (ev_hero / pot_f64) * 100.0
            } else {
                0.0
            },
            status: final_status.to_string(),
        };
        *self.latest_progress.lock().unwrap() = Some(final_event.clone());
        on_progress(&final_event);

        *self.session.lock().unwrap() = Some(active);
        drop(exclusive);

        let node_json = self.query_node()?;
        Ok(Json::obj([
            ("progress", final_event.to_json()),
            ("node", node_json),
        ]))
    }

    /// Steps additional iterations on the already-built `PloGame` without resetting regrets.
    pub fn step_iterations(&self, extra_iterations: u32) -> Result<Json, String> {
        let exclusive = self.acquire_exclusive_op("Solver is currently running")?;
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No active game built yet; build or solve first".to_string())?;
        self.stop_flag.store(false, Ordering::SeqCst);
        let t0 = Instant::now();
        let (_, _, stopped) = sess.game.solve_with_hooks(
            extra_iterations,
            0.0,
            extra_iterations.max(1),
            &self.stop_flag,
            |_, _, _| {},
        );
        let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
        let expl = sess.game.exploitability_f64();
        let evs = sess.game.root_expected_values_f64();
        let pot = sess.game.tree_config().starting_pot as f64;
        let hp = sess.hero_player;
        sess.convergence_log
            .push((sess.game.iterations_run(), expl, evs[hp]));
        let ev = SolveProgressEvent {
            iterations: sess.game.iterations_run(),
            max_iterations: sess.game.iterations_run(),
            iters_per_sec: (extra_iterations as f64) / elapsed,
            exploitability: expl,
            exploitability_sci: format!("{expl:.4e}"),
            ev_oop: evs[0],
            ev_ip: evs[1],
            ev_hero: evs[hp],
            ev_villain: evs[hp ^ 1],
            hero_ev_pct_pot: if pot > 0.0 { (evs[hp] / pot) * 100.0 } else { 0.0 },
            status: if stopped { "stopped" } else { "completed" }.to_string(),
        };
        drop(guard);
        drop(exclusive);
        let node = self.query_node()?;
        Ok(Json::obj([("progress", ev.to_json()), ("node", node)]))
    }

    fn prepare_and_build_unlocked(&self, cfg: &SpotConfig) -> Result<ActiveGameSession, String> {
        let prep = self.prepare_spot(cfg)?;
        if prep.mem_report.total_estimated_bytes > prep.budget_bytes {
            return Err(format_memory_budget_exceeded(&prep));
        }
        let abs_box = prep.chosen_abstraction.boxed();
        let build_opts = SolverBuildOptions {
            max_memory_bytes: prep.budget_bytes,
            enable_chance_isomorphism: true,
            rake_fraction: (cfg.rake_pct / 100.0).clamp(0.0, 0.5),
            rake_cap_chips: cfg.rake_cap.max(0.0),
        };
        let game = PloGame::with_options(
            prep.board,
            &prep.oop_range,
            &prep.ip_range,
            &prep.action_tree,
            abs_box.as_ref(),
            build_opts,
        )
        .map_err(|e| match e {
            SolverError::MemoryBudgetExceeded { .. } => format_memory_budget_exceeded(&prep),
            other => other.to_string(),
        })?;

        Ok(ActiveGameSession {
            game,
            spot_config: cfg.clone(),
            hero_hand: prep.hero_hand,
            hero_player: prep.hero_player,
            abstraction: prep.chosen_abstraction,
            warnings: prep.warnings,
            history_labels: Vec::new(),
            villain_filter: String::new(),
            convergence_log: Vec::new(),
            bot_hand: None,
            parent_session: None,
        })
    }

    // ========================================================================
    // 5. Subgame Resolving (`#1`), Multi-Flop Subset (`#5`), Class Node Locking (`#6`), & Tree Cursor (`#7`)
    // ========================================================================

    /// `#1` Exact Subgame Resolving:
    /// At the start of a Turn or River street (`committed[0] == committed[1]` and `node_type == "oop"`),
    /// extracts both players' exact current conditional `reach` distributions, builds a fresh `IdentityAbstraction`
    /// subgame with the remaining stack (`effective_stack - committed[0]`) and current pot (`report.pot`),
    /// solves it to `target_dev_pct` / `max_iterations`, and pushes the parent game onto `parent_session`
    /// so the user can return to the parent tree via `exit_subgame`!
    pub fn resolve_subgame(
        &self,
        max_iterations: u32,
        target_dev_pct: f64,
    ) -> Result<Json, String> {
        let _exclusive =
            self.acquire_exclusive_op("Solver is currently running; click Stop first")?;
        let mut guard = self.session.lock().unwrap();
        let prev = guard
            .take()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;

        let report = prev.game.inspect_current_node();
        let street = report.board.street();
        if !matches!(street, Street::Turn | Street::River) {
            *guard = Some(prev);
            return Err("Exact Subgame Resolve is intended for Turn or River nodes (step to a Turn/River card first)".to_string());
        }
        if report.committed[0] != report.committed[1] || report.node_type != "oop" {
            *guard = Some(prev);
            return Err("Step to the start of a Turn or River street (OOP first to act before bets on this street) to resolve the subgame".to_string());
        }

        let orig_eff_stack = prev.game.tree_config().effective_stack;
        let rem_stack = orig_eff_stack - report.committed[0];
        if rem_stack <= 0 {
            *guard = Some(prev);
            return Err("Players are already all-in; no remaining stack to resolve".to_string());
        }

        let reaches = prev.game.current_combo_reaches();
        let oop_pairs: Vec<(Hand4, f32)> = prev
            .game
            .combos(PLAYER_OOP)
            .iter()
            .zip(reaches[PLAYER_OOP].iter())
            .filter(|(_, &r)| r > 1e-7)
            .map(|(c, &r)| (c.hand, r))
            .collect();
        let ip_pairs: Vec<(Hand4, f32)> = prev
            .game
            .combos(PLAYER_IP)
            .iter()
            .zip(reaches[PLAYER_IP].iter())
            .filter(|(_, &r)| r > 1e-7)
            .map(|(c, &r)| (c.hand, r))
            .collect();

        if oop_pairs.is_empty() || ip_pairs.is_empty() {
            *guard = Some(prev);
            return Err("One player has zero reaching combos at this node; cannot resolve empty subgame".to_string());
        }

        let oop_range = match PloRange::from_hands(oop_pairs) {
            Ok(r) => r,
            Err(e) => {
                *guard = Some(prev);
                return Err(format!("Subgame OOP range error: {e}"));
            }
        };
        let ip_range = match PloRange::from_hands(ip_pairs) {
            Ok(r) => r,
            Err(e) => {
                *guard = Some(prev);
                return Err(format!("Subgame IP range error: {e}"));
            }
        };

        // Build `ActionTree` for the subgame using the spot's configured per-street sizings (with full River enabled!)
        let parent_tcfg = prev.game.tree_config().clone();
        let sub_tcfg = TreeConfig {
            initial_street: street,
            starting_pot: report.pot,
            effective_stack: rem_stack,
            active_betting_streets: [true, true, true],
            ..parent_tcfg
        };
        let sub_tree = match ActionTree::build(sub_tcfg) {
            Ok(t) => t,
            Err(e) => {
                *guard = Some(prev);
                return Err(format!("Subgame tree error: {e}"));
            }
        };

        let budget_bytes = prev.spot_config.budget_bytes();
        // Try exact `IdentityAbstraction` first; fall back to `NutBlocker100` only if huge Turn subgame exceeds RAM budget
        let id_est = match estimate_memory_for_config(
            report.board,
            &oop_range,
            &ip_range,
            &sub_tree,
            &IdentityAbstraction,
            true,
        ) {
            Ok(m) => m,
            Err(e) => {
                *guard = Some(prev);
                return Err(format!("Subgame memory estimate failed: {e}"));
            }
        };
        let chosen_abs = if id_est.total_estimated_bytes <= budget_bytes {
            ResolvedAbstraction::Identity
        } else {
            ResolvedAbstraction::NutBlocker100
        };
        let abs_box = chosen_abs.boxed();

        let (rake_frac, rake_cap) = prev.game.rake_config();
        let build_opts = SolverBuildOptions {
            max_memory_bytes: budget_bytes,
            enable_chance_isomorphism: true,
            rake_fraction: rake_frac,
            rake_cap_chips: rake_cap,
        };
        let mut sub_game = match PloGame::with_options(
            report.board,
            &oop_range,
            &ip_range,
            &sub_tree,
            abs_box.as_ref(),
            build_opts,
        ) {
            Ok(g) => g,
            Err(e) => {
                *guard = Some(prev);
                return Err(format!("Subgame build error: {e}"));
            }
        };

        let iters = max_iterations.max(50);
        let target_chips = if target_dev_pct > 0.0 && report.pot > 0 {
            (target_dev_pct / 100.0) * (report.pot as f64)
        } else {
            0.0
        };
        self.stop_flag.store(false, Ordering::SeqCst);
        let t0 = Instant::now();
        let _ = sub_game.solve_with_hooks(iters, target_chips, iters, &self.stop_flag, |_, _, _| {});
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut warnings = vec![format!(
            "Exact Subgame Resolved on {:?} {} (Pot {}, Rem Stack {}, {} in {:.0} ms).",
            street,
            report.board,
            report.pot,
            rem_stack,
            chosen_abs.as_label(),
            elapsed_ms
        )];
        warnings.extend(prev.warnings.iter().cloned());

        let mut new_sess = ActiveGameSession {
            game: sub_game,
            spot_config: prev.spot_config.clone(),
            hero_hand: prev.hero_hand,
            hero_player: prev.hero_player,
            abstraction: chosen_abs,
            warnings,
            history_labels: Vec::new(),
            villain_filter: prev.villain_filter.clone(),
            convergence_log: Vec::new(),
            bot_hand: None,
            parent_session: Some(Box::new(prev)),
        };
        let node_json = build_node_query_json(&mut new_sess);
        *guard = Some(new_sess);
        Ok(node_json)
    }

    /// Returns from an active resolved subgame back to the parent tree cursor (`#1`).
    pub fn exit_subgame(&self) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let cur = guard
            .take()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        if let Some(mut parent) = cur.parent_session {
            let j = build_node_query_json(&mut parent);
            *guard = Some(*parent);
            Ok(j)
        } else {
            let mut same = cur;
            let j = build_node_query_json(&mut same);
            *guard = Some(same);
            Err(format!("Not currently inside a resolved subgame (board {})", j.get("board").and_then(|v| v.as_str()).unwrap_or("")))
        }
    }

    /// `#6` Advanced Node Locking by PLO Hand Classes:
    /// `class_locks` maps `PloHandClass::id()` (e.g. `"fh_plus"`, `"flush"`, `"air"`, `"nut_fd"`, `"wrap"`, or `"*"` fallback)
    /// to an action-probability vector `[p0, p1, ...]`. Unlisted classes preserve their current equilibrium strategy!
    pub fn lock_node_by_classes(
        &self,
        class_locks: &std::collections::BTreeMap<String, Vec<f32>>,
        resolve_iterations: u32,
    ) -> Result<Json, String> {
        let _exclusive =
            self.acquire_exclusive_op("Solver is currently running; click Stop first")?;
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        let cur_board = sess.game.inspect_current_node().board;
        sess.game
            .lock_current_node_by_combo(|_idx, hand| {
                let feat = analyze_plo_hand(hand, &cur_board);
                let cid = feat.primary_class.id();
                if let Some(v) = class_locks.get(cid) {
                    Some(v.clone())
                } else {
                    class_locks.get("*").cloned()
                }
            })
            .map_err(|e| e.to_string())?;

        if resolve_iterations > 0 {
            self.stop_flag.store(false, Ordering::SeqCst);
            let _ = sess.game.solve_with_hooks(
                resolve_iterations,
                0.0,
                resolve_iterations.max(1),
                &self.stop_flag,
                |_, _, _| {},
            );
        }
        Ok(build_node_query_json(sess))
    }

    /// `#7` Jumps directly to any node on the current street's Tree Minimap (`street_root_depth` + `action_path`)!
    pub fn jump_street_path(&self, action_path: &[usize]) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        let (street_root_depth, _) = sess.game.current_street_tree_minimap();
        let prefix_steps: Vec<plo_cfr::HistoryStep> = sess
            .game
            .history()
            .iter()
            .take(street_root_depth)
            .cloned()
            .collect();
        sess.game.back_to_root();
        sess.history_labels.truncate(prefix_steps.len());
        for st in prefix_steps {
            match st {
                plo_cfr::HistoryStep::Action(idx) => {
                    sess.game.play_action(idx).map_err(|e| e.to_string())?;
                }
                plo_cfr::HistoryStep::Deal(c) => {
                    sess.game.play_chance_card(c).map_err(|e| e.to_string())?;
                }
            }
        }
        for &a_idx in action_path {
            let pre = sess.game.inspect_current_node();
            let actor_tag = match pre.node_type {
                "oop" => "OOP",
                "ip" => "IP",
                _ => "",
            };
            let act_name = pre
                .actions
                .get(a_idx)
                .map(|a| a.to_string())
                .unwrap_or_else(|| format!("Action#{a_idx}"));
            sess.game.play_action(a_idx).map_err(|e| e.to_string())?;
            sess.history_labels.push(format!("{actor_tag} {act_name}"));
        }
        Ok(build_node_query_json(sess))
    }

    /// `#5` Multi-Flop Subset C-Bet Report:
    /// Runs rapid `FlopOnly` solves across a list of PLO flops (excluding any flop that overlaps Hero's 4 hole cards)
    /// and returns a structured comparative table of Hero EV, `% Pot`, Villain Equity/EV, and Root Action Frequencies!
    pub fn run_flop_subset_report(
        &self,
        base_cfg: &SpotConfig,
        flops: &[String],
        iterations_per_flop: u32,
    ) -> Result<Json, String> {
        let _exclusive =
            self.acquire_exclusive_op("Solver is currently running; click Stop first")?;
        let default_subset = [
            "AsKs7d", // A-high Broadway Two-Tone
            "KsQhJc", // High Broadway Rainbow
            "QsJs9s", // Monotone High Connected
            "JhTh6d", // Mid-High Connected Two-Tone
            "Ts9d8c", // Mid Straight Connected Rainbow
            "9h9d4s", // Paired Mid Dry
            "As7d2c", // A-high Dry Rainbow
            "7s6s5d", // Low Connected Two-Tone
        ];
        let raw_list: Vec<String> = if flops.is_empty() {
            default_subset.iter().map(|s| s.to_string()).collect()
        } else {
            flops
                .iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .take(25)
                .collect()
        };

        let hero_mask = if !base_cfg.lab_two_range_mode {
            let (hh, _) = parse_hero_hand(&base_cfg.hero_hand)?;
            hh.mask()
        } else {
            CardMask::EMPTY
        };

        let iters = iterations_per_flop.clamp(20, 1000);
        let mut rows = Vec::new();
        let mut skipped = Vec::new();
        self.stop_flag.store(false, Ordering::SeqCst);

        for f_str in raw_list {
            if self.stop_flag.load(Ordering::SeqCst) {
                break;
            }
            let b = match parse_postflop_board(&f_str) {
                Ok(b) if b.street() == Street::Flop => b,
                Ok(_) => {
                    skipped.push(format!("{f_str} (not a 3-card flop)"));
                    continue;
                }
                Err(e) => {
                    skipped.push(format!("{f_str} ({e})"));
                    continue;
                }
            };
            if b.mask().intersects(hero_mask) {
                skipped.push(format!("{b} (overlaps Hero hole cards)"));
                continue;
            }

            let mut sub_cfg = base_cfg.clone();
            sub_cfg.board = b.to_string();
            sub_cfg.solve_streets = "FlopOnly".to_string();
            sub_cfg.max_iterations = iters;

            let mut sess = match self.prepare_and_build_unlocked(&sub_cfg) {
                Ok(s) => s,
                Err(e) => {
                    skipped.push(format!("{b} ({e})"));
                    continue;
                }
            };
            let pot_f64 = sess.game.tree_config().starting_pot as f64;
            let target_chips = if sub_cfg.target_dev_pct > 0.0 && pot_f64 > 0.0 {
                (sub_cfg.target_dev_pct / 100.0) * pot_f64
            } else {
                0.0
            };
            let _ = sess
                .game
                .solve_with_hooks(iters, target_chips, iters, &self.stop_flag, |_, _, _| {});

            let node_j = build_node_query_json(&mut sess);
            let root_ev_hero = node_j.get("root_ev_hero").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let root_ev_pct = node_j
                .get("root_ev_hero_pct_pot")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let expl = node_j.get("exploitability").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let hero_acts = node_j
                .get("hero_acts_here")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let root_strat = if hero_acts {
                node_j.get("hero_strategy").cloned().unwrap_or(Json::Arr(Vec::new()))
            } else {
                node_j
                    .get("villain_aggregate_strategy")
                    .cloned()
                    .unwrap_or(Json::Arr(Vec::new()))
            };

            rows.push(Json::obj([
                ("flop", Json::Str(b.to_string())),
                ("texture", Json::Str(classify_flop_texture(&b))),
                ("hero_acts_root", Json::Bool(hero_acts)),
                ("hero_class", node_j.get("hero_hand_class").cloned().unwrap_or(Json::Null)),
                ("hero_ev", Json::Num(root_ev_hero)),
                ("hero_ev_pct_pot", Json::Num(root_ev_pct)),
                ("exploitability", Json::Num(expl)),
                ("root_strategy", root_strat),
            ]));
        }

        Ok(Json::obj([
            ("count", Json::Num(rows.len() as f64)),
            ("iterations_per_flop", Json::Num(iters as f64)),
            ("rows", Json::Arr(rows)),
            (
                "skipped",
                Json::Arr(skipped.into_iter().map(Json::Str).collect()),
            ),
        ]))
    }

    // ========================================================================
    // Phase-3 Roadmap (`#1` Full Hand EV Audit, `#2` Play vs Solver Bot, `#3` Pure-Size Optimizer, `#5` Binary `.plotree` Snapshots)
    // ========================================================================

    /// `#1 (Phase 3)` Full Hand EV Audit:
    /// Parses a multi-street PLO-4 hand history, solves each street Hero acted on (Flop -> Turn -> River)
    /// using fast street-isolated trees (propagating Villain's reaching combos across streets and updating
    /// pot & effective stack at each street boundary!), maps every action in the hand history to the closest
    /// legal tree action, and grades every Hero decision (`Optimal` / `Inaccuracy` / `Blunder` + exact `ΔEV` in chips & BB)!
    pub fn audit_hand_history(
        &self,
        hh_text: &str,
        villain_override_opt: &str,
        iters_per_street: u32,
    ) -> Result<Json, String> {
        let _exclusive =
            self.acquire_exclusive_op("Solver is currently running; click Stop first")?;
        let parsed = parse_plo4_hand_history(hh_text)?;
        let iters = iters_per_street.clamp(40, 1000);
        let chips_per_bb = parsed.chips_per_bb.max(1e-6);

        // Build base SpotConfig from parsed HH (using pack if loaded & matching, else override or broad PLO macro)
        let pack_has_spot = {
            let pg = self.pack.lock().unwrap();
            pg.as_ref()
                .map(|p| p.has_available_spot(&parsed.suggested_spot_id))
                .unwrap_or(false)
        };
        let mut base_cfg = SpotConfig {
            board: parsed.flop_board.to_string(),
            hero_hand: parsed.hero_hand.to_string(),
            hero_seat: parsed.hero_seat.clone(),
            use_pack: pack_has_spot && villain_override_opt.trim().is_empty(),
            villain_spot_id: parsed.suggested_spot_id.clone(),
            villain_override_range: if !villain_override_opt.trim().is_empty() {
                villain_override_opt.trim().to_string()
            } else if !pack_has_spot {
                "AA82,KK97,QQJT,JT98,AQT9,KQT9".to_string()
            } else {
                String::new()
            },
            starting_pot: parsed.starting_pot,
            effective_stack: parsed.effective_stack,
            flop_oop_bets: "33%,75%".to_string(),
            flop_ip_bets: "33%,75%".to_string(),
            turn_oop_bets: "50%,100%".to_string(),
            turn_ip_bets: "50%,100%".to_string(),
            river_oop_bets: "50%,100%".to_string(),
            river_ip_bets: "50%,100%".to_string(),
            solve_streets: "FlopOnly".to_string(),
            max_iterations: iters,
            ..SpotConfig::default()
        };
        base_cfg.memory_budget_mib = self.effective_memory_budget_mib();

        let street_defs: Vec<(&str, Board, &str)> = {
            let mut v = vec![("Flop", parsed.flop_board, "FlopOnly")];
            if let Some(tc) = parsed.turn_card {
                if let Ok(tb) = parsed.flop_board.deal(tc) {
                    v.push(("Turn", tb, "TurnOnly"));
                    if let Some(rc) = parsed.river_card {
                        if let Ok(rb) = tb.deal(rc) {
                            v.push(("River", rb, "RiverOnly"));
                        }
                    }
                }
            }
            v
        };

        let mut decisions_json = Vec::new();
        let mut total_ev_loss_chips = 0.0f64;
        let mut cur_pot = parsed.starting_pot;
        let mut cur_rem_stack = parsed.effective_stack;
        self.stop_flag.store(false, Ordering::SeqCst);

        for (st_name, st_board, st_mode) in street_defs {
            let st_acts: Vec<_> = parsed
                .postflop_actions
                .iter()
                .filter(|a| a.street == st_name)
                .collect();
            if st_acts.is_empty() || cur_rem_stack <= 0 {
                break;
            }

            let mut st_cfg = base_cfg.clone();
            st_cfg.board = st_board.to_string();
            st_cfg.starting_pot = cur_pot;
            st_cfg.effective_stack = cur_rem_stack;
            st_cfg.solve_streets = st_mode.to_string();
            let st_iters = if st_name == "Flop" {
                iters.clamp(15, 35)
            } else {
                iters
            };
            st_cfg.max_iterations = st_iters;

            let mut sess = self.prepare_and_build_unlocked(&st_cfg)?;
            let _ = sess
                .game
                .solve_with_hooks(st_iters, 0.0, st_iters, &self.stop_flag, |_, _, _| {});

            // Walk recorded actions on `st_name`
            for rec in st_acts {
                let node_j = build_node_query_json(&mut sess);
                let ntype = node_j.get("node_type").and_then(|v| v.as_str()).unwrap_or("");
                if ntype != "oop" && ntype != "ip" {
                    break;
                }
                let node_acts = node_j
                    .get("actions")
                    .and_then(|v| v.as_arr())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                if node_acts.is_empty() {
                    break;
                }

                let chosen_idx = match_hh_action_to_tree_index(&rec.action_kind, rec.amount_chips, &node_acts);
                if rec.actor_is_hero && node_j.get("hero_acts_here").and_then(|v| v.as_bool()) == Some(true) {
                    let h_strat = node_j
                        .get("hero_strategy")
                        .and_then(|v| v.as_arr())
                        .map(|s| s.to_vec())
                        .unwrap_or_default();
                    let picked = h_strat
                        .iter()
                        .find(|s| s.get("action_index").and_then(|v| v.as_usize()) == Some(chosen_idx));
                    let best = h_strat.iter().max_by(|a, b| {
                        let ea = a.get("ev").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let eb = b.get("ev").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        ea.partial_cmp(&eb).unwrap_or(core::cmp::Ordering::Equal)
                    });

                    let gto_freq = picked
                        .and_then(|s| s.get("probability"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let ev_loss = picked
                        .and_then(|s| s.get("ev_loss"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let ev_loss_bb = ev_loss / chips_per_bb;
                    let pot_now = node_j.get("pot").and_then(|v| v.as_f64()).unwrap_or(cur_pot as f64).max(1.0);
                    let loss_pct_pot = (ev_loss / pot_now) * 100.0;
                    let grade = if loss_pct_pot <= 0.6 {
                        "Optimal"
                    } else if loss_pct_pot <= 2.5 {
                        "Inaccuracy"
                    } else {
                        "Blunder"
                    };
                    let best_act = best
                        .and_then(|s| s.get("action"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    let best_freq = best
                        .and_then(|s| s.get("probability"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let mapped_lbl = node_acts
                        .get(chosen_idx)
                        .and_then(|a| a.get("label"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");

                    total_ev_loss_chips += ev_loss;
                    decisions_json.push(Json::obj([
                        ("street", Json::Str(st_name.to_string())),
                        ("board", Json::Str(st_board.to_string())),
                        ("pot", Json::Num(pot_now)),
                        ("effective_stack", Json::Num(cur_rem_stack as f64)),
                        ("played_raw", Json::Str(rec.raw_desc.clone())),
                        ("raw_hh_action", Json::Str(rec.raw_desc.clone())),
                        ("mapped_action", Json::Str(mapped_lbl.to_string())),
                        ("mapped_tree_action", Json::Str(mapped_lbl.to_string())),
                        ("gto_frequency", Json::Num(gto_freq)),
                        ("gto_freq_pct", Json::Num(gto_freq * 100.0)),
                        ("best_action", Json::Str(best_act.to_string())),
                        ("best_gto_action", Json::Str(best_act.to_string())),
                        ("best_gto_freq", Json::Num(best_freq)),
                        ("best_freq_pct", Json::Num(best_freq * 100.0)),
                        ("ev_loss_chips", Json::Num(ev_loss)),
                        ("ev_loss_bb", Json::Num(ev_loss_bb)),
                        ("ev_loss_pct_pot", Json::Num(loss_pct_pot)),
                        ("grade", Json::Str(grade.to_string())),
                        ("spot_config", st_cfg.to_json()),
                    ]));
                }

                let _ = sess.game.play_action(chosen_idx);
            }

            // Update `cur_pot` and `cur_rem_stack` from the end-of-street node in `sess.game`
            let end_rep = sess.game.inspect_current_node();
            cur_pot = end_rep.pot;
            cur_rem_stack = (cur_rem_stack - end_rep.committed[0]).max(0);
        }

        let dec_cnt = decisions_json.len();
        Ok(Json::obj([
            ("parsed", parsed.to_json()),
            ("hero_hand", Json::Str(parsed.hero_hand.to_string())),
            ("hero_seat", Json::Str(parsed.hero_seat.clone())),
            ("villain_spot_id", Json::Str(parsed.suggested_spot_id.clone())),
            ("decision_count", Json::Num(dec_cnt as f64)),
            ("decisions", Json::Arr(decisions_json)),
            ("total_ev_loss_chips", Json::Num(total_ev_loss_chips)),
            ("total_ev_loss_bb", Json::Num(total_ev_loss_chips / chips_per_bb)),
        ]))
    }

    /// `#2 (Phase 3)` Play vs Solver Bot:
    /// Starts or steps an interactive playout against the solved GTO equilibrium!
    /// - `start_bot_hand(seed)`: resets cursor to Root, samples one concrete non-overlapping Villain 4-card hand
    ///   proportional to Villain's root weights, and auto-plays any Villain / Chance turns until Hero acts or Terminal!
    /// - `bot_play_hero_action(action_idx, seed)`: records Hero's chosen action (`ev_loss` graded), advances Hero's
    ///   action, and auto-plays Villain (consistent with `bot_hand`'s exact equilibrium strategy!) and random live
    ///   Turn/River Chance runouts (`disjoint from board | hero_hand | bot_hand`) until Hero's next decision or Terminal!
    pub fn start_bot_hand(&self, seed: u64) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "Solve a spot first before starting Play vs Solver".to_string())?;
        sess.game.back_to_root();
        sess.history_labels.clear();

        let hp = sess.hero_player;
        let vp = hp ^ 1;
        let hero_mask = sess.hero_hand.map(|h| h.mask()).unwrap_or(CardMask::EMPTY);
        let bmask = sess.game.inspect_current_node().board.mask();
        let v_combos = sess.game.combos(vp).to_vec();

        let mut rng_state = mix_seed(seed);
        let candidates: Vec<(Hand4, f64)> = v_combos
            .iter()
            .filter(|c| !c.mask.intersects(bmask) && !c.mask.intersects(hero_mask) && c.weight > 0.0)
            .map(|c| (c.hand, c.weight as f64))
            .collect();
        if candidates.is_empty() {
            return Err("No valid non-overlapping Villain hands available for bot playout".to_string());
        }
        let tot_w: f64 = candidates.iter().map(|x| x.1).sum();
        let mut pick_r = next_unit_f64(&mut rng_state) * tot_w;
        let mut chosen_bot = candidates[0].0;
        for (h, w) in &candidates {
            if pick_r <= *w {
                chosen_bot = *h;
                break;
            }
            pick_r -= *w;
        }
        sess.bot_hand = Some(chosen_bot);

        let bot_events = advance_bot_and_chance_until_hero_or_end(sess, &mut rng_state)?;
        let node_j = build_node_query_json(sess);
        Ok(Json::obj([
            ("node", node_j),
            ("bot_events", Json::Arr(bot_events.into_iter().map(Json::Str).collect())),
        ]))
    }

    pub fn bot_play_hero_action(&self, action_idx: usize, seed: u64) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No active Play vs Solver session".to_string())?;
        if sess.bot_hand.is_none() {
            return Err("Bot hand not initialized; click 'Play Hand' first".to_string());
        }

        // Grade Hero's chosen action BEFORE stepping
        let pre_j = build_node_query_json(sess);
        let h_strat = pre_j
            .get("hero_strategy")
            .and_then(|v| v.as_arr())
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let picked = h_strat
            .iter()
            .find(|s| s.get("action_index").and_then(|v| v.as_usize()) == Some(action_idx))
            .cloned()
            .unwrap_or(Json::Null);

        let pre_rep = sess.game.inspect_current_node();
        let actor_tag = if pre_rep.node_type == "oop" { "OOP" } else { "IP" };
        let act_name = pre_rep
            .actions
            .get(action_idx)
            .map(|a| a.to_string())
            .unwrap_or_else(|| format!("Action#{action_idx}"));
        sess.game
            .play_action(action_idx)
            .map_err(|e| e.to_string())?;
        sess.history_labels.push(format!("{actor_tag} {act_name}"));

        let mut rng_state = mix_seed(seed);
        let bot_events = advance_bot_and_chance_until_hero_or_end(sess, &mut rng_state)?;
        let node_j = build_node_query_json(sess);

        Ok(Json::obj([
            ("hero_action_grade", picked.clone()),
            ("picked_grade", picked),
            ("node", node_j),
            ("bot_events", Json::Arr(bot_events.into_iter().map(Json::Str).collect())),
        ]))
    }

    pub fn stop_bot_hand(&self) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        sess.bot_hand = None;
        Ok(build_node_query_json(sess))
    }

    /// `#3 (Phase 3)` Auto-Simplify Strategy / Pure-Size Optimizer:
    /// Tests candidate single-bet-size trees (`33%`, `50%`, `67%`, `75%`, `100%`) against the multi-size baseline
    /// on the current street (`FlopOnly` / `TurnOnly` / `RiverOnly`), measuring exact Hero Root EV retention (`%` of baseline EV
    /// and `ΔEV` in `% pot`) so the player knows which single bet size captures `99.5%+` of full GTO EV!
    pub fn compare_bet_simplifications(
        &self,
        base_cfg: &SpotConfig,
        candidate_sizes: &[String],
        iterations: u32,
    ) -> Result<Json, String> {
        let _exclusive =
            self.acquire_exclusive_op("Solver is currently running; click Stop first")?;
        let board = parse_postflop_board(&base_cfg.board)?;
        let street_mode = match board.street() {
            Street::Flop => "FlopOnly",
            Street::Turn => "TurnOnly",
            Street::River => "RiverOnly",
        };
        let iters = iterations.clamp(40, 1000);
        let defaults = ["33%", "50%", "67%", "75%", "100%"];
        let cands: Vec<String> = if candidate_sizes.is_empty() {
            defaults.iter().map(|s| s.to_string()).collect()
        } else {
            candidate_sizes
                .iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .take(8)
                .collect()
        };

        self.stop_flag.store(false, Ordering::SeqCst);

        // 1. Solve Full Multi-Size Baseline (`33%,67%,100%`)
        let mut full_cfg = base_cfg.clone();
        full_cfg.solve_streets = street_mode.to_string();
        full_cfg.max_iterations = iters;
        apply_street_bet_size_to_cfg(&mut full_cfg, board.street(), "33%,67%,100%");

        let mut base_sess = self.prepare_and_build_unlocked(&full_cfg)?;
        let _ = base_sess
            .game
            .solve_with_hooks(iters, 0.0, iters, &self.stop_flag, |_, _, _| {});
        let base_j = build_node_query_json(&mut base_sess);
        let baseline_ev = base_j.get("root_ev_hero").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let base_hero_acts = base_j.get("hero_acts_here").and_then(|v| v.as_bool()).unwrap_or(false);
        let base_strat = if base_hero_acts {
            base_j.get("hero_strategy").cloned().unwrap_or(Json::Arr(Vec::new()))
        } else {
            base_j
                .get("villain_aggregate_strategy")
                .cloned()
                .unwrap_or(Json::Arr(Vec::new()))
        };
        let pot_f64 = full_cfg.starting_pot.max(1) as f64;

        // 2. Solve each Single-Size candidate (prepend Baseline row first)
        let mut rows = vec![Json::obj([
            ("label", Json::Str("Full Multi-Size (33%, 67%, 100%)".to_string())),
            ("sizes", Json::Str("33%, 67%, 100%".to_string())),
            ("size", Json::Str("33%, 67%, 100%".to_string())),
            ("is_baseline", Json::Bool(true)),
            ("hero_ev", Json::Num(baseline_ev)),
            ("ev_diff_chips", Json::Num(0.0)),
            ("ev_diff_pct_pot", Json::Num(0.0)),
            ("ev_retention_pct", Json::Num(100.0)),
            ("root_strategy", base_strat.clone()),
            ("strategy", base_strat),
        ])];
        let mut best_size = String::new();
        let mut best_ev = f64::NEG_INFINITY;

        for sz in cands {
            if self.stop_flag.load(Ordering::SeqCst) {
                break;
            }
            let mut c_cfg = full_cfg.clone();
            apply_street_bet_size_to_cfg(&mut c_cfg, board.street(), &sz);
            let mut c_sess = match self.prepare_and_build_unlocked(&c_cfg) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let _ = c_sess
                .game
                .solve_with_hooks(iters, 0.0, iters, &self.stop_flag, |_, _, _| {});
            let cj = build_node_query_json(&mut c_sess);
            let ev = cj.get("root_ev_hero").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let diff_chips = ev - baseline_ev;
            let diff_pct_pot = (diff_chips / pot_f64) * 100.0;
            let retention_pct = if baseline_ev.abs() > 1e-6 {
                (ev / baseline_ev) * 100.0
            } else {
                100.0
            };
            let hero_acts = cj.get("hero_acts_here").and_then(|v| v.as_bool()).unwrap_or(false);
            let strat = if hero_acts {
                cj.get("hero_strategy").cloned().unwrap_or(Json::Arr(Vec::new()))
            } else {
                cj.get("villain_aggregate_strategy")
                    .cloned()
                    .unwrap_or(Json::Arr(Vec::new()))
            };
            if ev > best_ev {
                best_ev = ev;
                best_size = sz.clone();
            }
            rows.push(Json::obj([
                ("label", Json::Str(format!("Pure {sz}"))),
                ("sizes", Json::Str(sz.clone())),
                ("size", Json::Str(sz)),
                ("is_baseline", Json::Bool(false)),
                ("hero_ev", Json::Num(ev)),
                ("ev_diff_chips", Json::Num(diff_chips)),
                ("ev_diff_pct_pot", Json::Num(diff_pct_pot)),
                ("ev_retention_pct", Json::Num(retention_pct)),
                ("root_strategy", strat.clone()),
                ("strategy", strat),
            ]));
        }

        Ok(Json::obj([
            ("street", Json::Str(format!("{:?}", board.street()))),
            ("board", Json::Str(board.to_string())),
            ("baseline_sizes", Json::Str("33%,67%,100%".to_string())),
            ("baseline_ev", Json::Num(baseline_ev)),
            ("baseline_hero_ev", Json::Num(baseline_ev)),
            ("recommended_size", Json::Str(best_size.clone())),
            ("recommended_single_size", Json::Str(best_size)),
            ("rows", Json::Arr(rows)),
        ]))
    }

    /// `#5 (Phase 3)` Saves the currently solved `PloGame` (`SpotConfig` + binary `regrets`/`cum_strategy` + `convergence_log`)
    /// to a compact `.plotree` snapshot file (or returns base64/hex-free portable `.plotree` file on disk)!
    pub fn save_tree_snapshot(&self, file_path: &str) -> Result<Json, String> {
        let _exclusive =
            self.acquire_exclusive_op("Solver is currently running; click Stop first")?;
        let guard = self.session.lock().unwrap();
        let sess = guard
            .as_ref()
            .ok_or_else(|| "No solved game in memory to snapshot".to_string())?;

        let trimmed = file_path.trim();
        if trimmed.is_empty() {
            return Err("Snapshot file path is empty".to_string());
        }
        let path = PathBuf::from(trimmed);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create snapshot directory: {e}"))?;
            }
        }

        // Binary `.plotree` container format (`PLOTREE2` + `PLOSNAP2`):
        // - `PLOTREE2` (8 bytes)
        // - `cfg_json_len: u32 LE` + UTF-8 `SpotConfig` JSON bytes
        // - `weights_len: u32 LE` + `PloGame::export_weights_binary()` (`PLOSNAP2` + 64-bit hash) bytes
        let mut cfg_to_save = sess.spot_config.clone();
        // Record exact resolved abstraction so `load_tree_snapshot` builds identical bucket topology!
        cfg_to_save.abstraction_mode = match sess.abstraction {
            ResolvedAbstraction::Identity => "Identity".to_string(),
            ResolvedAbstraction::NutBlocker50 => "NutBlocker50".to_string(),
            ResolvedAbstraction::NutBlocker100 => "NutBlocker100".to_string(),
        };
        let cfg_bytes = cfg_to_save.to_json().to_json_string().into_bytes();
        let weight_bytes = sess.game.export_weights_binary();

        let mut container = Vec::with_capacity(16 + cfg_bytes.len() + weight_bytes.len());
        container.extend_from_slice(b"PLOTREE2");
        container.extend_from_slice(&(cfg_bytes.len() as u32).to_le_bytes());
        container.extend_from_slice(&cfg_bytes);
        container.extend_from_slice(&(weight_bytes.len() as u32).to_le_bytes());
        container.extend_from_slice(&weight_bytes);

        fs::write(&path, &container)
            .map_err(|e| format!("Failed to write .plotree snapshot {}: {e}", path.display()))?;

        Ok(Json::obj([
            ("path", Json::Str(path.display().to_string())),
            ("bytes", Json::Num(container.len() as f64)),
            ("bytes_written", Json::Num(container.len() as f64)),
            ("iterations", Json::Num(sess.game.iterations_run() as f64)),
        ]))
    }

    /// `#5 (Phase 3)` Loads a `.plotree` binary tree snapshot from disk in milliseconds without re-running DCFR!
    pub fn load_tree_snapshot(&self, file_path: &str) -> Result<Json, String> {
        let _exclusive =
            self.acquire_exclusive_op("Solver is currently running; click Stop first")?;
        let path = PathBuf::from(file_path.trim());
        let data = fs::read(&path)
            .map_err(|e| format!("Failed to read .plotree snapshot {}: {e}", path.display()))?;
        if data.len() >= 8 && &data[0..8] == b"PLOTREE1" {
            return Err(
                "Legacy PLOTREE1 snapshot is incompatible with v0.1.1+ (expected PLOTREE2 with config hash); please re-save the spot"
                    .to_string(),
            );
        }
        if data.len() < 16 || &data[0..8] != b"PLOTREE2" {
            return Err("Invalid .plotree file header (expected PLOTREE2)".to_string());
        }
        let cfg_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        if 12 + cfg_len + 4 > data.len() {
            return Err("Corrupted .plotree config segment".to_string());
        }
        let cfg_str = std::str::from_utf8(&data[12..12 + cfg_len])
            .map_err(|e| format!("Invalid UTF-8 in .plotree config: {e}"))?;
        let cfg_json = Json::parse(cfg_str)?;
        let cfg = SpotConfig::from_json(&cfg_json)?;

        let w_off = 12 + cfg_len;
        let w_len =
            u32::from_le_bytes([data[w_off], data[w_off + 1], data[w_off + 2], data[w_off + 3]])
                as usize;
        if w_off + 4 + w_len > data.len() {
            return Err("Corrupted .plotree weights segment".to_string());
        }
        let weight_bytes = &data[w_off + 4..w_off + 4 + w_len];

        let t0 = Instant::now();
        let mut restored = self.prepare_and_build_unlocked(&cfg)?;
        restored
            .game
            .import_weights_binary(weight_bytes)
            .map_err(|e| e.to_string())?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let expl = restored.game.exploitability_f64();
        let evs = restored.game.root_expected_values_f64();
        let hp = restored.hero_player;
        restored
            .convergence_log
            .push((restored.game.iterations_run(), expl, evs[hp]));
        restored.warnings.insert(
            0,
            format!(
                "Loaded solved .plotree snapshot ({} iters, {:.1} KiB) in {:.1} ms.",
                restored.game.iterations_run(),
                data.len() as f64 / 1024.0,
                elapsed_ms
            ),
        );

        let iters_restored = restored.game.iterations_run();
        let node_j = build_node_query_json(&mut restored);
        let cfg_out = cfg.to_json();
        *self.session.lock().unwrap() = Some(restored);

        Ok(Json::obj([
            ("path", Json::Str(path.display().to_string())),
            ("iterations", Json::Num(iters_restored as f64)),
            ("config", cfg_out),
            ("node", node_j),
            ("elapsed_ms", Json::Num(elapsed_ms)),
        ]))
    }

    pub fn query_node(&self) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        Ok(build_node_query_json(sess))
    }

    pub fn set_villain_filter(&self, filter: &str) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        sess.villain_filter = filter.trim().to_string();
        Ok(build_node_query_json(sess))
    }

    /// Locks the current decision node to fixed `action_probs` (`#2` Node Locking) and optionally runs
    /// `resolve_iterations` of DCFR immediately to compute maximally exploitative counter-strategies!
    pub fn lock_current_node(
        &self,
        action_probs: &[f32],
        resolve_iterations: u32,
    ) -> Result<Json, String> {
        let _exclusive =
            self.acquire_exclusive_op("Solver is currently running; click Stop first")?;
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        sess.game
            .lock_current_node_uniform(action_probs)
            .map_err(|e| e.to_string())?;
        if resolve_iterations > 0 {
            self.stop_flag.store(false, Ordering::SeqCst);
            let _ = sess.game.solve_with_hooks(
                resolve_iterations,
                0.0,
                resolve_iterations.max(1),
                &self.stop_flag,
                |_, _, _| {},
            );
        }
        Ok(build_node_query_json(sess))
    }

    /// Unlocks the current decision node (or all nodes when `all == true`) and optionally runs `resolve_iterations`.
    pub fn unlock_node(&self, all: bool, resolve_iterations: u32) -> Result<Json, String> {
        let _exclusive =
            self.acquire_exclusive_op("Solver is currently running; click Stop first")?;
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        if all {
            sess.game.unlock_all_nodes();
        } else {
            sess.game
                .unlock_current_node()
                .map_err(|e| e.to_string())?;
        }
        if resolve_iterations > 0 {
            self.stop_flag.store(false, Ordering::SeqCst);
            let _ = sess.game.solve_with_hooks(
                resolve_iterations,
                0.0,
                resolve_iterations.max(1),
                &self.stop_flag,
                |_, _, _| {},
            );
        }
        Ok(build_node_query_json(sess))
    }

    pub fn play_action(&self, action_idx: usize) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        let pre = sess.game.inspect_current_node();
        let actor_tag = match pre.node_type {
            "oop" => "OOP",
            "ip" => "IP",
            _ => "",
        };
        let act_name = pre
            .actions
            .get(action_idx)
            .map(|a| a.to_string())
            .unwrap_or_else(|| format!("Action#{action_idx}"));
        sess.game
            .play_action(action_idx)
            .map_err(|e| e.to_string())?;
        sess.history_labels.push(format!("{actor_tag} {act_name}"));
        Ok(build_node_query_json(sess))
    }

    pub fn play_chance(&self, card_str: &str) -> Result<Json, String> {
        let card: Card = card_str
            .trim()
            .parse()
            .map_err(|e| format!("Invalid turn/river card {card_str:?}: {e}"))?;
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        sess.game
            .play_chance_card(card)
            .map_err(|e| e.to_string())?;
        sess.history_labels.push(format!("Card {card}"));
        Ok(build_node_query_json(sess))
    }

    /// Rewinds the tree cursor to `depth` (`0 == Root`, `1 == after 1st step`, etc.).
    pub fn rewind_to(&self, depth: usize) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        let steps: Vec<plo_cfr::HistoryStep> = sess
            .game
            .history()
            .iter()
            .take(depth)
            .cloned()
            .collect();
        sess.game.back_to_root();
        sess.history_labels.truncate(steps.len());
        for st in steps {
            match st {
                plo_cfr::HistoryStep::Action(idx) => {
                    sess.game.play_action(idx).map_err(|e| e.to_string())?;
                }
                plo_cfr::HistoryStep::Deal(c) => {
                    sess.game.play_chance_card(c).map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(build_node_query_json(sess))
    }

    pub fn reset_cursor(&self) -> Result<Json, String> {
        let mut guard = self.session.lock().unwrap();
        let sess = guard
            .as_mut()
            .ok_or_else(|| "No solved game in memory yet".to_string())?;
        sess.game.back_to_root();
        sess.history_labels.clear();
        Ok(build_node_query_json(sess))
    }

    // ========================================================================
    // 6. Save / Load Config JSON
    // ========================================================================

    pub fn save_spot_config_to_file(&self, cfg: &SpotConfig, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        fs::write(path, cfg.to_json().to_json_pretty())
            .map_err(|e| format!("Failed to save config {}: {e}", path.display()))
    }

    pub fn load_spot_config_from_file(&self, path: &Path) -> Result<SpotConfig, String> {
        let raw = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
        let j = Json::parse(&raw)?;
        SpotConfig::from_json(&j)
    }

    // ========================================================================
    // Internal Range Resolution (Hero 1-combo vs Villain pack/override, or Lab 2-range)
    // ========================================================================

    fn resolve_ranges(
        &self,
        cfg: &SpotConfig,
        board: &Board,
    ) -> Result<(PloRange, PloRange, usize, usize, usize), String> {
        let hero_player = if cfg.hero_seat.trim().eq_ignore_ascii_case("IP") {
            PLAYER_IP
        } else {
            PLAYER_OOP
        };

        if cfg.lab_two_range_mode {
            let oop_r = cfg
                .lab_oop_range
                .parse::<PloRange>()
                .map_err(|e| format!("Invalid Lab OOP range: {e}"))?;
            let ip_r = cfg
                .lab_ip_range
                .parse::<PloRange>()
                .map_err(|e| format!("Invalid Lab IP range: {e}"))?;
            let live_oop = oop_r.without_dead(board.mask());
            let live_ip = ip_r.without_dead(board.mask());
            if live_oop.is_empty() || live_ip.is_empty() {
                return Err("Lab OOP or IP range has 0 live combos on this board".to_string());
            }
            let raw_v = if hero_player == PLAYER_OOP {
                ip_r.len()
            } else {
                oop_r.len()
            };
            let live_v = if hero_player == PLAYER_OOP {
                live_ip.len()
            } else {
                live_oop.len()
            };
            return Ok((live_oop, live_ip, hero_player, raw_v, live_v));
        }

        // Standard Home UX: Hero exact 4 cards (weight 1.0) vs Villain range
        let (hero_hand, hero_range) = parse_hero_hand(&cfg.hero_hand)?;
        if hero_hand.is_blocked_by(board.mask()) {
            return Err(format!(
                "Hero hand {hero_hand} overlaps with board {board}"
            ));
        }

        let raw_villain = if cfg.use_pack {
            let eff_spot = if cfg.villain_spot_id.trim().is_empty() {
                self.default_spot_for_seat(&cfg.hero_seat)
                    .unwrap_or_else(|| canonical_default_spot_for_seat(&cfg.hero_seat).to_string())
            } else {
                cfg.villain_spot_id.trim().to_string()
            };
            let raw = {
                let pack_guard = self.pack.lock().unwrap();
                let pack = pack_guard.as_ref().ok_or_else(|| {
                    "No range pack folder configured. Select your range pack folder in Settings, or uncheck 'Use pack' to enter a custom Villain range.".to_string()
                })?;
                pack.range_for_spot(&eff_spot)?.clone()
            };
            let _ = self.select_spot(&eff_spot);
            raw
        } else {
            cfg.villain_override_range
                .parse::<PloRange>()
                .map_err(|e| format!("Invalid Villain override range: {e}"))?
        };

        if raw_villain.is_empty() {
            return Err("Villain range is empty".to_string());
        }
        let raw_v_count = raw_villain.len();
        let dead = board.mask() | hero_hand.mask();
        let live_villain = raw_villain.without_dead(dead);
        if live_villain.is_empty() {
            return Err(format!(
                "All {raw_v_count} Villain combos are blocked by board {board} + Hero {hero_hand}"
            ));
        }
        let live_v_count = live_villain.len();

        if hero_player == PLAYER_OOP {
            Ok((hero_range, live_villain, PLAYER_OOP, raw_v_count, live_v_count))
        } else {
            Ok((live_villain, hero_range, PLAYER_IP, raw_v_count, live_v_count))
        }
    }
}

fn format_memory_budget_exceeded(prep: &PreparedSpot) -> String {
    let est_mib = prep.mem_report.total_estimated_mib_val();
    let bud_mib = prep.budget_bytes as f64 / (1024.0 * 1024.0);
    let hint = match prep.chosen_abstraction {
        ResolvedAbstraction::Identity => {
            "Hint: switch Abstraction to NutBlocker k=50 (or Auto), pick a tighter Villain spot, solve on Turn/River, or raise your RAM budget in Settings."
        }
        _ => {
            "Hint: pick a tighter Villain spot (fewer live combos), simplify bet/raise sizes, solve on Turn/River, or raise your RAM budget in Settings."
        }
    };
    format!(
        "MemoryBudgetExceeded: estimated {:.1} MiB ({}) exceeds budget {:.0} MiB ({}) across {} decision nodes. {}",
        est_mib,
        format_mib_human(est_mib),
        bud_mib,
        format_mib_human(bud_mib),
        prep.mem_report.decision_nodes,
        hint
    )
}

trait MemoryReportExt {
    fn total_estimated_mib_val(&self) -> f64;
}

impl MemoryReportExt for MemoryReport {
    fn total_estimated_mib_val(&self) -> f64 {
        self.total_estimated_bytes as f64 / (1024.0 * 1024.0)
    }
}

fn resolve_solve_streets(
    full_board: &Board,
    mode_raw: &str,
) -> Result<(Board, [bool; 3], String, Option<String>), String> {
    let key = mode_raw.trim().to_ascii_lowercase();
    match key.as_str() {
        "" | "all" | "full" => {
            let lbl = match full_board.street() {
                Street::Flop => "Flop + Turn + River",
                Street::Turn => "Turn + River",
                Street::River => "River",
            };
            Ok((*full_board, [true, true, true], lbl.to_string(), None))
        }
        "floponly" | "flop_only" | "flop" => {
            let flop_b = Board::flop(full_board.flop_cards()).map_err(|e| e.to_string())?;
            let note = if full_board.len() > 3 {
                format!(
                    "Flop Only mode: using flop {flop_b} (trimmed from {full_board}); Turn & River check down to exact 2+3 showdown."
                )
            } else {
                "Flop Only mode: Flop betting solved with Turn & River checked down to exact 2+3 showdown.".to_string()
            };
            Ok((
                flop_b,
                [true, false, false],
                "Flop Only".to_string(),
                Some(note),
            ))
        }
        "turnonly" | "turn_only" | "turn" => {
            let turn_cards = full_board.turn_cards().ok_or_else(|| {
                format!(
                    "Turn Only requires at least 4 board cards (Flop + Turn); current board '{full_board}' has only 3 cards."
                )
            })?;
            let turn_b = Board::turn(turn_cards).map_err(|e| e.to_string())?;
            let note = if full_board.len() > 4 {
                format!(
                    "Turn Only mode: using 4-card turn {turn_b} (trimmed from {full_board}); River checks down to exact 2+3 showdown."
                )
            } else {
                "Turn Only mode: Turn betting solved with River checked down to exact 2+3 showdown.".to_string()
            };
            Ok((
                turn_b,
                [false, true, false],
                "Turn Only".to_string(),
                Some(note),
            ))
        }
        "riveronly" | "river_only" | "river" => {
            if full_board.len() < 5 {
                return Err(format!(
                    "River Only requires 5 board cards (Flop + Turn + River); current board '{full_board}' has {} cards.",
                    full_board.len()
                ));
            }
            Ok((
                *full_board,
                [false, false, true],
                "River Only".to_string(),
                None,
            ))
        }
        "flopturn" | "flop_turn" => {
            let flop_b = Board::flop(full_board.flop_cards()).map_err(|e| e.to_string())?;
            let note = if full_board.len() > 3 {
                format!(
                    "Flop + Turn mode: using flop {flop_b} (trimmed from {full_board}); River checks down to exact 2+3 showdown."
                )
            } else {
                "Flop + Turn mode: Flop & Turn betting solved with River checked down to exact 2+3 showdown.".to_string()
            };
            Ok((
                flop_b,
                [true, true, false],
                "Flop + Turn".to_string(),
                Some(note),
            ))
        }
        other => Err(format!(
            "Unknown solve_streets mode {other:?}; expected All, FlopOnly, TurnOnly, RiverOnly, or FlopTurn"
        )),
    }
}

fn build_player_street_sizes(
    label: &str,
    bets_override: &str,
    raises_override: &str,
    donks_override: Option<&str>,
    fallback_bets: &str,
    fallback_raises: &str,
) -> Result<StreetBetSizes, String> {
    let eff_bets = if bets_override.trim().is_empty() {
        fallback_bets
    } else {
        bets_override
    };
    let eff_raises = if raises_override.trim().is_empty() {
        fallback_raises
    } else {
        raises_override
    };
    let bet_fractions = parse_fraction_list(eff_bets)
        .map_err(|e| format!("Invalid {label} bet sizes ({eff_bets:?}): {e}"))?;
    let raise_fractions = parse_fraction_list(eff_raises)
        .map_err(|e| format!("Invalid {label} raise sizes ({eff_raises:?}): {e}"))?;
    let donk_fractions = match donks_override.map(str::trim).filter(|s| !s.is_empty()) {
        Some(d_str) => Some(
            parse_fraction_list(d_str)
                .map_err(|e| format!("Invalid {label} donk sizes ({d_str:?}): {e}"))?,
        ),
        None => None,
    };
    Ok(StreetBetSizes {
        bet_fractions,
        raise_fractions,
        donk_fractions,
    })
}

fn default_iters_for_street(street: Street) -> u32 {
    match street {
        Street::River => 1000,
        Street::Turn => 500,
        Street::Flop => 200,
    }
}

fn build_range_preview_json(spot_id: &str, raw: &PloRange, live: &PloRange) -> Json {
    // Sort preview by highest weight, then ComboId, showing up to 20 combos (never dumping thousands!)
    let mut sample_entries = live.entries().to_vec();
    sample_entries.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.combo_id.cmp(&b.combo_id))
    });
    let sample_json: Vec<Json> = sample_entries
        .iter()
        .take(20)
        .map(|e| {
            Json::obj([
                ("hand", Json::Str(e.hand.to_string())),
                ("weight", Json::Num(e.weight as f64)),
            ])
        })
        .collect();

    Json::obj([
        ("spot_id", Json::Str(spot_id.to_string())),
        ("raw_combos", Json::Num(raw.len() as f64)),
        ("live_combos", Json::Num(live.len() as f64)),
        ("blocked_combos", Json::Num(raw.len().saturating_sub(live.len()) as f64)),
        ("live_weight_sum", Json::Num(live.total_weight())),
        ("sample_top_20", Json::Arr(sample_json)),
    ])
}

fn build_node_query_json(sess: &mut ActiveGameSession) -> Json {
    let report = sess.game.inspect_current_node();
    let reaches = sess.game.current_combo_reaches();
    let root_evs = sess.game.root_expected_values_f64();
    let expl = sess.game.exploitability_f64();
    let start_pot = sess.game.tree_config().starting_pot as f64;
    let is_locked = sess.game.is_current_node_locked();
    let locked_node_count = sess.game.locked_node_count();
    let (rake_frac, rake_cap) = sess.game.rake_config();
    let hp = sess.hero_player;
    let vp = hp ^ 1;

    let hero_combos = sess.game.combos(hp).to_vec();
    let villain_combos = sess.game.combos(vp).to_vec();

    // Current node Hero EV & reach
    let hero_evs_at_node = if hp == PLAYER_OOP {
        &report.ev_oop
    } else {
        &report.ev_ip
    };
    let villain_evs_at_node = if vp == PLAYER_OOP {
        &report.ev_oop
    } else {
        &report.ev_ip
    };

    let hero_node_ev = weighted_mean_ev(hero_evs_at_node, &reaches[hp]);
    let villain_node_ev = weighted_mean_ev(villain_evs_at_node, &reaches[vp]);

    let acting_opt = match report.node_type {
        "oop" => Some(PLAYER_OOP),
        "ip" => Some(PLAYER_IP),
        _ => None,
    };

    let actions_json: Vec<Json> = report
        .actions
        .iter()
        .enumerate()
        .map(|(idx, act)| {
            Json::obj([
                ("index", Json::Num(idx as f64)),
                ("label", Json::Str(act.to_string())),
                ("solver_label", Json::Str(act.to_solver_label())),
            ])
        })
        .collect();

    // Pre-analyze live Villain combos (`reach > 1e-9`) on `report.board` using exact PLO 2+3 rules
    // and evaluate `sess.villain_filter` (PPT/Monker-style filter query).
    let filter_query = sess.villain_filter.trim();
    let has_filter = !filter_query.is_empty();
    let mut v_features: Vec<Option<HandBoardFeatures>> = vec![None; villain_combos.len()];
    let mut v_matches_filter: Vec<bool> = vec![false; villain_combos.len()];
    let mut total_villain_reach = 0.0f64;
    let mut filtered_villain_reach = 0.0f64;
    let mut filtered_villain_combos = 0usize;

    for (i, vc) in villain_combos.iter().enumerate() {
        let r = reaches[vp][i] as f64;
        if r > 1e-9 {
            total_villain_reach += r;
            let feat = analyze_plo_hand(vc.hand, &report.board);
            let ok = if has_filter {
                matches_hand_filter(vc.hand, &feat, filter_query)
            } else {
                true
            };
            v_features[i] = Some(feat);
            v_matches_filter[i] = ok;
            if ok {
                filtered_villain_reach += r;
                filtered_villain_combos += 1;
            }
        }
    }

    // Build PLO-4 Hand & Draw Category Breakdown (`villain_hand_classes`) across filtered live Villain combos
    let villain_acts = acting_opt == Some(vp);
    let n_v_all = villain_combos.len();
    let n_actions = report.actions.len();
    let mut class_weights = [0.0f64; 15];
    let mut class_ev_num = [0.0f64; 15];
    let mut class_counts = [0usize; 15];
    let mut class_sample: [Option<usize>; 15] = [None; 15];
    let mut class_act_num = vec![vec![0.0f64; n_actions]; 15];

    for (i, feat_opt) in v_features.iter().enumerate() {
        if !v_matches_filter[i] {
            continue;
        }
        if let Some(feat) = feat_opt {
            let cid = feat.primary_class as usize;
            let r = reaches[vp][i] as f64;
            class_weights[cid] += r;
            class_ev_num[cid] += r * (villain_evs_at_node.get(i).copied().unwrap_or(0.0) as f64);
            class_counts[cid] += 1;
            if match class_sample[cid] {
                None => true,
                Some(prev) => reaches[vp][i] > reaches[vp][prev],
            } {
                class_sample[cid] = Some(i);
            }
            if villain_acts {
                for a in 0..n_actions {
                    class_act_num[cid][a] += r * (report.strategy[a * n_v_all + i] as f64);
                }
            }
        }
    }

    let denom_share = if filtered_villain_reach > 1e-12 {
        filtered_villain_reach
    } else {
        1.0
    };
    let mut villain_hand_classes_json = Vec::new();
    for cls in ALL_PLO_HAND_CLASSES {
        let cid = cls as usize;
        if class_counts[cid] == 0 || class_weights[cid] <= 1e-9 {
            continue;
        }
        let w = class_weights[cid];
        let share_pct = (w / denom_share) * 100.0;
        let avg_ev = class_ev_num[cid] / w;
        let sample_str = class_sample[cid]
            .map(|idx| villain_combos[idx].hand.to_string())
            .unwrap_or_default();
        let probs: Vec<Json> = if villain_acts {
            (0..n_actions)
                .map(|a| Json::Num(class_act_num[cid][a] / w))
                .collect()
        } else {
            Vec::new()
        };
        villain_hand_classes_json.push(Json::obj([
            ("class_id", Json::Str(cls.id().to_string())),
            ("label", Json::Str(cls.label().to_string())),
            ("short_tag", Json::Str(cls.short_tag().to_string())),
            ("combo_count", Json::Num(class_counts[cid] as f64)),
            ("weight", Json::Num(w)),
            ("share_pct", Json::Num(share_pct)),
            ("ev", Json::Num(avg_ev)),
            ("sample_hand", Json::Str(sample_str)),
            ("action_probs", Json::Arr(probs)),
        ]));
    }

    // Hero hand PLO-4 class tag on current board (if singleton Hero hand is set)
    let hero_hand_class_str = sess
        .hero_hand
        .and_then(|hh| {
            if hh.is_blocked_by(report.board.mask()) {
                None
            } else {
                Some(analyze_plo_hand(hh, &report.board).primary_class.label().to_string())
            }
        });

    // Hero action strategy at current node (if Hero is the acting player)
    let mut hero_strategy_json = Vec::new();
    // Villain aggregate + Strategy Detail (up to 30 rows + remaining count)
    let mut villain_aggregate_json = Vec::new();
    let mut villain_buckets_json = Vec::new();
    let mut villain_detail_mode = "combos";
    let mut villain_detail_rows = Vec::new();
    let mut villain_detail_more: usize = 0;

    if let Some(acting) = acting_opt {
        let n_combos = sess.game.combos(acting).len();
        let reach_acting = &reaches[acting];
        let reach_sum: f64 = reach_acting.iter().map(|&r| r as f64).sum();

        let agg_probs: Vec<f64> = (0..report.actions.len())
            .map(|a| {
                if reach_sum <= 1e-12 {
                    let s: f64 = (0..n_combos)
                        .map(|i| report.strategy[a * n_combos + i] as f64)
                        .sum();
                    s / (n_combos.max(1) as f64)
                } else {
                    let s: f64 = (0..n_combos)
                        .map(|i| (report.strategy[a * n_combos + i] as f64) * (reach_acting[i] as f64))
                        .sum();
                    s / reach_sum
                }
            })
            .collect();

        if acting == hp {
            // Compute exact counterfactual EV for each Hero action (`#6` Drill Mode EV-loss grading!)
            let saved_hist_len = sess.game.history().len();
            let mut action_evs = Vec::with_capacity(report.actions.len());
            for a in 0..report.actions.len() {
                let mut a_ev = hero_node_ev;
                if sess.game.play_action(a).is_ok() {
                    let child_rep = sess.game.inspect_current_node();
                    let child_reaches = sess.game.current_combo_reaches();
                    let ch_evs = if hp == PLAYER_OOP {
                        &child_rep.ev_oop
                    } else {
                        &child_rep.ev_ip
                    };
                    // Use current-node hero reach so unplayed (0%-freq) actions still grade their exact counterfactual EV!
                    a_ev = weighted_mean_ev(ch_evs, &reaches[hp]);
                    if !a_ev.is_finite() {
                        a_ev = weighted_mean_ev(ch_evs, &child_reaches[hp]);
                    }
                    // Restore cursor back to `saved_hist_len`
                    let steps: Vec<plo_cfr::HistoryStep> = sess
                        .game
                        .history()
                        .iter()
                        .take(saved_hist_len)
                        .cloned()
                        .collect();
                    sess.game.back_to_root();
                    for st in steps {
                        match st {
                            plo_cfr::HistoryStep::Action(idx) => {
                                let _ = sess.game.play_action(idx);
                            }
                            plo_cfr::HistoryStep::Deal(c) => {
                                let _ = sess.game.play_chance_card(c);
                            }
                        }
                    }
                }
                action_evs.push(a_ev);
            }
            let best_ev = action_evs
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            for (a, act) in report.actions.iter().enumerate() {
                // For 1-combo Hero, `n_combos == 1`, so `agg_probs[a]` is Hero's exact action probability!
                let prob = if hero_combos.len() == 1 {
                    report.strategy[a] as f64
                } else {
                    agg_probs[a]
                };
                let ev_val = action_evs.get(a).copied().unwrap_or(hero_node_ev);
                let ev_loss = if best_ev.is_finite() {
                    (best_ev - ev_val).max(0.0)
                } else {
                    0.0
                };
                hero_strategy_json.push(Json::obj([
                    ("action_index", Json::Num(a as f64)),
                    ("action", Json::Str(act.to_string())),
                    ("probability", Json::Num(prob)),
                    ("ev", Json::Num(ev_val)),
                    ("ev_loss", Json::Num(ev_loss)),
                ]));
            }
            // When Hero acts, show Villain's top reaching live combos matching `villain_filter` (capped at 30 rows)
            let mut active_v: Vec<usize> = (0..villain_combos.len())
                .filter(|&i| reaches[vp][i] > 1e-9 && v_matches_filter[i])
                .collect();
            active_v.sort_by(|&a, &b| {
                reaches[vp][b]
                    .partial_cmp(&reaches[vp][a])
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then_with(|| villain_combos[a].combo_id.cmp(&villain_combos[b].combo_id))
            });
            villain_detail_mode = "combos_reach";
            villain_detail_more = active_v.len().saturating_sub(30);
            for i in active_v.into_iter().take(30) {
                let tag = v_features[i]
                    .map(|f| f.primary_class.short_tag())
                    .unwrap_or("");
                villain_detail_rows.push(Json::obj([
                    ("id", Json::Str(villain_combos[i].hand.to_string())),
                    ("sample_hand", Json::Str(villain_combos[i].hand.to_string())),
                    ("hand_class", Json::Str(tag.to_string())),
                    ("weight", Json::Num(reaches[vp][i] as f64)),
                    ("ev", Json::Num(villain_evs_at_node.get(i).copied().unwrap_or(0.0) as f64)),
                    ("action_probs", Json::Arr(Vec::new())),
                ]));
            }
        } else {
            // If filter is active on Villain's turn, compute aggregate over filtered Villain combos!
            let (eff_agg_probs, _) = if has_filter {
                let mut f_sum = 0.0f64;
                for i in 0..n_combos {
                    if v_matches_filter[i] {
                        f_sum += reach_acting[i] as f64;
                    }
                }
                let probs: Vec<f64> = (0..report.actions.len())
                    .map(|a| {
                        if f_sum <= 1e-12 {
                            0.0
                        } else {
                            let num: f64 = (0..n_combos)
                                .filter(|&i| v_matches_filter[i])
                                .map(|i| {
                                    (report.strategy[a * n_combos + i] as f64)
                                        * (reach_acting[i] as f64)
                                })
                                .sum();
                            num / f_sum
                        }
                    })
                    .collect();
                (probs, f_sum)
            } else {
                (agg_probs.clone(), reach_sum)
            };

            for (a, act) in report.actions.iter().enumerate() {
                villain_aggregate_json.push(Json::obj([
                    ("action_index", Json::Num(a as f64)),
                    ("action", Json::Str(act.to_string())),
                    ("probability", Json::Num(eff_agg_probs[a])),
                ]));
            }
            // Use per-combo rows whenever Identity <= 500 OR when a user hand filter is active!
            let use_combo_rows = has_filter
                || (sess.abstraction == ResolvedAbstraction::Identity && n_combos <= 500);
            if use_combo_rows {
                villain_detail_mode = "combos";
                let mut active_c: Vec<usize> = (0..n_combos)
                    .filter(|&i| reach_acting[i] > 1e-9 && v_matches_filter[i])
                    .collect();
                active_c.sort_by(|&a, &b| {
                    reach_acting[b]
                        .partial_cmp(&reach_acting[a])
                        .unwrap_or(core::cmp::Ordering::Equal)
                        .then_with(|| villain_combos[a].combo_id.cmp(&villain_combos[b].combo_id))
                });
                villain_detail_more = active_c.len().saturating_sub(30);
                for i in active_c.into_iter().take(30) {
                    let probs: Vec<Json> = (0..report.actions.len())
                        .map(|a| Json::Num(report.strategy[a * n_combos + i] as f64))
                        .collect();
                    let tag = v_features[i]
                        .map(|f| f.primary_class.short_tag())
                        .unwrap_or("");
                    let row = Json::obj([
                        ("bucket_id", Json::Num(i as f64)),
                        ("id", Json::Str(villain_combos[i].hand.to_string())),
                        ("sample_hand", Json::Str(villain_combos[i].hand.to_string())),
                        ("hand_class", Json::Str(tag.to_string())),
                        ("weight", Json::Num(reach_acting[i] as f64)),
                        ("reach_weight", Json::Num(reach_acting[i] as f64)),
                        ("ev", Json::Num(villain_evs_at_node.get(i).copied().unwrap_or(0.0) as f64)),
                        ("action_probs", Json::Arr(probs)),
                    ]);
                    villain_buckets_json.push(row.clone());
                    villain_detail_rows.push(row);
                }
            } else if let Some((_, num_buckets, c2b)) = sess.game.current_node_bucket_assignment() {
                villain_detail_mode = "buckets";
                let mut bucket_weights = vec![0.0f64; num_buckets];
                let mut bucket_ev_num = vec![0.0f64; num_buckets];
                let mut bucket_rep_combo = vec![None::<usize>; num_buckets];
                let mut bucket_combo_counts = vec![0usize; num_buckets];
                for (i, &b_u32) in c2b.iter().enumerate() {
                    let b = b_u32 as usize;
                    if b < num_buckets && v_matches_filter[i] {
                        let r = reach_acting[i] as f64;
                        if r > 1e-9 {
                            bucket_weights[b] += r;
                            bucket_ev_num[b] += r * (villain_evs_at_node.get(i).copied().unwrap_or(0.0) as f64);
                            bucket_combo_counts[b] += 1;
                            if bucket_rep_combo[b].is_none() {
                                bucket_rep_combo[b] = Some(i);
                            }
                        }
                    }
                }
                // Cap at 30 rows + N more
                let mut active_b: Vec<usize> = (0..num_buckets)
                    .filter(|&b| bucket_weights[b] > 1e-9)
                    .collect();
                active_b.sort_by(|&a, &b| {
                    bucket_weights[b]
                        .partial_cmp(&bucket_weights[a])
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                villain_detail_more = active_b.len().saturating_sub(30);
                for b in active_b.into_iter().take(30) {
                    if let Some(rep_i) = bucket_rep_combo[b] {
                        let probs: Vec<Json> = (0..report.actions.len())
                            .map(|a| Json::Num(report.strategy[a * n_combos + rep_i] as f64))
                            .collect();
                        let b_ev = if bucket_weights[b] > 1e-12 {
                            bucket_ev_num[b] / bucket_weights[b]
                        } else {
                            0.0
                        };
                        let tag = v_features[rep_i]
                            .map(|f| f.primary_class.short_tag())
                            .unwrap_or("");
                        let row = Json::obj([
                            ("bucket_id", Json::Num(b as f64)),
                            ("id", Json::Str(format!("#{b}"))),
                            ("sample_hand", Json::Str(villain_combos[rep_i].hand.to_string())),
                            ("hand_class", Json::Str(tag.to_string())),
                            ("combo_count", Json::Num(bucket_combo_counts[b] as f64)),
                            ("weight", Json::Num(bucket_weights[b])),
                            ("reach_weight", Json::Num(bucket_weights[b])),
                            ("ev", Json::Num(b_ev)),
                            ("action_probs", Json::Arr(probs)),
                        ]);
                        villain_buckets_json.push(row.clone());
                        villain_detail_rows.push(row);
                    }
                }
            }
        }
    } else {
        // Chance or terminal node: still show Villain's top 30 reaching combos matching `villain_filter`
        let mut active_v: Vec<usize> = (0..villain_combos.len())
            .filter(|&i| reaches[vp][i] > 1e-9 && v_matches_filter[i])
            .collect();
        active_v.sort_by(|&a, &b| {
            reaches[vp][b]
                .partial_cmp(&reaches[vp][a])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| villain_combos[a].combo_id.cmp(&villain_combos[b].combo_id))
        });
        villain_detail_mode = "combos_reach";
        villain_detail_more = active_v.len().saturating_sub(30);
        for i in active_v.into_iter().take(30) {
            let tag = v_features[i]
                .map(|f| f.primary_class.short_tag())
                .unwrap_or("");
            villain_detail_rows.push(Json::obj([
                ("id", Json::Str(villain_combos[i].hand.to_string())),
                ("sample_hand", Json::Str(villain_combos[i].hand.to_string())),
                ("hand_class", Json::Str(tag.to_string())),
                ("weight", Json::Num(reaches[vp][i] as f64)),
                ("ev", Json::Num(villain_evs_at_node.get(i).copied().unwrap_or(0.0) as f64)),
                ("action_probs", Json::Arr(Vec::new())),
            ]));
        }
    }

    let legal_chance: Vec<Json> = sess
        .game
        .all_legal_chance_cards()
        .into_iter()
        .map(|c| Json::Str(c.to_string()))
        .collect();
    let rep_chance: Vec<Json> = sess
        .game
        .available_chance_cards()
        .iter()
        .map(|c| Json::Str(c.to_string()))
        .collect();

    // `#3` Turn / River `4×13` Runout EV Heatmap at Chance nodes
    let chance_runouts_json: Vec<Json> = if report.node_type == "chance" {
        let hero_combos_snapshot: Vec<_> = sess.game.combos(hp).to_vec();
        let raw_runouts = sess.game.inspect_chance_runouts();
        raw_runouts
            .into_iter()
            .map(|ro| {
                let cmask = ro.card.mask();
                let hero_blocked = hero_combos_snapshot.len() == 1
                    && hero_combos_snapshot[0].mask.intersects(cmask);
                let hero_ev_vec = if hp == PLAYER_OOP {
                    &ro.ev_oop
                } else {
                    &ro.ev_ip
                };
                let hero_ev = if hero_blocked {
                    f64::NAN
                } else if hero_combos_snapshot.len() == 1 {
                    hero_ev_vec.first().copied().unwrap_or(0.0) as f64
                } else if hp == PLAYER_OOP {
                    ro.range_ev_oop as f64
                } else {
                    ro.range_ev_ip as f64
                };
                let vill_ev = if vp == PLAYER_OOP {
                    ro.range_ev_oop as f64
                } else {
                    ro.range_ev_ip as f64
                };
                let next_acts: Vec<Json> = ro
                    .next_actions
                    .iter()
                    .zip(ro.next_agg_freqs.iter())
                    .map(|(act, &p)| {
                        Json::obj([
                            ("action", Json::Str(act.to_string())),
                            ("probability", Json::Num(p as f64)),
                        ])
                    })
                    .collect();
                Json::obj([
                    ("card", Json::Str(ro.card.to_string())),
                    ("hero_blocked", Json::Bool(hero_blocked)),
                    (
                        "hero_ev",
                        if hero_ev.is_finite() {
                            Json::Num(hero_ev)
                        } else {
                            Json::Null
                        },
                    ),
                    (
                        "hero_ev_pct_pot",
                        if hero_ev.is_finite() && report.pot > 0 {
                            Json::Num((hero_ev / (report.pot as f64)) * 100.0)
                        } else {
                            Json::Null
                        },
                    ),
                    ("villain_ev", Json::Num(vill_ev)),
                    ("next_node_type", Json::Str(ro.next_node_type.to_string())),
                    ("next_actions", Json::Arr(next_acts)),
                ])
            })
            .collect()
    } else {
        Vec::new()
    };

    let history_labels: Vec<Json> = sess
        .game
        .history()
        .iter()
        .enumerate()
        .map(|(i, h)| {
            if let Some(lbl) = sess.history_labels.get(i) {
                Json::Str(lbl.clone())
            } else {
                match h {
                    plo_cfr::HistoryStep::Action(idx) => Json::Str(format!("Action#{idx}")),
                    plo_cfr::HistoryStep::Deal(c) => Json::Str(format!("Card {c}")),
                }
            }
        })
        .collect();

    // `#4a` 52-Card Blocker Impact Matrix on Villain's Filtered Reaching Range (`Quads..Set` = Value, `ComboDraw..Wrap` = Strong Draws, `Air/Blocker` = Air)
    let mut blocker_impact_json = Vec::new();
    let bmask_cur = report.board.mask();
    let hero_mask_cur = sess.hero_hand.map(|h| h.mask()).unwrap_or(CardMask::EMPTY);
    if filtered_villain_reach > 1e-9 {
        let mut card_tot_w = [0.0f64; 52];
        let mut card_val_w = [0.0f64; 52];
        let mut card_draw_w = [0.0f64; 52];
        let mut card_air_w = [0.0f64; 52];
        let mut tot_val_all = 0.0f64;
        let mut tot_air_all = 0.0f64;

        for (i, vc) in villain_combos.iter().enumerate() {
            if !v_matches_filter[i] {
                continue;
            }
            let r = reaches[vp][i] as f64;
            if r <= 1e-9 {
                continue;
            }
            let cls = v_features[i]
                .map(|f| f.primary_class)
                .unwrap_or(PloHandClass::Air);
            let is_val = matches!(
                cls,
                PloHandClass::FullHousePlus
                    | PloHandClass::Flush
                    | PloHandClass::Straight
                    | PloHandClass::SetTrips
                    | PloHandClass::TwoPair
            );
            let is_draw = matches!(
                cls,
                PloHandClass::ComboDraw
                    | PloHandClass::NutFlushDraw
                    | PloHandClass::FlushDraw
                    | PloHandClass::WrapDraw
            );
            let is_air = matches!(cls, PloHandClass::NutBlockerAir | PloHandClass::Air);
            if is_val {
                tot_val_all += r;
            }
            if is_air {
                tot_air_all += r;
            }
            for c in vc.hand.cards() {
                let cid = c.id() as usize;
                card_tot_w[cid] += r;
                if is_val {
                    card_val_w[cid] += r;
                }
                if is_draw {
                    card_draw_w[cid] += r;
                }
                if is_air {
                    card_air_w[cid] += r;
                }
            }
        }

        for cid in 0..52u8 {
            let c = Card::new(cid).unwrap();
            let on_board = bmask_cur.contains(c);
            let in_hero = hero_mask_cur.contains(c);
            let idx = cid as usize;
            let blocked_total_pct = (card_tot_w[idx] / filtered_villain_reach) * 100.0;
            let blocked_value_pct = if tot_val_all > 1e-9 {
                (card_val_w[idx] / tot_val_all) * 100.0
            } else {
                0.0
            };
            let blocked_air_pct = if tot_air_all > 1e-9 {
                (card_air_w[idx] / tot_air_all) * 100.0
            } else {
                0.0
            };
            let val_share_within_card = if card_tot_w[idx] > 1e-9 {
                (card_val_w[idx] / card_tot_w[idx]) * 100.0
            } else {
                0.0
            };
            let draw_share_within_card = if card_tot_w[idx] > 1e-9 {
                (card_draw_w[idx] / card_tot_w[idx]) * 100.0
            } else {
                0.0
            };
            blocker_impact_json.push(Json::obj([
                ("card", Json::Str(c.to_string())),
                ("on_board", Json::Bool(on_board)),
                ("in_hero", Json::Bool(in_hero)),
                ("blocked_total_pct", Json::Num(blocked_total_pct)),
                ("blocked_value_pct", Json::Num(blocked_value_pct)),
                ("blocked_air_pct", Json::Num(blocked_air_pct)),
                ("value_concentration_pct", Json::Num(val_share_within_card)),
                ("draw_concentration_pct", Json::Num(draw_share_within_card)),
            ]));
        }
    }

    // `#4b` Villain Equity / EV Quintiles Distribution (`5 × 20%` decile/quintile bins from Weakest 0–20% to Strongest 80–100%)
    let mut equity_buckets_json = Vec::new();
    if filtered_villain_reach > 1e-9 {
        // Rank filtered Villain combos by exact node EV (or current 2+3 hand rank tie-breaker) ascending
        let mut ranked_v: Vec<(usize, f64, u16)> = (0..villain_combos.len())
            .filter(|&i| v_matches_filter[i] && reaches[vp][i] > 1e-9)
            .map(|i| {
                let ev = villain_evs_at_node.get(i).copied().unwrap_or(0.0) as f64;
                let hr = plo_eval::eval_plo(villain_combos[i].hand, &report.board)
                    .map(|r| r.raw())
                    .unwrap_or(0);
                (i, ev, hr)
            })
            .collect();
        ranked_v.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });

        let bin_labels = [
            "0–20% (Air / Bottom)",
            "20–40% (Weak / Low Eq)",
            "40–60% (Medium / Bluffcatch)",
            "60–80% (Strong / Draws+)",
            "80–100% (Top / Nuts)",
        ];
        let mut bin_w = [0.0f64; 5];
        let mut bin_ev_sum = [0.0f64; 5];
        let mut bin_cnt = [0usize; 5];
        let mut bin_act_sum = vec![vec![0.0f64; n_actions]; 5];
        let mut cum_w = 0.0f64;

        for (i, ev, _hr) in ranked_v {
            let r = reaches[vp][i] as f64;
            let mid_frac = (cum_w + 0.5 * r) / filtered_villain_reach;
            cum_w += r;
            let b_idx = ((mid_frac * 5.0).floor() as usize).min(4);
            bin_w[b_idx] += r;
            bin_ev_sum[b_idx] += r * ev;
            bin_cnt[b_idx] += 1;
            if villain_acts {
                for a in 0..n_actions {
                    bin_act_sum[b_idx][a] += r * (report.strategy[a * n_v_all + i] as f64);
                }
            }
        }

        for b_idx in (0..5).rev() {
            let w = bin_w[b_idx];
            let share = (w / filtered_villain_reach) * 100.0;
            let avg_ev = if w > 1e-12 { bin_ev_sum[b_idx] / w } else { 0.0 };
            let eq_pct_pot = if report.pot > 0 && w > 1e-12 {
                ((avg_ev / (report.pot as f64)) * 100.0).clamp(-100.0, 200.0)
            } else {
                0.0
            };
            let probs: Vec<Json> = if villain_acts && w > 1e-12 {
                (0..n_actions)
                    .map(|a| Json::Num(bin_act_sum[b_idx][a] / w))
                    .collect()
            } else {
                Vec::new()
            };
            equity_buckets_json.push(Json::obj([
                ("bucket_index", Json::Num(b_idx as f64)),
                ("label", Json::Str(bin_labels[b_idx].to_string())),
                ("share_pct", Json::Num(share)),
                ("combo_count", Json::Num(bin_cnt[b_idx] as f64)),
                ("avg_ev", Json::Num(avg_ev)),
                ("ev_pct_pot", Json::Num(eq_pct_pot)),
                ("action_probs", Json::Arr(probs)),
            ]));
        }
    }

    // `#7` Interactive Street Tree Minimap
    let (street_root_depth, raw_minimap) = sess.game.current_street_tree_minimap();
    let tree_minimap_json: Vec<Json> = raw_minimap
        .into_iter()
        .map(
            |(nid, parent_opt, path_vec, edge_lbl, ntype, npot, is_cur, is_on_path)| {
                Json::obj([
                    ("node_id", Json::Num(nid as f64)),
                    (
                        "parent_id",
                        parent_opt.map(|p| Json::Num(p as f64)).unwrap_or(Json::Null),
                    ),
                    (
                        "action_path",
                        Json::Arr(path_vec.into_iter().map(|idx| Json::Num(idx as f64)).collect()),
                    ),
                    ("edge_label", Json::Str(edge_lbl)),
                    ("node_type", Json::Str(ntype)),
                    ("pot", Json::Num(npot as f64)),
                    ("is_current", Json::Bool(is_cur)),
                    ("is_on_path", Json::Bool(is_on_path)),
                ])
            },
        )
        .collect();
    // `#4 (Phase 3)` Live Convergence History (`(iters, dEV, dEV % pot, hero_ev)`)
    let convergence_json: Vec<Json> = sess
        .convergence_log
        .iter()
        .map(|&(it, ex, hev)| {
            Json::obj([
                ("iteration", Json::Num(it as f64)),
                ("iterations", Json::Num(it as f64)),
                ("exploitability", Json::Num(ex)),
                (
                    "dev_pct_pot",
                    Json::Num(if start_pot > 0.0 {
                        (ex / start_pot) * 100.0
                    } else {
                        0.0
                    }),
                ),
                ("hero_ev", Json::Num(hev)),
            ])
        })
        .collect();

    // `#6 (Phase 3)` Hero Suit & Nut-Out Matrix (`Nut Outs` vs `2nd-Nut` vs `Dominated / Non-Nut Improving Outs` on Flop/Turn)
    let hero_nut_outs_json = if let Some(hh) = sess.hero_hand {
        compute_hero_nut_out_matrix(hh, &report.board, &villain_combos, &reaches[vp])
    } else {
        Json::Null
    };

    // `#2 (Phase 3)` If `bot_hand` is active (`Play vs Solver` mode) AND we reached a `terminal` node, reveal showdown!
    let bot_showdown_json = if report.node_type == "terminal" {
        if let (Some(hh), Some(bh)) = (sess.hero_hand, sess.bot_hand) {
            let h_class = analyze_plo_hand(hh, &report.board).primary_class.label();
            let b_class = analyze_plo_hand(bh, &report.board).primary_class.label();
            // A terminal is a showdown iff both players matched committed chips (`committed[0] == committed[1]`)!
            let went_to_sd = report.committed[0] == report.committed[1]
                && report.board.street() == Street::River;
            let outcome = if went_to_sd {
                match (plo_eval::eval_plo(hh, &report.board), plo_eval::eval_plo(bh, &report.board)) {
                    (Ok(hr), Ok(br)) => {
                        if hr > br {
                            "Hero Wins Showdown"
                        } else if hr < br {
                            "Solver Bot Wins Showdown"
                        } else {
                            "Split Pot (Chop)"
                        }
                    }
                    _ => "Hand Ended",
                }
            } else {
                "Fold before Showdown"
            };
            Json::obj([
                ("bot_hand", Json::Str(bh.to_string())),
                ("bot_class", Json::Str(b_class.to_string())),
                ("bot_category", Json::Str(b_class.to_string())),
                ("hero_category", Json::Str(h_class.to_string())),
                ("went_to_showdown", Json::Bool(went_to_sd)),
                ("winner", Json::Str(outcome.to_string())),
                ("outcome", Json::Str(outcome.to_string())),
            ])
        } else {
            Json::Null
        }
    } else {
        Json::Null
    };
    let can_subgame_resolve = matches!(report.board.street(), Street::Turn | Street::River)
        && report.committed[0] == report.committed[1]
        && report.node_type == "oop"
        && (sess.game.tree_config().effective_stack - report.committed[0]) > 0;
    let in_subgame = sess.parent_session.is_some();

    Json::obj([
        ("board", Json::Str(report.board.to_string())),
        ("street", Json::Str(format!("{:?}", report.board.street()))),
        ("pot", Json::Num(report.pot as f64)),
        (
            "committed",
            Json::Arr(vec![
                Json::Num(report.committed[0] as f64),
                Json::Num(report.committed[1] as f64),
            ]),
        ),
        ("node_type", Json::Str(report.node_type.to_string())),
        (
            "hero_seat",
            Json::Str(if hp == PLAYER_OOP { "OOP" } else { "IP" }.to_string()),
        ),
        (
            "hero_hand",
            sess.hero_hand
                .map(|h| Json::Str(h.to_string()))
                .unwrap_or(Json::Null),
        ),
        (
            "hero_hand_class",
            hero_hand_class_str.map(Json::Str).unwrap_or(Json::Null),
        ),
        ("hero_acts_here", Json::Bool(acting_opt == Some(hp))),
        ("iterations", Json::Num(sess.game.iterations_run() as f64)),
        ("exploitability", Json::Num(expl)),
        ("exploitability_sci", Json::Str(format!("{expl:.4e}"))),
        ("root_ev_hero", Json::Num(root_evs[hp])),
        ("root_ev_villain", Json::Num(root_evs[vp])),
        (
            "root_ev_hero_pct_pot",
            Json::Num(if start_pot > 0.0 {
                (root_evs[hp] / start_pot) * 100.0
            } else {
                0.0
            }),
        ),
        ("node_ev_hero", Json::Num(hero_node_ev)),
        ("node_ev_villain", Json::Num(villain_node_ev)),
        ("hero_reach", Json::Num(reaches[hp].iter().map(|&r| r as f64).sum())),
        ("villain_reach", Json::Num(total_villain_reach)),
        ("villain_filter", Json::Str(filter_query.to_string())),
        ("filtered_villain_reach", Json::Num(filtered_villain_reach)),
        ("filtered_villain_combos", Json::Num(filtered_villain_combos as f64)),
        ("is_locked", Json::Bool(is_locked)),
        ("locked_node_count", Json::Num(locked_node_count as f64)),
        ("rake_pct", Json::Num(rake_frac * 100.0)),
        ("rake_cap", Json::Num(rake_cap)),
        ("native_threads", Json::Num(native_solver_threads() as f64)),
        ("abstraction", Json::Str(sess.abstraction.as_label().to_string())),
        ("actions", Json::Arr(actions_json)),
        ("hero_strategy", Json::Arr(hero_strategy_json)),
        ("villain_aggregate_strategy", Json::Arr(villain_aggregate_json)),
        ("villain_hand_classes", Json::Arr(villain_hand_classes_json)),
        ("villain_bucket_mix", Json::Arr(villain_buckets_json)),
        ("villain_detail_mode", Json::Str(villain_detail_mode.to_string())),
        ("villain_detail_rows", Json::Arr(villain_detail_rows)),
        ("villain_detail_more", Json::Num(villain_detail_more as f64)),
        ("legal_chance_cards", Json::Arr(legal_chance)),
        ("representative_chance_cards", Json::Arr(rep_chance)),
        ("chance_runout_stats", Json::Arr(chance_runouts_json)),
        ("blocker_impact", Json::Arr(blocker_impact_json)),
        ("equity_buckets", Json::Arr(equity_buckets_json)),
        ("street_root_depth", Json::Num(street_root_depth as f64)),
        ("tree_minimap", Json::Arr(tree_minimap_json)),
        ("can_subgame_resolve", Json::Bool(can_subgame_resolve)),
        ("in_subgame", Json::Bool(in_subgame)),
        ("convergence_history", Json::Arr(convergence_json)),
        ("hero_nut_outs", hero_nut_outs_json),
        ("bot_active", Json::Bool(sess.bot_hand.is_some())),
        ("bot_showdown", bot_showdown_json),
        ("history", Json::Arr(history_labels)),
        (
            "warnings",
            Json::Arr(sess.warnings.iter().cloned().map(Json::Str).collect()),
        ),
        ("footer", Json::Str(MANDATORY_FOOTER_TEXT.to_string())),
    ])
}

fn weighted_mean_ev(evs: &[f32], reaches: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&ev, &r) in evs.iter().zip(reaches.iter()) {
        if r > 0.0 {
            num += (ev as f64) * (r as f64);
            den += r as f64;
        }
    }
    if den > 1e-12 {
        num / den
    } else if evs.len() == 1 {
        evs[0] as f64
    } else {
        0.0
    }
}

/// Unified JSON-RPC style dispatcher used identically by Tauri 2 IPC commands and the local preview server.
pub fn dispatch_command(state: &Arc<DesktopState>, cmd: &str, args: &Json) -> Result<Json, String> {
    match cmd {
        "get_state" | "list_spots" => Ok(state.list_spots()),
        "set_pack_folder" => {
            let folder = args
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'path' string".to_string())?;
            state.set_pack_folder(folder)?;
            Ok(state.list_spots())
        }
        "bind_spot" => {
            let spot_id = args
                .get("spot_id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'spot_id'".to_string())?;
            let source_key = args
                .get("source_key")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            state.bind_spot(spot_id, source_key)?;
            Ok(state.list_spots())
        }
        "default_spot_for_seat" => {
            let hero_seat = args
                .get("hero_seat")
                .and_then(|v| v.as_str())
                .unwrap_or("OOP");
            let canon = canonical_default_spot_for_seat(hero_seat);
            let resolved = state.default_spot_for_seat(hero_seat);
            Ok(Json::obj([
                ("hero_seat", Json::Str(hero_seat.to_string())),
                ("canonical_spot_id", Json::Str(canon.to_string())),
                (
                    "spot_id",
                    resolved.map(Json::Str).unwrap_or(Json::Null),
                ),
            ]))
        }
        "select_spot" => {
            let spot_id = args
                .get("spot_id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'spot_id'".to_string())?;
            state.select_spot(spot_id)?;
            Ok(Json::obj([("last_spot_id", Json::Str(spot_id.to_string()))]))
        }
        "set_memory_budget" => {
            let mib = if let Some(s) = args.get("budget").and_then(|v| v.as_str()) {
                parse_memory_budget_input(s)?
            } else if let Some(s) = args.get("memory_budget_mib").and_then(|v| v.as_str()) {
                parse_memory_budget_input(s)?
            } else if let Some(n) = args
                .get("memory_budget_mib")
                .or_else(|| args.get("budget"))
                .and_then(|v| v.as_u32())
            {
                n
            } else {
                return Err("missing 'memory_budget_mib' or 'budget'".to_string());
            };
            let saved = state.set_memory_budget(mib)?;
            Ok(Json::obj([
                ("memory_budget_mib", Json::Num(saved as f64)),
                (
                    "memory_budget_human",
                    Json::Str(format_mib_human(saved as f64)),
                ),
            ]))
        }
        "villain_from_spot" => {
            let spot_id = args
                .get("spot_id")
                .and_then(|v| v.as_str())
                .unwrap_or("RFI_BTN");
            let board = args.get("board").and_then(|v| v.as_str()).unwrap_or("");
            let hero = args.get("hero_hand").and_then(|v| v.as_str()).unwrap_or("");
            let (_, _, preview) = state.villain_from_spot(spot_id, board, hero)?;
            Ok(preview)
        }
        "preview_villain" => {
            let use_pack = args.get("use_pack").and_then(|v| v.as_bool()).unwrap_or(true);
            let spot_id = args
                .get("spot_id")
                .and_then(|v| v.as_str())
                .unwrap_or("RFI_BTN");
            let ov = args
                .get("villain_override_range")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let board = args.get("board").and_then(|v| v.as_str()).unwrap_or("");
            let hero = args.get("hero_hand").and_then(|v| v.as_str()).unwrap_or("");
            state.preview_villain_range(use_pack, spot_id, ov, board, hero)
        }
        "parse_hero" => {
            let h_str = args
                .get("hero_hand")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let (hand, _) = parse_hero_hand(h_str)?;
            Ok(Json::obj([("hand", Json::Str(hand.to_string()))]))
        }
        "parse_board" => {
            let b_str = args.get("board").and_then(|v| v.as_str()).unwrap_or("");
            let board = parse_postflop_board(b_str)?;
            Ok(Json::obj([
                ("board", Json::Str(board.to_string())),
                ("street", Json::Str(format!("{:?}", board.street()))),
            ]))
        }
        "compute_equity" => {
            let cfg = SpotConfig::from_json(args)?;
            state.compute_equity(&cfg)
        }
        "estimate_spot" => {
            let cfg = SpotConfig::from_json(args)?;
            state.estimate_spot(&cfg)
        }
        "build_game" => {
            let cfg = SpotConfig::from_json(args)?;
            state.build_game(&cfg)
        }
        "stop_solve" => {
            state.stop_solve();
            Ok(Json::obj([("stopped", Json::Bool(true))]))
        }
        "latest_progress" => {
            let p = state
                .latest_progress()
                .map(|ev| ev.to_json())
                .unwrap_or(Json::Null);
            Ok(Json::obj([
                ("solving", Json::Bool(state.is_solving())),
                ("progress", p),
            ]))
        }
        "step_iterations" => {
            let n = args
                .get("iterations")
                .and_then(|v| v.as_u32())
                .unwrap_or(100);
            state.step_iterations(n)
        }
        "query_node" => state.query_node(),
        "set_villain_filter" => {
            let f = args
                .get("filter")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            state.set_villain_filter(f)
        }
        "lock_node" => {
            let probs_arr = args
                .get("action_probs")
                .and_then(|v| match v {
                    Json::Arr(a) => Some(a),
                    _ => None,
                })
                .ok_or_else(|| "missing 'action_probs' array".to_string())?;
            let probs: Vec<f32> = probs_arr
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect();
            let iters = args
                .get("resolve_iterations")
                .and_then(|v| v.as_u32())
                .unwrap_or(200);
            state.lock_current_node(&probs, iters)
        }
        "unlock_node" => {
            let all = args
                .get("all")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let iters = args
                .get("resolve_iterations")
                .and_then(|v| v.as_u32())
                .unwrap_or(200);
            state.unlock_node(all, iters)
        }
        "play_action" => {
            let idx = args
                .get("action_index")
                .and_then(|v| v.as_usize())
                .ok_or_else(|| "missing 'action_index'".to_string())?;
            state.play_action(idx)
        }
        "play_chance" => {
            let card = args
                .get("card")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'card'".to_string())?;
            state.play_chance(card)
        }
        "rewind_to" => {
            let depth = args
                .get("depth")
                .and_then(|v| v.as_usize())
                .unwrap_or(0);
            state.rewind_to(depth)
        }
        "reset_cursor" => state.reset_cursor(),
        "resolve_subgame" => {
            let max_iters = args
                .get("max_iterations")
                .and_then(|v| v.as_u32())
                .unwrap_or(500);
            let target_dev_pct = args
                .get("target_dev_pct")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.2);
            state.resolve_subgame(max_iters, target_dev_pct)
        }
        "exit_subgame" => state.exit_subgame(),
        "lock_node_by_classes" => {
            let obj = args
                .get("class_locks")
                .and_then(|v| v.as_obj())
                .ok_or_else(|| "missing 'class_locks' object".to_string())?;
            let mut map = std::collections::BTreeMap::new();
            for (k, v) in obj {
                if let Some(arr) = v.as_arr() {
                    let probs: Vec<f32> = arr
                        .iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect();
                    map.insert(k.clone(), probs);
                }
            }
            let iters = args
                .get("resolve_iterations")
                .and_then(|v| v.as_u32())
                .unwrap_or(200);
            state.lock_node_by_classes(&map, iters)
        }
        "jump_street_path" => {
            let arr = args
                .get("action_path")
                .and_then(|v| v.as_arr())
                .ok_or_else(|| "missing 'action_path' array".to_string())?;
            let path: Vec<usize> = arr.iter().filter_map(|x| x.as_usize()).collect();
            state.jump_street_path(&path)
        }
        "list_saved_spots" => Ok(state.list_saved_spots()),
        "save_spot_bookmark" => {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let folder = args
                .get("folder")
                .and_then(|v| v.as_str())
                .unwrap_or("General");
            let cfg_src = args.get("config").unwrap_or(args);
            let cfg = SpotConfig::from_json(cfg_src)?;
            state.save_spot_bookmark(name, folder, cfg)
        }
        "delete_saved_spot" => {
            let spot_id = args
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'id'".to_string())?;
            state.delete_saved_spot(spot_id)
        }
        "export_node_report" => state.export_node_report(),
        "parse_hand_history" => {
            let raw = args
                .get("text")
                .or_else(|| args.get("hh_text"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'text' string".to_string())?;
            state.parse_hand_history(raw)
        }
        "run_flop_subset_report" => {
            let cfg_src = args.get("config").unwrap_or(args);
            let cfg = SpotConfig::from_json(cfg_src)?;
            let flops: Vec<String> = args
                .get("flops")
                .and_then(|v| v.as_arr())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let iters = args
                .get("iterations_per_flop")
                .and_then(|v| v.as_u32())
                .unwrap_or(100);
            state.run_flop_subset_report(&cfg, &flops, iters)
        }
        "audit_hand_history" => {
            let raw = args
                .get("text")
                .or_else(|| args.get("hh_text"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'text' string".to_string())?;
            let v_ov = args
                .get("villain_override_range")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let iters = args
                .get("iterations_per_street")
                .and_then(|v| v.as_u32())
                .unwrap_or(100);
            state.audit_hand_history(raw, v_ov, iters)
        }
        "start_bot_hand" => {
            let seed = args
                .get("seed")
                .and_then(|v| v.as_f64())
                .map(|x| x as u64)
                .unwrap_or_else(default_time_seed);
            state.start_bot_hand(seed)
        }
        "bot_play_hero_action" => {
            let idx = args
                .get("action_index")
                .and_then(|v| v.as_usize())
                .ok_or_else(|| "missing 'action_index'".to_string())?;
            let seed = args
                .get("seed")
                .and_then(|v| v.as_f64())
                .map(|x| x as u64)
                .unwrap_or_else(default_time_seed);
            state.bot_play_hero_action(idx, seed)
        }
        "stop_bot_hand" => state.stop_bot_hand(),
        "compare_bet_simplifications" => {
            let cfg_src = args.get("config").unwrap_or(args);
            let cfg = SpotConfig::from_json(cfg_src)?;
            let sizes: Vec<String> = args
                .get("candidate_sizes")
                .and_then(|v| v.as_arr())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let iters = args
                .get("iterations")
                .and_then(|v| v.as_u32())
                .unwrap_or(120);
            state.compare_bet_simplifications(&cfg, &sizes, iters)
        }
        "save_tree_snapshot" => {
            let p = args
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'path' string".to_string())?;
            state.save_tree_snapshot(p)
        }
        "load_tree_snapshot" => {
            let p = args
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'path' string".to_string())?;
            state.load_tree_snapshot(p)
        }
        "export_config" => {
            let cfg = SpotConfig::from_json(args)?;
            Ok(cfg.to_json())
        }
        "import_config" => {
            let raw = args
                .get("json_text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'json_text'".to_string())?;
            let parsed = Json::parse(raw)?;
            let cfg = SpotConfig::from_json(&parsed)?;
            Ok(cfg.to_json())
        }
        other => Err(format!("unknown desktop command: {other:?}")),
    }
}

fn default_time_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
}

fn mix_seed(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let out = z ^ (z >> 31);
    if out == 0 { 0xCAFE_BABE_1234_5678 } else { out }
}

fn next_unit_f64(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = if x == 0 { 1 } else { x };
    ((*state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn advance_bot_and_chance_until_hero_or_end(
    sess: &mut ActiveGameSession,
    rng_state: &mut u64,
) -> Result<Vec<String>, String> {
    let hp = sess.hero_player;
    let vp = hp ^ 1;
    let hero_mask = sess.hero_hand.map(|h| h.mask()).unwrap_or(CardMask::EMPTY);
    let bot_hand = sess
        .bot_hand
        .ok_or_else(|| "Bot hand missing".to_string())?;
    let bot_mask = bot_hand.mask();
    let v_combos = sess.game.combos(vp).to_vec();
    let bot_combo_idx = v_combos.iter().position(|c| c.hand == bot_hand);

    let mut events = Vec::new();
    for _safety in 0..32 {
        let rep = sess.game.inspect_current_node();
        match rep.node_type {
            "terminal" => break,
            "oop" if hp == PLAYER_OOP => break,
            "ip" if hp == PLAYER_IP => break,
            "oop" | "ip" => {
                // Solver Bot acts! Sample action from `bot_hand`'s exact solved strategy (or Villain aggregate fallback)
                let n_acts = rep.actions.len();
                let n_v = v_combos.len();
                let mut probs = vec![0.0f64; n_acts];
                let mut p_sum = 0.0f64;
                if let Some(bi) = bot_combo_idx {
                    for (a, slot) in probs.iter_mut().enumerate().take(n_acts) {
                        let p = (rep.strategy[a * n_v + bi] as f64).max(0.0);
                        *slot = p;
                        p_sum += p;
                    }
                }
                if p_sum <= 1e-9 {
                    let reaches = sess.game.current_combo_reaches();
                    for (a, slot) in probs.iter_mut().enumerate().take(n_acts) {
                        let num: f64 = (0..n_v)
                            .map(|i| (rep.strategy[a * n_v + i] as f64) * (reaches[vp][i] as f64))
                            .sum();
                        *slot = num.max(0.0);
                        p_sum += *slot;
                    }
                }
                let chosen_a = if p_sum <= 1e-9 {
                    0usize
                } else {
                    let mut r = next_unit_f64(rng_state) * p_sum;
                    let mut pick = n_acts - 1;
                    for (a, &w) in probs.iter().enumerate() {
                        if r <= w {
                            pick = a;
                            break;
                        }
                        r -= w;
                    }
                    pick
                };
                let actor_tag = if rep.node_type == "oop" { "OOP" } else { "IP" };
                let act_lbl = rep.actions[chosen_a].to_string();
                sess.game
                    .play_action(chosen_a)
                    .map_err(|e| e.to_string())?;
                let lbl = format!("{actor_tag} {act_lbl}");
                sess.history_labels.push(lbl.clone());
                events.push(format!("Bot ({actor_tag}): {act_lbl}"));
            }
            "chance" => {
                // Deal a uniformly random live card disjoint from `board | hero_mask | bot_mask`!
                let live_cards: Vec<Card> = sess
                    .game
                    .all_legal_chance_cards()
                    .into_iter()
                    .filter(|c| !hero_mask.contains(*c) && !bot_mask.contains(*c))
                    .collect();
                if live_cards.is_empty() {
                    return Err("No live deck cards remaining at Chance node".to_string());
                }
                let idx = ((next_unit_f64(rng_state) * (live_cards.len() as f64)).floor() as usize)
                    .min(live_cards.len() - 1);
                let card = live_cards[idx];
                sess.game
                    .play_chance_card(card)
                    .map_err(|e| e.to_string())?;
                sess.history_labels.push(format!("Card {card}"));
                events.push(format!("Dealt Card: {card}"));
            }
            _ => break,
        }
    }
    Ok(events)
}

fn match_hh_action_to_tree_index(kind: &str, amount_chips: i32, node_acts: &[Json]) -> usize {
    if node_acts.is_empty() {
        return 0;
    }
    let parsed_acts: Vec<(usize, String, Option<i32>)> = node_acts
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let lbl = a
                .get("label")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let amt = extract_int_in_parens(&lbl);
            (i, lbl.to_ascii_lowercase(), amt)
        })
        .collect();

    match kind {
        "fold" => {
            if let Some((i, _, _)) = parsed_acts.iter().find(|x| x.1.starts_with("fold")) {
                return *i;
            }
            if let Some((i, _, _)) = parsed_acts.iter().find(|x| x.1.starts_with("check")) {
                return *i;
            }
            0
        }
        "check" | "call" => {
            if let Some((i, _, _)) = parsed_acts
                .iter()
                .find(|x| x.1.starts_with("check") || x.1.starts_with("call"))
            {
                return *i;
            }
            0
        }
        "bet" | "raise" => {
            let aggr: Vec<_> = parsed_acts
                .iter()
                .filter(|x| {
                    x.1.starts_with("bet") || x.1.starts_with("raise") || x.1.starts_with("allin")
                })
                .collect();
            if aggr.is_empty() {
                return parsed_acts.last().map(|x| x.0).unwrap_or(0);
            }
            aggr.into_iter()
                .min_by_key(|x| (x.2.unwrap_or(amount_chips) - amount_chips).abs())
                .map(|x| x.0)
                .unwrap_or(0)
        }
        _ => 0,
    }
}

fn extract_int_in_parens(s: &str) -> Option<i32> {
    let open = s.find('(')?;
    let close = s[open + 1..].find(')')?;
    s[open + 1..open + 1 + close].trim().parse::<i32>().ok()
}

fn apply_street_bet_size_to_cfg(cfg: &mut SpotConfig, street: Street, sizes_str: &str) {
    match street {
        Street::Flop => {
            cfg.flop_oop_bets = sizes_str.to_string();
            cfg.flop_ip_bets = sizes_str.to_string();
        }
        Street::Turn => {
            cfg.turn_oop_bets = sizes_str.to_string();
            cfg.turn_ip_bets = sizes_str.to_string();
        }
        Street::River => {
            cfg.river_oop_bets = sizes_str.to_string();
            cfg.river_ip_bets = sizes_str.to_string();
        }
    }
}

/// `#6 (Phase 3)` Computes Hero's Suit Structure (`Double-Suited` / `Single-Suited` / `Rainbow`) and exact
/// `Nut Outs` (`1st Nut` vs all live reaching Villain combos), `2nd-Nut Outs`, and `Dominated / Non-Nut Outs` on Flop/Turn!
fn compute_hero_nut_out_matrix(
    hero_hand: Hand4,
    board: &Board,
    villain_combos: &[plo_range::WeightedHand],
    v_reaches: &[f32],
) -> Json {
    let h_cards = hero_hand.cards();
    let mut suit_counts = [0u8; 4];
    for c in h_cards {
        suit_counts[c.suit() as usize] += 1;
    }
    let pairs_of_suits = suit_counts.iter().filter(|&&cnt| cnt == 2).count();
    let max_suit = *suit_counts.iter().max().unwrap_or(&1);
    let suit_structure = if pairs_of_suits == 2 {
        "Double-Suited ($ds)"
    } else if max_suit == 2 {
        "Single-Suited ($ss)"
    } else if max_suit == 3 {
        "Triple-Suited (3-suit)"
    } else if max_suit == 4 {
        "Monotone (4-suit)"
    } else {
        "Rainbow ($r)"
    };

    let cur_hr = plo_eval::eval_plo(hero_hand, board).ok();
    let cur_cat = cur_hr.map(|r| format!("{:?}", r.category())).unwrap_or_default();

    if board.street() == Street::River {
        // On River, compute Hero's exact rank percentile vs Villain's live reaching combos
        let h_raw = cur_hr.map(|r| r.raw()).unwrap_or(0);
        let mut better_v_reach = 0.0f64;
        let mut tot_v_reach = 0.0f64;
        for (vc, &r) in villain_combos.iter().zip(v_reaches.iter()) {
            if r > 1e-9 && !vc.mask.intersects(hero_hand.mask()) && !vc.mask.intersects(board.mask()) {
                tot_v_reach += r as f64;
                if let Ok(vr) = plo_eval::eval_plo(vc.hand, board) {
                    if vr.raw() > h_raw {
                        better_v_reach += r as f64;
                    }
                }
            }
        }
        let is_effective_nuts = tot_v_reach > 0.0 && better_v_reach <= 1e-9;
        return Json::obj([
            ("suit_structure", Json::Str(suit_structure.to_string())),
            ("current_category", Json::Str(cur_cat)),
            ("street", Json::Str("River".to_string())),
            ("is_river_nuts_vs_villain", Json::Bool(is_effective_nuts)),
            ("nut_outs", Json::Arr(Vec::new())),
            ("second_nut_outs", Json::Arr(Vec::new())),
            ("non_nut_outs", Json::Arr(Vec::new())),
            ("nut_out_count", Json::Num(0.0)),
            ("second_nut_out_count", Json::Num(0.0)),
            ("non_nut_out_count", Json::Num(0.0)),
        ]);
    }

    // Flop or Turn: test all legal remaining deck cards `c` disjoint from `board | hero_hand`
    let dead = board.mask() | hero_hand.mask();
    let cur_cat_enum = cur_hr.map(|r| r.category());
    let mut nut_outs = Vec::new();
    let mut second_nut_outs = Vec::new();
    let mut non_nut_outs = Vec::new();

    // Active Villain combos with `reach > 1e-9` disjoint from `hero_hand`
    let active_v: Vec<Hand4> = villain_combos
        .iter()
        .zip(v_reaches.iter())
        .filter(|(vc, &r)| r > 1e-9 && !vc.mask.intersects(hero_hand.mask()))
        .map(|(vc, _)| vc.hand)
        .collect();

    for cid in 0..52u8 {
        let card = Card::new(cid).unwrap();
        if dead.contains(card) {
            continue;
        }
        let Ok(next_board) = board.deal(card) else {
            continue;
        };
        let Ok(h_next) = plo_eval::eval_plo(hero_hand, &next_board) else {
            continue;
        };
        let next_cat = h_next.category();
        // Consider `card` a real PLO improving out when Hero reaches Straight+ OR strictly upgrades category to Set+
        let improves_to_strong = next_cat >= plo_eval::HandCategory::Straight
            || (next_cat >= plo_eval::HandCategory::ThreeOfAKind
                && Some(next_cat) > cur_cat_enum);
        if !improves_to_strong {
            continue;
        }

        // Count how many distinct stronger `HandRank`s exist in Villain's live reaching range on `next_board`
        let cmask = card.mask();
        let mut stronger_ranks: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        for &vh in &active_v {
            if vh.mask().intersects(cmask) {
                continue;
            }
            if let Ok(vr) = plo_eval::eval_plo(vh, &next_board) {
                if vr.raw() > h_next.raw() {
                    stronger_ranks.insert(vr.raw());
                    if stronger_ranks.len() >= 2 {
                        break;
                    }
                }
            }
        }

        let item = Json::obj([
            ("card", Json::Str(card.to_string())),
            ("category", Json::Str(format!("{next_cat:?}"))),
        ]);
        match stronger_ranks.len() {
            0 => nut_outs.push(item),
            1 => second_nut_outs.push(item),
            _ => non_nut_outs.push(item),
        }
    }

    let n_nut = nut_outs.len();
    let n_sec = second_nut_outs.len();
    let n_non = non_nut_outs.len();
    Json::obj([
        ("suit_structure", Json::Str(suit_structure.to_string())),
        ("current_category", Json::Str(cur_cat)),
        ("street", Json::Str(format!("{:?}", board.street()))),
        ("is_river_nuts_vs_villain", Json::Bool(false)),
        ("nut_out_count", Json::Num(n_nut as f64)),
        ("second_nut_out_count", Json::Num(n_sec as f64)),
        ("non_nut_out_count", Json::Num(n_non as f64)),
        ("nut_outs", Json::Arr(nut_outs)),
        ("second_nut_outs", Json::Arr(second_nut_outs)),
        ("non_nut_outs", Json::Arr(non_nut_outs)),
    ])
}

/// Classifies a 3-card Flop into a concise human-readable PLO board texture tag (`#5`).
fn classify_flop_texture(board: &Board) -> String {
    let cards = board.flop_cards();
    let mut suits = [0u8; 4];
    let mut ranks = [cards[0].rank(), cards[1].rank(), cards[2].rank()];
    ranks.sort_unstable_by(|a, b| b.cmp(a));
    for c in cards {
        suits[c.suit() as usize] += 1;
    }
    let max_suit = *suits.iter().max().unwrap_or(&1);
    let suit_tag = match max_suit {
        3 => "Monotone",
        2 => "Two-Tone",
        _ => "Rainbow",
    };
    let paired = ranks[0] == ranks[1] || ranks[1] == ranks[2];
    let span = ranks[0].saturating_sub(ranks[2]);
    let conn_tag = if paired {
        "Paired"
    } else if span <= 4 {
        "Connected"
    } else if ranks[0] == 12 {
        "A-High"
    } else if ranks[0] >= 9 {
        "Broadway"
    } else {
        "Dry"
    };
    format!("{conn_tag} {suit_tag}")
}
