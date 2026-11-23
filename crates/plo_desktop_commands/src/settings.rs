//! OS application-data settings persistence (`pack_folder`, `spot_bindings`, `last_spot_id`,
//! `memory_budget_mib`) and portable `SpotConfig` JSON export/import.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::formats::Json;

/// Fallback desktop budget (`8 GiB`) when OS total RAM cannot be queried.
pub const FALLBACK_DESKTOP_BUDGET_MIB: u32 = 8192;
/// Minimum allowed user-entered desktop budget (`16 MiB`, so small unit-test guards like `128 MiB` work).
pub const MIN_DESKTOP_BUDGET_MIB: u32 = 16;
/// Maximum allowed user-entered desktop budget (`1 TiB` = `1,048,576 MiB` — no `512 MiB` WASM cap).
pub const MAX_DESKTOP_BUDGET_MIB: u32 = 1_048_576;

/// Pure desktop RAM budget policy given detected system RAM (`Some(total_mib)` or `None`):
/// - If `total_mib` is `None`: returns `8192 MiB` (`8 GiB`).
/// - Otherwise: `min(50% of system RAM, 16384 MiB)`, with a `4096 MiB` (`4 GiB`) floor whenever
///   system RAM `>= 8192 MiB` (`8 GiB`), else `50%` of system RAM.
pub fn compute_default_desktop_budget_mib(detected_system_ram_mib: Option<u64>) -> u32 {
    let Some(ram_mib) = detected_system_ram_mib.filter(|&m| m > 0) else {
        return FALLBACK_DESKTOP_BUDGET_MIB;
    };
    let half = ram_mib / 2;
    let capped = half.min(16_384);
    let chosen = if ram_mib >= 8_192 {
        capped.max(4_096)
    } else {
        capped.max(256)
    };
    chosen as u32
}

/// Queries physical host RAM in MiB without external crates (supports Linux `/proc/meminfo`,
/// Windows `GlobalMemoryStatusEx`, macOS `sysctl hw.memsize`, plus `PLO_DESKTOP_DETECTED_RAM_MIB`
/// test override).
pub fn detect_system_ram_mib() -> Option<u64> {
    if let Ok(ov) = env::var("PLO_DESKTOP_DETECTED_RAM_MIB") {
        if let Ok(mib) = ov.trim().parse::<u64>() {
            if mib > 0 {
                return Some(mib);
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let kb_str = rest
                        .split_whitespace()
                        .next()
                        .unwrap_or("");
                    if let Ok(kb) = kb_str.parse::<u64>() {
                        if kb > 0 {
                            return Some(kb / 1024);
                        }
                    }
                }
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            dw_length: u32,
            dw_memory_load: u32,
            ull_total_phys: u64,
            ull_avail_phys: u64,
            ull_total_page_file: u64,
            ull_avail_page_file: u64,
            ull_total_virtual: u64,
            ull_avail_virtual: u64,
            ull_avail_extended_virtual: u64,
        }
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(lpBuffer: *mut MemoryStatusEx) -> i32;
        }
        let mut st = MemoryStatusEx {
            dw_length: core::mem::size_of::<MemoryStatusEx>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };
        // SAFETY: `dw_length` is initialized to the exact struct size as required by Win32 ABI.
        let ok = unsafe { GlobalMemoryStatusEx(&mut st) };
        if ok != 0 && st.ull_total_phys > 0 {
            return Some(st.ull_total_phys / (1024 * 1024));
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            if out.status.success() {
                if let Ok(bytes) = String::from_utf8_lossy(&out.stdout).trim().parse::<u64>() {
                    if bytes > 0 {
                        return Some(bytes / (1024 * 1024));
                    }
                }
            }
        }
    }
    None
}

/// Returns the default desktop memory budget in MiB for the current machine.
pub fn default_desktop_budget_mib() -> u32 {
    compute_default_desktop_budget_mib(detect_system_ram_mib())
}

/// Parses a flexible user memory budget string such as `"16384"`, `"16 GiB"`, `"8GB"`, `"4096 MiB"` into MiB.
pub fn parse_memory_budget_input(raw: &str) -> Result<u32, String> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() {
        return Err("Memory budget cannot be empty (e.g. 8192 or 16 GiB)".to_string());
    }
    let (num_part, mult_mib) = if let Some(n) = s
        .strip_suffix("gib")
        .or_else(|| s.strip_suffix("gb"))
        .or_else(|| s.strip_suffix('g'))
    {
        (n.trim(), 1024.0)
    } else if let Some(n) = s
        .strip_suffix("mib")
        .or_else(|| s.strip_suffix("mb"))
        .or_else(|| s.strip_suffix('m'))
    {
        (n.trim(), 1.0)
    } else {
        (s.as_str(), 1.0)
    };
    let val: f64 = num_part
        .parse()
        .map_err(|_| format!("Invalid memory budget {raw:?}; enter MiB (e.g. 8192) or GiB (e.g. 16 GiB)"))?;
    if !val.is_finite() || val <= 0.0 {
        return Err(format!("Memory budget must be positive, got {raw:?}"));
    }
    let mib = (val * mult_mib).round() as u64;
    Ok(mib.clamp(MIN_DESKTOP_BUDGET_MIB as u64, MAX_DESKTOP_BUDGET_MIB as u64) as u32)
}

/// Formats a MiB value cleanly for desktop UI display (`"16 GiB"`, `"8.5 GiB"`, `"307 MiB"`).
pub fn format_mib_human(mib: f64) -> String {
    if mib >= 1024.0 {
        let gib = mib / 1024.0;
        if (gib - gib.round()).abs() < 0.05 {
            format!("{:.0} GiB", gib.round())
        } else {
            format!("{gib:.1} GiB")
        }
    } else if mib >= 10.0 {
        format!("{mib:.0} MiB")
    } else {
        format!("{mib:.1} MiB")
    }
}

/// Saved spot bookmark (`#2` Save / Load Spot Library) persisted inside `AppSettings`.
#[derive(Clone, Debug, PartialEq)]
pub struct SavedSpotEntry {
    pub id: String,
    pub name: String,
    pub folder: String,
    pub created_at: String,
    pub config: SpotConfig,
    pub summary_hero_ev: Option<f64>,
    pub summary_expl: Option<f64>,
}

impl SavedSpotEntry {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("id", Json::Str(self.id.clone())),
            ("name", Json::Str(self.name.clone())),
            ("folder", Json::Str(self.folder.clone())),
            ("created_at", Json::Str(self.created_at.clone())),
            ("config", self.config.to_json()),
            (
                "summary_hero_ev",
                self.summary_hero_ev.map(Json::Num).unwrap_or(Json::Null),
            ),
            (
                "summary_expl",
                self.summary_expl.map(Json::Num).unwrap_or(Json::Null),
            ),
        ])
    }

    pub fn from_json(j: &Json) -> Result<Self, String> {
        let id = j
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| "saved spot missing 'id'".to_string())?
            .to_string();
        let name = j
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(&id)
            .to_string();
        let folder = j
            .get("folder")
            .and_then(|v| v.as_str())
            .unwrap_or("General")
            .to_string();
        let created_at = j
            .get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let cfg_j = j
            .get("config")
            .ok_or_else(|| "saved spot missing 'config'".to_string())?;
        let config = SpotConfig::from_json(cfg_j)?;
        let summary_hero_ev = j.get("summary_hero_ev").and_then(|v| v.as_f64());
        let summary_expl = j.get("summary_expl").and_then(|v| v.as_f64());
        Ok(Self {
            id,
            name,
            folder,
            created_at,
            config,
            summary_hero_ev,
            summary_expl,
        })
    }
}

/// Persistent desktop settings saved in OS app-data config JSON.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AppSettings {
    pub pack_folder: Option<String>,
    pub spot_bindings: BTreeMap<String, String>,
    pub last_spot_id: Option<String>,
    pub memory_budget_mib: Option<u32>,
    pub saved_spots: Vec<SavedSpotEntry>,
}

impl AppSettings {
    pub fn default_config_path() -> PathBuf {
        if let Ok(p) = env::var("PLO_DESKTOP_CONFIG_PATH") {
            if !p.trim().is_empty() {
                return PathBuf::from(p);
            }
        }
        #[cfg(target_os = "windows")]
        {
            if let Ok(appdata) = env::var("APPDATA") {
                return PathBuf::from(appdata)
                    .join("plo_desktop")
                    .join("settings.json");
            }
        }
        #[cfg(target_os = "macos")]
        {
            if let Ok(home) = env::var("HOME") {
                return PathBuf::from(home)
                    .join("Library")
                    .join("Application Support")
                    .join("plo_desktop")
                    .join("settings.json");
            }
        }
        if let Ok(xdg) = env::var("XDG_CONFIG_HOME") {
            if !xdg.trim().is_empty() {
                return PathBuf::from(xdg).join("plo_desktop").join("settings.json");
            }
        }
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home)
                .join(".config")
                .join("plo_desktop")
                .join("settings.json");
        }
        PathBuf::from("plo_desktop_settings.json")
    }

    pub fn load_from_path(path: &Path) -> Self {
        let Ok(raw) = fs::read_to_string(path) else {
            return Self::default();
        };
        Self::from_json_str(&raw).unwrap_or_default()
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("failed to create config dir {}: {e}", parent.display()))?;
            }
        }
        fs::write(path, self.to_json().to_json_pretty())
            .map_err(|e| format!("failed to write settings {}: {e}", path.display()))
    }

    pub fn to_json(&self) -> Json {
        let mut bmap = BTreeMap::new();
        for (k, v) in &self.spot_bindings {
            bmap.insert(k.clone(), Json::Str(v.clone()));
        }
        let saved_arr: Vec<Json> = self.saved_spots.iter().map(|s| s.to_json()).collect();
        Json::obj([
            (
                "pack_folder",
                match &self.pack_folder {
                    Some(p) => Json::Str(p.clone()),
                    None => Json::Null,
                },
            ),
            ("spot_bindings", Json::Obj(bmap)),
            (
                "last_spot_id",
                match &self.last_spot_id {
                    Some(s) => Json::Str(s.clone()),
                    None => Json::Null,
                },
            ),
            (
                "memory_budget_mib",
                match self.memory_budget_mib {
                    Some(m) => Json::Num(m as f64),
                    None => Json::Null,
                },
            ),
            ("saved_spots", Json::Arr(saved_arr)),
        ])
    }

    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let j = Json::parse(s)?;
        let pack_folder = j
            .get("pack_folder")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let mut spot_bindings = BTreeMap::new();
        if let Some(obj) = j.get("spot_bindings").and_then(|v| v.as_obj()) {
            for (k, v) in obj {
                if let Some(vs) = v.as_str() {
                    spot_bindings.insert(k.clone(), vs.to_string());
                }
            }
        }
        let last_spot_id = j
            .get("last_spot_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string());
        let memory_budget_mib = j
            .get("memory_budget_mib")
            .and_then(|v| {
                v.as_u32()
                    .or_else(|| v.as_str().and_then(|s| parse_memory_budget_input(s).ok()))
            })
            .map(|m| m.clamp(MIN_DESKTOP_BUDGET_MIB, MAX_DESKTOP_BUDGET_MIB));
        let mut saved_spots = Vec::new();
        if let Some(arr) = j.get("saved_spots").and_then(|v| v.as_arr()) {
            for item in arr {
                if let Ok(se) = SavedSpotEntry::from_json(item) {
                    saved_spots.push(se);
                }
            }
        }
        Ok(Self {
            pack_folder,
            spot_bindings,
            last_spot_id,
            memory_budget_mib,
            saved_spots,
        })
    }

    pub fn effective_budget_mib(&self) -> u32 {
        self.memory_budget_mib
            .map(|m| m.clamp(MIN_DESKTOP_BUDGET_MIB, MAX_DESKTOP_BUDGET_MIB))
            .unwrap_or_else(default_desktop_budget_mib)
    }
}

/// Complete Spot / Solve configuration supporting JSON Save & Load.
#[derive(Clone, Debug, PartialEq)]
pub struct SpotConfig {
    pub board: String,
    pub hero_hand: String,
    /// `"OOP"` (`OOP = hero, IP = villain`) or `"IP"` (`OOP = villain, IP = hero`).
    pub hero_seat: String,
    pub use_pack: bool,
    pub villain_spot_id: String,
    pub villain_override_range: String,
    pub starting_pot: i32,
    pub effective_stack: i32,
    /// Legacy global fallback bet/raise sizes (used when a street-specific field is empty).
    pub bet_sizes: String,
    pub raise_sizes: String,
    /// Per-street GTO+-style Flop / Turn / River OOP & IP bet/raise/donk pot-% sizes.
    pub flop_oop_bets: String,
    pub flop_oop_raises: String,
    pub flop_ip_bets: String,
    pub flop_ip_raises: String,
    pub turn_oop_bets: String,
    pub turn_oop_raises: String,
    pub turn_oop_donks: String,
    pub turn_ip_bets: String,
    pub turn_ip_raises: String,
    pub river_oop_bets: String,
    pub river_oop_raises: String,
    pub river_oop_donks: String,
    pub river_ip_bets: String,
    pub river_ip_raises: String,
    /// Which streets to solve (`"All"`, `"FlopOnly"`, `"TurnOnly"`, `"RiverOnly"`, `"FlopTurn"`).
    pub solve_streets: String,
    /// `"Auto"`, `"Identity"`, `"NutBlocker50"`, `"NutBlocker100"`.
    pub abstraction_mode: String,
    pub max_iterations: u32,
    /// Optional early-stop target exploitability in `% of starting_pot` (`0.0` = run full `max_iterations`).
    pub target_dev_pct: f64,
    /// Postflop showdown rake fraction in `%` (`0.0` = zero-sum no rake; e.g. `5.0` = 5% pot rake).
    pub rake_pct: f64,
    /// Maximum showdown rake cap in chip units (`0.0` = uncapped when `rake_pct > 0`).
    pub rake_cap: f64,
    pub memory_budget_mib: u32,
    /// Collapsed unadvertised Advanced Lab two-range mode (`false` in normal Hero-vs-Range UX).
    pub lab_two_range_mode: bool,
    pub lab_oop_range: String,
    pub lab_ip_range: String,
}

impl Default for SpotConfig {
    fn default() -> Self {
        Self {
            board: "KsQsJs7d2c".to_string(),
            hero_hand: "AsTs9d8d".to_string(),
            hero_seat: "OOP".to_string(),
            use_pack: true,
            villain_spot_id: "RFI_BTN".to_string(),
            villain_override_range: "AAKK$ds, KQJT$ds, TT99$ds".to_string(),
            starting_pot: 100,
            effective_stack: 100,
            bet_sizes: "50%, 100%".to_string(),
            raise_sizes: "100%".to_string(),
            flop_oop_bets: String::new(),
            flop_oop_raises: String::new(),
            flop_ip_bets: String::new(),
            flop_ip_raises: String::new(),
            turn_oop_bets: String::new(),
            turn_oop_raises: String::new(),
            turn_oop_donks: String::new(),
            turn_ip_bets: String::new(),
            turn_ip_raises: String::new(),
            river_oop_bets: String::new(),
            river_oop_raises: String::new(),
            river_oop_donks: String::new(),
            river_ip_bets: String::new(),
            river_ip_raises: String::new(),
            solve_streets: "All".to_string(),
            abstraction_mode: "Auto".to_string(),
            max_iterations: 1000,
            target_dev_pct: 0.0,
            rake_pct: 0.0,
            rake_cap: 0.0,
            memory_budget_mib: default_desktop_budget_mib(),
            lab_two_range_mode: false,
            lab_oop_range: String::new(),
            lab_ip_range: String::new(),
        }
    }
}

impl SpotConfig {
    pub fn clamped_budget_mib(&self) -> u32 {
        self.memory_budget_mib
            .clamp(MIN_DESKTOP_BUDGET_MIB, MAX_DESKTOP_BUDGET_MIB)
    }

    pub fn budget_bytes(&self) -> u64 {
        (self.clamped_budget_mib() as u64) * 1024 * 1024
    }

    pub fn to_json(&self) -> Json {
        Json::obj([
            ("board", Json::Str(self.board.clone())),
            ("hero_hand", Json::Str(self.hero_hand.clone())),
            ("hero_seat", Json::Str(self.hero_seat.clone())),
            ("use_pack", Json::Bool(self.use_pack)),
            ("villain_spot_id", Json::Str(self.villain_spot_id.clone())),
            (
                "villain_override_range",
                Json::Str(self.villain_override_range.clone()),
            ),
            ("starting_pot", Json::Num(self.starting_pot as f64)),
            ("effective_stack", Json::Num(self.effective_stack as f64)),
            ("bet_sizes", Json::Str(self.bet_sizes.clone())),
            ("raise_sizes", Json::Str(self.raise_sizes.clone())),
            ("flop_oop_bets", Json::Str(self.flop_oop_bets.clone())),
            ("flop_oop_raises", Json::Str(self.flop_oop_raises.clone())),
            ("flop_ip_bets", Json::Str(self.flop_ip_bets.clone())),
            ("flop_ip_raises", Json::Str(self.flop_ip_raises.clone())),
            ("turn_oop_bets", Json::Str(self.turn_oop_bets.clone())),
            ("turn_oop_raises", Json::Str(self.turn_oop_raises.clone())),
            ("turn_oop_donks", Json::Str(self.turn_oop_donks.clone())),
            ("turn_ip_bets", Json::Str(self.turn_ip_bets.clone())),
            ("turn_ip_raises", Json::Str(self.turn_ip_raises.clone())),
            ("river_oop_bets", Json::Str(self.river_oop_bets.clone())),
            ("river_oop_raises", Json::Str(self.river_oop_raises.clone())),
            ("river_oop_donks", Json::Str(self.river_oop_donks.clone())),
            ("river_ip_bets", Json::Str(self.river_ip_bets.clone())),
            ("river_ip_raises", Json::Str(self.river_ip_raises.clone())),
            ("solve_streets", Json::Str(self.solve_streets.clone())),
            ("abstraction_mode", Json::Str(self.abstraction_mode.clone())),
            ("max_iterations", Json::Num(self.max_iterations as f64)),
            ("target_dev_pct", Json::Num(self.target_dev_pct)),
            ("rake_pct", Json::Num(self.rake_pct)),
            ("rake_cap", Json::Num(self.rake_cap)),
            (
                "memory_budget_mib",
                Json::Num(self.clamped_budget_mib() as f64),
            ),
            ("lab_two_range_mode", Json::Bool(self.lab_two_range_mode)),
            ("lab_oop_range", Json::Str(self.lab_oop_range.clone())),
            ("lab_ip_range", Json::Str(self.lab_ip_range.clone())),
        ])
    }

    pub fn from_json(j: &Json) -> Result<Self, String> {
        let def = Self::default();
        let opt_str = |key: &str| -> String {
            j.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        Ok(Self {
            board: j
                .get("board")
                .and_then(|v| v.as_str())
                .unwrap_or(&def.board)
                .to_string(),
            hero_hand: j
                .get("hero_hand")
                .and_then(|v| v.as_str())
                .unwrap_or(&def.hero_hand)
                .to_string(),
            hero_seat: j
                .get("hero_seat")
                .and_then(|v| v.as_str())
                .unwrap_or(&def.hero_seat)
                .to_string(),
            use_pack: j
                .get("use_pack")
                .and_then(|v| v.as_bool())
                .unwrap_or(def.use_pack),
            villain_spot_id: j
                .get("villain_spot_id")
                .and_then(|v| v.as_str())
                .unwrap_or(&def.villain_spot_id)
                .to_string(),
            villain_override_range: j
                .get("villain_override_range")
                .and_then(|v| v.as_str())
                .unwrap_or(&def.villain_override_range)
                .to_string(),
            starting_pot: j
                .get("starting_pot")
                .and_then(|v| v.as_i64())
                .map(|x| x as i32)
                .unwrap_or(def.starting_pot),
            effective_stack: j
                .get("effective_stack")
                .and_then(|v| v.as_i64())
                .map(|x| x as i32)
                .unwrap_or(def.effective_stack),
            bet_sizes: j
                .get("bet_sizes")
                .and_then(|v| v.as_str())
                .unwrap_or(&def.bet_sizes)
                .to_string(),
            raise_sizes: j
                .get("raise_sizes")
                .and_then(|v| v.as_str())
                .unwrap_or(&def.raise_sizes)
                .to_string(),
            flop_oop_bets: opt_str("flop_oop_bets"),
            flop_oop_raises: opt_str("flop_oop_raises"),
            flop_ip_bets: opt_str("flop_ip_bets"),
            flop_ip_raises: opt_str("flop_ip_raises"),
            turn_oop_bets: opt_str("turn_oop_bets"),
            turn_oop_raises: opt_str("turn_oop_raises"),
            turn_oop_donks: opt_str("turn_oop_donks"),
            turn_ip_bets: opt_str("turn_ip_bets"),
            turn_ip_raises: opt_str("turn_ip_raises"),
            river_oop_bets: opt_str("river_oop_bets"),
            river_oop_raises: opt_str("river_oop_raises"),
            river_oop_donks: opt_str("river_oop_donks"),
            river_ip_bets: opt_str("river_ip_bets"),
            river_ip_raises: opt_str("river_ip_raises"),
            solve_streets: j
                .get("solve_streets")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .unwrap_or(&def.solve_streets)
                .to_string(),
            abstraction_mode: j
                .get("abstraction_mode")
                .and_then(|v| v.as_str())
                .unwrap_or(&def.abstraction_mode)
                .to_string(),
            max_iterations: j
                .get("max_iterations")
                .and_then(|v| v.as_u32())
                .unwrap_or(def.max_iterations),
            target_dev_pct: j
                .get("target_dev_pct")
                .and_then(|v| v.as_f64())
                .map(|x| x.max(0.0))
                .unwrap_or(def.target_dev_pct),
            rake_pct: j
                .get("rake_pct")
                .and_then(|v| v.as_f64())
                .map(|x| x.clamp(0.0, 50.0))
                .unwrap_or(def.rake_pct),
            rake_cap: j
                .get("rake_cap")
                .and_then(|v| v.as_f64())
                .map(|x| x.max(0.0))
                .unwrap_or(def.rake_cap),
            memory_budget_mib: match j.get("memory_budget_mib") {
                Some(Json::Str(s)) => parse_memory_budget_input(s)?,
                Some(v) => v
                    .as_u32()
                    .unwrap_or(def.memory_budget_mib)
                    .clamp(MIN_DESKTOP_BUDGET_MIB, MAX_DESKTOP_BUDGET_MIB),
                None => def.memory_budget_mib,
            },
            lab_two_range_mode: j
                .get("lab_two_range_mode")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            lab_oop_range: j
                .get("lab_oop_range")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            lab_ip_range: j
                .get("lab_ip_range")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        })
    }
}
