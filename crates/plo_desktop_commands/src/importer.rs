//! Recursive Range Pack Importer (`RangeImporter`) supporting:
//! 1. Native OOXML `.xlsx` preflop chart workbooks (`JNandez Preflop Chart.xlsx` `NR 1..133`
//!    category-weight sheets as well as explicit combo/macro `Cards` rows in synthetic fixtures),
//! 2. Recursive `.txt`, `.plo`, `.csv`, `.tsv` range files (`combo@weight` / `combo,weight` / `PloRange` syntax),
//! 3. Canonical spot ID mapping + user manual spot-binding overrides persisted in OS app-data JSON.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use plo_cards::{Card, Hand4, NUM_PLO_COMBOS};
use plo_iso::{classify_suit_pattern, SuitPattern4};
use plo_range::{PloRange, RangeError};

use crate::formats::{read_xlsx_workbook, Json, XlsxSheet};

/// Canonical UI group names in display order for the Villain spot selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpotGroup {
    RfiOpen,
    BbDefense,
    ThreeBetIp,
    ThreeBetOop,
    ColdCallIp,
    Other,
}

impl SpotGroup {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::RfiOpen => "RFI (open)",
            Self::BbDefense => "BB defense",
            Self::ThreeBetIp => "3-bet IP",
            Self::ThreeBetOop => "3-bet OOP",
            Self::ColdCallIp => "Cold call IP",
            Self::Other => "Other / unmapped files",
        }
    }
}

/// Metadata for a single canonical or extra preflop spot exposed to the Villain selector.
#[derive(Clone, Debug)]
pub struct SpotDescriptor {
    /// Stable spot identifier (`RFI_BTN`, `BB_DEF_VS_1_RAISER`, or `RAW::<file>::<sheet>::<col>`).
    pub spot_id: String,
    /// UI group (`"RFI (open)"`, `"BB defense"`, `"3-bet IP"`, `"3-bet OOP"`, `"Cold call IP"`, `"Other / unmapped files"`).
    pub group: String,
    /// Human-readable label shown in the Villain spot dropdown (`"RFI · BTN"`, `"BB vs 1 raiser"`, etc.).
    pub label: String,
    /// Underlying source key inside the pack (`<rel_path>::<sheet>::<col>` or `<rel_path>`), or empty if missing from pack.
    pub source_key: String,
    /// Whether this spot is one of the 15 canonical preflop spots (`true`) or an extra raw file/column (`false`).
    pub is_canonical: bool,
    /// Whether this spot is present and resolvable in the currently loaded pack (`false` => show disabled `"not in pack"`).
    pub available: bool,
    /// Total unblocked combos with `weight > 0` on an empty board (`combo_count_full`).
    pub combo_count_full: usize,
    /// Alias kept for compatibility (`== combo_count_full`).
    pub raw_combos: usize,
    /// Sum of weights divided by `270,725` (`0.0..=100.0`%).
    pub frequency_pct: f64,
}

impl SpotDescriptor {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("id", Json::Str(self.spot_id.clone())),
            ("spot_id", Json::Str(self.spot_id.clone())),
            ("group", Json::Str(self.group.clone())),
            ("label", Json::Str(self.label.clone())),
            ("source_key", Json::Str(self.source_key.clone())),
            ("is_canonical", Json::Bool(self.is_canonical)),
            ("available", Json::Bool(self.available)),
            ("combo_count_full", Json::Num(self.combo_count_full as f64)),
            ("raw_combos", Json::Num(self.raw_combos as f64)),
            ("frequency_pct", Json::Num(self.frequency_pct)),
        ])
    }
}

/// Loaded in-memory range pack from a user-selected directory.
#[derive(Clone, Debug, Default)]
pub struct LoadedRangePack {
    pub folder: PathBuf,
    /// All raw discovered sources (`source_key -> (default_title, PloRange)`).
    pub raw_sources: BTreeMap<String, (String, PloRange)>,
    /// Ordered list of canonical + raw spots exposed to the UI.
    pub spots: Vec<SpotDescriptor>,
    /// Effective `spot_id -> source_key` resolution map (incorporating user custom bindings).
    pub bindings: BTreeMap<String, String>,
}

impl LoadedRangePack {
    /// Loads a range pack recursively from `folder`, applying optional user `custom_bindings`
    /// (`spot_id -> source_key`). Never modifies files inside `folder`.
    pub fn load_from_dir(
        folder: &Path,
        custom_bindings: &BTreeMap<String, String>,
    ) -> Result<Self, String> {
        if !folder.exists() || !folder.is_dir() {
            return Err(format!(
                "range pack folder does not exist or is not a directory: {}",
                folder.display()
            ));
        }

        let mut files = Vec::new();
        collect_files_recursive(folder, 0, &mut files)?;
        files.sort();

        let mut raw_sources: BTreeMap<String, (String, PloRange)> = BTreeMap::new();

        for path in files {
            let rel = path
                .strip_prefix(folder)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            match ext.as_str() {
                "xlsx" => {
                    if rel.split('/').any(|seg| seg.starts_with("~$")) {
                        continue; // Skip Excel lock files
                    }
                    let meta = fs::metadata(&path)
                        .map_err(|e| format!("failed to stat {}: {e}", path.display()))?;
                    if meta.len() > MAX_IMPORT_FILE_BYTES {
                        return Err(format!(
                            "workbook {} ({} bytes) exceeds 64 MiB safety limit",
                            rel,
                            meta.len()
                        ));
                    }
                    let bytes = fs::read(&path)
                        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
                    let sheets = read_xlsx_workbook(&bytes)
                        .map_err(|e| format!("failed to parse workbook {rel}: {e}"))?;
                    for sheet in sheets {
                        import_xlsx_sheet(&rel, &sheet, &mut raw_sources)?;
                    }
                }
                "txt" | "plo" | "csv" | "tsv" => {
                    let meta = fs::metadata(&path)
                        .map_err(|e| format!("failed to stat {}: {e}", path.display()))?;
                    if meta.len() > MAX_IMPORT_FILE_BYTES {
                        return Err(format!(
                            "range file {} ({} bytes) exceeds 64 MiB safety limit",
                            rel,
                            meta.len()
                        ));
                    }
                    let text = fs::read_to_string(&path)
                        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
                    if let Some(range) = parse_text_range_file(&text)? {
                        if !range.is_empty() {
                            let stem = path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or(&rel)
                                .replace('_', " ");
                            raw_sources.insert(rel.clone(), (stem, range));
                        }
                    }
                }
                _ => {}
            }
        }

        if raw_sources.is_empty() {
            return Err(format!(
                "no valid PLO-4 range workbooks (.xlsx) or range files (.txt/.plo/.csv) found in {}",
                folder.display()
            ));
        }

        let mut pack = Self {
            folder: folder.to_path_buf(),
            raw_sources,
            spots: Vec::new(),
            bindings: BTreeMap::new(),
        };
        pack.rebuild_spots(custom_bindings);
        Ok(pack)
    }

    /// Recomputes `spots` and `bindings` when user custom bindings change.
    pub fn rebuild_spots(&mut self, custom_bindings: &BTreeMap<String, String>) {
        self.spots.clear();
        self.bindings.clear();

        let mut bound_sources = std::collections::BTreeSet::new();

        // 1. All 15 canonical preflop spots in UI group order (`RFI (open)`, `BB defense`,
        //    `3-bet IP`, `3-bet OOP`, `Cold call IP`).
        //    If a canonical spot is absent from the loaded pack, emit it with `available = false`
        //    so the UI renders it disabled (`"not in pack"`) instead of silently dropping the group.
        for &(canon_id, group, canon_label, sheet_match, hdr_match) in CANONICAL_SPOT_RULES {
            let chosen_source = if let Some(user_src) = custom_bindings.get(canon_id) {
                if self.raw_sources.contains_key(user_src) {
                    Some(user_src.clone())
                } else {
                    None
                }
            } else {
                self.find_unambiguous_source(sheet_match, hdr_match)
            };

            if let Some(src_key) = chosen_source {
                if let Some((_, range)) = self.raw_sources.get(&src_key) {
                    bound_sources.insert(src_key.clone());
                    self.bindings.insert(canon_id.to_string(), src_key.clone());
                    let count = range.len();
                    self.spots.push(SpotDescriptor {
                        spot_id: canon_id.to_string(),
                        group: group.as_label().to_string(),
                        label: canon_label.to_string(),
                        source_key: src_key,
                        is_canonical: true,
                        available: true,
                        combo_count_full: count,
                        raw_combos: count,
                        frequency_pct: (range.total_weight() / NUM_PLO_COMBOS as f64) * 100.0,
                    });
                    continue;
                }
            }

            // Missing from this pack — keep canonical entry disabled (`not in pack`)
            self.spots.push(SpotDescriptor {
                spot_id: canon_id.to_string(),
                group: group.as_label().to_string(),
                label: canon_label.to_string(),
                source_key: String::new(),
                is_canonical: true,
                available: false,
                combo_count_full: 0,
                raw_combos: 0,
                frequency_pct: 0.0,
            });
        }

        // 2. Also include any custom user-defined spot IDs not in CANONICAL_SPOT_RULES
        for (spot_id, src_key) in custom_bindings {
            if self.bindings.contains_key(spot_id) {
                continue;
            }
            if let Some((raw_title, range)) = self.raw_sources.get(src_key) {
                bound_sources.insert(src_key.clone());
                self.bindings.insert(spot_id.clone(), src_key.clone());
                let count = range.len();
                self.spots.push(SpotDescriptor {
                    spot_id: spot_id.clone(),
                    group: SpotGroup::Other.as_label().to_string(),
                    label: format!("{spot_id} ({raw_title})"),
                    source_key: src_key.clone(),
                    is_canonical: false,
                    available: true,
                    combo_count_full: count,
                    raw_combos: count,
                    frequency_pct: (range.total_weight() / NUM_PLO_COMBOS as f64) * 100.0,
                });
            }
        }

        // 3. Expose all remaining raw file / sheet-column ranges under `"Other / unmapped files"`
        //    so NO extra sheet/column or `.txt/.plo/.csv` file in a user's folder is unreachable.
        for (src_key, (raw_title, range)) in &self.raw_sources {
            if bound_sources.contains(src_key) {
                continue;
            }
            // Skip duplicate summary columns from `RFI Overview` when dedicated `RFI EP/MP/CO/BTN` already bound
            if src_key.contains("::RFI Overview::") && !self.bindings.is_empty() {
                continue;
            }
            let spot_id = format!("RAW::{src_key}");
            self.bindings.insert(spot_id.clone(), src_key.clone());
            let count = range.len();
            self.spots.push(SpotDescriptor {
                spot_id,
                group: SpotGroup::Other.as_label().to_string(),
                label: raw_title.clone(),
                source_key: src_key.clone(),
                is_canonical: false,
                available: true,
                combo_count_full: count,
                raw_combos: count,
                frequency_pct: (range.total_weight() / NUM_PLO_COMBOS as f64) * 100.0,
            });
        }
    }

    fn find_unambiguous_source(&self, sheet_exact: &str, hdr_exact: &str) -> Option<String> {
        let target_suffix = format!("::{sheet_exact}::{hdr_exact}");
        let matches: Vec<&String> = self
            .raw_sources
            .keys()
            .filter(|k| k.eq_ignore_ascii_case(&target_suffix) || k.ends_with(&target_suffix))
            .collect();
        if matches.len() == 1 {
            Some(matches[0].clone())
        } else {
            None
        }
    }

    pub fn has_available_spot(&self, spot_id: &str) -> bool {
        let resolved_id = if spot_id == "RFI_EP" {
            "RFI_UTG"
        } else {
            spot_id
        };
        if let Some(src) = self.bindings.get(resolved_id) {
            return self.raw_sources.contains_key(src);
        }
        self.raw_sources.contains_key(resolved_id)
    }

    pub fn first_available_spot_id(&self) -> Option<String> {
        self.spots
            .iter()
            .find(|s| s.available)
            .map(|s| s.spot_id.clone())
    }

    pub fn range_for_spot(&self, spot_id: &str) -> Result<&PloRange, String> {
        let resolved_id = if spot_id == "RFI_EP" {
            "RFI_UTG"
        } else {
            spot_id
        };
        if let Some(src) = self.bindings.get(resolved_id) {
            if let Some((_, r)) = self.raw_sources.get(src) {
                return Ok(r);
            }
        }
        if let Some((_, r)) = self.raw_sources.get(resolved_id) {
            return Ok(r);
        }
        if CANONICAL_SPOT_RULES
            .iter()
            .any(|&(cid, _, _, _, _)| cid == resolved_id)
        {
            return Err(format!(
                "spot {resolved_id:?} is not in the currently loaded range pack"
            ));
        }
        Err(format!("unknown range spot id: {spot_id:?}"))
    }

    pub fn available_source_keys(&self) -> Vec<String> {
        self.raw_sources.keys().cloned().collect()
    }
}

/// Exact 15 canonical preflop spots from `JNandez Preflop Chart.xlsx`, ordered by UI group:
/// 1. `RFI (open)` (4 spots: `RFI_UTG`, `RFI_MP`, `RFI_CO`, `RFI_BTN` — workbook has no SB RFI sheet)
/// 2. `BB defense` (2 spots: `BB_DEF_VS_1_RAISER`, `BB_DEF_MULTIWAY`)
/// 3. `3-bet IP` (2 spots: `THREEBET_IP_VS_15_20`, `THREEBET_IP_VS_25_35`)
/// 4. `3-bet OOP` (4 spots: `THREEBET_OOP_BB_VS_30`, `THREEBET_OOP_SB_VS_30`, `THREEBET_OOP_BB_VS_60`, `THREEBET_OOP_SB_VS_60`)
/// 5. `Cold call IP` (3 spots: `CC_IP_MP_CO_VS_15`, `CC_IP_BTN_VS_15`, `CC_IP_BTN_VS_30`)
pub const CANONICAL_SPOT_RULES: &[(&str, SpotGroup, &str, &str, &str)] = &[
    // Group 1: RFI (open)
    (
        "RFI_UTG",
        SpotGroup::RfiOpen,
        "RFI · UTG",
        "RFI EP",
        "UTG",
    ),
    (
        "RFI_MP",
        SpotGroup::RfiOpen,
        "RFI · MP",
        "RFI MP",
        "MP",
    ),
    (
        "RFI_CO",
        SpotGroup::RfiOpen,
        "RFI · CO",
        "RFI CO",
        "CO",
    ),
    (
        "RFI_BTN",
        SpotGroup::RfiOpen,
        "RFI · BTN",
        "RFI BTN",
        "BTN",
    ),
    // Group 2: BB defense
    (
        "BB_DEF_VS_1_RAISER",
        SpotGroup::BbDefense,
        "BB vs 1 raiser",
        "BB Defense",
        "BB vs 1 Raiser",
    ),
    (
        "BB_DEF_MULTIWAY",
        SpotGroup::BbDefense,
        "BB multiway (vs raiser + callers)",
        "BB Defense",
        "BB vs Raiser + Callers",
    ),
    // Group 3: 3-bet IP
    (
        "THREEBET_IP_VS_15_20",
        SpotGroup::ThreeBetIp,
        "3-bet IP vs 15–20% open",
        "3-Betting IP",
        "Vs. 15-20%",
    ),
    (
        "THREEBET_IP_VS_25_35",
        SpotGroup::ThreeBetIp,
        "3-bet IP vs 25–35% open",
        "3-Betting IP",
        "Vs. 25-35%",
    ),
    // Group 4: 3-bet OOP
    (
        "THREEBET_OOP_BB_VS_30",
        SpotGroup::ThreeBetOop,
        "3-bet OOP · BB vs 30% open",
        "3-Betting OOP",
        "Vs. 30% in BB",
    ),
    (
        "THREEBET_OOP_SB_VS_30",
        SpotGroup::ThreeBetOop,
        "3-bet OOP · SB vs 30% open",
        "3-Betting OOP",
        "Vs. 30% in SB",
    ),
    (
        "THREEBET_OOP_BB_VS_60",
        SpotGroup::ThreeBetOop,
        "3-bet OOP · BB vs 60% open",
        "3-Betting OOP",
        "Vs. 60% in BB",
    ),
    (
        "THREEBET_OOP_SB_VS_60",
        SpotGroup::ThreeBetOop,
        "3-bet OOP · SB vs 60% open",
        "3-Betting OOP",
        "Vs. 60% in SB",
    ),
    // Group 5: Cold call IP
    (
        "CC_IP_MP_CO_VS_15",
        SpotGroup::ColdCallIp,
        "Cold call IP · MP/CO vs 15%",
        "Coldcalling IP",
        "Vs. 15% in MP/CO",
    ),
    (
        "CC_IP_BTN_VS_15",
        SpotGroup::ColdCallIp,
        "Cold call IP · BTN vs 15%",
        "Coldcalling IP",
        "Vs. 15% on BTN",
    ),
    (
        "CC_IP_BTN_VS_30",
        SpotGroup::ColdCallIp,
        "Cold call IP · BTN vs 30%",
        "Coldcalling IP",
        "Vs. 30% on BTN",
    ),
];

const MAX_IMPORT_RECURSION_DEPTH: usize = 8;
const MAX_IMPORT_FILES: usize = 2_048;
const MAX_IMPORT_FILE_BYTES: u64 = 64 * 1024 * 1024;

fn collect_files_recursive(
    dir: &Path,
    depth: usize,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    if depth > MAX_IMPORT_RECURSION_DEPTH {
        return Ok(());
    }
    let entries =
        fs::read_dir(dir).map_err(|e| format!("cannot read directory {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("directory iteration error: {e}"))?;
        let ftype = entry
            .file_type()
            .map_err(|e| format!("cannot inspect file type in {}: {e}", dir.display()))?;
        // Do not follow symlinks so cyclic directory links cannot loop or escape the pack folder.
        if ftype.is_symlink() {
            continue;
        }
        let path = entry.path();
        if ftype.is_dir() {
            collect_files_recursive(&path, depth + 1, out)?;
        } else if ftype.is_file() {
            if out.len() >= MAX_IMPORT_FILES {
                return Err(format!(
                    "range pack directory exceeds safety limit of {MAX_IMPORT_FILES} files"
                ));
            }
            out.push(path);
        }
    }
    Ok(())
}

/// Imports range columns from a worksheet if `Row 1` matches the JNandez category-chart layout
/// (`Cards` in Col `E`, `Text` in Col `D`, and one or more `[0.0..=1.0]` weight columns in `F..N`).
fn import_xlsx_sheet(
    rel_file: &str,
    sheet: &XlsxSheet,
    out_sources: &mut BTreeMap<String, (String, PloRange)>,
) -> Result<(), String> {
    // Skip non-range reference/overview sheets
    if sheet.name.eq_ignore_ascii_case("All categories overview")
        || sheet.name.eq_ignore_ascii_case("RFI Profiles")
        || sheet.name.eq_ignore_ascii_case("Preflop 3-Bet Calc")
    {
        return Ok(());
    }

    let Some(hdr_row) = sheet.rows.get(&1) else {
        return Ok(());
    };
    let e_hdr = hdr_row.get("E").map(|s| s.as_str()).unwrap_or("");
    if !e_hdr.eq_ignore_ascii_case("Cards") {
        return Ok(());
    }

    // Identify candidate spot frequency columns:
    // - Non-empty text header (not a numeric `=SUM(...)` cell, not `NR`/`Type`/`Text`/`Cards`/`% of total`,
    //   and not a ranking/equity metric column like `JN * Ranking`, `Rank *`, `EQ*`, `Δ`, `P Max*`, `N Max*`, `Holding A`)
    // - Every non-empty data cell in rows `2..=500` parses as `f64` in `0.0..=1.0`.
        let candidate_cols = ["F", "G", "H", "I", "J", "K", "L", "M", "N"];
    for col in candidate_cols {
        let Some(col_hdr) = hdr_row.get(col) else {
            continue;
        };
        if !is_spot_weight_header(col_hdr) {
            continue;
        }
        // In `RFI MP` / `RFI CO` / `RFI BTN`, Col G carries the prior position's reference column
        // (`UTG` / `MP` / `CO`) purely for visual comparison next to Col H (`MP` / `CO` / `BTN`).
        // Skip that reference column so it doesn't clutter raw spots.
        if col == "G"
            && ((sheet.name.eq_ignore_ascii_case("RFI MP") && col_hdr.eq_ignore_ascii_case("UTG"))
                || (sheet.name.eq_ignore_ascii_case("RFI CO")
                    && col_hdr.eq_ignore_ascii_case("MP"))
                || (sheet.name.eq_ignore_ascii_case("RFI BTN")
                    && col_hdr.eq_ignore_ascii_case("CO")))
        {
            continue;
        }

        let mut all_valid_weights = true;
        let mut has_positive = false;
        let mut row_weights: Vec<(u32, Option<u16>, String, f32)> = Vec::new();

        for (&row_idx, cells) in sheet.rows.range(2..) {
            let Some(cards_spec) = cells.get("E") else {
                continue;
            };
            let w_str = cells.get(col).map(|s| s.as_str()).unwrap_or("0");
            let Ok(w_val) = w_str.parse::<f64>() else {
                all_valid_weights = false;
                break;
            };
            if !w_val.is_finite() || !(0.0..=1.0).contains(&w_val) {
                all_valid_weights = false;
                break;
            }
            if w_val > 0.0 {
                has_positive = true;
            }
            let nr_opt = cells.get("A").and_then(|a| a.parse::<u16>().ok());
            row_weights.push((row_idx, nr_opt, cards_spec.clone(), w_val as f32));
        }

        if !all_valid_weights || !has_positive {
            continue;
        }

        let range = expand_sheet_column_to_range(&row_weights)?;
        if !range.is_empty() {
            let src_key = format!("{rel_file}::{}::{col_hdr}", sheet.name);
            let title = format!("{} — {col_hdr}", sheet.name);
            out_sources.insert(src_key, (title, range));
        }
    }

    Ok(())
}

fn is_spot_weight_header(hdr: &str) -> bool {
    let h = hdr.trim();
    if h.is_empty() {
        return false;
    }
    if h.parse::<f64>().is_ok() {
        return false; // Cached =SUM(...) float in Row 1
    }
    let lower = h.to_ascii_lowercase();
    let blocked_prefixes = [
        "nr",
        "type",
        "text",
        "cards",
        "% of total",
        "tier",
        "hc max",
        "c max",
        "s max",
        "pr max",
        "pr%",
        "p max",
        "n max",
        "3b max",
        "3b%",
        "rank ",
        "jn ",
        "eq ",
        "eq.",
        "holding ",
        "situation",
        "frequency",
        "winrate",
        "importance",
        "weighted",
        "δ",
    ];
    !(h == "Δ" || blocked_prefixes.iter().any(|p| lower.starts_with(p)))
}

fn expand_sheet_column_to_range(
    rows: &[(u32, Option<u16>, String, f32)],
) -> Result<PloRange, String> {
    // Detect whether this sheet uses the 133 JNandez category rows (`NR 1..=133` with rank-interval `AKQJ-AJT7`)
    // or explicit `PloRange` combo/macro rows (such as synthetic fixtures `NR >= 1000` or `AsKsQhJh`).
    let is_133_chart = rows.len() >= 100
        && rows
            .iter()
            .filter(|(_, _, c, _)| c.contains('-'))
            .count()
            >= 100;

    if is_133_chart {
        let mut cat_weights = [0.0f32; 134];
        for &(row_idx, nr_opt, _, w) in rows {
            // In JNandez sheets, Row 2..=134 corresponds 1-to-1 to NR 1..=133 (even on RFI BTN Row 89 where A89 is ".")
            let nr = nr_opt.unwrap_or((row_idx - 1) as u16) as usize;
            if (1..=133).contains(&nr) {
                cat_weights[nr] = w;
            }
        }
        let table = category_lookup_table();
        let mut weighted_hands = Vec::new();
        for (cid, hand) in Hand4::all_hands() {
            let cat = table[cid.index()] as usize;
            if (1..=133).contains(&cat) {
                let w = cat_weights[cat];
                if w > 0.0 {
                    weighted_hands.push((hand, w));
                }
            }
        }
        PloRange::from_hands(weighted_hands).map_err(|e| e.to_string())
    } else {
        // Explicit `PloRange` token rows (exact 4-card combos `AsKsQhJh`, macros `AAKK$ds`, etc.)
        let mut tokens = Vec::new();
        for (_, _, cards_spec, w) in rows {
            if *w <= 0.0 {
                continue;
            }
            for sub in cards_spec.split(',') {
                let st = sub.trim();
                if !st.is_empty() {
                    tokens.push(format!("{st}@{w}"));
                }
            }
        }
        if tokens.is_empty() {
            return Ok(PloRange::new());
        }
        tokens
            .join(",")
            .parse::<PloRange>()
            .map_err(|e: RangeError| e.to_string())
    }
}

/// Parses `.txt`, `.plo`, `.csv`, or `.tsv` range files:
/// - Comment lines starting with `#` or `//` are ignored.
/// - CSV/TSV lines `combo,weight` or `combo\tweight` (with optional header `hand,weight` / `combo,freq`)
/// - Standard comma/newline-separated `PloRange` tokens (`AsKsQhJh@1.0, AAKK$ds:50%`).
fn parse_text_range_file(content: &str) -> Result<Option<PloRange>, String> {
    let mut cleaned_tokens: Vec<String> = Vec::new();

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("combo,")
            || lower.starts_with("hand,")
            || lower.starts_with("combo\t")
            || lower.starts_with("hand\t")
        {
            continue;
        }

        // Check two-column CSV/TSV `AsKsQhJh,0.75` or `AAKK$ds\t0.5` where second column is a bare float/percent
        if let Some((left, right)) = split_two_col_csv(line) {
            cleaned_tokens.push(format!("{left}@{right}"));
            continue;
        }

        for part in line.split(',') {
            let tok = part.trim();
            if !tok.is_empty() {
                cleaned_tokens.push(tok.to_string());
            }
        }
    }

    if cleaned_tokens.is_empty() {
        return Ok(None);
    }

    let expr = cleaned_tokens.join(",");
    let range = expr
        .parse::<PloRange>()
        .map_err(|e| format!("range parse error: {e}"))?;
    Ok(Some(range))
}

fn split_two_col_csv(line: &str) -> Option<(&str, &str)> {
    let sep = if line.contains('\t') { '\t' } else { ',' };
    let parts: Vec<&str> = line.split(sep).map(|s| s.trim()).collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return None;
    }
    let r = parts[1].strip_suffix('%').unwrap_or(parts[1]).trim();
    if r.parse::<f64>().is_ok() && !parts[0].contains('@') && !parts[0].contains(':') {
        Some((parts[0], parts[1]))
    } else {
        None
    }
}

// ============================================================================
// Precomputed Disjoint 133-Category Classifier for `JNandez Preflop Chart.xlsx`
// ============================================================================

/// Cached `270,725`-entry lookup table mapping every `ComboId` (`0..270_725`) to its JNandez category
/// `1..=133` (`0` for 3-of-a-kind / quads, which are excluded from the 133-row chart).
fn category_lookup_table() -> &'static [u8; NUM_PLO_COMBOS] {
    static TABLE: OnceLock<Box<[u8; NUM_PLO_COMBOS]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut arr = vec![0u8; NUM_PLO_COMBOS].into_boxed_slice();
        for (cid, hand) in Hand4::all_hands() {
            arr[cid.index()] = classify_nandez_category_133(hand);
        }
        arr.try_into().expect("exact NUM_PLO_COMBOS length")
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NandezSuitClass {
    Ds,
    Ss,
    Rr,
}

fn classify_nandez_category_133(hand: Hand4) -> u8 {
    let cards: [Card; 4] = hand.cards();
    let suit_pat = classify_suit_pattern(hand);
    let sb = match suit_pat {
        SuitPattern4::DoubleSuited => NandezSuitClass::Ds,
        SuitPattern4::SingleSuited | SuitPattern4::TripsSuited | SuitPattern4::Monotone => {
            NandezSuitClass::Ss
        }
        SuitPattern4::Rainbow => NandezSuitClass::Rr,
    };

    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    for c in cards {
        rank_counts[c.rank() as usize] += 1;
        suit_counts[c.suit() as usize] += 1;
    }

    let max_mult = *rank_counts.iter().max().unwrap_or(&0);
    if max_mult >= 3 {
        return 0; // Trips / quads are 0% across all 133 categories
    }

    let sa = cards
        .iter()
        .any(|c| c.rank() == 12 && suit_counts[c.suit() as usize] >= 2);
    let bway_suit = cards
        .iter()
        .any(|c| c.rank() >= 8 && suit_counts[c.suit() as usize] >= 2);

    let num_pairs = rank_counts.iter().filter(|&&cnt| cnt == 2).count();

    if num_pairs == 0 {
        // 4 distinct ranks sorted descending: r0 > r1 > r2 > r3
        let mut r = [0u8; 4];
        let mut idx = 0;
        for rk in (0..13u8).rev() {
            if rank_counts[rk as usize] == 1 {
                r[idx] = rk;
                idx += 1;
            }
        }
        let (r0, r1, r2, r3) = (r[0], r[1], r[2], r[3]);
        let bway = r.iter().filter(|&&x| x >= 8).count();
        let span = r0 - r3;

        if r0 == 12 {
            let base: u8 = if bway >= 3 && (r1 - r3 <= 3 || (r1 == 9 && r2 == 8 && (5..=6).contains(&r3))) {
                1 // Connected A-high broadway (AKQJ-AJT7)
            } else if bway >= 3 {
                2 // Semi-connected A-high broadway (AKQ9-AJT2)
            } else if (r1 - r3 <= 4 && r1 <= 10) || r1 <= 4 {
                3 // Connected A-high non-broadway (AQ98-A432)
            } else if bway == 2 && (r2 - r3 <= 3 || r1 - r2 <= 3) {
                4 // Disconnected A-high (2-card) broadway (AK98-AT32)
            } else {
                5 // Ragged A-high (AK96-A643)
            };

            match sb {
                NandezSuitClass::Ds => base,
                NandezSuitClass::Rr => 34 + base,
                NandezSuitClass::Ss => match (base, sa) {
                    (1, true) => 15,
                    (1, false) => 16,
                    (2, true) => 17,
                    (2, false) => 18,
                    (3, false) => 19,
                    (3, true) => 20,
                    (5, true) => 21,
                    (5, false) => 22,
                    (4, true) => 23,
                    (4, false) => 24,
                    _ => unreachable!(),
                },
            }
        } else {
            let base: u8 = if r0 >= 10 && span <= 4 {
                6 // Very connected K/Q-high broadway (KQJT-QT98)
            } else if (8..=9).contains(&r0) && span <= 4 {
                7 // Very connected mid-high (JT98-T876)
            } else if r0 <= 7 && (span <= 3 || (span == 4 && r2 - r3 == 1 && r0 - r1 == 1)) {
                8 // Very connected low (9876-5432)
            } else if bway >= 2 && (span <= 6 || r0 - r2 <= 3 || r1 - r3 <= 3) {
                9 // Connected near-broadway (KQJ8-QT85)
            } else if r0 >= 10 && bway == 1 && r1 - r3 <= 4 {
                10 // Connected 3-low with broadway kicker (K987-Q542)
            } else if (8..=9).contains(&r0) && span <= 6 {
                11 // Connected mid-high (JT96-T754)
            } else if r0 <= 7 && span <= 5 {
                12 // Connected low (9874-6432)
            } else if bway >= 2 {
                13 // Disconnected K/Q/J-high, 2 card broadway (KQ96-JT32)
            } else {
                14 // Ragged disconnected (K975-8432)
            };

            match sb {
                NandezSuitClass::Ds => base,
                NandezSuitClass::Rr => 34 + base,
                NandezSuitClass::Ss => match base {
                    6 => 25,
                    7 => 26,
                    8 => 27,
                    9 => 28,
                    10 => {
                        if bway_suit {
                            29
                        } else {
                            30
                        }
                    }
                    11 => 31,
                    12 => 32,
                    13 => 33,
                    14 => 34,
                    _ => unreachable!(),
                },
            }
        }
    } else if num_pairs == 1 {
        let mut pair_r = 0u8;
        let mut kickers = [0u8; 2];
        let mut ki = 0;
        for rk in (0..13u8).rev() {
            if rank_counts[rk as usize] == 2 {
                pair_r = rk;
            } else if rank_counts[rk as usize] == 1 {
                kickers[ki] = rk;
                ki += 1;
            }
        }
        let (k_hi, k_lo) = (kickers[0], kickers[1]);
        let has_a = pair_r == 12 || k_hi == 12;
        let mut all3 = [pair_r, k_hi, k_lo];
        all3.sort_unstable_by(|a, b| b.cmp(a));
        let conn3 = (all3[0] - all3[2] <= 4) || (has_a && k_hi <= 4 && pair_r <= 4);

        if pair_r == 12 {
            let is_conn = (k_hi - k_lo <= 3 && (k_hi >= 9 || k_hi <= 4)) || (k_hi - k_lo <= 2);
            match sb {
                NandezSuitClass::Ds => {
                    if is_conn { 49 } else { 50 }
                }
                NandezSuitClass::Rr => {
                    if is_conn { 96 } else { 97 }
                }
                NandezSuitClass::Ss => match (is_conn, sa) {
                    (true, true) => 70,
                    (true, false) => 71,
                    (false, true) => 72,
                    (false, false) => 73,
                },
            }
        } else if pair_r == 11 || pair_r == 10 {
            if k_hi == 12 {
                let is_conn = k_lo >= 6;
                match sb {
                    NandezSuitClass::Ds => {
                        if is_conn { 51 } else { 54 }
                    }
                    NandezSuitClass::Rr => {
                        if is_conn { 98 } else { 101 }
                    }
                    NandezSuitClass::Ss => match (is_conn, sa) {
                        (true, true) => 74,
                        (true, false) => 75,
                        (false, true) => 78,
                        (false, false) => 79,
                    },
                }
            } else if conn3 && k_lo >= 6 {
                match sb {
                    NandezSuitClass::Ds => 52,
                    NandezSuitClass::Ss => 76,
                    NandezSuitClass::Rr => 99,
                }
            } else if k_hi - k_lo <= 4 || k_hi >= 8 {
                match sb {
                    NandezSuitClass::Ds => 56,
                    NandezSuitClass::Ss => 82,
                    NandezSuitClass::Rr => 103,
                }
            } else {
                match sb {
                    NandezSuitClass::Ds => 57,
                    NandezSuitClass::Ss => 83,
                    NandezSuitClass::Rr => 104,
                }
            }
        } else if pair_r == 9 || pair_r == 8 {
            if k_hi == 12 {
                let is_conn = k_lo >= 6;
                match sb {
                    NandezSuitClass::Ds => {
                        if is_conn { 51 } else { 55 }
                    }
                    NandezSuitClass::Rr => {
                        if is_conn { 98 } else { 102 }
                    }
                    NandezSuitClass::Ss => match (is_conn, sa) {
                        (true, true) => 74,
                        (true, false) => 75,
                        (false, true) => 80,
                        (false, false) => 81,
                    },
                }
            } else if k_hi >= 10 && k_lo >= 6 {
                match sb {
                    NandezSuitClass::Ds => 52,
                    NandezSuitClass::Ss => 76,
                    NandezSuitClass::Rr => 99,
                }
            } else if conn3 && k_lo >= 4 {
                match sb {
                    NandezSuitClass::Ds => 53,
                    NandezSuitClass::Ss => 77,
                    NandezSuitClass::Rr => 100,
                }
            } else if k_hi - k_lo <= 4 || k_hi >= 8 {
                match sb {
                    NandezSuitClass::Ds => 58,
                    NandezSuitClass::Ss => 84,
                    NandezSuitClass::Rr => 105,
                }
            } else {
                match sb {
                    NandezSuitClass::Ds => 59,
                    NandezSuitClass::Ss => 85,
                    NandezSuitClass::Rr => 106,
                }
            }
        } else if (4..=7).contains(&pair_r) {
            if conn3 && k_hi != 12 {
                match sb {
                    NandezSuitClass::Ds => 60,
                    NandezSuitClass::Ss => 86,
                    NandezSuitClass::Rr => 107,
                }
            } else if k_hi == 12 && k_lo >= 8 {
                match sb {
                    NandezSuitClass::Ds => 62,
                    NandezSuitClass::Ss => 87,
                    NandezSuitClass::Rr => 109,
                }
            } else if k_hi >= 9 && k_lo >= 8 {
                match sb {
                    NandezSuitClass::Ds => 63,
                    NandezSuitClass::Ss => 88,
                    NandezSuitClass::Rr => 108,
                }
            } else if k_hi == 12 {
                match sb {
                    NandezSuitClass::Ds => 61,
                    NandezSuitClass::Ss => {
                        if sa { 89 } else { 90 }
                    }
                    NandezSuitClass::Rr => 108,
                }
            } else {
                match sb {
                    NandezSuitClass::Ds => 64,
                    NandezSuitClass::Ss => 90,
                    NandezSuitClass::Rr => 108,
                }
            }
        } else if conn3 && k_hi <= 6 {
            match sb {
                NandezSuitClass::Ds => 65,
                NandezSuitClass::Ss => 93,
                NandezSuitClass::Rr => 110,
            }
        } else if k_hi == 12 && k_lo >= 8 {
            match sb {
                NandezSuitClass::Ds => 66,
                NandezSuitClass::Ss => 91,
                NandezSuitClass::Rr => 111,
            }
        } else if k_hi >= 9 && k_lo >= 8 {
            match sb {
                NandezSuitClass::Ds => 67,
                NandezSuitClass::Ss => 92,
                NandezSuitClass::Rr => 112,
            }
        } else if k_hi == 12 {
            match sb {
                NandezSuitClass::Ds => 68,
                NandezSuitClass::Ss => 95,
                NandezSuitClass::Rr => 112,
            }
        } else {
            match sb {
                NandezSuitClass::Ds => 69,
                NandezSuitClass::Ss => 94,
                NandezSuitClass::Rr => 112,
            }
        }
    } else {
        // Two-pair (`num_pairs == 2`)
        let mut pairs = [0u8; 2];
        let mut pi = 0;
        for rk in (0..13u8).rev() {
            if rank_counts[rk as usize] == 2 {
                pairs[pi] = rk;
                pi += 1;
            }
        }
        let (p_hi, p_lo) = (pairs[0], pairs[1]);
        if p_hi == 12 {
            match sb {
                NandezSuitClass::Ds => 114,
                NandezSuitClass::Ss => 121,
                NandezSuitClass::Rr => 128,
            }
        } else if p_hi >= 9 && p_lo >= 8 {
            match sb {
                NandezSuitClass::Ds => 115,
                NandezSuitClass::Ss => 122,
                NandezSuitClass::Rr => 129,
            }
        } else if p_hi >= 10 && p_lo <= 7 {
            match sb {
                NandezSuitClass::Ds => 116,
                NandezSuitClass::Ss => 123,
                NandezSuitClass::Rr => 130,
            }
        } else if (6..=9).contains(&p_hi) && p_lo >= 5 && p_hi - p_lo <= 2 {
            match sb {
                NandezSuitClass::Ds => 117,
                NandezSuitClass::Ss => 127,
                NandezSuitClass::Rr => 131,
            }
        } else if (8..=9).contains(&p_hi) && p_lo <= 4 {
            match sb {
                NandezSuitClass::Ds => 118,
                NandezSuitClass::Ss => 124,
                NandezSuitClass::Rr => 133,
            }
        } else if p_hi <= 6 && p_lo >= 1 && p_hi - p_lo <= 2 {
            match sb {
                NandezSuitClass::Ds => 119,
                NandezSuitClass::Ss => 125,
                NandezSuitClass::Rr => 132,
            }
        } else {
            match sb {
                NandezSuitClass::Ds => 120,
                NandezSuitClass::Ss => 126,
                NandezSuitClass::Rr => 133,
            }
        }
    }
}
