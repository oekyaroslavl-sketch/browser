# PLO-4 Postflop — Hero vs Range (Desktop Architecture & Range Pack Specification)

## 1. Product Model (`Hero vs Range`)

Unlike laboratory symmetric range-vs-range solvers (`BTN range` vs `BB range`), **`PLO-4 Postflop — Hero vs Range`** is built around how a real PLO-4 player studies a postflop hand:

- **Hero**: Exact **4 hole cards** (`PloRange` singleton of weight `1.0`, e.g., `AsKsQhJh`). Hold'em 2-card inputs (`AKs`, `AA`, `AsKs`) are strictly rejected.
- **Seat**: Hero chooses **`OOP`** (`OOP = Hero, IP = Villain`) or **`IP`** (`OOP = Villain, IP = Hero`).
- **Villain**: Assumed preflop **`PloRange`** loaded from a user-owned preflop range pack folder (defaulting to a mapped preflop spot dropdown) or entered manually when `"Use pack"` is unchecked.
- **Exact Dead-Card Blocker Removal**: Both the postflop board (`3`, `4`, or `5` cards) and Hero's `4` hole cards (`dead = board.mask() | hero.mask()`) filter Villain's range before equity calculation, memory estimation, or game-tree construction (`live_villain = raw_villain.without_dead(dead)`).
- **Frozen Solver Core**: Showdowns use exact `C(4,2) × C(5,3) = 60` two-plus-three evaluation (`plo_eval`), terminal fold/showdown nodes use exact 15-term 4-card inclusion-exclusion (`plo_cfr`), chance nodes use exact `52 - |board| - 8` combinatorics (`41` on Flop→Turn, `40` on Turn→River), and DCFR runs natively on a background Rust worker thread (`std::thread::spawn` + `AtomicBool` cancellation + progress callback). No `wasm32` runtime is embedded in the desktop application.

---

## 2. Phase 0 Inspection of Purchased Nandez Range Pack (`.local/nandez_pack/`)

The Google Drive folder `1AY1JudavsMQMO8H9joFUyJiZ0C6uDOq5` (`preflop nandez`) was inspected and downloaded exclusively into gitignored `.local/nandez_pack/` (never committed, bundled, or redistributed).

### 2.1 Directory & Workbook Inventory

Top-level Drive folder contents:
- `JNandez Preflop Chart.xlsx` (`234,963` bytes, OOXML `.xlsx` archive containing `xl/workbook.xml`, `xl/sharedStrings.xml`, and `xl/worksheets/sheet1.xml` .. `sheet12.xml`).

Workbook sheets (`12` sheets total):
1. `All categories overview` — Master reference table of `133` disjoint PLO-4 hand categories (`NR 1..133`) with equity/playability metrics.
2. `RFI Overview` — Summary matrix of `UTG`, `MP`, `CO`, `BTN` open-raise frequencies across the `133` categories.
3. `RFI Profiles` — Population tendency lookup table (non-range metadata; ignored by importer).
4. `RFI EP` — Early Position (`UTG`) RFI chart (`~11.37%` sheet weighted frequency).
5. `RFI MP` — Middle Position (`MP`) RFI chart (`~14.11%`).
6. `RFI CO` — Cutoff (`CO`) RFI chart (`~25.75%`).
7. `RFI BTN` — Button (`BTN`) RFI chart (`~52.15%`).
8. `BB Defense` — Two range columns: `BB vs 1 Raiser` (`~56.07%`) and `BB vs Raiser + Callers` (`~41.82%`).
9. `3-Betting IP` — Two range columns: `Vs. 15-20%` (`~5.01%`) and `Vs. 25-35%` (`~8.18%`).
10. `3-Betting OOP` — Four range columns: `Vs. 30% in BB` (`~5.98%`), `Vs. 30% in SB` (`~8.35%`), `Vs. 60% in BB` (`~12.17%`), `Vs. 60% in SB` (`~17.01%`).
11. `Coldcalling IP` — Three range columns: `Vs. 15% in MP/CO` (`~6.73%`), `Vs. 15% on BTN` (`~9.88%`), `Vs. 30% on BTN` (`~16.47%`).
12. `Preflop 3-Bet Calc` — EV calculator worksheet (non-range metadata; ignored by importer).

### 2.2 Exact Cell Schema (`15`-Row Structural Sample)

Every range worksheet (`RFI EP`, `RFI MP`, `RFI CO`, `RFI BTN`, `BB Defense`, `3-Betting IP`, `3-Betting OOP`, `Coldcalling IP`, plus `RFI Overview`) shares the exact same row layout (`Row 1` = header, `Rows 2..134` = categories `NR 1..133`):

| Row | Col `A` (`NR`) | Col `B` (`Type`) | Col `C` (`Type`) | Col `D` (`Text`) | Col `E` (`Cards`) | Col `F` (`% of total`) | Spot Freq Col (`0.0..=1.0`) |
|---:|---:|---|---|---|---|---:|---:|
| 2 | `1` | `unp-ds` | `A-high` | `Connected A-high broadway` | `AKQJ-AJT7` | `0.0016` | `1` |
| 4 | `3` | `unp-ds` | `A-high` | `Connected A-high non-broadway` | `AQ98-A432` | `0.0039` | `0.25` |
| 7 | `6` | `unp-ds` | `Very connected` | `Very connected K/Q-high broadway` | `KQJT-QT98` | `0.0011` | `1` |
| 16 | `15` | `unp-ss` | `A-high` | `Connected A-high broadway, suited ace` | `AKQJ-AJT7` | `0.0050` | `1` |
| 17 | `16` | `unp-ss` | `A-high` | `Connected A-high broadway, unsuited ace` | `AKQJ-AJT7` | `0.0037` | `1` |
| 30 | `29` | `unp-ss` | `Connected` | `Connected 3-low with broadway kicker, broadway suit` | `K987-Q542` | `0.0150` | `0` |
| 36 | `35` | `unp-rr` | `A-high` | `Connected A-high broadway` | `AKQJ-AJT7` | `0.0011` | `0` |
| 50 | `49` | `1pr` | `1p-ds` | `Connected Aaxx` | `AAKQ-AA43` | `0.0010` | `1` |
| 71 | `70` | `1pr` | `1p-ss` | `Connected AAxx, suited ace` | `AAKQ-AA43` | `0.0051` | `1` |
| 97 | `96` | `1pr` | `1p-rr` | `Connected Aaxx` | `AAKQ-AA43` | `0.0010` | `1` |
| 115 | `114` | `2pr` | `2p-ds` | `Doublepaired Aces` | `AAKK-AA22` | `0.0003` | `1` |
| 122 | `121` | `2pr` | `2p-ss` | `Doublepaired Aces` | `AAKQ-AA22` (`AAKK-AA22`) | `0.0011` | `1` |
| 129 | `128` | `2pr` | `2p-rr` | `Doublepaired Aces` | `AAKK-AA22` | `0.0003` | `1` |

Key structural facts verified directly from the `.xlsx` XML:
1. **Suit Partitioning**:
   - `*-ds`: Double-suited `(2, 2)` (`36` combos per unpaired 4-rank set).
   - `*-ss`: Single-suited `(2, 1, 1)` plus 3-flush `(3, 1)` and 4-flush `(4)` (`144 + 48 + 4 = 196` combos per unpaired 4-rank set; verified by exact `112 : 84 = 4 : 3` suited-Ace vs unsuited-Ace ratio in `NR 15..24`, e.g., `1.61% : 1.21%`, `1.24% : 0.93%`, `4.80% : 3.60%`).
   - `*-rr`: Rainbow `(1, 1, 1, 1)` (`24` combos per unpaired 4-rank set).
2. **Weights**: Each spot column stores a fractional weight `w ∈ [0.0, 1.0]` (`0`, `0.05`, `0.1`, `0.25`, `0.33`, `0.5`, `0.75`, `1`), paired with an adjacent summary column `= $F{row} * {Col}{row}` summed in `Row 1` (`=SUM(H2:H134)`).
3. **Additional Supported Text/CSV Formats**: Because users may also export custom `.txt` / `.plo` / `.csv` files into their pack folder (e.g., Monker/PPT comma-separated `AsKsQhJh@0.5, AAKK$ds:75%` or `combo,weight` lines), `RangeImporter` supports **both**:
   - Native OOXML `.xlsx` category-chart workbooks (`JNandez Preflop Chart.xlsx` and custom/fixture `.xlsx` workbooks with either `NR 1..133` category rows or explicit PLO range/macro tokens in `Cards`), **and**
   - Recursive `.txt`, `.plo`, `.csv`, `.tsv` range files inside the selected pack directory.

---

## 3. UI Group → `spot_id` → `.xlsx` Sheet/Column Mapping & Smart Seat Defaults

`RangeImporter` exposes **all 15 canonical preflop spots** from `JNandez Preflop Chart.xlsx` grouped into 5 structured `<optgroup>` categories (plus a 6th `"Other / unmapped files"` group for any additional sheets, columns, or `.txt/.plo/.csv` files). If a canonical spot is absent from a partial user pack, its slot stays visible and disabled (`"— not in pack"`) rather than silently dropping the group. Note that `JNandez Preflop Chart.xlsx` has no SB RFI worksheet, so none is invented.

| UI Group | `spot_id` | UI Display Label | `.xlsx` Sheet / Column in `JNandez Preflop Chart.xlsx` |
|---|---|---|---|
| **1. `RFI (open)`** | `RFI_UTG` (alias `RFI_EP`) | `RFI · UTG` | `RFI EP :: UTG (Col G)` |
| **1. `RFI (open)`** | `RFI_MP` | `RFI · MP` | `RFI MP :: MP (Col H)` |
| **1. `RFI (open)`** | `RFI_CO` | `RFI · CO` | `RFI CO :: CO (Col H)` |
| **1. `RFI (open)`** | `RFI_BTN` | `RFI · BTN` | `RFI BTN :: BTN (Col H)` |
| **2. `BB defense`** | `BB_DEF_VS_1_RAISER` | `BB vs 1 raiser` | `BB Defense :: BB vs 1 Raiser (Col G)` |
| **2. `BB defense`** | `BB_DEF_MULTIWAY` | `BB multiway (vs raiser + callers)` | `BB Defense :: BB vs Raiser + Callers (Col I)` |
| **3. `3-bet IP`** | `THREEBET_IP_VS_15_20` | `3-bet IP vs 15–20% open` | `3-Betting IP :: Vs. 15-20% (Col G)` |
| **3. `3-bet IP`** | `THREEBET_IP_VS_25_35` | `3-bet IP vs 25–35% open` | `3-Betting IP :: Vs. 25-35% (Col I)` |
| **4. `3-bet OOP`** | `THREEBET_OOP_BB_VS_30` | `3-bet OOP · BB vs 30% open` | `3-Betting OOP :: Vs. 30% in BB (Col G)` |
| **4. `3-bet OOP`** | `THREEBET_OOP_SB_VS_30` | `3-bet OOP · SB vs 30% open` | `3-Betting OOP :: Vs. 30% in SB (Col I)` |
| **4. `3-bet OOP`** | `THREEBET_OOP_BB_VS_60` | `3-bet OOP · BB vs 60% open` | `3-Betting OOP :: Vs. 60% in BB (Col K)` |
| **4. `3-bet OOP`** | `THREEBET_OOP_SB_VS_60` | `3-bet OOP · SB vs 60% open` | `3-Betting OOP :: Vs. 60% in SB (Col M)` |
| **5. `Cold call IP`** | `CC_IP_MP_CO_VS_15` | `Cold call IP · MP/CO vs 15%` | `Coldcalling IP :: Vs. 15% in MP/CO (Col G)` |
| **5. `Cold call IP`** | `CC_IP_BTN_VS_15` | `Cold call IP · BTN vs 15%` | `Coldcalling IP :: Vs. 15% on BTN (Col I)` |
| **5. `Cold call IP`** | `CC_IP_BTN_VS_30` | `Cold call IP · BTN vs 30%` | `Coldcalling IP :: Vs. 30% on BTN (Col K)` |
| **6. `Other / unmapped files`** | `RAW::<file>::<sheet>::<col>` / `RAW::<file>` | `<raw sheet/file title>` | Any extra `.xlsx` weight column or `.txt/.plo/.csv` file |

### Smart Default by Hero Seat & Persistence

Villain's range always represents the **opponent's** preflop line in the hand (never Hero's own chart):
- **Hero `OOP` (e.g. defending BB)** → smart default Villain spot is **`RFI_BTN`** (the opponent opened).
- **Hero `IP` (e.g. BTN/CO opener)** → smart default Villain spot is **`BB_DEF_VS_1_RAISER`** (the BB opponent defends).
- The user can freely override to **any** of the 15 canonical spots or extra files (3-bet pots, cold-call pots, EP/MP/CO opens, multiway BB calls).
- User settings (`pack_folder`, custom `spot_bindings`, and `last_spot_id`) persist to OS application config JSON (`$XDG_CONFIG_HOME/plo_desktop/settings.json` on Linux, `%APPDATA%\plo_desktop\settings.json` on Windows, `~/Library/Application Support/plo_desktop/settings.json` on macOS).

---

## 4. Architecture & Auto Guardrails

```text
crates/plo_cards, plo_eval, plo_iso, plo_range, plo_tree, plo_cfr   [Frozen verified core]
       │
       ▼
crates/plo_desktop_commands   [Pure-Rust headless command & state layer — NO poker math]
  ├── importer.rs             [OOXML .xlsx + .txt/.csv/.plo RangeImporter -> PloRange + spot bindings]
  ├── settings.rs             [OS app-data JSON persistence + SolveConfig save/load]
  └── lib.rs                  [DesktopEngine: parse_hero/board, compute_equity, estimate_spot,
                               Auto abstraction selector, worker thread + AtomicBool stop,
                               progress callback, tree cursor navigation, collapsed 2-range Lab mode]
       │
       ▼
apps/plo_desktop              [Desktop UI + Tauri 2 shell & local preview bridge]
  ├── src-tauri/              [Tauri 2 Windows/cross-platform Rust entrypoint + `"solve-progress"` emitter]
  └── ui/                     [Single-screen dark dense Hero-vs-Range interface]
```

### Desktop System-RAM Budget Policy (vs `512 MiB` WASM Guard)

- **`512 MiB` is strictly a WASM/browser linear-memory guard (`plo_wasm`)**: The native desktop solver (`plo_desktop_commands` + `apps/plo_desktop`) removes the `512 MiB` product ceiling and uses machine RAM.
- **Desktop Default Budget**: `min(50% of detected physical RAM, 16 GiB)`, floored at `4 GiB` (`4096 MiB`) whenever physical RAM `>= 8 GiB` (else `50%` of RAM, or `8192 MiB` fallback if OS RAM cannot be queried). Users can enter any MiB or GiB budget freely (e.g. `2048` … `65536` or `16 GiB`), persisted in `settings.json`, and displayed as `"Est. 69 MiB / Budget 16 GiB"`.
- **Mandatory Pre-Build Estimate & Honest Runtime Expectations**:
  - **More RAM ≠ instant full-pack Flop (`GTOWizard`)**: Unfolded `41 × 40` Flop→Turn→River chance trees remain CPU/traversal-bound on wide PLO-4 ranges regardless of how many gigabytes of RAM are available.
  - Every `Estimate` and `Solve` call runs `estimate_memory_for_config` **before** allocating `PloGame`. If `estimated > budget`, desktop refuses cleanly with `MemoryBudgetExceeded` (showing estimated MiB/GiB, budget, and actionable hints: `NutBlocker`, tighter Villain spot, or Turn/River) without crashing.
  - **River (`5` cards)**: `Auto` prefers `IdentityAbstraction` (`1,000` iters, `< 30 MiB` even vs `100,000+` live combos).
  - **Turn (`4` cards)**: `Auto` uses `Identity` up to `150` live combos (if within budget), switching to `NutBlocker k=50` (`iso = false`) above `150` live combos.
  - **Flop (`3` cards)**: `Auto` **never** blindly starts `Identity` on fat ranges (e.g. `500 vs 500`) just because desktop RAM is `16 GiB`; it uses `Identity` only on compact spots (`<= 150` live combos and `<= budget`) and otherwise selects `NutBlocker k=50` (`iso = false`), warning above `300` live combos. Explicit `Identity` mode is honored whenever `estimate <= budget` (e.g. `500 vs 500` `Identity` at `4096 MiB` passes cleanly).

---

## 5. Legal & Repository Hygiene Rules

1. **Purchased Range Packs Are Never Distributed**: `.gitignore` excludes `/.local/`, `/nandez_pack/`, and `*Nandez*.xlsx`. Neither git commits nor Tauri installer resources (`tauri.conf.json`) ever include `.local/` or user range files.
2. **Synthetic Test Fixtures Only**: Automated CI tests run against `tests/fixtures/fake_nandez/` (`Fake_Preflop_Chart.xlsx` + `custom_spot.txt`), which uses the exact OOXML `.xlsx` worksheet/cell schema with **only 6 invented combos** so every CI build verifies `.xlsx` parsing, spot mapping, and dead-card blocker removal without proprietary data.
3. **Optional Local Smoke Test**: Integration test `local_real_nandez_pack_parses_if_present` checks `.local/nandez_pack` and skips cleanly (`#[ignore]` + runtime existence check) when absent.

---

## 6. Building & Running

### Headless Command & Golden Verification (All Platforms)
```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
```

### Desktop UI (Tauri 2 Native App / Local Preview Server)
- **Windows 10/11 (Tauri 2 WebView2 native `.exe` / `.msi` installer)**:
  ```powershell
  cd apps/plo_desktop/src-tauri
  cargo tauri dev
  # Or build installer (never bundles range packs):
  cargo tauri build
  ```
- **Linux (Debian/Ubuntu) Prerequisites for Tauri 2 Native Window**:
  ```bash
  sudo apt-get update && sudo apt-get install -y \
    libwebkit2gtk-4.1-dev build-essential curl wget file \
    libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
  ```
- **Zero-Dependency Local Desktop Bridge (`plo_desktop_preview`)**:
  In headless Linux environments where `webkit2gtk` headers are not installed, `crates/plo_desktop_commands` also powers a local HTTP/SSE bridge serving the exact `apps/plo_desktop/ui/` frontend over `0.0.0.0`:
  ```bash
  cargo run -p plo_desktop_commands --bin plo_desktop_preview --release
  ```

---

## 7. Trainer, Analysis & Persistence Suite (Phases 1–3) & Release Pipeline

### 7.1 Interactive Study & Exploitative Tools
- **15-Category PLO-4 Classifier & PPT/Monker Filter (`classifier.rs`)**: Breaks down Villain's reaching combos into 15 PLO-specific 2+3 made-hand and draw buckets (`Quads / Full House+`, `Flush`, `Straight`, `Set / Trips`, `Two Pair`, `Combo Draw`, `Nut Flush Draw`, `Wrap (9+ Outs)`, `Bare Nut Blocker`, `Air`, etc.) with live query filtering (`AA`, `$ds`, `As`, `nfd`, `wrap`, `set`, `nutblk`).
- **Uniform & Per-Hand-Class Node Locking**: `lock_node` and `lock_node_by_classes` (`PloGame::lock_current_node_by_combo`) freeze selected action distributions on any decision node and re-solve counter-exploitative strategies around them.
- **Exact Subgame Resolving (`resolve_subgame` / `exit_subgame`)**: Rebuilds any Turn/River street root from a bucketed solve into a zero-bucket `Identity` subgame using exact Bayesian combo reaches, with non-destructive return to the parent tree.
- **4×13 Runout EV Heatmap, 52-Card Blocker Matrix & Equity Quintiles**: Visualizes Hero EV across all Turn/River runouts, measures how each of the 52 deck cards blocks Villain's Value / Total / Air reaches, and plots 5-quintile EV polarization.
- **Hero Suit & Nut-Out Matrix (`compute_hero_nut_out_matrix`)**: Evaluates every live unseen Turn/River card via `plo_eval::eval_plo` against Villain's reaching range to separate Hero's outs into **1st Nut Outs**, **2nd Nut Outs**, and **Dominated Outs**.

### 7.2 Hand Audit, Solver Bot & Strategy Simplification
- **Hand History Parser & Full Hand EV Audit (`hh_parser.rs`, `audit_hand_history`)**: Parses PokerStars, GGPoker, and CoinPoker 4-card PLO hand histories, auto-selects Villain's preflop chart spot, and walks Flop → Turn → River to grade every Hero action (`Optimal` / `Inaccuracy` / `Blunder`) with `ΔEV` in chips and `bb`.
- **Play vs GTO Solver Bot (`start_bot_hand`, `bot_play_hero_action`, `stop_bot_hand`)**: Samples a live non-overlapping 4-card Villain hand from root weights, plays Villain actions strictly from that combo's solved GTO distribution, deals random live Turn/River cards at `Chance` nodes, grades Hero's decisions, and reveals Villain's 2+3 hand at showdown.
- **Auto-Simplify Pure-Size Optimizer (`compare_bet_simplifications`) & Multi-Flop Subset Report (`run_flop_subset_report`)**: Benchmarks single bet sizes (`33%`–`100%`) against multi-size baselines (`EV Retention %`) and runs batch `FlopOnly` C-bet/EV reports across representative flop textures.

### 7.3 Binary `.plotree` Snapshots (`PLOTREE1` + `PLOSNAP1`)
- `save_tree_snapshot` and `load_tree_snapshot` serialize `SpotConfig` together with `PloGame::export_weights_binary()` (`regrets`, `cum_strategy`, and `locked_strategy` buffers) into compact `.plotree` files (gitignored via `*.plotree`) that reload in milliseconds without re-running DCFR.

### 7.4 Keyboard Shortcuts
- `1`–`5`: Choose action `1..5` at the active decision node
- `Backspace` / `←`: Step back one node in the tree history
- `R`: Reset cursor to tree root
- `M`: Toggle interactive horizontal Street Tree Minimap
- `F`: Focus the PPT/Monker Villain range filter box

### 7.5 GitHub Actions CI & Cross-Platform Release Installers
- **`.github/workflows/ci.yml`**: Runs on every push and PR — enforces zero proprietary range packs in Git, `cargo clippy --workspace --all-targets -- -D warnings`, `wasm32-unknown-unknown` build, `node --check apps/plo_desktop/ui/app.js`, golden tests (`G1–G4`, `D1–D2`), headless desktop suite (`t1–t18`), and Linux `webkit2gtk-4.1` Tauri 2 compile check.
- **`.github/workflows/release.yml`**: Triggered on `v*.*.*` tags or manual `workflow_dispatch` — verifies `bundle.resources == []` and builds native Tauri 2 installers across **Windows (`x64 .msi` / `-setup.exe`)**, **macOS (`Universal .dmg`)**, and **Linux (`x64 .AppImage` / `.deb`)**.
