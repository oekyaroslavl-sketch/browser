# plo-postflop — PLO-4 Rust → WebAssembly Postflop Solver Core

`plo-postflop` is a dedicated 4-card **Pot-Limit Omaha (`PLO-4`)** postflop solver core written in Rust (`wasm32-unknown-unknown` ready, `AGPL-3.0-or-later`). Every Texas Hold'em assumption (`1,326` two-card combos, 7-card "best 5 of 7" evaluation, 2-card blocker subtraction, and unconditional No-Limit overbet shoves) is absent: all evaluation uses **exact 2 hole cards + 3 board cards** (`6` flop, `24` turn, `60` river 5-card candidates per hand) and **exact 15-term 4-card inclusion-exclusion** at terminal nodes.

---

## 1. Hand Indexing (`ComboId`) & How to Write PLO-4 Ranges

Every 4-card hand `Hand4` (`c0 < c1 < c2 < c3` in `0..52`) maps bijectively to a canonical **`ComboId` in `0 .. 270,725`** ($\binom{52}{4} = 270{,}725$). Two-card Hold'em tokens (`"AA"`, `"AKs"`, `"AKo"`, `"AsKs"`) are rejected at parse time.

`PloRange::from_str` and JS `solver.setRange(player, rangeStr)` accept comma-separated tokens (left-to-right, last-write-wins) with optional `@weight` (`0.0..=1.0`) or `:pct%` (`0%..=100%`):

| Syntax | Example | Live Preflop Combos | Meaning |
|---|---|---:|---|
| Exact 4-card hand | `AsKsQhJh`, `Ah9s6d5d@0.5` | `1` | Exact 4 distinct cards (`@0.5` or `:50%` sets weight; `@0` excludes) |
| 4-rank macro | `AAKK`, `AKQJ` | `36` / `256` | All valid 4-card suitings in a 52-card deck |
| Suit-filtered macro | `AAKK$ds`, `AKQJ$ds` | `6` / `36` | `$ds` double-suited (`2-2`), `$ss` single-suited (`2-1-1`), `$rb` rainbow (`1-1-1-1`), `$ts` trips-suited (`3-1`), `$ms` monotone (`4-0`) |
| Suited rank groups | `(AK)(QJ)`, `(AK)QJ` | `12` / `36` | Ranks inside `(...)` share one suit; distinct `(...)` groups use distinct suits; ungrouped ranks do not share any group's suit |
| Full deck | `*` or `all` | `270,725` | All 4-card combinations |

---

## 2. Pot-Limit Default, Precision (`f32` Train vs `f64` Report) & Isomorphism

- **Pot-Limit Default & Bet Fractions (`plo_tree`)**:
  - `BettingLimit::PotLimit` is the default everywhere. Opening bets are capped at `min(pot, stack)` (`100%` pot max — e.g. `effectiveStack = 200` into `startingPot = 100` caps `100%` at `Bet:100`, never `200`). Raises are capped at `min(opp_invested + pot_after_call, stack)`.
  - Street bet/raise strings accept comma-separated pot percentages or fractions (e.g. `"50%, 100%"`, `"25%, 50%"`, `"0.33, 0.75, 1.0"`).
- **Training (`f32`, DCFR Variant) vs Reported EV (`f64`)**:
  - DCFR training (`solve_step` / `solve` / `stepIterations`) stores regrets, cumulative strategies, and traversal buffers in **`f32`** (`8` bytes per `(action, bucket)` pair) for cache throughput and compact WASM linear memory, using a variant in the spirit of Brown & Sandholm's `DCFR(α = 3/2, β = 0, γ = 2)`: positive regrets scale by $t^{1.5}/(t^{1.5}+1)$, negative regrets scale by `0.0` (CFR+-style reset), and cumulative strategy sums scale by $\gamma_t = ((t-1)/t)^2$ before adding the current strategy (yielding exact quadratic $(t/T)^2$ iteration weights, vs $(t/(t+1))^\gamma$ in Brown & Sandholm 2019).
  - Under bucketed `HandAbstraction`, instantaneous counterfactual regrets are aggregated across live combos in each bucket normalized by Hero's conditional bucket reach $\sum_{i \in b} \pi_p(h_i)$ (`1` division per bucket, `0/0` guarded; zero-weight combos filtered out at build time). When $k \ge N$ (1 combo per bucket), bucketed DCFR matches `IdentityAbstraction` trajectory-for-trajectory; for lossy $k < N$, conditional weights depend on $\sigma^t$ across imperfect-recall nodes (empirically converging to the abstraction's Nash-gap floor, without a theoretical convergence guarantee).
  - Expected game value reporting (`PloGame::root_expected_values_f64()` and `queryNode().rootEv` in WASM) traverses the average strategy in **`f64`** using exact `1.0f64 / 41.0` (flop→turn) and `1.0f64 / 40.0` (turn→river) conditional card probabilities (`52 - |board| - 8`), matching `equity_range_vs_range * pot` on check-down trees to **`<= 1e-9`**.
- **Chance Suit Isomorphism (`plo_iso`)**:
  - Enabled by default (`enable_chance_isomorphism = true`), losslessly folding suit-equivalent turn/river runouts while weighting each canonical class by `|live_cards_in_class| / (52 - |board| - 8)` (never `1 / num_canonical_cards`).
  - **Automatic Fallback**: Isomorphism folds two suits **only** when swapping those suits leaves both the board and **both players' weighted ranges** invariant (and `abstraction.is_suit_symmetric() == true`). Asymmetric ranges (e.g. `AsKsQhJh` singleton or lexicographic combo prefixes) or rainbow boards (`Ks9d4c`) automatically leave asymmetric suits unfolded with zero loss of exactness.

---

## 3. WASM Memory Guard (`512 MiB`) & Flop Scalability Numbers

`SolverBuildOptions::default()` and `WasmPloSession` enforce **`DEFAULT_MAX_MEMORY_BYTES = 512 MiB` (`536,870,912 B`)** *before* allocating tree nodes (`Err(SolverError::MemoryBudgetExceeded)` on overflow; configurable via `solver.setMemoryBudgetMb(mb)`).

Measured `estimate_memory_for_config` / `memory_report()` figures (`Pot = 100, Stack = 100`, 2 sizes `50%, 100%` + `100%` raise, including realistic `192 B` concrete-node overhead):
- **River `200 vs 200` (`Identity`, `iso = false`)**: `8` decision nodes — **`40,832 B` (`39.9 KiB`)**.
- **Turn `50 vs 50` (`Identity`, `iso = false`)**: `776` decision nodes — **`1,157,120 B` (`1.10 MiB`)**.
- **Flop Tiny `20 vs 20` (`Identity`, `iso = true`, two-tone `Ks9s4d`, 0.5 SPR 2 sizes)**: `34,880` decision nodes — runs in Node/WASM (`js/example_flop_tiny.mjs`) in `~31 MiB` linear memory without OOM.
- **Flop `50 vs 50` (`Identity`, two-tone `Ks9s4d`)**: `iso = false` → `57,240` nodes, **`82,614,272 B` (`78.79 MiB`)**; `iso = true` → `34,880` nodes, **`50,441,040 B` (`48.10 MiB`)**.
- **Flop `500 vs 500` (`Identity`)**:
  - `iso = false`: `57,240` decision nodes, `496,352,000 B` (`473.36 MiB`) regret+strat, **`542,031,872 B` (`516.92 MiB`)** total (`>= 500 vs 500` without iso trips the `512 MiB` WASM guard).
  - `iso = true` (two-tone `Ks9s4d` + suit-symmetric ranges): `34,880` decision nodes, **`331,316,640 B` (`315.97 MiB`)** total (`>= 900 vs 900` trips the `512 MiB` guard; trips `MemoryBudgetExceeded` at `128 MiB`).
  - With `50` buckets (`NutBlockerBucketAbstraction` `(25, 2)` / `setAbstractionBuckets(50, 50, 50)`), regret/strategy RAM drops `10x` to **`49,635,200 B` (`47.34 MiB`)**, giving **`88,809,472 B` (`84.70 MiB` est / `83.84 MiB` live)** at `200 vs 200` and **`100,219,072 B` (`95.58 MiB`)** at `500 vs 500` (`iso = false`), while wide `5k–50k` PLO flop ranges remain out of single-tab WASM reach without depth-limited turn/river value functions.

### Practical flop (native)

For hundreds-of-combos flop solves (`200..=500` live combos per side), run native Rust with `NutBlockerBucketAbstraction` (`crates/plo_cfr/examples/bench_flop_buckets.rs`), which clusters decision regrets/strategies into `k = 50` (`(25, 2)`) or `k = 100` (`(50, 2)`) 2D `(strength, nut_ceiling + bare_nut_flush_blocker)` buckets (deduplicated per `(board, player)` via `Arc<[u32]>`) while keeping **combo-exact reaches, turn/river card removal, and 15-term 4-card inclusion-exclusion showdowns** (with pooled thread-local `InclusionExclusionScratch` buffers eliminating per-branch heap/`mmap` churn):
- **Measured (`200 vs 200` live combos on `Ks9s4d`, `256 MiB` guard)**:
  - **`k = 50` (`(25, 2)`)**: `83.84 MiB` live (`172.16 MiB` headroom), **`1.21 iters/s`** (`2.2×` faster than `v0.1.0`), exploitability `2.498e1` (`0` iters) → `3.241e-1` (`100` iters) → **`3.274e-1` chips (`0.33%` pot abstraction floor at `500` iters)**, `root EV = [60.5355, 39.4645]`.
  - **`k = 100` (`(50, 2)`, `--all`)**: `131.18 MiB` live, **`1.14 iters/s`**, exploitability **`1.000e-1` chips (`0.10%` pot at `500` iters)**, `root EV = [60.5084, 39.4916]`.
  - **`Identity` (`200 vs 200`, `--all`)**: `223.98 MiB` live, **`1.12 iters/s`**, exploitability `8.471e-3` (`100` iters) → **`6.562e-4` chips (`0.00066%` pot at `500` iters)**, `root EV = [60.4685, 39.5315]` (`+0.067` chips `k = 50` EV gap vs exact `Identity`).
- **WASM Policy**: Bucketed `200 vs 200` flop (`84.70 MiB > 64 MiB`, single-threaded) is **native-only** (`HARDENING.md §7.4`); use `js/example_flop_tiny.mjs` (`20 vs 20`) in WASM/browsers.

---

## 4. Regression Golden Suite, Benchmarks & Commands

`tests/golden/golden_spots.rs` (`cargo test -p plo_cfr --test golden --release`) freezes deterministic external regression goldens (zero randomness):
- **`G1`**: River no-betting hand-vs-hand (`EV = equity * pot` against `brute_showdown_ev`: nuts vs PLO bare-suit air, true Broadway chop, and 1-card blocker pair `== 0.0` mass).
- **`G2` & `G4`**: Polar River betting equilibrium (`KsQsJs7d2c`, `Pot = 100, Stack = 100`, `100%` pot bet, `1000` fixed DCFR iterations: `EV = [75.0, 25.0] ± 0.05`, bluff/call `0.50 ± 0.02`, cross-checked against both `js/example_river.mjs` and brute-enumerated pure-strategy best response).
- **`G3`**: Turn 2-vs-2 check-down on two-tone `AsKs7c2c` (`PloGame::root_expected_values_f64() == equity_range_vs_range * pot` within `1e-9`, `iso = true` and `iso = false`).
- **Real-Range Macro Equity Goldens (`D1`)**: `AAKK$ds` vs `(AK)(QJ)` on flop `QsJh4c` (`5302 / 14760`) and unpaired rainbow flop `7c3d2h` (`27459 / 29520` via `slow_eval_plo_lex` only), both verified to `1e-12`.
- **Richer River Bet + Raise / Probe Matrix Goldens (`D2`, `tests/golden/d2_river_raise_or_probe.rs`)**: `29`-card blocker-free `2.0` SPR Pot-Limit spots (`Pot = 100, Stack = 200`, `Bet(100)`, `AllIn(200)` raise, `Probe(100)`) comparing closed-form/matrix Nash (`EV = [200/9, 700/9]` and `[650/9, 250/9]`) against `2,000` DCFR iterations.
- **Flop Tiny BR vs DCFR Check (`D3`)**: `20 vs 20` Flop Tiny (`FIXED_FLOP_TINY_ITERS = 100` in native & WASM, `1000` iters for `BR_value - EV == exploitability`).

```bash
# Lint, full release test suite (including regression goldens), and WASM build
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
cargo build -p plo_wasm --target wasm32-unknown-unknown --release

# Native & WASM examples / microbenchmarks (JS API in js/plo_postflop.mjs)
node js/example_river.mjs
cargo run -p plo_cfr --example bench_flop_tiny --release
cargo run -p plo_cfr --example bench_flop_buckets --release
node js/example_flop_tiny.mjs
```

---

## 5. Desktop — Hero vs Range (`apps/plo_desktop` & `crates/plo_desktop_commands`)

On top of the frozen solver core (separate from the `plo_wasm` / `js/*` browser demos — **no `wasm32` is embedded in desktop**), the workspace ships **`PLO-4 Postflop — Hero vs Range`** (`DESKTOP.md`):

- **Dense `wasm-postflop`-Inspired 3-Column Layout**: Full-window dark (`~12px`, monospace numbers) solver workspace adapted specifically to PLO-4 Hero-vs-spot (no invalid Hold'em `13×13` grids): left column (`~320px`) for Spot/SPR/sizes/abstraction/telemetry, center column for rendered board cards, action/chance navigation, EV, and Hero combo frequencies, and right column (`~360px`) for Villain combo/bucket detail (capped at `30` rows + `+N more`).
- **Product Model & Full 15-Spot Villain Selector**: Hero enters **exact 4 PLO hole cards** (`PloRange` singleton weight `1.0`; Hold'em 2-card tokens like `AKs`, `AA`, `AsKs` are rejected) and seat (`OOP` or `IP`). You then pick the **one Nandez preflop line that matches Villain's action in THIS hand** from the 5-group dropdown — **RFI (open)** (`UTG`, `MP`, `CO`, `BTN`), **BB defense** (`vs 1 raiser`, `multiway`), **3-bet IP**, **3-bet OOP**, or **Cold call IP** (plus any extra unmapped files; smart defaults select `RFI · BTN` when Hero is OOP and `BB vs 1 raiser` when Hero is IP, fully overridable and persisted in `settings.json`). The app **never** solves two full preflop ranges against each other (`RFI` vs `BB defense`).
- **Legal Notice — Range Pack Not Included**: Purchased range packs (e.g., Nandez preflop charts) are **NOT** included in git, CI fixtures, or the Tauri installer (`bundle.resources = []`). Users point Settings to their own local pack folder; automated CI tests run exclusively on the 7-combo synthetic fixture in `tests/fixtures/fake_nandez/`.
- **Desktop System-RAM Policy (vs `512 MiB` WASM Cap)**:
  - **`plo_wasm` / JS examples** keep the conservative `512 MiB` browser guard unchanged, whereas **Desktop (`plo_desktop_commands` + Tauri)** uses machine RAM: default budget `= min(50% of system RAM, 16 GiB)` with a `4 GiB` floor when RAM `>= 8 GiB` (or `8192 MiB` fallback if unknown), freely configurable in MiB/GiB (`2048` … `65536` / `16 GiB`) and persisted in `settings.json` (`"Est. 69 MiB / Budget 16 GiB"`).
  - **Honest Runtime Reality (More RAM ≠ `GTOWizard` Full-Range Flop)**: Raising RAM lets larger trees fit in memory (always checked via `estimate_memory_for_config` before building; overflows return a clean `MemoryBudgetExceeded` hint without crashing), but unfolded `41 × 40` Flop→Turn→River chance trees remain CPU-bound on wide `BTN vs BB` ranges. `Auto` still prefers `Identity` on River, switches to `NutBlocker k=50` (`iso = false`) on wide Turn/Flop spots, and never auto-starts `Identity` on fat `500 vs 500` flops unless explicitly selected by the user.

### Commercial PLO-4 Study & Trainer Suite (Phases 1–3)

- **15-Category PLO-4 Hand & Draw Classifier + PPT/Monker Filter (`classifier.rs`)**: Real-time 2+3 made-hand and draw classification (`Quads/FH+`, `Flush`, `Straight`, `Set/Trips`, `Two Pair`, `Combo Draw`, `Nut FD`, `Wrap 9+ Outs`, `Bare Nut Blocker`, `Air`, etc.) with live PPT/Monker filter syntax (`AA`, `$ds`, `As`, `nfd`, `wrap`, `set`, `nutblk`).
- **Node Locking (Uniform & Per-Hand-Class) + Counter-Exploitative Re-Solve**: Lock action frequencies uniformly or selectively per PLO hand class (`🔒 Lock Node`), re-solve DCFR around locked nodes, and unlock individual or all nodes.
- **4×13 Runout EV Heatmap, 52-Card Blocker Matrix & Equity Quintiles**: Color-coded Turn/River chance runout heatmap (`Hero EV` & `% pot`), 52-card blocker impact matrix (`Blocks Value % / Total % / Air %`), and 5-quintile Villain EV/equity distribution.
- **Exact Turn/River Subgame Resolving & Interactive Street Tree Minimap**: One-click upgrade (`⚡ Exact Subgame`) from bucketed parent solves into zero-bucket `Identity` Turn/River subgames using exact Bayesian reaches, plus a keyboard-driven street minimap (`1–5`, `⌫/←`, `R`, `M`, `F`).
- **Pot Rake (`%` + Cap) & Target `dEV % Pot` Auto-Stop + Live SVG Convergence Chart**: Configurable showdown pot rake (`rake_pct`, `rake_cap`), early stopping once exploitability reaches `target_dev_pct`, and real-time log-scale `dEV % Pot` convergence chart.
- **Hand History Parser & Full Hand EV Audit (`📋 Paste HH` → `🔍 Full Hand EV Audit`)**: Parses PokerStars / GGPoker / CoinPoker 4-card PLO hand histories, maps Villain's preflop spot, and audits every postflop Hero decision across Flop, Turn, and River (`Optimal` / `Inaccuracy` / `Blunder` + `ΔEV` in chips & `bb`).
- **Play vs GTO Solver Bot (`🤖 Play vs Bot`) & Drill Mode (`🎯 Drill Mode`)**: Interactive full-hand playout against a sampled live Villain 4-card hand acting strictly from its solved GTO strategy across live Turn/River runouts with 2+3 showdown reveal and `ΔEV` grading.
- **Auto-Simplify Pure-Size Optimizer (`⚡ Bet Sizes`) & Multi-Flop Subset Report (`📊 Flop Subset`)**: Benchmarks candidate single bet sizes (`33%`–`100%`) against multi-size baselines (`EV Retention %`), and runs batch `FlopOnly` C-bet/EV reports across flop textures.
- **Saved Spots Library, CSV/Markdown Export & Binary `.plotree` Snapshots (`📚 Library`)**: Bookmarks spots locally, exports node tables to CSV/Markdown, and serializes solved `PloGame` trees (`PLOTREE2` + `PLOSNAP2` with 64-bit structural config hash + finite-float validation; legacy `PLOTREE1`/`PLOSNAP1` snapshots are explicitly rejected) to `.plotree` binary snapshots for millisecond reloads without re-solving.
- **Hero Suit & Nut-Out Matrix (`🧬 Hero Suit & Nut-Out Matrix`)**: Evaluates all unseen deck cards on Flop/Turn against Villain's live range to partition Hero's improving cards into **1st Nut Outs**, **2nd Nut Outs**, and **Dominated Outs**.

```bash
# Headless desktop + range-importer + Phase 1-3 verification suite (t1..t18)
cargo test -p plo_desktop_commands --release

# Launch native Tauri 2 Windows/macOS/Linux desktop app
cd apps/plo_desktop/src-tauri && cargo tauri dev

# Or launch the zero-dependency native HTTP/SSE desktop UI preview on 127.0.0.1:4173 (pass --host 0.0.0.0 in isolated containers)
cargo run -p plo_desktop_commands --bin plo_desktop_preview --release
```

---

## 6. Explicitly Not Implemented (Out of Scope)

The following features are **not implemented**:
- Preflop solving & folded-hand **bunching** (only the single-street conditional denominator formula `52 - |board| - 8 - |dead|` is noted in `HARDENING.md`)
- **Multiway** (`>= 3` players) postflop solving
- `i16` regret compression

