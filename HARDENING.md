# HARDENING.md — PLO-4 Solver Core Verification, Invariants & WASM Memory Contract

## 1. Crate Map & Public APIs (A0 Inventory)

| Crate | File | Public API Surface | Responsibility |
|---|---|---|---|
| `plo_cards` | `crates/plo_cards/src/lib.rs` | `Card`, `CardMask`, `Hand4`, `ComboId` (`0..270_725`), `Board`, `Street`, `parse_cards`, `pair_subset_index`, `triple_subset_index`, `BINOM`, `NUM_PLO_COMBOS` | 52-card encoding (`4 * rank + suit`), 64-bit dead-card masks, strictly sorted 4-card Omaha hands, `C(52,4)` combinatorial bijection. Zero Hold'em 2-card hand types exist. |
| `plo_eval` | `crates/plo_eval/src/lib.rs` | `HandRank` (`1..=7462`), `HandCategory`, `eval_plo`, `eval_plo_flop` (6), `eval_plo_turn` (24), `eval_plo_river` (60), `RiverBoardCache`, `showdown_heads_up`, `showdown_multiway`, `slow_eval_plo_lex`, `slow_eval_5_lex_key`, `eval_5_cards`, `init_tables` | Dedicated PLO 2-hole + 3-board evaluator (`~96 KB` lookup tables) + pure table-free reference oracle. |
| `plo_iso` | `crates/plo_iso/src/lib.rs` | `SuitPattern4`, `classify_suit_pattern`, `permute_card`, `permute_hand4`, `permute_board`, `canonicalize_preflop_hand`, `canonicalize_board`, `canonicalize_hand_with_board`, `suits_symmetric_on_board`, `compute_chance_isomorphism`, `build_combo_swap_pairs` | 24-permutation $S_4$ suit canonicalization for `Hand4` (16,432 classes), joint `(Hand4, Board)` canonicalization, and lossless chance-node suit folding. |
| `plo_range` | `crates/plo_range/src/lib.rs` | `PloRange`, `WeightedHand`, `EquitySummary`, `equity_hand_vs_hand`, `equity_hand_vs_range`, `equity_range_vs_range`, `per_combo_river_equity` | Sparse sorted 4-card weighted ranges, PLO range grammar parser, `O(1)` bitmask blocker filtering, exact runout equity engine. |
| `plo_tree` | `crates/plo_tree/src/lib.rs` | `BettingLimit` (`PotLimit` default), `Action`, `StreetBetSizes`, `TreeConfig`, `ActionTree`, `ActionNode`, `ActionNodeKind`, `PLAYER_OOP`, `PLAYER_IP` | Pot-Limit Omaha bet/raise sizing (`max_open = min(pot, stack)`, `max_raise_to = min(opp_inv + pot_after_call, stack)`), raise caps, donk options, street/chance skeleton. |
| `plo_cfr` | `crates/plo_cfr/src/lib.rs` | `PloGame`, `HandAbstraction`, `IdentityAbstraction`, `EquityBucketAbstraction`, `NutBlockerBucketAbstraction`, `SolverBuildOptions`, `MemoryReport`, `NodeReport`, `HistoryStep`, `estimate_memory_for_config`, `ShowdownMassBreakdown` | `DCFR(3/2, 0, 2)` solver, 4-card 15-term inclusion-exclusion terminal evaluator, best-response exploitability, chance-card navigation, abstraction hooks. |
| `plo_wasm` | `crates/plo_wasm/src/lib.rs`, `js/plo_postflop.mjs` | `WasmPloSession` + C-ABI exports (`plo_set_board`, `plo_set_range`, `plo_set_tree_config`, `plo_build_game`, `plo_solve`, `plo_step_iterations`, `plo_exploitability`, `plo_play_action`, `plo_play_chance`, `plo_query_node_json`, `plo_compute_equity_json`) | Zero-proc-macro `wasm32-unknown-unknown` binary + frozen JS wrapper `PloPostflopWasm`. |

---

## 2. Exact Code Paths Audited in Phase A0

1. **Where Showdown EV is Computed**:
   - `crates/plo_cfr/src/lib.rs` → `PloGame::eval_terminal_showdown` (and `PloGame::eval_showdown_mass_breakdown`).
   - Uses `RiverEvalData::sorted_strengths` (ascending `(HandRank, combo_idx)` built by `ConcreteBuilder::get_or_build_river_eval` using `plo_eval::RiverBoardCache`).
   - Centered zero-sum payoff around `0.5 * starting_pot`:
     - Let `pot = starting_pot + 2 * committed` (`paid_oop == paid_ip == committed` at showdown).
     - Win payoff: `+(0.5 * starting_pot + committed)` (`chip EV = pot - committed = starting_pot + committed`).
     - Tie payoff: `0.0` (`chip EV = 0.5 * pot - committed = 0.5 * starting_pot`).
     - Lose payoff: `-(0.5 * starting_pot + committed)` (`chip EV = -committed`).
   - **Audit Finding on Ties (`A0/A2`)**: Because tie centered payoff is `0.0`, the previous 2-pass `eval_terminal_showdown` computed `win_mass - lose_mass` without ever computing `tie_mass` directly — meaning a bug where `win_mass` and `lose_mass` both shifted by `+\Delta` on overlapping tie hands would **cancel out in `EV`**! We therefore expose and test `ShowdownMassBreakdown` (`compatible_mass`, `win_mass`, `tie_mass`, `lose_mass`, `ev`) via 15-term IE (`tie_mass = compatible_mass - win_mass - lose_mass` AND independent `<= strength` sweep `le_mass - win_mass`) against `brute_showdown_ev`, asserting exact agreement on **all five quantities** (`win`, `tie`, `lose`, `compatible_mass`, `EV`)!

2. **Where Fold Terminal Uses 15-Term IE**:
   - `crates/plo_cfr/src/lib.rs` → `PloGame::eval_terminal_fold`.
   - Inserts all unblocked opponent combos weighted by `opp_cfreach[j]` into `InclusionExclusionScratch` (`total`, `w1[52]`, `w2[1326]`, `w3[22100]`), then queries each unblocked hero combo via `scratch.query_disjoint(&sub, opp_cfreach)`:
     $$W_{\text{disjoint}}(h) = W_\emptyset - \sum_{|S|=1} W_1[S] + \sum_{|S|=2} W_2[S] - \sum_{|S|=3} W_3[S] + W_4[h]$$
   - Note: `eval_terminal_fold` and `eval_terminal_showdown` had a `<= 512` small-range fast path that bypassed `InclusionExclusionScratch` when `N * M <= 512`! Any fixture with `< 22` combos per side previously **never exercised the 15-term IE code path at all**!
   - **Hardening Fix**: Extract dedicated `eval_fold_ie` and `eval_showdown_ie_breakdown` kernels that run unconditionally in `A2` unit tests even on `1 vs 1` and `4 vs 4` fixtures so small hand-crafted overlap/chop fixtures directly test the 15-term IE math.

3. **Where Chance Cards Are Enumerated & Weighted (`C1` Dealing Policy)**:
   - `crates/plo_cfr/src/lib.rs` → `ConcreteBuilder::expand_node` (`ActionNodeKind::Chance`), `dcfr_traverse`, `strategy_cfv_traverse`, `best_response_traverse`, and `play_chance_card`.
   - **Dealing Policy**:
     - Every turn/river chance node deals **losslessly from the `K = 52 - |board|` remaining deck cards** (`K = 49` on flop→turn, `K = 48` on turn→river).
     - Wait — let's check the combinatorial chance factor very carefully!
       In a heads-up PLO game on a board of size $B \in \{3, 4\}$:
       - Player 0 holds 4 cards (not on board), Player 1 holds 4 cards (disjoint from board and from Player 0!).
       - Conditional on **any specific compatible pair** $(h_0, h_1)$ (`|board| + 4 + 4 = B + 8` dead cards!), how many legal next-street cards exist in the physical deck?
         - Flop ($B = 3$): $52 - 3 - 4 - 4 = \mathbf{41}$ legal turn cards (NOT 49!).
         - Turn ($B = 4$): $52 - 4 - 4 - 4 = \mathbf{40}$ legal river cards (NOT 48!).
       - **CRITICAL DISCOVERY (`C1/C2`)**:
         Look at what `dcfr_traverse` / `strategy_cfv_traverse` / `best_response_traverse` was doing at a Chance node:
         It multiplied `opp_cfreach` by `inv_chance = 1.0 / total_branches` where `total_branches = 52 - B` (`48` on turn, `49` on flop)!
         Let's check whether `1 / 48` vs `1 / 40` matters!
         Suppose on the Turn ($B = 4$), OOP checks, IP can either:
         - **Action A (Bet & win a fold immediately on the Turn)** $\implies$ terminal fold evaluated ON THE TURN (before the river chance node!), which gets weight `1.0` per compatible pair $(h_0, h_1)$!
         - **Action B (Check back to River -> Showdown on all legal rivers)** $\implies$ goes through the River Chance node! For any specific compatible pair $(h_0, h_1)$, `8` of the `48` river branches hit a card in $h_0 \cup h_1$ (and are zeroed out by card removal!), leaving **only `40` active river branches** for $(h_0, h_1)$!
         - If each river branch is weighted by `1 / 48` instead of `1 / 40`, then the sum of weights across all `40` legal rivers for $(h_0, h_1)$ is **`40 / 48 = 5/6` instead of `1.0`**!!
         - That shrinks all River counterfactual values by `40/48 = 0.8333` relative to Turn fold counterfactual values — a **16.7% mathematical distortion between Turn folds and River showdowns** (and on the Flop, `(41/49) * (40/48) = 0.697` — a **30.3% distortion**)!
         - Look back at upstream Hold'em `postflop-solver/src/game/base.rs` line 55:
           ```rust
           fn chance_factor(&self, node: &Self::Node) -> usize {
               if node.turn == NOT_DEALT {
                   45 - self.bunching_num_dead_cards // 52 - 3 (flop) - 2 (OOP) - 2 (IP) = 45!
               } else {
                   44 - self.bunching_num_dead_cards // 52 - 4 (turn) - 2 (OOP) - 2 (IP) = 44!
               }
           }
           ```
           In Hold'em, upstream divided by `52 - B - 2 - 2 = 45` (flop) and `44` (turn) because each player has **2** hole cards!
           In PLO-4, each player has **4** hole cards, so for every compatible pair $(h_0, h_1)$ (`|h_0| = 4`, `|h_1| = 4`, `h_0 \cap h_1 = \emptyset`), there are **always** `52 - B - 8` remaining deck cards:
           - **Flop → Turn chance factor in PLO-4**: `52 - 3 - 8 = 41`!
           - **Turn → River chance factor in PLO-4**: `52 - 4 - 8 = 40`!
         - Let's verify this identity! For **every** non-overlapping pair $(h_0, h_1)$ on a 4-card Turn board:
           $|B| = 4, |h_0| = 4, |h_1| = 4$, pairwise disjoint $\implies |B \cup h_0 \cup h_1| = 12 \implies$ **every single valid pair $(h_0, h_1)$ is live on EXACTLY $52 - 12 = 40$ river cards**, regardless of what ranks/suits $h_0$ and $h_1$ hold!
           And on a 3-card Flop board: $|B \cup h_0 \cup h_1| = 11 \implies$ **every single valid pair $(h_0, h_1)$ is live on EXACTLY $52 - 11 = 41$ turn cards**!
         - Catching this in Phase A0/C1 is a **textbook example** of why Phase C's "identity turn solver EV == brute terminal EV" oracle is critical! With `total_branches = 48`, a turn check-down tree would return `40/48` of the true EV! With `chance_card_denominator = 52 - board.len() - 8` (`41` on flop, `40` on turn), the sum over all `52 - board.len()` dealt branches has `40` (resp. `41`) surviving branches per valid pair $(h_0, h_1)$, summing to **exact `40 * (1/40) = 1.0`**!

4. **How Suit Isomorphism Remaps Hole Cards With Board (`B1`)**:
   - `crates/plo_iso/src/lib.rs`:
     - `canonicalize_hand_with_board(hand, board)`: enumerates all $24$ suit bijections $\pi \in S_4$, maps `(flop_set, turn_opt, river_opt)` and `hand` simultaneously by $\pi$, and selects the lexicographically minimum `(board_canonical_key, permuted_hand)`.
     - `compute_chance_isomorphism(board, oop_combos, ip_combos)`: merges two suits `s1 < s2` at a chance node iff swapping `s1 <-> s2` fixes `board_canonical_key(board)` AND preserves both players' weighted combo lists.
     - **Audit Finding on `play_chance_card` + Isomorphism (`B4`)**:
       - First, when `HandAbstraction` is non-identity (`EquityBucketAbstraction` / `NutBlockerBucketAbstraction`), bucket assignments on a representative branch need not commute with `build_combo_swap_pairs` unless buckets are suit-canonical; we ensure chance isomorphism is only active when exact `IdentityAbstraction` is used (or suit-symmetric).
       - Second, `play_chance_card(card)` previously rejected isomorphic duplicate cards (only accepting `rep_cards`), making `queryNode` fail if a user dealt e.g. `2h` when `2d` was the representative! We will support dealing **any** legal remaining deck card in `play_chance_card` by tracking an active suit permutation on the cursor and remapping combos/strategies/EVs transparently!

5. **How Buckets Map `combo -> bucket` & How Reach Is Stored (`C3`)**:
   - `reach[player]` is always stored at **combo granularity** (`len == combos[player].len()`).
   - At a player node with `combo_to_bucket: Vec<u32>`, combo `i` looks up `b = combo_to_bucket[i]` to read `strategy[a * num_buckets + b]`, multiplies `reach[player][i] *= strategy[a, b]`, and passes exact per-combo `reach` to children and terminals.
   - At chance nodes, combos overlapping the dealt card have `reach[player][i] = 0.0` immediately.
   - **Audit Finding on `my_bucket_reach` / Instantaneous Regret Weighting (`C3`)**:
     - In `dcfr_traverse`, instantaneous bucket regret accumulated `self.initial_weights[player][i] * (cfv[a][i] - node_cfv[i])` instead of weighting by `my_reach[i]`! If combo `i1` and combo `i2` share a bucket on the Turn, or if earlier actions/card removals zeroed `my_reach[i1]` (for instance if a combo was blocked on the dealt turn card — wait: `board_mask` filtered board-blocked combos, what if two combos in a bucket had different initial weights or different upstream reach from an earlier street where they were in *different* flop buckets?!). Using `my_reach[i]` (normalized within the bucket or reach-weighted `my_reach[i] * inst`) properly reflects conditional reach at that infoset! Wait: in standard imperfect-recall / abstracted CFR, counterfactual regret at bucket $B$ is weighted by hero's reach probability $\pi_p^\sigma(h_i, z)$ normalized across the bucket (or `my_reach[i] / my_bucket_reach[b]` when `my_bucket_reach[b] > 0`, falling back to `initial_weights[player][i]`), so a combo that took an action with `0%` frequency on the Flop does NOT pollute a Turn bucket's regrets! That's another subtle abstraction bug caught during A0!

6. **Audit of Pot-Limit Tree (`D`)**:
   - `crates/plo_tree/src/lib.rs`:
     - When `effective_stack < starting_pot` (e.g. `pot = 100, stack = 40`) and `bet_fractions = [0.5, 1.0]`:
       `max_legal_to = min(100, 40) = 40`. Both `0.5 * 100 = 50` and `1.0 * 100 = 100` clamp to `40`, deduplicate to `[40]`, and emit `[Action::Check, Action::AllIn(40)]` — no phantom `Bet(50)` or `Bet(100)`.
     - Wait: what if `bet_fractions = []` (empty) and `add_allin_spr_threshold` triggers? We should test empty/custom fractions, `stack < pot`, `call < stack < pot_raise`, `stack <= call` (when `street_max == opp_inv`, opponent's bet already puts player all-in on a call — verify `Call` transitions to `is_allin = true` and NO `Raise` is generated!).
     - Wait: check `BettingLimit::NoLimitApprox`: let's ensure `PotLimit` is strictly default and `WasmPloSession` never exposes NL overbets unless explicitly opted in.

7. **Audit of Previous Tests That Could Pass Even If Showdown/Chance Were Wrong**:
   - `river_plo_polar_vs_bluffcatcher_nash_equilibrium` had `2 vs 1` combos (`2 * 1 = 2 <= 512`) → hit the small-range bitmask loop, bypassing 15-term IE, and had zero ties!
   - `exact_4card_15term_inclusion_exclusion_matches_pairwise_bruteforce` only asserted `(ie_sd - bf_sd).abs() < 1e-5` on net EV, NOT `win_mass`, `tie_mass`, `lose_mass` separately!
   - `turn_solver_with_chance_isomorphism_matches_unfolded_turn_solver` compared `game_iso` against `game_full`, which **both** shared the `1 / 48` (instead of `1 / 40`) chance denominator bug, so they agreed with each other while both under-weighting river showdowns vs turn folds!

---

## 3. Formal Specifications for Phases B, C, E, G

### 3.1 Board-Aware Suit Canonicalization Algorithm (`B1`)
Given `hand: Hand4` and `board: Board` ($B \in \{3, 4, 5\}$ cards):
1. Represent `board` street-aware as `(flop_sorted_3, turn_opt, river_opt)` so flop card permutation is invariant while street order (`flop` vs `turn` vs `river`) is strictly preserved.
2. Enumerate all $4! = 24$ suit bijections $\pi \in S_4$:
   - Compute $\pi(\text{board}) \mapsto \text{key}_\pi \in \mathbb{N}^5$ and $\pi(\text{hand}) \mapsto h_\pi \in \text{Hand4}$ (re-sorted ascending).
3. Select $\pi^* = \arg\min_{\pi \in S_4} (\text{key}_\pi, h_\pi)$ lexicographically.
4. Output `(h_{\pi^*}, board_{\pi^*}, \pi^*)`.
Because $\pi$ is applied **simultaneously** to every card of `board` and `hand`, two pairs `(h_1, B_1)` and `(h_2, B_2)` collide **if and only if** they are in the exact same $S_4$-orbit (identical flop suit multiplicities, identical turn/river suit relations, identical hole suit pairings, and identical hole-to-board suit incidences and blockers).

### 3.2 Exact Chance Measure, Isomorphism Suit-Class Weighting & Precision (`Step 0` / `C1`)

1. **Exact Conditional Next-Card Probability (`HOLE_CARDS_BOTH_PLAYERS = 8`)**:
   - In `dcfr_traverse`, `strategy_cfv_traverse` (`f32` and `strategy_cfv_traverse_f64`), and `best_response_traverse`, `chance_denominator(board_len)` is defined via `const HOLE_CARDS_BOTH_PLAYERS: usize = 8` (`52 - |B| - HOLE_CARDS_BOTH_PLAYERS`):
     $$P(c \mid B, h_0, h_1) = \begin{cases} \frac{1}{52 - 3 - 8} = \frac{1}{41} & \text{if } |B| = 3 \text{ (Flop} \to \text{Turn) and } c \notin B \cup h_0 \cup h_1 \\ \frac{1}{52 - 4 - 8} = \frac{1}{40} & \text{if } |B| = 4 \text{ (Turn} \to \text{River) and } c \notin B \cup h_0 \cup h_1 \\ 0 & \text{if } c \in B \cup h_0 \cup h_1 \end{cases}$$
   - For every compatible pair $(h_0, h_1)$ ($h_0 \cap B = \emptyset$, $h_1 \cap B = \emptyset$, $h_0 \cap h_1 = \emptyset$), $|B \cup h_0 \cup h_1| = |B| + 8$, so enumerating all $52 - |B|$ off-board cards yields **exact** $52 - |B| - 8$ live cards (`41` on flop, `40` on turn), and $\sum_{c} P(c \mid B, h_0, h_1) = 1.0$ identically (`step1_chance_successor_probabilities_sum_to_one_with_and_without_isomorphism` verifies `|sum - 1.0| <= 1e-12`).
   - *Note on bunching (not implemented, out of scope)*: with $d = |\text{dead}|$ known folded/bunching cards disjoint from $B \cup h_0 \cup h_1$, the conditional denominator generalizes to $52 - |\text{board}| - 8 - |\text{dead}|$.

2. **How Chance Isomorphism Weights Canonical Suit Classes (NEVER `1 / num_canonical_cards`)**:
   - Suppose a board + ranges have unequal suit-equivalence class sizes (e.g., on two-tone flop `KsQs4d`, spade and diamond turn cards each form size-1 classes `{c}`, while heart and club turn cards pair into size-2 classes `{r_h, r_c}` with `36` representative cards out of `49` remaining deck cards).
   - `PloGame` **never** divides by `rep_cards.len()` (`1 / 36` would catastrophically misweight size-1 spade/diamond classes vs size-2 heart/club classes!).
   - Instead, **every** representative branch `b` scales `opp_cfreach` by the **uniform physical card probability** `1 / DENOM` (`1/41` or `1/40`) and traverses its subtree **once**:
     - The representative card `rep_cards[b]` itself contributes `1 / DENOM` (if live for $(h_0, h_1)$).
     - Each folded duplicate card `dup` in `iso_duplicates` pointing to `rep_branch_idx == b` applies its exact combo-index transposition `dup.combo_swap_pairs[player]` (`s_rep <-> s_dup`, swapping `h` and `c` in hero's hand so card-removal checks `c_dup \notin h_0` and `c_dup \notin h_1` are exact!) and adds another `1 / DENOM` for each live card in the suit class!
     - Consequently, a canonical suit class $C$ contributes $\frac{|C \setminus (h_0 \cup h_1)|}{\text{DENOM}}$ — **mathematically identical** to summing $\frac{1}{\text{DENOM}}$ over all $52 - |B|$ unfolded cards (`step1_tight_checkdown_oracle_fixtures_a_through_e_within_1e9` verifies `iso = true` vs `iso = false` agree to `<= 1e-12` on unequal-class two-tone flops and turns!).

3. **Behavior When `abstraction.is_suit_symmetric() == false`**:
   - In `PloGame::with_options` and `estimate_memory_for_config`, `effective_iso = options.enable_chance_isomorphism && abstraction.is_suit_symmetric()`.
   - Whenever an abstraction is not suit-symmetric (`is_suit_symmetric() == false`), chance isomorphism is automatically disabled (`effective_iso == false`), expanding all $52 - |B|$ (`49` flop / `48` turn) concrete branches so `combo_swap_pairs` can never collide with asymmetric bucket IDs. (`IdentityAbstraction`, `EquityBucketAbstraction`, and `NutBlockerBucketAbstraction` are all suit-symmetric after canonical `ComboId` ordering.)

4. **Why Earlier Check-Down Tests Used `1e-4` Tolerance & How `1e-9` Was Achieved (`Step 1`)**:
   - During DCFR training (`solve_step`), counterfactual reaches and values are stored in `f32` (`7` significant decimal digits, machine epsilon `~1.19e-7`) for cache bandwidth and WASM memory compactness. Multiplying `opp_cfreach: Vec<f32>` by `1.0f32 / 41.0` (`0.024390243f32`) and summing `41 * 40 = 1,640` `f32` runout leaves accumulates `~1e-6` relative (`~1e-4` chip on a `100` pot) `f32` rounding noise — **not** any mathematical or algorithmic approximation!
   - Adding `PloGame::root_expected_values_f64()` (`strategy_cfv_traverse_f64` + `eval_terminal_showdown_f64` + `eval_terminal_fold_f64` + `InclusionExclusionScratch::query_disjoint_f64`, using exact `1.0f64 / (denom as f64)` without altering the `f32` DCFR training kernel) eliminated all `f32` accumulation error: all 5 check-down fixtures (`A`–`E`) and `phase_c2` turn/flop check-down trees now match `plo_range::equity_range_vs_range * starting_pot` within **`<= 1e-9` relative and `<= 1e-8` absolute** (actual observed error `< 1e-13`)!

### 3.3 Frozen PLO Range Grammar (`E`)
Tokens are comma-separated (`tok_1, tok_2, ...`), evaluated left-to-right with **last-write-wins** weight semantics (`@w` in `0.0..=1.0` or `:p%` in `0%..=100%`; `@0` removes combos):
1. `XcYdZhWs` (8 chars, 4 distinct valid cards): exact `Hand4` singleton (`1` combo).
2. `R1R2R3R4` (4 rank chars, e.g. `AAKK`, `AKQJ`, `AAAK`): all valid 4-card suit assignments in a standard deck (`C(4,2)*C(4,2) = 36` for `AAKK`; `4^4 = 256` for unpaired `AKQJ`; `C(4,3)*4 = 16` for `AAAK`; `1` for `AAAA`).
3. `R1R2R3R4$<mod>` suit filter on 4-rank macros:
   - `$ds`: `SuitPattern4::DoubleSuited` (`2-2-0-0`, e.g. `AAKK$ds` = `6` combos, `AKQJ$ds` = `36` combos).
   - `$ss`: `SuitPattern4::SingleSuited` (`2-1-1-0`, e.g. `AAKK$ss` = `24` combos, `AKQJ$ss` = `144` combos).
   - `$rb` (alias `$np`, `$rainbow`): `SuitPattern4::Rainbow` (`1-1-1-1`, e.g. `AAKK$rb` = `6` combos, `AKQJ$rb` = `24` combos).
   - `$ts`: `SuitPattern4::TripsSuited` (`3-1-0-0`, e.g. `AKQJ$ts` = `48` combos; error on `AAKK$ts` since impossible).
   - `$ms` (alias `$mono`): `SuitPattern4::Monotone` (`4-0-0-0`, e.g. `AKQJ$ms` = `4` combos; error on paired ranks).
   - Note: For `AKQJ`, `36 (ds) + 144 (ss) + 24 (rb) + 48 (ts) + 4 (ms) = 256` exact partition!
4. Parenthesized suited groups (`(R1R2)(R3R4)` or `(R1R2)R3R4`, 1 or 2 groups):
   - Ranks inside the same `(...)` MUST have the **same** suit.
   - Ranks in different `(...)` groups MUST have **different** suits.
   - Ungrouped ranks MUST NOT share a suit with any `(...)` group (so `(AK)QJ` means `A,K` share suit $s_0$ and `Q,J` both have suits $\neq s_0$, yielding $4 \times 3 \times 3 = 36$ combos: $12$ double-suited `(AK)(QJ)` + $24$ single-suited where `Q,J` have distinct non-$s_0$ suits).
5. `*` or `all`: all `270,725` combos.

---

## 4. Concrete WASM Memory Table: Iso OFF vs Iso ON & Honest Flop Limits (`Step 3` / `G1`)

Formula implemented by `plo_cfr::estimate_memory_for_config` (verified node-for-node and byte-for-byte against `PloGame::memory_report()` in `phase_g1_g3_concrete_memory_table_guardrail_and_determinism`):
$$\text{Bytes}_{\text{regret+strat}} = \sum_{n \in \mathcal{N}_{\text{dec}}, |A_n| > 1} |A_n| \times B_{p(n), s(n)} \times 8\text{ bytes}$$
$$\text{Bytes}_{\text{total}} = \text{Bytes}_{\text{regret+strat}} + |\mathcal{R}_{\text{rivers}}| \times (N_{\text{OOP}} + N_{\text{IP}}) \times 12 + |\mathcal{N}_{\text{all}}| \times 96$$

*Why board texture + range suit-symmetry govern `iso = true` vs `iso = false`*:
- On a **rainbow** flop (`Ks9d4c`), 3 distinct ranks occupy 3 suits `{s, d, c}` and only 1 suit (`h`) is absent, so zero suit permutations fix the board (`49` turn branches $\to$ `2,352` river subtrees under both `iso = false` and `iso = true`).
- On a **two-tone** flop (`Ks9s4d`, ~55% of all PLO flops) with suit-symmetric ranges, unrepresented suits `{h, c}` swap cleanly (`36` representative turns $\to$ `1,429` representative river subtrees and `799` unique river tables instead of `2,352` subtrees and `1,176` tables — a **39.1% exact lossless RAM reduction**!).

Concrete unit-tested rows (`Pot = 100, Stack = 100`, Pot-Limit `50%, 100%` bet / `100%` raise, `DEFAULT_MAX_MEMORY_BYTES = 512 MiB = 536,870,912 B`):

| Configuration | Street & Board | Iso Flag | Combos / Buckets (OOP + IP) | Decision Nodes | Regret+Strat Memory | Total Estimated Memory | Guardrail & Browser WASM Status (`<= 512 MiB` default) |
|---|---|---|---|---:|---:|---:|---|
| River Toy / Study | River `Ks9d4c7h2s` | `iso = false` | `20 + 20` exact | `8` | `3,200 B` (`3.1 KiB`) | **`5,696 B` (`5.6 KiB`)** | Fits (`< 10 ms`) |
| River Medium | River `Ks9d4c7h2s` | `iso = false` | `200 + 200` exact | `8` | `32,000 B` (`31.3 KiB`) | **`38,816 B` (`37.9 KiB`)** | Fits (`< 50 ms`) |
| Turn Subrange (2 sizes, lossless) | Turn `Ks9d4c7h` | `iso = false` | `50 + 50` exact | `776` | `699,200 B` (`682.8 KiB`) | **`956,960 B` (`934.5 KiB`)** | Fits (`< 0.5 s`) |
| Flop `50 vs 50` Identity (Iso OFF) | Flop `Ks9s4d` (two-tone) | `iso = false` | `50 + 50` exact | `57,240` | `49,635,200 B` (`47.34 MiB`) | **`66,830,336 B` (`63.73 MiB`)** | Fits (`63.7 MiB < 512 MiB`) |
| Flop `50 vs 50` Identity (Iso ON) | Flop `Ks9s4d` (two-tone) | `iso = true` | `50 + 50` exact | `34,880` | `30,249,600 B` (`28.85 MiB`) | **`40,824,720 B` (`38.93 MiB`)** | Fits (`39.1%` smaller lossless tree; exact node/byte match with `PloGame::memory_report()`) |
| Flop `500 vs 500` Identity (Iso OFF) | Flop `Ks9d4c` / `Ks9s4d` | `iso = false` | `500 + 500` exact | `57,240` | **`496,352,000 B` (`473.36 MiB`)** | **`526,247,936 B` (`501.87 MiB`)** | Borderline (`501.9 MiB`); **`>= 600 vs 600` trips `MemoryBudgetExceeded`** (`> 512 MiB`), `> 5 GiB` at `5k vs 5k`! |
| Flop `500 vs 500` Identity (Iso ON) | Flop `Ks9s4d` (two-tone) | `iso = true` | `500 + 500` exact | `34,880` | **`302,496,000 B` (`288.48 MiB`)** | **`321,700,320 B` (`306.80 MiB`)** | Fits at `500 vs 500` (`306.8 MiB`), but **`>= 900 vs 900` still trips `MemoryBudgetExceeded`** (`> 512 MiB`)! |
| Flop Bucketed `50 vs 50` (`Equity` / `NutBlocker`) | Flop `Ks9d4c` | `iso = false` | `50 + 50` buckets (over `500+500` combos) | `57,240` | **`49,635,200 B` (`47.34 MiB`)** | **`84,435,136 B` (`80.52 MiB`)** | Fits (`10x` regret+strategy cut + `4.68 MiB` deduplicated `(board, player)` `Arc<[u32]>` bucket maps while keeping exact 500-combo blockers at showdown). |

**Honest Remaining Limits for Full Flop WASM**:
- Even with 50 buckets (`75.85 MiB` iso OFF / `~47 MiB` iso ON on two-tone), traversing `1,432–2,352` turn+river subtrees per DCFR step in single-threaded WASM is practical only for compact trees / modest iteration counts (`~20–100` iters). Commercial-scale `10k–50k` combo PLO flop solving requires depth-limited turn/river value functions or multi-threaded WebWorker `SharedArrayBuffer` + precomputed EMD bucket tables.

---

## 5. Frozen WASM Polar River Equilibrium & Determinism Contract (`Steps 4–5`)

Verified in `js/example_river.mjs` (`node js/example_river.mjs`) and `crates/plo_wasm/src/lib.rs`:
- **Board & Ranges**: `KsQsJs7d2c`, `Pot = 100`, `EffectiveStack = 100`, Pot-Limit `100%` bet (`Check:0`, `Allin:100`):
  - OOP (`player = 0`): `AsTs4h3h` (nut spade flush) + `Ah9s6d5d` (PLO-specific bare-`9s` air — `A-K-Q-J-9` high, cannot make a 1-card flush).
  - IP (`player = 1`): `KdQd4c3c` (two-pair Kings & Queens bluff-catcher).
- **Fixed Iteration Count**: `FIXED_POLAR_ITERATIONS = 1000` pure DCFR steps via `stepIterations(1000)` (no early-exit threshold).
- **Tight Equilibrium Assertions**:
  - Root EV: `|EV_OOP - 75.0| <= 0.05` and `|EV_IP - 25.0| <= 0.05` (measured: `[75.0000, 25.0000]`).
  - Strategy frequencies: OOP `AsTs4h3h` bet `>= 0.99`, OOP `Ah9s6d5d` bluff within `0.50 ± 0.02` (measured `49.98%`), IP `KdQd4c3c` call vs `Allin:100` within `0.50 ± 0.02` (measured `49.97%`).
- **Determinism (`<= 1e-9`)**:
  - Both `EquityBucketAbstraction` and `NutBlockerBucketAbstraction` canonicalize `opponent_combos`, `live_indices`, and all quantile tie-breaks strictly by `ComboId` (`step4_bucket_assignment_invariance_under_slice_permutations_and_ties`), eliminating any slice-order or `HashMap` dependence while preserving the `phase_h` `k = 2` bare `As` nut-flush blocker split.
  - Repeated WASM runs (`Polar River` at `1000` iterations and 2D-bucketed `KsQs4d2c` Turn solves across reversed range token orderings) agree within `<= 1e-9` on `rootEv`, `ev`, and `strategy`.

---

## 6. External Calibration & Honest Same-`N` Native vs WASM Contract (`Part A` & `Part D`)

### 6.1 Same-`N` Native vs WASM Flop Tiny Contract (`FIXED_FLOP_TINY_ITERS = 100`)
- **Spot**: Two-tone flop `Ks9s4d`, `Pot = 100`, `Stack = 50` (`SPR = 0.50`), Pot-Limit `bet = [25%, 50%]`, `raise = [100%]`, `IdentityAbstraction`, `iso = true`, `20` vs `20` suit-symmetric combos (`AAQQ$ds, JJTT$ds, 8877$ds, AsJsQhTh, AsJsQcTc` vs `AAJJ$ds, QQTT$ds, 7766$ds, AsQsJhTh, AsQsJcTc`).
- **Why `decision_nodes == 34,880` (`iso = true`) and `57,240` (`iso = false`) matches `(Pot=100, Stack=100, bet=[50%, 100%])`**:
  - Both configs produce the exact same abstract `ActionTree` player-node counts `(8 flop, 16 turn, 24 river)`: `1` half-stack opening bet (`Bet:25` into `50` vs `Bet:50` into `100`) + `1` full-stack opening bet (`AllIn:50` vs `Bet:100`), where `100%` pot raise facing the smaller bet immediately reaches all-in and calling leaves `1/6` SPR on the next street so both bet fractions collapse to a single `AllIn`.
  - On `Ks9s4d` with `{h, c}`-symmetric ranges (`36` representative turns, `1,429` representative river subtrees), `game.memory_report().decision_nodes == 8*1 + 16*36 + 24*1429 == 34,880` (`22,038,768 B = 21.02 MiB` total).
- **Measured Same-`N` (`N = 100`) Native vs WASM Comparison** (replacing an earlier `~0.012` chip comparison that mixed `N = 100` WASM with `N = 1000` native):
  - **Native (`bench_flop_tiny`, `N = 100`)**: `EV = [48.780203512, 51.219796488]`, `exploitability = 6.743357e-3` chips (`1.566112e-5` chips at `N = 1000`, `EV = [48.792543577, 51.207456423]`).
  - **WASM (`js/example_flop_tiny.mjs`, `N = 100`, `26.19 MiB` linear memory)**: `EV = [48.780203512, 51.219796488]`, `exploitability = 6.743357e-3` chips.
  - **Measured Same-`N` Deltas at `N = 100`**: `|ΔEV_oop| = 2.890e-10` chips, `|ΔEV_ip| = 2.890e-10` chips (asserted `< 0.05`), `|Δexploit| = 4.145e-10` chips (asserted `< 0.02`).
- **D3 Independent BR vs DCFR Check (`d3_flop_tiny_20v20_1000_iters_best_response_vs_dcfr_ev_and_exploitability`)**:
  - At `N = 1000` on `20 vs 20` Flop Tiny (`IdentityAbstraction`), `BR_OOP = 48.792573929 >= EV_OOP = 48.792543577`, `BR_IP = 51.207457393 >= EV_IP = 51.207456423`, and `0.5 * ((BR_OOP - EV_OOP) + (BR_IP - EV_IP)) = 1.5661e-5` matches `game.exploitability()` within `<= 1e-4`.

### 6.2 D1 External Equity Snapshots (`slow_eval_plo_lex` Only vs Fast Range Equity)
- **Board 1 (`QsJh4c`, blocker-colliding Broadway flop)**: `AAKK$ds` (`6` live) vs `(AK)(QJ)` (`6` live), `18` compatible pairs × `820` runouts = `14,760` deals → `wins_oop = 4,231`, `wins_ip = 8,387`, `ties = 2,142`, exact rational equity **`5302 / 14760` (`0.359214092141`)** verified to `1e-12`.
- **Board 2 (`7c3d2h`, unpaired rainbow low flop — zero board rank collisions)**: `AAKK$ds` (`6` live) vs `(AK)(QJ)` (`12` live), `36` compatible pairs × `820` runouts = `29,520` deals enumerated purely with `plo_eval::slow_eval_plo_lex` (zero fast lookup tables) → `wins_oop = 27,459`, `wins_ip = 2,061`, `ties = 0`, exact rational equity **`27459 / 29520 == 3051 / 3280` (`0.930182926829`)** matching `equity_range_vs_range` within `<= 1e-12`.

### 6.3 D2 Richer River Matrix Nash (`Bet(100)` + `AllIn(200)` Raise + `Probe(100)`) vs DCFR (`tests/golden/d2_river_raise_or_probe.rs`)
Both fixtures use `29` pairwise-disjoint cards (`5` board + `6×4` hole cards, zero blockers), `Pot = 100`, `Stack = 200`, `15` infosets (`OOP [Check, Bet(100)]`, `IP vs Check [Check, Bet(100)]`, `OOP vs Probe [Fold, Call]`, `IP vs Bet [Fold, Call, AllIn(200)]`, `OOP vs Raise [Fold, Call]`), `IdentityAbstraction`, `iso = false`:
1. **`d2_part1_probe_and_raise_tree_exact_rational_200_over_9_and_grid_check` (Check-Trap + Active IP Probe Bet)**:
   - On `KsQsJs7d2c`, IP holds `#1` Nut Flush (`As9s4h3h`), `#3` Straight (`AdTh6h5h`), `#5` Air (`Ah8d6c5c`); OOP holds `#2` K-Flush (`Ts8s6d5d`), `#4` Two-Pair (`KdQd4c3c`), `#6` Air (`9h8h4d3d`).
   - Exact closed-form & `0.01`-grid Nash (`0.0` exploitability to `<= 1e-12`): OOP checks `100%`; IP probes `#1` `1.00`, checks `#3` `0.00`, bluff-probes `#5` `0.50`; `EV_OOP = 200 / 9 = 22.222222...`, `EV_IP = 700 / 9 = 77.777778...`.
   - `PloGame` DCFR (`2,000` iters): `EV = [22.222228, 77.777772]` (`|ΔEV| = 5.6e-6 < 0.2`), `exploitability = 5.39e-4 < 0.2`, probe frequencies within `0.001` of `[1.00, 0.00, 0.50]`.
2. **`d2_part2_active_on_path_bet_and_ip_allin_raise_matrix_nash_vs_dcfr` (Active Root Bet + Active IP `AllIn(200)` Value & Bluff Raise)**:
   - On `KsQsTs7d2c`, OOP holds `#1` Straight Flush `Js9s4h3h` (`w=0.125`), `#3` 8-Flush `8s6s5h4c` (`w=0.375`), `#6` Air `9h8h4d3d` (`w=0.250`); IP holds `#2` A-Flush `As4s6d5d` (`w=0.125`), `#4` Straight `AdJh6h5c` (`w=0.375`), `#5` Two-Pair `AhKdQd3c` (`w=0.250`).
   - Exact closed-form rational Nash (`0.0` exploitability to `<= 1e-12`, cross-checked by 250k-step sequence-form fictitious play): OOP bets `100%` of range; facing `Bet(100)`, IP raises `AllIn(200)` `100%` with `#2` A-Flush AND bluff-raises `AllIn(200)` at average frequency `0.05` (`1/20`) on `#4/#5` (calling `0.35`, folding `0.60`); facing `AllIn(200)`, OOP calls `1.00` with `#1` and `2/3` (`0.6667`) with `#3`; `EV_OOP = 650 / 9 = 72.222222...`, `EV_IP = 250 / 9 = 27.777778...`.
   - `PloGame` DCFR (`2,000` iters): `EV = [72.222219, 27.777781]` (`|ΔEV| < 1e-4 < 0.2`), `exploitability = 5.16e-4 < 0.2`, IP `AllIn(200)` value-raise (`1.00`), bluff-raise (`0.05`), and OOP call-vs-raise (`2/3`) all within `<= 0.05`.

### 6.4 What Is Still Not Externally Cross-Checked Against Commercial Solvers
- No comparison has been run against closed-source commercial PLO solvers (e.g., MonkerSolver, GTOWizard, Vision, PokerJuice) — commercial solver binaries/APIs are unavailable in this sandbox and scraping external services is out of scope. Instead, every layer is anchored to self-contained mathematical oracles (`slow_eval_plo_lex` 2+3 enumeration, `brute_showdown_ev` / `brute_fold_ev` 200-case fuzzing, `1e-9` `f64` check-down trees, and `0.0`-exploitability closed-form/sequence-form river matrix equilibria).

---

## 7. Practical Native Flop Path (`Part C` — `NutBlockerBucketAbstraction`)

### 7.1 Inventory of Abstraction, Memory Accounting & Combo-Exact Terminals (`Step 0`)

1. **`NutBlockerBucketAbstraction` & `EquityBucketAbstraction` (`crates/plo_cfr/src/lib.rs`)**:
   - **`NutBlockerBucketAbstraction::new(flop_dims, turn_dims, river_dims)`**:
     - Takes `(equity_bins, nut_blocker_bins)` per street (`target_b = equity_bins * nut_blocker_bins`; e.g., `k = 50` uses `(25, 2)` and `k = 100` uses `(50, 2)`, matching `WasmPloSession::split_2d`).
     - **`f_strength`**: On river, exact 4-card blocker-aware showdown equity against `opponent_combos` (`per_combo_river_equity`, canonicalized by `ComboId`); on flop/turn, normalized mean next-street `HandRank` (`raw / 7462.0`) across all `52 - |board| - 4` legal next-street cards.
     - **`f_nut_blocker`**: Combines (a) top-4 (`~90th-percentile`) next-street `HandRank` nut ceiling on flop/turn with (b) PLO-specific flush-draw & **bare nut-flush blocker** score on boards with `>= 2` cards of suit `s` (`+2.0` / `+1.2` / `+0.6` when holding `>= 2` hole cards of suit `s` for actual PLO flush / frontdoor FD; `+0.75` on 2-suit board or `+1.5` on 3+-suit board when holding **exactly `1` hole card of suit `s`** matching the highest off-board rank `As` — capturing PLO's bare nut-flush blocker even though `1` suit card can never make a PLO flush).
     - **Deterministic `ComboId` Canonicalization**: `opponent_combos`, `live_indices`, primary `f_strength` quantile slices, and secondary `f_nut_blocker` sub-slices all sort and tie-break strictly by `ComboId` (`0..270,725`), making bucket IDs 100% invariant to caller slice/token order (`step4` + `part_c_c4`). Because `ComboId` tie-breaking orders suits canonically (`c < d < h < s`), `HandAbstraction::is_suit_symmetric()` returns `false` for both bucket abstractions, so `effective_iso = options.enable_chance_isomorphism && abstraction.is_suit_symmetric()` safely disables chance suit-folding (`57,240` unfolded decision nodes instead of `34,880`) and prevents folding non-commuting buckets.
   - **`EquityBucketAbstraction::new(flop_buckets, turn_buckets, river_buckets)`**:
     - 1D quantile baseline (`max_buckets_for_street = Some(k)`) scoring river by `per_combo_river_equity` and flop/turn by immediate 2+3 `eval_plo` rank, also tie-broken strictly by `ComboId`.

2. **How `PloGame` Separates Bucket Regrets/Strategies from Exact Combo Reaches & Terminals**:
   - **Per-Node Regrets & Cumulative Strategies (Bucket-Indexed)**: Each `ConcreteNodeKind::Player` allocates `regrets: Vec<f32>` and `cum_strategy: Vec<f32>` of length `actions.len() * num_buckets` (`0` when `actions.len() <= 1`), costing `8 * actions.len() * num_buckets` bytes (`f32 + f32`).
   - **Deduplicated `(board, player)` Bucket Maps (`Arc<[u32]>`)**: During `Part C` inspection, we found `ConcreteBuilder` previously cloned a fresh `combo_to_bucket: Vec<u32>` (`4 * |R_p|` bytes) and re-ran `assign_buckets` at every `Player` decision node (`57,240` calls / `~114 MiB` at `500 vs 500`). `ConcreteBuilder::get_or_build_bucket_map` now caches one `Arc<[u32]>` per unique `(board_mask, player)` (`1` flop + `49` turns + `1,176` unordered rivers = `1,226` unique boards per player when `effective_iso == false`), cutting `200 vs 200` tree build time to `0.82 s` and keeping `combo_to_bucket` storage at `2 * 1,226 * |R| * 4` bytes (`1.87 MiB` at `200 vs 200`, `4.68 MiB` at `500 vs 500`).
   - **Traversal Reach & Terminals (Strictly Combo-Indexed)**: `dcfr_traverse`, `strategy_cfv_traverse_f64`, `best_response_traverse_f64`, `play_chance_card`, and `current_combo_reaches` propagate `reach_oop: [f32; |R_oop|]` and `reach_ip: [f32; |R_ip|]` per individual 4-card combo (zeroing any combo intersecting a dealt turn/river card, verified in `part_c_c2`), and evaluate every fold/showdown terminal via `RiverEvalData` (`2+3` `RiverBoardCache`) + 15-term 4-card inclusion-exclusion (`part_c_c1` verifies `PloGame::root_expected_values_f64() == equity_range_vs_range * pot` to `<= 1e-8` under `k = 2` and `k = 3` on `8 vs 8` overlapping combos).

3. **Memory Scaling & Default Guardrail (`estimate_tree_memory` vs `memory_report`)**:
   - **`DEFAULT_MAX_MEMORY_BYTES`**: `512 * 1024 * 1024` (`512 MiB = 536,870,912 B`), checked in `PloGame::with_options` *before* allocating `ConcreteBuilder`.
   - **Scaling Laws**:
     - `regret_and_strategy_bytes` scales with **`O(decision_nodes * avg_actions * k_buckets)`** (`49,635,200 B = 47.34 MiB` for `k = 50` across all `57,240` unfolded nodes regardless of whether `|R|` is `200` or `500`, vs `O(decision_nodes * |R|)` under `IdentityAbstraction`).
     - `bucket_map_bytes` (`2 * 1,226 * |R| * 4` when `k < |R|`, `0` for `Identity`) and `river_eval_bytes` (`river_tables * (|R_oop| + |R_ip|) * 12`) scale **linearly in `|R|`** across unique boards (`799` iso-on / `1,176` iso-off), never across `decision_nodes`.
     - `node_overhead` scales as `(decision_nodes + chance_nodes + terminal_nodes) * 96` (`9.75 MiB` iso-on / `15.05 MiB` iso-off).

---

### 7.2 Step 1 Memory Table (`part_c_memory_rows`)

Measured on two-tone flop `Ks9s4d`, `Pot = 100, Stack = 50` (`SPR = 0.50`), `PotLimit` `bet = [25%, 50%], raise = [100%]` (`[8 flop, 16 turn, 24 river]` abstract player skeleton), `{h, c}`-suit-symmetric ranges with `enable_chance_isomorphism = true` (`cargo test -p plo_cfr --lib part_c_memory_rows --release -- --nocapture`):

| Range Size | Abstraction | `effective_iso` | `decision_nodes` | `regret_and_strategy_bytes` | `total_estimated_bytes` | Total (`MiB`) | `128 MiB` Budget Check |
|---|---|---|---:|---:|---:|---:|---|
| `50 vs 50` | `Identity` | `true` | `34,880` | `30,249,600 B` | `40,824,720 B` | **`38.93 MiB`** | Fits (`< 64 MiB`) |
| `50 vs 50` | `NutBlocker k=50` `(25,2)` | `false` | `57,240` | `49,635,200 B` | `66,830,336 B` | **`63.73 MiB`** | Fits (`n <= k` fallback) |
| `50 vs 50` | `NutBlocker k=100` `(50,2)` | `false` | `57,240` | `49,635,200 B` | `66,830,336 B` | **`63.73 MiB`** | Fits (`n <= k` fallback) |
| `200 vs 200` | `Identity` | `true` | `34,880` | `120,998,400 B` | `134,449,920 B` | **`128.22 MiB`** | Exceeds `128 MiB` (`209.78 MiB` if non-suit-symmetric) |
| `200 vs 200` | `NutBlocker k=50` `(25,2)` | `false` | `57,240` | `49,635,200 B` | **`73,025,536 B`** | **`69.64 MiB`** | **Fits well under `128 MiB`** (live `memory_report()` = `72,132,736 B` / `68.79 MiB`) |
| `200 vs 200` | `NutBlocker k=100` `(50,2)` | `false` | `57,240` | `99,270,400 B` | `122,660,736 B` | **`116.98 MiB`** | Fits `< 128 MiB` (live `memory_report()` = `120,579,840 B` / `115.00 MiB`) |
| `500 vs 500` | `Identity` | `true` | `34,880` | `302,496,000 B` | **`321,700,320 B`** | **`306.80 MiB`** | **Trips `MemoryBudgetExceeded` at `128 MiB`** (`501.87 MiB` if `iso=false`) |
| `500 vs 500` | `NutBlocker k=50` `(25,2)` | `false` | `57,240` | `49,635,200 B` | **`84,435,136 B`** | **`80.52 MiB`** | **Fits well under `128 MiB`** (`6.2x` smaller regret RAM than `Identity iso=false`) |
| `500 vs 500` | `NutBlocker k=100` `(50,2)` | `false` | `57,240` | `99,270,400 B` | `134,070,336 B` | **`127.86 MiB`** | Borderline at `128 MiB`; fits easily under `256 MiB` / `512 MiB` |

---

### 7.3 Step 2 Native Practical Flop Benchmark (`bench_flop_buckets.rs`) & C5 EV Study Note

Command: `cargo run -p plo_cfr --example bench_flop_buckets --release`
- **Board & Tree**: `Ks9s4d` (two-tone), `PotLimit`, `Pot = 100, Stack = 50`, `bet = [25%, 50%], raise = [100%]` (`[8, 16, 24]` abstract player skeleton).
- **Ranges (`200 vs 200` live combos)**: Explicit legal PLO-4 macro unions (`AAKK$ds..8877$ds`, rundown wraps `AKQJ$ds..8765$ds`, suited Ace/Broadway groups `(AK)(QJ)..(QJ)(T8)`, `AAKQ$ss, AKJT$ss` for OOP `465` live -> capped to first **`200`** in canonical `ComboId` order; IP `546` live -> capped to first **`200`** in canonical `ComboId` order).
- **Measured Native Release Results (`max_memory_bytes = 256 MiB`)**:
  - **`NutBlockerBucketAbstraction k = 50` (`dims = (25, 2)`, `effective_iso = false`, `57,240` decision nodes)**:
    - Build time: **`0.82 s`**; `memory_report()`: `regret_and_strategy = 49,635,200 B` (`47.34 MiB`), `river_eval = 4,752,000 B` (`4.53 MiB`), **`total = 72,132,736 B` (`68.79 MiB`)**, headroom **`187.21 MiB`** under `256 MiB` guard.
    - `Iter   0`: `exploitability = 2.498430e1` chips (`24.984304` chips)
    - `Iter 100`: `exploitability = 3.608519e-1` chips (`0.360852` chips) | `root EV f64 = [60.535155232, 39.464844768]` | `214.73 s` (`0.47 iters/s`)
    - `Iter 500`: `exploitability = 3.356844e-1` chips (`0.335684` chips, **`0.34%` of starting pot** measured against exact unabstracted combo-level best response!) | `root EV f64 = [60.535412637, 39.464587363]` | `911.03 s` (**`0.55 iters/s`**).
  - **`IdentityAbstraction` Reference on the Same `200 vs 200` `ComboId`-Capped Ranges (`208.93 MiB < 256 MiB`, `57,240` decision nodes)**:
    - Build time: `0.15 s`; `memory_report()`: `regret_and_strategy = 198,540,800 B` (`189.34 MiB`, `4.0x` larger than `k = 50`), `total = 219,076,736 B` (`208.93 MiB`).
    - `Iter 100`: `exploitability = 8.470895e-3` chips | `root EV f64 = [60.469868918, 39.530131082]` (`0.60 iters/s`).
  - **C5 Study Note (Same-`N` Abstraction EV Gap at `N = 100`)**:
    - `EV_k50(100) = 60.535155` chips vs `EV_id(100) = 60.469869` chips → **`ΔEV_oop = +0.065286` chips (`0.065%` of pot)** while cutting regret/strategy RAM `4.0x` (`189.34 MiB -> 47.34 MiB`) at `200 vs 200` and `10.0x` (`473.36 MiB -> 47.34 MiB`) at `500 vs 500`. (Logged for study only; bucketed EV is never asserted equal to unabstracted EV.)
  - **Throughput Bottleneck Note**: Because `NutBlockerBucketAbstraction` has `effective_iso = false`, each single-threaded DCFR step visits all `106,336` unfolded terminal nodes twice (`212,672` terminal evaluations per iteration) using exact 15-term 4-card inclusion-exclusion (`O(|R_oop| + |R_ip|)`), yielding `0.55 iters/s` at `200 vs 200` (`~0.34 iters/s` at `300 vs 300`).

---

### 7.4 Step 4 Honest WASM Policy Decision

**WASM flop-buckets SKIPPED: `69.64 MiB > 64.00 MiB` (`73,025,536 B` estimated / `68.79 MiB` live on `200 vs 200` `k = 50`, plus single-threaded WASM runtime `> 100 s` for `50` unfolded flop iterations). Native only.**
- Per the Step 4 memory policy (`< 64 MiB` required to ship `js/example_flop_buckets.mjs`; otherwise skip and keep `js/example_flop_tiny.mjs` as the browser flop demo), no `js/example_flop_buckets.mjs` script is shipped so WASM callers never face sluggish multi-minute main-thread/worker stalls or tight linear-memory pressure. `js/example_flop_tiny.mjs` (`20 vs 20`, `21.02 MiB` tree / `26.19 MiB` WASM linear memory) remains the supported browser/Node WASM flop example, while `bench_flop_buckets.rs` provides the native hundreds-of-combos flop path.

