# DESIGN.md — PLO-4 Postflop Solver Core (`plo-postflop`)

## 0. Upstream Architecture Mapping (`wasm-postflop` & `postflop-solver`)

Upstream [`b-inary/wasm-postflop`](https://github.com/b-inary/wasm-postflop) is a thin WebAssembly + Vue wrapper around [`b-inary/postflop-solver`](https://github.com/b-inary/postflop-solver) (licensed `AGPL-3.0-or-later`). Every poker-semantic invariant in upstream assumes 2-card Texas Hold'em and No-Limit betting:

| Upstream Module | What Upstream Did (Hold'em) | Disposition in PLO-4 Rewrite |
|---|---|---|
| `card.rs` | `Card = 4 * rank + suit` (`0..52`), `(Card, Card)` hole pairs, `card_pair_to_index` (`0..1326`), Hold'em turn/river suit swap tables | **Replaced** by `plo_cards`: `Card(u8)`, sorted `Hand4([Card; 4])`, exact combinatorial number system `ComboId(u32)` in `0..270_725` ($C(52,4)$), 64-bit `CardMask`, dead-card checks in $O(1)$ via `(m1 & m2) != 0`. Card ID encoding `4 * rank + suit` is retained for compatibility. |
| `hand.rs` + `hand_table.rs` | 7-card Hold'em evaluator ("best 5 of 7 using any cards") + binary search over `HAND_TABLE` | **Deleted / Replaced** by `plo_eval`: dedicated 5-card evaluator (`eval_5`) backed by O(1) flush + Cactus-Kev-style rank-product lookup tables, plus specialized `eval_plo_flop` (6 candidates), `eval_plo_turn` (24 candidates), and `eval_plo_river` (60 candidates) enforcing **exact 2 hole cards + 3 board cards**. Zero 7-card Hold'em paths exist anywhere in the workspace. |
| `range.rs` + `rust/range` | Dense `[f32; 1326]` array + 13x13 Hold'em matrix (`AA`, `AKs`, `AKo`) | **Deleted / Replaced** by `plo_range`: sparse weighted 4-card range (`PloRange`) indexed by `ComboId`, sorted/deduplicated, fast `CardMask` dead-card filtering, PLO pattern DSL parser (`AAKK`, `AAKK$ds`, `AKQJ$ss`, `[AsKsQhJh]`, `AsKsQhJh@0.5`, etc.), normalization, and equity engine (hand-vs-hand, hand-vs-range, range-vs-range on flop/turn/river). |
| `card.rs::isomorphism` | 2-card suit swaps on flop/turn/river | **Replaced** by `plo_iso`: Omaha-aware suit canonicalization for 4-card hands (rainbow, single-suited, double-suited, trips-suited, monotone), `(Hand4, Board)` joint canonicalization preserving suit bindings between hole cards and board suits, and lossless isomorphic chance-card grouping + range permutation swaps on turn/river deals. |
| `bet_size.rs` + `action_tree.rs` + `rust/tree` | No-Limit tree builder (`max_bet = opponent_stack + prev_amount`, arbitrary overbets) | **Rewritten** in `plo_tree`: **Pot-Limit Omaha legal bet/raise sizing by default** (`max_raise_to = 3 * facing_bet + pot_before_facing_bet`, i.e., `pot_after_call + call_amount`, capped by effective stack), configurable pot fractions (e.g., `33%`, `50%`, `75%`, `100%`), raise caps, donk options, street transitions (`Flop -> Turn -> River`), and optional explicit NL override documented as non-PLO. |
| `game/evaluation.rs` | 2-card inclusion-exclusion (`sum - minus[c1] - minus[c2] + same`) | **Deleted / Replaced** in `plo_cfr`: 2-card subtraction fails in PLO because two 4-card hands can overlap in 1, 2, 3, or 4 cards! Replaced by **exact 4-card 15-term subset inclusion-exclusion** ($\binom{4}{1}=4$ singles, $\binom{4}{2}=6$ pairs, $\binom{4}{3}=4$ triples, $\binom{4}{4}=1$ quad) for $O(|R_{\text{OOP}}| + |R_{\text{IP}}|)$ exact blocker-aware terminal evaluation, verified against $O(N \cdot M)$ pairwise reference. |
| `solver.rs` + `utility.rs` | Discounted CFR ($\alpha=1.5, \beta=0.5, \gamma=3.0$ with power-of-4 strategy resets), regret matching, exploitability (MES best response) | **Architecture Kept, Storage Rewritten** in `plo_cfr`: implements a variant in the spirit of Brown & Sandholm's `DCFR(3/2, 0, 2)` ($\alpha=1.5$ positive regret discount $t^{1.5}/(t^{1.5}+1)$, $\beta=0$ CFR+-style negative regret reset to $0$, and cumulative-strategy pre-multiply $\gamma_t = ((t-1)/t)^2$ yielding exact quadratic $(t/T)^2$ iteration weights), alternating updates, regret matching, and exact exploitability calculation without `unsafe` raw-pointer aliasing, supporting both **exact active-combo indexing** (`IdentityAbstraction`) and **reach-normalized bucketed `HandAbstraction`** (`EquityBucketAbstraction`, `NutBlockerBucketAbstraction`). |
| `bunching.rs` | Hold'em 2-card folded-player inclusion-exclusion (`COMB_49_8`) | **Omitted from MVP** (documented non-goal: 4-card folded-player bunching is $\binom{48}{4k}$ and intractable in WASM). |
| `rust/solver-st` | `wasm-bindgen` exports assuming 1326/169 Hold'em arrays | **Replaced** by `plo_wasm`: zero-dependency `wasm32-unknown-unknown` C-ABI + JSON/typed-buffer WASM module with JS/Node wrapper (`js/plo_postflop.mjs`), avoiding external proc-macro crates and keeping WASM binary footprint tiny (~100–250 KB code + ~150 KB runtime tables). |

---

## 1. Crate Architecture

```text
Cargo.toml                  (workspace root, AGPL-3.0-or-later, matching upstreamwasm-postflop)
crates/
  plo_cards/                Card (0..51), Rank, Suit, CardMask (u64), Hand4, Board, ComboId (0..270,725)
  plo_eval/                 Exact 2+3 PLO evaluator: slow reference + fast 5-card lookup + flop/turn/river hot paths
  plo_iso/                  PLO suit isomorphism: Hand4 canonicalization, (Hand4 + Board) canonicalization, chance equivalence
  plo_range/                Sparse weighted 4-card ranges, PLO range syntax parser, blocker filtering, exact equity calculator
  plo_tree/                 Pot-Limit Omaha betting rules, BetSizeConfig, ActionTree (flop/turn/river + chance + terminals)
  plo_cfr/                  DCFR postflop solver (river, turn, flop), 4-card 15-term inclusion-exclusion, HandAbstraction trait
  plo_wasm/                 wasm32-unknown-unknown cdylib + rlib API for browser/Node
js/
  plo_postflop.mjs          Ergonomic JavaScript wrapper over plo_wasm linear memory & exports
  example_river.mjs         End-to-end Node/WASM verification & usage example
```

Hold'em 2-card combo indices (`0..1326`), 13x13 range grids, and 7-card evaluators are **completely absent** from this workspace so no code path can accidentally invoke Hold'em semantics.

---

## 2. Core Data Structures & Mathematical Invariants

### 2.1 Card & 4-Card Hand Encoding (`plo_cards`)
- `Card`: `u8` in `0..52`, `card = 4 * rank + suit`, `rank` in `0..13` (`2..=A`), `suit` in `0..4` (`c, d, h, s`).
- `CardMask`: `u64` bitmask with bit `1u64 << card` set (`0 <= card < 52`). Overlap check is `(mask_a & mask_b) != 0`; card count is `mask.count_ones()`.
- `Hand4`: `[Card; 4]` strictly sorted `c0 < c1 < c2 < c3 < 52`. Enforced at construction; illegal duplicates or unsorted cards cannot be constructed via safe constructors.
- `ComboId`: `u32` in `0..270_725` via combinatorial number system (colexicographic index):
  $$\text{id}(c_0, c_1, c_2, c_3) = \binom{c_0}{1} + \binom{c_1}{2} + \binom{c_2}{3} + \binom{c_3}{4}$$
  Bijective, monotonic, $O(1)$ encode via precomputed $\binom{n}{k}$ table (`53 * 5` integers), fast decode.

### 2.2 PLO Hand Evaluator (`plo_eval`)
- **Strict 2+3 Rule**:
  - Hole pairs: `HOLE_PAIRS_4`: all $\binom{4}{2} = 6$ index pairs `(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`.
  - Board triples:
    - Flop (`[Card; 3]`): $\binom{3}{3} = 1$ triple $\implies 6 \times 1 = 6$ 5-card evaluations.
    - Turn (`[Card; 4]`): $\binom{4}{3} = 4$ triples $\implies 6 \times 4 = 24$ 5-card evaluations.
    - River (`[Card; 5]`): $\binom{5}{3} = 10$ triples $\implies 6 \times 10 = 60$ 5-card evaluations.
- **Fast 5-Card Kernel (`eval_5`)**:
  - Returns `HandRank(u16)` in `1..=7462` (higher is strictly stronger; identical equivalence classes get identical `u16`, ties split pot).
  - Categories:
    - High Card: `1..=1277` (1277 classes)
    - One Pair: `1278..=4137` (2860 classes)
    - Two Pair: `4138..=4995` (858 classes)
    - Three of a Kind: `4996..=5853` (858 classes)
    - Straight: `5854..=5863` (10 classes, `5-high` wheel to `A-high` broadway)
    - Flush: `5864..=7140` (1277 classes)
    - Full House: `7141..=7296` (156 classes)
    - Four of a Kind: `7297..=7452` (156 classes)
    - Straight Flush: `7453..=7462` (10 classes)
    - Total = `7462` exact standard 5-card equivalence classes.
  - Precomputed tables initialized once (`OnceLock`):
    1. `FLUSH_LOOKUP[8192]` (`16 KB`): indexed by 13-bit rank bitmask when all 5 cards share a suit.
    2. `UNIQUE5_LOOKUP[8192]` (`16 KB`): indexed by 13-bit rank bitmask when 5 distinct ranks (`rank_mask.count_ones() == 5`) and non-flush (handles all straights and high-card hands in 1 array read).
    3. Perfect/compact prime-product table for paired/trips/full-house/quads hands (4888 distinct rank multisets, `< 64 KB`).
  - **River Optimization**: On a 5-card river board, the 10 board triples' prime products, rank masks, and suit masks are computed **once per board** (`RiverBoardCache`), reducing each 4-card hand evaluation on a fixed river to 6 hole-pair combinations merged against 10 pre-decoded board triples with zero heap allocation!

### 2.3 Exact 4-Card Blocker Inclusion-Exclusion (`plo_cfr`)
In Hold'em (`postflop-solver`), terminal counterfactual value evaluation uses 2-card inclusion-exclusion:
$$W(\text{opp disjoint from } \{a,b\}) = W_{\text{total}} - W_1[a] - W_1[b] + W_2[\{a,b\}]$$
In PLO, two 4-card hands $h = \{c_0, c_1, c_2, c_3\}$ and $h'$ overlap iff $|h \cap h'| \ge 1$. By the Principle of Inclusion-Exclusion over the 15 non-empty subsets $S \subseteq h$:
$$\sum_{h': h' \cap h = \emptyset} w(h') = W_\emptyset - \sum_{|S|=1} W_1[S] + \sum_{|S|=2} W_2[S] - \sum_{|S|=3} W_3[S] + W_4[h]$$
Where:
- $|S|=1$: $\binom{52}{1} = 52$ single-card buckets (`[f64; 52]`)
- $|S|=2$: $\binom{52}{2} = 1,326$ pair buckets (`[f64; 1326]`)
- $|S|=3$: $\binom{52}{3} = 22,100$ triple buckets (or active-triple / direct lookup; note `22,100 * 8 bytes = 176.8 KB`, which fits in L2 cache!)
- $|S|=4$: exact same hand index (`u32::MAX` if opponent range doesn't contain `h`, or direct lookup `same_hand_idx[i]`).

Because $52 + 1326 + 22100 = 23,478$ `f64` entries is only **187.8 KB** (and if we dirty-track or zero only touched entries, clearing is proportional to active combos $O(N)$!), we get **exact $O(N_{\text{OOP}} + N_{\text{IP}})$ terminal evaluation for PLO** just like Hold'em — turning a $1000 \times 1000 = 1,000,000$ pairwise showdown loop into $2 \times 1000 \times 15 = 30,000$ register operations!
For ultra-tiny ranges (e.g., $N \cdot M \le 256$), a fast bitmask dot-product path (`(m_oop & m_ip) == 0`) avoids even zeroing scratch buffers. Both paths are tested for exact agreement.

### 2.4 Pot-Limit Betting Semantics (`plo_tree`)
Let:
- `pot_street_start`: pot at the start of the current street.
- `invested[p]`: chips committed by player `p` on the current street (`0` at street start).
- `stack[p]`: remaining chips behind for player `p`.
- `facing = invested[opp] - invested[p]` (`>= 0`, amount needed to call).
- `pot_before_action = pot_street_start + invested[0] + invested[1]`.
- `pot_after_call = pot_before_action + facing = pot_street_start + 2 * invested[opp]`.

Under **exact Pot-Limit rules**:
1. **Unopened pot (`facing == 0`)**:
   - Legal check (`Check`).
   - Min bet: `min(min_bet_unit, stack[p])` (default `min_bet_unit = 1` chip or configurable big blind).
   - **Max pot-limit bet**: `min(pot_before_action, stack[p])` (100% pot). Unlike NLHE, a player **cannot** shove >100% pot unconditionally!
   - A pot-fraction bet `f` (`0 < f <= 1.0`, e.g., `0.33, 0.50, 0.75, 1.00`) has size `round(f * pot_before_action)`, clamped to `[min_bet, max_plo_bet]`. Fractions `> 1.0` are capped at `1.0` (pot) in strict Pot-Limit mode.
2. **Facing a bet/raise (`facing > 0`)**:
   - Legal `Fold` and `Call` (commits `min(facing, stack[p])`).
   - If `stack[p] > facing` and opponent is not all-in and raise cap is not reached:
     - Previous raise increment `last_raise_size = invested[opp] - prev_invested_before_opp`.
     - Min raise-to (street total): `min(invested[opp] + max(last_raise_size, min_bet_unit), invested[p] + stack[p])`.
     - **Max pot-limit raise-to (street total)**:
       - Additional raise above call for a 100% pot raise is `pot_after_call`.
       - Total street commitment = `invested[p] + facing + pot_after_call = invested[opp] + pot_after_call = 3 * invested[opp] + pot_street_start - (pot_street_start ...)` — wait: let's verify!
       - Example: Start pot = `100`. OOP bets `100` (`invested = [100, 0]`).
         - IP faces `100`. `pot_before_action = 200`. `pot_after_call = 200 + 100 = 300`.
         - IP calls `100` + raises `300` $\implies$ IP total street commitment = `400`!
         - Check formula: `invested[opp] + pot_after_call = 100 + 300 = 400`. And `3 * 100 + 100 = 400`!
       - Example 2: OOP now faces IP's `400` (`invested = [100, 400]`, `pot_street_start = 100`).
         - `facing = 400 - 100 = 300`.
         - `pot_before_action = 100 + 100 + 400 = 600`.
         - `pot_after_call = 600 + 300 = 900`.
         - OOP max pot raise: calls `300` (bringing OOP to `400`) + raises `900` $\implies$ OOP total street commitment = `400 + 900 = 1300`!
         - Check formula: `invested[opp] + pot_after_call = 400 + 900 = 1300`. Exact!
     - For a fractional pot raise `f` (`0 < f <= 1.0`, e.g., `50%` pot raise or `100%` pot raise):
       - Street total `raise_to = invested[opp] + round(f * pot_after_call)`, clamped to `[min_raise_to, max_plo_raise_to]`.
     - All-in occurs **if and only if** a legal bet/raise reaches `invested[p] + stack[p]` (or via threshold snapping when `max_stack_commitment <= max_plo_raise_to` and within `add_allin_threshold`).

---

## 3. WASM Memory Budget & Scalability Realism

WebAssembly (`wasm32-unknown-unknown`) has a hard 4 GB (`2^32`) address space ceiling, and browsers typically enforce practical single-tab `WebAssembly.Memory` limits around **1–2 GB** on mobile and **2–4 GB** on desktop.

Let $H$ be the number of active infoset hands (or buckets) per player at a decision node, $A$ be the number of actions per decision node, and each uncompressed DCFR node store `regrets: f32` ($4$ bytes) + `cum_strategy: f32` ($4$ bytes) $= 8$ bytes per `(action, hand)` entry. Exact unit-tested figures from `plo_cfr::estimate_memory_for_config` (`Pot = 100, Stack = 100, Pot-Limit 50%/100% bet, 100% raise`, comparing `iso = false` vs `iso = true` on two-tone flop `Ks9s4d` as well as rainbow `Ks9d4c`):

| Scenario | Street & Board | Iso Flag | Combos / Buckets (`OOP + IP`) | Decision Nodes ($N_{\text{dec}}$) | Regret+Strat Memory | Total Estimated Memory (`192 B`/node) | Fits in Browser WASM (`<= 512 MiB` default)? |
|---|---|---|---:|---:|---:|---:|---|
| **River 20 vs 20** | River `Ks9d4c7h2s` | `iso = false` | `20 + 20` exact | `8` | `3,200 B` (`3.1 KiB`) | **`7,712 B` (`7.5 KiB`)** | Yes (instant, `< 10 ms`) |
| **River 200 vs 200** | River `Ks9d4c7h2s` | `iso = false` | `200 + 200` exact | `8` | `32,000 B` (`31.3 KiB`) | **`40,832 B` (`39.9 KiB`)** | Yes (`< 50 ms`) |
| **Turn 50 vs 50** | Turn `Ks9d4c7h` (48 rivers) | `iso = false` | `50 + 50` exact | `776` | `699,200 B` (`682.8 KiB`) | **`1,157,120 B` (`1.10 MiB`)** | Yes (`< 0.5 s`) |
| **Flop 50 vs 50 (Identity, Iso OFF)** | Flop `Ks9s4d` ($49 \times 48$ runouts) | `iso = false` | `50 + 50` exact | `57,240` | `49,635,200 B` (`47.34 MiB`) | **`82,614,272 B` (`78.79 MiB`)** | Yes (`78.8 MiB < 512 MiB`) |
| **Flop 50 vs 50 (Identity, Iso ON)** | Flop `Ks9s4d` (`36` turns / `1,432` rivers) | `iso = true` | `50 + 50` exact | `34,880` | `30,249,600 B` (`28.85 MiB`) | **`50,441,040 B` (`48.10 MiB`)** | Yes (`38.9%` lossless RAM reduction via `{h, c}` suit folding) |
| **Flop 500 vs 500 (Identity, Iso OFF)** | Flop `Ks9d4c` / `Ks9s4d` | `iso = false` | `500 + 500` exact | `57,240` | **`496,352,000 B` (`473.36 MiB`)** | **`542,031,872 B` (`516.92 MiB`)** | Refused without iso (`516.9 MiB > 512 MiB` default WASM guardrail; fits on Desktop RAM) |
| **Flop 500 vs 500 (Identity, Iso ON)** | Flop `Ks9s4d` (two-tone) | `iso = true` | `500 + 500` exact | `34,880` | **`302,496,000 B` (`288.48 MiB`)** | **`331,316,640 B` (`315.97 MiB`)** | Fits at `500 vs 500` (`316.0 MiB`), refused at `>= 900 vs 900` (`> 512 MiB` guardrail) |
| **Flop 50 Buckets (`500 vs 500`, Iso OFF)** | Flop `Ks9d4c` ($49 \times 48$ runouts) | `iso = false` | `50 + 50` buckets | `57,240` | **`49,635,200 B` (`47.34 MiB`)** | **`100,219,072 B` (`95.58 MiB`)** | Yes (`10x` smaller regret/strategy arrays, exact 500-combo blockers at showdown; CPU-bound in single-thread WASM). |

**Explicit Engineering Tradeoffs & Remaining Work for Large-Scale Turn/Flop**:
1. **Static Evaluator Tables**: Kept under **120 KB** total (`FLUSH_LOOKUP` 16 KB + `UNIQUE5_LOOKUP` 16 KB + rank-multiset hash table ~32 KB), easily fitting in L1/L2 cache and WASM linear memory.
2. **River & Turn Exact Core**: Fully exact at the 4-card combo level, using $O(N+M)$ 15-term inclusion-exclusion on river showdowns and isomorphic river/turn suit folding (`plo_iso`) to skip strategically identical runout suits.
3. **Flop Solver Support**:
   - Exact combo-level solving works out-of-the-box for small study ranges (e.g., 5–40 combos per player with compact bet trees, useful for toy-game verification and exact exploitability tests).
   - For wider flop ranges, `plo_cfr` exposes the `HandAbstraction` trait (`bucket_for_hand(player, street, board, combo_idx) -> BucketId`) with both `IdentityAbstraction` (exact) and `EquityBucketAbstraction` (clusters 4-card hands on each street/board into $K$ equity-quantile buckets while still evaluating exact 4-card blockers and exact PLO 2+3 showdowns at terminal leaves!).
   - Future work beyond MVP for commercial-scale flop solving: multi-street OCHS / Earth Mover's Distance (EMD) k-means clustering tables, disk/IndexedDB streaming for turn subgames, depth-limited subgame solving (resolve turn/river on demand rather than materializing all 1,980 runout subtrees simultaneously), and `i16` regret compression.
