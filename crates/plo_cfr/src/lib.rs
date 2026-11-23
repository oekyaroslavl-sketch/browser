//! Discounted Counterfactual Regret Minimization (`DCFR`) postflop solver for Pot-Limit Omaha (`PLO-4`).
//!
//! # Key PLO Architectural Guarantees
//! 1. **Exact PLO 2+3 Showdown Utility**: All showdown utilities derive exclusively from `plo_eval::RiverBoardCache`
//!    (`60` legal 2-hole + 3-board candidates on every 5-card river).
//! 2. **Exact 4-Card Blocker Removal (`O(N + M)` 15-Term Inclusion-Exclusion)**:
//!    Unlike Hold'em (`sum - c1 - c2 + c12`), two 4-card PLO hands can overlap in `1, 2, 3, or 4` cards.
//!    Terminal fold and showdown evaluations use **exact 4-card 15-term inclusion-exclusion**:
//!    `W(disjoint) = W_0 - sum_{|S|=1} W_1[S] + sum_{|S|=2} W_2[S] - sum_{|S|=3} W_3[S] + W_4[hand]`
//!    with dirty-slot tracking so clearing scratch memory is strictly `O(N_active)`!
//!    For small ranges (`N_oop * N_ip <= 512`), a direct bitmask loop (`(m0 & m1) == 0`) avoids table overhead.
//! 3. **Chance-Node Suit Isomorphism**: Turn and river chance nodes fold suit-equivalent runout cards via
//!    `plo_iso::compute_chance_isomorphism`, solving one canonical branch and permuting counterfactual values
//!    via `build_combo_swap_pairs`.
//! 4. **Layered Memory & Abstraction Support**:
//!    - Exact combo-level solving (`IdentityAbstraction`) for river, turn, and compact flop ranges.
//!    - Bucketed solving (`HandAbstraction`, including `EquityBucketAbstraction`) so larger flop/turn trees
//!      store regrets & cumulative strategies per `(action, bucket)` while **still propagating exact per-combo
//!      reach probabilities and evaluating exact 4-card blockers & showdowns at terminal nodes**!

use core::fmt;

use plo_cards::{
    pair_subset_index, triple_subset_index, Board, Card, CardError, CardMask, ComboId, Street,
};
use plo_eval::{eval_plo, RiverBoardCache};
use plo_iso::{
    build_combo_swap_pairs, compute_chance_isomorphism, permute_board, permute_card, permute_hand4,
    suit_transposition, ChanceIsomorphism,
};
use plo_range::{per_combo_river_equity, PloRange, RangeError, WeightedHand};
use plo_tree::{Action, ActionNodeKind, ActionTree, TreeConfig, TreeError, PLAYER_IP, PLAYER_OOP};

/// Default memory guardrail (`512 MB`) so WASM callers get a clean `SolverError::MemoryBudgetExceeded`
/// with exact byte breakdown rather than an unrecoverable WASM OOM trap.
pub const DEFAULT_MAX_MEMORY_BYTES: u64 = 512 * 1024 * 1024;

/// Total hole cards held by both players in heads-up PLO-4 (`4` for OOP + `4` for IP = `8`).
///
/// For any compatible pair `(h_oop, h_ip)` (`h_oop ∩ board = ∅`, `h_ip ∩ board = ∅`, `h_oop ∩ h_ip = ∅`),
/// exactly `52 - board.len() - HOLE_CARDS_BOTH_PLAYERS` cards remain in the live deck (`41` on Flop→Turn,
/// `40` on Turn→River; with future dead-card bunching `|dead|`, `52 - board.len() - HOLE_CARDS_BOTH_PLAYERS - |dead|`).
pub const HOLE_CARDS_BOTH_PLAYERS: usize = 8;

/// Errors returned when building or querying a [`PloGame`].
#[derive(Debug, Clone)]
pub enum SolverError {
    Card(CardError),
    Range(RangeError),
    Tree(TreeError),
    StreetMismatch {
        board_street: Street,
        tree_street: Street,
    },
    NoValidMatchups,
    InvalidHistoryStep(String),
    MemoryBudgetExceeded {
        required_bytes: u64,
        budget_bytes: u64,
        decision_nodes: usize,
        avg_buckets: usize,
    },
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::Card(e) => write!(f, "{e}"),
            SolverError::Range(e) => write!(f, "{e}"),
            SolverError::Tree(e) => write!(f, "{e}"),
            SolverError::StreetMismatch {
                board_street,
                tree_street,
            } => write!(
                f,
                "board street ({board_street:?}) does not match tree initial_street ({tree_street:?})"
            ),
            SolverError::NoValidMatchups => write!(
                f,
                "no non-overlapping 4-card OOP/IP hand pairs exist on this board"
            ),
            SolverError::InvalidHistoryStep(msg) => write!(f, "invalid history step: {msg}"),
            SolverError::MemoryBudgetExceeded {
                required_bytes,
                budget_bytes,
                decision_nodes,
                avg_buckets,
            } => write!(
                f,
                "tree requires {:.2} MB ({decision_nodes} decision nodes, ~{avg_buckets} infoset hands/buckets), exceeding budget {:.2} MB; reduce range size, bet sizes, or use EquityBucketAbstraction",
                *required_bytes as f64 / (1024.0 * 1024.0),
                *budget_bytes as f64 / (1024.0 * 1024.0),
            ),
        }
    }
}

impl std::error::Error for SolverError {}

impl From<CardError> for SolverError {
    fn from(e: CardError) -> Self {
        SolverError::Card(e)
    }
}
impl From<RangeError> for SolverError {
    fn from(e: RangeError) -> Self {
        SolverError::Range(e)
    }
}
impl From<TreeError> for SolverError {
    fn from(e: TreeError) -> Self {
        SolverError::Tree(e)
    }
}

// ============================================================================
// Hand Abstraction Hook (Identity for Exact Solving, Bucketing for Flop/Large Trees)
// ============================================================================

/// Maps each active 4-card combo of `player` (`0..combos.len()`) on `board` to `(num_buckets, combo_to_bucket)`.
///
/// When `num_buckets == combos.len()` and `combo_to_bucket[i] == i as u32`, solving is **100% unabstracted**
/// at the 4-card combo level. When `num_buckets < combos.len()`, regrets and cumulative strategies at decision
/// nodes on `board` are shared within each bucket while reach probabilities and terminal 4-card blockers/showdowns
/// remain exact at the combo level.
pub trait HandAbstraction: Send + Sync {
    fn assign_buckets(
        &self,
        player: usize,
        board: &Board,
        player_combos: &[WeightedHand],
        opponent_combos: &[WeightedHand],
    ) -> (usize, Vec<u32>);

    /// Whether bucket assignments commute with board/range suit permutations (true for `IdentityAbstraction`
    /// and suit-canonical abstractions). Used to ensure chance isomorphism never folds non-commuting buckets.
    fn is_suit_symmetric(&self) -> bool {
        false
    }
    /// Maximum bucket cap on `street` (`None` for unabstracted `IdentityAbstraction`).
    /// Used by `estimate_tree_memory` as an exact upper bound across all runouts on `street`.
    fn max_buckets_for_street(&self, _street: Street) -> Option<usize> {
        None
    }
}

/// Exact combo-level identity abstraction (`1` infoset slot per 4-card combo).
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityAbstraction;

impl HandAbstraction for IdentityAbstraction {
    fn assign_buckets(
        &self,
        _player: usize,
        _board: &Board,
        player_combos: &[WeightedHand],
        _opponent_combos: &[WeightedHand],
    ) -> (usize, Vec<u32>) {
        let n = player_combos.len();
        (n, (0..n as u32).collect())
    }

    fn is_suit_symmetric(&self) -> bool {
        true
    }
}

/// Clusters active 4-card combos on each `(street, board)` into at most `max_buckets[street]` buckets
/// by hand strength / showdown equity quantiles on `board` (`u32::MAX` for board-blocked combos).
///
/// Designed for memory-bounded flop/turn solving when `IdentityAbstraction` would exceed WASM RAM.
#[derive(Clone, Copy, Debug)]
pub struct EquityBucketAbstraction {
    pub flop_buckets: usize,
    pub turn_buckets: usize,
    pub river_buckets: usize,
}

impl EquityBucketAbstraction {
    pub fn new(flop_buckets: usize, turn_buckets: usize, river_buckets: usize) -> Self {
        Self {
            flop_buckets: flop_buckets.max(1),
            turn_buckets: turn_buckets.max(1),
            river_buckets: river_buckets.max(1),
        }
    }
}

impl HandAbstraction for EquityBucketAbstraction {
    fn assign_buckets(
        &self,
        _player: usize,
        board: &Board,
        player_combos: &[WeightedHand],
        opponent_combos: &[WeightedHand],
    ) -> (usize, Vec<u32>) {
        let target_b = match board.street() {
            Street::Flop => self.flop_buckets,
            Street::Turn => self.turn_buckets,
            Street::River => self.river_buckets,
        };
        let n = player_combos.len();
        if n <= target_b {
            // Canonicalize by `ComboId` so bucket IDs `0..n` are invariant to slice permutation!
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by_key(|&i| player_combos[i].combo_id);
            let mut mapping = vec![0u32; n];
            for (b_id, &orig_idx) in order.iter().enumerate() {
                mapping[orig_idx] = b_id as u32;
            }
            return (n, mapping);
        }

        let bmask = board.mask();
        // Canonicalize `opponent_combos` by `ComboId` before computing `per_combo_river_equity` so
        // floating-point summation order in `pot_share` / `denom` is 100% invariant to caller slice order!
        let mut sorted_opp: Vec<WeightedHand> = opponent_combos.to_vec();
        sorted_opp.sort_by_key(|c| c.combo_id);

        // Score each unblocked hand: on river use exact equity vs opponent; on flop/turn use exact current
        // made-hand strength + draw potential proxy (immediate PLO 2+3 rank)
        // so bucket construction stays fast even across 1,980 runouts.
        let scores: Vec<f32> = if let Some(river_cards) = board.river_cards() {
            per_combo_river_equity(player_combos, &sorted_opp, river_cards)
        } else {
            player_combos
                .iter()
                .map(|c| {
                    if c.mask.intersects(bmask) {
                        -1.0
                    } else {
                        eval_plo(c.hand, board).map(|r| r.raw() as f32).unwrap_or(0.0)
                    }
                })
                .collect()
        };

        // Pre-sort live indices by `ComboId` and tie-break by `ComboId` (NEVER slice index!)
        let mut live_indices: Vec<usize> = (0..n)
            .filter(|&i| !player_combos[i].mask.intersects(bmask))
            .collect();
        live_indices.sort_by_key(|&i| player_combos[i].combo_id);
        live_indices.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| player_combos[a].combo_id.cmp(&player_combos[b].combo_id))
        });

        let actual_b = target_b.min(live_indices.len().max(1));
        let mut mapping = vec![0u32; n];
        let live_len = live_indices.len();
        for (rank_pos, &combo_idx) in live_indices.iter().enumerate() {
            let bucket = (rank_pos * actual_b) / live_len;
            mapping[combo_idx] = bucket.min(actual_b - 1) as u32;
        }
        (actual_b, mapping)
    }

    fn max_buckets_for_street(&self, street: Street) -> Option<usize> {
        Some(match street {
            Street::Flop => self.flop_buckets,
            Street::Turn => self.turn_buckets,
            Street::River => self.river_buckets,
        })
    }
}

/// Deterministic 2D PLO postflop abstraction (`NutBlockerBucketAbstraction`):
/// Keeps [`EquityBucketAbstraction`] intact while adding a second PLO-specific dimension capturing
/// **nuttedness, nut/second-nut flush draws & bare nut-flush blockers, and straight/wrap draw potential**!
///
/// Specifically, for a target bucket budget `K = equity_bins * nut_blocker_bins` (or general `target_buckets`),
/// each live 4-card combo on `board` is mapped to a deterministic 2D feature vector `(f_strength, f_nut_blocker)`:
/// 1. `f_strength`:
///    - On River: exact showdown equity against `opponent_combos` (`[0.0, 1.0]`).
///    - On Flop/Turn: exact next-street average PLO rank / tail strength across remaining deck cards
///      (normalized made + immediate 1-card improvement expectation).
/// 2. `f_nut_blocker`:
///    - Explicit PLO nuttedness & blocker score combining:
///      - **90th-percentile next-card nut ceiling** (separating nut wraps / nut flush draws that improve
///        to the nuts from medium bluff-catchers that have similar raw equity but zero nut outs);
///      - **Flush draw & bare nut-flush blocker score** on suited boards (`>= 2` cards of suit `s` on board:
///        holding `>= 2` hole cards of suit `s` with the live nut/2nd-nut rank scores actual nut flush draw;
///        holding **exactly 1** hole card of suit `s` when `>= 3` cards of suit `s` are on board — or `>= 2` on
///        flop/turn — scores **bare nut-flush blocker** (`As` / `Ks` when `As` is on board) even though 1 suit card
///        cannot make a flush in PLO!).
///
/// Combos are partitioned into `equity_bins` quantiles along `f_strength`, and within each equity quantile
/// partitioned into `nut_blocker_bins` quantiles along `f_nut_blocker` (producing up to `equity_bins * nut_blocker_bins`
/// deterministic buckets; when `target_b == 2` and `equity_bins == 1, nut_blocker_bins == 2` or general `(e, nb)`,
/// directly separates identical-equity combos that differ in nut flush blocker or nuttedness!).
#[derive(Clone, Copy, Debug)]
pub struct NutBlockerBucketAbstraction {
    pub flop_dims: (usize, usize),
    pub turn_dims: (usize, usize),
    pub river_dims: (usize, usize),
}

impl NutBlockerBucketAbstraction {
    /// Creates a 2D `(equity_bins, nut_blocker_bins)` abstraction per street (total buckets per street = `e * nb`).
    pub fn new(
        flop_dims: (usize, usize),
        turn_dims: (usize, usize),
        river_dims: (usize, usize),
    ) -> Self {
        Self {
            flop_dims: (flop_dims.0.max(1), flop_dims.1.max(1)),
            turn_dims: (turn_dims.0.max(1), turn_dims.1.max(1)),
            river_dims: (river_dims.0.max(1), river_dims.1.max(1)),
        }
    }

    /// Computes `(f_strength, f_nut_blocker)` for an unblocked `hand` on `board`.
    pub fn compute_2d_features(
        hand: plo_cards::Hand4,
        board: &Board,
        river_eq_opt: Option<f32>,
    ) -> (f32, f32) {
        let b_cards = board.as_slice();
        let bmask = board.mask();
        let hmask = hand.mask();
        let dead = bmask | hmask;

        // Feature B1: Board-aware flush draw & bare nut-flush blocker score!
        let mut board_suit_count = [0u8; 4];
        for &bc in b_cards {
            board_suit_count[bc.suit() as usize] += 1;
        }
        let mut hand_suit_count = [0u8; 4];
        for hc in hand.cards() {
            hand_suit_count[hc.suit() as usize] += 1;
        }

        let mut blocker_and_fd_score = 0.0f32;
        for s in 0..4u8 {
            let b_cnt = board_suit_count[s as usize];
            if b_cnt < 2 {
                continue;
            }
            // Find the highest and second-highest ranks of suit `s` NOT on the board (`nut_rank`, `second_nut_rank`)
            let mut top_off_board_ranks = [255u8; 2];
            let mut found = 0usize;
            for r in (0..13u8).rev() {
                let c = Card::from_rank_suit(r, s).unwrap();
                if !bmask.contains(c) {
                    top_off_board_ranks[found] = r;
                    found += 1;
                    if found == 2 {
                        break;
                    }
                }
            }
            let has_nut_card = top_off_board_ranks[0] != 255
                && hmask.contains(Card::from_rank_suit(top_off_board_ranks[0], s).unwrap());
            let has_second_nut_card = top_off_board_ranks[1] != 255
                && hmask.contains(Card::from_rank_suit(top_off_board_ranks[1], s).unwrap());

            let h_cnt = hand_suit_count[s as usize];
            if h_cnt >= 2 {
                // Legal 2-card PLO flush (if b_cnt >= 3) or frontdoor flush draw (if b_cnt == 2)!
                if has_nut_card {
                    blocker_and_fd_score += 2.0;
                } else if has_second_nut_card {
                    blocker_and_fd_score += 1.2;
                } else {
                    blocker_and_fd_score += 0.6;
                }
            } else if h_cnt == 1 {
                // PLO-specific BARE BLOCKER (`h_cnt == 1` CANNOT make a flush in PLO, yet blocks Villain's nut/2nd-nut flush!)
                let weight = if b_cnt >= 3 { 1.5 } else { 0.75 };
                if has_nut_card {
                    blocker_and_fd_score += weight;
                } else if has_second_nut_card {
                    blocker_and_fd_score += 0.4 * weight;
                }
            }
        }

        if let Some(eq) = river_eq_opt {
            return (eq, blocker_and_fd_score);
        }

        // Flop / Turn: evaluate all `52 - board.len() - 4` legal next-street cards to get:
        // - `mean_rank`: average next-card `HandRank` (`f_strength` proxy capturing made hand + all outs)
        // - `top4_rank`: average of top-4 best next-card `HandRank`s (`~90th percentile` nut/wrap ceiling!)
        let mut next_ranks = Vec::with_capacity(45);
        for cid in 0..52u8 {
            let c = Card::new(cid).unwrap();
            if dead.contains(c) {
                continue;
            }
            if let Ok(nb) = board.deal(c) {
                if let Ok(hr) = eval_plo(hand, &nb) {
                    next_ranks.push(hr.raw());
                }
            }
        }

        if next_ranks.is_empty() {
            let cur = eval_plo(hand, board).map(|r| r.raw() as f32 / 7462.0).unwrap_or(0.0);
            return (cur, blocker_and_fd_score);
        }

        next_ranks.sort_unstable();
        let sum_all: u32 = next_ranks.iter().map(|&x| x as u32).sum();
        let mean_norm = (sum_all as f32) / ((next_ranks.len() as f32) * 7462.0);

        let top_k = 4.min(next_ranks.len());
        let sum_top: u32 = next_ranks[next_ranks.len() - top_k..]
            .iter()
            .map(|&x| x as u32)
            .sum();
        let ceiling_norm = (sum_top as f32) / ((top_k as f32) * 7462.0);

        (mean_norm, ceiling_norm + blocker_and_fd_score)
    }
}

impl HandAbstraction for NutBlockerBucketAbstraction {
    fn assign_buckets(
        &self,
        _player: usize,
        board: &Board,
        player_combos: &[WeightedHand],
        opponent_combos: &[WeightedHand],
    ) -> (usize, Vec<u32>) {
        let (e_bins, nb_bins) = match board.street() {
            Street::Flop => self.flop_dims,
            Street::Turn => self.turn_dims,
            Street::River => self.river_dims,
        };
        let target_b = e_bins.saturating_mul(nb_bins).max(1);
        let n = player_combos.len();
        if n <= target_b {
            // Canonicalize by `ComboId` so bucket IDs `0..n` are invariant to slice permutation!
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by_key(|&i| player_combos[i].combo_id);
            let mut mapping = vec![0u32; n];
            for (b_id, &orig_idx) in order.iter().enumerate() {
                mapping[orig_idx] = b_id as u32;
            }
            return (n, mapping);
        }

        let bmask = board.mask();
        // Canonicalize `opponent_combos` by `ComboId` before computing `per_combo_river_equity` so
        // floating-point summation order in `pot_share` / `denom` is 100% invariant to caller slice order!
        let mut sorted_opp: Vec<WeightedHand> = opponent_combos.to_vec();
        sorted_opp.sort_by_key(|c| c.combo_id);

        let river_eqs = board
            .river_cards()
            .map(|rc| per_combo_river_equity(player_combos, &sorted_opp, rc));

        let mut features = vec![(0.0f32, 0.0f32); n];
        let mut live_indices = Vec::with_capacity(n);
        for (i, c) in player_combos.iter().enumerate() {
            if !c.mask.intersects(bmask) {
                let req = river_eqs.as_ref().map(|v| v[i]);
                features[i] = Self::compute_2d_features(c.hand, board, req);
                live_indices.push(i);
            }
        }

        if live_indices.is_empty() {
            return (1, vec![0u32; n]);
        }

        // Pre-sort live indices by `ComboId`, then primary sort by `f_strength` with deterministic `ComboId` tie-break!
        live_indices.sort_by_key(|&i| player_combos[i].combo_id);
        live_indices.sort_by(|&a, &b| {
            features[a]
                .0
                .partial_cmp(&features[b].0)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| player_combos[a].combo_id.cmp(&player_combos[b].combo_id))
        });

        let eff_e = e_bins.min(live_indices.len()).max(1);
        let mut mapping = vec![0u32; n];
        let mut next_bucket_id = 0u32;

        // Partition into `eff_e` strength slices, then sub-partition each slice into `nb_bins` by `f_nut_blocker`!
        let total_live = live_indices.len();
        for eb in 0..eff_e {
            let start = (eb * total_live) / eff_e;
            let end = ((eb + 1) * total_live) / eff_e;
            if start >= end {
                continue;
            }
            let slice = &mut live_indices[start..end];
            slice.sort_by(|&a, &b| {
                features[a]
                    .1
                    .partial_cmp(&features[b].1)
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then_with(|| player_combos[a].combo_id.cmp(&player_combos[b].combo_id))
            });
            let eff_nb = nb_bins.min(slice.len()).max(1);
            let slen = slice.len();
            for (pos, &cidx) in slice.iter().enumerate() {
                let sub = ((pos * eff_nb) / slen).min(eff_nb - 1) as u32;
                mapping[cidx] = next_bucket_id + sub;
            }
            next_bucket_id += eff_nb as u32;
        }

        (next_bucket_id.max(1) as usize, mapping)
    }

    fn max_buckets_for_street(&self, street: Street) -> Option<usize> {
        let (e, nb) = match street {
            Street::Flop => self.flop_dims,
            Street::Turn => self.turn_dims,
            Street::River => self.river_dims,
        };
        Some(e.saturating_mul(nb).max(1))
    }
}

// ============================================================================
// Precomputed River Showdown Data & Exact 4-Card 15-Term Inclusion-Exclusion
// ============================================================================

#[derive(Clone, Copy)]
struct SortedStrength {
    strength: u16,
    combo_idx: u32,
}

/// Precomputed per-river evaluation structure enabling `O(N_oop + N_ip)` exact PLO terminal evaluation!
#[derive(Clone)]
struct RiverEvalData {
    /// Unblocked combo indices `[OOP, IP]` on this 5-card river.
    valid_combos: [Vec<u32>; 2],
    /// Ascending `(HandRank, combo_idx)` for unblocked combos `[OOP, IP]` on this river.
    sorted_strengths: [Vec<SortedStrength>; 2],
}

/// Precomputed 15-subset indices (`4` singles, `6` pairs, `4` triples, `1` same-4-card-hand index in opp range)
/// for a single 4-card hand `c0 < c1 < c2 < c3`.
#[derive(Clone, Copy)]
struct ComboBlockerSubsets {
    singles: [u8; 4],
    pairs: [u16; 6],
    triples: [u16; 4],
    same_opp_idx: u32, // u32::MAX if opponent range does not contain this exact 4-card hand
}

impl ComboBlockerSubsets {
    fn build(hand: plo_cards::Hand4, opp_combos: &[ComboId]) -> Self {
        let [c0, c1, c2, c3] = hand.cards();
        let singles = [c0.id(), c1.id(), c2.id(), c3.id()];
        let pairs = [
            pair_subset_index(c0, c1) as u16,
            pair_subset_index(c0, c2) as u16,
            pair_subset_index(c0, c3) as u16,
            pair_subset_index(c1, c2) as u16,
            pair_subset_index(c1, c3) as u16,
            pair_subset_index(c2, c3) as u16,
        ];
        let triples = [
            triple_subset_index(c0, c1, c2) as u16,
            triple_subset_index(c0, c1, c3) as u16,
            triple_subset_index(c0, c2, c3) as u16,
            triple_subset_index(c1, c2, c3) as u16,
        ];
        let same_opp_idx = match opp_combos.binary_search(&hand.combo_id()) {
            Ok(idx) => idx as u32,
            Err(_) => u32::MAX,
        };
        Self {
            singles,
            pairs,
            triples,
            same_opp_idx,
        }
    }
}

/// Reusable scratchpad for `O(N + M)` 4-card 15-term inclusion-exclusion with dirty-index reset!
///
/// Total static footprint: `52*8 + 1326*8 + 22100*8 = 187,824` bytes (~183 KB), allocated ONCE
/// per solver call (never per node!), with `dirty_pairs` and `dirty_triples` tracking touched slots
/// so reset takes `O(touched)` instead of scanning `22,100` entries.
struct InclusionExclusionScratch {
    total: f64,
    w1: [f64; 52],
    w2: Vec<f64>, // len 1326
    w3: Vec<f64>, // len 22100
    dirty_w2: Vec<u16>,
    dirty_w3: Vec<u16>,
}

impl InclusionExclusionScratch {
    fn new() -> Self {
        Self {
            total: 0.0,
            w1: [0.0; 52],
            w2: vec![0.0; 1326],
            w3: vec![0.0; 22_100],
            dirty_w2: Vec::with_capacity(1024),
            dirty_w3: Vec::with_capacity(1024),
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.total = 0.0;
        self.w1.fill(0.0);
        for &idx in &self.dirty_w2 {
            self.w2[idx as usize] = 0.0;
        }
        self.dirty_w2.clear();
        for &idx in &self.dirty_w3 {
            self.w3[idx as usize] = 0.0;
        }
        self.dirty_w3.clear();
    }

    #[inline]
    fn add_combo(&mut self, sub: &ComboBlockerSubsets, weight: f64) {
        if weight == 0.0 {
            return;
        }
        self.total += weight;
        for &s in &sub.singles {
            self.w1[s as usize] += weight;
        }
        for &p in &sub.pairs {
            let slot = &mut self.w2[p as usize];
            if *slot == 0.0 {
                self.dirty_w2.push(p);
            }
            *slot += weight;
        }
        for &t in &sub.triples {
            let slot = &mut self.w3[t as usize];
            if *slot == 0.0 {
                self.dirty_w3.push(t);
            }
            *slot += weight;
        }
    }

    /// Queries the exact sum of weights of added opponent combos that share **ZERO** cards with `sub`!
    #[inline]
    fn query_disjoint(&self, sub: &ComboBlockerSubsets, opp_cfreach: &[f32]) -> f64 {
        let s1 = self.w1[sub.singles[0] as usize]
            + self.w1[sub.singles[1] as usize]
            + self.w1[sub.singles[2] as usize]
            + self.w1[sub.singles[3] as usize];
        let s2 = self.w2[sub.pairs[0] as usize]
            + self.w2[sub.pairs[1] as usize]
            + self.w2[sub.pairs[2] as usize]
            + self.w2[sub.pairs[3] as usize]
            + self.w2[sub.pairs[4] as usize]
            + self.w2[sub.pairs[5] as usize];
        let s3 = self.w3[sub.triples[0] as usize]
            + self.w3[sub.triples[1] as usize]
            + self.w3[sub.triples[2] as usize]
            + self.w3[sub.triples[3] as usize];
        let s4 = if sub.same_opp_idx != u32::MAX {
            opp_cfreach[sub.same_opp_idx as usize] as f64
        } else {
            0.0
        };
        (self.total - s1 + s2 - s3 + s4).max(0.0)
    }

    #[inline]
    fn query_disjoint_f64(&self, sub: &ComboBlockerSubsets, opp_cfreach: &[f64]) -> f64 {
        let s1 = self.w1[sub.singles[0] as usize]
            + self.w1[sub.singles[1] as usize]
            + self.w1[sub.singles[2] as usize]
            + self.w1[sub.singles[3] as usize];
        let s2 = self.w2[sub.pairs[0] as usize]
            + self.w2[sub.pairs[1] as usize]
            + self.w2[sub.pairs[2] as usize]
            + self.w2[sub.pairs[3] as usize]
            + self.w2[sub.pairs[4] as usize]
            + self.w2[sub.pairs[5] as usize];
        let s3 = self.w3[sub.triples[0] as usize]
            + self.w3[sub.triples[1] as usize]
            + self.w3[sub.triples[2] as usize]
            + self.w3[sub.triples[3] as usize];
        let s4 = if sub.same_opp_idx != u32::MAX {
            opp_cfreach[sub.same_opp_idx as usize]
        } else {
            0.0
        };
        (self.total - s1 + s2 - s3 + s4).max(0.0)
    }

    /// Queries disjoint weight when `same_opp_idx` may or may not have been inserted into `self` yet
    /// (e.g. during strict `< strength` or `> strength` showdown sweeps, `same_opp_idx` has identical
    /// 4 cards and therefore identical `HandRank`, so it is NEVER strictly `<` or strictly `>`!).
    #[inline]
    fn query_disjoint_strict_ineq(&self, sub: &ComboBlockerSubsets) -> f64 {
        let s1 = self.w1[sub.singles[0] as usize]
            + self.w1[sub.singles[1] as usize]
            + self.w1[sub.singles[2] as usize]
            + self.w1[sub.singles[3] as usize];
        let s2 = self.w2[sub.pairs[0] as usize]
            + self.w2[sub.pairs[1] as usize]
            + self.w2[sub.pairs[2] as usize]
            + self.w2[sub.pairs[3] as usize]
            + self.w2[sub.pairs[4] as usize]
            + self.w2[sub.pairs[5] as usize];
        let s3 = self.w3[sub.triples[0] as usize]
            + self.w3[sub.triples[1] as usize]
            + self.w3[sub.triples[2] as usize]
            + self.w3[sub.triples[3] as usize];
        // Note: `s4` is 0 because an identical 4-card hand has the EXACT SAME strength on the same board,
        // so it can never be inserted during `< strength` or `> strength` passes!
        (self.total - s1 + s2 - s3).max(0.0)
    }
}

// ============================================================================
// Concrete Expanded Game Tree & DCFR Storage
// ============================================================================

#[derive(Clone, Debug)]
struct ChanceIsoBranch {
    card: Card,
    rep_branch_idx: usize,
    swaps: [Vec<(u32, u32)>; 2],
}

#[derive(Clone, Debug)]
struct PlayerStorage {
    /// Length `actions.len() * num_buckets`.
    regrets: Vec<f32>,
    /// Length `actions.len() * num_buckets`.
    cum_strategy: Vec<f32>,
}

/// Thread-safe interior-mutable wrapper around `PlayerStorage` allowing disjoint chance-branch subtrees
/// (`rep_children`) in `dcfr_traverse` to update their own strict-tree decision nodes concurrently via `rayon`
/// while zero-locking single-threaded/WASM execution!
struct SyncStorageCell(std::cell::UnsafeCell<PlayerStorage>);

// SAFETY: `ConcreteNode` forms a strict tree (`expand_node` creates fresh `ConcreteNode` indices for every
// child branch; only `RiverEvalData` and `Arc<[u32]>` bucket maps are shared read-only). During `dcfr_traverse(&self)`,
// parallel tasks spawned across `rep_children` of a `Chance` node visit strictly disjoint sets of `node_id`s,
// so no two threads ever access `self.nodes[nid]`'s `SyncStorageCell` simultaneously.
unsafe impl Sync for SyncStorageCell {}

impl SyncStorageCell {
    #[inline]
    fn new(len: usize) -> Self {
        Self(std::cell::UnsafeCell::new(PlayerStorage {
            regrets: vec![0.0f32; len],
            cum_strategy: vec![0.0f32; len],
        }))
    }

    #[inline]
    fn as_ref(&self) -> &PlayerStorage {
        // SAFETY: Shared read-only callers (`inspect_current_node`, `strategy_cfv_traverse`, `best_response_traverse`)
        // hold `&self` on `PloGame` with no concurrent `dcfr_traverse` running.
        unsafe { &*self.0.get() }
    }

    #[inline]
    fn as_mut(&mut self) -> &mut PlayerStorage {
        self.0.get_mut()
    }

    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn as_disjoint_mut(&self) -> &mut PlayerStorage {
        unsafe { &mut *self.0.get() }
    }
}

impl std::fmt::Debug for SyncStorageCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}

/// Returns the number of native hardware worker threads used by `plo_cfr` chance-branch parallelism (`1` on `wasm32`).
pub fn native_solver_threads() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 16))
            .unwrap_or(1)
    }
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
}

#[cfg(not(target_arch = "wasm32"))]
std::thread_local! {
    static IN_PARALLEL_CHANCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

std::thread_local! {
    static TLS_IE_SCRATCH_POOL: std::cell::RefCell<Vec<InclusionExclusionScratch>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Executes `f` with a reusable thread-local [`InclusionExclusionScratch`] (`~183 KiB`),
/// eliminating thousands of per-chance-branch `mmap`/`munmap` allocations during Flop/Turn solves.
#[inline]
fn with_tls_ie_scratch<R>(f: impl FnOnce(&mut InclusionExclusionScratch) -> R) -> R {
    let mut scratch = TLS_IE_SCRATCH_POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_else(InclusionExclusionScratch::new);
    let res = f(&mut scratch);
    TLS_IE_SCRATCH_POOL.with(|pool| {
        let mut p = pool.borrow_mut();
        if p.len() < 4 {
            p.push(scratch);
        }
    });
    res
}

/// Zero-dependency native `std::thread::scope` parallel mapper over `(rep_cards, rep_children)`.
/// Parallelizes top-level chance nodes across up to `native_solver_threads()` worker threads without
/// nested thread oversubscription (`Turn->River` inside `Flop->Turn` runs inline on each worker thread!),
/// passing one reusable [`InclusionExclusionScratch`] per worker thread across all branches in its chunk!
#[cfg(not(target_arch = "wasm32"))]
fn map_chance_branches<T, F>(rep_cards: &[Card], rep_children: &[usize], f: F) -> Vec<T>
where
    T: Send,
    F: Fn(Card, usize, &mut InclusionExclusionScratch) -> T + Sync,
{
    let n = rep_cards.len();
    let n_threads = native_solver_threads();
    let already_par = IN_PARALLEL_CHANCE.with(|c| c.get());
    if n <= 2 || n_threads <= 1 || already_par {
        return with_tls_ie_scratch(|scratch| {
            rep_cards
                .iter()
                .zip(rep_children.iter())
                .map(|(&card, &cid)| f(card, cid, scratch))
                .collect()
        });
    }

    let workers = n_threads.min(n);
    let chunk_size = n.div_ceil(workers);
    let f_ref = &f;

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for (c_chunk, id_chunk) in rep_cards.chunks(chunk_size).zip(rep_children.chunks(chunk_size))
        {
            handles.push(s.spawn(move || {
                IN_PARALLEL_CHANCE.with(|cell| cell.set(true));
                let res: Vec<T> = with_tls_ie_scratch(|scratch| {
                    c_chunk
                        .iter()
                        .zip(id_chunk.iter())
                        .map(|(&card, &cid)| f_ref(card, cid, scratch))
                        .collect()
                });
                IN_PARALLEL_CHANCE.with(|cell| cell.set(false));
                res
            }));
        }
        let mut out = Vec::with_capacity(n);
        for h in handles {
            out.extend(h.join().expect("chance worker thread panicked"));
        }
        out
    })
}

#[derive(Debug)]
enum ConcreteNodeKind {
    Player {
        player: usize,
        actions: Vec<Action>,
        children: Vec<usize>,
        num_buckets: usize,
        /// `combo_to_bucket[combo_idx]` in `0..num_buckets` (empty slice when identity abstraction `b == combo_idx`).
        /// Shared via `Arc<[u32]>` across all player nodes of `(board, player)` so memory & build time scale
        /// with unique boards (`<= 1,226`) instead of concrete decision nodes (`34,880`..`57,240`)!
        combo_to_bucket: std::sync::Arc<[u32]>,
        storage: SyncStorageCell,
        /// Optional node-locked strategy (`actions.len() * num_buckets`, normalized per bucket).
        /// `None` in standard unexploited Nash solving (zero overhead / identical behavior).
        locked_strategy: Option<Vec<f32>>,
    },
    Chance {
        /// Dealt representative cards and their child `ConcreteNode` indices.
        rep_cards: Vec<Card>,
        rep_children: Vec<usize>,
        /// Isomorphic skipped cards that reuse `rep_children[rep_branch_idx]` via range swaps.
        iso_duplicates: Vec<ChanceIsoBranch>,
        /// Exact combinatorial chance denominator conditional on any compatible 4-card vs 4-card hand pair:
        /// `52 - board.len() - 8` (`41` on Flop->Turn, `40` on Turn->River)!
        chance_denominator: usize,
    },
    TerminalFold {
        folded_player: usize,
        half_pot_swing: f64, // `committed[folded_player]`? Wait: let's check zero-sum baseline!
        board_mask: CardMask,
    },
    TerminalShowdown {
        river_eval_id: usize,
        half_pot_swing: f64,
        rake_chips: f64,
    },
}

// Zero-sum payoff convention (matching `postflop-solver`):
// Let `starting_pot` be in the pot before root. Each player's neutral baseline is `0.5 * starting_pot`.
// - At a fold terminal where `folded_player` committed `c_fold`:
//   contested pot is `starting_pot + 2 * c_fold`, so relative to `0.5 * starting_pot`,
//   winner gets `+(0.5 * starting_pot + c_fold)` and folder gets `-(0.5 * starting_pot + c_fold)`.
// - At a showdown terminal where both committed `c`:
//   winner gets `+(0.5 * starting_pot + c)`, loser gets `-(0.5 * starting_pot + c)`, and split pot nets `0.0`.
// Adding `0.5 * starting_pot` converts centered zero-sum EV directly to postflop chip EV.

#[derive(Debug)]
struct ConcreteNode {
    board: Board,
    committed: [i32; 2],
    pot: i32,
    kind: ConcreteNodeKind,
}

/// Discounted CFR parameter schedule: a variant in the spirit of Brown & Sandholm (AAAI 2019,
/// arXiv:1809.04040) `DCFR(3/2, 0, 2)`, using:
/// - `alpha_t = t^{3/2} / (t^{3/2} + 1)` on positive cumulative regrets,
/// - `beta = 0` (CFR+-style zeroing of negative cumulative regrets), and
/// - multiplicative cumulative-strategy step discount `gamma_t = ((t - 1) / t)^2`, which yields
///   an exact quadratic `(t / T)^2` weight profile across iterations `t = 1..=T` (no power-of-4 resets).
#[derive(Clone, Copy)]
struct DcfrParams {
    alpha: f32,
    beta: f32,
    gamma: f32,
}

impl DcfrParams {
    /// Computes per-iteration discount factors (`t = iter + 1` in `1, 2, 3, ...`):
    /// - `alpha_t = t^{1.5} / (t^{1.5} + 1)`
    /// - `beta_t = 0.0` (zeroes negative cumulative regrets)
    /// - `gamma_t = ((t - 1) / t)^2` (gives iteration `t` final weight `(t / T)^2` after `T` steps)
    fn for_iteration(iter: u32) -> Self {
        let t = (iter + 1) as f64;
        let pow_alpha = t * t.sqrt();
        let gamma_ratio = (iter as f64) / t;
        Self {
            alpha: (pow_alpha / (pow_alpha + 1.0)) as f32,
            beta: 0.0,
            gamma: (gamma_ratio * gamma_ratio) as f32,
        }
    }
}

/// Navigation step in a concrete [`PloGame`] (`ActionIndex` at a player node or `Card` at a chance node).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoryStep {
    Action(usize),
    Deal(Card),
}

/// Summary of node state, strategy, and per-combo EVs at the currently inspected node of [`PloGame`].
#[derive(Clone, Debug)]
pub struct NodeReport {
    pub board: Board,
    pub pot: i32,
    pub committed: [i32; 2],
    /// `"oop"`, `"ip"`, `"chance"`, or `"terminal"`.
    pub node_type: &'static str,
    pub actions: Vec<Action>,
    /// Row-major `actions.len() * num_combos(acting_player)` average strategy probabilities
    /// (`strategy[action_idx * num_combos + combo_idx]`). Empty at chance/terminal nodes.
    pub strategy: Vec<f32>,
    /// Per-combo EV for OOP (`len == oop_combos.len()`), expressed in chips from start of current node's street/pot
    /// (at root: net chip EV where folding at root with 0 invested gives `0.0` and winningstarting pot gives `starting_pot`).
    pub ev_oop: Vec<f32>,
    /// Per-combo EV for IP (`len == ip_combos.len()`).
    pub ev_ip: Vec<f32>,
}

/// Per-card Turn/River runout summary at a `Chance` node (`#3` `4×13` Runout EV Heatmap).
#[derive(Clone, Debug)]
pub struct ChanceRunoutSummary {
    pub card: Card,
    pub next_node_type: &'static str,
    pub next_actions: Vec<Action>,
    /// Reach-weighted aggregate next-node action frequencies (`len == next_actions.len()`).
    pub next_agg_freqs: Vec<f32>,
    /// Per-combo conditional EVs for OOP on this dealt runout (`len == oop_combos.len()`).
    pub ev_oop: Vec<f32>,
    /// Per-combo conditional EVs for IP on this dealt runout (`len == ip_combos.len()`).
    pub ev_ip: Vec<f32>,
    /// Reach-weighted average OOP EV across live non-overlapping pairs on this runout.
    pub range_ev_oop: f32,
    /// Reach-weighted average IP EV across live non-overlapping pairs on this runout.
    pub range_ev_ip: f32,
}

/// Memory footprint report for a [`PloGame`] instance.
#[derive(Clone, Copy, Debug)]
pub struct MemoryReport {
    pub decision_nodes: usize,
    pub chance_nodes: usize,
    pub terminal_nodes: usize,
    pub river_eval_tables: usize,
    pub regret_and_strategy_bytes: u64,
    pub river_eval_bytes: u64,
    pub total_estimated_bytes: u64,
}

/// Main Pot-Limit Omaha postflop game & DCFR solver instance.
pub struct PloGame {
    root_board: Board,
    tree_config: TreeConfig,
    /// Filtered initial ranges (board-blocked combos removed at root), sorted by `ComboId`.
    combos: [Vec<WeightedHand>; 2],
    initial_weights: [Vec<f32>; 2],
    blocker_subsets: [Vec<ComboBlockerSubsets>; 2],
    /// Sum of initial `w_oop * w_ip` over all non-overlapping 4-card root combo pairs.
    valid_root_pair_weight: f64,
    nodes: Vec<ConcreteNode>,
    river_evals: Vec<RiverEvalData>,
    iterations_run: u32,
    enable_chance_isomorphism: bool,
    rake_fraction: f64,
    rake_cap_chips: f64,
    cursor_node: usize,
    /// Maps real-world suit `s` (`0..4`) to canonical node suit `cursor_suit_perm[s]` when navigating
    /// isomorphic chance branches via `play_chance_card`.
    cursor_suit_perm: [u8; 4],
    /// Maps real-world combo index `i` (`0..combos[p].len()`) to canonical node combo index `cursor_combo_map[p][i]`.
    cursor_combo_map: [Vec<usize>; 2],
    history: Vec<HistoryStep>,
    mem_report: MemoryReport,
}

/// Builder options for [`PloGame`].
#[derive(Clone, Debug)]
pub struct SolverBuildOptions {
    pub max_memory_bytes: u64,
    pub enable_chance_isomorphism: bool,
    /// Optional showdown pot rake fraction (`0.0` by default = exact zero-sum game).
    pub rake_fraction: f64,
    /// Optional showdown rake cap in chips (`0.0` = uncapped when `rake_fraction > 0.0`).
    pub rake_cap_chips: f64,
}

impl Default for SolverBuildOptions {
    fn default() -> Self {
        Self {
            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
            enable_chance_isomorphism: true,
            rake_fraction: 0.0,
            rake_cap_chips: 0.0,
        }
    }
}

impl PloGame {
    /// Builds an exact combo-level `PloGame` (`IdentityAbstraction`) with default memory guardrails.
    pub fn new(
        board: Board,
        oop_range: &PloRange,
        ip_range: &PloRange,
        action_tree: &ActionTree,
    ) -> Result<Self, SolverError> {
        Self::with_options(
            board,
            oop_range,
            ip_range,
            action_tree,
            &IdentityAbstraction,
            SolverBuildOptions::default(),
        )
    }

    /// Builds a `PloGame` with custom [`HandAbstraction`] and [`SolverBuildOptions`].
    pub fn with_options(
        board: Board,
        oop_range: &PloRange,
        ip_range: &PloRange,
        action_tree: &ActionTree,
        abstraction: &dyn HandAbstraction,
        options: SolverBuildOptions,
    ) -> Result<Self, SolverError> {
        if board.street() != action_tree.config().initial_street {
            return Err(SolverError::StreetMismatch {
                board_street: board.street(),
                tree_street: action_tree.config().initial_street,
            });
        }

        // Remove root-board-blocked and zero-weight combos immediately (`w > 0.0` strictly)
        let live_oop = oop_range.without_dead(board.mask());
        let live_ip = ip_range.without_dead(board.mask());
        let combos = [
            live_oop
                .entries()
                .iter()
                .copied()
                .filter(|c| c.weight > 0.0)
                .collect::<Vec<_>>(),
            live_ip
                .entries()
                .iter()
                .copied()
                .filter(|c| c.weight > 0.0)
                .collect::<Vec<_>>(),
        ];
        if combos[0].is_empty() || combos[1].is_empty() {
            return Err(SolverError::Range(RangeError::EmptyRange));
        }
        let n_oop = combos[0].len();
        let n_ip = combos[1].len();
        let combo_ids = [
            combos[0].iter().map(|c| c.combo_id).collect::<Vec<_>>(),
            combos[1].iter().map(|c| c.combo_id).collect::<Vec<_>>(),
        ];
        let initial_weights = [
            combos[0].iter().map(|c| c.weight).collect::<Vec<_>>(),
            combos[1].iter().map(|c| c.weight).collect::<Vec<_>>(),
        ];

        // Compute root valid non-overlapping combo pair weight sum
        let mut valid_root_pair_weight = 0.0f64;
        for c0 in &combos[0] {
            for c1 in &combos[1] {
                if !c0.mask.intersects(c1.mask) {
                    valid_root_pair_weight += (c0.weight as f64) * (c1.weight as f64);
                }
            }
        }
        if valid_root_pair_weight <= 0.0 {
            return Err(SolverError::NoValidMatchups);
        }

        let blocker_subsets = [
            combos[0]
                .iter()
                .map(|c| ComboBlockerSubsets::build(c.hand, &combo_ids[1]))
                .collect(),
            combos[1]
                .iter()
                .map(|c| ComboBlockerSubsets::build(c.hand, &combo_ids[0]))
                .collect(),
        ];

        let effective_iso = options.enable_chance_isomorphism && abstraction.is_suit_symmetric();

        let oop_iso_vec: Vec<(ComboId, f32)> =
            combos[0].iter().map(|c| (c.combo_id, c.weight)).collect();
        let ip_iso_vec: Vec<(ComboId, f32)> =
            combos[1].iter().map(|c| (c.combo_id, c.weight)).collect();
        let mut iso_cache: std::collections::HashMap<u64, ChanceIsomorphism> =
            std::collections::HashMap::new();

        // Estimate concrete memory BEFORE allocating giant Flop/Turn arrays!
        let est = estimate_tree_memory(
            &board,
            &combos,
            &combo_ids,
            action_tree,
            abstraction,
            effective_iso,
            &oop_iso_vec,
            &ip_iso_vec,
            &mut iso_cache,
        );
        if est.total_estimated_bytes > options.max_memory_bytes {
            let avg_b = (combos[0].len() + combos[1].len()) / 2;
            return Err(SolverError::MemoryBudgetExceeded {
                required_bytes: est.total_estimated_bytes,
                budget_bytes: options.max_memory_bytes,
                decision_nodes: est.decision_nodes,
                avg_buckets: avg_b,
            });
        }

        let mut builder = ConcreteBuilder {
            combos: &combos,
            combo_ids: &combo_ids,
            oop_iso_vec,
            ip_iso_vec,
            iso_cache,
            action_tree,
            abstraction,
            enable_iso: effective_iso,
            rake_fraction: options.rake_fraction.clamp(0.0, 0.5),
            rake_cap_chips: options.rake_cap_chips.max(0.0),
            nodes: Vec::with_capacity(est.decision_nodes + est.chance_nodes + est.terminal_nodes),
            river_evals: Vec::new(),
            river_cache_map: std::collections::HashMap::new(),
            bucket_cache_map: std::collections::HashMap::new(),
            empty_bucket_map: std::sync::Arc::from([] as [u32; 0]),
            regret_strat_bytes: 0,
            bucket_map_bytes: 0,
        };

        builder.expand_node(action_tree.root_id(), board);
        let ConcreteBuilder {
            nodes,
            river_evals,
            regret_strat_bytes,
            bucket_map_bytes,
            ..
        } = builder;

        let river_eval_bytes: u64 = river_evals
            .iter()
            .map(|re| {
                ((re.valid_combos[0].len() + re.valid_combos[1].len()) * 4
                    + (re.sorted_strengths[0].len() + re.sorted_strengths[1].len()) * 8)
                    as u64
            })
            .sum();

        let mut dec_nodes = 0usize;
        let mut ch_nodes = 0usize;
        let mut term_nodes = 0usize;
        for n in &nodes {
            match n.kind {
                ConcreteNodeKind::Player { .. } => dec_nodes += 1,
                ConcreteNodeKind::Chance { .. } => ch_nodes += 1,
                ConcreteNodeKind::TerminalFold { .. }
                | ConcreteNodeKind::TerminalShowdown { .. } => term_nodes += 1,
            }
        }

        let mem_report = MemoryReport {
            decision_nodes: dec_nodes,
            chance_nodes: ch_nodes,
            terminal_nodes: term_nodes,
            river_eval_tables: river_evals.len(),
            regret_and_strategy_bytes: regret_strat_bytes,
            river_eval_bytes,
            total_estimated_bytes: regret_strat_bytes
                + bucket_map_bytes
                + river_eval_bytes
                + (nodes.len() as u64) * 192,
        };

        Ok(Self {
            root_board: board,
            tree_config: action_tree.config().clone(),
            combos,
            initial_weights,
            blocker_subsets,
            valid_root_pair_weight,
            nodes,
            river_evals,
            iterations_run: 0,
            enable_chance_isomorphism: effective_iso,
            rake_fraction: options.rake_fraction.clamp(0.0, 0.5),
            rake_cap_chips: options.rake_cap_chips.max(0.0),
            cursor_node: 0,
            cursor_suit_perm: [0, 1, 2, 3],
            cursor_combo_map: [(0..n_oop).collect(), (0..n_ip).collect()],
            history: Vec::new(),
            mem_report,
        })
    }

    #[inline]
    pub fn root_board(&self) -> &Board {
        &self.root_board
    }

    #[inline]
    pub fn tree_config(&self) -> &TreeConfig {
        &self.tree_config
    }

    #[inline]
    pub fn combos(&self, player: usize) -> &[WeightedHand] {
        &self.combos[player]
    }

    #[inline]
    pub fn iterations_run(&self) -> u32 {
        self.iterations_run
    }

    #[inline]
    pub fn rake_config(&self) -> (f64, f64) {
        (self.rake_fraction, self.rake_cap_chips)
    }

    #[inline]
    pub fn memory_report(&self) -> MemoryReport {
        self.mem_report
    }

    #[inline]
    pub fn is_chance_isomorphism_enabled(&self) -> bool {
        self.enable_chance_isomorphism
    }

    /// Runs a single Discounted CFR iteration (`OOP` update followed by `IP` update).
    pub fn solve_step(&mut self) {
        let params = DcfrParams::for_iteration(self.iterations_run);

        with_tls_ie_scratch(|scratch| {
            for player in [PLAYER_OOP, PLAYER_IP] {
                let opp = player ^ 1;
                let _ = self.dcfr_traverse(
                    0,
                    player,
                    &self.initial_weights[player],
                    &self.initial_weights[opp],
                    &params,
                    scratch,
                );
            }
        });

        self.iterations_run += 1;
    }

    /// Runs up to `max_iterations` of DCFR, stopping early if exploitability (in chip units)
    /// drops to or below `target_exploitability` (checked every `25` iterations and at completion
    /// to amortize full best-response traversal cost; use [`Self::solve_with_hooks`] for a custom check stride).
    /// Returns final exploitability in chips.
    pub fn solve(&mut self, max_iterations: u32, target_exploitability: f32) -> f32 {
        let mut expl = self.exploitability();
        for i in 0..max_iterations {
            if expl <= target_exploitability {
                return expl;
            }
            self.solve_step();
            if (i + 1) % 25 == 0 || i + 1 == max_iterations {
                expl = self.exploitability();
            }
        }
        expl
    }

    /// Runs up to `max_iterations` of DCFR with cooperative `stop_flag` cancellation between iterations
    /// and a periodic `on_progress` hook (`(iterations_run, exploitability_f64, [ev_oop, ev_ip])`)
    /// invoked every `report_every.max(1)` iterations and on completion/stop.
    pub fn solve_with_hooks<F>(
        &mut self,
        max_iterations: u32,
        target_exploitability: f64,
        report_every: u32,
        stop_flag: &core::sync::atomic::AtomicBool,
        mut on_progress: F,
    ) -> (f64, [f64; 2], bool)
    where
        F: FnMut(u32, f64, [f64; 2]),
    {
        use core::sync::atomic::Ordering;
        let stride = report_every.max(1);
        let mut stopped = false;

        if max_iterations == 0 {
            let expl = self.exploitability_f64();
            let evs = self.root_expected_values_f64();
            on_progress(self.iterations_run, expl, evs);
            return (expl, evs, false);
        }

        let mut expl = self.exploitability_f64();
        let mut evs = self.root_expected_values_f64();

        for i in 0..max_iterations {
            if stop_flag.load(Ordering::Relaxed) {
                stopped = true;
                break;
            }
            if target_exploitability > 0.0 && expl <= target_exploitability {
                break;
            }
            self.solve_step();
            let done = i + 1;
            if done % stride == 0 || done == max_iterations {
                expl = self.exploitability_f64();
                evs = self.root_expected_values_f64();
                on_progress(self.iterations_run, expl, evs);
            }
        }

        if stopped || max_iterations % stride != 0 {
            expl = self.exploitability_f64();
            evs = self.root_expected_values_f64();
            on_progress(self.iterations_run, expl, evs);
        }

        (expl, evs, stopped)
    }

    /// Computes exact Nash exploitability (`0.5 * (MES_OOP + MES_IP)` in chip units; `0.0` at exact Nash equilibrium).
    ///
    /// Note: Even when `HandAbstraction` buckets decision strategies, `exploitability()` computes the
    /// **exact unabstracted combo-level best response** against the induced strategy!
    pub fn exploitability(&self) -> f32 {
        let mut mes_sum = [0.0f64; 2];
        with_tls_ie_scratch(|scratch| {
            for player in [PLAYER_OOP, PLAYER_IP] {
                let opp = player ^ 1;
                let best_cfv =
                    self.best_response_traverse(0, player, &self.initial_weights[opp], scratch);
                mes_sum[player] = best_cfv
                    .iter()
                    .zip(&self.initial_weights[player])
                    .map(|(&v, &w)| (v as f64) * (w as f64))
                    .sum();
            }
        });

        // Since zero-rake payoffs are strictly zero-sum (`EV_OOP + EV_IP == starting_pot`, even when
        // nodes are locked), centered `mes_sum[0] + mes_sum[1] == (BR_OOP - 0.5*pot) + (BR_IP - 0.5*pot)
        // == (BR_OOP + BR_IP) - (EV_OOP + EV_IP) >= 0`, so `0.5 * (mes_sum[0] + mes_sum[1])` is exact
        // Nash distance! When `rake_fraction > 0`, `EV_OOP + EV_IP < starting_pot`, so we subtract
        // `(EV_OOP, EV_IP)` explicitly to preserve `0.5 * ((BR_OOP - EV_OOP) + (BR_IP - EV_IP))`.
        if self.rake_fraction > 0.0 {
            let [ev_oop, ev_ip] = self.root_expected_values();
            let half_pot = 0.5 * (self.tree_config.starting_pot as f64);
            let br0 = half_pot + mes_sum[0];
            let br1 = half_pot + mes_sum[1];
            (0.5 * ((br0 - ev_oop as f64) + (br1 - ev_ip as f64))).max(0.0) as f32
        } else {
            (0.5 * (mes_sum[0] + mes_sum[1])).max(0.0) as f32
        }
    }

    /// Computes exact best-response expected values `[BR_OOP, BR_IP]` in chip units against the frozen
    /// average strategy (`BR_p >= EV_p`, and `0.5 * ((BR_OOP - EV_OOP) + (BR_IP - EV_IP)) == exploitability()`).
    pub fn best_response_values_f64(&self) -> [f64; 2] {
        let half_pot = 0.5 * (self.tree_config.starting_pot as f64);
        let mut br = [0.0f64; 2];

        with_tls_ie_scratch(|scratch| {
            for player in [PLAYER_OOP, PLAYER_IP] {
                let opp = player ^ 1;
                let best_cfv =
                    self.best_response_traverse(0, player, &self.initial_weights[opp], scratch);
                let mes_centered: f64 = best_cfv
                    .iter()
                    .zip(&self.initial_weights[player])
                    .map(|(&v, &w)| (v as f64) * (w as f64))
                    .sum();
                br[player] = half_pot + mes_centered;
            }
        });
        br
    }

    /// Computes exact Nash exploitability in `f64` (`0.5 * ((BR_OOP - EV_OOP) + (BR_IP - EV_IP))`).
    pub fn exploitability_f64(&self) -> f64 {
        let [br_oop, br_ip] = self.best_response_values_f64();
        if self.rake_fraction > 0.0 {
            let [ev_oop, ev_ip] = self.root_expected_values_f64();
            (0.5 * ((br_oop - ev_oop) + (br_ip - ev_ip))).max(0.0)
        } else {
            let pot = self.tree_config.starting_pot as f64;
            (0.5 * ((br_oop + br_ip) - pot)).max(0.0)
        }
    }

    /// Computes expected game values `[EV_OOP, EV_IP]` at the root under the current average strategy
    /// in full `f64` double precision (`EV_OOP + EV_IP == starting_pot` to `~1e-13`).
    pub fn root_expected_values_f64(&self) -> [f64; 2] {
        let mut centered = [0.0f64; 2];
        with_tls_ie_scratch(|scratch| {
            for player in [PLAYER_OOP, PLAYER_IP] {
                let opp = player ^ 1;
                let init_opp_f64: Vec<f64> = self.initial_weights[opp]
                    .iter()
                    .map(|&w| w as f64)
                    .collect();
                let cfv = self.strategy_cfv_traverse_f64(0, player, &init_opp_f64, scratch);
                centered[player] = cfv
                    .iter()
                    .zip(&self.initial_weights[player])
                    .map(|(&v, &w)| v * (w as f64))
                    .sum();
            }
        });
        let half_start = 0.5 * (self.tree_config.starting_pot as f64);
        [half_start + centered[0], half_start + centered[1]]
    }

    /// Computes expected game values `[EV_OOP, EV_IP]` at the root under the current average strategy,
    /// expressed as net postflop chip EV (`EV_OOP + EV_IP == starting_pot`).
    pub fn root_expected_values(&self) -> [f32; 2] {
        let [ev0, ev1] = self.root_expected_values_f64();
        [ev0 as f32, ev1 as f32]
    }

    // ========================================================================
    // Tree Cursor & Node Inspection API (analogous to wasm-postflop)
    // ========================================================================

    pub fn back_to_root(&mut self) {
        self.cursor_node = 0;
        self.cursor_suit_perm = [0, 1, 2, 3];
        self.cursor_combo_map = [
            (0..self.combos[0].len()).collect(),
            (0..self.combos[1].len()).collect(),
        ];
        self.history.clear();
    }

    pub fn history(&self) -> &[HistoryStep] {
        &self.history
    }

    pub fn play_action(&mut self, action_idx: usize) -> Result<(), SolverError> {
        let next = match &self.nodes[self.cursor_node].kind {
            ConcreteNodeKind::Player { children, .. } => {
                *children.get(action_idx).ok_or_else(|| {
                    SolverError::InvalidHistoryStep(format!(
                        "action index {action_idx} out of bounds (len {})",
                        children.len()
                    ))
                })?
            }
            _ => {
                return Err(SolverError::InvalidHistoryStep(
                    "current node is not a player action node".to_string(),
                ))
            }
        };
        self.cursor_node = next;
        self.history.push(HistoryStep::Action(action_idx));
        Ok(())
    }

    /// Advances a chance node by dealing `card` (supports ANY remaining deck card, transparently
    /// applying suit-isomorphism remapping when `card` belongs to a folded isomorphic duplicate branch).
    pub fn play_chance_card(&mut self, card: Card) -> Result<(), SolverError> {
        let mapped_card = permute_card(card, self.cursor_suit_perm);
        let (next, extra_swap) = match &self.nodes[self.cursor_node].kind {
            ConcreteNodeKind::Chance {
                rep_cards,
                rep_children,
                iso_duplicates,
                ..
            } => {
                if let Some(pos) = rep_cards.iter().position(|&c| c == mapped_card) {
                    (rep_children[pos], None)
                } else if let Some(dup) = iso_duplicates.iter().find(|d| d.card == mapped_card) {
                    let rep_card = rep_cards[dup.rep_branch_idx];
                    (
                        rep_children[dup.rep_branch_idx],
                        Some((rep_card.suit(), mapped_card.suit())),
                    )
                } else {
                    return Err(SolverError::InvalidHistoryStep(format!(
                        "card {card} is already on the board or invalid at this chance node"
                    )));
                }
            }
            _ => {
                return Err(SolverError::InvalidHistoryStep(
                    "current node is not a chance node".to_string(),
                ))
            }
        };

        if let Some((sa, sb)) = extra_swap {
            let trans = suit_transposition(sa, sb);
            let mut next_perm = [0u8; 4];
            for s in 0..4 {
                next_perm[s] = trans[self.cursor_suit_perm[s] as usize];
            }
            self.cursor_suit_perm = next_perm;
            for p in 0..2 {
                let cids: Vec<ComboId> = self.combos[p].iter().map(|c| c.combo_id).collect();
                for (i, c) in self.combos[p].iter().enumerate() {
                    let mh = permute_hand4(c.hand, self.cursor_suit_perm).combo_id();
                    if let Ok(pos) = cids.binary_search(&mh) {
                        self.cursor_combo_map[p][i] = pos;
                    }
                }
            }
        }

        self.cursor_node = next;
        self.history.push(HistoryStep::Deal(card));
        Ok(())
    }

    /// Returns representative chance cards available at the current cursor node (empty if not a chance node).
    pub fn available_chance_cards(&self) -> &[Card] {
        match &self.nodes[self.cursor_node].kind {
            ConcreteNodeKind::Chance { rep_cards, .. } => rep_cards,
            _ => &[],
        }
    }

    /// Returns ALL legal real-world deck cards dealable at the current chance node (including folded isomorphic cards).
    pub fn all_legal_chance_cards(&self) -> Vec<Card> {
        match &self.nodes[self.cursor_node].kind {
            ConcreteNodeKind::Chance { .. } => {
                let mut inv_perm = [0u8; 4];
                for s in 0..4 {
                    inv_perm[self.cursor_suit_perm[s] as usize] = s as u8;
                }
                let real_board = permute_board(&self.nodes[self.cursor_node].board, inv_perm);
                let bmask = real_board.mask();
                (0..52u8)
                    .filter_map(|id| Card::new(id).ok())
                    .filter(|&c| !bmask.contains(c))
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    /// Computes the full [`NodeReport`] (average strategy + per-combo EVs for OOP and IP)
    /// at the current cursor node (in real-world card/combo coordinates even after isomorphic deals)!
    pub fn inspect_current_node(&self) -> NodeReport {
        // Propagate both players' average-strategy reach probabilities along the canonical path
        let mut reach = [
            self.initial_weights[0].clone(),
            self.initial_weights[1].clone(),
        ];
        let mut nid = 0usize;
        let mut cur_perm = [0u8, 1, 2, 3];
        for &step in &self.history {
            match (&self.nodes[nid].kind, step) {
                (
                    ConcreteNodeKind::Player {
                        player,
                        actions,
                        children,
                        num_buckets,
                        combo_to_bucket,
                        storage,
                        locked_strategy,
                    },
                    HistoryStep::Action(a_idx),
                ) => {
                    let p = *player;
                    let n_combos = self.combos[p].len();
                    let st = storage.as_ref();
                    let avg_b_strat = average_or_current_bucket_strategy(
                        locked_strategy.as_deref(),
                        &st.cum_strategy,
                        &st.regrets,
                        actions.len(),
                        *num_buckets,
                    );
                    let row = &avg_b_strat[a_idx * (*num_buckets)..(a_idx + 1) * (*num_buckets)];
                    for i in 0..n_combos {
                        let b = bucket_of(combo_to_bucket, i);
                        reach[p][i] *= row[b];
                    }
                    nid = children[a_idx];
                }
                (
                    ConcreteNodeKind::Chance {
                        rep_cards,
                        rep_children,
                        iso_duplicates,
                        ..
                    },
                    HistoryStep::Deal(real_card),
                ) => {
                    let mapped = permute_card(real_card, cur_perm);
                    let (rep_pos, swap_opt) =
                        if let Some(pos) = rep_cards.iter().position(|&c| c == mapped) {
                            (pos, None)
                        } else {
                            let dup = iso_duplicates
                                .iter()
                                .find(|d| d.card == mapped)
                                .expect("validated in play_chance_card");
                            (
                                dup.rep_branch_idx,
                                Some((
                                    rep_cards[dup.rep_branch_idx].suit(),
                                    mapped.suit(),
                                    &dup.swaps,
                                )),
                            )
                        };
                    if let Some((sa, sb, swaps)) = swap_opt {
                        let trans = suit_transposition(sa, sb);
                        let mut next_perm = [0u8; 4];
                        for s in 0..4 {
                            next_perm[s] = trans[cur_perm[s] as usize];
                        }
                        cur_perm = next_perm;
                        for p in 0..2 {
                            for &(a, b) in &swaps[p] {
                                reach[p].swap(a as usize, b as usize);
                            }
                        }
                    }
                    let rep_card = rep_cards[rep_pos];
                    let bmask = rep_card.mask();
                    for (p, r_vec) in reach.iter_mut().enumerate() {
                        for (i, c) in self.combos[p].iter().enumerate() {
                            if c.mask.intersects(bmask) {
                                r_vec[i] = 0.0;
                            }
                        }
                    }
                    nid = rep_children[rep_pos];
                }
                _ => unreachable!(),
            }
        }
        debug_assert_eq!(nid, self.cursor_node);
        debug_assert_eq!(cur_perm, self.cursor_suit_perm);

        let node = &self.nodes[nid];
        let bmask = node.board.mask();
        for (p, r_vec) in reach.iter_mut().enumerate() {
            for (i, c) in self.combos[p].iter().enumerate() {
                if c.mask.intersects(bmask) {
                    r_vec[i] = 0.0;
                }
            }
        }

        let mut scratch = InclusionExclusionScratch::new();
        let cfv_oop = self.strategy_cfv_traverse(nid, PLAYER_OOP, &reach[1], &mut scratch);
        let cfv_ip = self.strategy_cfv_traverse(nid, PLAYER_IP, &reach[0], &mut scratch);

        let canon_ev_oop =
            self.conditional_chip_evs(&cfv_oop, PLAYER_OOP, &reach[1], bmask, &mut scratch);
        let canon_ev_ip =
            self.conditional_chip_evs(&cfv_ip, PLAYER_IP, &reach[0], bmask, &mut scratch);

        // Map canonical EVs and strategy back to real-world combo indices via `cursor_combo_map`!
        let ev_oop: Vec<f32> = self.cursor_combo_map[0]
            .iter()
            .map(|&ci| canon_ev_oop[ci])
            .collect();
        let ev_ip: Vec<f32> = self.cursor_combo_map[1]
            .iter()
            .map(|&ci| canon_ev_ip[ci])
            .collect();

        let mut inv_perm = [0u8; 4];
        for s in 0..4 {
            inv_perm[self.cursor_suit_perm[s] as usize] = s as u8;
        }
        let real_board = permute_board(&node.board, inv_perm);

        let (node_type, actions, strategy) = match &node.kind {
            ConcreteNodeKind::Player {
                player,
                actions,
                num_buckets,
                combo_to_bucket,
                storage,
                locked_strategy,
                ..
            } => {
                let p = *player;
                let n_combos = self.combos[p].len();
                let st = storage.as_ref();
                let b_strat = average_or_current_bucket_strategy(
                    locked_strategy.as_deref(),
                    &st.cum_strategy,
                    &st.regrets,
                    actions.len(),
                    *num_buckets,
                );
                let mut full_strat = vec![0.0f32; actions.len() * n_combos];
                for a in 0..actions.len() {
                    for i in 0..n_combos {
                        let canon_i = self.cursor_combo_map[p][i];
                        let b = bucket_of(combo_to_bucket, canon_i);
                        full_strat[a * n_combos + i] = b_strat[a * (*num_buckets) + b];
                    }
                }
                (
                    if p == PLAYER_OOP { "oop" } else { "ip" },
                    actions.clone(),
                    full_strat,
                )
            }
            ConcreteNodeKind::Chance { .. } => ("chance", Vec::new(), Vec::new()),
            ConcreteNodeKind::TerminalFold { .. } | ConcreteNodeKind::TerminalShowdown { .. } => {
                ("terminal", Vec::new(), Vec::new())
            }
        };

        NodeReport {
            board: real_board,
            pot: node.pot,
            committed: node.committed,
            node_type,
            actions,
            strategy,
            ev_oop,
            ev_ip,
        }
    }

    /// Returns `[reach_oop, reach_ip]` per combo (in real-world combo coordinates) along the current
    /// `history` cursor path, zeroing any combo blocked by dealt board cards.
    pub fn current_combo_reaches(&self) -> [Vec<f32>; 2] {
        let mut reach = [
            self.initial_weights[0].clone(),
            self.initial_weights[1].clone(),
        ];
        let mut nid = 0usize;
        let mut cur_perm = [0u8, 1, 2, 3];
        for &step in &self.history {
            match (&self.nodes[nid].kind, step) {
                (
                    ConcreteNodeKind::Player {
                        player,
                        actions,
                        children,
                        num_buckets,
                        combo_to_bucket,
                        storage,
                        locked_strategy,
                    },
                    HistoryStep::Action(a_idx),
                ) => {
                    let p = *player;
                    let n_combos = self.combos[p].len();
                    let st = storage.as_ref();
                    let avg_b_strat = average_or_current_bucket_strategy(
                        locked_strategy.as_deref(),
                        &st.cum_strategy,
                        &st.regrets,
                        actions.len(),
                        *num_buckets,
                    );
                    let row = &avg_b_strat[a_idx * (*num_buckets)..(a_idx + 1) * (*num_buckets)];
                    for i in 0..n_combos {
                        let b = bucket_of(combo_to_bucket, i);
                        reach[p][i] *= row[b];
                    }
                    nid = children[a_idx];
                }
                (
                    ConcreteNodeKind::Chance {
                        rep_cards,
                        rep_children,
                        iso_duplicates,
                        ..
                    },
                    HistoryStep::Deal(real_card),
                ) => {
                    let mapped = permute_card(real_card, cur_perm);
                    let (rep_pos, swap_opt) =
                        if let Some(pos) = rep_cards.iter().position(|&c| c == mapped) {
                            (pos, None)
                        } else {
                            let dup = iso_duplicates
                                .iter()
                                .find(|d| d.card == mapped)
                                .expect("validated in play_chance_card");
                            (
                                dup.rep_branch_idx,
                                Some((
                                    rep_cards[dup.rep_branch_idx].suit(),
                                    mapped.suit(),
                                    &dup.swaps,
                                )),
                            )
                        };
                    if let Some((sa, sb, swaps)) = swap_opt {
                        let trans = suit_transposition(sa, sb);
                        let mut next_perm = [0u8; 4];
                        for s in 0..4 {
                            next_perm[s] = trans[cur_perm[s] as usize];
                        }
                        cur_perm = next_perm;
                        for p in 0..2 {
                            for &(a, b) in &swaps[p] {
                                reach[p].swap(a as usize, b as usize);
                            }
                        }
                    }
                    let rep_card = rep_cards[rep_pos];
                    let bmask = rep_card.mask();
                    for (p, r_vec) in reach.iter_mut().enumerate() {
                        for (i, c) in self.combos[p].iter().enumerate() {
                            if c.mask.intersects(bmask) {
                                r_vec[i] = 0.0;
                            }
                        }
                    }
                    nid = rep_children[rep_pos];
                }
                _ => unreachable!(),
            }
        }
        let bmask = self.nodes[nid].board.mask();
        for (p, r_vec) in reach.iter_mut().enumerate() {
            for (i, c) in self.combos[p].iter().enumerate() {
                if c.mask.intersects(bmask) {
                    r_vec[i] = 0.0;
                }
            }
        }
        let real_oop: Vec<f32> = self.cursor_combo_map[0]
            .iter()
            .map(|&ci| reach[0][ci])
            .collect();
        let real_ip: Vec<f32> = self.cursor_combo_map[1]
            .iter()
            .map(|&ci| reach[1][ci])
            .collect();
        [real_oop, real_ip]
    }

    /// If the current cursor node is a `Player` decision node, returns `(acting_player, num_buckets, combo_to_bucket)`
    /// in real-world combo coordinates.
    pub fn current_node_bucket_assignment(&self) -> Option<(usize, usize, Vec<u32>)> {
        match &self.nodes[self.cursor_node].kind {
            ConcreteNodeKind::Player {
                player,
                num_buckets,
                combo_to_bucket,
                ..
            } => {
                let p = *player;
                let n_combos = self.combos[p].len();
                let mapped: Vec<u32> = (0..n_combos)
                    .map(|i| {
                        let canon_i = self.cursor_combo_map[p][i];
                        bucket_of(combo_to_bucket, canon_i) as u32
                    })
                    .collect();
                Some((p, *num_buckets, mapped))
            }
            _ => None,
        }
    }

    /// Returns whether the currently inspected cursor node has an active node-locked strategy.
    pub fn is_current_node_locked(&self) -> bool {
        matches!(
            &self.nodes[self.cursor_node].kind,
            ConcreteNodeKind::Player {
                locked_strategy: Some(_),
                ..
            }
        )
    }

    /// Returns the total number of locked decision nodes in the concrete tree.
    pub fn locked_node_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| {
                matches!(
                    &n.kind,
                    ConcreteNodeKind::Player {
                        locked_strategy: Some(_),
                        ..
                    }
                )
            })
            .count()
    }

    /// Locks the currently inspected `Player` decision node to uniform action frequencies `action_probs`
    /// (`len == actions.len()`, normalized to sum to `1.0`) across all buckets at this node, and resets
    /// unlocked nodes' regrets & cumulative strategies (`iterations_run = 0`) so subsequent DCFR steps
    /// converge cleanly to the exact counter-exploitative response!
    pub fn lock_current_node_uniform(&mut self, action_probs: &[f32]) -> Result<(), SolverError> {
        let nid = self.cursor_node;
        let new_lock = match &self.nodes[nid].kind {
            ConcreteNodeKind::Player {
                actions,
                num_buckets,
                ..
            } => {
                let n_act = actions.len();
                if action_probs.len() != n_act {
                    return Err(SolverError::InvalidHistoryStep(format!(
                        "expected {n_act} action probabilities for node lock, got {}",
                        action_probs.len()
                    )));
                }
                let mut clean: Vec<f64> = action_probs
                    .iter()
                    .map(|&p| if p.is_finite() && p > 0.0 { p as f64 } else { 0.0 })
                    .collect();
                let sum: f64 = clean.iter().sum();
                if sum <= 1e-12 {
                    return Err(SolverError::InvalidHistoryStep(
                        "locked action frequencies must sum to > 0".to_string(),
                    ));
                }
                for p in &mut clean {
                    *p /= sum;
                }
                let mut ls = vec![0.0f32; n_act * (*num_buckets)];
                for a in 0..n_act {
                    let v = clean[a] as f32;
                    ls[a * (*num_buckets)..(a + 1) * (*num_buckets)].fill(v);
                }
                ls
            }
            _ => {
                return Err(SolverError::InvalidHistoryStep(
                    "current node is not a player decision node".to_string(),
                ))
            }
        };

        if let ConcreteNodeKind::Player {
            locked_strategy, ..
        } = &mut self.nodes[nid].kind
        {
            *locked_strategy = Some(new_lock);
        }
        self.reset_unlocked_regrets_and_cum_strategies();
        Ok(())
    }

    /// Locks the currently inspected `Player` decision node using a custom per-combo action probability
    /// callback `combo_probs_fn(real_combo_idx, hand) -> Option<Vec<f32>>`!
    /// - Requires `IdentityAbstraction` (`num_buckets == combos[player].len()`), or aggregatescombo weights
    ///   per bucket if bucketed (`combo_to_bucket` averages `combo_probs_fn` weighted by current reach, falling back
    ///   to the node's current average strategy for any combo where `combo_probs_fn` returns `None`).
    /// - Resets unlocked nodes' regrets & cumulative strategies (`iterations_run = 0`) so subsequent DCFR steps
    ///   converge cleanly to the exact counter-exploitative response!
    pub fn lock_current_node_by_combo<F>(&mut self, mut combo_probs_fn: F) -> Result<(), SolverError>
    where
        F: FnMut(usize, plo_cards::Hand4) -> Option<Vec<f32>>,
    {
        let nid = self.cursor_node;
        let reaches = self.current_combo_reaches();
        let new_lock = match &self.nodes[nid].kind {
            ConcreteNodeKind::Player {
                player,
                actions,
                num_buckets,
                combo_to_bucket,
                storage,
                locked_strategy,
                ..
            } => {
                let p = *player;
                let n_act = actions.len();
                let n_combos = self.combos[p].len();
                let nb = *num_buckets;
                let st = storage.as_ref();
                let base_b_strat = average_or_current_bucket_strategy(
                    locked_strategy.as_deref(),
                    &st.cum_strategy,
                    &st.regrets,
                    n_act,
                    nb,
                );

                // Accumulate target probabilities per canonical bucket `b` from real-world combos `i`
                let mut b_acc = vec![0.0f64; n_act * nb];
                let mut b_weight = vec![0.0f64; nb];
                let mut any_locked = false;

                for i in 0..n_combos {
                    let hand = self.combos[p][i].hand;
                    let canon_i = self.cursor_combo_map[p][i];
                    let b = bucket_of(combo_to_bucket, canon_i);
                    let w = (reaches[p][i] as f64).max(1e-9);
                    if let Some(raw_probs) = combo_probs_fn(i, hand) {
                        if raw_probs.len() != n_act {
                            return Err(SolverError::InvalidHistoryStep(format!(
                                "expected {n_act} action probabilities for combo lock, got {}",
                                raw_probs.len()
                            )));
                        }
                        let mut clean: Vec<f64> = raw_probs
                            .iter()
                            .map(|&v| if v.is_finite() && v > 0.0 { v as f64 } else { 0.0 })
                            .collect();
                        let sum: f64 = clean.iter().sum();
                        if sum <= 1e-12 {
                            return Err(SolverError::InvalidHistoryStep(
                                "locked action frequencies for a class/combo must sum to > 0"
                                    .to_string(),
                            ));
                        }
                        for v in &mut clean {
                            *v /= sum;
                        }
                        for a in 0..n_act {
                            b_acc[a * nb + b] += w * clean[a];
                        }
                        b_weight[b] += w;
                        any_locked = true;
                    } else {
                        for a in 0..n_act {
                            b_acc[a * nb + b] += w * (base_b_strat[a * nb + b] as f64);
                        }
                        b_weight[b] += w;
                    }
                }

                if !any_locked {
                    return Err(SolverError::InvalidHistoryStep(
                        "no combos matched the requested class/combo lock rules".to_string(),
                    ));
                }

                let mut ls = vec![0.0f32; n_act * nb];
                for b in 0..nb {
                    if b_weight[b] > 0.0 {
                        let mut col_sum = 0.0f64;
                        for a in 0..n_act {
                            let val = b_acc[a * nb + b] / b_weight[b];
                            ls[a * nb + b] = val as f32;
                            col_sum += val;
                        }
                        if col_sum > 1e-12 {
                            for a in 0..n_act {
                                ls[a * nb + b] = (ls[a * nb + b] as f64 / col_sum) as f32;
                            }
                        }
                    } else {
                        for a in 0..n_act {
                            ls[a * nb + b] = base_b_strat[a * nb + b];
                        }
                    }
                }
                ls
            }
            _ => {
                return Err(SolverError::InvalidHistoryStep(
                    "current node is not a player decision node".to_string(),
                ))
            }
        };

        if let ConcreteNodeKind::Player {
            locked_strategy, ..
        } = &mut self.nodes[nid].kind
        {
            *locked_strategy = Some(new_lock);
        }
        self.reset_unlocked_regrets_and_cum_strategies();
        Ok(())
    }

    /// Returns a compact view of the immediate street tree starting from the current street's root
    /// (`(street_root_history_depth, nodes)`) for `#7` Interactive Street Tree Minimap!
    #[allow(clippy::type_complexity)]
    pub fn current_street_tree_minimap(
        &self,
    ) -> (
        usize,
        Vec<(
            usize,
            Option<usize>,
            Vec<usize>,
            String,
            String,
            i32,
            bool,
            bool,
        )>,
    ) {
        // Determine the root `nid` and `history` depth of the current street along `self.history`
        let mut street_root_nid = 0usize;
        let mut street_root_depth = 0usize;
        let mut nid = 0usize;
        let mut cur_perm = [0u8, 1, 2, 3];
        let mut hist_prefix_depths: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        hist_prefix_depths.insert(0, 0);

        for (step_idx, &step) in self.history.iter().enumerate() {
            match (&self.nodes[nid].kind, step) {
                (ConcreteNodeKind::Player { children, .. }, HistoryStep::Action(a_idx)) => {
                    nid = children[a_idx];
                    hist_prefix_depths.insert(nid, step_idx + 1);
                }
                (
                    ConcreteNodeKind::Chance {
                        rep_cards,
                        rep_children,
                        iso_duplicates,
                        ..
                    },
                    HistoryStep::Deal(real_card),
                ) => {
                    let mapped = permute_card(real_card, cur_perm);
                    let (rep_pos, swap_opt) =
                        if let Some(pos) = rep_cards.iter().position(|&c| c == mapped) {
                            (pos, None)
                        } else {
                            let dup = iso_duplicates
                                .iter()
                                .find(|d| d.card == mapped)
                                .expect("validated in play_chance_card");
                            (
                                dup.rep_branch_idx,
                                Some((rep_cards[dup.rep_branch_idx].suit(), mapped.suit())),
                            )
                        };
                    if let Some((sa, sb)) = swap_opt {
                        let trans = suit_transposition(sa, sb);
                        let mut next_perm = [0u8; 4];
                        for s in 0..4 {
                            next_perm[s] = trans[cur_perm[s] as usize];
                        }
                        cur_perm = next_perm;
                    }
                    nid = rep_children[rep_pos];
                    street_root_nid = nid;
                    street_root_depth = step_idx + 1;
                    hist_prefix_depths.insert(nid, step_idx + 1);
                }
                _ => break,
            }
        }

        // Collect up to 64 decision/chance/terminal nodes within the current street (`stop` recursion at `Chance`)
        let mut out = Vec::new();
        let mut stack: Vec<(usize, Option<usize>, Vec<usize>, String)> =
            vec![(street_root_nid, None, Vec::new(), "Root".to_string())];
        while let Some((cur, parent_opt, path_indices, edge_label)) = stack.pop() {
            if out.len() >= 64 {
                break;
            }
            let n = &self.nodes[cur];
            let is_on_cursor_path = hist_prefix_depths.contains_key(&cur);
            let ntype = match &n.kind {
                ConcreteNodeKind::Player {
                    player,
                    actions,
                    children,
                    ..
                } => {
                    for (a_idx, (&ch, act)) in children.iter().zip(actions.iter()).enumerate().rev()
                    {
                        let mut next_path = path_indices.clone();
                        next_path.push(a_idx);
                        stack.push((ch, Some(cur), next_path, act.to_string()));
                    }
                    if *player == PLAYER_OOP {
                        "oop"
                    } else {
                        "ip"
                    }
                }
                ConcreteNodeKind::Chance { .. } => "chance",
                ConcreteNodeKind::TerminalFold { .. } => "fold",
                ConcreteNodeKind::TerminalShowdown { .. } => "showdown",
            };
            out.push((
                cur,
                parent_opt,
                path_indices,
                edge_label,
                ntype.to_string(),
                n.pot,
                cur == self.cursor_node,
                is_on_cursor_path,
            ));
        }
        (street_root_depth, out)
    }

    /// Unlocks the currently inspected `Player` decision node (if locked) and resets regrets/cumulative
    /// strategies so subsequent DCFR iterations re-solve equilibrium cleanly.
    pub fn unlock_current_node(&mut self) -> Result<(), SolverError> {
        let nid = self.cursor_node;
        match &mut self.nodes[nid].kind {
            ConcreteNodeKind::Player {
                locked_strategy, ..
            } => {
                *locked_strategy = None;
            }
            _ => {
                return Err(SolverError::InvalidHistoryStep(
                    "current node is not a player decision node".to_string(),
                ))
            }
        }
        self.reset_unlocked_regrets_and_cum_strategies();
        Ok(())
    }

    /// Deterministic 64-bit FNV-1a structural & range fingerprint of this [`PloGame`] instance,
    /// used by `PLOSNAP2` binary snapshots to prevent loading weights from a mismatched spot!
    fn structural_config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        mix(&self.root_board.mask().bits().to_le_bytes());
        mix(&self.tree_config.starting_pot.to_le_bytes());
        mix(&self.tree_config.effective_stack.to_le_bytes());
        mix(&self.rake_fraction.to_bits().to_le_bytes());
        mix(&self.rake_cap_chips.to_bits().to_le_bytes());
        mix(&[self.enable_chance_isomorphism as u8]);
        for p in 0..2 {
            mix(&(self.combos[p].len() as u32).to_le_bytes());
            for c in &self.combos[p] {
                mix(&c.combo_id.raw().to_le_bytes());
                mix(&c.weight.to_bits().to_le_bytes());
            }
        }
        mix(&(self.nodes.len() as u32).to_le_bytes());
        for node in &self.nodes {
            mix(&node.board.mask().bits().to_le_bytes());
            mix(&node.committed[0].to_le_bytes());
            mix(&node.committed[1].to_le_bytes());
            match &node.kind {
                ConcreteNodeKind::Player {
                    player,
                    actions,
                    num_buckets,
                    ..
                } => {
                    mix(&[1u8, *player as u8]);
                    mix(&(*num_buckets as u32).to_le_bytes());
                    mix(&(actions.len() as u32).to_le_bytes());
                }
                ConcreteNodeKind::Chance { rep_cards, .. } => {
                    mix(&[2u8]);
                    mix(&(rep_cards.len() as u32).to_le_bytes());
                }
                ConcreteNodeKind::TerminalFold { folded_player, .. } => {
                    mix(&[3u8, *folded_player as u8]);
                }
                ConcreteNodeKind::TerminalShowdown { .. } => {
                    mix(&[4u8]);
                }
            }
        }
        h
    }

    /// Exports all learned solver weights (`iterations_run` plus every `Player` node's `regrets`, `cum_strategy`,
    /// and optional `locked_strategy`) as a compact binary byte vector (`PLOSNAP2` magic header + 64-bit structural
    /// config hash) so a solved tree can be saved to disk and restored in milliseconds without re-running DCFR!
    pub fn export_weights_binary(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"PLOSNAP2");
        out.extend_from_slice(&self.structural_config_hash().to_le_bytes());
        out.extend_from_slice(&self.iterations_run.to_le_bytes());
        let n_nodes = self.nodes.len() as u32;
        out.extend_from_slice(&n_nodes.to_le_bytes());

        for node in &self.nodes {
            match &node.kind {
                ConcreteNodeKind::Player {
                    storage,
                    locked_strategy,
                    ..
                } => {
                    out.push(1u8); // Player node marker
                    let st = storage.as_ref();
                    let len = st.regrets.len() as u32;
                    out.extend_from_slice(&len.to_le_bytes());
                    for &v in &st.regrets {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    for &v in &st.cum_strategy {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    if let Some(ls) = locked_strategy {
                        out.push(1u8);
                        let llen = ls.len() as u32;
                        out.extend_from_slice(&llen.to_le_bytes());
                        for &v in ls {
                            out.extend_from_slice(&v.to_le_bytes());
                        }
                    } else {
                        out.push(0u8);
                    }
                }
                _ => {
                    out.push(0u8); // Non-player node marker
                }
            }
        }
        out
    }

    /// Restores learned solver weights previously exported via [`PloGame::export_weights_binary`] into an
    /// identically structured `PloGame` (`PLOSNAP2` format; explicitly rejects legacy `PLOSNAP1`, mismatched
    /// structural hashes, and any `NaN`/`Inf` floats without mutating `self` on error)!
    pub fn import_weights_binary(&mut self, bytes: &[u8]) -> Result<(), SolverError> {
        if bytes.len() >= 8 && &bytes[0..8] == b"PLOSNAP1" {
            return Err(SolverError::InvalidHistoryStep(
                "Legacy PLOSNAP1 snapshot is incompatible with v0.1.1+ (expected PLOSNAP2 with config hash); please re-save the spot".to_string(),
            ));
        }
        if bytes.len() < 24 || &bytes[0..8] != b"PLOSNAP2" {
            return Err(SolverError::InvalidHistoryStep(
                "Invalid .plotree binary header (expected PLOSNAP2)".to_string(),
            ));
        }
        let mut pos = 8usize;
        let read_u32 = |buf: &[u8], p: &mut usize| -> Result<u32, SolverError> {
            if *p + 4 > buf.len() {
                return Err(SolverError::InvalidHistoryStep(
                    "Truncated .plotree binary snapshot".to_string(),
                ));
            }
            let v = u32::from_le_bytes([buf[*p], buf[*p + 1], buf[*p + 2], buf[*p + 3]]);
            *p += 4;
            Ok(v)
        };
        let read_u64 = |buf: &[u8], p: &mut usize| -> Result<u64, SolverError> {
            if *p + 8 > buf.len() {
                return Err(SolverError::InvalidHistoryStep(
                    "Truncated .plotree config hash".to_string(),
                ));
            }
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&buf[*p..*p + 8]);
            *p += 8;
            Ok(u64::from_le_bytes(arr))
        };
        let read_finite_f32 = |buf: &[u8], p: &mut usize| -> Result<f32, SolverError> {
            if *p + 4 > buf.len() {
                return Err(SolverError::InvalidHistoryStep(
                    "Truncated .plotree f32 array".to_string(),
                ));
            }
            let v = f32::from_le_bytes([buf[*p], buf[*p + 1], buf[*p + 2], buf[*p + 3]]);
            *p += 4;
            if !v.is_finite() {
                return Err(SolverError::InvalidHistoryStep(
                    "Corrupted .plotree snapshot: non-finite (NaN/Inf) weight encountered"
                        .to_string(),
                ));
            }
            Ok(v)
        };

        let snap_hash = read_u64(bytes, &mut pos)?;
        let expected_hash = self.structural_config_hash();
        if snap_hash != expected_hash {
            return Err(SolverError::InvalidHistoryStep(format!(
                "Snapshot configuration hash mismatch ({snap_hash:#018x} != {expected_hash:#018x}): snapshot belongs to a different board, range, or tree"
            )));
        }

        let iters = read_u32(bytes, &mut pos)?;
        let n_nodes = read_u32(bytes, &mut pos)? as usize;
        if n_nodes != self.nodes.len() {
            return Err(SolverError::InvalidHistoryStep(format!(
                "Tree node count mismatch: snapshot has {n_nodes} nodes, rebuilt game has {}",
                self.nodes.len()
            )));
        }

        // Parse & validate all buffers into staging storage first so `self` is never half-mutated on error
        struct StagedPlayerWeights {
            regrets: Vec<f32>,
            cum_strategy: Vec<f32>,
            locked_strategy: Option<Vec<f32>>,
        }
        let mut staged: Vec<Option<StagedPlayerWeights>> = Vec::with_capacity(n_nodes);

        for node in &self.nodes {
            if pos >= bytes.len() {
                return Err(SolverError::InvalidHistoryStep(
                    "Truncated .plotree node stream".to_string(),
                ));
            }
            let marker = bytes[pos];
            pos += 1;
            match (&node.kind, marker) {
                (
                    ConcreteNodeKind::Player {
                        actions,
                        num_buckets,
                        storage,
                        ..
                    },
                    1u8,
                ) => {
                    let len = read_u32(bytes, &mut pos)? as usize;
                    let st = storage.as_ref();
                    if st.regrets.len() != len {
                        return Err(SolverError::InvalidHistoryStep(format!(
                            "Player storage length mismatch: snapshot has {len}, game has {}",
                            st.regrets.len()
                        )));
                    }
                    let mut regs = Vec::with_capacity(len);
                    for _ in 0..len {
                        regs.push(read_finite_f32(bytes, &mut pos)?);
                    }
                    let mut cums = Vec::with_capacity(len);
                    for _ in 0..len {
                        let c = read_finite_f32(bytes, &mut pos)?;
                        if c < 0.0 {
                            return Err(SolverError::InvalidHistoryStep(
                                "Corrupted .plotree snapshot: negative cumulative strategy weight"
                                    .to_string(),
                            ));
                        }
                        cums.push(c);
                    }
                    if pos >= bytes.len() {
                        return Err(SolverError::InvalidHistoryStep(
                            "Truncated .plotree lock flag".to_string(),
                        ));
                    }
                    let has_lock = bytes[pos];
                    pos += 1;
                    let lock_opt = if has_lock == 1u8 {
                        let llen = read_u32(bytes, &mut pos)? as usize;
                        let expected_llen = actions.len() * (*num_buckets);
                        if llen != expected_llen {
                            return Err(SolverError::InvalidHistoryStep(format!(
                                "Locked strategy length mismatch: snapshot has {llen}, expected {expected_llen}"
                            )));
                        }
                        let mut ls = Vec::with_capacity(llen);
                        for _ in 0..llen {
                            let p = read_finite_f32(bytes, &mut pos)?;
                            if !(0.0..=1.0001).contains(&p) {
                                return Err(SolverError::InvalidHistoryStep(
                                    "Corrupted .plotree snapshot: locked probability outside [0, 1]"
                                        .to_string(),
                                ));
                            }
                            ls.push(p);
                        }
                        Some(ls)
                    } else if has_lock == 0u8 {
                        None
                    } else {
                        return Err(SolverError::InvalidHistoryStep(
                            "Invalid lock flag in .plotree snapshot".to_string(),
                        ));
                    };
                    staged.push(Some(StagedPlayerWeights {
                        regrets: regs,
                        cum_strategy: cums,
                        locked_strategy: lock_opt,
                    }));
                }
                (ConcreteNodeKind::Player { .. }, _) => {
                    return Err(SolverError::InvalidHistoryStep(
                        "Expected Player node marker in .plotree snapshot".to_string(),
                    ));
                }
                (_, 0u8) => {
                    staged.push(None);
                }
                _ => {
                    return Err(SolverError::InvalidHistoryStep(
                        "Unexpected Player marker on non-player node in .plotree snapshot"
                            .to_string(),
                    ));
                }
            }
        }

        // Commit validated staging buffers atomically
        for (node, entry) in self.nodes.iter_mut().zip(staged.into_iter()) {
            if let (
                ConcreteNodeKind::Player {
                    storage,
                    locked_strategy,
                    ..
                },
                Some(sw),
            ) = (&mut node.kind, entry)
            {
                let st = storage.as_mut();
                st.regrets = sw.regrets;
                st.cum_strategy = sw.cum_strategy;
                *locked_strategy = sw.locked_strategy;
            }
        }
        self.iterations_run = iters;
        self.back_to_root();
        Ok(())
    }

    /// Unlocks all locked decision nodes across the entire tree and resets regrets/cumulative strategies.
    pub fn unlock_all_nodes(&mut self) {
        for n in &mut self.nodes {
            if let ConcreteNodeKind::Player {
                locked_strategy, ..
            } = &mut n.kind
            {
                *locked_strategy = None;
            }
        }
        self.reset_unlocked_regrets_and_cum_strategies();
    }

    /// Evaluates every valid remaining Turn/River card at the current `Chance` node without cloning
    /// the solver arena, returning per-card EVs and next-node action frequencies for the `4×13` Runout EV Heatmap (`#3`).
    pub fn inspect_chance_runouts(&mut self) -> Vec<ChanceRunoutSummary> {
        if !matches!(
            &self.nodes[self.cursor_node].kind,
            ConcreteNodeKind::Chance { .. }
        ) {
            return Vec::new();
        }

        let saved_node = self.cursor_node;
        let saved_perm = self.cursor_suit_perm;
        let saved_map = self.cursor_combo_map.clone();
        let saved_hist_len = self.history.len();

        let legal_cards = self.all_legal_chance_cards();
        let mut results = Vec::with_capacity(legal_cards.len());
        for card in legal_cards {
            if self.play_chance_card(card).is_ok() {
                let rep = self.inspect_current_node();
                let [r_oop, r_ip] = self.current_combo_reaches();

                let next_agg_freqs = if !rep.actions.is_empty() {
                    let act_reach = if rep.node_type == "oop" { &r_oop } else { &r_ip };
                    let n_combos = act_reach.len();
                    let reach_sum: f64 = act_reach.iter().map(|&x| x as f64).sum();
                    if reach_sum > 1e-12 {
                        (0..rep.actions.len())
                            .map(|a| {
                                let row = &rep.strategy[a * n_combos..(a + 1) * n_combos];
                                let s: f64 = row
                                    .iter()
                                    .zip(act_reach.iter())
                                    .map(|(&p, &r)| (p as f64) * (r as f64))
                                    .sum();
                                (s / reach_sum) as f32
                            })
                            .collect()
                    } else {
                        vec![1.0 / (rep.actions.len() as f32); rep.actions.len()]
                    }
                } else {
                    Vec::new()
                };

                let oop_sum: f64 = r_oop.iter().map(|&x| x as f64).sum();
                let ip_sum: f64 = r_ip.iter().map(|&x| x as f64).sum();
                let range_ev_oop = if oop_sum > 1e-12 {
                    let num: f64 = rep
                        .ev_oop
                        .iter()
                        .zip(r_oop.iter())
                        .filter(|(&ev, _)| ev.is_finite())
                        .map(|(&ev, &r)| (ev as f64) * (r as f64))
                        .sum();
                    (num / oop_sum) as f32
                } else {
                    0.0
                };
                let range_ev_ip = if ip_sum > 1e-12 {
                    let num: f64 = rep
                        .ev_ip
                        .iter()
                        .zip(r_ip.iter())
                        .filter(|(&ev, _)| ev.is_finite())
                        .map(|(&ev, &r)| (ev as f64) * (r as f64))
                        .sum();
                    (num / ip_sum) as f32
                } else {
                    0.0
                };

                results.push(ChanceRunoutSummary {
                    card,
                    next_node_type: rep.node_type,
                    next_actions: rep.actions,
                    next_agg_freqs,
                    ev_oop: rep.ev_oop,
                    ev_ip: rep.ev_ip,
                    range_ev_oop,
                    range_ev_ip,
                });

                self.cursor_node = saved_node;
                self.cursor_suit_perm = saved_perm;
                self.cursor_combo_map = saved_map.clone();
                self.history.truncate(saved_hist_len);
            }
        }
        results
    }

    fn reset_unlocked_regrets_and_cum_strategies(&mut self) {
        for n in &mut self.nodes {
            if let ConcreteNodeKind::Player {
                storage,
                locked_strategy,
                ..
            } = &mut n.kind
            {
                if locked_strategy.is_none() {
                    let st = storage.as_mut();
                    st.regrets.fill(0.0);
                    st.cum_strategy.fill(0.0);
                }
            }
        }
        self.iterations_run = 0;
    }

    fn conditional_chip_evs(
        &self,
        cfv: &[f32],
        player: usize,
        opp_reach: &[f32],
        board_mask: CardMask,
        scratch: &mut InclusionExclusionScratch,
    ) -> Vec<f32> {
        let opp = player ^ 1;
        scratch.clear();
        for (j, c_opp) in self.combos[opp].iter().enumerate() {
            if !c_opp.mask.intersects(board_mask) && opp_reach[j] > 0.0 {
                scratch.add_combo(&self.blocker_subsets[opp][j], opp_reach[j] as f64);
            }
        }
        let half_start = 0.5 * (self.tree_config.starting_pot as f64);
        let scale = self.valid_root_pair_weight;

        self.combos[player]
            .iter()
            .enumerate()
            .map(|(i, c_me)| {
                if c_me.mask.intersects(board_mask) {
                    return 0.0;
                }
                let denom = scratch.query_disjoint(&self.blocker_subsets[player][i], opp_reach);
                if denom <= 1e-12 {
                    0.0
                } else {
                    let centered_ev = (cfv[i] as f64) * scale / denom;
                    (half_start + centered_ev) as f32
                }
            })
            .collect()
    }

    // ========================================================================
    // Core DCFR, Best Response, and Terminal 4-Card Inclusion-Exclusion Engine
    // ========================================================================

    fn dcfr_traverse(
        &self,
        node_id: usize,
        player: usize,
        my_reach: &[f32],
        opp_cfreach: &[f32],
        params: &DcfrParams,
        scratch: &mut InclusionExclusionScratch,
    ) -> Vec<f32> {
        let n_hands = self.combos[player].len();

        match &self.nodes[node_id].kind {
            ConcreteNodeKind::TerminalFold {
                folded_player,
                half_pot_swing,
                board_mask,
            } => {
                let mut out = vec![0.0f32; n_hands];
                self.eval_terminal_fold(
                    &mut out,
                    player,
                    *folded_player,
                    *half_pot_swing,
                    *board_mask,
                    opp_cfreach,
                    scratch,
                );
                out
            }
            ConcreteNodeKind::TerminalShowdown {
                river_eval_id,
                half_pot_swing,
                rake_chips,
            } => {
                let mut out = vec![0.0f32; n_hands];
                self.eval_terminal_showdown(
                    &mut out,
                    player,
                    *river_eval_id,
                    *half_pot_swing,
                    *rake_chips,
                    opp_cfreach,
                    scratch,
                );
                out
            }
            ConcreteNodeKind::Chance {
                rep_cards,
                rep_children,
                iso_duplicates,
                chance_denominator,
                ..
            } => {
                let inv_chance = 1.0f32 / (*chance_denominator as f32);
                let opp = player ^ 1;

                #[cfg(not(target_arch = "wasm32"))]
                let branch_cfvs: Vec<Vec<f32>> =
                    map_chance_branches(rep_cards, rep_children, |card, child_id, local_scratch| {
                        let cmask = card.mask();
                        let mut child_my_reach = vec![0.0f32; my_reach.len()];
                        for (i, &r) in my_reach.iter().enumerate() {
                            if r != 0.0 && !self.combos[player][i].mask.intersects(cmask) {
                                child_my_reach[i] = r;
                            }
                        }
                        let mut child_reach = vec![0.0f32; opp_cfreach.len()];
                        for (j, &r) in opp_cfreach.iter().enumerate() {
                            if r != 0.0 && !self.combos[opp][j].mask.intersects(cmask) {
                                child_reach[j] = r * inv_chance;
                            }
                        }
                        let mut cfv = self.dcfr_traverse(
                            child_id,
                            player,
                            &child_my_reach,
                            &child_reach,
                            params,
                            local_scratch,
                        );
                        for (i, c) in self.combos[player].iter().enumerate() {
                            if c.mask.intersects(cmask) {
                                cfv[i] = 0.0;
                            }
                        }
                        cfv
                    });

                #[cfg(target_arch = "wasm32")]
                let branch_cfvs: Vec<Vec<f32>> = {
                    let mut out_cfvs = Vec::with_capacity(rep_children.len());
                    for (&card, &child_id) in rep_cards.iter().zip(rep_children.iter()) {
                        let cmask = card.mask();
                        let mut child_my_reach = vec![0.0f32; my_reach.len()];
                        for (i, &r) in my_reach.iter().enumerate() {
                            if r != 0.0 && !self.combos[player][i].mask.intersects(cmask) {
                                child_my_reach[i] = r;
                            }
                        }
                        let mut child_reach = vec![0.0f32; opp_cfreach.len()];
                        for (j, &r) in opp_cfreach.iter().enumerate() {
                            if r != 0.0 && !self.combos[opp][j].mask.intersects(cmask) {
                                child_reach[j] = r * inv_chance;
                            }
                        }
                        let mut cfv = self.dcfr_traverse(
                            child_id,
                            player,
                            &child_my_reach,
                            &child_reach,
                            params,
                            scratch,
                        );
                        for (i, c) in self.combos[player].iter().enumerate() {
                            if c.mask.intersects(cmask) {
                                cfv[i] = 0.0;
                            }
                        }
                        out_cfvs.push(cfv);
                    }
                    out_cfvs
                };

                let mut sum_f64 = vec![0.0f64; n_hands];
                for cfv in &branch_cfvs {
                    for (acc, &v) in sum_f64.iter_mut().zip(cfv.iter()) {
                        *acc += v as f64;
                    }
                }
                for dup in iso_duplicates {
                    let src = &branch_cfvs[dup.rep_branch_idx];
                    let swaps = &dup.swaps[player];
                    let mut swapped = src.clone();
                    for &(a, b) in swaps {
                        swapped.swap(a as usize, b as usize);
                    }
                    let dmask = dup.card.mask();
                    for (i, &v) in swapped.iter().enumerate() {
                        if !self.combos[player][i].mask.intersects(dmask) {
                            sum_f64[i] += v as f64;
                        }
                    }
                }

                sum_f64.into_iter().map(|x| x as f32).collect()
            }
            ConcreteNodeKind::Player {
                player: node_player,
                actions,
                children,
                num_buckets,
                combo_to_bucket,
                storage,
                locked_strategy,
            } => {
                let acting = *node_player;
                let n_actions = actions.len();
                let num_buckets = *num_buckets;

                if n_actions == 1 {
                    return self.dcfr_traverse(
                        children[0],
                        player,
                        my_reach,
                        opp_cfreach,
                        params,
                        scratch,
                    );
                }

                let cur_b_strat = match locked_strategy.as_deref() {
                    Some(ls) => ls.to_vec(),
                    None => {
                        let st = storage.as_ref();
                        regret_match_buckets(&st.regrets, n_actions, num_buckets)
                    }
                };

                if acting == player {
                    let mut action_cfvs: Vec<Vec<f32>> = Vec::with_capacity(n_actions);
                    for (a, &child_id) in children.iter().enumerate() {
                        let b_row = &cur_b_strat[a * num_buckets..(a + 1) * num_buckets];
                        let mut child_my_reach = vec![0.0f32; n_hands];
                        for i in 0..n_hands {
                            let b = bucket_of(combo_to_bucket, i);
                            child_my_reach[i] = my_reach[i] * b_row[b];
                        }
                        action_cfvs.push(self.dcfr_traverse(
                            child_id,
                            player,
                            &child_my_reach,
                            opp_cfreach,
                            params,
                            scratch,
                        ));
                    }

                    // Expected CFV per combo under `cur_b_strat`
                    let mut node_cfv = vec![0.0f32; n_hands];
                    for a in 0..n_actions {
                        let b_row = &cur_b_strat[a * num_buckets..(a + 1) * num_buckets];
                        let cfv_row = &action_cfvs[a];
                        for i in 0..n_hands {
                            let b = bucket_of(combo_to_bucket, i);
                            node_cfv[i] += b_row[b] * cfv_row[i];
                        }
                    }

                    // Only update regrets & cumulative strategy when this node is NOT locked!
                    if locked_strategy.is_none() {
                        let board_mask = self.nodes[node_id].board.mask();
                        let bucketed = !combo_to_bucket.is_empty();
                        let mut my_bucket_reach = vec![0.0f32; num_buckets];
                        let mut bucket_inv_norm = Vec::new();
                        if bucketed {
                            let mut init_bucket_weight = vec![0.0f32; num_buckets];
                            for i in 0..n_hands {
                                if !self.combos[player][i].mask.intersects(board_mask) {
                                    let b = combo_to_bucket[i] as usize;
                                    my_bucket_reach[b] += my_reach[i];
                                    init_bucket_weight[b] += self.initial_weights[player][i];
                                }
                            }
                            bucket_inv_norm.resize(num_buckets, 0.0f32);
                            for b in 0..num_buckets {
                                let denom = if my_bucket_reach[b] > 0.0 {
                                    my_bucket_reach[b]
                                } else {
                                    init_bucket_weight[b]
                                };
                                bucket_inv_norm[b] = if denom > 0.0 { 1.0f32 / denom } else { 0.0 };
                            }
                        } else {
                            for i in 0..n_hands {
                                if !self.combos[player][i].mask.intersects(board_mask) {
                                    my_bucket_reach[i] = my_reach[i];
                                }
                            }
                        }

                        // SAFETY: `node_id` belongs strictly to the current thread's disjoint chance subtree!
                        let st = unsafe { storage.as_disjoint_mut() };
                        let regrets = &mut st.regrets;
                        let cum_strategy = &mut st.cum_strategy;

                        // Update cumulative strategy weighted by hero's own reach probability `my_reach`!
                        for a in 0..n_actions {
                            let cs_row = &mut cum_strategy[a * num_buckets..(a + 1) * num_buckets];
                            let s_row = &cur_b_strat[a * num_buckets..(a + 1) * num_buckets];
                            for b in 0..num_buckets {
                                cs_row[b] =
                                    cs_row[b] * params.gamma + my_bucket_reach[b] * s_row[b];
                            }
                        }

                        // Counterfactual regret update:
                        // - In exact `Identity` mode (`!bucketed`), instantaneous counterfactual regret is
                        //   unweighted by hero's own reach (`cfv_row[i] - node_cfv[i]`), matching classic CFR
                        //   (Zinkevich et al., 2007) and upstream `b-inary/postflop-solver`.
                        // - In `bucketed` mode (`k < N` or bijective `k >= N`), combos within bucket `b` are
                        //   averaged by their normalized conditional composition `p(i | b) = my_reach[i] / my_bucket_reach[b]`
                        //   (with normalized prior fallback `w0[i] / sum_{j in b} w0[j]` when `my_bucket_reach[b] == 0.0`,
                        //   guarded against `denom == 0.0`), using a single division per bucket.
                        //   Note: For lossy abstractions (`k < N`), `p(i | b)` evolves across iterations as earlier-street
                        //   strategies change (imperfect recall has no general convergence theorem); in the bijective limit
                        //   `k >= N`, `p(i | b) == 1.0` identically, recovering exact `Identity` updates!
                        for a in 0..n_actions {
                            let reg_row = &mut regrets[a * num_buckets..(a + 1) * num_buckets];
                            let cfv_row = &action_cfvs[a];
                            if bucketed {
                                let mut inst_bucket = vec![0.0f32; num_buckets];
                                for i in 0..n_hands {
                                    if !self.combos[player][i].mask.intersects(board_mask) {
                                        let b = combo_to_bucket[i] as usize;
                                        let w = if my_bucket_reach[b] > 0.0 {
                                            my_reach[i]
                                        } else {
                                            self.initial_weights[player][i]
                                        };
                                        inst_bucket[b] += w * (cfv_row[i] - node_cfv[i]);
                                    }
                                }
                                for b in 0..num_buckets {
                                    let prev = reg_row[b];
                                    let coef = if prev >= 0.0 { params.alpha } else { params.beta };
                                    reg_row[b] =
                                        prev * coef + inst_bucket[b] * bucket_inv_norm[b];
                                }
                            } else {
                                for i in 0..n_hands {
                                    if self.combos[player][i].mask.intersects(board_mask) {
                                        continue;
                                    }
                                    let prev = reg_row[i];
                                    let coef = if prev >= 0.0 { params.alpha } else { params.beta };
                                    reg_row[i] = prev * coef + (cfv_row[i] - node_cfv[i]);
                                }
                            }
                        }
                    }

                    node_cfv
                } else {
                    // Opponent acts (`acting == player ^ 1`): split `opp_cfreach` by opponent's current strategy
                    let n_opp = opp_cfreach.len();
                    let mut sum_cfv = vec![0.0f32; n_hands];

                    for a in 0..n_actions {
                        let b_row = &cur_b_strat[a * num_buckets..(a + 1) * num_buckets];
                        let mut next_reach = vec![0.0f32; n_opp];
                        for j in 0..n_opp {
                            let b = bucket_of(combo_to_bucket, j);
                            next_reach[j] = opp_cfreach[j] * b_row[b];
                        }
                        let child_cfv = self.dcfr_traverse(
                            children[a],
                            player,
                            my_reach,
                            &next_reach,
                            params,
                            scratch,
                        );
                        for (acc, &v) in sum_cfv.iter_mut().zip(child_cfv.iter()) {
                            *acc += v;
                        }
                    }

                    sum_cfv
                }
            }
        }
    }

    fn strategy_cfv_traverse(
        &self,
        node_id: usize,
        player: usize,
        opp_cfreach: &[f32],
        scratch: &mut InclusionExclusionScratch,
    ) -> Vec<f32> {
        let n_hands = self.combos[player].len();
        match &self.nodes[node_id].kind {
            ConcreteNodeKind::TerminalFold {
                folded_player,
                half_pot_swing,
                board_mask,
            } => {
                let mut out = vec![0.0f32; n_hands];
                self.eval_terminal_fold(
                    &mut out,
                    player,
                    *folded_player,
                    *half_pot_swing,
                    *board_mask,
                    opp_cfreach,
                    scratch,
                );
                out
            }
            ConcreteNodeKind::TerminalShowdown {
                river_eval_id,
                half_pot_swing,
                rake_chips,
            } => {
                let mut out = vec![0.0f32; n_hands];
                self.eval_terminal_showdown(
                    &mut out,
                    player,
                    *river_eval_id,
                    *half_pot_swing,
                    *rake_chips,
                    opp_cfreach,
                    scratch,
                );
                out
            }
            ConcreteNodeKind::Chance {
                rep_cards,
                rep_children,
                iso_duplicates,
                chance_denominator,
                ..
            } => {
                let inv_chance = 1.0f32 / (*chance_denominator as f32);
                let opp = player ^ 1;

                #[cfg(not(target_arch = "wasm32"))]
                let branch_cfvs: Vec<Vec<f32>> =
                    map_chance_branches(rep_cards, rep_children, |card, child_id, local_scratch| {
                        let cmask = card.mask();
                        let mut child_reach = vec![0.0f32; opp_cfreach.len()];
                        for (j, &r) in opp_cfreach.iter().enumerate() {
                            if r != 0.0 && !self.combos[opp][j].mask.intersects(cmask) {
                                child_reach[j] = r * inv_chance;
                            }
                        }
                        let mut cfv = self.strategy_cfv_traverse(
                            child_id,
                            player,
                            &child_reach,
                            local_scratch,
                        );
                        for (i, c) in self.combos[player].iter().enumerate() {
                            if c.mask.intersects(cmask) {
                                cfv[i] = 0.0;
                            }
                        }
                        cfv
                    });

                #[cfg(target_arch = "wasm32")]
                let branch_cfvs: Vec<Vec<f32>> = {
                    let mut out_cfvs = Vec::with_capacity(rep_children.len());
                    for (&card, &child_id) in rep_cards.iter().zip(rep_children.iter()) {
                        let cmask = card.mask();
                        let mut child_reach = vec![0.0f32; opp_cfreach.len()];
                        for (j, &r) in opp_cfreach.iter().enumerate() {
                            if r != 0.0 && !self.combos[opp][j].mask.intersects(cmask) {
                                child_reach[j] = r * inv_chance;
                            }
                        }
                        let mut cfv =
                            self.strategy_cfv_traverse(child_id, player, &child_reach, scratch);
                        for (i, c) in self.combos[player].iter().enumerate() {
                            if c.mask.intersects(cmask) {
                                cfv[i] = 0.0;
                            }
                        }
                        out_cfvs.push(cfv);
                    }
                    out_cfvs
                };

                let mut sum_f64 = vec![0.0f64; n_hands];
                for cfv in &branch_cfvs {
                    for (acc, &v) in sum_f64.iter_mut().zip(cfv.iter()) {
                        *acc += v as f64;
                    }
                }
                for dup in iso_duplicates {
                    let src = &branch_cfvs[dup.rep_branch_idx];
                    let swaps = &dup.swaps[player];
                    let mut swapped = src.clone();
                    for &(a, b) in swaps {
                        swapped.swap(a as usize, b as usize);
                    }
                    let dmask = dup.card.mask();
                    for (i, &v) in swapped.iter().enumerate() {
                        if !self.combos[player][i].mask.intersects(dmask) {
                            sum_f64[i] += v as f64;
                        }
                    }
                }
                sum_f64.into_iter().map(|x| x as f32).collect()
            }
            ConcreteNodeKind::Player {
                player: node_player,
                actions,
                children,
                num_buckets,
                combo_to_bucket,
                storage,
                locked_strategy,
            } => {
                let n_actions = actions.len();
                if n_actions == 1 {
                    return self.strategy_cfv_traverse(children[0], player, opp_cfreach, scratch);
                }
                let st = storage.as_ref();
                let avg_b_strat = average_or_current_bucket_strategy(
                    locked_strategy.as_deref(),
                    &st.cum_strategy,
                    &st.regrets,
                    n_actions,
                    *num_buckets,
                );

                if *node_player == player {
                    let mut node_cfv = vec![0.0f32; n_hands];
                    for (a, &child_id) in children.iter().enumerate() {
                        let child_cfv =
                            self.strategy_cfv_traverse(child_id, player, opp_cfreach, scratch);
                        let b_row = &avg_b_strat[a * (*num_buckets)..(a + 1) * (*num_buckets)];
                        for i in 0..n_hands {
                            let b = bucket_of(combo_to_bucket, i);
                            node_cfv[i] += b_row[b] * child_cfv[i];
                        }
                    }
                    node_cfv
                } else {
                    let n_opp = opp_cfreach.len();
                    let mut sum_cfv = vec![0.0f32; n_hands];
                    for (a, &child_id) in children.iter().enumerate() {
                        let b_row = &avg_b_strat[a * (*num_buckets)..(a + 1) * (*num_buckets)];
                        let mut next_reach = vec![0.0f32; n_opp];
                        for j in 0..n_opp {
                            let b = bucket_of(combo_to_bucket, j);
                            next_reach[j] = opp_cfreach[j] * b_row[b];
                        }
                        let child_cfv =
                            self.strategy_cfv_traverse(child_id, player, &next_reach, scratch);
                        for (acc, &v) in sum_cfv.iter_mut().zip(child_cfv.iter()) {
                            *acc += v;
                        }
                    }
                    sum_cfv
                }
            }
        }
    }

    /// Full `f64` precision strategy counterfactual value traversal (`1e-13` exactness across Flop/Turn chance trees).
    fn strategy_cfv_traverse_f64(
        &self,
        node_id: usize,
        player: usize,
        opp_cfreach: &[f64],
        scratch: &mut InclusionExclusionScratch,
    ) -> Vec<f64> {
        let n_hands = self.combos[player].len();
        match &self.nodes[node_id].kind {
            ConcreteNodeKind::TerminalFold {
                folded_player,
                half_pot_swing,
                board_mask,
            } => {
                let mut out = vec![0.0f64; n_hands];
                self.eval_terminal_fold_f64(
                    &mut out,
                    player,
                    *folded_player,
                    *half_pot_swing,
                    *board_mask,
                    opp_cfreach,
                    scratch,
                );
                out
            }
            ConcreteNodeKind::TerminalShowdown {
                river_eval_id,
                half_pot_swing,
                rake_chips,
            } => {
                let mut out = vec![0.0f64; n_hands];
                self.eval_terminal_showdown_f64(
                    &mut out,
                    player,
                    *river_eval_id,
                    *half_pot_swing,
                    *rake_chips,
                    opp_cfreach,
                    scratch,
                );
                out
            }
            ConcreteNodeKind::Chance {
                rep_cards,
                rep_children,
                iso_duplicates,
                chance_denominator,
            } => {
                let inv_chance = 1.0f64 / (*chance_denominator as f64);
                let opp = player ^ 1;

                #[cfg(not(target_arch = "wasm32"))]
                let branch_cfvs: Vec<Vec<f64>> =
                    map_chance_branches(rep_cards, rep_children, |card, child_id, local_scratch| {
                        let cmask = card.mask();
                        let mut child_reach = vec![0.0f64; opp_cfreach.len()];
                        for (j, &r) in opp_cfreach.iter().enumerate() {
                            if r != 0.0 && !self.combos[opp][j].mask.intersects(cmask) {
                                child_reach[j] = r * inv_chance;
                            }
                        }
                        let mut cfv = self.strategy_cfv_traverse_f64(
                            child_id,
                            player,
                            &child_reach,
                            local_scratch,
                        );
                        for (i, c) in self.combos[player].iter().enumerate() {
                            if c.mask.intersects(cmask) {
                                cfv[i] = 0.0;
                            }
                        }
                        cfv
                    });

                #[cfg(target_arch = "wasm32")]
                let branch_cfvs: Vec<Vec<f64>> = {
                    let mut out_cfvs = Vec::with_capacity(rep_children.len());
                    for (&card, &child_id) in rep_cards.iter().zip(rep_children.iter()) {
                        let cmask = card.mask();
                        let mut child_reach = vec![0.0f64; opp_cfreach.len()];
                        for (j, &r) in opp_cfreach.iter().enumerate() {
                            if r != 0.0 && !self.combos[opp][j].mask.intersects(cmask) {
                                child_reach[j] = r * inv_chance;
                            }
                        }
                        let mut cfv =
                            self.strategy_cfv_traverse_f64(child_id, player, &child_reach, scratch);
                        for (i, c) in self.combos[player].iter().enumerate() {
                            if c.mask.intersects(cmask) {
                                cfv[i] = 0.0;
                            }
                        }
                        out_cfvs.push(cfv);
                    }
                    out_cfvs
                };

                let mut sum_f64 = vec![0.0f64; n_hands];
                for cfv in &branch_cfvs {
                    for (acc, &v) in sum_f64.iter_mut().zip(cfv.iter()) {
                        *acc += v;
                    }
                }
                for dup in iso_duplicates {
                    let src = &branch_cfvs[dup.rep_branch_idx];
                    let swaps = &dup.swaps[player];
                    let mut swapped = src.clone();
                    for &(a, b) in swaps {
                        swapped.swap(a as usize, b as usize);
                    }
                    let dmask = dup.card.mask();
                    for (i, &v) in swapped.iter().enumerate() {
                        if !self.combos[player][i].mask.intersects(dmask) {
                            sum_f64[i] += v;
                        }
                    }
                }
                sum_f64
            }
            ConcreteNodeKind::Player {
                player: node_player,
                actions,
                children,
                num_buckets,
                combo_to_bucket,
                storage,
                locked_strategy,
            } => {
                let n_actions = actions.len();
                if n_actions == 1 {
                    return self.strategy_cfv_traverse_f64(
                        children[0],
                        player,
                        opp_cfreach,
                        scratch,
                    );
                }
                let st = storage.as_ref();
                let avg_b_strat = average_or_current_bucket_strategy(
                    locked_strategy.as_deref(),
                    &st.cum_strategy,
                    &st.regrets,
                    n_actions,
                    *num_buckets,
                );

                if *node_player == player {
                    let mut node_cfv = vec![0.0f64; n_hands];
                    for (a, &child_id) in children.iter().enumerate() {
                        let child_cfv =
                            self.strategy_cfv_traverse_f64(child_id, player, opp_cfreach, scratch);
                        let b_row = &avg_b_strat[a * (*num_buckets)..(a + 1) * (*num_buckets)];
                        for i in 0..n_hands {
                            let b = bucket_of(combo_to_bucket, i);
                            node_cfv[i] += (b_row[b] as f64) * child_cfv[i];
                        }
                    }
                    node_cfv
                } else {
                    let n_opp = opp_cfreach.len();
                    let mut sum_cfv = vec![0.0f64; n_hands];
                    for (a, &child_id) in children.iter().enumerate() {
                        let b_row = &avg_b_strat[a * (*num_buckets)..(a + 1) * (*num_buckets)];
                        let mut next_reach = vec![0.0f64; n_opp];
                        for j in 0..n_opp {
                            let b = bucket_of(combo_to_bucket, j);
                            next_reach[j] = opp_cfreach[j] * (b_row[b] as f64);
                        }
                        let child_cfv =
                            self.strategy_cfv_traverse_f64(child_id, player, &next_reach, scratch);
                        for (acc, &v) in sum_cfv.iter_mut().zip(child_cfv.iter()) {
                            *acc += v;
                        }
                    }
                    sum_cfv
                }
            }
        }
    }

    fn best_response_traverse(
        &self,
        node_id: usize,
        player: usize,
        opp_cfreach: &[f32],
        scratch: &mut InclusionExclusionScratch,
    ) -> Vec<f32> {
        let n_hands = self.combos[player].len();
        match &self.nodes[node_id].kind {
            ConcreteNodeKind::TerminalFold {
                folded_player,
                half_pot_swing,
                board_mask,
            } => {
                let mut out = vec![0.0f32; n_hands];
                self.eval_terminal_fold(
                    &mut out,
                    player,
                    *folded_player,
                    *half_pot_swing,
                    *board_mask,
                    opp_cfreach,
                    scratch,
                );
                out
            }
            ConcreteNodeKind::TerminalShowdown {
                river_eval_id,
                half_pot_swing,
                rake_chips,
            } => {
                let mut out = vec![0.0f32; n_hands];
                self.eval_terminal_showdown(
                    &mut out,
                    player,
                    *river_eval_id,
                    *half_pot_swing,
                    *rake_chips,
                    opp_cfreach,
                    scratch,
                );
                out
            }
            ConcreteNodeKind::Chance {
                rep_cards,
                rep_children,
                iso_duplicates,
                chance_denominator,
                ..
            } => {
                let inv_chance = 1.0f32 / (*chance_denominator as f32);
                let opp = player ^ 1;

                #[cfg(not(target_arch = "wasm32"))]
                let branch_cfvs: Vec<Vec<f32>> =
                    map_chance_branches(rep_cards, rep_children, |card, child_id, local_scratch| {
                        let cmask = card.mask();
                        let mut child_reach = vec![0.0f32; opp_cfreach.len()];
                        for (j, &r) in opp_cfreach.iter().enumerate() {
                            if r != 0.0 && !self.combos[opp][j].mask.intersects(cmask) {
                                child_reach[j] = r * inv_chance;
                            }
                        }
                        let mut cfv = self.best_response_traverse(
                            child_id,
                            player,
                            &child_reach,
                            local_scratch,
                        );
                        for (i, c) in self.combos[player].iter().enumerate() {
                            if c.mask.intersects(cmask) {
                                cfv[i] = 0.0;
                            }
                        }
                        cfv
                    });

                #[cfg(target_arch = "wasm32")]
                let branch_cfvs: Vec<Vec<f32>> = {
                    let mut out_cfvs = Vec::with_capacity(rep_children.len());
                    for (&card, &child_id) in rep_cards.iter().zip(rep_children.iter()) {
                        let cmask = card.mask();
                        let mut child_reach = vec![0.0f32; opp_cfreach.len()];
                        for (j, &r) in opp_cfreach.iter().enumerate() {
                            if r != 0.0 && !self.combos[opp][j].mask.intersects(cmask) {
                                child_reach[j] = r * inv_chance;
                            }
                        }
                        let mut cfv =
                            self.best_response_traverse(child_id, player, &child_reach, scratch);
                        for (i, c) in self.combos[player].iter().enumerate() {
                            if c.mask.intersects(cmask) {
                                cfv[i] = 0.0;
                            }
                        }
                        out_cfvs.push(cfv);
                    }
                    out_cfvs
                };

                let mut sum_f64 = vec![0.0f64; n_hands];
                for cfv in &branch_cfvs {
                    for (acc, &v) in sum_f64.iter_mut().zip(cfv.iter()) {
                        *acc += v as f64;
                    }
                }
                for dup in iso_duplicates {
                    let src = &branch_cfvs[dup.rep_branch_idx];
                    let swaps = &dup.swaps[player];
                    let mut swapped = src.clone();
                    for &(a, b) in swaps {
                        swapped.swap(a as usize, b as usize);
                    }
                    let dmask = dup.card.mask();
                    for (i, &v) in swapped.iter().enumerate() {
                        if !self.combos[player][i].mask.intersects(dmask) {
                            sum_f64[i] += v as f64;
                        }
                    }
                }
                sum_f64.into_iter().map(|x| x as f32).collect()
            }
            ConcreteNodeKind::Player {
                player: node_player,
                actions,
                children,
                num_buckets,
                combo_to_bucket,
                storage,
                locked_strategy,
            } => {
                let n_actions = actions.len();
                if n_actions == 1 {
                    return self.best_response_traverse(children[0], player, opp_cfreach, scratch);
                }
                if *node_player == player {
                    if let Some(ls) = locked_strategy.as_deref() {
                        let mut node_cfv = vec![0.0f32; n_hands];
                        for (a, &child_id) in children.iter().enumerate() {
                            let child_cfv =
                                self.best_response_traverse(child_id, player, opp_cfreach, scratch);
                            let b_row = &ls[a * (*num_buckets)..(a + 1) * (*num_buckets)];
                            for i in 0..n_hands {
                                let b = bucket_of(combo_to_bucket, i);
                                node_cfv[i] += b_row[b] * child_cfv[i];
                            }
                        }
                        node_cfv
                    } else {
                        // Exact combo-wise max across actions!
                        let mut best =
                            self.best_response_traverse(children[0], player, opp_cfreach, scratch);
                        for &child_id in &children[1..] {
                            let cand =
                                self.best_response_traverse(child_id, player, opp_cfreach, scratch);
                            for (b, &c) in best.iter_mut().zip(cand.iter()) {
                                if c > *b {
                                    *b = c;
                                }
                            }
                        }
                        best
                    }
                } else {
                    let st = storage.as_ref();
                    let avg_b_strat = average_or_current_bucket_strategy(
                        locked_strategy.as_deref(),
                        &st.cum_strategy,
                        &st.regrets,
                        n_actions,
                        *num_buckets,
                    );
                    let n_opp = opp_cfreach.len();
                    let mut sum_cfv = vec![0.0f32; n_hands];
                    for (a, &child_id) in children.iter().enumerate() {
                        let b_row = &avg_b_strat[a * (*num_buckets)..(a + 1) * (*num_buckets)];
                        let mut next_reach = vec![0.0f32; n_opp];
                        for j in 0..n_opp {
                            let b = bucket_of(combo_to_bucket, j);
                            next_reach[j] = opp_cfreach[j] * b_row[b];
                        }
                        let child_cfv =
                            self.best_response_traverse(child_id, player, &next_reach, scratch);
                        for (acc, &v) in sum_cfv.iter_mut().zip(child_cfv.iter()) {
                            *acc += v;
                        }
                    }
                    sum_cfv
                }
            }
        }
    }

    /// Evaluates a fold terminal node using exact 4-card blocker removal (`O(N + M)` inclusion-exclusion
    /// for medium/large ranges, or direct `CardMask` bit-test for small ranges).
    #[allow(clippy::too_many_arguments)]
    fn eval_terminal_fold(
        &self,
        out: &mut [f32],
        player: usize,
        folded_player: usize,
        half_pot_swing: f64,
        board_mask: CardMask,
        opp_cfreach: &[f32],
        scratch: &mut InclusionExclusionScratch,
    ) {
        let payoff = if folded_player != player {
            half_pot_swing / self.valid_root_pair_weight
        } else {
            -half_pot_swing / self.valid_root_pair_weight
        };
        let opp = player ^ 1;
        let my_combos = &self.combos[player];
        let opp_combos = &self.combos[opp];

        // Fast bitmask path for tiny ranges (avoids touching w2/w3 arrays)
        if my_combos.len() * opp_combos.len() <= 512 {
            for (i, c_me) in my_combos.iter().enumerate() {
                if c_me.mask.intersects(board_mask) {
                    out[i] = 0.0;
                    continue;
                }
                let mut w_sum = 0.0f64;
                for (j, c_opp) in opp_combos.iter().enumerate() {
                    let r = opp_cfreach[j];
                    if r != 0.0
                        && !c_opp.mask.intersects(board_mask)
                        && !c_me.mask.intersects(c_opp.mask)
                    {
                        w_sum += r as f64;
                    }
                }
                out[i] = (payoff * w_sum) as f32;
            }
            return;
        }

        // Exact O(N + M) 15-term 4-card inclusion-exclusion!
        self.eval_terminal_fold_ie_unconditional(
            out,
            player,
            payoff,
            board_mask,
            opp_cfreach,
            scratch,
        );
    }

    /// Unconditional 15-term 4-card inclusion-exclusion fold evaluator (used by medium/large ranges in
    /// `eval_terminal_fold` and directly tested against `brute_fold_ev` even on tiny 1v1 / 4v4 fixtures).
    fn eval_terminal_fold_ie_unconditional(
        &self,
        out: &mut [f32],
        player: usize,
        payoff: f64,
        board_mask: CardMask,
        opp_cfreach: &[f32],
        scratch: &mut InclusionExclusionScratch,
    ) {
        let opp = player ^ 1;
        let my_combos = &self.combos[player];
        let opp_combos = &self.combos[opp];

        scratch.clear();
        let opp_subs = &self.blocker_subsets[opp];
        for (j, c_opp) in opp_combos.iter().enumerate() {
            let r = opp_cfreach[j];
            if r != 0.0 && !c_opp.mask.intersects(board_mask) {
                scratch.add_combo(&opp_subs[j], r as f64);
            }
        }
        if scratch.total == 0.0 {
            out.fill(0.0);
            return;
        }

        let my_subs = &self.blocker_subsets[player];
        for (i, c_me) in my_combos.iter().enumerate() {
            if c_me.mask.intersects(board_mask) {
                out[i] = 0.0;
            } else {
                let disjoint_reach = scratch.query_disjoint(&my_subs[i], opp_cfreach);
                out[i] = (payoff * disjoint_reach) as f32;
            }
        }
    }

    /// Evaluates a river showdown terminal node using 2-pass strength-sorted exact 4-card inclusion-exclusion!
    #[allow(clippy::too_many_arguments)]
    fn eval_terminal_showdown(
        &self,
        out: &mut [f32],
        player: usize,
        river_eval_id: usize,
        half_pot_swing: f64,
        rake_chips: f64,
        opp_cfreach: &[f32],
        scratch: &mut InclusionExclusionScratch,
    ) {
        let re = &self.river_evals[river_eval_id];
        let my_sorted = &re.sorted_strengths[player];
        let opp_sorted = &re.sorted_strengths[player ^ 1];
        out.fill(0.0);

        if my_sorted.is_empty() || opp_sorted.is_empty() {
            return;
        }

        let win_payoff = (half_pot_swing - rake_chips) / self.valid_root_pair_weight;
        let lose_payoff = -half_pot_swing / self.valid_root_pair_weight;
        let tie_payoff = -0.5 * rake_chips / self.valid_root_pair_weight;
        let opp = player ^ 1;

        // Fast bitmask path for tiny active river sets
        if my_sorted.len() * opp_sorted.len() <= 512 {
            let my_combos = &self.combos[player];
            let opp_combos = &self.combos[opp];
            for &SortedStrength {
                strength: s_me,
                combo_idx: i_u32,
            } in my_sorted
            {
                let i = i_u32 as usize;
                let m_me = my_combos[i].mask;
                let mut val = 0.0f64;
                for &SortedStrength {
                    strength: s_opp,
                    combo_idx: j_u32,
                } in opp_sorted
                {
                    let j = j_u32 as usize;
                    let r = opp_cfreach[j];
                    if r != 0.0 && !m_me.intersects(opp_combos[j].mask) {
                        if s_me > s_opp {
                            val += win_payoff * (r as f64);
                        } else if s_me < s_opp {
                            val += lose_payoff * (r as f64);
                        } else if rake_chips != 0.0 {
                            val += tie_payoff * (r as f64);
                        }
                    }
                }
                out[i] = val as f32;
            }
            return;
        }

        // Exact O(N + M) 2-pass 4-card 15-term inclusion-exclusion!
        self.eval_terminal_showdown_ie_unconditional(
            out,
            player,
            river_eval_id,
            win_payoff,
            lose_payoff,
            tie_payoff,
            opp_cfreach,
            scratch,
        );
    }

    /// Unconditional 15-term 4-card inclusion-exclusion showdown evaluator (`O(N + M)`).
    #[allow(clippy::too_many_arguments)]
    fn eval_terminal_showdown_ie_unconditional(
        &self,
        out: &mut [f32],
        player: usize,
        river_eval_id: usize,
        win_payoff: f64,
        lose_payoff: f64,
        tie_payoff: f64,
        opp_cfreach: &[f32],
        scratch: &mut InclusionExclusionScratch,
    ) {
        let re = &self.river_evals[river_eval_id];
        let my_sorted = &re.sorted_strengths[player];
        let opp_sorted = &re.sorted_strengths[player ^ 1];
        out.fill(0.0);

        if my_sorted.is_empty() || opp_sorted.is_empty() {
            return;
        }

        let opp = player ^ 1;
        let my_subs = &self.blocker_subsets[player];
        let opp_subs = &self.blocker_subsets[opp];

        // Pass 1 (ascending): accumulate strictly weaker opponent hands (`opp.strength < me.strength`)
        scratch.clear();
        let mut p_idx = 0usize;
        for &SortedStrength {
            strength: s_me,
            combo_idx: i_u32,
        } in my_sorted
        {
            while p_idx < opp_sorted.len() && opp_sorted[p_idx].strength < s_me {
                let opp_i = opp_sorted[p_idx].combo_idx as usize;
                let r = opp_cfreach[opp_i];
                if r != 0.0 {
                    scratch.add_combo(&opp_subs[opp_i], r as f64);
                }
                p_idx += 1;
            }
            if scratch.total != 0.0 {
                let win_w = scratch.query_disjoint_strict_ineq(&my_subs[i_u32 as usize]);
                out[i_u32 as usize] = (win_payoff * win_w) as f32;
            }
        }

        // Pass 2 (descending): accumulate strictly stronger opponent hands (`opp.strength > me.strength`)
        scratch.clear();
        let mut q_idx = opp_sorted.len();
        for &SortedStrength {
            strength: s_me,
            combo_idx: i_u32,
        } in my_sorted.iter().rev()
        {
            while q_idx > 0 && opp_sorted[q_idx - 1].strength > s_me {
                q_idx -= 1;
                let opp_i = opp_sorted[q_idx].combo_idx as usize;
                let r = opp_cfreach[opp_i];
                if r != 0.0 {
                    scratch.add_combo(&opp_subs[opp_i], r as f64);
                }
            }
            if scratch.total != 0.0 {
                let lose_w = scratch.query_disjoint_strict_ineq(&my_subs[i_u32 as usize]);
                out[i_u32 as usize] += (lose_payoff * lose_w) as f32;
            }
        }

        // Optional Pass 3 (raked ties only): subtract half-rake on exact split-pot ties (`opp.strength == me.strength`)
        if tie_payoff != 0.0 {
            let my_combos = &self.combos[player];
            let opp_combos = &self.combos[opp];
            let mut t_start = 0usize;
            for &SortedStrength {
                strength: s_me,
                combo_idx: i_u32,
            } in my_sorted
            {
                while t_start < opp_sorted.len() && opp_sorted[t_start].strength < s_me {
                    t_start += 1;
                }
                let m_me = my_combos[i_u32 as usize].mask;
                let mut tie_w = 0.0f64;
                let mut k = t_start;
                while k < opp_sorted.len() && opp_sorted[k].strength == s_me {
                    let opp_i = opp_sorted[k].combo_idx as usize;
                    let r = opp_cfreach[opp_i];
                    if r != 0.0 && !m_me.intersects(opp_combos[opp_i].mask) {
                        tie_w += r as f64;
                    }
                    k += 1;
                }
                if tie_w != 0.0 {
                    out[i_u32 as usize] += (tie_payoff * tie_w) as f32;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_terminal_fold_f64(
        &self,
        out: &mut [f64],
        player: usize,
        folded_player: usize,
        half_pot_swing: f64,
        board_mask: CardMask,
        opp_cfreach: &[f64],
        scratch: &mut InclusionExclusionScratch,
    ) {
        let payoff = if folded_player != player {
            half_pot_swing / self.valid_root_pair_weight
        } else {
            -half_pot_swing / self.valid_root_pair_weight
        };
        let opp = player ^ 1;
        let my_combos = &self.combos[player];
        let opp_combos = &self.combos[opp];

        scratch.clear();
        let opp_subs = &self.blocker_subsets[opp];
        for (j, c_opp) in opp_combos.iter().enumerate() {
            let r = opp_cfreach[j];
            if r != 0.0 && !c_opp.mask.intersects(board_mask) {
                scratch.add_combo(&opp_subs[j], r);
            }
        }
        if scratch.total == 0.0 {
            out.fill(0.0);
            return;
        }

        let my_subs = &self.blocker_subsets[player];
        for (i, c_me) in my_combos.iter().enumerate() {
            if c_me.mask.intersects(board_mask) {
                out[i] = 0.0;
            } else {
                let disjoint_reach = scratch.query_disjoint_f64(&my_subs[i], opp_cfreach);
                out[i] = payoff * disjoint_reach;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_terminal_showdown_f64(
        &self,
        out: &mut [f64],
        player: usize,
        river_eval_id: usize,
        half_pot_swing: f64,
        rake_chips: f64,
        opp_cfreach: &[f64],
        scratch: &mut InclusionExclusionScratch,
    ) {
        let re = &self.river_evals[river_eval_id];
        let my_sorted = &re.sorted_strengths[player];
        let opp_sorted = &re.sorted_strengths[player ^ 1];
        out.fill(0.0);

        if my_sorted.is_empty() || opp_sorted.is_empty() {
            return;
        }

        let win_payoff = (half_pot_swing - rake_chips) / self.valid_root_pair_weight;
        let lose_payoff = -half_pot_swing / self.valid_root_pair_weight;
        let tie_payoff = -0.5 * rake_chips / self.valid_root_pair_weight;
        let opp = player ^ 1;
        let my_subs = &self.blocker_subsets[player];
        let opp_subs = &self.blocker_subsets[opp];

        scratch.clear();
        let mut p_idx = 0usize;
        for &SortedStrength {
            strength: s_me,
            combo_idx: i_u32,
        } in my_sorted
        {
            while p_idx < opp_sorted.len() && opp_sorted[p_idx].strength < s_me {
                let opp_i = opp_sorted[p_idx].combo_idx as usize;
                let r = opp_cfreach[opp_i];
                if r != 0.0 {
                    scratch.add_combo(&opp_subs[opp_i], r);
                }
                p_idx += 1;
            }
            if scratch.total != 0.0 {
                let win_w = scratch.query_disjoint_strict_ineq(&my_subs[i_u32 as usize]);
                out[i_u32 as usize] = win_payoff * win_w;
            }
        }

        scratch.clear();
        let mut q_idx = opp_sorted.len();
        for &SortedStrength {
            strength: s_me,
            combo_idx: i_u32,
        } in my_sorted.iter().rev()
        {
            while q_idx > 0 && opp_sorted[q_idx - 1].strength > s_me {
                q_idx -= 1;
                let opp_i = opp_sorted[q_idx].combo_idx as usize;
                let r = opp_cfreach[opp_i];
                if r != 0.0 {
                    scratch.add_combo(&opp_subs[opp_i], r);
                }
            }
            if scratch.total != 0.0 {
                let lose_w = scratch.query_disjoint_strict_ineq(&my_subs[i_u32 as usize]);
                out[i_u32 as usize] += lose_payoff * lose_w;
            }
        }

        if tie_payoff != 0.0 {
            let my_combos = &self.combos[player];
            let opp_combos = &self.combos[opp];
            let mut t_start = 0usize;
            for &SortedStrength {
                strength: s_me,
                combo_idx: i_u32,
            } in my_sorted
            {
                while t_start < opp_sorted.len() && opp_sorted[t_start].strength < s_me {
                    t_start += 1;
                }
                let m_me = my_combos[i_u32 as usize].mask;
                let mut tie_w = 0.0f64;
                let mut k = t_start;
                while k < opp_sorted.len() && opp_sorted[k].strength == s_me {
                    let opp_i = opp_sorted[k].combo_idx as usize;
                    let r = opp_cfreach[opp_i];
                    if r != 0.0 && !m_me.intersects(opp_combos[opp_i].mask) {
                        tie_w += r;
                    }
                    k += 1;
                }
                if tie_w != 0.0 {
                    out[i_u32 as usize] += tie_payoff * tie_w;
                }
            }
        }
    }
}

// ============================================================================
// Tree Expansion & Memory Estimation Helpers
// ============================================================================

type CachedBucketMap = (usize, std::sync::Arc<[u32]>);

struct ConcreteBuilder<'a> {
    combos: &'a [Vec<WeightedHand>; 2],
    combo_ids: &'a [Vec<ComboId>; 2],
    oop_iso_vec: Vec<(ComboId, f32)>,
    ip_iso_vec: Vec<(ComboId, f32)>,
    iso_cache: std::collections::HashMap<u64, ChanceIsomorphism>,
    action_tree: &'a ActionTree,
    abstraction: &'a dyn HandAbstraction,
    enable_iso: bool,
    rake_fraction: f64,
    rake_cap_chips: f64,
    nodes: Vec<ConcreteNode>,
    river_evals: Vec<RiverEvalData>,
    river_cache_map: std::collections::HashMap<[u8; 5], usize>,
    bucket_cache_map: std::collections::HashMap<(u64, u8), CachedBucketMap>,
    empty_bucket_map: std::sync::Arc<[u32]>,
    regret_strat_bytes: u64,
    bucket_map_bytes: u64,
}

impl<'a> ConcreteBuilder<'a> {
    fn get_or_build_bucket_map(&mut self, p: usize, board: &Board) -> (usize, std::sync::Arc<[u32]>) {
        let key = (board.mask().bits(), p as u8);
        if let Some(cached) = self.bucket_cache_map.get(&key) {
            return cached.clone();
        }
        let (num_buckets, raw_map) =
            self.abstraction
                .assign_buckets(p, board, &self.combos[p], &self.combos[p ^ 1]);
        let is_identity = num_buckets == self.combos[p].len()
            && raw_map.iter().enumerate().all(|(i, &b)| b as usize == i);
        let arc_map: std::sync::Arc<[u32]> = if is_identity {
            self.empty_bucket_map.clone()
        } else {
            self.bucket_map_bytes += (raw_map.len() as u64) * 4;
            std::sync::Arc::from(raw_map)
        };
        let entry = (num_buckets, arc_map);
        self.bucket_cache_map.insert(key, entry.clone());
        entry
    }

    fn get_or_build_river_eval(&mut self, board: &Board) -> usize {
        let rc = board.river_cards().expect("showdown requires 5-card river board");
        let mut key = [
            rc[0].id(),
            rc[1].id(),
            rc[2].id(),
            rc[3].id(),
            rc[4].id(),
        ];
        key.sort_unstable();
        if let Some(&id) = self.river_cache_map.get(&key) {
            return id;
        }

        let cache = RiverBoardCache::new(rc);
        let bmask = cache.mask();
        let mut valid_combos: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        let mut sorted_strengths: [Vec<SortedStrength>; 2] = [Vec::new(), Vec::new()];

        for p in 0..2 {
            valid_combos[p] = Vec::with_capacity(self.combos[p].len());
            sorted_strengths[p] = Vec::with_capacity(self.combos[p].len());
            for (idx, c) in self.combos[p].iter().enumerate() {
                if !c.mask.intersects(bmask) {
                    let rank = cache.eval_hand_unchecked(c.hand).raw();
                    valid_combos[p].push(idx as u32);
                    sorted_strengths[p].push(SortedStrength {
                        strength: rank,
                        combo_idx: idx as u32,
                    });
                }
            }
            sorted_strengths[p].sort_unstable_by_key(|x| (x.strength, x.combo_idx));
            valid_combos[p].shrink_to_fit();
            sorted_strengths[p].shrink_to_fit();
        }

        let id = self.river_evals.len();
        self.river_evals.push(RiverEvalData {
            valid_combos,
            sorted_strengths,
        });
        self.river_cache_map.insert(key, id);
        id
    }

    fn expand_node(&mut self, action_node_id: usize, board: Board) -> usize {
        let anode = self.action_tree.node(action_node_id);
        let committed = anode.committed;
        let pot = anode.pot;
        let half_start = 0.5 * (self.action_tree.config().starting_pot as f64);

        let concrete_id = self.nodes.len();
        // Push placeholder to preserve parent-before-children arena ordering
        self.nodes.push(ConcreteNode {
            board,
            committed,
            pot,
            kind: ConcreteNodeKind::TerminalFold {
                folded_player: 0,
                half_pot_swing: 0.0,
                board_mask: board.mask(),
            },
        });

        let kind = match &anode.kind {
            ActionNodeKind::TerminalFold { folded_player, .. } => {
                let fp = *folded_player;
                let half_pot_swing = half_start + (committed[fp] as f64);
                ConcreteNodeKind::TerminalFold {
                    folded_player: fp,
                    half_pot_swing,
                    board_mask: board.mask(),
                }
            }
            ActionNodeKind::TerminalShowdown => {
                assert_eq!(
                    committed[0], committed[1],
                    "TerminalShowdown requires equal player commitments"
                );
                let river_eval_id = self.get_or_build_river_eval(&board);
                let half_pot_swing = half_start + (committed[0] as f64);
                let rake_chips = if self.rake_fraction > 0.0 {
                    let raw_r = (pot as f64) * self.rake_fraction;
                    if self.rake_cap_chips > 0.0 {
                        raw_r.min(self.rake_cap_chips)
                    } else {
                        raw_r
                    }
                } else {
                    0.0
                };
                ConcreteNodeKind::TerminalShowdown {
                    river_eval_id,
                    half_pot_swing,
                    rake_chips,
                }
            }
            ActionNodeKind::Chance { child, .. } => {
                let next_anode = *child;
                // In 4-card PLO heads-up, every compatible pair `(h_oop, h_ip)` removes `4 + 4 = 8` hole cards
                // from the `52 - board.len()` remaining deck cards, leaving `52 - board.len() - 8` equally likely
                // next-street cards (`41` on Flop->Turn, `40` on Turn->River)!
                let chance_denominator = 52usize
                    .saturating_sub(board.len())
                    .saturating_sub(HOLE_CARDS_BOTH_PLAYERS);
                debug_assert!(chance_denominator == 41 || chance_denominator == 40);
                if self.enable_iso {
                    let bkey = board.mask().bits();
                    let iso = if let Some(cached) = self.iso_cache.get(&bkey) {
                        cached.clone()
                    } else {
                        let computed = compute_chance_isomorphism(
                            &board,
                            &self.oop_iso_vec,
                            &self.ip_iso_vec,
                        );
                        self.iso_cache.insert(bkey, computed.clone());
                        computed
                    };
                    let mut rep_children = Vec::with_capacity(iso.representative_cards.len());
                    for &rcard in &iso.representative_cards {
                        let next_board = board.deal(rcard).expect("chance card cannot overlap board");
                        rep_children.push(self.expand_node(next_anode, next_board));
                    }
                    let iso_duplicates = iso
                        .duplicates
                        .into_iter()
                        .map(|d| ChanceIsoBranch {
                            card: d.card,
                            rep_branch_idx: d.rep_index,
                            swaps: [
                                build_combo_swap_pairs(&self.combo_ids[0], d.suit_a, d.suit_b),
                                build_combo_swap_pairs(&self.combo_ids[1], d.suit_a, d.suit_b),
                            ],
                        })
                        .collect();
                    ConcreteNodeKind::Chance {
                        rep_cards: iso.representative_cards,
                        rep_children,
                        iso_duplicates,
                        chance_denominator,
                    }
                } else {
                    let mut rep_cards = Vec::with_capacity(49);
                    let mut rep_children = Vec::with_capacity(49);
                    for cid in 0..52u8 {
                        let c = Card::new(cid).unwrap();
                        if board.mask().contains(c) {
                            continue;
                        }
                        rep_cards.push(c);
                        let next_board = board.deal(c).unwrap();
                        rep_children.push(self.expand_node(next_anode, next_board));
                    }
                    ConcreteNodeKind::Chance {
                        rep_cards,
                        rep_children,
                        iso_duplicates: Vec::new(),
                        chance_denominator,
                    }
                }
            }
            ActionNodeKind::Player {
                player,
                actions,
                children,
                ..
            } => {
                let p = *player;
                let (num_buckets, combo_to_bucket) = self.get_or_build_bucket_map(p, &board);

                let len = if actions.len() <= 1 {
                    0
                } else {
                    actions.len() * num_buckets
                };
                self.regret_strat_bytes += (len as u64) * 8;

                let mut concrete_children = Vec::with_capacity(children.len());
                for &ch in children {
                    concrete_children.push(self.expand_node(ch, board));
                }

                ConcreteNodeKind::Player {
                    player: p,
                    actions: actions.clone(),
                    children: concrete_children,
                    num_buckets,
                    combo_to_bucket,
                    storage: SyncStorageCell::new(len),
                    locked_strategy: None,
                }
            }
        };

        self.nodes[concrete_id].kind = kind;
        concrete_id
    }
}

#[allow(clippy::too_many_arguments)]
fn estimate_tree_memory(
    root_board: &Board,
    combos: &[Vec<WeightedHand>; 2],
    _combo_ids: &[Vec<ComboId>; 2],
    action_tree: &ActionTree,
    abstraction: &dyn HandAbstraction,
    enable_iso: bool,
    oop_iso: &[(ComboId, f32)],
    ip_iso: &[(ComboId, f32)],
    iso_cache: &mut std::collections::HashMap<u64, ChanceIsomorphism>,
) -> MemoryReport {
    // Exact representative runout counts per street (`turn_mult` on Turn, `total_river_branches` on River),
    // caching `ChanceIsomorphism` by `board.mask().bits()` so `ConcreteBuilder` reuses it with zero duplicate work!
    let mut unique_river_boards: std::collections::HashSet<[u8; 5]> =
        std::collections::HashSet::new();
    let (turn_mult, total_river_branches): (usize, usize) = match root_board.street() {
        Street::Flop => {
            if enable_iso {
                let flop_iso = compute_chance_isomorphism(root_board, oop_iso, ip_iso);
                let rep_turns = flop_iso.representative_cards.clone();
                iso_cache.insert(root_board.mask().bits(), flop_iso);
                let mut sum_rivers = 0usize;
                for &tc in &rep_turns {
                    let tb = root_board.deal(tc).unwrap();
                    let turn_iso = compute_chance_isomorphism(&tb, oop_iso, ip_iso);
                    sum_rivers += turn_iso.representative_cards.len();
                    for &rc in &turn_iso.representative_cards {
                        let rb = tb.deal(rc).unwrap().river_cards().unwrap();
                        let mut key = [rb[0].id(), rb[1].id(), rb[2].id(), rb[3].id(), rb[4].id()];
                        key.sort_unstable();
                        unique_river_boards.insert(key);
                    }
                    iso_cache.insert(tb.mask().bits(), turn_iso);
                }
                (rep_turns.len(), sum_rivers)
            } else {
                (49, 49 * 48)
            }
        }
        Street::Turn => {
            if enable_iso {
                let t_iso = compute_chance_isomorphism(root_board, oop_iso, ip_iso);
                let r_len = t_iso.representative_cards.len();
                iso_cache.insert(root_board.mask().bits(), t_iso);
                (1, r_len)
            } else {
                (1, 48)
            }
        }
        Street::River => (1, 1),
    };

    // Upper-bound `(num_buckets, is_non_identity)` per street for `[OOP, IP]`
    let sample_buckets = |street: Street, player: usize| -> (usize, bool) {
        let n_p = combos[player].len();
        if let Some(cap) = abstraction.max_buckets_for_street(street) {
            let nb = cap.min(n_p.max(1));
            return (nb, nb < n_p);
        }
        // Fallback for custom `HandAbstraction` implementations without `max_buckets_for_street`
        let mut sb = *root_board;
        for cid in 0..52u8 {
            if sb.street() >= street {
                break;
            }
            let c = Card::new(cid).unwrap();
            if !sb.mask().contains(c) {
                sb = sb.deal(c).unwrap();
            }
        }
        let (nb, map) =
            abstraction.assign_buckets(player, &sb, &combos[player], &combos[player ^ 1]);
        let is_id = nb == n_p && map.iter().enumerate().all(|(i, &b)| b as usize == i);
        (nb, !is_id)
    };

    let s_flop = [
        sample_buckets(Street::Flop, 0),
        sample_buckets(Street::Flop, 1),
    ];
    let s_turn = [
        sample_buckets(Street::Turn, 0),
        sample_buckets(Street::Turn, 1),
    ];
    let s_river = [
        sample_buckets(Street::River, 0),
        sample_buckets(Street::River, 1),
    ];
    let b_flop = [s_flop[0].0, s_flop[1].0];
    let b_turn = [s_turn[0].0, s_turn[1].0];
    let b_river = [s_river[0].0, s_river[1].0];

    let mut dec_nodes = 0usize;
    let mut chance_nodes = 0usize;
    let mut term_nodes = 0usize;
    let mut regret_bytes = 0u64;

    for anode in action_tree.nodes() {
        match &anode.kind {
            ActionNodeKind::Player {
                player,
                street,
                actions,
                ..
            } => {
                let (mult, nb) = match street {
                    Street::Flop => (1, b_flop[*player]),
                    Street::Turn => (turn_mult, b_turn[*player]),
                    Street::River => (total_river_branches, b_river[*player]),
                };
                dec_nodes += mult;
                if actions.len() > 1 {
                    regret_bytes += (mult as u64) * (actions.len() as u64) * (nb as u64) * 8;
                }
            }
            ActionNodeKind::Chance { next_street, .. } => {
                let mult = match next_street {
                    Street::Turn => 1,
                    Street::River => {
                        if root_board.street() == Street::Flop {
                            turn_mult
                        } else {
                            1
                        }
                    }
                    Street::Flop => 0,
                };
                chance_nodes += mult;
            }
            ActionNodeKind::TerminalFold { street, .. } => {
                let mult = match street {
                    Street::Flop => 1,
                    Street::Turn => turn_mult,
                    Street::River => total_river_branches,
                };
                term_nodes += mult;
            }
            ActionNodeKind::TerminalShowdown => {
                term_nodes += total_river_branches;
            }
        }
    }

    let river_tables = match root_board.street() {
        Street::River => 1,
        Street::Turn => total_river_branches,
        Street::Flop => {
            if enable_iso {
                unique_river_boards.len()
            } else {
                1176 // C(49, 2) = 1176 unique unordered turn+river pairs
            }
        }
    };
    let bucket_map_bytes = {
        let mut bm = 0u64;
        for p in 0..2 {
            let n_p = (combos[p].len() as u64) * 4;
            match root_board.street() {
                Street::Flop => {
                    if s_flop[p].1 {
                        bm += n_p;
                    }
                    if s_turn[p].1 {
                        bm += (turn_mult as u64) * n_p;
                    }
                    if s_river[p].1 {
                        bm += (river_tables as u64) * n_p;
                    }
                }
                Street::Turn => {
                    if s_turn[p].1 {
                        bm += n_p;
                    }
                    if s_river[p].1 {
                        bm += (river_tables as u64) * n_p;
                    }
                }
                Street::River => {
                    if s_river[p].1 {
                        bm += n_p;
                    }
                }
            }
        }
        bm
    };
    let river_eval_bytes =
        (river_tables as u64) * ((combos[0].len() + combos[1].len()) as u64) * 12;
    // Realistic concrete-node overhead (`size_of::<ConcreteNode>()` ~ 160 B + Vec action/children headers ~ 192 B)
    let node_overhead = ((dec_nodes + chance_nodes + term_nodes) as u64) * 192;

    MemoryReport {
        decision_nodes: dec_nodes,
        chance_nodes,
        terminal_nodes: term_nodes,
        river_eval_tables: river_tables,
        regret_and_strategy_bytes: regret_bytes,
        river_eval_bytes,
        total_estimated_bytes: regret_bytes + bucket_map_bytes + river_eval_bytes + node_overhead,
    }
}

#[inline]
fn bucket_of(combo_to_bucket: &[u32], combo_idx: usize) -> usize {
    if combo_to_bucket.is_empty() {
        combo_idx
    } else {
        combo_to_bucket[combo_idx] as usize
    }
}

/// Computes current iteration strategy via positive Regret Matching (`R^+ / sum R^+`, uniform if all `<= 0`).
fn regret_match_buckets(regrets: &[f32], n_actions: usize, num_buckets: usize) -> Vec<f32> {
    let mut strat = vec![0.0f32; n_actions * num_buckets];
    let mut denom = vec![0.0f32; num_buckets];

    for a in 0..n_actions {
        let r_row = &regrets[a * num_buckets..(a + 1) * num_buckets];
        let s_row = &mut strat[a * num_buckets..(a + 1) * num_buckets];
        for b in 0..num_buckets {
            let pos = r_row[b].max(0.0);
            s_row[b] = pos;
            denom[b] += pos;
        }
    }

    let uniform = 1.0f32 / (n_actions as f32);
    for a in 0..n_actions {
        let s_row = &mut strat[a * num_buckets..(a + 1) * num_buckets];
        for b in 0..num_buckets {
            if denom[b] > 0.0 {
                s_row[b] /= denom[b];
            } else {
                s_row[b] = uniform;
            }
        }
    }
    strat
}

/// Normalizes `cum_strategy` across actions (`sum_a pi(a, b) == 1.0`), falling back to regret matching / uniform
/// at iteration 0 or unvisited buckets.
fn average_or_current_bucket_strategy(
    locked_strategy: Option<&[f32]>,
    cum_strategy: &[f32],
    regrets: &[f32],
    n_actions: usize,
    num_buckets: usize,
) -> Vec<f32> {
    if let Some(ls) = locked_strategy {
        return ls.to_vec();
    }
    if n_actions == 1 {
        return vec![1.0f32; num_buckets];
    }
    let mut denom = vec![0.0f32; num_buckets];
    for a in 0..n_actions {
        let row = &cum_strategy[a * num_buckets..(a + 1) * num_buckets];
        for b in 0..num_buckets {
            denom[b] += row[b];
        }
    }

    let fallback = regret_match_buckets(regrets, n_actions, num_buckets);
    let mut out = vec![0.0f32; n_actions * num_buckets];
    for a in 0..n_actions {
        let c_row = &cum_strategy[a * num_buckets..(a + 1) * num_buckets];
        let f_row = &fallback[a * num_buckets..(a + 1) * num_buckets];
        let o_row = &mut out[a * num_buckets..(a + 1) * num_buckets];
        for b in 0..num_buckets {
            if denom[b] > 0.0 {
                o_row[b] = c_row[b] / denom[b];
            } else {
                o_row[b] = f_row[b];
            }
        }
    }
    out
}

// ============================================================================
// Public Memory Estimation & Showdown / Fold Exact Verification Breakdown
// ============================================================================

/// Standalone memory estimator matching [`PloGame::with_options`] without allocating `ConcreteNode`s.
pub fn estimate_memory_for_config(
    board: Board,
    oop_range: &PloRange,
    ip_range: &PloRange,
    action_tree: &ActionTree,
    abstraction: &dyn HandAbstraction,
    enable_chance_isomorphism: bool,
) -> Result<MemoryReport, SolverError> {
    if board.street() != action_tree.config().initial_street {
        return Err(SolverError::StreetMismatch {
            board_street: board.street(),
            tree_street: action_tree.config().initial_street,
        });
    }
    let live_oop = oop_range.without_dead(board.mask());
    let live_ip = ip_range.without_dead(board.mask());
    if live_oop.is_empty() || live_ip.is_empty() {
        return Err(SolverError::Range(RangeError::EmptyRange));
    }
    let combos = [
        live_oop
            .entries()
            .iter()
            .copied()
            .filter(|c| c.weight > 0.0)
            .collect::<Vec<_>>(),
        live_ip
            .entries()
            .iter()
            .copied()
            .filter(|c| c.weight > 0.0)
            .collect::<Vec<_>>(),
    ];
    if combos[0].is_empty() || combos[1].is_empty() {
        return Err(SolverError::Range(RangeError::EmptyRange));
    }
    let combo_ids = [
        combos[0].iter().map(|c| c.combo_id).collect::<Vec<_>>(),
        combos[1].iter().map(|c| c.combo_id).collect::<Vec<_>>(),
    ];
    let oop_iso_vec: Vec<(ComboId, f32)> = combos[0].iter().map(|c| (c.combo_id, c.weight)).collect();
    let ip_iso_vec: Vec<(ComboId, f32)> = combos[1].iter().map(|c| (c.combo_id, c.weight)).collect();
    let mut iso_cache = std::collections::HashMap::new();
    let effective_iso = enable_chance_isomorphism && abstraction.is_suit_symmetric();
    Ok(estimate_tree_memory(
        &board,
        &combos,
        &combo_ids,
        action_tree,
        abstraction,
        effective_iso,
        &oop_iso_vec,
        &ip_iso_vec,
        &mut iso_cache,
    ))
}

/// Exact mass and chip-EV breakdown of a river showdown (`O(N + M)` 15-term inclusion-exclusion
/// or `O(N * M)` pairwise brute-force oracle).
///
/// Convention for `pot`, `paid_oop`, `paid_ip` (`pot` is total pot at showdown, including `paid_oop + paid_ip`;
/// e.g. if starting pot was `P_0 = pot - paid_oop - paid_ip` and players invested `paid_oop`, `paid_ip` postflop):
/// - Net chip EV for OOP on a win: `pot - paid_oop`
/// - Net chip EV for OOP on a tie: `0.5 * pot - paid_oop`
/// - Net chip EV for OOP on a loss: `-paid_oop`
/// - `ev_oop + ev_ip == pot - paid_oop - paid_ip == starting_pot` (or if measured in total pot share without subtracting `paid`, add `paid_oop` / `paid_ip`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShowdownMassBreakdown {
    pub ev_oop: f64,
    pub ev_ip: f64,
    pub win_mass: f64,
    pub tie_mass: f64,
    pub lose_mass: f64,
    pub compatible_mass: f64,
}

/// Computes `ShowdownMassBreakdown` using ONLY the solver's **15-term 4-card inclusion-exclusion** kernel
/// (never taking any pairwise `O(N * M)` shortcut, even for `1 vs 1` ranges!), verifying:
/// 1. `compatible_mass` via full 15-term IE (`query_disjoint`, including `+W_4[same_hand]`);
/// 2. `win_mass` via ascending `< strength` 15-term IE sweep;
/// 3. `lose_mass` via descending `> strength` 15-term IE sweep;
/// 4. `tie_mass` via ascending `<= strength` 15-term IE sweep (`query_disjoint` minus `win_mass`),
///    and cross-checked against `compatible_mass - win_mass - lose_mass`!
pub fn ie_showdown_ev(
    range_oop: &PloRange,
    range_ip: &PloRange,
    board: [Card; 5],
    pot: f64,
    paid_oop: f64,
    paid_ip: f64,
) -> ShowdownMassBreakdown {
    let rcache = RiverBoardCache::new(board);
    let bmask = rcache.mask();

    let combos_oop = range_oop.entries();
    let combos_ip = range_ip.entries();
    let cids_oop: Vec<ComboId> = combos_oop.iter().map(|c| c.combo_id).collect();
    let cids_ip: Vec<ComboId> = combos_ip.iter().map(|c| c.combo_id).collect();

    let subs_oop: Vec<ComboBlockerSubsets> = combos_oop
        .iter()
        .map(|c| ComboBlockerSubsets::build(c.hand, &cids_ip))
        .collect();
    let subs_ip: Vec<ComboBlockerSubsets> = combos_ip
        .iter()
        .map(|c| ComboBlockerSubsets::build(c.hand, &cids_oop))
        .collect();

    let ip_weights: Vec<f32> = combos_ip
        .iter()
        .map(|c| if c.mask.intersects(bmask) { 0.0 } else { c.weight })
        .collect();

    let mut sorted_oop: Vec<SortedStrength> = combos_oop
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.mask.intersects(bmask) && c.weight > 0.0)
        .map(|(i, c)| SortedStrength {
            strength: rcache.eval_hand_unchecked(c.hand).raw(),
            combo_idx: i as u32,
        })
        .collect();
    sorted_oop.sort_unstable_by_key(|x| (x.strength, x.combo_idx));

    let mut sorted_ip: Vec<SortedStrength> = combos_ip
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.mask.intersects(bmask) && c.weight > 0.0)
        .map(|(j, c)| SortedStrength {
            strength: rcache.eval_hand_unchecked(c.hand).raw(),
            combo_idx: j as u32,
        })
        .collect();
    sorted_ip.sort_unstable_by_key(|x| (x.strength, x.combo_idx));

    let mut scratch = InclusionExclusionScratch::new();

    // 1) Total compatible mass via 15-term IE
    scratch.clear();
    for &SortedStrength { combo_idx: j, .. } in &sorted_ip {
        let ju = j as usize;
        scratch.add_combo(&subs_ip[ju], ip_weights[ju] as f64);
    }
    let mut compatible_mass = 0.0f64;
    for &SortedStrength { combo_idx: i, .. } in &sorted_oop {
        let iu = i as usize;
        let disj = scratch.query_disjoint(&subs_oop[iu], &ip_weights);
        compatible_mass += (combos_oop[iu].weight as f64) * disj;
    }

    // 2) Strict win mass (`ip.strength < oop.strength`) via ascending 15-term IE
    scratch.clear();
    let mut win_per_oop = vec![0.0f64; combos_oop.len()];
    let mut p_idx = 0usize;
    for &SortedStrength {
        strength: s_me,
        combo_idx: i,
    } in &sorted_oop
    {
        while p_idx < sorted_ip.len() && sorted_ip[p_idx].strength < s_me {
            let ju = sorted_ip[p_idx].combo_idx as usize;
            scratch.add_combo(&subs_ip[ju], ip_weights[ju] as f64);
            p_idx += 1;
        }
        if scratch.total != 0.0 {
            win_per_oop[i as usize] = scratch.query_disjoint_strict_ineq(&subs_oop[i as usize]);
        }
    }

    // 3) Weak inequality (`ip.strength <= oop.strength`) via ascending 15-term IE to measure `tie_mass` directly!
    // Notice that when `ip.strength <= s_me`, an identical 4-card hand `same_opp_idx` (if present in IP range)
    // HAS strength `== s_me`, so it IS in `scratch` and we MUST use `query_disjoint` (with `+W_4[same_opp_idx]`)!
    scratch.clear();
    let mut le_per_oop = vec![0.0f64; combos_oop.len()];
    let mut le_idx = 0usize;
    for &SortedStrength {
        strength: s_me,
        combo_idx: i,
    } in &sorted_oop
    {
        while le_idx < sorted_ip.len() && sorted_ip[le_idx].strength <= s_me {
            let ju = sorted_ip[le_idx].combo_idx as usize;
            scratch.add_combo(&subs_ip[ju], ip_weights[ju] as f64);
            le_idx += 1;
        }
        if scratch.total != 0.0 {
            le_per_oop[i as usize] = scratch.query_disjoint(&subs_oop[i as usize], &ip_weights);
        }
    }

    // 4) Strict lose mass (`ip.strength > oop.strength`) via descending 15-term IE
    scratch.clear();
    let mut lose_per_oop = vec![0.0f64; combos_oop.len()];
    let mut q_idx = sorted_ip.len();
    for &SortedStrength {
        strength: s_me,
        combo_idx: i,
    } in sorted_oop.iter().rev()
    {
        while q_idx > 0 && sorted_ip[q_idx - 1].strength > s_me {
            q_idx -= 1;
            let ju = sorted_ip[q_idx].combo_idx as usize;
            scratch.add_combo(&subs_ip[ju], ip_weights[ju] as f64);
        }
        if scratch.total != 0.0 {
            lose_per_oop[i as usize] = scratch.query_disjoint_strict_ineq(&subs_oop[i as usize]);
        }
    }

    let mut win_mass = 0.0f64;
    let mut tie_mass = 0.0f64;
    let mut lose_mass = 0.0f64;
    for &SortedStrength { combo_idx: i, .. } in &sorted_oop {
        let iu = i as usize;
        let w0 = combos_oop[iu].weight as f64;
        win_mass += w0 * win_per_oop[iu];
        tie_mass += w0 * (le_per_oop[iu] - win_per_oop[iu]);
        lose_mass += w0 * lose_per_oop[iu];
    }

    if compatible_mass <= 0.0 {
        return ShowdownMassBreakdown {
            ev_oop: 0.0,
            ev_ip: 0.0,
            win_mass: 0.0,
            tie_mass: 0.0,
            lose_mass: 0.0,
            compatible_mass: 0.0,
        };
    }

    let oop_share = (win_mass * pot + tie_mass * (0.5 * pot)) / compatible_mass;
    let ip_share = (lose_mass * pot + tie_mass * (0.5 * pot)) / compatible_mass;
    ShowdownMassBreakdown {
        ev_oop: oop_share - paid_oop,
        ev_ip: ip_share - paid_ip,
        win_mass,
        tie_mass,
        lose_mass,
        compatible_mass,
    }
}

/// Computes exact fold terminal mass and EV using ONLY the 15-term 4-card inclusion-exclusion kernel
/// (`folded_player == 0` means OOP folded; `folded_player == 1` means IP folded).
pub fn ie_fold_ev(
    range_oop: &PloRange,
    range_ip: &PloRange,
    board: &Board,
    pot: f64,
    paid_oop: f64,
    paid_ip: f64,
    folded_player: usize,
) -> (f64, f64, f64) {
    let bmask = board.mask();
    let combos_oop = range_oop.entries();
    let combos_ip = range_ip.entries();
    let cids_ip: Vec<ComboId> = combos_ip.iter().map(|c| c.combo_id).collect();
    let cids_oop: Vec<ComboId> = combos_oop.iter().map(|c| c.combo_id).collect();

    let subs_oop: Vec<ComboBlockerSubsets> = combos_oop
        .iter()
        .map(|c| ComboBlockerSubsets::build(c.hand, &cids_ip))
        .collect();
    let subs_ip: Vec<ComboBlockerSubsets> = combos_ip
        .iter()
        .map(|c| ComboBlockerSubsets::build(c.hand, &cids_oop))
        .collect();

    let ip_weights: Vec<f32> = combos_ip
        .iter()
        .map(|c| if c.mask.intersects(bmask) { 0.0 } else { c.weight })
        .collect();

    let mut scratch = InclusionExclusionScratch::new();
    scratch.clear();
    for (j, c) in combos_ip.iter().enumerate() {
        if !c.mask.intersects(bmask) && c.weight > 0.0 {
            scratch.add_combo(&subs_ip[j], c.weight as f64);
        }
    }

    let mut compatible_mass = 0.0f64;
    for (i, c) in combos_oop.iter().enumerate() {
        if !c.mask.intersects(bmask) && c.weight > 0.0 {
            let disj = scratch.query_disjoint(&subs_oop[i], &ip_weights);
            compatible_mass += (c.weight as f64) * disj;
        }
    }

    if compatible_mass <= 0.0 {
        return (0.0, 0.0, 0.0);
    }

    if folded_player == PLAYER_OOP {
        (-paid_oop, pot - paid_ip, compatible_mass)
    } else {
        (pot - paid_oop, -paid_ip, compatible_mass)
    }
}

/// Independent `O(N * M)` pairwise reference oracle for river showdown using `plo_eval::slow_eval_plo_lex`
/// (zero inclusion-exclusion, zero lookup-table shortcuts).
pub fn brute_showdown_ev(
    range_oop: &PloRange,
    range_ip: &PloRange,
    board: [Card; 5],
    pot: f64,
    paid_oop: f64,
    paid_ip: f64,
) -> ShowdownMassBreakdown {
    let board_obj = Board::river(board).unwrap();
    let bmask = board_obj.mask();
    let mut win_mass = 0.0f64;
    let mut tie_mass = 0.0f64;
    let mut lose_mass = 0.0f64;
    let mut compatible_mass = 0.0f64;

    for c0 in range_oop.entries() {
        if c0.weight <= 0.0 || c0.mask.intersects(bmask) {
            continue;
        }
        let k0 = plo_eval::slow_eval_plo_lex(c0.hand, &board_obj)
            .unwrap()
            .0;
        for c1 in range_ip.entries() {
            if c1.weight <= 0.0 || c1.mask.intersects(bmask) || c0.mask.intersects(c1.mask) {
                continue;
            }
            let w = (c0.weight as f64) * (c1.weight as f64);
            compatible_mass += w;
            let k1 = plo_eval::slow_eval_plo_lex(c1.hand, &board_obj)
                .unwrap()
                .0;
            match k0.cmp(&k1) {
                core::cmp::Ordering::Greater => win_mass += w,
                core::cmp::Ordering::Equal => tie_mass += w,
                core::cmp::Ordering::Less => lose_mass += w,
            }
        }
    }

    if compatible_mass <= 0.0 {
        return ShowdownMassBreakdown {
            ev_oop: 0.0,
            ev_ip: 0.0,
            win_mass: 0.0,
            tie_mass: 0.0,
            lose_mass: 0.0,
            compatible_mass: 0.0,
        };
    }

    let oop_share = (win_mass * pot + tie_mass * (0.5 * pot)) / compatible_mass;
    let ip_share = (lose_mass * pot + tie_mass * (0.5 * pot)) / compatible_mass;
    ShowdownMassBreakdown {
        ev_oop: oop_share - paid_oop,
        ev_ip: ip_share - paid_ip,
        win_mass,
        tie_mass,
        lose_mass,
        compatible_mass,
    }
}

/// Independent `O(N * M)` pairwise reference oracle for fold terminal EV.
pub fn brute_fold_ev(
    range_oop: &PloRange,
    range_ip: &PloRange,
    board: &Board,
    pot: f64,
    paid_oop: f64,
    paid_ip: f64,
    folded_player: usize,
) -> (f64, f64, f64) {
    let bmask = board.mask();
    let mut compatible_mass = 0.0f64;
    for c0 in range_oop.entries() {
        if c0.weight <= 0.0 || c0.mask.intersects(bmask) {
            continue;
        }
        for c1 in range_ip.entries() {
            if c1.weight <= 0.0 || c1.mask.intersects(bmask) || c0.mask.intersects(c1.mask) {
                continue;
            }
            compatible_mass += (c0.weight as f64) * (c1.weight as f64);
        }
    }
    if compatible_mass <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    if folded_player == PLAYER_OOP {
        (-paid_oop, pot - paid_ip, compatible_mass)
    } else {
        (pot - paid_oop, -paid_ip, compatible_mass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plo_cards::Hand4;
    use plo_tree::{BettingLimit, StreetBetSizes};

    #[test]
    fn exact_4card_15term_inclusion_exclusion_matches_pairwise_bruteforce() {
        // Test on a dense overlapping set of 4-card hands where combos share 1, 2, 3, AND all 4 cards!
        let river: Board = "AsKsTd7h2c".parse().unwrap();
        let oop_range: PloRange = "AAQQ, KQJT, (QJ)(98), 7766, QhJh9s8s, QdJd9c8c".parse().unwrap();
        let ip_range: PloRange = "AAJJ, KQJT, (QT)(98), 7755, QhJh9s8s, QhJh9s6s".parse().unwrap();

        assert!(oop_range.without_dead(river.mask()).len() > 40);
        assert!(ip_range.without_dead(river.mask()).len() > 40);

        let cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 100,
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let tree = ActionTree::build(cfg).unwrap();
        let game = PloGame::new(river, &oop_range, &ip_range, &tree).unwrap();

        // Force both ranges > 512 product (`>40 * >40 > 1600 > 512`), so `eval_terminal_fold` and
        // `eval_terminal_showdown` execute the 15-term inclusion-exclusion kernel!
        assert!(game.combos(0).len() * game.combos(1).len() > 1500);

        // Give non-uniform pseudo-random reach weights to IP
        let opp_reach: Vec<f32> = (0..game.combos(1).len())
            .map(|j| ((j * 37 + 11) % 19) as f32 * 0.07)
            .collect();

        let mut scratch = InclusionExclusionScratch::new();
        let mut ie_fold = vec![0.0f32; game.combos(0).len()];
        game.eval_terminal_fold(
            &mut ie_fold,
            PLAYER_OOP,
            PLAYER_IP,
            100.0,
            river.mask(),
            &opp_reach,
            &mut scratch,
        );

        let mut ie_showdown = vec![0.0f32; game.combos(0).len()];
        game.eval_terminal_showdown(
            &mut ie_showdown,
            PLAYER_OOP,
            0,
            150.0,
            0.0,
            &opp_reach,
            &mut scratch,
        );

        // Compare element-by-element against exact O(N * M) pairwise brute-force!
        let rcards = river.river_cards().unwrap();
        let rcache = RiverBoardCache::new(rcards);
        let fold_payoff = 100.0 / game.valid_root_pair_weight;
        let sd_payoff = 150.0 / game.valid_root_pair_weight;

        for (i, c0) in game.combos(0).iter().enumerate() {
            let s0 = rcache.eval_hand_unchecked(c0.hand).raw();
            let mut bf_fold = 0.0f64;
            let mut bf_sd = 0.0f64;
            for (j, c1) in game.combos(1).iter().enumerate() {
                if c0.mask.intersects(c1.mask) {
                    continue;
                }
                let r = opp_reach[j] as f64;
                bf_fold += fold_payoff * r;
                let s1 = rcache.eval_hand_unchecked(c1.hand).raw();
                if s0 > s1 {
                    bf_sd += sd_payoff * r;
                } else if s0 < s1 {
                    bf_sd -= sd_payoff * r;
                }
            }
            assert!(
                (ie_fold[i] as f64 - bf_fold).abs() < 1e-5,
                "fold IE mismatch at {i}: {} vs {bf_fold}",
                ie_fold[i]
            );
            assert!(
                (ie_showdown[i] as f64 - bf_sd).abs() < 1e-5,
                "showdown IE mismatch at {i}: {} vs {bf_sd}",
                ie_showdown[i]
            );
        }
    }

    #[test]
    fn river_plo_polar_vs_bluffcatcher_nash_equilibrium() {
        // Classic polarized vs bluff-catcher river toy game in PLO!
        // Board: [Ks Qs Js 7d 2c] (3 spades on board).
        // - OOP Polarized Range (2 combos, disjoint from IP):
        //   1) Nut Flush: [As Ts 4h 3h] (unbeatable A-high spade flush)
        //   2) Missed Wrap Air: [9h 8h 6d 5d] (9-high — loses to IP's bluff-catcher; notice in Hold'em
        //      3 spades on board + nothing wouldn't matter, or we can even give OOP a PLO-specific air hand:
        //      [Ah 9s 6d 5d] — bare 9s CANNOT make a flush in PLO even with 3 spades on board + Ace in hand!
        //      Wait: [Ah 9s 6d 5d] + Ks Qs Js = Ace-high `A-K-Q-J-9`, NO flush and NO straight!)
        // - IP Bluff-Catcher Range (1 combo, completely disjoint cards!):
        //   1) Two Pair Kings & Queens: [Kd Qd 4c 3c] (beats OOP's [Ah 9s 6d 5d] air, loses to [As Ts 4h 3h] nut flush!)
        //
        // Game setup:
        // Starting pot = 100, Effective stack = 100, 100% Pot-only sizing (`Check` or `AllIn(100)`),
        // and IP is in position: if OOP checks, IP should NEVER bet with a pure medium bluff-catcher
        // against `{50% nuts, 50% air}` (checks back 100%).
        // Analytical Nash Equilibrium for Half-Street Polarized `{1 Nut, 1 Air}` vs `{1 Bluff-Catcher}`
        // with Pot-size bet `B = P = 100`:
        // - Pot odds for IP facing `Bet(100)` into `100` (calling `100` to win `200`, total `300`):
        //   IP needs `100 / 300 = 1/3` equity -> OOP bluffs `1/2` as often as value (`alpha = B / (P + B) = 100 / 200 = 0.5`)!
        //   So OOP bets Nut Flush `100%` (`1.0`), bluffs Air `50%` (`0.5`), checks Air `50%` (`0.5`)!
        // - IP calls `Bet(100)` with frequency `P / (P + B) = 100 / 200 = 50%` (`0.5`) to make OOP's bluff indifferent!
        // - Analytical EVs:
        //   - OOP Nut Flush EV: `0.5 * 100 (when IP folds) + 0.5 * 200 (when IP calls) = 150.0`!
        //   - OOP Air EV: `0.0` (checks and loses, or bluffs into 50% call: `0.5 * 100 + 0.5 * (-100) = 0.0`)!
        //   - Total OOP average EV: `0.5 * 150 + 0.5 * 0 = 75.0`!
        //   - Total IP EV: `100.0 - 75.0 = 25.0`!
        let river: Board = "KsQsJs7d2c".parse().unwrap();
        let nut_flush: Hand4 = "AsTs4h3h".parse().unwrap();
        let plo_air_bare_spade: Hand4 = "Ah9s6d5d".parse().unwrap();
        let bluff_catcher: Hand4 = "KdQd4c3c".parse().unwrap();

        let oop = PloRange::from_hands([(nut_flush, 1.0), (plo_air_bare_spade, 1.0)]).unwrap();
        let ip = PloRange::from_hands([(bluff_catcher, 1.0)]).unwrap();

        let cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 100,
            betting_limit: BettingLimit::PotLimit,
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let tree = ActionTree::build(cfg).unwrap();
        let mut game = PloGame::new(river, &oop, &ip, &tree).unwrap();

        let expl = game.solve(600, 0.001);
        assert!(expl < 0.05, "expected near-zero exploitability, got {expl}");

        let [ev_oop, ev_ip] = game.root_expected_values();
        assert!((ev_oop + ev_ip - 100.0).abs() < 1e-3);
        assert!((ev_oop - 75.0).abs() < 0.25, "OOP EV={ev_oop}, expected 75.0");
        assert!((ev_ip - 25.0).abs() < 0.25, "IP EV={ev_ip}, expected 25.0");

        let root_rep = game.inspect_current_node();
        assert_eq!(root_rep.actions, vec![Action::Check, Action::AllIn(100)]);

        // Strategy sums to 1.0 for every hand
        let n_oop = game.combos(0).len();
        assert_eq!(n_oop, 2);
        for h in 0..n_oop {
            let s_sum = root_rep.strategy[h] + root_rep.strategy[n_oop + h];
            assert!((s_sum - 1.0).abs() < 1e-5);
        }

        // Identify which index is nut_flush and which is plo_air_bare_spade
        let nut_idx = game
            .combos(0)
            .iter()
            .position(|c| c.hand == nut_flush)
            .unwrap();
        let air_idx = nut_idx ^ 1;

        let bet_nut = root_rep.strategy[n_oop + nut_idx];
        let bet_air = root_rep.strategy[n_oop + air_idx];
        assert!(bet_nut > 0.99, "nut flush must value-bet ~100%, got {bet_nut}");
        assert!(
            (bet_air - 0.5).abs() < 0.03,
            "air must bluff ~50% in pot-limit polar game, got {bet_air}"
        );
        assert!((root_rep.ev_oop[nut_idx] - 150.0).abs() < 0.5);
        assert!(root_rep.ev_oop[air_idx].abs() < 0.5);

        // Inspect IP node after OOP bets 100 (`action_idx = 1`)
        game.play_action(1).unwrap();
        let ip_rep = game.inspect_current_node();
        assert_eq!(ip_rep.actions, vec![Action::Fold, Action::Call]);
        let ip_call = ip_rep.strategy[1]; // action 1 = Call, 1 combo
        assert!(
            (ip_call - 0.5).abs() < 0.03,
            "IP bluff-catcher must call ~50%, got {ip_call}"
        );
    }

    #[test]
    fn turn_solver_with_chance_isomorphism_matches_unfolded_turn_solver() {
        // Verify that solving a Turn PLO game WITH suit isomorphism (`enable_chance_isomorphism = true`)
        // produces the EXACT SAME root EVs and exploitability convergence as WITHOUT suit isomorphism!
        let turn: Board = "AsKs7c2c".parse().unwrap(); // Spades & clubs on board; diamonds (1) & hearts (2) symmetric!
        // Suit-symmetric PLO subranges (`AAQQ$ds, (QJ)(T9)`)
        let oop: PloRange = "AAQQ$ds, (QJ)(T9)".parse().unwrap();
        let ip: PloRange = "KKJJ$ds, (QT)(98)".parse().unwrap();

        let cfg = TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 100,
            betting_limit: BettingLimit::PotLimit,
            turn_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let tree = ActionTree::build(cfg).unwrap();

        let mut game_iso = PloGame::with_options(
            turn,
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
        .unwrap();

        let mut game_full = PloGame::with_options(
            turn,
            &oop,
            &ip,
            &tree,
            &IdentityAbstraction,
            SolverBuildOptions {
                max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                enable_chance_isomorphism: false,
                ..Default::default()
            },
        )
        .unwrap();

        // Isomorphism folds 13 heart rivers into the 13 diamond rivers (`48 - 13 = 35` representative rivers < 48)!
        assert!(game_iso.memory_report().decision_nodes < game_full.memory_report().decision_nodes);

        let expl_iso = game_iso.solve(120, 0.0);
        let expl_full = game_full.solve(120, 0.0);

        assert!((expl_iso - expl_full).abs() < 1e-3, "{expl_iso} vs {expl_full}");
        let ev_iso = game_iso.root_expected_values();
        let ev_full = game_full.root_expected_values();
        assert!(
            (ev_iso[0] - ev_full[0]).abs() < 1e-2,
            "EV mismatch: {ev_iso:?} vs {ev_full:?}"
        );
        assert!((ev_iso[0] + ev_iso[1] - 100.0).abs() < 1e-2);
        assert!(expl_iso < 1.0, "turn exploitability converged to {expl_iso}");
    }

    #[test]
    fn flop_solver_runs_with_bucket_abstraction_and_enforces_memory_guardrail() {
        let flop: Board = "AcKc7c".parse().unwrap(); // Monotone flop -> 23 canonical turn cards!
        let oop: PloRange = "AAQQ$ds, (QJ)(T9)".parse().unwrap();
        let ip: PloRange = "KKJJ$ds, (QT)(98)".parse().unwrap();

        let cfg = TreeConfig {
            initial_street: Street::Flop,
            starting_pot: 100,
            effective_stack: 50, // 0.5 SPR short-stack flop spot for fast exact test
            betting_limit: BettingLimit::PotLimit,
            flop_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            turn_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let tree = ActionTree::build(cfg).unwrap();

        // 1) Verify memory budget guardrail cleanly rejects tiny 1 KB cap
        let too_small = PloGame::with_options(
            flop,
            &oop,
            &ip,
            &tree,
            &IdentityAbstraction,
            SolverBuildOptions {
                max_memory_bytes: 1024,
                enable_chance_isomorphism: true,
                ..Default::default()
            },
        );
        assert!(matches!(
            too_small,
            Err(SolverError::MemoryBudgetExceeded { .. })
        ));

        // 2) Build and run iterations on Flop using EquityBucketAbstraction(4, 4, 4)
        let bucket_abs = EquityBucketAbstraction::new(4, 4, 4);
        let mut flop_game = PloGame::with_options(
            flop,
            &oop,
            &ip,
            &tree,
            &bucket_abs,
            SolverBuildOptions::default(),
        )
        .unwrap();

        let expl0 = flop_game.exploitability();
        let expl1 = flop_game.solve(25, 0.0);
        assert!(
            expl1 < expl0,
            "flop DCFR must reduce exploitability: {expl0} -> {expl1}"
        );
        let [ev0, ev1] = flop_game.root_expected_values();
        assert!((ev0 + ev1 - 100.0).abs() < 0.05);
    }

    // ========================================================================
    // PHASE A1 & A2: Pairwise Brute-Force Oracles & 6 Explicit Overlap/Chop/Blocked Fixtures
    // ========================================================================

    fn assert_showdown_and_fold_ie_match_oracle(
        label: &str,
        oop: &PloRange,
        ip: &PloRange,
        river: Board,
        pot: f64,
        paid_oop: f64,
        paid_ip: f64,
    ) {
        let rc = river.river_cards().unwrap();
        let bf = brute_showdown_ev(oop, ip, rc, pot, paid_oop, paid_ip);
        let ie = ie_showdown_ev(oop, ip, rc, pot, paid_oop, paid_ip);

        let tol = 1e-9;
        assert!(
            (ie.compatible_mass - bf.compatible_mass).abs() < tol,
            "[{label}] compatible_mass mismatch: IE={} vs BF={}",
            ie.compatible_mass,
            bf.compatible_mass
        );
        assert!(
            (ie.win_mass - bf.win_mass).abs() < tol,
            "[{label}] win_mass mismatch: IE={} vs BF={}",
            ie.win_mass,
            bf.win_mass
        );
        assert!(
            (ie.tie_mass - bf.tie_mass).abs() < tol,
            "[{label}] tie_mass mismatch: IE={} vs BF={}",
            ie.tie_mass,
            bf.tie_mass
        );
        assert!(
            (ie.lose_mass - bf.lose_mass).abs() < tol,
            "[{label}] lose_mass mismatch: IE={} vs BF={}",
            ie.lose_mass,
            bf.lose_mass
        );
        assert!(
            ((ie.win_mass + ie.tie_mass + ie.lose_mass) - ie.compatible_mass).abs() < tol,
            "[{label}] IE mass partition error: win+tie+lose={} vs compat={}",
            ie.win_mass + ie.tie_mass + ie.lose_mass,
            ie.compatible_mass
        );
        assert!(
            (ie.ev_oop - bf.ev_oop).abs() < 1e-7 && (ie.ev_ip - bf.ev_ip).abs() < 1e-7,
            "[{label}] showdown EV mismatch: IE=({}, {}) vs BF=({}, {})",
            ie.ev_oop,
            ie.ev_ip,
            bf.ev_oop,
            bf.ev_ip
        );

        // Also check fold IE for both OOP fold and IP fold
        for fp in [PLAYER_OOP, PLAYER_IP] {
            let (bf_f0, bf_f1, bf_m) = brute_fold_ev(oop, ip, &river, pot, paid_oop, paid_ip, fp);
            let (ie_f0, ie_f1, ie_m) = ie_fold_ev(oop, ip, &river, pot, paid_oop, paid_ip, fp);
            assert!(
                (ie_m - bf_m).abs() < tol
                    && (ie_f0 - bf_f0).abs() < 1e-7
                    && (ie_f1 - bf_f1).abs() < 1e-7,
                "[{label}] fold(fp={fp}) mismatch: IE=({ie_f0},{ie_f1},{ie_m}) vs BF=({bf_f0},{bf_f1},{bf_m})"
            );
        }

        // Also check PloGame's internal per-combo `eval_terminal_showdown_ie_unconditional` & `eval_terminal_fold_ie_unconditional`
        // whenever compatible_mass > 0!
        if bf.compatible_mass > 0.0 {
            let starting_pot = (pot - paid_oop - paid_ip).round() as i32;
            if starting_pot > 0 && (paid_oop - paid_ip).abs() < 1e-9 {
                let committed = paid_oop.round() as i32;
                let cfg = TreeConfig {
                    initial_street: Street::River,
                    starting_pot,
                    effective_stack: committed.max(10),
                    river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
                    ..Default::default()
                };
                let tree = ActionTree::build(cfg).unwrap();
                let game = PloGame::new(river, oop, ip, &tree).unwrap();
                let half_swing = 0.5 * (starting_pot as f64) + (committed as f64);
                let win_payoff = half_swing / game.valid_root_pair_weight;
                let mut scratch = InclusionExclusionScratch::new();
                let mut out_sd = vec![0.0f32; game.combos(0).len()];
                game.eval_terminal_showdown_ie_unconditional(
                    &mut out_sd,
                    PLAYER_OOP,
                    0,
                    win_payoff,
                    -win_payoff,
                    0.0,
                    &game.initial_weights[1],
                    &mut scratch,
                );
                let centered_oop: f64 = out_sd
                    .iter()
                    .zip(&game.initial_weights[0])
                    .map(|(&v, &w)| (v as f64) * (w as f64))
                    .sum();
                let game_sd_ev_oop = 0.5 * (starting_pot as f64) + centered_oop;
                assert!(
                    (game_sd_ev_oop - bf.ev_oop).abs() < 1e-4,
                    "[{label}] PloGame unconditional IE showdown EV={game_sd_ev_oop} vs BF={}",
                    bf.ev_oop
                );
            }
        }
    }

    #[test]
    fn phase_a2_six_explicit_tie_overlap_blocked_fixtures_match_oracle() {
        let river: Board = "KsQsJs4d2c".parse().unwrap();

        // Fixture 1: Disjoint ranges, pure nuts vs air (100% win mass, 0 tie, 0 lose)
        let f1_oop: PloRange = "AsTs8h7h, As9s8h7h".parse().unwrap(); // Nut spade flush
        let f1_ip: PloRange = "KdQd6c5c, Td9d6c5c".parse().unwrap(); // Two pair / J-high straight
        assert_showdown_and_fold_ie_match_oracle("1_disjoint_nuts_vs_air", &f1_oop, &f1_ip, river, 300.0, 100.0, 100.0);
        let b1 = ie_showdown_ev(&f1_oop, &f1_ip, river.river_cards().unwrap(), 300.0, 100.0, 100.0);
        assert!((b1.win_mass - 4.0).abs() < 1e-9 && b1.tie_mass == 0.0 && b1.lose_mass == 0.0);
        assert!((b1.ev_oop - 200.0).abs() < 1e-9 && (b1.ev_ip - (-100.0)).abs() < 1e-9);

        // Fixture 2: Chop with identical hand strength (`A-K-Q-J-T` broadway straight, rainbow suits), NO card overlap!
        let f2_oop: PloRange = "AhTd8h7h".parse().unwrap();
        let f2_ip: PloRange = "AdTc6c5c".parse().unwrap();
        assert_showdown_and_fold_ie_match_oracle("2_chop_no_overlap", &f2_oop, &f2_ip, river, 300.0, 100.0, 100.0);
        let b2 = ie_showdown_ev(&f2_oop, &f2_ip, river.river_cards().unwrap(), 300.0, 100.0, 100.0);
        assert!(b2.win_mass == 0.0 && (b2.tie_mass - 1.0).abs() < 1e-9 && b2.lose_mass == 0.0);
        assert!((b2.ev_oop - 50.0).abs() < 1e-9 && (b2.ev_ip - 50.0).abs() < 1e-9);

        // Fixture 3: Chop with identical hand strength AND 1-card overlap on some pairs (`Ah` or `Td` shared)!
        // OOP has 2 broadway combos `{AhTd8h7h, AcTc8d7d}`, IP has `{AhTc6c5c (shares Ah with 1st, Tc with 2nd!), AdTd6h5h (shares Td with 1st, disjoint from 2nd!)}` -> only 1 compatible pair (`AcTc8d7d` vs `AdTd6h5h`)!
        let f3_oop: PloRange = "AhTd8h7h, AcTc8d7d".parse().unwrap();
        let f3_ip: PloRange = "AhTc6c5c, AdTd6h5h".parse().unwrap();
        assert_showdown_and_fold_ie_match_oracle("3_chop_1card_overlap", &f3_oop, &f3_ip, river, 300.0, 100.0, 100.0);
        let b3 = ie_showdown_ev(&f3_oop, &f3_ip, river.river_cards().unwrap(), 300.0, 100.0, 100.0);
        assert!((b3.compatible_mass - 1.0).abs() < 1e-9 && (b3.tie_mass - 1.0).abs() < 1e-9);

        // Fixture 4: 2-card, 3-card, AND 4-card identical-hand overlaps across identical-strength chop & non-chop pairs!
        // Both OOP and IP hold the EXACT same 4 combos `{AhTd8h7h, AhTd8c7c (2-card overlap), AhTd8h6c (3-card overlap), AdTc6h5h (disjoint from all 3!)}`!
        let f4_same: PloRange = "AhTd8h7h@0.8, AhTd8c7c@0.6, AhTd8h6c@0.4, AdTc6h5h@1.0".parse().unwrap();
        assert_showdown_and_fold_ie_match_oracle("4_2card_3card_4card_overlap_chop", &f4_same, &f4_same, river, 300.0, 100.0, 100.0);
        let b4 = ie_showdown_ev(&f4_same, &f4_same, river.river_cards().unwrap(), 300.0, 100.0, 100.0);
        // Only pairs between `{AhTd...}` and `AdTc6h5h` are disjoint (`2 * (0.8 + 0.6 + 0.4) * 1.0 = 3.6` tie mass, `0` win, `0` lose)!
        assert!(
            (b4.compatible_mass - 3.6).abs() < 1e-6 && (b4.tie_mass - 3.6).abs() < 1e-6,
            "b4={b4:?}"
        );
        assert!(b4.win_mass == 0.0 && b4.lose_mass == 0.0);

        // Fixture 5: Combo blocked by board (`Ks` and `2c` are on `KsQsJs4d2c`) mixed with live combos!
        let f5_oop: PloRange = "KsAhTd9d@1.0, AsTs8h7h@0.7, AhTd8c7c@0.5".parse().unwrap();
        let f5_ip: PloRange = "2cAdTc9c@1.0, AdTc6d5d@0.9, KdQd6h5h@0.4".parse().unwrap();
        assert_showdown_and_fold_ie_match_oracle("5_board_blocked_combos", &f5_oop, &f5_ip, river, 240.0, 70.0, 70.0);

        // Fixture 6: Dense mixed win + lose + chop + 1/2/3/4-card overlaps + fractional weights!
        let f6_oop: PloRange = "AsTs8h7h@0.9, As9s8c7c@0.4, AhTd8h7h@1.0, AdTc8c7c@0.65, KdQd6h5h@0.5, 9h8h7d6d@0.35, KsAh9d8d@1.0"
            .parse()
            .unwrap();
        let f6_ip: PloRange = "AsTs8h7h@0.7, As8s6h5h@0.55, AhTd8h7h@0.8, AhTd8h6c@0.45, AcTc6d5d@0.9, KdQd6c5c@0.6, 9c8c7h6h@0.25, QsAd9c8c@1.0"
            .parse()
            .unwrap();
        assert_showdown_and_fold_ie_match_oracle("6_mixed_win_lose_chop_blocked", &f6_oop, &f6_ip, river, 300.0, 100.0, 100.0);
        let b6 = ie_showdown_ev(&f6_oop, &f6_ip, river.river_cards().unwrap(), 300.0, 100.0, 100.0);
        assert!(b6.win_mass > 0.1 && b6.tie_mass > 0.1 && b6.lose_mass > 0.1);
    }

    // ========================================================================
    // PHASE B4, C2, C3, F, G1, G3: Iso vs No-Iso Per-Branch, Chance Card Removal,
    // Abstraction Invariants, Closed-Form Sanity, Memory Table, and Determinism
    // ========================================================================

    #[test]
    fn phase_b4_turn_iso_on_vs_iso_off_matches_ev_and_every_dealt_river_strategy() {
        // Turn board with symmetric diamonds (`d`) and hearts (`h`)
        let turn: Board = "AsKs7c2c".parse().unwrap();
        let oop: PloRange = "AAQQ$ds, (QJ)(T9)".parse().unwrap();
        let ip: PloRange = "KKJJ$ds, (QT)(98)".parse().unwrap();

        let cfg = TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 100,
            betting_limit: BettingLimit::PotLimit,
            turn_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let tree = ActionTree::build(cfg).unwrap();

        let mut g_iso = PloGame::with_options(
            turn,
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
        .unwrap();

        let mut g_no_iso = PloGame::with_options(
            turn,
            &oop,
            &ip,
            &tree,
            &IdentityAbstraction,
            SolverBuildOptions {
                max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                enable_chance_isomorphism: false,
                ..Default::default()
            },
        )
        .unwrap();

        let expl_iso = g_iso.solve(160, 0.0);
        let expl_no_iso = g_no_iso.solve(160, 0.0);
        assert!(
            (expl_iso - expl_no_iso).abs() < 1e-4,
            "exploitability diverged: iso={expl_iso} vs no_iso={expl_no_iso}"
        );

        let r_iso = g_iso.inspect_current_node();
        let r_no_iso = g_no_iso.inspect_current_node();
        for (a, b) in r_iso.strategy.iter().zip(&r_no_iso.strategy) {
            assert!((a - b).abs() < 1e-4, "turn root strategy mismatch: {a} vs {b}");
        }
        for (a, b) in r_iso.ev_oop.iter().zip(&r_no_iso.ev_oop) {
            assert!((a - b).abs() < 1e-3, "turn root EV mismatch: {a} vs {b}");
        }

        // Advance both games through Turn Check -> IP Check -> River Chance node!
        g_iso.play_action(0).unwrap();
        g_iso.play_action(0).unwrap();
        g_no_iso.play_action(0).unwrap();
        g_no_iso.play_action(0).unwrap();

        let all_48 = g_iso.all_legal_chance_cards();
        assert_eq!(all_48.len(), 48);
        assert_eq!(g_iso.available_chance_cards().len(), 35); // 13 hearts folded into 13 diamonds!

        // Check EVERY SINGLE ONE of the 48 river cards (including all 13 folded heart cards!)
        // to prove that `g_iso` after canonical remapping matches `g_no_iso` on both EV and strategy!
        for &rcard in &all_48 {
            g_iso.back_to_root();
            g_iso.play_action(0).unwrap();
            g_iso.play_action(0).unwrap();
            g_iso.play_chance_card(rcard).unwrap();

            g_no_iso.back_to_root();
            g_no_iso.play_action(0).unwrap();
            g_no_iso.play_action(0).unwrap();
            g_no_iso.play_chance_card(rcard).unwrap();

            let rep_i = g_iso.inspect_current_node();
            let rep_n = g_no_iso.inspect_current_node();
            assert_eq!(rep_i.board, rep_n.board);
            for (k, (&si, &sn)) in rep_i.strategy.iter().zip(&rep_n.strategy).enumerate() {
                assert!(
                    (si - sn).abs() < 1e-4,
                    "river {rcard} OOP strategy[{k}] mismatch: iso={si} vs no_iso={sn}"
                );
            }
            for (k, (&ei, &en)) in rep_i.ev_oop.iter().zip(&rep_n.ev_oop).enumerate() {
                assert!(
                    (ei - en).abs() < 1e-2,
                    "river {rcard} OOP EV[{k}] mismatch: iso={ei} vs no_iso={en}"
                );
            }
        }
    }

    #[test]
    fn phase_c2_chance_card_removal_zero_leakage_and_exact_checkdown_ev_vs_turn_fold() {
        // Crucial test proving the 1/40 (NOT 1/48!) PLO turn->river chance denominator AND zero card leakage:
        //1) On a pure Check-Down Turn->River tree (`effective_stack` doesn't matter when `bet_fractions = []` or 0 bets allowed),
        //   `PloGame::root_expected_values()` MUST equal `starting_pot * equity_range_vs_range(oop, ip, &turn)` to `< 1e-4`!
        //   (Note: if the chance denominator were `1/48` instead of `1/40`, centered EV would be shrunk by `40/48 = 5/6`, failing by >15%!)
        let turn: Board = "Ks9s5d2c".parse().unwrap();
        let oop: PloRange = "AsTs8h7h@1.0, AdKdQhJh@0.7, 9h9d6c4c@0.5".parse().unwrap();
        let ip: PloRange = "QsJs4h3h@0.9, KdQdTc8c@1.0, 6s6d4s3s@0.6".parse().unwrap();

        let checkdown_cfg = TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 100,
            max_bets_per_street: 0,
            turn_sizes: [
                StreetBetSizes {
                    bet_fractions: vec![],
                    raise_fractions: vec![],
                    donk_fractions: None,
                },
                StreetBetSizes {
                    bet_fractions: vec![],
                    raise_fractions: vec![],
                    donk_fractions: None,
                },
            ],
            river_sizes: [
                StreetBetSizes {
                    bet_fractions: vec![],
                    raise_fractions: vec![],
                    donk_fractions: None,
                },
                StreetBetSizes {
                    bet_fractions: vec![],
                    raise_fractions: vec![],
                    donk_fractions: None,
                },
            ],
            add_allin_spr_threshold: 0.0,
            ..Default::default()
        };
        // Note: `max_bets_per_street = 0` means unlimited in `TreeConfig`! Let's use `bet_fractions = []` + `add_allin_spr_threshold = 0.0` -> only `[Check]`!
        let check_tree = ActionTree::build(checkdown_cfg).unwrap();
        let game_cd = PloGame::with_options(
            turn,
            &oop,
            &ip,
            &check_tree,
            &IdentityAbstraction,
            SolverBuildOptions {
                max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                enable_chance_isomorphism: false,
                ..Default::default()
            },
        )
        .unwrap();

        let eq = plo_range::equity_range_vs_range(&oop, &ip, &turn).unwrap();
        let [ev0, ev1] = game_cd.root_expected_values_f64();
        let expected_ev0 = 100.0 * eq.equity_p0;
        let expected_ev1 = 100.0 * eq.equity_p1;
        assert!(
            (ev0 - expected_ev0).abs() <= 1e-9 && (ev1 - expected_ev1).abs() <= 1e-9,
            "turn checkdown EV=({ev0}, {ev1}) vs exact equity*pot=({expected_ev0}, {expected_ev1}) — verifies exact 1/40 PLO chance weight!"
        );

        // 2) Verify ZERO blocked-combo leakage after dealing a turn/river card `As`:
        // `AsTs8h7h` contains `As`! When `As` is dealt on the river, `AsTs8h7h` MUST have 0.0 EV and contribute 0.0 reach!
        let mut g_nav = game_cd;
        g_nav.play_action(0).unwrap(); // Turn OOP Check
        g_nav.play_action(0).unwrap(); // Turn IP Check
        let ace_spades: Card = "As".parse().unwrap();
        g_nav.play_chance_card(ace_spades).unwrap();
        let rep_after_as = g_nav.inspect_current_node();
        for (i, c) in g_nav.combos(0).iter().enumerate() {
            if c.mask.contains(ace_spades) {
                assert_eq!(
                    rep_after_as.ev_oop[i], 0.0,
                    "combo {} containing dealt river As leaked non-zero EV!",
                    c.hand
                );
            }
        }

        // 3) Also verify Flop -> Turn -> River 2-street check-down (`1/41 * 1/40 = 1/1640` ordered runout weight == `1/820` unordered)!
        let flop: Board = "Ks9s5d".parse().unwrap();
        let mut flop_cd_cfg = check_tree.config().clone();
        flop_cd_cfg.initial_street = Street::Flop;
        flop_cd_cfg.flop_sizes = [
            StreetBetSizes {
                bet_fractions: vec![],
                raise_fractions: vec![],
                donk_fractions: None,
            },
            StreetBetSizes {
                bet_fractions: vec![],
                raise_fractions: vec![],
                donk_fractions: None,
            },
        ];
        let flop_cd_tree = ActionTree::build(flop_cd_cfg).unwrap();
        let flop_cd_game = PloGame::with_options(
            flop,
            &oop,
            &ip,
            &flop_cd_tree,
            &IdentityAbstraction,
            SolverBuildOptions {
                max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                enable_chance_isomorphism: false,
                ..Default::default()
            },
        )
        .unwrap();
        let flop_eq = plo_range::equity_range_vs_range(&oop, &ip, &flop).unwrap();
        let [fev0, fev1] = flop_cd_game.root_expected_values_f64();
        assert!(
            (fev0 - 100.0 * flop_eq.equity_p0).abs() <= 1e-9
                && (fev1 - 100.0 * flop_eq.equity_p1).abs() <= 1e-9,
            "flop checkdown EV=({fev0}, {fev1}) vs exact flop equity*pot=({}, {}) — verifies exact 1/41 * 1/40 PLO chance weights!",
            100.0 * flop_eq.equity_p0,
            100.0 * flop_eq.equity_p1
        );
    }

    #[test]
    fn phase_c3_identity_and_equity_bucket_abstraction_preserve_combo_blockers_and_weights() {
        let river: Board = "KsQsJs4d2c".parse().unwrap();
        // OOP has 4 combos with fractional weights and card blockers overlapping IP; bucket them into 2 buckets!
        let oop: PloRange = "AsTs8h7h@0.9, As9s8c7c@0.4, KdQd6h5h@0.7, 9h8h7d6d@0.5"
            .parse()
            .unwrap();
        let ip: PloRange = "AsTs8h7h@0.8, KhQh6c5c@1.0, Td9d6s5s@0.6"
            .parse()
            .unwrap();

        // 1) IdentityAbstraction check-down terminal EV == brute_showdown_ev (`paid = 0`, `pot = 100`)
        let cd_cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 100,
            river_sizes: [
                StreetBetSizes {
                    bet_fractions: vec![],
                    raise_fractions: vec![],
                    donk_fractions: None,
                },
                StreetBetSizes {
                    bet_fractions: vec![],
                    raise_fractions: vec![],
                    donk_fractions: None,
                },
            ],
            add_allin_spr_threshold: 0.0,
            ..Default::default()
        };
        let cd_tree = ActionTree::build(cd_cfg).unwrap();
        let id_cd = PloGame::new(river, &oop, &ip, &cd_tree).unwrap();
        let bf = brute_showdown_ev(&oop, &ip, river.river_cards().unwrap(), 100.0, 0.0, 0.0);
        let [id_ev0, id_ev1] = id_cd.root_expected_values();
        assert!((id_ev0 as f64 - bf.ev_oop).abs() < 1e-4);
        assert!((id_ev1 as f64 - bf.ev_ip).abs() < 1e-4);

        // 2) EquityBucketAbstraction(2, 2, 2) on a real betting tree:
        //    - Combos in the same bucket MUST share identical action probabilities at decision nodes;
        //    - Showdown & blockers are STILL evaluated per-combo (`AsTs8h7h` blocks IP's `AsTs8h7h`, whereas `As9s8c7c` in the same nut-flush bucket does NOT block IP's `AsTs8h7h` — so their per-combo EVs differ cleanly even when sharing a bucket strategy!).
        let bet_cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 100,
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let bet_tree = ActionTree::build(bet_cfg).unwrap();
        let abs2 = EquityBucketAbstraction::new(2, 2, 2);
        let (_, map0) = abs2.assign_buckets(0, &river, id_cd.combos(0), id_cd.combos(1));

        let mut b_game = PloGame::with_options(
            river,
            &oop,
            &ip,
            &bet_tree,
            &abs2,
            SolverBuildOptions::default(),
        )
        .unwrap();
        let _ = b_game.solve(100, 0.0);
        let rep = b_game.inspect_current_node();
        let n0 = b_game.combos(0).len();
        for i in 0..n0 {
            for j in (i + 1)..n0 {
                if map0[i] == map0[j] {
                    for a in 0..rep.actions.len() {
                        assert!(
                            (rep.strategy[a * n0 + i] - rep.strategy[a * n0 + j]).abs() < 1e-6,
                            "combos {i} and {j} in bucket {} must share identical strategy!",
                            map0[i]
                        );
                    }
                }
            }
        }
        // Zero-sum conservation holds with exact post-removal weights (`ev0 + ev1 == 100.0`)
        let [bev0, bev1] = b_game.root_expected_values();
        assert!((bev0 + bev1 - 100.0).abs() < 1e-3);
    }

    #[test]
    fn phase_f_chop_heavy_river_tiny_turn_br_and_non_negative_exploitability() {
        // F2: Chop-heavy river game where almost all hands make the exact same Broadway straight (`A-K-Q-J-T`),
        // plus one nut flush combo for OOP:
        // - Neither player goes crazy jamming pure chop hands into a range with nut flushes;
        // - If both ranges are 100% identical Broadway straights (`AhTd...` vs `AdTc...`), EV is exact `50.0 / 50.0`!
        let river: Board = "KsQsJs4d2c".parse().unwrap();
        let chop_oop: PloRange = "AhTd8h7h, AcTc8d7d".parse().unwrap();
        let chop_ip: PloRange = "AdTd6h5h, AhTc6c5c".parse().unwrap();
        let cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 100,
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let tree = ActionTree::build(cfg).unwrap();
        let mut chop_game = PloGame::new(river, &chop_oop, &chop_ip, &tree).unwrap();
        let chop_expl = chop_game.solve(200, 0.001);
        assert!(chop_expl < 0.01, "chop game exploitability={chop_expl}");
        let [cev0, cev1] = chop_game.root_expected_values();
        assert!((cev0 - 50.0).abs() < 0.05 && (cev1 - 50.0).abs() < 0.05);

        // F3 & F4: Tiny turn game (`3 vs 3` combos, `1` bet size) solved to near-zero exploitability,
        // AND asserting `BR_p >= solver_EV_p` (non-negative exploitability invariant at EVERY checkpoint: 0, 1, 10, 50, 200 iters)!
        let turn: Board = "KsQs7d2c".parse().unwrap();
        let t_oop: PloRange = "AsTs8h6h, KdQd5h4h, 9h8h6c5c".parse().unwrap();
        let t_ip: PloRange = "JsTs4c3c, KhQh9d8d, 6d5d4s3s".parse().unwrap();
        let t_cfg = TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 100,
            turn_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let t_tree = ActionTree::build(t_cfg).unwrap();
        let mut t_game = PloGame::new(turn, &t_oop, &t_ip, &t_tree).unwrap();

        for step_target in [0u32, 1, 10, 50, 250] {
            while t_game.iterations_run() < step_target {
                t_game.solve_step();
            }
            let expl = t_game.exploitability();
            assert!(
                expl >= -1e-5,
                "negative exploitability impossible in exact zero-sum game, got {expl} at iter {}",
                t_game.iterations_run()
            );

            // Also explicitly verify `BR_OOP >= EV_OOP` and `BR_IP >= EV_IP`!
            let mut scratch = InclusionExclusionScratch::new();
            let half_start = 50.0f64;
            let [ev0, ev1] = t_game.root_expected_values();
            let br0: f64 = half_start
                + t_game
                    .best_response_traverse(0, 0, &t_game.initial_weights[1], &mut scratch)
                    .iter()
                    .zip(&t_game.initial_weights[0])
                    .map(|(&v, &w)| (v as f64) * (w as f64))
                    .sum::<f64>();
            let br1: f64 = half_start
                + t_game
                    .best_response_traverse(0, 1, &t_game.initial_weights[0], &mut scratch)
                    .iter()
                    .zip(&t_game.initial_weights[1])
                    .map(|(&v, &w)| (v as f64) * (w as f64))
                    .sum::<f64>();
            assert!(br0 + 1e-4 >= ev0 as f64, "BR0={br0} < EV0={ev0}");
            assert!(br1 + 1e-4 >= ev1 as f64, "BR1={br1} < EV1={ev1}");
        }
        assert!(
            t_game.exploitability() < 0.25,
            "tiny turn solver must converge within tight BR bound, got {}",
            t_game.exploitability()
        );
    }

    #[test]
    fn phase_g1_g3_concrete_memory_table_guardrail_and_determinism() {
        // G1: Verify all 5 required memory scenarios match documented order-of-magnitude budgets in HARDENING.md!
        let two_sizes = StreetBetSizes {
            bet_fractions: vec![0.5, 1.0],
            raise_fractions: vec![1.0],
            donk_fractions: None,
        };

        // Helper to pick first `k` live combos from `*` on `board`
        let first_k_range = |board: Board, k: usize| -> PloRange {
            let mut v = Vec::with_capacity(k);
            let bmask = board.mask();
            for (_, h) in Hand4::all_hands() {
                if !h.is_blocked_by(bmask) {
                    v.push((h, 1.0f32));
                    if v.len() == k {
                        break;
                    }
                }
            }
            PloRange::from_hands(v).unwrap()
        };

        // Row 1: River 20 vs 20, 2 bet sizes (`8` decision nodes, `3,200` regret+strat bytes, `7,712` total bytes)
        let r_board: Board = "Ks9d4c7h2s".parse().unwrap();
        let r_tree = ActionTree::build(TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 100,
            river_sizes: [two_sizes.clone(), two_sizes.clone()],
            ..Default::default()
        })
        .unwrap();
        let r20 = first_k_range(r_board, 20);
        let m_r20 = estimate_memory_for_config(r_board, &r20, &r20, &r_tree, &IdentityAbstraction, false).unwrap();
        assert_eq!(m_r20.decision_nodes, 8);
        assert_eq!(m_r20.regret_and_strategy_bytes, 3_200);
        assert_eq!(m_r20.total_estimated_bytes, 7_712);

        // Row 2: River 200 vs 200, 2 bet sizes (`8` decision nodes, `32,000` regret+strat bytes, `40,832` total bytes)
        let r200 = first_k_range(r_board, 200);
        let m_r200 = estimate_memory_for_config(r_board, &r200, &r200, &r_tree, &IdentityAbstraction, false).unwrap();
        assert_eq!(m_r200.decision_nodes, 8);
        assert_eq!(m_r200.regret_and_strategy_bytes, 32_000);
        assert_eq!(m_r200.total_estimated_bytes, 40_832);

        // Row 3: Turn 50 vs 50, 2 bet sizes, lossless (`776` decision nodes, `699,200` regret+strat bytes, `1,157,120` total bytes)
        let t_board: Board = "Ks9d4c7h".parse().unwrap();
        let t_tree = ActionTree::build(TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 100,
            turn_sizes: [two_sizes.clone(), two_sizes.clone()],
            river_sizes: [two_sizes.clone(), two_sizes.clone()],
            ..Default::default()
        })
        .unwrap();
        let t50 = first_k_range(t_board, 50);
        let m_t50 = estimate_memory_for_config(t_board, &t50, &t50, &t_tree, &IdentityAbstraction, false).unwrap();
        assert_eq!(m_t50.decision_nodes, 776);
        assert_eq!(m_t50.regret_and_strategy_bytes, 699_200);
        assert_eq!(m_t50.total_estimated_bytes, 1_157_120);

        // Row 4: Flop 500 vs 500, 2 bet sizes, IdentityAbstraction (`57,240` decision nodes, `496,352,000` regret+strat, `542,031,872` total bytes > `512 MB` default cap!)
        let f_board: Board = "Ks9d4c".parse().unwrap();
        let f_tree = ActionTree::build(TreeConfig {
            initial_street: Street::Flop,
            starting_pot: 100,
            effective_stack: 100,
            flop_sizes: [two_sizes.clone(), two_sizes.clone()],
            turn_sizes: [two_sizes.clone(), two_sizes.clone()],
            river_sizes: [two_sizes.clone(), two_sizes.clone()],
            ..Default::default()
        })
        .unwrap();
        let f500 = first_k_range(f_board, 500);
        let m_f500_id = estimate_memory_for_config(f_board, &f500, &f500, &f_tree, &IdentityAbstraction, false).unwrap();
        assert_eq!(m_f500_id.decision_nodes, 57_240);
        assert_eq!(m_f500_id.regret_and_strategy_bytes, 496_352_000);
        assert_eq!(m_f500_id.total_estimated_bytes, 542_031_872);

        // Verify `max_memory_bytes` guardrail (`500,000,000` bytes < `542,031,872` bytes, or `600 vs 600` under default `512 MB`)
        // cleanly refuses oversized Flop `IdentityAbstraction` without panicking!
        let f600 = first_k_range(f_board, 600);
        let refused = PloGame::with_options(
            f_board,
            &f600,
            &f600,
            &f_tree,
            &IdentityAbstraction,
            SolverBuildOptions {
                max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                enable_chance_isomorphism: false,
                ..Default::default()
            },
        );
        assert!(matches!(refused, Err(SolverError::MemoryBudgetExceeded { .. })));

        // Row 5: Flop 50 buckets (`EquityBucketAbstraction(50, 50, 50)`) over `500 vs 500` combos (`iso = false`):
        // Cuts `regret_and_strategy_bytes` EXACTLY 10x (`49,635,200` bytes = `47.34 MiB`), plus `1 + 49 + 1,176 = 1,226`
        // deduplicated `(board, player)` `Arc<[u32]>` bucket maps (`4,904,000` bytes) -> `100,219,072` total (`95.58 MiB` < `512 MiB` cap)!
        let abs50 = EquityBucketAbstraction::new(50, 50, 50);
        let m_f500_b50 = estimate_memory_for_config(f_board, &f500, &f500, &f_tree, &abs50, false).unwrap();
        assert_eq!(m_f500_b50.regret_and_strategy_bytes, 49_635_200);
        assert_eq!(m_f500_b50.total_estimated_bytes, 100_219_072);
        assert!(m_f500_b50.total_estimated_bytes < DEFAULT_MAX_MEMORY_BYTES);

        // Step 3: Explicit Iso OFF (`iso = false`) vs Iso ON (`iso = true`) Memory Table on Suit-Symmetric Ranges!
        // Construct deterministic suit-symmetric `50 vs 50` and `500 vs 500` ranges on rainbow flop `Ks9d4c`:
        // Note: `first_k_range` takes a lexicographic prefix of `ComboId`, which breaks suit symmetry, whereas real PLO
        // preflop ranges (e.g., `"AAKK, QQJJ"` or `"AABB$ds"`) are suit-symmetric across unrepresented suits `{h, s}` (and `{c, d}` where unblocked).
        let build_suit_symmetric_k_range = |board: Board, target_k: usize| -> PloRange {
            // Group all live combos into suit-orbits under permutations that fix the board (`Ks9d4c` fixes `{s, d, c}` and swaps? Wait:
            // on rainbow `Ks9d4c`, `h` is the ONLY unrepresented suit! A single suit cannot be swapped with itself!
            //Wait: on a TWO-TONE flop `Ks9s4d`, `{h, c}` are BOTH unrepresented on the board, so swapping `h <-> c` fixes `Ks9s4d`!
            // And on a MONOTONE flop `Ks9s4s`, `{h, d, c}` are ALL THREE unrepresented (`3! = 6` suit symmetries)!
            let bmask = board.mask();
            let mut used = vec![false; plo_cards::NUM_PLO_COMBOS];
            let mut chosen: Vec<(Hand4, f32)> = Vec::with_capacity(target_k);

            // Generate all 24 suit permutations and keep those that map `board` to the exact same card set!
            let all_perms: [[u8; 4]; 24] = [
                [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
                [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
                [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
                [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
            ];
            let valid_perms: Vec<[u8; 4]> = all_perms
                .into_iter()
                .filter(|&perm| {
                    board
                        .as_slice()
                        .iter()
                        .all(|&c| bmask.contains(permute_card(c, perm)))
                })
                .collect();

            // Collect orbits sorted by size ascending so we can hit exact `target_k`!
            let mut orbits: Vec<Vec<Hand4>> = Vec::new();
            for (cid, h) in Hand4::all_hands() {
                if h.is_blocked_by(bmask) || used[cid.index()] {
                    continue;
                }
                let mut orb = Vec::new();
                for &perm in &valid_perms {
                    let mh = permute_hand4(h, perm);
                    let mcid = mh.combo_id().index();
                    if !used[mcid] {
                        used[mcid] = true;
                        orb.push(mh);
                    }
                }
                orbits.push(orb);
            }
            // Pick size-2 orbits (or size-1 fixed combos) to reach exact `target_k` while preserving 100% suit symmetry!
            for orb in &orbits {
                if chosen.len() + orb.len() <= target_k && orb.len() > 1 {
                    for &mh in orb {
                        chosen.push((mh, 1.0));
                    }
                }
                if chosen.len() == target_k {
                    break;
                }
            }
            for orb in &orbits {
                if chosen.len() == target_k {
                    break;
                }
                if orb.len() == 1 {
                    chosen.push((orb[0], 1.0));
                }
            }
            assert_eq!(chosen.len(), target_k);
            PloRange::from_hands(chosen).unwrap()
        };

        // Note why rainbow flop `Ks9d4c` has 3 distinct ranks in 3 distinct suits `{s, d, c}`:
        // only 1 suit (`h`) is absent on the flop, so Turn chance has 0 flop-level suit swaps (`49` turn reps),
        // EXCEPT when a turn card pairs a board suit or leaves two suits unrepresented? No, once 3 suits `{s, d, c}` have
        // distinct ranks (`K`, `9`, `4`), NO suit permutation can swap any two suits!
        // On a TWO-TONE flop `Ks9s4d` (the most common PLO flop texture ~55% of flops), suits `{h, c}` are isomorphic (`h <-> c` swap),
        // folding 13 club turns into 13 heart turns (`36` turn reps instead of `49`, and `1,584` river subtrees instead of `2,352`!).
        let f_twotone: Board = "Ks9s4d".parse().unwrap();
        let sym50 = build_suit_symmetric_k_range(f_twotone, 50);
        let sym500 = build_suit_symmetric_k_range(f_twotone, 500);

        // Flop 50 vs 50 (`IdentityAbstraction`, 2 bet sizes) on two-tone `Ks9s4d`: `iso = false` vs `iso = true`
        let m_f50_noiso =
            estimate_memory_for_config(f_twotone, &sym50, &sym50, &f_tree, &IdentityAbstraction, false)
                .unwrap();
        assert_eq!(m_f50_noiso.decision_nodes, 57_240);
        assert_eq!(m_f50_noiso.chance_nodes, 840);
        assert_eq!(m_f50_noiso.terminal_nodes, 106_336);
        assert_eq!(m_f50_noiso.river_eval_tables, 1_176);
        assert_eq!(m_f50_noiso.regret_and_strategy_bytes, 49_635_200);
        assert_eq!(m_f50_noiso.river_eval_bytes, 1_411_200);
        assert_eq!(m_f50_noiso.total_estimated_bytes, 82_614_272);

        let m_f50_iso =
            estimate_memory_for_config(f_twotone, &sym50, &sym50, &f_tree, &IdentityAbstraction, true)
                .unwrap();
        assert_eq!(m_f50_iso.decision_nodes, 34_880);
        assert_eq!(m_f50_iso.chance_nodes, 619);
        assert_eq!(m_f50_iso.terminal_nodes, 64_671);
        assert_eq!(m_f50_iso.river_eval_tables, 799);
        assert_eq!(m_f50_iso.regret_and_strategy_bytes, 30_249_600);
        assert_eq!(m_f50_iso.river_eval_bytes, 958_800);
        assert_eq!(m_f50_iso.total_estimated_bytes, 50_441_040);

        // Also build the ACTUAL `PloGame` for `Flop 50 vs 50` (`iso = true`) to prove `estimate_memory_for_config`
        // matches `game.memory_report()` node-for-node and byte-for-byte on decision/chance/terminal/table counts & regret/strategy bytes!
        let g_f50_iso = PloGame::with_options(
            f_twotone,
            &sym50,
            &sym50,
            &f_tree,
            &IdentityAbstraction,
            SolverBuildOptions {
                max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                enable_chance_isomorphism: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(m_f50_iso.decision_nodes, g_f50_iso.memory_report().decision_nodes);
        assert_eq!(m_f50_iso.chance_nodes, g_f50_iso.memory_report().chance_nodes);
        assert_eq!(m_f50_iso.terminal_nodes, g_f50_iso.memory_report().terminal_nodes);
        assert_eq!(m_f50_iso.river_eval_tables, g_f50_iso.memory_report().river_eval_tables);
        assert_eq!(
            m_f50_iso.regret_and_strategy_bytes,
            g_f50_iso.memory_report().regret_and_strategy_bytes
        );

        // Flop 500 vs 500 (`IdentityAbstraction`, 2 bet sizes) on two-tone `Ks9s4d`: `iso = false` vs `iso = true`
        let m_f500_twotone_noiso =
            estimate_memory_for_config(f_twotone, &sym500, &sym500, &f_tree, &IdentityAbstraction, false)
                .unwrap();
        assert_eq!(m_f500_twotone_noiso.decision_nodes, 57_240);
        assert_eq!(m_f500_twotone_noiso.regret_and_strategy_bytes, 496_352_000);
        assert_eq!(m_f500_twotone_noiso.total_estimated_bytes, 542_031_872);

        let m_f500_twotone_iso =
            estimate_memory_for_config(f_twotone, &sym500, &sym500, &f_tree, &IdentityAbstraction, true)
                .unwrap();
        assert_eq!(m_f500_twotone_iso.decision_nodes, 34_880);
        assert_eq!(m_f500_twotone_iso.chance_nodes, 619);
        assert_eq!(m_f500_twotone_iso.terminal_nodes, 64_671);
        assert_eq!(m_f500_twotone_iso.river_eval_tables, 799);
        assert_eq!(m_f500_twotone_iso.regret_and_strategy_bytes, 302_496_000);
        assert_eq!(m_f500_twotone_iso.river_eval_bytes, 9_588_000);
        assert_eq!(m_f500_twotone_iso.total_estimated_bytes, 331_316_640);

        // Also verify `max_memory_bytes` guard triggers cleanly under BOTH `iso = false` (`600 vs 600` > `512 MiB`)
        // AND `iso = true` (`900 vs 900` > `512 MiB` = `536,870,912 B`)!
        let sym900 = build_suit_symmetric_k_range(f_twotone, 900);
        let refused_iso = PloGame::with_options(
            f_twotone,
            &sym900,
            &sym900,
            &f_tree,
            &IdentityAbstraction,
            SolverBuildOptions {
                max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                enable_chance_isomorphism: true,
                ..Default::default()
            },
        );
        assert!(matches!(refused_iso, Err(SolverError::MemoryBudgetExceeded { .. })));

        // G3: Native determinism: running identical inputs & iterations twice produces bitwise/1e-9 identical strategies & EVs!
        let oop_d: PloRange = "AAQQ$ds, (QJ)(T9)".parse().unwrap();
        let ip_d: PloRange = "KKJJ$ds, (QT)(98)".parse().unwrap();
        let mut run1 = PloGame::new(r_board, &oop_d, &ip_d, &r_tree).unwrap();
        let mut run2 = PloGame::new(r_board, &oop_d, &ip_d, &r_tree).unwrap();
        let e1 = run1.solve(150, 0.0);
        let e2 = run2.solve(150, 0.0);
        assert!((e1 - e2).abs() < 1e-9);
        let rep1 = run1.inspect_current_node();
        let rep2 = run2.inspect_current_node();
        for (&s1, &s2) in rep1.strategy.iter().zip(&rep2.strategy) {
            assert!((s1 - s2).abs() < 1e-9);
        }
        for (&v1, &v2) in rep1.ev_oop.iter().zip(&rep2.ev_oop) {
            assert!((v1 - v2).abs() < 1e-9);
        }
    }

    // ========================================================================
    // PHASE H: 2D PLO Nut/Blocker Abstraction Separation & Solver Test
    // ========================================================================

    #[test]
    fn phase_h_nut_blocker_2d_abstraction_separates_bare_nut_flush_blocker_with_k_eq_2() {
        // Classic PLO blocker separation test (`k = 2` buckets via `(1, 2)`):
        // 1) On a 3-spade river `[Ks Qs Js 4d 2c]`:
        //    - `h_blocker1 = [As 9h 8d 7c]` (bare `As` — NO spade flush in PLO, 0% showdown equity vs IP's two-pair `[Kd Qd 6c 5c]`, YET blocks Villain's nut spade flushes!)
        //    - `h_blocker2 = [As 9d 8h 7c]` (also bare `As` blocker!)
        //    - `h_noblk1   = [Ah 9c 8d 7h]` (NO spade blocker — exact same 0% equity vs `[Kd Qd 6c 5c]`)
        //    - `h_noblk2   = [Ad 9h 8c 7d]` (NO spade blocker — exact same 0% equity vs `[Kd Qd 6c 5c]`)
        //    Notice `EquityBucketAbstraction` sees 0.0 equity for ALL FOUR hands (grouping by arbitrary index order),
        //    whereas `NutBlockerBucketAbstraction` with `k = 2` (`(1, 2)`) deterministically puts BOTH `As` blocker hands
        //    into bucket `1` and BOTH non-blocker hands into bucket `0`!
        let river: Board = "KsQsJs4d2c".parse().unwrap();
        let oop_r: PloRange = "As9h8d7c, As9d8h7c, Ah9c8d7h, Ad9h8c7d".parse().unwrap();
        let ip_r: PloRange = "KdQd6c5c, AsTs6h5h".parse().unwrap();

        let abs_2d = NutBlockerBucketAbstraction::new((1, 2), (1, 2), (1, 2));
        let (nb_r, map_r) = abs_2d.assign_buckets(0, &river, oop_r.entries(), ip_r.entries());
        assert_eq!(nb_r, 2);

        let ace_spades: Card = "As".parse().unwrap();
        for (i, c) in oop_r.entries().iter().enumerate() {
            let expected_bucket = if c.mask.contains(ace_spades) { 1 } else { 0 };
            assert_eq!(
                map_r[i], expected_bucket,
                "river hand {} (has As={}) assigned to bucket {}, expected {expected_bucket}",
                c.hand,
                c.mask.contains(ace_spades),
                map_r[i]
            );
        }

        // 2) Also verify on a Two-Tone Turn `[Ks Qs 7d 2c]` (`k = 2` via `(1, 2)`):
        //    Four missed air hands with identical rank structure and no turn pair,
        //    two holding bare `As` nut-flush blocker (`As 9h 8h 4c`, `As 9c 8c 4h`) and two without `As` (`Ah 9h 8d 4c`, `Ac 9c 8h 4d`)!
        let turn: Board = "KsQs7d2c".parse().unwrap();
        let oop_t: PloRange = "As9h8h4c, As9c8c4h, Ah9h8d4c, Ac9c8h4d".parse().unwrap();
        let (nb_t, map_t) = abs_2d.assign_buckets(0, &turn, oop_t.entries(), ip_r.entries());
        assert_eq!(nb_t, 2);
        for (i, c) in oop_t.entries().iter().enumerate() {
            let expected_bucket = if c.mask.contains(ace_spades) { 1 } else { 0 };
            assert_eq!(
                map_t[i], expected_bucket,
                "turn hand {} (has As={}) assigned to bucket {}, expected {expected_bucket}",
                c.hand,
                c.mask.contains(ace_spades),
                map_t[i]
            );
        }

        // 3) Verify `PloGame` solves cleanly with `NutBlockerBucketAbstraction` and OOP bluffs the `As` nut-flush-blocker bucket
        //    strictly MORE frequently than the non-blocker air bucket against `[Kd Qd 6c 5c, As Ts 6h 5h]`!
        //    Why? Because when OOP holds `As`, IP CANNOT hold `As Ts 6h 5h` (the nut flush that snap-calls!), so IP ONLY holds `[Kd Qd 6c 5c]`!
        let oop_game_range: PloRange =
            "AsTs3h3d, As9h8d7c, As9d8h7c, Ah9c8d7h, Ad9h8c7d".parse().unwrap();
        let cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 100,
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let tree = ActionTree::build(cfg).unwrap();
        // Use `(2, 2)` = up to 4 buckets so nut flush (`AsTs3h3d`) gets high-equity bucket and the 4 air hands split into `{no As}` vs `{bare As}`!
        let abs_game = NutBlockerBucketAbstraction::new((2, 2), (2, 2), (2, 2));
        let mut game = PloGame::with_options(
            river,
            &oop_game_range,
            &ip_r,
            &tree,
            &abs_game,
            SolverBuildOptions::default(),
        )
        .unwrap();
        let _ = game.solve(400, 0.0);
        let rep = game.inspect_current_node();
        let n0 = game.combos(0).len();
        let blk_idx = game
            .combos(0)
            .iter()
            .position(|c| c.hand == "As9h8d7c".parse::<Hand4>().unwrap())
            .unwrap();
        let noblk_idx = game
            .combos(0)
            .iter()
            .position(|c| c.hand == "Ah9c8d7h".parse::<Hand4>().unwrap())
            .unwrap();
        let bet_blk = rep.strategy[n0 + blk_idx];
        let bet_noblk = rep.strategy[n0 + noblk_idx];
        assert!(
            bet_blk > bet_noblk + 0.25,
            "PLO solver with 2D NutBlockerBucketAbstraction must bluff bare As nut-flush blocker ({bet_blk}) far more than unblocked air ({bet_noblk})!"
        );
    }

    // ========================================================================
    // STEP 1 (BLOCKING): Exact `sum_c P(c) == 1.0` (±1e-12) & Tight Check-Down
    // EV vs `equity_range_vs_range` (`<= 1e-9` relative / `<= 1e-8` abs on f64)
    // ========================================================================

    fn build_checkdown_tree(street: Street, starting_pot: i32) -> ActionTree {
        let empty_sizes = StreetBetSizes {
            bet_fractions: vec![],
            raise_fractions: vec![],
            donk_fractions: None,
        };
        ActionTree::build(TreeConfig {
            initial_street: street,
            starting_pot,
            effective_stack: starting_pot,
            flop_sizes: [empty_sizes.clone(), empty_sizes.clone()],
            turn_sizes: [empty_sizes.clone(), empty_sizes.clone()],
            river_sizes: [empty_sizes.clone(), empty_sizes],
            add_allin_spr_threshold: 0.0,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn step1_chance_successor_probabilities_sum_to_one_with_and_without_isomorphism() {
        // Verify `sum_c P(c) == 1.0` (±1e-12) for EVERY compatible pair `(h0, h1)` across:
        // 1) Monotone flop (`AcKc7c`, unequal suit classes: `1` for clubs, `3` for `d/h/s`)
        // 2) Two-tone flop (`KsQs4d`, unequal suit classes: `1` for spades, `1` for diamonds, `2` for `c/h`)
        // 3) Rainbow flop (`Ks9d4c`, all suit classes size `1`)
        // 4) Two-tone turn (`AsKs7c2c`, unequal suit classes: `1` for `s`, `1` for `c`, `2` for `d/h`)
        let boards: [&str; 4] = ["AcKc7c", "KsQs4d", "Ks9d4c", "AsKs7c2c"];
        let oop_sym: PloRange = "AAQQ$ds, (QJ)(T9)".parse().unwrap();
        let ip_sym: PloRange = "KKJJ$ds, (QT)(98)".parse().unwrap();

        for b_str in boards {
            let board: Board = b_str.parse().unwrap();
            let tree = build_checkdown_tree(board.street(), 100);

            for enable_iso in [false, true] {
                let game = PloGame::with_options(
                    board,
                    &oop_sym,
                    &ip_sym,
                    &tree,
                    &IdentityAbstraction,
                    SolverBuildOptions {
                        max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                        enable_chance_isomorphism: enable_iso,
                        ..Default::default()
                    },
                )
                .unwrap();

                // Locate the first `ConcreteNodeKind::Chance` node in `game.nodes`
                let chance_node = game
                    .nodes
                    .iter()
                    .find(|n| matches!(n.kind, ConcreteNodeKind::Chance { .. }))
                    .expect("flop/turn check-down tree has chance node");

                let ConcreteNodeKind::Chance {
                    rep_cards,
                    iso_duplicates,
                    chance_denominator,
                    ..
                } = &chance_node.kind
                else {
                    unreachable!();
                };

                let expected_denom = 52 - board.len() - HOLE_CARDS_BOTH_PLAYERS;
                assert_eq!(*chance_denominator, expected_denom);
                let inv_denom = 1.0f64 / (*chance_denominator as f64);

                // Count suit-class sizes `|suit_class|` for each `rep_idx` (`1` + number of duplicates pointing to `rep_idx`)
                let mut class_sizes = vec![1usize; rep_cards.len()];
                for d in iso_duplicates {
                    class_sizes[d.rep_branch_idx] += 1;
                }
                assert_eq!(class_sizes.iter().sum::<usize>(), 52 - board.len());
                if b_str == "KsQs4d" && enable_iso {
                    // Two-tone board MUST exhibit UNEQUAL suit-class sizes (`1` for spades/diamonds, `2` for clubs+hearts)!
                    assert!(class_sizes.contains(&1) && class_sizes.contains(&2));
                }

                // A) Sum live probabilities across all unfolded/folded branches for EVERY compatible `(c0, c1)` pair!
                let mut checked_pairs = 0usize;
                for (i, c0) in game.combos(0).iter().enumerate() {
                    for (j, c1) in game.combos(1).iter().enumerate() {
                        if c0.mask.intersects(c1.mask) {
                            continue;
                        }
                        checked_pairs += 1;
                        let dead_holes = c0.mask | c1.mask;

                        // Direct deck probability sum over `rep_cards` + `iso_duplicates`
                        let mut prob_sum = 0.0f64;
                        let mut live_count = 0usize;
                        for &rc in rep_cards {
                            if !dead_holes.contains(rc) {
                                prob_sum += inv_denom;
                                live_count += 1;
                            }
                        }
                        for dup in iso_duplicates {
                            if !dead_holes.contains(dup.card) {
                                prob_sum += inv_denom;
                                live_count += 1;
                            }
                        }
                        assert_eq!(live_count, expected_denom);
                        assert!(
                            (prob_sum - 1.0).abs() <= 1e-12,
                            "board={b_str} iso={enable_iso} pair=({i},{j}): sum_c P(c) = {prob_sum}"
                        );
                    }
                }
                assert!(checked_pairs > 20);

                // B) Also verify via the solver's actual `iso_duplicates` index-swap machinery:
                //    If a synthetic child branch evaluates `1.0` for every live compatible pair on `rep_card`,
                //    then `rep + sum_{dup} swap(rep)` weighted by `1 / chance_denominator` MUST yield `1.0` (±1e-12)
                //    for every `i` (divided by `c0`'s compatible opponent weight)!
                let n0 = game.combos(0).len();
                let n1 = game.combos(1).len();
                let mut branch_ones: Vec<Vec<f64>> = Vec::with_capacity(rep_cards.len());
                for &rc in rep_cards {
                    let rmask = rc.mask();
                    let mut vec_i = vec![0.0f64; n0];
                    for (i, slot) in vec_i.iter_mut().enumerate() {
                        if game.combos(0)[i].mask.intersects(rmask) {
                            continue;
                        }
                        let mut w_live_opp = 0.0f64;
                        for j in 0..n1 {
                            if !game.combos(1)[j].mask.intersects(rmask)
                                && !game.combos(0)[i].mask.intersects(game.combos(1)[j].mask)
                            {
                                w_live_opp += (game.initial_weights[1][j] as f64) * inv_denom;
                            }
                        }
                        *slot = w_live_opp;
                    }
                    branch_ones.push(vec_i);
                }

                let mut total_for_i = vec![0.0f64; n0];
                for v in &branch_ones {
                    for (acc, &x) in total_for_i.iter_mut().zip(v.iter()) {
                        *acc += x;
                    }
                }
                for dup in iso_duplicates {
                    let mut swapped = branch_ones[dup.rep_branch_idx].clone();
                    for &(a, b) in &dup.swaps[0] {
                        swapped.swap(a as usize, b as usize);
                    }
                    let dmask = dup.card.mask();
                    for (i, &x) in swapped.iter().enumerate() {
                        if !game.combos(0)[i].mask.intersects(dmask) {
                            total_for_i[i] += x;
                        }
                    }
                }

                for (i, &tot_i) in total_for_i.iter().enumerate() {
                    let mut expected_opp_w = 0.0f64;
                    for j in 0..n1 {
                        if !game.combos(0)[i].mask.intersects(game.combos(1)[j].mask) {
                            expected_opp_w += game.initial_weights[1][j] as f64;
                        }
                    }
                    if expected_opp_w > 0.0 {
                        let ratio = tot_i / expected_opp_w;
                        assert!(
                            (ratio - 1.0).abs() <= 1e-12,
                            "board={b_str} iso={enable_iso} combo {i}: swapped chance probability sum = {ratio} != 1.0"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn step1_tight_checkdown_oracle_fixtures_a_through_e_within_1e9() {
        let assert_checkdown_matches_equity_1e9 =
            |label: &str, board_str: &str, oop: &PloRange, ip: &PloRange, enable_iso: bool| {
                let board: Board = board_str.parse().unwrap();
                let pot = 100i32;
                let tree = build_checkdown_tree(board.street(), pot);
                let game = PloGame::with_options(
                    board,
                    oop,
                    ip,
                    &tree,
                    &IdentityAbstraction,
                    SolverBuildOptions {
                        max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                        enable_chance_isomorphism: enable_iso,
                        ..Default::default()
                    },
                )
                .unwrap();

                let [ev_oop, ev_ip] = game.root_expected_values_f64();
                let eq = plo_range::equity_range_vs_range(oop, ip, &board).unwrap();
                let exp_oop = (pot as f64) * eq.equity_p0;
                let exp_ip = (pot as f64) * eq.equity_p1;

                let diff_oop = (ev_oop - exp_oop).abs();
                let diff_ip = (ev_ip - exp_ip).abs();
                let rel_oop = diff_oop / (pot as f64);
                let rel_ip = diff_ip / (pot as f64);

                assert!(
                    rel_oop <= 1e-9 && diff_oop <= 1e-8 && rel_ip <= 1e-9 && diff_ip <= 1e-8,
                    "[{label}] iso={enable_iso}: solver EV=({ev_oop:.12}, {ev_ip:.12}) vs equity*pot=({exp_oop:.12}, {exp_ip:.12}), abs_diff=({diff_oop:.3e}, {diff_ip:.3e})"
                );
            };

        // Fixture A: Flop, 2 vs 2 disjoint combos, rainbow board (`Ks 9d 4c`)
        let fa_oop: PloRange = "AhAd7s6s, QhJhTd8d".parse().unwrap();
        let fa_ip: PloRange = "KhKd5h3h, JcTc8c2c".parse().unwrap();
        assert_checkdown_matches_equity_1e9("A_flop_2v2_disjoint_rainbow", "Ks9d4c", &fa_oop, &fa_ip, false);

        // Fixture B: Flop, 3 vs 3 with 1-card overlaps + non-uniform weights
        let fb_oop: PloRange = "AhAd7s6s@1.0, QhJhTd8d@0.75, AsQdJcTc@0.5".parse().unwrap();
        let fb_ip: PloRange = "AhKd5h3h@0.8, QhTc8c2c@1.0, JsTs7d6d@0.6".parse().unwrap();
        assert_checkdown_matches_equity_1e9("B_flop_3v3_1card_overlaps", "Ks9d4c", &fb_oop, &fb_ip, false);

        // Fixture C: Turn, 3 vs 3 with 1-card overlaps
        assert_checkdown_matches_equity_1e9("C_turn_3v3_1card_overlaps", "Ks9d4c2h", &fb_oop, &fb_ip, false);

        // Fixture D & E: Two-tone Flop (`Ks Qs 4d`, unequal suit-class sizes: `1` for `s`, `1` for `d`, `2` for `c/h`)
        // AND Two-tone Turn (`As Ks 7c 2c`, unequal suit-class sizes: `1` for `s`, `1` for `c`, `2` for `d/h`),
        // with suit-symmetric overlapping ranges so `iso=true` actively folds `c <-> h` (on flop) and `d <-> h` (on turn)!
        let fd_oop: PloRange = "AAQQ$ds@0.9, (JT)(98)@0.6".parse().unwrap();
        let fd_ip: PloRange = "AAJJ$ds@0.7, (T9)(87)@1.0".parse().unwrap();

        // Confirm `iso=true` actually folds branches on both boards!
        let flop_2tone: Board = "KsQs4d".parse().unwrap();
        let t_flop = build_checkdown_tree(Street::Flop, 100);
        let g_d_iso = PloGame::with_options(
            flop_2tone,
            &fd_oop,
            &fd_ip,
            &t_flop,
            &IdentityAbstraction,
            SolverBuildOptions {
                max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                enable_chance_isomorphism: true,
                ..Default::default()
            },
        )
        .unwrap();
        let g_e_noiso = PloGame::with_options(
            flop_2tone,
            &fd_oop,
            &fd_ip,
            &t_flop,
            &IdentityAbstraction,
            SolverBuildOptions {
                max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                enable_chance_isomorphism: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            g_d_iso.memory_report().terminal_nodes < g_e_noiso.memory_report().terminal_nodes,
            "Fixture D must exercise non-trivial suit folding on two-tone flop!"
        );

        // D: iso = true (two-tone flop & two-tone turn)
        assert_checkdown_matches_equity_1e9("D_flop_twotone_iso_on", "KsQs4d", &fd_oop, &fd_ip, true);
        assert_checkdown_matches_equity_1e9("D_turn_twotone_iso_on", "AsKs7c2c", &fd_oop, &fd_ip, true);

        // E: iso = false (same two-tone flop & two-tone turn)
        assert_checkdown_matches_equity_1e9("E_flop_twotone_iso_off", "KsQs4d", &fd_oop, &fd_ip, false);
        assert_checkdown_matches_equity_1e9("E_turn_twotone_iso_off", "AsKs7c2c", &fd_oop, &fd_ip, false);

        // Also assert D vs E match each other to `1e-12`!
        let ev_d = g_d_iso.root_expected_values_f64();
        let ev_e = g_e_noiso.root_expected_values_f64();
        assert!(
            (ev_d[0] - ev_e[0]).abs() <= 1e-12 && (ev_d[1] - ev_e[1]).abs() <= 1e-12,
            "iso=true {ev_d:?} vs iso=false {ev_e:?}"
        );
    }

    // ========================================================================
    // STEP 2: 200-Case Deterministic Fuzz Oracles (IE Showdown, IE Fold, Tree vs Brute)
    // ========================================================================

    #[test]
    fn step2_fuzz_200_random_overlapping_ranges_ie_and_30_checkdown_trees_vs_brute() {
        // Fixed deterministic SplitMix64 PRNG
        let mut seed = 0xD1B5_4A32_D192_ED03u64;
        let mut next_u64 = || -> u64 {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        let mut deck = [Card::new(0).unwrap(); 52];
        for (i, slot) in deck.iter_mut().enumerate() {
            *slot = Card::new(i as u8).unwrap();
        }

        let draw_weighted_hand_from_pool =
            |pool: &[Card], rng: &mut dyn FnMut() -> u64| -> (Hand4, f32) {
                let len = pool.len();
                let mut indices = [0usize; 4];
                let mut k = 0usize;
                while k < 4 {
                    let cand = (rng() as usize) % len;
                    if !indices[..k].contains(&cand) {
                        indices[k] = cand;
                        k += 1;
                    }
                }
                let h = Hand4::new([
                    pool[indices[0]],
                    pool[indices[1]],
                    pool[indices[2]],
                    pool[indices[3]],
                ])
                .unwrap();
                // Non-uniform weight in `0.05..=1.00`
                let w = (((rng() % 20) + 1) as f32) * 0.05;
                (h, w)
            };

        let mut valid_cases = 0usize;
        let mut tree_cases = 0usize;
        let mut attempts = 0usize;

        while valid_cases < 200 {
            attempts += 1;
            assert!(attempts < 2_000, "unexpectedly many empty-after-removal draws");

            // Shuffle deck to pick 5-card river board + compact 14-card hot pool (forces heavy 1/2/3/4-card overlaps,
            // paired hands, suited/offsuit mixes, AND occasional board-blocked combos!)
            for i in 0..22 {
                let j = i + ((next_u64() as usize) % (52 - i));
                deck.swap(i, j);
            }
            let river_arr = [deck[0], deck[1], deck[2], deck[3], deck[4]];
            let river_board = Board::river(river_arr).unwrap();

            // Pool includes `deck[3..19]` (2 board cards `deck[3], deck[4]` + 14 live cards) so ranges routinely include
            // board-blocked hands AND 1-, 2-, 3-, and 4-card overlaps AND chops!
            let hot_pool = &deck[3..19];

            let (n_oop, n_ip) = if tree_cases < 30 {
                (
                    2 + ((next_u64() as usize) % 7), // 2..=8
                    2 + ((next_u64() as usize) % 7), // 2..=8
                )
            } else {
                (
                    2 + ((next_u64() as usize) % 23), // 2..=24
                    2 + ((next_u64() as usize) % 23), // 2..=24
                )
            };

            let mut oop_vec = Vec::with_capacity(n_oop);
            for _ in 0..n_oop {
                oop_vec.push(draw_weighted_hand_from_pool(hot_pool, &mut next_u64));
            }
            let mut ip_vec = Vec::with_capacity(n_ip);
            for _ in 0..n_ip {
                ip_vec.push(draw_weighted_hand_from_pool(hot_pool, &mut next_u64));
            }

            let Ok(oop_range) = PloRange::from_hands(oop_vec) else {
                continue;
            };
            let Ok(ip_range) = PloRange::from_hands(ip_vec) else {
                continue;
            };

            let starting_pot = 50 + ((next_u64() % 151) as i32); // 50..=200
            let paid_oop = ((next_u64() % 120) as f64) + 10.0;
            let paid_ip = paid_oop; // symmetric showdown investment
            let total_pot = (starting_pot as f64) + paid_oop + paid_ip;

            let bf_sd = brute_showdown_ev(
                &oop_range,
                &ip_range,
                river_arr,
                total_pot,
                paid_oop,
                paid_ip,
            );
            if bf_sd.compatible_mass <= 1e-12 {
                continue; // skip empty-after-removal cases per spec
            }

            // 2a: Unconditional 15-term IE Showdown (`ie_showdown_ev`) vs `brute_showdown_ev`
            let ie_sd = ie_showdown_ev(
                &oop_range,
                &ip_range,
                river_arr,
                total_pot,
                paid_oop,
                paid_ip,
            );
            assert!(
                (ie_sd.compatible_mass - bf_sd.compatible_mass).abs() <= 1e-9
                    && (ie_sd.win_mass - bf_sd.win_mass).abs() <= 1e-9
                    && (ie_sd.tie_mass - bf_sd.tie_mass).abs() <= 1e-9
                    && (ie_sd.lose_mass - bf_sd.lose_mass).abs() <= 1e-9
                    && (ie_sd.ev_oop - bf_sd.ev_oop).abs() <= 1e-8
                    && (ie_sd.ev_ip - bf_sd.ev_ip).abs() <= 1e-8,
                "Fuzz #{valid_cases} showdown IE mismatch:\n  IE={ie_sd:?}\n  BF={bf_sd:?}"
            );

            // 2b: Unconditional 15-term IE Fold (`ie_fold_ev`) vs `brute_fold_ev` on asymmetric investments too!
            let fold_paid_oop = 25.0;
            let fold_paid_ip = 75.0;
            let fold_pot = (starting_pot as f64) + fold_paid_oop + fold_paid_ip;
            for fp in [PLAYER_OOP, PLAYER_IP] {
                let bf_f = brute_fold_ev(
                    &oop_range,
                    &ip_range,
                    &river_board,
                    fold_pot,
                    fold_paid_oop,
                    fold_paid_ip,
                    fp,
                );
                let ie_f = ie_fold_ev(
                    &oop_range,
                    &ip_range,
                    &river_board,
                    fold_pot,
                    fold_paid_oop,
                    fold_paid_ip,
                    fp,
                );
                assert!(
                    (ie_f.2 - bf_f.2).abs() <= 1e-9
                        && (ie_f.0 - bf_f.0).abs() <= 1e-9
                        && (ie_f.1 - bf_f.1).abs() <= 1e-9,
                    "Fuzz #{valid_cases} fold(fp={fp}) IE mismatch: IE={ie_f:?} vs BF={bf_f:?}"
                );
            }

            // 2c: For 30 smaller cases (`N, M <= 8`), build a River Check-Down `PloGame` (`IdentityAbstraction`)
            // and verify BOTH `root_expected_values_f64()` AND `eval_terminal_showdown_ie_unconditional` match `brute_showdown_ev` to `1e-9`!
            if tree_cases < 30 && oop_range.len() <= 8 && ip_range.len() <= 8 {
                let cd_tree = build_checkdown_tree(Street::River, starting_pot);
                let game = PloGame::new(river_board, &oop_range, &ip_range, &cd_tree).unwrap();
                let [g_ev0, g_ev1] = game.root_expected_values_f64();
                let bf_cd = brute_showdown_ev(
                    &oop_range,
                    &ip_range,
                    river_arr,
                    starting_pot as f64,
                    0.0,
                    0.0,
                );
                assert!(
                    (g_ev0 - bf_cd.ev_oop).abs() <= 1e-9 && (g_ev1 - bf_cd.ev_ip).abs() <= 1e-9,
                    "Fuzz tree #{tree_cases} checkdown EV=({g_ev0}, {g_ev1}) vs BF=({}, {})",
                    bf_cd.ev_oop,
                    bf_cd.ev_ip
                );

                // Also directly invoke `eval_terminal_showdown_f64` (100% unconditional 15-term IE, zero pairwise fast path!)
                let mut scratch = InclusionExclusionScratch::new();
                let mut out_ie_f64 = vec![0.0f64; game.combos(0).len()];
                let init_opp_f64: Vec<f64> = game.initial_weights[1]
                    .iter()
                    .map(|&w| w as f64)
                    .collect();
                game.eval_terminal_showdown_f64(
                    &mut out_ie_f64,
                    PLAYER_OOP,
                    0,
                    0.5 * (starting_pot as f64),
                    0.0,
                    &init_opp_f64,
                    &mut scratch,
                );
                let centered: f64 = out_ie_f64
                    .iter()
                    .zip(&game.initial_weights[0])
                    .map(|(&v, &w)| v * (w as f64))
                    .sum();
                let ie_tree_ev0 = 0.5 * (starting_pot as f64) + centered;
                assert!(
                    (ie_tree_ev0 - bf_cd.ev_oop).abs() <= 1e-9,
                    "Fuzz tree #{tree_cases} unconditional IE={ie_tree_ev0} vs BF={}",
                    bf_cd.ev_oop
                );

                tree_cases += 1;
            }

            valid_cases += 1;
        }

        assert_eq!(valid_cases, 200);
        assert_eq!(tree_cases, 30);
    }

    // ========================================================================
    // STEP 4: Deterministic 2D & 1D Bucket Assignments Under Arbitrary Combo Permutation
    // ========================================================================

    #[test]
    fn step4_bucket_assignment_invariance_under_slice_permutations_and_ties() {
        // Construct ranges with many exact strength ties AND exact nut/blocker ties across flop, turn, and river!
        let flop: Board = "KsQs4d".parse().unwrap();
        let turn: Board = "KsQs4d2c".parse().unwrap();
        let river: Board = "KsQsJs4d2c".parse().unwrap();

        let hero_range: PloRange =
            "AA99, KQJT$ds, (QJ)(98), As9h8d7c, As9d8h7c, Ah9c8d7h, Ad9h8c7d, 6655"
                .parse()
                .unwrap();
        let opp_range: PloRange = "AAKK, QQTT, (JT)(98), KdQd6c5c, AsTs6h5h".parse().unwrap();

        let hero_base = hero_range.entries().to_vec();
        let opp_base = opp_range.entries().to_vec();
        assert!(hero_base.len() > 40);
        assert!(opp_base.len() > 40);

        let abs_1d = EquityBucketAbstraction::new(8, 8, 8);
        let abs_2d = NutBlockerBucketAbstraction::new((4, 3), (4, 3), (3, 3));
        let abs_cap_exceeds = NutBlockerBucketAbstraction::new((500, 2), (500, 2), (500, 2));

        let mut rng_seed = 0x7A11_BEEF_CAFE_5511u64;
        let mut shuffle = |slice: &mut [WeightedHand]| {
            for i in 0..slice.len() {
                rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = i + ((rng_seed >> 32) as usize % (slice.len() - i));
                slice.swap(i, j);
            }
        };

        for board in [flop, turn, river] {
            for abs in [
                &abs_1d as &dyn HandAbstraction,
                &abs_2d as &dyn HandAbstraction,
                &abs_cap_exceeds as &dyn HandAbstraction,
            ] {
                let (nb_ref, map_ref) = abs.assign_buckets(0, &board, &hero_base, &opp_base);
                let mut ref_by_cid = std::collections::BTreeMap::new();
                for (i, c) in hero_base.iter().enumerate() {
                    ref_by_cid.insert(c.combo_id, map_ref[i]);
                }

                // Permute BOTH `hero_combos` AND `opp_combos` across 10 independent shuffles + exact reverse!
                for trial in 0..10 {
                    let mut h_perm = hero_base.clone();
                    let mut o_perm = opp_base.clone();
                    if trial == 0 {
                        h_perm.reverse();
                        o_perm.reverse();
                    } else {
                        shuffle(&mut h_perm);
                        shuffle(&mut o_perm);
                    }

                    let (nb_p, map_p) = abs.assign_buckets(0, &board, &h_perm, &o_perm);
                    assert_eq!(nb_p, nb_ref, "num_buckets differed on {board:?} trial {trial}");
                    for (i, c) in h_perm.iter().enumerate() {
                        assert_eq!(
                            map_p[i],
                            ref_by_cid[&c.combo_id],
                            "bucket ID differed for hand {} ({:?}) on {board:?} trial {trial}",
                            c.hand,
                            c.combo_id
                        );
                    }
                }
            }
        }
    }

    // ========================================================================
    // PART C — STEP 1: Memory Estimates Table (`part_c_memory_rows`)
    // ========================================================================

    #[test]
    fn part_c_memory_rows() {
        let flop: Board = "Ks9s4d".parse().unwrap();
        let two_sizes = StreetBetSizes {
            bet_fractions: vec![0.25, 0.5],
            raise_fractions: vec![1.0],
            donk_fractions: None,
        };
        let tree = ActionTree::build(TreeConfig {
            initial_street: Street::Flop,
            starting_pot: 100,
            effective_stack: 50,
            flop_sizes: [two_sizes.clone(), two_sizes.clone()],
            turn_sizes: [two_sizes.clone(), two_sizes.clone()],
            river_sizes: [two_sizes.clone(), two_sizes],
            ..Default::default()
        })
        .unwrap();

        // Build deterministic `{h, c}`-suit-symmetric ranges of exact size `target_k` on `Ks9s4d`
        let build_sym_range = |target_k: usize| -> PloRange {
            let bmask = flop.mask();
            let swap_hc = [2u8, 1, 0, 3]; // c(0) <-> h(2), d(1)->1, s(3)->3 fixes `Ks9s4d`
            let mut used = vec![false; plo_cards::NUM_PLO_COMBOS];
            let mut chosen: Vec<(Hand4, f32)> = Vec::with_capacity(target_k);

            // Pass 1: size-2 `{h, c}` orbits so pairs fit evenly
            for (cid, h) in Hand4::all_hands() {
                if chosen.len() + 2 > target_k {
                    break;
                }
                if h.is_blocked_by(bmask) || used[cid.index()] {
                    continue;
                }
                let h_sw = permute_hand4(h, swap_hc);
                let cid_sw = h_sw.combo_id();
                if cid_sw != cid {
                    used[cid.index()] = true;
                    used[cid_sw.index()] = true;
                    chosen.push((h, 1.0));
                    chosen.push((h_sw, 1.0));
                }
            }
            // Pass 2: `{h, c}`-self-symmetric singletons if `target_k` is odd
            for (cid, h) in Hand4::all_hands() {
                if chosen.len() >= target_k {
                    break;
                }
                if h.is_blocked_by(bmask) || used[cid.index()] {
                    continue;
                }
                let h_sw = permute_hand4(h, swap_hc);
                if h_sw.combo_id() == cid {
                    used[cid.index()] = true;
                    chosen.push((h, 1.0));
                }
            }
            assert_eq!(chosen.len(), target_k);
            PloRange::from_hands(chosen).unwrap()
        };

        let r50 = build_sym_range(50);
        let r200 = build_sym_range(200);
        let r500 = build_sym_range(500);

        // `NutBlocker k=50` -> `(25, 2)` (`25 * 2 = 50`, matching `WasmPloSession::split_2d(50)`);
        // `NutBlocker k=100` -> `(50, 2)` (`50 * 2 = 100`, matching `WasmPloSession::split_2d(100)`).
        let nb50 = NutBlockerBucketAbstraction::new((25, 2), (25, 2), (25, 2));
        let nb100 = NutBlockerBucketAbstraction::new((50, 2), (50, 2), (50, 2));

        println!("\n=== Part C Step 1 Memory Estimates (Flop Ks9s4d, Pot=100, Stack=50, bet=[25%,50%], raise=[100%], req_iso=true) ===");
        println!(
            "{:<12} | {:<18} | {:<7} | {:>10} | {:>14} | {:>14} | {:>10}",
            "Range", "Abstraction", "EffIso", "DecNodes", "RegretStrat(B)", "TotalEst(B)", "Total(MiB)"
        );

        for (label_r, rng, n_c) in [("50 vs 50", &r50, 50usize), ("200 vs 200", &r200, 200), ("500 vs 500", &r500, 500)] {
            for (label_a, abs) in [
                ("Identity", &IdentityAbstraction as &dyn HandAbstraction),
                ("NutBlocker k=50", &nb50 as &dyn HandAbstraction),
                ("NutBlocker k=100", &nb100 as &dyn HandAbstraction),
            ] {
                let eff_iso = abs.is_suit_symmetric();
                let est = estimate_memory_for_config(flop, rng, rng, &tree, abs, true).unwrap();
                let mib = (est.total_estimated_bytes as f64) / (1024.0 * 1024.0);
                println!(
                    "{:<12} | {:<18} | {:<7} | {:>10} | {:>14} | {:>14} | {:>9.2}",
                    label_r,
                    label_a,
                    eff_iso,
                    est.decision_nodes,
                    est.regret_and_strategy_bytes,
                    est.total_estimated_bytes,
                    mib
                );

                if eff_iso {
                    assert_eq!(est.decision_nodes, 34_880);
                } else {
                    assert_eq!(est.decision_nodes, 57_240);
                }

                // Cross-check `estimate_memory_for_config` against a live built `PloGame::memory_report()` on the `200 vs 200` `k=50` row!
                if n_c == 200 && label_a == "NutBlocker k=50" {
                    assert!(
                        est.total_estimated_bytes < 128 * 1024 * 1024,
                        "200 vs 200 NutBlocker k=50 must fit well under 128 MiB, got {} B",
                        est.total_estimated_bytes
                    );
                    let g = PloGame::with_options(
                        flop,
                        rng,
                        rng,
                        &tree,
                        abs,
                        SolverBuildOptions {
                            max_memory_bytes: 128 * 1024 * 1024,
                            enable_chance_isomorphism: true,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let actual = g.memory_report();
                    println!(
                        "  [Live PloGame 200v200 k=50 check] est_total={} B vs actual_total={} B (regret_est={} B vs regret_actual={} B, river_est={} B vs river_actual={} B)",
                        est.total_estimated_bytes,
                        actual.total_estimated_bytes,
                        est.regret_and_strategy_bytes,
                        actual.regret_and_strategy_bytes,
                        est.river_eval_bytes,
                        actual.river_eval_bytes
                    );
                    assert_eq!(est.decision_nodes, actual.decision_nodes);
                    assert_eq!(est.regret_and_strategy_bytes, actual.regret_and_strategy_bytes);
                    assert!(actual.total_estimated_bytes <= est.total_estimated_bytes);
                }
                if n_c == 200 && label_a == "NutBlocker k=100" {
                    assert_eq!(est.regret_and_strategy_bytes, 99_270_400);
                    let g100 = PloGame::with_options(
                        flop,
                        rng,
                        rng,
                        &tree,
                        abs,
                        SolverBuildOptions {
                            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
                            enable_chance_isomorphism: true,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let actual100 = g100.memory_report();
                    println!(
                        "  [Live PloGame 200v200 k=100 check] est_total={} B vs actual_total={} B (regret_est={} B vs regret_actual={} B)",
                        est.total_estimated_bytes,
                        actual100.total_estimated_bytes,
                        est.regret_and_strategy_bytes,
                        actual100.regret_and_strategy_bytes
                    );
                    assert!(actual100.regret_and_strategy_bytes <= est.regret_and_strategy_bytes);
                    assert!(actual100.total_estimated_bytes <= est.total_estimated_bytes);
                }

                if n_c == 500 && label_a == "Identity" {
                    let refused_128 = PloGame::with_options(
                        flop,
                        rng,
                        rng,
                        &tree,
                        abs,
                        SolverBuildOptions {
                            max_memory_bytes: 128 * 1024 * 1024,
                            enable_chance_isomorphism: true,
                            ..Default::default()
                        },
                    );
                    assert!(
                        matches!(refused_128, Err(SolverError::MemoryBudgetExceeded { .. })),
                        "Identity 500 vs 500 must trip MemoryBudgetExceeded under 128 MiB budget!"
                    );
                }
            }
        }
    }

    // ========================================================================
    // PART C — STEP 3: Blocking Correctness Tests (C1, C2, C3, C4)
    // ========================================================================

    #[test]
    fn part_c_c1_overlapping_8v8_checkdown_with_nut_blocker_k2_and_k3_matches_exact_equity() {
        // C1: Tiny flop AND river check-down, `8 vs 8` overlapping combos, `NutBlocker k = 2` and `k = 3`:
        // Terminal evaluation under buckets MUST remain combo-exact (`2+3` PLO eval + 15-term 4-card IE),
        // matching `equity_range_vs_range * pot` within `<= 1e-8`!
        let flop: Board = "Ks9s4d".parse().unwrap();
        let river: Board = "Ks9s4d7c2h".parse().unwrap();
        let pot = 100;

        // `8` vs `8` heavily overlapping 4-card PLO combos (sharing `As, Qs, Js, Ts, Ah, Qh, Jh, Th, Ad, Qd` in `1, 2, 3, and 4` cards!)
        let oop: PloRange = [
            "AsQsJhTh@1.0",
            "AsJsQhTc@0.75",
            "AhQhJdTd@0.5",
            "AcQcJdTd@1.0",
            "AsTs8h7h@0.625",
            "AhTh8c6c@1.0",
            "AdQdJcTc@0.875",
            "QsJs8d6d@0.5",
        ]
        .join(", ")
        .parse()
        .unwrap();
        let ip: PloRange = [
            "AsQsJhTh@0.5",  // 4-card identical overlap with OOP #0!
            "AsQsJdTd@1.0",  // 2-card overlap (`AsQs` or `JdTd`)
            "AsJhTh8c@0.75", // 3-card overlap (`AsJhTh`) with OOP #0!
            "AhQhJcTc@1.0",  // 2-card overlap (`AhQh` or `JcTc`)
            "AdQd8h6h@0.625",
            "AcTs8d6c@1.0",
            "QsTs7h6h@0.5",
            "JsTs6d5c@0.875",
        ]
        .join(", ")
        .parse()
        .unwrap();

        assert_eq!(oop.without_dead(flop.mask()).len(), 8);
        assert_eq!(ip.without_dead(flop.mask()).len(), 8);
        assert_eq!(oop.without_dead(river.mask()).len(), 8);
        assert_eq!(ip.without_dead(river.mask()).len(), 8);

        let abs_k2 = NutBlockerBucketAbstraction::new((1, 2), (1, 2), (1, 2));
        let abs_k3 = NutBlockerBucketAbstraction::new((3, 1), (3, 1), (3, 1));

        for board in [river, flop] {
            let tree = build_checkdown_tree(board.street(), pot);
            let eq = plo_range::equity_range_vs_range(&oop, &ip, &board).unwrap();
            let exact_ev_oop = eq.equity_p0 * (pot as f64);
            let exact_ev_ip = eq.equity_p1 * (pot as f64);

            for (k_label, abs) in [("k=2", &abs_k2), ("k=3", &abs_k3)] {
                let game = PloGame::with_options(
                    board,
                    &oop,
                    &ip,
                    &tree,
                    abs,
                    SolverBuildOptions::default(),
                )
                .unwrap();
                let [ev_oop, ev_ip] = game.root_expected_values_f64();
                assert!(
                    (ev_oop - exact_ev_oop).abs() <= 1e-8,
                    "C1 mismatch on {board} ({k_label}): solver OOP EV={ev_oop:.12} vs exact={exact_ev_oop:.12}"
                );
                assert!(
                    (ev_ip - exact_ev_ip).abs() <= 1e-8,
                    "C1 mismatch on {board} ({k_label}): solver IP EV={ev_ip:.12} vs exact={exact_ev_ip:.12}"
                );
                assert!((ev_oop + ev_ip - (pot as f64)).abs() <= 1e-11);
            }
        }
    }

    #[test]
    fn part_c_c2_shared_bucket_combo_blocked_by_river_card_gets_zero_reach_while_live_peer_keeps_strategy() {
        // C2: Construct two OOP combos that share the EXACT SAME `bucket_id` on turn `Ks9s4d2c` (`k = 2`),
        // where one combo contains `Ah` (`AhQh8d7d`) and its bucket-mate (`AcQc8d7d`) does NOT contain `Ah`.
        // After check-checking (or betting/calling) the turn and calling `play_chance_card(Ah)` onto the river:
        // - `AhQh8d7d` MUST have reach `0.0` (blocked by dealt river card `Ah`)!
        // - `AcQc8d7d` MUST retain positive reach (`> 0.0`) and a valid normalized river strategy (`sum_a p(a) == 1.0`)!
        let turn: Board = "Ks9s4d2c".parse().unwrap();
        let oop: PloRange = "AsQsJsTs, AhQh8d7d, AcQc8d7d, 6h6c5h5c".parse().unwrap();
        let ip: PloRange = "AAJJ$ds, QQTT$ds".parse().unwrap();

        let two_sizes = StreetBetSizes {
            bet_fractions: vec![0.5, 1.0],
            raise_fractions: vec![1.0],
            donk_fractions: None,
        };
        let tree = ActionTree::build(TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 100,
            turn_sizes: [two_sizes.clone(), two_sizes.clone()],
            river_sizes: [two_sizes.clone(), two_sizes],
            ..Default::default()
        })
        .unwrap();

        let abs_k2 = NutBlockerBucketAbstraction::new((1, 2), (2, 1), (2, 1));
        let mut game = PloGame::with_options(
            turn,
            &oop,
            &ip,
            &tree,
            &abs_k2,
            SolverBuildOptions::default(),
        )
        .unwrap();
        let _ = game.solve(40, 0.0);

        let h_blocked: Hand4 = "AhQh8d7d".parse().unwrap();
        let h_live: Hand4 = "AcQc8d7d".parse().unwrap();
        let i_blocked = game
            .combos(0)
            .iter()
            .position(|c| c.hand == h_blocked)
            .unwrap();
        let i_live = game
            .combos(0)
            .iter()
            .position(|c| c.hand == h_live)
            .unwrap();

        // 1) Verify at Turn root decision node that `AhQh8d7d` and `AcQc8d7d` share the SAME `bucket_id`!
        let (acting_p, num_b, turn_bmap) = game.current_node_bucket_assignment().unwrap();
        assert_eq!(acting_p, 0);
        assert_eq!(num_b, 2);
        assert_eq!(
            turn_bmap[i_blocked], turn_bmap[i_live],
            "AhQh8d7d and AcQc8d7d have identical turn strength on Ks9s4d2c and must share a bucket!"
        );

        // Find a turn action `a_oop` where this shared bucket has positive strategy weight (`>= 0.25`)
        let root_rep = game.inspect_current_node();
        let n_oop = game.combos(0).len();
        let chosen_oop_a = (0..root_rep.actions.len())
            .max_by(|&a, &b| {
                root_rep.strategy[a * n_oop + i_live]
                    .partial_cmp(&root_rep.strategy[b * n_oop + i_live])
                    .unwrap()
            })
            .unwrap();
        let turn_prob = root_rep.strategy[chosen_oop_a * n_oop + i_live];
        assert!(turn_prob >= 0.25);
        assert_eq!(
            root_rep.strategy[chosen_oop_a * n_oop + i_blocked],
            turn_prob
        );

        // Advance to river chance node (`Check -> Check` if `chosen_oop_a == 0`, else `Bet -> Call`)
        game.play_action(chosen_oop_a).unwrap();
        let ip_rep = game.inspect_current_node();
        // Action 0 is Check (vs Check), Action 1 is Call (vs Bet)
        let call_or_check_idx = if chosen_oop_a == 0 { 0 } else { 1 };
        assert!(matches!(
            ip_rep.actions[call_or_check_idx],
            Action::Check | Action::Call
        ));
        game.play_action(call_or_check_idx).unwrap();
        assert_eq!(game.inspect_current_node().node_type, "chance");

        // Deal river card `Ah` (which blocks `AhQh8d7d` while leaving `AcQc8d7d` live!)
        let river_ah: Card = "Ah".parse().unwrap();
        game.play_chance_card(river_ah).unwrap();

        let [reach_oop, _] = game.current_combo_reaches();
        assert_eq!(
            reach_oop[i_blocked], 0.0,
            "Blocked combo AhQh8d7d MUST have exact 0.0 reach after dealing Ah!"
        );
        assert!(
            (reach_oop[i_live] - turn_prob).abs() <= 1e-6 && reach_oop[i_live] > 0.2,
            "Unblocked bucket-mate AcQc8d7d MUST retain positive reach ({} vs expected {turn_prob})!",
            reach_oop[i_live]
        );

        let river_rep = game.inspect_current_node();
        assert_eq!(river_rep.node_type, "oop");
        let river_prob_sum_live: f32 = (0..river_rep.actions.len())
            .map(|a| river_rep.strategy[a * n_oop + i_live])
            .sum();
        assert!(
            (river_prob_sum_live - 1.0).abs() <= 1e-5,
            "Live combo strategy probabilities at river node must sum to 1.0, got {river_prob_sum_live}"
        );
        assert_eq!(
            river_rep.ev_oop[i_blocked], 0.0,
            "Blocked combo conditional EV on river Ah must be 0.0"
        );
    }

    #[test]
    fn part_c_c3_k2_nut_blocker_splits_bare_ace_of_flush_suit_from_unblocking_air_on_flop() {
        // C3: On two-tone flop `Ks9s4d` AND monotone flop `Ks9s4s` (`k = 2` via `dims = (1, 2)`),
        // bare nut-flush-blocker `As` (`As7h6d2c` — holds ONLY 1 spade so it CANNOT make a PLO flush,
        // yet blocks Villain's nut spade flush!) MUST receive a distinct `bucket_id` (`1`) from
        // identical-rank unblocking air (`Ah7s6d2c` / `Ah7c6d2c` — no `As`, bucket `0`)!
        let abs_k2 = NutBlockerBucketAbstraction::new((1, 2), (1, 2), (1, 2));

        // 1) Two-tone flop `Ks9s4d`: compare TWO bare-`As` hands (`As8h6c2h`, `As8c6h2c`) vs TWO non-blocker air hands (`Ah8c6h2c`, `Ac8h6c2h`)!
        let flop_twotone: Board = "Ks9s4d".parse().unwrap();
        let oop_twotone: PloRange = "As8h6c2h, As8c6h2c, Ah8c6h2c, Ac8h6c2h".parse().unwrap();
        let ip_twotone: PloRange = "QsJsThTd, QhJhTsTd".parse().unwrap();
        let oop_entries_tt = oop_twotone
            .without_dead(flop_twotone.mask())
            .entries()
            .to_vec();
        let ip_entries_tt = ip_twotone
            .without_dead(flop_twotone.mask())
            .entries()
            .to_vec();
        assert_eq!(oop_entries_tt.len(), 4);

        let (nb_tt, map_tt) =
            abs_k2.assign_buckets(0, &flop_twotone, &oop_entries_tt, &ip_entries_tt);
        assert_eq!(nb_tt, 2);
        let ace_spades: Card = "As".parse().unwrap();
        for (i, c) in oop_entries_tt.iter().enumerate() {
            let expected = if c.mask.contains(ace_spades) { 1 } else { 0 };
            assert_eq!(
                map_tt[i], expected,
                "C3 two-tone flop Ks9s4d: hand {} (has bare As={}) got bucket {}, expected {expected}",
                c.hand,
                c.mask.contains(ace_spades),
                map_tt[i]
            );
        }

        // 2) Monotone flop `Ks9s4s`: TWO bare-`As` hands (`As8h6d2c`, `As8d6h2c` — still 0 spades in final 2+3 flush since PLO requires 2 hole spades!)
        //    vs TWO zero-spade air hands (`Ah8d6c2h`, `Ad8c6h2d`)
        let flop_mono: Board = "Ks9s4s".parse().unwrap();
        let oop_mono: PloRange = "As8h6d2c, As8d6h2c, Ah8d6c2h, Ad8c6h2d".parse().unwrap();
        let oop_entries_m = oop_mono.without_dead(flop_mono.mask()).entries().to_vec();
        let ip_entries_m = ip_twotone.without_dead(flop_mono.mask()).entries().to_vec();
        let (nb_m, map_m) = abs_k2.assign_buckets(0, &flop_mono, &oop_entries_m, &ip_entries_m);
        assert_eq!(nb_m, 2);
        for (i, c) in oop_entries_m.iter().enumerate() {
            let expected = if c.mask.contains(ace_spades) { 1 } else { 0 };
            assert_eq!(
                map_m[i], expected,
                "C3 monotone flop Ks9s4s: hand {} (has bare As={}) got bucket {}, expected {expected}",
                c.hand,
                c.mask.contains(ace_spades),
                map_m[i]
            );
        }
    }

    #[test]
    fn part_c_c4_shuffled_range_input_order_deterministic_buckets_on_part_c_fixture_board() {
        // C4: On the Part C fixture board `Ks9s4d` (plus turn `Ks9s4d7c` and river `Ks9s4d7c2h`),
        // shuffling macro tokens AND arbitrary `WeightedHand` slice permutations on hundreds of combos
        // produces 100% identical `bucket_id` per `ComboId` under `NutBlocker k = 50` and `k = 100`!
        let flop: Board = "Ks9s4d".parse().unwrap();
        let turn: Board = "Ks9s4d7c".parse().unwrap();
        let river: Board = "Ks9s4d7c2h".parse().unwrap();

        let oop_tokens = [
            "AAKK$ds",
            "AAQQ$ds",
            "AAJJ$ds",
            "AATT$ds",
            "KKQQ$ds",
            "QQJJ$ds",
            "JJTT$ds",
            "TT99$ds",
            "9988$ds",
            "8877$ds",
            "AKQJ$ds",
            "KQJT$ds",
            "QJT9$ds",
            "JT98$ds",
            "T987$ds",
            "9876$ds",
            "(AK)(QT)",
            "(AQ)(JT)",
            "(AJ)(T9)",
        ];
        let ip_tokens = [
            "AAKK$ss",
            "QQTT$ds",
            "JJ99$ds",
            "8866$ds",
            "7766$ds",
            "AKQT$ds",
            "KQT9$ds",
            "QT98$ds",
            "J987$ds",
            "(KQ)(J9)",
            "(QJ)(T8)",
            "(JT)(97)",
        ];

        let oop_fwd: PloRange = oop_tokens.join(", ").parse().unwrap();
        let mut oop_rev_tokens = oop_tokens;
        oop_rev_tokens.reverse();
        let oop_rev: PloRange = oop_rev_tokens.join(", ").parse().unwrap();
        let ip_fwd: PloRange = ip_tokens.join(", ").parse().unwrap();

        let hero_base = oop_fwd.without_dead(flop.mask()).entries().to_vec();
        let hero_from_rev = oop_rev.without_dead(flop.mask()).entries().to_vec();
        let opp_base = ip_fwd.without_dead(flop.mask()).entries().to_vec();
        assert!(
            hero_base.len() >= 200,
            "expected >= 200 live hero combos on Ks9s4d, got {}",
            hero_base.len()
        );
        assert_eq!(hero_base.len(), hero_from_rev.len());
        for (a, b) in hero_base.iter().zip(hero_from_rev.iter()) {
            assert_eq!(a.combo_id, b.combo_id);
        }

        let nb50 = NutBlockerBucketAbstraction::new((25, 2), (25, 2), (25, 2));
        let nb100 = NutBlockerBucketAbstraction::new((50, 2), (50, 2), (50, 2));

        let mut rng_seed = 0xC4_2026_1122_99AAu64;
        let mut shuffle = |slice: &mut [WeightedHand]| {
            for i in 0..slice.len() {
                rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = i + ((rng_seed >> 32) as usize % (slice.len() - i));
                slice.swap(i, j);
            }
        };

        for board in [flop, turn, river] {
            for abs in [&nb50, &nb100] {
                let (nb_ref, map_ref) = abs.assign_buckets(0, &board, &hero_base, &opp_base);
                let mut ref_by_cid = std::collections::BTreeMap::new();
                for (i, c) in hero_base.iter().enumerate() {
                    ref_by_cid.insert(c.combo_id, map_ref[i]);
                }

                for trial in 0..8 {
                    let mut h_perm = hero_base.clone();
                    let mut o_perm = opp_base.clone();
                    if trial == 0 {
                        h_perm.reverse();
                        o_perm.reverse();
                    } else {
                        shuffle(&mut h_perm);
                        shuffle(&mut o_perm);
                    }
                    let (nb_p, map_p) = abs.assign_buckets(0, &board, &h_perm, &o_perm);
                    assert_eq!(nb_p, nb_ref);
                    for (i, c) in h_perm.iter().enumerate() {
                        assert_eq!(
                            map_p[i],
                            ref_by_cid[&c.combo_id],
                            "C4 bucket mismatch for {} on {board} trial {trial}",
                            c.hand
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn p0_bucketed_lossless_trajectory_matches_identity_and_filters_zero_weights() {
        // P0 verification:
        // 1. On a 2-street Turn -> River game with mixed fractional prior weights (0.25, 0.5, 0.75, 1.0),
        //    `EquityBucketAbstraction(k >= N)` assigns 1 combo per bucket (`is_identity = false` code path).
        //    Its DCFR iteration trajectory (regrets, average strategies, EVs, exploitability) MUST match
        //    `IdentityAbstraction` (`is_identity = true` code path) across every iteration!
        // 2. Appending `weight = 0.0` hands via `PloRange::set_weight(..., 0.0)` must produce identical
        //    combo counts, memory estimates, and DCFR trajectories (build-time zero-weight filtering).
        let board: Board = "Ks9s4d2h".parse().unwrap();
        let oop = PloRange::from_hands(vec![
            ("AsQsJsTs".parse().unwrap(), 1.0),
            ("KhKdQhQd".parse().unwrap(), 0.75),
            ("9h9c8h8c".parse().unwrap(), 0.50),
            ("7s6s5c3c".parse().unwrap(), 0.25),
            ("AcAdJcJd".parse().unwrap(), 0.60),
            ("TcTd8d7d".parse().unwrap(), 0.40),
        ])
        .unwrap();
        let ip = PloRange::from_hands(vec![
            ("AhQhJhTh".parse().unwrap(), 0.90),
            ("KcKsQcQd".parse().unwrap(), 0.80),
            ("Ts8s7c6c".parse().unwrap(), 0.35),
            ("6h6d5h5d".parse().unwrap(), 0.50),
            ("AdAsTdTc".parse().unwrap(), 1.0),
            ("QdJdTd9c".parse().unwrap(), 0.45),
        ])
        .unwrap();

        let mut oop_with_zeros = oop.clone();
        oop_with_zeros.set_weight("2c3d5h6c".parse::<Hand4>().unwrap(), 0.0);
        let mut ip_with_zeros = ip.clone();
        ip_with_zeros.set_weight("2d3h5s6d".parse::<Hand4>().unwrap(), 0.0);

        let tree = ActionTree::build(TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 100,
            turn_sizes: [
                StreetBetSizes {
                    bet_fractions: vec![0.5, 1.0],
                    raise_fractions: vec![1.0],
                    donk_fractions: None,
                },
                StreetBetSizes {
                    bet_fractions: vec![0.75],
                    raise_fractions: vec![1.0],
                    donk_fractions: None,
                },
            ],
            river_sizes: [
                StreetBetSizes {
                    bet_fractions: vec![0.75],
                    raise_fractions: vec![1.0],
                    donk_fractions: None,
                },
                StreetBetSizes {
                    bet_fractions: vec![0.75],
                    raise_fractions: vec![1.0],
                    donk_fractions: None,
                },
            ],
            ..Default::default()
        })
        .unwrap();

        let lossless_buckets = EquityBucketAbstraction::new(64, 64, 64);
        let opts = SolverBuildOptions {
            enable_chance_isomorphism: false,
            ..Default::default()
        };

        let mut g_id =
            PloGame::with_options(board, &oop, &ip, &tree, &IdentityAbstraction, opts.clone()).unwrap();
        let mut g_bk =
            PloGame::with_options(board, &oop, &ip, &tree, &lossless_buckets, opts.clone()).unwrap();
        let mut g_zw = PloGame::with_options(
            board,
            &oop_with_zeros,
            &ip_with_zeros,
            &tree,
            &lossless_buckets,
            opts,
        )
        .unwrap();

        assert_eq!(g_id.combos(0).len(), g_zw.combos(0).len());
        assert_eq!(g_id.combos(1).len(), g_zw.combos(1).len());
        let m_clean = estimate_memory_for_config(board, &oop, &ip, &tree, &IdentityAbstraction, false).unwrap();
        let m_zeros = estimate_memory_for_config(
            board,
            &oop_with_zeros,
            &ip_with_zeros,
            &tree,
            &IdentityAbstraction,
            false,
        )
        .unwrap();
        assert_eq!(m_clean.total_estimated_bytes, m_zeros.total_estimated_bytes);

        for checkpoint in [1u32, 5, 25, 100] {
            let cur = g_id.iterations_run();
            let delta = checkpoint - cur;
            for _ in 0..delta {
                g_id.solve_step();
                g_bk.solve_step();
                g_zw.solve_step();
            }

            let exp_id = g_id.exploitability();
            let exp_bk = g_bk.exploitability();
            let exp_zw = g_zw.exploitability();
            assert!(
                (exp_id - exp_bk).abs() < 1e-5,
                "iter {checkpoint}: Identity exploitability {exp_id} != lossless Bucketed {exp_bk}"
            );
            assert!(
                (exp_id - exp_zw).abs() < 1e-6,
                "iter {checkpoint}: zero-weight range perturbed exploitability ({exp_id} vs {exp_zw})"
            );

            let rep_id = g_id.inspect_current_node();
            let rep_bk = g_bk.inspect_current_node();
            for (i, (&s_id, &s_bk)) in rep_id.strategy.iter().zip(rep_bk.strategy.iter()).enumerate() {
                assert!(
                    (s_id - s_bk).abs() < 1e-5,
                    "iter {checkpoint} root strategy[{i}] mismatch: Identity={s_id} vs Bucketed={s_bk}"
                );
            }
        }
    }

    #[test]
    fn p0_two_street_turn_to_river_polar_vs_bluffcatcher_converges_to_exact_nash() {
        // Multi-street (Turn -> 44-card Chance -> River) analytical Nash verification:
        // Board on Turn: `Ks 9d 4c 2h`.
        // - OOP holds pure Polarized range:
        //   * Nut Top Set `Kh Kd Qh Jd` (unbeatable on every non-Ace? Wait: if Ace hits river, could IP make higher?
        //     Let's give OOP `As Ad Ks Kd` — Aces full / Top Set, wait `Ks` is on board so `Ah Ad Kh Kd` = Kings full / Top Set,
        //     and IP holds Medium Pair `9h 9s?` no `9d` on board -> `9h 8h 7d 6c`? No, give IP `Qc Qd Jc Td`!
        //     Wait: on Q/J/T/A rivers, `QcQdJcTd` can make QJJ/straight! Let's pick cards with ZERO river outdraws!):
        //   Board: `As Ks Qs 2d` (Turn, 3 spades on board!).
        //   - OOP Nuts: `Js Ts 8c 7c` -> Royal/Spade Flush (`As Ks Qs Js Ts` = Royal Flush on Turn! 100% unbeatable on ALL 44 rivers!).
        //   - OOP Air:  `3h 4h 6c 7d` -> Complete air (best 2+3 on any river is at most One Pair of 3/4/6/7 or Ace-high; cannot make straight or flush!).
        //   - IP Bluffcatcher: `Ad Ah Kc Kd` -> Top Two Pair / Aces Full (`Ad Ah` + `As Ks Qs` = Trip Aces `AAA K Q` on Turn, or Full House on A/K/Q/2 river;
        //     ALWAYS beats OOP Air `3h 4h 6c 7d` on all 44 rivers, and ALWAYS loses to OOP Royal Flush `Js Ts 8c 7c` on all 44 rivers!).
        //   Notice too: OOP's two hands (`JsTs8c7c` and `3h4h6c7c`? Make disjoint: `JsTs8c9c` and `3h4h6d7d`) share ZERO cards with IP's `AdAhKcKd`
        //   and ZERO cards with the board `AsKsQs2d`! So all 41 remaining deck cards deal cleanly with uniform probability and zero card-removal asymmetry!
        //
        // Game tree: Starting Pot = 100, Effective Stack = 100.
        // Turn: OOP Check only (`turn_sizes[0] = []`), IP Check only (`turn_sizes[1] = []`) -> deals all 44 River runouts ->
        // River: OOP can Check or Bet Pot (100, all-in), IP faces Check or Call/Fold!
        // Wait: let's allow betting on BOTH Turn AND River (`turn_sizes = [Pot], [Pot]`, `river_sizes = [Pot], [Pot]`)!
        // With Pot = 100 and Stack = 100, a Pot bet (100) is ALL-IN!
        // Analytical Nash game value across Turn + 44 Rivers:
        // - Regardless of whether OOP shoves Turn or checks to River and shoves River (or mixes across Turn/River),
        //   OOP has 50% Royal Flush (`JsTs9c8c`, always wins) and 50% pure Air (`3h4h6d7d`, always loses when called).
        // - Facing a pot-sized shove (risking 100 to win 200 -> pot odds 1:2), IP is indifferent when OOP bluffs half as often as value
        //   (`alpha = B / (P + B) = 100 / 200 = 1/2` -> with 1 value combo and 1 bluff combo, OOP bluffs `3h4h6d7d` with total probability `1/2`
        //   across the game tree, and IP calls `AdAhKcKd` such that OOP's bluff EV = 0!).
        // - Exact Nash EV for OOP:
        //   * Royal Flush wins the full 100 pot whenever IP checks back + half the time IP calls a 100 shove = `100 + 0.5 * 100 = +150` chips (from 0 baseline:
        //     in zero-sum chip gain relative to start of Turn where pot=100: Checkdown gives +100 to Royal, 0 to Air -> baseline EV = +50;
        //     with optimal bluffing +50% of pot on Royal = `0.5 * (100 + 50) + 0.5 * 0 = +75.0` chips!).
        //   Let's verify: at start of Turn, pot = 100 (50 each? No, `plo_cfr` terminal EV is net chips won from start of postflop + pot share:
        //   specifically, checkdown wins `+50.0` for winner and `-50.0` for loser! Wait: let's check `river_plo_polar_vs_bluffcatcher_nash_equilibrium` convention!).
        let board: Board = "AsKsQs2d".parse().unwrap();
        // Choose completely disjoint cards where Air NEVER outdraws Bluffcatcher on ANY of the 40 live rivers:
        // - Royal Flush: `JsTs9c8c` (always Royal Flush `AsKsQsJsTs` on all 40 live rivers)
        // - Pure Air:    `8h6h4c3c` (cannot make straight, flush, or two pair on ANY river:
        //     * Spades: 0 spades in hand -> no flush
        //     * Straights: 3 non-Ace board cards are `K(11), Q(10), 2(0)` -> any river card gives at most TWO ranks in `{2..9}`,
        //       so 3 board cards can NEVER form a 5-card straight with `{8,6,4,3}` (even on `5` river, `{A,2,5}` + `{3,4}` is Wheel, wait: `A,2,5` IS on board if `5` hits!).
        //       Good catch! `As` + `2d` + river `5x` + hole `4c3c` = `5-4-3-2-A` Wheel Straight!
        //       Replace `4c3c` with `9h7h6c4c`! Let's check `9h7h6c4c` on `AsKsQs2d`:
        //       - Broadway `{T,J,Q,K,A}`: hand has NO `{T,J,Q,K,A}` -> max 3 broadway cards from board -> no broadway straight!
        //       - Non-broadway straight (`5-high`..`K-high`): requires >= 3 board cards in a 5-rank window <= K (`9..K` needs `{9,T,J,Q,K}`, board has `K,Q`, river adds at most 1 of `{T,J}` -> need 2 from hand, hand only has `9` -> max 4 cards!).
        //       - Lower windows (`<= Q`, i.e. `8..Q` down to `A..5`): board `AsKsQs2d` has at most TWO cards in any window (`Q` alone, or `A,2` in wheel `A..5`), plus 1 river card = max 3 board cards (`A,2,5` or `A,2,3` or `A,2,4`); hand `9h7h6c4c` has ONLY `4` in `{A,2,3,4,5}`, so `4` + 1 other card cannot complete `A-2-3-4-5`!
        //       - Pairs/Trips: hand `9,7,6,4` is unpaired and disjoint from `{A,K,Q,2}` -> on `9/7/6/4` river makes One Pair (`99AKQ`); on paired board river makes Two Pair (`AAKK9`), NEVER beats `AdAhKcKd` (Set of Aces `AAAKQ` / Full House `AAAKK`)!
        //     Therefore `9h7h6c4c` has ZERO equity (0/40 rivers!) against `AdAhKcKd`!)
        let oop: PloRange = "JsTs9c8c, 9h7h6c4c".parse().unwrap();
        let ip: PloRange = "AdAhKcKd".parse().unwrap();

        let no_bet = StreetBetSizes {
            bet_fractions: vec![],
            raise_fractions: vec![],
            donk_fractions: None,
        };
        let pot_size = StreetBetSizes {
            bet_fractions: vec![1.0],
            raise_fractions: vec![1.0],
            donk_fractions: None,
        };
        // Notice `JsTs9c8c` and `9h7h6c4c` share ZERO cards with `AdAhKcKd` or `AsKsQs2d`, so all 40 live rivers
        // have exact uniform probability `1/40`!
        // Look at `Ts`? Wait: `JsTs9c8c` holds `Js, Ts`; what if the river is `9s` (`As Ks Qs 2d 9s`)?
        // Board has 4 spades (`As Ks Qs 9s`), and `AdAhKcKd` has NO spades (`Ad, Ah, Kc, Kd`)! Still loses to `JsTs`!
        // What if the river is `Td` or `Jd` or `8s`? Wait: what if the river is `Ts`? `Ts` is in `JsTs9c8c`!
        // Wait why was EV exactly `72.50` (`29/40 = 0.725`, i.e. 1 river out of 40 differed by `-100`?! `75 - 2.5 = 72.5`!!)?
        // Aha!! `2.5 = 100 / 40`!! ONE river out of 40 blocked `JsTs9c8c` vs `9h7h6c4c`!!
        // Look at the 40 cards not in `AsKsQs2d` (4) and `AdAhKcKd` (4): `52 - 8 = 44` cards!
        // On the 4 rivers `{Js, Ts, 9c, 8c}`, `JsTs9c8c` IS BLOCKED (so OOP ONLY has Air `9h7h6c4c`, IP calls 100%, OOP EV = 0)!
        // On the 4 rivers `{9h, 7h, 6c, 4c}`, `9h7h6c4c` IS BLOCKED (so OOP ONLY has Royal `JsTs9c8c`, IP folds 100%, OOP EV = 100)!
        // On the remaining `44 - 8 = 36` rivers, BOTH OOP combos are unblocked (polar vs bluffcatcher -> OOP EV = `0.5 * 150 + 0.5 * 0 = 75`)!
        // Let's compute the EXACT card-removal analytical Nash EV across all 40 valid `(oop, ip, river)` triples (`2 * 40 = 80` triples):
        // - `36` unblocked rivers (`72` triples): Royal wins `150`, Air wins `0` -> sum = `36 * 150 = 5,400`
        // - `4` Air-blocked rivers `{9h, 7h, 6c, 4c}` (`4` Royal triples): IP knows OOP cannot have Air (`9h7h6c4c` is blocked by the river card!!),
        //   so IP's exact Nash response on `{9h, 7h, 6c, 4c}` is FOLD 100%! Royal wins `100` (not `150`!) -> sum = `4 * 100 = 400`!
        // - `4` Royal-blocked rivers `{Js, Ts, 9c, 8c}` (`4` Air triples): IP knows OOP cannot have Royal (`JsTs9c8c` is blocked by the river card!!),
        //   so OOP checks Air (EV = `0`) -> sum = `4 * 0 = 0`!
        // TOTAL EXACT ANALYTICAL NASH EV = `(5,400 + 400 + 0) / 80 = 5,800 / 80 = 72.500000` chips for OOP, and `27.500000` chips for IP!!
        // And on the 2-street game where OOP can shove TURN *before* the river card reveals blockers, OOP shoves Turn to avoid river blocker revelation and gets `75.0`!!
        let tree_check_turn = ActionTree::build(TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 100,
            active_betting_streets: [true, false, true],
            turn_sizes: [no_bet.clone(), no_bet],
            river_sizes: [pot_size.clone(), pot_size.clone()],
            ..Default::default()
        })
        .unwrap();

        let mut g_id = PloGame::new(board, &oop, &ip, &tree_check_turn).unwrap();
        let exp_id = g_id.solve(600, 0.0);
        assert!(
            exp_id < 0.10,
            "2-street Turn->River Identity exploitability did not converge: {exp_id}"
        );
        let [ev_oop, ev_ip] = g_id.root_expected_values();
        assert!(
            (ev_oop - 72.5).abs() < 0.15 && (ev_ip - 27.5).abs() < 0.15,
            "expected exact card-removal analytical Nash EV [72.5, 27.5], got [{ev_oop}, {ev_ip}]"
        );

        let nb_abs = NutBlockerBucketAbstraction::new((4, 2), (4, 2), (4, 2));
        let mut g_nb = PloGame::with_options(
            board,
            &oop,
            &ip,
            &tree_check_turn,
            &nb_abs,
            SolverBuildOptions::default(),
        )
        .unwrap();
        let exp_nb = g_nb.solve(600, 0.0);
        assert!(
            exp_nb < 0.10,
            "2-street Turn->River Bucketed exploitability did not converge: {exp_nb}"
        );
        let [ev_oop_nb, ev_ip_nb] = g_nb.root_expected_values();
        assert!(
            (ev_oop_nb - 72.5).abs() < 0.15 && (ev_ip_nb - 27.5).abs() < 0.15,
            "expected exact card-removal analytical Bucketed Nash EV [72.5, 27.5], got [{ev_oop_nb}, {ev_ip_nb}]"
        );

        // Full 2-street active betting tree (Turn + River pot-sized bets/raises on BOTH streets):
        // Because OOP can shove on the Turn *before* a `{9h,7h,6c,4c}` river card exposes that Air is blocked,
        // OOP's exact 2-street Nash EV rises from `72.5` (forced Turn check) to `75.0` (Turn polar shove)!
        let tree_two_street_bets = ActionTree::build(TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 100,
            turn_sizes: [pot_size.clone(), pot_size.clone()],
            river_sizes: [pot_size.clone(), pot_size],
            ..Default::default()
        })
        .unwrap();
        let mut g_full_id = PloGame::new(board, &oop, &ip, &tree_two_street_bets).unwrap();
        let exp_full_id = g_full_id.solve(600, 0.0);
        let mut g_full_nb = PloGame::with_options(
            board,
            &oop,
            &ip,
            &tree_two_street_bets,
            &nb_abs,
            SolverBuildOptions::default(),
        )
        .unwrap();
        let exp_full_nb = g_full_nb.solve(600, 0.0);
        assert!(exp_full_id < 0.15 && exp_full_nb < 0.15);
        let [f_oop_id, f_ip_id] = g_full_id.root_expected_values();
        let [f_oop_nb, _f_ip_nb] = g_full_nb.root_expected_values();
        assert!(
            (f_oop_id - 75.0).abs() < 0.25 && (f_ip_id - 25.0).abs() < 0.25,
            "expected full 2-street betting Nash EV [75.0, 25.0], got [{f_oop_id}, {f_ip_id}]"
        );
        assert!((f_oop_id - f_oop_nb).abs() < 0.05);
    }

    #[test]
    fn f8_plosnap2_validates_config_hash_rejects_v1_and_nan_without_corrupting_game() {
        let board: Board = "Ks9d4c7h2s".parse().unwrap();
        let oop: PloRange = "AAKK$ds, QQJJ$ds".parse().unwrap();
        let ip: PloRange = "KKQQ$ds, JJTT$ds".parse().unwrap();
        let half_pot = StreetBetSizes {
            bet_fractions: vec![0.5],
            raise_fractions: vec![1.0],
            donk_fractions: None,
        };
        let full_pot = StreetBetSizes {
            bet_fractions: vec![1.0],
            raise_fractions: vec![1.0],
            donk_fractions: None,
        };
        let tree_a = ActionTree::build(TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 100,
            river_sizes: [half_pot.clone(), half_pot],
            ..Default::default()
        })
        .unwrap();
        // Same node count & bucket counts on River, but DIFFERENT bet size (1.0x pot instead of 0.5x pot)!
        let tree_b = ActionTree::build(TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 100,
            river_sizes: [full_pot.clone(), full_pot],
            ..Default::default()
        })
        .unwrap();

        let mut game_a = PloGame::new(board, &oop, &ip, &tree_a).unwrap();
        game_a.solve(80, 0.0);
        let snap_a = game_a.export_weights_binary();
        assert_eq!(&snap_a[..8], b"PLOSNAP2");

        // 1. Roundtrip into fresh game_a2 succeeds and preserves exact exploitability & strategy
        let mut game_a2 = PloGame::new(board, &oop, &ip, &tree_a).unwrap();
        game_a2.import_weights_binary(&snap_a).unwrap();
        assert_eq!(game_a2.iterations_run(), 80);
        assert!((game_a.exploitability() - game_a2.exploitability()).abs() < 1e-6);

        // 2. Importing snap_a (0.5x pot) into game_b (1.0x pot, identical node/bucket dimensions!)
        //    MUST be rejected by structural_config_hash!
        let mut game_b = PloGame::new(board, &oop, &ip, &tree_b).unwrap();
        game_b.solve(25, 0.0);
        let exp_before = game_b.exploitability();
        let err_hash = game_b.import_weights_binary(&snap_a).unwrap_err().to_string();
        assert!(
            err_hash.contains("configuration hash mismatch"),
            "unexpected error: {err_hash}"
        );
        // Atomic staging check: game_b state is completely untouched!
        assert_eq!(game_b.iterations_run(), 25);
        assert!((game_b.exploitability() - exp_before).abs() < 1e-9);

        // 3. Legacy PLOSNAP1 magic is rejected with an explicit human-readable upgrade message
        let mut legacy = snap_a.clone();
        legacy[..8].copy_from_slice(b"PLOSNAP1");
        let err_v1 = game_a2.import_weights_binary(&legacy).unwrap_err().to_string();
        assert!(err_v1.contains("PLOSNAP1") && err_v1.contains("PLOSNAP2"));

        // 4. Corrupting the root Player node's first regret f32 to NaN is rejected without mutating game_a2
        let mut nan_snap = snap_a.clone();
        // Header: 8 (magic) + 8 (hash) + 4 (iters) + 4 (n_nodes) + 1 (node 0 Player marker) + 4 (len) = byte 29
        nan_snap[29..33].copy_from_slice(&f32::NAN.to_le_bytes());
        let err_nan = game_a2.import_weights_binary(&nan_snap).unwrap_err().to_string();
        assert!(
            err_nan.contains("non-finite") || err_nan.contains("negative"),
            "unexpected NaN error: {err_nan}"
        );
        assert_eq!(game_a2.iterations_run(), 80);
    }
}
