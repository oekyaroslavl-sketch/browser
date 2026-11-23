//! Suit isomorphism and canonicalization for 4-card Pot-Limit Omaha (`PLO-4`).
//!
//! # PLO Suit Semantics
//! Unlike 2-card Hold'em (where starting hands are only paired, suited, or offsuit),
//! 4-card Omaha hands exhibit rich suit topologies:
//! - Rainbow (`1-1-1-1`): 4 distinct suits (`[As Kd Qh Jc]`)
//! - Single-suited (`2-1-1-0`): exactly two cards share a suit; **which** two ranks are suited matters!
//!   For example, `[As Ks Qd Jc]` (Ace-high suited) is NOT isomorphic to `[Ad Ks Qs Jc]` (King-high suited)!
//! - Double-suited (`2-2-0-0`): two distinct 2-card suits; **which** pairs of ranks share suits matters!
//!   For example, `[As Ks Qh Jh]` (`{A,K}` suited + `{Q,J}` suited) is isomorphic to `[Ah Kh Qs Js]`,
//!   NOT to `[As Qs Kh Jh]` (`{A,Q}` suited + `{K,J}` suited)! Those have different flush equities and MUST NOT be merged.
//! - Trips-suited (`3-1-0-0`): three cards share one suit (only 2 can be used at once, plus self-blockers!).
//! - Monotone (`4-0-0-0`): all 4 cards share one suit (heavily self-blocked!).
//!
//! Furthermore, once a community board is dealt, suits are no longer symmetric unless two suits have
//! the **exact same rank bitmask on the board** (and, for game-tree folding, identical player range weights
//! under the suit transposition).
//!
//! Therefore, `plo_iso` computes canonical forms by explicitly testing or constructing the lexicographically
//! smallest image over all valid suit permutations $\pi \in S_4$ (`24` permutations total — tiny and exact!).
//! Brute-forcing or symmetry-filtering over $|S_4| = 24$ guarantees **zero false merges** of non-equivalent
//! PLO suit structures.

use plo_cards::{Board, Card, ComboId, Hand4};

/// All `4! = 24` permutations of suits `[0, 1, 2, 3]`.
pub const SUIT_PERMS_24: [[u8; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

/// Suit pattern classification for a 4-card PLO hole hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SuitPattern4 {
    /// 4 distinct suits (`1-1-1-1`).
    Rainbow,
    /// Exactly 2 cards share a suit, other 2 have distinct suits (`2-1-1-0`).
    SingleSuited,
    /// Two pairs of suits (`2-2-0-0`).
    DoubleSuited,
    /// Three cards share one suit (`3-1-0-0`).
    TripsSuited,
    /// All four cards share one suit (`4-0-0-0`).
    Monotone,
}

/// Classifies the 4-card suit pattern of `hand`.
#[inline]
pub fn classify_suit_pattern(hand: Hand4) -> SuitPattern4 {
    let mut counts = [0u8; 4];
    for c in hand.cards() {
        counts[c.suit() as usize] += 1;
    }
    counts.sort_unstable_by(|a, b| b.cmp(a));
    match counts {
        [4, 0, 0, 0] => SuitPattern4::Monotone,
        [3, 1, 0, 0] => SuitPattern4::TripsSuited,
        [2, 2, 0, 0] => SuitPattern4::DoubleSuited,
        [2, 1, 1, 0] => SuitPattern4::SingleSuited,
        [1, 1, 1, 1] => SuitPattern4::Rainbow,
        _ => unreachable!(),
    }
}

/// Applies a suit permutation `perm` (`new_suit = perm[old_suit]`) to a single `Card`.
#[inline]
pub fn permute_card(card: Card, perm: [u8; 4]) -> Card {
    let new_suit = perm[card.suit() as usize];
    Card::from_rank_suit(card.rank(), new_suit).expect("suit perm entry must be in 0..4")
}

/// Applies a bijection `perm` (`new_suit = perm[old_suit]`) to a `Hand4`, returning the re-sorted `Hand4`.
#[inline]
pub fn permute_hand4(hand: Hand4, perm: [u8; 4]) -> Hand4 {
    let [c0, c1, c2, c3] = hand.cards();
    Hand4::new([
        permute_card(c0, perm),
        permute_card(c1, perm),
        permute_card(c2, perm),
        permute_card(c3, perm),
    ])
    .expect("bijective suit permutation preserves card distinctness")
}

/// Applies a bijection `perm` (`new_suit = perm[old_suit]`) to a `Board` (preserving street order).
#[inline]
pub fn permute_board(board: &Board, perm: [u8; 4]) -> Board {
    let mut tmp = [Card::new(0).unwrap(); 5];
    let slice = board.as_slice();
    for (i, &c) in slice.iter().enumerate() {
        tmp[i] = permute_card(c, perm);
    }
    Board::from_slice(&tmp[..slice.len()])
        .expect("bijective suit permutation preserves board distinctness")
}

/// Preflop canonicalization of a 4-card hand (`16,432` exact strategic PLO preflop equivalence classes
/// out of `270,725` total 4-card combos!).
///
/// Returns `(canonical_hand, suit_perm_applied)`.
/// Two `Hand4`s have the same canonical `Hand4` if and only if there exists a suit permutation $\pi \in S_4$
/// mapping one hand to the other.
#[inline]
pub fn canonicalize_preflop_hand(hand: Hand4) -> (Hand4, [u8; 4]) {
    let mut best_hand = hand;
    let mut best_perm = SUIT_PERMS_24[0];
    for &perm in &SUIT_PERMS_24[1..] {
        let cand = permute_hand4(hand, perm);
        if cand < best_hand {
            best_hand = cand;
            best_perm = perm;
        }
    }
    (best_hand, best_perm)
}

/// Board suit signature used to canonicalize a board (respecting street boundaries:
/// flop suits as a multiset of cards, turn card, river card).
///
/// For postflop solving on a fixed flop (where flop card order does not matter, e.g. `AsKd2d == KdAs2d`),
/// flop cards are sorted before comparing canonical keys; turn and river cards are kept in street order.
#[inline]
fn board_canonical_key(board: &Board) -> [u8; 5] {
    let s = board.as_slice();
    let mut flop = [s[0].id(), s[1].id(), s[2].id()];
    flop.sort_unstable();
    let t = if s.len() >= 4 { s[3].id() } else { u8::MAX };
    let r = if s.len() >= 5 { s[4].id() } else { u8::MAX };
    [flop[0], flop[1], flop[2], t, r]
}

/// Canonicalizes a community `Board` up to suit relabeling, returning `(canonical_board, suit_perm, stabilizers)`.
///
/// - `canonical_board`: board with suits relabeled by `suit_perm` (and flop sorted).
/// - `suit_perm`: the canonicalizing suit map (`new_suit = suit_perm[old_suit]`).
/// - `stabilizers`: all suit permutations $\sigma \in S_4$ that leave `board`'s per-street suit ranksets invariant!
///   Any `Hand4` can be further canonicalized across `stabilizers` without altering its relationship to `board`.
pub fn canonicalize_board(board: &Board) -> (Board, [u8; 4], Vec<[u8; 4]>) {
    let orig_key = board_canonical_key(board);
    let mut best_key = orig_key;
    let mut best_perm = SUIT_PERMS_24[0];
    let mut stabilizers = Vec::with_capacity(6);

    for &perm in &SUIT_PERMS_24 {
        let p_board = permute_board(board, perm);
        let key = board_canonical_key(&p_board);
        if key == orig_key {
            stabilizers.push(perm);
        }
        if key < best_key {
            best_key = key;
            best_perm = perm;
        }
    }

    // Reconstruct canonical board with sorted flop
    let mut cards = Vec::with_capacity(board.len());
    for &id in best_key.iter().take(board.len()) {
        cards.push(Card::new(id).unwrap());
    }
    let canon_board = Board::from_slice(&cards).unwrap();
    (canon_board, best_perm, stabilizers)
}

/// Jointly canonicalizes `(Hand4, Board)` up to simultaneous suit permutation $\pi \in S_4$.
///
/// Two `(Hand4, Board)` pairs produce the same `(canon_hand, canon_board)` **if and only if**
/// there exists $\pi \in S_4$ mapping `(flop_set, turn, river)` of board 1 to board 2 AND
/// mapping `hand1` to `hand2` simultaneously!
///
/// This guarantees that:
/// 1. Board-flush and nut-flush blockers match identically.
/// 2. Backdoor flush draws on flop/turn match identically.
/// 3. Non-equivalent double-suited or single-suited hands are NEVER merged.
pub fn canonicalize_hand_with_board(hand: Hand4, board: &Board) -> (Hand4, Board, [u8; 4]) {
    let mut best_board_key = [u8::MAX; 5];
    let mut best_hand = hand;
    let mut best_perm = SUIT_PERMS_24[0];
    let mut first = true;

    for &perm in &SUIT_PERMS_24 {
        let p_board = permute_board(board, perm);
        let b_key = board_canonical_key(&p_board);
        let p_hand = permute_hand4(hand, perm);

        if first || (b_key, p_hand) < (best_board_key, best_hand) {
            best_board_key = b_key;
            best_hand = p_hand;
            best_perm = perm;
            first = false;
        }
    }

    let mut b_cards = Vec::with_capacity(board.len());
    for &id in &best_board_key[..board.len()] {
        b_cards.push(Card::new(id).unwrap());
    }
    (best_hand, Board::from_slice(&b_cards).unwrap(), best_perm)
}

/// Transposes two suits `s1` and `s2` (`0..4`).
#[inline]
pub const fn suit_transposition(s1: u8, s2: u8) -> [u8; 4] {
    let mut perm = [0, 1, 2, 3];
    perm[s1 as usize] = s2;
    perm[s2 as usize] = s1;
    perm
}

/// Computes which suits on `board` have identical board ranksets on all streets dealt so far.
///
/// Specifically, `suits_symmetric_on_board(board, s1, s2)` is `true` iff swapping `s1 <-> s2`
/// maps the flop card set to itself AND maps turn (if present) and river (if present) to themselves.
#[inline]
pub fn suits_symmetric_on_board(board: &Board, s1: u8, s2: u8) -> bool {
    if s1 == s2 {
        return true;
    }
    let perm = suit_transposition(s1, s2);
    let p_board = permute_board(board, perm);
    board_canonical_key(board) == board_canonical_key(&p_board)
}

/// Checks whether a sparse list of `(ComboId, f32)` weights (sorted by `ComboId`) is invariant
/// under swapping `s1 <-> s2`.
pub fn is_weighted_combo_list_suit_symmetric(
    combos: &[(ComboId, f32)],
    s1: u8,
    s2: u8,
    tol: f32,
) -> bool {
    if s1 == s2 {
        return true;
    }
    let perm = suit_transposition(s1, s2);
    for &(cid, w) in combos {
        if w == 0.0 {
            continue;
        }
        let swapped_cid = permute_hand4(cid.to_hand(), perm).combo_id();
        match combos.binary_search_by_key(&swapped_cid, |&(id, _)| id) {
            Ok(pos) => {
                if (combos[pos].1 - w).abs() > tol {
                    return false;
                }
            }
            Err(_) => return false,
        }
    }
    true
}

/// Representation of chance-card isomorphism at a Turn or River chance node.
///
/// When dealing the next street card from `live_cards`:
/// - `representative_cards`: the canonical subset of chance cards that must actually be solved.
/// - `duplicates`: for each skipped chance card `dup_card`, stores `(rep_index_in_representatives, s1, s2)`
///   meaning `dup_card` is obtained from `representative_cards[rep_idx]` by transposing suits `s1 <-> s2`.
#[derive(Clone, Debug)]
pub struct ChanceIsomorphism {
    pub representative_cards: Vec<Card>,
    pub duplicates: Vec<ChanceDuplicate>,
}

#[derive(Clone, Debug)]
pub struct ChanceDuplicate {
    pub card: Card,
    pub rep_index: usize,
    pub suit_a: u8,
    pub suit_b: u8,
}

/// Computes exact lossless chance-card isomorphism for dealing the next card after `board`,
/// given OOP and IP sparse `(ComboId, weight)` ranges.
///
/// Two suits `s1 < s2` are interchangeable at this chance node if and only if:
/// 1. `suits_symmetric_on_board(board, s1, s2)` holds; AND
/// 2. Both OOP and IP initial ranges are symmetric under `s1 <-> s2`.
///
/// Under those conditions, dealing `(rank, s2)` is 100% strategically isomorphic to dealing `(rank, s1)`
/// with player hand indices permuted by `s1 <-> s2`!
pub fn compute_chance_isomorphism(
    board: &Board,
    oop_combos: &[(ComboId, f32)],
    ip_combos: &[(ComboId, f32)],
) -> ChanceIsomorphism {
    // Partition suits 0..4 into equivalence classes
    let mut canonical_suit = [0u8, 1, 2, 3];
    for s2 in 1u8..4 {
        for s1 in 0..s2 {
            if suits_symmetric_on_board(board, s1, s2)
                && is_weighted_combo_list_suit_symmetric(oop_combos, s1, s2, 1e-6)
                && is_weighted_combo_list_suit_symmetric(ip_combos, s1, s2, 1e-6)
            {
                canonical_suit[s2 as usize] = canonical_suit[s1 as usize];
                break;
            }
        }
    }

    let mut representative_cards = Vec::with_capacity(48);
    let mut duplicates = Vec::new();
    let mut card_to_rep_idx = [usize::MAX; 52];

    for id in 0..52u8 {
        let card = Card::new(id).unwrap();
        if board.mask().contains(card) {
            continue;
        }
        let s = card.suit();
        let cs = canonical_suit[s as usize];
        if cs == s {
            let idx = representative_cards.len();
            representative_cards.push(card);
            card_to_rep_idx[id as usize] = idx;
        } else {
            let rep_card = Card::from_rank_suit(card.rank(), cs).unwrap();
            let rep_index = card_to_rep_idx[rep_card.id() as usize];
            debug_assert_ne!(rep_index, usize::MAX);
            duplicates.push(ChanceDuplicate {
                card,
                rep_index,
                suit_a: cs,
                suit_b: s,
            });
        }
    }

    ChanceIsomorphism {
        representative_cards,
        duplicates,
    }
}

/// Builds the index swap list `(i, j)` (`i < j`) for permuting a player's sorted `ComboId` slice
/// under suit transposition `suit_a <-> suit_b`.
///
/// Only valid when the player's combo set is closed under `suit_a <-> suit_b` (which is guaranteed
/// whenever `compute_chance_isomorphism` marks `suit_a` and `suit_b` equivalent).
pub fn build_combo_swap_pairs(combos: &[ComboId], suit_a: u8, suit_b: u8) -> Vec<(u32, u32)> {
    if suit_a == suit_b {
        return Vec::new();
    }
    let perm = suit_transposition(suit_a, suit_b);
    let mut swaps = Vec::new();
    for (idx, &cid) in combos.iter().enumerate() {
        let mapped = permute_hand4(cid.to_hand(), perm).combo_id();
        if mapped > cid {
            if let Ok(other_idx) = combos.binary_search(&mapped) {
                swaps.push((idx as u32, other_idx as u32));
            }
        }
    }
    swaps
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn exact_16432_preflop_plo_canonical_classes() {
        // Mathematical theorem for 4-card Omaha:
        // Burnside's lemma on C(52, 4) = 270,725 under S_4 (24 suit permutations) yields
        // EXACTLY 16,432 canonical 4-card starting hands!
        // Let's verify our canonicalizer produces literally 16,432 distinct classes!
        let mut classes = HashSet::with_capacity(16_500);
        for (_, hand) in Hand4::all_hands() {
            let (canon, _) = canonicalize_preflop_hand(hand);
            classes.insert(canon);
        }
        assert_eq!(classes.len(), 16_432);
    }

    #[test]
    fn non_equivalent_suited_plo_hands_never_merge() {
        // 1) [As Ks Qd Jc] (Ace-King suited) vs [Ad Ks Qs Jc] (King-Queen suited)
        let ak_ss: Hand4 = "AsKsQdJc".parse().unwrap();
        let kq_ss: Hand4 = "AdKsQsJc".parse().unwrap();
        assert_ne!(
            canonicalize_preflop_hand(ak_ss).0,
            canonicalize_preflop_hand(kq_ss).0
        );

        // 2) [As Ks Qh Jh] ({A,K} suited + {Q,J} suited)
        // vs [As Qs Kh Jh] ({A,Q} suited + {K,J} suited)
        // vs [As Js Kh Qh] ({A,J} suited + {K,Q} suited)
        let ds_ak_qj: Hand4 = "AsKsQhJh".parse().unwrap();
        let ds_aq_kj: Hand4 = "AsQsKhJh".parse().unwrap();
        let ds_aj_kq: Hand4 = "AsJsKhQh".parse().unwrap();
        // Meanwhile [Ah Kh Qs Js] IS isomorphic to [As Ks Qh Jh]
        let ds_ak_qj_swapped: Hand4 = "AhKhQsJs".parse().unwrap();

        let c1 = canonicalize_preflop_hand(ds_ak_qj).0;
        let c2 = canonicalize_preflop_hand(ds_aq_kj).0;
        let c3 = canonicalize_preflop_hand(ds_aj_kq).0;
        let c1_swap = canonicalize_preflop_hand(ds_ak_qj_swapped).0;

        assert_eq!(c1, c1_swap);
        assert_ne!(c1, c2);
        assert_ne!(c1, c3);
        assert_ne!(c2, c3);

        // 3) Trips-suited vs double-suited vs monotone
        assert_eq!(
            classify_suit_pattern("AsKsQsJh".parse().unwrap()),
            SuitPattern4::TripsSuited
        );
        assert_eq!(
            classify_suit_pattern("AsKsQsJs".parse().unwrap()),
            SuitPattern4::Monotone
        );
        assert_eq!(
            classify_suit_pattern("AsKdQhJc".parse().unwrap()),
            SuitPattern4::Rainbow
        );
    }

    #[test]
    fn board_and_hand_joint_isomorphism_respects_flush_suits() {
        let board_spade_draw: Board = "Ks9s2d".parse().unwrap();
        let board_heart_draw: Board = "Kh9h2c".parse().unwrap();

        // Nut spade draw on Ks9s2d == Nut heart draw on Kh9h2c
        let nut_spade: Hand4 = "AsQsJdTc".parse().unwrap();
        let nut_heart: Hand4 = "AhQhJcTd".parse().unwrap();
        let (ch1, cb1, _) = canonicalize_hand_with_board(nut_spade, &board_spade_draw);
        let (ch2, cb2, _) = canonicalize_hand_with_board(nut_heart, &board_heart_draw);
        assert_eq!((ch1, cb1), (ch2, cb2));

        // Backdoor clubs vs frontdoor spades on Ks9s2d MUST NOT merge!
        let backdoor: Hand4 = "AcQcJdTd".parse().unwrap();
        let (ch_bd, cb_bd, _) = canonicalize_hand_with_board(backdoor, &board_spade_draw);
        assert_ne!((ch1, cb1), (ch_bd, cb_bd));

        // Backdoor diamond (matches 2d on Ks9s2d!) vs dead offsuit club (no club on Ks9s2d!) MUST NOT merge!
        let bdfd_diamond: Hand4 = "AdQdJhTc".parse().unwrap();
        let dead_club: Hand4 = "AcQcJhTd".parse().unwrap();
        assert_ne!(
            canonicalize_hand_with_board(bdfd_diamond, &board_spade_draw).0,
            canonicalize_hand_with_board(dead_club, &board_spade_draw).0
        );
    }

    #[test]
    fn monotone_flop_chance_isomorphism_folds_three_offsuits() {
        // On a monotone club flop [Ac Kc 7c] with suit-symmetric ranges (empty or full),
        // diamonds (1), hearts (2), and spades (3) are mutually isomorphic!
        // Out of 49 remaining turn cards:
        // - 10 remaining clubs (unique)
        // - 13 diamonds (representatives for d/h/s)
        // - 26 hearts + spades folded into the 13 diamond representatives!
        // Total representatives = 10 + 13 = 23 (instead of 49)!
        let flop: Board = "AcKc7c".parse().unwrap();
        let iso = compute_chance_isomorphism(&flop, &[], &[]);
        assert_eq!(iso.representative_cards.len(), 23);
        assert_eq!(iso.duplicates.len(), 26);
        assert_eq!(iso.representative_cards.len() + iso.duplicates.len(), 49);
    }

    #[test]
    fn phase_b2_b3_board_aware_negative_and_positive_isomorphism_cases() {
        // B2.1: Double-suited hand on a two-tone board (`Ts 9s 4h`):
        // - `[As Ks Qh Jh]` has `{A,K}` in the frontdoor flush-draw suit (spades) and `{Q,J}` in the backdoor suit (hearts).
        // - `[Ah Kh Qs Js]` has `{Q,J}` in the frontdoor flush-draw suit (spades, second-nut/third-nut!) and `{A,K}` in backdoor hearts!
        // They MUST NOT MERGE!
        let two_tone_flop: Board = "Ts9s4h".parse().unwrap();
        let nut_fd_bdfd: Hand4 = "AsKsQhJh".parse().unwrap();
        let qhigh_fd_nut_bdfd: Hand4 = "AhKhQsJs".parse().unwrap();
        assert_ne!(
            canonicalize_hand_with_board(nut_fd_bdfd, &two_tone_flop).0,
            canonicalize_hand_with_board(qhigh_fd_nut_bdfd, &two_tone_flop).0
        );
        // Also vs `[As Ks Qd Jd]` (frontdoor spades + dead offsuit diamonds instead of backdoor hearts!): MUST NOT MERGE!
        let nut_fd_dead_ds: Hand4 = "AsKsQdJd".parse().unwrap();
        let nut_fd_dead_clubs: Hand4 = "AsKsQcJc".parse().unwrap();
        assert_ne!(
            canonicalize_hand_with_board(nut_fd_bdfd, &two_tone_flop).0,
            canonicalize_hand_with_board(nut_fd_dead_ds, &two_tone_flop).0
        );
        // B3 positive: `[As Ks Qd Jd]` and `[As Ks Qc Jc]` on `Ts 9s 4h` DO merge because neither `d` nor `c` appears on `Ts 9s 4h`!
        assert_eq!(
            canonicalize_hand_with_board(nut_fd_dead_ds, &two_tone_flop).0,
            canonicalize_hand_with_board(nut_fd_dead_clubs, &two_tone_flop).0
        );

        // B2.2: `(AK)(QJ)` vs `(AQ)(KJ)` on BOTH rainbow (`9s 6d 2c`) AND suited (`9s 6s 2c`) boards MUST NOT MERGE!
        let rainbow_flop: Board = "9s6d2c".parse().unwrap();
        let suited_flop: Board = "9s6s2c".parse().unwrap();
        let ak_qj: Hand4 = "AsKsQhJh".parse().unwrap();
        let aq_kj: Hand4 = "AsQsKhJh".parse().unwrap();
        assert_ne!(
            canonicalize_hand_with_board(ak_qj, &rainbow_flop).0,
            canonicalize_hand_with_board(aq_kj, &rainbow_flop).0
        );
        assert_ne!(
            canonicalize_hand_with_board(ak_qj, &suited_flop).0,
            canonicalize_hand_with_board(aq_kj, &suited_flop).0
        );

        // B2.3: Nut flush blocker (`[As Kd Qh Jc]` bare Ace of spades!) vs second-nut flush blocker (`[Ah Ks Qd Jc]` bare King of spades!)
        // on a 3-spade river `[Qs Js Ts 4d 2c]`: neither hand has a spade flush (only 1 spade in hand!), both make the exact same
        // A-high broadway straight `A-K-Q-J-T`, YET `[As Kd Qh Jc]` blocks Villain's nut spade flush (`AsXs`) while `[Ah Ks Qd Jc]`
        // only blocks `KsXs`! Board-aware canonicalization MUST NOT MERGE them!
        let three_spade_river: Board = "QsJsTs4d2c".parse().unwrap();
        let nut_spade_blocker: Hand4 = "AsKdQhJc".parse().unwrap();
        let second_nut_spade_blocker: Hand4 = "AhKsQdJc".parse().unwrap();
        let no_spade_blocker: Hand4 = "AhKdQcJc".parse().unwrap();
        assert_ne!(
            canonicalize_hand_with_board(nut_spade_blocker, &three_spade_river).0,
            canonicalize_hand_with_board(second_nut_spade_blocker, &three_spade_river).0
        );
        assert_ne!(
            canonicalize_hand_with_board(nut_spade_blocker, &three_spade_river).0,
            canonicalize_hand_with_board(no_spade_blocker, &three_spade_river).0
        );

        // B3 positive: Isomorphic turn/river runouts with symmetric unused suits!
        // On rainbow flop `Ks 9d 4c`? Wait: on `Ks 9d 4c`, `h` is absent; on paired rainbow `Ks Kd 4c`? No, `s` and `d` both have `K`!
        // Wait: on `Ks Kd 4c`, swapping `s <-> d` swaps `Ks <-> Kd`, leaving the flop set `{Ks, Kd, 4c}` IDENTICAL!
        // So dealing `7s` vs `7d` on `Ks Kd 4c` IS isomorphic (`[As Qs Jh Th]` on `Ks Kd 4c 7s` == `[Ad Qd Jh Th]` on `Ks Kd 4c 7d`)!
        let paired_flop_turn_s: Board = "KsKd4c7s".parse().unwrap();
        let paired_flop_turn_d: Board = "KsKd4c7d".parse().unwrap();
        let hs: Hand4 = "AsQsJhTh".parse().unwrap();
        let hd: Hand4 = "AdQdJhTh".parse().unwrap();
        assert_eq!(
            canonicalize_hand_with_board(hs, &paired_flop_turn_s).0,
            canonicalize_hand_with_board(hd, &paired_flop_turn_d).0
        );
        assert_eq!(
            canonicalize_hand_with_board(hs, &paired_flop_turn_s).1,
            canonicalize_hand_with_board(hd, &paired_flop_turn_d).1
        );

        // B2 negative: Asymmetric player ranges prevent chance isomorphism from merging board-symmetric suits!
        let turn_board: Board = "AsKs7c2c".parse().unwrap(); // `d` (1) and `h` (2) symmetric on board
        let asym_oop = vec![("AdKdQhJh".parse::<Hand4>().unwrap().combo_id(), 1.0f32)]; // NOT symmetric under d <-> h!
        let sym_ip = Vec::new();
        let iso_asym = compute_chance_isomorphism(&turn_board, &asym_oop, &sym_ip);
        assert_eq!(
            iso_asym.representative_cards.len(),
            48,
            "asymmetric range must disable suit folding (all 48 river cards kept separate)"
        );
        assert!(iso_asym.duplicates.is_empty());
    }
}
