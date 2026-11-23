//! Dedicated Pot-Limit Omaha (`PLO-4`) hand evaluator.
//!
//! # PLO Evaluation Invariant (STRICT)
//! Every evaluated 5-card hand MUST consist of **EXACTLY 2 hole cards** and **EXACTLY 3 board cards**:
//! - Flop (`3` board cards): `C(4,2) * C(3,3) = 6` candidate 5-card hands.
//! - Turn (`4` board cards): `C(4,2) * C(4,3) = 24` candidate 5-card hands.
//! - River (`5` board cards): `C(4,2) * C(5,3) = 60` candidate 5-card hands.
//!
//! No 7-card Hold'em evaluator ("best 5 of 7") exists anywhere in this crate.
//!
//! # Output Ranking (`HandRank`)
//! Every valid 5-card hand maps to an exact equivalence class `HandRank(u16)` in `1..=7462`,
//! where higher is strictly stronger and equal values split the pot:
//! - `1..=1277`: High Card (`1277` classes)
//! - `1278..=4137`: One Pair (`2860` classes)
//! - `4138..=4995`: Two Pair (`858` classes)
//! - `4996..=5853`: Three of a Kind (`858` classes)
//! - `5854..=5863`: Straight (`10` classes, `5-high` wheel to `A-high` broadway)
//! - `5864..=7140`: Flush (`1277` classes)
//! - `7141..=7296`: Full House (`156` classes)
//! - `7297..=7452`: Four of a Kind (`156` classes)
//! - `7453..=7462`: Straight Flush (`10` classes, `5-high` steel wheel to Royal Flush `7462`)

use core::fmt;
use std::sync::OnceLock;

use plo_cards::{Board, Card, CardError, Hand4};

/// Distinct prime assigned to each rank `0..13` (`2..=A`), enabling unique multiset products
/// for all non-flush 5-card rank combinations (`max product = 41^4 * 37 = 104,553,157 < u32::MAX`).
const RANK_PRIMES: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// Precomputed per-card descriptor (`8` bytes per card, `416` bytes total for 52 cards).
///
/// Layout:
/// - `prime`: rank prime (`2..=41`)
/// - `rank_bit`: `1u16 << rank` (`0..13`)
/// - `suit_bit`: `1u8 << suit` (`0..4`), so bitwise AND across 5 cards is non-zero iff all 5 share a suit!
#[derive(Clone, Copy)]
struct CardInfo {
    prime: u32,
    rank_bit: u16,
    suit_bit: u8,
}

const CARD_INFO: [CardInfo; 52] = build_card_info();

const fn build_card_info() -> [CardInfo; 52] {
    let mut arr = [CardInfo {
        prime: 2,
        rank_bit: 1,
        suit_bit: 1,
    }; 52];
    let mut id = 0usize;
    while id < 52 {
        let rank = id >> 2;
        let suit = id & 3;
        arr[id] = CardInfo {
            prime: RANK_PRIMES[rank],
            rank_bit: 1u16 << rank,
            suit_bit: 1u8 << suit,
        };
        id += 1;
    }
    arr
}

/// All `C(4, 2) = 6` hole-card index pairs.
pub const HOLE_INDEX_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// All `C(4, 3) = 4` turn board index triples.
pub const TURN_BOARD_TRIPLES: [(usize, usize, usize); 4] =
    [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)];

/// All `C(5, 3) = 10` river board index triples.
pub const RIVER_BOARD_TRIPLES: [(usize, usize, usize); 10] = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 1, 4),
    (0, 2, 3),
    (0, 2, 4),
    (0, 3, 4),
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (2, 3, 4),
];

/// High-level poker hand category (`HighCard < OnePair < ... < StraightFlush`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum HandCategory {
    HighCard = 0,
    OnePair = 1,
    TwoPair = 2,
    ThreeOfAKind = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfAKind = 7,
    StraightFlush = 8,
}

/// Comparable 5-card poker hand strength in `1..=7462` (`higher == stronger`, equal ties chop).
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct HandRank(pub u16);

impl HandRank {
    pub const MIN: Self = Self(1);
    pub const MAX: Self = Self(7462);

    #[inline]
    pub const fn raw(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn category(self) -> HandCategory {
        debug_assert!(
            self.0 >= 1 && self.0 <= 7462,
            "HandRank out of valid 1..=7462 range"
        );
        match self.0 {
            1..=1277 => HandCategory::HighCard,
            1278..=4137 => HandCategory::OnePair,
            4138..=4995 => HandCategory::TwoPair,
            4996..=5853 => HandCategory::ThreeOfAKind,
            5854..=5863 => HandCategory::Straight,
            5864..=7140 => HandCategory::Flush,
            7141..=7296 => HandCategory::FullHouse,
            7297..=7452 => HandCategory::FourOfAKind,
            7453..=7462 => HandCategory::StraightFlush,
            _ => HandCategory::HighCard,
        }
    }
}

impl fmt::Debug for HandRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HandRank({}: {:?})", self.0, self.category())
    }
}

// ============================================================================
// Reference / Slow Exact 5-card & PLO 2+3 Evaluator (Zero Tables, Pure Logic)
// ============================================================================

/// Computes a strictly monotonic lexicographic score `(category, k0, k1, k2, k3, k4)` for a 5-card hand
/// directly from first principles, without using any lookup tables.
#[inline]
pub fn slow_eval_5_lex_key(cards: [Card; 5]) -> u32 {
    let mut rank_counts = [0u8; 13];
    let first_suit = cards[0].suit();
    let mut is_flush = true;
    let mut rank_mask = 0u16;

    for c in cards {
        let r = c.rank() as usize;
        rank_counts[r] += 1;
        rank_mask |= 1u16 << r;
        if c.suit() != first_suit {
            is_flush = false;
        }
    }

    let straight_high = detect_straight_high(rank_mask);

    if is_flush {
        if let Some(sh) = straight_high {
            return pack_lex(HandCategory::StraightFlush as u8, [sh, 0, 0, 0, 0]);
        }
        let kickers = top_ranks_by_count(&rank_counts);
        return pack_lex(HandCategory::Flush as u8, kickers);
    }

    if let Some(sh) = straight_high {
        return pack_lex(HandCategory::Straight as u8, [sh, 0, 0, 0, 0]);
    }

    // Group ranks by multiplicity descending, then by rank descending
    let mut groups: [(u8, u8); 5] = [(0, 0); 5];
    let mut g_len = 0usize;
    for r in (0..13u8).rev() {
        let cnt = rank_counts[r as usize];
        if cnt > 0 {
            groups[g_len] = (cnt, r);
            g_len += 1;
        }
    }
    // Insertion sort `groups[..g_len]` descending by `(count, rank)`
    for i in 1..g_len {
        let mut j = i;
        while j > 0 && groups[j] > groups[j - 1] {
            groups.swap(j, j - 1);
            j -= 1;
        }
    }

    match (groups[0].0, groups[1].0) {
        (4, 1) => pack_lex(
            HandCategory::FourOfAKind as u8,
            [groups[0].1, groups[1].1, 0, 0, 0],
        ),
        (3, 2) => pack_lex(
            HandCategory::FullHouse as u8,
            [groups[0].1, groups[1].1, 0, 0, 0],
        ),
        (3, 1) => pack_lex(
            HandCategory::ThreeOfAKind as u8,
            [groups[0].1, groups[1].1, groups[2].1, 0, 0],
        ),
        (2, 2) => pack_lex(
            HandCategory::TwoPair as u8,
            [groups[0].1, groups[1].1, groups[2].1, 0, 0],
        ),
        (2, 1) => pack_lex(
            HandCategory::OnePair as u8,
            [groups[0].1, groups[1].1, groups[2].1, groups[3].1, 0],
        ),
        _ => pack_lex(
            HandCategory::HighCard as u8,
            [
                groups[0].1,
                groups[1].1,
                groups[2].1,
                groups[3].1,
                groups[4].1,
            ],
        ),
    }
}

#[inline]
const fn pack_lex(cat: u8, k: [u8; 5]) -> u32 {
    ((cat as u32) << 20)
        | ((k[0] as u32) << 16)
        | ((k[1] as u32) << 12)
        | ((k[2] as u32) << 8)
        | ((k[3] as u32) << 4)
        | (k[4] as u32)
}

#[inline]
const fn detect_straight_high(rank_mask: u16) -> Option<u8> {
    if rank_mask.count_ones() != 5 {
        return None;
    }
    let m = rank_mask & (rank_mask << 1) & (rank_mask << 2) & (rank_mask << 3) & (rank_mask << 4);
    if m != 0 {
        Some((15 - m.leading_zeros()) as u8)
    } else if rank_mask == 0b1_0000_0000_1111 {
        // A-2-3-4-5 wheel: 5-high straight (`rank('5') == 3`)
        Some(3)
    } else {
        None
    }
}

#[inline]
fn top_ranks_by_count(rank_counts: &[u8; 13]) -> [u8; 5] {
    let mut out = [0u8; 5];
    let mut idx = 0;
    for r in (0..13u8).rev() {
        for _ in 0..rank_counts[r as usize] {
            out[idx] = r;
            idx += 1;
        }
    }
    out
}

/// Slow, crystal-clear brute-force PLO evaluator that enumerates every `C(4,2) * C(B,3)`
/// 2-hole + 3-board combination and computes the maximum lexicographic score without tables.
///
/// Used as an independent oracle to test the fast lookup evaluator across 100k+ random hands.
pub fn slow_eval_plo_lex(hand: Hand4, board: &Board) -> Result<(u32, usize), CardError> {
    if hand.is_blocked_by(board.mask()) {
        return Err(CardError::BoardOverlap);
    }
    let h = hand.cards();
    let b = board.as_slice();
    let mut best = 0u32;
    let mut candidates = 0usize;

    for &(i0, i1) in &HOLE_INDEX_PAIRS {
        for j0 in 0..b.len() {
            for j1 in (j0 + 1)..b.len() {
                for j2 in (j1 + 1)..b.len() {
                    let score = slow_eval_5_lex_key([h[i0], h[i1], b[j0], b[j1], b[j2]]);
                    if score > best {
                        best = score;
                    }
                    candidates += 1;
                }
            }
        }
    }
    Ok((best, candidates))
}

// ============================================================================
// Fast Lookup Table Evaluator (O(1) 5-card kernel + specialized 2+3 PLO loops)
// ============================================================================

/// Open-addressed power-of-two hash table mapping paired/two-pair/trips/full-house/quads
/// 5-card prime products (`4,888` distinct products in `8,192` slots, load factor `~0.60`,
/// `64 KiB` total) to `HandRank(u16)` in `O(1)`.
const HASH_BITS: u32 = 13;
const HASH_SIZE: usize = 1 << HASH_BITS; // 8192 slots * 8 bytes = 64 KiB
const HASH_MASK: usize = HASH_SIZE - 1;
const FIB_MULT: u32 = 2_654_435_769;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct PrimeSlot {
    product: u32,
    rank: u16,
    _pad: u16,
}

struct EvalTables {
    /// Indexed by 13-bit rank mask (`0..8192`) for 5 suited cards (`Flush` or `StraightFlush`).
    flush_lookup: [u16; 8192],
    /// Indexed by 13-bit rank mask (`0..8192`) for 5 distinct non-suited ranks (`HighCard` or `Straight`).
    unique5_lookup: [u16; 8192],
    /// Open-addressed table for paired/two-pair/trips/full-house/quads prime products (`4,888` entries in `8,192` slots).
    prime_hash: [PrimeSlot; HASH_SIZE],
}

static TABLES: OnceLock<Box<EvalTables>> = OnceLock::new();

#[inline]
fn tables() -> &'static EvalTables {
    TABLES.get_or_init(build_eval_tables)
}

/// Eagerly initializes the `plo_eval` lookup tables (~96 KB total).
pub fn init_tables() {
    let _ = tables();
}

fn build_eval_tables() -> Box<EvalTables> {
    let mut flush_lookup = [0u16; 8192];
    let mut unique5_lookup = [0u16; 8192];

    // Enumerate all valid 5-card rank multisets (with count <= 4 per rank):
    // C(13+5-1, 5) - 13 (five-of-a-kind) = 6188 - 13 = 6175 non-flush + 1287 flush = 7462 exact classes!
    struct ClassEntry {
        lex_key: u32,
        is_flush: bool,
        rank_mask: u16,
        prime_prod: u32,
    }

    let mut entries = Vec::with_capacity(7462);

    // 1) All C(13, 5) = 1287 5-distinct-rank masks: each produces one non-flush class and one flush class.
    for r4 in 4u8..13 {
        for r3 in 3..r4 {
            for r2 in 2..r3 {
                for r1 in 1..r2 {
                    for r0 in 0..r1 {
                        let mask = (1u16 << r0)
                            | (1u16 << r1)
                            | (1u16 << r2)
                            | (1u16 << r3)
                            | (1u16 << r4);
                        let prod = RANK_PRIMES[r0 as usize]
                            * RANK_PRIMES[r1 as usize]
                            * RANK_PRIMES[r2 as usize]
                            * RANK_PRIMES[r3 as usize]
                            * RANK_PRIMES[r4 as usize];

                        let non_flush_cards = [
                            Card::from_rank_suit(r0, 0).unwrap(),
                            Card::from_rank_suit(r1, 1).unwrap(),
                            Card::from_rank_suit(r2, 2).unwrap(),
                            Card::from_rank_suit(r3, 3).unwrap(),
                            Card::from_rank_suit(r4, 0).unwrap(),
                        ];
                        entries.push(ClassEntry {
                            lex_key: slow_eval_5_lex_key(non_flush_cards),
                            is_flush: false,
                            rank_mask: mask,
                            prime_prod: prod,
                        });

                        let flush_cards = [
                            Card::from_rank_suit(r0, 0).unwrap(),
                            Card::from_rank_suit(r1, 0).unwrap(),
                            Card::from_rank_suit(r2, 0).unwrap(),
                            Card::from_rank_suit(r3, 0).unwrap(),
                            Card::from_rank_suit(r4, 0).unwrap(),
                        ];
                        entries.push(ClassEntry {
                            lex_key: slow_eval_5_lex_key(flush_cards),
                            is_flush: true,
                            rank_mask: mask,
                            prime_prod: prod,
                        });
                    }
                }
            }
        }
    }

    // 2) All paired/trips/full-house/quads rank multisets (non-flush only):
    // r0 <= r1 <= r2 <= r3 <= r4, not all distinct, and not all 5 equal.
    for r4 in 0u8..13 {
        for r3 in 0..=r4 {
            for r2 in 0..=r3 {
                for r1 in 0..=r2 {
                    for r0 in 0..=r1 {
                        if r0 < r1 && r1 < r2 && r2 < r3 && r3 < r4 {
                            continue; // already added above
                        }
                        if r0 == r4 {
                            continue; // 5-of-a-kind impossible in 52-card deck
                        }
                        let rs = [r0, r1, r2, r3, r4];
                        let mut counts = [0u8; 13];
                        let mut prod = 1u32;
                        let mut mask = 0u16;
                        for &r in &rs {
                            counts[r as usize] += 1;
                            prod *= RANK_PRIMES[r as usize];
                            mask |= 1u16 << r;
                        }
                        // Assign suits 0..count[r] for each rank so cards are valid and never 5-suited
                        let mut used_per_rank = [0u8; 13];
                        let mut cards = [Card::new(0).unwrap(); 5];
                        for (idx, &r) in rs.iter().enumerate() {
                            let s = used_per_rank[r as usize];
                            used_per_rank[r as usize] += 1;
                            cards[idx] = Card::from_rank_suit(r, s).unwrap();
                        }
                        // Ensure not accidentally all suit 0 (impossible since at least two cards share a rank)
                        entries.push(ClassEntry {
                            lex_key: slow_eval_5_lex_key(cards),
                            is_flush: false,
                            rank_mask: mask,
                            prime_prod: prod,
                        });
                    }
                }
            }
        }
    }

    debug_assert_eq!(entries.len(), 7462);
    entries.sort_unstable_by_key(|e| e.lex_key);

    let mut prime_hash = [PrimeSlot::default(); HASH_SIZE];

    for (idx, e) in entries.iter().enumerate() {
        let hand_rank = (idx + 1) as u16;
        if e.is_flush {
            flush_lookup[e.rank_mask as usize] = hand_rank;
        } else if e.rank_mask.count_ones() == 5 {
            unique5_lookup[e.rank_mask as usize] = hand_rank;
        } else {
            let mut slot = ((e.prime_prod.wrapping_mul(FIB_MULT)) >> (32 - HASH_BITS)) as usize;
            while prime_hash[slot].product != 0 {
                slot = (slot + 1) & HASH_MASK;
            }
            prime_hash[slot] = PrimeSlot {
                product: e.prime_prod,
                rank: hand_rank,
                _pad: 0,
            };
        }
    }

    Box::new(EvalTables {
        flush_lookup,
        unique5_lookup,
        prime_hash,
    })
}

#[inline(always)]
fn lookup_prime(prime_hash: &[PrimeSlot; HASH_SIZE], prod: u32) -> u16 {
    let mut slot = ((prod.wrapping_mul(FIB_MULT)) >> (32 - HASH_BITS)) as usize;
    loop {
        let entry = prime_hash[slot];
        if entry.product == prod {
            return entry.rank;
        }
        assert!(
            entry.product != 0,
            "unrecognized 5-card prime product (duplicate or invalid cards passed to evaluator)"
        );
        slot = (slot + 1) & HASH_MASK;
    }
}

/// Evaluates a 5-card poker hand in `O(1)` with zero heap allocations.
///
/// # Panics
/// Panics if `cards` contains duplicate cards (`count_ones() != 5`).
#[inline]
pub fn eval_5_cards(cards: [Card; 5]) -> HandRank {
    let mask = cards[0].mask()
        | cards[1].mask()
        | cards[2].mask()
        | cards[3].mask()
        | cards[4].mask();
    assert_eq!(
        mask.len(),
        5,
        "eval_5_cards requires 5 distinct cards, got duplicates"
    );
    let tbl = tables();
    eval_5_with_tables(tbl, cards)
}

#[inline(always)]
fn eval_5_with_tables(tbl: &EvalTables, cards: [Card; 5]) -> HandRank {
    let i0 = CARD_INFO[cards[0].id() as usize];
    let i1 = CARD_INFO[cards[1].id() as usize];
    let i2 = CARD_INFO[cards[2].id() as usize];
    let i3 = CARD_INFO[cards[3].id() as usize];
    let i4 = CARD_INFO[cards[4].id() as usize];

    let rank_mask = i0.rank_bit | i1.rank_bit | i2.rank_bit | i3.rank_bit | i4.rank_bit;
    let suit_and = i0.suit_bit & i1.suit_bit & i2.suit_bit & i3.suit_bit & i4.suit_bit;

    if suit_and != 0 {
        return HandRank(tbl.flush_lookup[rank_mask as usize]);
    }

    let u = tbl.unique5_lookup[rank_mask as usize];
    if u != 0 {
        return HandRank(u);
    }

    let prod = i0.prime * i1.prime * i2.prime * i3.prime * i4.prime;
    HandRank(lookup_prime(&tbl.prime_hash, prod))
}

#[derive(Clone, Copy)]
struct PartialDesc {
    prime: u32,
    rank_mask: u16,
    suit_and: u8,
}

#[inline(always)]
fn pair_desc(c0: Card, c1: Card) -> PartialDesc {
    let a = CARD_INFO[c0.id() as usize];
    let b = CARD_INFO[c1.id() as usize];
    PartialDesc {
        prime: a.prime * b.prime,
        rank_mask: a.rank_bit | b.rank_bit,
        suit_and: a.suit_bit & b.suit_bit,
    }
}

#[inline(always)]
fn triple_desc(c0: Card, c1: Card, c2: Card) -> PartialDesc {
    let a = CARD_INFO[c0.id() as usize];
    let b = CARD_INFO[c1.id() as usize];
    let c = CARD_INFO[c2.id() as usize];
    PartialDesc {
        prime: a.prime * b.prime * c.prime,
        rank_mask: a.rank_bit | b.rank_bit | c.rank_bit,
        suit_and: a.suit_bit & b.suit_bit & c.suit_bit,
    }
}

#[inline(always)]
fn eval_2_plus_3(tbl: &EvalTables, hole2: PartialDesc, board3: PartialDesc) -> u16 {
    let rank_mask = hole2.rank_mask | board3.rank_mask;
    if (hole2.suit_and & board3.suit_and) != 0 {
        return tbl.flush_lookup[rank_mask as usize];
    }
    let u = tbl.unique5_lookup[rank_mask as usize];
    if u != 0 {
        return u;
    }
    lookup_prime(&tbl.prime_hash, hole2.prime * board3.prime)
}

/// Pre-decoded 5-card river board (`10` legal 3-card board subsets).
///
/// Precomputing `RiverBoardCache` once for a river board accelerates evaluating
/// an entire PLO range (`N` hands) by avoiding redundant board-triple decoding.
#[derive(Clone, Copy)]
pub struct RiverBoardCache {
    triples: [PartialDesc; 10],
    mask: plo_cards::CardMask,
}

impl RiverBoardCache {
    #[inline]
    pub fn new(board: [Card; 5]) -> Self {
        let mut mask = plo_cards::CardMask::EMPTY;
        for &c in &board {
            mask.insert(c);
        }
        let mut triples = [PartialDesc {
            prime: 1,
            rank_mask: 0,
            suit_and: 0,
        }; 10];
        for (idx, &(i, j, k)) in RIVER_BOARD_TRIPLES.iter().enumerate() {
            triples[idx] = triple_desc(board[i], board[j], board[k]);
        }
        Self { triples, mask }
    }

    #[inline]
    pub const fn mask(&self) -> plo_cards::CardMask {
        self.mask
    }

    /// Evaluates a `Hand4` (caller guarantees `!hand.is_blocked_by(self.mask())`) in 60 register-only steps.
    #[inline]
    pub fn eval_hand_unchecked(&self, hand: Hand4) -> HandRank {
        let tbl = tables();
        let [c0, c1, c2, c3] = hand.cards();
        let holes = [
            pair_desc(c0, c1),
            pair_desc(c0, c2),
            pair_desc(c0, c3),
            pair_desc(c1, c2),
            pair_desc(c1, c3),
            pair_desc(c2, c3),
        ];

        let mut best = 0u16;
        for &hp in &holes {
            for &bt in &self.triples {
                let r = eval_2_plus_3(tbl, hp, bt);
                if r > best {
                    best = r;
                }
            }
        }
        HandRank(best)
    }
}

/// Specialized PLO flop evaluator: evaluates all `C(4,2) * C(3,3) = 6` legal 2+3 combinations.
///
/// # Panics
/// Panics if `flop` contains duplicate cards or overlaps `hand` (use [`eval_plo`] for fallible `Result`).
#[inline]
pub fn eval_plo_flop(hand: Hand4, flop: [Card; 3]) -> HandRank {
    let fmask = flop[0].mask() | flop[1].mask() | flop[2].mask();
    assert_eq!(fmask.len(), 3, "eval_plo_flop: flop has duplicate cards");
    assert!(
        !hand.is_blocked_by(fmask),
        "eval_plo_flop: hand overlaps flop board"
    );
    let tbl = tables();
    let bt = triple_desc(flop[0], flop[1], flop[2]);
    let [c0, c1, c2, c3] = hand.cards();
    let holes = [
        pair_desc(c0, c1),
        pair_desc(c0, c2),
        pair_desc(c0, c3),
        pair_desc(c1, c2),
        pair_desc(c1, c3),
        pair_desc(c2, c3),
    ];

    let mut best = 0u16;
    for &hp in &holes {
        let r = eval_2_plus_3(tbl, hp, bt);
        if r > best {
            best = r;
        }
    }
    HandRank(best)
}

/// Specialized PLO turn evaluator: evaluates all `C(4,2) * C(4,3) = 24` legal 2+3 combinations.
///
/// # Panics
/// Panics if `turn` contains duplicate cards or overlaps `hand` (use [`eval_plo`] for fallible `Result`).
#[inline]
pub fn eval_plo_turn(hand: Hand4, turn: [Card; 4]) -> HandRank {
    let tmask = turn[0].mask() | turn[1].mask() | turn[2].mask() | turn[3].mask();
    assert_eq!(tmask.len(), 4, "eval_plo_turn: turn has duplicate cards");
    assert!(
        !hand.is_blocked_by(tmask),
        "eval_plo_turn: hand overlaps turn board"
    );
    let tbl = tables();
    let triples = [
        triple_desc(turn[0], turn[1], turn[2]),
        triple_desc(turn[0], turn[1], turn[3]),
        triple_desc(turn[0], turn[2], turn[3]),
        triple_desc(turn[1], turn[2], turn[3]),
    ];
    let [c0, c1, c2, c3] = hand.cards();
    let holes = [
        pair_desc(c0, c1),
        pair_desc(c0, c2),
        pair_desc(c0, c3),
        pair_desc(c1, c2),
        pair_desc(c1, c3),
        pair_desc(c2, c3),
    ];

    let mut best = 0u16;
    for &hp in &holes {
        for &bt in &triples {
            let r = eval_2_plus_3(tbl, hp, bt);
            if r > best {
                best = r;
            }
        }
    }
    HandRank(best)
}

/// Specialized PLO river evaluator: evaluates all `C(4,2) * C(5,3) = 60` legal 2+3 combinations.
///
/// # Panics
/// Panics if `river` contains duplicate cards or overlaps `hand` (use [`eval_plo`] for fallible `Result`).
#[inline]
pub fn eval_plo_river(hand: Hand4, river: [Card; 5]) -> HandRank {
    let rmask =
        river[0].mask() | river[1].mask() | river[2].mask() | river[3].mask() | river[4].mask();
    assert_eq!(rmask.len(), 5, "eval_plo_river: river has duplicate cards");
    assert!(
        !hand.is_blocked_by(rmask),
        "eval_plo_river: hand overlaps river board"
    );
    RiverBoardCache::new(river).eval_hand_unchecked(hand)
}

/// Evaluates a `Hand4` on a 3-, 4-, or 5-card `Board`, returning `Err(CardError::BoardOverlap)`
/// if any hole card duplicates a board card.
#[inline]
pub fn eval_plo(hand: Hand4, board: &Board) -> Result<HandRank, CardError> {
    if hand.is_blocked_by(board.mask()) {
        return Err(CardError::BoardOverlap);
    }
    let b = board.as_slice();
    Ok(match b.len() {
        3 => eval_plo_flop(hand, [b[0], b[1], b[2]]),
        4 => eval_plo_turn(hand, [b[0], b[1], b[2], b[3]]),
        5 => eval_plo_river(hand, [b[0], b[1], b[2], b[3], b[4]]),
        n => return Err(CardError::WrongBoardLength(n)),
    })
}

/// Outcome of a heads-up PLO showdown on a complete board.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShowdownOutcome {
    Player0Wins,
    Player1Wins,
    Split,
}

/// Compares two 4-card PLO hands on a board (`Flop`, `Turn`, or `River`), returning an error if
/// either hand overlaps the board or the two hands share any hole card.
#[inline]
pub fn showdown_heads_up(
    hand0: Hand4,
    hand1: Hand4,
    board: &Board,
) -> Result<ShowdownOutcome, CardError> {
    if hand0.is_blocked_by(board.mask())
        || hand1.is_blocked_by(board.mask())
        || hand0.is_blocked_by(hand1.mask())
    {
        return Err(CardError::BoardOverlap);
    }
    let r0 = eval_plo(hand0, board)?;
    let r1 = eval_plo(hand1, board)?;
    Ok(match r0.cmp(&r1) {
        core::cmp::Ordering::Greater => ShowdownOutcome::Player0Wins,
        core::cmp::Ordering::Less => ShowdownOutcome::Player1Wins,
        core::cmp::Ordering::Equal => ShowdownOutcome::Split,
    })
}

/// Evaluates an N-way PLO showdown, returning the winning player indices (`len == 1` for sole winner,
/// `len > 1` for split pot) and the winning `HandRank`.
pub fn showdown_multiway(hands: &[Hand4], board: &Board) -> Result<(Vec<usize>, HandRank), CardError> {
    let mut used = board.mask();
    let mut best = HandRank(0);
    let mut winners = Vec::new();

    for (idx, &h) in hands.iter().enumerate() {
        if h.is_blocked_by(used) {
            return Err(CardError::BoardOverlap);
        }
        used |= h.mask();
        let rank = eval_plo(h, board)?;
        if rank > best {
            best = rank;
            winners.clear();
            winners.push(idx);
        } else if rank == best {
            winners.push(idx);
        }
    }
    Ok((winners, best))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn all_2598960_five_card_hands_exact_class_counts() {
        // Exhaustively verify all C(52, 5) = 2,598,960 5-card hands!
        let tbl = tables();
        let mut cat_counts = [0usize; 9];

        for c4 in 4u8..52 {
            for c3 in 3..c4 {
                for c2 in 2..c3 {
                    for c1 in 1..c2 {
                        for c0 in 0..c1 {
                            let cards = [
                                Card::new(c0).unwrap(),
                                Card::new(c1).unwrap(),
                                Card::new(c2).unwrap(),
                                Card::new(c3).unwrap(),
                                Card::new(c4).unwrap(),
                            ];
                            let rank = eval_5_with_tables(tbl, cards);
                            assert!((1..=7462).contains(&rank.raw()));
                            cat_counts[rank.category() as usize] += 1;
                        }
                    }
                }
            }
        }

        // Exact combinatorial counts of 5-card poker hands out of C(52, 5) = 2,598,960:
        assert_eq!(cat_counts[HandCategory::HighCard as usize], 1_302_540);
        assert_eq!(cat_counts[HandCategory::OnePair as usize], 1_098_240);
        assert_eq!(cat_counts[HandCategory::TwoPair as usize], 123_552);
        assert_eq!(cat_counts[HandCategory::ThreeOfAKind as usize], 54_912);
        assert_eq!(cat_counts[HandCategory::Straight as usize], 10_200);
        assert_eq!(cat_counts[HandCategory::Flush as usize], 5_108);
        assert_eq!(cat_counts[HandCategory::FullHouse as usize], 3_744);
        assert_eq!(cat_counts[HandCategory::FourOfAKind as usize], 624);
        assert_eq!(cat_counts[HandCategory::StraightFlush as usize], 40);
        assert_eq!(cat_counts.iter().sum::<usize>(), 2_598_960);
    }

    #[test]
    fn plo_candidate_counts_6_24_60() {
        let hand: Hand4 = "AsKsQhJh".parse().unwrap();
        let flop: Board = "Ts9s2d".parse().unwrap();
        let turn: Board = "Ts9s2d3c".parse().unwrap();
        let river: Board = "Ts9s2d3c4h".parse().unwrap();

        assert_eq!(slow_eval_plo_lex(hand, &flop).unwrap().1, 6);
        assert_eq!(slow_eval_plo_lex(hand, &turn).unwrap().1, 24);
        assert_eq!(slow_eval_plo_lex(hand, &river).unwrap().1, 60);
    }

    #[test]
    fn plo_mandatory_2_hole_3_board_traps_where_holdem_fails() {
        // TRAP 1: Four spades on the board (Ks Qs Js Ts 2d), player holds [As 9c 8c 7c].
        // In Hold'em, As plays with 4 board spades for a Royal Flush!
        // In PLO, player MUST use 2 hole cards + 3 board cards: only ONE spade in hand -> NO FLUSH!
        // Best PLO hand is Ace-high Broadway straight (As + 9c? No: As + X with Qs Js Ts? Need K!
        // Wait: As + 9c + Ks Qs Js = A-K-Q-J-9 high card! Or 9c + 8c + Qs Js Ts = Q-J-T-9-8 straight!)
        let board_4_spades: Board = "KsQsJsTs2d".parse().unwrap();
        let bare_ace_spade: Hand4 = "As9c8c7c".parse().unwrap();
        let real_spade_flush: Hand4 = "9s2s5h4h".parse().unwrap();

        let r_bare_as = eval_plo(bare_ace_spade, &board_4_spades).unwrap();
        let r_flush = eval_plo(real_spade_flush, &board_4_spades).unwrap();

        assert_eq!(r_bare_as.category(), HandCategory::Straight); // 9c8c + QsJsTs = Q-high straight!
        assert_eq!(r_flush.category(), HandCategory::Flush); // 9s2s + KsQsJs = K-high flush!
        assert!(r_flush > r_bare_as);

        // TRAP 2: Four hearts in hole cards [Ah Kh Qh Jh] on dry board [2h 7c 8d Ts 3s].
        // Cannot use 4 hole hearts + 2h to make a flush! Must use 2 hole cards + 3 board cards -> Pair / Ace-high!
        let four_hearts_hand: Hand4 = "AhKhQhJh".parse().unwrap();
        let dry_board: Board = "2h7c8dTs3s".parse().unwrap();
        let r_four_hearts = eval_plo(four_hearts_hand, &dry_board).unwrap();
        assert_eq!(r_four_hearts.category(), HandCategory::HighCard);

        // TRAP 3: Quads in hand [Ac Ad Ah As] on [Kc Kd 7s 4h 2c].
        // Cannot play quads from hand! Must use exactly two Aces from hand + Kc Kd 7s -> Aces & Kings Two Pair!
        let four_aces_hand: Hand4 = "AcAdAhAs".parse().unwrap();
        let paired_kings_board: Board = "KcKd7s4h2c".parse().unwrap();
        let r_four_aces = eval_plo(four_aces_hand, &paired_kings_board).unwrap();
        assert_eq!(r_four_aces.category(), HandCategory::TwoPair);

        // Meanwhile [Kc Qs Jd Tc] has Three Kings (Kc + Qs + Kc? wait no duplicate Kc: [Kh Qs Jd Tc])
        let trips_kings: Hand4 = "KhQsJdTc".parse().unwrap();
        let r_trips_kings = eval_plo(trips_kings, &paired_kings_board).unwrap();
        assert_eq!(r_trips_kings.category(), HandCategory::ThreeOfAKind);
        assert!(r_trips_kings > r_four_aces);

        // TRAP 4: Playing the board is ILLEGAL in PLO!
        // Board is a Royal Flush [As Ks Qs Js Ts]. Player holds [2c 3c 4d 5d].
        // In Hold'em, player plays the board for a Royal Flush chop.
        // In PLO, player MUST use 2 hole cards + 3 board cards -> best hand is As Ks Qs + 5d 4d (Ace-high)!
        // Meanwhile [9s 8s 2h 3h] makes Qs Js Ts + 9s 8s = Queen-high Straight Flush!
        let royal_board: Board = "AsKsQsJsTs".parse().unwrap();
        let low_cards: Hand4 = "2c3c4d5d".parse().unwrap();
        let low_sf: Hand4 = "9s8s2h3h".parse().unwrap();
        assert_eq!(
            eval_plo(low_cards, &royal_board).unwrap().category(),
            HandCategory::HighCard
        );
        assert_eq!(
            eval_plo(low_sf, &royal_board).unwrap().category(),
            HandCategory::StraightFlush
        );

        // TRAP 5: 3 hole cards needed for straight is ILLEGAL.
        // Hand [9c 8d 7h 2s] on board [Ac Ks 6d 2c 2h]:
        // 9-8-7 + 6 on board is only 4 straight cards unless 3 hole cards were used.
        // Legal 2+3 gives Pair of Deuces (or Three Deuces using 2s + 2c 2h).
        let wrap_hand: Hand4 = "9c8d7h2s".parse().unwrap();
        let paired_deuces_board: Board = "AcKs6d2c2h".parse().unwrap();
        assert_eq!(
            eval_plo(wrap_hand, &paired_deuces_board).unwrap().category(),
            HandCategory::ThreeOfAKind
        );

        // TRAP 6: Quads on the board (`7c 7d 7h 7s Kc`) with `[Ac Ad Qh Js]` vs `[2c 2d 5h 4s]`!
        // In Hold'em, `[Ac Ad ...]` plays the 4 board sevens + Ace kicker to beat `[2c 2d ...]`.
        // In PLO, you can ONLY take THREE sevens from the board + TWO hole cards:
        // `[Ac Ad Qh Js]` MUST use `Ac Ad` + `777` = `7s full of Aces` (`FullHouse`), NOT Quads!
        // Meanwhile `[8c 7c? no, 7s all on board: Kh Qh 5d 4s]` uses `Kh`? Wait: `Kh` + `Qh` + `7c 7d Kc` = `7s`?
        // No: `7c 7d 7h` + `Kh Qh` = Three Sevens (`ThreeOfAKind`), OR `7c 7d Kc` + `Kh Qh` = Kings & Sevens (`TwoPair`)!
        // So `[2c 2d 5h 4s]` (`77722` Full House) ACTUALLY BEATS `[Kh Qh 5d 4s]` (`777KQ` Three of a Kind) in PLO!
        let quads_board: Board = "7c7d7h7sKc".parse().unwrap();
        let pocket_aces: Hand4 = "AcAdQhJs".parse().unwrap();
        let pocket_deuces: Hand4 = "2c2d5h4s".parse().unwrap();
        let bare_king_queen: Hand4 = "KhQh5d4c".parse().unwrap();

        let r_aa = eval_plo(pocket_aces, &quads_board).unwrap();
        let r_22 = eval_plo(pocket_deuces, &quads_board).unwrap();
        let r_kq = eval_plo(bare_king_queen, &quads_board).unwrap();
        assert_eq!(r_aa.category(), HandCategory::FullHouse); // 777AA (NOT Quads!)
        assert_eq!(r_22.category(), HandCategory::FullHouse); // 77722
        assert_eq!(r_kq.category(), HandCategory::ThreeOfAKind); // 777KQ (cannot use Kc from board AND 777 from board!)
        assert!(r_aa > r_22 && r_22 > r_kq);

        // TRAP 7: Wheel straight `A-2-3-4-5` using mandatory 2 hole + 3 board (`[Ah 2d Kc Qc]` on `[3s 4h 5c Jd Td]`)
        // vs missed 3-hole wheel (`[Ah 2d 3c Kc]` on `[4h 5c Jd Td 9s]` -> only One Pair / Ace-high!).
        let wheel_board: Board = "3s4h5cJdTd".parse().unwrap();
        let wheel_hand: Hand4 = "Ah2dKcQc".parse().unwrap();
        assert_eq!(
            eval_plo(wheel_hand, &wheel_board).unwrap().category(),
            HandCategory::Straight
        );
        let non_wheel_board: Board = "4h5cJdTd9s".parse().unwrap();
        let three_wheel_cards_in_hand: Hand4 = "Ah2d3cKc".parse().unwrap();
        assert_eq!(
            eval_plo(three_wheel_cards_in_hand, &non_wheel_board)
                .unwrap()
                .category(),
            HandCategory::HighCard
        );
    }

    #[test]
    fn identical_strengths_chop_pot() {
        let board: Board = "AsKsQs7d2c".parse().unwrap();
        // Both hands make Broadway straight (A-K-Q-J-T) with no flush possible
        let h0: Hand4 = "JhTd4c3c".parse().unwrap();
        let h1: Hand4 = "JdTc5h6h".parse().unwrap();
        assert_eq!(eval_plo(h0, &board).unwrap(), eval_plo(h1, &board).unwrap());
        assert_eq!(
            showdown_heads_up(h0, h1, &board).unwrap(),
            ShowdownOutcome::Split
        );
    }

    #[test]
    fn fast_plo_matches_slow_2_plus_3_on_120k_random_hands() {
        // Deterministic xorshift64* PRNG for reproducible tests across native and WASM
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_u64 = || -> u64 {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            rng.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };

        let mut lex_to_fast: BTreeMap<u32, u16> = BTreeMap::new();
        let mut deck = [Card::new(0).unwrap(); 52];
        for (i, slot) in deck.iter_mut().enumerate() {
            *slot = Card::new(i as u8).unwrap();
        }

        // 120,000 random river deals + 20,000 turn deals + 20,000 flop deals = 160,000 deals!
        for iter in 0..160_000usize {
            let need = if iter < 120_000 {
                9
            } else if iter < 140_000 {
                8
            } else {
                7
            };
            // Partial Fisher-Yates for `need` cards
            for i in 0..need {
                let j = i + (next_u64() as usize % (52 - i));
                deck.swap(i, j);
            }

            let hand = Hand4::new([deck[0], deck[1], deck[2], deck[3]]).unwrap();
            let board = Board::from_slice(&deck[4..need]).unwrap();

            let fast_rank = eval_plo(hand, &board).unwrap().raw();
            let (slow_lex, _) = slow_eval_plo_lex(hand, &board).unwrap();

            if let Some(&prev_rank) = lex_to_fast.get(&slow_lex) {
                assert_eq!(
                    fast_rank, prev_rank,
                    "inconsistent rank for hand={hand} board={board}"
                );
            } else {
                lex_to_fast.insert(slow_lex, fast_rank);
            }
        }

        // Verify strict monotonicity across every observed equivalence class in `lex_to_fast`:
        // slow_lex_1 < slow_lex_2 <==> fast_rank_1 < fast_rank_2!
        let mut prev: Option<(u32, u16)> = None;
        for (&lex, &r) in &lex_to_fast {
            if let Some((p_lex, p_r)) = prev {
                assert!(p_lex < lex);
                assert!(
                    p_r < r,
                    "monotonicity violation: ({p_lex:#x} -> {p_r}) vs ({lex:#x} -> {r})"
                );
            }
            prev = Some((lex, r));
        }
    }

    #[test]
    #[should_panic(expected = "eval_5_cards requires 5 distinct cards")]
    fn eval_5_cards_rejects_duplicate_cards_in_release() {
        let ace_spades: Card = "As".parse().unwrap();
        let _ = eval_5_cards([ace_spades; 5]);
    }

    #[test]
    #[should_panic(expected = "eval_plo_flop: hand overlaps flop board")]
    fn eval_plo_flop_rejects_hand_board_overlap_in_release() {
        let hand: Hand4 = "AsKsQhJh".parse().unwrap();
        let flop = [
            "As".parse().unwrap(),
            "7d".parse().unwrap(),
            "2c".parse().unwrap(),
        ];
        let _ = eval_plo_flop(hand, flop);
    }

    #[test]
    #[should_panic(expected = "eval_plo_river: hand overlaps river board")]
    fn eval_plo_river_rejects_hand_board_overlap_in_release() {
        let hand: Hand4 = "AsKsQhJh".parse().unwrap();
        let river = [
            "Qh".parse().unwrap(),
            "Td".parse().unwrap(),
            "9c".parse().unwrap(),
            "7d".parse().unwrap(),
            "2c".parse().unwrap(),
        ];
        let _ = eval_plo_river(hand, river);
    }
}
