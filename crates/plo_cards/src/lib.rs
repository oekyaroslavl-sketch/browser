//! Core card, bitmask, board, sorted 4-card Omaha hand, and `C(52,4) = 270,725`
//! combinatorial index types for Pot-Limit Omaha (`PLO-4`).
//!
//! Card encoding matches `wasm-postflop` (`card_id = 4 * rank + suit`, `0..52`):
//! - `rank`: `0..13` (`2 => 0, 3 => 1, ..., T => 8, J => 9, Q => 10, K => 11, A => 12`)
//! - `suit`: `0..4` (`c => 0, d => 1, h => 2, s => 3`)
//!
//! Hold'em 2-card combo tables (`1326`) are intentionally absent; only 4-card Omaha
//! hands (`Hand4`, `ComboId` in `0..270_725`) exist as hole-card representations.

use core::fmt;
use core::str::FromStr;

/// Number of cards in a standard deck.
pub const NUM_CARDS: usize = 52;
/// Number of ranks (`2..=A`).
pub const NUM_RANKS: usize = 13;
/// Number of suits (`c, d, h, s`).
pub const NUM_SUITS: usize = 4;
/// Total number of distinct 4-card Omaha starting hands: `C(52, 4) = 270,725`.
pub const NUM_PLO_COMBOS: usize = 270_725;

const RANK_CHARS: [u8; 13] = *b"23456789TJQKA";
const SUIT_CHARS: [u8; 4] = *b"cdhs";

/// Precomputed binomial coefficients `BINOM[n][k] = C(n, k)` for `0 <= n <= 52`, `0 <= k <= 4`.
pub const BINOM: [[u32; 5]; 53] = build_binom_table();

const fn build_binom_table() -> [[u32; 5]; 53] {
    let mut table = [[0u32; 5]; 53];
    let mut n = 0;
    while n <= 52 {
        table[n][0] = 1;
        let mut k = 1;
        while k <= 4 && k <= n {
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
            k += 1;
        }
        n += 1;
    }
    table
}

/// Errors encountered when constructing or parsing cards, 4-card hands, or boards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardError {
    InvalidCardId(u8),
    InvalidRank(u8),
    InvalidSuit(u8),
    InvalidCardString(String),
    DuplicateCard(Card),
    WrongHandLength(usize),
    WrongBoardLength(usize),
    BoardOverlap,
    InvalidComboId(u32),
}

impl fmt::Display for CardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CardError::InvalidCardId(id) => write!(f, "card id {id} out of range 0..52"),
            CardError::InvalidRank(r) => write!(f, "rank {r} out of range 0..13"),
            CardError::InvalidSuit(s) => write!(f, "suit {s} out of range 0..4"),
            CardError::InvalidCardString(s) => write!(f, "invalid card string: {s:?}"),
            CardError::DuplicateCard(c) => write!(f, "duplicate card: {c}"),
            CardError::WrongHandLength(n) => {
                write!(f, "PLO hand requires exactly 4 cards, got {n}")
            }
            CardError::WrongBoardLength(n) => {
                write!(f, "postflop board requires 3, 4, or 5 cards, got {n}")
            }
            CardError::BoardOverlap => write!(f, "cards overlap on board or hand"),
            CardError::InvalidComboId(id) => {
                write!(f, "PLO combo id {id} out of range 0..{NUM_PLO_COMBOS}")
            }
        }
    }
}

impl std::error::Error for CardError {}

/// A single playing card encoded in `0..52` as `4 * rank + suit`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Card(u8);

impl Card {
    /// Creates a `Card` from raw id `0..52`.
    #[inline]
    pub const fn new(id: u8) -> Result<Self, CardError> {
        if id < 52 {
            Ok(Self(id))
        } else {
            Err(CardError::InvalidCardId(id))
        }
    }

    /// Creates a `Card` from `rank` (`0..13`, `2..=A`) and `suit` (`0..4`, `c,d,h,s`).
    #[inline]
    pub const fn from_rank_suit(rank: u8, suit: u8) -> Result<Self, CardError> {
        if rank >= 13 {
            return Err(CardError::InvalidRank(rank));
        }
        if suit >= 4 {
            return Err(CardError::InvalidSuit(suit));
        }
        Ok(Self((rank << 2) | suit))
    }

    /// Raw card id in `0..52`.
    #[inline]
    pub const fn id(self) -> u8 {
        self.0
    }

    /// Card rank in `0..13` (`0 = '2', ..., 12 = 'A'`).
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 >> 2
    }

    /// Card suit in `0..4` (`0 = 'c', 1 = 'd', 2 = 'h', 3 = 's'`).
    #[inline]
    pub const fn suit(self) -> u8 {
        self.0 & 3
    }

    /// Single-card 64-bit mask (`1u64 << id`).
    #[inline]
    pub const fn mask(self) -> CardMask {
        CardMask(1u64 << self.0)
    }

    /// Rank character (`'2'..='9', 'T', 'J', 'Q', 'K', 'A'`).
    #[inline]
    pub const fn rank_char(self) -> char {
        RANK_CHARS[(self.0 >> 2) as usize] as char
    }

    /// Suit character (`'c', 'd', 'h', 's'`).
    #[inline]
    pub const fn suit_char(self) -> char {
        SUIT_CHARS[(self.0 & 3) as usize] as char
    }

    /// Parses a rank character (case-insensitive for `T, J, Q, K, A`).
    #[inline]
    pub fn parse_rank_char(ch: u8) -> Option<u8> {
        match ch {
            b'2'..=b'9' => Some(ch - b'2'),
            b'T' | b't' => Some(8),
            b'J' | b'j' => Some(9),
            b'Q' | b'q' => Some(10),
            b'K' | b'k' => Some(11),
            b'A' | b'a' => Some(12),
            _ => None,
        }
    }

    /// Parses a suit character (case-insensitive for `c, d, h, s`).
    #[inline]
    pub fn parse_suit_char(ch: u8) -> Option<u8> {
        match ch {
            b'c' | b'C' => Some(0),
            b'd' | b'D' => Some(1),
            b'h' | b'H' => Some(2),
            b's' | b'S' => Some(3),
            _ => None,
        }
    }

    /// Iterator over all 52 cards in ascending order `0..52`.
    #[inline]
    pub fn all() -> impl Iterator<Item = Card> {
        (0u8..52).map(Card)
    }
}

impl FromStr for Card {
    type Err = CardError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return Err(CardError::InvalidCardString(s.to_string()));
        }
        let rank = Card::parse_rank_char(bytes[0])
            .ok_or_else(|| CardError::InvalidCardString(s.to_string()))?;
        let suit = Card::parse_suit_char(bytes[1])
            .ok_or_else(|| CardError::InvalidCardString(s.to_string()))?;
        Card::from_rank_suit(rank, suit)
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let buf = [RANK_CHARS[self.rank() as usize], SUIT_CHARS[self.suit() as usize]];
        let s = core::str::from_utf8(&buf).unwrap_or("??");
        f.write_str(s)
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Card({self})")
    }
}

/// 64-bit bitset of cards (`bit i` corresponds to `Card(i)` for `0 <= i < 52`).
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CardMask(u64);

impl CardMask {
    pub const EMPTY: Self = Self(0);
    pub const FULL_DECK: Self = Self((1u64 << 52) - 1);

    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits & ((1u64 << 52) - 1))
    }

    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub const fn contains(self, card: Card) -> bool {
        (self.0 & (1u64 << card.0)) != 0
    }

    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub const fn without(self, dead: Self) -> Self {
        Self(self.0 & !dead.0)
    }

    #[inline]
    pub fn insert(&mut self, card: Card) -> bool {
        let bit = 1u64 << card.0;
        let newly_inserted = (self.0 & bit) == 0;
        self.0 |= bit;
        newly_inserted
    }

    pub fn from_cards(cards: &[Card]) -> Result<Self, CardError> {
        let mut mask = Self::EMPTY;
        for &c in cards {
            if !mask.insert(c) {
                return Err(CardError::DuplicateCard(c));
            }
        }
        Ok(mask)
    }

    /// Iterates over all `Card`s in this mask in ascending `Card` id order.
    #[inline]
    pub const fn iter(self) -> CardMaskIter {
        CardMaskIter(self.0)
    }
}

impl core::ops::BitOr for CardMask {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for CardMask {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl core::ops::BitAnd for CardMask {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl fmt::Debug for CardMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CardMask(")?;
        for (i, c) in self.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{c}")?;
        }
        write!(f, ")")
    }
}

/// Iterator over cards set in a [`CardMask`].
pub struct CardMaskIter(u64);

impl Iterator for CardMaskIter {
    type Item = Card;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            None
        } else {
            let tz = self.0.trailing_zeros() as u8;
            self.0 &= self.0 - 1;
            Some(Card(tz))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for CardMaskIter {}

/// Parses a concatenated or whitespace-separated sequence of cards (e.g. `"AsKsQhJh"` or `"As Ks Qh"`).
pub fn parse_cards(s: &str) -> Result<Vec<Card>, CardError> {
    let compact: Vec<u8> = s
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b',' && *b != b'[' && *b != b']')
        .collect();
    if compact.len() % 2 != 0 {
        return Err(CardError::InvalidCardString(s.to_string()));
    }
    let mut out = Vec::with_capacity(compact.len() / 2);
    let mut seen = CardMask::EMPTY;
    for chunk in compact.chunks_exact(2) {
        let rank = Card::parse_rank_char(chunk[0])
            .ok_or_else(|| CardError::InvalidCardString(s.to_string()))?;
        let suit = Card::parse_suit_char(chunk[1])
            .ok_or_else(|| CardError::InvalidCardString(s.to_string()))?;
        let card = Card::from_rank_suit(rank, suit)?;
        if !seen.insert(card) {
            return Err(CardError::DuplicateCard(card));
        }
        out.push(card);
    }
    Ok(out)
}

/// Stable identifier in `0..270_725` (`C(52, 4)`) for a 4-card Omaha hand.
///
/// Uses the combinatorial number system on sorted cards `c0 < c1 < c2 < c3`:
/// `id = C(c0, 1) + C(c1, 2) + C(c2, 3) + C(c3, 4)`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ComboId(u32);

impl ComboId {
    #[inline]
    pub const fn new(id: u32) -> Result<Self, CardError> {
        if (id as usize) < NUM_PLO_COMBOS {
            Ok(Self(id))
        } else {
            Err(CardError::InvalidComboId(id))
        }
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Encodes a sorted [`Hand4`] into its unique `ComboId` in `O(1)`.
    #[inline]
    pub const fn from_hand(hand: Hand4) -> Self {
        let c = hand.cards();
        let id = BINOM[c[0].0 as usize][1]
            + BINOM[c[1].0 as usize][2]
            + BINOM[c[2].0 as usize][3]
            + BINOM[c[3].0 as usize][4];
        Self(id)
    }

    /// Decodes a `ComboId` back into its unique sorted [`Hand4`].
    #[inline]
    pub fn to_hand(self) -> Hand4 {
        let mut rem = self.0;
        let mut out = [Card(0); 4];

        // Find c3 (largest n in 3..52 with C(n, 4) <= rem), then c2, c1, c0.
        let mut upper = 52usize;
        for k in (1..=4).rev() {
            // Binary search in (k - 1)..upper for largest c with BINOM[c][k] <= rem.
            let mut lo = k - 1;
            let mut hi = upper;
            while lo + 1 < hi {
                let mid = (lo + hi) >> 1;
                if BINOM[mid][k] <= rem {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            rem -= BINOM[lo][k];
            out[k - 1] = Card(lo as u8);
            upper = lo;
        }
        Hand4(out)
    }

    /// Iterator over all `270,725` `ComboId`s.
    #[inline]
    pub fn all() -> impl Iterator<Item = ComboId> {
        (0..NUM_PLO_COMBOS as u32).map(ComboId)
    }
}

impl fmt::Debug for ComboId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComboId({}: {})", self.0, self.to_hand())
    }
}

/// A 4-card Pot-Limit Omaha hole hand, guaranteed to contain 4 distinct cards
/// stored in strictly ascending order `c0 < c1 < c2 < c3`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Hand4([Card; 4]);

impl Hand4 {
    /// Constructs a sorted `Hand4` from 4 cards, returning an error if any two cards are identical.
    #[inline]
    pub fn new(mut cards: [Card; 4]) -> Result<Self, CardError> {
        // 4-element sorting network (5 comparisons, branch-predictable, no heap)
        if cards[0] > cards[1] {
            cards.swap(0, 1);
        }
        if cards[2] > cards[3] {
            cards.swap(2, 3);
        }
        if cards[0] > cards[2] {
            cards.swap(0, 2);
        }
        if cards[1] > cards[3] {
            cards.swap(1, 3);
        }
        if cards[1] > cards[2] {
            cards.swap(1, 2);
        }

        if cards[0] == cards[1] {
            return Err(CardError::DuplicateCard(cards[0]));
        }
        if cards[1] == cards[2] {
            return Err(CardError::DuplicateCard(cards[1]));
        }
        if cards[2] == cards[3] {
            return Err(CardError::DuplicateCard(cards[2]));
        }
        Ok(Self(cards))
    }

    /// Constructs a `Hand4` from a slice that must have length 4.
    #[inline]
    pub fn from_slice(slice: &[Card]) -> Result<Self, CardError> {
        if slice.len() != 4 {
            return Err(CardError::WrongHandLength(slice.len()));
        }
        Self::new([slice[0], slice[1], slice[2], slice[3]])
    }

    /// Returns the 4 sorted cards `c0 < c1 < c2 < c3`.
    #[inline]
    pub const fn cards(self) -> [Card; 4] {
        self.0
    }

    /// 64-bit `CardMask` of the 4 hole cards.
    #[inline]
    pub const fn mask(self) -> CardMask {
        CardMask(
            (1u64 << self.0[0].0)
                | (1u64 << self.0[1].0)
                | (1u64 << self.0[2].0)
                | (1u64 << self.0[3].0),
        )
    }

    /// Returns `true` if this 4-card hand shares at least one card with `dead`.
    #[inline]
    pub const fn is_blocked_by(self, dead: CardMask) -> bool {
        self.mask().intersects(dead)
    }

    /// Encodes this 4-card hand into its unique `ComboId` in `0..270_725`.
    #[inline]
    pub const fn combo_id(self) -> ComboId {
        ComboId::from_hand(self)
    }

    /// Decodes a `ComboId` into a `Hand4`.
    #[inline]
    pub fn from_combo_id(id: ComboId) -> Self {
        id.to_hand()
    }

    /// Returns all `C(4, 2) = 6` distinct 2-card hole subsets required by PLO's "exact 2 hole cards" rule.
    #[inline]
    pub const fn two_card_subsets(self) -> [[Card; 2]; 6] {
        let [c0, c1, c2, c3] = self.0;
        [
            [c0, c1],
            [c0, c2],
            [c0, c3],
            [c1, c2],
            [c1, c3],
            [c2, c3],
        ]
    }

    /// Fast zero-allocation iterator over all `270,725` `Hand4` combinations in `ComboId` order (`0..270,725`).
    pub fn all_hands() -> impl Iterator<Item = (ComboId, Hand4)> {
        (3u8..52).flat_map(|c3| {
            (2u8..c3).flat_map(move |c2| {
                (1u8..c2).flat_map(move |c1| {
                    (0u8..c1).map(move |c0| {
                        let hand = Hand4([Card(c0), Card(c1), Card(c2), Card(c3)]);
                        (hand.combo_id(), hand)
                    })
                })
            })
        })
    }
}

impl FromStr for Hand4 {
    type Err = CardError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let cards = parse_cards(s)?;
        Self::from_slice(&cards)
    }
}

impl fmt::Display for Hand4 {
    /// Displays in human-friendly descending-rank order (e.g. `AsKsQhJh`) while internal storage remains sorted ascending.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in (0..4).rev() {
            write!(f, "{}", self.0[i])?;
        }
        Ok(())
    }
}

impl fmt::Debug for Hand4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hand4({self})")
    }
}

/// Postflop street (`Flop = 3`, `Turn = 4`, `River = 5`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Street {
    Flop = 3,
    Turn = 4,
    River = 5,
}

impl Street {
    #[inline]
    pub const fn board_len(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn next(self) -> Option<Self> {
        match self {
            Street::Flop => Some(Street::Turn),
            Street::Turn => Some(Street::River),
            Street::River => None,
        }
    }
}

/// Community board of 3 (flop), 4 (turn), or 5 (river) unique cards.
///
/// Preserves deal order (flop[0..3], turn, river) while tracking a `CardMask` for `O(1)` blocker checks.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Board {
    cards: [Card; 5],
    len: u8,
    mask: CardMask,
}

impl Board {
    /// Constructs a `Board` from 3, 4, or 5 unique cards.
    pub fn from_slice(slice: &[Card]) -> Result<Self, CardError> {
        if !matches!(slice.len(), 3..=5) {
            return Err(CardError::WrongBoardLength(slice.len()));
        }
        let mut cards = [Card(0); 5];
        let mut mask = CardMask::EMPTY;
        for (i, &c) in slice.iter().enumerate() {
            if !mask.insert(c) {
                return Err(CardError::DuplicateCard(c));
            }
            cards[i] = c;
        }
        Ok(Self {
            cards,
            len: slice.len() as u8,
            mask,
        })
    }

    #[inline]
    pub fn flop(cards: [Card; 3]) -> Result<Self, CardError> {
        Self::from_slice(&cards)
    }

    #[inline]
    pub fn turn(cards: [Card; 4]) -> Result<Self, CardError> {
        Self::from_slice(&cards)
    }

    #[inline]
    pub fn river(cards: [Card; 5]) -> Result<Self, CardError> {
        Self::from_slice(&cards)
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub const fn street(&self) -> Street {
        match self.len {
            3 => Street::Flop,
            4 => Street::Turn,
            5 => Street::River,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[Card] {
        &self.cards[..self.len as usize]
    }

    #[inline]
    pub const fn mask(&self) -> CardMask {
        self.mask
    }

    #[inline]
    pub fn flop_cards(&self) -> [Card; 3] {
        [self.cards[0], self.cards[1], self.cards[2]]
    }

    #[inline]
    pub fn turn_cards(&self) -> Option<[Card; 4]> {
        if self.len >= 4 {
            Some([self.cards[0], self.cards[1], self.cards[2], self.cards[3]])
        } else {
            None
        }
    }

    #[inline]
    pub fn river_cards(&self) -> Option<[Card; 5]> {
        if self.len == 5 {
            Some(self.cards)
        } else {
            None
        }
    }

    /// Deals one more community card (`Flop -> Turn` or `Turn -> River`), checking uniqueness.
    #[inline]
    pub fn deal(mut self, card: Card) -> Result<Self, CardError> {
        if self.len >= 5 {
            return Err(CardError::WrongBoardLength(self.len as usize + 1));
        }
        if !self.mask.insert(card) {
            return Err(CardError::DuplicateCard(card));
        }
        self.cards[self.len as usize] = card;
        self.len += 1;
        Ok(self)
    }
}

impl FromStr for Board {
    type Err = CardError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let cards = parse_cards(s)?;
        Self::from_slice(&cards)
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for c in self.as_slice() {
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Board({self})")
    }
}

/// Encodes a pair of distinct cards `c0 < c1` into `0..1326` (`C(52, 2)`),
/// used internally as a 2-card blocker subset index in 4-card inclusion-exclusion.
///
/// NOTE: This is NOT a Hold'em hand identifier and is not exposed as a hole-card type.
/// Strict compile-fail guard ensuring `Hand4` cannot be constructed from a 2-card Hold'em array `[Card; 2]`:
///
/// ```compile_fail
/// use plo_cards::{Card, Hand4};
/// let c0: Card = "As".parse().unwrap();
/// let c1: Card = "Ks".parse().unwrap();
/// let _illegal_holdem_hand = Hand4::new([c0, c1]);
/// ```
///
/// ```compile_fail
/// use plo_cards::Hand2;
/// ```
#[inline]
pub const fn pair_subset_index(c0: Card, c1: Card) -> usize {
    debug_assert!(c0.0 < c1.0);
    BINOM[c0.0 as usize][1] as usize + BINOM[c1.0 as usize][2] as usize
}

/// Encodes a triple of distinct cards `c0 < c1 < c2` into `0..22_100` (`C(52, 3)`),
/// used internally as a 3-card blocker subset index in 4-card inclusion-exclusion.
#[inline]
pub const fn triple_subset_index(c0: Card, c1: Card, c2: Card) -> usize {
    debug_assert!(c0.0 < c1.0 && c1.0 < c2.0);
    BINOM[c0.0 as usize][1] as usize
        + BINOM[c1.0 as usize][2] as usize
        + BINOM[c2.0 as usize][3] as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_roundtrip_and_ordering() {
        for id in 0..52u8 {
            let c = Card::new(id).unwrap();
            let s = c.to_string();
            let parsed: Card = s.parse().unwrap();
            assert_eq!(c, parsed);
            assert_eq!(c.id(), id);
            assert_eq!(
                Card::from_rank_suit(c.rank(), c.suit()).unwrap(),
                c
            );
        }
        assert!(Card::new(52).is_err());
        assert!("2c".parse::<Card>().unwrap().id() == 0);
        assert!("As".parse::<Card>().unwrap().id() == 51);
    }

    #[test]
    fn hand4_rejects_duplicates_and_sorts() {
        let h1: Hand4 = "AsKsQhJh".parse().unwrap();
        let h2: Hand4 = "JhAsQhKs".parse().unwrap();
        assert_eq!(h1, h2);
        let cards = h1.cards();
        assert!(cards[0] < cards[1] && cards[1] < cards[2] && cards[2] < cards[3]);

        // Duplicate card must fail
        assert!("AsAsQhJh".parse::<Hand4>().is_err());
        // Hold'em 2-card strings & 13x13 Hold'em tokens (`"AsKs"`, `"AKs"`, `"AA"`, `"AKo"`) MUST fail!
        assert!("AsKs".parse::<Hand4>().is_err());
        assert!("AKs".parse::<Hand4>().is_err());
        assert!("AA".parse::<Hand4>().is_err());
        assert!("AKo".parse::<Hand4>().is_err());
        // 5-card string must fail
        assert!("AsKsQhJhTd".parse::<Hand4>().is_err());
    }

    #[test]
    fn combo_id_bijection_all_270725_hands() {
        assert_eq!(BINOM[52][4] as usize, NUM_PLO_COMBOS);
        let mut count = 0usize;
        for (expected_id, hand) in Hand4::all_hands() {
            assert_eq!(expected_id.index(), count);
            let encoded = hand.combo_id();
            assert_eq!(encoded, expected_id);
            let decoded = encoded.to_hand();
            assert_eq!(decoded, hand);
            count += 1;
        }
        assert_eq!(count, NUM_PLO_COMBOS);
        assert!(ComboId::new(NUM_PLO_COMBOS as u32).is_err());
    }

    #[test]
    fn dead_card_mask_filtering() {
        let board: Board = "AhKd7s".parse().unwrap();
        let blocked_hand: Hand4 = "AhQcJcTc".parse().unwrap();
        let live_hand: Hand4 = "AcQcJcTc".parse().unwrap();
        assert!(blocked_hand.is_blocked_by(board.mask()));
        assert!(!live_hand.is_blocked_by(board.mask()));

        // Count live 4-card combos on a 3-card flop: C(49, 4) = 211,876
        let live_count = Hand4::all_hands()
            .filter(|(_, h)| !h.is_blocked_by(board.mask()))
            .count();
        assert_eq!(live_count, 211_876);
    }
}
