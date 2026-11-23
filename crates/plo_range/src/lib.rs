//! Sparse weighted 4-card ranges, PLO range syntax parser, dead-card blocker filtering,
//! and exact PLO equity calculator (`hand vs hand`, `hand vs range`, `range vs range`
//! on flop, turn, and river).
//!
//! # Memory Design
//! Never stores a `270,725`-element dense array by default. Ranges are stored sparsely as
//! sorted `Vec<WeightedHand>` (`16` bytes per active combo: `ComboId`, `Hand4`, `CardMask`, `f32` weight),
//! giving cache-linear iteration, `O(log N)` membership lookup, and `O(1)` bitmask blocker filtering.
//!
//! # Supported PLO Range String Syntax
//! Comma-separated tokens, each optionally suffixed with `@weight` or `:weight` (`0.0 < weight <= 1.0`,
//! or percentage `0%..=100%`):
//! - Exact 8-char 4-card combo: `AsKsQhJh`, `AsKsQhJh@0.5`, `AsKsQhJh:75%`
//! - 4-rank macro (all valid 4-card suitings of those 4 ranks): `AAKK`, `AKQJ`, `JT98`, `AAJT@0.5`
//! - 4-rank macro with suit filter (`$ds`, `$ss`, `$rb` / `$np`, `$ms`, `$ts`):
//!   - `AAKK$ds` — double-suited Aces-Kings (`6` combos)
//!   - `AAKK$ss` — single-suited Aces-Kings (`24` combos)
//!   - `AAKK$rb` — rainbow Aces-Kings (`6` combos)
//!   - `AKQJ$ds` — all double-suited Broadway rundowns (`36` combos)
//! - Suit-grouped bracket syntax (`[AsKs]QhJh` or `(AK)(QJ)` for double-suited rank groups!):
//!   - `(AK)(QJ)` — Ace+King suited in one suit, Queen+Jack suited in a second distinct suit (`12` combos!)
//!   - `(AK)QJ` — at least Ace+King suited (`(AK)` share a suit; `Q, J` any valid cards)
//! - `*` or `100%` — full `270,725`-combo universe (for range exploration; solver enforces memory caps).

use core::fmt;
use core::str::FromStr;

use plo_cards::{Board, Card, CardError, CardMask, ComboId, Hand4};
use plo_eval::{eval_plo_river, RiverBoardCache};
use plo_iso::{classify_suit_pattern, SuitPattern4};

/// A single active 4-card combo in a sparse PLO range, pre-equipped with its `CardMask`
/// for `O(1)` blocker checks in hot inner loops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedHand {
    pub combo_id: ComboId,
    pub hand: Hand4,
    pub mask: CardMask,
    pub weight: f32,
}

impl WeightedHand {
    #[inline]
    pub fn new(hand: Hand4, weight: f32) -> Self {
        Self {
            combo_id: hand.combo_id(),
            hand,
            mask: hand.mask(),
            weight,
        }
    }
}

/// Errors when parsing or constructing a [`PloRange`].
#[derive(Debug, Clone, PartialEq)]
pub enum RangeError {
    Card(CardError),
    InvalidToken(String),
    InvalidWeight(String),
    EmptyRange,
}

impl fmt::Display for RangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RangeError::Card(e) => write!(f, "{e}"),
            RangeError::InvalidToken(t) => write!(f, "invalid PLO range token: {t:?}"),
            RangeError::InvalidWeight(w) => write!(f, "invalid combo weight: {w:?}"),
            RangeError::EmptyRange => write!(f, "range has no valid unblocked 4-card combos"),
        }
    }
}

impl std::error::Error for RangeError {}

impl From<CardError> for RangeError {
    fn from(e: CardError) -> Self {
        RangeError::Card(e)
    }
}

/// Sparse weighted set of 4-card PLO hands, kept strictly sorted and deduplicated by `ComboId`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PloRange {
    entries: Vec<WeightedHand>,
}

impl PloRange {
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Builds a `PloRange` from `(Hand4, f32)` pairs (`0.0 <= weight <= 1.0`). Duplicate hands
    /// have their latest positive weight kept; zero weights are dropped; negative, `> 1.0`, or
    /// non-finite weights return `Err(RangeError::InvalidWeight)`.
    pub fn from_hands(iter: impl IntoIterator<Item = (Hand4, f32)>) -> Result<Self, RangeError> {
        let mut entries = Vec::new();
        for (hand, weight) in iter {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(RangeError::InvalidWeight(weight.to_string()));
            }
            if weight > 0.0 {
                entries.push(WeightedHand::new(hand, weight));
            }
        }
        Self::from_raw_entries(entries)
    }

    fn from_raw_entries(mut entries: Vec<WeightedHand>) -> Result<Self, RangeError> {
        entries.sort_by_key(|e| e.combo_id);
        // Deduplicate keeping last assigned weight (standard overwrite semantics in range expressions like `AAKK, AAKK$ds@0.5`)
        let mut out: Vec<WeightedHand> = Vec::with_capacity(entries.len());
        for e in entries {
            if let Some(last) = out.last_mut() {
                if last.combo_id == e.combo_id {
                    last.weight = e.weight;
                    continue;
                }
            }
            if e.weight > 0.0 {
                out.push(e);
            }
        }
        out.shrink_to_fit();
        Ok(Self { entries: out })
    }

    /// Inserts or updates a single 4-card hand's weight (`weight <= 0.0` removes it;
    /// positive weights are clamped to `<= 1.0`, ignoring `NaN`).
    pub fn set_weight(&mut self, hand: Hand4, weight: f32) {
        if !weight.is_finite() {
            return;
        }
        let w = weight.min(1.0);
        let cid = hand.combo_id();
        match self.entries.binary_search_by_key(&cid, |e| e.combo_id) {
            Ok(idx) => {
                if w <= 0.0 {
                    self.entries.remove(idx);
                } else {
                    self.entries[idx].weight = w;
                }
            }
            Err(idx) => {
                if w > 0.0 {
                    self.entries.insert(idx, WeightedHand::new(hand, w));
                }
            }
        }
    }

    /// Looks up the weight of `hand` in `O(log N)` (`0.0` if absent).
    #[inline]
    pub fn weight_of(&self, hand: Hand4) -> f32 {
        let cid = hand.combo_id();
        match self.entries.binary_search_by_key(&cid, |e| e.combo_id) {
            Ok(idx) => self.entries[idx].weight,
            Err(_) => 0.0,
        }
    }

    /// Number of distinct non-zero 4-card combos in the range.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Slice of sorted [`WeightedHand`] entries.
    #[inline]
    pub fn entries(&self) -> &[WeightedHand] {
        &self.entries
    }

    /// Sum of all combo weights.
    #[inline]
    pub fn total_weight(&self) -> f64 {
        self.entries.iter().map(|e| e.weight as f64).sum()
    }

    /// Normalizes weights in-place so their sum equals `1.0` (no-op if empty).
    pub fn normalize(&mut self) {
        let sum = self.total_weight();
        if sum > 0.0 {
            let inv = (1.0 / sum) as f32;
            for e in &mut self.entries {
                e.weight *= inv;
            }
        }
    }

    /// Returns a new `PloRange` with all combos sharing any card with `dead` removed (`O(N)` bitmask scan).
    pub fn without_dead(&self, dead: CardMask) -> Self {
        let entries = self
            .entries
            .iter()
            .copied()
            .filter(|e| !e.mask.intersects(dead))
            .collect();
        Self { entries }
    }

    /// Removes in-place all combos sharing any card with `dead`.
    pub fn retain_unblocked(&mut self, dead: CardMask) {
        self.entries.retain(|e| !e.mask.intersects(dead));
    }

    /// Extracts `(ComboId, f32)` pairs (useful for isomorphism checks).
    pub fn to_combo_weight_vec(&self) -> Vec<(ComboId, f32)> {
        self.entries
            .iter()
            .map(|e| (e.combo_id, e.weight))
            .collect()
    }
}

impl FromStr for PloRange {
    type Err = RangeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut raw_entries: Vec<(ComboId, Hand4, f32)> = Vec::new();

        for raw_tok in s.split(',') {
            let tok = raw_tok.trim();
            if tok.is_empty() {
                continue;
            }
            let (pattern, weight) = split_pattern_weight(tok)?;
            expand_pattern_into(pattern, weight, &mut raw_entries)?;
        }

        // Apply overwrite semantics in token order (`AAKK, AAKK$rb@0` removes rainbow AAKK!)
        let mut map = std::collections::BTreeMap::<ComboId, (Hand4, f32)>::new();
        for (cid, hand, w) in raw_entries {
            if w <= 0.0 {
                map.remove(&cid);
            } else {
                map.insert(cid, (hand, w));
            }
        }

        let entries = map
            .into_iter()
            .map(|(combo_id, (hand, weight))| WeightedHand {
                combo_id,
                hand,
                mask: hand.mask(),
                weight,
            })
            .collect();
        Ok(Self { entries })
    }
}

impl fmt::Display for PloRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, e) in self.entries.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            if (e.weight - 1.0).abs() < 1e-6 {
                write!(f, "{}", e.hand)?;
            } else {
                write!(f, "{}@{:.4}", e.hand, e.weight)?;
            }
        }
        Ok(())
    }
}

fn split_pattern_weight(tok: &str) -> Result<(&str, f32), RangeError> {
    let sep_pos = tok.rfind('@').or_else(|| tok.rfind(':'));
    if let Some(pos) = sep_pos {
        let pat = tok[..pos].trim();
        let w_str = tok[pos + 1..].trim();
        let weight = if let Some(pct) = w_str.strip_suffix('%') {
            let v: f32 = pct
                .trim()
                .parse()
                .map_err(|_| RangeError::InvalidWeight(w_str.to_string()))?;
            v / 100.0
        } else {
            w_str
                .parse::<f32>()
                .map_err(|_| RangeError::InvalidWeight(w_str.to_string()))?
        };
        if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
            return Err(RangeError::InvalidWeight(w_str.to_string()));
        }
        Ok((pat, weight))
    } else {
        Ok((tok, 1.0))
    }
}

fn expand_pattern_into(
    pat: &str,
    weight: f32,
    out: &mut Vec<(ComboId, Hand4, f32)>,
) -> Result<(), RangeError> {
    let pat = pat.trim();
    if pat == "*" || pat == "100%" || pat.eq_ignore_ascii_case("all") {
        for (cid, hand) in Hand4::all_hands() {
            out.push((cid, hand, weight));
        }
        return Ok(());
    }

    // Check `$` suit modifier suffix (e.g. `AAKK$ds`, `AKQJ$ss`, `JT98$rb`)
    let (core, suit_filter) = if let Some(dollar_idx) = pat.find('$') {
        let filt_str = pat[dollar_idx + 1..].to_ascii_lowercase();
        let filter = match filt_str.as_str() {
            "ds" => Some(SuitPattern4::DoubleSuited),
            "ss" => Some(SuitPattern4::SingleSuited),
            "rb" | "np" | "rainbow" => Some(SuitPattern4::Rainbow),
            "ts" => Some(SuitPattern4::TripsSuited),
            "ms" | "mono" => Some(SuitPattern4::Monotone),
            _ => return Err(RangeError::InvalidToken(pat.to_string())),
        };
        (&pat[..dollar_idx], filter)
    } else {
        (pat, None)
    };

    // Case 1: Parenthesized suited rank groups, e.g. `(AK)(QJ)` or `(AK)QJ`
    if core.contains('(') {
        if suit_filter.is_some() {
            return Err(RangeError::InvalidToken(pat.to_string()));
        }
        return expand_grouped_ranks(core, weight, out);
    }

    // Case 2: Exact 8-character specific combo (`AsKsQhJh`)
    if core.len() == 8 && suit_filter.is_none() {
        if let Ok(hand) = core.parse::<Hand4>() {
            out.push((hand.combo_id(), hand, weight));
            return Ok(());
        }
    }

    // Case 3: 4-rank pattern (`AAKK`, `AKQJ`, `KKQ9`, etc.)
    let bytes = core.as_bytes();
    if bytes.len() == 4 {
        let mut ranks = [0u8; 4];
        for i in 0..4 {
            ranks[i] = Card::parse_rank_char(bytes[i])
                .ok_or_else(|| RangeError::InvalidToken(pat.to_string()))?;
        }
        return expand_four_ranks(ranks, suit_filter, weight, out);
    }

    Err(RangeError::InvalidToken(pat.to_string()))
}

fn expand_four_ranks(
    ranks: [u8; 4],
    suit_filter: Option<SuitPattern4>,
    weight: f32,
    out: &mut Vec<(ComboId, Hand4, f32)>,
) -> Result<(), RangeError> {
    let mut seen = std::collections::BTreeSet::new();
    for s0 in 0..4u8 {
        let c0 = Card::from_rank_suit(ranks[0], s0)?;
        for s1 in 0..4u8 {
            let c1 = Card::from_rank_suit(ranks[1], s1)?;
            if c1 == c0 {
                continue;
            }
            for s2 in 0..4u8 {
                let c2 = Card::from_rank_suit(ranks[2], s2)?;
                if c2 == c0 || c2 == c1 {
                    continue;
                }
                for s3 in 0..4u8 {
                    let c3 = Card::from_rank_suit(ranks[3], s3)?;
                    if c3 == c0 || c3 == c1 || c3 == c2 {
                        continue;
                    }
                    let hand = Hand4::new([c0, c1, c2, c3])?;
                    if let Some(req) = suit_filter {
                        if classify_suit_pattern(hand) != req {
                            continue;
                        }
                    }
                    seen.insert(hand);
                }
            }
        }
    }
    if seen.is_empty() {
        return Err(RangeError::InvalidToken(format!(
            "impossible 4-rank pattern {ranks:?} with filter {suit_filter:?}"
        )));
    }
    for hand in seen {
        out.push((hand.combo_id(), hand, weight));
    }
    Ok(())
}

/// Expands suited rank groups like `(AK)(QJ)` (group 0 shares suit `s_a`, group 1 shares suit `s_b != s_a`,
/// ungrouped ranks cannot share a suit with any group or each other unless unconstrained? Standard PPI/Monker
/// semantics: each `(...)` assigns a distinct suit to all ranks inside `(...)`, while outside ranks take
/// suits distinct from the parenthesized suits so `(AK)QJ` means `A,K` share suit `s_0` and `Q,J` do NOT have `s_0`!).
fn expand_grouped_ranks(
    core: &str,
    weight: f32,
    out: &mut Vec<(ComboId, Hand4, f32)>,
) -> Result<(), RangeError> {
    // Parse into 4 rank slots, each tagged with `Option<usize>` group id
    let mut ranks = Vec::with_capacity(4);
    let mut group_ids: Vec<Option<usize>> = Vec::with_capacity(4);
    let mut current_group: Option<usize> = None;
    let mut next_group = 0usize;

    for b in core.bytes() {
        match b {
            b'(' => {
                if current_group.is_some() {
                    return Err(RangeError::InvalidToken(core.to_string()));
                }
                current_group = Some(next_group);
                next_group += 1;
            }
            b')' => {
                if current_group.is_none() {
                    return Err(RangeError::InvalidToken(core.to_string()));
                }
                current_group = None;
            }
            ch => {
                let r = Card::parse_rank_char(ch)
                    .ok_or_else(|| RangeError::InvalidToken(core.to_string()))?;
                ranks.push(r);
                group_ids.push(current_group);
            }
        }
    }

    if current_group.is_some() || ranks.len() != 4 || next_group == 0 || next_group > 2 {
        return Err(RangeError::InvalidToken(core.to_string()));
    }

    let mut seen = std::collections::BTreeSet::new();
    for s0 in 0..4u8 {
        for s1 in 0..4u8 {
            for s2 in 0..4u8 {
                for s3 in 0..4u8 {
                    let suits = [s0, s1, s2, s3];
                    // Verify group constraints:
                    // 1) Same group => same suit
                    // 2) Different groups => different suits
                    // 3) Ungrouped card => cannot use any group's suit (preserves exact suitedness of groups)
                    let mut ok = true;
                    for i in 0..4 {
                        for j in (i + 1)..4 {
                            match (group_ids[i], group_ids[j]) {
                                (Some(gi), Some(gj)) if gi == gj => {
                                    if suits[i] != suits[j] {
                                        ok = false;
                                    }
                                }
                                (Some(_), Some(_)) => {
                                    if suits[i] == suits[j] {
                                        ok = false;
                                    }
                                }
                                (Some(_), None) | (None, Some(_)) => {
                                    if suits[i] == suits[j] {
                                        ok = false;
                                    }
                                }
                                (None, None) => {}
                            }
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let c0 = Card::from_rank_suit(ranks[0], s0)?;
                    let c1 = Card::from_rank_suit(ranks[1], s1)?;
                    let c2 = Card::from_rank_suit(ranks[2], s2)?;
                    let c3 = Card::from_rank_suit(ranks[3], s3)?;
                    if let Ok(hand) = Hand4::new([c0, c1, c2, c3]) {
                        seen.insert(hand);
                    }
                }
            }
        }
    }

    if seen.is_empty() {
        return Err(RangeError::InvalidToken(core.to_string()));
    }
    for hand in seen {
        out.push((hand.combo_id(), hand, weight));
    }
    Ok(())
}

// ============================================================================
// Exact PLO Equity Engine (Hand vs Hand, Hand vs Range, Range vs Range)
// ============================================================================

/// Result of an exact PLO equity calculation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquitySummary {
    /// Player 0's share of the pot in `[0.0, 1.0]` (`win_prob + 0.5 * tie_prob`).
    pub equity_p0: f64,
    /// Player 1's share of the pot in `[0.0, 1.0]` (`1.0 - equity_p0`).
    pub equity_p1: f64,
    /// Probability that Player 0 wins outright.
    pub win_p0: f64,
    /// Probability that Player 1 wins outright.
    pub win_p1: f64,
    /// Probability of a split pot.
    pub tie: f64,
    /// Total valid weighted runout-combo realizations summed over (`0.0` if completely blocked).
    pub valid_weight_sum: f64,
}

/// Computes exact heads-up PLO equity for `hand0` vs `hand1` on `board` (`Flop`, `Turn`, or `River`),
/// enumerating all remaining deck runouts (`C(41,2) = 820` on flop, `40` on turn, `1` on river)
/// without replacement!
pub fn equity_hand_vs_hand(
    hand0: Hand4,
    hand1: Hand4,
    board: &Board,
) -> Result<EquitySummary, RangeError> {
    let dead = board.mask();
    if hand0.is_blocked_by(dead) || hand1.is_blocked_by(dead) || hand0.is_blocked_by(hand1.mask()) {
        return Err(RangeError::Card(CardError::BoardOverlap));
    }
    let used = dead | hand0.mask() | hand1.mask();
    let b = board.as_slice();

    let mut wins0 = 0u64;
    let mut wins1 = 0u64;
    let mut ties = 0u64;

    match b.len() {
        5 => {
            let river = [b[0], b[1], b[2], b[3], b[4]];
            let cache = RiverBoardCache::new(river);
            let r0 = cache.eval_hand_unchecked(hand0);
            let r1 = cache.eval_hand_unchecked(hand1);
            match r0.cmp(&r1) {
                core::cmp::Ordering::Greater => wins0 += 1,
                core::cmp::Ordering::Less => wins1 += 1,
                core::cmp::Ordering::Equal => ties += 1,
            }
        }
        4 => {
            for r_id in 0..52u8 {
                let rc = Card::new(r_id).unwrap();
                if used.contains(rc) {
                    continue;
                }
                let river = [b[0], b[1], b[2], b[3], rc];
                let cache = RiverBoardCache::new(river);
                let r0 = cache.eval_hand_unchecked(hand0);
                let r1 = cache.eval_hand_unchecked(hand1);
                match r0.cmp(&r1) {
                    core::cmp::Ordering::Greater => wins0 += 1,
                    core::cmp::Ordering::Less => wins1 += 1,
                    core::cmp::Ordering::Equal => ties += 1,
                }
            }
        }
        3 => {
            for t_id in 0..52u8 {
                let tc = Card::new(t_id).unwrap();
                if used.contains(tc) {
                    continue;
                }
                for r_id in (t_id + 1)..52u8 {
                    let rc = Card::new(r_id).unwrap();
                    if used.contains(rc) {
                        continue;
                    }
                    let river = [b[0], b[1], b[2], tc, rc];
                    let cache = RiverBoardCache::new(river);
                    let r0 = cache.eval_hand_unchecked(hand0);
                    let r1 = cache.eval_hand_unchecked(hand1);
                    match r0.cmp(&r1) {
                        core::cmp::Ordering::Greater => wins0 += 1,
                        core::cmp::Ordering::Less => wins1 += 1,
                        core::cmp::Ordering::Equal => ties += 1,
                    }
                }
            }
        }
        n => return Err(RangeError::Card(CardError::WrongBoardLength(n))),
    }

    let total = (wins0 + wins1 + ties) as f64;
    let win_p0 = wins0 as f64 / total;
    let win_p1 = wins1 as f64 / total;
    let tie = ties as f64 / total;
    let eq0 = win_p0 + 0.5 * tie;
    Ok(EquitySummary {
        equity_p0: eq0,
        equity_p1: 1.0 - eq0,
        win_p0,
        win_p1,
        tie,
        valid_weight_sum: total,
    })
}

/// Computes exact PLO `hand` vs `villain_range` equity on `board` (`Flop`, `Turn`, or `River`),
/// strictly removing all villain combos blocked by `board`, `hand`, or dealt runout cards.
pub fn equity_hand_vs_range(
    hand: Hand4,
    villain_range: &PloRange,
    board: &Board,
) -> Result<EquitySummary, RangeError> {
    if hand.is_blocked_by(board.mask()) {
        return Err(RangeError::Card(CardError::BoardOverlap));
    }
    let hero_range = PloRange::from_hands([(hand, 1.0)])?;
    equity_range_vs_range(&hero_range, villain_range, board)
}

/// Computes exact PLO `range0` vs `range1` equity on `board` (`Flop`, `Turn`, or `River`)
/// with full card-removal / blocker accounting on both the board, runout cards, and opposing hole cards.
pub fn equity_range_vs_range(
    range0: &PloRange,
    range1: &PloRange,
    board: &Board,
) -> Result<EquitySummary, RangeError> {
    let live0 = range0.without_dead(board.mask());
    let live1 = range1.without_dead(board.mask());
    if live0.is_empty() || live1.is_empty() {
        return Err(RangeError::EmptyRange);
    }

    let b = board.as_slice();
    let mut win_w0 = 0.0f64;
    let mut win_w1 = 0.0f64;
    let mut tie_w = 0.0f64;

    let mut eval_river_slice = |river: [Card; 5], extra_dead: CardMask| {
        let cache = RiverBoardCache::new(river);
        // Pre-evaluate active combos on this river once (`O(N + M)` evaluations instead of `O(N * M)`!)
        let r0s: Vec<(CardMask, f64, u16)> = live0
            .entries()
            .iter()
            .filter(|e| !e.mask.intersects(extra_dead))
            .map(|e| {
                (
                    e.mask,
                    e.weight as f64,
                    cache.eval_hand_unchecked(e.hand).raw(),
                )
            })
            .collect();
        let r1s: Vec<(CardMask, f64, u16)> = live1
            .entries()
            .iter()
            .filter(|e| !e.mask.intersects(extra_dead))
            .map(|e| {
                (
                    e.mask,
                    e.weight as f64,
                    cache.eval_hand_unchecked(e.hand).raw(),
                )
            })
            .collect();

        for &(m0, w0, s0) in &r0s {
            for &(m1, w1, s1) in &r1s {
                if m0.intersects(m1) {
                    continue; // Exact 4-card hole-card blocker removal!
                }
                let ww = w0 * w1;
                if s0 > s1 {
                    win_w0 += ww;
                } else if s0 < s1 {
                    win_w1 += ww;
                } else {
                    tie_w += ww;
                }
            }
        }
    };

    match b.len() {
        5 => {
            eval_river_slice([b[0], b[1], b[2], b[3], b[4]], CardMask::EMPTY);
        }
        4 => {
            let bmask = board.mask();
            for r_id in 0..52u8 {
                let rc = Card::new(r_id).unwrap();
                if bmask.contains(rc) {
                    continue;
                }
                eval_river_slice([b[0], b[1], b[2], b[3], rc], rc.mask());
            }
        }
        3 => {
            let bmask = board.mask();
            for t_id in 0..52u8 {
                let tc = Card::new(t_id).unwrap();
                if bmask.contains(tc) {
                    continue;
                }
                for r_id in (t_id + 1)..52u8 {
                    let rc = Card::new(r_id).unwrap();
                    if bmask.contains(rc) {
                        continue;
                    }
                    eval_river_slice([b[0], b[1], b[2], tc, rc], tc.mask() | rc.mask());
                }
            }
        }
        n => return Err(RangeError::Card(CardError::WrongBoardLength(n))),
    }

    let total = win_w0 + win_w1 + tie_w;
    if total <= 0.0 {
        return Err(RangeError::EmptyRange);
    }
    let win_p0 = win_w0 / total;
    let win_p1 = win_w1 / total;
    let tie = tie_w / total;
    let eq0 = win_p0 + 0.5 * tie;
    Ok(EquitySummary {
        equity_p0: eq0,
        equity_p1: 1.0 - eq0,
        win_p0,
        win_p1,
        tie,
        valid_weight_sum: total,
    })
}

/// Computes per-combo river equity (`[0.0, 1.0]`) for each hand in `hero_combos` against weighted
/// `villain_combos` on a 5-card `river` board, returning `(equities, live_matchup_weights)`.
pub fn per_combo_river_equity(
    hero_combos: &[WeightedHand],
    villain_combos: &[WeightedHand],
    river: [Card; 5],
) -> Vec<f32> {
    let river_mask =
        river[0].mask() | river[1].mask() | river[2].mask() | river[3].mask() | river[4].mask();
    let cache = RiverBoardCache::new(river);

    let v_eval: Vec<(CardMask, f64, u16)> = villain_combos
        .iter()
        .filter(|v| v.weight > 0.0 && !v.mask.intersects(river_mask))
        .map(|v| (v.mask, v.weight as f64, cache.eval_hand_unchecked(v.hand).raw()))
        .collect();

    hero_combos
        .iter()
        .map(|h| {
            if h.mask.intersects(river_mask) {
                return 0.0;
            }
            let hr = eval_plo_river(h.hand, river).raw();
            let mut pot_share = 0.0f64;
            let mut denom = 0.0f64;
            for &(vm, vw, vr) in &v_eval {
                if h.mask.intersects(vm) {
                    continue;
                }
                denom += vw;
                if hr > vr {
                    pot_share += vw;
                } else if hr == vr {
                    pot_share += 0.5 * vw;
                }
            }
            if denom > 0.0 {
                (pot_share / denom) as f32
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use plo_eval::slow_eval_plo_lex;

    #[test]
    fn plo_range_syntax_combinatorics() {
        // AAKK total in PLO: C(4,2) * C(4,2) = 6 * 6 = 36 combos!
        let aakk_all: PloRange = "AAKK".parse().unwrap();
        assert_eq!(aakk_all.len(), 36);

        // AAKK$ds (double-suited): C(4,2) = 6 combos!
        let aakk_ds: PloRange = "AAKK$ds".parse().unwrap();
        assert_eq!(aakk_ds.len(), 6);

        // AAKK$ss (single-suited): 24 combos!
        let aakk_ss: PloRange = "AAKK$ss".parse().unwrap();
        assert_eq!(aakk_ss.len(), 24);

        // AAKK$rb (rainbow): 6 combos! (6 + 24 + 6 = 36, since AAKK cannot be monotone or trips-suited!)
        let aakk_rb: PloRange = "AAKK$rb".parse().unwrap();
        assert_eq!(aakk_rb.len(), 6);

        // AKQJ total: 4^4 = 256 combos!
        let akqj: PloRange = "AKQJ".parse().unwrap();
        assert_eq!(akqj.len(), 256);

        // (AK)(QJ): 4 choices for AK's suit * 3 choices for QJ's suit = 12 combos!
        let ak_qj_grouped: PloRange = "(AK)(QJ)".parse().unwrap();
        assert_eq!(ak_qj_grouped.len(), 12);

        // AKQJ$ds: 3 ways to pair the 4 ranks ({AK,QJ}, {AQ,KJ}, {AJ,KQ}) * 12 = 36 combos!
        let akqj_ds: PloRange = "AKQJ$ds".parse().unwrap();
        assert_eq!(akqj_ds.len(), 36);

        // Weighted overwrite: AAKK@0.5, AAKK$ds@1.0 -> 30 combos @ 0.5 + 6 combos @ 1.0 = total weight 21.0
        let mixed: PloRange = "AAKK@50%, AAKK$ds@100%".parse().unwrap();
        assert_eq!(mixed.len(), 36);
        assert!((mixed.total_weight() - 21.0).abs() < 1e-5);
    }

    #[test]
    fn blockers_remove_board_and_shared_hole_cards() {
        let board: Board = "AsKs2d".parse().unwrap();
        let aakk: PloRange = "AAKK".parse().unwrap(); // 36 combos before board
        // With As and Ks on board, remaining A's = {Ac,Ad,Ah} (C(3,2)=3) and remaining K's = {Kc,Kd,Kh} (C(3,2)=3) -> 3*3 = 9 live combos!
        let live = aakk.without_dead(board.mask());
        assert_eq!(live.len(), 9);
        for e in live.entries() {
            assert!(!e.hand.is_blocked_by(board.mask()));
        }

        // Now if Hero holds [AcAdKcKd], Villain's remaining AAKK combos must use {Ah} and {Kh} — impossible to have two Aces!
        // So Hero's [AcAdKcKd] + board [AsKs2d] completely blocks 100% of Villain's AAKK!
        let hero: Hand4 = "AcAdKcKd".parse().unwrap();
        assert!(equity_hand_vs_range(hero, &aakk, &board).is_err());
    }

    #[test]
    fn hand_vs_hand_equity_matches_independent_slow_bruteforce() {
        // Classic PLO flop war: Top set of Aces [AhAd7c2s] vs Monster wrap + flush draw [QhJhTd9s] on [AsTc8h]
        let set_aces: Hand4 = "AhAd7c2s".parse().unwrap();
        let monster_wrap: Hand4 = "QhJhTd9s".parse().unwrap();
        let flop: Board = "AsTc8h".parse().unwrap();

        let fast_eq = equity_hand_vs_hand(set_aces, monster_wrap, &flop).unwrap();
        // C(52 - 3 - 4 - 4, 2) = C(41, 2) = 820 exact turn+river runouts!
        assert_eq!(fast_eq.valid_weight_sum as u64, 820);

        // Verify independently with slow_eval_plo_lex over all 820 runouts
        let used = flop.mask() | set_aces.mask() | monster_wrap.mask();
        let mut slow_w0 = 0u64;
        let mut slow_w1 = 0u64;
        let mut slow_tie = 0u64;
        let b = flop.as_slice();
        for t in 0..52u8 {
            let tc = Card::new(t).unwrap();
            if used.contains(tc) {
                continue;
            }
            for r in (t + 1)..52u8 {
                let rc = Card::new(r).unwrap();
                if used.contains(rc) {
                    continue;
                }
                let rboard = Board::river([b[0], b[1], b[2], tc, rc]).unwrap();
                let s0 = slow_eval_plo_lex(set_aces, &rboard).unwrap().0;
                let s1 = slow_eval_plo_lex(monster_wrap, &rboard).unwrap().0;
                match s0.cmp(&s1) {
                    core::cmp::Ordering::Greater => slow_w0 += 1,
                    core::cmp::Ordering::Less => slow_w1 += 1,
                    core::cmp::Ordering::Equal => slow_tie += 1,
                }
            }
        }
        let expected_eq0 = (slow_w0 as f64 + 0.5 * slow_tie as f64) / 820.0;
        let expected_eq1 = (slow_w1 as f64 + 0.5 * slow_tie as f64) / 820.0;
        assert!((fast_eq.equity_p0 - expected_eq0).abs() < 1e-12);
        assert!((fast_eq.equity_p1 - expected_eq1).abs() < 1e-12);
        // Sanity check PLO intuition: big wrap + flush draw has ~45-55% equity vs bare top set!
        assert!(fast_eq.equity_p0 > 0.35 && fast_eq.equity_p0 < 0.65);
    }

    #[test]
    fn range_vs_range_symmetry_and_turn_verification() {
        let turn: Board = "Ks9s5d2c".parse().unwrap();
        let r0: PloRange = "AAKK$ds, (AQ)(JT)".parse().unwrap();
        let r1: PloRange = "QQJJ$ds, (AJ)(T9)".parse().unwrap();

        // Symmetric range vs itself MUST have exact 50.000000% equity!
        let self_eq = equity_range_vs_range(&r0, &r0, &turn).unwrap();
        assert!((self_eq.equity_p0 - 0.5).abs() < 1e-12);

        // Swapping P0 and P1 must swap equities (`eq0 + eq1 == 1.0`)
        let eq_01 = equity_range_vs_range(&r0, &r1, &turn).unwrap();
        let eq_10 = equity_range_vs_range(&r1, &r0, &turn).unwrap();
        assert!((eq_01.equity_p0 - eq_10.equity_p1).abs() < 1e-12);
        assert!((eq_01.win_p0 - eq_10.win_p1).abs() < 1e-12);
        assert!((eq_01.tie - eq_10.tie).abs() < 1e-12);
    }

    #[test]
    fn phase_a4_turn_and_flop_range_vs_range_equity_match_independent_slow_bruteforce() {
        // Independent pairwise + slow_eval_plo_lex oracle for Turn (`40` rivers per valid pair) and Flop (`820` runouts per valid pair),
        // specifically including overlapping 4-card hands, board-blocked hands, and non-uniform fractional weights!
        let r0: PloRange = "AsKsQhJh@0.8, AdKd9h8h@1.0, QcJcTh9h@0.5, 7s6s5d4d@0.35"
            .parse()
            .unwrap();
        let r1: PloRange = "AsKsQhJh@0.6, AcKcTh9h@0.9, QdJd8c7c@0.75, 2c3c4h5h@1.0"
            .parse()
            .unwrap();

        // 1) Turn board (`7s` on board blocks `7s6s5d4d` in `r0`!)
        let turn: Board = "7s8d9s2h".parse().unwrap();
        let fast_turn = equity_range_vs_range(&r0, &r1, &turn).unwrap();

        let b_turn = turn.as_slice();
        let mut t_win0 = 0.0f64;
        let mut t_win1 = 0.0f64;
        let mut t_tie = 0.0f64;
        for c0 in r0.entries() {
            if c0.mask.intersects(turn.mask()) {
                continue;
            }
            for c1 in r1.entries() {
                if c1.mask.intersects(turn.mask()) || c0.mask.intersects(c1.mask) {
                    continue;
                }
                let dead = turn.mask() | c0.mask | c1.mask;
                let ww = (c0.weight as f64) * (c1.weight as f64);
                let mut pair_rivers = 0usize;
                for rid in 0..52u8 {
                    let rc = Card::new(rid).unwrap();
                    if dead.contains(rc) {
                        continue;
                    }
                    pair_rivers += 1;
                    let rb = Board::river([b_turn[0], b_turn[1], b_turn[2], b_turn[3], rc]).unwrap();
                    let s0 = slow_eval_plo_lex(c0.hand, &rb).unwrap().0;
                    let s1 = slow_eval_plo_lex(c1.hand, &rb).unwrap().0;
                    match s0.cmp(&s1) {
                        core::cmp::Ordering::Greater => t_win0 += ww,
                        core::cmp::Ordering::Less => t_win1 += ww,
                        core::cmp::Ordering::Equal => t_tie += ww,
                    }
                }
                assert_eq!(pair_rivers, 40, "every valid PLO turn pair has exactly 40 river cards");
            }
        }
        let t_tot = t_win0 + t_win1 + t_tie;
        assert!((fast_turn.valid_weight_sum - t_tot).abs() < 1e-9);
        assert!((fast_turn.win_p0 - t_win0 / t_tot).abs() < 1e-12);
        assert!((fast_turn.win_p1 - t_win1 / t_tot).abs() < 1e-12);
        assert!((fast_turn.tie - t_tie / t_tot).abs() < 1e-12);
        assert!((fast_turn.equity_p0 - (t_win0 + 0.5 * t_tie) / t_tot).abs() < 1e-12);

        // 2) Flop board (`820` unordered turn+river runouts per valid pair!)
        let flop: Board = "7s8d9s".parse().unwrap();
        let fast_flop = equity_range_vs_range(&r0, &r1, &flop).unwrap();
        let b_flop = flop.as_slice();
        let mut f_win0 = 0.0f64;
        let mut f_win1 = 0.0f64;
        let mut f_tie = 0.0f64;
        for c0 in r0.entries() {
            if c0.mask.intersects(flop.mask()) {
                continue;
            }
            for c1 in r1.entries() {
                if c1.mask.intersects(flop.mask()) || c0.mask.intersects(c1.mask) {
                    continue;
                }
                let dead = flop.mask() | c0.mask | c1.mask;
                let ww = (c0.weight as f64) * (c1.weight as f64);
                let mut pair_runouts = 0usize;
                for tid in 0..52u8 {
                    let tc = Card::new(tid).unwrap();
                    if dead.contains(tc) {
                        continue;
                    }
                    for rid in (tid + 1)..52u8 {
                        let rc = Card::new(rid).unwrap();
                        if dead.contains(rc) {
                            continue;
                        }
                        pair_runouts += 1;
                        let rb = Board::river([b_flop[0], b_flop[1], b_flop[2], tc, rc]).unwrap();
                        let s0 = slow_eval_plo_lex(c0.hand, &rb).unwrap().0;
                        let s1 = slow_eval_plo_lex(c1.hand, &rb).unwrap().0;
                        match s0.cmp(&s1) {
                            core::cmp::Ordering::Greater => f_win0 += ww,
                            core::cmp::Ordering::Less => f_win1 += ww,
                            core::cmp::Ordering::Equal => f_tie += ww,
                        }
                    }
                }
                assert_eq!(pair_runouts, 820, "every valid PLO flop pair has exactly C(41,2)=820 runouts");
            }
        }
        let f_tot = f_win0 + f_win1 + f_tie;
        assert!((fast_flop.valid_weight_sum - f_tot).abs() < 1e-8);
        assert!((fast_flop.equity_p0 - (f_win0 + 0.5 * f_tie) / f_tot).abs() < 1e-12);
        assert!((fast_flop.tie - f_tie / f_tot).abs() < 1e-12);
    }

    // ========================================================================
    // PHASE E: Frozen PLO Range Parser Golden Tests & Blocker Weight Accounting
    // ========================================================================

    #[test]
    fn phase_e_range_parser_golden_combo_lists_and_blocker_accounting() {
        // 1) Exact combo literal + weight
        let single: PloRange = "AsKsQhJh".parse().unwrap();
        assert_eq!(single.len(), 1);
        let expected_h: Hand4 = "AsKsQhJh".parse().unwrap();
        assert_eq!(single.entries()[0].hand, expected_h);
        assert_eq!(single.entries()[0].combo_id, expected_h.combo_id());
        assert!((single.entries()[0].weight - 1.0).abs() < 1e-6);

        let weighted_single: PloRange = "Ah9s6d5d@0.5".parse().unwrap();
        assert_eq!(weighted_single.len(), 1);
        assert_eq!(
            weighted_single.entries()[0].hand,
            "Ah9s6d5d".parse::<Hand4>().unwrap()
        );
        assert!((weighted_single.entries()[0].weight - 0.5).abs() < 1e-6);

        // 2) Golden exact 6-combo list for `AAKK$ds`!
        let aakk_ds: PloRange = "AAKK$ds".parse().unwrap();
        let golden_aakk_ds: Vec<Hand4> = [
            "AcKcAdKd",
            "AcKcAhKh",
            "AcKcAsKs",
            "AdKdAhKh",
            "AdKdAsKs",
            "AhKhAsKs",
        ]
        .iter()
        .map(|s| s.parse::<Hand4>().unwrap())
        .collect();
        let mut expected_cids: Vec<ComboId> = golden_aakk_ds.iter().map(|h| h.combo_id()).collect();
        expected_cids.sort_unstable();
        let actual_cids: Vec<ComboId> = aakk_ds.entries().iter().map(|e| e.combo_id).collect();
        assert_eq!(actual_cids, expected_cids);

        // 3) Golden exact 12-combo list for `(AK)(QJ)` AND `(AJ)(98)`!
        let ak_qj: PloRange = "(AK)(QJ)".parse().unwrap();
        assert_eq!(ak_qj.len(), 12);
        let suits = ['c', 'd', 'h', 's'];
        let mut expected_ak_qj = Vec::new();
        let mut expected_aj_98 = Vec::new();
        for &s1 in &suits {
            for &s2 in &suits {
                if s1 == s2 {
                    continue;
                }
                let h1: Hand4 = format!("A{s1}K{s1}Q{s2}J{s2}").parse().unwrap();
                let h2: Hand4 = format!("A{s1}J{s1}9{s2}8{s2}").parse().unwrap();
                expected_ak_qj.push(h1.combo_id());
                expected_aj_98.push(h2.combo_id());
            }
        }
        expected_ak_qj.sort_unstable();
        expected_aj_98.sort_unstable();
        assert_eq!(
            ak_qj.entries().iter().map(|e| e.combo_id).collect::<Vec<_>>(),
            expected_ak_qj
        );
        let aj_98: PloRange = "(AJ)(98)".parse().unwrap();
        assert_eq!(
            aj_98.entries().iter().map(|e| e.combo_id).collect::<Vec<_>>(),
            expected_aj_98
        );

        // Disjointness between `(AK)(QJ)` and `(AQ)(KJ)`: ZERO overlapping ComboIds!
        let aq_kj: PloRange = "(AQ)(KJ)".parse().unwrap();
        assert_eq!(aq_kj.len(), 12);
        for e in ak_qj.entries() {
            assert_eq!(aq_kj.weight_of(e.hand), 0.0);
        }

        // 4) Exact partition of `AKQJ` (256 combos) = `$ds` (36) + `$ss` (144) + `$rb` (24) + `$ts` (48) + `$ms` (4)!
        let akqj_all: PloRange = "AKQJ".parse().unwrap();
        let p_ds: PloRange = "AKQJ$ds".parse().unwrap();
        let p_ss: PloRange = "AKQJ$ss".parse().unwrap();
        let p_rb: PloRange = "AKQJ$rb".parse().unwrap();
        let p_ts: PloRange = "AKQJ$ts".parse().unwrap();
        let p_ms: PloRange = "AKQJ$ms".parse().unwrap();
        assert_eq!(
            (p_ds.len(), p_ss.len(), p_rb.len(), p_ts.len(), p_ms.len()),
            (36, 144, 24, 48, 4)
        );
        assert_eq!(
            p_ds.len() + p_ss.len() + p_rb.len() + p_ts.len() + p_ms.len(),
            256
        );
        let union: PloRange = "AKQJ$ds, AKQJ$ss, AKQJ$rb, AKQJ$ts, AKQJ$ms".parse().unwrap();
        assert_eq!(
            union.entries().iter().map(|e| e.combo_id).collect::<Vec<_>>(),
            akqj_all.entries().iter().map(|e| e.combo_id).collect::<Vec<_>>()
        );

        // 5) Weighted mix + removal (`@0`) + board & opponent-hand blocker filtering!
        let mix: PloRange = "AAKK@0.25, AAKK$ds@1.0, AcKcAdKd@0".parse().unwrap();
        // 36 total - 1 removed = 35 combos (`5` ds @ 1.0 + `30` non-ds @ 0.25 = `12.5` total weight)
        assert_eq!(mix.len(), 35);
        assert!((mix.total_weight() - 12.5).abs() < 1e-6);

        // Board blocker filtering on `[As Ks Qh]` + opponent hand `[Ad Kd Jh Th]`:
        let flop: Board = "AsKsQh".parse().unwrap();
        let opp_hand: Hand4 = "AdKdJhTh".parse().unwrap();
        // Remaining Aces in deck after `As` (board) and `Ad` (opp) = `{Ac, Ah}` (only 1 pair of Aces!)
        // Remaining Kings in deck after `Ks` (board) and `Kd` (opp) = `{Kc, Kh}` (only 1 pair of Kings!)
        // So ONLY ONE `AAKK` combo (`AcKcAhKh`, which is `$ds` with weight `1.0` in `mix`) survives both board + opp_hand!
        let live_after_board_and_opp = mix.without_dead(flop.mask() | opp_hand.mask());
        assert_eq!(live_after_board_and_opp.len(), 1);
        assert_eq!(
            live_after_board_and_opp.entries()[0].hand,
            "AcKcAhKh".parse::<Hand4>().unwrap()
        );
        assert!((live_after_board_and_opp.total_weight() - 1.0).abs() < 1e-6);

        // 6) Explicit anti-Hold'em test-fail lock: 2-card / 13x13 Hold'em range strings MUST be rejected!
        for illegal_holdem_token in ["AA", "AKs", "AKo", "QQ+", "AsKs", "JJ-99"] {
            assert!(
                illegal_holdem_token.parse::<PloRange>().is_err(),
                "Hold'em token `{illegal_holdem_token}` must NEVER parse as a PLO-4 range!"
            );
        }

        // 7) `100%` full-universe token & `from_hands` `[0.0..=1.0]` weight validation:
        let full_pct: PloRange = "100%".parse().unwrap();
        assert_eq!(full_pct.len(), 270_725);
        let h_sample: Hand4 = "AsKsQhJh".parse().unwrap();
        assert!(PloRange::from_hands([(h_sample, 1.5)]).is_err());
        assert!(PloRange::from_hands([(h_sample, -0.1)]).is_err());
        assert!(PloRange::from_hands([(h_sample, f32::NAN)]).is_err());
    }
}
