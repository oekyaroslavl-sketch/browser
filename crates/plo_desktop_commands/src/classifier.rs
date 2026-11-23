//! PLO-4 postflop hand & draw classifier and PPT/Monker-style hand filter for the Desktop UI.
//!
//! Strictly obeys the **PLO 2-hole + 3-board** evaluation rule via `plo_eval::eval_plo`
//! (never Hold'em 7-card evaluation). Used purely for presentation, grouping, and filtering
//! in `plo_desktop_commands` — zero impact on solver regrets or game-tree math.

use plo_cards::{Board, Card, CardMask, Hand4, Street};
use plo_eval::{eval_plo, HandCategory};

/// Ordered PLO-4 postflop hand/draw categories (strongest made hands first, then key PLO draws & blockers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PloHandClass {
    FullHousePlus = 0,
    Flush = 1,
    Straight = 2,
    SetTrips = 3,
    TwoPair = 4,
    Overpair = 5,
    TopPair = 6,
    ComboDraw = 7,
    NutFlushDraw = 8,
    FlushDraw = 9,
    WrapDraw = 10,
    StraightDraw = 11,
    OtherPair = 12,
    NutBlockerAir = 13,
    Air = 14,
}

pub const ALL_PLO_HAND_CLASSES: [PloHandClass; 15] = [
    PloHandClass::FullHousePlus,
    PloHandClass::Flush,
    PloHandClass::Straight,
    PloHandClass::SetTrips,
    PloHandClass::TwoPair,
    PloHandClass::Overpair,
    PloHandClass::TopPair,
    PloHandClass::ComboDraw,
    PloHandClass::NutFlushDraw,
    PloHandClass::FlushDraw,
    PloHandClass::WrapDraw,
    PloHandClass::StraightDraw,
    PloHandClass::OtherPair,
    PloHandClass::NutBlockerAir,
    PloHandClass::Air,
];

impl PloHandClass {
    pub const fn id(self) -> &'static str {
        match self {
            Self::FullHousePlus => "fh_plus",
            Self::Flush => "flush",
            Self::Straight => "straight",
            Self::SetTrips => "set_trips",
            Self::TwoPair => "two_pair",
            Self::Overpair => "overpair",
            Self::TopPair => "top_pair",
            Self::ComboDraw => "combo_draw",
            Self::NutFlushDraw => "nut_fd",
            Self::FlushDraw => "fd",
            Self::WrapDraw => "wrap",
            Self::StraightDraw => "oesd_gs",
            Self::OtherPair => "other_pair",
            Self::NutBlockerAir => "nut_blocker",
            Self::Air => "air",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::FullHousePlus => "Quads / Full House+",
            Self::Flush => "Flush",
            Self::Straight => "Straight",
            Self::SetTrips => "Set / Trips",
            Self::TwoPair => "Two Pair",
            Self::Overpair => "Overpair",
            Self::TopPair => "Top Pair",
            Self::ComboDraw => "Combo Draw (FD + Str)",
            Self::NutFlushDraw => "Nut Flush Draw",
            Self::FlushDraw => "Non-Nut Flush Draw",
            Self::WrapDraw => "Wrap (9+ Str Outs)",
            Self::StraightDraw => "Straight Draw (4–8 Outs)",
            Self::OtherPair => "Mid / Low Pair",
            Self::NutBlockerAir => "Bare Nut Flush Blocker",
            Self::Air => "Air / Backdoors",
        }
    }

    pub const fn short_tag(self) -> &'static str {
        match self {
            Self::FullHousePlus => "FH+",
            Self::Flush => "Flush",
            Self::Straight => "Str",
            Self::SetTrips => "Set",
            Self::TwoPair => "2Pair",
            Self::Overpair => "OverP",
            Self::TopPair => "TopP",
            Self::ComboDraw => "ComboD",
            Self::NutFlushDraw => "NutFD",
            Self::FlushDraw => "FD",
            Self::WrapDraw => "Wrap",
            Self::StraightDraw => "StrD",
            Self::OtherPair => "Pair",
            Self::NutBlockerAir => "NutBlk",
            Self::Air => "Air",
        }
    }
}

/// Detailed PLO-4 structural tags for a `(Hand4, Board)` pair (used both for primary bucket
/// classification and for rich search/filter queries like `"nfd"`, `"wrap"`, `"set"`, `"$ds"`).
#[derive(Clone, Copy, Debug)]
pub struct HandBoardFeatures {
    pub primary_class: PloHandClass,
    pub made_category: HandCategory,
    pub has_nut_fd: bool,
    pub has_any_fd: bool,
    pub straight_out_ranks: u8,
    pub has_nut_flush_blocker: bool,
    pub is_double_suited: bool,
    pub is_single_suited: bool,
    pub is_rainbow: bool,
}

/// Classifies a live 4-card PLO hand on `board` (`3`, `4`, or `5` cards) using exact 2+3 rules.
pub fn analyze_plo_hand(hand: Hand4, board: &Board) -> HandBoardFeatures {
    let cards = hand.cards();
    let b_slice = board.as_slice();
    let street = board.street();

    // Suit pattern of the 4 hole cards ($ds / $ss / $r)
    let mut h_suit_counts = [0u8; 4];
    for c in cards {
        h_suit_counts[c.suit() as usize] += 1;
    }
    let pairs_of_suits = h_suit_counts.iter().filter(|&&n| n >= 2).count();
    let max_h_suit = *h_suit_counts.iter().max().unwrap_or(&0);
    let is_double_suited = pairs_of_suits == 2;
    let is_rainbow = max_h_suit == 1;
    let is_single_suited = !is_double_suited && !is_rainbow;

    // Exact PLO 2+3 current made-hand evaluation
    let rank = eval_plo(hand, board).unwrap_or_default();
    let made_cat = rank.category();

    // Board suit & rank structure
    let mut b_suit_counts = [0u8; 4];
    let mut b_rank_counts = [0u8; 13];
    let mut max_board_rank = 0u8;
    for &bc in b_slice {
        b_suit_counts[bc.suit() as usize] += 1;
        b_rank_counts[bc.rank() as usize] += 1;
        if bc.rank() > max_board_rank {
            max_board_rank = bc.rank();
        }
    }

    // Flush draws (only on Flop/Turn when board has exactly 2 of a suit and hand has >= 2 of that suit)
    // and bare nut-flush blocker (board has >= 2 of suit `s` on Flop/Turn or >= 3 on River,
    // and hand holds the highest non-board card of suit `s` with < 2 cards of suit `s` on Flop/Turn or ==1 on 3-flush river).
    let mut has_any_fd = false;
    let mut has_nut_fd = false;
    let mut has_nut_flush_blocker = false;
    let bmask = board.mask();

    for s in 0..4u8 {
        let bs = b_suit_counts[s as usize];
        let hs = h_suit_counts[s as usize];

        // Highest available rank of suit `s` not on the board
        let mut nut_card_opt: Option<Card> = None;
        for r in (0..13u8).rev() {
            if let Ok(c) = Card::from_rank_suit(r, s) {
                if !bmask.contains(c) {
                    nut_card_opt = Some(c);
                    break;
                }
            }
        }
        let holds_nut_card = nut_card_opt
            .map(|nc| hand.mask().contains(nc))
            .unwrap_or(false);

        if street != Street::River && bs == 2 && hs >= 2 && made_cat < HandCategory::Flush {
            has_any_fd = true;
            if holds_nut_card {
                has_nut_fd = true;
            }
        }

        // Bare nut flush blocker:
        // - On River with >=3 suited board cards: holding the nut card of that suit without making a flush (`hs == 1`).
        // - On Flop/Turn with >=2 suited board cards: holding the nut card of that suit with `hs == 1`.
        if holds_nut_card && hs == 1 && ((street == Street::River && bs >= 3) || (street != Street::River && bs >= 2)) {
            has_nut_flush_blocker = true;
        }
    }

    // Straight outs on Flop / Turn (exact count of distinct next-card ranks `0..13` that give a 2+3 Straight
    // when dealt in an offsuit/non-flush-completing suit, when hand doesn't already have Straight+).
    let mut straight_out_ranks = 0u8;
    if street != Street::River && made_cat < HandCategory::Straight {
        let dead: CardMask = bmask | hand.mask();
        for r in 0..13u8 {
            // Pick a live card of rank `r` that ideally doesn't complete a 3-suit board so we count true straight outs
            let mut test_card: Option<Card> = None;
            for s in 0..4u8 {
                if let Ok(c) = Card::from_rank_suit(r, s) {
                    if !dead.contains(c) {
                        if b_suit_counts[s as usize] < 2 {
                            test_card = Some(c);
                            break;
                        } else if test_card.is_none() {
                            test_card = Some(c);
                        }
                    }
                }
            }
            if let Some(tc) = test_card {
                if let Ok(next_b) = board.deal(tc) {
                    if let Ok(nr) = eval_plo(hand, &next_b) {
                        if nr.category() == HandCategory::Straight {
                            straight_out_ranks += 1;
                        }
                    }
                }
            }
        }
    }

    // Distinguish Overpair vs Top Pair vs Other Pair when `made_cat == HandCategory::OnePair`
    let mut is_overpair = false;
    let mut is_top_pair = false;
    if made_cat == HandCategory::OnePair {
        let board_paired = b_rank_counts.iter().any(|&n| n >= 2);
        let mut h_rank_counts = [0u8; 13];
        for c in cards {
            h_rank_counts[c.rank() as usize] += 1;
        }
        // In PLO 2+3, if the board is unpaired (or the best 5-card hand uses a pocket pair > max_board_rank):
        for r in ((max_board_rank as usize + 1)..13).rev() {
            if h_rank_counts[r] >= 2 {
                is_overpair = true;
                break;
            }
        }
        if !is_overpair && !board_paired && h_rank_counts[max_board_rank as usize] >= 1 {
            is_top_pair = true;
        } else if !is_overpair && board_paired {
            // If board is paired (e.g. K77) and player hits K (`h_rank_counts[K] >= 1`), that's TwoPair, not OnePair.
            // So OnePair on a paired board means using the board pair + 2 hole kickers, or a pocket pair on a 3-card unpaired subset.
            if h_rank_counts[max_board_rank as usize] >= 1 && b_rank_counts[max_board_rank as usize] == 1 {
                is_top_pair = true;
            }
        }
    }

    // Primary mutually-exclusive GTO+ breakdown class:
    // Strong made hands (TwoPair+) take precedence; for OnePair / HighCard on Flop/Turn, strong draws
    // (ComboDraw, NutFD, Wrap >=3 ranks / 9+ outs, FD, StraightDraw >=2 ranks / 6+ outs) take precedence
    // over weak pairs so PLO draw-heavy hands are categorized the way PLO players actually think!
    let has_wrap = straight_out_ranks >= 3; // >= 3 distinct completing ranks (~9–20 outs)
    let has_str_draw = straight_out_ranks >= 1; // 1–2 completing ranks (4–8 outs)

    let primary_class = match made_cat {
        HandCategory::StraightFlush | HandCategory::FourOfAKind | HandCategory::FullHouse => {
            PloHandClass::FullHousePlus
        }
        HandCategory::Flush => PloHandClass::Flush,
        HandCategory::Straight => PloHandClass::Straight,
        HandCategory::ThreeOfAKind => PloHandClass::SetTrips,
        HandCategory::TwoPair => PloHandClass::TwoPair,
        HandCategory::OnePair | HandCategory::HighCard => {
            if street != Street::River && has_any_fd && straight_out_ranks >= 2 {
                PloHandClass::ComboDraw
            } else if is_overpair {
                PloHandClass::Overpair
            } else if street != Street::River && has_nut_fd {
                PloHandClass::NutFlushDraw
            } else if street != Street::River && has_wrap {
                PloHandClass::WrapDraw
            } else if is_top_pair {
                PloHandClass::TopPair
            } else if street != Street::River && has_any_fd {
                PloHandClass::FlushDraw
            } else if street != Street::River && has_str_draw {
                PloHandClass::StraightDraw
            } else if made_cat == HandCategory::OnePair {
                PloHandClass::OtherPair
            } else if has_nut_flush_blocker {
                PloHandClass::NutBlockerAir
            } else {
                PloHandClass::Air
            }
        }
    };

    HandBoardFeatures {
        primary_class,
        made_category: made_cat,
        has_nut_fd,
        has_any_fd,
        straight_out_ranks,
        has_nut_flush_blocker,
        is_double_suited,
        is_single_suited,
        is_rainbow,
    }
}

/// Evaluates whether a 4-card PLO `hand` (with precomputed `features` on `board`) matches a user
/// filter query string (supports comma-separated OR clauses and `+` / space-separated AND tokens:
/// e.g. `"AA"`, `"$ds"`, `"As"`, `"nfd"`, `"wrap"`, `"set"`, `"AA $ds"`, `"flush, fullhouse"`).
pub fn matches_hand_filter(hand: Hand4, features: &HandBoardFeatures, query: &str) -> bool {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return true;
    }
    let hand_str_lower = hand.to_string().to_ascii_lowercase();
    let cards = hand.cards();
    let mut rank_counts = [0u8; 13];
    for c in cards {
        rank_counts[c.rank() as usize] += 1;
    }

    // Comma separates OR clauses; whitespace or `+` separates AND conditions within a clause.
    for or_clause in trimmed.split(',') {
        let clause = or_clause.trim();
        if clause.is_empty() {
            continue;
        }
        let and_tokens: Vec<&str> = clause
            .split(|ch: char| ch.is_whitespace() || ch == '+')
            .filter(|t| !t.is_empty())
            .collect();
        if and_tokens.is_empty() {
            continue;
        }
        if and_tokens
            .iter()
            .all(|tok| match_single_token(hand, &hand_str_lower, &rank_counts, features, tok))
        {
            return true;
        }
    }
    false
}

fn match_single_token(
    hand: Hand4,
    hand_str_lower: &str,
    rank_counts: &[u8; 13],
    f: &HandBoardFeatures,
    raw_tok: &str,
) -> bool {
    let (negated, tok) = if let Some(rest) = raw_tok.strip_prefix('!') {
        (true, rest)
    } else if let Some(rest) = raw_tok.strip_prefix('-') {
        (true, rest)
    } else {
        (false, raw_tok)
    };
    if tok.is_empty() {
        return true;
    }
    let res = match_positive_token(hand, hand_str_lower, rank_counts, f, tok);
    if negated {
        !res
    } else {
        res
    }
}

fn match_positive_token(
    hand: Hand4,
    hand_str_lower: &str,
    rank_counts: &[u8; 13],
    f: &HandBoardFeatures,
    tok: &str,
) -> bool {
    let low = tok.to_ascii_lowercase();
    match low.as_str() {
        "$ds" | "ds" | "doublesuited" => return f.is_double_suited,
        "$ss" | "ss" | "singlesuited" => return f.is_single_suited,
        "$r" | "$rb" | "rb" | "rainbow" => return f.is_rainbow,
        "$np" | "np" => return rank_counts.iter().all(|&c| c <= 1),
        "$op" | "paired" => return rank_counts.iter().any(|&c| c >= 2),
        "fh" | "fh+" | "fullhouse" | "quads" | "boat" => {
            return f.primary_class == PloHandClass::FullHousePlus
        }
        "flush" | "fl" => return f.made_category == HandCategory::Flush,
        "straight" | "str" => return f.made_category == HandCategory::Straight,
        "set" | "trips" | "set_trips" => return f.made_category == HandCategory::ThreeOfAKind,
        "2p" | "twopair" | "two_pair" => return f.made_category == HandCategory::TwoPair,
        "overpair" | "op" => return f.primary_class == PloHandClass::Overpair,
        "toppair" | "tp" | "top_pair" => return f.primary_class == PloHandClass::TopPair,
        "pair" | "1p" => return f.made_category == HandCategory::OnePair,
        "combodraw" | "combo" | "combo_draw" => return f.primary_class == PloHandClass::ComboDraw,
        "nfd" | "nutfd" | "nut_fd" | "nutflushdraw" => return f.has_nut_fd,
        "fd" | "flushdraw" => return f.has_any_fd,
        "wrap" => return f.straight_out_ranks >= 3,
        "sd" | "oesd" | "gs" | " gutshot" | "straightdraw" => return f.straight_out_ranks >= 1,
        "blocker" | "nutblocker" | "nut_blocker" | "nfb" => return f.has_nut_flush_blocker,
        "air" => return f.primary_class == PloHandClass::Air,
        _ => {}
    }

    // Exact 2-char specific card token like `As`, `Td`, `9h` (case-insensitive rank, lowercase suit)
    if tok.len() == 2 {
        let bytes = tok.as_bytes();
        if let (Some(r), Some(s)) = (
            Card::parse_rank_char(bytes[0].to_ascii_uppercase()),
            Card::parse_suit_char(bytes[1].to_ascii_lowercase()),
        ) {
            if let Ok(card) = Card::from_rank_suit(r, s) {
                return hand.mask().contains(card);
            }
        }
    }

    // Pure rank multiset token like `AA`, `KK`, `AKQ`, `JT98` (requires holding at least those rank multiplicities)
    if tok
        .bytes()
        .all(|b| Card::parse_rank_char(b.to_ascii_uppercase()).is_some())
        && tok.len() <= 4
    {
        let mut req = [0u8; 13];
        for b in tok.bytes() {
            if let Some(r) = Card::parse_rank_char(b.to_ascii_uppercase()) {
                req[r as usize] += 1;
            }
        }
        return (0..13).all(|r| rank_counts[r] >= req[r]);
    }

    // Fallback: substring match on exact hand string (e.g. `AsKs`)
    hand_str_lower.contains(&low)
}
