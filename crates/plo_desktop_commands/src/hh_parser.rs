//! Hand History Parser (`#3`) for PokerStars, GGPoker / Rush & Cash, and CoinPoker PLO-4 logs!
//!
//! Pure-`std`, zero-dependency parser extracting:
//! - Hero's exact 4 hole cards (`Dealt to <Hero> [...]`)
//! - Postflop Board (`Flop` / `Turn` / `River` up to the latest dealt street or requested street)
//! - Table positions (`BTN`, `SB`, `BB`, `UTG`, `MP`, `CO`), Hero Seat (`OOP` / `IP` heads-up on flop),
//!   and automatic mapping to one of the 15 canonical `SpotCatalog` IDs (`RFI_BTN`, `BB_DEF_VS_1_RAISER`,
//!   `THREEBET_OOP_SB_VS_30`, `THREEBET_IP_VS_15_20`, `CC_IP_BTN_VS_15`, etc.)
//! - Starting Pot at the start of postflop (or target street) and Effective Stack (converted cleanly
//!   to integer chips or `100bb`-normalized integer chips when stakes use `$0.50/$1.00` decimals!).

use plo_cards::{Board, Card, Hand4};
use crate::formats::Json;

#[derive(Clone, Debug)]
pub struct PostflopActionRecord {
    pub street: String,      // "Flop" | "Turn" | "River"
    pub actor_is_hero: bool,
    pub action_kind: String, // "check" | "call" | "fold" | "bet" | "raise"
    pub amount_chips: i32,   // scaled integer bet/raise-to/call amount in `SpotConfig` chip units
    pub raw_desc: String,
}

#[derive(Clone, Debug)]
pub struct ParsedHandHistory {
    pub room: String,
    pub hero_name: String,
    pub hero_hand: Hand4,
    pub flop_board: Board,
    pub turn_card: Option<Card>,
    pub river_card: Option<Card>,
    pub final_board: Board,
    pub hero_seat: String, // "OOP" or "IP"
    pub hero_table_pos: String,
    pub villain_table_pos: String,
    pub suggested_spot_id: String,
    pub starting_pot: i32,
    pub effective_stack: i32,
    pub chips_per_bb: f64,
    pub scale_note: Option<String>,
    pub postflop_line_notes: Vec<String>,
    pub postflop_actions: Vec<PostflopActionRecord>,
}

impl ParsedHandHistory {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("room", Json::Str(self.room.clone())),
            ("hero_name", Json::Str(self.hero_name.clone())),
            ("hero_hand", Json::Str(self.hero_hand.to_string())),
            ("flop_board", Json::Str(self.flop_board.to_string())),
            ("final_board", Json::Str(self.final_board.to_string())),
            ("hero_seat", Json::Str(self.hero_seat.clone())),
            ("hero_table_pos", Json::Str(self.hero_table_pos.clone())),
            ("villain_table_pos", Json::Str(self.villain_table_pos.clone())),
            ("suggested_spot_id", Json::Str(self.suggested_spot_id.clone())),
            ("starting_pot", Json::Num(self.starting_pot as f64)),
            ("effective_stack", Json::Num(self.effective_stack as f64)),
            ("chips_per_bb", Json::Num(self.chips_per_bb)),
            (
                "turn_card",
                self.turn_card
                    .map(|c| Json::Str(c.to_string()))
                    .unwrap_or(Json::Null),
            ),
            (
                "river_card",
                self.river_card
                    .map(|c| Json::Str(c.to_string()))
                    .unwrap_or(Json::Null),
            ),
            (
                "scale_note",
                self.scale_note
                    .clone()
                    .map(Json::Str)
                    .unwrap_or(Json::Null),
            ),
            (
                "postflop_line_notes",
                Json::Arr(
                    self.postflop_line_notes
                        .iter()
                        .cloned()
                        .map(Json::Str)
                        .collect(),
                ),
            ),
        ])
    }
}

#[derive(Clone, Debug)]
struct SeatEntry {
    seat_no: usize,
    name: String,
    stack: f64,
    pos_label: String, // "BTN", "SB", "BB", "UTG", "MP", "CO"
}

const MAX_HH_INPUT_BYTES: usize = 2 * 1024 * 1024;

/// Parses a PokerStars / GGPoker / CoinPoker 4-card Pot-Limit Omaha hand history text.
pub fn parse_plo4_hand_history(raw: &str) -> Result<ParsedHandHistory, String> {
    if raw.len() > MAX_HH_INPUT_BYTES {
        return Err("Hand history text exceeds 2 MiB safety limit".to_string());
    }
    let text = raw.trim();
    if text.is_empty() {
        return Err("Hand history text is empty".to_string());
    }

    let room = if text.contains("PokerStars") {
        "PokerStars"
    } else if text.contains("GGPoker") || text.contains("Poker Hand #HD") || text.contains("Poker Hand #RC") {
        "GGPoker"
    } else if text.contains("CoinPoker") || text.contains("₮") {
        "CoinPoker"
    } else {
        "Generic PLO"
    }
    .to_string();

    let lines: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();

    // 1. Detect Big Blind size from header (e.g. `($0.50/$1.00)`, `(100/200)`, `(€1/€2)`)
    let mut bb_amount = detect_big_blind(lines.first().copied().unwrap_or("")).unwrap_or(1.0);
    if bb_amount <= 0.0 {
        bb_amount = 1.0;
    }

    // 2. Detect button seat number (`Seat #3 is the button`)
    let mut button_seat: Option<usize> = None;
    for l in &lines {
        if l.to_ascii_lowercase().contains("is the button") {
            if let Some(num) = extract_seat_hash_number(l) {
                button_seat = Some(num);
                break;
            }
        }
    }

    // 3. Parse `Seat X: PlayerName ($100.50 in chips)` before `*** HOLE CARDS ***`
    let mut seats: Vec<SeatEntry> = Vec::new();
    for l in &lines {
        if l.starts_with("*** HOLE CARDS") {
            break;
        }
        if let Some(se) = parse_seat_line(l) {
            seats.push(se);
        }
    }
    seats.sort_by_key(|s| s.seat_no);

    // Assign canonical table positions (`BTN`, `SB`, `BB`, `UTG`, `MP`, `CO`) relative to `button_seat`
    if !seats.is_empty() {
        let btn_idx = button_seat
            .and_then(|bs| seats.iter().position(|s| s.seat_no == bs))
            .unwrap_or(0);
        let n = seats.len();
        for offset in 0..n {
            let idx = (btn_idx + offset) % n;
            let pos = match (n, offset) {
                (2, 0) => "BTN", // HU BTN/SB
                (2, 1) => "BB",
                (_, 0) => "BTN",
                (_, 1) => "SB",
                (_, 2) => "BB",
                (4, 3) => "CO",
                (5, 3) => "MP",
                (5, 4) => "CO",
                (_, o) if o == n - 1 => "CO",
                (_, o) if o == n - 2 => "MP",
                _ => "UTG",
            };
            seats[idx].pos_label = pos.to_string();
        }
    }

    // 4. Parse `Dealt to <HeroName> [Ah Ks Qd Jc]`
    let mut hero_name = String::new();
    let mut hero_hand_opt: Option<Hand4> = None;
    for l in &lines {
        if let Some(rest) = l.strip_prefix("Dealt to ") {
            if let Some(open_br) = rest.find('[') {
                if let Some(close_br) = rest[open_br + 1..].find(']') {
                    let candidate_name = rest[..open_br].trim();
                    let cards_inside = &rest[open_br + 1..open_br + 1 + close_br];
                    let cards = parse_card_tokens(cards_inside)?;
                    if cards.len() == 2 {
                        return Err(
                            "Hold'em 2-card hand history detected — PLO-4 requires 4 hole cards"
                                .to_string(),
                        );
                    }
                    if cards.len() != 4 {
                        return Err(format!(
                            "Expected exactly 4 PLO hole cards in 'Dealt to', got {}",
                            cards.len()
                        ));
                    }
                    let h = Hand4::new([cards[0], cards[1], cards[2], cards[3]])
                        .map_err(|e| format!("Invalid Hero 4-card hand: {e}"))?;
                    hero_name = candidate_name.to_string();
                    hero_hand_opt = Some(h);
                    break;
                }
            }
        }
    }
    let hero_hand = hero_hand_opt
        .ok_or_else(|| "Could not find 'Dealt to <Hero> [4 cards]' in hand history".to_string())?;

    // 5. Walk Preflop & Postflop streets
    #[derive(PartialEq, Eq)]
    enum Section {
        Header,
        Preflop,
        Flop,
        Turn,
        River,
        Summary,
    }
    let mut sec = Section::Header;
    let mut preflop_invested: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    let mut preflop_antes: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    let mut preflop_raises: Vec<String> = Vec::new();
    let mut preflop_calls: Vec<String> = Vec::new();
    let mut folded_preflop: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut flop_cards: Vec<Card> = Vec::new();
    let mut turn_card: Option<Card> = None;
    let mut river_card: Option<Card> = None;
    let mut postflop_notes: Vec<String> = Vec::new();
    let mut raw_postflop_acts: Vec<(String, String, String, f64)> = Vec::new();
    let mut postflop_actors_order: Vec<String> = Vec::new();

    for l in &lines {
        if l.starts_with("*** HOLE CARDS") {
            sec = Section::Preflop;
            continue;
        } else if l.starts_with("*** FLOP") {
            sec = Section::Flop;
            let bcards = extract_all_bracket_cards(l)?;
            if bcards.len() >= 3 {
                flop_cards = bcards[..3].to_vec();
            }
            continue;
        } else if l.starts_with("*** TURN") {
            sec = Section::Turn;
            let bcards = extract_all_bracket_cards(l)?;
            if let Some(&last) = bcards.last() {
                if bcards.len() >= 4 && flop_cards.is_empty() {
                    flop_cards = bcards[..3].to_vec();
                }
                turn_card = Some(last);
            }
            continue;
        } else if l.starts_with("*** RIVER") {
            sec = Section::River;
            let bcards = extract_all_bracket_cards(l)?;
            if let Some(&last) = bcards.last() {
                river_card = Some(last);
            }
            continue;
        } else if l.starts_with("*** SHOW DOWN")
            || l.starts_with("*** SHOWDOWN")
            || l.starts_with("*** SUMMARY")
        {
            sec = Section::Summary;
            continue;
        }

        // Blinds / Antes before `*** HOLE CARDS ***` or inside Header/Preflop
        if matches!(sec, Section::Header | Section::Preflop) {
            if let Some((actor, amt, is_uncalled, is_blind_or_ante)) =
                parse_money_contribution(l, &seats)
            {
                if is_uncalled {
                    let cur = preflop_invested.entry(actor).or_insert(0.0);
                    *cur = (*cur - amt).max(0.0);
                } else if l.contains("raises ") && l.contains(" to ") {
                    // `raises $2 to $3` sets the player's street investment to `to $3`,
                    // while preserving any pre-posted ante (`posts the ante`) that is not part of the blind/bet level.
                    let prev_ante = preflop_antes.get(&actor).copied().unwrap_or(0.0);
                    preflop_invested.insert(actor.clone(), prev_ante + amt);
                    preflop_raises.push(actor);
                } else {
                    *preflop_invested.entry(actor.clone()).or_insert(0.0) += amt;
                    if is_blind_or_ante && l.to_ascii_lowercase().contains("ante") {
                        *preflop_antes.entry(actor.clone()).or_insert(0.0) += amt;
                    }
                    if sec == Section::Preflop && l.contains(": calls ") {
                        preflop_calls.push(actor);
                    }
                }
                continue;
            }
            if sec == Section::Preflop && l.contains(": folds") {
                if let Some(actor) = match_known_player_prefix(l, &seats) {
                    folded_preflop.insert(actor);
                }
            }
        } else if matches!(sec, Section::Flop | Section::Turn | Section::River) {
            if let Some(actor) = match_known_player_prefix(l, &seats) {
                if sec == Section::Flop && !postflop_actors_order.contains(&actor) {
                    postflop_actors_order.push(actor.clone());
                }
                let street_tag = match sec {
                    Section::Flop => "Flop",
                    Section::Turn => "Turn",
                    Section::River => "River",
                    _ => "",
                };
                if let Some(after_actor) = l.strip_prefix(&actor) {
                    if let Some(act_raw) = after_actor.trim_start().strip_prefix(':') {
                        let act_desc = act_raw.trim();
                        if act_desc.starts_with("checks") {
                            postflop_notes.push(format!("{street_tag}: {actor} {act_desc}"));
                            raw_postflop_acts.push((
                                street_tag.to_string(),
                                actor,
                                "check".to_string(),
                                0.0,
                            ));
                        } else if act_desc.starts_with("folds") {
                            postflop_notes.push(format!("{street_tag}: {actor} {act_desc}"));
                            raw_postflop_acts.push((
                                street_tag.to_string(),
                                actor,
                                "fold".to_string(),
                                0.0,
                            ));
                        } else if act_desc.starts_with("calls") {
                            let amt = parse_money_before_all_in(act_desc).unwrap_or(0.0);
                            postflop_notes.push(format!("{street_tag}: {actor} {act_desc}"));
                            raw_postflop_acts.push((
                                street_tag.to_string(),
                                actor,
                                "call".to_string(),
                                amt,
                            ));
                        } else if act_desc.starts_with("bets") {
                            let amt = parse_money_before_all_in(act_desc).unwrap_or(0.0);
                            postflop_notes.push(format!("{street_tag}: {actor} {act_desc}"));
                            raw_postflop_acts.push((
                                street_tag.to_string(),
                                actor,
                                "bet".to_string(),
                                amt,
                            ));
                        } else if act_desc.starts_with("raises") {
                            let amt = if let Some(ti) = act_desc.find(" to ") {
                                parse_first_number(&act_desc[ti + 4..]).unwrap_or(0.0)
                            } else {
                                parse_money_before_all_in(act_desc).unwrap_or(0.0)
                            };
                            postflop_notes.push(format!("{street_tag}: {actor} {act_desc}"));
                            raw_postflop_acts.push((
                                street_tag.to_string(),
                                actor,
                                "raise".to_string(),
                                amt,
                            ));
                        }
                    }
                }
            }
        }
    }

    if flop_cards.len() != 3 {
        return Err("Hand did not reach Flop (or '*** FLOP *** [3 cards]' missing)".to_string());
    }

    let flop_board = Board::from_slice(&flop_cards)
        .map_err(|e| format!("Invalid Flop board in hand history: {e}"))?;
    let mut all_board_cards = flop_cards.clone();
    if let Some(tc) = turn_card {
        all_board_cards.push(tc);
        if let Some(rc) = river_card {
            all_board_cards.push(rc);
        }
    }
    let final_board = Board::from_slice(&all_board_cards)
        .map_err(|e| format!("Invalid final board in hand history: {e}"))?;

    // Determine active postflop players (`!folded_preflop` who invested > 0 or acted on flop)
    let active_seats: Vec<&SeatEntry> = seats
        .iter()
        .filter(|s| {
            !folded_preflop.contains(&s.name)
                && (preflop_invested.get(&s.name).copied().unwrap_or(0.0) > 0.0
                    || s.name == hero_name
                    || postflop_actors_order.contains(&s.name))
        })
        .collect();

    let hero_entry = seats.iter().find(|s| s.name == hero_name);
    let villain_entry = active_seats
        .iter()
        .copied()
        .find(|s| s.name != hero_name)
        .or_else(|| seats.iter().find(|s| s.name != hero_name));

    let hero_table_pos = hero_entry
        .map(|s| s.pos_label.clone())
        .unwrap_or_else(|| "IP".to_string());
    let villain_table_pos = villain_entry
        .map(|s| s.pos_label.clone())
        .unwrap_or_else(|| "OOP".to_string());

    // Determine Hero OOP vs IP:
    // Priority 1: actual flop action order (`postflop_actors_order[0]` is OOP!)
    // Priority 2: canonical postflop positional order (`SB < BB < UTG < MP < CO < BTN`)
    let hero_seat = if postflop_actors_order.len() >= 2 {
        if postflop_actors_order[0] == hero_name {
            "OOP".to_string()
        } else {
            "IP".to_string()
        }
    } else {
        let h_ord = postflop_order_rank(&hero_table_pos);
        let v_ord = postflop_order_rank(&villain_table_pos);
        if h_ord <= v_ord {
            "OOP".to_string()
        } else {
            "IP".to_string()
        }
    };

    // Infer Villain's canonical preflop spot ID (`SpotCatalog` 15 spots!)
    let suggested_spot_id = infer_villain_spot_id(
        &hero_name,
        &hero_seat,
        &hero_table_pos,
        &villain_table_pos,
        &preflop_raises,
        &preflop_calls,
        active_seats.len(),
    );

    // Compute Flop Starting Pot & Remaining Effective Stack
    let raw_pot: f64 = preflop_invested.values().sum();
    let fallback_pot = if raw_pot > 0.0 { raw_pot } else { 2.5 * bb_amount };

    let rem_stacks: Vec<f64> = active_seats
        .iter()
        .map(|s| {
            let inv = preflop_invested.get(&s.name).copied().unwrap_or(0.0);
            (s.stack - inv).max(0.0)
        })
        .filter(|&rem| rem > 0.0)
        .collect();
    let raw_eff_stack = if rem_stacks.len() >= 2 {
        rem_stacks.into_iter().fold(f64::INFINITY, f64::min)
    } else if let Some(he) = hero_entry {
        let inv = preflop_invested.get(&he.name).copied().unwrap_or(0.0);
        (he.stack - inv).max(bb_amount * 10.0)
    } else {
        100.0 * bb_amount
    };

    // Scale to integer chips (`SpotConfig` uses integer `starting_pot` & `effective_stack`):
    // If fractional dollars/cents are present or `fallback_pot < 20`, normalize `1 BB = 2 chips` (or `10 chips` for micro fractions)
    // so pot odds & SPR are preserved to `< 0.5%` rounding error!
    let has_decimals = (fallback_pot.fract().abs() > 1e-4) || (raw_eff_stack.fract().abs() > 1e-4);
    let (starting_pot, effective_stack, mult, chips_per_bb, scale_note) =
        if has_decimals || fallback_pot < 15.0 {
            // Normalize in `0.1 BB` integer units (`1 BB = 10 chips` gives 0.1bb precision!)
            let m = 10.0 / bb_amount;
            let p = (fallback_pot * m).round().max(2.0) as i32;
            let s = (raw_eff_stack * m).round().max(1.0) as i32;
            (
                p,
                s,
                m,
                10.0,
                Some(format!(
                    "Scaled stakes (BB={bb_amount}) to integer units (1 BB = 10 chips): Pot={p}, EffStack={s}"
                )),
            )
        } else {
            (
                fallback_pot.round().max(2.0) as i32,
                raw_eff_stack.round().max(1.0) as i32,
                1.0,
                bb_amount,
                None,
            )
        };

    let postflop_actions: Vec<PostflopActionRecord> = raw_postflop_acts
        .into_iter()
        .map(|(st, actor, kind, raw_amt)| {
            let scaled = (raw_amt * mult).round().max(0.0) as i32;
            let raw_desc = if scaled > 0 {
                format!("{kind} {scaled}")
            } else {
                kind.clone()
            };
            PostflopActionRecord {
                street: st,
                actor_is_hero: actor == hero_name,
                action_kind: kind,
                amount_chips: scaled,
                raw_desc,
            }
        })
        .collect();

    Ok(ParsedHandHistory {
        room,
        hero_name,
        hero_hand,
        flop_board,
        turn_card,
        river_card,
        final_board,
        hero_seat,
        hero_table_pos,
        villain_table_pos,
        suggested_spot_id,
        starting_pot,
        effective_stack,
        chips_per_bb,
        scale_note,
        postflop_line_notes: postflop_notes,
        postflop_actions,
    })
}

fn postflop_order_rank(pos: &str) -> u8 {
    match pos {
        "SB" => 0,
        "BB" => 1,
        "UTG" => 2,
        "MP" => 3,
        "CO" => 4,
        "BTN" => 5,
        _ => 3,
    }
}

fn infer_villain_spot_id(
    hero_name: &str,
    hero_seat: &str,
    _hero_pos: &str,
    villain_pos: &str,
    raises: &[String],
    _calls: &[String],
    num_postflop_players: usize,
) -> String {
    let n_raises = raises.len();
    let last_aggressor = raises.last().map(|s| s.as_str()).unwrap_or("");
    let villain_was_last_aggressor = !last_aggressor.is_empty() && last_aggressor != hero_name;

    if n_raises >= 2 {
        // 3-bet+ pot!
        if villain_was_last_aggressor {
            // Villain 3-bet!
            if hero_seat == "OOP" {
                // Hero is OOP -> Villain 3-bet IP
                return match villain_pos {
                    "BTN" => "THREEBET_IP_VS_25_35".to_string(),
                    _ => "THREEBET_IP_VS_15_20".to_string(),
                };
            } else {
                // Hero is IP -> Villain 3-bet OOP (from SB or BB)
                return match villain_pos {
                    "SB" => "THREEBET_OOP_SB_VS_30".to_string(),
                    _ => "THREEBET_OOP_BB_VS_30".to_string(),
                };
            }
        } else {
            // Hero 3-bet and Villain called preflop (use Villain's open range as baseline)
            return match villain_pos {
                "UTG" => "RFI_UTG".to_string(),
                "MP" => "RFI_MP".to_string(),
                "CO" => "RFI_CO".to_string(),
                _ => "RFI_BTN".to_string(),
            };
        }
    }

    // Single-raised pot (SRP) or limped pot
    if villain_was_last_aggressor {
        // Villain opened RFI, Hero defended!
        match villain_pos {
            "UTG" => "RFI_UTG".to_string(),
            "MP" => "RFI_MP".to_string(),
            "CO" => "RFI_CO".to_string(),
            _ => "RFI_BTN".to_string(),
        }
    } else {
        // Hero opened RFI, Villain called!
        if villain_pos == "BB" {
            if num_postflop_players >= 3 {
                "BB_DEF_MULTIWAY".to_string()
            } else {
                "BB_DEF_VS_1_RAISER".to_string()
            }
        } else if villain_pos == "BTN" {
            "CC_IP_BTN_VS_30".to_string()
        } else {
            "CC_IP_MP_CO_VS_15".to_string()
        }
    }
}

fn detect_big_blind(header: &str) -> Option<f64> {
    // Look for `X/Y` inside `(...)` or anywhere in header
    let mut best: Option<f64> = None;
    for tok in header.split(|c: char| c.is_whitespace() || c == '(' || c == ')') {
        if let Some(slash) = tok.find('/') {
            let rhs = &tok[slash + 1..];
            if let Some(v) = parse_first_number(rhs) {
                if v > 0.0 {
                    best = Some(v);
                    break;
                }
            }
        }
    }
    best
}

fn extract_seat_hash_number(line: &str) -> Option<usize> {
    let idx = line.find("Seat #")?;
    let rest = &line[idx + 6..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<usize>().ok()
}

fn parse_seat_line(line: &str) -> Option<SeatEntry> {
    let rest = line.strip_prefix("Seat ")?;
    let colon = rest.find(':')?;
    let seat_no = rest[..colon].trim().parse::<usize>().ok()?;
    let after_colon = rest[colon + 1..].trim();
    // Find last `(` containing stack (`($100.00 in chips)` or `(10000)`)
    let open_paren = after_colon.rfind('(')?;
    let name = after_colon[..open_paren].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let inside = &after_colon[open_paren + 1..];
    let stack = parse_first_number(inside)?;
    Some(SeatEntry {
        seat_no,
        name,
        stack,
        pos_label: "MP".to_string(),
    })
}

fn match_known_player_prefix(line: &str, seats: &[SeatEntry]) -> Option<String> {
    for s in seats {
        if let Some(after) = line.strip_prefix(&s.name) {
            if after.starts_with(':') || after.starts_with(' ') {
                return Some(s.name.clone());
            }
        }
    }
    if let Some(colon) = line.find(':') {
        let cand = line[..colon].trim();
        if !cand.is_empty() && !cand.starts_with("Seat ") && !cand.starts_with("***") {
            return Some(cand.to_string());
        }
    }
    None
}

fn parse_money_contribution(line: &str, seats: &[SeatEntry]) -> Option<(String, f64, bool, bool)> {
    // Handle `Uncalled bet ($2.00) returned to PlayerName`
    if line.starts_with("Uncalled bet") {
        if let Some(ret_idx) = line.find("returned to ") {
            let player = line[ret_idx + "returned to ".len()..].trim().to_string();
            let amt = parse_first_number(line)?;
            return Some((player, amt, true, false));
        }
    }

    let actor = match_known_player_prefix(line, seats)?;
    // Use `actor.len()` (or the colon right after `actor`) so player names containing `:` work cleanly!
    let after_actor = line.strip_prefix(&actor)?.trim_start();
    let act = after_actor.strip_prefix(':')?.trim();

    if act.starts_with("posts ") {
        let amt = parse_money_before_all_in(act)?;
        return Some((actor, amt, false, true));
    }
    if act.starts_with("calls ") || act.starts_with("bets ") {
        let amt = parse_money_before_all_in(act)?;
        return Some((actor, amt, false, false));
    }
    if act.starts_with("raises ") {
        if let Some(to_idx) = act.find(" to ") {
            let amt = parse_first_number(&act[to_idx + 4..])?;
            return Some((actor, amt, false, false));
        }
    }
    None
}

fn parse_money_before_all_in(act: &str) -> Option<f64> {
    let trimmed = if let Some(ai) = act.find(" and is all-in") {
        &act[..ai]
    } else {
        act
    };
    parse_last_number(trimmed)
}

fn parse_first_number(s: &str) -> Option<f64> {
    let mut buf = String::new();
    let mut started = false;
    for ch in s.chars() {
        if ch.is_ascii_digit() || (started && ch == '.') {
            buf.push(ch);
            started = true;
        } else if ch == ',' && started {
            continue;
        } else if started {
            break;
        }
    }
    buf.parse::<f64>().ok()
}

fn parse_last_number(s: &str) -> Option<f64> {
    let mut last = None;
    for tok in s.split_whitespace() {
        if let Some(v) = parse_first_number(tok) {
            last = Some(v);
        }
    }
    last
}

fn parse_card_tokens(inside: &str) -> Result<Vec<Card>, String> {
    let mut out = Vec::new();
    for tok in inside
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
    {
        let card: Card = tok
            .parse()
            .map_err(|e| format!("Invalid card token {tok:?}: {e}"))?;
        out.push(card);
    }
    Ok(out)
}

fn extract_all_bracket_cards(line: &str) -> Result<Vec<Card>, String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        if let Some(close) = after.find(']') {
            let inside = &after[..close];
            out.extend(parse_card_tokens(inside)?);
            rest = &after[close + 1..];
        } else {
            break;
        }
    }
    Ok(out)
}
