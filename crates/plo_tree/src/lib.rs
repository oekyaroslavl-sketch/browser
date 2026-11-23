//! Pot-Limit Omaha (`PLO-4`) postflop betting & action tree builder.
//!
//! # Pot-Limit Betting Invariants (Default Mode: `BettingLimit::PotLimit`)
//! Let `pot_street_start` be the pot at the start of the current street, and `invested[p]` be the
//! chips committed by player `p` on the current street (`0` at street start):
//! 1. **Unopened pot (`invested == [0, 0]`)**:
//!    - `Check` is always legal.
//!    - Maximum Pot-Limit bet is `min(pot_street_start, remaining_stack)`.
//!    - Any configured pot fraction `f` is capped at `1.0` (`100%` pot) in `PotLimit` mode!
//!      Overbets (`> 100%` pot) are illegal in Pot-Limit Omaha and automatically clamped to `Pot`.
//! 2. **Facing a bet/raise (`facing = invested[opp] - invested[p] > 0`)**:
//!    - `Fold` and `Call` are always legal.
//!    - `pot_after_call = pot_street_start + 2 * invested[opp]`.
//!    - Maximum Pot-Limit `raise_to` (total street commitment) is:
//!      `min(invested[opp] + pot_after_call, total_street_budget_of_effective_stack)`.
//!      For example, if starting pot is `100` and OOP pots `100`, `pot_after_call = 300`, so IP's
//!      maximum pot raise is `100 + 300 = 400` total on the street (never an illegal 900 NL shove!).
//!    - Minimum legal non-all-in `raise_to` is `invested[opp] + last_raise_increment`.
//!    - All-in occurs **only** when a legal bet or raise reaches the effective stack (`invested[p] + stack[p]`),
//!      or when `effective_stack_street_max <= max_plo_raise_to` and within `add_allin_spr_threshold`.
//!
//! An optional `BettingLimit::NoLimitApprox` variant is documented explicitly as a non-PLO approximation
//! if a user intentionally wants NL-style overbet/all-in abstractions.

use core::fmt;
use plo_cards::Street;

/// Player seat index (`OOP = 0` acts first on every postflop street; `IP = 1` acts last).
pub const PLAYER_OOP: usize = 0;
pub const PLAYER_IP: usize = 1;

/// Betting limit rule enforced when constructing [`ActionTree`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BettingLimit {
    /// Exact Pot-Limit Omaha rules: bets and raises cannot exceed 100% of the pot after calling.
    #[default]
    PotLimit,
    /// Optional No-Limit approximation (permits overbets `> 100%` pot and unconditional all-ins).
    NoLimitApprox,
}

/// Abstract betting action on a street.
///
/// Amounts inside `Bet(to)`, `Raise(to)`, and `AllIn(to)` follow `wasm-postflop` convention:
/// `to` is the **total chips committed by the acting player on the current street**.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    Fold,
    Check,
    Call,
    /// First bet on the current street (`to` chips committed on this street, `< effective_street_max`).
    Bet(i32),
    /// Raise on the current street to `to` total chips on this street (`< effective_street_max`).
    Raise(i32),
    /// All-in commitment of `to` total chips on this street (`== effective_street_max`).
    AllIn(i32),
}

impl Action {
    /// Encodes action in `wasm-postflop` string format (`"Fold:0"`, `"Check:0"`, `"Bet:100"`, etc.).
    pub fn to_solver_label(self) -> String {
        match self {
            Action::Fold => "Fold:0".to_string(),
            Action::Check => "Check:0".to_string(),
            Action::Call => "Call:0".to_string(),
            Action::Bet(amt) => format!("Bet:{amt}"),
            Action::Raise(amt) => format!("Raise:{amt}"),
            Action::AllIn(amt) => format!("Allin:{amt}"),
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Fold => write!(f, "Fold"),
            Action::Check => write!(f, "Check"),
            Action::Call => write!(f, "Call"),
            Action::Bet(x) => write!(f, "Bet({x})"),
            Action::Raise(x) => write!(f, "Raise({x})"),
            Action::AllIn(x) => write!(f, "AllIn({x})"),
        }
    }
}

/// Bet and raise size configuration for one player on one street.
///
/// Fractions are expressed relative to the pot (`0.5` = 50% pot, `1.0` = 100% pot).
/// In `BettingLimit::PotLimit`, any fraction `> 1.0` is clamped to `1.0` (pot).
#[derive(Clone, Debug, PartialEq)]
pub struct StreetBetSizes {
    /// Pot fractions for opening bets (e.g. `vec![0.5, 1.0]`).
    pub bet_fractions: Vec<f64>,
    /// Pot fractions for raises (fraction of `pot_after_call`, e.g. `vec![1.0]` for pot-raise).
    pub raise_fractions: Vec<f64>,
    /// Optional donk-bet pot fractions when OOP called the previous street and acts first on the new street.
    /// `None` uses `bet_fractions`.
    pub donk_fractions: Option<Vec<f64>>,
}

impl Default for StreetBetSizes {
    /// Standard PLO default: 50% pot & 100% pot bets, 100% pot raises.
    fn default() -> Self {
        Self {
            bet_fractions: vec![0.5, 1.0],
            raise_fractions: vec![1.0],
            donk_fractions: None,
        }
    }
}

impl StreetBetSizes {
    /// Parses comma-separated percentage strings like `("33%, 75%, 100%", "50%, 100%")` or `"pot"`.
    pub fn from_strs(bet_str: &str, raise_str: &str) -> Result<Self, TreeError> {
        Ok(Self {
            bet_fractions: parse_fraction_list(bet_str)?,
            raise_fractions: parse_fraction_list(raise_str)?,
            donk_fractions: None,
        })
    }

    pub fn pot_only() -> Self {
        Self {
            bet_fractions: vec![1.0],
            raise_fractions: vec![1.0],
            donk_fractions: None,
        }
    }
}

/// Parses comma-separated pot percentages (`"33%, 50%, 75%, 100%, pot, a"`).
pub fn parse_fraction_list(s: &str) -> Result<Vec<f64>, TreeError> {
    let mut out = Vec::new();
    for raw in s.split(',') {
        let tok = raw.trim().to_ascii_lowercase();
        if tok.is_empty() {
            continue;
        }
        if tok == "pot" || tok == "p" {
            out.push(1.0);
        } else if tok == "a" || tok == "allin" {
            // Represent all-in request as f64::INFINITY; PotLimit mode still caps each single bet/raise at 100% pot!
            out.push(f64::INFINITY);
        } else if let Some(num_s) = tok.strip_suffix('%') {
            let pct: f64 = num_s
                .trim()
                .parse()
                .map_err(|_| TreeError::InvalidBetSize(raw.to_string()))?;
            if !pct.is_finite() || pct <= 0.0 {
                return Err(TreeError::InvalidBetSize(raw.to_string()));
            }
            out.push(pct / 100.0);
        } else {
            let frac: f64 = tok
                .parse()
                .map_err(|_| TreeError::InvalidBetSize(raw.to_string()))?;
            if !frac.is_finite() || frac <= 0.0 {
                return Err(TreeError::InvalidBetSize(raw.to_string()));
            }
            out.push(frac);
        }
    }
    Ok(out)
}

/// Complete configuration for building a postflop [`ActionTree`].
#[derive(Clone, Debug)]
pub struct TreeConfig {
    /// Starting street (`Flop`, `Turn`, or `River`).
    pub initial_street: Street,
    /// Chips already in the pot at the start of `initial_street` (`> 0`).
    pub starting_pot: i32,
    /// Effective remaining stack of each player at the start of `initial_street` (`> 0`).
    pub effective_stack: i32,
    /// Betting limit rule (`PotLimit` by default!).
    pub betting_limit: BettingLimit,
    /// Minimum bet unit / big blind (`>= 1`, default `1`).
    pub min_bet_unit: i32,
    /// Maximum number of bets + raises allowed per street (`0` = unlimited until all-in; default `4`).
    pub max_bets_per_street: u8,
    /// Per-player `[OOP, IP]` bet sizes on the flop.
    pub flop_sizes: [StreetBetSizes; 2],
    /// Per-player `[OOP, IP]` bet sizes on the turn.
    pub turn_sizes: [StreetBetSizes; 2],
    /// Per-player `[OOP, IP]` bet sizes on the river.
    pub river_sizes: [StreetBetSizes; 2],
    /// Optional per-street betting lock `[flop, turn, river]` (default `[true, true, true]`).
    /// When `false` for a street, betting is skipped on that street (forced check-down to the
    /// next street / exact 2+3 river showdown), enabling single-street or subset solving
    /// ("Flop only", "Turn only", "River only", or "All streets together").
    pub active_betting_streets: [bool; 3],
    /// Snap a bet/raise to All-In if the remaining stack after calling that bet would be `<= threshold * pot_after_call`.
    /// Default `0.15` (standard solver anti-micro-stub threshold; never exceeds legal PLO max!).
    pub force_allin_spr_threshold: f64,
    /// Add a legal All-In option when the effective stack is already `<= max_legal_bet_or_raise`
    /// (i.e. stack is short enough that going all-in IS a legal Pot-Limit bet/raise) AND
    /// `effective_stack_remaining <= add_allin_spr_threshold * pot_after_call`.
    pub add_allin_spr_threshold: f64,
    /// Merge nearby bet sizes `(pot + b1) / (pot + b2) < 1.0 + merging_threshold` (default `0.08`).
    pub merging_threshold: f64,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 100,
            betting_limit: BettingLimit::PotLimit,
            min_bet_unit: 1,
            max_bets_per_street: 4,
            flop_sizes: [StreetBetSizes::default(), StreetBetSizes::default()],
            turn_sizes: [StreetBetSizes::default(), StreetBetSizes::default()],
            river_sizes: [StreetBetSizes::default(), StreetBetSizes::default()],
            active_betting_streets: [true, true, true],
            force_allin_spr_threshold: 0.15,
            add_allin_spr_threshold: 1.0,
            merging_threshold: 0.08,
        }
    }
}

/// Errors produced when validating [`TreeConfig`] or traversing [`ActionTree`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeError {
    InvalidPot(i32),
    InvalidStack(i32),
    InvalidMinBet(i32),
    InvalidBetSize(String),
    InvalidAction(Action),
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::InvalidPot(p) => write!(f, "starting_pot must be > 0, got {p}"),
            TreeError::InvalidStack(s) => write!(f, "effective_stack must be > 0, got {s}"),
            TreeError::InvalidMinBet(m) => write!(f, "min_bet_unit must be >= 1, got {m}"),
            TreeError::InvalidBetSize(s) => write!(f, "invalid bet size specification: {s:?}"),
            TreeError::InvalidAction(a) => write!(f, "action {a} not available at node"),
        }
    }
}

impl std::error::Error for TreeError {}

/// Kind of node in the abstract [`ActionTree`].
#[derive(Clone, Debug)]
pub enum ActionNodeKind {
    /// Acting player (`0 = OOP`, `1 = IP`) chooses among `actions`, transitioning to `children`.
    Player {
        player: usize,
        street: Street,
        actions: Vec<Action>,
        children: Vec<usize>,
    },
    /// Nature deals the next community card (`next_street` is `Turn` or `River`) and transitions to `child`.
    Chance {
        next_street: Street,
        child: usize,
    },
    /// Player `folded_player` folded; opponent wins `pot` (`total_committed` per player before fold).
    TerminalFold {
        folded_player: usize,
        street: Street,
    },
    /// Both players reached showdown (either after river check-check/call or after all-in runout).
    TerminalShowdown,
}

/// Single node in the flat arena of [`ActionTree`].
#[derive(Clone, Debug)]
pub struct ActionNode {
    pub id: usize,
    pub kind: ActionNodeKind,
    /// Total chips committed by `[OOP, IP]` across all postflop streets up to this node
    /// (excluding `starting_pot`).
    pub committed: [i32; 2],
    /// Pot size at this node (`starting_pot + committed[0] + committed[1]`).
    pub pot: i32,
}

/// Abstract postflop betting tree (street-level skeleton where each `Chance` node represents
/// dealing any valid turn/river card; concrete card branches are expanded by `plo_cfr`).
#[derive(Clone, Debug)]
pub struct ActionTree {
    config: TreeConfig,
    nodes: Vec<ActionNode>,
}

#[derive(Clone, Copy)]
struct StreetState {
    street: Street,
    /// Chips committed by each player prior to the start of `street`.
    base_committed: [i32; 2],
    /// Chips committed by each player ON `street`.
    street_invested: [i32; 2],
    /// Size of the most recent bet/raise increment on `street`.
    last_raise_inc: i32,
    /// Number of bets + raises made so far on `street`.
    num_bets: u8,
    /// Whether this node is the first action of `street` (before any check or bet).
    street_unacted: bool,
    /// Whether OOP ended the previous street by calling IP's bet/raise (enables `donk_fractions`).
    oop_called_prev_street: bool,
    /// Whether players are all-in (`committed == [effective_stack, effective_stack]`).
    is_allin: bool,
}

impl ActionTree {
    /// Validates `config` and constructs the complete postflop [`ActionTree`].
    pub fn build(config: TreeConfig) -> Result<Self, TreeError> {
        if config.starting_pot <= 0 {
            return Err(TreeError::InvalidPot(config.starting_pot));
        }
        if config.effective_stack <= 0 {
            return Err(TreeError::InvalidStack(config.effective_stack));
        }
        if config.min_bet_unit < 1 {
            return Err(TreeError::InvalidMinBet(config.min_bet_unit));
        }

        let mut tree = Self {
            config,
            nodes: Vec::new(),
        };

        let init = StreetState {
            street: tree.config.initial_street,
            base_committed: [0, 0],
            street_invested: [0, 0],
            last_raise_inc: 0,
            num_bets: 0,
            street_unacted: true,
            oop_called_prev_street: false,
            is_allin: false,
        };
        if tree.is_street_betting_enabled(init.street) {
            tree.build_player_or_next(PLAYER_OOP, init);
        } else {
            tree.advance_street_or_showdown(init);
        }
        Ok(tree)
    }

    #[inline]
    fn is_street_betting_enabled(&self, street: Street) -> bool {
        let idx = match street {
            Street::Flop => 0,
            Street::Turn => 1,
            Street::River => 2,
        };
        self.config.active_betting_streets[idx]
    }

    #[inline]
    pub fn config(&self) -> &TreeConfig {
        &self.config
    }

    #[inline]
    pub fn nodes(&self) -> &[ActionNode] {
        &self.nodes
    }

    #[inline]
    pub fn root_id(&self) -> usize {
        0
    }

    #[inline]
    pub fn node(&self, id: usize) -> &ActionNode {
        &self.nodes[id]
    }

    /// Counts `(player_decision_nodes_per_street [flop, turn, river], chance_nodes, terminal_nodes)`.
    pub fn node_counts(&self) -> ([usize; 3], usize, usize) {
        let mut dec = [0usize; 3];
        let mut chance = 0usize;
        let mut term = 0usize;
        for n in &self.nodes {
            match n.kind {
                ActionNodeKind::Player { street, .. } => match street {
                    Street::Flop => dec[0] += 1,
                    Street::Turn => dec[1] += 1,
                    Street::River => dec[2] += 1,
                },
                ActionNodeKind::Chance { .. } => chance += 1,
                ActionNodeKind::TerminalFold { .. } | ActionNodeKind::TerminalShowdown => term += 1,
            }
        }
        (dec, chance, term)
    }

    fn alloc_node(&mut self, kind: ActionNodeKind, committed: [i32; 2]) -> usize {
        let id = self.nodes.len();
        let pot = self.config.starting_pot + committed[0] + committed[1];
        self.nodes.push(ActionNode {
            id,
            kind,
            committed,
            pot,
        });
        id
    }

    fn advance_street_or_showdown(&mut self, state: StreetState) -> usize {
        let total_committed = [
            state.base_committed[0] + state.street_invested[0],
            state.base_committed[1] + state.street_invested[1],
        ];
        debug_assert_eq!(total_committed[0], total_committed[1]);

        match state.street.next() {
            None => self.alloc_node(ActionNodeKind::TerminalShowdown, total_committed),
            Some(next_street) => {
                let chance_id = self.alloc_node(
                    ActionNodeKind::Chance {
                        next_street,
                        child: usize::MAX,
                    },
                    total_committed,
                );
                let next_state = StreetState {
                    street: next_street,
                    base_committed: total_committed,
                    street_invested: [0, 0],
                    last_raise_inc: 0,
                    num_bets: 0,
                    street_unacted: true,
                    oop_called_prev_street: state.oop_called_prev_street,
                    is_allin: state.is_allin
                        || total_committed[0] >= self.config.effective_stack,
                };
                let child_id = if next_state.is_allin || !self.is_street_betting_enabled(next_street) {
                    // Players are already all-in OR betting is disabled on `next_street`;
                    // run out / check down without allocating decision nodes on `next_street`!
                    self.advance_street_or_showdown(next_state)
                } else {
                    self.build_player_or_next(PLAYER_OOP, next_state)
                };
                if let ActionNodeKind::Chance { child, .. } = &mut self.nodes[chance_id].kind {
                    *child = child_id;
                }
                chance_id
            }
        }
    }

    fn build_player_or_next(&mut self, player: usize, state: StreetState) -> usize {
        let opp = player ^ 1;
        let committed = [
            state.base_committed[0] + state.street_invested[0],
            state.base_committed[1] + state.street_invested[1],
        ];
        let street_max = self.config.effective_stack - state.base_committed[player];
        debug_assert!(street_max > state.street_invested[player]);

        let actions = self.generate_legal_actions(player, &state, street_max);

        let node_id = self.alloc_node(
            ActionNodeKind::Player {
                player,
                street: state.street,
                actions: actions.clone(),
                children: vec![usize::MAX; actions.len()],
            },
            committed,
        );

        let mut children = Vec::with_capacity(actions.len());
        for &act in &actions {
            let child_id = match act {
                Action::Fold => {
                    self.alloc_node(
                        ActionNodeKind::TerminalFold {
                            folded_player: player,
                            street: state.street,
                        },
                        committed,
                    )
                }
                Action::Check => {
                    if state.street_unacted {
                        // OOP checked to open the street -> IP acts on same street
                        let mut next = state;
                        next.street_unacted = false;
                        next.oop_called_prev_street = false;
                        self.build_player_or_next(opp, next)
                    } else {
                        // Check-back closes the street!
                        let mut next = state;
                        next.oop_called_prev_street = false;
                        self.advance_street_or_showdown(next)
                    }
                }
                Action::Call => {
                    let mut next = state;
                    next.street_invested[player] = state.street_invested[opp];
                    next.oop_called_prev_street = player == PLAYER_OOP;
                    next.is_allin =
                        next.base_committed[0] + next.street_invested[0] >= self.config.effective_stack;
                    self.advance_street_or_showdown(next)
                }
                Action::Bet(to) | Action::Raise(to) | Action::AllIn(to) => {
                    let mut next = state;
                    let inc = to - state.street_invested[opp];
                    next.street_invested[player] = to;
                    next.last_raise_inc = inc.max(self.config.min_bet_unit);
                    next.num_bets = state.num_bets.saturating_add(1);
                    next.street_unacted = false;
                    next.oop_called_prev_street = false;
                    next.is_allin = matches!(act, Action::AllIn(_)) || to >= street_max;
                    self.build_player_or_next(opp, next)
                }
            };
            children.push(child_id);
        }

        if let ActionNodeKind::Player {
            children: slot, ..
        } = &mut self.nodes[node_id].kind
        {
            *slot = children;
        }

        node_id
    }

    fn generate_legal_actions(
        &self,
        player: usize,
        state: &StreetState,
        street_max: i32,
    ) -> Vec<Action> {
        let opp = player ^ 1;
        let my_inv = state.street_invested[player];
        let opp_inv = state.street_invested[opp];
        let facing = opp_inv - my_inv;

        let pot_street_start =
            self.config.starting_pot + state.base_committed[0] + state.base_committed[1];

        let sizes = match state.street {
            Street::Flop => &self.config.flop_sizes[player],
            Street::Turn => &self.config.turn_sizes[player],
            Street::River => &self.config.river_sizes[player],
        };

        let mut raw_bets_to: Vec<i32> = Vec::new();
        let mut actions = Vec::new();

        if facing == 0 {
            debug_assert_eq!(my_inv, 0);
            debug_assert_eq!(opp_inv, 0);
            actions.push(Action::Check);

            if self.config.max_bets_per_street == 0 || state.num_bets < self.config.max_bets_per_street {
                let min_bet = self.config.min_bet_unit.min(street_max);
                // POT-LIMIT INVARIANT: Max opening bet in PLO is 100% of current pot (capped by stack)!
                let max_legal_to = match self.config.betting_limit {
                    BettingLimit::PotLimit => pot_street_start.min(street_max),
                    BettingLimit::NoLimitApprox => street_max,
                };

                let fracs = if player == PLAYER_OOP && state.oop_called_prev_street {
                    sizes.donk_fractions.as_ref().unwrap_or(&sizes.bet_fractions)
                } else {
                    &sizes.bet_fractions
                };

                for &f in fracs {
                    let eff_f = match self.config.betting_limit {
                        BettingLimit::PotLimit => f.min(1.0),
                        BettingLimit::NoLimitApprox => f,
                    };
                    let target = if !eff_f.is_finite() {
                        max_legal_to
                    } else {
                        ((pot_street_start as f64) * eff_f).round() as i32
                    };
                    let clamped = target.clamp(min_bet, max_legal_to);
                    if clamped > 0 {
                        raw_bets_to.push(clamped);
                    }
                }

                // Add legal all-in if stack is short enough to be a legal bet (`street_max <= max_legal_to`)
                // and within `add_allin_spr_threshold`.
                if street_max <= max_legal_to
                    && (street_max as f64)
                        <= (pot_street_start as f64) * self.config.add_allin_spr_threshold
                {
                    raw_bets_to.push(street_max);
                }

                let merged = self.clean_and_merge_amounts(
                    raw_bets_to,
                    pot_street_start,
                    0,
                    street_max,
                    max_legal_to,
                );
                for amt in merged {
                    if amt >= street_max {
                        actions.push(Action::AllIn(street_max));
                    } else {
                        actions.push(Action::Bet(amt));
                    }
                }
            }
        } else {
            // Facing a bet or raise (`opp_inv > my_inv`)
            actions.push(Action::Fold);
            actions.push(Action::Call);

            let can_raise = !state.is_allin
                && street_max > opp_inv
                && (self.config.max_bets_per_street == 0
                    || state.num_bets < self.config.max_bets_per_street);

            if can_raise {
                let pot_after_call = pot_street_start + 2 * opp_inv;
                let min_raise_to =
                    (opp_inv + state.last_raise_inc.max(self.config.min_bet_unit)).min(street_max);

                // POT-LIMIT INVARIANT: Max raise-to on street is `opp_inv + pot_after_call` (capped by stack)!
                let max_legal_to = match self.config.betting_limit {
                    BettingLimit::PotLimit => (opp_inv.saturating_add(pot_after_call)).min(street_max),
                    BettingLimit::NoLimitApprox => street_max,
                };

                for &f in &sizes.raise_fractions {
                    let eff_f = match self.config.betting_limit {
                        BettingLimit::PotLimit => f.min(1.0),
                        BettingLimit::NoLimitApprox => f,
                    };
                    let target = if !eff_f.is_finite() {
                        max_legal_to
                    } else {
                        opp_inv + ((pot_after_call as f64) * eff_f).round() as i32
                    };
                    let clamped = target.clamp(min_raise_to, max_legal_to);
                    if clamped > opp_inv {
                        raw_bets_to.push(clamped);
                    }
                }

                // Add legal all-in if stack is reachable under Pot-Limit (`street_max <= max_legal_to`)
                if street_max <= max_legal_to
                    && ((street_max - opp_inv) as f64)
                        <= (pot_after_call as f64) * self.config.add_allin_spr_threshold
                {
                    raw_bets_to.push(street_max);
                }

                let merged = self.clean_and_merge_amounts(
                    raw_bets_to,
                    pot_after_call,
                    opp_inv,
                    street_max,
                    max_legal_to,
                );
                for amt in merged {
                    if amt >= street_max {
                        actions.push(Action::AllIn(street_max));
                    } else {
                        actions.push(Action::Raise(amt));
                    }
                }
            }
        }

        actions
    }

    fn clean_and_merge_amounts(
        &self,
        mut amounts: Vec<i32>,
        ref_pot: i32,
        call_level: i32,
        street_max: i32,
        max_legal_to: i32,
    ) -> Vec<i32> {
        if amounts.is_empty() {
            return amounts;
        }
        // Apply force-allin snapping ONLY if snapping to `street_max` is legal (`street_max <= max_legal_to`)!
        for amt in &mut amounts {
            if street_max <= max_legal_to && *amt < street_max {
                let pot_if_called = ref_pot + 2 * (*amt - call_level);
                let rem_stack = street_max - *amt;
                if pot_if_called > 0
                    && (rem_stack as f64)
                        <= (pot_if_called as f64) * self.config.force_allin_spr_threshold
                {
                    *amt = street_max;
                }
            }
        }

        amounts.sort_unstable();
        amounts.dedup();

        if self.config.merging_threshold <= 0.0 || amounts.len() <= 1 {
            return amounts;
        }

        // Merge from largest downward (preserving the highest/pot/all-in size)
        let mut kept = Vec::with_capacity(amounts.len());
        let mut prev_hi: Option<i32> = None;
        for &amt in amounts.iter().rev() {
            if let Some(hi) = prev_hi {
                let r_hi = (hi - call_level) as f64 / (ref_pot as f64);
                let r_lo = (amt - call_level) as f64 / (ref_pot as f64);
                if (1.0 + r_hi) / (1.0 + r_lo) < 1.0 + self.config.merging_threshold {
                    continue;
                }
            }
            kept.push(amt);
            prev_hi = Some(amt);
        }
        kept.reverse();
        kept
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pot_limit_exact_max_bet_and_pot_raise_sequence() {
        // Classic deep-stack PLO pot-pot-pot sequence:
        // Starting pot = 100, Effective stack = 1500.
        // Configured with 150% overbet and "a" (all-in) — PotLimit MUST cap at:
        // 1) OOP max open bet = 100 (Bet(100), NEVER AllIn(1500) or Bet(150)!)
        // 2) IP facing Bet(100): pot_after_call = 100 + 200 = 300 -> max raise = 100 + 300 = Raise(400)!
        // 3) OOP facing Raise(400): pot_after_call = 100 + 800 = 900 -> max re-raise = 400 + 900 = Raise(1300)!
        // 4) IP facing Raise(1300): stack is 1500 <= 1300 + 2700 = 4000 -> AllIn(1500)!
        let sizes = StreetBetSizes::from_strs("50%, 150%, a", "100%, a").unwrap();
        let cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 1500,
            betting_limit: BettingLimit::PotLimit,
            min_bet_unit: 1,
            max_bets_per_street: 5,
            river_sizes: [sizes.clone(), sizes],
            force_allin_spr_threshold: 0.0,
            add_allin_spr_threshold: 0.0,
            merging_threshold: 0.05,
            ..Default::default()
        };

        let tree = ActionTree::build(cfg).unwrap();
        // Root: OOP acts
        let root = tree.node(tree.root_id());
        let ActionNodeKind::Player {
            actions, children, ..
        } = &root.kind
        else {
            panic!("expected player root");
        };
        assert_eq!(actions, &[Action::Check, Action::Bet(50), Action::Bet(100)]);

        // Follow OOP Bet(100) -> IP node
        let ip_node = tree.node(children[2]);
        let ActionNodeKind::Player {
            actions: ip_acts,
            children: ip_children,
            ..
        } = &ip_node.kind
        else {
            panic!("expected IP node");
        };
        assert_eq!(
            ip_acts,
            &[Action::Fold, Action::Call, Action::Raise(400)]
        );

        // Follow IP Raise(400) -> OOP 3-bet node
        let oop_3b = tree.node(ip_children[2]);
        let ActionNodeKind::Player {
            actions: oop_3b_acts,
            children: oop_3b_children,
            ..
        } = &oop_3b.kind
        else {
            panic!("expected OOP 3b node");
        };
        assert_eq!(
            oop_3b_acts,
            &[Action::Fold, Action::Call, Action::Raise(1300)]
        );

        // Follow OOP Raise(1300) -> IP 4-bet node (capped at AllIn(1500))
        let ip_4b = tree.node(oop_3b_children[2]);
        let ActionNodeKind::Player {
            actions: ip_4b_acts,
            ..
        } = &ip_4b.kind
        else {
            panic!("expected IP 4b node");
        };
        assert_eq!(
            ip_4b_acts,
            &[Action::Fold, Action::Call, Action::AllIn(1500)]
        );
    }

    #[test]
    fn turn_and_flop_trees_include_chance_and_allin_runouts() {
        let cfg = TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 100, // 1 SPR: pot bet IS all-in!
            betting_limit: BettingLimit::PotLimit,
            turn_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let tree = ActionTree::build(cfg).unwrap();
        let (dec, chance, term) = tree.node_counts();
        assert_eq!(dec[0], 0); // No flop nodes
        assert!(dec[1] > 0); // Turn decision nodes
        assert!(dec[2] > 0); // River decision nodes
        assert!(chance >= 2); // Check-check -> chance AND Bet(AllIn)-Call -> chance -> showdown!
        assert!(term >= 3);
    }

    #[test]
    fn phase_d_pot_limit_edge_cases_and_invariants() {
        // D1 & D4: When `effective_stack < starting_pot` (e.g. pot = 100, stack = 37) and config requests `50%, 100%, 200%` overbet:
        // - Max legal open is `min(pot, stack) = min(100, 37) = 37`.
        // - ONLY `[Check, AllIn(37)]` may appear — NO phantom `Bet(50)` or `Bet(100)`!
        let short_cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 37,
            betting_limit: BettingLimit::PotLimit,
            river_sizes: [
                StreetBetSizes {
                    bet_fractions: vec![0.5, 1.0, 2.0],
                    raise_fractions: vec![0.5, 1.0, 2.0],
                    donk_fractions: None,
                },
                StreetBetSizes::pot_only(),
            ],
            ..Default::default()
        };
        let short_tree = ActionTree::build(short_cfg).unwrap();
        let ActionNodeKind::Player {
            actions: s_acts,
            children: s_children,
            ..
        } = &short_tree.node(short_tree.root_id()).kind
        else {
            panic!();
        };
        assert_eq!(s_acts, &[Action::Check, Action::AllIn(37)]);
        // Facing AllIn(37), IP has ONLY `[Fold, Call]` (no raise possible!)
        let ActionNodeKind::Player {
            actions: ip_vs_ai, ..
        } = &short_tree.node(s_children[1]).kind
        else {
            panic!();
        };
        assert_eq!(ip_vs_ai, &[Action::Fold, Action::Call]);

        // D2 & D5: Partial stack strictly between call amount and full pot-raise, AND non-round numbers!
        // Starting pot = 73, effective stack = 195.
        // - OOP opens 100% pot -> `Bet(73)`.
        // - Facing `Bet(73)`, IP's `pot_after_call = 73 + 2*73 = 219`. Uncapped pot-raise-to would be `73 + 219 = 292`.
        // - Since `effective_stack = 195` (`73 < 195 < 292`), IP's `50%` pot raise is `73 + round(0.5 * 219) = 73 + 110 = 183`
        //   (Wait: remaining stack `195 - 183 = 12`, let's set `force_allin_spr_threshold = 0.0` and `merging_threshold = 0.0`
        //   to verify exact un-snapped arithmetic: `Raise(183)` AND `AllIn(195)` when `100%` clamps to `min(292, 195) = 195`!).
        let partial_cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 73,
            effective_stack: 195,
            betting_limit: BettingLimit::PotLimit,
            river_sizes: [
                StreetBetSizes {
                    bet_fractions: vec![1.0, 3.0], // 3.0x overbet must clamp to 1.0x pot = 73!
                    raise_fractions: vec![1.0],
                    donk_fractions: None,
                },
                StreetBetSizes {
                    bet_fractions: vec![1.0],
                    raise_fractions: vec![0.5, 1.0],
                    donk_fractions: None,
                },
            ],
            add_allin_spr_threshold: 0.0,
            force_allin_spr_threshold: 0.0,
            merging_threshold: 0.0,
            ..Default::default()
        };
        let partial_tree = ActionTree::build(partial_cfg).unwrap();
        let ActionNodeKind::Player {
            actions: p_root_acts,
            children: p_root_ch,
            ..
        } = &partial_tree.node(partial_tree.root_id()).kind
        else {
            panic!();
        };
        // Overbet `3.0` was strictly clamped to Pot-Limit `Bet(73)`!
        assert_eq!(p_root_acts, &[Action::Check, Action::Bet(73)]);

        let ActionNodeKind::Player {
            actions: ip_partial_acts,
            ..
        } = &partial_tree.node(p_root_ch[1]).kind
        else {
            panic!();
        };
        assert_eq!(
            ip_partial_acts,
            &[
                Action::Fold,
                Action::Call,
                Action::Raise(183),
                Action::AllIn(195)
            ]
        );

        // Also check incomplete all-in raise (`73 < stack = 110 < min_raise_to = 146`):
        // When OOP bets `73` and effective stack is `110`, `min_raise_to = min(73 + 73, 110) = 110`,
        // so IP can still shove `AllIn(110)` legally, and facing `AllIn(110)` OOP only has `[Fold, Call]`!
        let short_raise_cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 73,
            effective_stack: 110,
            betting_limit: BettingLimit::PotLimit,
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            ..Default::default()
        };
        let sr_tree = ActionTree::build(short_raise_cfg).unwrap();
        let ActionNodeKind::Player { children: sr_ch, .. } =
            &sr_tree.node(sr_tree.root_id()).kind
        else {
            panic!();
        };
        let ActionNodeKind::Player {
            actions: sr_ip_acts,
            children: sr_ip_ch,
            ..
        } = &sr_tree.node(sr_ch[1]).kind
        else {
            panic!();
        };
        assert_eq!(sr_ip_acts, &[Action::Fold, Action::Call, Action::AllIn(110)]);
        let ActionNodeKind::Player {
            actions: sr_oop_vs_ai,
            ..
        } = &sr_tree.node(sr_ip_ch[2]).kind
        else {
            panic!();
        };
        assert_eq!(sr_oop_vs_ai, &[Action::Fold, Action::Call]);

        // D6: Raise count cap (`max_bets_per_street = 2` -> Bet + 1 Raise allowed, 3-bet forbidden!)
        let cap_cfg = TreeConfig {
            initial_street: Street::River,
            starting_pot: 100,
            effective_stack: 2000,
            max_bets_per_street: 2,
            river_sizes: [StreetBetSizes::pot_only(), StreetBetSizes::pot_only()],
            add_allin_spr_threshold: 0.0,
            ..Default::default()
        };
        let cap_tree = ActionTree::build(cap_cfg).unwrap();
        let ActionNodeKind::Player { children: c0, .. } = &cap_tree.node(cap_tree.root_id()).kind
        else {
            panic!();
        };
        // OOP Bet(100) -> num_bets = 1 -> IP can Raise(400) -> num_bets = 2 -> OOP CANNOT 3-bet!
        let ActionNodeKind::Player { children: c1, .. } = &cap_tree.node(c0[1]).kind else {
            panic!();
        };
        let ActionNodeKind::Player {
            actions: capped_oop_acts,
            ..
        } = &cap_tree.node(c1[2]).kind
        else {
            panic!();
        };
        assert_eq!(capped_oop_acts, &[Action::Fold, Action::Call]);

        // D7: Donk bet toggle (`Some(vec![])` disables OOP turn donk lead after OOP check-called flop;
        // `Some(vec![0.5])` enables ONLY 50% donk lead on turn, while after flop check-check OOP still has full `bet_fractions`!).
        let donk_cfg = TreeConfig {
            initial_street: Street::Turn,
            starting_pot: 100,
            effective_stack: 500,
            turn_sizes: [
                StreetBetSizes::pot_only(),
                StreetBetSizes::pot_only(),
            ],
            river_sizes: [
                StreetBetSizes {
                    bet_fractions: vec![1.0],
                    raise_fractions: vec![1.0],
                    donk_fractions: Some(vec![]), // No river donk after OOP check-called turn!
                },
                StreetBetSizes::pot_only(),
            ],
            add_allin_spr_threshold: 0.0,
            ..Default::default()
        };
        let donk_tree = ActionTree::build(donk_cfg).unwrap();
        // Path A: Turn Check -> Check -> Chance -> River OOP root (oop_called_prev_street == false -> has `Bet(100)`)!
        let ActionNodeKind::Player { children: t_root_ch, .. } =
            &donk_tree.node(donk_tree.root_id()).kind
        else {
            panic!();
        };
        let ActionNodeKind::Player { children: t_ip_ch, .. } = &donk_tree.node(t_root_ch[0]).kind
        else {
            panic!();
        };
        let ActionNodeKind::Chance { child: r_after_xx, .. } = donk_tree.node(t_ip_ch[0]).kind
        else {
            panic!();
        };
        let ActionNodeKind::Player {
            actions: r_xx_acts, ..
        } = &donk_tree.node(r_after_xx).kind
        else {
            panic!();
        };
        assert_eq!(r_xx_acts, &[Action::Check, Action::Bet(100)]);

        // Path B: Turn Check -> IP Bet(100) -> OOP Call -> Chance -> River OOP root (`oop_called_prev_street == true` -> `donk_fractions = []` -> ONLY `[Check]`!)
        let ActionNodeKind::Player {
            children: t_oop_vs_bet_ch,
            ..
        } = &donk_tree.node(t_ip_ch[1]).kind
        else {
            panic!();
        };
        let ActionNodeKind::Chance { child: r_after_xc, .. } =
            donk_tree.node(t_oop_vs_bet_ch[1]).kind
        else {
            panic!();
        };
        let ActionNodeKind::Player {
            actions: r_xc_acts, ..
        } = &donk_tree.node(r_after_xc).kind
        else {
            panic!();
        };
        assert_eq!(r_xc_acts, &[Action::Check]);

        // D8: Global tree walk verifying ZERO illegal Pot-Limit bets/raises exist at ANY node of a multi-street tree!
        assert_every_node_obeys_pot_limit(&partial_tree);
        assert_every_node_obeys_pot_limit(&donk_tree);
    }

    fn assert_every_node_obeys_pot_limit(tree: &ActionTree) {
        assert_eq!(tree.config().betting_limit, BettingLimit::PotLimit);
        let stack = tree.config().effective_stack;
        for node in tree.nodes() {
            if let ActionNodeKind::Player {
                player, actions, ..
            } = &node.kind
            {
                let p = *player;
                let opp = p ^ 1;
                let my_c = node.committed[p];
                let opp_c = node.committed[opp];
                let call_amt = opp_c - my_c;
                assert!(call_amt >= 0);
                // Current pot before acting is `node.pot = starting_pot + my_c + opp_c`.
                // After calling `call_amt`, pot is `node.pot + call_amt`.
                // Max legal additional raise above `opp_c` in Pot-Limit is `pot_after_call = node.pot + call_amt`,
                // capped by remaining stack `stack - opp_c`!
                let pot_after_call = node.pot + call_amt;
                let max_total_commit = (opp_c + pot_after_call).min(stack);

                for &a in actions {
                    match a {
                        Action::Check => assert_eq!(call_amt, 0, "Check illegal when facing bet"),
                        Action::Fold | Action::Call => {
                            assert!(call_amt > 0, "Fold/Call illegal when facing == 0")
                        }
                        Action::Bet(to) | Action::Raise(to) | Action::AllIn(to) => {
                            // Note `to` is street_invested; let's deduce street base_committed:
                            // `my_c >= 0` and `opp_c - my_c == opp_inv - my_inv == call_amt`, so
                            // additional chips added by `to` beyond `my_inv` is `(base + to) - my_c`.
                            // Wait: `base` is at least 0, and on street `to <= stack - base`, while
                            // raise increment over `opp_inv` (`to - opp_inv`) MUST satisfy:
                            // `to - opp_inv <= pot_after_call` AND `base + to <= stack`!
                            // Since `opp_c - my_c = opp_inv - my_inv` and at street open `my_inv == 0 => base == my_c`,
                            // let's check child node's `committed[p]` directly!
                            let _ = (to, max_total_commit);
                        }
                    }
                }
            }
        }
        // Check parent -> child total commitment delta against exact `max_total_commit`!
        for node in tree.nodes() {
            if let ActionNodeKind::Player {
                player,
                actions,
                children,
                ..
            } = &node.kind
            {
                let p = *player;
                let opp = p ^ 1;
                let call_amt = node.committed[opp] - node.committed[p];
                let max_total_commit = (node.committed[opp] + node.pot + call_amt).min(stack);
                for (&act, &ch_id) in actions.iter().zip(children.iter()) {
                    let ch = tree.node(ch_id);
                    match act {
                        Action::Bet(_) | Action::Raise(_) => {
                            assert!(
                                ch.committed[p] <= max_total_commit && ch.committed[p] < stack,
                                "illegal Bet/Raise commit {} > max_legal {} (stack {})",
                                ch.committed[p],
                                max_total_commit,
                                stack
                            );
                            assert!(ch.committed[p] > node.committed[opp]);
                        }
                        Action::AllIn(_) => {
                            assert_eq!(ch.committed[p], stack);
                            assert!(
                                ch.committed[p] <= max_total_commit,
                                "over-pot NL AllIn({}) leaked into PotLimit tree (max_legal={max_total_commit})!",
                                ch.committed[p]
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
