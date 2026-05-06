use crate::config::Config;
use crate::error::EngineError;
use crate::proto;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Card {
    pub color: u32,
    pub rank: u32,
}

impl Card {
    pub fn is_handshake(self) -> bool {
        self.rank == 0
    }

    pub fn numeric_value(self, min_rank: u32) -> u32 {
        if self.is_handshake() {
            0
        } else {
            min_rank + self.rank - 1
        }
    }

    pub fn label(self, min_rank: u32) -> String {
        if self.is_handshake() {
            format!("[{}]H", self.color)
        } else {
            format!("[{}]{}", self.color, self.numeric_value(min_rank))
        }
    }

    pub fn to_proto(self, config: &Config) -> proto::Card {
        proto::Card {
            color: self.color,
            rank: self.rank,
            numeric_value: self.numeric_value(config.min_rank),
            label: self.label(config.min_rank),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Card,
    Draw,
}

impl Phase {
    pub fn to_proto(self) -> i32 {
        match self {
            Self::Card => proto::Phase::Card as i32,
            Self::Draw => proto::Phase::Draw as i32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub config: Config,
    pub deck: Vec<Card>,
    pub hands: [Vec<Card>; 2],
    pub expeditions: [Vec<Vec<Card>>; 2],
    pub discards: Vec<Vec<Card>>,
    pub current_player: usize,
    pub phase: Phase,
    pub pending_discarded_color: Option<u32>,
    pub turn_count: u32,
    pub terminal: bool,
}

impl GameState {
    pub fn new_game(config: Config) -> Result<Self, EngineError> {
        config.validate()?;
        let mut deck = build_deck(&config);
        match config.seed {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                deck.shuffle(&mut rng);
            }
            None => {
                let mut rng = rand::thread_rng();
                deck.shuffle(&mut rng);
            }
        }

        Self::new_game_from_deck(config, deck)
    }

    pub fn new_game_from_deck(config: Config, deck: Vec<Card>) -> Result<Self, EngineError> {
        config.validate()?;
        let mut expected = build_deck(&config);
        let mut actual = deck.clone();
        expected.sort();
        actual.sort();
        if actual != expected {
            return Err(EngineError::invalid_argument(
                "deck must contain exactly the cards defined by config",
            ));
        }

        let mut state = Self::empty(config)?;
        state.deck = deck;
        for _ in 0..state.config.hand_size {
            for player in 0..2 {
                let card = state
                    .deck
                    .pop()
                    .expect("validated deck must contain both initial hands");
                state.hands[player].push(card);
            }
        }
        state.sort_hands();
        state
            .validate_invariants()
            .map_err(EngineError::invalid_argument)?;
        Ok(state)
    }

    pub fn empty(config: Config) -> Result<Self, EngineError> {
        config.validate()?;
        let n_colors = config.n_colors;
        Ok(Self {
            config,
            deck: Vec::new(),
            hands: [Vec::new(), Vec::new()],
            expeditions: std::array::from_fn(|_| vec![Vec::new(); n_colors]),
            discards: vec![Vec::new(); n_colors],
            current_player: 0,
            phase: Phase::Card,
            pending_discarded_color: None,
            turn_count: 0,
            terminal: false,
        })
    }

    pub fn sort_hands(&mut self) {
        self.sort_hand(0);
        self.sort_hand(1);
    }

    pub fn sort_hand(&mut self, player: usize) {
        self.hands[player].sort_by_key(|card| (card.color, card.rank));
    }

    pub fn last_numeric_rank(&self, player: usize, color: usize) -> u32 {
        self.expeditions[player][color]
            .iter()
            .filter(|card| !card.is_handshake())
            .map(|card| card.rank)
            .max()
            .unwrap_or(0)
    }

    pub fn can_play_card(&self, player: usize, card: Card) -> bool {
        let color = match usize::try_from(card.color) {
            Ok(value) if value < self.config.n_colors => value,
            _ => return false,
        };
        if card.rank > self.config.n_ranks {
            return false;
        }
        let last_numeric = self.last_numeric_rank(player, color);
        if card.is_handshake() {
            last_numeric == 0
        } else {
            card.rank > last_numeric
        }
    }

    pub fn legal_card_mask_phase(&self) -> Vec<bool> {
        let mut mask = vec![false; self.config.card_action_size()];
        if self.terminal {
            return mask;
        }

        for (slot, card) in self.hands[self.current_player].iter().copied().enumerate() {
            let play_id = Self::play_action_id(slot);
            mask[play_id] = self.can_play_card(self.current_player, card);
            mask[Self::discard_action_id(slot)] = true;
        }
        mask
    }

    pub fn legal_draw_mask_phase(&self) -> Vec<bool> {
        let mut mask = vec![false; self.config.draw_action_size()];
        if self.terminal {
            return mask;
        }

        mask[0] = !self.deck.is_empty();
        for color in 0..self.config.n_colors {
            let pending = self.pending_discarded_color == Some(color as u32);
            mask[1 + color] = !self.discards[color].is_empty() && !pending;
        }
        mask
    }

    pub fn legal_unified_mask(&self) -> Vec<bool> {
        if self.terminal {
            return vec![false; self.config.action_space_size()];
        }
        match self.phase {
            Phase::Card => {
                let mut mask = self.legal_card_mask_phase();
                mask.resize(self.config.action_space_size(), false);
                mask
            }
            Phase::Draw => {
                let mut mask = vec![false; self.config.card_action_size()];
                mask.extend(self.legal_draw_mask_phase());
                mask
            }
        }
    }

    pub fn apply_unified_action(&mut self, action_id: u32) -> Result<(), EngineError> {
        if self.terminal {
            return Err(EngineError::failed_precondition("game is already terminal"));
        }

        let action_index = usize::try_from(action_id)
            .map_err(|_| EngineError::failed_precondition("action_id is out of range"))?;
        let mask = self.legal_unified_mask();
        if action_index >= mask.len() || !mask[action_index] {
            return Err(EngineError::failed_precondition(format!(
                "illegal action {} in phase {:?} for player {}",
                action_id, self.phase, self.current_player
            )));
        }

        match self.phase {
            Phase::Card => self.apply_card_action(action_index),
            Phase::Draw => self.apply_draw_action(action_index),
        }
        Ok(())
    }

    pub fn total_score(&self, player: usize) -> i32 {
        self.expeditions[player]
            .iter()
            .map(|expedition| score_expedition(expedition, &self.config))
            .sum()
    }

    pub fn score_diff(&self, player: usize) -> i32 {
        self.total_score(player) - self.total_score(1 - player)
    }

    pub fn validate_invariants(&self) -> Result<(), String> {
        self.config.validate().map_err(|err| err.to_string())?;
        if self.current_player > 1 {
            return Err("current_player must be 0 or 1".to_string());
        }
        if self.discards.len() != self.config.n_colors {
            return Err("discard pile count must match n_colors".to_string());
        }

        let mut all_cards = Vec::new();
        all_cards.extend(self.deck.iter().copied());
        for (player_index, hand) in self.hands.iter().enumerate() {
            if hand.len() > self.config.hand_size {
                return Err(format!("hand {} exceeds hand_size", player_index));
            }
            if !hand.windows(2).all(|pair| pair[0] <= pair[1]) {
                return Err(format!("hand {} is not sorted", player_index));
            }
            all_cards.extend(hand.iter().copied());
        }

        for (player_index, expeditions) in self.expeditions.iter().enumerate() {
            if expeditions.len() != self.config.n_colors {
                return Err("expedition color count must match n_colors".to_string());
            }
            for (color, expedition) in expeditions.iter().enumerate() {
                let mut seen_numeric = false;
                let mut last_numeric = 0;
                for card in expedition {
                    if card.color as usize != color {
                        return Err(format!(
                            "player {} expedition {} contains wrong color",
                            player_index, color
                        ));
                    }
                    if card.is_handshake() {
                        if seen_numeric {
                            return Err(format!(
                                "player {} expedition {} has handshake after numeric",
                                player_index, color
                            ));
                        }
                        continue;
                    }
                    seen_numeric = true;
                    if card.rank <= last_numeric {
                        return Err(format!(
                            "player {} expedition {} is not strictly increasing",
                            player_index, color
                        ));
                    }
                    last_numeric = card.rank;
                }
                all_cards.extend(expedition.iter().copied());
            }
        }
        for discard in &self.discards {
            all_cards.extend(discard.iter().copied());
        }

        let mut expected = build_deck(&self.config);
        expected.sort();
        all_cards.sort();
        if all_cards != expected {
            return Err("card conservation failed".to_string());
        }

        if self.phase == Phase::Card && self.pending_discarded_color.is_some() {
            return Err("pending_discarded_color must be None during card phase".to_string());
        }
        if let Some(color) = self.pending_discarded_color {
            let color = color as usize;
            if color >= self.config.n_colors {
                return Err("pending_discarded_color is out of range".to_string());
            }
            if self.discards[color].is_empty() {
                return Err("pending discard color must have a discard pile card".to_string());
            }
        }

        let any_legal = self.legal_unified_mask().into_iter().any(|value| value);
        if self.terminal && any_legal {
            return Err("terminal state must have no legal actions".to_string());
        }
        if !self.terminal && !any_legal {
            return Err("non-terminal state must have at least one legal action".to_string());
        }

        Ok(())
    }

    pub(crate) fn build_legal_action_set(
        &self,
        state_version: u64,
        include_actions: bool,
    ) -> proto::LegalActionSet {
        if self.terminal || !include_actions {
            return self.empty_legal_action_set(state_version);
        }

        let mask = self.legal_unified_mask();
        let actions = match self.phase {
            Phase::Card => self.build_card_actions(&mask),
            Phase::Draw => self.build_draw_actions(&mask),
        };

        proto::LegalActionSet {
            state_version,
            actions,
            mask,
            action_space_size: self.config.action_space_size() as u32,
            phase: self.phase.to_proto(),
        }
    }

    fn apply_card_action(&mut self, action_index: usize) {
        let slot = action_index / 2;
        let play = action_index.is_multiple_of(2);
        let card = self.hands[self.current_player].remove(slot);
        let color = card.color as usize;

        if play {
            self.expeditions[self.current_player][color].push(card);
            self.pending_discarded_color = None;
        } else {
            self.discards[color].push(card);
            self.pending_discarded_color = Some(card.color);
        }

        self.phase = Phase::Draw;
        if self.deck.is_empty() && !self.has_draw_source() {
            self.terminal = true;
        }
    }

    fn apply_draw_action(&mut self, action_index: usize) {
        let draw_index = action_index - self.draw_action_offset();
        let card = if draw_index == 0 {
            self.deck.pop().expect("legal deck draw")
        } else {
            let color = draw_index - 1;
            self.discards[color].pop().expect("legal discard draw")
        };

        self.hands[self.current_player].push(card);
        self.sort_hand(self.current_player);
        self.pending_discarded_color = None;
        self.turn_count += 1;

        if self.deck.is_empty() {
            self.terminal = true;
            return;
        }

        self.current_player = 1 - self.current_player;
        self.phase = Phase::Card;
    }

    fn build_card_actions(&self, mask: &[bool]) -> Vec<proto::Action> {
        let mut actions = Vec::new();
        for (slot, card) in self.hands[self.current_player].iter().copied().enumerate() {
            let play_id = Self::play_action_id(slot);
            if mask[play_id] {
                actions.push(self.card_action(
                    play_id,
                    proto::ActionKind::PlayCard,
                    slot,
                    card,
                    "Play",
                ));
            }
            let discard_id = Self::discard_action_id(slot);
            if mask[discard_id] {
                actions.push(self.card_action(
                    discard_id,
                    proto::ActionKind::DiscardCard,
                    slot,
                    card,
                    "Discard",
                ));
            }
        }
        actions
    }

    fn build_draw_actions(&self, mask: &[bool]) -> Vec<proto::Action> {
        let mut actions = Vec::new();
        let deck_draw_id = self.draw_action_offset();
        if mask[deck_draw_id] {
            actions.push(self.draw_action(deck_draw_id, None));
        }

        for color in 0..self.config.n_colors {
            let action_id = self.discard_draw_action_id(color);
            if mask[action_id] {
                actions.push(self.draw_action(action_id, Some(color)));
            }
        }
        actions
    }

    fn empty_legal_action_set(&self, state_version: u64) -> proto::LegalActionSet {
        proto::LegalActionSet {
            state_version,
            actions: Vec::new(),
            mask: vec![false; self.config.action_space_size()],
            action_space_size: self.config.action_space_size() as u32,
            phase: self.phase.to_proto(),
        }
    }

    fn has_draw_source(&self) -> bool {
        self.legal_draw_mask_phase().into_iter().any(|value| value)
    }

    fn draw_action_offset(&self) -> usize {
        self.config.card_action_size()
    }

    fn play_action_id(slot: usize) -> usize {
        slot * 2
    }

    fn discard_action_id(slot: usize) -> usize {
        Self::play_action_id(slot) + 1
    }

    fn discard_draw_action_id(&self, color: usize) -> usize {
        self.draw_action_offset() + 1 + color
    }

    fn card_action(
        &self,
        id: usize,
        kind: proto::ActionKind,
        slot: usize,
        card: Card,
        verb: &str,
    ) -> proto::Action {
        proto::Action {
            id: id as u32,
            kind: kind as i32,
            hand_slot: slot as u32,
            card: Some(card.to_proto(&self.config)),
            discard_color: 0,
            description: format!("{verb} {}", card.label(self.config.min_rank)),
        }
    }

    fn draw_action(&self, id: usize, discard_color: Option<usize>) -> proto::Action {
        let (kind, discard_color, description) = match discard_color {
            Some(color) => (
                proto::ActionKind::DrawDiscard,
                color as u32,
                format!("Draw discard {color}"),
            ),
            None => (proto::ActionKind::DrawDeck, 0, "Draw deck".to_string()),
        };

        proto::Action {
            id: id as u32,
            kind: kind as i32,
            hand_slot: 0,
            card: None,
            discard_color,
            description,
        }
    }
}

pub fn build_deck(config: &Config) -> Vec<Card> {
    let mut deck = Vec::with_capacity(config.deck_size());
    for color in 0..config.n_colors as u32 {
        for _ in 0..config.n_handshakes {
            deck.push(Card { color, rank: 0 });
        }
        for rank in 1..=config.n_ranks {
            deck.push(Card { color, rank });
        }
    }
    deck
}

pub fn score_expedition(expedition: &[Card], config: &Config) -> i32 {
    if expedition.is_empty() {
        return 0;
    }

    let handshakes = expedition.iter().filter(|card| card.is_handshake()).count() as i32;
    let numeric_sum = expedition
        .iter()
        .map(|card| card.numeric_value(config.min_rank) as i32)
        .sum::<i32>();
    let mut score = (numeric_sum + config.expedition_penalty) * (handshakes + 1);
    if expedition.len() >= config.bonus_threshold {
        score += config.bonus_amount;
    }
    score
}
