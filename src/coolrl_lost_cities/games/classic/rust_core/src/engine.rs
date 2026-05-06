use std::collections::HashMap;

use crate::config::Config;
use crate::error::EngineError;
use crate::proto;
use crate::state::{GameState, Phase};

#[derive(Clone, Debug)]
struct SessionState {
    game: GameState,
    state_version: u64,
}

impl SessionState {
    fn observation(&self, session_id: &str, observer: usize) -> proto::GameObservation {
        let game = &self.game;
        let can_act = !game.terminal && observer == game.current_player;
        let my_score = game.total_score(observer);
        let opponent_score = game.total_score(1 - observer);

        proto::GameObservation {
            session_id: session_id.to_string(),
            config: Some((&game.config).into()),
            state_version: self.state_version,
            observer_player: observer as u32,
            current_player: game.current_player as u32,
            phase: game.phase.to_proto(),
            hand: game.hands[observer]
                .iter()
                .copied()
                .map(|card| card.to_proto(&game.config))
                .collect(),
            opponent_hand_size: game.hands[1 - observer].len() as u32,
            my_expeditions: Self::build_expeditions(observer, game),
            opponent_expeditions: Self::build_expeditions(1 - observer, game),
            discards: Self::build_discards(game),
            deck_size: game.deck.len() as u32,
            pending_discarded_color: if game.phase == Phase::Draw {
                game.pending_discarded_color
            } else {
                None
            },
            legal_actions: Some(game.build_legal_action_set(self.state_version, can_act)),
            turn_count: game.turn_count,
            terminal: game.terminal,
            my_score,
            opponent_score,
            score_diff: my_score - opponent_score,
        }
    }

    fn final_scores(&self) -> HashMap<u32, i32> {
        HashMap::from([(0, self.game.total_score(0)), (1, self.game.total_score(1))])
    }

    fn build_expeditions(player: usize, game: &GameState) -> Vec<proto::Expedition> {
        game.expeditions[player]
            .iter()
            .enumerate()
            .map(|(color, cards)| proto::Expedition {
                color: color as u32,
                cards: cards
                    .iter()
                    .copied()
                    .map(|card| card.to_proto(&game.config))
                    .collect(),
                current_score: crate::score_expedition(cards, &game.config),
            })
            .collect()
    }

    fn build_discards(game: &GameState) -> Vec<proto::DiscardPile> {
        game.discards
            .iter()
            .enumerate()
            .map(|(color, cards)| proto::DiscardPile {
                color: color as u32,
                cards: cards
                    .iter()
                    .copied()
                    .map(|card| card.to_proto(&game.config))
                    .collect(),
                size: cards.len() as u32,
            })
            .collect()
    }
}

#[derive(Default)]
pub struct LostCitiesEngine {
    sessions: HashMap<String, SessionState>,
}

impl LostCitiesEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn new_game(
        &mut self,
        request: proto::NewGameRequest,
    ) -> Result<proto::GameObservation, EngineError> {
        let session_id = request.session_id;
        Self::validate_session_id(&session_id)?;
        if self.sessions.contains_key(&session_id) {
            return Err(EngineError::already_exists(format!(
                "session {} already exists",
                session_id
            )));
        }

        let config_proto = request
            .config
            .ok_or_else(|| EngineError::invalid_argument("config is required"))?;
        let config = Config::try_from(config_proto)?;
        let session = SessionState {
            game: GameState::new_game(config)?,
            state_version: 0,
        };

        let observation = session.observation(&session_id, 0);
        self.sessions.insert(session_id, session);
        Ok(observation)
    }

    pub fn get_observation(
        &self,
        request: proto::SessionRef,
    ) -> Result<proto::GameObservation, EngineError> {
        let session_id = request.session_id;
        Self::validate_session_id(&session_id)?;
        let session = self
            .sessions
            .get(&session_id)
            .ok_or_else(|| EngineError::not_found(format!("unknown session {}", session_id)))?;
        let observer = Self::requested_or_default_observer(
            request.observer_player,
            session.game.current_player,
        )?;
        Ok(session.observation(&session_id, observer))
    }

    pub fn apply_action(
        &mut self,
        request: proto::ApplyActionRequest,
    ) -> Result<proto::StepResult, EngineError> {
        let session_id = request.session_id;
        Self::validate_session_id(&session_id)?;
        let requested_observer = request
            .observer_player
            .map(Self::validate_observer)
            .transpose()?;
        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| EngineError::not_found(format!("unknown session {}", session_id)))?;

        if session.game.terminal {
            return Err(EngineError::failed_precondition("game is already terminal"));
        }
        if request.expected_state_version != session.state_version {
            return Err(EngineError::failed_precondition(format!(
                "expected state_version {}, got {}",
                session.state_version, request.expected_state_version
            )));
        }

        session.game.apply_unified_action(request.action_id)?;
        session.state_version += 1;

        let observer = requested_observer.unwrap_or(session.game.current_player);
        let observation = session.observation(&session_id, observer);
        let terminal = session.game.terminal;
        let reward = if terminal {
            f64::from(observation.score_diff)
        } else {
            0.0
        };
        let final_scores = if terminal {
            session.final_scores()
        } else {
            HashMap::new()
        };

        Ok(proto::StepResult {
            observation: Some(observation),
            reward,
            terminal,
            final_scores,
        })
    }

    pub fn end_session(&mut self, request: proto::SessionRef) -> Result<(), EngineError> {
        Self::validate_session_id(&request.session_id)?;
        self.sessions.remove(&request.session_id);
        Ok(())
    }

    fn validate_session_id(session_id: &str) -> Result<(), EngineError> {
        if session_id.trim().is_empty() {
            return Err(EngineError::invalid_argument(
                "session_id must not be empty",
            ));
        }
        Ok(())
    }

    fn requested_or_default_observer(
        observer_player: Option<u32>,
        default_player: usize,
    ) -> Result<usize, EngineError> {
        observer_player
            .map(Self::validate_observer)
            .transpose()
            .map(|observer| observer.unwrap_or(default_player))
    }

    fn validate_observer(observer: u32) -> Result<usize, EngineError> {
        match observer {
            0 => Ok(0),
            1 => Ok(1),
            _ => Err(EngineError::invalid_argument(
                "observer_player must be 0 or 1",
            )),
        }
    }
}
