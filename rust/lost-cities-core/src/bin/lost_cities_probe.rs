use std::env;
use std::error::Error;
use std::fs;
use std::io;

use lost_cities_core::proto::{
    self, lost_cities_client::LostCitiesClient, lost_cities_server::LostCitiesServer,
};
use lost_cities_core::{
    Card, Config, EngineErrorKind, GameState, LostCitiesEngine, LostCitiesGrpcService, Phase,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Channel, Endpoint, Server};

type ProbeResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct CardJson {
    color: u32,
    rank: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ConfigJson {
    n_colors: usize,
    n_ranks: u32,
    min_rank: u32,
    n_handshakes: u32,
    hand_size: usize,
    expedition_penalty: i32,
    bonus_threshold: usize,
    bonus_amount: i32,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct FixtureInput {
    config: ConfigJson,
    initial_deck: Vec<CardJson>,
    steps: Vec<FixtureStepInput>,
}

#[derive(Debug, Deserialize)]
struct FixtureStepInput {
    action: Option<u32>,
}

#[derive(Debug, Serialize)]
struct TraceOutput {
    config: ConfigJson,
    steps: Vec<StateTraceStep>,
}

#[derive(Debug, Serialize)]
struct StateTraceStep {
    action: Option<u32>,
    phase: &'static str,
    current_player: usize,
    turn_count: u32,
    terminal: bool,
    pending_discarded_color: Option<u32>,
    score_diff_player0: i32,
    legal_mask: Vec<bool>,
    deck: Vec<CardJson>,
    hands: Vec<Vec<CardJson>>,
    expeditions: Vec<Vec<Vec<CardJson>>>,
    discards: Vec<Vec<CardJson>>,
}

#[derive(Debug, Serialize)]
struct ObservationMini {
    state_version: u64,
    current_player: u32,
    observer_player: u32,
    phase: String,
    terminal: bool,
}

#[derive(Debug, Serialize, PartialEq)]
struct ObservationSummary {
    state_version: u64,
    current_player: u32,
    observer_player: u32,
    phase: String,
    hand: Vec<CardJson>,
    opponent_hand_size: u32,
    deck_size: u32,
    discards: Vec<Vec<CardJson>>,
    my_expeditions: Vec<Vec<CardJson>>,
    opponent_expeditions: Vec<Vec<CardJson>>,
    legal_mask: Vec<bool>,
    terminal: bool,
    my_score: i32,
    opponent_score: i32,
    score_diff: i32,
}

#[derive(Debug, Serialize)]
struct EngineProbeOutput {
    duplicate_kind: String,
    missing_config_kind: String,
    empty_session_kind: String,
    unknown_session_kind: String,
    invalid_observer_kind: String,
    invalid_observer_state_unchanged: bool,
    invalid_observer_action_still_applies: bool,
    phase_flow: Vec<ObservationMini>,
    stale_kind: String,
    end_session_counts: Vec<usize>,
    off_turn_legal_empty: bool,
    full_session_terminal_reward_matches: bool,
    full_session_final_scores_match: bool,
    terminal_reject_kind: String,
    deterministic_match: bool,
}

#[derive(Debug, Serialize)]
struct GrpcProbeOutput {
    round_trip_phase: String,
    opponent_legal_empty: bool,
    stale_code: String,
    invalid_observer_code: String,
    invalid_observer_state_unchanged: bool,
    ended_session_code: String,
}

fn main() -> ProbeResult<()> {
    let mut args = env::args().skip(1);
    let command = args
        .next()
        .ok_or_else(|| io::Error::other("expected command"))?;
    match command.as_str() {
        "defaults" => print_json(&ConfigJson::from_config(&Config::default()))?,
        "trace" => {
            let path = args
                .next()
                .ok_or_else(|| io::Error::other("expected fixture path"))?;
            print_json(&run_trace(&path)?)?;
        }
        "engine" => print_json(&run_engine_probe()?)?,
        "grpc" => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            print_json(&runtime.block_on(run_grpc_probe())?)?;
        }
        _ => return Err(io::Error::other(format!("unknown command: {command}")).into()),
    }
    Ok(())
}

fn print_json<T: Serialize>(value: &T) -> ProbeResult<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn run_trace(path: &str) -> ProbeResult<TraceOutput> {
    let fixture: FixtureInput = serde_json::from_str(&fs::read_to_string(path)?)?;
    let config = fixture.config.to_config();
    let deck = fixture
        .initial_deck
        .iter()
        .copied()
        .map(Card::from)
        .collect::<Vec<_>>();
    let mut state = GameState::new_game_from_deck(config, deck)?;
    let mut steps = Vec::with_capacity(fixture.steps.len());

    for step in fixture.steps {
        if let Some(action) = step.action {
            state.apply_unified_action(action)?;
        }
        state.validate_invariants().map_err(io::Error::other)?;
        steps.push(StateTraceStep::from_state(step.action, &state));
    }

    Ok(TraceOutput {
        config: ConfigJson::from_config(&state.config),
        steps,
    })
}

fn run_engine_probe() -> ProbeResult<EngineProbeOutput> {
    let duplicate_kind = {
        let mut engine = LostCitiesEngine::new();
        engine.new_game(new_game_request("dup", 7, true))?;
        kind_name(
            engine
                .new_game(new_game_request("dup", 7, true))
                .expect_err("duplicate session must fail")
                .kind(),
        )
    };

    let missing_config_kind = {
        let mut engine = LostCitiesEngine::new();
        kind_name(
            engine
                .new_game(new_game_request("missing-config", 7, false))
                .expect_err("missing config must fail")
                .kind(),
        )
    };

    let empty_session_kind = {
        let mut engine = LostCitiesEngine::new();
        kind_name(
            engine
                .new_game(new_game_request("   ", 7, true))
                .expect_err("empty session id must fail")
                .kind(),
        )
    };

    let unknown_session_kind = {
        let engine = LostCitiesEngine::new();
        kind_name(
            engine
                .get_observation(proto::SessionRef {
                    session_id: "unknown".to_string(),
                    observer_player: None,
                })
                .expect_err("unknown session must fail")
                .kind(),
        )
    };

    let (
        invalid_observer_kind,
        invalid_observer_state_unchanged,
        invalid_observer_action_still_applies,
    ) = {
        let mut engine = LostCitiesEngine::new();
        let observation = engine.new_game(new_game_request("bad-observer", 7, true))?;
        let action_id = first_action(&observation)?;
        let err = engine
            .apply_action(proto::ApplyActionRequest {
                session_id: "bad-observer".to_string(),
                action_id,
                expected_state_version: observation.state_version,
                observer_player: Some(2),
            })
            .expect_err("invalid observer must fail");
        let after_error = engine.get_observation(proto::SessionRef {
            session_id: "bad-observer".to_string(),
            observer_player: Some(0),
        })?;
        let state_unchanged = after_error.state_version == observation.state_version
            && after_error.phase == observation.phase
            && after_error.current_player == observation.current_player;
        let still_applies = engine
            .apply_action(proto::ApplyActionRequest {
                session_id: "bad-observer".to_string(),
                action_id,
                expected_state_version: observation.state_version,
                observer_player: Some(0),
            })
            .is_ok();
        (kind_name(err.kind()), state_unchanged, still_applies)
    };

    let phase_flow = {
        let mut engine = LostCitiesEngine::new();
        let observation = engine.new_game(new_game_request("phase-flow", 11, true))?;
        let first = engine
            .apply_action(proto::ApplyActionRequest {
                session_id: "phase-flow".to_string(),
                action_id: first_action(&observation)?,
                expected_state_version: observation.state_version,
                observer_player: None,
            })?
            .observation
            .ok_or_else(|| io::Error::other("missing first observation"))?;
        let second = engine
            .apply_action(proto::ApplyActionRequest {
                session_id: "phase-flow".to_string(),
                action_id: first_action(&first)?,
                expected_state_version: first.state_version,
                observer_player: None,
            })?
            .observation
            .ok_or_else(|| io::Error::other("missing second observation"))?;
        vec![
            ObservationMini::from_observation(&observation),
            ObservationMini::from_observation(&first),
            ObservationMini::from_observation(&second),
        ]
    };

    let stale_kind = {
        let mut engine = LostCitiesEngine::new();
        let observation = engine.new_game(new_game_request("stale", 3, true))?;
        let action_id = first_action(&observation)?;
        engine.apply_action(proto::ApplyActionRequest {
            session_id: "stale".to_string(),
            action_id,
            expected_state_version: observation.state_version,
            observer_player: None,
        })?;
        kind_name(
            engine
                .apply_action(proto::ApplyActionRequest {
                    session_id: "stale".to_string(),
                    action_id,
                    expected_state_version: observation.state_version,
                    observer_player: None,
                })
                .expect_err("stale version must fail")
                .kind(),
        )
    };

    let end_session_counts = {
        let mut engine = LostCitiesEngine::new();
        engine.new_game(new_game_request("cleanup", 5, true))?;
        let before = engine.session_count();
        engine.end_session(session_ref("cleanup", None))?;
        engine.end_session(session_ref("cleanup", None))?;
        vec![before, engine.session_count()]
    };

    let off_turn_legal_empty = {
        let mut engine = LostCitiesEngine::new();
        let config = small_config(9);
        engine.new_game(proto::NewGameRequest {
            session_id: "hidden".to_string(),
            config: Some(config),
        })?;
        let hidden = engine.get_observation(session_ref("hidden", Some(1)))?;
        hidden
            .legal_actions
            .as_ref()
            .map(|legal| legal.actions.is_empty() && legal.mask.iter().all(|value| !value))
            .unwrap_or(false)
    };

    let (
        full_session_terminal_reward_matches,
        full_session_final_scores_match,
        terminal_reject_kind,
    ) = run_full_session_probe()?;

    let deterministic_match = run_deterministic_probe()?;

    Ok(EngineProbeOutput {
        duplicate_kind,
        missing_config_kind,
        empty_session_kind,
        unknown_session_kind,
        invalid_observer_kind,
        invalid_observer_state_unchanged,
        invalid_observer_action_still_applies,
        phase_flow,
        stale_kind,
        end_session_counts,
        off_turn_legal_empty,
        full_session_terminal_reward_matches,
        full_session_final_scores_match,
        terminal_reject_kind,
        deterministic_match,
    })
}

async fn run_grpc_probe() -> ProbeResult<GrpcProbeOutput> {
    let (mut client, server) = spawn_client().await?;
    let observation = client
        .new_game(new_game_request("grpc-round-trip", 13, true))
        .await?
        .into_inner();
    let opponent_view = client
        .get_observation(session_ref("grpc-round-trip", Some(1)))
        .await?
        .into_inner();
    let round_trip_phase = phase_name_proto(observation.phase).to_string();
    let opponent_legal_empty = opponent_view
        .legal_actions
        .map(|legal| legal.actions.is_empty())
        .unwrap_or(false);
    server.abort();

    let (mut client, server) = spawn_client().await?;
    let observation = client
        .new_game(new_game_request("grpc-stale", 21, true))
        .await?
        .into_inner();
    let action_id = first_action(&observation)?;
    client
        .apply_action(proto::ApplyActionRequest {
            session_id: "grpc-stale".to_string(),
            action_id,
            expected_state_version: observation.state_version,
            observer_player: None,
        })
        .await?;
    let stale_code = format!(
        "{:?}",
        client
            .apply_action(proto::ApplyActionRequest {
                session_id: "grpc-stale".to_string(),
                action_id,
                expected_state_version: observation.state_version,
                observer_player: None,
            })
            .await
            .expect_err("stale state_version must fail")
            .code()
    );
    server.abort();

    let (mut client, server) = spawn_client().await?;
    let observation = client
        .new_game(new_game_request("grpc-bad-observer", 31, true))
        .await?
        .into_inner();
    let action_id = first_action(&observation)?;
    let invalid_observer_code = format!(
        "{:?}",
        client
            .apply_action(proto::ApplyActionRequest {
                session_id: "grpc-bad-observer".to_string(),
                action_id,
                expected_state_version: observation.state_version,
                observer_player: Some(2),
            })
            .await
            .expect_err("invalid observer must fail")
            .code()
    );
    let after_error = client
        .get_observation(session_ref("grpc-bad-observer", Some(0)))
        .await?
        .into_inner();
    let invalid_observer_state_unchanged = after_error.state_version == observation.state_version
        && after_error.phase == observation.phase
        && after_error.current_player == observation.current_player;
    server.abort();

    let (mut client, server) = spawn_client().await?;
    client
        .new_game(new_game_request("grpc-end-session", 41, true))
        .await?;
    client
        .end_session(session_ref("grpc-end-session", None))
        .await?;
    client
        .end_session(session_ref("grpc-end-session", None))
        .await?;
    let ended_session_code = format!(
        "{:?}",
        client
            .get_observation(session_ref("grpc-end-session", None))
            .await
            .expect_err("ended session should not be readable")
            .code()
    );
    server.abort();

    Ok(GrpcProbeOutput {
        round_trip_phase,
        opponent_legal_empty,
        stale_code,
        invalid_observer_code,
        invalid_observer_state_unchanged,
        ended_session_code,
    })
}

fn run_full_session_probe() -> ProbeResult<(bool, bool, String)> {
    let mut engine = LostCitiesEngine::new();
    let mut observation = engine.new_game(new_game_request("loop", 17, true))?;

    loop {
        let action_id = first_action(&observation)?;
        let step = engine.apply_action(proto::ApplyActionRequest {
            session_id: "loop".to_string(),
            action_id,
            expected_state_version: observation.state_version,
            observer_player: None,
        })?;
        let next_observation = step
            .observation
            .ok_or_else(|| io::Error::other("missing loop observation"))?;

        if step.terminal {
            let observer = next_observation.observer_player;
            let other = 1 - observer;
            let reward_matches = step.reward as i32 == next_observation.score_diff;
            let scores_match = step.final_scores.len() == 2
                && step.final_scores.get(&observer).copied() == Some(next_observation.my_score)
                && step.final_scores.get(&other).copied() == Some(next_observation.opponent_score);
            let terminal_reject_kind = kind_name(
                engine
                    .apply_action(proto::ApplyActionRequest {
                        session_id: "loop".to_string(),
                        action_id: 0,
                        expected_state_version: next_observation.state_version,
                        observer_player: None,
                    })
                    .expect_err("terminal game must reject further actions")
                    .kind(),
            );
            return Ok((reward_matches, scores_match, terminal_reject_kind));
        }

        observation = next_observation;
    }
}

fn run_deterministic_probe() -> ProbeResult<bool> {
    let config = small_config(29);
    let mut left = LostCitiesEngine::new();
    let mut right = LostCitiesEngine::new();
    let mut left_observation = left.new_game(proto::NewGameRequest {
        session_id: "det-left".to_string(),
        config: Some(config.clone()),
    })?;
    let mut right_observation = right.new_game(proto::NewGameRequest {
        session_id: "det-right".to_string(),
        config: Some(config),
    })?;

    loop {
        if observation_summary(&left_observation) != observation_summary(&right_observation) {
            return Ok(false);
        }
        if left_observation.terminal {
            return Ok(right_observation.terminal);
        }

        let action_id = first_action(&left_observation)?;
        let left_step = left.apply_action(proto::ApplyActionRequest {
            session_id: "det-left".to_string(),
            action_id,
            expected_state_version: left_observation.state_version,
            observer_player: None,
        })?;
        let right_step = right.apply_action(proto::ApplyActionRequest {
            session_id: "det-right".to_string(),
            action_id,
            expected_state_version: right_observation.state_version,
            observer_player: None,
        })?;
        if left_step.terminal != right_step.terminal
            || left_step.final_scores != right_step.final_scores
            || left_step.reward != right_step.reward
        {
            return Ok(false);
        }
        left_observation = left_step
            .observation
            .ok_or_else(|| io::Error::other("missing left observation"))?;
        right_observation = right_step
            .observation
            .ok_or_else(|| io::Error::other("missing right observation"))?;
    }
}

async fn spawn_client() -> ProbeResult<(LostCitiesClient<Channel>, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let incoming = TcpListenerStream::new(listener);

    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(LostCitiesServer::new(LostCitiesGrpcService::default()))
            .serve_with_incoming(incoming)
            .await
            .expect("gRPC server should run");
    });

    let endpoint = Endpoint::from_shared(format!("http://{}", addr))?;
    let client = LostCitiesClient::new(endpoint.connect().await?);
    Ok((client, server))
}

fn observation_summary(observation: &proto::GameObservation) -> ObservationSummary {
    ObservationSummary {
        state_version: observation.state_version,
        current_player: observation.current_player,
        observer_player: observation.observer_player,
        phase: phase_name_proto(observation.phase).to_string(),
        hand: observation.hand.iter().map(CardJson::from_proto).collect(),
        opponent_hand_size: observation.opponent_hand_size,
        deck_size: observation.deck_size,
        discards: observation
            .discards
            .iter()
            .map(|discard| discard.cards.iter().map(CardJson::from_proto).collect())
            .collect(),
        my_expeditions: observation
            .my_expeditions
            .iter()
            .map(|expedition| expedition.cards.iter().map(CardJson::from_proto).collect())
            .collect(),
        opponent_expeditions: observation
            .opponent_expeditions
            .iter()
            .map(|expedition| expedition.cards.iter().map(CardJson::from_proto).collect())
            .collect(),
        legal_mask: observation
            .legal_actions
            .as_ref()
            .map(|legal| legal.mask.clone())
            .unwrap_or_default(),
        terminal: observation.terminal,
        my_score: observation.my_score,
        opponent_score: observation.opponent_score,
        score_diff: observation.score_diff,
    }
}

fn first_action(observation: &proto::GameObservation) -> ProbeResult<u32> {
    observation
        .legal_actions
        .as_ref()
        .and_then(|actions| actions.actions.first())
        .map(|action| action.id)
        .ok_or_else(|| io::Error::other("observation has no legal action").into())
}

fn small_config(seed: u64) -> proto::GameConfig {
    proto::GameConfig {
        n_colors: 2,
        n_ranks: 2,
        min_rank: 1,
        n_handshakes: 0,
        hand_size: 1,
        expedition_penalty: 0,
        bonus_threshold: 99,
        bonus_amount: 0,
        seed: Some(seed),
    }
}

fn new_game_request(session_id: &str, seed: u64, include_config: bool) -> proto::NewGameRequest {
    proto::NewGameRequest {
        session_id: session_id.to_string(),
        config: include_config.then(|| small_config(seed)),
    }
}

fn session_ref(session_id: &str, observer_player: Option<u32>) -> proto::SessionRef {
    proto::SessionRef {
        session_id: session_id.to_string(),
        observer_player,
    }
}

fn kind_name(kind: EngineErrorKind) -> String {
    match kind {
        EngineErrorKind::AlreadyExists => "AlreadyExists",
        EngineErrorKind::NotFound => "NotFound",
        EngineErrorKind::FailedPrecondition => "FailedPrecondition",
        EngineErrorKind::InvalidArgument => "InvalidArgument",
    }
    .to_string()
}

fn phase_name_state(phase: Phase) -> &'static str {
    match phase {
        Phase::Card => "card",
        Phase::Draw => "draw",
    }
}

fn phase_name_proto(phase: i32) -> &'static str {
    match proto::Phase::try_from(phase) {
        Ok(proto::Phase::Card) => "card",
        Ok(proto::Phase::Draw) => "draw",
        _ => "unspecified",
    }
}

impl ConfigJson {
    fn to_config(&self) -> Config {
        Config {
            n_colors: self.n_colors,
            n_ranks: self.n_ranks,
            min_rank: self.min_rank,
            n_handshakes: self.n_handshakes,
            hand_size: self.hand_size,
            expedition_penalty: self.expedition_penalty,
            bonus_threshold: self.bonus_threshold,
            bonus_amount: self.bonus_amount,
            seed: self.seed,
        }
    }

    fn from_config(config: &Config) -> Self {
        Self {
            n_colors: config.n_colors,
            n_ranks: config.n_ranks,
            min_rank: config.min_rank,
            n_handshakes: config.n_handshakes,
            hand_size: config.hand_size,
            expedition_penalty: config.expedition_penalty,
            bonus_threshold: config.bonus_threshold,
            bonus_amount: config.bonus_amount,
            seed: config.seed,
        }
    }
}

impl From<CardJson> for Card {
    fn from(value: CardJson) -> Self {
        Self {
            color: value.color,
            rank: value.rank,
        }
    }
}

impl From<Card> for CardJson {
    fn from(value: Card) -> Self {
        Self {
            color: value.color,
            rank: value.rank,
        }
    }
}

impl CardJson {
    fn from_proto(value: &proto::Card) -> Self {
        Self {
            color: value.color,
            rank: value.rank,
        }
    }
}

impl StateTraceStep {
    fn from_state(action: Option<u32>, state: &GameState) -> Self {
        Self {
            action,
            phase: phase_name_state(state.phase),
            current_player: state.current_player,
            turn_count: state.turn_count,
            terminal: state.terminal,
            pending_discarded_color: state.pending_discarded_color,
            score_diff_player0: state.score_diff(0),
            legal_mask: state.legal_unified_mask(),
            deck: state.deck.iter().copied().map(CardJson::from).collect(),
            hands: state
                .hands
                .iter()
                .map(|hand| hand.iter().copied().map(CardJson::from).collect())
                .collect(),
            expeditions: state
                .expeditions
                .iter()
                .map(|expeditions| {
                    expeditions
                        .iter()
                        .map(|expedition| expedition.iter().copied().map(CardJson::from).collect())
                        .collect()
                })
                .collect(),
            discards: state
                .discards
                .iter()
                .map(|discard| discard.iter().copied().map(CardJson::from).collect())
                .collect(),
        }
    }
}

impl ObservationMini {
    fn from_observation(observation: &proto::GameObservation) -> Self {
        Self {
            state_version: observation.state_version,
            current_player: observation.current_player,
            observer_player: observation.observer_player,
            phase: phase_name_proto(observation.phase).to_string(),
            terminal: observation.terminal,
        }
    }
}
