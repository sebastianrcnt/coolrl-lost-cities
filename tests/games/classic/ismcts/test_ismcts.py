from __future__ import annotations

import random

import numpy as np
import torch
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state, input_dim
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.ismcts.config import IsMctsConfig, MctsConfig
from coolrl_lost_cities.games.classic.ismcts.determinization import sample_determinization
from coolrl_lost_cities.games.classic.ismcts.info_set import canonical_info_set_key
from coolrl_lost_cities.games.classic.ismcts.mcts import IsMctsSearcher
from coolrl_lost_cities.games.classic.ismcts.network import AlphaZeroNet
from coolrl_lost_cities.games.classic.ismcts.replay_buffer import ReplayBuffer, ReplaySample
from coolrl_lost_cities.games.classic.ismcts.self_play import play_self_play_game
from coolrl_lost_cities.games.classic.ismcts.trainer import IsMctsTrainer


def mini_config(seed: int = 1) -> LostCitiesConfig:
    return LostCitiesConfig(
        n_colors=3,
        n_ranks=5,
        n_handshakes=1,
        hand_size=4,
        bonus_threshold=4,
        seed=seed,
    )


def test_canonical_key_ignores_opponent_hand_identity() -> None:
    config = mini_config()
    state = GameState.new_game(config, seed=3)
    snap = state.to_snapshot()
    snap["hands"][1] = list(reversed(snap["hands"][1]))
    other = GameState.from_snapshot(snap)
    assert canonical_info_set_key(state, 0) == canonical_info_set_key(other, 0)


def test_determinization_consistent_with_info_set() -> None:
    config = mini_config()
    state = GameState.new_game(config, seed=4)
    rng = random.Random(5)
    key = canonical_info_set_key(state, 0)
    for _ in range(100):
        det = sample_determinization(state, 0, rng)
        assert canonical_info_set_key(det, 0) == key
        assert len(det.hands[1]) == config.hand_size
        all_cards = det.hands[0] + det.hands[1] + det.deck
        for player_expeditions in det.expeditions:
            for expedition in player_expeditions:
                all_cards.extend(expedition)
        for discard in det.discards:
            all_cards.extend(discard)
        assert len(all_cards) == config.deck_size


def test_network_shapes_and_mask() -> None:
    state = GameState.new_game(mini_config(), seed=6)
    dim = input_dim(state)
    net = AlphaZeroNet(dim, state.action_size, hidden_size=16, num_layers=1)
    x = torch.as_tensor(encode_info_state(state, 0)[None, :], dtype=torch.float32)
    mask = torch.as_tensor(np.asarray(state.unified_legal_mask(), dtype=bool)[None, :])
    logits, value = net(x, mask)
    probs = net.policy_distribution(x, mask)
    assert logits.shape == (1, state.action_size)
    assert value.shape == (1,)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
    assert torch.all(probs[~mask] == 0)


def test_mcts_prior_drives_visits() -> None:
    state = GameState.new_game(mini_config(), seed=7)
    dim = input_dim(state)
    net = AlphaZeroNet(dim, state.action_size, hidden_size=8, num_layers=0)
    for param in net.parameters():
        param.data.zero_()
    legal = state.unified_legal_actions()
    favored = legal[0]
    net.policy_head.bias.data[favored] = 5.0
    searcher = IsMctsSearcher(
        net,
        MctsConfig(n_simulations=12, c_puct=2.0, use_rollout_value=False),
        rng=random.Random(8),
    )
    visits = searcher.search(state, state.current_player)
    assert visits[favored] == max(visits.values())


def test_replay_buffer_capacity_and_sample() -> None:
    sample = ReplaySample(
        info_state=np.zeros(4, dtype=np.float32),
        legal_mask=np.ones(3, dtype=bool),
        pi_target=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        v_target=1.0,
        player=0,
    )
    buffer = ReplayBuffer(2, seed=1)
    buffer.add([sample, sample, sample])
    assert len(buffer) == 2
    assert len(buffer.sample(2)) == 2


def test_self_play_game_returns_signed_targets() -> None:
    config = mini_config()
    state = GameState.new_game(config, seed=9)
    net = AlphaZeroNet(input_dim(state), state.action_size, hidden_size=8, num_layers=1)
    samples = play_self_play_game(
        net,
        MctsConfig(n_simulations=2),
        config,
        random.Random(10),
    )
    assert samples
    assert {sample.player for sample in samples} <= {0, 1}
    assert all(sample.pi_target.sum() > 0 for sample in samples)


def test_trainer_one_iteration_smoke(tmp_path) -> None:
    config = IsMctsConfig.model_validate(
        {
            "run": {"max_iterations": 1, "seed": 11, "device": "cpu"},
            "rules": {
                "n_colors": 3,
                "n_ranks": 5,
                "n_handshakes": 1,
                "hand_size": 4,
                "bonus_threshold": 4,
            },
            "network": {"hidden_size": 16, "num_layers": 1},
            "mcts": {"n_simulations": 2},
            "training": {"games_per_iter": 1, "gradient_steps_per_iter": 1, "batch_size": 8},
            "checkpoint": {"save_every": 0},
            "evaluation": {"eval_every": 0, "num_workers": 1, "max_steps": 80},
        }
    )
    trainer = IsMctsTrainer(
        config,
        config.rules.to_lost_cities_config(seed=config.run.seed),
        run_dir=tmp_path,
    )
    metrics = trainer.train()
    assert len(metrics) == 1
    assert (tmp_path / "metrics.jsonl").exists()
