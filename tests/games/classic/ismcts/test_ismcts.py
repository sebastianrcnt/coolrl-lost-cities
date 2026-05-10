from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state, input_dim
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.bots.heuristic import HeuristicBot
from coolrl_lost_cities.games.classic.bots.heuristic_py import (
    HeuristicBot as PythonHeuristicBot,
)
from coolrl_lost_cities.games.classic.ismcts.config import IsMctsConfig, MctsConfig
from coolrl_lost_cities.games.classic.ismcts.determinization import sample_determinization
from coolrl_lost_cities.games.classic.ismcts.info_set import canonical_info_set_key
from coolrl_lost_cities.games.classic.ismcts.interleaved_self_play import (
    play_self_play_iteration,
)
from coolrl_lost_cities.games.classic.ismcts.mcts import IsMctsSearcher, MctsNode
from coolrl_lost_cities.games.classic.ismcts.network import AlphaZeroLogitsView, AlphaZeroNet
from coolrl_lost_cities.games.classic.ismcts.replay_buffer import ReplayBuffer, ReplaySample
from coolrl_lost_cities.games.classic.ismcts.self_play import play_self_play_game
from coolrl_lost_cities.games.classic.ismcts.trainer import IsMctsTrainer


def _python_mcts_searcher():
    module_name = "coolrl_lost_cities.games.classic.ismcts._mcts_python_baseline"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing.IsMctsSearcher
    path = (
        Path(__file__).parents[4]
        / "src"
        / "coolrl_lost_cities"
        / "games"
        / "classic"
        / "ismcts"
        / "mcts.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.IsMctsSearcher


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


def test_logits_view_adapter() -> None:
    state = GameState.new_game(mini_config(), seed=6)
    dim = input_dim(state)
    net = AlphaZeroNet(dim, state.action_size, hidden_size=16, num_layers=1)
    logits_view = AlphaZeroLogitsView(net)
    x = torch.as_tensor(encode_info_state(state, 0)[None, :], dtype=torch.float32)
    logits = logits_view(x)
    assert logits.shape == (1, state.action_size)


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


def test_search_correctness_vs_sequential() -> None:
    for n_sims in (8, 16, 64):
        state = GameState.new_game(mini_config(), seed=17)
        dim = input_dim(state)
        net = AlphaZeroNet(dim, state.action_size, hidden_size=8, num_layers=1)
        config = MctsConfig(
            n_simulations=n_sims,
            parallel_simulations=1,
            use_rollout_value=False,
        )
        left = IsMctsSearcher(net, config, rng=random.Random(18))
        right = IsMctsSearcher(net, config, rng=random.Random(18))
        assert left.search(state, state.current_player) == right.search(state, state.current_player)


def test_cython_sequential_matches_python_sequential_visit_counts() -> None:
    PythonIsMctsSearcher = _python_mcts_searcher()
    for n_sims in (8, 32, 128):
        state = GameState.new_game(mini_config(), seed=23)
        dim = input_dim(state)
        torch.manual_seed(24)
        net = AlphaZeroNet(dim, state.action_size, hidden_size=8, num_layers=1)
        config = MctsConfig(
            n_simulations=n_sims,
            parallel_simulations=1,
            use_rollout_value=False,
        )
        python_searcher = PythonIsMctsSearcher(net, config, rng=random.Random(25))
        cython_searcher = IsMctsSearcher(net, config, rng=random.Random(25))

        assert cython_searcher.search(state, state.current_player) == python_searcher.search(
            state, state.current_player
        )


def test_search_visit_counts_match_with_parallel_simulations() -> None:
    for n_sims in (8, 32, 128):
        state = GameState.new_game(mini_config(), seed=26)
        dim = input_dim(state)
        torch.manual_seed(27)
        net = AlphaZeroNet(dim, state.action_size, hidden_size=8, num_layers=1)
        sequential = IsMctsSearcher(
            net,
            MctsConfig(n_simulations=n_sims, parallel_simulations=1, use_rollout_value=False),
            rng=random.Random(28),
        )
        batched = IsMctsSearcher(
            net,
            MctsConfig(n_simulations=n_sims, parallel_simulations=8, use_rollout_value=False),
            rng=random.Random(28),
        )

        assert batched.search(state, state.current_player) == sequential.search(
            state, state.current_player
        )


def test_search_with_virtual_loss_diversity() -> None:
    state = GameState.new_game(mini_config(), seed=19)
    dim = input_dim(state)
    net = AlphaZeroNet(dim, state.action_size, hidden_size=8, num_layers=0)
    for param in net.parameters():
        param.data.zero_()
    searcher = IsMctsSearcher(
        net,
        MctsConfig(n_simulations=64, parallel_simulations=4, virtual_loss_value=1.0),
        rng=random.Random(20),
    )
    first = searcher.prepare_simulation_batch(state, state.current_player, 1)
    searcher.evaluate_and_backup(first)

    pending = searcher.prepare_simulation_batch(state, state.current_player, 4)
    first_actions = [item.path[0].action for item in pending if item.path]
    assert len(set(first_actions)) >= 2


def test_heuristic_cython_fast_path_matches_python_for_random_states() -> None:
    configs = [mini_config(seed=31), LostCitiesConfig(seed=32)]
    py_bot = PythonHeuristicBot()
    cy_bot = HeuristicBot()

    for config in configs:
        rng = random.Random(33)
        checked = 0
        attempts = 0
        while checked < 100 and attempts < 1000:
            attempts += 1
            state = GameState.new_game(config, seed=rng.randrange(2**31))
            for _ in range(rng.randrange(40)):
                if state.terminal:
                    break
                legal = state.unified_legal_actions()
                if not legal:
                    break
                state.apply_unified_action(rng.choice(legal))
            if state.terminal or not state.unified_legal_actions():
                continue

            assert cy_bot.act_cython(state) == py_bot.act(state)
            checked += 1

        assert checked == 100


def test_game_state_push_pop_unified_round_trip_snapshot() -> None:
    rng = random.Random(34)
    for config in (mini_config(seed=35), LostCitiesConfig(seed=36)):
        state = GameState.new_game(config, seed=37)
        for _ in range(100):
            if state.terminal:
                break
            before = state.to_snapshot()
            unified = rng.choice(state.unified_legal_actions())
            local = state.from_unified_action(unified)
            state.push_action(local)
            state.pop_action()
            assert state.to_snapshot() == before
            state.apply_unified_action(unified)


def test_mcts_node_c_array_maps_are_dict_like() -> None:
    node = MctsNode(b"root", player=0, action_size=16)
    node.priors[3] = 0.25
    node.visits.setdefault(3, 0)
    node.value_sum[3] = 1.5
    node.virtual_visits[3] = 2
    node.visits[3] = node.visits.get(3, 0) + 4

    assert bool(node.priors)
    assert node.priors.get(3, 0.0) == 0.25
    assert node.visits.get(3, 0) == 4
    assert node.value_sum[3] == 1.5
    assert node.virtual_visits[3] == 2
    assert 3 in node.visits
    assert dict(node.visits.items()) == {3: 4}


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
    assert all(sample.prior is not None for sample in samples)


def test_interleaved_self_play_yields_complete_games() -> None:
    config = mini_config()
    state = GameState.new_game(config, seed=21)
    net = AlphaZeroNet(input_dim(state), state.action_size, hidden_size=8, num_layers=1)
    ismcts_config = IsMctsConfig.model_validate(
        {
            "mcts": {"n_simulations": 2, "parallel_simulations": 2},
            "training": {"games_per_iter": 4, "interleave_games": 4, "interleave_max_batch": 16},
        }
    )
    samples = play_self_play_iteration(
        net,
        ismcts_config.mcts,
        ismcts_config.training,
        config,
        random.Random(22),
        max_steps=80,
    )
    assert samples
    assert {sample.game_index for sample in samples} == {0, 1, 2, 3}
    assert all(np.isfinite(sample.v_target) for sample in samples)


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


def test_trainer_emits_full_eval_metrics(tmp_path) -> None:
    config = IsMctsConfig.model_validate(
        {
            "run": {"max_iterations": 1, "seed": 12, "device": "cpu"},
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
            "evaluation": {
                "eval_every": 1,
                "games": 2,
                "opponents": ["random"],
                "num_workers": 1,
                "max_steps": 80,
            },
        }
    )
    trainer = IsMctsTrainer(
        config,
        config.rules.to_lost_cities_config(seed=config.run.seed),
        run_dir=tmp_path,
    )
    metrics = trainer.train()[0].to_dict()
    assert "eval/random/avg_score_diff0" in metrics
    assert "eval/random/play_action_rate" in metrics
    assert "eval/random/win_rate0" in metrics


def test_trainer_emits_mcts_metrics(tmp_path) -> None:
    config = IsMctsConfig.model_validate(
        {
            "run": {"max_iterations": 1, "seed": 13, "device": "cpu"},
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
    metrics = trainer.train()[0].to_dict()
    for key in (
        "mcts/avg_visit_entropy",
        "mcts/value_prediction_error",
        "mcts/policy_mcts_kl",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_smoke_iter_with_batching(tmp_path) -> None:
    config = IsMctsConfig.model_validate(
        {
            "run": {"max_iterations": 1, "seed": 14, "device": "cpu"},
            "rules": {
                "n_colors": 3,
                "n_ranks": 5,
                "n_handshakes": 1,
                "hand_size": 4,
                "bonus_threshold": 4,
            },
            "network": {"hidden_size": 16, "num_layers": 1},
            "mcts": {"n_simulations": 4, "parallel_simulations": 4},
            "training": {
                "games_per_iter": 1,
                "gradient_steps_per_iter": 1,
                "batch_size": 8,
                "interleave_games": 4,
                "interleave_max_batch": 16,
            },
            "checkpoint": {"save_every": 0},
            "evaluation": {"eval_every": 0, "num_workers": 1, "max_steps": 80},
        }
    )
    trainer = IsMctsTrainer(
        config,
        config.rules.to_lost_cities_config(seed=config.run.seed),
        run_dir=tmp_path,
    )
    metrics = trainer.train()[0].to_dict()
    assert metrics["samples/added"] > 0
    assert "mcts/avg_visit_entropy" in metrics
    assert "mcts/value_prediction_error" in metrics
    assert "mcts/policy_mcts_kl" in metrics
