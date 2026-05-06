from __future__ import annotations

import numpy as np
import pytest
from coolrl_lost_cities.games.classic.deep_cfr.cfr_math import (
    normalize_legal_policy,
    regret_matching,
    sample_policy,
)
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state, input_dim
from coolrl_lost_cities.games.classic.deep_cfr.traversal import (
    random_rollout_value,
    root_action_values,
)
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig


def test_swap_deck_cards_swaps_internal_deck_order_and_validates_bounds() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=7))
    before = state.to_snapshot()
    first = before["deck"][0]
    last = before["deck"][-1]

    state.swap_deck_cards(0, len(before["deck"]) - 1)

    after = state.to_snapshot()
    assert after["deck"][0] == last
    assert after["deck"][-1] == first
    state.validate_invariants()

    state.swap_deck_cards(0, len(before["deck"]) - 1)
    assert state.to_snapshot() == before

    with pytest.raises(IndexError, match="deck index out of range"):
        state.swap_deck_cards(-1, 0)
    with pytest.raises(IndexError, match="deck index out of range"):
        state.swap_deck_cards(0, len(before["deck"]))


def test_regret_matching_uses_positive_legal_regrets() -> None:
    policy = regret_matching(
        np.asarray([1.0, -2.0, 3.0, 5.0], dtype=np.float32),
        np.asarray([True, True, False, True]),
    )

    np.testing.assert_allclose(policy, [1.0 / 6.0, 0.0, 0.0, 5.0 / 6.0])


def test_regret_matching_falls_back_to_uniform_legal_policy() -> None:
    policy = regret_matching(
        np.asarray([-1.0, 0.0, 3.0, 5.0], dtype=np.float32),
        np.asarray([True, True, False, False]),
    )
    no_legal = regret_matching(
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([False, False]),
    )

    np.testing.assert_allclose(policy, [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(no_legal, [0.0, 0.0])


def test_normalize_legal_policy_clamps_and_normalizes_legal_weights() -> None:
    policy = normalize_legal_policy(
        np.asarray([2.0, -1.0, 4.0, 6.0], dtype=np.float32),
        np.asarray([True, True, False, True]),
    )
    fallback = normalize_legal_policy(
        np.asarray([0.0, -1.0, 3.0], dtype=np.float32),
        np.asarray([True, True, False]),
    )

    np.testing.assert_allclose(policy, [0.25, 0.0, 0.0, 0.75])
    np.testing.assert_allclose(fallback, [0.5, 0.5, 0.0])


def test_sample_policy_uses_cumulative_probability_boundaries() -> None:
    policy = np.asarray([0.2, 0.3, 0.5], dtype=np.float32)

    assert sample_policy(policy, 0.0) == 0
    assert sample_policy(policy, 0.21) == 1
    assert sample_policy(policy, 0.51) == 2
    assert sample_policy(policy, 0.999) == 2


def test_encode_info_state_is_deterministic_and_matches_legal_mask_tail() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=13))

    encoded = encode_info_state(state, 0)
    encoded_again = encode_info_state(state, 0)
    legal_mask = np.asarray(state.unified_legal_mask(), dtype=np.float32)

    assert encoded.dtype == np.float32
    assert encoded.shape == (input_dim(state),)
    np.testing.assert_array_equal(encoded, encoded_again)
    np.testing.assert_array_equal(encoded[-len(legal_mask) :], legal_mask)

    with pytest.raises(ValueError, match="invalid player"):
        encode_info_state(state, 2)


def test_random_rollout_value_restores_state_and_is_deterministic() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=17))
    before = state.to_snapshot()

    value = random_rollout_value(state, 0, seed=101, max_steps=128)
    same_value = random_rollout_value(state, 0, seed=101, max_steps=128)

    assert value == same_value
    assert state.to_snapshot() == before


def test_root_action_values_restores_state_and_marks_legal_actions() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=19))
    before = state.to_snapshot()

    values, legal = root_action_values(state, 0, seed=202, rollouts_per_action=1, max_steps=128)

    assert values.dtype == np.float32
    assert legal.dtype == np.uint8
    assert values.shape == legal.shape
    np.testing.assert_array_equal(legal.astype(bool), state.unified_legal_mask())
    assert state.to_snapshot() == before
