import pytest

from coolrl_lost_cities.games.classic import (
    LostCitiesConfig,
    ReferenceLostCitiesCard,
    ReferenceLostCitiesState,
)


def test_reference_card_is_plain_ordered_value() -> None:
    assert ReferenceLostCitiesCard(0, 1) < ReferenceLostCitiesCard(1, 0)


def test_reference_state_placeholder_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError):
        ReferenceLostCitiesState.new_game(LostCitiesConfig())
