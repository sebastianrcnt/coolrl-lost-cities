# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Core Lost Cities classic types.

``GameState`` is provided by the C-array fast engine.
"""

from dataclasses import dataclass, fields
from typing import Any, Literal

cimport cython


Phase = Literal["card", "draw"]


class IllegalMoveError(ValueError):
    """Raised when an action id is not legal for the current state."""


@cython.freelist(256)
cdef class Card:
    cdef readonly int color
    cdef readonly int rank

    def __cinit__(self, color, rank):
        self.color = int(color)
        self.rank = int(rank)

    @property
    def is_handshake(self):
        return self.rank == 0

    cpdef int numeric_value(self, int min_rank):
        if self.rank == 0:
            return 0
        return min_rank + self.rank - 1

    def label(self, int min_rank):
        if self.rank == 0:
            return f"[{self.color}]H"
        return f"[{self.color}]{self.numeric_value(min_rank)}"

    def to_snapshot(self):
        return {"color": self.color, "rank": self.rank}

    @classmethod
    def from_snapshot(cls, data):
        if isinstance(data, Card):
            return data
        if isinstance(data, dict):
            return cls(int(data["color"]), int(data["rank"]))
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return cls(int(data[0]), int(data[1]))
        raise ValueError(f"invalid card snapshot: {data!r}")

    def __hash__(self):
        return (self.color << 8) | self.rank

    def __richcmp__(self, other, int op):
        if not isinstance(other, Card):
            return NotImplemented
        cdef Card o = <Card>other
        cdef bint eq = self.color == o.color and self.rank == o.rank
        if op == 2:  # ==
            return eq
        if op == 3:  # !=
            return not eq
        cdef bint lt
        if self.color != o.color:
            lt = self.color < o.color
        else:
            lt = self.rank < o.rank
        if op == 0:  # <
            return lt
        if op == 1:  # <=
            return lt or eq
        if op == 4:  # >
            return not lt and not eq
        if op == 5:  # >=
            return not lt
        return NotImplemented

    def __repr__(self):
        return f"Card(color={self.color}, rank={self.rank})"

    def __reduce__(self):
        return (Card, (self.color, self.rank))


@dataclass(frozen=True)
class LostCitiesConfig:
    n_colors: int = 5
    n_ranks: int = 9
    min_rank: int = 2
    n_handshakes: int = 3
    hand_size: int = 8
    expedition_penalty: int = -20
    bonus_threshold: int = 8
    bonus_amount: int = 20
    seed: int | None = None

    @property
    def deck_size(self) -> int:
        return self.n_colors * (self.n_ranks + self.n_handshakes)

    @property
    def max_rank(self) -> int:
        return self.min_rank + self.n_ranks - 1

    @property
    def card_action_size(self) -> int:
        return 2 * self.hand_size

    @property
    def draw_action_size(self) -> int:
        return 1 + self.n_colors

    @property
    def action_size(self) -> int:
        return self.card_action_size + self.draw_action_size

    def validate(self) -> None:
        if self.n_colors <= 0:
            raise ValueError("n_colors must be positive")
        if self.n_ranks <= 0:
            raise ValueError("n_ranks must be positive")
        if self.min_rank <= 0:
            raise ValueError("min_rank must be positive")
        if self.n_handshakes < 0:
            raise ValueError("n_handshakes cannot be negative")
        if self.hand_size <= 0:
            raise ValueError("hand_size must be positive")
        if self.deck_size < 2 * self.hand_size:
            raise ValueError("deck must contain at least both initial hands")
        if self.bonus_threshold <= 0:
            raise ValueError("bonus_threshold must be positive")

    def to_snapshot(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def classic_config(*, seed=None):
    return LostCitiesConfig(seed=seed)


def config_from_mapping(data):
    allowed = LostCitiesConfig.__dataclass_fields__.keys()
    kwargs = {key: value for key, value in data.items() if key in allowed}
    config = LostCitiesConfig(**kwargs)
    config.validate()
    return config


def config_to_mapping(config):
    return config.to_snapshot()


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to load Lost Cities YAML configs") from exc

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in config file: {path}")
    return config_from_mapping(data)


def build_deck(config):
    config.validate()
    cdef list deck = []
    cdef int color, rank
    cdef int n_colors = config.n_colors
    cdef int n_handshakes = config.n_handshakes
    cdef int n_ranks = config.n_ranks
    for color in range(n_colors):
        for _ in range(n_handshakes):
            deck.append(Card(color, 0))
        for rank in range(1, n_ranks + 1):
            deck.append(Card(color, rank))
    return deck


cpdef int score_expedition(list expedition, config):
    cdef int n = len(expedition)
    if n == 0:
        return 0
    cdef int min_rank = config.min_rank
    cdef int handshakes = 0
    cdef int numeric_sum = 0
    cdef int i
    cdef Card card
    for i in range(n):
        card = <Card>expedition[i]
        if card.rank == 0:
            handshakes += 1
        else:
            numeric_sum += min_rank + card.rank - 1
    cdef int score = (numeric_sum + config.expedition_penalty) * (handshakes + 1)
    if n >= config.bonus_threshold:
        score += config.bonus_amount
    return score


from .engines.fast import FastGameState as GameState
