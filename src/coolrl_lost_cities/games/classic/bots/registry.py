from __future__ import annotations

from collections.abc import Callable

from ..policy import LostCitiesPolicy
from .heuristic import SafeHeuristicBot
from .passive import PassiveDiscardBot
from .random import RandomBot

BotName = str
DEFAULT_BOT: BotName = "random"
PolicyFactory = Callable[[int | None], LostCitiesPolicy]

BOT_REGISTRY: dict[BotName, PolicyFactory] = {
    DEFAULT_BOT: RandomBot,
    "passive-discard": lambda seed: PassiveDiscardBot(),
    "safe-heuristic": lambda seed: SafeHeuristicBot(),
}


def available_bot_names() -> list[BotName]:
    return sorted(BOT_REGISTRY)


def build_bot(name: BotName, *, seed: int | None = None) -> LostCitiesPolicy:
    try:
        policy_factory = BOT_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown Lost Cities bot: {name}") from exc
    return policy_factory(seed)
