from __future__ import annotations

from collections.abc import Callable

from ..interfaces import LostCitiesBot
from .heuristic import SafeHeuristicBot
from .passive import PassiveDiscardBot
from .random import RandomBot

BotName = str
DEFAULT_BOT: BotName = "random"
BotFactory = Callable[[int | None], LostCitiesBot]

BOT_REGISTRY: dict[BotName, BotFactory] = {
    DEFAULT_BOT: RandomBot,
    "passive-discard": lambda seed: PassiveDiscardBot(),
    "safe-heuristic": lambda seed: SafeHeuristicBot(),
}


def available_bot_names() -> list[BotName]:
    return sorted(BOT_REGISTRY)


def build_bot(name: BotName, *, seed: int | None = None) -> LostCitiesBot:
    try:
        bot_factory = BOT_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown Lost Cities bot: {name}") from exc
    return bot_factory(seed)
