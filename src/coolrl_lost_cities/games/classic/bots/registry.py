from __future__ import annotations

from collections.abc import Callable

from ..policy import LostCitiesPolicy, PolicyInput
from .heuristic import SafeHeuristicBot, SafeHeuristicParams
from .passive import PassiveDiscardBot
from .random import RandomBot

BotName = str
DEFAULT_BOT: BotName = "random"
PolicyFactory = Callable[[int | None], LostCitiesPolicy]


class NoisyPolicy(LostCitiesPolicy):
    def __init__(
        self,
        base: LostCitiesPolicy,
        random_policy: RandomBot,
        *,
        epsilon: float = 0.15,
    ):
        self.base = base
        self.random_policy = random_policy
        self.epsilon = epsilon

    def act(self, obs_or_state: PolicyInput) -> int:
        if self.random_policy.rng.random() < self.epsilon:
            return self.random_policy.act(obs_or_state)
        return self.base.act(obs_or_state)


LOOSE_SAFE_HEURISTIC_PARAMS = SafeHeuristicParams(
    open_target_ratio=0.42,
    open_min_card_ratio=0.30,
    handshake_target_multiplier=1.00,
    handshake_min_card_ratio=0.25,
    late_open_block_ratio=0.12,
)

STRICT_SAFE_HEURISTIC_PARAMS = SafeHeuristicParams(
    open_target_ratio=0.62,
    open_min_card_ratio=0.50,
    handshake_target_multiplier=1.35,
    handshake_min_card_ratio=0.45,
    late_open_block_ratio=0.30,
)


BOT_REGISTRY: dict[BotName, PolicyFactory] = {
    DEFAULT_BOT: RandomBot,
    "passive-discard": lambda seed: PassiveDiscardBot(),
    "safe-heuristic": lambda seed: SafeHeuristicBot(),
    "safe-heuristic-loose": lambda seed: SafeHeuristicBot(LOOSE_SAFE_HEURISTIC_PARAMS),
    "safe-heuristic-strict": lambda seed: SafeHeuristicBot(STRICT_SAFE_HEURISTIC_PARAMS),
    "noisy-safe": lambda seed: NoisyPolicy(
        SafeHeuristicBot(),
        RandomBot(seed),
    ),
}


def canonical_bot_name(name: BotName) -> BotName:
    return name.strip().lower().replace("_", "-")


def available_bot_names() -> list[BotName]:
    return sorted(BOT_REGISTRY)


def build_bot(name: BotName, *, seed: int | None = None) -> LostCitiesPolicy:
    canonical = canonical_bot_name(name)
    try:
        policy_factory = BOT_REGISTRY[canonical]
    except KeyError as exc:
        raise ValueError(f"unknown Lost Cities bot: {name}") from exc
    return policy_factory(seed)
