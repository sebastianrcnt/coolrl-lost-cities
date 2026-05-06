from __future__ import annotations

import json
import logging
import random
import subprocess
import tempfile
from pathlib import Path

from ..game import Card, LostCitiesConfig, build_deck
from ..interfaces import BackendName, Snapshot
from .common import snapshot_from_trace, snapshot_summary

LOGGER = logging.getLogger("coolrl_lost_cities.games.classic.backends.rust")
RUST_CORE_DIR = Path(__file__).resolve().parents[5] / "rust" / "lost-cities-core"


class RustLostCitiesBackend:
    name: BackendName = "rust"

    def __init__(self, config: LostCitiesConfig, seed: int | None):
        self.config = config
        self.seed = seed
        self.initial_deck = _shuffled_deck(config, seed)
        self.actions: list[int] = []
        self._snapshot = self._run_trace()
        LOGGER.debug("러스트 백엔드 초기화: %s", snapshot_summary(self.snapshot()))

    def snapshot(self) -> Snapshot:
        return self._snapshot

    def apply(self, action_id: int) -> None:
        before = self.snapshot()
        self.actions.append(action_id)
        try:
            self._snapshot = self._run_trace()
        except Exception:
            self.actions.pop()
            raise
        LOGGER.debug(
            "러스트 액션 적용: 액션=%s 이전={%s} 이후={%s} 되돌리기깊이=%s",
            action_id,
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.actions),
        )

    def can_undo(self) -> bool:
        return bool(self.actions)

    def undo(self) -> bool:
        if not self.actions:
            LOGGER.debug("러스트 되돌리기 무시: 액션 기록이 비어 있음")
            return False
        before = self.snapshot()
        removed = self.actions.pop()
        self._snapshot = self._run_trace()
        LOGGER.debug(
            "러스트 되돌리기: 제거한액션=%s 이전={%s} 이후={%s} 되돌리기깊이=%s",
            removed,
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.actions),
        )
        return True

    def _run_trace(self) -> Snapshot:
        fixture = {
            "config": self.config.to_snapshot(),
            "initial_deck": [card.to_snapshot() for card in self.initial_deck],
            "steps": [{"action": None}] + [{"action": action} for action in self.actions],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(fixture, handle)
            fixture_path = Path(handle.name)
        try:
            result = subprocess.run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "--bin",
                    "lost_cities_probe",
                    "--",
                    "trace",
                    str(fixture_path),
                ],
                cwd=RUST_CORE_DIR,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise RuntimeError(f"rust backend failed: {message}") from exc
        finally:
            fixture_path.unlink(missing_ok=True)

        trace = json.loads(result.stdout)
        return snapshot_from_trace(trace["config"], trace["steps"][-1])


def _shuffled_deck(config: LostCitiesConfig, seed: int | None) -> list[Card]:
    deck = build_deck(config)
    rng = random.Random(config.seed if seed is None else seed)
    rng.shuffle(deck)
    return deck
