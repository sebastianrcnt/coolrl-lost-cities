# coolrl-lost-cities

Focused Lost Cities extraction from the legacy `coolrl` repository.

The current implementation starts with the classic two-player card game:

- classic 5-expedition rules by default
- Python/Cython game engine
- env wrapper
- random, passive-discard, and safe-heuristic bots
- core rule, scoring, mask, env, canonical-state, bot, and GUI smoke tests

Training code, Deep CFR, learned-policy evaluation, GUI, and web client are
intentionally outside the first port.

## Development

```bash
uv run pytest tests/games/classic
uv run lost-cities-classic
```

For future GUI work, install the optional GUI dependencies:

```bash
uv sync --extra gui
```

Run the classic pygame GUI:

```bash
uv run lost-cities-classic-gui --mode pvc --bot safe-heuristic
```

The GUI uses the in-process Python backend.

## Basic Usage

```python
from coolrl_lost_cities.games.classic import GameState, build_bot, classic_config

state = GameState.new_game(classic_config(seed=1))
bot = build_bot("random", seed=1)

while not state.terminal:
    state.apply_action(bot.act(state))

print(state.total_score(0), state.total_score(1))
```

Backends use the same snapshot/apply/undo interface:

```python
from coolrl_lost_cities.games.classic import build_backend, classic_config

backend = build_backend("python", classic_config(), seed=1)
snapshot = backend.snapshot()
```

See [classic port notes](docs/classic-port-notes.md) for the current direction.
