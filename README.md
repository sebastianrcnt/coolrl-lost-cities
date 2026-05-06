# coolrl-lost-cities

Focused Lost Cities extraction from the legacy `coolrl` repository.

The current implementation starts with the classic two-player card game:

- classic 5-expedition rules by default
- Python/Cython game engine
- Rust core parity checks
- env wrapper
- random, passive-discard, and safe-heuristic bots
- core rule, scoring, mask, env, canonical-state, bot, and Rust parity tests

Training code, Deep CFR, learned-policy evaluation, GUI, and web client are
intentionally outside the first port.

## Development

```bash
uv run pytest tests/games/classic
uv run lost-cities-classic
```

See [classic port notes](docs/classic-port-notes.md) for the current direction.
