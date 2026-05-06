from __future__ import annotations

import pytest

from coolrl_lost_cities.games.classic import pygame_pvp


def test_gui_argparser_accepts_classic_options() -> None:
    args = pygame_pvp.build_argparser().parse_args(
        [
            "--mode",
            "pvc",
            "--bot",
            "safe-heuristic",
            "--seed",
            "7",
            "--width",
            "1024",
            "--height",
            "768",
        ]
    )

    assert args.mode == "pvc"
    assert args.bot == "safe-heuristic"
    assert args.seed == 7
    assert args.width == 1024
    assert args.height == 768


def test_gui_argparser_rejects_removed_backend_option() -> None:
    with pytest.raises(SystemExit):
        pygame_pvp.build_argparser().parse_args(["--backend", "rust"])
