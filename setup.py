from __future__ import annotations

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Cython is required to build coolrl-lost-cities") from exc


extensions = cythonize(
    [
        Extension(
            "coolrl_lost_cities.games.classic.game",
            ["src/coolrl_lost_cities/games/classic/game.pyx"],
        ),
        Extension(
            "coolrl_lost_cities.games.classic.deep_cfr.cfr_math",
            ["src/coolrl_lost_cities/games/classic/deep_cfr/cfr_math.pyx"],
        ),
        Extension(
            "coolrl_lost_cities.games.classic.deep_cfr.encoding",
            ["src/coolrl_lost_cities/games/classic/deep_cfr/encoding.pyx"],
        ),
        Extension(
            "coolrl_lost_cities.games.classic.deep_cfr.traversal",
            ["src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx"],
        ),
        Extension(
            "coolrl_lost_cities.games.classic.bots.heuristic_cy",
            ["src/coolrl_lost_cities/games/classic/bots/heuristic_cy.pyx"],
        ),
        Extension(
            "coolrl_lost_cities.games.classic.ismcts.mcts",
            ["src/coolrl_lost_cities/games/classic/ismcts/mcts.pyx"],
        ),
    ],
    language_level=3,
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "initializedcheck": False,
    },
)


setup(ext_modules=extensions)
