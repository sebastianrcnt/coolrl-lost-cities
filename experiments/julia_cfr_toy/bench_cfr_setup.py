from __future__ import annotations

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent

setup(
    name="bench_cfr",
    ext_modules=cythonize(
        [
            Extension(
                "bench_cfr",
                [str(ROOT / "bench_cfr.pyx")],
                include_dirs=[np.get_include()],
            )
        ],
        compiler_directives={"language_level": "3"},
    ),
)
