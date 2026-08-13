"""Small shared environment defaults for focused Python tests."""

from __future__ import annotations

import os


# Avoid surprising numerical-library thread pools during unit tests.
for _thread_limit_variable in (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_limit_variable, "1")
