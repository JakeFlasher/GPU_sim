"""Repository-wide pytest isolation from inherited adjacent bytecode."""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path


# Timestamp-based .pyc validation does not bind bytecode to a checkout. Point
# imports at a unique, nonexistent cache root before test collection so Python
# cannot consume inherited adjacent caches, then suppress all cache writes.
_PYCACHE_PREFIX = os.fspath(
    Path(tempfile.gettempdir())
    / f"gpu-simulator-pycache-disabled-{os.getpid()}-{uuid.uuid4().hex}"
)
os.environ["PYTHONPYCACHEPREFIX"] = _PYCACHE_PREFIX
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
# External tracer build tests exercise a production boundary that requires a
# single-threaded parent. Keep numerical runtimes from creating ambient worker
# pools during earlier tests in the same pytest process.
for _thread_limit_variable in (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_limit_variable] = "1"
sys.pycache_prefix = _PYCACHE_PREFIX
sys.dont_write_bytecode = True
