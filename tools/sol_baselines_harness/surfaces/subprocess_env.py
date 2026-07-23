"""Curated subprocess env builder for surface plugins.

Each surface invokes the upstream ``sol-execbench`` evaluator (or a
wrapper around it) as a host subprocess. The subprocess env is built
from three layers:

1. ``curated_base_env`` — a fixed allow-list of host env vars copied
   from :data:`os.environ`. Anything not on the allow-list is
   excluded from inheritance so unrelated env vars (debug flags,
   shell-specific paths, transient surface state from a prior run)
   cannot leak into the eval driver.
2. ``plan_env_overrides`` — the plan-wide overrides (FLASHINFER trace
   dir, GPU/DRAM clock hints, etc.) shared across surfaces.
3. ``surface_env_overrides`` — the per-surface overrides
   (``CUDA_INJECTION64_PATH``, ``TRACES_FOLDER``,
   ``SOLBENCH_CUPTI_ACTIVITY_OUT``, ``NSYS_OUTPUT_FILE``, etc.) that
   the surface OWNS and that must NOT survive into another surface's
   subprocess.

After merging, the denylist scrubs any denied key that arrived via
``os.environ`` or ``plan_env_overrides``. Surface-owned overrides
bypass the denylist by construction — the surface that explicitly
sets ``CUDA_INJECTION64_PATH`` does so intentionally; only a
*different* surface inheriting it constitutes leakage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from baselines_harness.native_env import default_cutlass_dir  # noqa: E402


ALLOW_LIST = (
    "PATH",
    "HOME",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
    "TMPDIR",
)

# Build-toolchain selection vars. None of these carry profiling-
# surface state (the denylist still scrubs ``LD_PRELOAD`` /
# ``CUDA_INJECTION64_PATH`` / ``CUPTI_*`` / ``NSYS_*`` regardless);
# they are deterministic build inputs the eval driver's host-compile
# path needs:
#
# * ``CUTLASS_DIR`` — the eval driver's CUTLASS build path
#   (``sol-execbench/src/sol_execbench/driver/templates/build_ext.py``)
#   reads it. The Docker image used to provision
#   ``/usr/local/cutlass``; post-M4 the harness injects
#   :func:`default_cutlass_dir` (``REPO_ROOT/cutlass``, the bundled
#   v4.5.0 tree) when neither the operator env nor the overrides
#   carry one, so a fresh ``uv sync`` host compiles CUTLASS
#   baselines without manual operator setup. The cache hash
#   (:func:`baselines_harness.native_env.cutlass_tree_hash`) reads
#   the same default so the hash matches what the subprocess
#   actually built.
# * ``NVCC_CCBIN``, ``CXX``, ``CC`` — host-compiler selection. On
#   distros where the default ``g++`` is newer than what the
#   installed CUDA toolkit supports (e.g. a g++16 system header
#   tripping an nvcc-13 codegen path), the operator pins a
#   compatible compiler via these vars. Stripping them silently
#   breaks the build with confusing system-header errors.
#
# Callers that drive an ``nvcc`` build path or the ``sol-execbench``
# eval driver should pass ``extra_allowed=BUILD_TOOLCHAIN_ALLOWED``
# to :func:`build_subprocess_env`. The vars stay OUT of
# :data:`ALLOW_LIST` so the curated baseline allow-list stays
# minimal and surfaces that do not need them (cache replay,
# manifest writes) do not pick them up.
BUILD_TOOLCHAIN_ALLOWED = (
    "CUTLASS_DIR",
    "NVCC_CCBIN",
    "CXX",
    "CC",
)

DENYLIST_TOKENS = (
    "CUDA_INJECTION64_PATH",
    "NVBIT_TOOL",
    "TRACES_FOLDER",
    "SOLBENCH_CUPTI_ACTIVITY_OUT",
    "LD_PRELOAD",
    "NSYS_OUTPUT_FILE",
)

DENYLIST_PREFIXES = (
    "CUPTI_",
    "NSYS_",
)


def curated_base_env(
    source_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return the curated subset of ``source_env`` keyed by :data:`ALLOW_LIST`.

    Defaults to reading from :data:`os.environ` when ``source_env`` is
    omitted. Keys not on the allow-list are NEVER copied.
    """
    env = source_env if source_env is not None else os.environ
    return {key: env[key] for key in ALLOW_LIST if key in env}


def build_subprocess_env(
    plan_env_overrides: Optional[Mapping[str, str]] = None,
    surface_env_overrides: Optional[Mapping[str, str]] = None,
    *,
    source_env: Optional[Mapping[str, str]] = None,
    extra_allowed: Iterable[str] = (),
) -> Dict[str, str]:
    """Return the subprocess env for one surface invocation.

    Parameters
    ----------
    plan_env_overrides
        Plan-wide env overrides shared across surfaces (FLASHINFER
        trace dir, GPU/DRAM clock hints, etc.).
    surface_env_overrides
        Surface-owned env overrides that bypass the denylist. The
        surface that sets ``CUDA_INJECTION64_PATH`` etc. lists it
        here; another surface that does NOT list it sees the key
        scrubbed by the denylist even if a prior surface or
        ``os.environ`` carried it.
    source_env
        Defaults to :data:`os.environ`. Override for testing.
    extra_allowed
        Additional allow-list keys for cases where a surface
        genuinely needs an env var that is not on the curated
        :data:`ALLOW_LIST` (e.g., a future ``CXX`` requirement). Use
        sparingly — every entry weakens the isolation contract.
    """
    plan_overrides = dict(plan_env_overrides or {})
    surface_overrides = dict(surface_env_overrides or {})
    base_env = source_env if source_env is not None else os.environ
    allowed = set(ALLOW_LIST) | set(extra_allowed)
    base = {key: base_env[key] for key in allowed if key in base_env}
    merged: Dict[str, str] = {**base, **plan_overrides, **surface_overrides}
    # Native CUTLASS_DIR default: when a build-toolchain call site
    # opted into BUILD_TOOLCHAIN_ALLOWED but no upstream layer
    # supplied CUTLASS_DIR, fall back to the bundled tree so the
    # eval driver's build_ext.py picks up REPO_ROOT/cutlass instead
    # of the absent Docker-era /usr/local/cutlass. native_env's
    # cutlass_tree_hash reads the same default, so the cache hash
    # and the subprocess build path agree by construction.
    if "CUTLASS_DIR" in allowed and "CUTLASS_DIR" not in merged:
        merged["CUTLASS_DIR"] = str(default_cutlass_dir())
    for key in list(merged.keys()):
        if key in surface_overrides:
            continue
        if key in DENYLIST_TOKENS:
            del merged[key]
            continue
        if any(key.startswith(prefix) for prefix in DENYLIST_PREFIXES):
            del merged[key]
    return merged
