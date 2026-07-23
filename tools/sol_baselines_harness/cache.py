"""Harness cache-key derivation and on-disk index helpers.

The key tuple is:

    (solution_source_hash, sol_execbench_source_hash,
     native_env_hash, environment_libs, build_mode, surface_set)

Cache identity is **strict**: every component must read a real value
before a cached result is replay-eligible. There is no asymmetric
"unknown digest matches anything" fallback — a degraded native-env
fingerprint raises :class:`DegradedNativeEnvError` at key derivation
time so the harness fails closed.

``build_mode ∈ {"judge", "profiling"}`` separates the canonical judge
build (byte-identical to the landed ``solution.json``) from the
counter-free framework's profiling build (staged ``solution.json``
with ``-DSOLBENCH_PROFILING`` always and ``-DSOLBENCH_HAS_NVTX``
conditional on host NVTX probes). ``surface_set`` is the sorted tuple
of surfaces in the run: a partial-surface run (e.g. ``{"static"}``)
must not collide with a full-set run (``{"static", "sanitizer",
"nvbit_metrics", ...}``) because the bundle the report consumes differs.

A cached harness result is replay-eligible only when every component
of the tuple matches AND every row in the cached trace is ``PASSED``.
Speedup vs the PyTorch reference is **diagnostic only** under the
correctness-only landing contract — a PASSED-but-slow row replays
normally. Non-PASSED rows are never replayed; they are recomputed.

``sol_execbench_source_hash`` is the **Git tree hash** of the vendored
``sol-execbench`` subdirectory at the current HEAD, computed via
``git rev-parse HEAD:sol-execbench``. Top-level commits that only
touch ``scripts/baselines_harness/``, ``baselines/``, or report
artifacts do NOT invalidate cached PASS traces.

``native_env_hash`` is the SHA-256 of a canonical-order JSON over the
seven host-environment components defined in
:mod:`baselines_harness.native_env` (python / torch / nvcc / runtime
CUDA / CUTLASS tree / cuDNN / sol-execbench package). The raw
component dict is persisted alongside the hash in
``cache_record_for_key`` so downstream debugging can inspect which
component differed without re-probing the host.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
REPO_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from baselines_harness.native_env import (  # noqa: E402
    DegradedNativeEnvError,
    compute_native_env_hash,
    missing_components,
    native_env_components,
)

VALID_BUILD_MODES = frozenset({"judge", "profiling"})
EMPTY_SURFACE_SET: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class CacheKey:
    """Hashable cache key for a single harness run.

    The 6-tuple is:

    * ``solution_source_hash`` — SHA-256 over ``solution.json`` source
      contents in canonical order (name + definition + spec fields +
      sources).
    * ``sol_execbench_source_hash`` — Git tree hash of the vendored
      ``sol-execbench/`` directory.
    * ``native_env_hash`` — SHA-256 of the seven-component native-env
      fingerprint computed by :mod:`baselines_harness.native_env`.
      Strict identity, no asymmetric fallback.
    * ``environment_libs`` — sorted ``(name, version)`` pairs from
      ``evaluation.environment.libs``.
    * ``build_mode`` — ``"judge"`` or ``"profiling"``.
    * ``surface_set`` — sorted tuple of surface names that participated
      in the run; empty tuple for the legacy trace-only path.

    All six components participate in :meth:`as_sha256` and
    :func:`key_matches_record` so a profiling cache entry never serves
    a judge query, a partial-surface entry never serves a full-set
    query, and a degraded-env entry never matches anything.
    """

    solution_source_hash: str
    sol_execbench_source_hash: str
    native_env_hash: str
    environment_libs: Tuple[Tuple[str, str], ...]
    build_mode: str
    surface_set: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.build_mode not in VALID_BUILD_MODES:
            raise ValueError(
                f"CacheKey.build_mode must be one of {sorted(VALID_BUILD_MODES)}; "
                f"got {self.build_mode!r}"
            )
        if list(self.surface_set) != sorted(self.surface_set):
            raise ValueError(
                f"CacheKey.surface_set must be a sorted tuple; got "
                f"{list(self.surface_set)!r}. Convert with "
                f"`tuple(sorted(my_set))` at the call site."
            )

    def as_sha256(self) -> str:
        """Return a stable hex digest of the key."""
        payload = json.dumps(
            {
                "solution_source_hash": self.solution_source_hash,
                "sol_execbench_source_hash": self.sol_execbench_source_hash,
                "native_env_hash": self.native_env_hash,
                "environment_libs": list(self.environment_libs),
                "build_mode": self.build_mode,
                "surface_set": list(self.surface_set),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def solution_source_hash(solution_dict: Dict[str, Any]) -> str:
    """SHA-256 over the solution's source contents in canonical order.

    Mirrors the live ``Solution._compute_hash`` ordering (name +
    definition + languages + entry_point + binding + dependencies +
    compile_options + per-source (path, content)).

    ``spec.compile_options`` is hashed after ``dependencies``: compile
    flags directly change the produced binary's behavior (`-O3` vs
    `-O0`, `-gencode`, linker flags), so omitting them from the cache
    key would let `report.py` replay latency/correctness evidence from
    a different binary on a flag edit. Within ``compile_options`` the
    three list-valued keys (``cflags``, ``cuda_cflags``, ``ld_flags``)
    are iterated in sorted key order, but element order within each
    list is preserved verbatim because compile-flag order can be
    semantically significant.
    """
    h = hashlib.sha256()
    h.update(solution_dict.get("name", "").encode("utf-8"))
    h.update(b"\x00")
    h.update(solution_dict.get("definition", "").encode("utf-8"))
    h.update(b"\x00")
    spec = solution_dict.get("spec", {})
    for lang in spec.get("languages", []) or []:
        h.update(str(lang).encode("utf-8"))
        h.update(b"\x00")
    h.update(spec.get("entry_point", "").encode("utf-8"))
    h.update(b"\x00")
    binding = spec.get("binding") or ""
    h.update(str(binding).encode("utf-8"))
    h.update(b"\x00")
    for dep in spec.get("dependencies", []) or []:
        h.update(str(dep).encode("utf-8"))
        h.update(b"\x00")
    compile_options = spec.get("compile_options") or {}
    for co_key in sorted(compile_options.keys()):
        h.update(str(co_key).encode("utf-8"))
        h.update(b"\x00")
        for flag in compile_options.get(co_key) or []:
            h.update(str(flag).encode("utf-8"))
            h.update(b"\x00")
        h.update(b"\x01")
    h.update(b"\x02")
    for src in solution_dict.get("sources", []) or []:
        h.update(src.get("path", "").encode("utf-8"))
        h.update(b"\x00")
        h.update(src.get("content", "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def sol_execbench_source_hash() -> str:
    """Resolve the vendored ``sol-execbench`` subdirectory's Git tree hash.

    Returns the tree hash from ``git rev-parse HEAD:sol-execbench``
    (anchored on the repo root) so the value changes only when the
    evaluator source tree actually changes.

    Falls back to ``"unknown"`` when git is unavailable or the
    invocation fails; the cache key then mismatches every prior record
    and forces a fresh run.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD:sol-execbench"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def environment_libs_tuple(
    libs: Optional[Dict[str, str]],
) -> Tuple[Tuple[str, str], ...]:
    """Convert ``evaluation.environment.libs`` into a sorted tuple."""
    if not libs:
        return ()
    return tuple(sorted(libs.items()))


def derive(
    solution_dict: Dict[str, Any],
    environment_libs: Optional[Dict[str, str]] = None,
    build_mode: str = "judge",
    surface_set: Optional[FrozenSet[str]] = None,
    native_env_hash: Optional[str] = None,
    native_env_components_value: Optional[Dict[str, str]] = None,
) -> CacheKey:
    """Build a ``CacheKey`` from the components currently in scope.

    Strict identity rule: if ``native_env_hash`` is not supplied, the
    live host components are probed; any component reading ``"unknown"``
    raises :class:`DegradedNativeEnvError` so a degraded fingerprint
    is never silently reused.
    """
    if native_env_hash is None:
        components = (
            dict(native_env_components_value)
            if native_env_components_value is not None
            else native_env_components()
        )
        missing = missing_components(components)
        if missing:
            raise DegradedNativeEnvError(missing, components)
        native_env_hash = compute_native_env_hash(components)
    sset = tuple(sorted(surface_set)) if surface_set else tuple()
    return CacheKey(
        solution_source_hash=solution_source_hash(solution_dict),
        sol_execbench_source_hash=sol_execbench_source_hash(),
        native_env_hash=native_env_hash,
        environment_libs=environment_libs_tuple(environment_libs),
        build_mode=build_mode,
        surface_set=sset,
    )


def is_passed_trace(rows: list) -> bool:
    """True when every report row has status ``PASSED``."""
    if not rows:
        return False
    return all(r.get("status") == "PASSED" for r in rows)


def current_baseline_key(
    baseline_dir: Path,
    environment_libs: Optional[Dict[str, str]] = None,
    build_mode: str = "judge",
    surface_set: Optional[FrozenSet[str]] = None,
    native_env_hash: Optional[str] = None,
    native_env_components_value: Optional[Dict[str, str]] = None,
) -> CacheKey:
    """Derive a ``CacheKey`` from a baseline's current on-disk state.

    Reads ``baseline_dir/solution.json`` for ``solution_source_hash``,
    resolves the current ``sol-execbench`` subdirectory tree hash, and
    computes the native-env fingerprint (or accepts a pre-computed one
    via the optional ``native_env_hash`` / ``native_env_components_value``
    pair). Used by the aggregate report's cache-keyed replay gate.
    """
    solution_path = baseline_dir / "solution.json"
    solution_dict = json.loads(solution_path.read_text(encoding="utf-8"))
    return derive(
        solution_dict,
        environment_libs=environment_libs,
        build_mode=build_mode,
        surface_set=surface_set,
        native_env_hash=native_env_hash,
        native_env_components_value=native_env_components_value,
    )


_FORBIDDEN_CACHE_RECORD_FIELDS = (
    "docker_image_digest",
    "profiling_image_digest",
    "dispatch_mode",
    "dispatch_kind",
    "docker_digest",
)


_REQUIRED_CACHE_RECORD_FIELDS = (
    "native_env_hash",
    "build_mode",
    "surface_set",
)


def _record_is_v2_native(record: Any) -> bool:
    """Return True iff ``record`` is a v2-native cache record.

    The strict shape: a dict carrying every required field
    (``native_env_hash``, ``build_mode``, ``surface_set``) AND no
    Docker-era forbidden field
    (``docker_image_digest``, ``profiling_image_digest``,
    ``dispatch_mode``, ``dispatch_kind``, ``docker_digest``).
    """
    if not isinstance(record, dict):
        return False
    for field_name in _REQUIRED_CACHE_RECORD_FIELDS:
        if field_name not in record:
            return False
    if any(k in record for k in _FORBIDDEN_CACHE_RECORD_FIELDS):
        return False
    return True


def load_cache_index(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read the per-trace cache index. Returns ``{}`` when absent/invalid.

    Docker-era records are dropped at load time: any record that is
    missing one of :data:`_REQUIRED_CACHE_RECORD_FIELDS` or that
    carries one of :data:`_FORBIDDEN_CACHE_RECORD_FIELDS` is silently
    omitted from the returned dict. This implements the plan's
    hard-invalidate policy: an old `.harness_cache.json` from a
    pre-native-dispatch dev box cannot serve a v2 replay query.

    Loaded records are NOT persisted back here; callers (typically
    ``report.py``) overwrite the on-disk index with their own
    freshly-stamped records, so a dropped Docker-era record stays
    dropped on the next ``save_cache_index`` call.
    """
    if not cache_path.is_file():
        return {}
    try:
        records = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(records, dict):
        return {}
    return {k: v for k, v in records.items() if _record_is_v2_native(v)}


def save_cache_index(
    cache_path: Path,
    index: Dict[str, Dict[str, Any]],
) -> None:
    """Persist the cache index. Caller owns ``cache_path.parent`` mkdir."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cache_record_for_key(
    key: CacheKey,
    native_env_components_value: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Convert a CacheKey to the serialised index entry shape.

    When the raw component dict is supplied, it is persisted alongside
    the hash so downstream debugging can inspect which component
    differed without re-probing the host.
    """
    record: Dict[str, Any] = {
        "solution_source_hash": key.solution_source_hash,
        "sol_execbench_source_hash": key.sol_execbench_source_hash,
        "native_env_hash": key.native_env_hash,
        "environment_libs": [list(p) for p in key.environment_libs],
        "build_mode": key.build_mode,
        "surface_set": list(key.surface_set),
    }
    if native_env_components_value is not None:
        record["native_env_components"] = dict(native_env_components_value)
    return record


def key_matches_record(key: CacheKey, record: Dict[str, Any]) -> bool:
    """True when ``key`` matches ``record`` under the strict cache contract.

    Every component requires exact equality. There is no asymmetric
    fallback for any component — a degraded ``native_env_hash`` cannot
    be derived in the first place (``derive`` raises
    :class:`DegradedNativeEnvError`), so by the time a ``CacheKey``
    reaches this function its identity is fully resolved.
    """
    stored_libs = tuple(
        tuple(pair) for pair in record.get("environment_libs", [])
    )
    if record.get("solution_source_hash") != key.solution_source_hash:
        return False
    if record.get("sol_execbench_source_hash") != key.sol_execbench_source_hash:
        return False
    if stored_libs != key.environment_libs:
        return False
    if record.get("build_mode", "judge") != key.build_mode:
        return False
    if tuple(record.get("surface_set", [])) != key.surface_set:
        return False
    if record.get("native_env_hash") != key.native_env_hash:
        return False
    return True
