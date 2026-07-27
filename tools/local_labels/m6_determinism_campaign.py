#!/usr/bin/env python3
"""Offline planner, analyzer, and verifier for the sm_86 determinism study.

This source deliberately contains no NCU, GPU, shell, or hardware invocation.
The ``collect`` command validates its prerequisite identity and then fails with a
stable disabled-source error.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

try:
    from . import build_sm86_capture_matrix as matrix_builder
    from . import campaign_done_manifest as m4_done
    from . import import_label as label_import
    from . import immutable_artifact as immutable
    from . import metric_registry as metric_registry_lib
    from . import ncu_capture_contract
    from . import parse_ncu_metrics as ncu_parser
    from . import sidecar_schema_authority
    from .m6_determinism_stats import (
        DETERMINISM_CLASSES,
        benjamini_hochberg,
        c4,
        classify_sigma,
        dispersion_gate,
        median_bootstrap_ci,
        modality_result,
        pooled_p90_upper,
        sigma_rel,
        sigma_upper_factor,
        trigger_codes,
    )
except ImportError:  # direct script execution
    import build_sm86_capture_matrix as matrix_builder  # type: ignore[no-redef]
    import campaign_done_manifest as m4_done  # type: ignore[no-redef]
    import import_label as label_import  # type: ignore[no-redef]
    import immutable_artifact as immutable  # type: ignore[no-redef]
    import metric_registry as metric_registry_lib  # type: ignore[no-redef]
    import ncu_capture_contract  # type: ignore[no-redef]
    import parse_ncu_metrics as ncu_parser  # type: ignore[no-redef]
    import sidecar_schema_authority  # type: ignore[no-redef]
    from m6_determinism_stats import (  # type: ignore[no-redef]
        DETERMINISM_CLASSES,
        benjamini_hochberg,
        c4,
        classify_sigma,
        dispersion_gate,
        median_bootstrap_ci,
        modality_result,
        pooled_p90_upper,
        sigma_rel,
        sigma_upper_factor,
        trigger_codes,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "data/local_labels/campaign/sm86_reference_capture_matrix.json"
MATRIX_SCHEMA_PATH = (
    REPO_ROOT / "workloads/simbench_bw/schemas/sm86_capture_matrix.schema.json"
)
REGISTRY_PATH = REPO_ROOT / "tools/local_labels/metric_registry.json"
FROZEN_REGISTRY_PATH = (
    REPO_ROOT / "data/local_labels/campaign/frozen_v1_metric_registry.json"
)
FROZEN_SIDECAR_SCHEMA_PATH = (
    REPO_ROOT
    / "data/local_labels/campaign/frozen_v1_local_label_sidecar.schema.json"
)
PROBE_PATH = REPO_ROOT / "data/local_labels/ncu_metric_availability_probe_sm_86.json"
SIDECAR_SCHEMA_PATH = REPO_ROOT / "workloads/simbench_bw/schemas/local_label_sidecar.schema.json"
FROZEN_SOURCE_ALIAS_PATH = (
    REPO_ROOT / "data/local_labels/campaign/frozen_v1_source_aliases.v2.json"
)
FROZEN_SOURCE_ALIAS_SHA256 = (
    "e4e46c1a77e855e8858dbdb27876738d61b0fc28227297a3470e83b2a9df2014"
)
FROZEN_SOURCE_ALIAS_SCHEMA_VERSION = 2
FROZEN_V1_SOURCE_COMMIT = "124ce358733f97ba6e0af808d10ab69ffe4d5a78"
STATS_PATH = Path(__file__).resolve().with_name("m6_determinism_stats.py")
IMMUTABLE_HELPER_PATH = Path(__file__).resolve().with_name("immutable_artifact.py")
SCHEMA_DIR = REPO_ROOT / "workloads/simbench_bw/schemas"
RECOVERY_SCHEMA_PATH = SCHEMA_DIR / "m6_recovery_receipt.schema.json"
RECOVERY_SCHEMA_SHA256 = "54613e4e4c879bb037d401ec3dafd8d6bba5628e61cea9627f08e517c8a27311"
SCHEMA_PATHS_BY_VERSION = {
    1: {
        "m4_binding": SCHEMA_DIR / "m6_m4_binding_receipt.schema.json",
        "schedule": SCHEMA_DIR / "m6_determinism_schedule.schema.json",
        "observation_journal": SCHEMA_DIR / "m6_determinism_observation_journal.schema.json",
        "analysis": SCHEMA_DIR / "m6_determinism_analysis.schema.json",
        "done": SCHEMA_DIR / "m6_determinism_done.schema.json",
    },
    2: {
        "m4_binding": SCHEMA_DIR / "m6_m4_binding_receipt.schema.json",
        "schedule": SCHEMA_DIR / "m6_determinism_schedule.v2.schema.json",
        "observation_journal": SCHEMA_DIR / "m6_determinism_observation_journal.v2.schema.json",
        "historical_admission_set": SCHEMA_DIR
        / "m6_historical_admission_set.v1.schema.json",
        "analysis": SCHEMA_DIR / "m6_determinism_analysis.v2.schema.json",
        "done": SCHEMA_DIR / "m6_determinism_done.v2.schema.json",
    },
}
SCHEMA_IDS_BY_VERSION = {
    1: {
        "m4_binding": "simbench_bw/schemas/m6_m4_binding_receipt.schema.json",
        "schedule": "simbench_bw/schemas/m6_determinism_schedule.schema.json",
        "observation_journal": "simbench_bw/schemas/m6_determinism_observation_journal.schema.json",
        "analysis": "simbench_bw/schemas/m6_determinism_analysis.schema.json",
        "done": "simbench_bw/schemas/m6_determinism_done.schema.json",
    },
    2: {
        "m4_binding": "simbench_bw/schemas/m6_m4_binding_receipt.schema.json",
        "schedule": "simbench_bw/schemas/m6_determinism_schedule.v2.schema.json",
        "observation_journal": "simbench_bw/schemas/m6_determinism_observation_journal.v2.schema.json",
        "historical_admission_set": "simbench_bw/schemas/m6_historical_admission_set.v1.schema.json",
        "analysis": "simbench_bw/schemas/m6_determinism_analysis.v2.schema.json",
        "done": "simbench_bw/schemas/m6_determinism_done.v2.schema.json",
    },
}
SCHEMA_SHA256_BY_VERSION = {
    1: {
        "m4_binding": "d1a6338eec3ec59755b5ed769d9fac6ddcc1e790caac31091336207bb4648dc2",
        "schedule": "5aef059b1175d971c7d7c92f63c292a31cc206d3cb7dd458244a4382ea9efabc",
        "observation_journal": "2056ca961d70c3a8fdd773945bd09c8f6c61dab5b3f298cd152134ce135d21a0",
        "analysis": "35cd406d202f09087426144b74383490c6bf10f9b78ca11457ee7fc486636bb4",
        "done": "60b2223002710a71db483cc1f8a4d4e15929e4a4d283f044a44b04b921662e2a",
    },
    2: {
        "m4_binding": "d1a6338eec3ec59755b5ed769d9fac6ddcc1e790caac31091336207bb4648dc2",
        "schedule": "0cd16ac70046b984306d8e02e8f601f9cbbeafa37f98f2c8ee3e814e8565f3cf",
        "observation_journal": "58afa2fcbcc850e6bafcfb8af0027bdf349ecbfd793b9b3812914d5bae9e85c5",
        "historical_admission_set": "675054f6866d89161260595b57dac61f612a59f580a037770b1d4c2818c78aae",
        "analysis": "004b4e2eaf7a9b4fcd7b2de3c523852376677699e6bd409712e98392215d3401",
        "done": "cad84152efc311538d2ce6f71c6d22a4d0d34255795cef3674ee01c3329e92d5",
    },
}
# Compatibility aliases deliberately retain the historical v1 paths.  Active
# producers and publication gates use the explicit v2 dispatch below.
SCHEMA_PATHS_V1 = SCHEMA_PATHS_BY_VERSION[1]
SCHEMA_PATHS_V2 = SCHEMA_PATHS_BY_VERSION[2]
SCHEMA_PATHS = SCHEMA_PATHS_V1
ACTIVE_SCHEMA_VERSION = 2
REFERENCE_CONFIG = "cfg=clock_none__cache_all"
ZERO_SHA256 = "0" * 64
TIER1_SLOTS = 8_540
TIER2_CANDIDATES = 1_220
MAXIMUM_SLOTS = 9_760
EXPECTED_RUNNABLE = 122
EXPECTED_UNIVERSE = 344
EXPECTED_EXCLUDED = 222
EXPECTED_REGISTRY_METRICS = 61
EXPECTED_SELECTED_METRICS = 55
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
STATISTICAL_CONTRACT_VERSION = "m6_statistical_override_v1"
STATISTICAL_CONTRACT_SHA256 = "17ac0986bc090532ce1898eed5223e02f8e8944ca08ec134e6f58a8a6f9707b5"
HISTORICAL_ADMISSION_PROJECTION = "reference_config_cell_metric_dispersion_gate_v1"
# Populated only by a separate, independently reviewed authority-admission change.
HISTORICAL_ADMISSION_AUTHORITY_ARTIFACT_SHA256: str | None = None
# These are explicit per-key operands, not unit- or class-derived defaults.  The
# list is sealed into every projected metric contract and therefore into the
# schedule metric identity.  A missing or overlapping key fails closed.
STATISTICAL_DISCRETE_KEYS = frozenset({
    "traffic_gmem_ld_requests",
    "traffic_gmem_st_requests",
    "traffic_gmem_ld_sectors",
    "traffic_gmem_st_sectors",
    "traffic_dram_read_sectors",
    "traffic_dram_write_sectors",
    "traffic_l2_total_requests_from_tex",
    "traffic_l2_read_sectors_from_l1",
    "traffic_l1_write_sectors_to_l2",
    "traffic_gmem_atom_requests",
    "traffic_gmem_red_requests",
    "traffic_l1_total_requests",
    "occupancy_limit_warps_theoretical",
    "occupancy_limit_blocks_theoretical",
    "occupancy_limit_registers_theoretical",
    "occupancy_limit_shared_mem_theoretical",
    "smsp_inst_executed_sum",
    "traffic_local_ld_requests",
    "traffic_local_st_requests",
    "traffic_tex_requests",
    "traffic_surface_ld_requests",
    "traffic_surface_st_requests",
    "traffic_surface_red_requests",
    "traffic_surface_atom_requests",
})
STATISTICAL_CONTINUOUS_FINE_KEYS = frozenset({
    "cache_hit_rate_l2_sector",
    "cache_hit_rate_l1tex_sector",
    "cache_hit_rate_l1tex_read_sector",
    "ipc_inst_per_cycle",
    "dram_bytes_per_second",
    "throughput_compute_mem_access_pct_peak",
    "throughput_compute_mem_request_pct_peak",
    "throughput_sm_memory_pct_peak",
    "traffic_dram_sectors_op_read",
    "traffic_dram_sectors_op_write",
    "warp_stall_barrier_per_issue_active",
    "warp_stall_branch_resolving_per_issue_active",
    "warp_stall_dispatch_stall_per_issue_active",
    "warp_stall_drain_per_issue_active",
    "warp_stall_imc_miss_per_issue_active",
    "warp_stall_lg_throttle_per_issue_active",
    "warp_stall_long_scoreboard_per_issue_active",
    "warp_stall_math_pipe_throttle_per_issue_active",
    "warp_stall_membar_per_issue_active",
    "warp_stall_mio_throttle_per_issue_active",
    "warp_stall_misc_per_issue_active",
    "warp_stall_no_instruction_per_issue_active",
    "warp_stall_not_selected_per_issue_active",
    "warp_stall_selected_per_issue_active",
    "warp_stall_short_scoreboard_per_issue_active",
    "warp_stall_sleeping_per_issue_active",
    "warp_stall_tex_throttle_per_issue_active",
    "warp_stall_wait_per_issue_active",
    "throughput_sm_pct_peak",
    "throughput_compute_memory_pct_peak",
})
STATISTICAL_CONTINUOUS_UNIT_KEYS = frozenset({
    "exec_cycles_gpc_max",
    "exec_duration_ns",
    "occupancy_achieved_warps_pct",
    "exec_cycles_sm_active_avg",
    "warps_active_per_cycle_active_avg",
    "sm_clock_hz_covariate",
    "mem_clock_hz_covariate",
})
STATISTICAL_CONTRACT_KEYS = (
    STATISTICAL_DISCRETE_KEYS
    | STATISTICAL_CONTINUOUS_FINE_KEYS
    | STATISTICAL_CONTINUOUS_UNIT_KEYS
)


class M6Error(ValueError):
    """Stable fail-closed controller error."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> None:
    raise M6Error(code, detail)


class InputSnapshotCache:
    """Bind each declared input to one verified byte snapshot per invocation."""

    def __init__(self) -> None:
        self._entries: dict[
            str,
            tuple[str, int, bytes, immutable.MetadataSnapshot],
        ] = {}
        self._witnesses: dict[str, dict[str, Any]] = {}
        self._roles: dict[str, set[str]] = {}

    @staticmethod
    def _identity_object(
        identity: immutable.MetadataSnapshot,
    ) -> dict[str, int | str]:
        names = (
            "device",
            "inode",
            "mode",
            "nlink",
            "size_bytes",
            "mtime_ns",
            "ctime_ns",
        )
        numeric = {name: int(value) for name, value in zip(names, identity)}
        for name in ("device", "inode", "mtime_ns", "ctime_ns"):
            numeric[name] = str(numeric[name])
        return numeric

    def read(
        self,
        path: Path,
        *,
        role: str,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> bytes:
        absolute = Path(os.path.abspath(os.fspath(path)))
        key = os.fspath(absolute)
        entry = self._entries.get(key)
        if entry is not None:
            digest, size, payload, identity = entry
            if (
                expected_sha256 is not None and expected_sha256 != digest
            ) or (expected_size is not None and expected_size != size):
                _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: declared identity changed")
            try:
                observed = immutable.snapshot_regular_identity(absolute)
            except immutable.ImmutableArtifactError as exc:
                _fail("M6_INPUT_SNAPSHOT", f"{role}: {absolute}: {exc}")
            if observed != identity:
                _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: path generation changed")
            witness = self._witnesses[key]
            self._roles.setdefault(key, set()).add(role)
            if expected_sha256 is not None:
                declared = witness.get("declared_sha256")
                if declared not in (None, expected_sha256):
                    _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: digest declaration changed")
                witness["declared_sha256"] = expected_sha256
            if expected_size is not None:
                declared_size = witness.get("declared_size_bytes")
                if declared_size not in (None, expected_size):
                    _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: size declaration changed")
                witness["declared_size_bytes"] = expected_size
            return payload
        try:
            snapshot = immutable.read_regular_snapshot(absolute)
        except immutable.ImmutableArtifactError as exc:
            _fail("M6_INPUT_SNAPSHOT", f"{role}: {absolute}: {exc}")
        return self.bind_pinned(
            absolute,
            snapshot.payload,
            identity=snapshot.metadata,
            role=role,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )

    def lookup_bound(
        self,
        path: Path,
        *,
        identity: immutable.MetadataSnapshot,
        role: str,
    ) -> bytes | None:
        """Return bytes already bound to this exact pinned generation."""
        absolute = Path(os.path.abspath(os.fspath(path)))
        key = os.fspath(absolute)
        existing = self._entries.get(key)
        if existing is None:
            return None
        if existing[3] != identity:
            _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: path generation changed")
        self._roles.setdefault(key, set()).add(role)
        return existing[2]

    def bind_pinned(
        self,
        path: Path,
        payload: bytes,
        *,
        identity: immutable.MetadataSnapshot,
        role: str,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> bytes:
        """Bind bytes and identity consumed through a pinned regular leaf."""
        absolute = Path(os.path.abspath(os.fspath(path)))
        if (
            len(identity) != 7
            or not stat.S_ISREG(identity[2])
            or identity[3] != 1
        ):
            _fail("M6_INPUT_SNAPSHOT", f"{role}: regular single-link file required")
        if len(payload) != identity[4]:
            _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: bound payload size differs")
        digest = sha256_bytes(payload)
        if expected_sha256 is not None and digest != expected_sha256:
            _fail("M6_SOURCE_IDENTITY_MISMATCH", role)
        if expected_size is not None and len(payload) != expected_size:
            _fail("M6_SOURCE_IDENTITY_MISMATCH", role)
        key = os.fspath(absolute)
        existing = self._entries.get(key)
        if existing is not None:
            if existing[:2] != (digest, len(payload)) or existing[2] != payload:
                _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: conflicting bound payload")
            if existing[3] != identity:
                _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: path generation changed")
            self._roles.setdefault(key, set()).add(role)
            witness = self._witnesses[key]
            if expected_sha256 is not None:
                declared = witness.get("declared_sha256")
                if declared not in (None, expected_sha256):
                    _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: digest declaration changed")
                witness["declared_sha256"] = expected_sha256
            if expected_size is not None:
                declared_size = witness.get("declared_size_bytes")
                if declared_size not in (None, expected_size):
                    _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: size declaration changed")
                witness["declared_size_bytes"] = expected_size
            return existing[2]
        self._entries[key] = (digest, len(payload), payload, identity)
        self._roles[key] = {role}
        self._witnesses[key] = {
            "canonical_path": key,
            "role": role,
            "declared_sha256": expected_sha256,
            "declared_size_bytes": expected_size,
            "observed_sha256": digest,
            "observed_size_bytes": len(payload),
            "pre_identity": self._identity_object(identity),
            "post_identity": self._identity_object(identity),
        }
        return payload

    def witness(self) -> list[dict[str, Any]]:
        """Return a deterministic, payload-free snapshot witness."""
        rows: list[dict[str, Any]] = []
        for path in sorted(self._witnesses):
            row = copy.deepcopy(self._witnesses[path])
            roles = sorted(self._roles.get(path, {str(row.get("role", "input"))}))
            row["role"] = " | ".join(roles)
            rows.append(row)
        return rows

    def witness_payload(self) -> bytes:
        return canonical_bytes(self.witness())


def _read_regular_cached(
    path: Path,
    *,
    role: str,
    cache: InputSnapshotCache | None = None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> bytes:
    if cache is None:
        return _read_regular(path, role=role)
    return cache.read(
        path,
        role=role,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )


def _verify_existing_source_snapshot(
    path: Path,
    expected_payload: bytes,
    *,
    cache: InputSnapshotCache,
) -> None:
    """Pin and verify a previously published source witness.

    The witness itself is derived from the invocation's source cache, so the
    retained output is tracked in a separate cache to avoid self-reference.
    Reusing that cache across DONE inventory and final construction detects an
    inode or generation change even when replacement bytes are identical.
    """
    retained = _read_regular_cached(
        path,
        role="published source snapshot",
        cache=cache,
        expected_sha256=sha256_bytes(expected_payload),
        expected_size=len(expected_payload),
    )
    if retained != expected_payload:
        _fail("M6_SOURCE_SNAPSHOT_MISMATCH", "published witness differs")


def _validate_json_domain(
    value: Any,
    *,
    role: str,
    depth: int = 0,
    ancestors: frozenset[int] = frozenset(),
) -> None:
    if depth > 256:
        _fail("M6_JSON_DEPTH", role)
    if type(value) is int and abs(value) > MAX_SAFE_JSON_INTEGER:
        _fail("M6_JSON_UNSAFE_INTEGER", f"{role}: {value}")
    if type(value) is float:
        if not math.isfinite(value):
            _fail("M6_JSON_NONFINITE", f"{role}: {value}")
        if abs(value) > MAX_SAFE_JSON_INTEGER:
            _fail("M6_JSON_UNSAFE_FLOAT", f"{role}: {value}")
    if type(value) is str and any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        _fail("M6_JSON_SURROGATE", role)
    if type(value) in {list, tuple} or type(value) is dict:
        identity = id(value)
        if identity in ancestors:
            _fail("M6_JSON_CYCLE", role)
        nested_ancestors = ancestors | {identity}
        if isinstance(value, dict):
            for key, item in value.items():
                if type(key) is not str:
                    _fail("M6_JSON_OBJECT_KEY", f"{role}: {key!r}")
                _validate_json_domain(
                    key,
                    role=role,
                    depth=depth + 1,
                    ancestors=nested_ancestors,
                )
                _validate_json_domain(
                    item,
                    role=role,
                    depth=depth + 1,
                    ancestors=nested_ancestors,
                )
        else:
            for item in value:
                _validate_json_domain(
                    item,
                    role=role,
                    depth=depth + 1,
                    ancestors=nested_ancestors,
                )


def canonical_bytes(value: Any) -> bytes:
    _validate_json_domain(value, role="canonical JSON")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeEncodeError) as exc:
        _fail("M6_CANONICAL_JSON", str(exc))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            _fail("M6_JSON_DUPLICATE_KEY", key)
        out[key] = value
    return out


def _reject_constant(token: str) -> None:
    _fail("M6_JSON_NONFINITE", token)


def loads_strict(payload: bytes, *, role: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail("M6_JSON_UTF8", f"{role}: {exc}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except M6Error:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        _fail("M6_JSON_PARSE", f"{role}: {exc}")
    _validate_json_domain(value, role=role)
    return value


def _read_regular(path: Path, *, role: str) -> bytes:
    try:
        return immutable.read_regular(Path(path))
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_INPUT_SNAPSHOT", f"{role}: {path}: {exc}")


def load_json(
    path: Path,
    *,
    role: str,
    cache: InputSnapshotCache | None = None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[Any, bytes]:
    payload = _read_regular_cached(
        path,
        role=role,
        cache=cache,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
    return loads_strict(payload, role=role), payload


def _repo_relative(path: Path, repo_root: Path) -> str:
    repository = _resolve_no_symlink_chain(repo_root, role="repository root")
    actual = _resolve_no_symlink_chain(path, role="repository file")
    try:
        relative = actual.relative_to(repository)
    except ValueError as exc:
        _fail("M6_SOURCE_OUTSIDE_REPO", f"{path}: {exc}")
    return relative.as_posix()


def _file_identity(
    path: Path,
    repo_root: Path,
    payload: bytes | None = None,
    *,
    cache: InputSnapshotCache | None = None,
    role: str = "source identity",
) -> dict[str, Any]:
    if payload is None:
        data = _read_regular_cached(path, role=role, cache=cache)
    elif cache is not None:
        data = cache.read(path, role=role)
        if data != payload:
            _fail("M6_INPUT_SNAPSHOT_MIXED", f"{role}: supplied payload changed")
    else:
        data = payload
    return {
        "path": _repo_relative(path, repo_root),
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }


def _resolve_repo_file(
    identity: Mapping[str, Any],
    repo_root: Path,
    *,
    role: str,
    cache: InputSnapshotCache | None = None,
) -> bytes:
    rel = _safe_relative(str(identity.get("path", "")), role=role)
    path = repo_root / rel
    payload = _read_regular_cached(
        path,
        role=role,
        cache=cache,
        expected_sha256=identity.get("sha256"),
        expected_size=identity.get("size_bytes"),
    )
    if len(payload) != identity.get("size_bytes") or sha256_bytes(payload) != identity.get("sha256"):
        _fail("M6_SOURCE_IDENTITY_MISMATCH", role)
    return payload


def _load_frozen_source_alias(
    *,
    repo_root: Path,
    cache: InputSnapshotCache | None = None,
) -> tuple[dict[str, dict[str, Any]], bytes]:
    """Load the admitted local alias table for frozen-v1 source payloads.

    The historical Git object is unavailable in this checkout.  This table is
    the only accepted relocation path: its own bytes are pinned, and every
    entry independently pins the canonical regular file's bytes and size.
    """
    alias_relative = FROZEN_SOURCE_ALIAS_PATH.relative_to(REPO_ROOT)
    alias_file = _canonical_repo_file(
        repo_root / alias_relative,
        FROZEN_SOURCE_ALIAS_PATH,
        repo_root=repo_root,
        role="frozen source alias table",
    )
    alias_payload = _read_regular_cached(
        alias_file,
        role="frozen source alias table",
        cache=cache,
        expected_sha256=FROZEN_SOURCE_ALIAS_SHA256,
    )
    value = loads_strict(alias_payload, role="frozen source alias table")
    if not isinstance(value, dict):
        _fail("M6_SOURCE_ALIAS_SHAPE", "alias table must be an object")
    expected_top_level = {"$id", "alias_id", "entries", "schema_version"}
    if set(value) != expected_top_level:
        _fail("M6_SOURCE_ALIAS_SHAPE", "alias table keys differ")
    if value.get("$id") != "simbench_bw/data/frozen_v1_source_aliases.v2.json":
        _fail("M6_SOURCE_ALIAS_IDENTITY", "alias table $id differs")
    if value.get("alias_id") != "sm86_reference_capture_matrix_v1_source_aliases_v2":
        _fail("M6_SOURCE_ALIAS_IDENTITY", "alias table id differs")
    if value.get("schema_version") != FROZEN_SOURCE_ALIAS_SCHEMA_VERSION:
        _fail("M6_SOURCE_ALIAS_VERSION", repr(value.get("schema_version")))
    rows = value.get("entries")
    if not isinstance(rows, list) or len(rows) != 2:
        _fail("M6_SOURCE_ALIAS_SHAPE", "exactly two frozen source aliases are required")
    expected_paths = {
        REGISTRY_PATH.relative_to(REPO_ROOT).as_posix(): (
            FROZEN_REGISTRY_PATH.relative_to(REPO_ROOT).as_posix()
        ),
        SIDECAR_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix(): (
            FROZEN_SIDECAR_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix()
        ),
    }
    aliases: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            _fail("M6_SOURCE_ALIAS_SHAPE", "alias entry must be an object")
        expected_keys = {
            "canonical_path",
            "historical_path",
            "match",
            "requires_sha256_match",
            "sha256",
            "size_bytes",
            "source_commit",
        }
        if set(row) != expected_keys:
            _fail("M6_SOURCE_ALIAS_SHAPE", "alias entry keys differ")
        historical = row.get("historical_path")
        canonical = row.get("canonical_path")
        if (
            not isinstance(historical, str)
            or not isinstance(canonical, str)
            or expected_paths.get(historical) != canonical
        ):
            _fail("M6_SOURCE_ALIAS_PATH", repr((historical, canonical)))
        if row.get("match") != "exact" or row.get("requires_sha256_match") is not True:
            _fail("M6_SOURCE_ALIAS_POLICY", historical)
        if row.get("source_commit") != FROZEN_V1_SOURCE_COMMIT:
            _fail("M6_SOURCE_ALIAS_COMMIT", historical)
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if not _sha256_text(digest) or type(size) is not int or size <= 0:
            _fail("M6_SOURCE_ALIAS_IDENTITY", historical)
        if historical in aliases:
            _fail("M6_SOURCE_ALIAS_DUPLICATE", historical)
        aliases[historical] = dict(row)
    if set(aliases) != set(expected_paths):
        _fail("M6_SOURCE_ALIAS_COVERAGE", "registry and sidecar aliases are required")
    return aliases, alias_payload


def _frozen_source_payload(
    repo_root: Path,
    source_commit: str,
    relative_path: str,
    *,
    role: str,
    cache: InputSnapshotCache | None = None,
) -> bytes:
    """Resolve one frozen source only through the hash-bound alias table."""
    if source_commit != FROZEN_V1_SOURCE_COMMIT:
        _fail("M6_SOURCE_ALIAS_COMMIT", f"{role}: {source_commit!r}")
    relative = _safe_relative(relative_path, role=role).as_posix()
    aliases, _alias_payload = _load_frozen_source_alias(
        repo_root=repo_root,
        cache=cache,
    )
    entry = aliases.get(relative)
    if entry is None:
        _fail("M6_SOURCE_ALIAS_MISSING", f"{role}: {relative}")
    canonical = entry["canonical_path"]
    expected = REPO_ROOT / _safe_relative(canonical, role=f"{role} canonical path")
    actual = _canonical_repo_file(
        repo_root / Path(canonical),
        expected,
        repo_root=repo_root,
        role=role,
    )
    payload = _read_regular_cached(
        actual,
        role=role,
        cache=cache,
        expected_sha256=entry["sha256"],
        expected_size=entry["size_bytes"],
    )
    if sha256_bytes(payload) != entry["sha256"] or len(payload) != entry["size_bytes"]:
        _fail("M6_SOURCE_ALIAS_IDENTITY", role)
    return payload


def _lexical_absolute(path: Path, *, role: str) -> Path:
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        _fail("M6_ABSOLUTE_PATH", f"{role}: lexical absolute path required: {path}")
    return path


def _resolve_no_symlink_chain(path: Path, *, role: str) -> Path:
    path = _lexical_absolute(path, role=role)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            observed = current.lstat()
        except OSError as exc:
            _fail("M6_PATH_IDENTITY", f"{role}: {current}: {exc}")
        if stat.S_ISLNK(observed.st_mode):
            _fail("M6_PATH_SYMLINK", f"{role}: {current}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        _fail("M6_PATH_IDENTITY", f"{role}: {path}: {exc}")


def _require_pinned_directory(path: Path, *, role: str) -> None:
    try:
        pinned = immutable.PinnedDirectoryReader(path)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_OUTPUT_PARENT", f"{role}: {path}: {exc}")
    try:
        pinned.verify_path()
    finally:
        pinned.close()


def _canonical_repo_root(repo_root: Path) -> Path:
    actual = _resolve_no_symlink_chain(repo_root, role="repository root")
    expected = _resolve_no_symlink_chain(REPO_ROOT, role="canonical repository root")
    actual_info = actual.stat()
    expected_info = expected.stat()
    if actual != expected or (
        actual_info.st_dev,
        actual_info.st_ino,
    ) != (
        expected_info.st_dev,
        expected_info.st_ino,
    ):
        _fail("M6_REPOSITORY_AUTHORITY", f"expected {expected}, observed {actual}")
    return actual


def _contains_path(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def _external_m4_root(archive_root: Path, *, repo_root: Path) -> Path:
    archive = _resolve_no_symlink_chain(archive_root, role="M4 archive root")
    repository = _canonical_repo_root(repo_root)
    if _contains_path(repository, archive) or _contains_path(archive, repository):
        _fail(
            "M6_M4_ARCHIVE_NOT_EXTERNAL",
            f"archive={archive} repository={repository}",
        )
    return archive


def _authorize_m4_receipt_output(
    output: Path, *, archive_root: Path, repo_root: Path,
) -> immutable.PinnedDirectoryReader:
    output = _lexical_absolute(output, role="M4 receipt output")
    parent = _resolve_no_symlink_chain(output.parent, role="M4 receipt parent")
    try:
        pinned = immutable.PinnedDirectoryReader(parent)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_OUTPUT_PARENT", f"{parent}: {exc}")
    try:
        pinned.verify_path()
        if _entry_metadata(pinned.descriptor, output.name) is not None:
            _fail("M6_NO_OVERWRITE", str(output))
        archive = _external_m4_root(archive_root, repo_root=repo_root)
        repository = _canonical_repo_root(repo_root)
        candidate = parent / output.name
        for role, protected in (("M4 archive", archive), ("repository", repository)):
            if _contains_path(protected, candidate) or _contains_path(candidate, protected):
                _fail(
                    "M6_M4_RECEIPT_OUTPUT_OVERLAP",
                    f"output={candidate} overlaps {role}={protected}",
                )
        pinned.verify_path()
        return pinned
    except BaseException:
        pinned.close()
        raise

def _safe_relative(raw: str, *, role: str) -> Path:
    if not isinstance(raw, str):
        _fail("M6_RELATIVE_PATH", f"{role}: {raw!r}")
    try:
        pure = immutable.safe_relative_path(raw)
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_RELATIVE_PATH", f"{role}: {raw!r}: {exc}")
    return Path(*pure.parts)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(value: Any, *, role: str) -> float:
    if not _is_number(value):
        _fail("M6_NUMERIC_VALUE", f"{role}: JSON number required (booleans forbidden)")
    return float(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _artifact_schema_version(value: Any, *, kind: str) -> int:
    if not isinstance(value, Mapping) or type(value.get("schema_version")) is not int:
        _fail("M6_SCHEMA_VERSION", f"{kind}: integer schema_version required")
    version = value["schema_version"]
    if version not in SCHEMA_PATHS_BY_VERSION:
        _fail("M6_SCHEMA_VERSION", f"{kind}: unsupported version {version}")
    return version


def _load_schema(
    kind: str,
    repo_root: Path = REPO_ROOT,
    *,
    version: int = ACTIVE_SCHEMA_VERSION,
    cache: InputSnapshotCache | None = None,
) -> tuple[dict[str, Any], bytes]:
    paths = SCHEMA_PATHS_BY_VERSION.get(version)
    if paths is None or kind not in paths:
        _fail("M6_SCHEMA_VERSION", f"{kind}: unsupported version {version}")
    relative = paths[kind].relative_to(REPO_ROOT)
    repository = _canonical_repo_root(repo_root)
    path = _canonical_repo_file(
        repository / relative,
        paths[kind],
        repo_root=repository,
        role=f"{kind} schema v{version}",
    )
    expected_sha = SCHEMA_SHA256_BY_VERSION[version][kind]
    value, payload = load_json(
        path,
        role=f"{kind} schema v{version}",
        cache=cache,
        expected_sha256=expected_sha,
    )
    if not isinstance(value, dict):
        _fail("M6_SCHEMA_SHAPE", f"{kind} v{version}")
    expected_id = SCHEMA_IDS_BY_VERSION[version][kind]
    if value.get("$id") != expected_id:
        _fail("M6_SCHEMA_IDENTITY", f"{kind} v{version}: $id")
    try:
        Draft202012Validator.check_schema(value)
    except Exception as exc:
        _fail("M6_SCHEMA_INVALID", f"{kind} v{version}: {exc}")
    return value, payload


def _validate_schema(
    kind: str,
    value: Any,
    repo_root: Path = REPO_ROOT,
    *,
    cache: InputSnapshotCache | None = None,
    require_active: bool = False,
) -> None:
    version = _artifact_schema_version(value, kind=kind)
    if require_active and version != ACTIVE_SCHEMA_VERSION:
        _fail("M6_LEGACY_SCHEMA", f"{kind} v{version} cannot authorize publication")
    schema, _ = _load_schema(kind, repo_root, version=version, cache=cache)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(value), key=lambda err: list(err.absolute_path))
    if errors:
        err = errors[0]
        location = "/" + "/".join(str(part) for part in err.absolute_path)
        _fail("M6_SCHEMA_VALIDATION", f"{kind}{location}: {err.message}")


def _require_active_schema(kind: str, value: Any) -> None:
    version = _artifact_schema_version(value, kind=kind)
    if version != ACTIVE_SCHEMA_VERSION:
        _fail("M6_LEGACY_SCHEMA", f"{kind} v{version} cannot authorize publication")


def _canonical_repo_file(
    path: Path, expected: Path, *, repo_root: Path, role: str,
) -> Path:
    repository = _canonical_repo_root(repo_root)
    actual = _resolve_no_symlink_chain(path, role=role)
    canonical = _resolve_no_symlink_chain(
        repository / expected.relative_to(REPO_ROOT), role=f"canonical {role}",
    )
    actual_info = actual.stat()
    canonical_info = canonical.stat()
    if actual != canonical or (
        actual_info.st_dev,
        actual_info.st_ino,
    ) != (
        canonical_info.st_dev,
        canonical_info.st_ino,
    ):
        _fail("M6_SOURCE_AUTHORITY", f"{role}: expected {canonical}, observed {actual}")
    return actual


def _load_active_sidecar_schema(
    *,
    repo_root: Path,
    cache: InputSnapshotCache | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Load the pinned active schema used to validate newly captured sidecars."""
    schema_file = _canonical_repo_file(
        repo_root / SIDECAR_SCHEMA_PATH.relative_to(REPO_ROOT),
        SIDECAR_SCHEMA_PATH,
        repo_root=repo_root,
        role="active v5 sidecar schema",
    )
    payload = _read_regular_cached(
        schema_file,
        role="active v5 sidecar schema",
        cache=cache,
        expected_sha256=sidecar_schema_authority.SCHEMA_SHA256,
    )
    try:
        schema = sidecar_schema_authority.parse_authoritative_schema_bytes(payload)
        Draft202012Validator.check_schema(schema)
    except (SchemaError, TypeError, ValueError) as exc:
        _fail("M6_M4_SIDECAR_SCHEMA_AUTHORITY", str(exc))
    return schema, payload


def _load_canonical_matrix(
    matrix_path: Path,
    *,
    repo_root: Path,
    require_semantic_rederivation: bool,
    cache: InputSnapshotCache | None = None,
) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    matrix_file = _canonical_repo_file(
        matrix_path, MATRIX_PATH, repo_root=repo_root, role="canonical matrix",
    )
    value, payload = load_json(matrix_file, role="canonical matrix", cache=cache)
    if not isinstance(value, dict):
        _fail("M6_SOURCE_SHAPE", "canonical matrix must be an object")
    schema_file = _canonical_repo_file(
        repo_root / MATRIX_SCHEMA_PATH.relative_to(REPO_ROOT),
        MATRIX_SCHEMA_PATH,
        repo_root=repo_root,
        role="canonical matrix schema",
    )
    schema, schema_payload = load_json(schema_file, role="canonical matrix schema", cache=cache)
    if not isinstance(schema, dict):
        _fail("M6_SCHEMA_SHAPE", "canonical matrix schema")
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(value), key=lambda err: list(err.absolute_path))
    except M6Error:
        raise
    except Exception as exc:
        _fail("M6_MATRIX_SCHEMA_INVALID", f"{type(exc).__name__}: {exc}")
    if errors:
        err = errors[0]
        location = "/" + "/".join(str(part) for part in err.absolute_path)
        _fail("M6_MATRIX_SCHEMA", f"{location}: {err.message}")
    if require_semantic_rederivation:
        try:
            semantic_problems = matrix_builder.validate_matrix(
                value,
                repo_root=repo_root,
                validation_mode="frozen_v1",
            )
        except Exception as exc:  # authority decoder boundary
            _fail("M6_MATRIX_REDERIVATION", f"{type(exc).__name__}: {exc}")
        if semantic_problems:
            _fail("M6_MATRIX_REDERIVATION", "; ".join(semantic_problems))
    if cache is not None:
        _read_regular_cached(
            matrix_file,
            role="canonical matrix post-validation",
            cache=cache,
            expected_sha256=sha256_bytes(payload),
            expected_size=len(payload),
        )
        _read_regular_cached(
            schema_file,
            role="canonical matrix schema post-validation",
            cache=cache,
            expected_sha256=sha256_bytes(schema_payload),
            expected_size=len(schema_payload),
        )
    return value, payload, _matrix_semantics(value)


def _load_matrix_authority_sources(
    matrix: Mapping[str, Any],
    *,
    repo_root: Path,
    cache: InputSnapshotCache,
) -> dict[str, Any]:
    """Bind the matrix's complete local authority closure.

    The matrix is a projection, not permission to omit the files that made the
    projection.  Every pinned local input is read once through the invocation
    cache.  The registry has an intentional frozen-source pin, so both the
    current authority and the exact alias-resolved frozen payload are retained.
    """
    def local_identity(relative: str, declared: str | None, role: str) -> dict[str, Any]:
        if not isinstance(relative, str) or not relative:
            _fail("M6_SOURCE_AUTHORITY", f"{role}: missing relative path")
        expected = REPO_ROOT / relative
        actual = _canonical_repo_file(
            repo_root / relative,
            expected,
            repo_root=repo_root,
            role=role,
        )
        payload = _read_regular_cached(actual, role=role, cache=cache)
        digest = sha256_bytes(payload)
        if declared is not None and digest != declared:
            _fail("M6_INPUT_PIN_IDENTITY", f"{role}: declared digest differs")
        return {
            "path": relative,
            "sha256": digest,
            "size_bytes": len(payload),
        }

    authority = matrix.get("authority")
    if not isinstance(authority, Mapping):
        _fail("M6_MATRIX_AUTHORITY", "roster authority is absent")
    roster_path = authority.get("path")
    roster = local_identity(
        roster_path,
        authority.get("sha256"),
        "capture roster authority",
    )
    schema_relative = MATRIX_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix()
    matrix_schema = local_identity(
        schema_relative,
        None,
        "canonical matrix schema",
    )
    pins = matrix.get("input_pins")
    if not isinstance(pins, Mapping):
        _fail("M6_MATRIX_INPUT_PINS", "matrix input pins are absent")
    pin_rows: list[dict[str, Any]] = []
    for name, pin in sorted(pins.items()):
        if not isinstance(pin, Mapping) or not isinstance(pin.get("path"), str):
            _fail("M6_MATRIX_INPUT_PINS", str(name))
        # The metric registry is validated against its frozen source alias
        # below; its active authority is already bound separately.
        if name == "metric_registry":
            continue
        identity = local_identity(
            pin["path"],
            pin.get("sha256"),
            f"matrix input pin {name}",
        )
        pin_rows.append({"name": str(name), **identity})
    registry_pin = pins.get("metric_registry")
    if not isinstance(registry_pin, Mapping):
        _fail("M6_MATRIX_REGISTRY_PIN", "metric registry pin is absent")
    registry_path = registry_pin.get("path")
    expected_registry_path = REGISTRY_PATH.relative_to(REPO_ROOT).as_posix()
    if registry_path != expected_registry_path:
        _fail("M6_MATRIX_REGISTRY_PIN", "metric registry path differs")
    source_commit = matrix.get("source_commit")
    if source_commit != FROZEN_V1_SOURCE_COMMIT:
        _fail("M6_MATRIX_SOURCE_COMMIT", repr(source_commit))
    frozen_payload = _frozen_source_payload(
        repo_root,
        source_commit,
        expected_registry_path,
        role="frozen matrix metric registry",
        cache=cache,
    )
    frozen_digest = sha256_bytes(frozen_payload)
    if registry_pin.get("sha256") != frozen_digest:
        _fail("M6_MATRIX_REGISTRY_PIN", "frozen registry digest differs")
    frozen_registry = {
        "path": expected_registry_path,
        "source_commit": source_commit,
        "sha256": frozen_digest,
        "size_bytes": len(frozen_payload),
    }
    frozen_sidecar_relative = SIDECAR_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix()
    frozen_sidecar_payload = _frozen_source_payload(
        repo_root,
        source_commit,
        frozen_sidecar_relative,
        role="frozen matrix sidecar schema",
        cache=cache,
    )
    frozen_sidecar = {
        "path": frozen_sidecar_relative,
        "source_commit": source_commit,
        "sha256": sha256_bytes(frozen_sidecar_payload),
        "size_bytes": len(frozen_sidecar_payload),
    }
    pin_rows.sort(key=lambda row: row["name"])
    return {
        "matrix_schema": matrix_schema,
        "roster_authority": roster,
        "matrix_input_pins": pin_rows,
        "frozen_metric_registry": frozen_registry,
        "frozen_sidecar_schema": frozen_sidecar,
    }
def _load_sm86_probe(
    *,
    repo_root: Path,
    cache: InputSnapshotCache,
) -> tuple[set[str], dict[str, Any], bytes]:
    probe_path = _canonical_repo_file(
        PROBE_PATH,
        PROBE_PATH,
        repo_root=repo_root,
        role="sm86 metric availability probe",
    )
    value, payload = load_json(
        probe_path,
        role="sm86 metric availability probe",
        cache=cache,
    )
    if not isinstance(value, dict) or value.get("arch") != "sm_86":
        _fail("M6_PROBE_IDENTITY", "sm_86 availability probe is required")
    rows = value.get("metrics")
    if not isinstance(rows, list):
        _fail("M6_PROBE_IDENTITY", "probe metrics must be an array")
    keys: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("M6_PROBE_IDENTITY", "probe metric row must be an object")
        key = row.get("registry_key")
        if key is None:
            continue
        if not isinstance(key, str) or not key or key in keys:
            _fail("M6_PROBE_IDENTITY", repr(key))
        keys.add(key)
    return (
        keys,
        {
            "path": PROBE_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        },
        payload,
    )


def _entry_metadata(parent_descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _pretty_json_bytes(value: Any) -> bytes:
    _validate_json_domain(value, role="presentation JSON")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError, RecursionError, UnicodeEncodeError) as exc:
        _fail("M6_PRESENTATION_JSON", str(exc))


def _write_new(
    path: Path,
    value: Any,
    *,
    pinned_parent: immutable.PinnedDirectoryReader | None = None,
) -> None:
    """Publish one M6 JSON output through the shared immutable helper."""
    payload = _pretty_json_bytes(value)
    output = Path(os.path.abspath(os.fspath(path)))
    if output.parent == output:
        _fail("M6_OUTPUT_PARENT", str(output))
    try:
        immutable.publish_regular_file(
            output.parent,
            output.name,
            payload,
            partial_suffix=".m6-partial",
            pinned_parent=pinned_parent,
        )
    except immutable.ImmutableArtifactError as exc:
        detail = str(exc)
        if "publication parent is unavailable" in detail:
            _fail("M6_OUTPUT_PARENT", f"{output.parent}: {detail}")
        if "publication already exists" in detail:
            _fail("M6_NO_OVERWRITE", str(output))
        if "partial" in detail and "preserved" in detail:
            _fail("M6_PARTIAL_PRESERVED", str(output.parent / f".{output.name}.m6-partial"))
        if "short immutable-artifact write" in detail or "publication I/O failed" in detail:
            _fail("M6_OUTPUT_WRITE", f"{output}: {detail}")
        if detail.startswith(immutable.DURABILITY_UNPROVEN):
            _fail(immutable.DURABILITY_UNPROVEN, f"{output}: {detail}")
        if "named regular-file path changed" in detail or "locked deterministic partial" in detail:
            _fail("M6_OUTPUT_REPLACED", f"{output}: {detail}")
        _fail("M6_OUTPUT_VERIFY", f"{output}: {detail}")


class _PinnedTreeTransaction:
    """Read one evidence tree against one stable no-follow inventory snapshot."""

    def __init__(
        self,
        root: Path,
        *,
        role: str,
        require_readonly_files: bool = False,
        require_readonly_directories: bool = False,
        input_cache: InputSnapshotCache | None = None,
    ) -> None:
        self.root = Path(os.path.abspath(os.fspath(root)))
        self.role = role
        self.require_readonly_files = require_readonly_files
        self.require_readonly_directories = require_readonly_directories
        self.input_cache = input_cache
        try:
            self.reader = immutable.PinnedDirectoryReader(self.root)
        except (OSError, immutable.ImmutableArtifactError) as exc:
            _fail("M6_TREE_SNAPSHOT", f"{role}: {self.root}: {exc}")
        try:
            self._initial = self._snapshot_inventory()
        except BaseException:
            self.reader.close()
            raise

    def _snapshot_inventory(
        self,
    ) -> dict[
        str,
        tuple[str, immutable.MetadataSnapshot],
    ]:
        try:
            snapshot = self.reader.snapshot_tree(
                require_readonly_files=self.require_readonly_files,
                require_readonly_directories=self.require_readonly_directories,
            )
        except (OSError, immutable.ImmutableArtifactError) as exc:
            _fail("M6_TREE_SNAPSHOT", f"{self.role}: {exc}")
        observed = {
            "" if path == "." else path: ("directory", metadata)
            for path, metadata in snapshot.directories
        }
        observed.update(
            (path, ("file", metadata))
            for path, metadata in snapshot.files
        )
        return observed

    @property
    def inventory(self) -> Mapping[
        str, tuple[str, immutable.MetadataSnapshot]
    ]:
        return self._initial

    def read(self, relative_path: str, *, role: str) -> bytes:
        try:
            relative = immutable.safe_relative_path(relative_path).as_posix()
            initial = self._initial.get(relative)
            if initial is None or initial[0] != "file":
                raise immutable.ImmutableArtifactError(
                    f"pinned regular file is absent from initial inventory: {relative}"
                )
            expected = initial[1]
            if self.input_cache is None:
                return self.reader.read(relative)
            self.reader.verify_regular(relative, expected)
            cached = self.input_cache.lookup_bound(
                self.root / relative,
                identity=expected,
                role=role,
            )
            if cached is not None:
                return cached
            snapshot = self.reader.read_snapshot(relative)
            if snapshot.metadata != expected:
                _fail(
                    "M6_INPUT_SNAPSHOT_MIXED",
                    f"{role}: changed since initial tree inventory",
                )
            return self.input_cache.bind_pinned(
                self.root / relative,
                snapshot.payload,
                identity=snapshot.metadata,
                role=role,
            )
        except (OSError, immutable.ImmutableArtifactError) as exc:
            code = "M6_INPUT_SNAPSHOT" if self.input_cache is not None else "M6_TREE_READ"
            _fail(code, f"{role}: {relative_path}: {exc}")

    def verify(self) -> None:
        final = self._snapshot_inventory()
        if final != self._initial:
            changed = sorted(set(final) ^ set(self._initial))
            if not changed:
                changed = sorted(
                    path
                    for path in final
                    if final[path] != self._initial[path]
                )
            _fail(
                "M6_TREE_CHANGED",
                f"{self.role}: changed paths={changed[:8]}",
            )

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> "_PinnedTreeTransaction":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def _validate_seal(value: Mapping[str, Any], field: str, *, role: str) -> None:
    if value.get(field) != canonical_sha256(_without(value, field)):
        _fail("M6_SEAL_MISMATCH", f"{role}.{field}")


def _matrix_semantics(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    _validate_seal(matrix, "matrix_seal_sha256", role="canonical_matrix")
    if matrix.get("matrix_id") != "sm86_reference_capture_matrix_v1":
        _fail("M6_MATRIX_ID", repr(matrix.get("matrix_id")))
    counts = matrix.get("counts")
    expected = {
        "universe": EXPECTED_UNIVERSE,
        "runnable": EXPECTED_RUNNABLE,
        "excluded": EXPECTED_EXCLUDED,
        "capture_ready": EXPECTED_RUNNABLE,
        "capture_blocked": 0,
    }
    if (
        not isinstance(counts, dict)
        or any(type(counts.get(key)) is not int for key in expected)
        or any(counts.get(key) != expected_value for key, expected_value in expected.items())
    ):
        _fail("M6_MATRIX_COUNTS", repr(counts))
    cells = matrix.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_UNIVERSE:
        _fail("M6_MATRIX_CELLS", "expected 344 cells")
    identities: set[tuple[str, str]] = set()
    runnable: list[dict[str, Any]] = []
    excluded = 0
    for cell in cells:
        if not isinstance(cell, dict):
            _fail("M6_MATRIX_CELL", "non-object")
        identity = cell.get("identity")
        if not isinstance(identity, dict):
            _fail("M6_MATRIX_CELL_IDENTITY", "missing")
        key = (identity.get("roster_source"), identity.get("slug"))
        if not all(isinstance(part, str) and part for part in key) or key in identities:
            _fail("M6_MATRIX_CELL_IDENTITY", repr(key))
        identities.add(key)  # type: ignore[arg-type]
        disposition = cell.get("disposition")
        if disposition == "runnable":
            if (cell.get("capture_readiness") or {}).get("status") != "ready":
                _fail("M6_MATRIX_RUNNABLE_NOT_READY", repr(key))
            if not isinstance(cell.get("invocation"), dict):
                _fail("M6_MATRIX_RUNNABLE_INVOCATION", repr(key))
            runnable.append(cell)
        elif disposition == "excluded":
            excluded += 1
        else:
            _fail("M6_MATRIX_DISPOSITION", repr(disposition))
    if len(runnable) != EXPECTED_RUNNABLE or excluded != EXPECTED_EXCLUDED:
        _fail("M6_MATRIX_PARTITION", f"{len(runnable)}/{excluded}")
    chunks = matrix.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 18:
        _fail("M6_MATRIX_CHUNKS", "expected 18")
    flat: list[dict[str, Any]] = []
    chunk_ids: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            _fail("M6_MATRIX_CHUNK", repr(chunk))
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in chunk_ids:
            _fail("M6_MATRIX_CHUNK_ID", repr(chunk_id))
        chunk_ids.add(chunk_id)
        members = chunk.get("cells")
        if not isinstance(members, list) or not members:
            _fail("M6_MATRIX_CHUNK_MEMBERS", repr(chunk))
        if chunk.get("membership_sha256") != canonical_sha256(members):
            _fail("M6_MATRIX_CHUNK_SEAL", chunk_id)
        flat.extend(members)
    runnable_keys = {
        (cell["identity"]["roster_source"], cell["identity"]["slug"])
        for cell in runnable
    }
    flat_keys = {(member.get("roster_source"), member.get("slug")) for member in flat}
    if len(flat_keys) != EXPECTED_RUNNABLE or flat_keys != runnable_keys:
        _fail("M6_MATRIX_RUNNABLE_MEMBERSHIP", "chunk coverage differs")
    if (matrix.get("chunking") or {}).get("runnable_membership_sha256") != canonical_sha256(flat):
        _fail("M6_MATRIX_RUNNABLE_SEAL", "mismatch")
    return runnable


def _metric_capture_status(metric: Mapping[str, Any]) -> str:
    per_arch = (metric.get("per_arch_availability") or {}).get("sm_86")
    if metric.get("availability") != "excluded" and per_arch == "available":
        return "selected"
    if per_arch == "requires_runtime_carveout":
        return "requires_runtime_carveout"
    if per_arch != "available":
        return "unavailable_on_sm86"
    return "static_derived_not_ncu_collected"


def _metric_statistical_contract(metric: Mapping[str, Any], *, key: str) -> tuple[str, float, str]:
    """Return the sealed per-key statistical operands.

    Unit and metric class are descriptive metadata only.  They are never used
    as a fallback for trigger/modality behavior: every registry key must appear
    in the explicit override sets above, and any future registry declaration
    must agree with that sealed per-key contract.
    """
    if key not in STATISTICAL_CONTRACT_KEYS:
        _fail("M6_METRIC_RESOLUTION_UNDECLARED", key)
    if key in STATISTICAL_DISCRETE_KEYS:
        domain = "discrete"
        resolution = 1.0
    elif key in STATISTICAL_CONTINUOUS_FINE_KEYS:
        domain = "continuous"
        resolution = 0.0001
    elif key in STATISTICAL_CONTINUOUS_UNIT_KEYS:
        domain = "continuous"
        resolution = 1.0
    else:
        _fail("M6_METRIC_RESOLUTION_UNDECLARED", key)
    expected_semantics = STATISTICAL_CONTRACT_VERSION
    declared_domain = metric.get("statistical_domain")
    declared_resolution = metric.get("absolute_resolution")
    declared_semantics = metric.get("resolution_semantics")
    if declared_domain is not None and declared_domain != domain:
        _fail("M6_METRIC_DOMAIN_DRIFT", key)
    if declared_resolution is not None:
        if not _is_number(declared_resolution) or float(declared_resolution) != resolution:
            _fail("M6_METRIC_RESOLUTION_DRIFT", key)
    if declared_semantics is not None and declared_semantics != expected_semantics:
        _fail("M6_METRIC_RESOLUTION_SEMANTICS", key)
    return domain, resolution, expected_semantics


def _registry_metrics(
    registry: Mapping[str, Any],
    matrix: Mapping[str, Any],
    *,
    probe_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    contract_rows = []
    for contract_key in sorted(STATISTICAL_CONTRACT_KEYS):
        domain, resolution, semantics = _metric_statistical_contract({}, key=contract_key)
        contract_rows.append({
            "key": contract_key,
            "statistical_domain": domain,
            "absolute_resolution": resolution,
            "resolution_semantics": semantics,
        })
    if canonical_sha256(contract_rows) != STATISTICAL_CONTRACT_SHA256:
        _fail("M6_STATISTICAL_CONTRACT_DRIFT", "sealed per-key contract changed")
    if len(contract_rows) != EXPECTED_REGISTRY_METRICS:
        _fail("M6_STATISTICAL_CONTRACT_DRIFT", "contract key count differs")
    raw = registry.get("metrics")
    if (
        registry.get("registry_version") != 2
        or registry.get("label_schema_version") != 5
        or not isinstance(raw, list)
        or len(raw) != EXPECTED_REGISTRY_METRICS
    ):
        _fail("M6_REGISTRY_IDENTITY", "registry version/count mismatch")
    thresholds = registry.get("repeatability_thresholds")
    if not isinstance(thresholds, dict):
        _fail("M6_REGISTRY_THRESHOLDS", "missing")
    matrix_rows = (matrix.get("metric_registry") or {}).get("metrics")
    matrix_by_key = {
        row.get("key"): row for row in matrix_rows if isinstance(row, dict)
    } if isinstance(matrix_rows, list) else {}
    identities: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    keys: set[str] = set()
    ncus: set[str] = set()
    for metric in raw:
        if not isinstance(metric, dict):
            _fail("M6_REGISTRY_METRIC", "non-object")
        key = metric.get("key")
        ncu = metric.get("ncu_metric")
        if not isinstance(key, str) or not key or key in keys:
            _fail("M6_REGISTRY_KEY", repr(key))
        if not isinstance(ncu, str) or not ncu or ncu in ncus:
            _fail("M6_REGISTRY_NCU", repr(ncu))
        keys.add(key)
        ncus.add(ncu)
        status = _metric_capture_status(metric)
        identity = {
            "key": key,
            "ncu_metric": ncu,
            "unit": metric.get("unit"),
            "carrier": metric.get("carrier"),
            "availability": metric.get("availability"),
            "sm86_availability": (metric.get("per_arch_availability") or {}).get("sm_86"),
            "capture_status": status,
        }
        identities.append(identity)
        if matrix_by_key.get(key) != identity:
            _fail("M6_METRIC_MATRIX_LOCKSTEP", key)
        rule = metric.get("repeatability_rule")
        threshold = thresholds.get(rule)
        if not _is_number(threshold):
            _fail("M6_METRIC_THRESHOLD", key)
        unit = metric.get("unit")
        statistical_domain, resolution, resolution_semantics = _metric_statistical_contract(
            metric, key=key,
        )
        parser_transform = metric.get("parser_transform")
        model_io_role = metric.get("model_io_role")
        deployability = metric.get("deployability")
        if parser_transform not in {"percent_to_fraction", "identity_nonneg"}:
            _fail("M6_METRIC_PARSER_TRANSFORM", key)
        if model_io_role not in {"target", "input", "either"}:
            _fail("M6_METRIC_ROLE", key)
        if deployability not in {"label_only", "feature_eligible"}:
            _fail("M6_METRIC_ROLE", key)
        expected_transform = (
            "percent_to_fraction" if metric.get("unit") == "pct" else "identity_nonneg"
        )
        if parser_transform != expected_transform:
            _fail("M6_METRIC_PARSER_TRANSFORM", key)
        probe_backed = key in probe_keys if probe_keys is not None else False
        expected_probe_linkage = "sm86_probe_v1" if probe_backed else "sm86_probe_unbacked"
        declared_probe_linkage = metric.get("probe_linkage")
        if declared_probe_linkage is not None and declared_probe_linkage != expected_probe_linkage:
            _fail("M6_METRIC_PROBE_LINKAGE", key)
        probe_linkage = expected_probe_linkage
        contract = {
            "key": key,
            "ncu_metric": ncu,
            "unit": unit,
            "carrier": metric.get("carrier"),
            "sidecar_field": metric.get("sidecar_field"),
            "family": metric.get("family"),
            "metric_class": metric.get("metric_class"),
            "parser_transform": parser_transform,
            "model_io_role": model_io_role,
            "deployability": deployability,
            "probe_linkage": probe_linkage,
            "probe_backed": probe_backed,
            "statistical_domain": statistical_domain,
            "resolution_semantics": resolution_semantics,
            "statistical_contract_sha256": STATISTICAL_CONTRACT_SHA256,
            "repeatability_rule": rule,
            "admission_threshold": threshold,
            "valid_min": metric.get("valid_min"),
            "valid_max": metric.get("valid_max"),
            "absolute_resolution": resolution,
            "capture_status": status,
        }
        if status == "selected" and (
            model_io_role != "target" or deployability != "label_only"
        ):
            _fail("M6_METRIC_ROLE", key)
        for field in ("admission_threshold", "valid_min", "absolute_resolution"):
            _number(contract[field], role=f"metric {key}.{field}")
        if contract["valid_max"] is not None:
            _number(contract["valid_max"], role=f"metric {key}.valid_max")
        contract["metric_sha256"] = canonical_sha256(contract)
        contracts.append(contract)
    identities.sort(key=lambda row: row["key"])
    expected_identity = (matrix.get("metric_registry") or {}).get("metric_identity_sha256")
    if canonical_sha256(identities) != expected_identity:
        _fail("M6_METRIC_IDENTITY", "semantic metric identity mismatch")
    contracts.sort(key=lambda row: row["key"])
    if sum(row["capture_status"] == "selected" for row in contracts) != EXPECTED_SELECTED_METRICS:
        _fail("M6_SELECTED_METRIC_COUNT", "expected 55")
    return contracts


def _configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for clock in ("none", "base", "boost"):
        for cache in ("all", "none"):
            reference = clock == "none" and cache == "all"
            config = {
                "config_id": f"cfg=clock_{clock}__cache_{cache}",
                "clock_control": clock,
                "cache_control": cache,
                "replay_mode": "kernel",
                "target_processes": "all",
                "filter_mode": "global",
                "profile_from_start": False,
                "reference": reference,
                "tier1_repeats": 20 if reference else 10,
                "fresh_process_per_profile": True,
            }
            config["config_sha256"] = canonical_sha256(config)
            configs.append(config)
    return configs


def _schedule_cells(runnable: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for raw in runnable:
        identity = raw["identity"]
        invocation = raw["invocation"]
        logical = invocation.get("logical_workload") or {}
        cell = {
            "cell_id": f"{identity['roster_source']}:{identity['slug']}",
            "roster_source": identity["roster_source"],
            "slug": identity["slug"],
            "shape_point": str(logical.get("value")),
            "authority_cell_sha256": raw["authority_cell_sha256"],
            "binary_sha256": invocation["executable_sha256"],
        }
        cell["cell_identity_sha256"] = canonical_sha256(cell)
        cells.append(cell)
    cells.sort(key=lambda row: row["cell_id"])
    return cells


def _make_slot(
    *,
    ordinal: int,
    tier: int,
    sweep: int,
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    # The ordinal is part of the identity, not presentation metadata.  This
    # prevents a resealed schedule from moving a slot while retaining its
    # apparent slot hash.
    core = {
        "ordinal": ordinal,
        "tier": tier,
        "sweep_index": sweep,
        "repeat_index": sweep,
        "cell_id": cell["cell_id"],
        "cell_identity_sha256": cell["cell_identity_sha256"],
        "config_id": config["config_id"],
        "config_sha256": config["config_sha256"],
        "process_isolation": "fresh_process",
    }
    identity = canonical_sha256(core)
    return {
        "ordinal": ordinal,
        "slot_id": f"m6-t{tier}-{identity[:24]}",
        **core,
        "slot_sha256": identity,
    }


def _shuffle_key(seed: int, sweep: int, tier: int, slot: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        f"{seed}\0{tier}\0{sweep}\0{slot['cell_id']}\0{slot['config_id']}".encode()
    ).hexdigest()


def _build_slots(
    cells: Sequence[Mapping[str, Any]], configs: Sequence[Mapping[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = next(config for config in configs if config["reference"])
    configs_by_id = {config["config_id"]: config for config in configs}
    cells_by_id = {cell["cell_id"]: cell for cell in cells}
    tier1: list[dict[str, Any]] = []
    ordinal = 0
    for sweep in range(20):
        sweep_configs = configs if sweep < 10 else [reference]
        pending = [
            _make_slot(ordinal=0, tier=1, sweep=sweep, cell=cell, config=config)
            for cell in cells
            for config in sweep_configs
        ]
        pending.sort(key=lambda slot: _shuffle_key(seed, sweep, 1, slot))
        for pending_slot in pending:
            cell = cells_by_id[pending_slot["cell_id"]]
            config = configs_by_id[pending_slot["config_id"]]
            tier1.append(
                _make_slot(
                    ordinal=ordinal,
                    tier=1,
                    sweep=sweep,
                    cell=cell,
                    config=config,
                )
            )
            ordinal += 1
    candidates: list[dict[str, Any]] = []
    for sweep in range(20, 30):
        pending = [
            _make_slot(ordinal=0, tier=2, sweep=sweep, cell=cell, config=reference)
            for cell in cells
        ]
        pending.sort(key=lambda slot: _shuffle_key(seed, sweep, 2, slot))
        for pending_slot in pending:
            cell = cells_by_id[pending_slot["cell_id"]]
            candidates.append(
                _make_slot(
                    ordinal=ordinal,
                    tier=2,
                    sweep=sweep,
                    cell=cell,
                    config=reference,
                )
            )
            ordinal += 1
    if len(tier1) != TIER1_SLOTS or len(candidates) != TIER2_CANDIDATES:
        _fail("M6_SLOT_COUNT", f"{len(tier1)}/{len(candidates)}")
    if [slot["ordinal"] for slot in tier1 + candidates] != list(range(MAXIMUM_SLOTS)):
        _fail("M6_SLOT_COORDINATE", "slot ordinals are not globally contiguous")
    return tier1, candidates


def _default_m4_identity(matrix: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "status": "incomplete",
        "archive_root": None,
        "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "roster_authority_sha256": matrix["authority"]["sha256"],
        "runnable_membership_sha256": matrix["chunking"]["runnable_membership_sha256"],
        "protocol_version": "2.1",
        "expected_cells": EXPECTED_RUNNABLE,
        "completed_cells": 0,
        "chunk_done_manifests": [],
        "cell_launch_identities": [],
    }
    value["identity_sha256"] = canonical_sha256(value)
    return value


def _sha256_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _m4_artifact(row: Mapping[str, Any], kind: str, *, cell_id: str) -> Mapping[str, Any]:
    artifacts = row.get("artifacts")
    matches = [
        artifact
        for artifact in artifacts if isinstance(artifact, dict)
        and artifact.get("artifact_kind") == kind
    ] if isinstance(artifacts, list) else []
    if len(matches) != 1:
        _fail("M6_M4_ARTIFACT_CARDINALITY", f"{cell_id}:{kind}:{len(matches)}")
    return matches[0]


def _load_m4_artifact(
    chunk_root: Path,
    artifact: Mapping[str, Any],
    *,
    role: str,
    archive_root: Path,
    archive_tree: _PinnedTreeTransaction,
) -> tuple[bytes, Path]:
    relative = _safe_relative(artifact.get("relative_path"), role=role)
    path = chunk_root / relative
    try:
        archive_relative = path.relative_to(archive_root).as_posix()
    except ValueError as exc:
        _fail("M6_M4_ARTIFACT_PATH", f"{role}: {exc}")
    payload = archive_tree.read(archive_relative, role=role)
    if (
        type(artifact.get("size_bytes")) is not int
        or artifact["size_bytes"] != len(payload)
        or not _sha256_text(artifact.get("sha256"))
        or artifact["sha256"] != sha256_bytes(payload)
    ):
        _fail("M6_M4_ARTIFACT_IDENTITY", role)
    return payload, path


def _m4_expected_controls(
    matrix_cell: Mapping[str, Any], launch_index: int,
) -> tuple[dict[str, Any], bool, str]:
    invocation = matrix_cell.get("invocation")
    if not isinstance(invocation, dict):
        _fail("M6_M4_INVOCATION", repr(matrix_cell.get("identity")))
    shape_delivery = invocation.get("shape_delivery")
    if shape_delivery == "application_args":
        profile_from_start = True
        shape_binding = "application-args"
    elif (
        shape_delivery == "adapter_consumes_logical_shape"
        and (matrix_cell.get("identity") or {}).get("roster_source")
        == "sol_execbench"
        and invocation.get("adapter_kind") == "sol_execbench"
    ):
        profile_from_start = False
        shape_binding = "logical-workload"
    else:
        _fail("M6_M4_SHAPE_DELIVERY", repr(shape_delivery))
    controls = {
        "target_processes": "all",
        "filter_mode": "global",
        "replay_mode": "kernel",
        "cache_control": "all",
        "clock_control": "none",
        "profile_from_start": "on" if profile_from_start else "off",
        "launch_skip": launch_index,
        "launch_count": 1,
    }
    return controls, profile_from_start, shape_binding


def _validate_m4_sidecar(
    sidecar: Mapping[str, Any],
    *,
    sidecar_schema: Mapping[str, Any],
    registry: Mapping[str, Any],
    matrix: Mapping[str, Any],
    matrix_cell: Mapping[str, Any],
    schedule_cell: Mapping[str, Any],
    launch_index: int,
) -> None:
    try:
        parsed_registry = metric_registry_lib.parse_registry(dict(registry))
        validation = label_import.validate_with_bound_authorities(
            sidecar,
            schema=dict(sidecar_schema),
            registry=parsed_registry,
        )
    except (TypeError, ValueError) as exc:
        _fail(
            "M6_M4_REGISTRY_AUTHORITY",
            f"{schedule_cell['cell_id']}: {exc}",
        )
    if not validation.ok:
        _fail(
            "M6_M4_SIDECAR_VALIDATION",
            f"{schedule_cell['cell_id']}: {validation.typed_reason}: {validation.detail}",
        )
    expected = {
        "schema_version": 5,
        "device": "sm_86",
        "slug": schedule_cell["slug"],
        "shape_point": schedule_cell["shape_point"],
        "repeat_index": 0,
        "launch_index": launch_index,
        "binary_sha256": schedule_cell["binary_sha256"],
    }
    controls, profile_from_start, _ = _m4_expected_controls(
        matrix_cell, launch_index,
    )
    protocol = sidecar.get("measurement_protocol")
    expected_protocol = {
        "ncu_target_processes": controls["target_processes"],
        "ncu_filter_mode": controls["filter_mode"],
        "ncu_replay_mode": controls["replay_mode"],
        "ncu_cache_control": controls["cache_control"],
        "ncu_clock_control": controls["clock_control"],
        "ncu_profile_from_start": profile_from_start,
        "ncu_launch_skip": launch_index,
        "ncu_launch_count": 1,
    }
    collection = (matrix.get("metric_registry") or {}).get("collection") or {}
    selected_unit = collection.get("selected_unit_metrics")
    selected_flat = collection.get("selected_flat_metrics")
    observed_unit = [
        row.get("ncu_metric")
        for row in sidecar.get("unit_tagged_metrics", [])
        if isinstance(row, dict)
    ]
    registry_rows = registry.get("metrics")
    flat_fields = {
        row.get("sidecar_field")
        for row in registry_rows
        if isinstance(row, dict)
        and row.get("carrier") == "flat_field"
        and row.get("ncu_metric") in set(selected_flat or [])
    } if isinstance(registry_rows, list) else set()
    if (
        any(sidecar.get(key) != value for key, value in expected.items())
        or not isinstance(protocol, dict)
        or any(protocol.get(key) != value for key, value in expected_protocol.items())
        or not isinstance(selected_unit, list)
        or observed_unit != selected_unit
        or not isinstance(selected_flat, list)
        or len(flat_fields) != len(selected_flat)
        or any(field not in sidecar for field in flat_fields)
    ):
        _fail("M6_M4_SIDECAR_BINDING", schedule_cell["cell_id"])


def _validate_m4_capture_manifest(
    capture: Mapping[str, Any],
    *,
    records: Mapping[str, Mapping[str, Any]],
    observation: Mapping[str, Any],
    run: Mapping[str, Any],
    matrix: Mapping[str, Any],
    matrix_cell: Mapping[str, Any],
    schedule_cell: Mapping[str, Any],
    source_commit: str,
    registry_payload: bytes,
) -> None:
    fields = {
        "schema_version", "identity", "command", "binary_sha256",
        "source_commit", "kernel_signature_sha256",
        "kernel_signature_source", "validated_artifact_receipt",
        "selected_unit_metrics", "selected_flat_metrics",
        "toolchain_contract", "publication_metric_scope",
        "logical_workload_sha256", "capture_launch_contract",
        "requested_launch_index", "ncu_controls", "shape_binding",
        "shape_forwarded_to_target", "sol_phase_evidence",
        "elapsed_seconds",
    }
    launch_index = run["launch_index"]
    controls, _profile_from_start, shape_binding = _m4_expected_controls(
        matrix_cell, launch_index,
    )
    invocation = matrix_cell["invocation"]
    logical = invocation["logical_workload"]
    expected_signature = hashlib.sha256(
        f"{schedule_cell['slug']}|{invocation['executable_path']}".encode("utf-8")
    ).hexdigest()
    expected_receipt = {
        "label_sidecar": {
            "sha256": records["label_sidecar"]["sha256"],
            "size_bytes": records["label_sidecar"]["size_bytes"],
        },
        "label_run_manifest": {
            "sha256": records["label_run_manifest"]["sha256"],
            "size_bytes": records["label_run_manifest"]["size_bytes"],
        },
        "ncu_report": {
            "sha256": records["ncu_report"]["sha256"],
            "size_bytes": records["ncu_report"]["size_bytes"],
        },
        "ncu_launch_observation": {
            "sha256": records["ncu_launch_observation"]["sha256"],
            "size_bytes": records["ncu_launch_observation"]["size_bytes"],
            "observation_sha256": observation["observation_sha256"],
        },
        "metric_registry": {
            "sha256": sha256_bytes(registry_payload),
            "size_bytes": len(registry_payload),
        },
    }
    toolchain = capture.get("toolchain_contract")
    command = capture.get("command")
    elapsed = capture.get("elapsed_seconds")
    collection = matrix["metric_registry"]["collection"]
    if (
        set(capture) != fields
        or capture.get("schema_version") != 1
        or capture.get("identity") != matrix_cell.get("identity")
        or not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) for item in command)
        or capture.get("binary_sha256") != schedule_cell["binary_sha256"]
        or capture.get("source_commit") != source_commit
        or capture.get("kernel_signature_sha256") != expected_signature
        or capture.get("kernel_signature_source") != "fallback_legacy"
        or run.get("kernel_signature_sha256") != expected_signature
        or run.get("kernel_signature_source") != "fallback_legacy"
        or capture.get("validated_artifact_receipt") != expected_receipt
        or capture.get("selected_unit_metrics")
        != collection["selected_unit_metrics"]
        or capture.get("selected_flat_metrics")
        != collection["selected_flat_metrics"]
        or not isinstance(toolchain, dict)
        or set(toolchain) != {
            "expected_cuda_release", "observed_cuda_release",
            "expected_ncu_version_prefix", "observed_ncu_version",
        }
        or capture.get("publication_metric_scope") is not None
        or capture.get("logical_workload_sha256") != canonical_sha256(logical)
        or capture.get("capture_launch_contract")
        != invocation.get("capture_launch_contract")
        or capture.get("requested_launch_index") != launch_index
        or capture.get("ncu_controls") != controls
        or capture.get("shape_binding") != shape_binding
        or capture.get("shape_forwarded_to_target")
        is not (shape_binding == "application-args")
        or not _is_number(elapsed)
        or float(elapsed) < 0
    ):
        _fail("M6_M4_CAPTURE_MANIFEST_BINDING", schedule_cell["cell_id"])


def _derive_m4_launch(
    *,
    row: Mapping[str, Any],
    chunk_root: Path,
    archive_root: Path,
    archive_tree: _PinnedTreeTransaction,
    matrix: Mapping[str, Any],
    matrix_cell: Mapping[str, Any],
    schedule_cell: Mapping[str, Any],
    source_commit: str,
    sidecar_schema: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_payload: bytes,
) -> dict[str, Any]:
    cell_id = schedule_cell["cell_id"]
    records = {
        kind: _m4_artifact(row, kind, cell_id=cell_id)
        for kind in (
            "label_sidecar",
            "label_run_manifest",
            "ncu_launch_observation",
            "ncu_report",
            "label_capture_manifest",
        )
    }
    sidecar_record = records["label_sidecar"]
    run_record = records["label_run_manifest"]
    observation_record = records["ncu_launch_observation"]
    report_record = records["ncu_report"]
    capture_record = records["label_capture_manifest"]
    sidecar_raw, _ = _load_m4_artifact(
        chunk_root,
        sidecar_record,
        role=f"M4 sidecar {cell_id}",
        archive_root=archive_root,
        archive_tree=archive_tree,
    )
    run_raw, _ = _load_m4_artifact(
        chunk_root,
        run_record,
        role=f"M4 run manifest {cell_id}",
        archive_root=archive_root,
        archive_tree=archive_tree,
    )
    observation_raw, _ = _load_m4_artifact(
        chunk_root,
        observation_record,
        role=f"M4 launch observation {cell_id}",
        archive_root=archive_root,
        archive_tree=archive_tree,
    )
    _load_m4_artifact(
        chunk_root,
        report_record,
        role=f"M4 NCU report {cell_id}",
        archive_root=archive_root,
        archive_tree=archive_tree,
    )
    capture_raw, _ = _load_m4_artifact(
        chunk_root,
        capture_record,
        role=f"M4 capture manifest {cell_id}",
        archive_root=archive_root,
        archive_tree=archive_tree,
    )
    sidecar = loads_strict(sidecar_raw, role=f"M4 sidecar {cell_id}")
    run = loads_strict(run_raw, role=f"M4 run manifest {cell_id}")
    observation = loads_strict(
        observation_raw, role=f"M4 launch observation {cell_id}",
    )
    capture = loads_strict(capture_raw, role=f"M4 capture manifest {cell_id}")
    if not all(
        isinstance(value, dict) for value in (sidecar, run, observation, capture)
    ):
        _fail("M6_M4_LAUNCH_DOCUMENT", cell_id)
    launch_index = sidecar.get("launch_index")
    logical = (matrix_cell.get("invocation") or {}).get("logical_workload") or {}
    launch_contract = (matrix_cell.get("invocation") or {}).get(
        "capture_launch_contract"
    )
    expected_launch_index = (
        launch_contract.get("label_launch_index")
        if isinstance(launch_contract, dict) else None
    )
    if (
        type(launch_index) is not int
        or launch_index < 0
        or type(expected_launch_index) is not int
        or launch_index != expected_launch_index
        or str(logical.get("value")) != schedule_cell["shape_point"]
    ):
        _fail("M6_M4_SIDECAR_BINDING", cell_id)
    _validate_m4_sidecar(
        sidecar,
        sidecar_schema=sidecar_schema,
        registry=registry,
        matrix=matrix,
        matrix_cell=matrix_cell,
        schedule_cell=schedule_cell,
        launch_index=launch_index,
    )
    controls, _profile_from_start, _shape_binding = _m4_expected_controls(
        matrix_cell, launch_index,
    )
    expected_run = {
        "schema_version": 1,
        "source_commit": source_commit,
        "binary_sha256": schedule_cell["binary_sha256"],
        "arch": "sm_86",
        "slug": schedule_cell["slug"],
        "shape_point": schedule_cell["shape_point"],
        "repeat_index": 0,
        "launch_index": launch_index,
        "ncu_controls": controls,
    }
    kernel_signature = run.get("kernel_signature_sha256")
    kernel_signature_source = run.get("kernel_signature_source")
    expected_run_fields = set(expected_run) | {
        "kernel_signature_sha256", "kernel_signature_source",
    }
    if (
        set(run) != expected_run_fields
        or any(run.get(key) != expected for key, expected in expected_run.items())
        or not _sha256_text(kernel_signature)
        or not isinstance(kernel_signature_source, str)
        or not kernel_signature_source
    ):
        _fail("M6_M4_RUN_MANIFEST_BINDING", cell_id)
    identity = observation.get("identity")
    kernel_name = identity.get("Kernel Name") if isinstance(identity, dict) else None
    try:
        ncu_parser.validate_launch_observation_document(
            observation,
            requested_launch_skip=launch_index,
            requested_launch_count=1,
            report_sha256=report_record["sha256"],
        )
    except (TypeError, ValueError) as exc:
        _fail("M6_M4_LAUNCH_OBSERVATION_BINDING", f"{cell_id}: {exc}")
    if not isinstance(kernel_name, str) or not kernel_name:
        _fail("M6_M4_LAUNCH_OBSERVATION_BINDING", cell_id)
    _validate_m4_capture_manifest(
        capture,
        records=records,
        observation=observation,
        run=run,
        matrix=matrix,
        matrix_cell=matrix_cell,
        schedule_cell=schedule_cell,
        source_commit=source_commit,
        registry_payload=registry_payload,
    )
    launch = {
        "launch_index": launch_index,
        "kernel_name": kernel_name,
        "kernel_signature_sha256": kernel_signature,
        "kernel_signature_source": kernel_signature_source,
        "label_sidecar_sha256": sidecar_record["sha256"],
        "label_run_manifest_sha256": run_record["sha256"],
        "label_capture_manifest_sha256": capture_record["sha256"],
        "ncu_launch_observation_sha256": observation_record["sha256"],
        "ncu_observation_sha256": observation["observation_sha256"],
        "ncu_report_sha256": report_record["sha256"],
    }
    launch["launch_identity_sha256"] = canonical_sha256(launch)
    return launch


def _derive_m4_binding_receipt(
    *,
    archive_root: Path,
    matrix: Mapping[str, Any],
    matrix_sha256: str,
    cells: Sequence[Mapping[str, Any]],
    generated_at_utc: str,
    repo_root: Path,
    archive_tree: _PinnedTreeTransaction | None = None,
    input_cache: InputSnapshotCache | None = None,
) -> dict[str, Any]:
    if archive_tree is None:
        with _PinnedTreeTransaction(
            archive_root,
            role="M4 archive",
            require_readonly_files=True,
            require_readonly_directories=True,
            input_cache=input_cache,
        ) as transaction:
            return _derive_m4_binding_receipt(
                archive_root=archive_root,
                matrix=matrix,
                matrix_sha256=matrix_sha256,
                cells=cells,
                generated_at_utc=generated_at_utc,
                repo_root=repo_root,
                archive_tree=transaction,
                input_cache=input_cache,
            )
    if (
        not archive_root.is_absolute()
        or archive_root != Path(os.path.abspath(archive_root))
    ):
        _fail(
            "M6_M4_ARCHIVE_ROOT",
            "a lexical absolute external M4 root is required",
        )
    source_commit = matrix.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        _fail("M6_M4_SOURCE_COMMIT", repr(source_commit))
    registry_pin = ((matrix.get("input_pins") or {}).get("metric_registry") or {})
    registry_relative = REGISTRY_PATH.relative_to(REPO_ROOT).as_posix()
    schema_relative = SIDECAR_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix()
    if registry_pin.get("path") != registry_relative:
        _fail("M6_M4_REGISTRY_AUTHORITY", "matrix registry path mismatch")
    registry_payload = _frozen_source_payload(
        repo_root,
        source_commit,
        registry_relative,
        role="source metric registry",
        cache=input_cache,
    )
    sidecar_schema_payload = _frozen_source_payload(
        repo_root,
        source_commit,
        schema_relative,
        role="source v5 sidecar schema",
        cache=input_cache,
    )
    registry = loads_strict(registry_payload, role="source metric registry")
    frozen_sidecar_schema = loads_strict(
        sidecar_schema_payload, role="source v5 sidecar schema",
    )
    if (
        not isinstance(registry, dict)
        or not isinstance(frozen_sidecar_schema, dict)
        or registry_pin.get("sha256") != sha256_bytes(registry_payload)
        or frozen_sidecar_schema.get("schema_version") != 5
        or frozen_sidecar_schema.get("$id")
        != "simbench_bw/schemas/local_label_sidecar.schema.json"
    ):
        _fail("M6_M4_REGISTRY_AUTHORITY", "source authority mismatch")
    active_sidecar_schema, active_sidecar_schema_payload = (
        _load_active_sidecar_schema(repo_root=repo_root, cache=input_cache)
    )
    active_sidecar_schema_identity = {
        "path": SIDECAR_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_bytes(active_sidecar_schema_payload),
        "size_bytes": len(active_sidecar_schema_payload),
    }
    _registry_metrics(registry, matrix)
    chunks_root = archive_root / "chunks"
    if archive_tree.inventory.get("chunks", (None,))[0] != "directory":
        _fail("M6_M4_CHUNK_ROOT", str(chunks_root))
    expected_chunks = {
        chunk["chunk_id"]: chunk for chunk in matrix.get("chunks", [])
        if isinstance(chunk, dict)
    }
    if len(expected_chunks) != 18:
        _fail("M6_M4_CHUNK_COUNT", f"matrix declares {len(expected_chunks)} chunks")
    actual_chunk_entries = {
        PurePosixPath(path).name: kind
        for path, (kind, _snapshot) in archive_tree.inventory.items()
        if path and PurePosixPath(path).parent == PurePosixPath("chunks")
    }
    actual_chunk_names = set(actual_chunk_entries)
    if (
        actual_chunk_names != set(expected_chunks)
        or any(kind != "directory" for kind in actual_chunk_entries.values())
    ):
        _fail(
            "M6_M4_CHUNK_INVENTORY",
            f"missing={sorted(set(expected_chunks) - actual_chunk_names)} "
            f"extra={sorted(actual_chunk_names - set(expected_chunks))}",
        )
    matrix_cells = {
        f"{cell['identity']['roster_source']}:{cell['identity']['slug']}": cell
        for cell in matrix.get("cells", [])
        if isinstance(cell, dict) and cell.get("disposition") == "runnable"
    }
    schedule_cells = {cell["cell_id"]: cell for cell in cells}
    if len(matrix_cells) != EXPECTED_RUNNABLE or set(matrix_cells) != set(schedule_cells):
        _fail("M6_M4_CELL_SOURCE", "matrix and schedule runnable cells differ")
    chunk_receipts: list[dict[str, Any]] = []
    cell_launches: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    preflight_hashes: set[str] = set()
    for chunk_id, expected_chunk in sorted(expected_chunks.items()):
        chunk_root = chunks_root / chunk_id
        chunk_relative = f"chunks/{chunk_id}"
        if archive_tree.inventory.get(chunk_relative, (None,))[0] != "directory":
            _fail("M6_M4_CHUNK_ROOT", chunk_id)
        done_path = chunk_root / "DONE.json"
        done_relative = done_path.relative_to(archive_root).as_posix()
        done_raw = archive_tree.read(done_relative, role=f"M4 DONE {chunk_id}")
        done = loads_strict(done_raw, role=f"M4 DONE {chunk_id}")
        if not isinstance(done, dict):
            _fail("M6_M4_DONE_CONTENT", chunk_id)
        try:
            problems = m4_done.verify_seal(
                done,
                repo_root=repo_root,
                rehash_references=True,
                require_current_head=True,
                require_clean_tree=True,
            )
        except Exception as exc:  # evidence decoder boundary
            _fail(
                "M6_M4_DONE_VERIFIER_FAILURE",
                f"{chunk_id}: {type(exc).__name__}: {exc}",
            )
        if not isinstance(problems, list):
            _fail("M6_M4_DONE_VERIFIER_FAILURE", f"{chunk_id}: invalid result")
        if problems:
            _fail("M6_M4_DONE_UNVERIFIED", f"{chunk_id}: {'; '.join(problems)}")
        expected_members = {
            f"{member['roster_source']}:{member['slug']}": {
                **member,
                "authority_cell_sha256": matrix_cells[
                    f"{member['roster_source']}:{member['slug']}"
                ]["authority_cell_sha256"],
            }
            for member in expected_chunk["cells"]
        }
        result_rows = done.get("cell_results")
        result_objects = (
            [row for row in result_rows if isinstance(row, dict)]
            if isinstance(result_rows, list) else []
        )
        actual_rows = {
            f"{row.get('roster_source')}:{row.get('slug')}": row
            for row in result_objects
        }
        fixed = {
            "schema_version": 2,
            "manifest_kind": "campaign_chunk_done",
            "chunk_id": chunk_id,
            "output_root": str(chunk_root),
            "source_commit_pre": source_commit,
            "source_commit_post": source_commit,
            "source_head_at_seal": source_commit,
            "source_commit_stable": True,
            "working_tree_clean": True,
            "protocol_version": "2.1",
            "arch": "sm_86",
            "capture_matrix_sha256": matrix_sha256,
            "matrix_seal_sha256": matrix["matrix_seal_sha256"],
            "roster_authority_sha256": matrix["authority"]["sha256"],
            "chunk_membership_sha256": expected_chunk["membership_sha256"],
            "expected_cells": len(expected_members),
        }
        if (
            any(done.get(key) != expected for key, expected in fixed.items())
            or len(result_objects) != len(result_rows or [])
            or len(actual_rows) != len(result_objects)
            or set(actual_rows) != set(expected_members)
        ):
            _fail("M6_M4_DONE_BINDING", chunk_id)
        preflight_sha = done.get("preflight_sha256")
        if not _sha256_text(preflight_sha):
            _fail("M6_M4_PREFLIGHT_BINDING", chunk_id)
        preflight_hashes.add(preflight_sha)
        for cell_id, member in sorted(expected_members.items()):
            if cell_id in seen_cells:
                _fail("M6_M4_CELL_DUPLICATE", cell_id)
            seen_cells.add(cell_id)
            row = actual_rows[cell_id]
            if (
                row.get("authority_cell_sha256")
                != member["authority_cell_sha256"]
                or not (
                    row.get("disposition") == "admitted"
                    or (
                        row.get("disposition") == "failed"
                        and row.get("typed_reason") == "trace_unsupported"
                    )
                )
            ):
                _fail("M6_M4_CELL_OUTCOME", cell_id)
            schedule_cell = schedule_cells[cell_id]
            launch = _derive_m4_launch(
                row=row,
                chunk_root=chunk_root,
                archive_root=archive_root,
                archive_tree=archive_tree,
                matrix=matrix,
                matrix_cell=matrix_cells[cell_id],
                schedule_cell=schedule_cell,
                source_commit=source_commit,
                sidecar_schema=active_sidecar_schema,
                registry=registry,
                registry_payload=registry_payload,
            )
            launches = [launch]
            cell_launches.append({
                "cell_id": cell_id,
                "cell_identity_sha256": schedule_cell["cell_identity_sha256"],
                "authority_cell_sha256": schedule_cell["authority_cell_sha256"],
                "binary_sha256": schedule_cell["binary_sha256"],
                "launch_set_sha256": canonical_sha256(launches),
                "launches": launches,
            })
        chunk_receipts.append({
            "chunk_id": chunk_id,
            "membership_sha256": expected_chunk["membership_sha256"],
            "done_path": done_path.relative_to(archive_root).as_posix(),
            "done_sha256": sha256_bytes(done_raw),
            "done_size_bytes": len(done_raw),
            "source_commit": source_commit,
            "capture_matrix_sha256": matrix_sha256,
            "preflight_sha256": preflight_sha,
            "cell_count": len(expected_members),
        })
    cell_launches.sort(key=lambda row: row["cell_id"])
    if len(seen_cells) != EXPECTED_RUNNABLE or seen_cells != set(schedule_cells):
        _fail("M6_M4_CELL_COUNT", f"verified {len(seen_cells)} cells")
    preflight_list = sorted(preflight_hashes)
    receipt = {
        "schema_version": 1,
        "artifact_kind": "m6_verified_m4_binding",
        "generated_at_utc": generated_at_utc,
        "status": "complete",
        "archive_root": str(archive_root),
        "source_commit": source_commit,
        "capture_matrix_sha256": matrix_sha256,
        "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "roster_authority_sha256": matrix["authority"]["sha256"],
        "runnable_membership_sha256": matrix["chunking"]["runnable_membership_sha256"],
        "active_sidecar_schema": active_sidecar_schema_identity,
        "preflight_sha256s": preflight_list,
        "preflight_set_sha256": canonical_sha256(preflight_list),
        "protocol_version": "2.1",
        "expected_cells": EXPECTED_RUNNABLE,
        "completed_cells": EXPECTED_RUNNABLE,
        "chunk_done_manifests": chunk_receipts,
        "cell_launch_identities": cell_launches,
    }
    receipt["identity_sha256"] = canonical_sha256(receipt)
    _validate_schema("m4_binding", receipt, repo_root)
    archive_tree.verify()
    return receipt


def build_m4_binding_receipt(
    *,
    archive_root: Path,
    matrix_path: Path = MATRIX_PATH,
    generated_at_utc: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repository = _canonical_repo_root(repo_root)
    archive = _external_m4_root(archive_root, repo_root=repository)
    matrix, matrix_raw, runnable = _load_canonical_matrix(
        matrix_path,
        repo_root=repository,
        require_semantic_rederivation=True,
    )
    return _derive_m4_binding_receipt(
        archive_root=archive,
        matrix=matrix,
        matrix_sha256=sha256_bytes(matrix_raw),
        cells=_schedule_cells(runnable),
        generated_at_utc=generated_at_utc or _now(),
        repo_root=repository,
    )


def _validate_m4_identity(
    value: Mapping[str, Any],
    *,
    matrix: Mapping[str, Any],
    matrix_sha256: str,
    cells: Sequence[Mapping[str, Any]],
    repo_root: Path,
    require_complete: bool,
    input_cache: InputSnapshotCache | None = None,
) -> None:
    _validate_seal(value, "identity_sha256", role="m4_identity")
    fixed = {
        "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "roster_authority_sha256": matrix["authority"]["sha256"],
        "runnable_membership_sha256": matrix["chunking"]["runnable_membership_sha256"],
        "protocol_version": "2.1",
        "expected_cells": EXPECTED_RUNNABLE,
    }
    if any(value.get(key) != expected for key, expected in fixed.items()):
        _fail("M6_M4_IDENTITY_BINDING", "canonical identity mismatch")
    complete = value.get("status") == "complete"
    if require_complete and not complete:
        _fail("M6_M4_INCOMPLETE", "complete M4 identity is required")
    if not complete:
        expected_incomplete = _default_m4_identity(matrix)
        if dict(value) != expected_incomplete:
            _fail(
                "M6_M4_INCOMPLETE_IDENTITY",
                "incomplete identity must equal the canonical zero-credit default",
            )
        return
    archive_root = value.get("archive_root")
    if not isinstance(archive_root, str) or not archive_root:
        _fail("M6_M4_ARCHIVE_ROOT", "complete identity requires an archive root")
    _external_m4_root(Path(archive_root), repo_root=repo_root)
    _validate_schema("m4_binding", value, repo_root)
    expected = _derive_m4_binding_receipt(
        archive_root=Path(value.get("archive_root", "")),
        matrix=matrix,
        matrix_sha256=matrix_sha256,
        cells=cells,
        generated_at_utc=value.get("generated_at_utc"),
        repo_root=repo_root,
        input_cache=input_cache,
    )
    if expected != value:
        _fail("M6_M4_RECEIPT_REDERIVATION", "verified M4 receipt differs")


def _derive_schedule_id(
    source_identity_sha256: str,
    config_identity_sha256: str,
    metric_identity_sha256: str,
    seed: int,
) -> str:
    return f"m6-determinism-{canonical_sha256([
        source_identity_sha256,
        config_identity_sha256,
        metric_identity_sha256,
        seed,
    ])[:16]}"


def _derive_journal_id(
    schedule_sha256: str,
    journal_payload: Mapping[str, Any],
) -> str:
    identity = {
        key: value
        for key, value in journal_payload.items()
        if key not in {"journal_id", "journal_sha256"}
    }
    identity["schedule_sha256"] = schedule_sha256
    return f"m6-journal-{canonical_sha256(identity)[:16]}"


def build_schedule(
    *,
    matrix_path: Path = MATRIX_PATH,
    registry_path: Path = REGISTRY_PATH,
    sidecar_schema_path: Path = SIDECAR_SCHEMA_PATH,
    m4_receipt_path: Path | None = None,
    seed: int,
    max_new_profiles_per_resume: int = TIER1_SLOTS,
    generated_at_utc: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= MAX_SAFE_JSON_INTEGER
    ):
        _fail(
            "M6_SEED",
            f"seed must be an integer in [0, {MAX_SAFE_JSON_INTEGER}]",
        )
    if (
        isinstance(max_new_profiles_per_resume, bool)
        or not isinstance(max_new_profiles_per_resume, int)
        or not 1 <= max_new_profiles_per_resume <= MAXIMUM_SLOTS
    ):
        _fail("M6_COST_GATE", "max profiles must be in [1,9760]")
    repository = _canonical_repo_root(repo_root)
    input_cache = InputSnapshotCache()
    matrix, matrix_bytes, runnable = _load_canonical_matrix(
        matrix_path,
        repo_root=repository,
        require_semantic_rederivation=True,
        cache=input_cache,
    )
    authority_sources = _load_matrix_authority_sources(
        matrix,
        repo_root=repository,
        cache=input_cache,
    )
    registry_file = _canonical_repo_file(
        registry_path, REGISTRY_PATH, repo_root=repository, role="metric registry",
    )
    sidecar_schema_file = _canonical_repo_file(
        sidecar_schema_path,
        SIDECAR_SCHEMA_PATH,
        repo_root=repository,
        role="v5 sidecar schema",
    )
    registry, registry_bytes = load_json(
        registry_file,
        role="metric registry",
        cache=input_cache,
    )
    sidecar_schema, sidecar_schema_bytes = load_json(
        sidecar_schema_file,
        role="v5 sidecar schema",
        cache=input_cache,
    )
    if not isinstance(registry, dict) or not isinstance(sidecar_schema, dict):
        _fail("M6_SOURCE_SHAPE", "source documents must be objects")
    try:
        Draft202012Validator.check_schema(sidecar_schema)
    except Exception as exc:
        _fail("M6_SIDECAR_SCHEMA_INVALID", f"{type(exc).__name__}: {exc}")
    probe_keys, probe_identity, probe_bytes = _load_sm86_probe(
        repo_root=repository,
        cache=input_cache,
    )
    metrics = _registry_metrics(registry, matrix, probe_keys=probe_keys)
    if sidecar_schema.get("schema_version") != 5 or sidecar_schema.get("$id") != "simbench_bw/schemas/local_label_sidecar.schema.json":
        _fail("M6_SIDECAR_SCHEMA_IDENTITY", "expected active v5 sidecar schema")
    configs = _configs()
    cells = _schedule_cells(runnable)
    slots, candidates = _build_slots(cells, configs, seed)
    if m4_receipt_path is None:
        m4_identity = _default_m4_identity(matrix)
    else:
        m4_identity, _ = load_json(
            m4_receipt_path,
            role="verified M4 binding receipt",
            cache=input_cache,
        )
        if not isinstance(m4_identity, dict):
            _fail("M6_M4_IDENTITY_SHAPE", "object required")
    _validate_m4_identity(
        m4_identity,
        matrix=matrix,
        matrix_sha256=sha256_bytes(matrix_bytes),
        cells=cells,
        repo_root=repository,
        require_complete=m4_receipt_path is not None,
        input_cache=input_cache,
    )
    controller_path = _canonical_repo_file(
        Path(__file__).resolve(),
        Path(__file__).resolve(),
        repo_root=repository,
        role="M6 controller",
    )
    stats_path = _canonical_repo_file(
        STATS_PATH, STATS_PATH, repo_root=repository, role="M6 statistics module",
    )
    immutable_helper_path = _canonical_repo_file(
        IMMUTABLE_HELPER_PATH,
        IMMUTABLE_HELPER_PATH,
        repo_root=repository,
        role="immutable artifact helper",
    )
    controller_bytes = _read_regular_cached(
        controller_path, role="M6 controller", cache=input_cache,
    )
    stats_bytes = _read_regular_cached(
        stats_path, role="M6 statistics module", cache=input_cache,
    )
    immutable_helper_bytes = _read_regular_cached(
        immutable_helper_path,
        role="immutable artifact helper",
        cache=input_cache,
    )
    legacy_schema_identities = [
        _file_identity(
            repository / path.relative_to(REPO_ROOT),
            repository,
            cache=input_cache,
            role=f"legacy M6 schema {path}",
        )
        for path in SCHEMA_PATHS_V1.values()
    ]
    legacy_schema_identities.sort(key=lambda item: item["path"])
    active_schema_identities = [
        _file_identity(
            repository / path.relative_to(REPO_ROOT),
            repository,
            cache=input_cache,
            role=f"active M6 schema {path}",
        )
        for path in SCHEMA_PATHS_V2.values()
    ]
    active_schema_identities.sort(key=lambda item: item["path"])
    source_identity = {
        "canonical_matrix": _file_identity(
            MATRIX_PATH, repository, matrix_bytes, cache=input_cache, role="canonical matrix"
        ),
        "metric_registry": _file_identity(
            registry_file, repository, registry_bytes, cache=input_cache, role="metric registry"
        ),
        "sidecar_schema": _file_identity(
            sidecar_schema_file,
            repository,
            sidecar_schema_bytes,
            cache=input_cache,
            role="v5 sidecar schema",
        ),
        "availability_probe": probe_identity,
        "controller_source": _file_identity(
            controller_path, repository, controller_bytes, cache=input_cache, role="M6 controller"
        ),
        "stats_source": _file_identity(
            stats_path, repository, stats_bytes, cache=input_cache, role="M6 statistics module"
        ),
        "immutable_helper": _file_identity(
            immutable_helper_path,
            repository,
            immutable_helper_bytes,
            cache=input_cache,
            role="immutable artifact helper",
        ),
        **authority_sources,
        "m6_schemas": legacy_schema_identities,
        "active_m6_schemas": active_schema_identities,
        "matrix_id": matrix["matrix_id"],
        "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "matrix_source_commit": matrix["source_commit"],
        "roster_authority_sha256": matrix["authority"]["sha256"],
        "runnable_membership_sha256": matrix["chunking"]["runnable_membership_sha256"],
        "sidecar_schema_version": 5,
        "registry_version": registry["registry_version"],
        "label_schema_version": registry["label_schema_version"],
    }
    source_sha = canonical_sha256(source_identity)
    config_sha = canonical_sha256(configs)
    metric_sha = canonical_sha256(metrics)
    schedule_id = _derive_schedule_id(source_sha, config_sha, metric_sha, seed)
    schedule = {
        "schema_version": ACTIVE_SCHEMA_VERSION,
        "artifact_kind": "m6_determinism_schedule",
        "schedule_id": schedule_id,
        "generated_at_utc": generated_at_utc or _now(),
        "seed": seed,
        "shuffle_algorithm": "sha256_seeded_sweep_v1",
        "source_identity": source_identity,
        "source_identity_sha256": source_sha,
        "config_identity_sha256": config_sha,
        "metric_identity_sha256": metric_sha,
        "m4_identity": m4_identity,
        "configs": configs,
        "metrics": metrics,
        "cells": cells,
        "policy": {
            "reference_config_id": REFERENCE_CONFIG,
            "reference_tier1_repeats": 20,
            "other_tier1_repeats": 10,
            "tier2_additional_reference_repeats": 10,
            "fresh_process_per_profile": True,
            "adaptive_dispersion_stopping": False,
            "cross_day_replication": False,
            "trigger_codes": [
                "boundary_proximity",
                "discrete_multiple_values",
                "modality_flag",
                "out_of_range",
                "single_point_outlier",
            ],
        },
        "cost_gate": {
            "unit": "fresh_profile_process",
            "max_new_profiles_per_resume": max_new_profiles_per_resume,
            "completed_slots_recomputed": False,
            "conditioned_on_dispersion": False,
        },
        "slots": slots,
        "tier2_candidates": candidates,
        "counts": {
            "canonical_cells": EXPECTED_UNIVERSE,
            "runnable_cells": EXPECTED_RUNNABLE,
            "excluded_cells": EXPECTED_EXCLUDED,
            "configs": 6,
            "selected_metrics": EXPECTED_SELECTED_METRICS,
            "registry_metrics": EXPECTED_REGISTRY_METRICS,
            "tier1_slots": TIER1_SLOTS,
            "tier2_candidate_slots": TIER2_CANDIDATES,
            "maximum_slots": MAXIMUM_SLOTS,
            "tier1_fresh_processes": TIER1_SLOTS,
            "maximum_fresh_processes": MAXIMUM_SLOTS,
        },
        "slots_sha256": canonical_sha256(slots),
        "tier2_candidates_sha256": canonical_sha256(candidates),
    }
    schedule["schedule_sha256"] = canonical_sha256(schedule)
    validate_schedule(
        schedule,
        repo_root=repository,
        rehash_sources=True,
        _input_cache=input_cache,
    )
    return schedule


def validate_schedule(
    schedule: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    rehash_sources: bool = True,
    _input_cache: InputSnapshotCache | None = None,
) -> None:
    input_cache = _input_cache or InputSnapshotCache()
    _validate_schema("schedule", schedule, repo_root, cache=input_cache)
    _validate_seal(schedule, "schedule_sha256", role="schedule")
    if schedule.get("source_identity_sha256") != canonical_sha256(schedule["source_identity"]):
        _fail("M6_SOURCE_SEAL", "source identity")
    expected_schedule_id = _derive_schedule_id(
        schedule["source_identity_sha256"],
        schedule["config_identity_sha256"],
        schedule["metric_identity_sha256"],
        schedule["seed"],
    )
    if schedule.get("schedule_id") != expected_schedule_id:
        _fail("M6_SCHEDULE_ID", "schedule_id does not rederive from bound identities")
    if schedule.get("config_identity_sha256") != canonical_sha256(schedule["configs"]):
        _fail("M6_CONFIG_SEAL", "config identity")
    if schedule.get("metric_identity_sha256") != canonical_sha256(schedule["metrics"]):
        _fail("M6_METRIC_SEAL", "metric identity")
    if schedule.get("slots_sha256") != canonical_sha256(schedule["slots"]):
        _fail("M6_SLOT_SEAL", "tier 1")
    if schedule.get("tier2_candidates_sha256") != canonical_sha256(schedule["tier2_candidates"]):
        _fail("M6_SLOT_SEAL", "tier 2")
    if schedule["configs"] != _configs():
        _fail("M6_CONFIG_SET", "not the exact six kernel-replay configurations")
    cells = schedule["cells"]
    expected_slots, expected_candidates = _build_slots(cells, schedule["configs"], schedule["seed"])
    if schedule["slots"] != expected_slots or schedule["tier2_candidates"] != expected_candidates:
        _fail("M6_SCHEDULE_ORDER", "seeded shuffled sweep re-derivation differs")
    all_slots = schedule["slots"] + schedule["tier2_candidates"]
    if [slot["ordinal"] for slot in all_slots] != list(range(MAXIMUM_SLOTS)):
        _fail("M6_SLOT_COORDINATE", "schedule ordinals must be the contiguous global range")
    if any(slot["ordinal"] != index for index, slot in enumerate(all_slots)):
        _fail("M6_SLOT_COORDINATE", "slot ordinal does not bind its schedule position")
    if len({slot["slot_id"] for slot in schedule["slots"] + schedule["tier2_candidates"]}) != MAXIMUM_SLOTS:
        _fail("M6_SLOT_IDENTITY", "duplicate slot")
    if rehash_sources:
        source = schedule["source_identity"]
        matrix_payload = _resolve_repo_file(
            source["canonical_matrix"],
            repo_root,
            role="canonical matrix",
            cache=input_cache,
        )
        registry_payload = _resolve_repo_file(
            source["metric_registry"],
            repo_root,
            role="metric registry",
            cache=input_cache,
        )
        sidecar_payload = _resolve_repo_file(
            source["sidecar_schema"],
            repo_root,
            role="sidecar schema",
            cache=input_cache,
        )
        probe_keys, probe_identity, probe_payload = _load_sm86_probe(
            repo_root=repo_root,
            cache=input_cache,
        )
        expected_controller_path = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
        expected_stats_path = STATS_PATH.relative_to(REPO_ROOT).as_posix()
        expected_immutable_helper_path = IMMUTABLE_HELPER_PATH.relative_to(
            REPO_ROOT
        ).as_posix()
        if (
            source["controller_source"]["path"] != expected_controller_path
            or source["stats_source"]["path"] != expected_stats_path
            or source["immutable_helper"]["path"]
            != expected_immutable_helper_path
        ):
            _fail(
                "M6_CONTROLLER_SOURCE_IDENTITY",
                "controller/statistics/immutable-helper path differs",
            )
        _resolve_repo_file(
            source["controller_source"],
            repo_root,
            role="controller source",
            cache=input_cache,
        )
        _resolve_repo_file(
            source["stats_source"],
            repo_root,
            role="statistics source",
            cache=input_cache,
        )
        _resolve_repo_file(
            source["immutable_helper"],
            repo_root,
            role="immutable artifact helper",
            cache=input_cache,
        )
        expected_legacy_paths = {
            path.relative_to(REPO_ROOT).as_posix() for path in SCHEMA_PATHS_V1.values()
        }
        if len(source.get("m6_schemas", [])) != len(expected_legacy_paths):
            _fail("M6_SCHEMA_SOURCE_COUNT", f"expected {len(expected_legacy_paths)} legacy M6 schemas")
        actual_legacy_paths = {identity["path"] for identity in source["m6_schemas"]}
        if actual_legacy_paths != expected_legacy_paths:
            _fail("M6_SCHEMA_SOURCE_IDENTITY", "legacy schema path set differs")
        legacy_by_path = {
            path.relative_to(REPO_ROOT).as_posix(): kind
            for kind, path in SCHEMA_PATHS_V1.items()
        }
        for identity in source["m6_schemas"]:
            kind = legacy_by_path.get(identity.get("path"))
            if kind is None or identity.get("sha256") != SCHEMA_SHA256_BY_VERSION[1][kind]:
                _fail("M6_SCHEMA_SOURCE_IDENTITY", "legacy schema digest/path differs")
            _resolve_repo_file(
                identity,
                repo_root,
                role=f"legacy M6 schema {identity['path']}",
                cache=input_cache,
            )
        if schedule.get("schema_version") == ACTIVE_SCHEMA_VERSION:
            active_rows = source.get("active_m6_schemas")
            expected_active_paths = {
                path.relative_to(REPO_ROOT).as_posix() for path in SCHEMA_PATHS_V2.values()
            }
            if not isinstance(active_rows, list) or len(active_rows) != len(expected_active_paths):
                _fail("M6_SCHEMA_SOURCE_COUNT", "active v2 schema set is incomplete")
            actual_active_paths = {identity["path"] for identity in active_rows}
            if actual_active_paths != expected_active_paths:
                _fail("M6_SCHEMA_SOURCE_IDENTITY", "active schema path set differs")
            active_by_path = {
                path.relative_to(REPO_ROOT).as_posix(): kind
                for kind, path in SCHEMA_PATHS_V2.items()
            }
            for identity in active_rows:
                kind = active_by_path.get(identity.get("path"))
                if kind is None or identity.get("sha256") != SCHEMA_SHA256_BY_VERSION[2][kind]:
                    _fail("M6_SCHEMA_SOURCE_IDENTITY", "active schema digest/path differs")
                _resolve_repo_file(
                    identity,
                    repo_root,
                    role=f"active M6 schema {identity['path']}",
                    cache=input_cache,
                )
        matrix, verified_matrix_payload, _ = _load_canonical_matrix(
            repo_root / MATRIX_PATH.relative_to(REPO_ROOT),
            repo_root=repo_root,
            require_semantic_rederivation=True,
            cache=input_cache,
        )
        if verified_matrix_payload != matrix_payload:
            _fail("M6_INPUT_SNAPSHOT_MIXED", "canonical matrix generation changed")
        registry = loads_strict(registry_payload, role="metric registry")
        sidecar_schema = loads_strict(sidecar_payload, role="sidecar schema")
        try:
            Draft202012Validator.check_schema(sidecar_schema)
        except Exception as exc:
            _fail("M6_SIDECAR_SCHEMA_INVALID", f"{type(exc).__name__}: {exc}")
        registry_pin = (matrix.get("input_pins") or {}).get("metric_registry") or {}
        expected_registry_path = REGISTRY_PATH.relative_to(REPO_ROOT).as_posix()
        if registry_pin.get("path") != expected_registry_path:
            _fail("M6_MATRIX_REGISTRY_PIN", "matrix metric registry path differs")
        validation_registry_payload = _frozen_source_payload(
            repo_root,
            matrix.get("source_commit"),
            expected_registry_path,
            role="frozen matrix metric registry",
            cache=input_cache,
        )
        validation_registry = loads_strict(
            validation_registry_payload,
            role="frozen matrix metric registry",
        )
        validation_registry_digest = sha256_bytes(validation_registry_payload)
        if registry_pin.get("sha256") != validation_registry_digest:
            _fail("M6_MATRIX_REGISTRY_PIN", "frozen registry pin differs")
        try:
            semantic_problems = matrix_builder.validate_matrix(
                matrix,
                repo_root=repo_root,
                registry_document=validation_registry,
                registry_payload=validation_registry_payload,
                registry_sha256=validation_registry_digest,
                validation_mode="frozen_v1",
            )
        except Exception as exc:
            _fail("M6_MATRIX_REDERIVATION", f"{type(exc).__name__}: {exc}")
        if semantic_problems:
            _fail("M6_MATRIX_REDERIVATION", "; ".join(semantic_problems))
        authority_sources = _load_matrix_authority_sources(
            matrix,
            repo_root=repo_root,
            cache=input_cache,
        )
        source = schedule["source_identity"]
        for authority_key, authority_value in authority_sources.items():
            if source.get(authority_key) != authority_value:
                _fail("M6_SOURCE_AUTHORITY_DRIFT", authority_key)
        source = schedule["source_identity"]
        expected_source_paths = {
            "canonical_matrix": MATRIX_PATH.relative_to(REPO_ROOT).as_posix(),
            "metric_registry": REGISTRY_PATH.relative_to(REPO_ROOT).as_posix(),
            "sidecar_schema": SIDECAR_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix(),
            "availability_probe": PROBE_PATH.relative_to(REPO_ROOT).as_posix(),
        }
        for field, expected_path in expected_source_paths.items():
            if source[field]["path"] != expected_path:
                _fail("M6_SOURCE_PATH_CLOSED", f"{field}: {source[field]['path']}")
        expected_source_values = {
            "matrix_id": matrix.get("matrix_id"),
            "matrix_seal_sha256": matrix.get("matrix_seal_sha256"),
            "matrix_source_commit": matrix.get("source_commit"),
            "roster_authority_sha256": (matrix.get("authority") or {}).get("sha256"),
            "runnable_membership_sha256": (matrix.get("chunking") or {}).get(
                "runnable_membership_sha256"
            ),
            "sidecar_schema_version": sidecar_schema.get("schema_version"),
            "registry_version": registry.get("registry_version"),
            "label_schema_version": registry.get("label_schema_version"),
        }
        if any(source.get(key) != expected for key, expected in expected_source_values.items()):
            _fail("M6_SOURCE_METADATA_DRIFT", "duplicated source metadata differs")
        for field, payload in (
            ("canonical_matrix", matrix_payload),
            ("metric_registry", registry_payload),
            ("sidecar_schema", sidecar_payload),
            ("availability_probe", probe_payload),
        ):
            identity = source[field]
            if identity.get("size_bytes") != len(payload) or identity.get("sha256") != sha256_bytes(payload):
                _fail("M6_SOURCE_IDENTITY_MISMATCH", field)
        expected_source_sha = canonical_sha256(source)
        if schedule["source_identity_sha256"] != expected_source_sha:
            _fail("M6_SOURCE_SEAL", "rehashed source identity differs")
        runnable = _matrix_semantics(matrix)
        if _schedule_cells(runnable) != schedule["cells"]:
            _fail("M6_CELL_SOURCE_DRIFT", "cell projection differs")
        if _registry_metrics(registry, matrix, probe_keys=probe_keys) != schedule["metrics"]:
            _fail("M6_METRIC_SOURCE_DRIFT", "metric projection differs")
        if sidecar_schema.get("schema_version") != 5:
            _fail("M6_SIDECAR_SCHEMA_IDENTITY", "not v5")
    else:
        if schedule["m4_identity"].get("status") == "complete":
            _fail(
                "M6_SOURCE_REHASH_REQUIRED",
                "structural inspection cannot certify a complete M4 identity",
            )
        matrix = {
            "matrix_seal_sha256": schedule["source_identity"]["matrix_seal_sha256"],
            "authority": {"sha256": schedule["source_identity"]["roster_authority_sha256"]},
            "chunking": {"runnable_membership_sha256": schedule["source_identity"]["runnable_membership_sha256"]},
            "chunks": [],
        }
        # Structural validation of an incomplete identity remains possible when
        # retained source files are intentionally unavailable.
        _validate_seal(schedule["m4_identity"], "identity_sha256", role="m4_identity")
        return
    _validate_m4_identity(
        schedule["m4_identity"],
        matrix=matrix,
        matrix_sha256=sha256_bytes(matrix_payload),
        cells=schedule["cells"],
        repo_root=repo_root,
        require_complete=False,
        input_cache=input_cache,
    )


def _relaxed_sidecar_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    relaxed = copy.deepcopy(schema)
    properties = relaxed["properties"]
    for field in ("cache_hit_rate_l2_sector", "cache_hit_rate_l1tex_sector"):
        properties[field].pop("minimum", None)
        properties[field].pop("maximum", None)
    for name in ("confidence_interval", "unit_tagged_confidence_interval"):
        definition = relaxed["$defs"][name]
        for field in ("lower", "upper"):
            definition["properties"][field].pop("minimum", None)
            definition["properties"][field].pop("maximum", None)
    value_schema = relaxed["$defs"]["unit_tagged_metric"]["properties"]["value"]
    value_schema.pop("minimum", None)
    value_schema.pop("maximum", None)
    return relaxed


def _metric_maps(schedule: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key = {metric["key"]: metric for metric in schedule["metrics"]}
    by_ncu = {metric["ncu_metric"]: metric for metric in schedule["metrics"]}
    return by_key, by_ncu


def _extract_observations(
    sidecar: Mapping[str, Any], sidecar_sha: str, schedule: Mapping[str, Any]
) -> list[dict[str, Any]]:
    _, by_ncu = _metric_maps(schedule)
    selected = {metric["key"] for metric in schedule["metrics"] if metric["capture_status"] == "selected"}
    observations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for metric in schedule["metrics"]:
        if metric["capture_status"] != "selected" or metric["carrier"] != "flat_field":
            continue
        field = metric["sidecar_field"]
        if field not in sidecar:
            continue
        value = _number(sidecar[field], role=f"sidecar.{field}")
        observations.append({
            "sidecar_sha256": sidecar_sha,
            "launch_index": sidecar["launch_index"],
            "metric_key": metric["key"],
            "metric_sha256": metric["metric_sha256"],
            "ncu_metric": metric["ncu_metric"],
            "unit": metric["unit"],
            "carrier": metric["carrier"],
            "source_pointer": f"/{field}",
            "value": value,
            "source_emitted_value": None,
            "source_emitted_unit": None,
        })
        seen.add(metric["key"])
    unit_metrics = sidecar.get("unit_tagged_metrics", [])
    if not isinstance(unit_metrics, list):
        _fail("M6_SIDECAR_UNIT_METRICS", "array required")
    for index, item in enumerate(unit_metrics):
        if not isinstance(item, dict):
            _fail("M6_SIDECAR_UNIT_METRIC", str(index))
        metric = by_ncu.get(item.get("ncu_metric"))
        if metric is None:
            _fail("M6_UNKNOWN_METRIC", repr(item.get("ncu_metric")))
        if metric["key"] not in selected:
            continue
        if metric["carrier"] != "unit_tagged_metric" or item.get("unit") != metric["unit"]:
            _fail("M6_METRIC_UNIT_IDENTITY", metric["key"])
        if metric["key"] in seen:
            _fail("M6_DUPLICATE_METRIC_OBSERVATION", metric["key"])
        value = _number(item.get("value"), role=f"unit_tagged_metrics/{index}/value")
        observations.append({
            "sidecar_sha256": sidecar_sha,
            "launch_index": sidecar["launch_index"],
            "metric_key": metric["key"],
            "metric_sha256": metric["metric_sha256"],
            "ncu_metric": metric["ncu_metric"],
            "unit": metric["unit"],
            "carrier": metric["carrier"],
            "source_pointer": f"/unit_tagged_metrics/{index}/value",
            "value": value,
            "source_emitted_value": None,
            "source_emitted_unit": None,
        })
        seen.add(metric["key"])
    diagnostics = sidecar.get("parser_diagnostics") or {}
    diagnostic_seen: set[str] = set()
    for diagnostic_name in (
        "ncu_sector_hit_rate_clamped_to_100",
        "ncu_lts_pct_overcount_raw_counter_ratio_used",
    ):
        records = diagnostics.get(diagnostic_name, []) if isinstance(diagnostics, dict) else []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                _fail("M6_PARSER_DIAGNOSTIC", diagnostic_name)
            metric = by_ncu.get(record.get("metric_label"))
            if metric is None or metric["key"] not in selected:
                continue
            if metric["unit"] != "pct" or metric["key"] in diagnostic_seen:
                _fail("M6_PARSER_DIAGNOSTIC_IDENTITY", metric["key"])
            raw_value = _number(
                record.get("raw_value"),
                role=f"parser_diagnostics/{diagnostic_name}/{index}/raw_value",
            )
            existing = next(
                (row for row in observations if row["metric_key"] == metric["key"]),
                None,
            )
            replacement = {
                "sidecar_sha256": sidecar_sha,
                "launch_index": sidecar["launch_index"],
                "metric_key": metric["key"],
                "metric_sha256": metric["metric_sha256"],
                "ncu_metric": metric["ncu_metric"],
                "unit": metric["unit"],
                "carrier": metric["carrier"],
                "source_pointer": f"/parser_diagnostics/{diagnostic_name}/{index}/raw_value",
                "value": raw_value / 100.0,
                "source_emitted_value": raw_value,
                "source_emitted_unit": "percent",
            }
            if existing is None:
                observations.append(replacement)
                seen.add(metric["key"])
            else:
                observations[observations.index(existing)] = replacement
            diagnostic_seen.add(metric["key"])
    observations.sort(key=lambda row: (row["metric_key"], row["source_pointer"]))
    return observations


def _lexical_relative_to_root(path: Path, root: Path, *, role: str) -> tuple[Path, str]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    root_absolute = Path(os.path.abspath(os.fspath(root)))
    try:
        relative = absolute.relative_to(root_absolute).as_posix()
    except ValueError as exc:
        _fail("M6_EVIDENCE_OUTSIDE_JOURNAL_ROOT", f"{role}: {path}: {exc}")
    _safe_relative(relative, role=role)
    return absolute, relative


def _journal_file_reference(
    path: Path,
    *,
    journal_dir: Path,
    role: str,
    evidence_tree: _PinnedTreeTransaction | None = None,
    input_cache: InputSnapshotCache | None = None,
) -> tuple[dict[str, Any], bytes]:
    absolute, relative = _lexical_relative_to_root(
        path, journal_dir, role=role,
    )
    if evidence_tree is None:
        payload = _read_regular_cached(
            absolute,
            role=role,
            cache=input_cache,
        )
    else:
        try:
            tree_relative = absolute.relative_to(evidence_tree.root).as_posix()
        except ValueError as exc:
            _fail("M6_EVIDENCE_TRANSACTION_ROOT", f"{role}: {exc}")
        payload = evidence_tree.read(tree_relative, role=role)
    return {
        "relative_path": relative,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }, payload


def _m6_repeat_controls(
    config: Mapping[str, Any], launch_index: int,
) -> dict[str, Any]:
    return {
        "target_processes": "all",
        "filter_mode": "global",
        "replay_mode": "kernel",
        "cache_control": config["cache_control"],
        "clock_control": config["clock_control"],
        "profile_from_start": "off",
        "launch_skip": launch_index,
        "launch_count": 1,
    }


def _repeat_launch_evidence(
    paths: Mapping[str, Path],
    *,
    sidecar_ref: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    journal_dir: Path,
    schedule: Mapping[str, Any],
    slot: Mapping[str, Any],
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    evidence_tree: _PinnedTreeTransaction | None,
    input_cache: InputSnapshotCache | None = None,
) -> tuple[dict[str, Any], str | None]:
    expected_roles = {
        "label_run_manifest",
        "ncu_report",
        "ncu_launch_observation",
        "label_capture_manifest",
    }
    if set(paths) != expected_roles or not all(
        isinstance(path, Path) for path in paths.values()
    ):
        _fail("M6_REPEAT_EVIDENCE_PATHS", slot["slot_id"])
    references: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for role in sorted(expected_roles):
        reference, payload = _journal_file_reference(
            paths[role],
            journal_dir=journal_dir,
            role=f"M6 {role}",
            evidence_tree=evidence_tree,
            input_cache=input_cache,
        )
        references[role] = reference
        payloads[role] = payload
    run = loads_strict(payloads["label_run_manifest"], role="M6 run manifest")
    observation = loads_strict(
        payloads["ncu_launch_observation"], role="M6 launch observation",
    )
    capture = loads_strict(
        payloads["label_capture_manifest"], role="M6 capture manifest",
    )
    if not all(isinstance(value, dict) for value in (run, observation, capture)):
        _fail("M6_REPEAT_EVIDENCE_SHAPE", slot["slot_id"])
    launch_index = sidecar["launch_index"]
    controls = _m6_repeat_controls(config, launch_index)
    expected_run = {
        "schema_version": 1,
        "source_commit": schedule["source_identity"]["matrix_source_commit"],
        "binary_sha256": cell["binary_sha256"],
        "arch": "sm_86",
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "repeat_index": slot["repeat_index"],
        "launch_index": launch_index,
        "ncu_controls": controls,
    }
    expected_run_fields = set(expected_run) | {
        "kernel_signature_sha256", "kernel_signature_source",
    }
    signature = run.get("kernel_signature_sha256")
    signature_source = run.get("kernel_signature_source")
    if (
        set(run) != expected_run_fields
        or any(run.get(key) != value for key, value in expected_run.items())
        or not _sha256_text(signature)
        or not isinstance(signature_source, str)
        or not signature_source
    ):
        _fail("M6_REPEAT_RUN_MANIFEST", slot["slot_id"])
    try:
        ncu_parser.validate_launch_observation_document(
            observation,
            requested_launch_skip=launch_index,
            requested_launch_count=1,
            report_sha256=references["ncu_report"]["sha256"],
        )
    except (TypeError, ValueError) as exc:
        _fail("M6_REPEAT_LAUNCH_OBSERVATION", f"{slot['slot_id']}: {exc}")
    identity = observation.get("identity")
    kernel_name = identity.get("Kernel Name") if isinstance(identity, dict) else None
    if not isinstance(kernel_name, str) or not kernel_name:
        _fail("M6_REPEAT_KERNEL_NAME", slot["slot_id"])
    artifact_receipt = {
        "label_sidecar": {
            "sha256": sidecar_ref["sha256"],
            "size_bytes": sidecar_ref["size_bytes"],
        },
        "label_run_manifest": {
            "sha256": references["label_run_manifest"]["sha256"],
            "size_bytes": references["label_run_manifest"]["size_bytes"],
        },
        "ncu_report": {
            "sha256": references["ncu_report"]["sha256"],
            "size_bytes": references["ncu_report"]["size_bytes"],
        },
        "ncu_launch_observation": {
            "sha256": references["ncu_launch_observation"]["sha256"],
            "size_bytes": references["ncu_launch_observation"]["size_bytes"],
            "observation_sha256": observation["observation_sha256"],
        },
    }
    expected_capture = {
        "schema_version": 1,
        "artifact_kind": "m6_repeat_capture_manifest",
        "source_commit": schedule["source_identity"]["matrix_source_commit"],
        "schedule_sha256": schedule["schedule_sha256"],
        "slot_id": slot["slot_id"],
        "slot_sha256": slot["slot_sha256"],
        "cell_id": cell["cell_id"],
        "cell_identity_sha256": cell["cell_identity_sha256"],
        "config_id": config["config_id"],
        "config_sha256": config["config_sha256"],
        "repeat_index": slot["repeat_index"],
        "launch_index": launch_index,
        "binary_sha256": cell["binary_sha256"],
        "kernel_name": kernel_name,
        "kernel_signature_sha256": signature,
        "kernel_signature_source": signature_source,
        "ncu_controls": controls,
        "validated_artifact_receipt": artifact_receipt,
    }
    if (
        set(capture) != set(expected_capture) | {"capture_manifest_sha256"}
        or any(capture.get(key) != value for key, value in expected_capture.items())
        or capture.get("capture_manifest_sha256")
        != canonical_sha256(_without(capture, "capture_manifest_sha256"))
    ):
        _fail("M6_REPEAT_CAPTURE_MANIFEST", slot["slot_id"])
    m4_launch_identity: str | None = None
    if schedule["m4_identity"]["status"] == "complete":
        m4_cells = {
            item["cell_id"]: item
            for item in schedule["m4_identity"]["cell_launch_identities"]
        }
        m4_cell = m4_cells.get(cell["cell_id"])
        if not isinstance(m4_cell, dict):
            _fail("M6_M4_LAUNCH_IDENTITY", slot["slot_id"])
        matches = [
            launch for launch in m4_cell["launches"]
            if launch["launch_index"] == launch_index
        ]
        if len(matches) != 1:
            _fail("M6_M4_LAUNCH_IDENTITY", slot["slot_id"])
        m4_launch = matches[0]
        expected_kernel = {
            "kernel_name": kernel_name,
            "kernel_signature_sha256": signature,
            "kernel_signature_source": signature_source,
        }
        if any(
            m4_launch.get(key) != value for key, value in expected_kernel.items()
        ):
            _fail("M6_M4_KERNEL_IDENTITY", slot["slot_id"])
        m4_launch_identity = m4_launch["launch_identity_sha256"]
    launch_evidence = {
        "label_run_manifest": references["label_run_manifest"],
        "ncu_report": references["ncu_report"],
        "ncu_launch_observation": references["ncu_launch_observation"],
        "label_capture_manifest": references["label_capture_manifest"],
        "kernel_name": kernel_name,
        "kernel_signature_sha256": signature,
        "kernel_signature_source": signature_source,
        "ncu_observation_sha256": observation["observation_sha256"],
        "capture_manifest_sha256": capture["capture_manifest_sha256"],
    }
    return launch_evidence, m4_launch_identity


def _sidecar_ref_and_observations(
    path: Path,
    *,
    journal_dir: Path,
    schedule: Mapping[str, Any],
    slot: Mapping[str, Any],
    repo_root: Path,
    launch_evidence_paths: Mapping[str, Path] | None,
    evidence_tree: _PinnedTreeTransaction | None = None,
    input_cache: InputSnapshotCache | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    artifact_ref, payload = _journal_file_reference(
        path,
        journal_dir=journal_dir,
        role="external v5 sidecar",
        evidence_tree=evidence_tree,
        input_cache=input_cache,
    )
    sidecar = loads_strict(payload, role="external v5 sidecar")
    if not isinstance(sidecar, dict):
        _fail("M6_SIDECAR_SHAPE", str(path))
    source_schema_payload = _resolve_repo_file(
        schedule["source_identity"]["sidecar_schema"],
        repo_root,
        role="sidecar schema",
        cache=input_cache,
    )
    source_schema = loads_strict(source_schema_payload, role="sidecar schema")
    validator = Draft202012Validator(_relaxed_sidecar_schema(source_schema), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(sidecar), key=lambda err: list(err.absolute_path))
    if errors:
        _fail("M6_SIDECAR_V5_STRUCTURE", errors[0].message)
    cell = next(cell for cell in schedule["cells"] if cell["cell_id"] == slot["cell_id"])
    config = next(config for config in schedule["configs"] if config["config_id"] == slot["config_id"])
    expected_identity = {
        "schema_version": 5,
        "device": "sm_86",
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "repeat_index": slot["repeat_index"],
        "binary_sha256": cell["binary_sha256"],
    }
    if any(sidecar.get(key) != value for key, value in expected_identity.items()):
        _fail("M6_SIDECAR_SLOT_IDENTITY", slot["slot_id"])
    protocol = sidecar.get("measurement_protocol")
    try:
        ncu_capture_contract.validate_measurement_protocol(protocol)
    except ValueError as exc:
        _fail("M6_SIDECAR_CONFIG_IDENTITY", f"{slot['slot_id']}: {exc}")
    required_protocol = {
        "ncu_clock_control": config["clock_control"],
        "ncu_cache_control": config["cache_control"],
        "ncu_replay_mode": "kernel",
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_profile_from_start": False,
        "ncu_launch_skip": sidecar["launch_index"],
        "ncu_launch_count": 1,
    }
    if not isinstance(protocol, dict) or any(protocol.get(k) != v for k, v in required_protocol.items()):
        _fail("M6_SIDECAR_CONFIG_IDENTITY", slot["slot_id"])
    for field in ("host_sm_clock_mhz", "host_mem_clock_mhz", "host_temperature_c"):
        _number(protocol.get(field), role=f"measurement_protocol.{field}")
    if config["reference"]:
        if protocol.get("host_sm_clock_mhz") != 1350 or protocol.get("host_mem_clock_mhz") != 10501:
            _fail("M6_REFERENCE_CLOCK_LOCK", slot["slot_id"])
        if protocol.get("host_persistence_mode") != "Enabled":
            _fail("M6_REFERENCE_PERSISTENCE", slot["slot_id"])
    sidecar_sha = artifact_ref["sha256"]
    ref = {
        "relative_path": artifact_ref["relative_path"],
        "sha256": sidecar_sha,
        "size_bytes": artifact_ref["size_bytes"],
        "schema_version": 5,
        "device": sidecar["device"],
        "slug": sidecar["slug"],
        "shape_point": sidecar["shape_point"],
        "repeat_index": sidecar["repeat_index"],
        "launch_index": sidecar["launch_index"],
        "binary_sha256": sidecar["binary_sha256"],
        "launch_evidence": None,
        "m4_launch_identity_sha256": None,
    }
    if launch_evidence_paths is not None:
        launch_evidence, m4_launch_identity = _repeat_launch_evidence(
            launch_evidence_paths,
            sidecar_ref=ref,
            sidecar=sidecar,
            journal_dir=journal_dir,
            schedule=schedule,
            slot=slot,
            cell=cell,
            config=config,
            evidence_tree=evidence_tree,
            input_cache=input_cache,
        )
        ref["launch_evidence"] = launch_evidence
        ref["m4_launch_identity_sha256"] = m4_launch_identity
    elif schedule["m4_identity"]["status"] == "complete":
        _fail("M6_REPEAT_EVIDENCE_REQUIRED", slot["slot_id"])
    observations = _extract_observations(sidecar, sidecar_sha, schedule)
    return ref, observations, sidecar


def build_observed_event(
    schedule: Mapping[str, Any],
    slot: Mapping[str, Any],
    sidecar_paths: Sequence[Path],
    *,
    journal_dir: Path,
    observed_at_utc: str,
    ncu_replay_pass_count: int,
    launch_evidence_paths: Sequence[Mapping[str, Path] | None] | None = None,
    resume_epoch: int = 0,
    repo_root: Path = REPO_ROOT,
    _evidence_tree: _PinnedTreeTransaction | None = None,
    _input_cache: InputSnapshotCache | None = None,
) -> dict[str, Any]:
    input_cache = _input_cache or InputSnapshotCache()
    if len(sidecar_paths) != 1:
        _fail("M6_M4_LAUNCH_CARDINALITY", slot["slot_id"])
    if not sidecar_paths:
        _fail("M6_EVENT_SIDECARS", "at least one sidecar is required")
    if launch_evidence_paths is None:
        evidence_paths: list[Mapping[str, Path] | None] = [None] * len(
            sidecar_paths
        )
    else:
        evidence_paths = list(launch_evidence_paths)
        if len(evidence_paths) != len(sidecar_paths):
            _fail("M6_REPEAT_EVIDENCE_CARDINALITY", slot["slot_id"])
    if _evidence_tree is None:
        transaction_paths: list[Path] = []
        for path in sidecar_paths:
            if not isinstance(path, Path):
                _fail("M6_EVIDENCE_PATH", f"sidecar: {path!r}")
            transaction_paths.append(Path(os.path.abspath(os.fspath(path))))
        for bundle in evidence_paths:
            if bundle is None:
                continue
            if not isinstance(bundle, Mapping):
                _fail("M6_REPEAT_EVIDENCE_PATHS", slot["slot_id"])
            for path in bundle.values():
                if not isinstance(path, Path):
                    _fail("M6_REPEAT_EVIDENCE_PATHS", slot["slot_id"])
                transaction_paths.append(Path(os.path.abspath(os.fspath(path))))
        journal_root = Path(os.path.abspath(os.fspath(journal_dir)))
        common_root = Path(os.path.commonpath([
            os.fspath(path.parent) for path in transaction_paths
        ]))
        try:
            common_root.relative_to(journal_root)
        except ValueError as exc:
            _fail("M6_EVIDENCE_OUTSIDE_JOURNAL_ROOT", str(exc))
        with _PinnedTreeTransaction(
            common_root,
            role=f"M6 event evidence {slot['slot_id']}",
            input_cache=input_cache,
        ) as transaction:
            event = build_observed_event(
                schedule,
                slot,
                sidecar_paths,
                journal_dir=journal_dir,
                observed_at_utc=observed_at_utc,
                ncu_replay_pass_count=ncu_replay_pass_count,
                launch_evidence_paths=evidence_paths,
                resume_epoch=resume_epoch,
                repo_root=repo_root,
                _evidence_tree=transaction,
                _input_cache=input_cache,
            )
            transaction.verify()
            return event
    refs: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    for path, launch_paths in zip(sidecar_paths, evidence_paths, strict=True):
        ref, rows, sidecar = _sidecar_ref_and_observations(
            path,
            journal_dir=journal_dir,
            schedule=schedule,
            slot=slot,
            repo_root=repo_root,
            launch_evidence_paths=launch_paths,
            evidence_tree=_evidence_tree,
            input_cache=input_cache,
        )
        refs.append(ref)
        observations.extend(rows)
        sidecars.append(sidecar)
    launches = [ref["launch_index"] for ref in refs]
    if len(launches) != len(set(launches)):
        _fail("M6_EVENT_LAUNCH_DUPLICATE", slot["slot_id"])
    protocols = [sidecar["measurement_protocol"] for sidecar in sidecars]
    sm_clocks = {p["host_sm_clock_mhz"] for p in protocols}
    mem_clocks = {p["host_mem_clock_mhz"] for p in protocols}
    temperatures = {p["host_temperature_c"] for p in protocols}
    if len(sm_clocks) != 1 or len(mem_clocks) != 1 or len(temperatures) != 1:
        _fail("M6_EVENT_TELEMETRY_INCONSISTENT", slot["slot_id"])
    observations.sort(key=lambda row: (row["launch_index"], row["metric_key"], row["source_pointer"]))
    return {
        "resume_epoch": resume_epoch,
        "ordinal": slot["ordinal"],
        "slot_id": slot["slot_id"],
        "slot_sha256": slot["slot_sha256"],
        "tier": slot["tier"],
        "cell_id": slot["cell_id"],
        "cell_identity_sha256": slot["cell_identity_sha256"],
        "config_id": slot["config_id"],
        "config_sha256": slot["config_sha256"],
        "repeat_index": slot["repeat_index"],
        "sweep_index": slot["sweep_index"],
        "fresh_process": True,
        "outcome": "observed",
        "sidecars": refs,
        "raw_observations": observations,
        "covariates": {
            "observed_at_utc": observed_at_utc,
            "sweep_index": slot["sweep_index"],
            "sm_clock_mhz": next(iter(sm_clocks)),
            "mem_clock_mhz": next(iter(mem_clocks)),
            "temperature_c": next(iter(temperatures)),
            "ncu_replay_pass_count": ncu_replay_pass_count,
        },
        "cost": {"fresh_processes": 1, "ncu_replay_passes": ncu_replay_pass_count},
        "typed_exclusion": None,
    }


def _validate_typed_exclusion_contract(
    *,
    reason: Any,
    detail: Any,
    fresh_processes: Any,
    ncu_replay_passes: Any,
    operator_authority_receipt: Any = None,
    role: str,
) -> None:
    reasons = {
        "profile_failed",
        "sidecar_missing",
        "identity_mismatch",
        "metric_missing",
        "source_unavailable",
        "operator_stop",
    }
    if reason not in reasons or not isinstance(detail, str) or not detail:
        _fail("M6_TYPED_EXCLUSION_CONTRACT", f"{role}: reason/detail")
    if reason == "operator_stop" and (
        not isinstance(operator_authority_receipt, str)
        or len(operator_authority_receipt) != 64
        or any(character not in "0123456789abcdef" for character in operator_authority_receipt)
    ):
        _fail("M6_OPERATOR_AUTHORITY_REQUIRED", role)
    if (
        type(fresh_processes) is not int
        or fresh_processes not in {0, 1}
        or type(ncu_replay_passes) is not int
        or ncu_replay_passes < 0
    ):
        _fail("M6_TYPED_EXCLUSION_CONTRACT", f"{role}: cost")
    if reason in {"operator_stop", "source_unavailable"}:
        if fresh_processes != 0 or ncu_replay_passes != 0:
            _fail(
                "M6_TYPED_EXCLUSION_CONTRACT",
                f"{role}: {reason} must not claim collection cost",
            )
    elif reason == "profile_failed":
        if fresh_processes != 1 or ncu_replay_passes < 1:
            _fail(
                "M6_TYPED_EXCLUSION_CONTRACT",
                f"{role}: profile_failed requires one attempted fresh process and replay",
            )
    elif fresh_processes != 1 or ncu_replay_passes < 1:
        _fail(
            "M6_TYPED_EXCLUSION_CONTRACT",
            f"{role}: {reason} requires one profiled fresh process",
        )


def build_excluded_event(
    slot: Mapping[str, Any], *, reason: str, detail: str, resume_epoch: int = 0,
    fresh_processes: int = 0, ncu_replay_passes: int = 0,
    operator_authority_receipt: str | None = None,
) -> dict[str, Any]:
    _validate_typed_exclusion_contract(
        reason=reason,
        detail=detail,
        fresh_processes=fresh_processes,
        ncu_replay_passes=ncu_replay_passes,
        operator_authority_receipt=operator_authority_receipt,
        role=str(slot.get("slot_id", "excluded event")),
    )
    return {
        "resume_epoch": resume_epoch,
        "ordinal": slot["ordinal"],
        "slot_id": slot["slot_id"],
        "slot_sha256": slot["slot_sha256"],
        "tier": slot["tier"],
        "cell_id": slot["cell_id"],
        "cell_identity_sha256": slot["cell_identity_sha256"],
        "config_id": slot["config_id"],
        "config_sha256": slot["config_sha256"],
        "repeat_index": slot["repeat_index"],
        "sweep_index": slot["sweep_index"],
        "fresh_process": fresh_processes == 1,
        "outcome": "excluded_typed",
        "sidecars": [],
        "raw_observations": [],
        "covariates": None,
        "cost": {"fresh_processes": fresh_processes, "ncu_replay_passes": ncu_replay_passes},
        "typed_exclusion": {
            "reason": reason,
            "detail": detail,
            **(
                {"operator_authority_receipt": operator_authority_receipt}
                if operator_authority_receipt is not None
                else {}
            ),
        },
    }


def _decision_map(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {decision["cell_id"]: decision for decision in decisions}


def _active_slots(schedule: Mapping[str, Any], decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    triggered = {decision["cell_id"] for decision in decisions if decision["triggered"]}
    return list(schedule["slots"]) + [
        slot for slot in schedule["tier2_candidates"] if slot["cell_id"] in triggered
    ]


def _causal_prefix_closed(
    schedule: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    trigger_state: str,
    trigger_decisions: Sequence[Mapping[str, Any]],
) -> bool:
    active = _active_slots(schedule, trigger_decisions)
    eligible = [slot["ordinal"] for slot in active]
    by_id = {slot["slot_id"]: slot for slot in active}
    observed = [by_id[event["slot_id"]]["ordinal"] for event in events]
    return observed == eligible[: len(observed)]


def _seal_causal_record(record: Mapping[str, Any], previous: str, sequence: int) -> dict[str, Any]:
    sealed = {
        "sequence": sequence,
        "previous_record_sha256": previous,
        **dict(record),
    }
    sealed["record_sha256"] = canonical_sha256(sealed)
    return sealed


def _build_causal_records(
    schedule: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    trigger_state: str,
    trigger_decisions: Sequence[Mapping[str, Any]],
    enforce_prefix: bool = False,
) -> list[dict[str, Any]]:
    """Build the causal append stream from already sealed terminal events."""
    by_id = {
        slot["slot_id"]: slot
        for slot in _active_slots(schedule, trigger_decisions)
    }
    ordered_ordinals = [slot["ordinal"] for slot in schedule["slots"]]
    if trigger_state == "decided":
        triggered = {
            decision["cell_id"]
            for decision in trigger_decisions
            if decision["triggered"]
        }
        ordered_ordinals.extend(
            slot["ordinal"]
            for slot in schedule["tier2_candidates"]
            if slot["cell_id"] in triggered
        )
    event_ordinals: list[int] = []
    for event in events:
        slot = by_id.get(event.get("slot_id"))
        if slot is None:
            _fail("M6_JOURNAL_SLOT", repr(event.get("slot_id")))
        event_ordinals.append(slot["ordinal"])
    expected_prefix = ordered_ordinals[: len(event_ordinals)]
    if enforce_prefix and event_ordinals != expected_prefix:
        _fail("M6_JOURNAL_PREFIX", "terminal events must close an exact eligible prefix")
    records: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    epoch = 0
    records.append(
        _seal_causal_record(
            {"record_kind": "epoch_open", "epoch": epoch}, previous, len(records)
        )
    )
    previous = records[-1]["record_sha256"]
    for event in events:
        slot = by_id[event["slot_id"]]
        attempt_id = canonical_sha256([epoch, slot["ordinal"], slot["slot_sha256"]])
        started = _seal_causal_record(
            {
                "record_kind": "attempt_started",
                "epoch": epoch,
                "ordinal": slot["ordinal"],
                "slot_id": slot["slot_id"],
                "attempt_id": attempt_id,
            },
            previous,
            len(records),
        )
        records.append(started)
        previous = started["record_sha256"]
        terminal = _seal_causal_record(
            {
                "record_kind": "attempt_terminal",
                "epoch": epoch,
                "ordinal": slot["ordinal"],
                "slot_id": slot["slot_id"],
                "attempt_id": attempt_id,
                "event_sha256": event["event_sha256"],
                "cost": event["cost"],
            },
            previous,
            len(records),
        )
        records.append(terminal)
        previous = terminal["record_sha256"]
    if len(event_ordinals) >= TIER1_SLOTS:
        record = _seal_causal_record(
            {"record_kind": "tier1_closed", "epoch": epoch},
            previous,
            len(records),
        )
        records.append(record)
        previous = record["record_sha256"]
        if trigger_state == "decided":
            for decision in trigger_decisions:
                record = _seal_causal_record(
                    {
                        "record_kind": "trigger_decision",
                        "epoch": epoch,
                        "cell_id": decision["cell_id"],
                        "cell_identity_sha256": decision["cell_identity_sha256"],
                        "triggered": decision["triggered"],
                        "reason_codes": decision["reason_codes"],
                        "evidence_metric_keys": decision["evidence_metric_keys"],
                        "decision_sha256": decision["decision_sha256"],
                    },
                    previous,
                    len(records),
                )
                records.append(record)
                previous = record["record_sha256"]
            triggered = sorted(
                decision["cell_id"]
                for decision in trigger_decisions
                if decision["triggered"]
            )
            if triggered:
                record = _seal_causal_record(
                    {
                        "record_kind": "tier2_activated",
                        "epoch": epoch,
                        "cell_id": triggered[0],
                    },
                    previous,
                    len(records),
                )
                records.append(record)
                previous = record["record_sha256"]
    record = _seal_causal_record(
        {
            "record_kind": "checkpoint",
            "epoch": epoch,
            "cost": {
                "fresh_processes": sum(
                    event["cost"]["fresh_processes"] for event in events
                ),
                "ncu_replay_passes": sum(
                    event["cost"]["ncu_replay_passes"] for event in events
                ),
            },
        },
        previous,
        len(records),
    )
    records.append(record)
    records.append(
        _seal_causal_record(
            {"record_kind": "epoch_closed", "epoch": epoch},
            previous,
            len(records),
        )
    )
    # Every causal record is bound to the schedule and terminal outcome.  Seal
    # the complete stream once more after adding those bindings so callers
    # cannot transplant a valid record chain between schedules.
    outcome_by_event = {
        event["event_sha256"]: event["outcome"] for event in events
    }
    activated_cells = sorted(
        decision["cell_id"]
        for decision in trigger_decisions
        if decision.get("triggered")
    )
    normalized: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    for index, record in enumerate(records):
        body = _without(record, "record_sha256")
        body.pop("sequence", None)
        body.pop("previous_record_sha256", None)
        body["schedule_sha256"] = schedule["schedule_sha256"]
        if body.get("record_kind") == "attempt_terminal":
            body["outcome"] = outcome_by_event.get(body.get("event_sha256"))
        if body.get("record_kind") == "trigger_decision":
            decision = next(
                item for item in trigger_decisions
                if item.get("cell_id") == body.get("cell_id")
            )
            body["cell_identity_sha256"] = decision["cell_identity_sha256"]
            body["triggered"] = decision["triggered"]
            body["reason_codes"] = decision["reason_codes"]
            body["evidence_metric_keys"] = decision["evidence_metric_keys"]
            body["decision_sha256"] = decision["decision_sha256"]
        if body.get("record_kind") == "tier2_activated":
            body["activated_cell_ids"] = activated_cells
            body["activated_ordinals"] = sorted(
                slot["ordinal"]
                for slot in schedule["tier2_candidates"]
                if slot["cell_id"] in activated_cells
            )
        sealed = _seal_causal_record(body, previous, index)
        normalized.append(sealed)
        previous = sealed["record_sha256"]
    return normalized



def _build_causal_records_v2(
    schedule: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    trigger_state: str,
    trigger_decisions: Sequence[Mapping[str, Any]],
    enforce_prefix: bool = False,
) -> list[dict[str, Any]]:
    """Replay terminal events into the predecessor-bound causal stream.

    The event list is only an observation input.  Epoch numbers and every
    causal record are derived here from the bound schedule and terminal costs;
    caller-supplied resume labels are never trusted as authority.
    """
    active = _active_slots(schedule, trigger_decisions)
    by_id = {slot["slot_id"]: slot for slot in active}
    active_order = [slot["ordinal"] for slot in active]
    event_ordinals: list[int] = []
    for event in events:
        slot = by_id.get(event.get("slot_id"))
        if slot is None:
            _fail("M6_JOURNAL_SLOT", repr(event.get("slot_id")))
        event_ordinals.append(slot["ordinal"])
    if enforce_prefix and event_ordinals != active_order[:len(event_ordinals)]:
        _fail("M6_JOURNAL_PREFIX", "terminal events must close an exact eligible prefix")
    if any(left >= right for left, right in zip(event_ordinals, event_ordinals[1:])):
        _fail("M6_JOURNAL_ORDINAL", "terminal ordinals must be strictly increasing")

    epochs: list[int] = []
    for event in events:
        epoch = event.get("resume_epoch")
        if type(epoch) is not int or epoch < 0:
            _fail("M6_JOURNAL_EPOCH", "event epoch was not derived")
        epochs.append(epoch)
    if epochs and epochs[0] != 0:
        _fail("M6_JOURNAL_EPOCH", "epochs must begin at zero")
    if any(right < left or right > left + 1 for left, right in zip(epochs, epochs[1:])):
        _fail("M6_JOURNAL_EPOCH", "epochs must be contiguous and monotonic")

    records: list[dict[str, Any]] = []
    previous = ZERO_SHA256

    def append(body: Mapping[str, Any]) -> None:
        nonlocal previous
        sealed = _seal_causal_record(
            {"schedule_sha256": schedule["schedule_sha256"], **dict(body)},
            previous,
            len(records),
        )
        records.append(sealed)
        previous = sealed["record_sha256"]

    events_by_epoch: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event, epoch in zip(events, epochs, strict=True):
        events_by_epoch[epoch].append(event)
    max_epoch = max(epochs, default=0)
    tier1_terminal: set[str] = set()
    tier1_closed = False
    triggers_emitted = False
    activation_emitted = False
    triggered_cells = sorted(
        decision["cell_id"] for decision in trigger_decisions if decision.get("triggered")
    )
    activated_ordinals = sorted(
        slot["ordinal"]
        for slot in schedule["tier2_candidates"]
        if slot["cell_id"] in triggered_cells
    )

    for epoch in range(max_epoch + 1):
        group = events_by_epoch.get(epoch, [])
        append({"record_kind": "epoch_open", "epoch": epoch})
        group_cost = {"fresh_processes": 0, "ncu_replay_passes": 0}
        for event in group:
            slot = by_id[event["slot_id"]]
            attempt_id = canonical_sha256([epoch, slot["ordinal"], slot["slot_sha256"]])
            append({
                "record_kind": "attempt_started",
                "epoch": epoch,
                "ordinal": slot["ordinal"],
                "slot_id": slot["slot_id"],
                "attempt_id": attempt_id,
            })
            cost = event.get("cost")
            if not isinstance(cost, Mapping):
                _fail("M6_JOURNAL_COST", event["slot_id"])
            group_cost["fresh_processes"] += int(cost.get("fresh_processes", 0))
            group_cost["ncu_replay_passes"] += int(cost.get("ncu_replay_passes", 0))
            append({
                "record_kind": "attempt_terminal",
                "epoch": epoch,
                "ordinal": slot["ordinal"],
                "slot_id": slot["slot_id"],
                "attempt_id": attempt_id,
                "event_sha256": event["event_sha256"],
                "outcome": event["outcome"],
                "cost": dict(cost),
            })
            if slot["tier"] == 1:
                tier1_terminal.add(slot["slot_id"])
            if not tier1_closed and len(tier1_terminal) == len(schedule["slots"]):
                append({"record_kind": "tier1_closed", "epoch": epoch})
                tier1_closed = True
                if trigger_state == "decided":
                    for decision in trigger_decisions:
                        append({
                            "record_kind": "trigger_decision",
                            "epoch": epoch,
                            "cell_id": decision["cell_id"],
                            "cell_identity_sha256": decision["cell_identity_sha256"],
                            "triggered": decision["triggered"],
                            "reason_codes": decision["reason_codes"],
                            "evidence_metric_keys": decision["evidence_metric_keys"],
                            "decision_sha256": decision["decision_sha256"],
                        })
                    triggers_emitted = True
                    if triggered_cells:
                        append({
                            "record_kind": "tier2_activated",
                            "epoch": epoch,
                            "activated_cell_ids": triggered_cells,
                            "activated_ordinals": activated_ordinals,
                        })
                        activation_emitted = True
        append({
            "record_kind": "checkpoint",
            "epoch": epoch,
            "cost": group_cost,
        })
        append({"record_kind": "epoch_closed", "epoch": epoch})

    if len(tier1_terminal) == len(schedule["slots"]) and not tier1_closed:
        _fail("M6_JOURNAL_CAUSALITY", "tier1 closure was not emitted")
    if trigger_state == "decided" and tier1_closed and not triggers_emitted:
        _fail("M6_JOURNAL_CAUSALITY", "trigger decisions were not emitted")
    if triggered_cells and tier1_closed and not activation_emitted:
        _fail("M6_JOURNAL_CAUSALITY", "tier2 activation was not emitted")
    return records


def _validate_causal_records(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    *,
    require_prefix: bool = True,
) -> None:
    records = journal.get("causal_events")
    if not isinstance(records, list) or not records:
        _fail("M6_JOURNAL_CAUSALITY", "causal record stream is required")
    previous = ZERO_SHA256
    starts: dict[str, str] = {}
    terminals: set[str] = set()
    event_by_hash = {
        event.get("event_sha256"): event for event in journal.get("events", [])
    }
    slots_by_id = {
        slot["slot_id"]: slot
        for slot in _active_slots(schedule, journal.get("trigger_decisions", []))
    }
    tier1_closed_seen = False
    trigger_records: list[Mapping[str, Any]] = []
    activation_records: list[Mapping[str, Any]] = []
    terminal_ordinals: list[int] = []
    open_epoch: int | None = None
    closed_epoch = -1
    epoch_checkpoint_seen = False
    last_ordinal = -1
    for index, record in enumerate(records):
        if (
            record.get("sequence") != index
            or record.get("previous_record_sha256") != previous
            or record.get("schedule_sha256") != journal.get("schedule_sha256")
            or record.get("record_sha256")
            != canonical_sha256(_without(record, "record_sha256"))
        ):
            _fail("M6_JOURNAL_CAUSALITY", str(index))
        previous = record["record_sha256"]
        kind = record["record_kind"]
        record_epoch = record.get("epoch")
        if type(record_epoch) is not int or record_epoch < 0:
            _fail("M6_JOURNAL_EPOCH", "invalid epoch")
        if kind == "epoch_open":
            if open_epoch is not None or record_epoch != (0 if index == 0 else (closed_epoch + 1)):
                _fail("M6_JOURNAL_EPOCH", "invalid epoch open transition")
            open_epoch = record_epoch
            epoch_checkpoint_seen = False
        elif kind == "epoch_closed":
            if open_epoch != record_epoch or not epoch_checkpoint_seen:
                _fail("M6_JOURNAL_EPOCH", "epoch closed without checkpoint")
            closed_epoch = record_epoch
            open_epoch = None
        else:
            if open_epoch != record_epoch:
                _fail("M6_JOURNAL_EPOCH", "record outside open epoch")
        if kind == "attempt_started":
            if open_epoch is None:
                _fail("M6_JOURNAL_EPOCH", "attempt outside epoch")
            ordinal = record.get("ordinal")
            slot_id = record.get("slot_id")
            slot = slots_by_id.get(slot_id)
            if (
                not isinstance(ordinal, int)
                or ordinal <= last_ordinal
                or slot is None
                or slot["ordinal"] != ordinal
            ):
                _fail("M6_JOURNAL_ORDINAL", str(ordinal))
            last_ordinal = ordinal
            attempt_id = record.get("attempt_id")
            expected_attempt = canonical_sha256(
                [record.get("epoch"), ordinal, slot["slot_sha256"]]
            )
            if (
                slot_id in starts
                or not _sha256_text(attempt_id)
                or attempt_id != expected_attempt
            ):
                _fail(
                    "M6_CAUSAL_UNSUPPORTED_ATTEMPT_STATE",
                    "retry or duplicate attempt identity is not representable in v2",
                )
            starts[slot_id] = attempt_id
        elif kind == "attempt_terminal":
            slot_id = record.get("slot_id")
            slot = slots_by_id.get(slot_id)
            event = event_by_hash.get(record.get("event_sha256"))
            if (
                slot is None
                or slot_id not in starts
                or slot_id in terminals
                or event is None
                or record.get("ordinal") != slot["ordinal"]
                or record.get("outcome") != event.get("outcome")
                or record.get("cost") != event.get("cost")
                or event.get("resume_epoch") != record_epoch
            ):
                _fail("M6_CAUSAL_UNSUPPORTED_ATTEMPT_STATE", "terminal lacks unique predecessor")
            if record.get("attempt_id") != starts[slot_id]:
                _fail("M6_JOURNAL_CAUSALITY", "attempt identity mismatch")
            terminals.add(slot_id)
            terminal_ordinals.append(slot["ordinal"])
        elif kind == "checkpoint":
            if epoch_checkpoint_seen:
                _fail("M6_JOURNAL_EPOCH", "duplicate checkpoint")
            cost = record.get("cost")
            if not isinstance(cost, dict) or any(
                type(cost.get(field)) is not int or cost[field] < 0
                for field in ("fresh_processes", "ncu_replay_passes")
            ):
                _fail("M6_JOURNAL_COST", "checkpoint cost is malformed")
            epoch_checkpoint_seen = True
        elif kind == "tier1_closed":
            if tier1_closed_seen or not all(
                slot["slot_id"] in terminals for slot in schedule["slots"]
            ):
                _fail("M6_JOURNAL_EPOCH", "tier1 closure precedes terminal closure")
            tier1_closed_seen = True
        elif kind == "trigger_decision":
            if not tier1_closed_seen or journal.get("trigger_state") != "decided":
                _fail("M6_JOURNAL_CAUSALITY", "trigger decision before tier1 closure")
            trigger_records.append(record)
        elif kind == "tier2_activated":
            if not tier1_closed_seen or not trigger_records:
                _fail("M6_JOURNAL_CAUSALITY", "tier2 activation lacks trigger predecessor")
            activation_records.append(record)
    if starts.keys() - terminals:
        _fail(
            "M6_CAUSAL_UNSUPPORTED_ATTEMPT_STATE",
            "open or interrupted attempts require a future causal schema version",
        )
    if records[0]["record_kind"] != "epoch_open" or records[-1]["record_kind"] != "epoch_closed":
        _fail("M6_JOURNAL_EPOCH", "unbalanced epoch boundary")
    if open_epoch is not None:
        _fail("M6_JOURNAL_EPOCH", "final epoch remains open")
    active_order = [slot["ordinal"] for slot in _active_slots(
        schedule, journal.get("trigger_decisions", [])
    )]
    if require_prefix and terminal_ordinals != active_order[: len(terminal_ordinals)]:
        _fail("M6_JOURNAL_PREFIX", "terminal ordinals are not an exact eligible prefix")
    expected_trigger_records = [
        {
            "cell_id": decision["cell_id"],
            "cell_identity_sha256": decision["cell_identity_sha256"],
            "triggered": decision["triggered"],
            "reason_codes": decision["reason_codes"],
            "evidence_metric_keys": decision["evidence_metric_keys"],
            "decision_sha256": decision["decision_sha256"],
        }
        for decision in journal.get("trigger_decisions", [])
    ]
    actual_trigger_records = [
        {
            "cell_id": record.get("cell_id"),
            "cell_identity_sha256": record.get("cell_identity_sha256"),
            "triggered": record.get("triggered"),
            "reason_codes": record.get("reason_codes"),
            "evidence_metric_keys": record.get("evidence_metric_keys"),
            "decision_sha256": record.get("decision_sha256"),
        }
        for record in trigger_records
    ]
    if actual_trigger_records != expected_trigger_records:
        _fail("M6_JOURNAL_CAUSALITY", "trigger record stream differs")
    expected_activated = sorted(
        decision["cell_id"]
        for decision in journal.get("trigger_decisions", [])
        if decision.get("triggered")
    )
    if activation_records:
        expected_ordinals = sorted(
            slot["ordinal"]
            for slot in schedule["tier2_candidates"]
            if slot["cell_id"] in expected_activated
        )
        if len(activation_records) != 1 or activation_records[0].get(
            "activated_cell_ids"
        ) != expected_activated or activation_records[0].get(
            "activated_ordinals"
        ) != expected_ordinals:
            _fail("M6_JOURNAL_CAUSALITY", "tier2 activation set differs")
    elif expected_activated and tier1_closed_seen and journal.get("trigger_state") == "decided":
        _fail("M6_JOURNAL_CAUSALITY", "triggered tier2 set was not activated")
    expected = _build_causal_records_v2(
        schedule,
        journal["events"],
        trigger_state=journal["trigger_state"],
        trigger_decisions=journal["trigger_decisions"],
        enforce_prefix=require_prefix,
    )
    if expected != records:
        _fail("M6_JOURNAL_CAUSALITY", "causal replay differs from terminal events")


def _derive_event_epochs(
    events: Sequence[Mapping[str, Any]],
    *,
    max_profiles_per_epoch: int,
) -> list[int]:
    """Derive resume epochs from terminal costs, never from caller labels."""
    if type(max_profiles_per_epoch) is not int or max_profiles_per_epoch < 1:
        _fail("M6_COST_GATE", "positive epoch profile limit required")
    epochs: list[int] = []
    epoch = 0
    used = 0
    for index, event in enumerate(events):
        cost = event.get("cost")
        fresh = cost.get("fresh_processes") if isinstance(cost, Mapping) else None
        if type(fresh) is not int or fresh not in {0, 1}:
            _fail("M6_TYPED_EXCLUSION_CONTRACT", f"event {index}: fresh process cost")
        if fresh and used + fresh > max_profiles_per_epoch:
            epoch += 1
            used = 0
        epochs.append(epoch)
        used += fresh
    return epochs


def build_journal(    schedule: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    trigger_decisions: Sequence[Mapping[str, Any]] = (),
    trigger_state: str = "pending",
    created_at_utc: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    input_cache = InputSnapshotCache()
    validate_schedule(
        schedule,
        repo_root=repo_root,
        rehash_sources=True,
        _input_cache=input_cache,
    )
    _require_active_schema("schedule", schedule)
    if trigger_state not in {"pending", "decided"}:
        _fail("M6_TRIGGER_STATE", trigger_state)
    if trigger_state == "pending" and trigger_decisions:
        _fail("M6_TRIGGER_PENDING_ROWS", "pending journal cannot carry decisions")
    if trigger_state == "decided" and len(trigger_decisions) != EXPECTED_RUNNABLE:
        _fail("M6_TRIGGER_DECISION_COUNT", "decided journal requires 122 decisions")
    decisions = [dict(decision) for decision in trigger_decisions]
    if trigger_state == "decided":
        decision_cells = [decision.get("cell_id") for decision in decisions]
        expected_cells = [cell["cell_id"] for cell in schedule["cells"]]
        if len(set(decision_cells)) != EXPECTED_RUNNABLE or set(decision_cells) != set(expected_cells):
            _fail("M6_TRIGGER_DECISION_SET", "decisions must cover each runnable cell exactly once")
        if decision_cells != expected_cells:
            _fail("M6_TRIGGER_DECISION_ORDER", "decisions must follow canonical cell order")
    for decision in decisions:
        if decision.get("decision_sha256") != canonical_sha256(_without(decision, "decision_sha256")):
            _fail("M6_TRIGGER_DECISION_SEAL", str(decision.get("cell_id")))
    active = _active_slots(schedule, decisions)
    active_by_id = {slot["slot_id"]: slot for slot in active}
    sealed_events: list[dict[str, Any]] = []
    derived_epochs = _derive_event_epochs(
        events,
        max_profiles_per_epoch=schedule["cost_gate"]["max_new_profiles_per_resume"],
    )
    previous = ZERO_SHA256
    seen: set[str] = set()
    for sequence, (raw, derived_epoch) in enumerate(zip(events, derived_epochs, strict=True)):
        event = dict(raw)
        event["resume_epoch"] = derived_epoch
        if event.get("outcome") == "excluded_typed":
            exclusion = event.get("typed_exclusion")
            cost = event.get("cost")
            _validate_typed_exclusion_contract(
                reason=(exclusion or {}).get("reason")
                if isinstance(exclusion, dict)
                else None,
                detail=(exclusion or {}).get("detail")
                if isinstance(exclusion, dict)
                else None,
                fresh_processes=(cost or {}).get("fresh_processes")
                if isinstance(cost, dict)
                else None,
                ncu_replay_passes=(cost or {}).get("ncu_replay_passes")
                if isinstance(cost, dict)
                else None,
                operator_authority_receipt=(exclusion or {}).get(
                    "operator_authority_receipt"
                )
                if isinstance(exclusion, dict)
                else None,
                role=str(event.get("slot_id", sequence)),
            )
        slot_id = event.get("slot_id")
        if slot_id not in active_by_id or slot_id in seen:
            _fail("M6_JOURNAL_SLOT", repr(slot_id))
        seen.add(slot_id)
        event["sequence"] = sequence
        event["previous_event_sha256"] = previous
        event.pop("event_sha256", None)
        event["event_sha256"] = canonical_sha256(event)
        previous = event["event_sha256"]
        sealed_events.append(event)
    if trigger_state == "decided":
        derived_decisions = _trigger_decisions(
            schedule, _trigger_rows_from_events(schedule, sealed_events)
        )
        if decisions != derived_decisions:
            _fail(
                "M6_TRIGGER_DECISION_MISMATCH",
                "trigger decisions must rederive from terminal observations",
            )
    if trigger_state == "decided" and not all(
        slot["slot_id"] in seen for slot in schedule["slots"]
    ):
        _fail("M6_TRIGGER_PREMATURE", "trigger decisions require all Tier-1 terminals")
    if trigger_state == "decided":
        triggered_cells = {
            decision["cell_id"] for decision in decisions if decision["triggered"]
        }
        tier2_counts = Counter(
            next(slot for slot in schedule["tier2_candidates"] if slot["slot_id"] == event["slot_id"])["cell_id"]
            for event in sealed_events
            if any(slot["slot_id"] == event["slot_id"] for slot in schedule["tier2_candidates"])
        )
        if any(cell not in triggered_cells for cell in tier2_counts):
            _fail("M6_TIER2_UNAUTHORIZED", "Tier-2 event lacks a triggered cell")
        if any(count > 10 for count in tier2_counts.values()):
            _fail("M6_TIER2_CARDINALITY", "more than ten Tier-2 candidates for a cell")
    terminal_ids = [slot["slot_id"] for slot in active if slot["slot_id"] in seen]
    pending_ids = [slot["slot_id"] for slot in active if slot["slot_id"] not in seen]
    epoch_counts = Counter(
        event["resume_epoch"] for event in sealed_events if event["cost"]["fresh_processes"]
    )
    resume_epoch_count = (
        max((event["resume_epoch"] for event in sealed_events), default=-1) + 1
    )
    journal = {
        "schema_version": ACTIVE_SCHEMA_VERSION,
        "artifact_kind": "m6_raw_observation_journal",
        "journal_id": None,
        "created_at_utc": created_at_utc or _now(),
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity_sha256": schedule["source_identity_sha256"],
        "config_identity_sha256": schedule["config_identity_sha256"],
        "metric_identity_sha256": schedule["metric_identity_sha256"],
        "m4_identity_sha256": schedule["m4_identity"]["identity_sha256"],
        "trigger_state": trigger_state,
        "trigger_decisions": decisions,
        "events": sealed_events,
        "resume": {
            "active_slots": len(active),
            "terminal_slots": len(terminal_ids),
            "pending_slots": len(pending_ids),
            "resume_epochs": resume_epoch_count,
            "max_profiles_in_one_epoch": max(epoch_counts.values(), default=0),
            "terminal_slot_ids_sha256": canonical_sha256(terminal_ids),
            "pending_slot_ids_sha256": canonical_sha256(pending_ids),
            "completed_slots_recomputed": False,
        },
        "cost": {
            "fresh_processes": sum(event["cost"]["fresh_processes"] for event in sealed_events),
            "ncu_replay_passes": sum(event["cost"]["ncu_replay_passes"] for event in sealed_events),
            "observed_slots": sum(event["outcome"] == "observed" for event in sealed_events),
            "excluded_typed_slots": sum(event["outcome"] == "excluded_typed" for event in sealed_events),
        },
    }
    journal["causal_prefix_closed"] = _causal_prefix_closed(
        schedule,
        sealed_events,
        trigger_state=trigger_state,
        trigger_decisions=decisions,
    )
    journal["causal_events"] = _build_causal_records_v2(
        schedule,
        sealed_events,
        trigger_state=trigger_state,
        trigger_decisions=decisions,
    )
    journal["journal_id"] = _derive_journal_id(
        schedule["schedule_sha256"], journal,
    )
    journal["journal_sha256"] = canonical_sha256(journal)
    _validate_causal_records(
        schedule,
        journal,
        require_prefix=bool(journal["causal_prefix_closed"]),
    )
    _validate_schema("observation_journal", journal, repo_root, cache=input_cache)
    return journal


def publish_journal(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    *,
    output: Path,
    repo_root: Path = REPO_ROOT,
    _input_cache: InputSnapshotCache | None = None,
) -> None:
    repository = _canonical_repo_root(repo_root)
    output = _lexical_absolute(output, role="M6 journal output")
    validate_schedule(
        schedule,
        repo_root=repository,
        rehash_sources=True,
        _input_cache=_input_cache,
    )
    _require_active_schema("schedule", schedule)
    _require_active_schema("observation_journal", journal)
    if journal.get("schema_version") != schedule.get("schema_version"):
        _fail("M6_SCHEMA_MIXED", "schedule and journal schema versions differ")
    if not journal.get("causal_prefix_closed", False):
        _fail("M6_JOURNAL_PREFIX", "only an exact eligible prefix may be published")
    validate_journal(
        schedule,
        journal,
        journal_path=output,
        repo_root=repository,
        rehash_sidecars=True,
        _input_cache=_input_cache,
    )
    _write_new(output, journal)


def validate_journal(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    *,
    journal_path: Path,
    repo_root: Path = REPO_ROOT,
    rehash_sidecars: bool = True,
    evidence_root: Path | None = None,
    _evidence_tree: _PinnedTreeTransaction | None = None,
    _input_cache: InputSnapshotCache | None = None,
) -> None:
    input_cache = _input_cache or InputSnapshotCache()
    _validate_schema("observation_journal", journal, repo_root, cache=input_cache)
    schedule_version = _artifact_schema_version(schedule, kind="schedule")
    journal_version = _artifact_schema_version(journal, kind="observation_journal")
    if schedule_version != journal_version:
        _fail("M6_SCHEMA_MIXED", "schedule and journal schema versions differ")
    validate_schedule(
        schedule,
        repo_root=repo_root,
        rehash_sources=True,
        _input_cache=input_cache,
    )
    if rehash_sidecars and _evidence_tree is None:
        journal_root = Path(
            os.path.abspath(
                os.fspath(evidence_root if evidence_root is not None else journal_path.parent)
            )
        )
        evidence_paths: list[Path] = []
        for event in journal["events"]:
            if event["outcome"] != "observed":
                continue
            for ref in event["sidecars"]:
                relative = _safe_relative(
                    ref["relative_path"], role="external v5 sidecar",
                )
                evidence_paths.append(journal_root / relative)
                launch_evidence = ref.get("launch_evidence")
                if isinstance(launch_evidence, dict):
                    for role in (
                        "label_run_manifest", "ncu_report",
                        "ncu_launch_observation", "label_capture_manifest",
                    ):
                        evidence_relative = _safe_relative(
                            launch_evidence[role]["relative_path"],
                            role=f"M6 {role}",
                        )
                        evidence_paths.append(journal_root / evidence_relative)
        if evidence_paths:
            # Evidence references are journal-root-relative.  The pinned
            # transaction must therefore be rooted at that same namespace;
            # using a deeper common parent would make every relative lookup
            # resolve one directory too deep during relocation verification.
            common_root = journal_root
            with _PinnedTreeTransaction(
                common_root,
                role="M6 retained journal evidence",
                input_cache=input_cache,
            ) as transaction:
                validate_journal(
                    schedule,
                    journal,
                    journal_path=journal_path,
                    repo_root=repo_root,
                    rehash_sidecars=True,
                    evidence_root=journal_root,
                    _evidence_tree=transaction,
                    _input_cache=input_cache,
                )
                transaction.verify()
                return
    _validate_seal(journal, "journal_sha256", role="journal")
    bindings = {
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity_sha256": schedule["source_identity_sha256"],
        "config_identity_sha256": schedule["config_identity_sha256"],
        "metric_identity_sha256": schedule["metric_identity_sha256"],
        "m4_identity_sha256": schedule["m4_identity"]["identity_sha256"],
    }
    if any(journal.get(key) != value for key, value in bindings.items()):
        _fail("M6_JOURNAL_BINDING", "schedule/source/config/metric/M4 identity")
    decisions = journal["trigger_decisions"]
    if journal["trigger_state"] == "pending" and decisions:
        _fail("M6_TRIGGER_PENDING_ROWS", "pending journal carries decisions")
    if journal["trigger_state"] == "decided":
        if len(decisions) != EXPECTED_RUNNABLE:
            _fail("M6_TRIGGER_DECISION_COUNT", "expected 122")
        cells = {cell["cell_id"]: cell for cell in schedule["cells"]}
        if {decision["cell_id"] for decision in decisions} != set(cells):
            _fail("M6_TRIGGER_DECISION_COVERAGE", "cell set differs")
        if [decision["cell_id"] for decision in decisions] != list(cells):
            _fail("M6_TRIGGER_DECISION_ORDER", "decision order differs")
        for decision in decisions:
            if decision["cell_identity_sha256"] != cells[decision["cell_id"]]["cell_identity_sha256"]:
                _fail("M6_TRIGGER_CELL_IDENTITY", decision["cell_id"])
            _validate_seal(decision, "decision_sha256", role="trigger decision")
            if decision["triggered"] != bool(decision["reason_codes"]):
                _fail("M6_TRIGGER_BOOLEAN", decision["cell_id"])
    active = _active_slots(schedule, decisions)
    active_by_id = {slot["slot_id"]: slot for slot in active}
    metrics_by_key, _ = _metric_maps(schedule)
    seen: set[str] = set()
    previous = ZERO_SHA256
    epoch_counts: Counter[int] = Counter()
    derived_epochs = _derive_event_epochs(
        journal["events"],
        max_profiles_per_epoch=schedule["cost_gate"]["max_new_profiles_per_resume"],
    )
    journal_dir = Path(
        os.path.abspath(
            os.fspath(evidence_root if evidence_root is not None else journal_path.parent)
        )
    )
    for sequence, (event, derived_epoch) in enumerate(zip(journal["events"], derived_epochs, strict=True)):
        if event.get("resume_epoch") != derived_epoch:
            _fail("M6_JOURNAL_EPOCH", event.get("slot_id", sequence))
        if event["sequence"] != sequence or event["previous_event_sha256"] != previous:
            _fail("M6_JOURNAL_CHAIN", str(sequence))
        if event["event_sha256"] != canonical_sha256(_without(event, "event_sha256")):
            _fail("M6_EVENT_SEAL", str(sequence))
        previous = event["event_sha256"]
        slot = active_by_id.get(event["slot_id"])
        if slot is None or event["slot_id"] in seen:
            _fail("M6_JOURNAL_SLOT", event["slot_id"])
        seen.add(event["slot_id"])
        if event["ordinal"] != slot["ordinal"]:
            _fail("M6_EVENT_SLOT_BINDING", f"{event['slot_id']}:ordinal")
        for field in ("slot_sha256", "tier", "cell_id", "cell_identity_sha256", "config_id", "config_sha256", "repeat_index", "sweep_index"):
            if event[field] != slot[field]:
                _fail("M6_EVENT_SLOT_BINDING", f"{event['slot_id']}:{field}")
        if event["cost"]["fresh_processes"]:
            epoch_counts[event["resume_epoch"]] += event["cost"]["fresh_processes"]
        if event["outcome"] == "observed":
            if (
                event["fresh_process"] is not True
                or event["cost"]["fresh_processes"] != 1
                or event["covariates"]["sweep_index"] != slot["sweep_index"]
            ):
                _fail("M6_FRESH_PROCESS_CONTRACT", event["slot_id"])
            if event["cost"]["ncu_replay_passes"] != event["covariates"]["ncu_replay_pass_count"]:
                _fail("M6_EVENT_COST", event["slot_id"])
            expected_refs: list[dict[str, Any]] = []
            expected_rows: list[dict[str, Any]] = []
            for ref in event["sidecars"]:
                rel = _safe_relative(ref["relative_path"], role="external v5 sidecar")
                sidecar_path = journal_dir / rel
                launch_evidence = ref.get("launch_evidence")
                complete_m4 = schedule["m4_identity"]["status"] == "complete"
                if complete_m4 != (
                    isinstance(launch_evidence, dict)
                    and _sha256_text(ref.get("m4_launch_identity_sha256"))
                ):
                    _fail("M6_M4_REPEAT_EVIDENCE_STATE", event["slot_id"])
                launch_paths: dict[str, Path] | None = None
                if isinstance(launch_evidence, dict):
                    launch_paths = {}
                    for role in (
                        "label_run_manifest", "ncu_report",
                        "ncu_launch_observation", "label_capture_manifest",
                    ):
                        evidence_ref = launch_evidence.get(role)
                        if not isinstance(evidence_ref, dict):
                            _fail("M6_REPEAT_EVIDENCE_REFERENCE", event["slot_id"])
                        evidence_rel = _safe_relative(
                            evidence_ref.get("relative_path"), role=f"M6 {role}",
                        )
                        launch_paths[role] = journal_dir / evidence_rel
                if rehash_sidecars:
                    built_ref, rows, sidecar = _sidecar_ref_and_observations(
                        sidecar_path,
                        journal_dir=journal_dir,
                        schedule=schedule,
                        slot=slot,
                        repo_root=repo_root,
                        launch_evidence_paths=launch_paths,
                        evidence_tree=_evidence_tree,
                        input_cache=input_cache,
                    )
                    if built_ref != ref:
                        _fail("M6_SIDECAR_REFERENCE", event["slot_id"])
                    protocol = sidecar["measurement_protocol"]
                    cov = event["covariates"]
                    if (
                        cov["sm_clock_mhz"] != protocol["host_sm_clock_mhz"]
                        or cov["mem_clock_mhz"] != protocol["host_mem_clock_mhz"]
                        or cov["temperature_c"] != protocol["host_temperature_c"]
                    ):
                        _fail("M6_COVARIATE_BINDING", event["slot_id"])
                    expected_refs.append(built_ref)
                    expected_rows.extend(rows)
            launches = [ref["launch_index"] for ref in event["sidecars"]]
            if len(launches) != len(set(launches)):
                _fail("M6_EVENT_LAUNCH_DUPLICATE", event["slot_id"])
            expected_rows.sort(key=lambda row: (row["launch_index"], row["metric_key"], row["source_pointer"]))
            if rehash_sidecars and event["raw_observations"] != expected_rows:
                _fail("M6_RAW_OBSERVATION_MISMATCH", event["slot_id"])
            for row in event["raw_observations"]:
                metric = metrics_by_key.get(row["metric_key"])
                if metric is None or row["metric_sha256"] != metric["metric_sha256"]:
                    _fail("M6_RAW_METRIC_IDENTITY", event["slot_id"])
                _number(row["value"], role=f"raw observation {event['slot_id']}")
                source_value = row["source_emitted_value"]
                source_unit = row["source_emitted_unit"]
                if (source_value is None) != (source_unit is None):
                    _fail("M6_RAW_SOURCE_VALUE_PAIR", event["slot_id"])
                if source_value is not None:
                    _number(source_value, role=f"raw source observation {event['slot_id']}")
        else:
            if (
                event["sidecars"]
                or event["raw_observations"]
                or event["typed_exclusion"] is None
                or event.get("covariates") is not None
            ):
                _fail("M6_TYPED_EXCLUSION_SHAPE", event["slot_id"])
            _validate_typed_exclusion_contract(
                reason=event["typed_exclusion"].get("reason"),
                detail=event["typed_exclusion"].get("detail"),
                fresh_processes=event["cost"].get("fresh_processes"),
                ncu_replay_passes=event["cost"].get("ncu_replay_passes"),
                operator_authority_receipt=event["typed_exclusion"].get(
                    "operator_authority_receipt"
                ),
                role=event["slot_id"],
            )
            if event["fresh_process"] is not (
                event["cost"]["fresh_processes"] == 1
            ):
                _fail("M6_TYPED_EXCLUSION_CONTRACT", event["slot_id"])
    if journal["trigger_state"] == "decided":
        derived_decisions = _trigger_decisions(
            schedule, _trigger_rows_from_events(schedule, journal["events"])
        )
        if decisions != derived_decisions:
            _fail(
                "M6_TRIGGER_DECISION_MISMATCH",
                "journal decisions do not rederive from uncensored terminal observations",
            )
    if epoch_counts and max(epoch_counts.values()) > schedule["cost_gate"]["max_new_profiles_per_resume"]:
        _fail("M6_COST_GATE_EXCEEDED", str(max(epoch_counts.values())))
    terminal_ids = [slot["slot_id"] for slot in active if slot["slot_id"] in seen]
    pending_ids = [slot["slot_id"] for slot in active if slot["slot_id"] not in seen]
    derived_epoch_count = max(derived_epochs, default=-1) + 1
    resume = journal["resume"]
    derived_resume = {
        "active_slots": len(active),
        "terminal_slots": len(terminal_ids),
        "pending_slots": len(pending_ids),
        "resume_epochs": derived_epoch_count,
        "max_profiles_in_one_epoch": max(epoch_counts.values(), default=0),
        "terminal_slot_ids_sha256": canonical_sha256(terminal_ids),
        "pending_slot_ids_sha256": canonical_sha256(pending_ids),
        "completed_slots_recomputed": False,
    }
    if resume != derived_resume:
        _fail("M6_RESUME_ACCOUNTING", "journal resume summary differs")
    events = journal["events"]
    derived_cost = {
        "fresh_processes": sum(event["cost"]["fresh_processes"] for event in events),
        "ncu_replay_passes": sum(event["cost"]["ncu_replay_passes"] for event in events),
        "observed_slots": sum(event["outcome"] == "observed" for event in events),
        "excluded_typed_slots": sum(event["outcome"] == "excluded_typed" for event in events),
    }
    if journal["cost"] != derived_cost:
        _fail("M6_COST_ACCOUNTING", "journal cost summary differs")
    expected_journal_id = _derive_journal_id(
        schedule["schedule_sha256"], journal,
    )
    if journal.get("journal_id") != expected_journal_id:
        _fail("M6_JOURNAL_ID", "journal identity differs from schedule/events")
    if journal.get("causal_events") is not None:
        _validate_causal_records(
            schedule,
            journal,
            require_prefix=bool(journal.get("causal_prefix_closed")),
        )


def _validity(value: float, metric: Mapping[str, Any]) -> tuple[str, str | None]:
    if value < float(metric["valid_min"]):
        return "out_of_range_low", "registry_lower_bound"
    valid_max = metric["valid_max"]
    if valid_max is not None and value > float(valid_max):
        return "out_of_range_high", "ncu_pct_overflow" if metric["unit"] == "pct" else "registry_upper_bound"
    return "in_range", None


def _row(base: dict[str, Any]) -> dict[str, Any]:
    identity = canonical_sha256(base)
    result = {"row_id": f"m6-row-{identity[:24]}", **base}
    result["row_sha256"] = canonical_sha256(result)
    return result


def _expected_launches(schedule: Mapping[str, Any], event: Mapping[str, Any] | None, cell_id: str) -> list[int | None]:
    if schedule["m4_identity"]["status"] == "complete":
        item = next(item for item in schedule["m4_identity"]["cell_launch_identities"] if item["cell_id"] == cell_id)
        launches = item["launches"]
        if len(launches) != 1:
            _fail("M6_M4_LAUNCH_CARDINALITY", cell_id)
        return [launches[0]["launch_index"]]
    if event is not None and event["outcome"] == "observed":
        return [ref["launch_index"] for ref in event["sidecars"]]
    return [None]


def _trigger_decisions(
    schedule: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["config_id"] == REFERENCE_CONFIG and row["repeat_index"] < 20 and row["value"] is not None:
            by_group[(row["cell_id"], row["metric_key"])].append(row)
    metrics = {metric["key"]: metric for metric in schedule["metrics"]}
    decisions: list[dict[str, Any]] = []
    for cell in schedule["cells"]:
        reason_codes: set[str] = set()
        evidence: set[str] = set()
        for (cell_id, metric_key), group in by_group.items():
            if cell_id != cell["cell_id"]:
                continue
            metric = metrics[metric_key]
            in_range = [float(row["value"]) for row in group if row["validity"] == "in_range"]
            any_oor = any(row["validity"] != "in_range" for row in group)
            if len(in_range) >= 2 or any_oor:
                codes = trigger_codes(
                    in_range,
                    rule=metric["repeatability_rule"],
                    absolute_resolution=metric["absolute_resolution"],
                    discrete=metric["statistical_domain"] == "discrete",
                    any_out_of_range=any_oor,
                )
                if codes:
                    evidence.add(metric_key)
                    reason_codes.update(codes)
        decision = {
            "cell_id": cell["cell_id"],
            "cell_identity_sha256": cell["cell_identity_sha256"],
            "triggered": bool(reason_codes),
            "reason_codes": sorted(reason_codes),
            "evidence_metric_keys": sorted(evidence),
        }
        decision["decision_sha256"] = canonical_sha256(decision)
        decisions.append(decision)
    return decisions


def _trigger_rows_from_events(
    schedule: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project uncensored terminal observations into trigger operands.

    Journal summaries are not an authority for adaptive activation.  This
    projection is intentionally small and deterministic; it mirrors the row
    fields consumed by ``_trigger_decisions`` without consulting analysis
    output or any caller-provided decision summary.
    """
    metrics = {metric["key"]: metric for metric in schedule["metrics"]}
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("outcome") != "observed":
            continue
        for observation in event.get("raw_observations", []):
            if not isinstance(observation, Mapping):
                _fail("M6_RAW_OBSERVATION_MISMATCH", "observation must be an object")
            metric = metrics.get(observation.get("metric_key"))
            if metric is None:
                _fail("M6_RAW_METRIC_IDENTITY", str(observation.get("metric_key")))
            value = _number(
                observation.get("value"),
                role=f"trigger observation {event.get('slot_id')}",
            )
            validity, _reason = _validity(value, metric)
            rows.append(
                {
                    "cell_id": event["cell_id"],
                    "config_id": event["config_id"],
                    "metric_key": metric["key"],
                    "repeat_index": event["repeat_index"],
                    "value": value,
                    "validity": validity,
                }
            )
    return rows


def _completion_state(
    *,
    tier1_pending: int,
    trigger_state: str,
    triggered_cells: int,
    tier2_pending: int,
    observed_slots: int,
    excluded_typed_slots: int,
    missing_selected_rows: int,
) -> str:
    if tier1_pending:
        return "incomplete_cost_gate"
    if trigger_state != "decided":
        return "trigger_decisions_required"
    if triggered_cells and tier2_pending:
        return "tier2_required"
    if (
        observed_slots == 0
        or excluded_typed_slots != 0
        or missing_selected_rows != 0
    ):
        return "incomplete_evidence"
    return "complete"


def _historical_admission_key(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": row.get("cell_id"),
        "cell_identity_sha256": row.get("cell_identity_sha256"),
        "metric_key": row.get("metric_key"),
        "metric_sha256": row.get("metric_sha256"),
    }


def _validate_historical_admission_authority_schema(
    authority: Mapping[str, Any],
    *,
    repo_root: Path,
    input_cache: InputSnapshotCache,
) -> None:
    if authority.get("schema_version") != 1:
        _fail("M6_SCHEMA_VERSION", "historical_admission_set: version 1 required")
    schema, _ = _load_schema(
        "historical_admission_set",
        repo_root,
        version=ACTIVE_SCHEMA_VERSION,
        cache=input_cache,
    )
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(authority), key=lambda error: list(error.absolute_path)
    )
    if errors:
        error = errors[0]
        location = "/" + "/".join(str(part) for part in error.absolute_path)
        _fail(
            "M6_SCHEMA_VALIDATION",
            f"historical_admission_set{location}: {error.message}",
        )


def _historical_admission_reconciliation(
    schedule: Mapping[str, Any],
    classifications: Sequence[Mapping[str, Any]],
    *,
    authority_path: Path | None,
    repo_root: Path,
    input_cache: InputSnapshotCache,
) -> dict[str, Any]:
    """Compare the replayed reference projection to a separate exact-set authority."""
    selected_metrics = {
        metric["key"]: metric
        for metric in schedule["metrics"]
        if metric["capture_status"] == "selected"
    }
    expected_keys = [
        {
            "cell_id": cell["cell_id"],
            "cell_identity_sha256": cell["cell_identity_sha256"],
            "metric_key": metric_key,
            "metric_sha256": selected_metrics[metric_key]["metric_sha256"],
        }
        for cell in schedule["cells"]
        for metric_key in sorted(selected_metrics)
    ]
    expected_keys.sort(key=lambda row: (row["cell_id"], row["metric_key"]))
    expected_by_identity = {
        (row["cell_id"], row["metric_key"]): row for row in expected_keys
    }
    projection_rows = [
        row for row in classifications if row.get("config_id") == REFERENCE_CONFIG
    ]
    projected_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in projection_rows:
        identity = (row.get("cell_id"), row.get("metric_key"))
        expected = expected_by_identity.get(identity)
        if expected is None or _historical_admission_key(row) != expected:
            _fail("M6_HISTORICAL_PROJECTION_SCOPE", repr(identity))
        if identity in projected_by_identity:
            _fail("M6_HISTORICAL_PROJECTION_DUPLICATE", repr(identity))
        projected_by_identity[identity] = row
    projection_complete = (
        len(projected_by_identity) == len(expected_by_identity)
        and set(projected_by_identity) == set(expected_by_identity)
        and all(
            type(row.get("would_have_been_admitted")) is bool
            for row in projected_by_identity.values()
        )
    )
    projected_admitted = [
        expected_by_identity[identity]
        for identity, row in sorted(projected_by_identity.items())
        if row.get("would_have_been_admitted") is True
    ]
    reconciliation: dict[str, Any] = {
        "status": "external_pending",
        "projection_contract": HISTORICAL_ADMISSION_PROJECTION,
        "reference_config_id": REFERENCE_CONFIG,
        "expected_projection_count": len(expected_keys),
        "observed_projection_count": len(projected_by_identity),
        "projection_complete": projection_complete,
        "projected_admitted_count": len(projected_admitted),
        "projected_admitted_set_sha256": canonical_sha256(projected_admitted),
        "authority": None,
        "exact_set_match": None,
        "typed_reason": "historical_admitted_set_not_supplied",
    }
    if authority_path is None:
        reconciliation["reconciliation_sha256"] = canonical_sha256(reconciliation)
        return reconciliation

    authority, authority_payload = load_json(
        authority_path,
        role="historical admission set authority",
        cache=input_cache,
    )
    authority_artifact_sha256 = sha256_bytes(authority_payload)
    admitted_artifact_sha256 = HISTORICAL_ADMISSION_AUTHORITY_ARTIFACT_SHA256
    if admitted_artifact_sha256 is None:
        _fail(
            "M6_HISTORICAL_AUTHORITY_UNADMITTED",
            "no historical admission authority artifact has an admitted trust anchor",
        )
    if authority_artifact_sha256 != admitted_artifact_sha256:
        _fail(
            "M6_HISTORICAL_AUTHORITY_UNADMITTED",
            "historical admission authority artifact does not match the admitted trust anchor",
        )
    if not isinstance(authority, dict):
        _fail("M6_HISTORICAL_AUTHORITY_SHAPE", "object required")
    _validate_historical_admission_authority_schema(
        authority,
        repo_root=repo_root,
        input_cache=input_cache,
    )
    _validate_seal(
        authority, "authority_sha256", role="historical admission set authority"
    )
    admitted_keys = authority.get("admitted_keys")
    if not isinstance(admitted_keys, list):
        _fail("M6_HISTORICAL_AUTHORITY_SHAPE", "admitted_keys")
    normalized_admitted = sorted(
        admitted_keys, key=lambda row: (row["cell_id"], row["metric_key"])
    )
    if admitted_keys != normalized_admitted:
        _fail("M6_HISTORICAL_AUTHORITY_ORDER", "admitted_keys")
    identities = [(row["cell_id"], row["metric_key"]) for row in admitted_keys]
    if len(identities) != len(set(identities)):
        _fail("M6_HISTORICAL_AUTHORITY_DUPLICATE", "admitted_keys")
    for row, identity in zip(admitted_keys, identities):
        if expected_by_identity.get(identity) != row:
            _fail("M6_HISTORICAL_AUTHORITY_SCOPE", repr(identity))
    if authority.get("admitted_keys_sha256") != canonical_sha256(admitted_keys):
        _fail("M6_HISTORICAL_AUTHORITY_SET_SEAL", "admitted_keys")
    source = schedule["source_identity"]
    expected_bindings = {
        "canonical_matrix_sha256": source["canonical_matrix"]["sha256"],
        "matrix_seal_sha256": source["matrix_seal_sha256"],
        "roster_authority_sha256": source["roster_authority_sha256"],
        "metric_identity_sha256": schedule["metric_identity_sha256"],
        "reference_config_id": REFERENCE_CONFIG,
    }
    if authority.get("projection_contract") != HISTORICAL_ADMISSION_PROJECTION:
        _fail("M6_HISTORICAL_AUTHORITY_BINDING", "projection_contract")
    if authority.get("bindings") != expected_bindings:
        _fail("M6_HISTORICAL_AUTHORITY_BINDING", "schedule authority identities")
    if authority.get("expected_projection_count") != len(expected_keys):
        _fail("M6_HISTORICAL_AUTHORITY_BINDING", "expected projection count")

    exact_match = projection_complete and admitted_keys == projected_admitted
    reconciliation.update(
        {
            "status": "reconciled" if exact_match else (
                "set_mismatch" if projection_complete else "projection_incomplete"
            ),
            "authority": {
                "authority_id": authority["authority_id"],
                "issued_at_utc": authority["issued_at_utc"],
                "artifact_sha256": authority_artifact_sha256,
                "size_bytes": len(authority_payload),
                "authority_sha256": authority["authority_sha256"],
                "admitted_count": len(admitted_keys),
                "admitted_set_sha256": authority["admitted_keys_sha256"],
                "bindings": expected_bindings,
            },
            "exact_set_match": exact_match if projection_complete else None,
            "typed_reason": None if exact_match else (
                "historical_admitted_set_mismatch"
                if projection_complete
                else "historical_projection_incomplete"
            ),
        }
    )
    reconciliation["reconciliation_sha256"] = canonical_sha256(reconciliation)
    return reconciliation


def analyze(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    *,
    journal_path: Path,
    generated_at_utc: str | None = None,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int | None = None,
    historical_admission_path: Path | None = None,
    evidence_root: Path | None = None,
    repo_root: Path = REPO_ROOT,
    _input_cache: InputSnapshotCache | None = None,
) -> dict[str, Any]:
    input_cache = _input_cache or InputSnapshotCache()
    validate_schedule(
        schedule,
        repo_root=repo_root,
        rehash_sources=True,
        _input_cache=input_cache,
    )
    _require_active_schema("schedule", schedule)
    _require_active_schema("observation_journal", journal)
    if journal.get("schema_version") != schedule.get("schema_version"):
        _fail("M6_SCHEMA_MIXED", "schedule and journal schema versions differ")
    validate_journal(
        schedule,
        journal,
        journal_path=journal_path,
        repo_root=repo_root,
        rehash_sidecars=True,
        evidence_root=evidence_root,
        _input_cache=input_cache,
    )
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        _fail("M6_BOOTSTRAP_REPLICATES", "positive integer required")
    seed = schedule["seed"] if bootstrap_seed is None else bootstrap_seed
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= MAX_SAFE_JSON_INTEGER
    ):
        _fail("M6_BOOTSTRAP_SEED", "safe nonnegative integer required")
    metrics = {metric["key"]: metric for metric in schedule["metrics"]}
    selected_metrics = [metric for metric in schedule["metrics"] if metric["capture_status"] == "selected"]
    events = {event["slot_id"]: event for event in journal["events"]}
    rows: list[dict[str, Any]] = []
    for event in journal["events"]:
        if event["outcome"] == "observed":
            present: set[tuple[int, str]] = set()
            for observation in event["raw_observations"]:
                metric = metrics[observation["metric_key"]]
                value = _number(observation["value"], role="analysis raw value")
                validity, reason = _validity(value, metric)
                base = {
                    "slot_id": event["slot_id"],
                    "slot_sha256": event["slot_sha256"],
                    "cell_id": event["cell_id"],
                    "cell_identity_sha256": event["cell_identity_sha256"],
                    "config_id": event["config_id"],
                    "config_sha256": event["config_sha256"],
                    "metric_key": metric["key"],
                    "metric_sha256": metric["metric_sha256"],
                    "repeat_index": event["repeat_index"],
                    "sweep_index": event["sweep_index"],
                    "launch_index": observation["launch_index"],
                    "sidecar_sha256": observation["sidecar_sha256"],
                    "value": value,
                    "source_emitted_value": observation["source_emitted_value"],
                    "source_emitted_unit": observation["source_emitted_unit"],
                    "validity": validity,
                    "disposition": "classified" if validity == "in_range" else "excluded_typed",
                    "typed_reason": reason,
                }
                rows.append(_row(base))
                present.add((observation["launch_index"], metric["key"]))
            for launch in _expected_launches(schedule, event, event["cell_id"]):
                if launch is None:
                    continue
                for metric in selected_metrics:
                    if (launch, metric["key"]) in present:
                        continue
                    rows.append(_row({
                        "slot_id": event["slot_id"],
                        "slot_sha256": event["slot_sha256"],
                        "cell_id": event["cell_id"],
                        "cell_identity_sha256": event["cell_identity_sha256"],
                        "config_id": event["config_id"],
                        "config_sha256": event["config_sha256"],
                        "metric_key": metric["key"],
                        "metric_sha256": metric["metric_sha256"],
                        "repeat_index": event["repeat_index"],
                        "sweep_index": event["sweep_index"],
                        "launch_index": launch,
                        "sidecar_sha256": None,
                        "value": None,
                        "source_emitted_value": None,
                        "source_emitted_unit": None,
                        "validity": "not_collected",
                        "disposition": "not_collected_typed",
                        "typed_reason": "metric_missing_from_sidecar",
                    }))
        else:
            for launch in _expected_launches(schedule, event, event["cell_id"]):
                for metric in selected_metrics:
                    rows.append(_row({
                        "slot_id": event["slot_id"],
                        "slot_sha256": event["slot_sha256"],
                        "cell_id": event["cell_id"],
                        "cell_identity_sha256": event["cell_identity_sha256"],
                        "config_id": event["config_id"],
                        "config_sha256": event["config_sha256"],
                        "metric_key": metric["key"],
                        "metric_sha256": metric["metric_sha256"],
                        "repeat_index": event["repeat_index"],
                        "sweep_index": event["sweep_index"],
                        "launch_index": launch,
                        "sidecar_sha256": None,
                        "value": None,
                        "source_emitted_value": None,
                        "source_emitted_unit": None,
                        "validity": "not_collected",
                        "disposition": "not_collected_typed",
                        "typed_reason": event["typed_exclusion"]["reason"],
                    }))
    rows.sort(key=lambda row: (row["cell_id"], row["config_id"], row["metric_key"], row["repeat_index"], -1 if row["launch_index"] is None else row["launch_index"], row["row_id"]))
    decisions = _trigger_decisions(schedule, rows)
    if journal["trigger_state"] == "decided" and journal["trigger_decisions"] != decisions:
        _fail("M6_TRIGGER_DECISION_MISMATCH", "journal decisions do not rederive from uncensored tier-1 rows")
    triggered_cells = {decision["cell_id"] for decision in decisions if decision["triggered"]}
    tier1_ids = {slot["slot_id"] for slot in schedule["slots"]}
    required_tier2 = {
        slot["slot_id"] for slot in schedule["tier2_candidates"] if slot["cell_id"] in triggered_cells
    }
    event_ids = set(events)
    tier1_pending = len(tier1_ids - event_ids)
    tier2_pending = len(required_tier2 - event_ids)
    completion = _completion_state(
        tier1_pending=tier1_pending,
        trigger_state=journal["trigger_state"],
        triggered_cells=len(triggered_cells),
        tier2_pending=tier2_pending,
        observed_slots=journal["cost"]["observed_slots"],
        excluded_typed_slots=journal["cost"]["excluded_typed_slots"],
        missing_selected_rows=sum(
            row["disposition"] == "not_collected_typed" for row in rows
        ),
    )
    group_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_rows[(row["cell_id"], row["config_id"], row["metric_key"])].append(row)
    if completion == "complete":
        group_keys = [
            (cell["cell_id"], config["config_id"], metric["key"])
            for cell in schedule["cells"]
            for config in schedule["configs"]
            for metric in selected_metrics
        ]
    else:
        terminal_pairs = {(event["cell_id"], event["config_id"]) for event in journal["events"]}
        group_keys = sorted(set(group_rows) | {
            (cell_id, config_id, metric["key"])
            for cell_id, config_id in terminal_pairs
            for metric in selected_metrics
        })
    cells_by_id = {cell["cell_id"]: cell for cell in schedule["cells"]}
    configs_by_id = {config["config_id"]: config for config in schedule["configs"]}
    classifications: list[dict[str, Any]] = []
    class_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    mean_by_key: dict[tuple[str, str, str], float] = {}
    for cell_id, config_id, metric_key in sorted(group_keys):
        metric = metrics[metric_key]
        group = group_rows.get((cell_id, config_id, metric_key), [])
        by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_slot[row["slot_id"]].append(row)
        in_range_values: list[float] = []
        out_of_range_repeats = 0
        for slot_rows in by_slot.values():
            valid = [float(row["value"]) for row in slot_rows if row["validity"] == "in_range"]
            if any(row["validity"].startswith("out_of_range") for row in slot_rows):
                out_of_range_repeats += 1
            if valid:
                in_range_values.append(statistics.fmean(valid))
        gate = dispersion_gate(metric["repeatability_rule"], in_range_values) if len(in_range_values) >= 2 else None
        rel = sigma_rel(metric["repeatability_rule"], in_range_values, absolute_resolution=metric["absolute_resolution"]) if len(in_range_values) >= 2 else None
        rel_upper = (
            rel * sigma_upper_factor(len(in_range_values))
            if rel is not None and len(in_range_values) >= 2
            else None
        )
        dispersion_class = classify_sigma(
            rel_upper if rel_upper is not None else rel, in_range_values
        )
        tier2 = config_id == REFERENCE_CONFIG and len(in_range_values) >= 30
        modality = modality_result(
            in_range_values,
            discrete=metric["statistical_domain"] == "discrete",
            absolute_resolution=metric["absolute_resolution"],
            tier2=tier2,
        )
        regime_means: list[float] = []
        regime_counts: list[int] = []
        within_regime_sigmas: list[float] = []
        if modality.certified:
            sorted_values = sorted(in_range_values)
            if modality.split_index is not None:
                regimes = [
                    sorted_values[: modality.split_index],
                    sorted_values[modality.split_index :],
                ]
            else:
                regimes = [
                    [value for value in sorted_values if value == distinct]
                    for distinct in sorted(set(sorted_values))
                    if sorted_values.count(distinct) >= 3
                ]
            for regime in regimes:
                if not regime:
                    continue
                regime_mean = statistics.fmean(regime)
                regime_means.append(regime_mean)
                regime_counts.append(len(regime))
                if len(regime) >= 2:
                    denominator = max(abs(regime_mean), metric["absolute_resolution"])
                    within_regime_sigmas.append(statistics.stdev(regime) / denominator)
        within_regime_sigma = max(within_regime_sigmas, default=0.0) if modality.certified else None
        if out_of_range_repeats:
            determinism_class = "vendor_defect"
        elif len(in_range_values) < 5:
            determinism_class = "insufficient_evidence"
        elif modality.certified:
            determinism_class = "multimodal"
        else:
            determinism_class = dispersion_class
        annotations: list[str] = []
        if classify_sigma(rel, in_range_values) != dispersion_class:
            annotations.append("uncertain_boundary")
        if modality.suspected:
            annotations.append("suspected_multimodal")
        if modality.outlier_flag:
            annotations.append("outlier_flag")
        if out_of_range_repeats:
            annotations.append("oor_floor_lower_bound")
        if cell_id in triggered_cells and config_id == REFERENCE_CONFIG and len(in_range_values) < 30:
            annotations.append("tier2_not_collected")
        expected_repeats = 30 if cell_id in triggered_cells and config_id == REFERENCE_CONFIG else (20 if config_id == REFERENCE_CONFIG else 10)
        row = {
            "cell_id": cell_id,
            "cell_identity_sha256": cells_by_id[cell_id]["cell_identity_sha256"],
            "config_id": config_id,
            "config_sha256": configs_by_id[config_id]["config_sha256"],
            "metric_key": metric_key,
            "metric_sha256": metric["metric_sha256"],
            "repeatability_rule": metric["repeatability_rule"],
            "expected_repeats": expected_repeats,
            "collected_repeats": len(by_slot),
            "in_range_repeats": len(in_range_values),
            "out_of_range_repeats": out_of_range_repeats,
            "values_sha256": canonical_sha256(in_range_values),
            "mean_in_range": statistics.fmean(in_range_values) if in_range_values else None,
            "dispersion_gate": gate,
            "admission_threshold": metric["admission_threshold"],
            "would_have_been_admitted": None if gate is None else gate <= metric["admission_threshold"],
            "sigma_rel": rel,
            "sigma_rel_upper_95": rel_upper,
            "determinism_class": determinism_class,
            "dispersion_class": dispersion_class,
            "modality": {
                "flagged": modality.flagged,
                "outlier_flag": modality.outlier_flag,
                "certified": modality.certified,
                "suspected": modality.suspected,
                "split_index": modality.split_index,
                "gap": modality.gap,
                "reason": modality.reason,
                "regime_means": regime_means,
                "regime_counts": regime_counts,
                "within_regime_sigma_rel": within_regime_sigma,
            },
            "annotations": sorted(set(annotations)),
        }
        row["classification_sha256"] = canonical_sha256(row)
        classifications.append(row)
        class_by_key[(cell_id, config_id, metric_key)] = row
        if row["mean_in_range"] is not None:
            mean_by_key[(cell_id, config_id, metric_key)] = row["mean_in_range"]
    oor_rates: list[dict[str, Any]] = []
    for metric in selected_metrics:
        for config in schedule["configs"]:
            subset = [row for row in rows if row["metric_key"] == metric["key"] and row["config_id"] == config["config_id"] and row["value"] is not None]
            count = sum(row["validity"] != "in_range" for row in subset)
            oor_rates.append({
                "metric_key": metric["key"],
                "config_id": config["config_id"],
                "observed_rows": len(subset),
                "out_of_range_rows": count,
                "rate": count / len(subset) if subset else None,
                "floor_is_lower_bound": count > 0,
                "rule_of_three_upper_bound": min(1.0, 3.0 / len(subset)) if subset else None,
                "rare_event_upper_bound": 1.0 / len(subset) if subset else None,
            })
    reference_oor_rows = [
        item for item in oor_rates if item["config_id"] == REFERENCE_CONFIG
    ]
    reference_oor_zero = (
        completion == "complete"
        and len(reference_oor_rows) == len(selected_metrics)
        and all(
            item["observed_rows"] > 0 and item["out_of_range_rows"] == 0
            for item in reference_oor_rows
        )
    )
    noise_floors: list[dict[str, Any]] = []
    floor_by_metric: dict[str, float | None] = {}
    p90_by_metric: dict[str, float | None] = {}
    for metric in selected_metrics:
        values: list[float] = []
        lower_bound = next(item for item in oor_rates if item["metric_key"] == metric["key"] and item["config_id"] == REFERENCE_CONFIG)["out_of_range_rows"] > 0
        for cell in schedule["cells"]:
            classification = class_by_key.get((cell["cell_id"], REFERENCE_CONFIG, metric["key"]))
            if classification and classification["sigma_rel"] is not None and classification["determinism_class"] not in {"vendor_defect", "insufficient_evidence", "multimodal"}:
                n = classification["in_range_repeats"]
                values.append(classification["sigma_rel"] / c4(n) if n >= 2 else classification["sigma_rel"])
        if len(values) >= 30:
            metric_seed = int(hashlib.sha256(f"{seed}:{metric['key']}:floor".encode()).hexdigest()[:16], 16)
            p90, upper = pooled_p90_upper(values, seed=metric_seed, replicates=bootstrap_replicates)
            floor = upper
            status = "available"
        else:
            p90 = upper = floor = None
            status = "insufficient_cells"
        p90_by_metric[metric["key"]] = p90
        floor_by_metric[metric["key"]] = floor
        noise_floors.append({
            "metric_key": metric["key"],
            "valid_cells": len(values),
            "cross_cell_p90": p90,
            "p90_upper_90": upper,
            "absolute_resolution": metric["absolute_resolution"],
            "floor": floor,
            "lower_bound": lower_bound,
            "status": status,
            "fallback_authorized": False,
            "fallback_reason": None,
        })
    denominators: list[dict[str, Any]] = []
    for metric in selected_metrics:
        floor = floor_by_metric[metric["key"]]
        p90 = p90_by_metric[metric["key"]]
        for cell in schedule["cells"]:
            classification = class_by_key.get((cell["cell_id"], REFERENCE_CONFIG, metric["key"]))
            if classification is None:
                continue
            cls = classification["determinism_class"]
            if cls == "insufficient_evidence":
                denominator = override = None
                denominator_kind = "none"
                gate_target = "none"
                status = "no_gate_insufficient_evidence"
            elif cls == "deterministic":
                denominator = metric["absolute_resolution"]
                override = None
                denominator_kind = "absolute"
                gate_target = "exact_value"
                status = "available"
            elif floor is None:
                denominator = override = None
                denominator_kind = "none"
                gate_target = "none"
                status = "pending"
            else:
                sigma = classification["sigma_rel"]
                sigma_upper = classification["sigma_rel_upper_95"]
                override = (
                    sigma_upper
                    if sigma is not None
                    and sigma_upper is not None
                    and p90 is not None
                    and sigma > p90
                    else None
                )
                mean = classification["mean_in_range"]
                resolution_relative = (
                    metric["absolute_resolution"]
                    if mean is None or abs(mean) < metric["absolute_resolution"]
                    else metric["absolute_resolution"] / abs(mean)
                )
                if cls == "multimodal":
                    within = classification["modality"]["within_regime_sigma_rel"]
                    if within is not None:
                        override = max(override or 0.0, within)
                denominator = max(resolution_relative, floor, override or 0.0)
                denominator_kind = "relative"
                gate_target = "nearest_regime_mean" if cls == "multimodal" else "mean"
                status = "available"
            denominators.append({
                "cell_id": cell["cell_id"],
                "metric_key": metric["key"],
                "determinism_class": cls,
                "absolute_resolution": metric["absolute_resolution"],
                "pooled_floor": floor,
                "cell_override": override,
                "denominator": denominator,
                "denominator_kind": denominator_kind,
                "gate_target": gate_target,
                "exact_match": cls == "deterministic",
                "status": status,
            })
    sensitivity: list[dict[str, Any]] = []
    comparisons = [
        ("clock", "cfg=clock_none__cache_all", "cfg=clock_base__cache_all"),
        ("clock", "cfg=clock_none__cache_all", "cfg=clock_boost__cache_all"),
        ("clock", "cfg=clock_none__cache_none", "cfg=clock_base__cache_none"),
        ("clock", "cfg=clock_none__cache_none", "cfg=clock_boost__cache_none"),
        ("cache", "cfg=clock_none__cache_all", "cfg=clock_none__cache_none"),
        ("cache", "cfg=clock_base__cache_all", "cfg=clock_base__cache_none"),
        ("cache", "cfg=clock_boost__cache_all", "cfg=clock_boost__cache_none"),
    ]
    for metric in selected_metrics:
        floor = floor_by_metric[metric["key"]]
        band = max(0.005, 2.0 * floor) if floor is not None else None
        for axis, baseline, comparison in comparisons:
            shifts: list[float] = []
            for cell in schedule["cells"]:
                ref = mean_by_key.get((cell["cell_id"], baseline, metric["key"]))
                other = mean_by_key.get((cell["cell_id"], comparison, metric["key"]))
                if ref is not None and other is not None and ref != 0:
                    shifts.append((other - ref) / abs(ref))
            if shifts and band is not None:
                metric_seed = int(hashlib.sha256(f"{seed}:{metric['key']}:{comparison}".encode()).hexdigest()[:16], 16)
                median, low, high = median_bootstrap_ci(shifts, seed=metric_seed, replicates=bootstrap_replicates)
                if low >= -band and high <= band:
                    label = "protocol_invariant"
                elif low > band or high < -band:
                    label = "protocol_sensitive"
                else:
                    label = "undetermined"
            else:
                median = low = high = None
                label = "undetermined"
            sensitivity.append({
                "metric_key": metric["key"],
                "axis": axis,
                "baseline_config_id": baseline,
                "comparison_config_id": comparison,
                "paired_cells": len(shifts),
                "median_relative_shift": median,
                "ci_lower": low,
                "ci_upper": high,
                "equivalence_band": band,
                "label": label,
            })
    migration_counts: Counter[tuple[str, str, str, str]] = Counter()
    for metric in selected_metrics:
        for config in schedule["configs"]:
            if config["config_id"] == REFERENCE_CONFIG:
                continue
            for cell in schedule["cells"]:
                ref = class_by_key.get((cell["cell_id"], REFERENCE_CONFIG, metric["key"]))
                other = class_by_key.get((cell["cell_id"], config["config_id"], metric["key"]))
                if ref and other:
                    migration_counts[(metric["key"], config["config_id"], ref["determinism_class"], other["determinism_class"])] += 1
    migrations = [
        {
            "metric_key": key[0],
            "from_config_id": REFERENCE_CONFIG,
            "to_config_id": key[1],
            "from_class": key[2],
            "to_class": key[3],
            "cell_count": count,
        }
        for key, count in sorted(migration_counts.items())
    ]
    active_slots = TIER1_SLOTS + len(triggered_cells) * 10
    terminal_slots = len(event_ids & (tier1_ids | required_tier2))
    pending_slots = active_slots - terminal_slots
    selected_raw = [row for row in rows if row["value"] is not None]
    classified_count = sum(row["disposition"] == "classified" for row in selected_raw)
    excluded_count = sum(row["disposition"] == "excluded_typed" for row in selected_raw)
    not_collected = sum(row["disposition"] == "not_collected_typed" for row in rows)
    events_by_slot = {event["slot_id"]: event for event in journal["events"]}
    launch_count_by_slot: dict[str, int] = {}
    expected_rows = 0
    for slot in _active_slots(schedule, decisions):
        event = events_by_slot.get(slot["slot_id"])
        if event is not None and event.get("outcome") == "observed":
            launch_count = len(event.get("sidecars", []))
        elif schedule["m4_identity"]["status"] == "complete":
            launch_count = len(_expected_launches(schedule, None, slot["cell_id"]))
        else:
            launch_count = 1
        if launch_count < 1:
            _fail("M6_DENOMINATOR_LAUNCH_COUNT", slot["slot_id"])
        launch_count_by_slot[slot["slot_id"]] = launch_count
        expected_rows += launch_count * len(selected_metrics)
    pending_expected = expected_rows - len(selected_raw) - not_collected
    if pending_expected < 0:
        _fail("M6_DENOMINATOR_OVERFLOW", str(pending_expected))
    bucket_rows: list[dict[str, Any]] = []
    active = _active_slots(schedule, decisions)
    for config in schedule["configs"]:
        for metric in selected_metrics:
            relevant_active = [slot for slot in active if slot["config_id"] == config["config_id"]]
            expected = sum(
                launch_count_by_slot[slot["slot_id"]] for slot in relevant_active
            )
            subset = [row for row in rows if row["config_id"] == config["config_id"] and row["metric_key"] == metric["key"]]
            collected = sum(row["value"] is not None for row in subset)
            not_coll = sum(row["disposition"] == "not_collected_typed" for row in subset)
            bucket_rows.append({
                "config_id": config["config_id"],
                "metric_key": metric["key"],
                "expected_rows": expected,
                "collected_rows": collected,
                "classified_rows": sum(row["disposition"] == "classified" for row in subset),
                "excluded_typed_rows": sum(row["disposition"] == "excluded_typed" for row in subset),
                "not_collected_typed_rows": not_coll,
                "pending_rows": expected - collected - not_coll,
            })
    accounting = {
        "tier1_slots": TIER1_SLOTS,
        "triggered_cells": len(triggered_cells),
        "active_tier2_slots": len(triggered_cells) * 10,
        "active_slots": active_slots,
        "terminal_slots": terminal_slots,
        "pending_slots": pending_slots,
        "observed_slots": journal["cost"]["observed_slots"],
        "excluded_typed_slots": journal["cost"]["excluded_typed_slots"],
        "raw_rows_collected": len(selected_raw),
        "classified_raw_rows": classified_count,
        "excluded_typed_raw_rows": excluded_count,
        "not_collected_typed_rows": not_collected,
        "expected_rows": expected_rows,
        "pending_expected_rows": pending_expected,
        "denominator_scope": (
            "complete_m4_launch_set"
            if schedule["m4_identity"]["status"] == "complete"
            else "provisional_incomplete_m4"
        ),
        "collected_identity_holds": len(selected_raw) == classified_count + excluded_count,
        "terminal_identity_holds": terminal_slots == journal["cost"]["observed_slots"] + journal["cost"]["excluded_typed_slots"],
        "denominator_identity_holds": expected_rows == len(selected_raw) + not_collected + pending_expected,
        "per_config_metric": bucket_rows,
    }
    if not all(accounting[key] for key in ("collected_identity_holds", "terminal_identity_holds", "denominator_identity_holds")):
        _fail("M6_ACCOUNTING_IDENTITY", repr(accounting))
    multiplicity_discoveries = benjamini_hochberg([], q=0.05)
    multiplicity_policy = {
        "method": "BH-FDR",
        "q": 0.05,
        "family_definition": "metric_key_within_config_v1",
        "status": "external_pending",
        "hypothesis_count": 0,
        "discoveries": list(multiplicity_discoveries),
    }
    multiplicity_policy["discoveries_sha256"] = canonical_sha256(
        multiplicity_policy["discoveries"]
    )
    multiplicity_policy["policy_sha256"] = canonical_sha256(multiplicity_policy)
    historical_reconciliation = _historical_admission_reconciliation(
        schedule,
        classifications,
        authority_path=historical_admission_path,
        repo_root=repo_root,
        input_cache=input_cache,
    )
    if historical_reconciliation["status"] == "reconciled":
        historical_limitation = (
            "reference-configuration dispersion-gate projection exactly matches "
            "the separately supplied historical admission set"
        )
    elif historical_reconciliation["status"] == "set_mismatch":
        historical_limitation = (
            "failed: replayed reference-configuration dispersion-gate projection "
            "does not match the supplied historical admission set"
        )
    elif historical_reconciliation["status"] == "projection_incomplete":
        historical_limitation = (
            "external_pending: the historical authority is supplied but the local "
            "reference-configuration projection is incomplete"
        )
    else:
        historical_limitation = (
            "external_pending: historical admitted set is not supplied"
        )
    analysis = {
        "schema_version": ACTIVE_SCHEMA_VERSION,
        "artifact_kind": "determinism_study",
        "study_id": f"m6-study-{canonical_sha256([schedule['schedule_sha256'], journal['journal_sha256'], seed, bootstrap_replicates])[:16]}",
        "generated_at_utc": generated_at_utc or _now(),
        "schedule_sha256": schedule["schedule_sha256"],
        "journal_sha256": journal["journal_sha256"],
        "source_identity_sha256": schedule["source_identity_sha256"],
        "config_identity_sha256": schedule["config_identity_sha256"],
        "metric_identity_sha256": schedule["metric_identity_sha256"],
        "m4_identity_sha256": schedule["m4_identity"]["identity_sha256"],
        "completion_state": completion,
        "publication_state": (
            "M6_INCOMPLETE_EVIDENCE"
            if completion == "incomplete_evidence"
            else "DIAGNOSTIC_PUBLICATION"
        ),
        "m7_state": "M7_BLOCKED",
        "simulator_scoreability": "UNAVAILABLE_REQUIRED_CYCLES",
        "publication_verdict": "NOT_EVALUATED",
        "taxonomy": list(DETERMINISM_CLASSES),
        "accounting": accounting,
        "trigger_decisions": decisions,
        "row_accounting": rows,
        "classifications": classifications,
        "out_of_range_rates": oor_rates,
        "noise_floors": noise_floors,
        "denominators": denominators,
        "protocol_sensitivity": sensitivity,
        "class_migrations": migrations,
        "analysis_policy": {
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
            "floor_quantile": 0.9,
            "floor_upper_confidence": 0.9,
            "within_session_only": True,
            "raw_values_clamped": False,
            "admitted_summary_consumed": False,
            "reference_oor_zero_confirmed": reference_oor_zero,
            "multiplicity_policy": multiplicity_policy,
            "historical_admission_reconciliation": historical_reconciliation,
            "limitations": {
                "censoring": "finite out-of-range values retained uncensored; floors are lower bounds when OOR exists",
                "finite_n": "per-cell rare-event bounds use rule_of_three_3_over_n and 1_over_n annotations",
                "within_session": "replicates and protocol sensitivity are within one controlled session",
                "multiplicity": "BH-FDR policy is sealed as external_pending; no hypothesis family is admitted until p-values and family membership are supplied",
                "historical_reconciliation": historical_limitation,
            },
            "floor_fallback_authorizations": [],
        },
    }
    analysis["scientific_gates"] = _scientific_gate_snapshot(analysis)
    if (
        completion == "complete"
        and schedule["m4_identity"].get("status") == "complete"
        and analysis["scientific_gates"].get("done_eligible") is True
    ):
        analysis["publication_state"] = "M6_DONE"
    elif completion == "incomplete_evidence":
        analysis["publication_state"] = "M6_INCOMPLETE_EVIDENCE"
    else:
        analysis["publication_state"] = "DIAGNOSTIC_PUBLICATION"
    analysis["m6_m7_handoff_candidate"] = False
    analysis["m7_consumer_contract_version"] = None
    analysis["analysis_sha256"] = canonical_sha256(analysis)
    _validate_schema("analysis", analysis, repo_root, cache=input_cache)
    return analysis


def _artifact_payload_record(payload: bytes) -> dict[str, Any]:
    return {"artifact_sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def _artifact_file_record(path: Path) -> dict[str, Any]:
    payload = _read_regular(path, role="artifact")
    return _artifact_payload_record(payload)


def _scientific_gate_snapshot(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Derive nonvacuity gates without treating caller summaries as authority."""
    floors = analysis.get("noise_floors")
    denominators = analysis.get("denominators")
    classifications = analysis.get("classifications")
    accounting = analysis.get("accounting") or {}
    policies = analysis.get("analysis_policy") or {}
    floor_rows = floors if isinstance(floors, list) else []
    denominator_rows = denominators if isinstance(denominators, list) else []
    floor_checks: list[dict[str, Any]] = []
    authorizations = policies.get("floor_fallback_authorizations")
    authorization_by_metric = {
        row.get("metric_key"): row
        for row in authorizations
        if isinstance(row, dict)
    } if isinstance(authorizations, list) else {}
    for row in floor_rows:
        floor = row.get("floor")
        finite_nonnegative = (
            isinstance(floor, (int, float))
            and not isinstance(floor, bool)
            and math.isfinite(float(floor))
            and float(floor) >= 0.0
        )
        valid_cells = row.get("valid_cells")
        status = row.get("status")
        authorization = authorization_by_metric.get(row.get("metric_key"))
        fallback = bool(row.get("fallback_authorized", False)) and (
            isinstance(authorization, dict)
            and authorization.get("authorized") is True
            and _sha256_text(authorization.get("authority_receipt"))
        )
        status_consistent = (
            status == "available"
            and isinstance(valid_cells, int)
            and valid_cells >= 30
            and finite_nonnegative
        )
        floor_checks.append({
            "metric_key": row.get("metric_key"),
            "valid_cells": valid_cells,
            "finite_nonnegative": finite_nonnegative,
            "fallback_authorized": fallback,
            "fallback_receipt_valid": fallback,
            "passes": status_consistent or fallback,
        })
    denominator_checks: list[dict[str, Any]] = []
    for row in denominator_rows:
        deterministic = row.get("determinism_class") == "deterministic"
        resolution = row.get("absolute_resolution")
        resolution_positive = (
            isinstance(resolution, (int, float))
            and not isinstance(resolution, bool)
            and math.isfinite(float(resolution))
            and float(resolution) > 0.0
        )
        status = row.get("status")
        denominator = row.get("denominator")
        denominator_kind = row.get("denominator_kind")
        denominator_numeric = (
            isinstance(denominator, (int, float))
            and not isinstance(denominator, bool)
            and math.isfinite(float(denominator))
            and float(denominator) >= 0.0
        )
        status_consistent = (
            status == "pending"
            and denominator is None
            and denominator_kind == "none"
        ) or (
            status == "no_gate_insufficient_evidence"
            and denominator is None
            and denominator_kind == "none"
        ) or (
            status == "available"
            and denominator_kind in {"absolute", "relative"}
            and denominator_numeric
        )
        denominator_checks.append({
            "cell_id": row.get("cell_id"),
            "metric_key": row.get("metric_key"),
            "pending": status == "pending" or not status_consistent,
            "deterministic_exact": (
                status_consistent
                and (
                    not deterministic
                    or (
                        denominator_kind == "absolute"
                        and resolution_positive
                        and row.get("exact_match") is True
                    )
                )
            ),
        })
    denominator_keys = {
        (row.get("cell_id"), row.get("metric_key"))
        for row in denominator_rows
        if isinstance(row, dict)
    } if isinstance(denominator_rows, list) else set()
    bucket_rows = accounting.get("per_config_metric")
    bucket_keys = {
        (row.get("config_id"), row.get("metric_key"))
        for row in bucket_rows
        if isinstance(row, dict)
    } if isinstance(bucket_rows, list) else set()
    expected_bucket_count = 6 * EXPECTED_SELECTED_METRICS
    limitations = policies.get("limitations")
    limitations_sealed = (
        isinstance(limitations, dict)
        and all(
            isinstance(limitations.get(key), str) and limitations[key]
            for key in (
                "censoring",
                "finite_n",
                "within_session",
                "multiplicity",
                "historical_reconciliation",
            )
        )
    )
    floor_keys = [
        row.get("metric_key")
        for row in floor_rows
        if isinstance(row, dict) and isinstance(row.get("metric_key"), str)
    ]
    multiplicity = policies.get("multiplicity_policy")
    multiplicity_sealed = False
    if isinstance(multiplicity, dict):
        discoveries = multiplicity.get("discoveries")
        identity = dict(multiplicity)
        identity.pop("policy_sha256", None)
        multiplicity_sealed = (
            multiplicity.get("method") == "BH-FDR"
            and isinstance(multiplicity.get("q"), (int, float))
            and not isinstance(multiplicity.get("q"), bool)
            and 0.0 < float(multiplicity["q"]) <= 1.0
            and isinstance(multiplicity.get("family_definition"), str)
            and bool(multiplicity["family_definition"])
            and multiplicity.get("status") == "sealed"
            and isinstance(multiplicity.get("hypothesis_count"), int)
            and multiplicity["hypothesis_count"] >= 0
            and isinstance(discoveries, list)
            and len(discoveries) == multiplicity["hypothesis_count"]
            and multiplicity.get("discoveries_sha256") == canonical_sha256(discoveries)
            and multiplicity.get("policy_sha256") == canonical_sha256(identity)
        )
    historical = policies.get("historical_admission_reconciliation")
    historical_reconciliation_closed = False
    if isinstance(historical, dict):
        historical_identity = dict(historical)
        historical_identity.pop("reconciliation_sha256", None)
        projection_rows = [
            row
            for row in classifications
            if isinstance(row, Mapping)
            and row.get("config_id") == REFERENCE_CONFIG
        ] if isinstance(classifications, list) else []
        projection_identities = [
            (row.get("cell_id"), row.get("metric_key")) for row in projection_rows
        ]
        projected_admitted = sorted(
            (
                _historical_admission_key(row)
                for row in projection_rows
                if row.get("would_have_been_admitted") is True
            ),
            key=lambda row: (str(row["cell_id"]), str(row["metric_key"])),
        )
        authority = historical.get("authority")
        bindings = authority.get("bindings") if isinstance(authority, Mapping) else None
        historical_reconciliation_closed = (
            historical.get("status") == "reconciled"
            and historical.get("projection_contract")
            == HISTORICAL_ADMISSION_PROJECTION
            and historical.get("reference_config_id") == REFERENCE_CONFIG
            and historical.get("expected_projection_count")
            == EXPECTED_RUNNABLE * EXPECTED_SELECTED_METRICS
            and historical.get("observed_projection_count")
            == EXPECTED_RUNNABLE * EXPECTED_SELECTED_METRICS
            and historical.get("projection_complete") is True
            and len(projection_rows)
            == EXPECTED_RUNNABLE * EXPECTED_SELECTED_METRICS
            and len(set(projection_identities)) == len(projection_identities)
            and all(
                type(row.get("would_have_been_admitted")) is bool
                for row in projection_rows
            )
            and historical.get("projected_admitted_count")
            == len(projected_admitted)
            and historical.get("projected_admitted_set_sha256")
            == canonical_sha256(projected_admitted)
            and isinstance(authority, Mapping)
            and _sha256_text(authority.get("artifact_sha256"))
            and type(authority.get("size_bytes")) is int
            and authority["size_bytes"] > 0
            and _sha256_text(authority.get("authority_sha256"))
            and authority.get("admitted_count") == len(projected_admitted)
            and authority.get("admitted_set_sha256")
            == canonical_sha256(projected_admitted)
            and isinstance(bindings, Mapping)
            and bindings.get("metric_identity_sha256")
            == analysis.get("metric_identity_sha256")
            and bindings.get("reference_config_id") == REFERENCE_CONFIG
            and historical.get("exact_set_match") is True
            and historical.get("typed_reason") is None
            and historical.get("reconciliation_sha256")
            == canonical_sha256(historical_identity)
        )
    result = {
        "floor_checks": floor_checks,
        "denominator_checks": denominator_checks,
        "bucket_set_exact": (
            len(bucket_keys) == expected_bucket_count
            and isinstance(bucket_rows, list)
            and len(bucket_rows) == expected_bucket_count
            and len(bucket_keys) == len(bucket_rows)
        ),
        "floor_nonvacuous": (
            len(floor_rows) == EXPECTED_SELECTED_METRICS
            and len(set(floor_keys)) == len(floor_keys)
            and all(row["passes"] for row in floor_checks)
        ),
        "denominator_closed": (
            bool(denominator_checks)
            and len(set(denominator_keys)) == len(denominator_keys)
            and not any(
                row["pending"] or not row["deterministic_exact"]
                for row in denominator_checks
            )
        ),
        "limitations_sealed": limitations_sealed,
        "multiplicity_sealed": multiplicity_sealed,
        "historical_reconciliation": historical_reconciliation_closed,
        "reference_oor_zero": bool(
            policies.get("reference_oor_zero_confirmed", False)
        ),
    }
    result["done_eligible"] = all(
        result[key]
        for key in (
            "bucket_set_exact",
            "floor_nonvacuous",
            "denominator_closed",
            "limitations_sealed",
            "multiplicity_sealed",
            "historical_reconciliation",
            "reference_oor_zero",
        )
    )
    return result


def _validate_snapshot_witness(witness: Any, *, role: str = "source snapshot") -> None:
    if not isinstance(witness, list) or not witness:
        _fail("M6_SOURCE_SNAPSHOT_INCOMPLETE", role)
    paths: set[str] = set()
    for index, row in enumerate(witness):
        if not isinstance(row, Mapping):
            _fail("M6_SOURCE_SNAPSHOT_INCOMPLETE", f"{role}[{index}]")
        path = row.get("canonical_path")
        if not isinstance(path, str) or not path.startswith("/") or path in paths:
            _fail("M6_SOURCE_SNAPSHOT_DUPLICATE", f"{role}[{index}]")
        paths.add(path)
        observed_digest = row.get("observed_sha256")
        observed_size = row.get("observed_size_bytes")
        declared_digest = row.get("declared_sha256")
        declared_size = row.get("declared_size_bytes")
        if not _sha256_text(observed_digest) or type(observed_size) is not int or observed_size < 0:
            _fail("M6_SOURCE_SNAPSHOT_INCOMPLETE", f"{role}[{index}]")
        if declared_digest is not None and declared_digest != observed_digest:
            _fail("M6_SOURCE_SNAPSHOT_IDENTITY", f"{role}[{index}]: digest declaration")
        if declared_size is not None and declared_size != observed_size:
            _fail("M6_SOURCE_SNAPSHOT_IDENTITY", f"{role}[{index}]: size declaration")
        identities: list[Mapping[str, Any]] = []
        for identity_name in ("pre_identity", "post_identity"):
            identity = row.get(identity_name)
            if not isinstance(identity, Mapping):
                _fail("M6_SOURCE_SNAPSHOT_INCOMPLETE", f"{role}[{index}]/{identity_name}")
            for field in ("device", "inode"):
                value = identity.get(field)
                if not (
                    (type(value) is int and value >= 0)
                    or (
                        isinstance(value, str)
                        and value.isascii()
                        and value.isdigit()
                        and (value == "0" or not value.startswith("0"))
                    )
                ):
                    _fail(
                        "M6_SOURCE_SNAPSHOT_INCOMPLETE",
                        f"{role}[{index}]/{identity_name}/{field}",
                    )
            if any(
                type(identity.get(field)) is not int or identity[field] < 0
                for field in ("mode", "size_bytes")
            ) or type(identity.get("nlink")) is not int or identity["nlink"] != 1:
                _fail("M6_SOURCE_SNAPSHOT_INCOMPLETE", f"{role}[{index}]/{identity_name}")
            for field in ("mtime_ns", "ctime_ns"):
                value = identity.get(field)
                if value is not None and not (
                    (type(value) is int and value >= 0)
                    or (
                        isinstance(value, str)
                        and value.isascii()
                        and value.isdigit()
                        and (value == "0" or not value.startswith("0"))
                    )
                ):
                    _fail(
                        "M6_SOURCE_SNAPSHOT_INCOMPLETE",
                        f"{role}[{index}]/{identity_name}/{field}",
                    )
            if not stat.S_ISREG(int(identity["mode"])):
                _fail("M6_SOURCE_SNAPSHOT_IDENTITY", f"{role}[{index}]/{identity_name}: regular file required")
            if identity["size_bytes"] != observed_size:
                _fail("M6_SOURCE_SNAPSHOT_IDENTITY", f"{role}[{index}]/{identity_name}: size")
            identities.append(identity)
        if dict(identities[0]) != dict(identities[1]):
            _fail("M6_SOURCE_SNAPSHOT_IDENTITY", f"{role}[{index}]: generation changed")


def _source_authority_rows(
    schedule: Mapping[str, Any],
) -> list[tuple[str, str, int | None]]:
    source = schedule.get("source_identity")
    if not isinstance(source, Mapping):
        _fail("M6_SOURCE_SNAPSHOT_INCOMPLETE", "schedule source identity")
    keys = {
        "canonical_matrix",
        "matrix_schema",
        "roster_authority",
        "matrix_input_pins",
        "frozen_metric_registry",
        "frozen_sidecar_schema",
        "metric_registry",
        "sidecar_schema",
        "availability_probe",
        "controller_source",
        "stats_source",
        "immutable_helper",
        "m6_schemas",
        "active_m6_schemas",
    }
    rows: set[tuple[str, str, int | None]] = set()
    for key in sorted(keys):
        value = source.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, Mapping):
                _fail("M6_SOURCE_SNAPSHOT_INCOMPLETE", f"source:{key}")
            path = item.get("path")
            digest = item.get("sha256")
            size = item.get("size_bytes")
            if key == "frozen_metric_registry":
                expected_historical = REGISTRY_PATH.relative_to(REPO_ROOT).as_posix()
                if path != expected_historical:
                    _fail("M6_SOURCE_SNAPSHOT_INCOMPLETE", f"source:{key}")
                path = FROZEN_REGISTRY_PATH.relative_to(REPO_ROOT).as_posix()
            elif key == "frozen_sidecar_schema":
                expected_historical = SIDECAR_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix()
                if path != expected_historical:
                    _fail("M6_SOURCE_SNAPSHOT_INCOMPLETE", f"source:{key}")
                path = FROZEN_SIDECAR_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix()
            if (
                not isinstance(path, str)
                or not path
                or not _sha256_text(digest)
                or (size is not None and (type(size) is not int or size < 0))
            ):
                _fail("M6_SOURCE_SNAPSHOT_INCOMPLETE", f"source:{key}")
            _safe_relative(path, role=f"source:{key}")
            rows.add((path, digest, size))
    return sorted(rows)


def _validate_snapshot_source_bindings(
    witness: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    *,
    repo_root: Path | None,
) -> None:
    authorities = _source_authority_rows(schedule)
    for relative_path, digest, size in authorities:
        candidates = []
        for row in witness:
            canonical = row.get("canonical_path")
            if not isinstance(canonical, str):
                continue
            if repo_root is not None:
                expected = os.path.abspath(
                    os.fspath(Path(repo_root) / relative_path)
                )
                path_matches = canonical == expected
            else:
                path_matches = canonical == relative_path or canonical.endswith(
                    "/" + relative_path
                )
            if (
                path_matches
                and row.get("observed_sha256") == digest
                and (size is None or row.get("observed_size_bytes") == size)
            ):
                candidates.append(row)
        if not candidates:
            _fail(
                "M6_SOURCE_SNAPSHOT_BINDING",
                f"source authority is absent or mismatched: {relative_path}",
            )


def _validate_snapshot_historical_authority_binding(
    witness: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any] | None,
) -> None:
    if not isinstance(authority, Mapping):
        _fail(
            "M6_SOURCE_SNAPSHOT_BINDING",
            "historical admission authority descriptor is absent",
        )
    digest = authority.get("artifact_sha256")
    size = authority.get("size_bytes")
    role_rows = [
        row
        for row in witness
        if row.get("role") == "historical admission set authority"
    ]
    if len(role_rows) != 1 or (
        role_rows[0].get("observed_sha256") != digest
        or role_rows[0].get("observed_size_bytes") != size
    ):
        _fail(
            "M6_SOURCE_SNAPSHOT_BINDING",
            "historical admission authority witness is absent or ambiguous",
        )


def _require_bound_payload(
    payload: bytes | None,
    expected: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    if payload is None:
        _fail("M6_ARTIFACT_PAYLOAD_UNBOUND", role)
    value = loads_strict(payload, role=role)
    if not isinstance(value, dict) or value != dict(expected):
        _fail("M6_ARTIFACT_PAYLOAD_MISMATCH", role)
    return _artifact_payload_record(payload)


def _pinned_tree_manifest(
    root: Path,
    *,
    role: str,
    input_cache: InputSnapshotCache,
    readonly: bool = False,
) -> tuple[dict[str, Any], bytes]:
    """Return a byte/topology-complete manifest for one external tree."""
    with _PinnedTreeTransaction(
        root,
        role=role,
        require_readonly_files=readonly,
        require_readonly_directories=readonly,
        input_cache=input_cache,
    ) as transaction:
        entries: list[dict[str, Any]] = []
        for relative, (kind, identity) in sorted(transaction.inventory.items()):
            if relative == "":
                continue
            item: dict[str, Any] = {
                "relative_path": relative,
                "kind": kind,
                "device": str(identity[0]),
                "inode": str(identity[1]),
                "mode": identity[2],
                "nlink": identity[3],
                "size_bytes": identity[4],
                "mtime_ns": str(identity[5]),
                "ctime_ns": str(identity[6]),
            }
            if kind == "file":
                payload = transaction.read(relative, role=f"{role} {relative}")
                item["sha256"] = sha256_bytes(payload)
                item["size_bytes"] = len(payload)
            else:
                item["sha256"] = canonical_sha256(item)
            entries.append(item)
        transaction.verify()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "root": ".",
        "entries": entries,
    }
    manifest["tree_sha256"] = canonical_sha256(manifest)
    return manifest, canonical_bytes(manifest)
def _validate_recovery_receipt_value(
    receipt: Any,
    *,
    schema_value: Mapping[str, Any],
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        validator = Draft202012Validator(
            schema_value, format_checker=FormatChecker()
        )
        errors = sorted(
            validator.iter_errors(receipt),
            key=lambda error: list(error.absolute_path),
        )
    except Exception as exc:
        _fail("M6_RECOVERY_SCHEMA", str(exc))
    if errors:
        error = errors[0]
        location = "/" + "/".join(str(part) for part in error.absolute_path)
        _fail("M6_RECOVERY_SCHEMA", f"receipt{location}: {error.message}")
    if not isinstance(receipt, dict):
        _fail("M6_RECOVERY_SCHEMA", "receipt must be an object")
    if receipt.get("decision") not in {
        "EXACT_PREFIX_COMPLETE",
        "EXACT_PREFIX_PARTIAL",
    }:
        _fail(
            "M6_RECOVERY_DECISION",
            "only a successful exact-prefix recovery can authorize DONE",
        )
    if (
        receipt.get("schedule_sha256") != schedule.get("schedule_sha256")
        or receipt.get("journal_sha256") != journal.get("journal_sha256")
        or receipt.get("analysis_sha256") != analysis.get("analysis_sha256")
    ):
        _fail("M6_RECOVERY_IDENTITY_MISMATCH", "receipt owner seals differ")
    try:
        immutable.verify_model_seal(receipt, "receipt_sha256")
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_RECOVERY_SEAL", str(exc))
    return receipt


def _validate_recovery_receipt_payload(
    payload: bytes,
    *,
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    analysis: Mapping[str, Any],
    repo_root: Path,
    cache: InputSnapshotCache,
) -> dict[str, Any]:
    """Admit only a sealed, semantically bound successful recovery receipt."""
    schema_path = repo_root / RECOVERY_SCHEMA_PATH.relative_to(REPO_ROOT)
    schema_value, _schema_payload = load_json(
        schema_path,
        role="M6 recovery receipt schema",
        cache=cache,
        expected_sha256=RECOVERY_SCHEMA_SHA256,
    )
    if not isinstance(schema_value, dict):
        _fail("M6_RECOVERY_SCHEMA", "recovery receipt schema is not an object")
    receipt = loads_strict(payload, role="M6 recovery receipt")
    return _validate_recovery_receipt_value(
        receipt,
        schema_value=schema_value,
        schedule=schedule,
        journal=journal,
        analysis=analysis,
    )


def _done_inventory(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    schedule_path: Path,
    journal_path: Path,
    analysis_path: Path,
    schedule_record: Mapping[str, Any],
    journal_record: Mapping[str, Any],
    analysis_record: Mapping[str, Any],
    schema_records: Sequence[Mapping[str, Any]],
    schedule_payload: bytes | None = None,
    journal_payload: bytes | None = None,
    analysis_payload: bytes | None = None,
    repo_root: Path = REPO_ROOT,
    input_cache: InputSnapshotCache | None = None,
    source_snapshot_cache: InputSnapshotCache | None = None,
    recovery_path: Path | None = None,
    historical_admission_path: Path | None = None,
    m4_receipt_path: Path | None = None,
    source_snapshot_path: Path | None = None,
    evidence_root: Path | None = None,
    derived_seals: dict[str, str] | None = None,
    publication_material: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    """Build a safe-relative, dependency-closed inventory.

    Inventory paths are logical bundle-relative names, never ambient absolute
    paths.  Every external payload is nevertheless re-read through the
    invocation snapshot cache before its declared digest/size is admitted.
    Retained evidence is deliberately rooted in a dedicated tree separate
    from the journal/input namespace; otherwise the journal itself would be
    mistaken for retained evidence and closure could never be exact.
    """
    cache = input_cache or InputSnapshotCache()
    source_cache = source_snapshot_cache or InputSnapshotCache()
    if evidence_root is None:
        _fail(
            "M6_EVIDENCE_ROOT_REQUIRED",
            "DONE inventory requires a dedicated retained-evidence root",
        )
    evidence_root_absolute = Path(os.path.abspath(os.fspath(evidence_root)))
    journal_absolute = Path(os.path.abspath(os.fspath(journal_path)))
    if _contains_path(evidence_root_absolute, journal_absolute):
        _fail(
            "M6_EVIDENCE_NAMESPACE",
            "journal must not be inside the retained-evidence root",
        )
    schedule_path = Path(os.path.abspath(os.fspath(schedule_path)))
    journal_path = journal_absolute
    analysis_path = Path(os.path.abspath(os.fspath(analysis_path)))
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    def add(
        kind: str,
        logical_path: str,
        digest: Any,
        size: Any,
        *,
        actual_path: Path | None = None,
        payload: bytes | None = None,
        allow_missing_actual: bool = False,
    ) -> None:
        try:
            relative = _safe_relative(logical_path, role=f"DONE inventory {kind}")
        except M6Error as exc:
            _fail("M6_FINAL_INVENTORY_PATH", f"{kind}: {exc.detail}")
        if relative.as_posix() in seen_paths:
            _fail("M6_FINAL_INVENTORY_DUPLICATE", relative.as_posix())
        if actual_path is not None and not (
            allow_missing_actual and not os.path.lexists(actual_path)
        ):
            observed = _read_regular_cached(
                actual_path,
                role=f"DONE inventory {kind}",
                cache=cache,
                expected_sha256=digest if _sha256_text(digest) else None,
                expected_size=size if type(size) is int else None,
            )
            if payload is not None and observed != payload:
                _fail("M6_INPUT_SNAPSHOT_MIXED", f"DONE inventory {kind}")
            payload = observed
        if payload is not None:
            observed_digest = sha256_bytes(payload)
            observed_size = len(payload)
            if digest != observed_digest or (
                size is not None and size != observed_size
            ):
                _fail("M6_FINAL_INVENTORY_IDENTITY", kind)
            digest, size = observed_digest, observed_size
        if not _sha256_text(digest):
            _fail("M6_FINAL_INVENTORY_IDENTITY", f"{kind}: sha256")
        if size is not None and (type(size) is not int or size < 0):
            _fail("M6_FINAL_INVENTORY_IDENTITY", f"{kind}: size")
        seen_paths.add(relative.as_posix())
        entries.append({
            "kind": kind,
            "path": relative.as_posix(),
            "sha256": digest,
            "size_bytes": size,
        })
        if publication_material is not None and payload is not None:
            prior_payload = publication_material.get(relative.as_posix())
            if prior_payload is not None and prior_payload != payload:
                _fail("M6_FINAL_INVENTORY_DUPLICATE", relative.as_posix())
            publication_material[relative.as_posix()] = bytes(payload)

    add(
        "schedule",
        "schedule.json",
        schedule_record.get("artifact_sha256"),
        schedule_record.get("size_bytes"),
        actual_path=schedule_path,
        payload=schedule_payload,
    )
    add(
        "journal",
        "observation_journal.json",
        journal_record.get("artifact_sha256"),
        journal_record.get("size_bytes"),
        actual_path=journal_path,
        payload=journal_payload,
    )
    add(
        "analysis",
        "determinism_study.json",
        analysis_record.get("artifact_sha256"),
        analysis_record.get("size_bytes"),
        payload=analysis_payload,
    )
    historical = analysis.get("analysis_policy", {}).get(
        "historical_admission_reconciliation"
    )
    historical_authority = (
        historical.get("authority") if isinstance(historical, Mapping) else None
    )
    if historical_admission_path is None or not isinstance(
        historical_authority, Mapping
    ):
        _fail(
            "M6_FINAL_INVENTORY_INCOMPLETE",
            "historical admission authority is absent",
        )
    historical_payload = _read_regular_cached(
        historical_admission_path,
        role="historical admission set authority",
        cache=cache,
        expected_sha256=historical_authority.get("artifact_sha256"),
        expected_size=historical_authority.get("size_bytes"),
    )
    retained_historical = loads_strict(
        historical_payload, role="historical admission set authority"
    )
    if (
        not isinstance(retained_historical, Mapping)
        or retained_historical.get("authority_sha256")
        != historical_authority.get("authority_sha256")
        or retained_historical.get("admitted_keys_sha256")
        != historical_authority.get("admitted_set_sha256")
    ):
        _fail(
            "M6_HISTORICAL_AUTHORITY_BINDING",
            "retained authority differs from analysis",
        )
    add(
        "historical_admission_authority",
        "historical/historical_admission_set.json",
        sha256_bytes(historical_payload),
        len(historical_payload),
        payload=historical_payload,
    )

    m4 = schedule["m4_identity"]
    if not isinstance(m4, Mapping) or m4.get("status") != "complete":
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "complete M4 identity is required")
    if m4_receipt_path is None:
        _fail(
            "M6_FINAL_INVENTORY_INCOMPLETE",
            "raw M4 receipt path is required for dependency closure",
        )
    receipt_payload = _read_regular_cached(
        m4_receipt_path,
        role="M4 binding receipt",
        cache=cache,
    )
    receipt_value = loads_strict(receipt_payload, role="M4 binding receipt")
    if not isinstance(receipt_value, Mapping) or dict(receipt_value) != dict(m4):
        _fail("M6_M4_RECEIPT_REDERIVATION", "retained receipt bytes differ")
    add(
        "m4_receipt",
        "m4/m4_binding_receipt.json",
        sha256_bytes(receipt_payload),
        len(receipt_payload),
        payload=receipt_payload,
    )
    chunks = m4.get("chunk_done_manifests")
    if not isinstance(chunks, list) or len(chunks) != 18:
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "M4 must contain exactly 18 DONE hashes")
    archive_root = m4.get("archive_root")
    if not isinstance(archive_root, str) or not archive_root:
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "M4 archive root is absent")
    archive_root_path = _external_m4_root(Path(archive_root), repo_root=repo_root)
    archive_manifest, archive_manifest_payload = _pinned_tree_manifest(
        archive_root_path,
        role="M4 archive inventory",
        input_cache=cache,
        readonly=True,
    )
    add(
        "m4_archive_tree",
        "m4/archive_tree/manifest.json",
        sha256_bytes(archive_manifest_payload),
        len(archive_manifest_payload),
        payload=archive_manifest_payload,
    )
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"M4 chunk {index}")
        chunk_id = chunk.get("chunk_id")
        done_path = chunk.get("done_path") or (
            f"chunks/{chunk_id}/DONE.json" if isinstance(chunk_id, str) else ""
        )
        try:
            safe_done = _safe_relative(str(done_path), role="M4 DONE inventory")
        except M6Error as exc:
            _fail("M6_FINAL_INVENTORY_PATH", f"M4 DONE: {exc.detail}")
        if not safe_done.as_posix().startswith("chunks/"):
            _fail("M6_FINAL_INVENTORY_PATH", safe_done.as_posix())
        actual_done = archive_root_path / safe_done
        done_payload = _read_regular_cached(
            actual_done,
            role=f"M4 DONE {chunk_id}",
            cache=cache,
            expected_sha256=chunk.get("done_sha256"),
            expected_size=chunk.get("done_size_bytes", chunk.get("size_bytes")),
        )
        add(
            "m4_chunk_done",
            f"m4/archive/{safe_done.as_posix()}",
            sha256_bytes(done_payload),
            len(done_payload),
            payload=done_payload,
        )
    source = schedule.get("source_identity")
    if not isinstance(source, Mapping):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "source identity is absent")
    alias_relative = FROZEN_SOURCE_ALIAS_PATH.relative_to(REPO_ROOT).as_posix()
    alias_file = _canonical_repo_file(
        repo_root / alias_relative,
        FROZEN_SOURCE_ALIAS_PATH,
        repo_root=repo_root,
        role="frozen source alias table inventory",
    )
    _aliases, alias_payload = _load_frozen_source_alias(
        repo_root=repo_root,
        cache=cache,
    )
    add(
        "source:frozen_source_aliases",
        alias_relative,
        FROZEN_SOURCE_ALIAS_SHA256,
        len(alias_payload),
        actual_path=alias_file,
        payload=alias_payload,
    )
    structured_source_keys = {
        "canonical_matrix",
        "matrix_schema",
        "roster_authority",
        "matrix_input_pins",
        "frozen_metric_registry",
        "frozen_sidecar_schema",
        "metric_registry",
        "sidecar_schema",
        "availability_probe",
        "controller_source",
        "stats_source",
        "immutable_helper",
        "m6_schemas",
        "active_m6_schemas",
    }
    for key, value in source.items():
        if key not in structured_source_keys:
            continue
        values = value if isinstance(value, list) else [value]
        for index, item in enumerate(values):
            if not isinstance(item, Mapping) or "path" not in item or "sha256" not in item:
                _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"malformed source:{key}[{index}]")
            path = item.get("path")
            if not isinstance(path, str) or not path:
                _fail("M6_FINAL_INVENTORY_PATH", f"source:{key}[{index}]")
            if key in {"frozen_metric_registry", "frozen_sidecar_schema"}:
                commit = item.get("source_commit")
                if not isinstance(commit, str):
                    _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"frozen {key} commit")
                frozen_payload = _frozen_source_payload(
                    repo_root,
                    commit,
                    path,
                    role=f"{key} inventory",
                    cache=cache,
                )
                if sha256_bytes(frozen_payload) != item.get("sha256"):
                    _fail("M6_FINAL_INVENTORY_IDENTITY", key)
                add(
                    f"source:{key}",
                    f"source/{key}.json",
                    sha256_bytes(frozen_payload),
                    len(frozen_payload),
                    payload=frozen_payload,
                )
                continue
            if key == "m6_schemas":
                logical = f"source/legacy_schemas/{PurePosixPath(path).name}"
            elif key == "active_m6_schemas":
                logical = f"source/active_schemas/{PurePosixPath(path).name}"
            else:
                logical = path
            actual = repo_root / _safe_relative(path, role=f"source:{key}")
            add(
                f"source:{key}",
                logical,
                item.get("sha256"),
                item.get("size_bytes"),
                actual_path=actual,
            )

    evidence_seen: dict[str, tuple[str, int, str]] = {}
    evidence_refs: list[tuple[str, Mapping[str, Any], Path, str]] = []
    for event in journal.get("events", []):
        if not isinstance(event, Mapping):
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", "malformed journal event")
        for ref in event.get("sidecars", []):
            if not isinstance(ref, Mapping):
                _fail("M6_FINAL_INVENTORY_INCOMPLETE", "malformed sidecar reference")
            refs: list[tuple[str, Mapping[str, Any]]] = [("sidecar", ref)]
            launch = ref.get("launch_evidence")
            if isinstance(launch, Mapping):
                refs.extend(
                    (str(role), evidence)
                    for role, evidence in launch.items()
                    if isinstance(evidence, Mapping)
                )
            elif launch is not None:
                _fail("M6_FINAL_INVENTORY_INCOMPLETE", "malformed launch evidence")
            for role, evidence in refs:
                relative_raw = evidence.get("relative_path")
                digest = evidence.get("sha256")
                size = evidence.get("size_bytes")
                if not isinstance(relative_raw, str) or not _sha256_text(digest) or type(size) is not int:
                    _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"retained {role}")
                relative = _safe_relative(relative_raw, role=f"retained {role}")
                actual = evidence_root_absolute / relative
                logical = f"retained_evidence/{relative.as_posix()}"
                identity = (digest, size, os.path.abspath(os.fspath(actual)))
                prior = evidence_seen.get(logical)
                if prior is not None:
                    if prior != identity:
                        _fail("M6_FINAL_INVENTORY_DUPLICATE", logical)
                    continue
                evidence_seen[logical] = identity
                evidence_refs.append((role, evidence, actual, relative.as_posix()))
    if not evidence_refs:
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "retained evidence tree is empty")
    common_root = evidence_root_absolute
    for _role, _evidence, actual, _relative in evidence_refs:
        try:
            actual.relative_to(common_root)
        except ValueError as exc:
            _fail("M6_EVIDENCE_OUTSIDE_ROOT", str(exc))
    evidence_manifest, evidence_manifest_payload = _pinned_tree_manifest(
        common_root,
        role="retained evidence inventory",
        input_cache=cache,
    )
    referenced_files = {
        path.relative_to(common_root).as_posix()
        for _, _, path, _ in evidence_refs
    }
    manifest_files = {
        entry["relative_path"]
        for entry in evidence_manifest["entries"]
        if entry.get("kind") == "file"
    }
    if manifest_files != referenced_files:
        _fail(
            "M6_FINAL_INVENTORY_EVIDENCE_CLOSURE",
            f"missing={sorted(referenced_files - manifest_files)} "
            f"extra={sorted(manifest_files - referenced_files)}",
        )
    for role, evidence, actual, _relative in evidence_refs:
        add(
            f"retained:{role}",
            f"retained_evidence/{_relative}",
            evidence.get("sha256"),
            evidence.get("size_bytes"),
            actual_path=actual,
        )
    try:
        with _PinnedTreeTransaction(
            common_root,
            role="retained evidence final closure",
            input_cache=cache,
        ) as evidence_transaction:
            evidence_transaction.verify()
    except M6Error:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_TREE_CHANGED", f"retained evidence final closure: {exc}")
    add(
        "retained_evidence_tree",
        "retained_evidence/manifest.json",
        sha256_bytes(evidence_manifest_payload),
        len(evidence_manifest_payload),
        payload=evidence_manifest_payload,
    )
    if derived_seals is not None:
        derived_seals["retained_evidence_tree_sha256"] = str(
            evidence_manifest["tree_sha256"]
        )

    for schema in schema_records:
        if not isinstance(schema, Mapping):
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", "invalid schema identity")
        schema_path = schema.get("path")
        if not isinstance(schema_path, str):
            _fail("M6_FINAL_INVENTORY_PATH", "schema path")
        safe_schema = _safe_relative(schema_path, role="schema inventory")
        add(
            f"schema:{schema.get('schema_id')}",
            safe_schema.as_posix(),
            schema.get("sha256"),
            None,
            actual_path=repo_root / safe_schema,
        )

    if recovery_path is not None:
        recovery_payload = _read_regular_cached(
            recovery_path, role="recovery receipt", cache=cache,
        )
        add(
            "recovery_receipt",
            "recovery/m6_recovery_receipt.json",
            sha256_bytes(recovery_payload),
            len(recovery_payload),
            payload=recovery_payload,
        )
        recovery_schema_path = repo_root / RECOVERY_SCHEMA_PATH.relative_to(REPO_ROOT)
        recovery_schema_payload = _read_regular_cached(
            recovery_schema_path,
            role="M6 recovery receipt schema",
            cache=cache,
            expected_sha256=RECOVERY_SCHEMA_SHA256,
        )
        add(
            "schema:recovery_receipt",
            RECOVERY_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix(),
            RECOVERY_SCHEMA_SHA256,
            len(recovery_schema_payload),
            payload=recovery_schema_payload,
        )
    snapshot_payload = cache.witness_payload()
    if source_snapshot_path is not None and os.path.lexists(source_snapshot_path):
        _verify_existing_source_snapshot(
            source_snapshot_path,
            snapshot_payload,
            cache=source_cache,
        )
    add(
        "source_snapshot_identity",
        "source_snapshot/identity.json",
        sha256_bytes(snapshot_payload),
        len(snapshot_payload),
        payload=snapshot_payload,
    )
    entries.sort(key=lambda row: (row["kind"], row["path"]))
    return {
        "schema_version": 1,
        "entries": entries,
        "inventory_sha256": canonical_sha256(entries),
    }


def build_done(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    schedule_path: Path,
    journal_path: Path,
    analysis_path: Path,
    schedule_payload: bytes | None = None,
    journal_payload: bytes | None = None,
    analysis_payload: bytes | None = None,
    generated_at_utc: str | None = None,
    repo_root: Path = REPO_ROOT,
    _input_cache: InputSnapshotCache | None = None,
    _source_snapshot_cache: InputSnapshotCache | None = None,
    m4_receipt_path: Path | None = None,
    source_snapshot_path: Path | None = None,
    recovery_path: Path | None = None,
    historical_admission_path: Path | None = None,
    evidence_root: Path | None = None,
    _publication_material: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    if analysis["completion_state"] != "complete":
        _fail("M6_DONE_INCOMPLETE", analysis["completion_state"])
    if journal["trigger_state"] != "decided":
        _fail("M6_DONE_TRIGGER_STATE", journal["trigger_state"])
    if schedule["m4_identity"]["status"] != "complete":
        _fail("M6_DONE_M4_INCOMPLETE", "complete M4 identity required")
    if analysis["accounting"]["denominator_scope"] != "complete_m4_launch_set":
        _fail("M6_DONE_DENOMINATOR_SCOPE", analysis["accounting"]["denominator_scope"])
    accounting = analysis["accounting"]
    selected_metric_keys = {
        metric["key"]
        for metric in schedule["metrics"]
        if metric["capture_status"] == "selected"
    }
    expected_bucket_keys = {
        (config["config_id"], metric_key)
        for config in schedule["configs"]
        for metric_key in selected_metric_keys
    }
    buckets = accounting.get("per_config_metric")
    actual_bucket_keys = {
        (bucket.get("config_id"), bucket.get("metric_key"))
        for bucket in buckets
        if (
            isinstance(bucket, dict)
            and isinstance(bucket.get("config_id"), str)
            and isinstance(bucket.get("metric_key"), str)
        )
    } if isinstance(buckets, list) else set()
    nonvacuous = (
        accounting.get("active_slots", 0) > 0
        and accounting.get("observed_slots") == accounting.get("active_slots")
        and accounting.get("terminal_slots") == accounting.get("active_slots")
        and accounting.get("pending_slots") == 0
        and accounting.get("excluded_typed_slots") == 0
        and accounting.get("raw_rows_collected", 0) > 0
        and accounting.get("raw_rows_collected") == accounting.get("expected_rows")
        and accounting.get("not_collected_typed_rows") == 0
        and accounting.get("pending_expected_rows") == 0
        and accounting.get("classified_raw_rows", 0)
        + accounting.get("excluded_typed_raw_rows", 0)
        == accounting.get("raw_rows_collected")
        and isinstance(buckets, list)
        and len(buckets) == len(expected_bucket_keys)
        and actual_bucket_keys == expected_bucket_keys
        and all(
            bucket.get("expected_rows", 0) > 0
            and bucket.get("collected_rows") == bucket.get("expected_rows")
            and bucket.get("not_collected_typed_rows") == 0
            and bucket.get("pending_rows") == 0
            and bucket.get("classified_rows", 0)
            + bucket.get("excluded_typed_rows", 0)
            == bucket.get("collected_rows")
            for bucket in buckets
            if isinstance(bucket, dict)
        )
    )
    if not nonvacuous:
        _fail(
            "M6_DONE_NONVACUOUS_EVIDENCE",
            "every active slot and selected metric requires a retained observation",
        )
    floor_rows = analysis.get("noise_floors")
    floor_keys = {
        row.get("metric_key")
        for row in floor_rows
        if isinstance(row, Mapping) and isinstance(row.get("metric_key"), str)
    } if isinstance(floor_rows, list) else set()
    if (
        not isinstance(floor_rows, list)
        or len(floor_rows) != len(selected_metric_keys)
        or floor_keys != selected_metric_keys
    ):
        _fail("M6_DONE_FLOOR_COVERAGE", "exactly one floor row is required per selected metric")
    denominator_rows = analysis.get("denominators")
    expected_denominator_keys = {
        (cell["cell_id"], metric_key)
        for cell in schedule["cells"]
        for metric_key in selected_metric_keys
    }
    denominator_keys = {
        (row.get("cell_id"), row.get("metric_key"))
        for row in denominator_rows
        if (
            isinstance(row, Mapping)
            and isinstance(row.get("cell_id"), str)
            and isinstance(row.get("metric_key"), str)
        )
    } if isinstance(denominator_rows, list) else set()
    if (
        not isinstance(denominator_rows, list)
        or len(denominator_rows) != len(expected_denominator_keys)
        or denominator_keys != expected_denominator_keys
    ):
        _fail(
            "M6_DONE_DENOMINATOR_COVERAGE",
            "exactly one denominator row is required per cell and selected metric",
        )
    expected_analysis_bindings = {
        "schedule_sha256": schedule.get("schedule_sha256"),
        "journal_sha256": journal.get("journal_sha256"),
        "source_identity_sha256": schedule.get("source_identity_sha256"),
        "config_identity_sha256": schedule.get("config_identity_sha256"),
        "metric_identity_sha256": schedule.get("metric_identity_sha256"),
        "m4_identity_sha256": schedule.get("m4_identity", {}).get("identity_sha256"),
    }
    if any(
        analysis.get(key) != expected for key, expected in expected_analysis_bindings.items()
    ):
        _fail("M6_ANALYSIS_BINDING", "analysis is bound to a different schedule/journal")
    policy = analysis.get("analysis_policy")
    if not isinstance(policy, Mapping):
        _fail("M6_ANALYSIS_BINDING", "analysis policy is absent")
    bootstrap_seed = policy.get("bootstrap_seed")
    bootstrap_replicates = policy.get("bootstrap_replicates")
    study_identity = [
        schedule["schedule_sha256"],
        journal["journal_sha256"],
        bootstrap_seed,
        bootstrap_replicates,
    ]
    expected_study_id = f"m6-study-{canonical_sha256(study_identity)[:16]}"
    if analysis.get("study_id") != expected_study_id:
        _fail("M6_ANALYSIS_ID", "analysis study identity differs")
    recomputed_scientific = _scientific_gate_snapshot(analysis)
    if analysis.get("scientific_gates") != recomputed_scientific:
        _fail("M6_DONE_SCIENTIFIC_REDERIVATION", "scientific gates were not derived")
    if analysis.get("publication_state") != "M6_DONE" or analysis.get("m7_state") != "M7_BLOCKED":
        _fail("M6_DONE_STATE", "analysis state is not an active M6 DONE state")
    _require_active_schema("observation_journal", journal)
    _require_active_schema("analysis", analysis)
    if (
        journal.get("schema_version") != schedule.get("schema_version")
        or analysis.get("schema_version") != schedule.get("schema_version")
    ):
        _fail("M6_SCHEMA_MIXED", "DONE inputs use mixed top-level schema versions")
    if not journal.get("causal_prefix_closed", False):
        _fail("M6_JOURNAL_PREFIX", "DONE requires an exact eligible journal prefix")
    scientific = analysis.get("scientific_gates")
    if not isinstance(scientific, dict) or not scientific.get("done_eligible"):
        _fail("M6_DONE_SCIENTIFIC_VACUITY", "scientific nonvacuity gates are not closed")
    if historical_admission_path is None:
        _fail(
            "M6_HISTORICAL_AUTHORITY_UNBOUND",
            "DONE requires the separately supplied historical admission set",
        )
    if any(payload is None for payload in (schedule_payload, journal_payload, analysis_payload)):
        _fail("M6_ARTIFACT_PAYLOAD_UNBOUND", "DONE requires bound schedule/journal/analysis payloads")
    if not analysis["analysis_policy"]["reference_oor_zero_confirmed"]:
        _fail("M6_DONE_REFERENCE_OOR", "host-locked reference has OOR observations")
    input_cache = _input_cache or InputSnapshotCache()
    source_snapshot_cache = _source_snapshot_cache or InputSnapshotCache()
    if recovery_path is not None:
        recovery_payload = _read_regular_cached(
            recovery_path,
            role="recovery receipt",
            cache=input_cache,
        )
        _validate_recovery_receipt_payload(
            recovery_payload,
            schedule=schedule,
            journal=journal,
            analysis=analysis,
            repo_root=repo_root,
            cache=input_cache,
        )
    validate_schedule(
        schedule,
        repo_root=repo_root,
        rehash_sources=True,
        _input_cache=input_cache,
    )
    validate_journal(
        schedule,
        journal,
        journal_path=journal_path,
        repo_root=repo_root,
        rehash_sidecars=True,
        evidence_root=evidence_root,
        _input_cache=input_cache,
    )
    _validate_schema("analysis", analysis, repo_root, cache=input_cache)
    _validate_seal(analysis, "analysis_sha256", role="analysis")
    regenerated_analysis = analyze(
        schedule,
        journal,
        journal_path=journal_path,
        generated_at_utc=analysis.get("generated_at_utc"),
        bootstrap_replicates=analysis.get("analysis_policy", {}).get(
            "bootstrap_replicates"
        ),
        bootstrap_seed=analysis.get("analysis_policy", {}).get("bootstrap_seed"),
        historical_admission_path=historical_admission_path,
        evidence_root=evidence_root,
        repo_root=repo_root,
        _input_cache=input_cache,
    )
    if regenerated_analysis != dict(analysis):
        _fail("M6_ANALYSIS_REDERIVATION", "analysis differs from independent replay")
    _require_bound_payload(schedule_payload, schedule, role="schedule payload")
    _require_bound_payload(journal_payload, journal, role="journal payload")
    _require_bound_payload(analysis_payload, analysis, role="analysis payload")
    matrix_payload = _resolve_repo_file(
        schedule["source_identity"]["canonical_matrix"],
        repo_root,
        role="canonical matrix",
        cache=input_cache,
    )
    _validate_m4_identity(
        schedule["m4_identity"],
        matrix=loads_strict(matrix_payload, role="canonical matrix"),
        matrix_sha256=sha256_bytes(matrix_payload),
        cells=schedule["cells"],
        repo_root=repo_root,
        require_complete=True,
        input_cache=input_cache,
    )
    schemas = []
    for kind, path in SCHEMA_PATHS_V2.items():
        actual = repo_root / path.relative_to(REPO_ROOT)
        payload = _read_regular_cached(
            actual,
            role=f"{kind} schema",
            cache=input_cache,
        )
        schemas.append({
            "schema_id": kind,
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_bytes(payload),
        })
    schemas.sort(key=lambda row: row["schema_id"])
    schedule_record = {
        "relative_path": "schedule.json",
        **_require_bound_payload(
            schedule_payload,
            schedule,
            role="schedule payload",
        ),
    }
    journal_record = {
        "relative_path": "observation_journal.json",
        **_require_bound_payload(
            journal_payload,
            journal,
            role="journal payload",
        ),
    }
    analysis_record = {
        "relative_path": "determinism_study.json",
        **_require_bound_payload(
            analysis_payload,
            analysis,
            role="analysis payload",
        ),
    }
    derived_seals: dict[str, str] = {}
    inventory = _done_inventory(
        schedule,
        journal,
        analysis,
        schedule_path=schedule_path,
        journal_path=journal_path,
        analysis_path=analysis_path,
        schedule_record=schedule_record,
        journal_record=journal_record,
        analysis_record=analysis_record,
        schema_records=schemas,
        schedule_payload=schedule_payload,
        journal_payload=journal_payload,
        analysis_payload=analysis_payload,
        repo_root=repo_root,
        input_cache=input_cache,
        source_snapshot_cache=source_snapshot_cache,
        m4_receipt_path=m4_receipt_path,
        source_snapshot_path=source_snapshot_path,
        recovery_path=recovery_path,
        historical_admission_path=historical_admission_path,
        evidence_root=evidence_root,
        derived_seals=derived_seals,
        publication_material=_publication_material,
    )
    snapshot_payload = input_cache.witness_payload()
    witness = input_cache.witness()
    _validate_snapshot_witness(witness)
    _validate_snapshot_source_bindings(
        witness,
        schedule,
        repo_root=repo_root,
    )
    historical_reconciliation = analysis.get("analysis_policy", {}).get(
        "historical_admission_reconciliation"
    )
    _validate_snapshot_historical_authority_binding(
        witness,
        historical_reconciliation.get("authority")
        if isinstance(historical_reconciliation, Mapping)
        else None,
    )
    if source_snapshot_path is not None and os.path.lexists(source_snapshot_path):
        _verify_existing_source_snapshot(
            source_snapshot_path,
            snapshot_payload,
            cache=source_snapshot_cache,
        )
    inventory_entries = inventory.get("entries")
    if not isinstance(inventory_entries, list):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "inventory entries are absent")
    inventory_by_kind = {
        entry.get("kind"): entry
        for entry in inventory_entries
        if isinstance(entry, Mapping)
    }
    m4_receipt_entry = inventory_by_kind.get("m4_receipt")
    m4_archive_entry = inventory_by_kind.get("m4_archive_tree")
    evidence_tree_entry = inventory_by_kind.get("retained_evidence_tree")
    chunk_entries = sorted(
        (entry for entry in inventory_entries
         if isinstance(entry, Mapping) and entry.get("kind") == "m4_chunk_done"),
        key=lambda entry: str(entry.get("path")),
    )
    if (
        not isinstance(m4_receipt_entry, Mapping)
        or not isinstance(m4_archive_entry, Mapping)
        or not isinstance(evidence_tree_entry, Mapping)
        or len(chunk_entries) != 18
    ):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "required M4/evidence inventory entries are absent")
    m4_descriptor = {
        "relative_path": "m4/m4_binding_receipt.json",
        "artifact_sha256": m4_receipt_entry["sha256"],
        "size_bytes": m4_receipt_entry["size_bytes"],
        "archive_root": schedule["m4_identity"]["archive_root"],
        "chunk_done_sha256s": [entry["sha256"] for entry in chunk_entries],
    }
    evidence_tree_sha256 = derived_seals.get("retained_evidence_tree_sha256")
    if not _sha256_text(evidence_tree_sha256):
        _fail("M6_FINAL_INVENTORY_EVIDENCE_SEAL", "retained evidence tree seal is absent")
    evidence_descriptor = {
        "relative_path": "retained_evidence/manifest.json",
        "artifact_sha256": evidence_tree_entry["sha256"],
        "size_bytes": evidence_tree_entry["size_bytes"],
        "tree_sha256": evidence_tree_sha256,
    }
    _validate_snapshot_witness(witness)
    done = {
        "schema_version": ACTIVE_SCHEMA_VERSION,
        "artifact_kind": "m6_determinism_done",
        "status": "done",
        "m6_state": "M6_DONE",
        "m7_state": "M7_BLOCKED",
        "m6_m7_handoff_candidate": False,
        "simulator_scoreability": "UNAVAILABLE_REQUIRED_CYCLES",
        "publication_verdict": "NOT_EVALUATED",
        "generated_at_utc": generated_at_utc or _now(),
        "schedule": schedule_record,
        "journal": journal_record,
        "analysis": {"relative_path": "determinism_study.json", **analysis_record},
        "source_snapshot": {
            "relative_path": "source_snapshot/identity.json",
            "artifact_sha256": sha256_bytes(snapshot_payload),
            "size_bytes": len(snapshot_payload),
            "witness": witness,
        },
        "m4": m4_descriptor,
        "retained_evidence": evidence_descriptor,
        "schemas": schemas,
        "inventory": inventory,
        "source_identity_sha256": schedule["source_identity_sha256"],
        "config_identity_sha256": schedule["config_identity_sha256"],
        "metric_identity_sha256": schedule["metric_identity_sha256"],
        "m4_identity_sha256": schedule["m4_identity"]["identity_sha256"],
        "m4_status": "complete",
        "counts": {
            "runnable_cells": EXPECTED_RUNNABLE,
            "configs": 6,
            "tier1_slots": TIER1_SLOTS,
            "triggered_cells": analysis["accounting"]["triggered_cells"],
            "tier2_slots": analysis["accounting"]["active_tier2_slots"],
            "total_slots": analysis["accounting"]["active_slots"],
            "raw_rows_collected": analysis["accounting"]["raw_rows_collected"],
            "classified_raw_rows": analysis["accounting"]["classified_raw_rows"],
            "excluded_typed_raw_rows": analysis["accounting"]["excluded_typed_raw_rows"],
            "not_collected_typed_rows": analysis["accounting"]["not_collected_typed_rows"],
        },
        "accounting_identity_holds": True,
        "denominator_identity_holds": True,
        "taxonomy": list(DETERMINISM_CLASSES),
        "collect_implementation": "disabled_offline_source",
    }
    done["done_sha256"] = canonical_sha256(done)
    _validate_schema("done", done, repo_root, cache=input_cache)
    return done


def _bundle_expected_topology(paths: Iterable[str]) -> dict[str, str]:
    """Derive the complete file/directory topology for a published bundle."""
    expected: dict[str, str] = {}
    for raw_path in paths:
        relative = _safe_relative(str(raw_path), role="bundle topology")
        key = relative.as_posix()
        prior = expected.get(key)
        if prior is not None and prior != "file":
            _fail("M6_FINAL_INVENTORY_PATH", f"file/directory collision: {key}")
        expected[key] = "file"
        parts = relative.parts
        for depth in range(1, len(parts)):
            parent = "/".join(parts[:depth])
            prior = expected.get(parent)
            if prior == "file":
                _fail("M6_FINAL_INVENTORY_PATH", f"file/directory collision: {parent}")
            expected[parent] = "directory"
    return expected


def _verify_done_inventory_bundle(
    done: Mapping[str, Any],
    *,
    bundle_root: Path,
) -> None:
    """Verify every byte named by a relocated DONE inventory.

    This relocation-only path deliberately does not reinterpret an external M4
    archive root.  The inventory's sealed receipt, archive manifest, chunk DONE
    bytes, and retained-evidence manifest are the portable trust surface; full
    semantic M4 admission still requires the operator-supplied archive path.
    """
    inventory = done.get("inventory")
    if not isinstance(inventory, Mapping):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "inventory object is absent")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or inventory.get(
        "inventory_sha256"
    ) != canonical_sha256(entries):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "inventory seal differs")
    expected_topology = _bundle_expected_topology(
        [
            *(
                entry.get("path")
                for entry in entries
                if isinstance(entry, Mapping)
            ),
            "M6_DONE.json",
        ]
    )
    try:
        with _PinnedTreeTransaction(
            bundle_root,
            role="relocated M6 DONE bundle",
            require_readonly_files=True,
            require_readonly_directories=True,
        ) as transaction:
            observed_topology = {
                path: kind
                for path, (kind, _identity) in transaction.inventory.items()
                if path
            }
            if observed_topology != expected_topology:
                _fail(
                    "M6_ANALYSIS_INVENTORY",
                    f"observed={sorted(observed_topology.items())} "
                    f"expected={sorted(expected_topology.items())}",
                )
            transaction.verify()
    except M6Error:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_ANALYSIS_IMMUTABILITY", f"{bundle_root}: {exc}")
    by_path: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"entry {index}")
        try:
            relative = _safe_relative(str(entry.get("path")), role="bundle inventory")
        except M6Error:
            raise
        key = relative.as_posix()
        if key in by_path:
            _fail("M6_FINAL_INVENTORY_DUPLICATE", key)
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if not _sha256_text(digest) or type(size) is not int or size < 0:
            _fail("M6_FINAL_INVENTORY_IDENTITY", key)
        payload = _read_regular(bundle_root / relative, role=f"bundle inventory {key}")
        if len(payload) != size or sha256_bytes(payload) != digest:
            _fail("M6_FINAL_INVENTORY_IDENTITY", key)
        by_path[key] = entry

    def require_entry(kind: str, path: str) -> Mapping[str, Any]:
        entry = next(
            (candidate for candidate in entries
             if isinstance(candidate, Mapping)
             and candidate.get("kind") == kind
             and candidate.get("path") == path),
            None,
        )
        if entry is None:
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"{kind}:{path}")
        return entry

    def require_descriptor(
        descriptor: Any,
        *,
        kind: str,
        path: str,
        role: str,
    ) -> Mapping[str, Any]:
        if not isinstance(descriptor, Mapping):
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"{role}: descriptor")
        entry = require_entry(kind, path)
        if (
            descriptor.get("relative_path") != path
            or descriptor.get("artifact_sha256") != entry.get("sha256")
            or descriptor.get("size_bytes") != entry.get("size_bytes")
        ):
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"{role}: descriptor binding")
        payload = _read_regular(
            bundle_root / path,
            role=f"{role} payload",
        )
        if (
            len(payload) != entry.get("size_bytes")
            or sha256_bytes(payload) != entry.get("sha256")
        ):
            _fail("M6_FINAL_INVENTORY_IDENTITY", role)
        return entry

    require_descriptor(
        done.get("schedule"),
        kind="schedule",
        path="schedule.json",
        role="schedule",
    )
    require_descriptor(
        done.get("journal"),
        kind="journal",
        path="observation_journal.json",
        role="journal",
    )
    require_descriptor(
        done.get("analysis"),
        kind="analysis",
        path="determinism_study.json",
        role="analysis",
    )
    historical_entries = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("kind") == "historical_admission_authority"
    ]
    if len(historical_entries) != 1 or historical_entries[0].get("path") != (
        "historical/historical_admission_set.json"
    ):
        _fail(
            "M6_FINAL_INVENTORY_INCOMPLETE",
            "historical admission authority singleton",
        )
    historical_entry = require_entry(
        "historical_admission_authority",
        "historical/historical_admission_set.json",
    )

    bundled_schedule_payload = _read_regular(
        bundle_root / "schedule.json", role="relocated schedule"
    )
    bundled_schedule = loads_strict(bundled_schedule_payload, role="relocated schedule")
    if not isinstance(bundled_schedule, Mapping):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "relocated schedule is not an object")
    schedule_bindings = {
        "source_identity_sha256": bundled_schedule.get("source_identity_sha256"),
        "config_identity_sha256": bundled_schedule.get("config_identity_sha256"),
        "metric_identity_sha256": bundled_schedule.get("metric_identity_sha256"),
        "m4_identity_sha256": bundled_schedule.get("m4_identity", {}).get(
            "identity_sha256"
        ),
    }
    if any(
        schedule_bindings[field] != done.get(field)
        for field in schedule_bindings
    ):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "DONE/schedule identity bindings differ")
    bundled_journal_payload = _read_regular(
        bundle_root / "observation_journal.json", role="relocated journal"
    )
    bundled_journal = loads_strict(
        bundled_journal_payload, role="relocated journal"
    )
    bundled_analysis_payload = _read_regular(
        bundle_root / "determinism_study.json", role="relocated analysis"
    )
    bundled_analysis = loads_strict(
        bundled_analysis_payload, role="relocated analysis"
    )
    if not isinstance(bundled_journal, Mapping) or not isinstance(
        bundled_analysis, Mapping
    ):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "relocated journal/analysis shape")
    historical_reconciliation = bundled_analysis.get("analysis_policy", {}).get(
        "historical_admission_reconciliation"
    )
    historical_authority = (
        historical_reconciliation.get("authority")
        if isinstance(historical_reconciliation, Mapping)
        else None
    )
    if not isinstance(historical_authority, Mapping) or (
        historical_authority.get("artifact_sha256")
        != historical_entry.get("sha256")
        or historical_authority.get("size_bytes")
        != historical_entry.get("size_bytes")
    ):
        _fail(
            "M6_FINAL_INVENTORY_INCOMPLETE",
            "historical admission authority descriptor binding",
        )

    schema_descriptors = done.get("schemas")
    if (
        not isinstance(schema_descriptors, list)
        or len(schema_descriptors) != len(SCHEMA_PATHS_V2)
    ):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "schema descriptor closure")
    schema_ids: set[str] = set()
    for descriptor in schema_descriptors:
        if not isinstance(descriptor, Mapping):
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", "malformed schema descriptor")
        schema_id = descriptor.get("schema_id")
        schema_path = descriptor.get("path")
        if not isinstance(schema_id, str) or schema_id in schema_ids:
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", "duplicate schema descriptor")
        schema_ids.add(schema_id)
        schema_entry = require_entry(f"schema:{schema_id}", str(schema_path))
        expected_schema_sha = SCHEMA_SHA256_BY_VERSION[ACTIVE_SCHEMA_VERSION].get(
            schema_id
        )
        if expected_schema_sha is None:
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"unknown schema:{schema_id}")
        if descriptor.get("sha256") != expected_schema_sha:
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"schema authority:{schema_id}")
        if (
            descriptor.get("sha256") != schema_entry.get("sha256")
            or descriptor.get("sha256") != sha256_bytes(
                _read_regular(bundle_root / str(schema_path), role=f"schema {schema_id}")
            )
        ):
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"schema:{schema_id}")
    if schema_ids != set(SCHEMA_PATHS_V2):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "schema descriptor set")

    recovery_entry = next(
        (
            entry
            for entry in entries
            if isinstance(entry, Mapping)
            and entry.get("kind") == "recovery_receipt"
        ),
        None,
    )
    if recovery_entry is not None:
        if recovery_entry.get("path") != "recovery/m6_recovery_receipt.json":
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", "recovery receipt path")
        recovery_schema_path = RECOVERY_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix()
        recovery_schema_entry = require_entry(
            "schema:recovery_receipt", recovery_schema_path
        )
        recovery_schema_payload = _read_regular(
            bundle_root / recovery_schema_path,
            role="relocated recovery receipt schema",
        )
        if (
            recovery_schema_entry.get("sha256") != RECOVERY_SCHEMA_SHA256
            or sha256_bytes(recovery_schema_payload) != RECOVERY_SCHEMA_SHA256
        ):
            _fail("M6_RECOVERY_SCHEMA", "recovery schema authority differs")
        recovery_schema_value = loads_strict(
            recovery_schema_payload, role="relocated recovery receipt schema"
        )
        if not isinstance(recovery_schema_value, Mapping):
            _fail("M6_RECOVERY_SCHEMA", "recovery schema must be an object")
        recovery_payload = _read_regular(
            bundle_root / "recovery/m6_recovery_receipt.json",
            role="relocated recovery receipt",
        )
        recovery_receipt = loads_strict(
            recovery_payload, role="relocated recovery receipt"
        )
        _validate_recovery_receipt_value(
            recovery_receipt,
            schema_value=recovery_schema_value,
            schedule=bundled_schedule,
            journal=bundled_journal,
            analysis=bundled_analysis,
        )

    structured_source_keys = {
        "canonical_matrix",
        "matrix_schema",
        "roster_authority",
        "matrix_input_pins",
        "frozen_metric_registry",
        "frozen_sidecar_schema",
        "metric_registry",
        "sidecar_schema",
        "availability_probe",
        "controller_source",
        "stats_source",
        "immutable_helper",
        "m6_schemas",
        "active_m6_schemas",
    }
    source_identity = bundled_schedule.get("source_identity")
    if not isinstance(source_identity, Mapping):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "schedule source identity")
    for key in sorted(structured_source_keys):
        value = source_identity.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, Mapping):
                _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"source:{key}")
            source_path = item.get("path")
            source_digest = item.get("sha256")
            source_size = item.get("size_bytes")
            if not isinstance(source_path, str) or not _sha256_text(source_digest):
                _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"source:{key}")
            if key in {"frozen_metric_registry", "frozen_sidecar_schema"}:
                logical_path = f"source/{key}.json"
            elif key == "m6_schemas":
                logical_path = f"source/legacy_schemas/{PurePosixPath(source_path).name}"
            elif key == "active_m6_schemas":
                logical_path = f"source/active_schemas/{PurePosixPath(source_path).name}"
            else:
                logical_path = source_path
            source_entry = require_entry(f"source:{key}", logical_path)
            if (
                source_entry.get("sha256") != source_digest
                or source_entry.get("size_bytes") != source_size
            ):
                _fail("M6_FINAL_INVENTORY_INCOMPLETE", f"source:{key}:{source_path}")

    snapshot_descriptor = done.get("source_snapshot")
    snapshot_entry = require_entry(
        "source_snapshot_identity", "source_snapshot/identity.json"
    )
    snapshot_payload = _read_regular(
        bundle_root / "source_snapshot/identity.json",
        role="source snapshot",
    )
    snapshot_value = loads_strict(snapshot_payload, role="source snapshot")
    if (
        not isinstance(snapshot_descriptor, Mapping)
        or snapshot_descriptor.get("relative_path") != snapshot_entry.get("path")
        or snapshot_descriptor.get("artifact_sha256") != snapshot_entry.get("sha256")
        or snapshot_descriptor.get("size_bytes") != snapshot_entry.get("size_bytes")
        or len(snapshot_payload) != snapshot_entry.get("size_bytes")
        or sha256_bytes(snapshot_payload) != snapshot_entry.get("sha256")
        or not isinstance(snapshot_value, list)
        or snapshot_descriptor.get("witness") != snapshot_value
    ):
        _fail("M6_SOURCE_SNAPSHOT_MISMATCH", "relocated witness differs from DONE")
    _validate_snapshot_witness(snapshot_value)
    _validate_snapshot_source_bindings(
        snapshot_value,
        bundled_schedule,
        repo_root=None,
    )
    _validate_snapshot_historical_authority_binding(
        snapshot_value,
        historical_authority,
    )

    m4 = done.get("m4")
    receipt = require_entry("m4_receipt", "m4/m4_binding_receipt.json")
    archive = require_entry("m4_archive_tree", "m4/archive_tree/manifest.json")
    chunks = sorted(
        (
            entry
            for entry in entries
            if isinstance(entry, Mapping) and entry.get("kind") == "m4_chunk_done"
        ),
        key=lambda entry: str(entry.get("path")),
    )
    if (
        not isinstance(m4, Mapping)
        or m4.get("relative_path") != receipt.get("path")
        or m4.get("artifact_sha256") != receipt.get("sha256")
        or m4.get("size_bytes") != receipt.get("size_bytes")
        or len(chunks) != 18
        or m4.get("chunk_done_sha256s") != [entry.get("sha256") for entry in chunks]
    ):
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "M4 descriptor")
    receipt_payload = _read_regular(
        bundle_root / str(receipt["path"]), role="relocated M4 receipt"
    )
    receipt_value = loads_strict(receipt_payload, role="relocated M4 receipt")
    if (
        not isinstance(receipt_value, Mapping)
        or receipt_value.get("status") != "complete"
        or receipt_value.get("identity_sha256") != done.get("m4_identity_sha256")
        or receipt_value.get("identity_sha256") != canonical_sha256(
            {key: value for key, value in receipt_value.items() if key != "identity_sha256"}
        )
        or receipt_value.get("archive_root") != m4.get("archive_root")
    ):
        _fail("M6_M4_RECEIPT_REDERIVATION", "relocated M4 receipt identity differs")
    receipt_chunks = receipt_value.get("chunk_done_manifests")
    if not isinstance(receipt_chunks, list) or len(receipt_chunks) != 18:
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "M4 receipt chunk closure")
    receipt_chunks_by_path = {
        str(row.get("done_path")): row
        for row in receipt_chunks
        if isinstance(row, Mapping)
    }
    if len(receipt_chunks_by_path) != 18:
        _fail("M6_FINAL_INVENTORY_INCOMPLETE", "M4 receipt chunk paths")
    archive_payload = _read_regular(
        bundle_root / str(archive["path"]), role="relocated M4 archive manifest"
    )
    archive_value = loads_strict(archive_payload, role="relocated M4 archive manifest")
    archive_entries = (
        archive_value.get("entries") if isinstance(archive_value, Mapping) else None
    )
    if (
        not isinstance(archive_value, Mapping)
        or archive_value.get("schema_version") != 1
        or archive_value.get("root") != "."
        or not isinstance(archive_entries, list)
        or not _sha256_text(archive_value.get("tree_sha256"))
        or archive_value.get("tree_sha256") != canonical_sha256(
            {key: value for key, value in archive_value.items() if key != "tree_sha256"}
        )
    ):
        _fail("M6_M4_ARCHIVE_MANIFEST", "relocated archive manifest seal differs")
    archive_files = {
        str(row.get("relative_path")): row
        for row in archive_entries
        if isinstance(row, Mapping) and row.get("kind") == "file"
    } if isinstance(archive_entries, list) else {}
    for chunk_entry in chunks:
        path = str(chunk_entry["path"])
        done_relative = path.removeprefix("m4/archive/")
        receipt_chunk = receipt_chunks_by_path.get(done_relative)
        if not isinstance(receipt_chunk, Mapping):
            _fail("M6_M4_RECEIPT_REDERIVATION", f"missing chunk {done_relative}")
        if (
            receipt_chunk.get("done_sha256") != chunk_entry.get("sha256")
            or receipt_chunk.get("done_size_bytes") != chunk_entry.get("size_bytes")
            or archive_files.get(done_relative, {}).get("sha256") != chunk_entry.get("sha256")
            or archive_files.get(done_relative, {}).get("size_bytes") != chunk_entry.get("size_bytes")
        ):
            _fail("M6_M4_RECEIPT_REDERIVATION", f"chunk identity {done_relative}")
    evidence = done.get("retained_evidence")
    evidence_entry = require_entry(
        "retained_evidence_tree", "retained_evidence/manifest.json",
    )
    evidence_payload = _read_regular(
        bundle_root / "retained_evidence/manifest.json",
        role="retained evidence manifest",
    )
    manifest = loads_strict(evidence_payload, role="retained evidence manifest")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("artifact_sha256") != evidence_entry.get("sha256")
        or evidence.get("size_bytes") != evidence_entry.get("size_bytes")
        or not isinstance(manifest, Mapping)
        or not _sha256_text(manifest.get("tree_sha256"))
        or manifest.get("tree_sha256") != canonical_sha256(
            {key: value for key, value in manifest.items() if key != "tree_sha256"}
        )
        or evidence.get("tree_sha256") != manifest.get("tree_sha256")
    ):
        _fail("M6_FINAL_INVENTORY_EVIDENCE_SEAL", "retained evidence descriptor")
    manifest_entries = manifest.get("entries") if isinstance(manifest, Mapping) else None
    if not isinstance(manifest_entries, list):
        _fail("M6_FINAL_INVENTORY_EVIDENCE_SEAL", "retained evidence entries")
    manifest_files = {
        str(row.get("relative_path")): row
        for row in manifest_entries
        if isinstance(row, Mapping) and row.get("kind") == "file"
    }
    retained_entries = {
        str(entry.get("path")): entry
        for entry in entries
        if isinstance(entry, Mapping)
        and isinstance(entry.get("kind"), str)
        and entry["kind"].startswith("retained:")
    }
    expected_manifest_files = {
        path.removeprefix("retained_evidence/"): entry
        for path, entry in retained_entries.items()
    }
    if set(manifest_files) != set(expected_manifest_files):
        _fail("M6_FINAL_INVENTORY_EVIDENCE_CLOSURE", "retained file set differs")
    for relative, entry in expected_manifest_files.items():
        manifest_entry = manifest_files.get(relative)
        if (
            not isinstance(manifest_entry, Mapping)
            or manifest_entry.get("sha256") != entry.get("sha256")
            or manifest_entry.get("size_bytes") != entry.get("size_bytes")
        ):
            _fail("M6_FINAL_INVENTORY_EVIDENCE_CLOSURE", relative)
    try:
        with _PinnedTreeTransaction(
            bundle_root,
            role="relocated M6 DONE final closure",
            require_readonly_files=True,
            require_readonly_directories=True,
        ) as final_transaction:
            final_transaction.verify()
    except M6Error:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_ANALYSIS_IMMUTABILITY", f"final bundle closure: {exc}")


def verify_analysis_bundle(
    *,
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    analysis: Mapping[str, Any],
    schedule_path: Path,
    journal_path: Path,
    analysis_path: Path,
    done: Mapping[str, Any] | None,
    done_path: Path | None,
    schedule_payload: bytes | None = None,
    journal_payload: bytes | None = None,
    analysis_payload: bytes | None = None,
    m4_receipt_path: Path | None = None,
    source_snapshot_path: Path | None = None,
    recovery_path: Path | None = None,
    historical_admission_path: Path | None = None,
    evidence_root: Path | None = None,
    bundle_root: Path | None = None,
    repo_root: Path = REPO_ROOT,
    _input_cache: InputSnapshotCache | None = None,
) -> None:
    input_cache = _input_cache or InputSnapshotCache()
    if bundle_root is not None:
        bundle_root = Path(os.path.abspath(os.fspath(bundle_root)))
        bundled_evidence_root = bundle_root / "retained_evidence"
        if evidence_root is None or Path(
            os.path.abspath(os.fspath(evidence_root))
        ) != bundled_evidence_root:
            _fail(
                "M6_EVIDENCE_NAMESPACE",
                "relocated DONE verification requires bundled retained_evidence",
            )
        evidence_root = bundled_evidence_root
        if done is not None:
            bundled_historical = (
                bundle_root / "historical/historical_admission_set.json"
            )
            if historical_admission_path is None:
                historical_admission_path = bundled_historical
            elif Path(
                os.path.abspath(os.fspath(historical_admission_path))
            ) != bundled_historical:
                _fail(
                    "M6_HISTORICAL_AUTHORITY_NAMESPACE",
                    "relocated DONE verification requires the bundled historical authority",
                )
    _require_active_schema("schedule", schedule)
    _require_active_schema("observation_journal", journal)
    _require_active_schema("analysis", analysis)
    if (
        journal.get("schema_version") != schedule.get("schema_version")
        or analysis.get("schema_version") != schedule.get("schema_version")
    ):
        _fail("M6_SCHEMA_MIXED", "bundle inputs use mixed top-level schema versions")
    validate_schedule(
        schedule,
        repo_root=repo_root,
        rehash_sources=True,
        _input_cache=input_cache,
    )
    validate_journal(
        schedule,
        journal,
        journal_path=journal_path,
        repo_root=repo_root,
        rehash_sidecars=True,
        evidence_root=evidence_root,
        _input_cache=input_cache,
    )
    _validate_schema("analysis", analysis, repo_root, cache=input_cache)
    _validate_seal(analysis, "analysis_sha256", role="analysis")
    regenerated = analyze(
        schedule,
        journal,
        journal_path=journal_path,
        generated_at_utc=analysis["generated_at_utc"],
        bootstrap_replicates=analysis["analysis_policy"]["bootstrap_replicates"],
        bootstrap_seed=analysis["analysis_policy"]["bootstrap_seed"],
        historical_admission_path=historical_admission_path,
        evidence_root=evidence_root,
        repo_root=repo_root,
        _input_cache=input_cache,
    )
    if regenerated != analysis:
        _fail("M6_ANALYSIS_REDERIVATION", "determinism study differs")
    if done is None:
        return
    if done_path is None:
        _fail("M6_DONE_PATH", "missing")
    declared_done_path = Path(os.path.abspath(os.fspath(done_path)))
    if bundle_root is not None:
        expected_done_path = Path(
            os.path.abspath(os.fspath(bundle_root / "M6_DONE.json"))
        )
        if declared_done_path != expected_done_path:
            _fail("M6_DONE_PATH", "path is not the canonical bundle DONE")
        bound_done_path = expected_done_path
    else:
        bound_done_path = declared_done_path
    try:
        bound_done_payload = _read_regular(
            bound_done_path, role="bound DONE artifact"
        )
    except M6Error as exc:
        _fail("M6_DONE_PATH", str(exc))
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_DONE_PATH", f"cannot read {bound_done_path}: {exc}")
    try:
        bound_done = loads_strict(bound_done_payload, role="bound DONE artifact")
    except M6Error:
        raise
    if not isinstance(bound_done, Mapping) or dict(bound_done) != dict(done):
        _fail("M6_DONE_PATH", "DONE mapping differs from bound artifact bytes")
    _validate_schema("done", done, repo_root, cache=input_cache)
    _require_active_schema("done", done)
    _validate_seal(done, "done_sha256", role="done")
    if m4_receipt_path is None:
        if bundle_root is None:
            _fail(
                "M6_M4_RECEIPT_UNBOUND",
                "relocation verification requires the bundled inventory root",
            )
        _verify_done_inventory_bundle(done, bundle_root=bundle_root)
        return
    expected_done = build_done(
        schedule,
        journal,
        analysis,
        schedule_path=schedule_path,
        journal_path=journal_path,
        analysis_path=analysis_path,
        schedule_payload=schedule_payload,
        journal_payload=journal_payload,
        analysis_payload=analysis_payload,
        generated_at_utc=done["generated_at_utc"],
        repo_root=repo_root,
        _input_cache=input_cache,
        m4_receipt_path=m4_receipt_path,
        source_snapshot_path=source_snapshot_path,
        recovery_path=recovery_path,
        historical_admission_path=historical_admission_path,
        evidence_root=evidence_root,
    )
    if expected_done != done:
        _fail("M6_DONE_REDERIVATION", "completion seal differs")
    inventory = done.get("inventory")
    if isinstance(inventory, dict):
        if inventory.get("inventory_sha256") != canonical_sha256(inventory.get("entries")):
            _fail("M6_FINAL_INVENTORY_INCOMPLETE", "inventory seal differs")


def _publish_analysis_directory(
    output: Path,
    *,
    analysis_payload: bytes,
    done_payload: bytes | None,
    source_snapshot_payload: bytes | None = None,
    dependency_material: Mapping[str, bytes] | None = None,
    repo_root: Path,
) -> Path:
    repository = _canonical_repo_root(repo_root)
    output = _lexical_absolute(output, role="M6 analysis output")
    parent = _resolve_no_symlink_chain(output.parent, role="M6 analysis parent")
    if not parent.is_dir():
        _fail("M6_OUTPUT_PARENT", str(parent))
    final = parent / output.name
    if _contains_path(repository, final) or _contains_path(final, repository):
        _fail(
            "M6_ANALYSIS_OUTPUT_OVERLAP",
            f"output={final} repository={repository}",
        )
    try:
        immutable.safe_relative_path(output.name)
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_OUTPUT_PATH", f"{output}: {exc}")
    if output.name.startswith(".") or output.name.endswith(".staging"):
        _fail("M6_OUTPUT_PATH", f"reserved publication name: {output.name}")
    stage_name = f".{output.name}.staging"
    try:
        pinned_parent = immutable.PinnedDirectoryReader(parent)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_OUTPUT_PARENT", f"{parent}: {exc}")
    try:
        pinned_parent.verify_path()
        if _entry_metadata(pinned_parent.descriptor, output.name) is not None:
            _fail("M6_NO_OVERWRITE", str(output))
        if _entry_metadata(pinned_parent.descriptor, stage_name) is not None:
            _fail("M6_PARTIAL_PRESERVED", str(parent / stage_name))
        material_payloads: dict[str, bytes] = {
            "determinism_study.json": bytes(analysis_payload),
        }
        if done_payload is not None:
            if source_snapshot_payload is None:
                _fail("M6_SOURCE_SNAPSHOT_UNBOUND", "DONE requires source snapshot bytes")
            material_payloads["source_snapshot/identity.json"] = bytes(
                source_snapshot_payload
            )
            material_payloads["M6_DONE.json"] = bytes(done_payload)
            if dependency_material is None:
                _fail(
                    "M6_FINAL_INVENTORY_INCOMPLETE",
                    "complete DONE publication requires dependency material",
                )
            for relative, payload in dependency_material.items():
                try:
                    safe_relative = immutable.safe_relative_path(relative).as_posix()
                except (TypeError, immutable.ImmutableArtifactError) as exc:
                    _fail("M6_ANALYSIS_MATERIAL_PATH", f"{relative!r}: {exc}")
                if not isinstance(payload, bytes):
                    _fail("M6_ANALYSIS_MATERIAL_PAYLOAD", safe_relative)
                prior = material_payloads.get(safe_relative)
                if prior is not None and prior != payload:
                    _fail("M6_ANALYSIS_MATERIAL_MISMATCH", safe_relative)
                material_payloads[safe_relative] = bytes(payload)
        material = [
            immutable.ImmutableFile.from_bytes(relative, payload)
            for relative, payload in sorted(material_payloads.items())
        ]
        authorize_calls = 0

        def authorize(_root: Path, root_descriptor: int) -> None:
            nonlocal authorize_calls
            authorize_calls += 1
            if (
                authorize_calls == 1
                and _entry_metadata(root_descriptor, stage_name) is not None
            ):
                _fail("M6_PARTIAL_PRESERVED", str(parent / stage_name))

        try:
            published = immutable.publish_directory(
                parent,
                output.name,
                material,
                before_commit=authorize,
                pinned_root=pinned_parent,
            )
            immutable.verify_published_files(published, material)
        except M6Error:
            raise
        except (OSError, immutable.ImmutableArtifactError) as exc:
            _fail("M6_ANALYSIS_PUBLICATION", f"{output}: {exc}")
        return published
    finally:
        pinned_parent.close()


def _load_mapping(
    path: Path,
    role: str,
    *,
    cache: InputSnapshotCache | None = None,
) -> tuple[dict[str, Any], bytes]:
    value, payload = load_json(path, role=role, cache=cache)
    if not isinstance(value, dict):
        _fail("M6_ARTIFACT_SHAPE", f"{role}: object required")
    return value, payload


def _cmd_bind_m4(args: argparse.Namespace) -> int:
    repository = _canonical_repo_root(Path(args.repo_root))
    archive_root = Path(args.m4_root)
    output = Path(args.out)
    pinned_output = _authorize_m4_receipt_output(
        output, archive_root=archive_root, repo_root=repository,
    )
    try:
        receipt = build_m4_binding_receipt(
            archive_root=archive_root,
            generated_at_utc=args.generated_at_utc,
            repo_root=repository,
        )
        _write_new(output, receipt, pinned_parent=pinned_output)
    finally:
        pinned_output.close()
    print(
        f"m4_identity_sha256={receipt['identity_sha256']} "
        f"chunks={len(receipt['chunk_done_manifests'])} "
        f"cells={len(receipt['cell_launch_identities'])}"
    )
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    schedule = build_schedule(
        matrix_path=Path(args.matrix),
        registry_path=Path(args.metric_registry),
        sidecar_schema_path=Path(args.sidecar_schema),
        m4_receipt_path=Path(args.m4_receipt) if args.m4_receipt else None,
        seed=args.seed,
        max_new_profiles_per_resume=args.max_new_profiles_per_resume,
        generated_at_utc=args.generated_at_utc,
        repo_root=Path(args.repo_root),
    )
    _write_new(Path(args.out), schedule)
    print(f"schedule_sha256={schedule['schedule_sha256']} tier1_slots=8540 maximum_slots=9760")
    return 0


def _cmd_publish_journal(args: argparse.Namespace) -> int:
    repository = _canonical_repo_root(Path(args.repo_root))
    input_cache = InputSnapshotCache()
    schedule, _ = _load_mapping(
        Path(args.schedule), "schedule", cache=input_cache,
    )
    journal, _ = _load_mapping(
        Path(args.journal_draft), "journal draft", cache=input_cache,
    )
    publish_journal(
        schedule,
        journal,
        output=Path(args.out),
        repo_root=repository,
        _input_cache=input_cache,
    )
    print(f"journal_sha256={journal['journal_sha256']}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    repo_root = _canonical_repo_root(Path(args.repo_root))
    schedule_path = Path(args.schedule)
    journal_path = Path(args.journal)
    evidence_root = Path(args.evidence_root) if args.evidence_root else None
    historical_admission_path = (
        Path(args.historical_admission_set)
        if args.historical_admission_set
        else None
    )
    input_cache = InputSnapshotCache()
    schedule, schedule_payload = _load_mapping(
        schedule_path, "schedule", cache=input_cache,
    )
    journal, journal_payload = _load_mapping(
        journal_path, "journal", cache=input_cache,
    )
    analysis = analyze(
        schedule,
        journal,
        journal_path=journal_path,
        generated_at_utc=args.generated_at_utc,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        historical_admission_path=historical_admission_path,
        evidence_root=evidence_root,
        repo_root=repo_root,
        _input_cache=input_cache,
    )
    analysis_payload = _pretty_json_bytes(analysis)
    emitted_done = (
        analysis.get("publication_state") == "M6_DONE"
        and analysis.get("scientific_gates", {}).get("done_eligible") is True
        and schedule["m4_identity"]["status"] == "complete"
    )
    if args.require_done and not emitted_done:
        _fail(
            "M6_DONE_NOT_EMITTED",
            f"completion={analysis['completion_state']} m4={schedule['m4_identity']['status']}",
        )
    out_dir = Path(args.out_dir)
    done_payload: bytes | None = None
    source_snapshot_payload: bytes | None = None
    publication_material: dict[str, bytes] = {}
    m4_receipt_path = Path(args.m4_receipt) if args.m4_receipt else None
    recovery_path = (
        Path(args.recovery_receipt) if args.recovery_receipt else None
    )
    source_snapshot_path = Path(args.source_snapshot) if args.source_snapshot else out_dir / "source_snapshot" / "identity.json"
    if emitted_done and m4_receipt_path is None:
        _fail("M6_M4_RECEIPT_UNBOUND", "--m4-receipt is required for DONE publication")
    if emitted_done:
        done = build_done(
            schedule,
            journal,
            analysis,
            schedule_path=schedule_path,
            journal_path=journal_path,
            analysis_path=out_dir / "determinism_study.json",
            schedule_payload=schedule_payload,
            journal_payload=journal_payload,
            analysis_payload=analysis_payload,
            generated_at_utc=args.generated_at_utc,
            repo_root=repo_root,
            _input_cache=input_cache,
            m4_receipt_path=m4_receipt_path,
            source_snapshot_path=source_snapshot_path,
            recovery_path=recovery_path,
            historical_admission_path=historical_admission_path,
            evidence_root=evidence_root,
            _publication_material=publication_material,
        )
        done_payload = _pretty_json_bytes(done)
        source_snapshot_payload = canonical_bytes(done["source_snapshot"]["witness"])
    _publish_analysis_directory(
        out_dir,
        analysis_payload=analysis_payload,
        done_payload=done_payload,
        source_snapshot_payload=source_snapshot_payload,
        dependency_material=publication_material if emitted_done else None,
        repo_root=repo_root,
    )
    print(
        f"analysis_sha256={analysis['analysis_sha256']} "
        f"completion={analysis['completion_state']} "
        f"done={str(emitted_done).lower()}"
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    repo_root = _canonical_repo_root(Path(args.repo_root))
    input_cache = InputSnapshotCache()
    bundle_root = (
        _lexical_absolute(Path(args.analysis_root), role="M6 analysis root")
        if args.analysis_root else None
    )
    if args.schedule:
        schedule_path = Path(args.schedule)
    elif bundle_root is not None:
        schedule_path = bundle_root / "schedule.json"
    else:
        _fail("M6_SCHEDULE_REQUIRED", "--schedule or a complete analysis bundle is required")
    schedule, schedule_payload = _load_mapping(
        schedule_path, "schedule", cache=input_cache,
    )
    validate_schedule(
        schedule,
        repo_root=repo_root,
        rehash_sources=True,
        _input_cache=input_cache,
    )
    if args.journal:
        journal_path = Path(args.journal)
    elif bundle_root is not None and os.path.lexists(bundle_root / "observation_journal.json"):
        journal_path = bundle_root / "observation_journal.json"
    else:
        if not args.analysis_root:
            print(f"verified_schedule_sha256={schedule['schedule_sha256']}")
            return 0
        _fail("M6_JOURNAL_REQUIRED", "analysis bundle has no journal payload")
    journal, journal_payload = _load_mapping(
        journal_path, "journal", cache=input_cache,
    )
    relocated_evidence_root = (
        bundle_root / "retained_evidence"
        if bundle_root is not None
        and os.path.lexists(bundle_root / "retained_evidence")
        else None
    )
    requested_evidence_root = (
        Path(args.evidence_root) if args.evidence_root else relocated_evidence_root
    )
    validate_journal(
        schedule,
        journal,
        journal_path=journal_path,
        repo_root=repo_root,
        rehash_sidecars=True,
        evidence_root=requested_evidence_root,
        _input_cache=input_cache,
    )
    if not args.analysis_root:
        print(f"verified_journal_sha256={journal['journal_sha256']}")
        return 0
    root = bundle_root
    if root is None:
        _fail("M6_ANALYSIS_ROOT", "analysis root is required")
    with _PinnedTreeTransaction(
        root,
        role="M6 analysis bundle",
        require_readonly_files=True,
        require_readonly_directories=True,
        input_cache=input_cache,
    ) as transaction:
        observed = {
            path: kind
            for path, (kind, _snapshot) in transaction.inventory.items()
            if path
        }
        has_done = "M6_DONE.json" in observed
        has_snapshot = "source_snapshot/identity.json" in observed
        done: dict[str, Any] | None = None
        done_payload: bytes | None = None
        source_snapshot_payload: bytes | None = None
        if has_done:
            done_payload = transaction.read("M6_DONE.json", role="done")
            loaded_done = loads_strict(done_payload, role="done")
            if not isinstance(loaded_done, dict):
                _fail("M6_ARTIFACT_SHAPE", "done: object required")
            done = loaded_done
            inventory = done.get("inventory")
            inventory_entries = inventory.get("entries") if isinstance(inventory, Mapping) else None
            if not isinstance(inventory_entries, list):
                _fail("M6_FINAL_INVENTORY_INCOMPLETE", "bundle DONE inventory is absent")
            expected_paths = [
                *(
                    str(entry.get("path"))
                    for entry in inventory_entries
                    if isinstance(entry, Mapping)
                ),
                "M6_DONE.json",
            ]
            expected = _bundle_expected_topology(expected_paths)
            if observed != expected:
                _fail(
                    "M6_ANALYSIS_INVENTORY",
                    f"observed={sorted(observed.items())} expected={sorted(expected.items())}",
                )
            if not has_snapshot:
                _fail("M6_SOURCE_SNAPSHOT_MISSING", str(root / "source_snapshot/identity.json"))
            source_snapshot_payload = transaction.read(
                "source_snapshot/identity.json", role="source snapshot",
            )
        else:
            expected = {"determinism_study.json": "file"}
            if observed != expected:
                _fail(
                    "M6_ANALYSIS_INVENTORY",
                    f"observed={sorted(observed.items())}",
                )
            if args.require_done:
                _fail("M6_DONE_MISSING", str(root / "M6_DONE.json"))
        if "schedule.json" in observed:
            bundle_schedule_payload = transaction.read("schedule.json", role="bundle schedule")
            if bundle_schedule_payload != schedule_payload:
                _fail("M6_BUNDLE_INPUT_MISMATCH", "schedule")
        if "observation_journal.json" in observed:
            bundle_journal_payload = transaction.read(
                "observation_journal.json", role="bundle journal",
            )
            if bundle_journal_payload != journal_payload:
                _fail("M6_BUNDLE_INPUT_MISMATCH", "journal")
        analysis_payload = transaction.read(
            "determinism_study.json", role="analysis",
        )
        analysis = loads_strict(analysis_payload, role="analysis")
        if not isinstance(analysis, dict):
            _fail("M6_ARTIFACT_SHAPE", "analysis: object required")
        transaction.verify()
    analysis_path = root / "determinism_study.json"
    done_path = root / "M6_DONE.json"
    verify_analysis_bundle(
        schedule=schedule,
        journal=journal,
        analysis=analysis,
        schedule_path=schedule_path,
        journal_path=journal_path,
        analysis_path=analysis_path,
        done=done,
        done_path=done_path if done is not None else None,
        schedule_payload=schedule_payload,
        journal_payload=journal_payload,
        analysis_payload=analysis_payload,
        m4_receipt_path=Path(args.m4_receipt) if args.m4_receipt else None,
        source_snapshot_path=(
            Path(args.source_snapshot)
            if args.source_snapshot
            else (root / "source_snapshot" / "identity.json" if done is not None else None)
        ),
        recovery_path=Path(args.recovery_receipt) if args.recovery_receipt else None,
        historical_admission_path=(
            Path(args.historical_admission_set)
            if args.historical_admission_set
            else None
        ),
        evidence_root=requested_evidence_root,
        bundle_root=root if done is not None else None,
        repo_root=repo_root,
        _input_cache=input_cache,
    )
    material = [
        immutable.ImmutableFile.from_bytes(
            "determinism_study.json", analysis_payload,
        )
    ]
    if done_payload is not None:
        if source_snapshot_payload is None:
            _fail("M6_SOURCE_SNAPSHOT_MISSING", str(root / "source_snapshot/identity.json"))
        material.append(
            immutable.ImmutableFile.from_bytes(
                "source_snapshot/identity.json", source_snapshot_payload,
            )
        )
        material.append(
            immutable.ImmutableFile.from_bytes("M6_DONE.json", done_payload)
        )
    try:
        immutable.verify_published_files(root, material)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_ANALYSIS_IMMUTABILITY", f"{root}: {exc}")
    print(
        f"verified_analysis_sha256={analysis['analysis_sha256']} "
        f"done={str(done is not None).lower()}"
    )
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    schedule, _ = _load_mapping(Path(args.schedule), "schedule")
    validate_schedule(schedule, repo_root=repo_root, rehash_sources=True)
    matrix_payload = _resolve_repo_file(
        schedule["source_identity"]["canonical_matrix"],
        repo_root,
        role="canonical matrix",
    )
    matrix = loads_strict(matrix_payload, role="canonical matrix")
    _validate_m4_identity(
        schedule["m4_identity"],
        matrix=matrix,
        matrix_sha256=sha256_bytes(matrix_payload),
        cells=schedule["cells"],
        repo_root=repo_root,
        require_complete=True,
    )
    _fail(
        "M6_COLLECT_DISABLED_OFFLINE_SOURCE",
        "collection is intentionally absent; use the separately reviewed hardware owner after complete M4 identity",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    bind_m4 = sub.add_parser(
        "bind-m4",
        help="derive one verified receipt from the exact 18 canonical M4 DONEs",
    )
    bind_m4.add_argument("--m4-root", required=True)
    bind_m4.add_argument("--generated-at-utc")
    bind_m4.add_argument("--out", required=True)
    bind_m4.set_defaults(func=_cmd_bind_m4)

    plan = sub.add_parser("plan", help="emit a sealed offline schedule")
    plan.add_argument("--matrix", default=str(MATRIX_PATH))
    plan.add_argument("--metric-registry", default=str(REGISTRY_PATH))
    plan.add_argument("--sidecar-schema", default=str(SIDECAR_SCHEMA_PATH))
    plan.add_argument("--m4-receipt")
    plan.add_argument("--seed", required=True, type=int)
    plan.add_argument("--max-new-profiles-per-resume", type=int, default=TIER1_SLOTS)
    plan.add_argument("--generated-at-utc")
    plan.add_argument("--out", required=True)
    plan.set_defaults(func=_cmd_plan)

    publish_journal_cmd = sub.add_parser(
        "publish-journal",
        help="validate a complete draft and publish one immutable raw journal",
    )
    publish_journal_cmd.add_argument("--schedule", required=True)
    publish_journal_cmd.add_argument("--journal-draft", required=True)
    publish_journal_cmd.add_argument("--out", required=True)
    publish_journal_cmd.set_defaults(func=_cmd_publish_journal)

    analyze_cmd = sub.add_parser("analyze", help="analyze immutable external v5 sidecars through a sealed journal")
    analyze_cmd.add_argument("--schedule", required=True)
    analyze_cmd.add_argument("--journal", required=True)
    analyze_cmd.add_argument("--out-dir", required=True)
    analyze_cmd.add_argument("--bootstrap-replicates", type=int, default=1000)
    analyze_cmd.add_argument("--bootstrap-seed", type=int)
    analyze_cmd.add_argument("--generated-at-utc")
    analyze_cmd.add_argument("--require-done", action="store_true")
    analyze_cmd.add_argument("--m4-receipt")
    analyze_cmd.add_argument("--source-snapshot")
    analyze_cmd.add_argument("--recovery-receipt")
    analyze_cmd.add_argument(
        "--historical-admission-set",
        help="separately supplied exact historical admission set authority",
    )
    analyze_cmd.add_argument(
        "--evidence-root",
        help="dedicated root containing the journal-referenced sidecars and launch evidence",
    )
    analyze_cmd.set_defaults(func=_cmd_analyze)

    verify = sub.add_parser("verify", help="rehash and semantically rederive offline artifacts")
    verify.add_argument("--schedule")
    verify.add_argument("--journal")
    verify.add_argument("--analysis-root")
    verify.add_argument("--m4-receipt")
    verify.add_argument("--source-snapshot")
    verify.add_argument("--recovery-receipt")
    verify.add_argument(
        "--historical-admission-set",
        help="external authority path; relocated DONE bundles use their retained copy",
    )
    verify.add_argument(
        "--evidence-root",
        help="dedicated root containing external journal-referenced evidence",
    )
    verify.add_argument("--require-done", action="store_true")
    verify.set_defaults(func=_cmd_verify)

    collect = sub.add_parser("collect", help="disabled in this offline source")
    collect.add_argument("--schedule", required=True)
    collect.set_defaults(func=_cmd_collect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except M6Error as exc:
        print(f"ERROR {exc.code}: {exc.detail}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
