#!/usr/bin/env python3
"""Trace-budget sidecar verifier for SimBench-BW.

Walks every trace artifact under SUITE_ROOT / results / and enforces:

1. **Trace volume** (existing): any oversize trace on disk must be paired
   with a `trace_volume_exceeded.json` sidecar matching the committed
   schema. An oversize trace without a matching sidecar is
   `BudgetViolationUntracked`.

2. **Slowdown** (C-AC-H): every per-shape `trace_*.json` run summary is
   verified by recomputing
   ``actual_wall_time_s / baseline_wall_time_s`` from the raw evidence
   the controller persisted, comparing the recomputed ratio to the fixed
   threshold ``NVBIT_SLOWDOWN_BUDGET_MAX_RATIO`` declared in
   ``tools/cloud/trace_budgets.py``, and rejecting any breach without a
   matching `nvbit_slowdown_budget_exceeded.json` sidecar bound to the
   same `slug`/`shape_point`.

3. **Drop rate** (C-AC-H): same shape but for
   ``1 - admitted_record_count / expected_record_count`` recomputed from
   the raw evidence, against ``NVBIT_DROP_RATE_BUDGET_MAX``.

The R11 strict-schema discipline applies: every present typed sidecar's
on-disk bytes must admit under
``jsonschema.Draft202012Validator``. The R12 closure ensures cached
ratio fields cannot be falsified to bypass enforcement: cached values
are consistency-checked against the recomputation, and absent cached
values do not relax enforcement when raw evidence is present.

Exit codes:
    0  all trace artifacts are within budget OR paired with a binding sidecar.
    1  at least one BudgetViolationUntracked or schema mismatch.
    2  usage error.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import sys
import tomllib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import jsonschema


SUITE_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = SUITE_ROOT / "schemas"
_REPO_ROOT = SUITE_ROOT.parent.parent

_TOOLS_CLOUD_DIR = SUITE_ROOT.parent.parent / "tools" / "cloud"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_TOOLS_CLOUD_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_CLOUD_DIR))
import trace_budgets as _trace_budgets  # type: ignore[import-not-found]  # noqa: E402
from tools.local_labels.immutable_artifact import (  # noqa: E402
    DirectoryGenerationFile,
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
    ImmutableDirectoryGeneration,
    PinnedDirectoryReader,
    PinnedPathSetMonitor,
    PinnedTreeSnapshot,
    inspect_immutable_directory_generation,
    strict_json_loads,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    DEFAULT_TRACE_ADMISSION_LIMITS,
    TraceAdmission,
    TraceAdmissionError,
    raw_trace_generation_limits,
    require_trace_admission,
)
from workloads.simbench_bw.nvbit_trace_summary_adapter import (  # noqa: E402
    compute_kernel_signature_sha256,
)
from workloads.simbench_bw.phase_cycle_ledger_aggregator import (  # noqa: E402
    PhaseLedgerSchemaInvalid,
    serialize_phase_cycle_ledger,
)
from workloads.simbench_bw.phase_cycle_ledger_consumer import (  # noqa: E402
    NvbitPhaseInjectionOverheadExceeded,
    PhaseBoundaryMappingUnresolved,
    admit_phase_cycle_ledger,
)


# Numeric tolerance for cached-ratio consistency checks. The cached
# ratio fields are produced by floating-point arithmetic on the same
# raw fields the verifier recomputes from; differences smaller than
# this tolerance are accepted as numerically equivalent. A larger
# discrepancy is BudgetViolationUntracked-equivalent because the
# cached field cannot have been derived from the raw evidence.
_RATIO_CONSISTENCY_TOL: float = 1e-6
_RAW_TRACE_GENERATION_NAMESPACE = "raw_trace_generations"
_DERIVED_TRACE_GENERATION_NAMESPACE = "trace_derived_generations"
_SUMMARY_GENERATION_NAMESPACE = "nvbit_trace_summary_generations"
_PHASE_LEDGER_GENERATION_NAMESPACE = "phase_cycle_ledger_generations"
_TRACE_CAPTURE_ATTEMPT_NAMESPACE = "trace_capture_attempts"
_CANONICAL_RESERVED_NAMESPACE_COMPONENTS = frozenset({
    _RAW_TRACE_GENERATION_NAMESPACE,
    _DERIVED_TRACE_GENERATION_NAMESPACE,
    _SUMMARY_GENERATION_NAMESPACE,
    _PHASE_LEDGER_GENERATION_NAMESPACE,
    _TRACE_CAPTURE_ATTEMPT_NAMESPACE,
})
_CANONICAL_PLATFORM_KEYS = frozenset({
    "h100",
    "rtx_5090_blackwell",
    "rtx_5060_laptop",
    "rtx_3090_ti",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_DATETIME_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})$"
)
_SUMMARY_FILENAME_RE = re.compile(
    r"^nvbit_trace_summary_([0-9]+)_([0-9]+)\.json$"
)
_PHASE_LEDGER_FILENAME_RE = re.compile(
    r"^phase_cycle_ledger_([0-9]+)_([0-9]+)\.json$"
)
_CANONICAL_DISCOVERY_MAX_FILES = 250_000
_CANONICAL_DISCOVERY_MAX_DIRECTORIES = 100_000
_CANONICAL_DISCOVERY_MAX_TOTAL_FILE_BYTES = 4 * 1024**4
_CANONICAL_DISCOVERY_MAX_RELATIVE_PATH_BYTES = 64 * 1024 * 1024
_CANONICAL_AUTHORITY_JSON_MAX_BYTES = 16 * 1024 * 1024
_CANONICAL_AUTHORITY_JSON_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_CANONICAL_WORKING_SET_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_CANONICAL_MATERIALIZED_MAX_OBJECTS = 4_000_000
_CANONICAL_MATERIALIZATION_TRACKER_ENTRY_BYTES = sys.getsizeof({0: None})
_DIRECTORY_GENERATION_DESCRIPTOR_FIELDS = frozenset({
    "schema_version",
    "attempt_id",
    "path",
    "content_manifest_sha256",
    "filesystem_generation_sha256",
    "file_count",
    "directory_count",
    "total_file_bytes",
})
_RAW_GENERATION_DESCRIPTOR_FIELDS = _DIRECTORY_GENERATION_DESCRIPTOR_FIELDS | {
    "trace_admission_sha256",
    "trace_filesystem_generation_sha256",
    "dynamic_trace_sha256",
}
_SUMMARY_GENERATION_DESCRIPTOR_FIELDS = frozenset({
    "schema_version",
    "path",
    "content_manifest_sha256",
    "filesystem_generation_sha256",
    "file_count",
    "directory_count",
    "total_file_bytes",
    "files",
    "manifest_sha256",
})
_MEASUREMENT_MANIFEST_SCHEMA_PATH = (
    SCHEMAS_DIR / "measurement_manifest.schema.json"
)


class _CanonicalAuthorityReadBudget:
    """Bound retained JSON authority before each payload allocation."""

    def __init__(self) -> None:
        self.total_bytes = 0
        self.reserved: dict[str, int] = {}

    def reserve_total(self, path: Path, size: int, *, role: str) -> None:
        if type(size) is not int or size < 0:
            raise ImmutableArtifactError(
                f"{role} has an invalid inventoried size: {path}"
            )
        key = str(path.absolute())
        prior = self.reserved.get(key)
        if prior is not None:
            if prior != size:
                raise ImmutableArtifactError(
                    f"{role} retained size changed for {path}"
                )
            return
        attempted = self.total_bytes + size
        if attempted > _CANONICAL_AUTHORITY_JSON_MAX_TOTAL_BYTES:
            raise ImmutableArtifactLimitError(
                f"{role} aggregate retained JSON authority is {attempted} "
                "bytes; maximum is "
                f"{_CANONICAL_AUTHORITY_JSON_MAX_TOTAL_BYTES}"
            )
        self.reserved[key] = size
        self.total_bytes = attempted

    def reserve_file(self, path: Path, size: int, *, role: str) -> None:
        if type(size) is not int or size < 0:
            raise ImmutableArtifactError(
                f"{role} has an invalid inventoried size: {path}"
            )
        if size > _CANONICAL_AUTHORITY_JSON_MAX_BYTES:
            raise ImmutableArtifactLimitError(
                f"{role} file {path} is {size} bytes; maximum is "
                f"{_CANONICAL_AUTHORITY_JSON_MAX_BYTES}"
            )
        self.reserve_total(path, size, role=role)


def _retained_path_references(
    path: PurePosixPath,
    *,
    role: str,
) -> list[object]:
    references: list[object] = []
    has_component_storage = False
    for path_type in type(path).__mro__:
        slots = path_type.__dict__.get("__slots__", ())
        if type(slots) is str:
            slots = (slots,)
        for slot in slots:
            if slot == "__weakref__":
                continue
            descriptor = path_type.__dict__.get(slot)
            if descriptor is None:
                raise ImmutableArtifactError(
                    f"{role} cannot inspect pathlib retained storage"
                )
            try:
                retained = descriptor.__get__(path, type(path))
            except AttributeError:
                continue
            if slot in {"_raw_paths", "_parts"}:
                if type(retained) not in {list, tuple}:
                    raise ImmutableArtifactError(
                        f"{role} cannot account pathlib retained storage"
                    )
                has_component_storage = True
            references.append(retained)
    if not has_component_storage:
        raise ImmutableArtifactError(
            f"{role} cannot account pathlib retained storage"
        )
    return references


@dataclass(slots=True)
class _CanonicalAuthorityMaterializationBudget:
    """Bound decoded authority graphs retained beside their source bytes."""

    total_bytes: int = 0
    object_count: int = 0
    seen_objects: dict[int, object] = field(default_factory=dict)

    def reserve(
        self,
        value: object,
        *,
        retained_input_bytes: int,
        role: str,
    ) -> None:
        pending = [value]
        added_objects_by_id: dict[int, object] = {}
        expanded_references: set[int] = set()
        added_bytes = 0
        added_objects = 0
        while pending:
            current = pending.pop()
            identity = id(current)
            if (
                identity not in self.seen_objects
                and identity not in added_objects_by_id
            ):
                added_objects_by_id[identity] = current
                added_objects += 1
                if (
                    self.object_count + added_objects
                    > _CANONICAL_MATERIALIZED_MAX_OBJECTS
                ):
                    raise ImmutableArtifactLimitError(
                        f"{role} retains more than "
                        f"{_CANONICAL_MATERIALIZED_MAX_OBJECTS} decoded objects"
                    )
                added_bytes += (
                    sys.getsizeof(current)
                    + _CANONICAL_MATERIALIZATION_TRACKER_ENTRY_BYTES
                )
                attempted = retained_input_bytes + self.total_bytes + added_bytes
                if attempted > _CANONICAL_WORKING_SET_MAX_TOTAL_BYTES:
                    raise ImmutableArtifactLimitError(
                        f"{role} retained input plus decoded authority material is "
                        f"{attempted} bytes; maximum is "
                        f"{_CANONICAL_WORKING_SET_MAX_TOTAL_BYTES}"
                    )
            if identity in expanded_references:
                continue
            expanded_references.add(identity)
            if type(current) is dict:
                for key, item in current.items():
                    pending.append(key)
                    pending.append(item)
            elif type(current) in {list, tuple, set, frozenset}:
                pending.extend(current)
            elif isinstance(current, PurePosixPath):
                pending.extend(
                    _retained_path_references(current, role=role)
                )
            elif type(current) is PinnedTreeSnapshot:
                pending.extend((current.files, current.directories))
            elif type(current) is ImmutableDirectoryGeneration:
                pending.extend((
                    current.path,
                    current.content_manifest_sha256,
                    current.filesystem_generation_sha256,
                    current.file_count,
                    current.directory_count,
                    current.total_file_bytes,
                    current.files,
                ))
            elif type(current) is DirectoryGenerationFile:
                pending.extend((
                    current.relative_path,
                    current.sha256,
                    current.size_bytes,
                ))
            elif type(current) not in {
                type(None),
                bool,
                int,
                float,
                str,
                bytes,
            }:
                raise ImmutableArtifactError(
                    f"{role} contains unsupported authority material type "
                    f"{type(current).__name__}"
                )
        self.seen_objects.update(added_objects_by_id)
        self.object_count += added_objects
        self.total_bytes += added_bytes

    def reserve_bytes(
        self,
        size: int,
        *,
        retained_input_bytes: int,
        role: str,
    ) -> None:
        if type(size) is not int or size < 0:
            raise ImmutableArtifactError(
                f"{role} has an invalid retained byte charge"
            )
        attempted = retained_input_bytes + self.total_bytes + size
        if attempted > _CANONICAL_WORKING_SET_MAX_TOTAL_BYTES:
            raise ImmutableArtifactLimitError(
                f"{role} retained input plus decoded authority material is "
                f"{attempted} bytes; maximum is "
                f"{_CANONICAL_WORKING_SET_MAX_TOTAL_BYTES}"
            )
        self.total_bytes += size


_KNOWN_NONAUTHORITY_TRACE_FILENAMES = frozenset({
    "trace_volume_exceeded.json",
})


def _is_rfc3339_datetime(value: object) -> bool:
    if type(value) is not str or _RFC3339_DATETIME_RE.fullmatch(value) is None:
        return False
    normalized = value[:-1] + "+00:00" if value[-1] in "Zz" else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _exact_json_equal(observed: object, expected: object) -> bool:
    """Compare decoded JSON without Python bool/int/float coercion."""
    pending = [(observed, expected)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if type(left) is dict:
            if set(left) != set(right):
                return False
            pending.extend((left[key], right[key]) for key in left)
        elif type(left) is list:
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right, strict=True))
        elif left != right:
            return False
    return True


# Cache-predictor verifier defect-token frozenset. Every typed
# defect ``_verify_cache_predictor_run_summaries`` can produce MUST
# be enumerated here. The bidirectional structural lockstep test
# in ``tests/test_verify_cache_predictor_run_summaries.py`` asserts
# ``set(asserted_verifier_defect_tokens) == _VERIFIER_DEFECT_TOKENS``
# (set equality, not subset-of). The vocabulary is intentionally
# separate from ``cache_predictor._TYPED_REASONS`` (the predictor's
# typed-reason enum) so verifier-level defects do not conflate with
# predictor-level admit-failure reasons; a parity test asserts
# disjointness. Pattern source:
# ``BL-20260509-strict-admission-contract-first``.
_VERIFIER_DEFECT_TOKENS: frozenset[str] = frozenset({
    "cache_predictor_manifest_pairing_index_mismatch",
    "cache_predictor_sidecar_digest_drift",
    "cache_predictor_circular_predictor_input_rejected",
    "cache_predictor_bounded_arm_inversion",
    "cache_predictor_bundle_underconfigured_with_admitted_arm",
    "cache_predictor_schema_invalid",
    "cache_predictor_run_summary_missing_path",
})

# Forbidden keys in a predictor bundle's ``input_sidecar_digests``
# (the non-circular discipline). Mirrors
# ``cache_predictor._CIRCULAR_INPUT_KEYS`` and the schema's
# ``additionalProperties: false`` clause; the verifier records the
# typed defect so the failure surfaces with a stable token even
# when the schema validator already rejected.
_CACHE_PREDICTOR_FORBIDDEN_DIGEST_KEYS: frozenset[str] = frozenset({
    "hrd_profile",
    "hrd_profile_path",
    "hit_count",
    "hit_rate",
    "model_hit",
    "modeled_hit_rate",
    "predicted_hit_rate",
})


def _load_roster() -> dict[str, dict]:
    with open(SUITE_ROOT / "roster.toml", "rb") as f:
        return {r["slug"]: r for r in tomllib.load(f)["rows"]}


def _trace_budgets_sha256() -> str:
    p = SUITE_ROOT / "trace_budgets.toml"
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_trace_dir(
    trace_dir: Path,
    *,
    require_canonical_trace_summary: bool = False,
) -> list[str]:
    """Check every trace file in trace_dir against its row's budget.

    A trace file is identified by filename ending `.trace` or `.trace.zst`.
    Budget is read from roster.toml trace_budget_mib.
    If trace size > budget, there must be a `trace_volume_exceeded.json` sidecar
    in the same directory with consistent content.

    ``require_canonical_trace_summary`` gates the strict canonical-capture
    admission scan (`_verify_canonical_trace_pb_admission`). That scan asserts
    every ``trace_<shape>/dynamic_trace.pb`` was produced by the
    ``run_suite.py --mode trace`` orchestrator (a sibling ``trace_<shape>.json``
    binder + ``measurement_manifest.json``). It is a FRESH-CAPTURE gate, not a
    property of arbitrary in-tree results: a host's accumulated `results/` tree
    (gitignored, host-local) legitimately holds legacy/audit captures from other
    capture paths (e.g. the activation orchestrator, or raw-pb + analysis-sidecar
    audit flows) that carry a different binder layout or none, and which are
    indistinguishable BY CONTENT from a genuine bypass-orchestrator capture.
    Freshness is invocation context, not artifact content, so the default
    operator sweep over `results/` leaves this scan OFF; the capture/CI workflow
    passes ``require_canonical_trace_summary=True`` against the artifact root it
    just produced to keep a real bypass failing.
    """
    roster = _load_roster()
    errors: list[str] = []
    budgets_sha = _trace_budgets_sha256()
    for trace in trace_dir.rglob("*.trace*"):
        # Infer slug from directory name two levels up: <results>/<slug>/<...>/trace
        slug = None
        for part in trace.parts[::-1]:
            if part in roster:
                slug = part
                break
        if slug is None:
            errors.append(f"slug_inference_failed: cannot identify row slug for {trace}")
            continue
        budget_bytes = roster[slug]["trace_budget_mib"] * 1024 * 1024
        size = trace.stat().st_size
        if size <= budget_bytes:
            continue
        sidecar = trace.parent / "trace_volume_exceeded.json"
        if not sidecar.exists():
            errors.append(f"BudgetViolationUntracked: {trace} exceeds budget ({size} > {budget_bytes}) with no trace_volume_exceeded.json sidecar")
            continue
        # Validate sidecar content
        try:
            with open(sidecar) as f:
                data = json.load(f)
        except Exception as e:
            errors.append(f"BudgetViolationUntracked: sidecar {sidecar} unreadable ({e})")
            continue
        if data.get("slug") != slug:
            errors.append(f"BudgetViolationUntracked: sidecar {sidecar} slug={data.get('slug')!r} != expected {slug!r}")
        if data.get("actual_trace_bytes") is None or data["actual_trace_bytes"] < size:
            errors.append(f"BudgetViolationUntracked: sidecar {sidecar} actual_trace_bytes={data.get('actual_trace_bytes')} does not account for {size}")
        if not (data.get("exceeded_trace_bytes") or data.get("exceeded_wall_time")):
            errors.append(f"BudgetViolationUntracked: sidecar {sidecar} has no exceeded_* flag set")
        if budgets_sha and data.get("trace_budgets_sha256") != budgets_sha:
            errors.append(f"BudgetViolationUntracked: sidecar {sidecar} trace_budgets_sha256 mismatch")
    errors.extend(_verify_slowdown_sidecars(trace_dir, budgets_sha))
    errors.extend(_verify_drop_rate_sidecars(trace_dir, budgets_sha))
    errors.extend(_verify_run_summaries_against_budgets(trace_dir))
    errors.extend(_verify_hrd_run_summaries(trace_dir))
    errors.extend(_verify_order_projections_run_summaries(trace_dir))
    errors.extend(_verify_divergence_proxies_run_summaries(trace_dir))
    errors.extend(_verify_sharing_locality_run_summaries(trace_dir))
    errors.extend(_verify_perturbation_run_ledger_run_summaries(trace_dir))
    errors.extend(_verify_trace_convergence_run_summaries(trace_dir))
    # Fresh-capture gate only (see verify_trace_dir docstring): the default
    # operator sweep over a host-local `results/` tree must not hard-fail
    # legacy/audit raw-only trees that other capture paths legitimately left
    # without a `run_suite --mode trace` binder.
    if require_canonical_trace_summary:
        errors.extend(_verify_canonical_trace_pb_admission(trace_dir))
    errors.extend(_verify_cache_predictor_run_summaries(trace_dir))
    return errors


def _inspect_recorded_attempt_generation(
    *,
    trace_record_path: Path,
    trace_record: dict[str, Any],
    descriptor: object,
    namespace: str,
    label: str,
    require_nonempty: bool,
    max_files: int,
    max_directories: int,
    max_total_file_bytes: int,
    max_total_relative_path_bytes: int,
    read_budget: _CanonicalAuthorityReadBudget | None = None,
    materialization_budget: (
        _CanonicalAuthorityMaterializationBudget | None
    ) = None,
    retained_input_bytes: int = 0,
    max_file_bytes: int | None = None,
    max_file_bytes_by_path: dict[str, int] | None = None,
) -> tuple[Any | None, str | None]:
    """Inspect one exact attempt-scoped generation named by an active record."""
    if type(descriptor) is not dict:
        return None, f"{label} descriptor is absent or not an object"
    schema_version = descriptor.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        return None, f"{label} schema_version must equal one"
    shape_point = trace_record.get("shape_point")
    attempt_id = trace_record.get("trace_capture_attempt_id")
    relative_value = descriptor.get("path")
    if type(relative_value) is not str:
        return None, f"{label} path must be a string"
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_value
        or any(part in {"", ".", ".."} for part in relative.parts)
        or len(relative.parts) != 3
        or relative.parts[0] != namespace
        or relative.parts[1] != shape_point
        or relative.parts[2] != attempt_id
        or descriptor.get("attempt_id") != attempt_id
    ):
        return None, f"{label} path is not the exact active attempt path"
    for field in ("content_manifest_sha256", "filesystem_generation_sha256"):
        value = descriptor.get(field)
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            return None, f"{label} {field} must be lowercase 64-hex"
    bounds = {
        "file_count": max_files,
        "directory_count": max_directories,
        "total_file_bytes": max_total_file_bytes,
    }
    for field, maximum in bounds.items():
        value = descriptor.get(field)
        minimum = (
            1
            if field == "directory_count"
            or (field == "file_count" and require_nonempty)
            else 0
        )
        if type(value) is not int or not minimum <= value <= maximum:
            return None, f"{label} {field} is invalid or exceeds its hard bound"
    generation_path = trace_record_path.parent.joinpath(*relative.parts)
    try:
        if read_budget is not None:
            read_budget.reserve_total(
                generation_path,
                descriptor["total_file_bytes"],
                role=f"{label} JSON authority",
            )
        observed = inspect_immutable_directory_generation(
            generation_path,
            require_nonempty=require_nonempty,
            max_files=descriptor["file_count"],
            max_directories=descriptor["directory_count"],
            max_total_file_bytes=descriptor["total_file_bytes"],
            max_total_relative_path_bytes=max_total_relative_path_bytes,
            max_file_bytes=max_file_bytes,
            max_file_bytes_by_path=max_file_bytes_by_path,
        )
        if materialization_budget is not None:
            materialization_budget.reserve(
                observed,
                retained_input_bytes=max(
                    retained_input_bytes,
                    read_budget.total_bytes if read_budget is not None else 0,
                ),
                role=f"inspected {label} identity",
            )
    except ImmutableArtifactResourceError as exc:
        return None, (
            f"{label} resource unavailable: {type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        return None, f"{label} is absent or invalid: {type(exc).__name__}: {exc}"
    if (
        observed.content_manifest_sha256
        != descriptor["content_manifest_sha256"]
        or observed.filesystem_generation_sha256
        != descriptor["filesystem_generation_sha256"]
        or observed.file_count != descriptor["file_count"]
        or observed.directory_count != descriptor["directory_count"]
        or observed.total_file_bytes != descriptor["total_file_bytes"]
    ):
        return None, f"{label} content or filesystem-generation seal is stale"
    return observed, None


def _admitted_launch_indices(
    admission: TraceAdmission,
) -> tuple[int, ...] | None:
    """Return the exact contiguous launch authority admitted from raw bytes."""
    if type(admission.launch_count) is not int or admission.launch_count <= 0:
        return None
    indices = tuple(launch.launch_index for launch in admission.launches)
    if any(type(index) is not int for index in indices):
        return None
    if indices != tuple(range(admission.launch_count)):
        return None
    return indices


def _expected_summary_body(
    trace_record: dict[str, Any],
    trace_admission: TraceAdmission,
    *,
    repeat_index: int,
    launch_index: int,
) -> dict[str, Any]:
    """Build the exact schema-v2 body authorized by raw launch bytes."""
    launch = trace_admission.launches[launch_index]
    return {
        "schema_version": 2,
        "slug": trace_record.get("slug"),
        "shape_point": trace_record.get("shape_point"),
        "repeat_index": repeat_index,
        "launch_index": launch_index,
        "kernel_name": launch.kernel_name,
        "kernel_signature_sha256": compute_kernel_signature_sha256(
            launch.kernel_signature_inputs()
        ),
        "inst_count": launch.inst_count,
        "binary_sha256": trace_record.get("binary_sha256"),
        "nvbit_trace_pb_path": trace_admission.dynamic_trace_path,
        "nvbit_trace_pb_sha256": trace_admission.dynamic_trace_sha256,
        "nvbit_tracer_build_hash": trace_record.get(
            "nvbit_tracer_build_hash"
        ),
        "nvbit_threadblock_payload_size_bytes": (
            trace_admission.threadblock_payload_size_bytes
        ),
        "nvbit_threadblock_payload_manifest_sha256": (
            trace_admission.threadblock_payload_manifest_sha256
        ),
        "trace_admission_sha256": trace_admission.trace_admission_sha256,
    }


def _verify_active_summary_generation(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    trace_admission: TraceAdmission,
    read_budget: _CanonicalAuthorityReadBudget | None = None,
    materialization_budget: (
        _CanonicalAuthorityMaterializationBudget | None
    ) = None,
) -> list[str]:
    errors: list[str] = []
    budget = read_budget or _CanonicalAuthorityReadBudget()
    material_budget = (
        materialization_budget
        or _CanonicalAuthorityMaterializationBudget()
    )
    admitted_launch_indices = _admitted_launch_indices(trace_admission)
    if admitted_launch_indices is None:
        return ["admitted raw launch authority is malformed"]
    descriptor = trace_record.get("nvbit_trace_summary_generation")
    if type(descriptor) is not dict:
        return ["summary generation descriptor is absent or not an object"]
    if set(descriptor) != _SUMMARY_GENERATION_DESCRIPTOR_FIELDS:
        return ["summary generation descriptor is not exact and closed"]
    schema_version = descriptor.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        return ["summary generation schema_version must equal one"]
    shape_point = trace_record.get("shape_point")
    admission_sha256 = trace_record.get("trace_admission_sha256")
    filesystem_sha256 = trace_record.get(
        "trace_filesystem_generation_sha256"
    )
    relative_value = descriptor.get("path")
    expected = PurePosixPath(
        _SUMMARY_GENERATION_NAMESPACE,
        str(shape_point),
        str(admission_sha256),
        str(filesystem_sha256),
    )
    if (
        type(relative_value) is not str
        or PurePosixPath(relative_value) != expected
        or expected.as_posix() != relative_value
    ):
        return ["summary generation path is not the exact admitted generation"]
    for field in ("content_manifest_sha256", "filesystem_generation_sha256"):
        value = descriptor.get(field)
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            errors.append(f"summary generation {field} is absent or invalid")
    file_count = descriptor.get("file_count")
    directory_count = descriptor.get("directory_count")
    total_file_bytes = descriptor.get("total_file_bytes")
    if type(file_count) is not int or not 1 <= file_count <= 4096:
        errors.append("summary generation file_count is invalid")
    if type(directory_count) is not int or directory_count != 1:
        errors.append("summary generation directory_count must equal one")
    if (
        type(total_file_bytes) is not int
        or not 0 <= total_file_bytes <= 256 * 1024 * 1024
    ):
        errors.append("summary generation total_file_bytes is invalid")
    if errors:
        return errors

    generation_path = trace_record_path.parent.joinpath(*expected.parts)
    try:
        budget.reserve_total(
            generation_path,
            total_file_bytes,
            role="summary generation JSON authority",
        )
        observed = inspect_immutable_directory_generation(
            generation_path,
            require_nonempty=True,
            max_files=file_count,
            max_directories=directory_count,
            max_total_file_bytes=total_file_bytes,
            max_total_relative_path_bytes=4 * 1024 * 1024,
            max_file_bytes=_CANONICAL_AUTHORITY_JSON_MAX_BYTES,
        )
        material_budget.reserve(
            observed,
            retained_input_bytes=budget.total_bytes,
            role="inspected summary generation identity",
        )
    except ImmutableArtifactResourceError as exc:
        return [
            "summary generation resource unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]
    except (ImmutableArtifactError, OSError) as exc:
        return [
            "summary generation is absent or invalid: "
            f"{type(exc).__name__}: {exc}"
        ]
    if (
        observed.content_manifest_sha256
        != descriptor["content_manifest_sha256"]
        or observed.filesystem_generation_sha256
        != descriptor["filesystem_generation_sha256"]
        or observed.file_count != file_count
        or observed.directory_count != directory_count
        or observed.total_file_bytes != total_file_bytes
    ):
        return ["summary generation content or filesystem seal is stale"]

    observed_pairs: set[tuple[int, int]] = set()
    for identity in observed.files:
        match = _SUMMARY_FILENAME_RE.fullmatch(identity.relative_path)
        if match is None:
            errors.append(
                "summary generation contains noncanonical file "
                f"{identity.relative_path}"
            )
            continue
        repeat_text, launch_text = match.groups()
        repeat_index = int(repeat_text)
        launch_index = int(launch_text)
        if (
            str(repeat_index) != repeat_text
            or str(launch_index) != launch_text
        ):
            errors.append(
                "summary generation contains noncanonical numeric filename "
                f"{identity.relative_path}"
            )
            continue
        observed_pairs.add((repeat_index, launch_index))
    expected_pairs = {
        (0, launch_index) for launch_index in admitted_launch_indices
    }
    if observed_pairs != expected_pairs or observed.file_count != len(
        expected_pairs
    ):
        errors.append(
            "summary generation does not equal the admitted raw launch authority"
        )

    expected_identities = [
        {
            "path": (expected / identity.relative_path).as_posix(),
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }
        for identity in observed.files
    ]
    if descriptor.get("files") != expected_identities:
        errors.append("summary generation file identities are stale or incomplete")
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            expected_identities,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if descriptor.get("manifest_sha256") != manifest_sha256:
        errors.append("summary generation manifest_sha256 is stale")
    references = trace_record.get("nvbit_trace_summary_paths")
    if references != [identity["path"] for identity in expected_identities]:
        errors.append("active summary path list is stale or incomplete")

    identities = {identity.relative_path: identity for identity in observed.files}
    material_budget.reserve(
        (observed_pairs, expected_pairs, expected_identities, identities),
        retained_input_bytes=budget.total_bytes,
        role="retained summary generation projections",
    )
    try:
        with (
            _load_canonical_schema_validator(
                SCHEMAS_DIR / "nvbit_trace_summary.schema.json",
                label="NVBit trace summary schema",
                read_budget=budget,
                materialization_budget=material_budget,
            ) as summary_validator,
            PinnedDirectoryReader(generation_path) as reader,
        ):
            snapshot = reader.snapshot_tree(
                require_readonly_files=True,
                require_readonly_directories=True,
                max_files=file_count,
                max_directories=directory_count,
                max_total_file_bytes=total_file_bytes,
                max_total_relative_path_bytes=4 * 1024 * 1024,
            )
            files = dict(snapshot.files)
            directories = dict(snapshot.directories)
            material_budget.reserve(
                (snapshot, files, directories),
                retained_input_bytes=budget.total_bytes,
                role="retained summary generation inventory",
            )
            if set(files) != set(identities) or set(directories) != {"."}:
                errors.append(
                    "summary generation inventory changed before read"
                )
            for relative_path, metadata in sorted(files.items()):
                match = _SUMMARY_FILENAME_RE.fullmatch(relative_path)
                if match is None:
                    errors.append(
                        "summary generation contains noncanonical file "
                        f"{relative_path}"
                    )
                    continue
                payload = reader.read_snapshot(
                    relative_path,
                    max_bytes=_CANONICAL_AUTHORITY_JSON_MAX_BYTES,
                    expected_metadata=metadata,
                ).payload
                identity = identities.get(relative_path)
                if identity is None or (
                    len(payload) != identity.size_bytes
                    or hashlib.sha256(payload).hexdigest() != identity.sha256
                ):
                    errors.append(
                        "summary generation file "
                        f"{relative_path} disagrees with its immutable seal"
                    )
                    continue
                try:
                    summary = _decode_authority_json(
                        payload,
                        materialization_budget=material_budget,
                        retained_input_bytes=budget.total_bytes,
                        role="NVBit trace summary JSON authority",
                    )
                except MemoryError as exc:
                    errors.append(
                        "summary generation resource unavailable during JSON "
                        f"decode for {relative_path}: {type(exc).__name__}: {exc}"
                    )
                    continue
                except (UnicodeDecodeError, ValueError) as exc:
                    errors.append(
                        "summary generation file "
                        f"{relative_path} is invalid JSON: {exc}"
                    )
                    continue
                schema_errors = list(summary_validator.iter_errors(summary))
                if schema_errors:
                    first = schema_errors[0]
                    errors.append(
                        "summary generation file "
                        f"{relative_path} fails schema validation at "
                        f"{list(first.absolute_path)}: {first.message}"
                    )
                    continue
                repeat_index = int(match.group(1))
                launch_index = int(match.group(2))
                if launch_index not in admitted_launch_indices:
                    errors.append(
                        f"summary generation file {relative_path} "
                        "names an unadmitted launch"
                    )
                    continue
                expected_summary = _expected_summary_body(
                    trace_record,
                    trace_admission,
                    repeat_index=repeat_index,
                    launch_index=launch_index,
                )
                if (
                    type(summary) is not dict
                    or not _exact_json_equal(summary, expected_summary)
                ):
                    errors.append(
                        f"summary generation file {relative_path} "
                        "disagrees with admitted raw launch and execution authority"
                    )
            reader.verify_path()
    except ImmutableArtifactResourceError as exc:
        errors.append(
            "summary generation resource unavailable during descriptor read: "
            f"{type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        errors.append(
            "summary generation changed or became invalid during descriptor read: "
            f"{type(exc).__name__}: {exc}"
        )
    return errors


def _verify_active_phase_generation(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    trace_admission: TraceAdmission,
    read_budget: _CanonicalAuthorityReadBudget | None = None,
    materialization_budget: (
        _CanonicalAuthorityMaterializationBudget | None
    ) = None,
) -> list[str]:
    errors: list[str] = []
    budget = read_budget or _CanonicalAuthorityReadBudget()
    material_budget = (
        materialization_budget
        or _CanonicalAuthorityMaterializationBudget()
    )
    phase = trace_record.get("phase_inject_ab")
    if phase is None:
        phase_namespace = (
            trace_record_path.parent / _PHASE_LEDGER_GENERATION_NAMESPACE
        )
        try:
            phase_namespace.lstat()
        except FileNotFoundError:
            return errors
        except OSError as exc:
            return [
                "phase generation namespace cannot be authenticated: "
                f"{type(exc).__name__}: {exc}"
            ]
        return [
            "phase generation authority exists but phase_inject_ab "
            "authority is absent"
        ]

    if type(phase) is not dict or type(phase.get("requested")) is not bool:
        return ["phase_inject_ab authority is malformed"]
    repeat_count = phase.get("repeat_count")
    descriptor = phase.get("generation")
    evidence = phase.get("evidence")
    if not phase["requested"]:
        if repeat_count is not None and (
            type(repeat_count) is not int or repeat_count != 0
        ):
            errors.append("unrequested phase evidence has an invalid repeat_count")
        if descriptor is not None:
            errors.append("unrequested phase evidence carries active authority")
        if evidence is not None and (
            type(evidence) is not list or evidence
        ):
            errors.append("unrequested phase evidence carries active authority")
        return errors

    if type(repeat_count) is not int or repeat_count <= 0:
        return ["requested phase evidence has no positive repeat_count"]
    if type(descriptor) is not dict:
        return ["requested phase evidence has no generation descriptor"]
    if set(descriptor) != _DIRECTORY_GENERATION_DESCRIPTOR_FIELDS:
        return ["phase generation descriptor is not exact and closed"]
    schema_version = descriptor.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        return ["phase generation schema_version must equal one"]
    admitted_launch_indices = _admitted_launch_indices(trace_admission)
    if admitted_launch_indices is None:
        return ["admitted raw launch authority is malformed"]
    expected_file_count = repeat_count * len(admitted_launch_indices)
    if expected_file_count > 4096:
        return [
            "requested phase repeat-by-launch authority exceeds the hard "
            "phase generation file bound"
        ]
    phase_observed, phase_error = _inspect_recorded_attempt_generation(
        trace_record_path=trace_record_path,
        trace_record=trace_record,
        descriptor=descriptor,
        namespace=_PHASE_LEDGER_GENERATION_NAMESPACE,
        label="phase ledger generation",
        require_nonempty=True,
        max_files=4096,
        max_directories=1,
        max_total_file_bytes=256 * 1024 * 1024,
        max_total_relative_path_bytes=4 * 1024 * 1024,
        read_budget=budget,
        materialization_budget=material_budget,
        retained_input_bytes=budget.total_bytes,
        max_file_bytes=_CANONICAL_AUTHORITY_JSON_MAX_BYTES,
    )
    if phase_error is not None:
        return [phase_error]
    assert phase_observed is not None and type(descriptor) is dict

    if phase_observed.file_count != expected_file_count:
        errors.append(
            "phase ledger generation file count does not equal the admitted "
            "repeat-by-launch authority"
        )
    expected_phase = [
        {
            "path": (
                PurePosixPath(descriptor["path"]) / identity.relative_path
            ).as_posix(),
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }
        for identity in phase_observed.files
    ]
    if evidence != expected_phase:
        errors.append("phase ledger evidence list is stale or incomplete")

    generation_path = trace_record_path.parent.joinpath(
        *PurePosixPath(descriptor["path"]).parts
    )
    identities = {
        identity.relative_path: identity for identity in phase_observed.files
    }
    observed_pairs: set[tuple[int, int]] = set()
    material_budget.reserve(
        (expected_phase, identities),
        retained_input_bytes=budget.total_bytes,
        role="retained phase generation projections",
    )
    try:
        with PinnedDirectoryReader(generation_path) as reader:
            snapshot = reader.snapshot_tree(
                require_readonly_files=True,
                require_readonly_directories=True,
                max_files=descriptor["file_count"],
                max_directories=descriptor["directory_count"],
                max_total_file_bytes=descriptor["total_file_bytes"],
                max_total_relative_path_bytes=4 * 1024 * 1024,
            )
            files = dict(snapshot.files)
            directories = dict(snapshot.directories)
            material_budget.reserve(
                (snapshot, files, directories),
                retained_input_bytes=budget.total_bytes,
                role="retained phase generation inventory",
            )
            if set(files) != set(identities) or set(directories) != {"."}:
                errors.append(
                    "phase ledger generation inventory changed before read"
                )
            else:
                for relative_path, identity in sorted(identities.items()):
                    match = _PHASE_LEDGER_FILENAME_RE.fullmatch(relative_path)
                    if match is None:
                        errors.append(
                            "phase ledger generation contains noncanonical file "
                            f"{relative_path}"
                        )
                        continue
                    repeat_text, launch_text = match.groups()
                    repeat_index = int(repeat_text)
                    launch_index = int(launch_text)
                    if (
                        str(repeat_index) != repeat_text
                        or str(launch_index) != launch_text
                    ):
                        errors.append(
                            "phase ledger generation contains noncanonical numeric "
                            f"filename {relative_path}"
                        )
                        continue
                    payload = reader.read_snapshot(
                        relative_path,
                        max_bytes=_CANONICAL_AUTHORITY_JSON_MAX_BYTES,
                        expected_metadata=files[relative_path],
                    ).payload
                    if (
                        len(payload) != identity.size_bytes
                        or hashlib.sha256(payload).hexdigest() != identity.sha256
                    ):
                        errors.append(
                            "phase ledger generation file "
                            f"{relative_path} disagrees with its immutable seal"
                        )
                        continue
                    try:
                        ledger = _decode_authority_json(
                            payload,
                            materialization_budget=material_budget,
                            retained_input_bytes=budget.total_bytes,
                            role="phase ledger JSON authority",
                        )
                    except MemoryError as exc:
                        errors.append(
                            "phase ledger generation resource unavailable during "
                            f"JSON decode for {relative_path}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                    except (UnicodeDecodeError, ValueError) as exc:
                        errors.append(
                            "phase ledger generation file "
                            f"{relative_path} is invalid JSON: {exc}"
                        )
                        continue
                    if type(ledger) is not dict:
                        errors.append(
                            f"phase ledger generation file {relative_path} "
                            "is not an object"
                        )
                        continue
                    try:
                        serialize_phase_cycle_ledger(ledger)
                    except (PhaseLedgerSchemaInvalid, TypeError, ValueError) as exc:
                        errors.append(
                            "phase ledger generation file "
                            f"{relative_path} fails schema validation: {exc}"
                        )
                        continue
                    try:
                        admit_phase_cycle_ledger(ledger)
                    except (
                        NvbitPhaseInjectionOverheadExceeded,
                        PhaseBoundaryMappingUnresolved,
                    ) as exc:
                        errors.append(
                            "phase ledger generation file "
                            f"{relative_path} is semantically inadmissible: "
                            f"{exc.TOKEN}: {exc}"
                        )
                        continue
                    if launch_index not in admitted_launch_indices:
                        errors.append(
                            "phase ledger generation file "
                            f"{relative_path} names an unadmitted launch"
                        )
                        continue
                    if (
                        ledger.get("slug") != trace_record.get("slug")
                        or ledger.get("shape_point")
                        != trace_record.get("shape_point")
                        or type(ledger.get("repeat_index")) is not int
                        or ledger.get("repeat_index") != repeat_index
                        or type(ledger.get("launch_index")) is not int
                        or ledger.get("launch_index") != launch_index
                        or ledger.get("kernel_signature_sha256")
                        != trace_record.get("kernel_signature_sha256")
                        or ledger.get("binary_sha256")
                        != trace_record.get("binary_sha256")
                        or ledger.get("nvbit_tracer_build_hash")
                        != trace_record.get("nvbit_tracer_build_hash")
                    ):
                        errors.append(
                            "phase ledger generation file "
                            f"{relative_path} disagrees with admitted raw launch "
                            "and execution authority"
                        )
                        continue
                    observed_pairs.add((repeat_index, launch_index))
            reader.verify_path()
    except ImmutableArtifactResourceError as exc:
        errors.append(
            "phase ledger generation resource unavailable during authenticated "
            f"read: {type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        errors.append(
            "phase ledger generation changed or became invalid during "
            f"authenticated read: {type(exc).__name__}: {exc}"
        )

    expected_pairs = {
        (repeat_index, launch_index)
        for repeat_index in range(repeat_count)
        for launch_index in admitted_launch_indices
    }
    material_budget.reserve(
        (observed_pairs, expected_pairs),
        retained_input_bytes=budget.total_bytes,
        role="retained phase repeat-by-launch projections",
    )
    if observed_pairs != expected_pairs:
        errors.append(
            "phase ledger generation does not equal the admitted "
            "repeat-by-launch authority"
        )
    return errors


@contextmanager
def _load_canonical_schema_validator(
    path: Path,
    *,
    label: str,
    read_budget: _CanonicalAuthorityReadBudget | None = None,
    materialization_budget: (
        _CanonicalAuthorityMaterializationBudget | None
    ) = None,
) -> Iterator[jsonschema.Draft202012Validator]:
    """Retain one bounded schema capability through every validator use."""
    if (read_budget is None) != (materialization_budget is None):
        raise TypeError("schema read and materialization budgets must be paired")
    if read_budget is None:
        read_budget = _CanonicalAuthorityReadBudget()
        materialization_budget = _CanonicalAuthorityMaterializationBudget()
    assert materialization_budget is not None
    with ExitStack() as retained_schema:
        reader = retained_schema.enter_context(PinnedDirectoryReader(path.parent))
        monitor = retained_schema.enter_context(
            PinnedPathSetMonitor.from_reader(
                reader,
                regular_paths=(path.name,),
                recursive_directories=(),
            )
        )
        monitor.verify_unchanged()
        snapshot = reader.read_snapshot(
            path.name,
            max_bytes=_CANONICAL_AUTHORITY_JSON_MAX_BYTES,
        )
        read_budget.reserve_file(
            path,
            snapshot.metadata[4],
            role=label,
        )
        monitor.verify_unchanged()
        schema = _decode_authority_json(
            snapshot.payload,
            materialization_budget=materialization_budget,
            retained_input_bytes=read_budget.total_bytes,
            role=label,
            label=label,
        )
        if type(schema) is not dict:
            raise ImmutableArtifactError(f"{label} root is not an object")
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise ImmutableArtifactError(
                f"{label} authority is invalid: {exc.message}"
            ) from exc
        validator = jsonschema.Draft202012Validator(
            schema,
            format_checker=jsonschema.FormatChecker(),
        )
        monitor.verify_unchanged()
        try:
            yield validator
        finally:
            monitor.verify_unchanged()


@contextmanager
def _load_measurement_manifest_validator(
    *,
    read_budget: _CanonicalAuthorityReadBudget | None = None,
    materialization_budget: (
        _CanonicalAuthorityMaterializationBudget | None
    ) = None,
) -> Iterator[jsonschema.Draft202012Validator]:
    with _load_canonical_schema_validator(
        _MEASUREMENT_MANIFEST_SCHEMA_PATH,
        label="measurement manifest schema",
        read_budget=read_budget,
        materialization_budget=materialization_budget,
    ) as validator:
        yield validator


def _verify_active_measurement_manifest(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    authority_root: Path,
    authority_reader: PinnedDirectoryReader,
    authority_file_inventory: dict[
        str, tuple[int, int, int, int, int, int, int]
    ],
    expected_platform: str | None,
    read_budget: _CanonicalAuthorityReadBudget,
    materialization_budget: _CanonicalAuthorityMaterializationBudget,
) -> list[str]:
    """Apply the same closed v2 execution-manifest contract as merge."""
    descriptor = trace_record.get("measurement_manifest")
    if type(descriptor) is not dict or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        return ["measurement manifest descriptor is not exact and closed"]
    shape_point = trace_record.get("shape_point")
    expected_name = f"measurement_manifest_{shape_point}.json"
    if descriptor.get("path") != expected_name:
        return [
            "measurement manifest path does not name the exact active shape authority"
        ]
    sha256 = descriptor.get("sha256")
    size_bytes = descriptor.get("size_bytes")
    if (
        type(sha256) is not str
        or _SHA256_RE.fullmatch(sha256) is None
        or type(size_bytes) is not int
        or size_bytes < 0
    ):
        return ["measurement manifest descriptor identity is invalid"]

    try:
        record_relative = trace_record_path.relative_to(authority_root)
        manifest_relative = (
            PurePosixPath(record_relative.as_posix()).parent / expected_name
        ).as_posix()
        metadata = authority_file_inventory.get(manifest_relative)
        if metadata is None:
            raise ImmutableArtifactError(
                "measurement manifest is absent from the retained authority inventory"
            )
        if stat.S_IMODE(metadata[2]) != 0o444 or metadata[3] != 1:
            return [
                "measurement manifest is not a single-link 0444 committed file"
            ]
        read_budget.reserve_file(
            authority_root.joinpath(*PurePosixPath(manifest_relative).parts),
            metadata[4],
            role="measurement manifest JSON authority",
        )
        payload = authority_reader.read_snapshot(
            manifest_relative,
            max_bytes=_CANONICAL_AUTHORITY_JSON_MAX_BYTES,
            expected_metadata=metadata,
        ).payload
        if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != sha256:
            return ["measurement manifest descriptor is stale"]
        body = _decode_authority_json(
            payload,
            materialization_budget=materialization_budget,
            retained_input_bytes=read_budget.total_bytes,
            role="measurement manifest JSON authority",
            label="measurement manifest",
        )
        if type(body) is not dict:
            return ["measurement manifest root is not an object"]
        if (
            type(body.get("schema_version")) is not int
            or body["schema_version"] != 2
        ):
            return [
                "measurement manifest fails closed v2 schema: schema_version "
                "must be the exact integer 2"
            ]
        if not _is_rfc3339_datetime(body.get("timestamp_utc")):
            return [
                "measurement manifest fails closed v2 schema: timestamp_utc "
                "is not an RFC-3339 date-time"
            ]
        with _load_measurement_manifest_validator(
            read_budget=read_budget,
            materialization_budget=materialization_budget,
        ) as validator:
            validation_error = next(validator.iter_errors(body), None)
        if validation_error is not None:
            return [
                "measurement manifest fails closed v2 schema: "
                f"{validation_error.message}"
            ]
        expected = {
            "schema_version": 2,
            "mode": "trace",
            "slug": trace_record.get("slug"),
            "shape_point": shape_point,
            "binary_sha256": trace_record.get("binary_sha256"),
        }
        if expected_platform is not None:
            expected["gpu_sku"] = expected_platform
        for field, expected_value in expected.items():
            if body.get(field) != expected_value:
                return [
                    f"measurement manifest field {field!r} disagrees with "
                    "active trace execution authority"
                ]
        manifest_signature = body.get("kernel_signature_sha256")
        if (
            manifest_signature is not None
            and manifest_signature != trace_record.get("kernel_signature_sha256")
        ):
            return [
                "measurement manifest field 'kernel_signature_sha256' "
                "disagrees with active trace execution authority"
            ]
    except (ImmutableArtifactResourceError, MemoryError) as exc:
        return [
            "measurement manifest resource unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]
    except ImmutableArtifactLimitError as exc:
        return [
            "measurement manifest exceeds the bounded authority JSON limit: "
            f"{type(exc).__name__}: {exc}"
        ]
    except OSError as exc:
        if exc.errno in {errno.EMFILE, errno.ENFILE}:
            return [
                "measurement manifest resource unavailable: "
                f"{type(exc).__name__}: {exc}"
            ]
        return [
            "measurement manifest is absent or invalid: "
            f"{type(exc).__name__}: {exc}"
        ]
    except (
        ImmutableArtifactError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        return [
            "measurement manifest is absent or invalid: "
            f"{type(exc).__name__}: {exc}"
        ]
    return []


def _platform_authority_from_record_path(
    trace_record_path: Path,
) -> tuple[str | None, str | None]:
    """Derive one canonical platform key from the active record's path."""
    matches = sorted({
        part
        for part in trace_record_path.parent.parts
        if part in _CANONICAL_PLATFORM_KEYS
    })
    if len(matches) > 1:
        return None, (
            "active trace record path contains multiple canonical platform "
            f"authorities: {matches}"
        )
    return (matches[0], None) if matches else (None, None)


def _verify_active_attempt_event(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    authority_root: Path,
    authority_reader: PinnedDirectoryReader,
    authority_file_inventory: dict[
        str, tuple[int, int, int, int, int, int, int]
    ],
    attempt_namespace_present: bool,
    read_budget: _CanonicalAuthorityReadBudget,
    materialization_budget: _CanonicalAuthorityMaterializationBudget,
) -> list[str]:
    """Authenticate the exact activation-ready event named by the record."""
    descriptor = trace_record.get("trace_capture_attempt_event")
    if descriptor is None:
        if attempt_namespace_present or trace_record.get(
            "trace_derived_generation"
        ) is not None:
            return [
                "trace-attempt authority exists but the active trace record has "
                "no trace_capture_attempt_event reference"
            ]
        return []
    if type(descriptor) is not dict or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        return ["trace capture attempt event descriptor is not exact and closed"]
    shape_point = trace_record.get("shape_point")
    attempt_id = trace_record.get("trace_capture_attempt_id")
    expected_relative = PurePosixPath(
        _TRACE_CAPTURE_ATTEMPT_NAMESPACE,
        str(shape_point),
        str(attempt_id),
        "activation_ready.json",
    )
    relative_value = descriptor.get("path")
    sha256 = descriptor.get("sha256")
    size_bytes = descriptor.get("size_bytes")
    if (
        type(relative_value) is not str
        or PurePosixPath(relative_value) != expected_relative
        or expected_relative.as_posix() != relative_value
    ):
        return ["trace capture attempt event path is not activation_ready"]
    if (
        type(sha256) is not str
        or _SHA256_RE.fullmatch(sha256) is None
        or type(size_bytes) is not int
        or not 0 <= size_bytes <= _CANONICAL_AUTHORITY_JSON_MAX_BYTES
    ):
        return ["trace capture attempt event descriptor identity is invalid"]
    try:
        record_relative = trace_record_path.relative_to(authority_root)
        event_relative = (
            PurePosixPath(record_relative.as_posix()).parent / expected_relative
        ).as_posix()
        metadata = authority_file_inventory.get(event_relative)
        if metadata is None:
            raise ImmutableArtifactError(
                "trace capture attempt event is absent from retained inventory"
            )
        if stat.S_IMODE(metadata[2]) != 0o444 or metadata[3] != 1:
            return [
                "trace capture attempt event is not a single-link 0444 file"
            ]
        read_budget.reserve_file(
            authority_root.joinpath(*PurePosixPath(event_relative).parts),
            metadata[4],
            role="trace capture attempt event JSON authority",
        )
        payload = authority_reader.read_snapshot(
            event_relative,
            max_bytes=_CANONICAL_AUTHORITY_JSON_MAX_BYTES,
            expected_metadata=metadata,
        ).payload
        if (
            len(payload) != size_bytes
            or hashlib.sha256(payload).hexdigest() != sha256
        ):
            return ["trace capture attempt event descriptor is stale"]
        event = _decode_authority_json(
            payload,
            materialization_budget=materialization_budget,
            retained_input_bytes=read_budget.total_bytes,
            role="trace capture attempt event JSON authority",
            label="trace capture attempt event",
        )
        phase = trace_record.get("phase_inject_ab")
        phase_generation = (
            phase.get("generation") if type(phase) is dict else None
        )
        expected_event = {
            "schema_version": 1,
            "slug": trace_record.get("slug"),
            "shape_point": shape_point,
            "attempt_id": attempt_id,
            "status": "activation_ready",
            "typed_reason": None,
            "activation_state": "non_active",
            "raw_trace_generation": trace_record.get(
                "raw_trace_generation"
            ),
            "raw_staging_path": None,
            "raw_unresolved_committed_path": None,
            "trace_derived_generation": trace_record.get(
                "trace_derived_generation"
            ),
            "derived_staging_path": None,
            "phase_cycle_ledger_generation": phase_generation,
            "phase_ledger_staging_path": None,
            "nvbit_trace_summary_generation": trace_record.get(
                "nvbit_trace_summary_generation"
            ),
            "measurement_manifest": trace_record.get(
                "measurement_manifest"
            ),
        }
        if not _exact_json_equal(event, expected_event):
            return [
                "trace capture attempt event body disagrees with active "
                "generation authority"
            ]
        authority_reader.verify_path()
    except (ImmutableArtifactResourceError, MemoryError) as exc:
        return [
            "trace capture attempt event resource unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]
    except ImmutableArtifactLimitError as exc:
        return [
            "trace capture attempt event exceeds the bounded authority JSON "
            f"limit: {type(exc).__name__}: {exc}"
        ]
    except (
        ImmutableArtifactError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        return [
            "trace capture attempt event is absent or invalid: "
            f"{type(exc).__name__}: {exc}"
        ]
    return []


def _verify_active_generation_record(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    authority_root: Path,
    authority_reader: PinnedDirectoryReader,
    authority_file_inventory: dict[
        str, tuple[int, int, int, int, int, int, int]
    ],
    authority_directory_inventory: dict[
        str, tuple[int, int, int, int, int, int, int]
    ],
    read_budget: _CanonicalAuthorityReadBudget,
    materialization_budget: _CanonicalAuthorityMaterializationBudget,
) -> list[str]:
    """Freshly authenticate one active raw, derived, summary, and phase set."""
    errors: list[str] = []
    shape_point = trace_record.get("shape_point")
    attempt_id = trace_record.get("trace_capture_attempt_id")
    if (
        type(shape_point) is not str
        or not shape_point
        or type(attempt_id) is not str
        or not attempt_id
        or trace_record.get("activation_state") != "active"
    ):
        return ["canonical trace record lacks active shape/attempt authority"]

    try:
        record_relative = trace_record_path.relative_to(authority_root)
    except ValueError:
        return ["canonical trace record escapes retained authority root"]
    artifact_relative = PurePosixPath(record_relative.as_posix()).parent

    def namespace_present(namespace: str) -> bool:
        relative = (artifact_relative / namespace).as_posix()
        return (
            relative in authority_file_inventory
            or relative in authority_directory_inventory
        )

    expected_platform, platform_error = _platform_authority_from_record_path(
        trace_record_path
    )
    if platform_error is not None:
        return [platform_error]

    limits = DEFAULT_TRACE_ADMISSION_LIMITS
    raw_limits = raw_trace_generation_limits(limits)
    raw_descriptor = trace_record.get("raw_trace_generation")
    if type(raw_descriptor) is not dict:
        return ["raw trace generation descriptor is absent or not an object"]
    if set(raw_descriptor) != _RAW_GENERATION_DESCRIPTOR_FIELDS:
        return ["raw trace generation descriptor is not exact and closed"]
    raw_schema_version = raw_descriptor.get("schema_version")
    if type(raw_schema_version) is not int or raw_schema_version != 1:
        return ["raw trace generation schema_version must equal one"]
    raw_observed, raw_error = _inspect_recorded_attempt_generation(
        trace_record_path=trace_record_path,
        trace_record=trace_record,
        descriptor=raw_descriptor,
        namespace=_RAW_TRACE_GENERATION_NAMESPACE,
        label="raw trace generation",
        require_nonempty=True,
        max_files=raw_limits["max_files"],
        max_directories=raw_limits["max_directories"],
        max_total_file_bytes=raw_limits["max_total_file_bytes"],
        max_total_relative_path_bytes=raw_limits[
            "max_total_relative_path_bytes"
        ],
        materialization_budget=materialization_budget,
        retained_input_bytes=read_budget.total_bytes,
        max_file_bytes=raw_limits["max_file_bytes"],
        max_file_bytes_by_path=raw_limits["max_file_bytes_by_path"],
    )
    if raw_error is not None:
        errors.append(raw_error)
        return errors
    assert type(raw_descriptor) is dict and raw_observed is not None
    for field in (
        "trace_admission_sha256",
        "trace_filesystem_generation_sha256",
        "dynamic_trace_sha256",
    ):
        value = raw_descriptor.get(field)
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            errors.append(f"raw trace generation {field} is absent or invalid")
        if trace_record.get(field) != value:
            errors.append(f"top-level {field} disagrees with raw generation")
    expected_trace_pb = (
        PurePosixPath(str(raw_descriptor["path"])) / "dynamic_trace.pb"
    ).as_posix()
    if trace_record.get("trace_pb_path") != expected_trace_pb:
        errors.append("trace_pb_path does not name the exact raw generation")
    raw_path = trace_record_path.parent.joinpath(
        *PurePosixPath(str(raw_descriptor["path"])).parts
    )
    admitted: TraceAdmission | None = None
    try:
        admitted = require_trace_admission(raw_path)
    except TraceAdmissionError as exc:
        if exc.typed_reason == "trace_snapshot_resource_unavailable":
            errors.append(
                "raw trace generation admission resource unavailable: "
                f"{exc.typed_reason}: {exc.detail}"
            )
        else:
            errors.append(
                "raw trace generation fresh admission failed: "
                f"{exc.typed_reason}: {exc.detail}"
            )
    except ImmutableArtifactResourceError as exc:
        errors.append(
            "raw trace generation admission resource unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        errors.append(
            "raw trace generation fresh admission failed: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        if (
            admitted.trace_admission_sha256
            != raw_descriptor.get("trace_admission_sha256")
            or admitted.trace_filesystem_generation_sha256
            != raw_descriptor.get("trace_filesystem_generation_sha256")
            or admitted.dynamic_trace_sha256
            != raw_descriptor.get("dynamic_trace_sha256")
        ):
            errors.append("raw trace admission or filesystem seals are stale")
    if errors or admitted is None:
        return errors

    derived_descriptor = trace_record.get("trace_derived_generation")
    if derived_descriptor is None:
        if namespace_present(
            _DERIVED_TRACE_GENERATION_NAMESPACE
        ) or trace_record.get("trace_capture_attempt_event") is not None:
            errors.append(
                "derived generation authority exists but the active trace record "
                "has no trace_derived_generation reference"
            )
    elif type(derived_descriptor) is not dict or set(derived_descriptor) != {
        "schema_version",
        "attempt_id",
        "path",
        "content_manifest_sha256",
        "filesystem_generation_sha256",
        "file_count",
        "directory_count",
        "total_file_bytes",
    }:
        errors.append(
            "trace_derived_generation descriptor is not exact and closed"
        )
    elif (
        type(derived_descriptor.get("schema_version")) is not int
        or derived_descriptor.get("schema_version") != 1
    ):
        errors.append("trace_derived_generation schema_version must equal one")
    else:
        _derived_observed, derived_error = _inspect_recorded_attempt_generation(
            trace_record_path=trace_record_path,
            trace_record=trace_record,
            descriptor=derived_descriptor,
            namespace=_DERIVED_TRACE_GENERATION_NAMESPACE,
            label="derived trace generation",
            require_nonempty=False,
            max_files=limits.tree_files,
            max_directories=limits.tree_directories,
            max_total_file_bytes=raw_limits["max_total_file_bytes"],
            max_total_relative_path_bytes=limits.tree_relative_path_bytes,
            materialization_budget=materialization_budget,
            retained_input_bytes=read_budget.total_bytes,
        )
        if derived_error is not None:
            errors.append(derived_error)
    if errors:
        return errors

    errors.extend(
        _verify_active_summary_generation(
            trace_record_path,
            trace_record,
            trace_admission=admitted,
            read_budget=read_budget,
            materialization_budget=materialization_budget,
        )
    )

    errors.extend(
        _verify_active_measurement_manifest(
            trace_record_path,
            trace_record,
            authority_root=authority_root,
            authority_reader=authority_reader,
            authority_file_inventory=authority_file_inventory,
            expected_platform=expected_platform,
            read_budget=read_budget,
            materialization_budget=materialization_budget,
        )
    )

    errors.extend(
        _verify_active_phase_generation(
            trace_record_path,
            trace_record,
            trace_admission=admitted,
            read_budget=read_budget,
            materialization_budget=materialization_budget,
        )
    )
    errors.extend(
        _verify_active_attempt_event(
            trace_record_path,
            trace_record,
            authority_root=authority_root,
            authority_reader=authority_reader,
            authority_file_inventory=authority_file_inventory,
            attempt_namespace_present=namespace_present(
                _TRACE_CAPTURE_ATTEMPT_NAMESPACE
            ),
            read_budget=read_budget,
            materialization_budget=materialization_budget,
        )
    )
    return errors


def _snapshot_relative_path(root: Path, relative_path: str) -> Path:
    """Render one descriptor-inventoried path without resolving symlinks."""
    relative = PurePosixPath(relative_path)
    return root.joinpath(*relative.parts)


def _snapshot_join_relative(parent: str, *parts: str) -> str:
    base = PurePosixPath(parent)
    return (base.joinpath(*parts)).as_posix()


def _snapshot_direct_child_names(
    snapshot: PinnedTreeSnapshot,
    parent: str,
) -> set[str]:
    """Return direct children of one inventoried directory."""
    names: set[str] = set()
    for relative_path, _metadata in (*snapshot.files, *snapshot.directories):
        if relative_path == ".":
            continue
        relative = PurePosixPath(relative_path)
        if relative.parent.as_posix() == parent:
            names.add(relative.name)
    return names


def _decode_authority_json(
    payload: bytes,
    *,
    materialization_budget: (
        _CanonicalAuthorityMaterializationBudget | None
    ) = None,
    retained_input_bytes: int = 0,
    role: str = "canonical authority JSON",
    label: str = "canonical authority JSON",
) -> Any:
    """Strictly decode and immediately budget one authority JSON graph."""
    value = strict_json_loads(payload, label=label)
    if materialization_budget is not None:
        materialization_budget.reserve(
            value,
            retained_input_bytes=retained_input_bytes,
            role=f"decoded {role}",
        )
    return value


def _read_snapshot_json(
    reader: PinnedDirectoryReader,
    file_inventory: dict[str, tuple[int, int, int, int, int, int, int]],
    relative_path: str,
    *,
    authority_root: Path | None = None,
    read_budget: _CanonicalAuthorityReadBudget | None = None,
    materialization_budget: (
        _CanonicalAuthorityMaterializationBudget | None
    ) = None,
    role: str = "canonical JSON authority",
) -> Any:
    """Parse one bounded JSON leaf from the retained discovery inventory."""
    metadata = file_inventory[relative_path]
    if read_budget is not None:
        if authority_root is None:
            raise ImmutableArtifactError(
                "canonical JSON budget requires an authority root"
            )
        read_budget.reserve_file(
            authority_root.joinpath(*PurePosixPath(relative_path).parts),
            metadata[4],
            role=role,
        )
    payload = reader.read_snapshot(
        relative_path,
        max_bytes=_CANONICAL_AUTHORITY_JSON_MAX_BYTES,
        expected_metadata=metadata,
    ).payload
    return _decode_authority_json(
        payload,
        materialization_budget=materialization_budget,
        retained_input_bytes=(
            read_budget.total_bytes
            if read_budget is not None
            else len(payload)
        ),
        role=role,
    )


def _canonical_discovery_failure(trace_dir: Path, exc: BaseException) -> str:
    if isinstance(exc, (ImmutableArtifactResourceError, MemoryError)):
        return (
            "CanonicalGenerationAuthorityResourceUnavailable: "
            f"{trace_dir} bounded retained authority traversal failed: "
            f"{type(exc).__name__}: {exc}"
        )
    if isinstance(exc, ImmutableArtifactLimitError):
        return (
            "CanonicalGenerationAuthorityInvalid: "
            f"{trace_dir} bounded authority discovery limit exceeded: "
            f"{type(exc).__name__}: {exc}"
        )
    return (
        "CanonicalGenerationAuthorityInvalid: "
        f"{trace_dir} trace authority changed or cannot be inventoried without "
        "symlink indirection through bounded traversal: "
        f"{type(exc).__name__}: {exc}"
    )


def _verify_canonical_trace_pb_admission(trace_dir: Path) -> list[str]:
    """Verify nested generation authority from one retained bounded inventory."""
    errors: list[str] = []
    root = Path(trace_dir).absolute()
    namespaces = (
        _RAW_TRACE_GENERATION_NAMESPACE,
        _DERIVED_TRACE_GENERATION_NAMESPACE,
        _SUMMARY_GENERATION_NAMESPACE,
        _PHASE_LEDGER_GENERATION_NAMESPACE,
        _TRACE_CAPTURE_ATTEMPT_NAMESPACE,
    )
    namespace_set = frozenset(namespaces)
    invalid_ancestor = next(
        (
            component
            for component in root.parts
            if component not in {root.anchor, "/"}
            and (
                component.startswith(".")
                or component in _CANONICAL_RESERVED_NAMESPACE_COMPONENTS
            )
        ),
        None,
    )
    if invalid_ancestor is not None:
        return [
            "CanonicalGenerationAuthorityInvalid: "
            f"{root} scan root is inside hidden or reserved namespace "
            f"component {invalid_ancestor!r}"
        ]
    read_budget = _CanonicalAuthorityReadBudget()
    materialization_budget = _CanonicalAuthorityMaterializationBudget()

    try:
        with ExitStack() as retained_tree:
            reader = retained_tree.enter_context(PinnedDirectoryReader(root))
            monitor = retained_tree.enter_context(
                PinnedPathSetMonitor.from_reader(
                    reader,
                    regular_paths=(),
                    recursive_directories=(".",),
                    max_files=_CANONICAL_DISCOVERY_MAX_FILES,
                    max_directories=_CANONICAL_DISCOVERY_MAX_DIRECTORIES,
                    max_total_file_bytes=(
                        _CANONICAL_DISCOVERY_MAX_TOTAL_FILE_BYTES
                    ),
                    max_total_relative_path_bytes=(
                        _CANONICAL_DISCOVERY_MAX_RELATIVE_PATH_BYTES
                    ),
                )
            )
            snapshot = reader.snapshot_tree(
                max_files=_CANONICAL_DISCOVERY_MAX_FILES,
                max_directories=_CANONICAL_DISCOVERY_MAX_DIRECTORIES,
                max_total_file_bytes=(
                    _CANONICAL_DISCOVERY_MAX_TOTAL_FILE_BYTES
                ),
                max_total_relative_path_bytes=(
                    _CANONICAL_DISCOVERY_MAX_RELATIVE_PATH_BYTES
                ),
            )
            monitor.verify_unchanged()
            file_inventory = dict(snapshot.files)
            directory_inventory = dict(snapshot.directories)

            authority_dirs: set[Path] = set()
            invalid_namespace_entries: dict[Path, list[Path]] = {}
            invalid_record_entries: dict[Path, list[Path]] = {}
            record_candidates: dict[Path, list[str]] = {}

            def has_hidden_or_reserved_ancestor(
                relative: PurePosixPath,
            ) -> bool:
                return any(
                    part.startswith(".") or part in namespace_set
                    for part in relative.parts[:-1]
                )

            def is_trace_record_candidate(relative: PurePosixPath) -> bool:
                return (
                    relative.name.startswith("trace_")
                    and relative.name.endswith(".json")
                    and relative.name
                    not in _KNOWN_NONAUTHORITY_TRACE_FILENAMES
                    and not has_hidden_or_reserved_ancestor(relative)
                )
            for relative_path in directory_inventory:
                if relative_path == ".":
                    continue
                relative = PurePosixPath(relative_path)
                artifact_dir = _snapshot_relative_path(
                    root, relative.parent.as_posix()
                )
                if (
                    relative.name in namespace_set
                    and not has_hidden_or_reserved_ancestor(relative)
                ):
                    authority_dirs.add(artifact_dir)
                if is_trace_record_candidate(relative):
                    authority_dirs.add(artifact_dir)
                    invalid_record_entries.setdefault(
                        artifact_dir, []
                    ).append(_snapshot_relative_path(root, relative_path))
            for relative_path in file_inventory:
                relative = PurePosixPath(relative_path)
                if (
                    relative.name in namespace_set
                    and not has_hidden_or_reserved_ancestor(relative)
                ):
                    artifact_dir = _snapshot_relative_path(
                        root, relative.parent.as_posix()
                    )
                    authority_dirs.add(artifact_dir)
                    invalid_namespace_entries.setdefault(
                        artifact_dir, []
                    ).append(_snapshot_relative_path(root, relative_path))

            # Every canonical-looking trace leaf establishes authority in its own
            # artifact directory, including malformed and explicitly non-active
            # tombstones. Hidden stages and generation interiors are not locator
            # surfaces, but they remain covered by the retained tree monitor.
            for relative_path in file_inventory:
                relative = PurePosixPath(relative_path)
                if not is_trace_record_candidate(relative):
                    continue
                artifact_dir = _snapshot_relative_path(
                    root, relative.parent.as_posix()
                )
                authority_dirs.add(artifact_dir)
                record_candidates.setdefault(artifact_dir, []).append(
                    relative_path
                )

            materialization_budget.reserve(
                (
                    snapshot,
                    file_inventory,
                    directory_inventory,
                    authority_dirs,
                    invalid_namespace_entries,
                    invalid_record_entries,
                    record_candidates,
                ),
                retained_input_bytes=read_budget.total_bytes,
                role="retained canonical authority discovery inventories",
            )

            for artifact_dir in sorted(authority_dirs):
                try:
                    artifact_relative = artifact_dir.relative_to(root).as_posix()
                except ValueError as exc:
                    raise ImmutableArtifactError(
                        "inventoried authority directory escaped retained root"
                    ) from exc
                if artifact_relative == "":
                    artifact_relative = "."

                for invalid_namespace in sorted(
                    invalid_namespace_entries.get(artifact_dir, ())
                ):
                    errors.append(
                        "CanonicalGenerationAuthorityInvalid: "
                        f"{invalid_namespace} generation namespace is not a "
                        "no-follow directory"
                    )

                for invalid_record in sorted(
                    invalid_record_entries.get(artifact_dir, ())
                ):
                    errors.append(
                        "CanonicalGenerationAuthorityInvalid: "
                        f"{invalid_record} canonical trace-record candidate is "
                        "not a single-link regular file"
                    )

                active_records: list[tuple[Path, dict[str, Any]]] = []
                materialization_budget.reserve(
                    active_records,
                    retained_input_bytes=read_budget.total_bytes,
                    role="active canonical trace-record queue",
                )
                for relative_path in sorted(
                    record_candidates.get(artifact_dir, ())
                ):
                    candidate = _snapshot_relative_path(root, relative_path)
                    candidate_metadata = file_inventory[relative_path]
                    if (
                        not stat.S_ISREG(candidate_metadata[2])
                        or stat.S_IMODE(candidate_metadata[2]) != 0o444
                        or candidate_metadata[3] != 1
                    ):
                        errors.append(
                            "CanonicalGenerationAuthorityInvalid: "
                            f"{candidate} canonical trace-record candidate is not "
                            "a single-link 0444 regular file"
                        )
                        continue
                    try:
                        record = _read_snapshot_json(
                            reader,
                            file_inventory,
                            relative_path,
                            authority_root=root,
                            read_budget=read_budget,
                            materialization_budget=materialization_budget,
                            role="canonical trace-record JSON authority",
                        )
                    except (ImmutableArtifactResourceError, MemoryError) as exc:
                        errors.append(
                            "CanonicalGenerationAuthorityResourceUnavailable: "
                            f"{candidate} cannot be retained or decoded: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                    except ImmutableArtifactLimitError as exc:
                        errors.append(
                            "CanonicalGenerationAuthorityInvalid: "
                            f"{candidate} exceeds the bounded authority JSON "
                            f"limit: {type(exc).__name__}: {exc}"
                        )
                        continue
                    except (
                        ImmutableArtifactError,
                        OSError,
                        UnicodeDecodeError,
                        ValueError,
                    ) as exc:
                        errors.append(
                            "CanonicalGenerationAuthorityInvalid: "
                            f"{candidate} is unreadable: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                    if type(record) is not dict:
                        continue
                    has_generation_reference = any(
                        record.get(field) is not None
                        for field in (
                            "raw_trace_generation",
                            "trace_derived_generation",
                            "nvbit_trace_summary_generation",
                            "trace_capture_attempt_event",
                        )
                    )
                    if record.get("activation_state") == "active":
                        shape_point = record.get("shape_point")
                        schema_version = record.get("schema_version")
                        if (
                            type(schema_version) is not int
                            or schema_version != 1
                        ):
                            errors.append(
                                "CanonicalGenerationAuthorityInvalid: "
                                f"{candidate} schema_version must equal one"
                            )
                            continue
                        if record.get("shape_set") != "trace_shapes":
                            errors.append(
                                "CanonicalGenerationAuthorityInvalid: "
                                f"{candidate} shape_set must equal 'trace_shapes'"
                            )
                            continue
                        if record.get("nvbit_trace_summary_emit_status") != "ok":
                            errors.append(
                                "CanonicalGenerationAuthorityInvalid: "
                                f"{candidate} summary status must equal 'ok'"
                            )
                            continue
                        if (
                            type(shape_point) is not str
                            or not shape_point
                            or candidate.name != f"trace_{shape_point}.json"
                        ):
                            errors.append(
                                "CanonicalGenerationAuthorityInvalid: "
                                f"{candidate} has malformed active shape authority"
                            )
                            continue
                        queued_record = (candidate, record)
                        materialization_budget.reserve(
                            queued_record,
                            retained_input_bytes=read_budget.total_bytes,
                            role=(
                                "retained active canonical trace-record tuple "
                                f"{candidate}"
                            ),
                        )
                        prior_list_bytes = sys.getsizeof(active_records)
                        active_records.append(queued_record)
                        materialization_budget.reserve_bytes(
                            sys.getsizeof(active_records) - prior_list_bytes,
                            retained_input_bytes=read_budget.total_bytes,
                            role=(
                                "active canonical trace-record queue growth "
                                f"{candidate}"
                            ),
                        )
                    elif has_generation_reference:
                        errors.append(
                            "CanonicalGenerationAuthorityInvalid: "
                            f"{candidate} carries generation references but is "
                            "not active"
                        )

                if not active_records:
                    errors.append(
                        "CanonicalGenerationAuthorityMissing: "
                        f"{artifact_dir} contains generation authority without an "
                        "active canonical trace record"
                    )
                    continue

                active_shapes = {
                    record["shape_point"] for _path, record in active_records
                }
                materialization_budget.reserve(
                    active_shapes,
                    retained_input_bytes=read_budget.total_bytes,
                    role="retained active shape authority",
                )
                for namespace in namespaces:
                    namespace_relative = _snapshot_join_relative(
                        artifact_relative,
                        namespace,
                    )
                    if namespace_relative not in directory_inventory:
                        continue
                    namespace_path = _snapshot_relative_path(
                        root, namespace_relative
                    )
                    child_names = _snapshot_direct_child_names(
                        snapshot,
                        namespace_relative,
                    )
                    materialization_budget.reserve(
                        (namespace_path, child_names),
                        retained_input_bytes=read_budget.total_bytes,
                        role="retained generation namespace children",
                    )
                    for child_name in sorted(child_names):
                        if child_name.startswith("."):
                            continue
                        if child_name not in active_shapes:
                            errors.append(
                                "CanonicalGenerationAuthorityUnbound: "
                                f"{namespace_path / child_name} has no active "
                                "trace record for its shape"
                            )
                for record_path, record in active_records:
                    for detail in _verify_active_generation_record(
                        record_path,
                        record,
                        authority_root=root,
                        authority_reader=reader,
                        authority_file_inventory=file_inventory,
                        authority_directory_inventory=directory_inventory,
                        read_budget=read_budget,
                        materialization_budget=materialization_budget,
                    ):
                        token = (
                            "CanonicalGenerationAuthorityResourceUnavailable"
                            if (
                                "resource unavailable" in detail
                                or "resource_unavailable" in detail
                            )
                            else "CanonicalGenerationAuthorityInvalid"
                        )
                        errors.append(f"{token}: {record_path}: {detail}")
                materialization_budget.reserve(
                    active_records,
                    retained_input_bytes=read_budget.total_bytes,
                    role="terminal active canonical trace-record queue",
                )

            monitor.verify_unchanged()
            reader.verify_path()
            errors.extend(
                _verify_legacy_canonical_trace_pb_admission(
                    root,
                    generation_authority_dirs=authority_dirs,
                    reader=reader,
                    tree_snapshot=snapshot,
                    read_budget=read_budget,
                    materialization_budget=materialization_budget,
                )
            )
            monitor.verify_unchanged()
            reader.verify_path()
    except (
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
        MemoryError,
        OSError,
        ValueError,
    ) as exc:
        errors.append(_canonical_discovery_failure(root, exc))
    try:
        materialization_budget.reserve(
            errors,
            retained_input_bytes=read_budget.total_bytes,
            role="retained canonical verification diagnostics",
        )
    except (
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
        MemoryError,
        OSError,
        ValueError,
    ) as exc:
        errors.append(_canonical_discovery_failure(root, exc))
    return errors


def _verify_legacy_canonical_trace_pb_admission(
    trace_dir: Path,
    *,
    generation_authority_dirs: set[Path],
    reader: PinnedDirectoryReader | None = None,
    tree_snapshot: PinnedTreeSnapshot | None = None,
    read_budget: _CanonicalAuthorityReadBudget | None = None,
    materialization_budget: (
        _CanonicalAuthorityMaterializationBudget | None
    ) = None,
) -> list[str]:
    """Verify legacy raw trees from the same bounded retained inventory.

    Fallback remains local to each artifact directory: authority in one sibling
    cannot disable legacy checks in another. Discovery or retention failure is a
    hard canonical-authority error and never falls through to lexical probing.
    """
    root = Path(trace_dir).absolute()
    budget = read_budget or _CanonicalAuthorityReadBudget()
    material_budget = (
        materialization_budget
        or _CanonicalAuthorityMaterializationBudget()
    )
    if (reader is None) != (tree_snapshot is None):
        raise TypeError("reader and tree_snapshot must be supplied together")
    if reader is None:
        try:
            with ExitStack() as retained_tree:
                owned_reader = retained_tree.enter_context(
                    PinnedDirectoryReader(root)
                )
                monitor = retained_tree.enter_context(
                    PinnedPathSetMonitor.from_reader(
                        owned_reader,
                        regular_paths=(),
                        recursive_directories=(".",),
                        max_files=_CANONICAL_DISCOVERY_MAX_FILES,
                        max_directories=_CANONICAL_DISCOVERY_MAX_DIRECTORIES,
                        max_total_file_bytes=(
                            _CANONICAL_DISCOVERY_MAX_TOTAL_FILE_BYTES
                        ),
                        max_total_relative_path_bytes=(
                            _CANONICAL_DISCOVERY_MAX_RELATIVE_PATH_BYTES
                        ),
                    )
                )
                owned_snapshot = owned_reader.snapshot_tree(
                    max_files=_CANONICAL_DISCOVERY_MAX_FILES,
                    max_directories=_CANONICAL_DISCOVERY_MAX_DIRECTORIES,
                    max_total_file_bytes=(
                        _CANONICAL_DISCOVERY_MAX_TOTAL_FILE_BYTES
                    ),
                    max_total_relative_path_bytes=(
                        _CANONICAL_DISCOVERY_MAX_RELATIVE_PATH_BYTES
                    ),
                )
                monitor.verify_unchanged()
                result = _verify_legacy_canonical_trace_pb_admission(
                    root,
                    generation_authority_dirs=generation_authority_dirs,
                    reader=owned_reader,
                    tree_snapshot=owned_snapshot,
                    read_budget=budget,
                    materialization_budget=material_budget,
                )
                monitor.verify_unchanged()
                owned_reader.verify_path()
                return result
        except (
            ImmutableArtifactLimitError,
            ImmutableArtifactResourceError,
            ImmutableArtifactError,
            MemoryError,
            OSError,
            ValueError,
        ) as exc:
            return [_canonical_discovery_failure(root, exc)]

    assert tree_snapshot is not None
    errors: list[str] = []
    file_inventory = dict(tree_snapshot.files)
    directory_inventory = dict(tree_snapshot.directories)
    authority_dirs = {Path(path).absolute() for path in generation_authority_dirs}
    material_budget.reserve(
        (tree_snapshot, file_inventory, directory_inventory, authority_dirs),
        retained_input_bytes=budget.total_bytes,
        role="retained legacy canonical authority inventories",
    )

    for trace_relative in sorted(file_inventory):
        relative = PurePosixPath(trace_relative)
        if any(
            part.startswith(".")
            or part in _CANONICAL_RESERVED_NAMESPACE_COMPONENTS
            for part in relative.parts[:-1]
        ):
            continue
        if relative.name != "dynamic_trace.pb":
            continue
        shape_relative = relative.parent
        if not shape_relative.name.startswith("trace_"):
            continue
        artifact_relative = shape_relative.parent.as_posix()
        artifact_dir = _snapshot_relative_path(root, artifact_relative)
        trace_pb = _snapshot_relative_path(root, trace_relative)
        if artifact_dir.absolute() in authority_dirs:
            errors.append(
                "LegacyTraceWithGenerationAuthority: "
                f"{trace_pb} uses the legacy trace_<shape> layout beside "
                "active or generation authority; legacy fallback is disabled"
            )
            continue

        summary_relative = _snapshot_join_relative(
            artifact_relative,
            f"{shape_relative.name}.json",
        )
        summary_path = _snapshot_relative_path(root, summary_relative)
        if summary_relative not in file_inventory:
            errors.append(
                f"RawOnlyTraceTree: {trace_pb} has no binding "
                f"run summary at {summary_path}; the canonical "
                "`run_suite.py --mode trace` path emits a "
                "per-shape `trace_<shape>.json` alongside the raw "
                "`dynamic_trace.pb` — its absence indicates a "
                "bypass-orchestrator capture (e.g., direct "
                "CUDA_INJECTION64_PATH invocation) that did not "
                "go through `handle_trace`. Re-capture via "
                "`run_suite.py --mode trace` so the per-shape "
                "summary, manifest, and feature sidecars are emitted."
            )
            continue
        try:
            summary = _read_snapshot_json(
                reader,
                file_inventory,
                summary_relative,
                authority_root=root,
                read_budget=budget,
                materialization_budget=material_budget,
                role="legacy trace summary JSON authority",
            )
        except (ImmutableArtifactResourceError, MemoryError) as exc:
            errors.append(
                "CanonicalGenerationAuthorityResourceUnavailable: "
                f"{summary_path} cannot be retained or decoded: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        except (
            ImmutableArtifactLimitError,
            ImmutableArtifactError,
            OSError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            errors.append(
                f"RawOnlyTraceTree: {summary_path} is unparseable "
                f"({type(exc).__name__}: {exc}); the orchestrator-emitted run "
                "summary must be valid JSON."
            )
            continue

        shape_point = shape_relative.name[len("trace_") :]
        per_shape_manifest_relative = _snapshot_join_relative(
            artifact_relative,
            f"measurement_manifest_{shape_point}.json",
        )
        cell_level_manifest_relative = _snapshot_join_relative(
            artifact_relative,
            "measurement_manifest.json",
        )
        per_shape_manifest_path = _snapshot_relative_path(
            root, per_shape_manifest_relative
        )
        sibling_shape_dirs = [
            PurePosixPath(path)
            for path in directory_inventory
            if path != "."
            and PurePosixPath(path).parent.as_posix() == artifact_relative
            and PurePosixPath(path).name.startswith("trace_")
        ]
        material_budget.reserve(
            sibling_shape_dirs,
            retained_input_bytes=budget.total_bytes,
            role="retained legacy sibling shape inventory",
        )
        is_multi_shape = len(sibling_shape_dirs) > 1
        if per_shape_manifest_relative in file_inventory:
            pass
        elif (
            cell_level_manifest_relative in file_inventory
            and not is_multi_shape
        ):
            pass
        elif (
            cell_level_manifest_relative in file_inventory
            and is_multi_shape
        ):
            try:
                cell_manifest = _read_snapshot_json(
                    reader,
                    file_inventory,
                    cell_level_manifest_relative,
                    authority_root=root,
                    read_budget=budget,
                    materialization_budget=material_budget,
                    role="legacy measurement manifest JSON authority",
                )
                manifest_shape = (
                    cell_manifest.get("shape_point")
                    if type(cell_manifest) is dict
                    else None
                )
            except (ImmutableArtifactResourceError, MemoryError) as exc:
                errors.append(
                    "CanonicalGenerationAuthorityResourceUnavailable: "
                    f"{_snapshot_relative_path(root, cell_level_manifest_relative)} "
                    "cannot be retained or decoded: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            except (
                ImmutableArtifactLimitError,
                ImmutableArtifactError,
                OSError,
                UnicodeDecodeError,
                ValueError,
            ):
                manifest_shape = None
            if manifest_shape != shape_point:
                errors.append(
                    f"RawOnlyTraceTree: {trace_pb} is in a "
                    f"multi-shape cell ({len(sibling_shape_dirs)} "
                    "`trace_<shape>/` children at "
                    f"{artifact_dir}); the per-shape manifest "
                    f"`{per_shape_manifest_path.name}` is missing AND "
                    "the cell-level `measurement_manifest.json` "
                    f"records shape_point={manifest_shape!r} != "
                    f"current shape {shape_point!r}. The cell-level "
                    "manifest is stale (overwritten by another "
                    "shape's capture); the activation orchestrator "
                    "must emit the per-shape manifest to avoid "
                    "cross-shape overwrites."
                )
                continue
        else:
            errors.append(
                f"RawOnlyTraceTree: {trace_pb} has summary "
                f"{summary_path} but neither "
                f"`{per_shape_manifest_path.name}` nor "
                "`measurement_manifest.json` exists at "
                f"{artifact_dir}; the canonical capture path "
                "emits the manifest via `_emit_manifest` (or its "
                "per-shape variant for multi-shape activation "
                "slugs) — its absence indicates a partial / "
                "interrupted capture."
            )
            continue

        budget_exceeded = any(
            _snapshot_join_relative(shape_relative.as_posix(), filename)
            in file_inventory
            for filename in (
                "trace_volume_exceeded.json",
                "nvbit_slowdown_budget_exceeded.json",
                "nvbit_drop_rate_exceeded.json",
            )
        )
        if budget_exceeded:
            continue
        if type(summary) is not dict:
            # Valid JSON of the wrong shape cannot carry typed sidecar skips.
            summary = {}
        upstream_skip = bool(summary.get("hrd_skip_reason"))
        for sidecar_dir, sidecar_file, skip_field in (
            (
                "sector_stream",
                "sector_stream.jsonl",
                "sector_stream_skip_reason",
            ),
            ("hrd_profile", "hrd_profile.json", "hrd_skip_reason"),
            (
                "divergence_proxies",
                "divergence_proxies.json",
                "divergence_proxies_skip_reason",
            ),
            (
                "sharing_locality",
                "sharing_locality.json",
                "sharing_locality_skip_reason",
            ),
            (
                "order_projections",
                "order_projections.json",
                "order_projections_skip_reason",
            ),
        ):
            sidecar_relative = _snapshot_join_relative(
                shape_relative.as_posix(),
                sidecar_dir,
                sidecar_file,
            )
            skip_reason = summary.get(skip_field)
            if sidecar_relative in file_inventory:
                continue
            if skip_reason or upstream_skip:
                continue
            errors.append(
                f"CanonicalSidecarMissing: {trace_pb} run summary "
                f"{summary_path} admits but the canonical sidecar "
                f"`{sidecar_dir}/{sidecar_file}` is absent and no "
                f"`{skip_field}` is recorded. The canonical "
                "capture path emits all 5 per-shape feature sidecars feature "
                "sidecars (sector_stream, hrd_profile, "
                "divergence_proxies, sharing_locality, "
                "order_projections); a missing file with no "
                "typed-unavailable arm indicates a partial "
                "emission and is not admissible."
            )
    reader.verify_path()
    material_budget.reserve(
        errors,
        retained_input_bytes=budget.total_bytes,
        role="retained legacy canonical verification diagnostics",
    )
    return errors


def _load_schema_validator(schema_filename: str) -> jsonschema.Draft202012Validator:
    schema = json.loads((SCHEMAS_DIR / schema_filename).read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def _strict_validate_sidecar(
    sidecar: Path, schema_filename: str
) -> list[str]:
    """Validate `sidecar`'s on-disk bytes against the committed
    schema via `Draft202012Validator`. The R7 exact-artifact
    discipline applied to the C-AC-H sidecars: schema rejection
    is `BudgetViolationUntracked`-equivalent."""
    try:
        with open(sidecar) as f:
            data = json.load(f)
    except Exception as e:
        return [
            f"BudgetViolationUntracked: sidecar {sidecar} unreadable ({e})"
        ]
    validator = _load_schema_validator(schema_filename)
    errors = list(validator.iter_errors(data))
    if errors:
        first = errors[0]
        return [
            f"BudgetViolationUntracked: sidecar {sidecar} fails strict "
            f"schema validation at {list(first.absolute_path)}: "
            f"{first.message}"
        ]
    return []


# --- C-AC-H recomputation helpers --------------------------------------


class _InvalidEvidenceError(Exception):
    """Raised by `_recompute_slowdown` / `_recompute_drop_rate` when raw
    evidence is malformed (non-numeric, non-positive baseline, negative
    counts, admitted > expected). The caller turns the exception message
    into a `BudgetViolationUntracked` error string."""


def _recompute_slowdown(summary: dict[str, Any]) -> float | None:
    """Return the recomputed slowdown ratio for a per-shape run summary,
    or `None` when raw evidence is absent (typed-unavailable arm).

    Raises `_InvalidEvidenceError` for malformed raw evidence
    (non-numeric values, non-positive baseline, non-positive actual
    wall time). A valid recomputation result is `actual_wall_time_s /
    baseline_wall_time_s`. The summary's `actual_wall_time_s` field is
    preferred over the legacy `wall_time_s` field; both refer to the
    NVBit-instrumented run's wall time and are written to be
    interchangeable, but the explicit C-AC-H field documents the
    semantic.
    """
    baseline = summary.get("baseline_wall_time_s")
    actual_explicit = summary.get("actual_wall_time_s")
    actual_legacy = summary.get("wall_time_s")
    # A legacy summary pre-R11 carries `wall_time_s` but no
    # baseline; it cannot be recomputed against. Return None so the
    # caller routes through the typed-unavailable arm. R11+
    # summaries always carry both fields.
    if baseline is None:
        return None
    if not isinstance(baseline, (int, float)):
        raise _InvalidEvidenceError(
            f"baseline_wall_time_s={baseline!r} is not numeric"
        )
    if baseline <= 0:
        raise _InvalidEvidenceError(
            f"baseline_wall_time_s={baseline} is not strictly positive"
        )
    if actual_explicit is None and actual_legacy is None:
        return None
    actual = (
        actual_explicit if actual_explicit is not None else actual_legacy
    )
    if not isinstance(actual, (int, float)):
        raise _InvalidEvidenceError(
            f"actual wall time {actual!r} is not numeric"
        )
    if actual <= 0:
        raise _InvalidEvidenceError(
            f"actual wall time {actual} is not strictly positive"
        )
    return float(actual) / float(baseline)


def _recompute_drop_rate(summary: dict[str, Any]) -> float | None:
    """Return the recomputed drop rate, or `None` when raw evidence is
    absent. Raises `_InvalidEvidenceError` for malformed raw evidence."""
    expected = summary.get("expected_record_count")
    admitted = summary.get("admitted_record_count")
    if expected is None and admitted is None:
        return None
    if expected is None or admitted is None:
        raise _InvalidEvidenceError(
            "drop-rate evidence partially present; both "
            "expected_record_count and admitted_record_count required"
        )
    if not isinstance(expected, int) or not isinstance(admitted, int):
        raise _InvalidEvidenceError(
            f"record counts must be ints; got expected={expected!r}, "
            f"admitted={admitted!r}"
        )
    if expected < 0 or admitted < 0:
        raise _InvalidEvidenceError(
            f"record counts must be non-negative; got expected={expected}, "
            f"admitted={admitted}"
        )
    if admitted > expected:
        raise _InvalidEvidenceError(
            f"admitted_record_count={admitted} > expected_record_count="
            f"{expected}; producer-side counters are inconsistent"
        )
    if expected == 0:
        # Zero expected records means no producer activity at all; the
        # drop rate is undefined and treated as the typed-unavailable
        # arm rather than a divide-by-zero.
        return None
    return 1.0 - (admitted / expected)


def _ratio_close(a: float, b: float, tol: float = _RATIO_CONSISTENCY_TOL) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _verify_slowdown_sidecars(
    trace_dir: Path, budgets_sha: str
) -> list[str]:
    """C-AC-H slowdown sidecar standalone verifier.

    Walks `trace_dir` for `nvbit_slowdown_budget_exceeded.json`
    sidecars. Each sidecar must (a) admit under strict
    `Draft202012Validator` against the committed schema, (b) declare
    `max_slowdown_ratio == NVBIT_SLOWDOWN_BUDGET_MAX_RATIO` (the fixed
    constant; arbitrary row-local thresholds are rejected),
    (c) record `actual_slowdown_ratio > max_slowdown_ratio`, and
    (d) carry the canonical `trace_budgets_sha256`.
    """
    errors: list[str] = []
    for sidecar in trace_dir.rglob("nvbit_slowdown_budget_exceeded.json"):
        schema_errors = _strict_validate_sidecar(
            sidecar, "nvbit_slowdown_budget_exceeded.schema.json"
        )
        if schema_errors:
            errors.extend(schema_errors)
            continue
        with open(sidecar) as f:
            data = json.load(f)
        actual = data.get("actual_slowdown_ratio")
        max_ratio = data.get("max_slowdown_ratio")
        if max_ratio != _trace_budgets.NVBIT_SLOWDOWN_BUDGET_MAX_RATIO:
            errors.append(
                f"BudgetViolationUntracked: slowdown sidecar {sidecar} "
                f"max_slowdown_ratio={max_ratio} does not equal the "
                f"canonical fixed constant "
                f"NVBIT_SLOWDOWN_BUDGET_MAX_RATIO="
                f"{_trace_budgets.NVBIT_SLOWDOWN_BUDGET_MAX_RATIO}"
            )
            continue
        if (
            not isinstance(actual, (int, float))
            or not isinstance(max_ratio, (int, float))
            or actual <= max_ratio
        ):
            errors.append(
                f"BudgetViolationUntracked: slowdown sidecar {sidecar} "
                f"actual_slowdown_ratio={actual} does not exceed "
                f"max_slowdown_ratio={max_ratio}"
            )
        if budgets_sha and data.get("trace_budgets_sha256") != budgets_sha:
            errors.append(
                f"BudgetViolationUntracked: slowdown sidecar {sidecar} "
                f"trace_budgets_sha256 mismatch"
            )
    return errors


def _verify_drop_rate_sidecars(
    trace_dir: Path, budgets_sha: str
) -> list[str]:
    """C-AC-H drop-rate sidecar standalone verifier.

    Walks `trace_dir` for `nvbit_drop_rate_exceeded.json` sidecars.
    Each sidecar must (a) admit under strict
    `Draft202012Validator` against the committed schema, (b) declare
    `max_drop_rate == NVBIT_DROP_RATE_BUDGET_MAX`, (c) record
    `actual_drop_rate > max_drop_rate`, (d) have its per-source
    counters sum to `expected_record_count - admitted_record_count`,
    and (e) carry the canonical `trace_budgets_sha256`.
    """
    errors: list[str] = []
    for sidecar in trace_dir.rglob("nvbit_drop_rate_exceeded.json"):
        schema_errors = _strict_validate_sidecar(
            sidecar, "nvbit_drop_rate_exceeded.schema.json"
        )
        if schema_errors:
            errors.extend(schema_errors)
            continue
        with open(sidecar) as f:
            data = json.load(f)
        actual = data.get("actual_drop_rate")
        max_rate = data.get("max_drop_rate")
        if max_rate != _trace_budgets.NVBIT_DROP_RATE_BUDGET_MAX:
            errors.append(
                f"BudgetViolationUntracked: drop-rate sidecar {sidecar} "
                f"max_drop_rate={max_rate} does not equal the canonical "
                f"fixed constant NVBIT_DROP_RATE_BUDGET_MAX="
                f"{_trace_budgets.NVBIT_DROP_RATE_BUDGET_MAX}"
            )
            continue
        if (
            not isinstance(actual, (int, float))
            or not isinstance(max_rate, (int, float))
            or actual <= max_rate
        ):
            errors.append(
                f"BudgetViolationUntracked: drop-rate sidecar {sidecar} "
                f"actual_drop_rate={actual} does not exceed "
                f"max_drop_rate={max_rate}"
            )
        expected = data.get("expected_record_count")
        admitted = data.get("admitted_record_count")
        if isinstance(expected, int) and isinstance(admitted, int):
            gap = expected - admitted
            counter_sum = sum(
                int(data.get(k) or 0)
                for k in (
                    "channel_overflow_dropped",
                    "ring_buffer_loss_dropped",
                    "decoder_gap_dropped",
                    "expected_record_mismatch_dropped",
                )
            )
            if counter_sum != gap:
                errors.append(
                    f"BudgetViolationUntracked: drop-rate sidecar "
                    f"{sidecar} per-source counter sum {counter_sum} "
                    f"does not equal expected-admitted gap {gap}"
                )
        if budgets_sha and data.get("trace_budgets_sha256") != budgets_sha:
            errors.append(
                f"BudgetViolationUntracked: drop-rate sidecar {sidecar} "
                f"trace_budgets_sha256 mismatch"
            )
    return errors


def _check_slowdown_sidecar_binds_summary(
    sidecar_path: Path, summary: dict[str, Any], recomputed: float
) -> list[str]:
    """When a run summary breaches the slowdown budget, the matching
    sidecar must bind to the summary by `slug`, `shape_point`, the
    sidecar's `actual_slowdown_ratio` matching the recomputed value,
    and the sidecar's raw timing fields matching the summary's. A
    sidecar present but not bound is `BudgetViolationUntracked`."""
    errors: list[str] = []
    try:
        with open(sidecar_path) as f:
            data = json.load(f)
    except Exception as e:
        return [
            f"BudgetViolationUntracked: sidecar {sidecar_path} unreadable "
            f"({e})"
        ]
    if data.get("slug") != summary.get("slug"):
        errors.append(
            f"BudgetViolationUntracked: slowdown sidecar {sidecar_path} "
            f"slug={data.get('slug')!r} does not match run summary slug "
            f"{summary.get('slug')!r}"
        )
    if data.get("shape_point") != summary.get("shape_point"):
        errors.append(
            f"BudgetViolationUntracked: slowdown sidecar {sidecar_path} "
            f"shape_point={data.get('shape_point')!r} does not match run "
            f"summary shape_point {summary.get('shape_point')!r}"
        )
    sidecar_actual = data.get("actual_slowdown_ratio")
    if (
        not isinstance(sidecar_actual, (int, float))
        or not _ratio_close(float(sidecar_actual), recomputed)
    ):
        errors.append(
            f"BudgetViolationUntracked: slowdown sidecar {sidecar_path} "
            f"actual_slowdown_ratio={sidecar_actual} does not match the "
            f"recomputed run-summary ratio={recomputed}"
        )
    summary_baseline = summary.get("baseline_wall_time_s")
    summary_actual_wall = (
        summary.get("actual_wall_time_s") or summary.get("wall_time_s")
    )
    sidecar_baseline = data.get("baseline_wall_time_s")
    sidecar_actual_wall = data.get("actual_wall_time_s")
    if (
        isinstance(summary_baseline, (int, float))
        and isinstance(sidecar_baseline, (int, float))
        and not _ratio_close(float(sidecar_baseline), float(summary_baseline))
    ):
        errors.append(
            f"BudgetViolationUntracked: slowdown sidecar {sidecar_path} "
            f"baseline_wall_time_s={sidecar_baseline} does not match run "
            f"summary baseline_wall_time_s={summary_baseline}"
        )
    if (
        isinstance(summary_actual_wall, (int, float))
        and isinstance(sidecar_actual_wall, (int, float))
        and not _ratio_close(
            float(sidecar_actual_wall), float(summary_actual_wall)
        )
    ):
        errors.append(
            f"BudgetViolationUntracked: slowdown sidecar {sidecar_path} "
            f"actual_wall_time_s={sidecar_actual_wall} does not match "
            f"run summary actual wall time={summary_actual_wall}"
        )
    return errors


def _check_drop_rate_sidecar_binds_summary(
    sidecar_path: Path, summary: dict[str, Any], recomputed: float
) -> list[str]:
    """C-AC-H drop-rate analog of `_check_slowdown_sidecar_binds_summary`."""
    errors: list[str] = []
    try:
        with open(sidecar_path) as f:
            data = json.load(f)
    except Exception as e:
        return [
            f"BudgetViolationUntracked: sidecar {sidecar_path} unreadable "
            f"({e})"
        ]
    if data.get("slug") != summary.get("slug"):
        errors.append(
            f"BudgetViolationUntracked: drop-rate sidecar {sidecar_path} "
            f"slug={data.get('slug')!r} does not match run summary slug "
            f"{summary.get('slug')!r}"
        )
    if data.get("shape_point") != summary.get("shape_point"):
        errors.append(
            f"BudgetViolationUntracked: drop-rate sidecar {sidecar_path} "
            f"shape_point={data.get('shape_point')!r} does not match run "
            f"summary shape_point {summary.get('shape_point')!r}"
        )
    sidecar_actual = data.get("actual_drop_rate")
    if (
        not isinstance(sidecar_actual, (int, float))
        or not _ratio_close(float(sidecar_actual), recomputed)
    ):
        errors.append(
            f"BudgetViolationUntracked: drop-rate sidecar {sidecar_path} "
            f"actual_drop_rate={sidecar_actual} does not match the "
            f"recomputed run-summary rate={recomputed}"
        )
    if data.get("expected_record_count") != summary.get(
        "expected_record_count"
    ):
        errors.append(
            f"BudgetViolationUntracked: drop-rate sidecar {sidecar_path} "
            f"expected_record_count={data.get('expected_record_count')} "
            f"does not match run summary "
            f"expected_record_count={summary.get('expected_record_count')}"
        )
    if data.get("admitted_record_count") != summary.get(
        "admitted_record_count"
    ):
        errors.append(
            f"BudgetViolationUntracked: drop-rate sidecar {sidecar_path} "
            f"admitted_record_count={data.get('admitted_record_count')} "
            f"does not match run summary "
            f"admitted_record_count={summary.get('admitted_record_count')}"
        )
    return errors


def _budget_sidecar_dir_for_summary(
    summary_path: Path, summary: dict[str, Any],
) -> Path:
    shape_point = summary.get("shape_point")
    if isinstance(shape_point, str) and shape_point:
        return summary_path.parent / f"trace_{shape_point}"
    return summary_path.parent


def _verify_run_summaries_against_budgets(trace_dir: Path) -> list[str]:
    """C-AC-H run-summary verifier (Loop 2 R12 fail-closed rework).

    Walks `trace_dir` for per-shape `trace_*.json` run summaries
    written by `run_suite.py::handle_trace`. For each summary:

    1. Recompute slowdown from raw evidence
       (`actual_wall_time_s` / `baseline_wall_time_s`).
       Absent raw evidence is the typed-unavailable arm; invalid raw
       evidence is `BudgetViolationUntracked`. If the cached
       `actual_slowdown_ratio` is also present, require it to match
       the recomputed value within `_RATIO_CONSISTENCY_TOL`.
    2. Recompute drop rate the same way from
       `expected_record_count` / `admitted_record_count`. Same
       invalid / consistency rules.
    3. If a recomputed budget is breached, the matching typed
       sidecar must (a) exist in the shape-specific trace directory,
       (b) bind to the summary by `slug`, `shape_point`, and
       recomputed actual values, and (c) carry the fixed max
       constants. Absence or non-binding is
       `BudgetViolationUntracked`.

    The R12 closure is fail-closed: a falsified or missing cached
    ratio cannot bypass enforcement when the raw evidence shows a
    breach.
    """
    errors: list[str] = []
    for summary_path in trace_dir.rglob("trace_*.json"):
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        budget_sidecar_dir = _budget_sidecar_dir_for_summary(
            summary_path, summary
        )
        # Slowdown arm.
        try:
            recomputed_slowdown = _recompute_slowdown(summary)
        except _InvalidEvidenceError as exc:
            errors.append(
                f"BudgetViolationUntracked: run summary {summary_path} "
                f"slowdown evidence invalid: {exc}"
            )
            recomputed_slowdown = None
        if recomputed_slowdown is not None:
            cached = summary.get("actual_slowdown_ratio")
            if isinstance(cached, (int, float)) and not _ratio_close(
                float(cached), recomputed_slowdown
            ):
                errors.append(
                    f"BudgetViolationUntracked: run summary "
                    f"{summary_path} cached "
                    f"actual_slowdown_ratio={cached} disagrees with "
                    f"recomputed {recomputed_slowdown} (raw "
                    f"baseline_wall_time_s and actual wall time imply "
                    f"the recomputed value)"
                )
            if recomputed_slowdown > _trace_budgets.NVBIT_SLOWDOWN_BUDGET_MAX_RATIO:
                sidecar_path = (
                    budget_sidecar_dir / "nvbit_slowdown_budget_exceeded.json"
                )
                raw_baseline = summary.get("baseline_wall_time_s")
                baseline_above_floor = (
                    isinstance(raw_baseline, (int, float))
                    and not isinstance(raw_baseline, bool)
                    and float(raw_baseline)
                    >= _trace_budgets.NVBIT_SLOWDOWN_BUDGET_MIN_BASELINE_TIME_S
                )
                if not sidecar_path.is_file():
                    if baseline_above_floor:
                        errors.append(
                            f"BudgetViolationUntracked: run summary "
                            f"{summary_path} recomputed slowdown="
                            f"{recomputed_slowdown:.6f} > "
                            f"{_trace_budgets.NVBIT_SLOWDOWN_BUDGET_MAX_RATIO} "
                            f"with no nvbit_slowdown_budget_exceeded.json "
                            f"sidecar"
                        )
                    # Below the min-baseline-time floor the ratio is
                    # dominated by per-instruction NVBit overhead
                    # rather than a real budget breach; the producer
                    # symmetrically suppresses the sidecar in this
                    # case, so its absence is admissible.
                else:
                    errors.extend(
                        _check_slowdown_sidecar_binds_summary(
                            sidecar_path, summary, recomputed_slowdown
                        )
                    )
        # Drop-rate arm.
        try:
            recomputed_drop = _recompute_drop_rate(summary)
        except _InvalidEvidenceError as exc:
            errors.append(
                f"BudgetViolationUntracked: run summary {summary_path} "
                f"drop-rate evidence invalid: {exc}"
            )
            recomputed_drop = None
        if recomputed_drop is not None:
            cached = summary.get("actual_drop_rate")
            if isinstance(cached, (int, float)) and not _ratio_close(
                float(cached), recomputed_drop
            ):
                errors.append(
                    f"BudgetViolationUntracked: run summary "
                    f"{summary_path} cached actual_drop_rate={cached} "
                    f"disagrees with recomputed {recomputed_drop}"
                )
            if recomputed_drop > _trace_budgets.NVBIT_DROP_RATE_BUDGET_MAX:
                sidecar_path = (
                    budget_sidecar_dir / "nvbit_drop_rate_exceeded.json"
                )
                if not sidecar_path.is_file():
                    errors.append(
                        f"BudgetViolationUntracked: run summary "
                        f"{summary_path} recomputed drop_rate="
                        f"{recomputed_drop:.6f} > "
                        f"{_trace_budgets.NVBIT_DROP_RATE_BUDGET_MAX} "
                        f"with no nvbit_drop_rate_exceeded.json sidecar"
                    )
                else:
                    errors.extend(
                        _check_drop_rate_sidecar_binds_summary(
                            sidecar_path, summary, recomputed_drop
                        )
                    )
    return errors


# --- task11 / the HRD-profiler clause run-summary verifier --------------


_BUNDLE_UNDERCONFIGURED_TYPED_REASONS: frozenset[str] = frozenset({
    # Original-plan-enumerated scalar refusal: the bundle's
    # bins fail the per-bin CI half-width gate or sample-count
    # threshold; the bundle-level annotation MUST be paired
    # with at-least-one underconfigured bin in `bins[]`.
    "cache_model_underconfigured",
    # R54-added defense-in-depth: the producer's
    # `HRD_MAX_SECTOR_STREAM_RECORDS` cap fired, so the
    # reuse-distance computation was skipped entirely. The
    # bundle has empty `bins[]` by design (no MRC support is
    # available); the bundle-level annotation stands alone.
    "hrd_compute_budget_exceeded",
})


def _recompute_hrd_bundle_section10(bundle: dict[str, Any]) -> dict[str, Any]:
    """Recompute the Section 10 cached fields the run summary persists,
    from the bundle's authoritative ``bins[]`` and ``evidence[]``.

    R19 closure (Codex's R18 review directive #7) extended in R55
    (Codex's R54 review): the three surfaces that signal
    underconfigured bundle state — at-least-one
    ``cache_model_underconfigured`` bin in ``bins[]``, a
    bundle-level typed-unavailable annotation in ``evidence[]``
    whose ``typed_reason`` is in
    ``_BUNDLE_UNDERCONFIGURED_TYPED_REASONS``, and the top-level
    ``bundle_underconfigured: true`` cached flag — must agree.

    R55 broadened the evidence-surface admission set so the new
    ``hrd_compute_budget_exceeded`` typed reason is also a
    valid bundle-level scalar-refusal source. ``hrd_compute_
    budget_exceeded`` has no bin counterpart (the budget guard
    skips the reuse-distance compute, so ``bins[]`` is empty
    by design); the bin-vs-annotation pairing invariant only
    applies when the typed reason is
    ``cache_model_underconfigured``.

    The recompute exposes each surface independently so the
    caller can flag the specific mismatch class (bin without
    annotation, annotation without bin for the cache-model
    reason, top-level disagreeing with either).
    """
    bins = bundle.get("bins", [])
    evidence = bundle.get("evidence", [])
    typed_unavailable_count = sum(
        1
        for b in bins
        if isinstance(b, dict) and b.get("result_kind") == "typed_unavailable"
    )
    underconfigured_bin_count = sum(
        1
        for b in bins
        if isinstance(b, dict)
        and b.get("result_kind") == "typed_unavailable"
        and b.get("typed_reason") == "cache_model_underconfigured"
    )
    low_confidence_bin_count = sum(
        1
        for b in bins
        if isinstance(b, dict)
        and b.get("result_kind") == "ok"
        and b.get("low_confidence") is True
    )
    bin_implies_underconfigured = underconfigured_bin_count > 0
    matching_evidence_records = [
        e
        for e in evidence
        if isinstance(e, dict)
        and e.get("result_kind") == "typed_unavailable"
        and e.get("scope") == "bundle"
        and e.get("typed_reason") in _BUNDLE_UNDERCONFIGURED_TYPED_REASONS
    ]
    evidence_implies_underconfigured = len(matching_evidence_records) > 0
    evidence_record = (
        matching_evidence_records[0]
        if matching_evidence_records else None
    )
    # The bundle-level annotation's typed_reason; None when no
    # matching evidence record exists. The cached run-summary
    # field below mirrors this exactly so the verifier can
    # detect a stale cache.
    evidence_typed_reason = (
        evidence_record.get("typed_reason")
        if evidence_record is not None else None
    )
    # The bin/evidence pairing invariant only applies to the
    # cache_model_underconfigured reason. The
    # hrd_compute_budget_exceeded reason has no bin counterpart
    # (empty `bins[]` is intended), so we do NOT require the
    # bin surface to imply underconfigured when the evidence
    # reason is the budget arm.
    bin_evidence_must_pair = (
        evidence_typed_reason == "cache_model_underconfigured"
    )
    if bin_evidence_must_pair:
        # When the evidence carries the cache-model reason, the
        # full three-way agreement applies: bin-implies must
        # equal evidence-implies must equal cached top-level.
        three_way_agreement = (
            bin_implies_underconfigured
            == evidence_implies_underconfigured
            == bool(bundle.get("bundle_underconfigured"))
        )
    else:
        # When the evidence carries the budget reason (or no
        # bundle-level annotation at all), the bin-implies and
        # evidence-implies surfaces don't have to align. The
        # cached top-level must still equal
        # `bin_implies OR evidence_implies` so a stale or
        # tampered cache is rejected.
        cached_top_level = bool(bundle.get("bundle_underconfigured"))
        any_implies = (
            bin_implies_underconfigured
            or evidence_implies_underconfigured
        )
        three_way_agreement = cached_top_level == any_implies
    cached_top_level = bool(bundle.get("bundle_underconfigured"))
    # The "agreed" value is what the run summary should store. If
    # the three surfaces disagree, agreed_value is unstable and the
    # caller raises BudgetViolationUntracked rather than blessing
    # any one of them.
    agreed_value = (
        cached_top_level if three_way_agreement else None
    )
    return {
        "typed_unavailable_count": typed_unavailable_count,
        "underconfigured_bin_count": underconfigured_bin_count,
        "low_confidence_bin_count": low_confidence_bin_count,
        "bin_implies_underconfigured": bin_implies_underconfigured,
        "evidence_implies_underconfigured": evidence_implies_underconfigured,
        "cached_top_level_underconfigured": cached_top_level,
        "three_way_agreement": three_way_agreement,
        "evidence_record": evidence_record,
        "evidence_typed_reason": evidence_typed_reason,
        "bin_evidence_must_pair": bin_evidence_must_pair,
        # The downstream-friendly Section 10 fields the run
        # summary mirrors. When the three surfaces disagree, these
        # carry the cached top-level value so the summary's
        # cached-vs-recomputed comparison still works — but the
        # caller has already flagged the disagreement.
        "bundle_underconfigured": cached_top_level,
        # Field-name preserved for backwards compat with existing
        # cached summaries: the run summary's
        # `cache_model_underconfigured_typed_reason` field now
        # holds whichever typed reason actually fired (may be
        # `cache_model_underconfigured` or
        # `hrd_compute_budget_exceeded`); historical name reflects
        # the original use case.
        "cache_model_underconfigured_typed_reason": (
            evidence_typed_reason
        ),
        "agreed_value": agreed_value,
    }


# Section 10 fields the run summary persists alongside the HRD
# bundle. The verifier compares each duplicated field against the
# bundle's authoritative source and rejects disagreements.
# Mirrors `workloads/simbench_bw/run_suite.py::_HRD_SECTION10_FIELDS`.
_HRD_SECTION10_REQUIRED_FIELDS: tuple[str, ...] = (
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
    "launch_index",
    "kernel_name",
    "device_name",
    "compute_capability_major",
    "compute_capability_minor",
    "sm_count",
    "total_global_mem_bytes",
    "l2_cache_size_bytes",
    "cuda_l2_cache_size_bytes",
    "static_card_l2_cache_size_bytes",
    "cache_model_capacity_source",
    "cache_model_capacity_typed_reason",
    "sector_size_bytes",
    "line_size_bytes",
    "sectors_per_line",
    "geometry_source",
    "geometry_typed_reason",
    # Slice-geometry-aware effective capacity mirror fields
    # (the Round B sub-slice 2.1 refinement). The resolver-
    # side `cuda_num_l2_slices` / `static_card_num_l2_slices`
    # have no bundle counterpart (they are resolver inputs);
    # the resolved `cache_model_num_l2_slices` /
    # `_source` mirror the bundle's
    # `cache_model_spec.num_l2_slices` /
    # `num_l2_slices_source` so the verifier rejects a stale
    # cache.
    "cuda_num_l2_slices",
    "static_card_num_l2_slices",
    "cache_model_num_l2_slices",
    "cache_model_num_l2_slices_source",
    "associativity_policy",
    "fill_policy",
    "write_policy",
    "l1_bypass_cases",
    "ncu_cache_hit_rate_l2_sector",
    "ncu_cache_hit_rate_l1tex_sector",
    "hrd_profile_path",
    "hrd_schema_version",
    "bootstrap_resamples",
    "ci_half_width_target",
    "minimum_bin_sample_count",
    "trace_access_count",
    "trace_drop_count",
    "trace_drop_rate",
    "typed_unavailable_count",
    "underconfigured_bin_count",
    "low_confidence_bin_count",
    "bundle_underconfigured",
    "cache_model_underconfigured_typed_reason",
)


def _bundle_value_for_summary_field(
    bundle: dict[str, Any], field: str,
) -> Any:
    """Map a Section 10 run-summary field name to the bundle's
    authoritative source for that field. Returns the bundle's
    value, or a sentinel ``object()`` when the field is not
    duplicated in the bundle (some Section 10 fields live only in
    the run summary — e.g., NCU label placeholders, drop counts —
    and have no bundle counterpart)."""
    spec = bundle.get("cache_model_spec", {}) or {}
    binning = bundle.get("binning", {}) or {}
    device = bundle.get("device", {}) or {}
    mpi = bundle.get("manifest_pairing_index", {}) or {}
    if field in {"binary_sha256", "source_commit", "kernel_signature_sha256"}:
        return mpi.get(field)
    if field in {
        "device_name",
        "compute_capability_major",
        "compute_capability_minor",
        "sm_count",
        "total_global_mem_bytes",
        "l2_cache_size_bytes",
    }:
        return device.get(field)
    if field == "cache_model_capacity_source":
        return spec.get("capacity_source")
    if field == "geometry_source":
        return spec.get("geometry_source")
    if field == "cache_model_num_l2_slices":
        return spec.get("num_l2_slices")
    if field == "cache_model_num_l2_slices_source":
        return spec.get("num_l2_slices_source")
    if field in {
        "sector_size_bytes",
        "line_size_bytes",
        "sectors_per_line",
        "associativity_policy",
        "fill_policy",
        "write_policy",
        "l1_bypass_cases",
    }:
        return spec.get(field)
    if field == "hrd_schema_version":
        return bundle.get("schema_version")
    if field in {
        "bootstrap_resamples",
        "ci_half_width_target",
        "minimum_bin_sample_count",
    }:
        return binning.get(field)
    return _NO_BUNDLE_COUNTERPART


_NO_BUNDLE_COUNTERPART = object()


def _verify_hrd_run_summaries(trace_dir: Path) -> list[str]:
    """task11 / the HRD-profiler clause R12-style fail-closed verifier.

    Walks `trace_dir` for per-shape `trace_*.json` summaries and, for
    every summary carrying an `hrd_profile_path`:

    1. Resolve the bundle path relative to the summary directory and
       load it.
    2. Strict-validate the on-disk bytes against
       `hrd_profile.schema.json` via `Draft202012Validator`. Schema
       rejection is `BudgetViolationUntracked`-equivalent.
    3. Confirm the bundle's `manifest_pairing_index.slug` /
       `.shape_point` match the run summary.
    4. Confirm the bundle's `cache_model_spec.capacity_source` /
       `.geometry_source` match the cached fields the run summary
       persisted.
    5. Confirm the bundle's `schema_version` matches the cached
       `hrd_schema_version`.
    6. Recompute Section 10 counts + `bundle_underconfigured` from
       the bundle's `bins[]` / `evidence[]`. Reject if the bundle's
       cached top-level `bundle_underconfigured` disagrees with the
       recomputed value (R12 closure inside the bundle), or if the
       run summary's cached counts/flags disagree with the recomputed
       values.

    Summaries that do not carry an `hrd_profile_path` (because the
    sector-stream consumer didn't run, or the bundle emitter failed)
    fall through the typed-unavailable arm and are not enforced
    against. The `hrd_skip_reason` field is the operator-facing
    explanation in those cases.
    """
    errors: list[str] = []
    validator = _load_schema_validator("hrd_profile.schema.json")
    for summary_path in trace_dir.rglob("trace_*.json"):
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        bundle_path_str = summary.get("hrd_profile_path")
        if bundle_path_str is None:
            # Typed-unavailable arm: the sector-stream consumer didn't
            # run, or `emit_hrd_bundle` raised. The producer recorded
            # `hrd_skip_reason`; no bundle to verify.
            continue
        bundle_path_p = Path(bundle_path_str)
        if not bundle_path_p.is_absolute():
            bundle_path_p = (summary_path.parent / bundle_path_p).resolve()
        if not bundle_path_p.is_file():
            errors.append(
                f"BudgetViolationUntracked: run summary {summary_path} "
                f"references hrd_profile_path={bundle_path_str!r} but the "
                f"file is not present at {bundle_path_p}"
            )
            continue
        try:
            with open(bundle_path_p) as f:
                bundle = json.load(f)
        except Exception as exc:
            errors.append(
                f"BudgetViolationUntracked: hrd_profile bundle "
                f"{bundle_path_p} unreadable ({exc})"
            )
            continue
        schema_errors = list(validator.iter_errors(bundle))
        if schema_errors:
            first = schema_errors[0]
            errors.append(
                f"BudgetViolationUntracked: hrd_profile bundle "
                f"{bundle_path_p} fails strict schema validation at "
                f"{list(first.absolute_path)}: {first.message}"
            )
            continue
        mpi = bundle.get("manifest_pairing_index", {})
        if not isinstance(mpi, dict):
            mpi = {}
        if mpi.get("slug") != summary.get("slug"):
            errors.append(
                f"BudgetViolationUntracked: hrd_profile bundle "
                f"{bundle_path_p} manifest_pairing_index.slug="
                f"{mpi.get('slug')!r} does not match run summary slug "
                f"{summary.get('slug')!r}"
            )
        if mpi.get("shape_point") != summary.get("shape_point"):
            errors.append(
                f"BudgetViolationUntracked: hrd_profile bundle "
                f"{bundle_path_p} manifest_pairing_index.shape_point="
                f"{mpi.get('shape_point')!r} does not match run summary "
                f"shape_point {summary.get('shape_point')!r}"
            )
        # Section 10 duplicated-field comparison: every Section 10
        # field that has a bundle counterpart must agree with the
        # bundle's authoritative source.
        #
        # Conditional slice-geometry fields (Codex Round-19 review
        # P2 fix): the 4 slice-geometry fields are required ONLY
        # when the bundle is post-slice-geometry sub-slice (i.e.,
        # the bundle's `cache_model_spec.num_l2_slices` is
        # present). Pre-slice-geometry committed HRD summaries
        # (e.g. summaries that predate the slice-geometry
        # sub-slice landing) lack these fields by construction;
        # the verifier admits them without requiring the new
        # fields.
        bundle_spec_num_l2_slices = (
            bundle.get("cache_model_spec", {}).get("num_l2_slices")
            if isinstance(bundle.get("cache_model_spec"), dict)
            else None
        )
        slice_geometry_present = bundle_spec_num_l2_slices is not None
        slice_geometry_gated_fields = {
            "cuda_num_l2_slices",
            "static_card_num_l2_slices",
            "cache_model_num_l2_slices",
            "cache_model_num_l2_slices_source",
        }
        for field in _HRD_SECTION10_REQUIRED_FIELDS:
            if (
                field in slice_geometry_gated_fields
                and not slice_geometry_present
            ):
                # Pre-slice-geometry bundle: skip the slice-
                # geometry field requirement. The bundle itself
                # doesn't carry the slice-geometry provenance, so
                # the summary's cache cannot be required to
                # carry it either.
                continue
            if field not in summary:
                errors.append(
                    f"BudgetViolationUntracked: run summary "
                    f"{summary_path} is missing required Section 10 "
                    f"field {field!r}"
                )
                continue
            bundle_value = _bundle_value_for_summary_field(
                bundle, field
            )
            if bundle_value is _NO_BUNDLE_COUNTERPART:
                continue
            cached = summary[field]
            if cached != bundle_value:
                errors.append(
                    f"BudgetViolationUntracked: run summary "
                    f"{summary_path} Section 10 field {field}="
                    f"{cached!r} disagrees with bundle "
                    f"{bundle_value!r}"
                )
        # Three-way bundle-underconfigured invariant + count
        # recomputation.
        recomputed = _recompute_hrd_bundle_section10(bundle)
        if not recomputed["three_way_agreement"]:
            if (
                recomputed["bin_implies_underconfigured"]
                and not recomputed["evidence_implies_underconfigured"]
            ):
                errors.append(
                    f"BudgetViolationUntracked: hrd_profile bundle "
                    f"{bundle_path_p} has "
                    f"{recomputed['underconfigured_bin_count']} "
                    f"cache_model_underconfigured bin(s) but no "
                    f"matching bundle-level evidence annotation "
                    f"(scope='bundle', "
                    f"typed_reason='cache_model_underconfigured'); "
                    f"the original-plan HRD-profiler clause requires "
                    f"the bundle-level annotation when any bin is "
                    f"underconfigured"
                )
            if (
                recomputed["evidence_implies_underconfigured"]
                and not recomputed["bin_implies_underconfigured"]
                and recomputed["bin_evidence_must_pair"]
            ):
                # R55: only fire this mismatch for the cache-
                # model reason. The hrd_compute_budget_exceeded
                # reason has no bin counterpart by design (the
                # budget guard skips the reuse-distance
                # computation, so `bins[]` is empty); the
                # evidence-without-bin pattern is the EXPECTED
                # shape when the budget arm fires.
                errors.append(
                    f"BudgetViolationUntracked: hrd_profile bundle "
                    f"{bundle_path_p} carries a bundle-level "
                    f"cache_model_underconfigured evidence annotation "
                    f"but bins[] contains no "
                    f"cache_model_underconfigured record"
                )
            # R55: when the budget arm fires (or no bundle-level
            # annotation at all), the cached top-level must still
            # equal `bin_implies OR evidence_implies` but the bin
            # and evidence surfaces don't have to mirror each
            # other. The `bin_evidence_must_pair` branch covers
            # the cache-model reason's full three-way invariant;
            # the else branch covers the budget arm + the
            # no-annotation case.
            if recomputed["bin_evidence_must_pair"]:
                if (
                    recomputed["cached_top_level_underconfigured"]
                    != recomputed["bin_implies_underconfigured"]
                    or recomputed["cached_top_level_underconfigured"]
                    != recomputed["evidence_implies_underconfigured"]
                ):
                    errors.append(
                        f"BudgetViolationUntracked: hrd_profile bundle "
                        f"{bundle_path_p} top-level "
                        f"bundle_underconfigured="
                        f"{recomputed['cached_top_level_underconfigured']!r} "
                        f"disagrees with bin surface "
                        f"({recomputed['bin_implies_underconfigured']!r}) "
                        f"and/or evidence surface "
                        f"({recomputed['evidence_implies_underconfigured']!r})"
                    )
            else:
                any_implies = (
                    recomputed["bin_implies_underconfigured"]
                    or recomputed["evidence_implies_underconfigured"]
                )
                if (
                    recomputed["cached_top_level_underconfigured"]
                    != any_implies
                ):
                    errors.append(
                        f"BudgetViolationUntracked: hrd_profile bundle "
                        f"{bundle_path_p} top-level "
                        f"bundle_underconfigured="
                        f"{recomputed['cached_top_level_underconfigured']!r} "
                        f"disagrees with bin-or-evidence surface "
                        f"({any_implies!r})"
                    )
        # Verify the bundle-level evidence annotation's
        # underconfigured_bin_count + scalar_hit_rate_refusal
        # match the bins[] count.
        if (
            recomputed["evidence_record"] is not None
            and recomputed["bin_implies_underconfigured"]
        ):
            ev = recomputed["evidence_record"]
            ev_count = ev.get("underconfigured_bin_count")
            ev_refusal = ev.get("scalar_hit_rate_refusal")
            if ev_count != recomputed["underconfigured_bin_count"]:
                errors.append(
                    f"BudgetViolationUntracked: hrd_profile bundle "
                    f"{bundle_path_p} evidence record "
                    f"underconfigured_bin_count={ev_count!r} disagrees "
                    f"with bins[] count "
                    f"{recomputed['underconfigured_bin_count']!r}"
                )
            if ev_refusal is not True:
                errors.append(
                    f"BudgetViolationUntracked: hrd_profile bundle "
                    f"{bundle_path_p} evidence record "
                    f"scalar_hit_rate_refusal={ev_refusal!r}; the R16 "
                    f"task10 contract requires True when bins are "
                    f"underconfigured"
                )
        # Cached-vs-recomputed count comparison.
        for field in (
            "typed_unavailable_count",
            "underconfigured_bin_count",
            "low_confidence_bin_count",
            "bundle_underconfigured",
            "cache_model_underconfigured_typed_reason",
        ):
            cached = summary.get(field)
            recomputed_value = recomputed[field]
            if cached != recomputed_value:
                errors.append(
                    f"BudgetViolationUntracked: run summary "
                    f"{summary_path} cached {field}={cached!r} disagrees "
                    f"with recomputed {recomputed_value!r} from bundle "
                    f"{bundle_path_p}"
                )
    return errors


# --- task13 / the order-projections clause run-summary verifier ----


def _verify_order_projections_run_summaries(trace_dir: Path) -> list[str]:
    """task13 / the order-projections clause R12-style fail-closed
    verifier (R26 closure of Codex's R25 review action item #2).

    Walks `trace_dir` for per-shape `trace_*.json` summaries and,
    for every summary carrying an `order_projections_path`:

    1. Resolve the bundle path relative to the summary directory
       and load it.
    2. Strict-validate the on-disk bytes against
       `order_projections.schema.json` via
       `Draft202012Validator`. Schema rejection is
       `BudgetViolationUntracked`-equivalent.
    3. Confirm the bundle's `manifest_pairing_index.slug` /
       `.shape_point` / `.source_commit` match the run
       summary's cached fields.
    4. Recompute the cached Section fields from the bundle's
       `projections[]` and `disagreement_variance`. Reject any
       cached-vs-recomputed disagreement.

    Summaries that do not carry an `order_projections_path`
    (the typed-unavailable arm; the producer recorded an
    `order_projections_skip_reason`) are not enforced against.
    """
    errors: list[str] = []
    validator = _load_schema_validator(
        "order_projections.schema.json"
    )
    for summary_path in trace_dir.rglob("trace_*.json"):
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        bundle_path_str = summary.get("order_projections_path")
        if bundle_path_str is None:
            continue
        bundle_path_p = Path(bundle_path_str)
        if not bundle_path_p.is_absolute():
            bundle_path_p = (
                summary_path.parent / bundle_path_p
            ).resolve()
        if not bundle_path_p.is_file():
            errors.append(
                f"BudgetViolationUntracked: run summary "
                f"{summary_path} references "
                f"order_projections_path={bundle_path_str!r} but "
                f"the file is not present at {bundle_path_p}"
            )
            continue
        try:
            with open(bundle_path_p) as f:
                bundle = json.load(f)
        except Exception as exc:
            errors.append(
                f"BudgetViolationUntracked: order_projections "
                f"bundle {bundle_path_p} unreadable ({exc})"
            )
            continue
        schema_errors = list(validator.iter_errors(bundle))
        if schema_errors:
            first = schema_errors[0]
            errors.append(
                f"BudgetViolationUntracked: order_projections "
                f"bundle {bundle_path_p} fails strict schema "
                f"validation at "
                f"{list(first.absolute_path)}: {first.message}"
            )
            continue
        mpi = bundle.get("manifest_pairing_index", {})
        if not isinstance(mpi, dict):
            mpi = {}
        if mpi.get("slug") != summary.get("slug"):
            errors.append(
                f"BudgetViolationUntracked: order_projections "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.slug={mpi.get('slug')!r} "
                f"does not match run summary slug "
                f"{summary.get('slug')!r}"
            )
        if mpi.get("shape_point") != summary.get("shape_point"):
            errors.append(
                f"BudgetViolationUntracked: order_projections "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.shape_point="
                f"{mpi.get('shape_point')!r} does not match run "
                f"summary shape_point "
                f"{summary.get('shape_point')!r}"
            )
        cached_source_commit = summary.get(
            "order_projections_source_commit"
        )
        if mpi.get("source_commit") != cached_source_commit:
            errors.append(
                f"BudgetViolationUntracked: order_projections "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.source_commit="
                f"{mpi.get('source_commit')!r} does not match "
                f"run summary "
                f"order_projections_source_commit="
                f"{cached_source_commit!r}"
            )
        cached_schema_version = summary.get(
            "order_projections_schema_version"
        )
        if cached_schema_version != bundle.get("schema_version"):
            errors.append(
                f"BudgetViolationUntracked: run summary "
                f"{summary_path} "
                f"order_projections_schema_version="
                f"{cached_schema_version!r} does not match "
                f"bundle schema_version="
                f"{bundle.get('schema_version')!r}"
            )
        recomputed = _recompute_order_projections_section(bundle)
        for field in (
            "order_projections_admitted_count",
            "order_projections_typed_unavailable_count",
            "order_projections_disagreement_variance_result_kind",
            "order_projections_disagreement_variance_value",
            "order_projections_disagreement_variance_typed_reason",
            "order_projections_total_access_events",
        ):
            cached = summary.get(field)
            recomputed_value = recomputed[field]
            if cached != recomputed_value:
                errors.append(
                    f"BudgetViolationUntracked: run summary "
                    f"{summary_path} cached {field}={cached!r} "
                    f"disagrees with recomputed "
                    f"{recomputed_value!r} from bundle "
                    f"{bundle_path_p}"
                )
    return errors


# =================================================================
# R54 / Codex's R53 review blocking issue: verifier binding for the
# two newly-wired canonical sidecars (`divergence_proxies/` and
# `sharing_locality/`). Mirrors the HRD/order-projections pattern:
# resolve cached path, strict-validate on-disk bundle, recompute
# cached counts/totals, reject any mismatch.
# =================================================================


def _verify_divergence_proxies_run_summaries(
    trace_dir: Path,
) -> list[str]:
    """The divergence-proxies clause fail-closed verifier. For every
    per-shape `trace_*.json` summary carrying a
    `divergence_proxies_path`:

    1. Resolve the bundle path relative to the summary directory.
    2. Strict-validate the on-disk bytes against
       `divergence_proxies.schema.json`.
    3. Bind the bundle's `manifest_pairing_index.{slug,
       shape_point, source_commit}` to the run summary's `slug`,
       `shape_point`, and `divergence_proxies_source_commit`
       (the same provenance binding the order-projections,
       sharing-locality, HRD, and perturbation-ledger verifiers
       enforce; closes the C-AC-G2 review gap that the bundle
       had no manifest-pairing pin).
    4. Recompute the cached
       `divergence_proxies_admitted_count` and
       `divergence_proxies_typed_unavailable_count` from the
       on-disk bundle's 8 proxy records and reject any
       cached-vs-recomputed disagreement.

    Summaries that do not carry a `divergence_proxies_path` (the
    typed-unavailable arm with `divergence_proxies_skip_reason`
    recorded) are not enforced against — the canonical sidecar
    completeness check governs those.
    """
    errors: list[str] = []
    validator = _load_schema_validator(
        "divergence_proxies.schema.json"
    )
    _proxy_names = (
        "unique_sectors_per_warp_inst",
        "unique_lines_per_warp_inst",
        "sectors_per_request",
        "md_loads_per_kilo_inst",
        "outstanding_load_density_by_phase",
        "pc_repeated_miss_likely_streams",
        "stride_entropy",
        "throughput_saturation_slope_under_resident_warp_sweep",
    )
    for summary_path in trace_dir.rglob("trace_*.json"):
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        bundle_path_str = summary.get("divergence_proxies_path")
        if bundle_path_str is None:
            continue
        bundle_path_p = Path(bundle_path_str)
        if not bundle_path_p.is_absolute():
            bundle_path_p = (
                summary_path.parent / bundle_path_p
            ).resolve()
        if not bundle_path_p.is_file():
            errors.append(
                f"BudgetViolationUntracked: run summary "
                f"{summary_path} references "
                f"divergence_proxies_path={bundle_path_str!r} "
                f"but the file is not present at {bundle_path_p}"
            )
            continue
        try:
            with open(bundle_path_p) as f:
                bundle = json.load(f)
        except Exception as exc:
            errors.append(
                f"BudgetViolationUntracked: divergence_proxies "
                f"bundle {bundle_path_p} unreadable ({exc})"
            )
            continue
        schema_errors = list(validator.iter_errors(bundle))
        if schema_errors:
            first = schema_errors[0]
            errors.append(
                f"BudgetViolationUntracked: divergence_proxies "
                f"bundle {bundle_path_p} fails strict schema "
                f"validation at "
                f"{list(first.absolute_path)}: {first.message}"
            )
            continue
        cached_schema_version = summary.get(
            "divergence_proxies_schema_version"
        )
        if int(bundle.get("schema_version", -1)) != cached_schema_version:
            errors.append(
                f"BudgetViolationUntracked: divergence_proxies "
                f"bundle {bundle_path_p} schema_version="
                f"{bundle.get('schema_version')!r} disagrees "
                f"with cached "
                f"divergence_proxies_schema_version="
                f"{cached_schema_version!r} in {summary_path}"
            )
        mpi = bundle.get("manifest_pairing_index", {})
        if not isinstance(mpi, dict):
            mpi = {}
        if mpi.get("slug") != summary.get("slug"):
            errors.append(
                f"BudgetViolationUntracked: divergence_proxies "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.slug={mpi.get('slug')!r} "
                f"does not match run summary slug "
                f"{summary.get('slug')!r}"
            )
        if mpi.get("shape_point") != summary.get("shape_point"):
            errors.append(
                f"BudgetViolationUntracked: divergence_proxies "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.shape_point="
                f"{mpi.get('shape_point')!r} does not match run "
                f"summary shape_point "
                f"{summary.get('shape_point')!r}"
            )
        cached_source_commit = summary.get(
            "divergence_proxies_source_commit"
        )
        if mpi.get("source_commit") != cached_source_commit:
            errors.append(
                f"BudgetViolationUntracked: divergence_proxies "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.source_commit="
                f"{mpi.get('source_commit')!r} does not match "
                f"run summary "
                f"divergence_proxies_source_commit="
                f"{cached_source_commit!r}"
            )
        # Bundle schema requires {slug, shape_point, binary_sha256,
        # source_commit, kernel_signature_sha256} on manifest_pairing_index;
        # the run summary already carries binary_sha256 and
        # kernel_signature_sha256 alongside source_commit. Compare both
        # axes against the enclosing run summary so a bundle copied from
        # a different binary (or a different kernel signature) at the
        # same source commit cannot be silently admitted. Mirrors the
        # trace_convergence verifier's manifest_run_summary_drift block.
        run_binary_sha = summary.get("binary_sha256")
        if (
            run_binary_sha is not None
            and mpi.get("binary_sha256") is not None
            and mpi.get("binary_sha256") != run_binary_sha
        ):
            errors.append(
                f"BudgetViolationUntracked: divergence_proxies "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.binary_sha256="
                f"{mpi.get('binary_sha256')!r} does not match "
                f"run summary binary_sha256="
                f"{run_binary_sha!r}"
            )
        run_kernel_sig = summary.get("kernel_signature_sha256")
        if (
            run_kernel_sig is not None
            and mpi.get("kernel_signature_sha256") is not None
            and mpi.get("kernel_signature_sha256") != run_kernel_sig
        ):
            errors.append(
                f"BudgetViolationUntracked: divergence_proxies "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.kernel_signature_sha256="
                f"{mpi.get('kernel_signature_sha256')!r} does not "
                f"match run summary kernel_signature_sha256="
                f"{run_kernel_sig!r}"
            )
        # Recompute admitted + typed-unavailable counts from the
        # 8 proxy records (the bundle is FLAT — each proxy name
        # is a top-level field carrying `result_kind`).
        recomputed_admitted = 0
        recomputed_typed_unavailable = 0
        for name in _proxy_names:
            rec = bundle.get(name)
            if not isinstance(rec, dict):
                continue
            kind = rec.get("result_kind")
            if kind == "proxy_value":
                recomputed_admitted += 1
            elif kind == "typed_unavailable":
                recomputed_typed_unavailable += 1
        for field, cached_val, recomputed_val in (
            (
                "divergence_proxies_admitted_count",
                summary.get("divergence_proxies_admitted_count"),
                recomputed_admitted,
            ),
            (
                "divergence_proxies_typed_unavailable_count",
                summary.get(
                    "divergence_proxies_typed_unavailable_count"
                ),
                recomputed_typed_unavailable,
            ),
        ):
            if cached_val != recomputed_val:
                errors.append(
                    f"BudgetViolationUntracked: run summary "
                    f"{summary_path} cached {field}="
                    f"{cached_val!r} disagrees with recomputed "
                    f"{recomputed_val!r} from bundle "
                    f"{bundle_path_p}"
                )
    return errors


def _verify_sharing_locality_run_summaries(
    trace_dir: Path,
) -> list[str]:
    """task / the sharing-locality clause R12-style fail-closed
    verifier (R54 closure of Codex's R53 review blocking issue
    #1). For every per-shape `trace_*.json` summary carrying a
    `sharing_locality_path`:

    1. Resolve the bundle path relative to the summary directory.
    2. Strict-validate against
       `sharing_locality.schema.json`.
    3. Recompute the cached
       `sharing_locality_grounded_count`,
       `sharing_locality_projected_count`, and
       `sharing_locality_total_accesses` from the bundle's
       `grounded`, `projected`, and `evidence` arrays. Reject
       any cached-vs-recomputed disagreement.

    Summaries that do not carry a `sharing_locality_path` (the
    typed-unavailable arm with `sharing_locality_skip_reason`
    recorded) are not enforced against.
    """
    errors: list[str] = []
    validator = _load_schema_validator(
        "sharing_locality.schema.json"
    )
    for summary_path in trace_dir.rglob("trace_*.json"):
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        bundle_path_str = summary.get("sharing_locality_path")
        if bundle_path_str is None:
            continue
        bundle_path_p = Path(bundle_path_str)
        if not bundle_path_p.is_absolute():
            bundle_path_p = (
                summary_path.parent / bundle_path_p
            ).resolve()
        if not bundle_path_p.is_file():
            errors.append(
                f"BudgetViolationUntracked: run summary "
                f"{summary_path} references "
                f"sharing_locality_path={bundle_path_str!r} "
                f"but the file is not present at {bundle_path_p}"
            )
            continue
        try:
            with open(bundle_path_p) as f:
                bundle = json.load(f)
        except Exception as exc:
            errors.append(
                f"BudgetViolationUntracked: sharing_locality "
                f"bundle {bundle_path_p} unreadable ({exc})"
            )
            continue
        schema_errors = list(validator.iter_errors(bundle))
        if schema_errors:
            first = schema_errors[0]
            errors.append(
                f"BudgetViolationUntracked: sharing_locality "
                f"bundle {bundle_path_p} fails strict schema "
                f"validation at "
                f"{list(first.absolute_path)}: {first.message}"
            )
            continue
        cached_schema_version = summary.get(
            "sharing_locality_schema_version"
        )
        if int(bundle.get("schema_version", -1)) != cached_schema_version:
            errors.append(
                f"BudgetViolationUntracked: sharing_locality "
                f"bundle {bundle_path_p} schema_version="
                f"{bundle.get('schema_version')!r} disagrees "
                f"with cached "
                f"sharing_locality_schema_version="
                f"{cached_schema_version!r} in {summary_path}"
            )
        mpi = bundle.get("manifest_pairing_index", {})
        if not isinstance(mpi, dict):
            mpi = {}
        cached_source_commit = summary.get(
            "sharing_locality_source_commit"
        )
        if mpi.get("source_commit") != cached_source_commit:
            errors.append(
                f"BudgetViolationUntracked: sharing_locality "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.source_commit="
                f"{mpi.get('source_commit')!r} disagrees with "
                f"cached sharing_locality_source_commit="
                f"{cached_source_commit!r} in {summary_path}"
            )
        # Recompute grounded + projected counts from the bundle's
        # arrays. The schema's grounded array has 3 records and
        # projected has 4; we count those whose `result_kind ==
        # "ok"`.
        grounded = bundle.get("grounded") or []
        projected = bundle.get("projected") or []
        recomputed_grounded = sum(
            1 for r in grounded
            if isinstance(r, dict) and r.get("result_kind") == "ok"
        )
        recomputed_projected = sum(
            1 for r in projected
            if isinstance(r, dict) and r.get("result_kind") == "ok"
        )
        # Total accesses comes from the
        # `sharing_locality_pair_count_summary` evidence record
        # (per the schema's `addresses_observed` field — the R53
        # emitter incorrectly read `total_accesses` and reported
        # 0 on every canonical capture).
        evidence_records = bundle.get("evidence") or []
        recomputed_total_accesses = 0
        for ev in evidence_records:
            if not isinstance(ev, dict):
                continue
            if (
                ev.get("note_kind")
                == "sharing_locality_pair_count_summary"
            ):
                v = ev.get("addresses_observed")
                if isinstance(v, int):
                    recomputed_total_accesses += v
        for field, cached_val, recomputed_val in (
            (
                "sharing_locality_grounded_count",
                summary.get("sharing_locality_grounded_count"),
                recomputed_grounded,
            ),
            (
                "sharing_locality_projected_count",
                summary.get("sharing_locality_projected_count"),
                recomputed_projected,
            ),
            (
                "sharing_locality_total_accesses",
                summary.get("sharing_locality_total_accesses"),
                recomputed_total_accesses,
            ),
        ):
            if cached_val != recomputed_val:
                errors.append(
                    f"BudgetViolationUntracked: run summary "
                    f"{summary_path} cached {field}="
                    f"{cached_val!r} disagrees with recomputed "
                    f"{recomputed_val!r} from bundle "
                    f"{bundle_path_p}"
                )
    return errors


def _recompute_order_projections_section(
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the cached order-projections section fields
    from the bundle's authoritative `projections[]` and
    `disagreement_variance`. The R12 fail-closed pattern
    applied to the order-projections surface."""
    projections = bundle.get("projections", [])
    admitted = sum(
        1 for p in projections
        if isinstance(p, dict) and p.get("result_kind") == "ok"
    )
    typed_unavail = sum(
        1 for p in projections
        if isinstance(p, dict)
        and p.get("result_kind") == "typed_unavailable"
    )
    dv = bundle.get("disagreement_variance", {}) or {}
    if dv.get("result_kind") == "ok":
        dv_value = dv.get("value")
        dv_typed_reason = None
    else:
        dv_value = None
        dv_typed_reason = dv.get("typed_reason")
    total_events = 0
    for ev in bundle.get("evidence", []) or []:
        if (
            isinstance(ev, dict)
            and ev.get("note_kind") == "order_projections_access_summary"
        ):
            total_events = int(ev.get("total_access_events", 0))
            break
    return {
        "order_projections_admitted_count": admitted,
        "order_projections_typed_unavailable_count": typed_unavail,
        "order_projections_disagreement_variance_result_kind": (
            dv.get("result_kind")
        ),
        "order_projections_disagreement_variance_value": dv_value,
        "order_projections_disagreement_variance_typed_reason": (
            dv_typed_reason
        ),
        "order_projections_total_access_events": total_events,
    }


# --- task15 / the perturbation-harness clause verifier helpers ------------


_LDG_ANY_RE = re.compile(r"\bLDG(?:\.[A-Z0-9_]+)*\b")
# Accept BOTH Ampere-style (`LDG.E.CA`, `LDG.E.CG`) and Blackwell-style
# (`LDG.E.STRONG.SM` for `.ca`; `LDG.E.STRONG.GPU` for `.cg`;
# `LDG.E.EF` for `.cs` per R45) cache-modifier mnemonics. Mirrors
# `tools/perturbation_harness/sass_confirm.py` so the verifier's
# `_recompute_sass_confirmation_hash` matches the producer.
_LDG_CA_RE = re.compile(
    r"\bLDG(?:\.[A-Z0-9_]+)*\.(?:CA|STRONG\.SM)(?:\.[A-Z0-9_]+)*\b"
)
_LDG_CG_RE = re.compile(
    r"\bLDG(?:\.[A-Z0-9_]+)*\.(?:CG|STRONG\.GPU)(?:\.[A-Z0-9_]+)*\b"
)
_LDG_CS_RE = re.compile(
    r"\bLDG(?:\.[A-Z0-9_]+)*\.(?:CS|EF)(?:\.[A-Z0-9_]+)*\b"
)
_LEADING_ADDR_COMMENT_RE = re.compile(r"^/\*[0-9A-Fa-fx_ ]+\*/\s*")
_INTERNAL_WS_RE = re.compile(r"\s+")
_PERTURBATION_KERNEL_NAME = "dram_stream_stride_mix_kernel"


def _normalize_sass_confirmation_lines(
    sass_text: str, family: str, parameter: dict[str, Any],  # noqa: ARG001
) -> list[str]:
    """Apply the per-family kernel-scoped LDG extraction +
    normalization the producer uses, mirroring
    `tools/perturbation_harness/sass_confirm.py`. The verifier
    recomputes from the on-disk SASS dump so a tampered
    `sass_confirmation_hash` is rejected even if the cached
    `sass_match_count` looks consistent.
    """
    if family == "cache_modifier_ca":
        pattern = _LDG_CA_RE
    elif family == "cache_modifier_cg":
        pattern = _LDG_CG_RE
    elif family == "cache_modifier_cs":
        pattern = _LDG_CS_RE
    elif family == "working_set_sweep":
        pattern = _LDG_ANY_RE
    else:
        return []
    section_lines: list[str] = []
    in_section = False
    for line in sass_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Function :"):
            name = stripped[len("Function :"):].strip()
            in_section = (
                name == _PERTURBATION_KERNEL_NAME
                or _PERTURBATION_KERNEL_NAME in name
            )
            continue
        if in_section:
            section_lines.append(line)
    matching = [
        line for line in section_lines if pattern.search(line)
    ]
    out: list[str] = []
    for line in matching:
        s = line.strip()
        s = _LEADING_ADDR_COMMENT_RE.sub("", s)
        s = _INTERNAL_WS_RE.sub(" ", s)
        if s:
            out.append(s)
    return out


def _recompute_sass_confirmation_hash(
    sass_path: Path, family: str, parameter: dict[str, Any],
) -> tuple[str, int]:
    """Recompute `(sass_confirmation_hash, sass_match_count)` from
    the on-disk SASS dump referenced by the admitted record's
    `sass_artifact_path`. Used by the per-record verifier to reject
    cached-vs-recomputed hash drift."""
    sass_text = sass_path.read_text(encoding="utf-8")
    normalized = _normalize_sass_confirmation_lines(sass_text, family, parameter)
    payload = "\n".join(normalized) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest, len(normalized)


def _correctness_delta_exceeds_tolerance(delta: dict[str, Any]) -> bool:
    """Per-mode dispatch over `delta["comparison_mode"]`. Used to
    reject an admitted record whose delta crosses tolerance with no
    matching `correctness_equivalence_failed` typed-unavailable
    record. Mirrors
    `tools/perturbation_harness/correctness_check.py::delta_exceeds_tolerance`.
    """
    mode = delta.get("comparison_mode")
    if mode == "abs_and_rel":
        max_abs = float(delta["max_abs_delta"])
        abs_tol = float(delta["abs_tol"])
        rel_tol = float(delta["rel_tol"])
        ref = float(delta["first_mismatch_reference_value"])
        return max_abs > abs_tol + rel_tol * abs(ref)
    if mode == "exact_match":
        return float(delta["max_abs_delta"]) != 0
    if mode == "sha256_f32le":
        return delta["actual_sha256"] != delta["reference_sha256"]
    return False


def _perturbation_typed_reason_present(
    bundle: dict[str, Any], *, family: str, parameter: dict[str, Any],
    typed_reason: str,
) -> bool:
    """True iff the bundle contains a typed-unavailable record with
    the given (family, parameter, typed_reason) triple. Used to
    accept the case 'admitted record exceeded tolerance but a
    matching typed-unavailable record exists for the same
    parameter'."""
    for record in bundle.get("perturbations", []):
        if not isinstance(record, dict):
            continue
        if record.get("result_kind") != "typed_unavailable":
            continue
        if record.get("family") != family:
            continue
        if record.get("parameter") != parameter:
            continue
        if record.get("typed_reason") != typed_reason:
            continue
        return True
    return False


# The pinned default-loop perturbation parameter set per the
# task14 / R32 spec, extended in R45 with the cache_modifier_cs
# variant: one cache_modifier_ca, one cache_modifier_cg, one
# cache_modifier_cs, and one working_set_sweep record per MiB
# ladder slot. The default bundle MUST contain exactly these 10
# (family, parameter) pairs. Either result_kind=ok or schema-valid
# typed_unavailable per parameter; missing or duplicate parameters
# fail the verifier invariant.
_DEFAULT_LOOP_PERTURBATION_PARAMETER_SET: tuple[
    tuple[str, tuple[tuple[str, Any], ...]], ...
] = (
    ("cache_modifier_ca", (("modifier", "ca"),)),
    ("cache_modifier_cg", (("modifier", "cg"),)),
    ("cache_modifier_cs", (("modifier", "cs"),)),
    ("working_set_sweep", (("working_set_mib", 1),)),
    ("working_set_sweep", (("working_set_mib", 2),)),
    ("working_set_sweep", (("working_set_mib", 4),)),
    ("working_set_sweep", (("working_set_mib", 8),)),
    ("working_set_sweep", (("working_set_mib", 16),)),
    ("working_set_sweep", (("working_set_mib", 32),)),
    ("working_set_sweep", (("working_set_mib", 64),)),
)


def _perturbation_record_key(record: dict[str, Any]) -> tuple[str, tuple[tuple[str, Any], ...]] | None:
    """Return a hashable `(family, frozen_parameter_subset)` key the
    default-loop invariant uses to count parameter occurrences. The
    parameter subset is the discriminating subset (`modifier` for
    cache modifiers; `working_set_mib` for working_set_sweep) so the
    full parameter object can carry additional descriptive fields
    (`allocation_bytes`, `warps`, `accesses`, `stride_classes`)
    without affecting the parameter-set match."""
    family = record.get("family")
    parameter = record.get("parameter") or {}
    if family in {"cache_modifier_ca", "cache_modifier_cg", "cache_modifier_cs"}:
        modifier = parameter.get("modifier")
        if modifier is None:
            return None
        return (family, (("modifier", modifier),))
    if family == "working_set_sweep":
        mib = parameter.get("working_set_mib")
        if mib is None:
            return None
        return (family, (("working_set_mib", int(mib)),))
    return None


def _verify_perturbation_run_ledger_default_loop_parameter_set(
    bundle: dict[str, Any], *, bundle_path: Path,
) -> list[str]:
    """Recompute the bundle's `(family, parameter)` set and assert it
    equals the pinned 10-parameter loop scope. The default loop bundle
    MUST contain exactly the 10 pinned parameter pairs (R45 added
    cache_modifier_cs, growing from 9 to 10); missing or duplicate
    parameters are rejected. Either result_kind=ok or schema-valid
    typed_unavailable per parameter is acceptable. R32 closure of
    Codex's R31 review action item #2; R45 grew the loop scope."""
    errors: list[str] = []
    expected_keys = list(_DEFAULT_LOOP_PERTURBATION_PARAMETER_SET)
    expected_count: dict[tuple[str, tuple[tuple[str, Any], ...]], int] = {}
    for key in expected_keys:
        expected_count[key] = expected_count.get(key, 0) + 1
    seen_count: dict[tuple[str, tuple[tuple[str, Any], ...]], int] = {}
    for record in bundle.get("perturbations", []) or []:
        if not isinstance(record, dict):
            continue
        key = _perturbation_record_key(record)
        if key is None:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} record family="
                f"{record.get('family')!r} parameter="
                f"{record.get('parameter')!r} does not produce a "
                f"recognizable parameter-set key"
            )
            continue
        seen_count[key] = seen_count.get(key, 0) + 1
    for key, want in expected_count.items():
        got = seen_count.get(key, 0)
        if got < want:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} is missing the default-loop "
                f"parameter {key!r} ({got} of {want} present)"
            )
        elif got > want:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} has duplicate default-loop "
                f"parameter {key!r} ({got} of {want} expected)"
            )
    for key, got in seen_count.items():
        if key not in expected_count:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} carries an unexpected "
                f"parameter {key!r} that is not part of the default "
                f"loop scope"
            )
    return errors


def _verify_perturbation_run_ledger_working_set_metadata(
    bundle: dict[str, Any], *, bundle_path: Path,
) -> list[str]:
    """Enforce per-record `working_set_sweep` metadata invariants the
    parameter table in task14 Section 1 pins. R33 closure per
    Codex's R32 review: a producer bug could silently emit a record
    that claims a large `working_set_mib` while the actual
    correctness comparison covered only 128 elements (the R31
    failure class). The verifier's default-loop helper deliberately
    keys only on `(family, working_set_mib)` so it does not reject
    such drift.

    For every working_set_sweep record (admitted OR
    typed_unavailable that carries `correctness_delta`), assert:

    - `parameter.allocation_bytes == working_set_mib * 1048576`
    - `parameter.warps == 32`
    - `parameter.accesses == working_set_mib * 4096`
    - `parameter.stride_classes == 2`
    - `correctness_delta.element_count == working_set_mib * 262144`
    - `correctness_delta.comparison_mode == "sha256_f32le"` for
      `working_set_mib >= 16`; `"abs_and_rel"` otherwise.
    """
    errors: list[str] = []
    for record in bundle.get("perturbations", []) or []:
        if not isinstance(record, dict):
            continue
        if record.get("family") != "working_set_sweep":
            continue
        parameter = record.get("parameter") or {}
        mib = parameter.get("working_set_mib")
        if not isinstance(mib, int):
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} working_set_sweep record "
                f"parameter {parameter!r} missing or non-integer "
                f"`working_set_mib`"
            )
            continue
        expected_alloc = mib * 1048576
        expected_acc = mib * 4096
        expected_elements = mib * 262144
        if parameter.get("allocation_bytes") != expected_alloc:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} working_set_sweep mib={mib} "
                f"parameter.allocation_bytes="
                f"{parameter.get('allocation_bytes')!r} does not match "
                f"the expected mib*1048576={expected_alloc}"
            )
        if parameter.get("warps") != 32:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} working_set_sweep mib={mib} "
                f"parameter.warps={parameter.get('warps')!r} does not "
                f"match the pinned value 32"
            )
        if parameter.get("accesses") != expected_acc:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} working_set_sweep mib={mib} "
                f"parameter.accesses={parameter.get('accesses')!r} "
                f"does not match the expected mib*4096={expected_acc}"
            )
        if parameter.get("stride_classes") != 2:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} working_set_sweep mib={mib} "
                f"parameter.stride_classes="
                f"{parameter.get('stride_classes')!r} does not match "
                f"the pinned value 2"
            )
        delta = record.get("correctness_delta")
        if not isinstance(delta, dict):
            # Non-correctness typed-unavailable arms (e.g.
            # arch_feature_unavailable_*) skip the delta-shape
            # checks; the parameter checks above still apply.
            continue
        actual_elements = delta.get("element_count")
        if actual_elements != expected_elements:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} working_set_sweep mib={mib} "
                f"correctness_delta.element_count="
                f"{actual_elements!r} does not match the expected "
                f"mib*262144={expected_elements}"
            )
        actual_mode = delta.get("comparison_mode")
        expected_mode = (
            "sha256_f32le" if mib >= 16 else "abs_and_rel"
        )
        if actual_mode != expected_mode:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path} working_set_sweep mib={mib} "
                f"correctness_delta.comparison_mode="
                f"{actual_mode!r} does not match the expected "
                f"{expected_mode!r} (digest mode required for "
                f"mib >= 16)"
            )
    return errors


def _recompute_perturbation_run_ledger_section(
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the cached perturbation_run_ledger section fields
    from the bundle's authoritative `perturbations[]`. The R12
    fail-closed pattern applied to the perturbation-harness surface."""
    perturbations = bundle.get("perturbations", [])
    admitted = sum(
        1 for p in perturbations
        if isinstance(p, dict) and p.get("result_kind") == "ok"
    )
    typed_unavail = sum(
        1 for p in perturbations
        if isinstance(p, dict)
        and p.get("result_kind") == "typed_unavailable"
    )
    families = (
        "cache_modifier_ca", "cache_modifier_cg", "cache_modifier_cs",
        "working_set_sweep",
    )
    per_family = {f: 0 for f in families}
    for record in perturbations:
        if not isinstance(record, dict):
            continue
        if record.get("result_kind") != "ok":
            continue
        f = record.get("family")
        if f in per_family:
            per_family[f] += 1
    typed_reasons = (
        "modifier_optimized_away",
        "correctness_equivalence_failed",
        "arch_feature_unavailable_l2_prefetch_64b",
        "arch_feature_unavailable_l2_prefetch_128b",
        "arch_feature_unavailable_l2_prefetch_256b",
        "arch_feature_unavailable_l2_persisting",
        "arch_feature_unavailable_eviction_buffer",
        "arch_feature_unavailable_carveout",
        "perturbation_compile_failed",
    )
    per_reason = {r: 0 for r in typed_reasons}
    for record in perturbations:
        if not isinstance(record, dict):
            continue
        if record.get("result_kind") != "typed_unavailable":
            continue
        r = record.get("typed_reason")
        if r in per_reason:
            per_reason[r] += 1
    sass_pass = sum(
        1 for p in perturbations
        if isinstance(p, dict)
        and p.get("result_kind") == "ok"
        and p.get("sass_confirmation_status") == "pass"
    )
    correctness_pass = sum(
        1 for p in perturbations
        if isinstance(p, dict)
        and p.get("result_kind") == "ok"
        and p.get("correctness_equivalence_status") == "pass"
    )
    return {
        "perturbation_admitted_count": admitted,
        "perturbation_typed_unavailable_count": typed_unavail,
        "perturbation_per_family_admitted_count": per_family,
        "perturbation_typed_unavailable_breakdown": per_reason,
        "sass_confirmation_pass_count": sass_pass,
        "correctness_equivalence_pass_count": correctness_pass,
    }


def _verify_perturbation_run_ledger_run_summaries(
    trace_dir: Path,
) -> list[str]:
    """task15 / the perturbation-harness clause R12-style fail-closed
    verifier.

    For every `trace_*.json` summary carrying a
    `perturbation_run_ledger_path`:

    1. Resolve the bundle path relative to the summary directory and
       load it.
    2. Strict-validate the on-disk bytes against
       `perturbation_run_ledger.schema.json` via
       `Draft202012Validator`. Schema rejection is
       `BudgetViolationUntracked`-equivalent.
    3. Confirm `manifest_pairing_index.{slug, shape_point,
       source_commit}` match the summary.
    4. Recompute the cached section fields from `perturbations[]`;
       reject any cached-vs-recomputed disagreement.
    5. For each admitted record, recompute the SASS confirmation
       hash from the on-disk file at `sass_artifact_path` and
       reject when the recomputation disagrees with the cached
       hash.
    6. For each admitted record, evaluate
       `_correctness_delta_exceeds_tolerance(delta)`; reject when
       the delta exceeds tolerance and no matching
       `correctness_equivalence_failed` typed-unavailable record
       exists.

    Summaries that do not carry a `perturbation_run_ledger_path`
    (the typed-unavailable arm; the producer recorded a
    `perturbation_run_ledger_skip_reason`) are not enforced against.
    """
    errors: list[str] = []
    validator = _load_schema_validator(
        "perturbation_run_ledger.schema.json"
    )
    for summary_path in trace_dir.rglob("trace_*.json"):
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        bundle_path_str = summary.get("perturbation_run_ledger_path")
        if bundle_path_str is None:
            continue
        bundle_path_p = Path(bundle_path_str)
        if not bundle_path_p.is_absolute():
            bundle_path_p = (
                summary_path.parent / bundle_path_p
            ).resolve()
        if not bundle_path_p.is_file():
            errors.append(
                f"BudgetViolationUntracked: run summary "
                f"{summary_path} references "
                f"perturbation_run_ledger_path={bundle_path_str!r} "
                f"but the file is not present at {bundle_path_p}"
            )
            continue
        try:
            with open(bundle_path_p) as f:
                bundle = json.load(f)
        except Exception as exc:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path_p} unreadable ({exc})"
            )
            continue
        schema_errors = list(validator.iter_errors(bundle))
        if schema_errors:
            first = schema_errors[0]
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path_p} fails strict schema "
                f"validation at {list(first.absolute_path)}: "
                f"{first.message}"
            )
            continue
        # R32 default-loop invariant per Codex's R31 review action
        # item #2: the bundle MUST carry exactly the 9 pinned
        # (family, parameter) pairs.
        errors.extend(
            _verify_perturbation_run_ledger_default_loop_parameter_set(
                bundle, bundle_path=bundle_path_p,
            )
        )
        # R33 working_set_sweep metadata invariants per Codex's R32
        # review: parameter.{allocation_bytes,warps,accesses,
        # stride_classes}, correctness_delta.element_count, and
        # correctness_delta.comparison_mode must match the
        # working_set_mib ladder. Without these the producer could
        # silently emit a record claiming a large MiB while the
        # actual comparison covered only 128 elements (the R31
        # failure class).
        errors.extend(
            _verify_perturbation_run_ledger_working_set_metadata(
                bundle, bundle_path=bundle_path_p,
            )
        )
        mpi = bundle.get("manifest_pairing_index", {}) or {}
        if mpi.get("slug") != summary.get("slug"):
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.slug={mpi.get('slug')!r} "
                f"does not match run summary slug "
                f"{summary.get('slug')!r}"
            )
        if mpi.get("shape_point") != summary.get("shape_point"):
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.shape_point="
                f"{mpi.get('shape_point')!r} does not match run "
                f"summary shape_point "
                f"{summary.get('shape_point')!r}"
            )
        cached_source_commit = summary.get(
            "perturbation_run_ledger_source_commit"
        )
        if mpi.get("source_commit") != cached_source_commit:
            errors.append(
                f"BudgetViolationUntracked: perturbation_run_ledger "
                f"bundle {bundle_path_p} "
                f"manifest_pairing_index.source_commit="
                f"{mpi.get('source_commit')!r} does not match "
                f"run summary "
                f"perturbation_run_ledger_source_commit="
                f"{cached_source_commit!r}"
            )
        cached_schema_version = summary.get(
            "perturbation_run_ledger_schema_version"
        )
        if cached_schema_version != bundle.get("schema_version"):
            errors.append(
                f"BudgetViolationUntracked: run summary "
                f"{summary_path} "
                f"perturbation_run_ledger_schema_version="
                f"{cached_schema_version!r} does not match "
                f"bundle schema_version="
                f"{bundle.get('schema_version')!r}"
            )
        recomputed = _recompute_perturbation_run_ledger_section(bundle)
        for field in (
            "perturbation_admitted_count",
            "perturbation_typed_unavailable_count",
            "perturbation_per_family_admitted_count",
            "perturbation_typed_unavailable_breakdown",
            "sass_confirmation_pass_count",
            "correctness_equivalence_pass_count",
        ):
            cached = summary.get(field)
            recomputed_value = recomputed[field]
            if cached != recomputed_value:
                errors.append(
                    f"BudgetViolationUntracked: run summary "
                    f"{summary_path} cached {field}={cached!r} "
                    f"disagrees with recomputed "
                    f"{recomputed_value!r} from bundle "
                    f"{bundle_path_p}"
                )
        for record in bundle.get("perturbations", []):
            if not isinstance(record, dict):
                continue
            if record.get("result_kind") != "ok":
                continue
            sass_artifact_path = record.get("sass_artifact_path")
            if not sass_artifact_path:
                continue
            sass_path_p = Path(sass_artifact_path)
            if not sass_path_p.is_absolute():
                sass_path_p = (
                    summary_path.parent / sass_path_p
                ).resolve()
                if not sass_path_p.is_file():
                    sass_path_p = (
                        bundle_path_p.parent / Path(sass_artifact_path)
                    ).resolve()
            if not sass_path_p.is_file():
                errors.append(
                    f"BudgetViolationUntracked: perturbation_run_ledger "
                    f"bundle {bundle_path_p} admitted record family="
                    f"{record.get('family')!r} parameter="
                    f"{record.get('parameter')!r} references "
                    f"sass_artifact_path={sass_artifact_path!r} but "
                    f"the file is not present at {sass_path_p}"
                )
                continue
            recomputed_hash, recomputed_count = (
                _recompute_sass_confirmation_hash(
                    sass_path_p,
                    record["family"],
                    record.get("parameter", {}),
                )
            )
            if recomputed_hash != record.get("sass_confirmation_hash"):
                errors.append(
                    f"BudgetViolationUntracked: perturbation_run_ledger "
                    f"bundle {bundle_path_p} admitted record family="
                    f"{record.get('family')!r} parameter="
                    f"{record.get('parameter')!r} "
                    f"sass_confirmation_hash="
                    f"{record.get('sass_confirmation_hash')!r} "
                    f"disagrees with recomputed "
                    f"{recomputed_hash!r} from "
                    f"{sass_path_p}"
                )
            if recomputed_count != record.get("sass_match_count"):
                errors.append(
                    f"BudgetViolationUntracked: perturbation_run_ledger "
                    f"bundle {bundle_path_p} admitted record family="
                    f"{record.get('family')!r} parameter="
                    f"{record.get('parameter')!r} "
                    f"sass_match_count="
                    f"{record.get('sass_match_count')!r} "
                    f"disagrees with recomputed "
                    f"{recomputed_count!r} from "
                    f"{sass_path_p}"
                )
            delta = record.get("correctness_delta") or {}
            if (
                delta
                and _correctness_delta_exceeds_tolerance(delta)
                and not _perturbation_typed_reason_present(
                    bundle,
                    family=record["family"],
                    parameter=record.get("parameter", {}),
                    typed_reason="correctness_equivalence_failed",
                )
            ):
                errors.append(
                    f"BudgetViolationUntracked: perturbation_run_ledger "
                    f"bundle {bundle_path_p} admitted record family="
                    f"{record.get('family')!r} parameter="
                    f"{record.get('parameter')!r} carries a "
                    f"correctness_delta exceeding tolerance with no "
                    f"matching correctness_equivalence_failed "
                    f"typed-unavailable record"
                )
    return errors


_TRACE_CONVERGENCE_RUN_SUMMARY_FIELDS: tuple[str, ...] = (
    "trace_convergence_confidence_path",
    "trace_convergence_schema_version",
    "trace_convergence_admitted",
    "trace_convergence_typed_reason",
    "trace_convergence_windows_observed",
    "trace_convergence_windows_converged",
    "trace_convergence_final_residual",
    "trace_convergence_mrc_quantile_set",
    "trace_convergence_source_commit",
    "trace_convergence_windows_digest",
    "trace_convergence_manifest_slug",
    "trace_convergence_manifest_shape_point",
    "trace_convergence_manifest_binary_sha256",
    "trace_convergence_manifest_source_commit",
    "trace_convergence_manifest_kernel_signature_sha256",
)
_TRACE_CONVERGENCE_MRC_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
_TRACE_CONVERGENCE_DISTANCE_THRESHOLD: float = 0.02
_TRACE_CONVERGENCE_K: int = 3


def _trace_convergence_canonical_windows_digest(
    windows: list[dict[str, Any]],
) -> str:
    """Recompute the canonical-JSON sha256 over `windows[]` per
    Section 8 of `task16-ac11-analysis.md` — sorted keys + compact
    separators + per-quantile sort. The verifier MUST recompute
    rather than trust the cached
    `trace_convergence_windows_digest` field; cached-vs-recomputed
    disagreement is `BudgetViolationUntracked`-equivalent (R36
    round-added)."""
    records: list[dict[str, Any]] = []
    for w in windows:
        mrc = sorted(
            (
                {
                    "hierarchy_layer": q["hierarchy_layer"],
                    "quantile": q["quantile"],
                    "mrc_hit_rate_fraction": q["mrc_hit_rate_fraction"],
                    "quantile_observation_count": q["quantile_observation_count"],
                }
                for q in w["mrc_at_quantile"]
            ),
            key=lambda q: (q["hierarchy_layer"], q["quantile"]),
        )
        records.append({
            "window_index": w["window_index"],
            "observation_count_cumulative": w["observation_count_cumulative"],
            "trace_bytes_cumulative": w["trace_bytes_cumulative"],
            "mrc_at_quantile": list(mrc),
            "distance_to_previous": w["distance_to_previous"],
            "converged": w["converged"],
        })
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _trace_convergence_trailing_converged_count(
    windows: list[dict[str, Any]],
) -> int:
    """Trailing consecutive count of windows whose `converged` is
    true. Used to recompute `trace_convergence_windows_converged`."""
    n = 0
    for w in reversed(windows):
        if w.get("converged"):
            n += 1
        else:
            break
    return n


def _verify_trace_convergence_run_summaries(trace_dir: Path) -> list[str]:
    """Verify the adaptive-sampler clause's run-summary contract per
    Section 10 of `task16-ac11-analysis.md`. Implements the 12
    rejection arms: missing `windows[]`; cached-vs-recomputed
    `windows_converged` disagreement; `result_kind=ok` with missing
    `final_residual`; `result_kind=ok` with `final_residual > 0.02`;
    `result_kind=ok` with fewer than `K=3` trailing converged
    windows; `trace_convergence_not_reached_budget_cap` without a
    matching `trace_volume_exceeded.json` sidecar; cached
    `trace_convergence_mrc_quantile_set` disagreeing with the
    canonical numeric set; any `admission.oneOf` branch containing
    extra properties; non-baseline window with
    `distance_to_previous == null`; baseline window 0 with
    `distance_to_previous != null`; cached
    `trace_convergence_windows_digest` disagreeing with recomputed
    canonical-JSON digest (R36 round-added); any cached
    `trace_convergence_manifest_<key>` disagreeing with the sidecar
    or run-summary identity (R36 round-added).
    """
    errors: list[str] = []
    schema_validator = _load_schema_validator("trace_convergence_confidence.schema.json")
    for run_summary_path in sorted(trace_dir.rglob("trace_*.json")):
        try:
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(run_summary, dict):
            continue
        if "trace_convergence_skip_reason" in run_summary:
            continue
        if run_summary.get("trace_convergence_confidence_path") is None:
            continue

        bundle_rel = run_summary["trace_convergence_confidence_path"]
        bundle_path = (run_summary_path.parent / bundle_rel).resolve()
        if not bundle_path.is_file():
            errors.append(
                f"trace_convergence:bundle_missing:{bundle_path}"
            )
            continue
        try:
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(
                f"trace_convergence:bundle_unreadable:{bundle_path}:{exc}"
            )
            continue

        # Strict schema admission — the schema's per-mode oneOf with
        # additionalProperties: false on every closed branch handles
        # the "extra properties" rejection arm (Section 10 #8).
        for err in schema_validator.iter_errors(bundle):
            errors.append(
                f"trace_convergence:schema_invalid:{bundle_path}:{err.message}"
            )

        windows = bundle.get("windows")
        if not isinstance(windows, list) or not windows:
            errors.append(
                f"trace_convergence:windows_missing:{bundle_path}"
            )
            continue

        # Section 10 #9: non-baseline window with distance_to_previous == null.
        # Section 10 #10: baseline window 0 with distance_to_previous != null.
        for w in windows:
            idx = w.get("window_index")
            dist = w.get("distance_to_previous")
            if idx == 0 and dist is not None:
                errors.append(
                    f"trace_convergence:baseline_distance_not_null:"
                    f"{bundle_path}:{dist}"
                )
            elif idx != 0 and dist is None:
                errors.append(
                    f"trace_convergence:non_baseline_distance_null:"
                    f"{bundle_path}:window_index={idx}"
                )

        # Section 10 #2: cached-vs-recomputed windows_converged.
        cached_converged = run_summary.get("trace_convergence_windows_converged")
        recomputed_converged = _trace_convergence_trailing_converged_count(windows)
        if (
            cached_converged is not None
            and cached_converged != recomputed_converged
        ):
            errors.append(
                f"trace_convergence:windows_converged_cache_drift:"
                f"{bundle_path}:cached={cached_converged}:"
                f"recomputed={recomputed_converged}"
            )

        admission = bundle.get("admission")
        if isinstance(admission, dict):
            result_kind = admission.get("result_kind")
            if result_kind == "ok":
                # Section 10 #3: missing final_residual.
                final_residual = admission.get("final_residual")
                if final_residual is None:
                    errors.append(
                        f"trace_convergence:final_residual_missing:{bundle_path}"
                    )
                # Section 10 #4: final_residual > threshold.
                elif float(final_residual) > _TRACE_CONVERGENCE_DISTANCE_THRESHOLD:
                    errors.append(
                        f"trace_convergence:final_residual_above_threshold:"
                        f"{bundle_path}:value={final_residual}:"
                        f"threshold={_TRACE_CONVERGENCE_DISTANCE_THRESHOLD}"
                    )
                # Section 10 #5: trailing converged < K.
                if recomputed_converged < _TRACE_CONVERGENCE_K:
                    errors.append(
                        f"trace_convergence:trailing_converged_below_k:"
                        f"{bundle_path}:trailing={recomputed_converged}:"
                        f"k={_TRACE_CONVERGENCE_K}"
                    )
            elif result_kind == "typed_unavailable":
                typed_reason = admission.get("typed_reason")
                # Section 10 #6: budget-cap without matching sidecar.
                if typed_reason == "trace_convergence_not_reached_budget_cap":
                    sidecar_rel = admission.get("trace_volume_exceeded_sidecar_path")
                    if sidecar_rel:
                        sidecar_path = (
                            run_summary_path.parent / sidecar_rel
                        ).resolve()
                        if not sidecar_path.is_file():
                            errors.append(
                                f"trace_convergence:budget_cap_sidecar_missing:"
                                f"{bundle_path}:expected={sidecar_path}"
                            )

        # Section 10 #7: cached MRC quantile set disagreement.
        cached_quantiles = run_summary.get("trace_convergence_mrc_quantile_set")
        if (
            cached_quantiles is not None
            and tuple(cached_quantiles) != _TRACE_CONVERGENCE_MRC_QUANTILES
        ):
            errors.append(
                f"trace_convergence:mrc_quantile_set_drift:"
                f"{bundle_path}:cached={cached_quantiles}:"
                f"canonical={list(_TRACE_CONVERGENCE_MRC_QUANTILES)}"
            )
        # Bundle-level pin must also match the canonical set.
        if (
            bundle.get("mrc_quantiles") is not None
            and tuple(bundle["mrc_quantiles"]) != _TRACE_CONVERGENCE_MRC_QUANTILES
        ):
            errors.append(
                f"trace_convergence:bundle_mrc_quantiles_drift:{bundle_path}:"
                f"value={bundle['mrc_quantiles']}"
            )

        # R36 #11: cached digest vs recomputed canonical-JSON digest.
        cached_digest = run_summary.get("trace_convergence_windows_digest")
        recomputed_digest = _trace_convergence_canonical_windows_digest(windows)
        if cached_digest is not None and cached_digest != recomputed_digest:
            errors.append(
                f"trace_convergence:windows_digest_drift:"
                f"{bundle_path}:cached={cached_digest}:"
                f"recomputed={recomputed_digest}"
            )

        # R36 #12: cached manifest tuple vs sidecar manifest_pairing_index
        # AND vs the enclosing trace run-summary's manifest identity.
        manifest = bundle.get("manifest_pairing_index")
        if isinstance(manifest, dict):
            for key, summary_field in (
                ("slug", "trace_convergence_manifest_slug"),
                ("shape_point", "trace_convergence_manifest_shape_point"),
                ("binary_sha256", "trace_convergence_manifest_binary_sha256"),
                ("source_commit", "trace_convergence_manifest_source_commit"),
                (
                    "kernel_signature_sha256",
                    "trace_convergence_manifest_kernel_signature_sha256",
                ),
            ):
                cached = run_summary.get(summary_field)
                authoritative = manifest.get(key)
                if cached is not None and cached != authoritative:
                    errors.append(
                        f"trace_convergence:manifest_tuple_cache_drift:"
                        f"{bundle_path}:field={summary_field}:"
                        f"cached={cached}:sidecar={authoritative}"
                    )
            # Enclosing run-summary identity binding: the convergence
            # sidecar's manifest_pairing_index MUST match the enclosing
            # trace run-summary's identity for every field present
            # (R38 closure of Codex's R37-review blocking side issue:
            # the previous version only compared slug + shape_point,
            # admitting a sidecar with stale binary_sha256 /
            # source_commit / kernel_signature_sha256 from a re-emit
            # at a different HEAD).
            for key, run_field in (
                ("slug", "slug"),
                ("shape_point", "shape_point"),
                ("binary_sha256", "binary_sha256"),
                ("source_commit", "source_commit"),
                ("kernel_signature_sha256", "kernel_signature_sha256"),
            ):
                run_value = run_summary.get(run_field)
                authoritative = manifest.get(key)
                if (
                    run_value is not None
                    and authoritative is not None
                    and run_value != authoritative
                ):
                    errors.append(
                        f"trace_convergence:manifest_run_summary_drift:"
                        f"{bundle_path}:field={key}:"
                        f"run_summary={run_value}:sidecar={authoritative}"
                    )

    return errors


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_cache_predictor_run_summaries(
    trace_dir: Path,
) -> list[str]:
    """The cache-predictor verifier hook. Mirrors the structural
    pattern of ``_verify_divergence_proxies_run_summaries`` (R4
    commit ``d2ec483``): walks per-shape ``trace_*.json`` summaries,
    follows ``cache_predictor_bundle_path``, strict-validates the
    on-disk bundle against ``cache_predictor.schema.json``, binds
    the bundle's 5-field ``manifest_pairing_index`` to the
    enclosing run summary, recomputes on-disk ``sector_stream``
    SHA-256 against the bundle's recorded digest, rejects forbidden
    modeled-hit keys in ``input_sidecar_digests``, enforces the
    bounded-arm ``hit_rate_lower <= hit_rate_upper`` invariant,
    and cross-checks the sibling ``hrd_profile.json``'s
    ``bundle_underconfigured`` flag against bundles that emit a
    scalar / bounded admission arm.

    Returns a list of error strings, each carrying a token from
    ``_VERIFIER_DEFECT_TOKENS``. Summaries that lack
    ``cache_predictor_bundle_path`` admit silently (the cache
    predictor was not run for that trace; legitimate skip path)."""
    errors: list[str] = []
    schema_validator = _load_schema_validator(
        "cache_predictor.schema.json"
    )
    for summary_path in trace_dir.rglob("trace_*.json"):
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        bundle_path_str = summary.get("cache_predictor_bundle_path")
        if bundle_path_str is None:
            # Field absent: the legitimate skip path. The cache
            # predictor did not emit a bundle for this trace, and
            # the summary correctly reflects that.
            continue
        bundle_path_p = Path(bundle_path_str)
        if not bundle_path_p.is_absolute():
            bundle_path_p = (
                summary_path.parent / bundle_path_p
            ).resolve()
        if not bundle_path_p.is_file():
            errors.append(
                f"cache_predictor_run_summary_missing_path: "
                f"summary {summary_path} sets "
                f"cache_predictor_bundle_path={bundle_path_str!r} "
                f"but the file is not present at {bundle_path_p}"
            )
            continue
        try:
            with open(bundle_path_p) as f:
                bundle = json.load(f)
        except Exception as exc:
            errors.append(
                f"cache_predictor_schema_invalid: cache_predictor "
                f"bundle {bundle_path_p} unreadable ({exc})"
            )
            continue
        # Forbidden-key check FIRST (before schema validation), so
        # the typed circular_predictor_input_rejected token surfaces
        # even when the schema's additionalProperties:false would
        # also reject. Mirrors the bidirectional contract: the
        # token must fire on every offending input shape.
        if isinstance(bundle, dict):
            digests = bundle.get("input_sidecar_digests") or {}
            if isinstance(digests, dict):
                forbidden = (
                    set(digests.keys())
                    & _CACHE_PREDICTOR_FORBIDDEN_DIGEST_KEYS
                )
                if forbidden:
                    errors.append(
                        f"cache_predictor_circular_predictor_input_rejected: "
                        f"bundle {bundle_path_p} input_sidecar_digests "
                        f"carries forbidden modeled-hit key(s) "
                        f"{sorted(forbidden)!r}; the non-circular "
                        f"discipline forbids consuming "
                        f"hrd_profile-derived modeled fields"
                    )
        schema_errors = list(schema_validator.iter_errors(bundle))
        if schema_errors:
            first = schema_errors[0]
            errors.append(
                f"cache_predictor_schema_invalid: bundle "
                f"{bundle_path_p} fails strict schema validation at "
                f"{list(first.absolute_path)}: {first.message}"
            )
            continue
        mpi = bundle.get("manifest_pairing_index", {})
        if not isinstance(mpi, dict):
            mpi = {}
        for key in (
            "slug",
            "shape_point",
            "binary_sha256",
            "source_commit",
            "kernel_signature_sha256",
        ):
            run_value = summary.get(key)
            bundle_value = mpi.get(key)
            # Fail-closed equality check: when the summary opts in
            # to a cache_predictor bundle (sets
            # cache_predictor_bundle_path), the 5 provenance fields
            # become REQUIRED on the summary side. A missing field
            # (run_value=None while bundle_value is set) is
            # indistinguishable from a foreign bundle being copied
            # into the summary's slot — the bundle's MPI references
            # an arch/slug/commit the summary never claimed. Treat
            # missing as mismatch via a plain != comparison; both-None
            # is False so the no-op admission path stays clean (and
            # is unreachable in practice since the schema validator
            # rejects bundles whose MPI lacks any required field
            # earlier in this loop).
            if run_value != bundle_value:
                errors.append(
                    f"cache_predictor_manifest_pairing_index_mismatch: "
                    f"bundle {bundle_path_p} "
                    f"manifest_pairing_index.{key}="
                    f"{bundle_value!r} does not match run summary "
                    f"{key}={run_value!r}"
                )
        # Per-arm digest cross-check. The schema admits an empty
        # input_sidecar_digests on the typed_unavailable arm; the
        # scalar / bounded arms require sector_stream. The verifier
        # recomputes sha256 on every present digest. Codex R54
        # code-review P2: `launch_parameters_sidecar` was missing
        # from the loop, so the gridDim-correction path's
        # launch-parameter sidecar could drift without triggering
        # `cache_predictor_sidecar_digest_drift`.
        digests = bundle.get("input_sidecar_digests") or {}
        for digest_name in (
            "sector_stream",
            "dynamic_trace_pb",
            "launch_parameters_sidecar",
        ):
            entry = digests.get(digest_name) if isinstance(digests, dict) else None
            if not isinstance(entry, dict):
                continue
            recorded_sha = entry.get("sha256")
            recorded_path = entry.get("path")
            if not recorded_sha or not recorded_path:
                continue
            on_disk_p = Path(recorded_path)
            if not on_disk_p.is_absolute():
                # The cache_predictor producer's portability path
                # normalization emits paths under _REPO_ROOT as
                # repo-relative so committed bundles survive
                # checkout-location and cross-host moves. Resolve
                # repo-relative paths from _REPO_ROOT first; fall
                # back to bundle-parent for backward compatibility
                # with bundles that may have used bundle-relative
                # paths.
                repo_rooted = (_REPO_ROOT / on_disk_p).resolve()
                if repo_rooted.is_file():
                    on_disk_p = repo_rooted
                else:
                    on_disk_p = (
                        bundle_path_p.parent / on_disk_p
                    ).resolve()
            if not on_disk_p.is_file():
                errors.append(
                    f"cache_predictor_sidecar_digest_drift: bundle "
                    f"{bundle_path_p} input_sidecar_digests."
                    f"{digest_name}.path={recorded_path!r} not "
                    f"present at {on_disk_p}"
                )
                continue
            actual_sha = _sha256_of_file(on_disk_p)
            if actual_sha != recorded_sha:
                errors.append(
                    f"cache_predictor_sidecar_digest_drift: bundle "
                    f"{bundle_path_p} input_sidecar_digests."
                    f"{digest_name}.sha256={recorded_sha!r} disagrees "
                    f"with on-disk sha256={actual_sha!r} of "
                    f"{on_disk_p}"
                )
        result_kind = bundle.get("result_kind")
        if result_kind == "bounded":
            lower = bundle.get("hit_rate_lower")
            upper = bundle.get("hit_rate_upper")
            if (
                isinstance(lower, (int, float))
                and isinstance(upper, (int, float))
                and lower > upper
            ):
                errors.append(
                    f"cache_predictor_bounded_arm_inversion: bundle "
                    f"{bundle_path_p} "
                    f"hit_rate_lower={lower!r} > "
                    f"hit_rate_upper={upper!r}"
                )
        if result_kind in {"scalar", "bounded"}:
            # Cross-check sibling hrd_profile.json's structural
            # bundle_underconfigured flag against the bundle's
            # admission arm. The bundle's siblings sit one level
            # up from the bundle file (the canonical layout puts
            # cache_predictor/, hrd_profile/, etc. under a
            # trace_<shape>/ directory).
            hrd_path = (
                bundle_path_p.parent.parent
                / "hrd_profile"
                / "hrd_profile.json"
            )
            if hrd_path.is_file():
                try:
                    with open(hrd_path) as f:
                        hrd = json.load(f)
                except Exception:
                    hrd = None
                if (
                    isinstance(hrd, dict)
                    and hrd.get("bundle_underconfigured") is True
                ):
                    errors.append(
                        f"cache_predictor_bundle_underconfigured_with_admitted_arm: "
                        f"bundle {bundle_path_p} emits "
                        f"result_kind={result_kind!r} but sibling "
                        f"{hrd_path} reports "
                        f"bundle_underconfigured=true; the runtime "
                        f"admission gate must refuse to emit a "
                        f"scalar / bounded arm under structural "
                        f"underconfiguration"
                    )
    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SimBench-BW trace-budget sidecar verifier")
    p.add_argument("--results", metavar="DIR", default=str(SUITE_ROOT / "results"),
                   help="Root directory to scan for trace artifacts (default: ./results)")
    p.add_argument(
        "--require-canonical-trace-summary",
        action="store_true",
        help=(
            "Enable the strict fresh-capture gate: every "
            "trace_<shape>/dynamic_trace.pb MUST carry the "
            "run_suite.py --mode trace binder (trace_<shape>.json + "
            "measurement_manifest.json). OFF by default because a "
            "host-local results/ tree legitimately holds legacy/audit "
            "captures from other paths; the capture/CI workflow passes "
            "this flag against the artifact root it just produced."
        ),
    )
    args = p.parse_args(argv)

    trace_dir = Path(args.results)
    if not trace_dir.exists():
        # Nothing to verify — empty results tree is a pass.
        print("ok")
        return 0

    errors = verify_trace_dir(
        trace_dir,
        require_canonical_trace_summary=args.require_canonical_trace_summary,
    )
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"{len(errors)} trace-budget violation(s)", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
