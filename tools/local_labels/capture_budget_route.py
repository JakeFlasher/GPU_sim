#!/usr/bin/env python3
"""Campaign-facing AC-9 trace-budget routing.

This module is deliberately *not* a GPU launcher.  A caller supplies callbacks
for the counting, selective, and (optionally) reduced-shape captures.  The
router binds those callback results to deterministic requests and then consumes
the existing certified trace-sampling implementation:

``full trace -> lossless full zstd -> counting/gate -> selective subset``.

Every admitted compressed artifact is a replay envelope.  It retains all
non-threadblock files byte-for-byte, retains each threadblock as ``.pb.zst``,
records per-file raw and retained hashes, carries a sealed schema-v2 sampling
sidecar, and can be materialized back into the exact source trace tree with
``materialize_replay_bundle``.  The input full trace is never modified.

The public route fails closed on empty/vacuous coverage, producer drops,
sampling/truncation evidence in the purported full trace, stale callback
receipts, malformed schema-v2 sidecars, payload-hash mismatches, and any
retention/staging budget overrun.  Gate-forbidden cells require an explicit
``reduced_shape`` policy plus callback, or the explicit ``typed_unavailable``
policy; they are never silently subset-sampled.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACE_SAMPLING = REPO_ROOT / "tools" / "trace_sampling"
_NVBIT = REPO_ROOT / "tools" / "nvbit_tracer"
for _path in (str(_TRACE_SAMPLING), str(_NVBIT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from admissibility_gate import MIN_SEGMENTS, M_SUBSET, classify  # noqa: E402
from compression import CompressionResult, _zstd, compress_trace_dir  # noqa: E402
from counting_pass import analyze_trace_dir, cell_stats_from_capture  # noqa: E402
from orchestrate import (  # noqa: E402
    _certainty_serialized_bytes,
    _same_launch_identity,
    _selected_block_ids,
    _selection_auxiliary,
    verify_capture_selection,
    verify_pass2_payload,
)
from selection import SelectionPlanError, plan_from_stats  # noqa: E402
from sidecar import (  # noqa: E402
    build_sidecar,
    sidecar_sha256,
    validate_sidecar,
)
from import_trace_contract import validate_trace_dir  # noqa: E402

from tools.local_labels import campaign_preflight as preflight  # noqa: E402


ROUTE_SCHEMA_VERSION = 1
ENVELOPE_SCHEMA_VERSION = 1
TRACER_VERSION = 5
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
IDENTITY_SOURCES = frozenset({"portable_lcd", "simbench_bw", "sol_execbench"})
REDUCED_VARIANTS = frozenset({"grid_stride_footprint_preserving", "dataset_shrunk"})
GATE_POLICIES = frozenset({"typed_unavailable", "reduced_shape"})

__all__ = [
    "BudgetRouteRefusal",
    "CaptureReceipt",
    "CaptureRequest",
    "FullTraceEvidence",
    "RouteCallbacks",
    "RouteConfig",
    "materialize_replay_bundle",
    "route_trace_budget",
    "validate_capacity_margins",
    "validate_full_trace_capture",
    "verify_replay_bundle",
    "write_route_result",
]

_PROTOCOL_FIELDS = frozenset({
    "protocol_version",
    "source_commit",
    "authority_cell_sha256",
    "capture_matrix_sha256",
    "preflight_sha256",
    "pins_sha256",
    "trace_budget_sha256",
    "capture_manifest_sha256",
    "binary_sha256",
    "tracer_so_sha256",
    "args_hash",
})
_BUDGET_FIELDS = frozenset({
    "max_trace_bytes", "max_wall_time_s", "sampling_policy",
})


class BudgetRouteRefusal(RuntimeError):
    """Typed AC-9 refusal.  No ``output_dir`` admission exists on failure."""

    def __init__(self, reason: str, detail: str | None = None):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RouteConfig:
    """AC-9 policy values not present in the source cell budget.

    ``selection_max_raw_bytes`` is the schema-v2 subset payload ceiling.  It
    must be no greater than the cell's retained ``max_trace_bytes``; reserving
    the difference for the replay envelope prevents payload-only accounting
    from consuming all retained space.
    """

    selection_max_raw_bytes: int
    staging_limit_bytes: int
    gate_forbidden_policy: str
    reduced_shape_variant: str | None = None
    seed: int = 0
    sample_size: int = MIN_SEGMENTS
    max_reconcile_attempts: int = 3


@dataclass(frozen=True)
class CaptureRequest:
    """A semantic request passed to a GPU-owning callback."""

    kind: str
    identity: dict[str, str]
    cell_id: str
    protocol_sha256: str
    full_trace_tree_sha256: str
    output_dir: Path
    staging_remaining_bytes: int
    attempt: int | None = None
    selected_block_ids: tuple[int, ...] = ()
    reduced_shape_variant: str | None = None

    def semantic_record(self) -> dict:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "identity": self.identity,
            "cell_id": self.cell_id,
            "protocol_sha256": self.protocol_sha256,
            "full_trace_tree_sha256": self.full_trace_tree_sha256,
            "staging_remaining_bytes": self.staging_remaining_bytes,
            "attempt": self.attempt,
            "selected_block_ids": list(self.selected_block_ids),
            "reduced_shape_variant": self.reduced_shape_variant,
        }

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(self.semantic_record())


@dataclass(frozen=True)
class CaptureReceipt:
    """Callback attestation consumed before any returned trace is trusted."""

    trace_dir: Path
    request_sha256: str
    binary_sha256: str
    tracer_so_sha256: str
    capture_manifest: Path | None = None
    matched_label_evidence: Path | None = None


CaptureCallback = Callable[[CaptureRequest], CaptureReceipt]


@dataclass(frozen=True)
class RouteCallbacks:
    counting_capture: CaptureCallback
    selective_capture: CaptureCallback
    reduced_shape_capture: CaptureCallback | None = None


@dataclass(frozen=True)
class FullTraceEvidence:
    stats: Any
    inventory: tuple[dict, ...]
    tree_sha256: str
    tree_bytes: int
    root: dict
    expected_records: int
    admitted_records: int
    coverage: dict


def _require_plain_int(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive " if positive else "non-negative "
        raise BudgetRouteRefusal("route_config_invalid", f"{name} must be a {qualifier}integer")
    return value


def _validate_identity(identity: Any, cell_id: str) -> dict[str, str]:
    if not isinstance(identity, dict) or set(identity) != {"roster_source", "slug"}:
        raise BudgetRouteRefusal(
            "identity_invalid", "identity must contain exactly roster_source and slug")
    source, slug = identity.get("roster_source"), identity.get("slug")
    if source not in IDENTITY_SOURCES or not isinstance(slug, str) or not slug:
        raise BudgetRouteRefusal("identity_invalid", repr(identity))
    if cell_id != slug:
        raise BudgetRouteRefusal(
            "cell_identity_mismatch", f"cell_id={cell_id!r}, identity.slug={slug!r}")
    return {"roster_source": source, "slug": slug}


def _require_sha(value: Any, name: str, *, source_commit: bool = False) -> str:
    pattern = SHA1_RE if source_commit else SHA256_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise BudgetRouteRefusal("protocol_provenance_invalid", f"{name}={value!r}")
    return value


def _validate_protocol(protocol: Any) -> dict:
    if not isinstance(protocol, dict) or set(protocol) != _PROTOCOL_FIELDS:
        got = sorted(protocol) if isinstance(protocol, dict) else type(protocol).__name__
        raise BudgetRouteRefusal(
            "protocol_provenance_invalid",
            f"expected exact fields {sorted(_PROTOCOL_FIELDS)}, got {got}",
        )
    if not isinstance(protocol["protocol_version"], str) or not protocol["protocol_version"]:
        raise BudgetRouteRefusal("protocol_provenance_invalid", "protocol_version is empty")
    _require_sha(protocol["source_commit"], "source_commit", source_commit=True)
    for field in _PROTOCOL_FIELDS - {"protocol_version", "source_commit"}:
        _require_sha(protocol[field], field)
    return json.loads(json.dumps(protocol, sort_keys=True))


def _validate_budget(budget: Any) -> dict:
    if not isinstance(budget, dict) or set(budget) != _BUDGET_FIELDS:
        got = sorted(budget) if isinstance(budget, dict) else type(budget).__name__
        raise BudgetRouteRefusal(
            "cell_budget_invalid", f"expected exact fields {sorted(_BUDGET_FIELDS)}, got {got}")
    _require_plain_int(budget["max_trace_bytes"], "max_trace_bytes", positive=True)
    _require_plain_int(budget["max_wall_time_s"], "max_wall_time_s", positive=True)
    if budget["sampling_policy"] != "off":
        raise BudgetRouteRefusal(
            "legacy_stride_evidence_refused",
            f"sampling_policy={budget['sampling_policy']!r}; only exact 'off' is accepted",
        )
    return dict(budget)


def _validate_config(config: RouteConfig, retained_limit: int) -> None:
    _require_plain_int(
        config.selection_max_raw_bytes, "selection_max_raw_bytes", positive=True)
    _require_plain_int(config.staging_limit_bytes, "staging_limit_bytes", positive=True)
    _require_plain_int(config.sample_size, "sample_size", positive=True)
    _require_plain_int(
        config.max_reconcile_attempts, "max_reconcile_attempts", positive=True)
    if type(config.seed) is not int:
        raise BudgetRouteRefusal("route_config_invalid", "seed must be an integer")
    if config.selection_max_raw_bytes > retained_limit:
        raise BudgetRouteRefusal(
            "route_config_invalid", "selection_max_raw_bytes exceeds retained max_trace_bytes")
    if config.gate_forbidden_policy not in GATE_POLICIES:
        raise BudgetRouteRefusal(
            "route_config_invalid", f"unknown gate policy {config.gate_forbidden_policy!r}")
    if config.gate_forbidden_policy == "reduced_shape":
        if config.reduced_shape_variant not in REDUCED_VARIANTS:
            raise BudgetRouteRefusal(
                "route_config_invalid", "reduced_shape policy requires an explicit certified variant")
    elif config.reduced_shape_variant is not None:
        raise BudgetRouteRefusal(
            "route_config_invalid", "typed_unavailable cannot declare a reduced shape")


def _safe_relative(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise BudgetRouteRefusal("artifact_path_escape", str(path)) from exc
    pure = PurePosixPath(rel)
    if rel in {"", "."} or pure.is_absolute() or ".." in pure.parts:
        raise BudgetRouteRefusal("artifact_path_escape", rel)
    return rel


def _tree_inventory(root: Path, *, allow_empty: bool = False) -> tuple[tuple[dict, ...], int, str]:
    if not root.is_dir() or root.is_symlink():
        raise BudgetRouteRefusal("trace_unreadable", f"not a real directory: {root}")
    rows: list[dict] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BudgetRouteRefusal("trace_symlink_refused", str(path))
        if path.is_dir():
            continue
        if not path.is_file():
            raise BudgetRouteRefusal("trace_special_file_refused", str(path))
        rel = _safe_relative(path, root)
        size = path.stat().st_size
        rows.append({"relative_path": rel, "size_bytes": size, "sha256": sha256_file(path)})
        total += size
    if not rows and not allow_empty:
        raise BudgetRouteRefusal("trace_empty", str(root))
    frozen = tuple(rows)
    return frozen, total, canonical_sha256(list(frozen))


def _tree_bytes(root: Path) -> int:
    return _tree_inventory(root, allow_empty=True)[1]


def _sampling_sidecar_paths(trace_dir: Path) -> tuple[Path, Path]:
    return (
        trace_dir / "trace_sampling.json",
        trace_dir / "extra_info" / "trace_sampling_capture.json",
    )


def _manifest_bytes(path: Path) -> tuple[dict, str]:
    if not path.is_file() or path.is_symlink():
        raise BudgetRouteRefusal("capture_manifest_unreadable", str(path))
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise BudgetRouteRefusal("capture_manifest_unreadable", str(exc)) from exc
    if not isinstance(value, dict):
        raise BudgetRouteRefusal("capture_manifest_unreadable", "root is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def _manifest_field_equal(manifest: dict, field: str, expected: Any) -> None:
    if manifest.get(field) != expected:
        raise BudgetRouteRefusal(
            "capture_manifest_stale",
            f"{field}: manifest={manifest.get(field)!r}, derived={expected!r}",
        )


def validate_full_trace_capture(
    trace_dir: Path,
    capture_manifest: Path,
    *,
    identity: dict[str, str],
    budget: dict,
    protocol: dict,
) -> FullTraceEvidence:
    """Re-derive exact nonempty one-launch coverage and bind its manifest."""
    trace_dir = Path(trace_dir)
    inventory, tree_bytes, tree_sha = _tree_inventory(trace_dir)
    relpaths = {row["relative_path"] for row in inventory}
    if "dynamic_trace.pb" not in relpaths or "stats.csv" not in relpaths:
        raise BudgetRouteRefusal(
            "trace_envelope_incomplete", "dynamic_trace.pb and stats.csv are required")
    if any(rel.endswith(".pb.zst") for rel in relpaths):
        raise BudgetRouteRefusal(
            "legacy_or_precompressed_trace_refused", "purported full trace contains .pb.zst")
    sampling_paths = _sampling_sidecar_paths(trace_dir)
    if any(path.exists() or path.is_symlink() for path in sampling_paths):
        raise BudgetRouteRefusal(
            "legacy_stride_evidence_refused",
            "purported full trace already carries sampling evidence",
        )
    threadblocks = sorted((trace_dir / "threadblocks").rglob("*.pb")) \
        if (trace_dir / "threadblocks").is_dir() else []
    if not threadblocks:
        raise BudgetRouteRefusal(
            "trace_payload_empty", "zero contract-validated threadblock protobufs")

    try:
        contract = validate_trace_dir(trace_dir)
    except Exception as exc:  # noqa: BLE001 - surfaced as typed evidence failure
        raise BudgetRouteRefusal("trace_contract_validator_failed", str(exc)) from exc
    if not bool(getattr(contract, "ok", False)):
        raise BudgetRouteRefusal(
            getattr(contract, "typed_reason", None) or "trace_contract_invalid",
            getattr(contract, "detail", None),
        )
    try:
        root = preflight._validate_root_trace(trace_dir / "dynamic_trace.pb", REPO_ROOT)
        details = preflight._parse_stats_admission_details(trace_dir / "stats.csv")
        stats = analyze_trace_dir(trace_dir, identity["slug"])
    except Exception as exc:  # noqa: BLE001
        raise BudgetRouteRefusal("trace_coverage_decode_failed", str(exc)) from exc
    expected, admitted = details.get("expected"), details.get("admitted")
    coverage = preflight._analyze_full_trace_coverage(
        trace_dir, root, expected, admitted, details.get("row_count", 0))
    if root.get("status") != preflight.STATUS_OK:
        raise BudgetRouteRefusal("trace_root_invalid", root.get("detail", "unknown"))
    if coverage.get("status") != preflight.STATUS_OK:
        raise BudgetRouteRefusal(
            "trace_full_coverage_invalid", coverage.get("detail", "unknown"))
    if type(expected) is not int or expected <= 0 or type(admitted) is not int:
        raise BudgetRouteRefusal(
            "trace_stats_invalid", f"expected={expected!r}, admitted={admitted!r}")
    if admitted != expected:
        raise BudgetRouteRefusal(
            "trace_truncated_or_dropped", f"admitted={admitted}, expected={expected}")
    if stats.n_blocks <= 0 or stats.total_records <= 0:
        raise BudgetRouteRefusal("trace_payload_empty", "decoded stats are empty")

    manifest, manifest_sha = _manifest_bytes(capture_manifest)
    _manifest_field_equal(manifest, "identity", identity)
    _manifest_field_equal(manifest, "binary_sha256", protocol["binary_sha256"])
    _manifest_field_equal(manifest, "tracer_so_sha256", protocol["tracer_so_sha256"])
    _manifest_field_equal(manifest, "max_trace_bytes", budget["max_trace_bytes"])
    _manifest_field_equal(manifest, "trace_tree_bytes", tree_bytes)
    _manifest_field_equal(manifest, "launch_count", root["launch_count"])
    _manifest_field_equal(manifest, "threadblocks_pb_count", len(threadblocks))
    for field in (
        "grid_threadblock_count", "decoded_threadblock_count",
        "decoded_record_count", "zero_record_threadblock_count", "stats_row_count",
    ):
        _manifest_field_equal(manifest, field, coverage[field])
    _manifest_field_equal(manifest, "expected_records", expected)
    _manifest_field_equal(manifest, "admitted_records", admitted)
    _manifest_field_equal(manifest, "drop_rate", 0.0)
    if manifest_sha != protocol["capture_manifest_sha256"]:
        raise BudgetRouteRefusal(
            "capture_manifest_stale",
            f"sha256={manifest_sha}, expected={protocol['capture_manifest_sha256']}",
        )
    return FullTraceEvidence(
        stats=stats, inventory=inventory, tree_sha256=tree_sha,
        tree_bytes=tree_bytes, root=root, expected_records=expected,
        admitted_records=admitted, coverage=coverage,
    )


def _remaining_staging(full_trace_dir: Path, work_root: Path, limit: int) -> int:
    used = _tree_bytes(full_trace_dir)
    if work_root.exists():
        used += _tree_bytes(work_root)
    if used > limit:
        raise BudgetRouteRefusal(
            "staging_limit_exceeded", f"used={used}, limit={limit}")
    return limit - used


def _validate_sampling_root(trace_dir: Path, expected_stats: Any, *, mode: int) -> None:
    if not trace_dir.is_dir() or trace_dir.is_symlink():
        raise BudgetRouteRefusal("capture_trace_missing", str(trace_dir))
    _tree_inventory(trace_dir)
    root = preflight._validate_root_trace(trace_dir / "dynamic_trace.pb", REPO_ROOT)
    if root.get("status") != preflight.STATUS_OK or root.get("launch_count") != 1:
        raise BudgetRouteRefusal(
            "sampling_launch_envelope_invalid", root.get("detail", repr(root)))
    launch = root["launches"][0]
    if (launch.get("kernel_name") != expected_stats.kernel_name
            or tuple(launch.get("grid_dim", ())) != tuple(expected_stats.grid)
            or tuple(launch.get("block_dim", ())) != tuple(expected_stats.block)):
        raise BudgetRouteRefusal(
            "sampling_launch_identity_drift",
            f"root={launch}, expected kernel/grid/block="
            f"{expected_stats.kernel_name!r}/{expected_stats.grid}/{expected_stats.block}",
        )
    sidecar_path = trace_dir / "extra_info" / "trace_sampling_capture.json"
    try:
        capture = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise BudgetRouteRefusal("capture_sidecar_unreadable", str(exc)) from exc
    if not isinstance(capture, dict) or type(capture.get("mode")) is not int \
            or capture["mode"] != mode:
        raise BudgetRouteRefusal(
            "capture_mode_mismatch", f"expected={mode}, got={capture.get('mode')!r}")
    blocks = list((trace_dir / "threadblocks").rglob("*.pb")) \
        if (trace_dir / "threadblocks").is_dir() else []
    if mode == 1 and blocks:
        raise BudgetRouteRefusal(
            "counting_payload_unexpected", f"mode-1 capture dumped {len(blocks)} blocks")


def _validate_receipt(
    request: CaptureRequest,
    receipt: Any,
    protocol: dict,
) -> Path:
    if not isinstance(receipt, CaptureReceipt):
        raise BudgetRouteRefusal(
            "capture_receipt_invalid", f"callback returned {type(receipt).__name__}")
    if receipt.request_sha256 != request.request_sha256:
        raise BudgetRouteRefusal(
            "capture_receipt_stale",
            f"request={receipt.request_sha256}, expected={request.request_sha256}",
        )
    if receipt.binary_sha256 != protocol["binary_sha256"]:
        raise BudgetRouteRefusal("capture_receipt_stale", "binary_sha256 mismatch")
    if receipt.tracer_so_sha256 != protocol["tracer_so_sha256"]:
        raise BudgetRouteRefusal("capture_receipt_stale", "tracer_so_sha256 mismatch")
    expected = request.output_dir.resolve(strict=False)
    observed = Path(receipt.trace_dir).resolve(strict=False)
    if observed != expected:
        raise BudgetRouteRefusal(
            "capture_receipt_stale", f"trace_dir={observed}, requested={expected}")
    return Path(receipt.trace_dir)


def _validate_matched_label_evidence(
    path: Path,
    reduced_trace_dir: Path,
    *,
    identity: dict,
    reduced_shape_variant: str,
    protocol: dict,
) -> dict:
    """Require the reduced trace to carry its actually matched label artifacts."""
    expected_path = reduced_trace_dir / "extra_info" / "matched_reduced_label_evidence.json"
    if Path(path).resolve(strict=False) != expected_path.resolve(strict=False) \
            or not expected_path.is_file() or expected_path.is_symlink():
        raise BudgetRouteRefusal(
            "matched_reduced_label_evidence_missing", str(expected_path))
    try:
        evidence = json.loads(expected_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise BudgetRouteRefusal(
            "matched_reduced_label_evidence_invalid", str(exc)) from exc
    required = {
        "schema_version", "kind", "identity", "reduced_shape_variant",
        "args_hash", "binary_sha256", "artifacts",
    }
    if not isinstance(evidence, dict) or set(evidence) != required \
            or evidence.get("schema_version") != 1 \
            or evidence.get("kind") != "matched_reduced_shape_label_evidence" \
            or evidence.get("identity") != identity \
            or evidence.get("reduced_shape_variant") != reduced_shape_variant \
            or evidence.get("args_hash") != protocol["args_hash"] \
            or evidence.get("binary_sha256") != protocol["binary_sha256"]:
        raise BudgetRouteRefusal(
            "matched_reduced_label_evidence_invalid", "closed provenance mismatch")
    artifacts = evidence.get("artifacts")
    required_kinds = {"label_sidecar", "label_run_manifest", "ncu_report"}
    if not isinstance(artifacts, list) or len(artifacts) != len(required_kinds):
        raise BudgetRouteRefusal(
            "matched_reduced_label_evidence_invalid", "three label artifacts are required")
    seen_kinds: set[str] = set()
    seen_paths: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or set(artifact) != {
            "artifact_kind", "relative_path", "sha256", "size_bytes",
        }:
            raise BudgetRouteRefusal(
                "matched_reduced_label_evidence_invalid", f"artifact[{index}] fields")
        kind, rel = artifact["artifact_kind"], artifact["relative_path"]
        if kind not in required_kinds or kind in seen_kinds:
            raise BudgetRouteRefusal(
                "matched_reduced_label_evidence_invalid", f"artifact kind {kind!r}")
        if not isinstance(rel, str) or rel in seen_paths:
            raise BudgetRouteRefusal(
                "matched_reduced_label_evidence_invalid", f"artifact path {rel!r}")
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != rel:
            raise BudgetRouteRefusal(
                "matched_reduced_label_evidence_invalid", f"unsafe artifact path {rel!r}")
        artifact_path = reduced_trace_dir / rel
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise BudgetRouteRefusal("matched_reduced_label_artifact_missing", rel)
        size = artifact_path.stat().st_size
        if type(artifact.get("size_bytes")) is not int or size <= 0 \
                or artifact["size_bytes"] != size \
                or not isinstance(artifact.get("sha256"), str) \
                or SHA256_RE.fullmatch(artifact["sha256"]) is None \
                or artifact["sha256"] != sha256_file(artifact_path):
            raise BudgetRouteRefusal(
                "matched_reduced_label_artifact_stale", rel)
        seen_kinds.add(kind)
        seen_paths.add(rel)
    if seen_kinds != required_kinds:
        raise BudgetRouteRefusal(
            "matched_reduced_label_evidence_invalid", repr(sorted(seen_kinds)))
    return evidence


def _invoke_capture(
    callback: CaptureCallback,
    request: CaptureRequest,
    protocol: dict,
) -> tuple[Path, CaptureReceipt]:
    if request.output_dir.exists() or request.output_dir.is_symlink():
        raise BudgetRouteRefusal("capture_output_exists", str(request.output_dir))
    receipt = callback(request)
    trace_dir = _validate_receipt(request, receipt, protocol)
    return trace_dir, receipt


def _copy_identity_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)


def _seal_sidecar_block_manifest(sidecar: dict, block_manifest_sha256: str) -> dict:
    sealed = json.loads(json.dumps(sidecar))
    sealed["hashes"].pop("sidecar_sha256", None)
    sealed["hashes"]["per_block_raw_sha256_manifest"] = block_manifest_sha256
    sealed["hashes"]["sidecar_sha256"] = sidecar_sha256(sealed)
    validate_sidecar(sealed)
    return sealed


def _build_compressed_bundle(
    source_trace: Path,
    destination: Path,
    sidecar_factory: Callable[[CompressionResult, str], dict],
) -> dict:
    """Create and independently verify a replayable zstd trace envelope."""
    if destination.exists() or destination.is_symlink():
        raise BudgetRouteRefusal("route_candidate_exists", str(destination))
    source_inventory, source_bytes, source_tree_sha = _tree_inventory(source_trace)
    source_paths = {row["relative_path"] for row in source_inventory}
    if "trace_sampling.json" in source_paths:
        raise BudgetRouteRefusal(
            "stale_sampling_sidecar", f"source capture already has trace_sampling.json: {source_trace}")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        try:
            comp = compress_trace_dir(source_trace, destination)
        except BudgetRouteRefusal:
            # A typed refusal supplied by a stricter compression implementation
            # already carries the route semantics and must not be collapsed into
            # a generic codec failure.
            raise
        except (OSError, RuntimeError) as exc:
            # The compression layer reports missing/unusable zstd, read/write
            # failures, and failed byte-exact round trips through these types.
            # Convert them at the layer boundary so no raw host exception can
            # escape an AC-9 route decision.
            raise BudgetRouteRefusal(
                "lossless_compression_failed",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        entries: list[dict] = []
        pb_raw_manifest: list[dict] = []
        by_rel = {row["relative_path"]: row for row in source_inventory}
        for rel, row in sorted(by_rel.items()):
            src = source_trace / rel
            if rel.startswith("threadblocks/") and rel.endswith(".pb"):
                retained_rel = rel + ".zst"
                encoding = "zstd"
                pb_raw_manifest.append({
                    "relative_path": rel,
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                })
            else:
                retained_rel = rel
                encoding = "identity"
                _copy_identity_file(src, destination / retained_rel)
            retained = destination / retained_rel
            entries.append({
                "source_relative_path": rel,
                "retained_relative_path": retained_rel,
                "encoding": encoding,
                "raw_size_bytes": row["size_bytes"],
                "raw_sha256": row["sha256"],
                "retained_size_bytes": retained.stat().st_size,
                "retained_sha256": sha256_file(retained),
            })
        block_manifest_sha = canonical_sha256(pb_raw_manifest)
        sidecar = _seal_sidecar_block_manifest(
            sidecar_factory(comp, block_manifest_sha), block_manifest_sha)
        sidecar_path = destination / "trace_sampling.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        generated_entry = {
            "source_relative_path": None,
            "retained_relative_path": "trace_sampling.json",
            "encoding": "generated",
            "raw_size_bytes": None,
            "raw_sha256": None,
            "retained_size_bytes": sidecar_path.stat().st_size,
            "retained_sha256": sha256_file(sidecar_path),
        }
        entries.append(generated_entry)
        entries.sort(key=lambda row: row["retained_relative_path"])
        envelope = {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "kind": "ac9_replayable_zstd_trace",
            "method": sidecar["method"],
            "source_tree_sha256": source_tree_sha,
            "source_tree_bytes": source_bytes,
            "source_file_count": len(source_inventory),
            "threadblock_count": comp.n_blocks,
            "decompressed_threadblocks_sha256": comp.decompressed_sha256,
            "per_block_raw_sha256_manifest": block_manifest_sha,
            "entries": entries,
            "entries_sha256": canonical_sha256(entries),
        }
        envelope["envelope_sha256"] = canonical_sha256(envelope)
        envelope_path = destination / "ac9_replay_envelope.json"
        envelope_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        verification = verify_replay_bundle(destination)
        return {
            "sidecar": sidecar,
            "envelope": envelope,
            "verification": verification,
        }
    except Exception:
        # The path was exclusively created by this call; removing a rejected
        # partial candidate cannot destroy pre-existing user evidence.
        shutil.rmtree(destination, ignore_errors=True)
        raise


def verify_replay_bundle(bundle_dir: Path) -> dict:
    """Verify every retained/raw hash and the sealed schema-v2 payload hash."""
    bundle_dir = Path(bundle_dir)
    envelope_path = bundle_dir / "ac9_replay_envelope.json"
    try:
        envelope = json.loads(envelope_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise BudgetRouteRefusal("replay_envelope_unreadable", str(exc)) from exc
    required = {
        "schema_version", "kind", "method", "source_tree_sha256",
        "source_tree_bytes", "source_file_count", "threadblock_count",
        "decompressed_threadblocks_sha256", "per_block_raw_sha256_manifest",
        "entries", "entries_sha256", "envelope_sha256",
    }
    if not isinstance(envelope, dict) or set(envelope) != required \
            or envelope.get("schema_version") != ENVELOPE_SCHEMA_VERSION \
            or envelope.get("kind") != "ac9_replayable_zstd_trace":
        raise BudgetRouteRefusal("replay_envelope_malformed", "closed envelope mismatch")
    entries = envelope.get("entries")
    if not isinstance(entries, list) or not entries \
            or canonical_sha256(entries) != envelope.get("entries_sha256"):
        raise BudgetRouteRefusal("replay_envelope_malformed", "entries seal mismatch")
    claimed_envelope_sha = envelope["envelope_sha256"]
    unsigned_envelope = dict(envelope)
    unsigned_envelope.pop("envelope_sha256")
    if claimed_envelope_sha != canonical_sha256(unsigned_envelope):
        raise BudgetRouteRefusal("replay_envelope_malformed", "envelope seal mismatch")
    retained_seen: set[str] = set()
    source_seen: set[str] = set()
    raw_inventory: list[dict] = []
    pb_manifest: list[dict] = []
    payload_digest = hashlib.sha256()
    payload_raw = payload_compressed = payload_count = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "source_relative_path", "retained_relative_path", "encoding",
            "raw_size_bytes", "raw_sha256", "retained_size_bytes",
            "retained_sha256",
        }:
            raise BudgetRouteRefusal(
                "replay_envelope_malformed", f"entry[{index}] fields")
        retained_rel = entry["retained_relative_path"]
        if not isinstance(retained_rel, str) or retained_rel in retained_seen:
            raise BudgetRouteRefusal(
                "replay_envelope_malformed", f"duplicate/malformed retained path {retained_rel!r}")
        retained_seen.add(retained_rel)
        retained_path = bundle_dir / retained_rel
        if not retained_path.is_file() or retained_path.is_symlink():
            raise BudgetRouteRefusal("replay_payload_missing", retained_rel)
        if retained_path.stat().st_size != entry["retained_size_bytes"] \
                or sha256_file(retained_path) != entry["retained_sha256"]:
            raise BudgetRouteRefusal("replay_retained_hash_mismatch", retained_rel)
        encoding = entry["encoding"]
        if encoding == "generated":
            if entry["source_relative_path"] is not None \
                    or retained_rel != "trace_sampling.json":
                raise BudgetRouteRefusal("replay_envelope_malformed", "invalid generated entry")
            continue
        source_rel = entry["source_relative_path"]
        if not isinstance(source_rel, str) or source_rel in source_seen:
            raise BudgetRouteRefusal("replay_envelope_malformed", "missing source path")
        source_seen.add(source_rel)
        try:
            raw = (_zstd(retained_path.read_bytes(), decompress=True)
                   if encoding == "zstd" else retained_path.read_bytes())
        except Exception as exc:  # noqa: BLE001
            raise BudgetRouteRefusal("replay_decompression_failed", f"{retained_rel}: {exc}") from exc
        if encoding not in {"zstd", "identity"}:
            raise BudgetRouteRefusal("replay_envelope_malformed", f"encoding={encoding!r}")
        if encoding == "zstd" and retained_rel != source_rel + ".zst":
            raise BudgetRouteRefusal(
                "replay_envelope_malformed", "zstd path does not mirror its raw source")
        if encoding == "identity" and retained_rel != source_rel:
            raise BudgetRouteRefusal(
                "replay_envelope_malformed", "identity path does not mirror its source")
        if len(raw) != entry["raw_size_bytes"] \
                or hashlib.sha256(raw).hexdigest() != entry["raw_sha256"]:
            raise BudgetRouteRefusal("replay_raw_hash_mismatch", source_rel)
        raw_row = {
            "relative_path": source_rel,
            "size_bytes": len(raw),
            "sha256": entry["raw_sha256"],
        }
        raw_inventory.append(raw_row)
        if encoding == "zstd":
            if not source_rel.startswith("threadblocks/") or not source_rel.endswith(".pb"):
                raise BudgetRouteRefusal("replay_envelope_malformed", source_rel)
            pb_manifest.append(raw_row)
            payload_digest.update(raw)
            payload_raw += len(raw)
            payload_compressed += retained_path.stat().st_size
            payload_count += 1
    raw_inventory.sort(key=lambda row: row["relative_path"])
    pb_manifest.sort(key=lambda row: row["relative_path"])
    if canonical_sha256(raw_inventory) != envelope["source_tree_sha256"] \
            or sum(row["size_bytes"] for row in raw_inventory) != envelope["source_tree_bytes"] \
            or len(raw_inventory) != envelope["source_file_count"]:
        raise BudgetRouteRefusal("replay_source_tree_mismatch")
    if canonical_sha256(pb_manifest) != envelope["per_block_raw_sha256_manifest"] \
            or len(pb_manifest) != envelope["threadblock_count"]:
        raise BudgetRouteRefusal("replay_block_manifest_mismatch")
    payload_sha = payload_digest.hexdigest()
    if payload_sha != envelope["decompressed_threadblocks_sha256"]:
        raise BudgetRouteRefusal("replay_payload_hash_mismatch")
    if payload_count <= 0 or payload_raw <= 0:
        raise BudgetRouteRefusal("replay_payload_empty")
    try:
        sidecar = json.loads((bundle_dir / "trace_sampling.json").read_text())
        validate_sidecar(sidecar)
    except Exception as exc:  # noqa: BLE001
        raise BudgetRouteRefusal("trace_sampling_sidecar_invalid", str(exc)) from exc
    hashes = sidecar["hashes"]
    if sidecar["method"] != envelope["method"]:
        raise BudgetRouteRefusal("trace_sampling_method_mismatch")
    if hashes.get("decompressed_payload_sha256") != payload_sha \
            or hashes.get("per_block_raw_sha256_manifest") != canonical_sha256(pb_manifest):
        raise BudgetRouteRefusal("trace_sampling_payload_hash_mismatch")
    if sidecar["bytes"]["raw"] != payload_raw \
            or sidecar["bytes"]["compressed"] != payload_compressed:
        raise BudgetRouteRefusal("trace_sampling_byte_envelope_mismatch")
    inventory, retained_bytes, retained_tree_sha = _tree_inventory(bundle_dir)
    actual_paths = {row["relative_path"] for row in inventory}
    expected_paths = retained_seen | {"ac9_replay_envelope.json"}
    if actual_paths != expected_paths:
        raise BudgetRouteRefusal(
            "replay_envelope_unlisted_file",
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}",
        )
    return {
        "status": "ok",
        "method": sidecar["method"],
        "schema_version": sidecar["schema_version"],
        "sidecar_sha256": hashes["sidecar_sha256"],
        "payload_sha256": payload_sha,
        "threadblock_count": payload_count,
        "retained_bytes": retained_bytes,
        "retained_tree_sha256": retained_tree_sha,
        "retained_file_count": len(inventory),
    }


def materialize_replay_bundle(bundle_dir: Path, output_trace_dir: Path) -> dict:
    """Exclusively reconstruct the raw replay tree and verify its source seal."""
    verification = verify_replay_bundle(bundle_dir)
    output_trace_dir = Path(output_trace_dir)
    if output_trace_dir.exists() or output_trace_dir.is_symlink():
        raise BudgetRouteRefusal("materialize_output_exists", str(output_trace_dir))
    envelope = json.loads((Path(bundle_dir) / "ac9_replay_envelope.json").read_text())
    output_trace_dir.mkdir(parents=True, exist_ok=False)
    try:
        generated: list[dict] = []
        for entry in envelope["entries"]:
            retained = Path(bundle_dir) / entry["retained_relative_path"]
            if entry["encoding"] == "generated":
                generated.append(entry)
                continue
            destination = output_trace_dir / entry["source_relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw = (_zstd(retained.read_bytes(), decompress=True)
                   if entry["encoding"] == "zstd" else retained.read_bytes())
            with destination.open("xb") as stream:
                stream.write(raw)
        inventory, tree_bytes, tree_sha = _tree_inventory(output_trace_dir)
        if tree_sha != envelope["source_tree_sha256"] \
                or tree_bytes != envelope["source_tree_bytes"]:
            raise BudgetRouteRefusal("materialized_source_tree_mismatch")
        # Copy the generated sidecar only after checking the exact source-tree
        # seal.  It must remain beside a subset replay to sanction incomplete
        # full-grid coverage; lossless-full also retains it as provenance.
        for entry in generated:
            retained = Path(bundle_dir) / entry["retained_relative_path"]
            destination = output_trace_dir / entry["retained_relative_path"]
            _copy_identity_file(retained, destination)
        return {
            "status": "ok",
            "source_tree_sha256": tree_sha,
            "source_tree_bytes": tree_bytes,
            "source_file_count": len(inventory),
            "bundle_verification": verification,
        }
    except Exception:
        shutil.rmtree(output_trace_dir, ignore_errors=True)
        raise


def _build_unavailable_bundle(
    counting_trace: Path,
    destination: Path,
    sidecar: dict,
    *,
    gate: Any,
    typed_reason: str,
    terminal_detail: str | None,
) -> dict:
    if destination.exists() or destination.is_symlink():
        raise BudgetRouteRefusal("route_candidate_exists", str(destination))
    destination.mkdir(parents=True, exist_ok=False)
    try:
        evidence_root = destination / "counting_evidence"
        for source in sorted(counting_trace.rglob("*")):
            if source.is_symlink():
                raise BudgetRouteRefusal("trace_symlink_refused", str(source))
            if source.is_file():
                _copy_identity_file(source, evidence_root / source.relative_to(counting_trace))
        validate_sidecar(sidecar)
        sidecar_path = destination / "trace_sampling.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        evidence_inventory, evidence_bytes, evidence_sha = _tree_inventory(evidence_root)
        manifest = {
            "schema_version": 1,
            "kind": "ac9_typed_unavailable_evidence",
            "method": "unavailable",
            "typed_reason": typed_reason,
            "terminal_detail": terminal_detail,
            "gate_verdict": gate.verdict,
            "forbidding_gates": list(gate.forbidding_gates),
            "counting_evidence_tree_sha256": evidence_sha,
            "counting_evidence_bytes": evidence_bytes,
            "counting_evidence_files": list(evidence_inventory),
            "trace_sampling_sidecar_sha256": sidecar["hashes"]["sidecar_sha256"],
        }
        (destination / "ac9_unavailable_evidence.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {"sidecar": sidecar, "manifest": manifest}
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _artifact_kind(relative_path: str) -> str:
    if relative_path.endswith("dynamic_trace.pb"):
        return "trace_root"
    if relative_path.endswith("stats.csv"):
        return "trace_stats"
    if relative_path.endswith(".pb.zst"):
        return "trace_threadblock"
    return "trace_auxiliary"


def _result_record(
    *, identity: dict, cell_id: str, budget: dict, config: RouteConfig,
    protocol: dict, full: FullTraceEvidence, output_dir: Path,
    method: str, outcome: str, typed_reason: str | None,
) -> dict:
    inventory, retained_bytes, retained_sha = _tree_inventory(output_dir)
    matched_label_kinds: dict[str, str] = {}
    matched_evidence_path = (
        output_dir / "extra_info" / "matched_reduced_label_evidence.json")
    if matched_evidence_path.is_file() and not matched_evidence_path.is_symlink():
        matched_evidence = json.loads(matched_evidence_path.read_text())
        for artifact in matched_evidence.get("artifacts", []):
            if isinstance(artifact, dict):
                matched_label_kinds[artifact.get("relative_path")] = artifact.get(
                    "artifact_kind")
    artifacts = [
        {
            **row,
            "artifact_kind": matched_label_kinds.get(
                row["relative_path"], _artifact_kind(row["relative_path"])),
        }
        for row in inventory
    ]
    sidecar_path = output_dir / "trace_sampling.json"
    sidecar = json.loads(sidecar_path.read_text())
    validate_sidecar(sidecar)
    record = {
        "schema_version": ROUTE_SCHEMA_VERSION,
        "kind": "ac9_trace_budget_route",
        "cell_id": cell_id,
        "identity": identity,
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "budget": budget,
        "route_config": {
            "selection_max_raw_bytes": config.selection_max_raw_bytes,
            "staging_limit_bytes": config.staging_limit_bytes,
            "gate_forbidden_policy": config.gate_forbidden_policy,
            "reduced_shape_variant": config.reduced_shape_variant,
            "seed": config.seed,
            "sample_size": config.sample_size,
            "max_reconcile_attempts": config.max_reconcile_attempts,
        },
        "full_trace": {
            "tree_sha256": full.tree_sha256,
            "tree_bytes": full.tree_bytes,
            "threadblock_count": full.coverage["threadblocks_pb_count"],
            "decoded_record_count": full.coverage["decoded_record_count"],
            "expected_records": full.expected_records,
            "admitted_records": full.admitted_records,
        },
        "outcome": outcome,
        "typed_reason": typed_reason,
        "method": method,
        "trace_sampling_schema_version": sidecar["schema_version"],
        "trace_sampling_sidecar_sha256": sidecar["hashes"]["sidecar_sha256"],
        "retained": {
            "tree_sha256": retained_sha,
            "bytes": retained_bytes,
            "file_count": len(artifacts),
            "artifacts": artifacts,
        },
    }
    record["result_sha256"] = canonical_sha256(record)
    return record


def write_route_result(record: dict, path: Path) -> None:
    """Exclusively write a sealed route result without overwriting evidence."""
    expected = record.get("result_sha256") if isinstance(record, dict) else None
    unsigned = dict(record) if isinstance(record, dict) else {}
    unsigned.pop("result_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise BudgetRouteRefusal("route_result_seal_invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _accept_candidate(candidate: Path, output_dir: Path, work_root: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise BudgetRouteRefusal("route_output_exists", str(output_dir))
    candidate.rename(output_dir)
    shutil.rmtree(work_root, ignore_errors=True)


def route_trace_budget(
    full_trace_dir: Path,
    full_capture_manifest: Path,
    output_dir: Path,
    *,
    cell_id: str,
    identity: dict,
    budget: dict,
    protocol: dict,
    config: RouteConfig,
    callbacks: RouteCallbacks,
) -> dict:
    """Route one over-budget exact full trace through the certified AC-9 ladder.

    The function returns a sealed deterministic record.  ``output_dir`` exists
    only after a route is fully verified and is suitable for inventorying under
    a campaign DONE manifest.
    """
    identity = _validate_identity(identity, cell_id)
    budget = _validate_budget(budget)
    protocol = _validate_protocol(protocol)
    _validate_config(config, budget["max_trace_bytes"])
    full_trace_dir = Path(full_trace_dir).resolve()
    output_dir = Path(output_dir).resolve(strict=False)
    if output_dir == full_trace_dir or full_trace_dir in output_dir.parents \
            or output_dir in full_trace_dir.parents:
        raise BudgetRouteRefusal(
            "route_path_overlap", "input full trace and retained output must be disjoint")
    work_root = output_dir.parent / f".{output_dir.name}.ac9_work"
    if output_dir.exists() or output_dir.is_symlink():
        raise BudgetRouteRefusal("route_output_exists", str(output_dir))
    if work_root.exists() or work_root.is_symlink():
        raise BudgetRouteRefusal("route_work_output_exists", str(work_root))

    full = validate_full_trace_capture(
        full_trace_dir, Path(full_capture_manifest), identity=identity,
        budget=budget, protocol=protocol)
    if full.tree_bytes <= budget["max_trace_bytes"]:
        raise BudgetRouteRefusal(
            "budget_route_not_required",
            f"full trace bytes={full.tree_bytes} <= max={budget['max_trace_bytes']}",
        )
    if full.tree_bytes > config.staging_limit_bytes:
        raise BudgetRouteRefusal(
            "staging_limit_exceeded",
            f"full trace bytes={full.tree_bytes}, limit={config.staging_limit_bytes}",
        )

    work_root.mkdir(parents=True, exist_ok=False)
    protocol_sha = canonical_sha256(protocol)
    try:
        # Tier 1: always perform the complete byte-exact compression attempt.
        candidate = work_root / "lossless_full_candidate"
        full_gate = classify(full.stats, sample_size=config.sample_size)

        def full_sidecar(comp: CompressionResult, _block_manifest: str) -> dict:
            return build_sidecar(
                cell_id=cell_id, kernel_name=full.stats.kernel_name,
                args_hash=protocol["args_hash"], stats=full.stats,
                gate=full_gate, compression=comp,
                records_captured=full.stats.total_records,
                tracer_version=TRACER_VERSION,
                stats_pass_version=full.stats.stats_pass_version,
                method="lossless_compressed_full", reason="byte_budget",
            )

        _build_compressed_bundle(full_trace_dir, candidate, full_sidecar)
        _remaining_staging(full_trace_dir, work_root, config.staging_limit_bytes)
        compressed_bytes = _tree_bytes(candidate)
        if compressed_bytes <= budget["max_trace_bytes"]:
            _accept_candidate(candidate, output_dir, work_root)
            return _result_record(
                identity=identity, cell_id=cell_id, budget=budget, config=config,
                protocol=protocol, full=full, output_dir=output_dir,
                method="lossless_compressed_full", outcome="admitted",
                typed_reason=None,
            )
        shutil.rmtree(candidate)

        # Tier 2 pass 1: certified all-block counting capture.
        counting_request = CaptureRequest(
            kind="trace_sampling_counting_v2", identity=identity, cell_id=cell_id,
            protocol_sha256=protocol_sha, full_trace_tree_sha256=full.tree_sha256,
            output_dir=work_root / "counting",
            staging_remaining_bytes=_remaining_staging(
                full_trace_dir, work_root, config.staging_limit_bytes),
        )
        counting_dir, _ = _invoke_capture(
            callbacks.counting_capture, counting_request, protocol)
        try:
            counting_stats = cell_stats_from_capture(counting_dir, cell_id)
            _same_launch_identity(full.stats, counting_stats)
            _validate_sampling_root(counting_dir, counting_stats, mode=1)
        except BudgetRouteRefusal:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BudgetRouteRefusal("counting_evidence_invalid", str(exc)) from exc
        _remaining_staging(full_trace_dir, work_root, config.staging_limit_bytes)
        gate = classify(counting_stats, sample_size=config.sample_size)

        def finish_gate_forbidden(
            forbidden_stats: Any,
            forbidden_gate: Any,
            evidence_dir: Path,
            *,
            tag: str,
            route_reason: str = "gate_forced",
            typed_reason: str = "gate_forced",
            terminal_detail: str | None = None,
        ) -> dict:
            if config.gate_forbidden_policy == "typed_unavailable":
                unavailable = build_sidecar(
                    cell_id=cell_id, kernel_name=forbidden_stats.kernel_name,
                    args_hash=protocol["args_hash"], stats=forbidden_stats,
                    gate=forbidden_gate, compression=None, records_captured=0,
                    tracer_version=TRACER_VERSION,
                    stats_pass_version=forbidden_stats.stats_pass_version,
                    method="unavailable", reason=route_reason,
                )
                candidate = work_root / f"typed_unavailable_{tag}_candidate"
                _build_unavailable_bundle(
                    evidence_dir, candidate, unavailable, gate=forbidden_gate,
                    typed_reason=typed_reason,
                    terminal_detail=terminal_detail,
                )
                _remaining_staging(
                    full_trace_dir, work_root, config.staging_limit_bytes
                )
                if _tree_bytes(candidate) > budget["max_trace_bytes"]:
                    raise BudgetRouteRefusal(
                        "retained_limit_exceeded", "typed unavailable evidence exceeds cell budget")
                _accept_candidate(candidate, output_dir, work_root)
                return _result_record(
                    identity=identity, cell_id=cell_id, budget=budget,
                    config=config, protocol=protocol, full=full,
                    output_dir=output_dir, method="unavailable",
                    outcome="typed_unavailable", typed_reason=typed_reason,
                )
            if callbacks.reduced_shape_capture is None:
                raise BudgetRouteRefusal(
                    "reduced_shape_callback_missing", config.reduced_shape_variant)
            reduced_request = CaptureRequest(
                kind="trace_sampling_reduced_full_v2", identity=identity,
                cell_id=cell_id, protocol_sha256=protocol_sha,
                full_trace_tree_sha256=full.tree_sha256,
                output_dir=work_root / f"reduced_full_{tag}",
                staging_remaining_bytes=_remaining_staging(
                    full_trace_dir, work_root, config.staging_limit_bytes),
                reduced_shape_variant=config.reduced_shape_variant,
            )
            reduced_dir, reduced_receipt = _invoke_capture(
                callbacks.reduced_shape_capture, reduced_request, protocol)
            if reduced_receipt.capture_manifest is None:
                raise BudgetRouteRefusal("reduced_capture_manifest_missing")
            if reduced_receipt.matched_label_evidence is None:
                raise BudgetRouteRefusal("matched_reduced_label_evidence_missing")
            _validate_matched_label_evidence(
                reduced_receipt.matched_label_evidence, reduced_dir,
                identity=identity,
                reduced_shape_variant=config.reduced_shape_variant,
                protocol=protocol,
            )
            reduced = validate_full_trace_capture(
                reduced_dir, reduced_receipt.capture_manifest, identity=identity,
                budget=budget, protocol={
                    **protocol,
                    "capture_manifest_sha256": sha256_file(
                        reduced_receipt.capture_manifest),
                },
            )
            if reduced.stats.grid == full.stats.grid and reduced.stats.block == full.stats.block:
                raise BudgetRouteRefusal(
                    "reduced_shape_not_reduced", repr(reduced.stats.grid))
            candidate = work_root / "reduced_shape_candidate"

            def reduced_sidecar(comp: CompressionResult, _block_manifest: str) -> dict:
                sidecar = build_sidecar(
                    cell_id=cell_id, kernel_name=reduced.stats.kernel_name,
                    args_hash=protocol["args_hash"], stats=reduced.stats,
                    gate=forbidden_gate, compression=comp,
                    records_captured=reduced.stats.total_records,
                    tracer_version=TRACER_VERSION,
                    stats_pass_version=reduced.stats.stats_pass_version,
                    method="reduced_shape_matched_label", reason=route_reason,
                    reduced_shape_variant=config.reduced_shape_variant,
                )
                sidecar["hashes"].pop("sidecar_sha256", None)
                sidecar["shape"]["original_grid"] = {
                    "x": full.stats.grid[0], "y": full.stats.grid[1], "z": full.stats.grid[2]}
                sidecar["shape"]["original_block"] = {
                    "x": full.stats.block[0], "y": full.stats.block[1], "z": full.stats.block[2]}
                sidecar["hashes"]["sidecar_sha256"] = sidecar_sha256(sidecar)
                validate_sidecar(sidecar)
                return sidecar

            _build_compressed_bundle(reduced_dir, candidate, reduced_sidecar)
            _remaining_staging(full_trace_dir, work_root, config.staging_limit_bytes)
            if _tree_bytes(candidate) > budget["max_trace_bytes"]:
                raise BudgetRouteRefusal(
                    "retained_limit_exceeded", "reduced-shape bundle exceeds cell budget")
            _accept_candidate(candidate, output_dir, work_root)
            return _result_record(
                identity=identity, cell_id=cell_id, budget=budget, config=config,
                protocol=protocol, full=full, output_dir=output_dir,
                method="reduced_shape_matched_label", outcome="admitted",
                typed_reason=None,
            )

        if gate.method != M_SUBSET:
            return finish_gate_forbidden(
                counting_stats, gate, counting_dir, tag="counting")

        # Tier 2 pass 2: fixed-point whole-segment selective capture.
        try:
            candidate_plan = plan_from_stats(
                counting_stats, seed=config.seed, sample_size=config.sample_size,
                force_include_block_ids=tuple(gate.force_include_blocks))
        except SelectionPlanError as exc:
            raise BudgetRouteRefusal("bootstrap_selection_refused", str(exc)) from exc
        selected = _selected_block_ids(candidate_plan)
        force = tuple(gate.force_include_blocks)
        accepted: tuple | None = None
        for attempt in range(1, config.max_reconcile_attempts + 1):
            request = CaptureRequest(
                kind="trace_sampling_selective_v2", identity=identity,
                cell_id=cell_id, protocol_sha256=protocol_sha,
                full_trace_tree_sha256=full.tree_sha256,
                output_dir=work_root / f"selective_attempt_{attempt}",
                staging_remaining_bytes=_remaining_staging(
                    full_trace_dir, work_root, config.staging_limit_bytes),
                attempt=attempt, selected_block_ids=tuple(selected),
            )
            pass2_dir, _ = _invoke_capture(
                callbacks.selective_capture, request, protocol)
            try:
                pass2_stats = cell_stats_from_capture(pass2_dir, cell_id)
                _same_launch_identity(counting_stats, pass2_stats)
                _validate_sampling_root(pass2_dir, pass2_stats, mode=2)
                capture_stats_sha = verify_capture_selection(pass2_dir, selected)
                selected_set = set(selected)
                serialized = {
                    block.block_id: block.serialized_bytes
                    for block in pass2_stats.blocks if block.block_id in selected_set
                }
                actual_records, actual_bytes = verify_pass2_payload(
                    pass2_dir, pass2_stats.grid, selected, serialized)
                contract = validate_trace_dir(pass2_dir)
                if not bool(getattr(contract, "ok", False)):
                    raise BudgetRouteRefusal(
                        getattr(contract, "typed_reason", None) or "trace_contract_invalid",
                        getattr(contract, "detail", None),
                    )
            except BudgetRouteRefusal:
                raise
            except Exception as exc:  # noqa: BLE001
                raise BudgetRouteRefusal("selective_evidence_invalid", str(exc)) from exc
            provisional = classify(
                pass2_stats, sample_size=config.sample_size,
                force_include_block_ids=force)
            if provisional.method != M_SUBSET:
                return finish_gate_forbidden(
                    pass2_stats, provisional, pass2_dir, tag=f"pass2_{attempt}")
            force = tuple(provisional.force_include_blocks)
            try:
                desired_plan = plan_from_stats(
                    pass2_stats, seed=config.seed, sample_size=config.sample_size,
                    force_include_block_ids=force)
            except SelectionPlanError as exc:
                raise BudgetRouteRefusal("pass2_selection_refused", str(exc)) from exc
            desired = _selected_block_ids(desired_plan)
            certainty_bytes = _certainty_serialized_bytes(pass2_stats, desired_plan)
            final_gate = classify(
                pass2_stats, sample_size=config.sample_size,
                force_include_block_ids=force,
                certainty_bytes=certainty_bytes,
                max_trace_bytes=config.selection_max_raw_bytes,
            )
            if final_gate.method != M_SUBSET:
                return finish_gate_forbidden(
                    pass2_stats, final_gate, pass2_dir,
                    tag=f"pass2_exact_{attempt}")
            if desired != selected:
                selected = desired
                continue
            accepted = (
                attempt, pass2_dir, pass2_stats, desired_plan, final_gate,
                certainty_bytes, capture_stats_sha, actual_records, actual_bytes,
            )
            break
        if accepted is None:
            return finish_gate_forbidden(
                counting_stats, gate, counting_dir,
                tag="placement_reconciliation",
                route_reason="byte_budget",
                typed_reason="budget_limited_wide_ci",
                terminal_detail=(
                    "placement_reconciliation_exhausted: "
                    f"attempts={config.max_reconcile_attempts}"
                ),
            )
        (attempts, pass2_dir, pass2_stats, plan, gate, certainty_bytes,
         capture_stats_sha, actual_records, actual_bytes) = accepted
        selected = _selected_block_ids(plan)
        raw_selected = sum(actual_bytes.values())
        if raw_selected > config.selection_max_raw_bytes:
            return finish_gate_forbidden(
                pass2_stats, gate, pass2_dir,
                tag="selected_payload_over_budget",
                route_reason="byte_budget",
                typed_reason="budget_limited_wide_ci",
                terminal_detail=(
                    f"selected_payload_over_budget: raw={raw_selected}, "
                    f"max={config.selection_max_raw_bytes}"
                ),
            )
        candidate = work_root / "selective_subset_candidate"

        def subset_sidecar(comp: CompressionResult, _block_manifest: str) -> dict:
            if comp.raw_bytes != raw_selected:
                raise BudgetRouteRefusal(
                    "serialized_payload_size_mismatch",
                    f"compressed={comp.raw_bytes}, verified={raw_selected}",
                )
            return build_sidecar(
                cell_id=cell_id, kernel_name=pass2_stats.kernel_name,
                args_hash=protocol["args_hash"], stats=pass2_stats,
                gate=gate, compression=comp,
                records_captured=sum(actual_records.values()),
                tracer_version=TRACER_VERSION,
                stats_pass_version=pass2_stats.stats_pass_version,
                method=M_SUBSET, reason="byte_budget",
                segments=plan.sidecar_segments(), strata=plan.sidecar_strata(),
                estimator={
                    "mode": "weighted_full_shape_estimate",
                    "targets": ["sectors", "l2_sector_hit_rate"],
                    "not_recoverable": ["cycles", "sol_pct_timing"],
                    "convergence": {"status": "budget_limited_wide_ci"},
                },
                seed=config.seed, rng_algo="pcg64",
                selected_segment_ids=[segment.id for segment in plan.selected_segments],
                certainty_segments=[segment.id for segment in plan.certainty_segments],
                force_include_blocks=list(plan.population.force_include_block_ids),
                selection_reconciliation={
                    "status": "fixed_point", "attempts": attempts,
                    "max_attempts": config.max_reconcile_attempts,
                    "capture_stats_sha256": capture_stats_sha,
                },
                selection_budget={
                    "max_trace_bytes": config.selection_max_raw_bytes,
                    "basis": "raw_threadblock_pb",
                    "certainty_fraction_numerator": 2,
                    "certainty_fraction_denominator": 5,
                    "certainty_bytes": certainty_bytes,
                },
                selection_auxiliary=_selection_auxiliary(pass2_stats, selected),
            )

        _build_compressed_bundle(pass2_dir, candidate, subset_sidecar)
        _remaining_staging(full_trace_dir, work_root, config.staging_limit_bytes)
        if _tree_bytes(candidate) > budget["max_trace_bytes"]:
            shutil.rmtree(candidate)
            return finish_gate_forbidden(
                pass2_stats, gate, pass2_dir,
                tag="subset_retained_over_budget",
                route_reason="byte_budget",
                typed_reason="budget_limited_wide_ci",
                terminal_detail=(
                    "retained_limit_exceeded: verified subset replay bundle "
                    "exceeds cell budget"
                ),
            )
        _accept_candidate(candidate, output_dir, work_root)
        return _result_record(
            identity=identity, cell_id=cell_id, budget=budget, config=config,
            protocol=protocol, full=full, output_dir=output_dir,
            method=M_SUBSET, outcome="admitted", typed_reason=None,
        )
    except Exception:
        # Leave the deterministic work root for forensic inspection.  It is not
        # an admitted output and a retry cannot silently reuse it.
        raise


def validate_capacity_margins(
    *,
    declared_retained_bytes: Iterable[int],
    existing_retained_bytes: int,
    retention_capacity_bytes: int,
    retention_reserve_bytes: int,
    per_cell_staging_bytes: Iterable[int],
    existing_staging_bytes: int,
    staging_capacity_bytes: int,
    staging_reserve_bytes: int,
    max_concurrent_captures: int = 1,
) -> dict:
    """Fail-closed aggregate retained + peak-staging reservation ledger."""
    scalar_values = {
        "existing_retained_bytes": existing_retained_bytes,
        "retention_capacity_bytes": retention_capacity_bytes,
        "retention_reserve_bytes": retention_reserve_bytes,
        "existing_staging_bytes": existing_staging_bytes,
        "staging_capacity_bytes": staging_capacity_bytes,
        "staging_reserve_bytes": staging_reserve_bytes,
    }
    for name, value in scalar_values.items():
        _require_plain_int(value, name, positive="capacity" in name)
    _require_plain_int(max_concurrent_captures, "max_concurrent_captures", positive=True)
    retained = list(declared_retained_bytes)
    staging = list(per_cell_staging_bytes)
    for index, value in enumerate(retained):
        _require_plain_int(value, f"declared_retained_bytes[{index}]", positive=True)
    for index, value in enumerate(staging):
        _require_plain_int(value, f"per_cell_staging_bytes[{index}]", positive=True)
    if not retained:
        raise BudgetRouteRefusal("capacity_plan_empty", "no declared retained budgets")
    if not staging:
        raise BudgetRouteRefusal("capacity_plan_empty", "no staging budgets")
    retained_projected = existing_retained_bytes + sum(retained)
    retained_usable = retention_capacity_bytes - retention_reserve_bytes
    if retained_usable < 0 or retained_projected > retained_usable:
        raise BudgetRouteRefusal(
            "aggregate_retention_margin_exceeded",
            f"projected={retained_projected}, usable={retained_usable}",
        )
    concurrent = sorted(staging, reverse=True)[:max_concurrent_captures]
    staging_projected = existing_staging_bytes + sum(concurrent)
    staging_usable = staging_capacity_bytes - staging_reserve_bytes
    if staging_usable < 0 or staging_projected > staging_usable:
        raise BudgetRouteRefusal(
            "aggregate_staging_margin_exceeded",
            f"projected={staging_projected}, usable={staging_usable}",
        )
    ledger = {
        "schema_version": 1,
        "declared_cell_count": len(retained),
        "declared_retained_bytes": sum(retained),
        "existing_retained_bytes": existing_retained_bytes,
        "retention_capacity_bytes": retention_capacity_bytes,
        "retention_reserve_bytes": retention_reserve_bytes,
        "retention_projected_bytes": retained_projected,
        "retention_headroom_after_reserve_bytes": retained_usable - retained_projected,
        "staging_cell_count": len(staging),
        "max_concurrent_captures": max_concurrent_captures,
        "peak_concurrent_staging_bytes": sum(concurrent),
        "existing_staging_bytes": existing_staging_bytes,
        "staging_capacity_bytes": staging_capacity_bytes,
        "staging_reserve_bytes": staging_reserve_bytes,
        "staging_projected_bytes": staging_projected,
        "staging_headroom_after_reserve_bytes": staging_usable - staging_projected,
    }
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    return ledger
