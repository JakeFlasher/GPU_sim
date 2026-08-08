"""Compute per-cell L1 predictions for the admitted-cell cohort
via the production L1 head, and emit a schema-validated JSON
artifact for the interim diagnostic report to consume.

The interim diagnostic report at
``data/sm_86_predictor_accuracy_recovery_v2/interim_diagnostic_report.md``
previously hard-coded ``predicted_l1=None`` on every cell row
because the L1 head was not yet producing scalar bundles
end-to-end. With the L1 cache-op classifier honoring the sm_86
default-cache-modifier semantic and the L1 production helper
emitting correctly-tagged ``cache_layer=L1TEX`` bundles, the
L1 head is now load-bearing on the admitted-cell cohort. This
utility drives the L1 head per admitted cell, persists the
result, and lets the report wire L1 ``rel_err`` + per-bucket
geomeans onto real evidence.

Inputs:

  - ``data/local_labels/label_admitted.json`` (v4 admitted
    cells; the utility iterates rows whose ``arch`` is sm_86
    or ``None`` -- the Plan-1 cohort is sm_86 only).
  - ``data/local_labels/cache_op_classifier_manifests/sm_86/<slug>__<shape>.json``
    (one classifier manifest per admitted cell).
  - The cell's trace directory:
    ``workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/trace_<shape_point>/``
    OR (microbench convention) ``trace_trace_<shape_point>/``
    when ``shape_point`` starts with ``trace_``. The
    ``dynamic_trace.pb`` lives at the top of that dir; per-block
    proto files live at
    ``<trace_dir>/threadblocks/device_0/stream_0/kernel_1/*.pb``.

Output:

  ``data/sm_86_predictor_accuracy_recovery_v2/predicted_l1_per_admitted_cell.json``
  validates against
  ``predicted_l1_per_admitted_cell.schema.json`` (this
  utility's sibling schema). Keyed by ``(slug, shape_point)``
  with: ``result_kind`` enum
  (``scalar``/``bounded``/``typed_unavailable``),
  ``cache_layer`` const ``L1TEX``,
  ``predicted_hit_rate`` (nullable; non-null when scalar/bounded),
  ``typed_reason`` (nullable; non-null when typed_unavailable),
  ``evidence`` (passthrough of the helper bundle's evidence
  subset).

CLI:

    python3 tools/sm_86_predictor_accuracy_recovery_v2/\
        compute_predicted_l1_per_admitted_cell.py
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "cache_predictor"))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "local_labels"))

import l1_production_helper as lph  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.nvbit_tracer import trace_admission as trace_admission_contract  # noqa: E402
# Shared base-predictor-cohort boundary: register-cap build variants + pending-
# refit factorial cells are admitted in the corpus but not yet folded into the
# predictor head, so the per-admitted-cell L1 prediction skips them (no row)
# until the refit. Using the two underlying predicates (rather than
# is_base_predictor_fit_cell) preserves the legacy missing-arch tolerance of the
# arch filter below; the exclusion is the same boundary the head producer uses.
from corpus_cell_identity import (  # noqa: E402
    cell_variant_id,
    is_pending_refit_factorial,
)


# The 5 identity fields the predicted-L1 artifact mirrors
# from each admitted-cell row. Every field is required;
# missing OR empty values fail-close with a typed diagnostic
# rather than silently emitting a partial artifact. The
# identity tuple lets the interim diagnostic report's
# per-cell join check that the prediction was made against
# the SAME binary / source_commit / kernel as the admitted
# truth label, not just the same (slug, shape_point) name.
_IDENTITY_FIELDS: tuple[str, ...] = (
    "arch",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
    "ncu_metric_version",
)


class _ArtifactIdentityFieldMissing(Exception):
    """Raised when an admitted row lacks a required identity
    field for the predicted-L1 artifact."""


# sm_86 device facts. The L1 head receives these as fixed
# inputs because the admitted-cell cohort is sm_86-only and the
# 3090 Ti's parameters are stable. Tier-2 cross-arch work would
# parameterize these per arch.
_SM_86_SM_COUNT: int = 84  # RTX 3090 Ti
_SM_86_L1_CAPACITY_BYTES: int = 128 * 1024
_SM_86_L1TEX_CAPACITY_SOURCE: str = "static_card_constant"
_SM_86_L1_SHARED_CARVEOUT_SOURCE: str = "default"
_SM_86_CTA_TO_SM_POLICY: str = "round_robin_homogeneous"
_KERNEL_ID: int = 1

# Every scalar prediction is computed from one byte-exact, bounded snapshot.
# Reuse the forward trace-admission ceilings rather than inventing a looser
# predictor-only policy. The classifier manifest is a separate regular-file
# authority and receives its own bounded anchored read.
_TRACE_INPUT_LIMITS = trace_admission_contract.DEFAULT_TRACE_ADMISSION_LIMITS
_CLASSIFIER_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
_CLASSIFIER_MANIFEST_TREE_LIMITS = {
    "max_files": 4096,
    "max_directories": 64,
    "max_total_file_bytes": 64 * 1024 * 1024,
    "max_total_relative_path_bytes": 1024 * 1024,
}
_THREADBLOCK_ROOT = PurePosixPath(
    "threadblocks/device_0/stream_0/kernel_1"
)


@dataclasses.dataclass(frozen=True)
class _RetainedTraceInputs:
    """Exact trace bytes read once from one monitored tree generation."""

    dynamic_trace_path: Path
    dynamic_trace_payload: bytes
    threadblocks: tuple[tuple[Path, bytes], ...]


# R17 carveout-aware capacity: per-cell `carveout_percent` source.
# Cells whose binary calls
# `cudaFuncSetAttribute(cudaFuncAttributePreferredSharedMemoryCarveout, ...)`
# get an entry here; the value is sourced from the binary's
# `main.cu` default. Other cells fall back to the toolkit default
# (no entry → default carveout source remains "default" and
# L1 capacity is the static-card default_l1_bytes).
#
# carveout_sensitivity_probe: the CANONICAL trace_l2_accuracy_xl axes
# (workloads/simbench_bw/shapes/carveout_sensitivity_probe.toml) set
# carveout_percent=25 — the value the canonical trace AND the
# canonical-geometry NCU label were captured at. (The binary's main.cu
# default of 50 only applies to a bare default-geometry run, which is NOT
# the canonical cell; the prior 50 here predated the geometry-mislabel
# repair.) The walker computes effective L1 capacity by interpolating
# between the minimum bucket (default_l1_bytes) and the maximum bucket
# (carveout_max_l1_bytes) at carveout_percent=25.
_CARVEOUT_PERCENT_BY_CELL: dict[tuple[str, str], int] = {
    ("carveout_sensitivity_probe", "trace_l2_accuracy_xl"): 25,
}
# Default-bucket minimum capacity for the carveout-aware
# interpolation (sm_86 default carveout bucket = 32KB).
_SM_86_DEFAULT_BUCKET_L1_BYTES: int = 32 * 1024


def _trace_dir_for(slug: str, shape_point: str) -> Path:
    """Resolve the cell's trace directory.

    Microbench cells (``shape_point`` like ``trace_l2_accuracy_xl``)
    follow the legacy ``trace_<shape>`` convention which doubles
    the ``trace_`` prefix when ``shape_point`` already starts
    with ``trace_``. Activation cells (``timing_shape`` etc.)
    use the simple ``trace_<shape>`` convention.

    Both workload results roots are covered: simbench_bw cells live
    under ``workloads/simbench_bw/results``; portable_lcd cells
    (the activation/GEMM ``*_sm80`` slugs) live under
    ``workloads/portable_lcd/results``. The simbench root wins when
    both exist (no admitted slug is present in both trees).
    """
    for suite in ("simbench_bw", "portable_lcd"):
        cell_root = (
            _REPO_ROOT / "workloads" / suite / "results"
            / slug / "rtx_3090_ti" / "sm_86"
        )
        candidate = cell_root / f"trace_{shape_point}"
        if candidate.is_dir():
            return candidate
    # Preserve the legacy default (simbench path) for the caller's
    # typed missing-trace arm when neither root has the directory.
    return (
        _REPO_ROOT / "workloads" / "simbench_bw" / "results"
        / slug / "rtx_3090_ti" / "sm_86" / f"trace_{shape_point}"
    )


def _threadblock_pb_paths(trace_dir: Path) -> list[Path]:
    """Return the sorted per-threadblock proto paths."""
    nested = trace_dir / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
    if not nested.is_dir():
        return []
    return sorted(nested.glob("*.pb"))


def _manifest_path_for(slug: str, shape_point: str) -> Path:
    return (
        _REPO_ROOT / "data" / "local_labels"
        / "cache_op_classifier_manifests" / "sm_86"
        / f"{slug}__{shape_point}.json"
    )


def _read_manifest(slug: str, shape_point: str) -> dict | None:
    path = _manifest_path_for(slug, shape_point)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_sorted_files(paths: list[Path]) -> str:
    """Canonical-list digest of a sorted file set.

    Each file's path-relative-to-repo + sha256 contributes a
    line ``<repo-relative-path>\\t<sha256>\\n``; the digest is
    SHA-256 over the concatenation of those lines in sorted
    order. The result binds the predicted-L1 artifact to the
    EXACT set of threadblock proto files the L1 head consumed.
    A future regeneration that adds, drops, or reorders a
    threadblock proto produces a different digest.

    This path-reading compatibility helper remains imported by the separate
    warp-pool producer. The production per-admitted-cell path below uses
    ``_sha256_of_retained_files`` so parsing and serialization share bytes.
    """
    h = hashlib.sha256()
    for p in paths:
        rel = str(p.relative_to(_REPO_ROOT))
        sha = _sha256_of_file(p)
        h.update(f"{rel}\t{sha}\n".encode("utf-8"))
    return h.hexdigest()


def _trace_tree_generation_limits() -> dict[str, int]:
    policy = trace_admission_contract.raw_trace_generation_limits(
        _TRACE_INPUT_LIMITS
    )
    return {
        name: int(policy[name])
        for name in (
            "max_files",
            "max_directories",
            "max_total_file_bytes",
            "max_total_relative_path_bytes",
        )
    }


def _snapshot_classifier_manifest(path: Path) -> tuple[bytes, dict]:
    with immutable.PinnedDirectoryReader(path.parent) as reader:
        with reader.retain_tree_generation(
            **_CLASSIFIER_MANIFEST_TREE_LIMITS
        ) as tree:
            file_metadata = dict(tree.files)
            metadata = file_metadata.get(path.name)
            if metadata is None:
                raise immutable.ImmutableArtifactError(
                    f"classifier manifest generation lost {path.name}"
                )
            snapshot = reader.read_snapshot(
                path.name,
                max_bytes=_CLASSIFIER_MANIFEST_MAX_BYTES,
                expected_metadata=metadata,
            )
    manifest = immutable.strict_json_loads(
        snapshot.payload,
        label=f"classifier manifest {path}",
    )
    if type(manifest) is not dict:
        raise immutable.ImmutableArtifactError(
            f"classifier manifest {path} must be an exact JSON object"
        )
    return snapshot.payload, manifest


def _snapshot_trace_inputs(trace_dir: Path) -> _RetainedTraceInputs | None:
    """Read the root and selected threadblocks once from one generation.

    ``None`` preserves the producer's typed-unavailable arm for an otherwise
    valid trace directory that lacks ``dynamic_trace.pb``. Any unsafe node,
    resource exhaustion, limit violation, or mutation during the snapshot is a
    hard immutable-artifact failure rather than prediction evidence.
    """
    limits = _TRACE_INPUT_LIMITS
    with immutable.PinnedDirectoryReader(trace_dir) as reader:
        with reader.retain_tree_generation(
            **_trace_tree_generation_limits()
        ) as tree:
            file_metadata = dict(tree.files)
            dynamic_metadata = file_metadata.get("dynamic_trace.pb")
            if dynamic_metadata is None:
                return None
            if dynamic_metadata[4] > limits.trace_root_bytes:
                raise immutable.ImmutableArtifactLimitError(
                    "predictor dynamic_trace.pb exceeds the bounded byte limit"
                )

            threadblock_names = tuple(sorted(
                name
                for name in file_metadata
                if (
                    PurePosixPath(name).parent == _THREADBLOCK_ROOT
                    and PurePosixPath(name).suffix == ".pb"
                )
            ))
            if len(threadblock_names) > limits.threadblock_files:
                raise immutable.ImmutableArtifactLimitError(
                    "predictor threadblock set exceeds the bounded file limit"
                )
            threadblock_total = 0
            for name in threadblock_names:
                size = file_metadata[name][4]
                if size > limits.threadblock_bytes:
                    raise immutable.ImmutableArtifactLimitError(
                        f"predictor threadblock {name} exceeds the bounded "
                        "per-file byte limit"
                    )
                if threadblock_total > limits.total_threadblock_bytes - size:
                    raise immutable.ImmutableArtifactLimitError(
                        "predictor threadblock set exceeds the bounded total-byte "
                        "limit"
                    )
                threadblock_total += size

            dynamic = reader.read_snapshot(
                "dynamic_trace.pb",
                max_bytes=limits.trace_root_bytes,
                expected_metadata=dynamic_metadata,
            )
            threadblocks: list[tuple[Path, bytes]] = []
            for name in threadblock_names:
                retained = reader.read_snapshot(
                    name,
                    max_bytes=limits.threadblock_bytes,
                    expected_metadata=file_metadata[name],
                )
                threadblocks.append((
                    trace_dir.joinpath(*PurePosixPath(name).parts),
                    retained.payload,
                ))

    return _RetainedTraceInputs(
        dynamic_trace_path=trace_dir / "dynamic_trace.pb",
        dynamic_trace_payload=dynamic.payload,
        threadblocks=tuple(threadblocks),
    )


def _sha256_of_retained_files(
    files: tuple[tuple[Path, bytes], ...],
) -> str:
    """Canonical-list digest over the exact retained threadblock bytes."""
    h = hashlib.sha256()
    for path, payload in files:
        rel = str(path.relative_to(_REPO_ROOT))
        digest = immutable.sha256_bytes(payload)
        h.update(f"{rel}\t{digest}\n".encode("utf-8"))
    return h.hexdigest()


def _extract_identity_tuple(
    cell_row: dict, *, slug: str, shape_point: str,
) -> dict[str, str]:
    """Return the identity tuple dict for the cell row,
    raising ``_ArtifactIdentityFieldMissing`` when any
    required field is missing or empty."""
    out: dict[str, str] = {}
    for field in _IDENTITY_FIELDS:
        value = cell_row.get(field)
        if value is None or value == "":
            raise _ArtifactIdentityFieldMissing(
                f"admitted cell {slug}/{shape_point} is "
                f"missing required identity field {field!r}; "
                f"the predicted-L1 artifact cannot be emitted "
                f"without provenance binding"
            )
        out[field] = str(value)
    return out


def _build_mpi_skeleton(
    *, slug: str, shape_point: str, identity: dict[str, str],
) -> dict:
    """Mint a manifest_pairing_index skeleton from the admitted
    label row's identity tuple. The L1 head requires these 5
    keys to be present on the bundle's MPI; the identity tuple
    enforced by ``_extract_identity_tuple`` guarantees they
    are non-empty.
    """
    return {
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": identity["binary_sha256"],
        "source_commit": identity["source_commit"],
        "kernel_signature_sha256": identity["kernel_signature_sha256"],
    }


def _digest_skeleton(*, trace_pb_path: Path) -> dict:
    """Compatibility path reader for the separate warp-pool producer."""
    return {
        "dynamic_trace_pb": {
            "path": str(trace_pb_path.relative_to(_REPO_ROOT)),
            "sha256": _sha256_of_file(trace_pb_path),
        },
        "sector_stream": {
            "path": "n/a-l1-head-does-not-consume-sector-stream",
            "sha256": "0" * 64,
        },
    }


def _digest_skeleton_from_retained(
    *, trace_pb_path: Path, trace_pb_payload: bytes,
) -> dict:
    """Build predictor provenance from the exact retained root bytes."""
    return {
        "dynamic_trace_pb": {
            "path": str(trace_pb_path.relative_to(_REPO_ROOT)),
            "sha256": immutable.sha256_bytes(trace_pb_payload),
        },
        "sector_stream": {
            "path": "n/a-l1-head-does-not-consume-sector-stream",
            "sha256": "0" * 64,
        },
    }


def compute_for_cell(
    *,
    slug: str,
    shape_point: str,
    cell_row: dict,
) -> dict:
    """Return the per-cell predicted-L1 entry. Fail-closes
    with ``_ArtifactIdentityFieldMissing`` when the admitted
    row lacks a required identity field; the caller is
    expected to treat that as a fatal error."""
    identity = _extract_identity_tuple(
        cell_row, slug=slug, shape_point=shape_point,
    )

    manifest_path = _manifest_path_for(slug, shape_point)
    if not manifest_path.is_file():
        # No classifier manifest on disk -- emit
        # typed_unavailable WITH a placeholder digest set so
        # the schema validates. Use sha-256-of-empty for the
        # missing inputs to keep the structure consistent.
        empty_sha = hashlib.sha256(b"").hexdigest()
        return {
            "slug": slug,
            "shape_point": shape_point,
            **identity,
            "result_kind": "typed_unavailable",
            "cache_layer": "L1TEX",
            "predicted_hit_rate": None,
            "typed_reason": "predictor_l1_sass_provenance_missing",
            "evidence": {
                "diagnostic": (
                    f"classifier manifest not found at "
                    f"data/local_labels/cache_op_classifier_manifests/"
                    f"sm_86/{slug}__{shape_point}.json"
                ),
            },
            "input_digests": {
                "classifier_manifest_sha256": empty_sha,
                "dynamic_trace_pb_sha256": empty_sha,
                "threadblock_proto_set_sha256": empty_sha,
            },
        }

    manifest_payload, manifest = _snapshot_classifier_manifest(manifest_path)
    manifest_sha256 = immutable.sha256_bytes(manifest_payload)

    trace_dir = _trace_dir_for(slug, shape_point)
    trace_pb = trace_dir / "dynamic_trace.pb"
    if not trace_dir.is_dir():
        empty_sha = hashlib.sha256(b"").hexdigest()
        return {
            "slug": slug,
            "shape_point": shape_point,
            **identity,
            "result_kind": "typed_unavailable",
            "cache_layer": "L1TEX",
            "predicted_hit_rate": None,
            "typed_reason": "predictor_l1_sass_provenance_missing",
            "evidence": {
                "diagnostic": (
                    f"dynamic_trace.pb missing at {trace_pb}"
                ),
            },
            "input_digests": {
                "classifier_manifest_sha256": manifest_sha256,
                "dynamic_trace_pb_sha256": empty_sha,
                "threadblock_proto_set_sha256": empty_sha,
            },
        }

    trace_inputs = _snapshot_trace_inputs(trace_dir)
    if trace_inputs is None:
        empty_sha = hashlib.sha256(b"").hexdigest()
        return {
            "slug": slug,
            "shape_point": shape_point,
            **identity,
            "result_kind": "typed_unavailable",
            "cache_layer": "L1TEX",
            "predicted_hit_rate": None,
            "typed_reason": "predictor_l1_sass_provenance_missing",
            "evidence": {
                "diagnostic": (
                    f"dynamic_trace.pb missing at {trace_pb}"
                ),
            },
            "input_digests": {
                "classifier_manifest_sha256": manifest_sha256,
                "dynamic_trace_pb_sha256": empty_sha,
                "threadblock_proto_set_sha256": empty_sha,
            },
        }
    tb_paths = [path for path, _payload in trace_inputs.threadblocks]

    mpi = _build_mpi_skeleton(
        slug=slug, shape_point=shape_point, identity=identity,
    )
    isd = _digest_skeleton_from_retained(
        trace_pb_path=trace_inputs.dynamic_trace_path,
        trace_pb_payload=trace_inputs.dynamic_trace_payload,
    )

    # R17 carveout-aware capacity: if the cell has a per-cell
    # carveout_percent setting in `_CARVEOUT_PERCENT_BY_CELL`,
    # override the default `l1_shared_carveout_source` so the
    # walker computes the effective L1 capacity from the
    # carveout setting. Otherwise the default bucket applies.
    carveout_percent_for_cell = _CARVEOUT_PERCENT_BY_CELL.get(
        (slug, shape_point)
    )
    if carveout_percent_for_cell is not None:
        carveout_source = "cuda_func_attribute_carveout_percent"
        default_l1_for_helper = _SM_86_DEFAULT_BUCKET_L1_BYTES
    else:
        carveout_source = _SM_86_L1_SHARED_CARVEOUT_SOURCE
        default_l1_for_helper = _SM_86_L1_CAPACITY_BYTES

    bundle = lph.predict_l1_from_trace_dir(
        trace_pb_path=trace_inputs.dynamic_trace_path,
        threadblock_pb_paths=tb_paths,
        kernel_id=_KERNEL_ID,
        classifier_manifest=manifest,
        sm_count=_SM_86_SM_COUNT,
        l1tex_capacity_source=_SM_86_L1TEX_CAPACITY_SOURCE,
        l1_shared_carveout_source=carveout_source,
        arch_token="sm_86",
        manifest_pairing_index=mpi,
        input_sidecar_digests=isd,
        device="NVIDIA GeForce RTX 3090 Ti",
        cache_model_capacity_bytes=_SM_86_L1_CAPACITY_BYTES,
        cta_to_sm_schedule_policy=_SM_86_CTA_TO_SM_POLICY,
        carveout_max_l1_bytes=_SM_86_L1_CAPACITY_BYTES,
        default_l1_bytes=default_l1_for_helper,
        carveout_percent=carveout_percent_for_cell,
        warmup_launches=5,
        retained_trace_pb_payload=trace_inputs.dynamic_trace_payload,
        retained_threadblock_pb_payloads=[
            payload for _path, payload in trace_inputs.threadblocks
        ],
    )

    entry: dict = {
        "slug": slug,
        "shape_point": shape_point,
        **identity,
        "result_kind": bundle["result_kind"],
        "cache_layer": bundle.get("cache_layer", "L1TEX"),
        "predicted_hit_rate": bundle.get("predicted_hit_rate"),
        "typed_reason": bundle.get("typed_reason"),
        "evidence": bundle.get("evidence", {}),
        "input_digests": {
            "classifier_manifest_sha256": manifest_sha256,
            "dynamic_trace_pb_sha256": immutable.sha256_bytes(
                trace_inputs.dynamic_trace_payload
            ),
            "threadblock_proto_set_sha256": _sha256_of_retained_files(
                trace_inputs.threadblocks
            ),
        },
    }
    return entry


_DEFAULT_COHORT_ACCOUNTING = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_accounting_post_corpus_expansion.json"
)


def _load_cohort_priority_keys(
    cohort_accounting_path: Path = _DEFAULT_COHORT_ACCOUNTING,
) -> list[tuple[str, str]]:
    """Return the ordered ``(slug, shape_point)`` keys for the
    13-cell headline cohort. The downstream report iterates the
    cohort_accounting rows when applying its per-cell identity
    guard; emitting the predicted_l1 artifact with cohort cells
    FIRST keeps the artifact's ``cells[0]`` in the guarded
    iteration scope so identity drift on the first row is
    surfaced by the guard's existing per-cell check.
    """
    if not cohort_accounting_path.is_file():
        return []
    try:
        payload = json.loads(
            cohort_accounting_path.read_text(encoding="utf-8"),
        )
    except (OSError, json.JSONDecodeError):
        return []
    keys: list[tuple[str, str]] = []
    for row in payload.get("cells", []) or payload.get("rows", []) or []:
        slug = row.get("slug")
        shape_point = row.get("shape_point")
        if isinstance(slug, str) and isinstance(shape_point, str):
            keys.append((slug, shape_point))
    return keys


def build_artifact(
    *,
    label_admitted_path: Path,
    existing_artifact_path: Path | None = None,
) -> dict:
    """Build every row from the current admitted row and fresh inputs only.

    ``existing_artifact_path`` remains a source-compatible CLI/public-call
    parameter, but its bytes are deliberately not prediction authority. A prior
    scalar has no independent prediction digest and cannot be authenticated when
    the retained classifier/trace generation is unavailable. In that case the
    fresh ``typed_unavailable`` row must survive unchanged rather than acquiring
    stale prediction fields under a newly stamped identity.

    BL-20260708-guard-on-consumer-path: the producer that emits the scalar is the
    consumer path that must authenticate its inputs; historical output bytes are
    not a substitute for those inputs.
    """
    _ = existing_artifact_path
    with label_admitted_path.open("r", encoding="utf-8") as fh:
        admitted = json.load(fh)
    cells = admitted.get("cells", [])
    entries: list[dict] = []
    for cell in cells:
        # Filter to sm_86 (or legacy missing-arch). This is the
        # sm_86-only generator: cross-host merge workflows that
        # land sm_120 rows in label_admitted.json would otherwise
        # be routed through sm_86 trace/manifest paths AND emitted
        # as `arch="sm_120"` cells the predicted-L1 schema rejects
        # (the schema's arch enum is closed on sm_86). The legacy
        # missing-arch (None) case is preserved because earlier
        # Plan-1 label_admitted.json revisions omitted the arch
        # field and were sm_86-only by construction.
        cell_arch = cell.get("arch")
        if cell_arch is not None and cell_arch != "sm_86":
            continue
        # A pending-refit cell (the transitional boundary for FUTURE corpus
        # additions; empty since the controlled refit) stays out entirely.
        if is_pending_refit_factorial(cell):
            continue
        slug = cell.get("slug")
        shape_point = cell.get("shape_point")
        if not slug or not shape_point:
            raise ValueError(
                f"admitted row missing slug/shape_point: {cell!r}"
            )
        # Register-cap build variants are FOLDED corpus cells since the
        # controlled refit, so the 1:1 prediction-to-admitted mapping must
        # cover them — but the SASS-provenance manifest store is keyed
        # (slug, shape_point) without a build-variant axis and the per-cap
        # spill_variants binaries have no per-variant manifests yet, so each
        # variant lands as an honest TYPED row (never a silent skip). The
        # row carries the variant's own closed identity (per-cap
        # binary_sha256) + the canonical variant_id/register_cap markers.
        variant_id = cell_variant_id(cell)
        if variant_id:
            identity = _extract_identity_tuple(
                cell, slug=slug, shape_point=shape_point,
            )
            empty_sha = hashlib.sha256(b"").hexdigest()
            entries.append({
                "slug": slug,
                "shape_point": shape_point,
                **identity,
                "variant_id": variant_id,
                "register_cap": int(cell["register_cap"]),
                "result_kind": "typed_unavailable",
                "cache_layer": "L1TEX",
                "predicted_hit_rate": None,
                "typed_reason": "predictor_l1_sass_provenance_missing",
                "evidence": {
                    "diagnostic": (
                        "register-cap build variant: the cache-op classifier "
                        "manifest store is keyed (slug, shape_point) without "
                        "a build-variant axis, so no per-cap SASS provenance "
                        "manifest exists for the spill_variants binary; "
                        "extending the manifest store with a variant axis is "
                        "the typed follow-up that lifts these rows to scalar"
                    ),
                },
                "input_digests": {
                    "classifier_manifest_sha256": empty_sha,
                    "dynamic_trace_pb_sha256": empty_sha,
                    "threadblock_proto_set_sha256": empty_sha,
                },
            })
            continue
        fresh = compute_for_cell(
            slug=slug, shape_point=shape_point, cell_row=cell,
        )
        entries.append(fresh)
    # Sort entries so the 13-cell headline cohort lands first
    # (in cohort_accounting order), then any non-cohort cells in
    # the order ``label_admitted.json`` already provides
    # (alphabetical by slug). The downstream report iterates the
    # cohort_accounting rows for its per-cell identity guard;
    # placing cohort cells first guarantees the artifact's
    # ``cells[0]`` is in the guarded iteration scope, which the
    # identity-guard tests in
    # ``test_compute_predicted_l1_per_admitted_cell.py`` assert
    # by mutating a single field on ``cells[0]`` and expecting
    # the guard to fire.
    cohort_priority_keys = _load_cohort_priority_keys()
    if cohort_priority_keys:
        cohort_priority_index = {
            key: idx for idx, key in enumerate(cohort_priority_keys)
        }
        max_priority = len(cohort_priority_keys)
        def _sort_key(entry: dict) -> tuple[int, str, str]:
            key = (entry["slug"], entry["shape_point"])
            return (
                cohort_priority_index.get(key, max_priority),
                entry["slug"],
                entry["shape_point"],
            )
        entries.sort(key=_sort_key)
    return {
        "schema_version": 2,
        "generator": (
            "tools/sm_86_predictor_accuracy_recovery_v2/"
            "compute_predicted_l1_per_admitted_cell.py"
        ),
        "cohort_size": len(entries),
        "cells": entries,
    }


def _validate_artifact(
    artifact: dict, *, schema_path: Path,
) -> None:
    import jsonschema  # local import
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(artifact, schema)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-admitted-cell L1 predictions via the "
            "production L1 head and emit a schema-validated "
            "JSON artifact for the interim diagnostic report to consume."
        ),
    )
    parser.add_argument(
        "--label-admitted-path",
        type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "label_admitted.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
            / "predicted_l1_per_admitted_cell.json"
        ),
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=(
            _REPO_ROOT / "tools" / "sm_86_predictor_accuracy_recovery_v2"
            / "predicted_l1_per_admitted_cell.schema.json"
        ),
    )
    args = parser.parse_args(argv)

    try:
        artifact = build_artifact(
            label_admitted_path=args.label_admitted_path,
        )
    except _ArtifactIdentityFieldMissing as exc:
        sys.stderr.write(
            f"compute_predicted_l1_per_admitted_cell: identity "
            f"field missing on an admitted row -- the artifact "
            f"requires arch / binary_sha256 / source_commit / "
            f"kernel_signature_sha256 / ncu_metric_version on "
            f"every admitted row: {exc}\n"
        )
        return 2

    if args.schema_path.is_file():
        try:
            _validate_artifact(artifact, schema_path=args.schema_path)
        except Exception as exc:  # pylint: disable=broad-except
            sys.stderr.write(
                f"compute_predicted_l1_per_admitted_cell: schema "
                f"validation FAILED: {exc}\n"
            )
            return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")

    scalar_n = sum(
        1 for c in artifact["cells"]
        if c["result_kind"] in ("scalar", "bounded")
    )
    typed_unavailable_n = sum(
        1 for c in artifact["cells"]
        if c["result_kind"] == "typed_unavailable"
    )
    try:
        out_display = str(args.out.relative_to(_REPO_ROOT))
    except ValueError:
        # `--out` may point outside the repo (e.g., a tempdir
        # for scratch-output validation). Fall back to the
        # absolute path so the CLI does not crash on a
        # display-only formatting call after the file has been
        # written. Mirrors the same fix landed earlier for
        # ``tools/local_labels/emit_sm86_cohort_preflight.py``.
        out_display = str(args.out)
    sys.stderr.write(
        f"compute_predicted_l1_per_admitted_cell: wrote "
        f"{out_display} "
        f"(scalar/bounded={scalar_n}, typed_unavailable="
        f"{typed_unavailable_n})\n"
    )
    for c in artifact["cells"]:
        sys.stderr.write(
            f"  {c['slug']}/{c['shape_point']}: result_kind="
            f"{c['result_kind']!r}, predicted_hit_rate="
            f"{c['predicted_hit_rate']!r}, typed_reason="
            f"{c['typed_reason']!r}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
