"""Per-cell sidecar regeneration entry point for the sm_86 cohort
regeneration lane.

The bulk-cohort entry point at
``tools/local_labels/regenerate_v5_trace_sidecars.py`` consumes the
v5 staging manifest and dispatches every ``REGENERATE_*_FROM_TRACE``
action in one batch. The preflight matrix at
``data/local_labels/sm86_cohort_preflight.json`` instead emits a
PER-CELL command for the ``REGENERATE_SIDECARS_LOCAL`` proposed
action so the operator can replay one cell at a time without
fabricating a synthetic manifest.

This module is the per-cell surface. It accepts the cell-identity
3-tuple ``(arch, slug, shape_point)``, resolves the canonical
trace directory under
``workloads/<source>/results/<slug>/<sku>/<arch>/trace_<shape_point>/``,
loads the cell's full ``manifest_pairing_index`` 5-tuple directly
from ``data/local_labels/label_admitted.json``, and invokes the
three trace-sidecar producers in sequence with that label-derived
identity:

* ``compute_sharing_locality`` (in-process; the label MPI is the
  canonical identity; no placeholder fallback).
* ``emit_hrd_bundle`` (in-process; the label MPI is the canonical
  identity — no bootstrap dependency on an already-admitted
  predictor bundle, which would deadlock with sidecar production
  in the ``sidecar then bundle`` producer order).
* ``consume_trace_dir`` (subprocess via the bulk runner's
  subprocess wrapper).

After the JSONL is written, the entry point also writes the
sibling ``<trace_dir>/sector_stream/manifest_pairing_index.json``
with the same label MPI 5-tuple, satisfying the canonical
sector_stream identity-binding sibling.

The MPI-resolution path is the load-bearing difference from the
bulk runner. The bulk runner resolves HRD MPI from an
already-admitted predictor bundle (``_resolve_real_mpi_from_admitted_bundle``)
because it runs AFTER bundles exist. The per-cell runner runs
BEFORE bundles exist (sidecars are a prerequisite for bundle
emission); resolving MPI from the bundle here would deadlock with
typed reason ``hrd_mpi_unresolvable:no_admitted_predictor_bundle_for_cell``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tools.local_labels.regenerate_v5_trace_sidecars import (  # noqa: E402
    _DEVICE_BLOCKS,
    _HRD_CACHE_INPUTS,
    _HRD_DEVICE_BLOCKS,
    _resolve_trace_dir,
)
from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
    ensure_directory,
    read_regular,
    replace_regular_bytes,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    AdmittedTraceSources,
    TraceAdmissionError,
    open_admitted_trace_sources,
    require_trace_admission,
)
from tools.sector_stream.consume_trace import (  # noqa: E402
    AdmittedSectorStream,
    consume_admitted_trace_sources,
)

_HARD_TRACE_SOURCE_ERRORS = (
    TraceAdmissionError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
    ImmutableArtifactError,
)

# Shared mixed-arity build-variant identity helper (sibling import so the
# per-cell sidecar regenerator reasons about base-vs-variant identity through
# the SAME normalizer as the aggregator / selector / merge / staging — no
# re-inlined copy). ``cell_variant_id`` returns "" for a base cell and the
# canonical ``register_cap_<n>`` for a build variant, raising on an
# inconsistent register_cap/variant_id pair (fail-closed).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from corpus_cell_identity import cell_variant_id  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABEL_ADMITTED = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"

_LABEL_IDENTITY_KEYS: tuple[str, ...] = (
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
)

# Per-arch device blocks for the divergence_proxies and
# order_projections producers. The shape mirrors the canonical
# ``workloads/simbench_bw/run_suite.py`` device block emitted from
# live ``DeviceProps``. Values are hardcoded here because this
# regeneration entry runs against on-disk traces only — no live
# DeviceProps lookup is available. The 6 MiB sm_86 L2 figure is
# the RTX 3090 Ti static card capacity (matches the existing 3
# admitted simbench cells' on-disk
# ``order_projections/order_projections.json::device.l2_cache_size_bytes``
# = 6291456). When sm_120 cells get regenerated through this path
# the corresponding entry must be added; until then sm_120 calls
# fail closed with ``unknown_arch``.
_ORDER_PROJECTIONS_DEVICE_BLOCKS: dict[str, dict[str, Any]] = {
    "sm_86": {
        "device_name": "NVIDIA GeForce RTX 3090 Ti",
        "compute_capability_major": 8,
        "compute_capability_minor": 6,
        "sm_count": 84,
        "l2_cache_size_bytes": 6 * 1024 * 1024,
    },
}

# Per-arch capacity used by the order_projections walker. Mirrors
# the device block's ``l2_cache_size_bytes`` because the canonical
# run_suite.py path also passes the resolved L2 capacity as the
# walker's ``capacity_bytes``.
_ORDER_PROJECTIONS_CAPACITY_BYTES: dict[str, int] = {
    "sm_86": 6 * 1024 * 1024,
}

# Divergence_proxies receives only the trace_dir + a short device
# string + the MPI; no capacity_bytes / sm_count are required.
_DIVERGENCE_PROXIES_DEVICE_STRING: dict[str, str] = {
    "sm_86": "sm_86",
}


def label_mpi_for_cell(
    *,
    arch: str,
    slug: str,
    shape_point: str,
    variant_id: str | None = None,
    label_path: Path = _LABEL_ADMITTED,
) -> dict[str, str] | None:
    """Return the cell's manifest_pairing_index 5-tuple from
    ``label_admitted.json``, or ``None`` when no matching row
    exists or the row is missing any required identity field.

    The returned dict carries EXACTLY the canonical 5 identity
    fields per the sharing_locality / hrd_profile / sector_stream
    sibling schemas (which pin `manifest_pairing_index` with
    `additionalProperties: false`). ``arch`` is NOT included
    because adding it makes every sidecar schema-invalid; arch
    is passed separately as a routing argument to each
    downstream producer.

    ``variant_id`` selects among build-variant siblings of the same
    (arch, slug, shape_point): when None the base cell (no
    ``variant_id`` field) is matched; when set (e.g.
    ``register_cap_64``) the matching register-cap variant row is
    matched. The variant's distinct ``binary_sha256`` flows into the
    returned MPI, so a sibling cap's regenerated sidecars carry a
    DISTINCT identity 5-tuple from the base cell and from each other
    WITHOUT the closed sidecar MPI schema needing a variant field.
    """
    if not label_path.is_file():
        return None
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    want_variant = variant_id or ""
    for cell in payload.get("cells", []):
        if not (
            cell.get("arch") == arch
            and cell.get("slug") == slug
            and cell.get("shape_point") == shape_point
        ):
            continue
        # Mixed-arity disambiguation through the shared normalizer: a
        # base cell yields "" and a register-cap variant yields its
        # canonical ``register_cap_<n>``. cell_variant_id raises on an
        # inconsistent register_cap/variant_id pair (fail-closed).
        if cell_variant_id(cell) != want_variant:
            continue
        mpi: dict[str, str] = {}
        for key in _LABEL_IDENTITY_KEYS:
            val = cell.get(key)
            if not isinstance(val, str) or not val:
                return None
            mpi[key] = val
        return mpi
    return None


def _regenerate_sharing_locality_with_label_mpi(
    *,
    arch: str,
    trace_dir: Path,
    label_mpi: dict[str, str],
    sources: AdmittedTraceSources,
) -> dict[str, Any]:
    """Emit sharing-locality from the batch's admitted capability."""
    device_block = _DEVICE_BLOCKS.get(arch)
    if device_block is None:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": f"unknown_arch:{arch}",
            "output_path": None,
            "producer_invoked": "compute_sharing_locality",
        }
    try:
        from tools.sharing_locality.classifier import (
            compute_sharing_locality_from_admitted_sources,
        )
        bundle = compute_sharing_locality_from_admitted_sources(
            sources=sources,
            manifest_pairing_index=label_mpi,
            device=device_block,
        )
    except _HARD_TRACE_SOURCE_ERRORS:
        raise
    except (ImportError, OSError, RuntimeError) as exc:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"compute_sharing_locality_raised:"
                f"{type(exc).__name__}:{exc!s}"
            )[:512],
            "output_path": None,
            "producer_invoked": "compute_sharing_locality",
        }
    sl_dir = trace_dir / "sharing_locality"
    output_path = sl_dir / "sharing_locality.json"
    ensure_directory(sl_dir)
    replace_regular_bytes(
        output_path,
        (json.dumps(bundle, indent=2) + "\n").encode("utf-8"),
    )
    return {
        "status": "written",
        "typed_failure_reason": None,
        "output_path": str(output_path),
        "producer_invoked": "compute_sharing_locality",
    }


def _regenerate_hrd_profile_with_label_mpi(
    *,
    arch: str,
    trace_dir: Path,
    label_mpi: dict[str, str],
    sources: AdmittedTraceSources,
    sector_stream: AdmittedSectorStream,
) -> dict[str, Any]:
    """Invoke ``emit_hrd_bundle`` in-process with the label MPI
    5-tuple (no dependency on an already-admitted predictor
    bundle) and write the bundle to
    ``<trace_dir>/hrd_profile/hrd_profile.json``.

    The producer returns a status dict carrying ``contract_ok``
    and ``schema_admits``; this wrapper surfaces both as typed
    failures rather than reporting success based on file
    presence alone (a stale file on disk could mask a producer
    contract failure). The bulk runner already consults the
    status dict; the per-cell wrapper now mirrors that
    discipline.
    """
    device_block = _HRD_DEVICE_BLOCKS.get(arch)
    cache_inputs = _HRD_CACHE_INPUTS.get(arch)
    if device_block is None or cache_inputs is None:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": f"unknown_arch:{arch}",
            "output_path": None,
            "producer_invoked": "emit_hrd_bundle",
        }
    try:
        from tools.hrd_profiler.bundle_from_trace import (
            emit_hrd_bundle_from_admitted_sources,
        )
        producer_result = emit_hrd_bundle_from_admitted_sources(
            sources=sources,
            sector_stream=sector_stream,
            output_root=trace_dir,
            manifest_pairing_index=label_mpi,
            device=device_block,
            cuda_l2_cache_size_bytes=cache_inputs[
                "cuda_l2_cache_size_bytes"
            ],
            static_card_l2_cache_size_bytes=cache_inputs[
                "static_card_l2_cache_size_bytes"
            ],
            cuda_num_l2_slices=cache_inputs["cuda_num_l2_slices"],
            static_card_num_l2_slices=cache_inputs[
                "static_card_num_l2_slices"
            ],
        )
    except _HARD_TRACE_SOURCE_ERRORS:
        raise
    except (RuntimeError, ImportError, OSError, ValueError) as exc:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"emit_hrd_bundle_raised:{type(exc).__name__}:{exc!s}"
            )[:512],
            "output_path": None,
            "producer_invoked": "emit_hrd_bundle",
        }
    output_path = trace_dir / "hrd_profile" / "hrd_profile.json"
    contract_ok = bool(producer_result.get("contract_ok"))
    if not contract_ok:
        typed_reason = producer_result.get("contract_typed_reason")
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"emit_hrd_bundle_contract_not_ok:{typed_reason}"
            )[:512],
            "output_path": str(output_path),
            "producer_invoked": "emit_hrd_bundle",
            "contract_ok": False,
            "schema_admits": producer_result.get("schema_admits"),
        }
    schema_admits = bool(producer_result.get("schema_admits"))
    if not schema_admits:
        first_failure = producer_result.get("schema_first_failure")
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"emit_hrd_bundle_schema_invalid:{first_failure}"
            )[:512],
            "output_path": str(output_path),
            "producer_invoked": "emit_hrd_bundle",
            "contract_ok": True,
            "schema_admits": False,
        }
    return {
        "status": "written",
        "typed_failure_reason": None,
        "output_path": str(output_path),
        "producer_invoked": "emit_hrd_bundle",
        "contract_ok": True,
        "schema_admits": True,
    }


def _write_sector_stream_mpi_sibling(
    *, trace_dir: Path, label_mpi: dict[str, str],
) -> dict[str, Any]:
    """Write the sector_stream identity-binding sibling at
    ``<trace_dir>/sector_stream/manifest_pairing_index.json``.
    The sibling carries the cell's 5-tuple identity. ``arch`` is
    NOT written because the sharing_locality / hrd_profile
    schemas pin the MPI block to the 5 fields with
    `additionalProperties: false`; the sector_stream sibling
    follows the same shape for consistency. The sector_stream
    JSONL itself is record-level and does not embed identity.
    """
    sibling = trace_dir / "sector_stream" / "manifest_pairing_index.json"
    ensure_directory(sibling.parent)
    payload = {key: label_mpi[key] for key in _LABEL_IDENTITY_KEYS}
    replace_regular_bytes(
        sibling,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {
        "status": "written",
        "typed_failure_reason": None,
        "output_path": str(sibling),
        "producer_invoked": "sector_stream_mpi_sibling",
    }


def _regenerate_divergence_proxies_with_label_mpi(
    *,
    arch: str,
    trace_dir: Path,
    label_mpi: dict[str, str],
    sources: AdmittedTraceSources,
    sector_stream: AdmittedSectorStream,
) -> dict[str, Any]:
    """Emit divergence proxies from the batch's admitted capability."""
    device_str = _DIVERGENCE_PROXIES_DEVICE_STRING.get(arch)
    if device_str is None:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": f"unknown_arch:{arch}",
            "output_path": None,
            "producer_invoked": "divergence_proxies/consume_trace_dir",
        }
    try:
        from tools.divergence_proxies.bundle_from_trace import (
            consume_admitted_trace_sources,
        )
        result = consume_admitted_trace_sources(
            sources=sources,
            sector_stream=sector_stream,
            output_root=trace_dir,
            device=device_str,
            manifest_pairing_index=label_mpi,
        )
    except _HARD_TRACE_SOURCE_ERRORS:
        raise
    except (
        RuntimeError, ImportError, OSError, ValueError,
        FileNotFoundError,
    ) as exc:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"consume_trace_dir_raised:"
                f"{type(exc).__name__}:{exc!s}"
            )[:512],
            "output_path": None,
            "producer_invoked": "divergence_proxies/consume_trace_dir",
        }
    output_path = result.get("bundle_path")
    schema_admits = bool(result.get("schema_admits"))
    if not schema_admits:
        first_failure = result.get("schema_first_failure")
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"divergence_proxies_schema_invalid:{first_failure}"
            )[:512],
            "output_path": output_path,
            "producer_invoked": "divergence_proxies/consume_trace_dir",
            "schema_admits": False,
        }
    return {
        "status": "written",
        "typed_failure_reason": None,
        "output_path": output_path,
        "producer_invoked": "divergence_proxies/consume_trace_dir",
        "schema_admits": True,
    }


def _regenerate_order_projections_with_label_mpi(
    *,
    arch: str,
    trace_dir: Path,
    label_mpi: dict[str, str],
    sources: AdmittedTraceSources,
) -> dict[str, Any]:
    """Emit order projections from the batch's admitted capability."""
    device_block = _ORDER_PROJECTIONS_DEVICE_BLOCKS.get(arch)
    capacity_bytes = _ORDER_PROJECTIONS_CAPACITY_BYTES.get(arch)
    if device_block is None or capacity_bytes is None:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": f"unknown_arch:{arch}",
            "output_path": None,
            "producer_invoked": "order_projections/compute_order_projections",
        }
    try:
        from tools.order_projections.projector import (
            compute_order_projections_from_admitted_sources,
        )
        bundle = compute_order_projections_from_admitted_sources(
            sources=sources,
            manifest_pairing_index=label_mpi,
            device=device_block,
            capacity_bytes=capacity_bytes,
        )
    except _HARD_TRACE_SOURCE_ERRORS:
        raise
    except (
        RuntimeError, ImportError, OSError, ValueError,
        FileNotFoundError,
    ) as exc:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"compute_order_projections_raised:"
                f"{type(exc).__name__}:{exc!s}"
            )[:512],
            "output_path": None,
            "producer_invoked": "order_projections/compute_order_projections",
        }
    op_dir = trace_dir / "order_projections"
    output_path = op_dir / "order_projections.json"
    ensure_directory(op_dir)
    replace_regular_bytes(
        output_path,
        (json.dumps(bundle, indent=2) + "\n").encode("utf-8"),
    )
    try:
        import jsonschema  # noqa: E402
        schema_path = (
            _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
            / "order_projections.schema.json"
        )
        schema = json.loads(read_regular(schema_path))
        on_disk = json.loads(read_regular(output_path))
        validator = jsonschema.Draft202012Validator(schema)
        errors = list(validator.iter_errors(on_disk))
    except _HARD_TRACE_SOURCE_ERRORS:
        raise
    except (OSError, json.JSONDecodeError, ImportError) as exc:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"order_projections_schema_load_failed:"
                f"{type(exc).__name__}:{exc!s}"
            )[:512],
            "output_path": str(output_path),
            "producer_invoked": "order_projections/compute_order_projections",
        }
    if errors:
        first = errors[0]
        first_failure = (
            f"{list(first.absolute_path)}: {first.message}"
        )
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"order_projections_schema_invalid:{first_failure}"
            )[:512],
            "output_path": str(output_path),
            "producer_invoked": "order_projections/compute_order_projections",
            "schema_admits": False,
        }
    return {
        "status": "written",
        "typed_failure_reason": None,
        "output_path": str(output_path),
        "producer_invoked": "order_projections/compute_order_projections",
        "schema_admits": True,
    }


def regenerate_sidecars_for_cell(
    *,
    arch: str,
    slug: str,
    shape_point: str,
    variant_id: str | None = None,
    dry_run: bool = False,
    label_path: Path = _LABEL_ADMITTED,
) -> dict[str, Any]:
    """Regenerate the six trace sidecars for one cell.

    Returns a per-cell report dict with the cell key, a list of
    per-producer status entries (including the sector_stream MPI
    sibling), and a top-level overall_status summary. Admission, immutable
    artifact, resource, and limit failures escape unchanged and stop the batch;
    ordinary producer/schema failures retain explicit status entries.

    The six producers, in order:
      1. ``compute_sharing_locality`` →
         ``sharing_locality/sharing_locality.json``
      2. ``emit_hrd_bundle`` → ``hrd_profile/hrd_profile.json``
      3. ``sector_stream/consume_trace`` →
         ``sector_stream/sector_stream.jsonl``
      4. ``sector_stream_mpi_sibling`` →
         ``sector_stream/manifest_pairing_index.json``
      5. ``divergence_proxies/consume_trace_dir`` →
         ``divergence_proxies/divergence_proxies.json``
      6. ``order_projections/compute_order_projections`` →
         ``order_projections/order_projections.json``

    Producers 5 + 6 are the task4 sidecar surfaces beyond the
    initial sharing-locality/HRD/sector-stream trio: their bundles
    encode divergence proxies and the order-projections variance
    scalar respectively. Both schemas pin closed 5-field MPI
    objects, so the label MPI flows through without ``arch``
    leaking in.

    ``dry_run`` resolves the trace dir + the cell MPI WITHOUT
    invoking any producer. Useful for the replayability test path.
    """
    cell_key = (
        f"{arch}/{slug}/{shape_point}" if variant_id is None
        else f"{arch}/{slug}/{shape_point}/{variant_id}"
    )
    label_mpi = label_mpi_for_cell(
        arch=arch, slug=slug, shape_point=shape_point,
        variant_id=variant_id, label_path=label_path,
    )
    if label_mpi is None:
        return {
            "cell_key": cell_key,
            "trace_dir": None,
            "overall_status": "label_mpi_unresolved",
            "label_mpi_present": False,
            "producers": [],
        }
    trace_dir = _resolve_trace_dir(
        arch=arch, slug=slug, shape_point=shape_point, variant_id=variant_id,
    )
    if trace_dir is None:
        return {
            "cell_key": cell_key,
            "trace_dir": None,
            "overall_status": "trace_dir_unresolved",
            "label_mpi_present": True,
            "producers": [
                {
                    "producer": producer,
                    "status": "skipped",
                    "typed_failure_reason": "trace_dir_unresolved",
                    "output_path": None,
                }
                for producer in (
                    "compute_sharing_locality",
                    "emit_hrd_bundle",
                    "sector_stream/consume_trace",
                    "sector_stream_mpi_sibling",
                    "divergence_proxies/consume_trace_dir",
                    "order_projections/compute_order_projections",
                )
            ],
        }
    if dry_run:
        return {
            "cell_key": cell_key,
            "trace_dir": str(trace_dir),
            "overall_status": "dry_run_skipped",
            "label_mpi_present": True,
            "producers": [
                {
                    "producer": producer,
                    "status": "dry_run_skipped",
                    "typed_failure_reason": None,
                    "output_path": None,
                }
                for producer in (
                    "compute_sharing_locality",
                    "emit_hrd_bundle",
                    "sector_stream/consume_trace",
                    "sector_stream_mpi_sibling",
                    "divergence_proxies/consume_trace_dir",
                    "order_projections/compute_order_projections",
                )
            ],
        }
    admission = require_trace_admission(trace_dir)
    if admission.launch_count != 1:
        raise TraceAdmissionError(
            "trace_launch_count_unsupported",
            "coherent sidecar regeneration requires exactly one admitted "
            f"launch; observed launch_count={admission.launch_count}",
        )
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        sector_stream = consume_admitted_trace_sources(
            sources=sources,
            output_root=trace_dir,
            write_jsonl=True,
        )
        if not sector_stream.schema_admits_all:
            raise TraceAdmissionError(
                "sector_stream_schema_invalid",
                "sidecar regeneration sector-stream schema rejection: "
                f"{sector_stream.schema_first_failure}",
            )
        sector_stream.verify_for(sources)
        sector = {
            "status": "written",
            "typed_failure_reason": None,
            "output_path": str(sector_stream.jsonl_path),
        }
        sharing = _regenerate_sharing_locality_with_label_mpi(
            arch=arch,
            trace_dir=trace_dir,
            label_mpi=label_mpi,
            sources=sources,
        )
        hrd = _regenerate_hrd_profile_with_label_mpi(
            arch=arch,
            trace_dir=trace_dir,
            label_mpi=label_mpi,
            sources=sources,
            sector_stream=sector_stream,
        )
        sector_mpi = _write_sector_stream_mpi_sibling(
            trace_dir=trace_dir, label_mpi=label_mpi,
        )
        divergence = _regenerate_divergence_proxies_with_label_mpi(
            arch=arch,
            trace_dir=trace_dir,
            label_mpi=label_mpi,
            sources=sources,
            sector_stream=sector_stream,
        )
        order = _regenerate_order_projections_with_label_mpi(
            arch=arch,
            trace_dir=trace_dir,
            label_mpi=label_mpi,
            sources=sources,
        )
        sources.verify_unchanged()
    per_producer = [
        {
            "producer": "compute_sharing_locality",
            "status": sharing.get("status"),
            "typed_failure_reason": sharing.get("typed_failure_reason"),
            "output_path": sharing.get("output_path"),
        },
        {
            "producer": "emit_hrd_bundle",
            "status": hrd.get("status"),
            "typed_failure_reason": hrd.get("typed_failure_reason"),
            "output_path": hrd.get("output_path"),
        },
        {
            "producer": "sector_stream/consume_trace",
            "status": sector.get("status"),
            "typed_failure_reason": sector.get("typed_failure_reason"),
            "output_path": sector.get("output_path"),
        },
        {
            "producer": "sector_stream_mpi_sibling",
            "status": sector_mpi.get("status"),
            "typed_failure_reason": sector_mpi.get("typed_failure_reason"),
            "output_path": sector_mpi.get("output_path"),
        },
        {
            "producer": "divergence_proxies/consume_trace_dir",
            "status": divergence.get("status"),
            "typed_failure_reason": divergence.get("typed_failure_reason"),
            "output_path": divergence.get("output_path"),
        },
        {
            "producer": "order_projections/compute_order_projections",
            "status": order.get("status"),
            "typed_failure_reason": order.get("typed_failure_reason"),
            "output_path": order.get("output_path"),
        },
    ]
    all_written = all(p["status"] == "written" for p in per_producer)
    overall = "all_written" if all_written else "partial_or_failed"
    return {
        "cell_key": cell_key,
        "trace_dir": str(trace_dir),
        "overall_status": overall,
        "label_mpi_present": True,
        "producers": per_producer,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug", required=True,
        help="Slug name (e.g., bf16_hmma_gemm_traceable).",
    )
    parser.add_argument(
        "--shape-point", required=True,
        help="Shape point (e.g., trace_l2_accuracy_xl, timing_shape).",
    )
    parser.add_argument(
        "--arch", required=True,
        help="Cell architecture (sm_86 or sm_120).",
    )
    parser.add_argument(
        "--variant-id", default=None,
        help=(
            "Optional build-variant id (e.g. register_cap_64). When "
            "set, resolves the variants/<variant_id>/ trace subtree "
            "and the matching variant label row; when omitted, the "
            "base cell's canonical trace dir + label row are used."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Resolve the trace dir + label MPI + emit the report "
            "shape WITHOUT invoking any producer. Useful for the "
            "replayability test path."
        ),
    )
    args = parser.parse_args(argv)
    report = regenerate_sidecars_for_cell(
        arch=args.arch,
        slug=args.slug,
        shape_point=args.shape_point,
        variant_id=args.variant_id,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    return 0 if report.get("overall_status") == "all_written" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
