"""Regenerate the per-shape trace sidecars
(`sharing_locality.json`, `hrd_profile.json`, `sector_stream.jsonl`)
for the v5 cohort cells that the staging manifest marks as
``REGENERATE_*_FROM_TRACE``.

Reads `data/local_labels/v5_cohort_staging_manifest.json`, dispatches
each REGENERATE row to the appropriate producer, and emits a
machine-readable JSON report at
`data/local_labels/v5_trace_sidecar_regeneration_report.json`
with per-cell action, output paths, status, and typed failure
reason.

Producers invoked:
  - REGENERATE_SHARING_LOCALITY_FROM_TRACE ->
    `tools.sharing_locality.classifier.compute_sharing_locality`
    (in-process; the bundle is JSON-serialized + written to
    `trace_dir / "sharing_locality" / "sharing_locality.json"`).
  - REGENERATE_HRD_PROFILE_FROM_TRACE ->
    `tools.hrd_profiler.bundle_from_trace.emit_hrd_bundle` in-process
    call with REAL per-cell `manifest_pairing_index` resolved from
    the admitted predictor bundle on disk + arch-specific device
    block + L2 cache inputs. Fails the cell with a typed reason
    when MPI cannot be resolved (no placeholder identity is ever
    written). The earlier subprocess CLI path was retired because
    `bundle_from_trace.py`'s CLI fabricates placeholder MPI for
    ad-hoc operator inspection and would silently corrupt
    downstream provenance on a batch run (see BL-20260527-producer-
    cli-placeholder-mpi-vs-production-api).
  - REGENERATE_SECTOR_STREAM_FROM_TRACE ->
    `tools/sector_stream/consume_trace.py` subprocess invocation
    (`--trace-dir <PATH>`).

A regenerated sidecar lands on disk only when the producer succeeds;
on producer failure, the report records the typed failure reason
(no on-disk sidecar is written). This preserves the manifest's
audit visibility without partial state.

CLI:

    python tools/local_labels/regenerate_v5_trace_sidecars.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.local_labels.sm120_template_safety import (  # noqa: E402
    is_sm120_template_arch,
    unavailable_result,
)


_DEFAULT_MANIFEST = (
    REPO_ROOT / "data" / "local_labels"
    / "v5_cohort_staging_manifest.json"
)
_DEFAULT_REPORT = (
    REPO_ROOT / "data" / "local_labels"
    / "v5_trace_sidecar_regeneration_report.json"
)
_SECTOR_STREAM_PRODUCER = (
    REPO_ROOT / "tools" / "sector_stream" / "consume_trace.py"
)


# Device-property blocks the sharing_locality classifier requires
# (4-field minimal). Hardcoded for the project's two local devices
# per CLAUDE.md's two-device local development model.
_DEVICE_BLOCKS: dict[str, dict[str, Any]] = {
    "sm_86": {
        "device_name": "NVIDIA GeForce RTX 3090 Ti",
        "compute_capability_major": 8,
        "compute_capability_minor": 6,
        "sm_count": 84,
    },
    "sm_120": {
        "device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "compute_capability_major": 12,
        "compute_capability_minor": 0,
        "sm_count": 36,
    },
}


# Extended device blocks for the HRD bundle producer (which also
# needs total_global_mem_bytes + l2_cache_size_bytes per
# `emit_hrd_bundle`'s device shape). The values for sm_86 mirror
# `_static_card_l2_bytes_for_props` (6 MiB L2) + the RTX 3090 Ti's
# 24 GiB global memory; sm_120 mirrors the 32 MiB L2 +
# the RTX 5060 Laptop's 8 GiB global memory.
_HRD_DEVICE_BLOCKS: dict[str, dict[str, Any]] = {
    "sm_86": {
        **_DEVICE_BLOCKS["sm_86"],
        "total_global_mem_bytes": 24 * 1024 * 1024 * 1024,
        "l2_cache_size_bytes": 6 * 1024 * 1024,
    },
    "sm_120": {
        **_DEVICE_BLOCKS["sm_120"],
        "total_global_mem_bytes": 8 * 1024 * 1024 * 1024,
        "l2_cache_size_bytes": 32 * 1024 * 1024,
    },
}


# Per-arch L2 cache inputs the HRD producer's
# `resolve_cache_model_spec` expects. Mirror the static-card
# fallbacks in `workloads/simbench_bw/run_suite.py:_static_card_l2_bytes_for_props`
# and `_static_card_num_l2_slices_for_props`.
_HRD_CACHE_INPUTS: dict[str, dict[str, int | None]] = {
    "sm_86": {
        "cuda_l2_cache_size_bytes": 6 * 1024 * 1024,
        "static_card_l2_cache_size_bytes": 6 * 1024 * 1024,
        "cuda_num_l2_slices": 6,
        "static_card_num_l2_slices": 6,
    },
    "sm_120": {
        "cuda_l2_cache_size_bytes": 32 * 1024 * 1024,
        "static_card_l2_cache_size_bytes": 32 * 1024 * 1024,
        # The Blackwell L2 slice count is not yet captured in the
        # static card (per `run_suite._static_card_num_l2_slices_for_props`).
        "cuda_num_l2_slices": None,
        "static_card_num_l2_slices": None,
    },
}


def _device_dir_for_arch(arch: str) -> str:
    return {"sm_86": "rtx_3090_ti", "sm_120": "rtx_5060_laptop"}.get(
        arch, "unknown",
    )


def _resolve_trace_dir(
    *, arch: str, slug: str, shape_point: str, variant_id: str | None = None,
) -> Path | None:
    """Resolve the per-cell trace directory under the simbench or
    portable_lcd results roots. Returns None when the directory
    is not on disk under either root.

    ``variant_id`` is optional and purely additive: when None the
    canonical base trace dir ``.../<arch>/trace_<shape_point>/`` is
    resolved (byte-identical legacy behavior); when set to a
    canonical build-variant id (e.g. ``register_cap_64``) the
    register-cap variant subtree
    ``.../<arch>/variants/<variant_id>/trace_<shape_point>/`` is
    resolved instead. The variant producer
    (``trace_register_cap_variant_cells.py``) writes the per-cap
    trace under that ``variants/<variant_id>/`` segment, so a base
    cell and its sibling caps never collide on one trace dir."""
    device_dir = _device_dir_for_arch(arch)
    leaf = (
        f"trace_{shape_point}" if variant_id is None
        else Path("variants") / variant_id / f"trace_{shape_point}"
    )
    candidates = [
        REPO_ROOT / "workloads" / "simbench_bw" / "results" / slug
        / device_dir / arch / leaf,
        REPO_ROOT / "workloads" / "portable_lcd" / "results" / slug
        / device_dir / arch / leaf,
    ]
    for c in candidates:
        if (c / "dynamic_trace.pb").is_file():
            return c
    return None


_REGEN_LEDGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "local_labels" / "sm86_cohort_regen_ledger.json"
)
_SWEEP_TS_RE = re.compile(r"(\d{8}T\d{6}Z)")

# The closed identity 5-tuple that every manifest_pairing_index carries
# (and that `sharing_locality.schema.json` closes over via
# `additionalProperties: false`). Both MPI resolvers key off this set.
_MPI_REQUIRED_KEYS: tuple[str, ...] = (
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
)


def _sweep_timestamp_key(path: Path) -> str:
    """Sortable timestamp token embedded in a sweep-dir name (empty
    string when absent, sorting historical un-stamped dirs first)."""
    for part in path.parts:
        m = _SWEEP_TS_RE.search(part)
        if m:
            return m.group(1)
    return ""


def _admitted_bundle_paths_newest_first(
    *, arch: str, slug: str, shape_point: str,
) -> list[Path]:
    """Admitted predictor-bundle paths for the cell, ordered NEWEST
    sweep first by the timestamp token in the sweep-dir name, so a
    stale historical sibling can never shadow a newer bundle for the
    same (arch, slug, shape_point)."""
    return sorted(
        REPO_ROOT.glob(
            f"data/local_labels/sweeps/**/predictor_bundles/"
            f"{arch}__{slug}__{shape_point}.json",
        ),
        key=_sweep_timestamp_key,
        reverse=True,
    )


def _ledger_bound_mpi(
    *, arch: str, slug: str, shape_point: str,
    ledger_path: Path = _REGEN_LEDGER_PATH,
) -> dict[str, Any] | None:
    """Resolve the ACTIVE admitted bundle's manifest_pairing_index via
    the regen ledger's per-cell ``predictor_bundle_path`` binding —
    the authoritative active-bundle source. A bare filesystem glob
    across ``sweeps/**`` returns the FIRST match in filesystem order,
    which with historical sweeps present can be a STALE sibling for
    the same (arch, slug, shape_point); stamping regenerated sidecars
    with that identity corrupts downstream joins (a Codex review P2).
    Digest-verified when the ledger records the bundle digest.
    Returns None when the ledger lacks an admitted base entry for the
    cell (callers fall back to the timestamp-ordered glob)."""
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for cell in ledger.get("cells", []):
        if (
            cell.get("status") == "admitted"
            and str(cell.get("arch")) == arch
            and str(cell.get("slug")) == slug
            and str(cell.get("shape_point")) == shape_point
            and cell.get("variant_id") is None
        ):
            rel = cell.get("predictor_bundle_path")
            if not rel:
                return None
            bundle_path = Path(rel)
            if not bundle_path.is_absolute():
                bundle_path = ledger_path.parents[2] / rel
            try:
                raw = bundle_path.read_bytes()
            except OSError:
                return None
            expected = (cell.get("artifact_sha256") or {}).get(
                "predictor_bundle",
            )
            if expected:
                if hashlib.sha256(raw).hexdigest() != expected:
                    return None
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            mpi = payload.get("manifest_pairing_index")
            return dict(mpi) if isinstance(mpi, dict) else None
    return None


def _load_manifest_pairing_index_for_cell(
    *, arch: str, slug: str, shape_point: str,
) -> dict[str, Any]:
    """Build a manifest_pairing_index for the sharing_locality
    classifier: ledger-bound active bundle first, then admitted
    bundles on disk (newest sweep first), then a minimal
    closed-5-tuple placeholder payload when no bundle is admitted
    yet. The MPI deliberately carries NO ``arch`` key: the closed
    identity 5-tuple convention (and ``sharing_locality.schema.json``'s
    ``additionalProperties: false`` MPI) reject it — ``arch`` routes
    the device-block selection in the caller, not the pairing
    identity."""
    ledger_mpi = _ledger_bound_mpi(
        arch=arch, slug=slug, shape_point=shape_point,
    )
    if ledger_mpi is not None:
        return ledger_mpi
    # Fallback: scan admitted bundle paths, NEWEST sweep first (the
    # timestamp token in the sweep-dir name), so a stale historical
    # sibling can never shadow a newer bundle for the same cell.
    for p in _admitted_bundle_paths_newest_first(
        arch=arch, slug=slug, shape_point=shape_point,
    ):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mpi = payload.get("manifest_pairing_index")
        if isinstance(mpi, dict):
            return dict(mpi)
    # Fallback: minimal MPI that satisfies the schema's required
    # keys with placeholder hex values (the regenerated sidecar
    # carries this MPI verbatim; a downstream consumer that joins
    # on the 5-field tuple will reject the placeholder, surfacing
    # the missing-bundle gap cleanly). NO `arch` key: the schema's
    # MPI is closed (additionalProperties: false) over the 5-tuple,
    # so adding arch would make the tool report the sidecar as
    # written while emitting schema-invalid bytes on exactly this
    # missing-bundle path.
    return {
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": "0" * 64,
        "source_commit": "0" * 40,
        "kernel_signature_sha256": "0" * 64,
    }


def _regenerate_sharing_locality(
    *, arch: str, slug: str, shape_point: str, trace_dir: Path,
) -> dict[str, Any]:
    """Invoke compute_sharing_locality in-process + write to
    trace_dir / sharing_locality / sharing_locality.json. Returns a
    per-cell report dict."""
    if is_sm120_template_arch(arch):
        return unavailable_result(
            operation="regenerate_v5_trace_sidecars.sharing_locality",
            arch=arch,
            cell_key=f"{arch}/{slug}/{shape_point}",
            producer_invoked=None,
            output_path=None,
        )
    device_block = _DEVICE_BLOCKS.get(arch)
    if device_block is None:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": f"unknown_arch:{arch}",
            "output_path": None,
            "producer_invoked": "compute_sharing_locality",
        }
    mpi = _load_manifest_pairing_index_for_cell(
        arch=arch, slug=slug, shape_point=shape_point,
    )
    try:
        sys.path.insert(
            0,
            str(REPO_ROOT / "tools" / "sharing_locality"),
        )
        from classifier import (  # type: ignore[import-not-found]
            compute_sharing_locality,
        )
        bundle = compute_sharing_locality(
            trace_dir=trace_dir,
            manifest_pairing_index=mpi,
            device=device_block,
        )
    except (RuntimeError, ImportError, OSError) as exc:
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
    sl_dir.mkdir(parents=True, exist_ok=True)
    output_path = sl_dir / "sharing_locality.json"
    try:
        output_path.write_text(
            json.dumps(bundle, indent=2) + "\n", encoding="utf-8",
        )
    except OSError as exc:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"write_failed:{type(exc).__name__}:{exc!s}"
            )[:512],
            "output_path": str(output_path),
            "producer_invoked": "compute_sharing_locality",
        }
    return {
        "status": "written",
        "typed_failure_reason": None,
        "output_path": str(output_path),
        "producer_invoked": "compute_sharing_locality",
    }


def _regenerate_via_subprocess(
    *, producer_path: Path, trace_dir: Path,
    expected_output_subpath: str, producer_label: str,
) -> dict[str, Any]:
    """Invoke a producer script via subprocess with --trace-dir and
    verify the expected sidecar landed on disk."""
    try:
        proc = subprocess.run(
            [sys.executable, str(producer_path), "--trace-dir", str(trace_dir)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"subprocess_failed:{type(exc).__name__}:{exc!s}"
            )[:512],
            "output_path": None,
            "producer_invoked": producer_label,
        }
    if proc.returncode != 0:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"producer_exit_{proc.returncode}:"
                f"stderr_tail:{proc.stderr[-300:]!r}"
            )[:512],
            "output_path": None,
            "producer_invoked": producer_label,
        }
    expected_output = trace_dir / expected_output_subpath
    if not expected_output.is_file():
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"producer_returned_zero_but_no_output_at:"
                f"{expected_output}"
            )[:512],
            "output_path": str(expected_output),
            "producer_invoked": producer_label,
        }
    return {
        "status": "written",
        "typed_failure_reason": None,
        "output_path": str(expected_output),
        "producer_invoked": producer_label,
    }


def _resolve_real_mpi_from_admitted_bundle(
    *, arch: str, slug: str, shape_point: str,
) -> dict[str, Any] | None:
    """Find the admitted predictor bundle for (arch, slug, shape_point)
    and return its ``manifest_pairing_index`` dict. Returns None when
    no such bundle exists on disk — the caller fails the cell with
    a typed reason rather than fabricating placeholder identity.

    The HRD producer module documents that callers MUST supply real
    per-cell MPI; fabricating placeholder values (as the
    bundle_from_trace.py CLI does for ad-hoc inspection) silently
    corrupts downstream provenance.

    Resolution order (a Codex review P2: a bare first-match glob
    across ALL historical sweeps can return a STALE sibling's
    identity): the regen LEDGER's per-cell ``predictor_bundle_path``
    binding first (digest-verified — the authoritative active
    bundle), then on-disk bundles ordered NEWEST sweep first by the
    timestamp token in the sweep-dir name.
    """
    ledger_mpi = _ledger_bound_mpi(
        arch=arch, slug=slug, shape_point=shape_point,
    )
    if ledger_mpi is not None:
        if all(ledger_mpi.get(k) for k in _MPI_REQUIRED_KEYS):
            return ledger_mpi
    for p in _admitted_bundle_paths_newest_first(
        arch=arch, slug=slug, shape_point=shape_point,
    ):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mpi = payload.get("manifest_pairing_index")
        if not isinstance(mpi, dict):
            continue
        if all(mpi.get(k) for k in _MPI_REQUIRED_KEYS):
            return dict(mpi)
    return None


def _regenerate_hrd_profile(
    *, arch: str, slug: str, shape_point: str, trace_dir: Path,
) -> dict[str, Any]:
    """In-process HRD bundle producer call. Resolves real per-cell
    MPI from the admitted predictor bundle on disk + arch-specific
    device block + L2 cache inputs, then calls
    ``tools.hrd_profiler.bundle_from_trace.emit_hrd_bundle`` to
    write ``<trace_dir>/hrd_profile/hrd_profile.json``.

    Fails the cell with a typed reason when:
      - the arch is unknown (no _HRD_DEVICE_BLOCKS entry),
      - no admitted predictor bundle exists for the cell (so MPI
        cannot be resolved without fabricating placeholder values),
      - the producer raises any exception,
      - the producer returns ``contract_ok=False``.

    Replaces the Round-15 CLI subprocess path that the upstream
    bundle_from_trace.py module documents as "ad-hoc convenience"
    + that fabricates placeholder MPI / defaults to sm_120 device.
    """
    if is_sm120_template_arch(arch):
        return unavailable_result(
            operation="regenerate_v5_trace_sidecars.hrd_profile",
            arch=arch,
            cell_key=f"{arch}/{slug}/{shape_point}",
            producer_invoked=None,
            output_path=None,
        )
    device_block = _HRD_DEVICE_BLOCKS.get(arch)
    if device_block is None:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": f"unknown_arch:{arch}",
            "output_path": None,
            "producer_invoked": "emit_hrd_bundle",
        }
    cache_inputs = _HRD_CACHE_INPUTS.get(arch, {})
    mpi = _resolve_real_mpi_from_admitted_bundle(
        arch=arch, slug=slug, shape_point=shape_point,
    )
    if mpi is None:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                "hrd_mpi_unresolvable:no_admitted_predictor_bundle_for_cell"
            ),
            "output_path": None,
            "producer_invoked": "emit_hrd_bundle",
        }
    # Lazy in-process import. The producer module's path lives under
    # tools/hrd_profiler/; sys.path is injected so the bare
    # `bundle_from_trace` import that the module uses internally
    # also resolves.
    try:
        sys.path.insert(
            0, str(REPO_ROOT / "tools" / "hrd_profiler"),
        )
        import bundle_from_trace  # type: ignore[import-not-found]
        result = bundle_from_trace.emit_hrd_bundle(
            trace_dir=trace_dir,
            manifest_pairing_index=mpi,
            device=device_block,
            cuda_l2_cache_size_bytes=(
                cache_inputs.get("cuda_l2_cache_size_bytes")
            ),
            static_card_l2_cache_size_bytes=(
                cache_inputs.get("static_card_l2_cache_size_bytes")
            ),
            cuda_num_l2_slices=cache_inputs.get("cuda_num_l2_slices"),
            static_card_num_l2_slices=(
                cache_inputs.get("static_card_num_l2_slices")
            ),
        )
    except (RuntimeError, ImportError, OSError, ValueError) as exc:
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"emit_hrd_bundle_raised:{type(exc).__name__}:{exc!s}"
            )[:512],
            "output_path": None,
            "producer_invoked": "emit_hrd_bundle",
        }
    if not result.get("contract_ok", False):
        return {
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"emit_hrd_bundle_contract_failed:"
                f"{result.get('contract_typed_reason') or 'unknown'}"
            )[:512],
            "output_path": None,
            "producer_invoked": "emit_hrd_bundle",
        }
    output_path = trace_dir / "hrd_profile" / "hrd_profile.json"
    return {
        "status": "written",
        "typed_failure_reason": None,
        "output_path": str(output_path),
        "producer_invoked": "emit_hrd_bundle",
    }


def _regenerate_sector_stream(*, trace_dir: Path, arch: str | None = None) -> dict[str, Any]:
    if arch is not None and is_sm120_template_arch(arch):
        return unavailable_result(
            operation="regenerate_v5_trace_sidecars.sector_stream",
            arch=arch,
            producer_invoked=None,
            output_path=None,
        )
    return _regenerate_via_subprocess(
        producer_path=_SECTOR_STREAM_PRODUCER,
        trace_dir=trace_dir,
        expected_output_subpath="sector_stream/sector_stream.jsonl",
        producer_label="sector_stream/consume_trace.py",
    )


def _action_to_producer(
    action: str,
) -> Any | None:
    """Map a REGENERATE_*_FROM_TRACE action to the regeneration helper."""
    return {
        "REGENERATE_SHARING_LOCALITY_FROM_TRACE": (
            _regenerate_sharing_locality
        ),
        "REGENERATE_HRD_PROFILE_FROM_TRACE": _regenerate_hrd_profile,
        "REGENERATE_SECTOR_STREAM_FROM_TRACE": _regenerate_sector_stream,
    }.get(action)


def regenerate_manifest_regenerate_rows(
    *,
    manifest_path: Path,
    report_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Read the manifest, dispatch each REGENERATE_*_FROM_TRACE row,
    write a JSON report. Returns the report dict for testability."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", []) or []
    per_cell_reports: list[dict[str, Any]] = []
    counts = {
        "total_regenerate_rows": 0,
        "written": 0,
        "failed_with_typed_reason": 0,
        "trace_dir_unresolved": 0,
        "dry_run_skipped": 0,
    }
    for row in rows:
        action = row.get("suggested_operator_action", "")
        if not action.startswith("REGENERATE_"):
            continue
        counts["total_regenerate_rows"] += 1
        arch = str(row.get("arch", ""))
        slug = str(row.get("slug", ""))
        shape_point = str(row.get("shape_point", ""))
        cell_key = f"{arch}/{slug}/{shape_point}"
        if is_sm120_template_arch(arch):
            # Historical sm_120 sidecars remain untouched. Do not resolve a
            # trace or dispatch a producer merely because a manifest asks for
            # regeneration; represent the blocked lane explicitly.
            per_cell_reports.append({
                **unavailable_result(
                    operation="regenerate_v5_trace_sidecars",
                    arch=arch,
                    cell_key=cell_key,
                    producer_invoked=None,
                    output_path=None,
                    requested_action=action,
                ),
                "cell_key": cell_key,
                "requested_action": action,
            })
            counts["typed_unavailable"] = counts.get(
                "typed_unavailable", 0,
            ) + 1
            continue
        producer = _action_to_producer(action)
        if producer is None:
            per_cell_reports.append({
                "cell_key": cell_key,
                "requested_action": action,
                "producer_invoked": None,
                "output_path": None,
                "status": "failed_with_typed_reason",
                "typed_failure_reason": f"unknown_action:{action}",
            })
            counts["failed_with_typed_reason"] += 1
            continue
        trace_dir = _resolve_trace_dir(
            arch=arch, slug=slug, shape_point=shape_point,
        )
        if trace_dir is None:
            per_cell_reports.append({
                "cell_key": cell_key,
                "requested_action": action,
                "producer_invoked": None,
                "output_path": None,
                "status": "failed_with_typed_reason",
                "typed_failure_reason": (
                    "trace_dir_unresolved:dynamic_trace.pb_absent_under_"
                    "simbench_or_portable_lcd_roots"
                ),
            })
            counts["trace_dir_unresolved"] += 1
            counts["failed_with_typed_reason"] += 1
            continue
        if dry_run:
            per_cell_reports.append({
                "cell_key": cell_key,
                "requested_action": action,
                "producer_invoked": None,
                "output_path": None,
                "status": "dry_run_skipped",
                "typed_failure_reason": None,
                "trace_dir": str(trace_dir),
            })
            counts["dry_run_skipped"] += 1
            continue
        # Dispatch by action type. The sharing_locality + hrd_profile
        # helpers both need arch/slug/shape_point/trace_dir (the HRD
        # helper resolves real per-cell MPI from the admitted bundle
        # to avoid fabricating placeholder identity, per Codex Round
        # 15 review's HRD-fix demand); the sector_stream helper is
        # still subprocess-only because `consume_trace.py --trace-dir`
        # is a real producer CLI that does not fabricate cell
        # identity.
        if action in (
            "REGENERATE_SHARING_LOCALITY_FROM_TRACE",
            "REGENERATE_HRD_PROFILE_FROM_TRACE",
        ):
            result = producer(
                arch=arch, slug=slug, shape_point=shape_point,
                trace_dir=trace_dir,
            )
        else:
            result = producer(trace_dir=trace_dir)
        result["cell_key"] = cell_key
        result["requested_action"] = action
        per_cell_reports.append(result)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    report = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "dry_run": dry_run,
        "counts": counts,
        "per_cell_reports": per_cell_reports,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(_DEFAULT_MANIFEST))
    parser.add_argument("--report", default=str(_DEFAULT_REPORT))
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Resolve trace directories + emit the report WITHOUT "
            "invoking any producer. Useful for checking which cells "
            "are reachable before the live regeneration sweep."
        ),
    )
    args = parser.parse_args(argv)
    report = regenerate_manifest_regenerate_rows(
        manifest_path=Path(args.manifest),
        report_path=Path(args.report),
        dry_run=args.dry_run,
    )
    counts = report["counts"]
    print(
        f"[v5_trace_sidecar_regen] "
        f"rows={counts['total_regenerate_rows']} "
        f"written={counts['written']} "
        f"failed={counts['failed_with_typed_reason']} "
        f"typed_unavailable={counts.get('typed_unavailable', 0)} "
        f"dry_run_skipped={counts['dry_run_skipped']} "
        f"-> {args.report}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
