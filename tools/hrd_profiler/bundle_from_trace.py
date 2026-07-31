"""Live HRD profile bundle producer (task11 / the HRD-profiler clause live-evidence path).

The capability-backed producer walks admitted threadblock snapshots, computes
one HRD observation per active+predicated lane × touched 32B sector, and groups
observations by kernel-launch directory. Sector-stream records are supplied
in-process by the sector consumer and are never reloaded from a pre-existing
JSONL artifact.

The in-process API (`emit_hrd_bundle`) accepts real
`manifest_pairing_index`, `device`, and `cache_model_resolver_input`
from the caller (`run_suite.py::handle_trace`); a contract-failure
return path (``contract_ok=False``) propagates the typed reason so
the controller can surface a typed `hrd_skip_reason` rather than
emitting a bundle from broken evidence. The CLI keeps ad-hoc
convenience defaults for operator inspection.

The bundle is strict-validated against
``workloads/simbench_bw/schemas/hrd_profile.schema.json`` via
``jsonschema.Draft202012Validator`` (R7 exact-artifact discipline).

CLI:

    python tools/hrd_profiler/bundle_from_trace.py \\
        --trace-dir <PATH> \\
        [--device sm_86 | sm_120-laptop]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import jsonschema

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.hrd_profiler.cache_model_spec import (  # noqa: E402
    resolve_cache_model_spec,
)
from tools.hrd_profiler.observation_builder import (  # noqa: E402
    build_launches_from_admitted_sources,
)
from tools.hrd_profiler.profiler import (  # noqa: E402
    HRD_MAX_SECTOR_STREAM_RECORDS,
    compute_hrd_profile,
)
from tools.local_labels.immutable_artifact import (  # noqa: E402
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
    consume_admitted_trace_sources as consume_sector_stream_sources,
)
_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "hrd_profile.schema.json"
)


def _load_schema() -> dict[str, Any]:
    schema = json.loads(read_regular(_SCHEMA_PATH))
    if not isinstance(schema, dict):
        raise TypeError("HRD schema must be a JSON object")
    return schema


def _read_stats(raw: bytes, *, label: str = "stats.csv") -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        raise ValueError(f"{label} is empty")
    header = [h.strip() for h in rows[0]]
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(f"{label} has no data rows")
    cells = [c.strip() for c in data_rows[0]]
    return dict(zip(header, cells))


def _kernel_signature_sha256_from_stats(stats: dict[str, Any]) -> str:
    """Stable placeholder derived from the kernel mangled name.
    The simbench manifest-pairing index requires a string field
    that does not collide across kernels; a future round-added
    simbench writer extension will surface a real signature SHA."""
    import hashlib
    name = stats.get("kernel mangled name", "")
    if not name:
        return "0" * 64
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _write_bundle(
    output_root: Path, bundle: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    out_dir = Path(output_root) / "hrd_profile"
    ensure_directory(out_dir)
    out_path = out_dir / "hrd_profile.json"
    payload = (json.dumps(bundle, indent=2) + "\n").encode("utf-8")
    replace_regular_bytes(out_path, payload)
    on_disk = json.loads(read_regular(out_path))
    if not isinstance(on_disk, dict):
        raise TypeError("serialized HRD bundle must be a JSON object")
    return out_path, on_disk


def _budget_exceeded_typed_bundle(
    *,
    manifest_pairing_index: dict[str, str],
    device: dict[str, Any],
    cache_spec: dict[str, Any],
    sector_stream_record_count: int,
) -> dict[str, Any]:
    """Emit a schema-valid hrd_profile.json typed-unavailable
    bundle with bundle-level
    `typed_reason == "hrd_compute_budget_exceeded"`.

    The schema's top-level `result_kind` is pinned to "ok", so
    the typed-unavailable arm goes through the bundle-level
    `evidence` annotation (`scope == "bundle"`, `result_kind ==
    "typed_unavailable"`, `typed_reason ==
    "hrd_compute_budget_exceeded"`). `bins` is empty (the
    reuse-distance computation was skipped) and
    `bundle_underconfigured` is True so any consumer that
    reads the cached annotation as a fast path correctly
    refuses.

    The bundle still carries the manifest/device/cache-model
    identity so the verifier can bind the typed-unavailable
    arm to the run summary's manifest_pairing_index fields.
    """
    return {
        "schema_version": 1,
        "result_kind": "ok",
        "manifest_pairing_index": manifest_pairing_index,
        "device": {**device, "l2_cache_size_bytes": cache_spec["capacity_bytes"]},
        "cache_model_spec": cache_spec,
        "binning": {
            "unit": "sector_32b",
            "rd_infinity_sentinel": 9223372036854775807,
            "ci_method": "percentile_bootstrap",
            "bootstrap_resamples": 2000,
            "ci_confidence": 0.95,
            "ci_half_width_target": 0.05,
            "minimum_bin_sample_count": 50,
        },
        "resampling": {"unit": "kernel_launch"},
        "bins": [],
        "bundle_underconfigured": True,
        "evidence": [
            {
                "result_kind": "typed_unavailable",
                "typed_reason": "hrd_compute_budget_exceeded",
                "scope": "bundle",
                "scalar_hit_rate_refusal": True,
            }
        ],
    }


def _geometry_unavailable_typed_bundle(
    *,
    manifest_pairing_index: dict[str, str],
    device: dict[str, Any],
    cache_spec: dict[str, Any],
) -> dict[str, Any]:
    spec = dict(cache_spec)
    if spec.get("num_l2_slices") is None:
        spec.pop("num_l2_slices", None)
        spec.pop("num_l2_slices_source", None)
    return {
        "schema_version": 1,
        "result_kind": "ok",
        "manifest_pairing_index": manifest_pairing_index,
        "device": {**device, "l2_cache_size_bytes": spec["capacity_bytes"]},
        "cache_model_spec": spec,
        "binning": {
            "unit": "sector_32b",
            "rd_infinity_sentinel": 9223372036854775807,
            "ci_method": "percentile_bootstrap",
            "bootstrap_resamples": 2000,
            "ci_confidence": 0.95,
            "ci_half_width_target": 0.05,
            "minimum_bin_sample_count": 50,
        },
        "resampling": {"unit": "kernel_launch"},
        "bins": [],
        "bundle_underconfigured": True,
        "evidence": [{
            "result_kind": "typed_unavailable",
            "typed_reason": "cache_model_geometry_missing",
            "scope": "bundle",
        }],
    }


def emit_hrd_bundle_from_admitted_sources(
    *,
    sources: AdmittedTraceSources,
    sector_stream: AdmittedSectorStream,
    output_root: Path,
    manifest_pairing_index: dict[str, str],
    device: dict[str, Any],
    cuda_l2_cache_size_bytes: int | None,
    static_card_l2_cache_size_bytes: int | None = None,
    cuda_num_l2_slices: int | None = None,
    static_card_num_l2_slices: int | None = None,
    architecture: str | None = None,
) -> dict[str, Any]:
    """Capability-only HRD producer for one admitted launch.

    ``sources`` and the admission-bound ``sector_stream`` are mandatory
    keyword-only inputs. This core never opens a pre-existing JSONL artifact;
    ``output_root`` is used only as the generated bundle destination.
    """
    if not isinstance(sources, AdmittedTraceSources):
        raise TypeError("sources must be an AdmittedTraceSources capability")
    if not isinstance(sector_stream, AdmittedSectorStream):
        raise TypeError("sector_stream must be an AdmittedSectorStream")
    sources.verify_unchanged()
    admission = sources.admission
    if admission.launch_count != 1:
        raise TraceAdmissionError(
            "trace_launch_count_unsupported",
            "HRD profile schema requires exactly one admitted launch; "
            f"observed launch_count={admission.launch_count}",
        )
    sector_stream.verify_for(sources)
    if len(sources.threadblock_paths) != admission.threadblock_count:
        raise TraceAdmissionError(
            "trace_snapshot_changed",
            "admitted threadblock path count disagrees with admission",
        )
    sector_stream_record_count = len(sector_stream.records)

    cache_spec, cache_evidence = resolve_cache_model_spec(
        cuda_l2_cache_size_bytes=cuda_l2_cache_size_bytes,
        static_card_l2_cache_size_bytes=static_card_l2_cache_size_bytes,
        cuda_num_l2_slices=cuda_num_l2_slices,
        static_card_num_l2_slices=static_card_num_l2_slices,
        architecture=architecture,
    )

    geometry_unavailable = any(
        item.get("typed_reason") == "cache_model_geometry_missing"
        for item in cache_evidence
        if isinstance(item, dict)
    ) or cache_spec.get("num_l2_slices") is None
    if geometry_unavailable:
        bundle = _geometry_unavailable_typed_bundle(
            manifest_pairing_index=manifest_pairing_index,
            device=device,
            cache_spec=cache_spec,
        )
        out_path, on_disk = _write_bundle(output_root, bundle)
        errors = list(
            jsonschema.Draft202012Validator(_load_schema()).iter_errors(on_disk)
        )
        sources.verify_unchanged()
        return {
            "bundle_path": str(out_path),
            "bundle": bundle,
            "contract_ok": True,
            "contract_typed_reason": "cache_model_geometry_missing",
            "contract_detail": "cache-model geometry unresolved; walker not invoked",
            "schema_admits": not errors,
            "schema_first_failure": (
                f"{list(errors[0].absolute_path)}: {errors[0].message}"
                if errors else None
            ),
            "bundle_underconfigured": True,
            "bin_count": 0,
            "typed_unavailable_count": 0,
            "underconfigured_bin_count": 0,
            "low_confidence_bin_count": 0,
            "launches_consumed": 0,
            "total_observations": 0,
        }

    # Defense-in-depth budget guard uses the records produced in this process.
    if sector_stream_record_count > HRD_MAX_SECTOR_STREAM_RECORDS:
        bundle = _budget_exceeded_typed_bundle(
            manifest_pairing_index=manifest_pairing_index,
            device=device,
            cache_spec=cache_spec,
            sector_stream_record_count=sector_stream_record_count,
        )
        out_path, on_disk = _write_bundle(output_root, bundle)
        schema = _load_schema()
        validator = jsonschema.Draft202012Validator(schema)
        errors = list(validator.iter_errors(on_disk))
        schema_admits = len(errors) == 0
        first_failure = (
            f"{list(errors[0].absolute_path)}: {errors[0].message}"
            if errors else None
        )
        sources.verify_unchanged()
        return {
            "bundle_path": str(out_path),
            "bundle": bundle,
            "contract_ok": True,
            "contract_typed_reason": None,
            "contract_detail": None,
            "schema_admits": schema_admits,
            "schema_first_failure": first_failure,
            "bundle_underconfigured": bool(bundle["bundle_underconfigured"]),
            "bin_count": 0,
            "typed_unavailable_count": 0,
            "underconfigured_bin_count": 0,
            "low_confidence_bin_count": 0,
            "launches_consumed": 0,
            "total_observations": 0,
        }

    build_result = build_launches_from_admitted_sources(
        sources=sources,
        capacity_bytes=cache_spec["capacity_bytes"],
        num_l2_slices=cache_spec["num_l2_slices"],
    )
    if not build_result.contract_ok:
        sources.verify_unchanged()
        return {
            "bundle_path": None,
            "bundle": None,
            "contract_ok": False,
            "contract_typed_reason": build_result.contract_typed_reason,
            "contract_detail": build_result.contract_detail,
            "schema_admits": False,
            "schema_first_failure": None,
            "bundle_underconfigured": None,
            "bin_count": 0,
            "typed_unavailable_count": 0,
            "underconfigured_bin_count": 0,
            "low_confidence_bin_count": 0,
            "launches_consumed": 0,
            "total_observations": 0,
        }
    bundle = compute_hrd_profile(
        manifest_pairing_index=manifest_pairing_index,
        device={**device, "l2_cache_size_bytes": cache_spec["capacity_bytes"]},
        cache_model_spec=cache_spec,
        launches=build_result.launches,
        cache_model_evidence=cache_evidence,
    )
    out_path, on_disk = _write_bundle(output_root, bundle)
    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(on_disk))
    schema_admits = len(errors) == 0
    first_failure = (
        f"{list(errors[0].absolute_path)}: {errors[0].message}"
        if errors else None
    )
    typed_unavailable_count = sum(
        1 for b in bundle["bins"] if b["result_kind"] == "typed_unavailable"
    )
    underconfigured_bin_count = sum(
        1 for b in bundle["bins"]
        if b["result_kind"] == "typed_unavailable"
        and b.get("typed_reason") == "cache_model_underconfigured"
    )
    low_confidence_bin_count = sum(
        1 for b in bundle["bins"]
        if b["result_kind"] == "ok" and b.get("low_confidence")
    )
    sources.verify_unchanged()
    return {
        "bundle_path": str(out_path),
        "bundle": bundle,
        "contract_ok": True,
        "contract_typed_reason": None,
        "contract_detail": None,
        "schema_admits": schema_admits,
        "schema_first_failure": first_failure,
        "bundle_underconfigured": bool(bundle["bundle_underconfigured"]),
        "bin_count": len(bundle["bins"]),
        "typed_unavailable_count": typed_unavailable_count,
        "underconfigured_bin_count": underconfigured_bin_count,
        "low_confidence_bin_count": low_confidence_bin_count,
        "launches_consumed": len(build_result.launches),
        "total_observations": build_result.total_observations,
    }


def emit_hrd_bundle(
    *,
    trace_dir: Path,
    manifest_pairing_index: dict[str, str],
    device: dict[str, Any],
    cuda_l2_cache_size_bytes: int | None,
    static_card_l2_cache_size_bytes: int | None = None,
    cuda_num_l2_slices: int | None = None,
    static_card_num_l2_slices: int | None = None,
    architecture: str | None = None,
) -> dict[str, Any]:
    """Path-compatible wrapper that regenerates sector records in memory."""
    admission = require_trace_admission(trace_dir)
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        sector_result = consume_sector_stream_sources(
            sources=sources,
            output_root=None,
            write_jsonl=False,
        )
        if not sector_result.schema_admits_all:
            raise TraceAdmissionError(
                "sector_stream_schema_invalid",
                "sector-stream producer generated a schema-invalid record: "
                f"{sector_result.schema_first_failure}",
            )
        return emit_hrd_bundle_from_admitted_sources(
            sources=sources,
            sector_stream=sector_result,
            output_root=trace_dir,
            manifest_pairing_index=manifest_pairing_index,
            device=device,
            cuda_l2_cache_size_bytes=cuda_l2_cache_size_bytes,
            static_card_l2_cache_size_bytes=static_card_l2_cache_size_bytes,
            cuda_num_l2_slices=cuda_num_l2_slices,
            static_card_num_l2_slices=static_card_num_l2_slices,
            architecture=architecture,
        )


def consume_trace_dir(
    trace_dir: Path,
    device_arch: str,
) -> dict[str, Any]:
    """CLI convenience entry: read the trace directory, build a
    placeholder manifest/device block, call `emit_hrd_bundle`,
    return the status. Used by the CLI for ad-hoc operator
    inspection. Production callers (run_suite.py::handle_trace)
    use `emit_hrd_bundle` directly with real inputs."""
    admission = require_trace_admission(trace_dir)
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        stats = _read_stats(sources.read_stats_snapshot().payload)
        manifest_pairing_index = {
            "slug": "dram_stream_stride_mix",
            "shape_point": "trace_00",
            "binary_sha256": "0" * 64,
            "source_commit": "0" * 64,
            "kernel_signature_sha256": _kernel_signature_sha256_from_stats(stats),
        }
        static_l2 = _STATIC_CARD_L2_BYTES.get(device_arch, 0)
        device_block = {
            "device_name": stats.get("kernel mangled name", "unknown"),
            "compute_capability_major": (
                12 if device_arch == "sm_120-laptop" else 8
            ),
            "compute_capability_minor": (
                0 if device_arch == "sm_120-laptop" else 6
            ),
            "sm_count": 0,
            "total_global_mem_bytes": 0,
            "l2_cache_size_bytes": 0,
        }
        sector_result = consume_sector_stream_sources(
            sources=sources,
            output_root=None,
            write_jsonl=False,
        )
        if not sector_result.schema_admits_all:
            raise TraceAdmissionError(
                "sector_stream_schema_invalid",
                "sector-stream producer generated a schema-invalid record: "
                f"{sector_result.schema_first_failure}",
            )
        return emit_hrd_bundle_from_admitted_sources(
            sources=sources,
            sector_stream=sector_result,
            output_root=trace_dir,
            manifest_pairing_index=manifest_pairing_index,
            device=device_block,
            cuda_l2_cache_size_bytes=None,
            static_card_l2_cache_size_bytes=(
                static_l2 if static_l2 > 0 else None
            ),
            architecture=device_arch,
        )


# Static device-card L2 cache sizes per arch.
#   sm_86: RTX 3090 Ti has 6 MiB L2 (Ampere consumer card).
#   sm_120-laptop: RTX 5060 Laptop has 32 MiB L2 (Blackwell GB206
#     laptop; confirmed via the live `cuDeviceGetAttribute(L2_CACHE_SIZE)`
#     on driver 595.71.05 + CUDA 13.2).
_STATIC_CARD_L2_BYTES: dict[str, int] = {
    "sm_86": 6 * 1024 * 1024,
    "sm_120-laptop": 32 * 1024 * 1024,
}


__all__ = [
    "consume_trace_dir",
    "emit_hrd_bundle",
    "emit_hrd_bundle_from_admitted_sources",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/hrd_profiler/bundle_from_trace.py",
        description=(
            "Consume admitted trace snapshots plus in-memory sector records "
            "and emit the HRD profile bundle."
        ),
    )
    parser.add_argument(
        "--trace-dir", required=True, metavar="PATH",
    )
    parser.add_argument(
        "--device", required=True,
        choices=("sm_86", "sm_120-laptop"),
    )
    args = parser.parse_args(argv)
    trace_dir = Path(os.path.abspath(os.fspath(args.trace_dir)))
    status = consume_trace_dir(trace_dir, args.device)
    # Hide the bundle dict from CLI output (operator-facing
    # status only).
    cli_status = {k: v for k, v in status.items() if k != "bundle"}
    sys.stdout.write(json.dumps(cli_status, indent=2) + "\n")
    return 0 if status["schema_admits"] else 2


if __name__ == "__main__":
    sys.exit(main())
