"""Live HRD profile bundle producer (task11 / the HRD-profiler clause live-evidence path).

R19 refactor (per Codex's R18 review): the observation builder is
now proto-backed. ``emit_hrd_bundle`` walks the trace directory's
``threadblocks/`` proto records via
`import_trace_contract.validate_trace_dir` +
`_reconstruct_lane_vector`, emits one HRD observation per
active+predicated lane × per touched 32B sector with byte-address-
derived identities (sector / line / allocation / region), and
groups observations by kernel-launch directory. The R18 JSONL-
backed path is removed because it produced synthetic record-index
identities that did not represent real cache reuse.

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
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cache_model_spec import (  # type: ignore[import-not-found]  # noqa: E402
    resolve_cache_model_spec,
)
from observation_builder import (  # type: ignore[import-not-found]  # noqa: E402
    build_launches_from_trace_dir,
)
from profiler import compute_hrd_profile  # type: ignore[import-not-found]  # noqa: E402

_REPO_ROOT = _HERE.parent.parent
_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "hrd_profile.schema.json"
)


def _load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _read_stats(stats_path: Path) -> dict[str, Any]:
    with stats_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"stats.csv at {stats_path} is empty")
    header = [h.strip() for h in rows[0]]
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(f"stats.csv at {stats_path} has no data rows")
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


def _count_sector_stream_records(trace_dir: Path) -> int:
    """Count records in `<trace_dir>/sector_stream/sector_stream.jsonl`.

    The file is a JSONL (one record per line). Returns 0 when the
    file is missing — the caller decides how to handle absence.
    Counts non-empty lines only so trailing whitespace doesn't
    inflate the count. Reading the file is O(N) bytes; we don't
    materialize records since we only need the count for the
    R54 budget guard.
    """
    p = trace_dir / "sector_stream" / "sector_stream.jsonl"
    if not p.is_file():
        return 0
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


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
    """Production-path entry point. Walks the trace directory's
    threadblock proto records under the same wire contract used
    by `tools/sector_stream/consume_trace.py`, computes the bundle
    using launch-unit bootstrap over per-touched-sector
    observations with real byte-address-derived identities, writes
    it to `<trace_dir>/hrd_profile/hrd_profile.json`,
    strict-validates the on-disk bytes via Draft202012Validator,
    and returns a status dict for the run controller to persist
    into the run summary.

    On wire-contract failure, returns ``contract_ok=False`` and
    the typed reason; the bundle is not written. The controller
    surfaces a typed `hrd_skip_reason` rather than continuing on
    broken evidence.

    R54 defense-in-depth budget cap: when the materialized
    sector_stream.jsonl exceeds
    ``profiler.HRD_MAX_SECTOR_STREAM_RECORDS``, the producer
    skips the reuse-distance computation and emits a schema-
    valid typed-unavailable bundle with
    ``typed_reason == "hrd_compute_budget_exceeded"`` instead.
    The R53 Fenwick-tree complexity fix already keeps real
    canonical captures fast (warm_cold_pair_relaunch's ~33k
    sector_stream records complete in under a second), so the
    budget is set comfortably above observed inputs and only
    fires on adversarial / pathological streams.

    The caller (run_suite.py::handle_trace) supplies real values
    for manifest / device / cache-model resolver inputs; this
    function does NOT fabricate placeholders.
    """
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
        out_dir = trace_dir / "hrd_profile"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "hrd_profile.json"
        out_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        errors = list(jsonschema.Draft202012Validator(_load_schema()).iter_errors(bundle))
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

    # R54 budget guard runs BEFORE the expensive proto walk + reuse-distance
    # computation. If the sector stream exceeds the budget, emit a typed arm.
    from profiler import HRD_MAX_SECTOR_STREAM_RECORDS  # type: ignore[import-not-found]
    sector_stream_record_count = _count_sector_stream_records(trace_dir)
    if sector_stream_record_count > HRD_MAX_SECTOR_STREAM_RECORDS:
        bundle = _budget_exceeded_typed_bundle(
            manifest_pairing_index=manifest_pairing_index,
            device=device,
            cache_spec=cache_spec,
            sector_stream_record_count=sector_stream_record_count,
        )
        out_dir = trace_dir / "hrd_profile"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "hrd_profile.json"
        out_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        on_disk = json.loads(out_path.read_text(encoding="utf-8"))
        schema = _load_schema()
        validator = jsonschema.Draft202012Validator(schema)
        errors = list(validator.iter_errors(on_disk))
        schema_admits = len(errors) == 0
        first_failure = (
            f"{list(errors[0].absolute_path)}: {errors[0].message}"
            if errors else None
        )
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

    build_result = build_launches_from_trace_dir(
        trace_dir, capacity_bytes=cache_spec["capacity_bytes"],
        num_l2_slices=cache_spec["num_l2_slices"],
    )
    if not build_result.contract_ok:
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
    out_dir = trace_dir / "hrd_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hrd_profile.json"
    out_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
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


def consume_trace_dir(
    trace_dir: Path,
    device_arch: str,
) -> dict[str, Any]:
    """CLI convenience entry: read the trace directory, build a
    placeholder manifest/device block, call `emit_hrd_bundle`,
    return the status. Used by the CLI for ad-hoc operator
    inspection. Production callers (run_suite.py::handle_trace)
    use `emit_hrd_bundle` directly with real inputs."""
    stats_csv = trace_dir / "stats.csv"
    if not stats_csv.is_file():
        raise FileNotFoundError(
            f"Expected stats.csv at {stats_csv}"
        )
    stats = _read_stats(stats_csv)
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
    return emit_hrd_bundle(
        trace_dir=trace_dir,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/hrd_profiler/bundle_from_trace.py",
        description=(
            "Read a simbench NVBit trace directory's "
            "sector_stream.jsonl + stats.csv and emit the "
            "HRD profile bundle."
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
    trace_dir = Path(args.trace_dir).resolve()
    status = consume_trace_dir(trace_dir, args.device)
    # Hide the bundle dict from CLI output (operator-facing
    # status only).
    cli_status = {k: v for k, v in status.items() if k != "bundle"}
    sys.stdout.write(json.dumps(cli_status, indent=2) + "\n")
    return 0 if status["schema_admits"] else 2


if __name__ == "__main__":
    sys.exit(main())
