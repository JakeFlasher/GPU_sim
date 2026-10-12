"""Live divergence-proxies bundle producer (C-AC-G live-evidence path).

Consumes in-process sector-stream records plus the admitted ``stats.csv``
snapshot and writes a per-kernel-per-launch proxy bundle to
``<output_root>/divergence_proxies/divergence_proxies.json``. The generated
sector JSONL path is never treated as an input authority.

The bundle conforms to
``workloads/simbench_bw/schemas/divergence_proxies.schema.json`` under
strict ``jsonschema.Draft202012Validator`` validation. Proxies whose
required evidence is not present in the trace emit a typed-unavailable
record instead of a silent zero default — the C-AC-G negative-test
contract that distinguishes "0.0 measured" from "no data".

For a single-launch trace such as
``workloads/simbench_bw/l1/dram_stream_stride_mix/traces/`` the
expected bundle shape is:

* ``unique_sectors_per_warp_inst`` — admitted (sector-stream mean)
* ``unique_lines_per_warp_inst`` — admitted (sector-stream mean)
* ``sectors_per_request`` — admitted (sector-stream mean)
* ``md_loads_per_kilo_inst`` — admitted (uses ``stats.csv:total_insts``)
* ``outstanding_load_density_by_phase`` — typed-unavailable
  (``%clock64`` phase records not extracted onto disk)
* ``pc_repeated_miss_likely_streams`` — typed-unavailable
  (PC provenance is in-process only, not on the on-disk
  ``sector_stream.jsonl`` per the R7 ``_origin``-removal lesson)
* ``stride_entropy`` — typed-unavailable (lane vectors are
  reconstructed in-memory by the sector-stream consumer; the on-disk
  artifact does not persist the per-warp 32-element vector)
* ``throughput_saturation_slope_under_resident_warp_sweep`` —
  typed-unavailable (single-launch trace; no resident-warp sweep)

CLI:

    python tools/divergence_proxies/bundle_from_trace.py \\
        --trace-dir <PATH> \\
        [--device sm_86 | sm_120 | sm_120]

The CLI prints a small JSON status record (counts, schema-admit
result) and writes the bundle to disk.
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

from tools.divergence_proxies.proxies import (  # noqa: E402
    compute_divergence_proxies,
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
    / "divergence_proxies.schema.json"
)


def _load_schema() -> dict:
    schema = json.loads(read_regular(_SCHEMA_PATH))
    if not isinstance(schema, dict):
        raise TypeError("divergence-proxies schema must be a JSON object")
    return schema


def _read_stats(raw: bytes, *, label: str = "stats.csv") -> dict[str, Any]:
    """Parse the sole stats row required by the single-launch schema."""
    try:
        text = raw.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise TraceAdmissionError(
            "stats_schema_invalid", f"{label} is not valid UTF-8 CSV: {exc}"
        ) from exc
    if not rows:
        raise TraceAdmissionError("stats_empty", f"{label} is empty")
    header = [cell.strip() for cell in rows[0]]
    data_rows = [
        [cell.strip() for cell in row]
        for row in rows[1:]
        if any(cell.strip() for cell in row)
    ]
    if len(data_rows) != 1:
        raise TraceAdmissionError(
            "stats_schema_invalid",
            f"{label} must contain exactly one data row; observed "
            f"row_count={len(data_rows)}",
        )
    cells = data_rows[0]
    if len(cells) != len(header):
        raise TraceAdmissionError(
            "stats_schema_invalid",
            f"{label} row has {len(cells)} cells for {len(header)} headers",
        )
    stats = dict(zip(header, cells))
    for required in ("kernel id", "kernel mangled name", "total_insts"):
        if not stats.get(required):
            raise TraceAdmissionError(
                "stats_schema_invalid", f"{label} missing {required!r} cell"
            )
    return stats


def _kernel_unique_id_from_stats(stats: dict[str, Any]) -> int:
    """Parse the strict ``kernel-<id>.trace`` stats identity."""
    raw = stats.get("kernel id", "")
    prefix = "kernel-"
    suffix = ".trace"
    if not raw.startswith(prefix) or not raw.endswith(suffix):
        raise TraceAdmissionError(
            "stats_schema_invalid",
            f"stats.csv 'kernel id' is not kernel-<id>.trace: {raw!r}",
        )
    token = raw[len(prefix): -len(suffix)]
    if not token.isascii() or not token.isdigit():
        raise TraceAdmissionError(
            "stats_schema_invalid",
            f"stats.csv 'kernel id' has a non-decimal id: {raw!r}",
        )
    return int(token, 10)


def _total_instruction_count_from_stats(
    stats: dict[str, Any],
) -> int | None:
    """Parse ``total_insts`` from stats.csv. Missing or non-numeric
    cell returns ``None`` so the proxy module emits the documented
    ``md_loads_per_kilo_inst_unavailable_no_instruction_count``
    typed-unavailable record."""
    cell = stats.get("total_insts", "")
    if not cell:
        return None
    try:
        return int(cell)
    except ValueError:
        return None


_PLACEHOLDER_MANIFEST_PAIRING_INDEX: dict[str, str] = {
    "slug": "<unspecified>",
    "shape_point": "<unspecified>",
    "binary_sha256": "0" * 64,
    "source_commit": "0" * 64,
    "kernel_signature_sha256": "0" * 64,
}


def consume_admitted_trace_sources(
    *,
    sources: AdmittedTraceSources,
    sector_stream: AdmittedSectorStream,
    output_root: Path,
    device: str,
    manifest_pairing_index: dict[str, str],
) -> dict[str, Any]:
    """Build one divergence bundle from admission-bound in-memory sectors."""
    if not isinstance(sources, AdmittedTraceSources):
        raise TypeError("sources must be an AdmittedTraceSources capability")
    if not isinstance(sector_stream, AdmittedSectorStream):
        raise TypeError("sector_stream must be an AdmittedSectorStream")
    if not isinstance(manifest_pairing_index, dict):
        raise TypeError("manifest_pairing_index must be a dictionary")
    sources.verify_unchanged()
    admission = sources.admission
    if admission.launch_count != 1:
        raise TraceAdmissionError(
            "trace_launch_count_unsupported",
            "divergence-proxies schema requires exactly one admitted launch; "
            f"observed launch_count={admission.launch_count}",
        )
    if admission.stats_row_count != 1:
        raise TraceAdmissionError(
            "stats_schema_invalid",
            "divergence-proxies requires exactly one admitted stats row; "
            f"observed stats_row_count={admission.stats_row_count}",
        )
    sector_stream.verify_for(sources)
    if len(sources.threadblock_paths) != admission.threadblock_count:
        raise TraceAdmissionError(
            "trace_snapshot_changed",
            "admitted threadblock path count disagrees with admission",
        )
    records = sector_stream.mutable_records()
    stats = _read_stats(sources.read_stats_snapshot().payload)
    kernel_unique_id = _kernel_unique_id_from_stats(stats)
    stats_kernel_name = str(stats["kernel mangled name"])
    launch = admission.launches[0]
    if (
        kernel_unique_id != launch.kernel_id
        or stats_kernel_name != launch.kernel_name
    ):
        raise TraceAdmissionError(
            "trace_launch_identity_mismatch",
            "stats.csv identity does not match the sole admitted launch: "
            f"stats_kernel_id={kernel_unique_id} "
            f"admitted_kernel_id={launch.kernel_id} "
            f"stats_kernel_name={stats_kernel_name!r} "
            f"admitted_kernel_name={launch.kernel_name!r}",
        )
    total_instruction_count = _total_instruction_count_from_stats(stats)

    bundle = compute_divergence_proxies(
        manifest_pairing_index=dict(manifest_pairing_index),
        kernel_unique_id=kernel_unique_id,
        device=device,
        sector_stream_records=records,
        total_instruction_count=total_instruction_count,
    )

    out_dir = Path(output_root) / "divergence_proxies"
    ensure_directory(out_dir)
    out_path = out_dir / "divergence_proxies.json"
    payload = (json.dumps(bundle, indent=2) + "\n").encode("utf-8")
    replace_regular_bytes(out_path, payload)

    # Validate the exact serialized bytes read back from the output destination.
    on_disk = json.loads(read_regular(out_path))
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
        "kernel_unique_id": kernel_unique_id,
        "device": device,
        "sector_stream_records_consumed": len(records),
        "total_instruction_count": total_instruction_count,
        "schema_admits": schema_admits,
        "schema_first_failure": first_failure,
        "proxy_summary": _summarize_proxies(bundle),
    }


def consume_trace_dir(
    trace_dir: Path,
    device: str,
    manifest_pairing_index: dict[str, str] | None = None,
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
        effective_mpi = (
            dict(manifest_pairing_index)
            if manifest_pairing_index is not None
            else dict(_PLACEHOLDER_MANIFEST_PAIRING_INDEX)
        )
        return consume_admitted_trace_sources(
            sources=sources,
            sector_stream=sector_result,
            output_root=trace_dir,
            device=device,
            manifest_pairing_index=effective_mpi,
        )


def _summarize_proxies(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compact per-proxy summary for operator-facing CLI output:
    each proxy reports either its numeric value or its typed_reason."""
    summary: dict[str, Any] = {}
    for key, value in bundle.items():
        if not isinstance(value, dict):
            continue
        if value.get("result_kind") == "proxy_value":
            summary[key] = {
                "value": value["value"],
                "sample_count": value.get("sample_count"),
            }
        elif value.get("result_kind") == "typed_unavailable":
            summary[key] = {"typed_reason": value["typed_reason"]}
    return summary


__all__ = ["consume_admitted_trace_sources", "consume_trace_dir"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/divergence_proxies/bundle_from_trace.py",
        description=(
            "Consume admitted trace snapshots and in-memory sector records "
            "and emit the divergence-proxies bundle."
        ),
    )
    parser.add_argument(
        "--trace-dir", required=True, metavar="PATH",
        help=(
            "Path to the admitted trace directory. Sector records are "
            "regenerated from its capability-backed threadblock snapshots."
        ),
    )
    parser.add_argument(
        "--device", default="sm_120",
        choices=("sm_86", "sm_120", "sm_120"),
        help=(
            "Compute capability tag for the device that produced "
            "the trace. Default sm_120 (the R8 live-evidence "
            "device)."
        ),
    )
    args = parser.parse_args(argv)

    trace_dir = Path(os.path.abspath(os.fspath(args.trace_dir)))
    status = consume_trace_dir(trace_dir, args.device)
    sys.stdout.write(json.dumps(status, indent=2) + "\n")
    return 0 if status["schema_admits"] else 2


if __name__ == "__main__":
    sys.exit(main())
