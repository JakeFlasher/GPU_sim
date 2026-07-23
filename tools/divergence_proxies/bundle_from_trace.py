"""Live divergence-proxies bundle producer (C-AC-G live-evidence path).

Reads a simbench NVBit trace directory's already-emitted
``sector_stream/sector_stream.jsonl`` plus ``stats.csv`` (the
trace-level instruction count) and writes a per-kernel-per-launch
proxy bundle to ``<trace_dir>/divergence_proxies/divergence_proxies.json``.

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
        [--device sm_86 | sm_120-laptop]

The CLI prints a small JSON status record (counts, schema-admit
result) and writes the bundle to disk.
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

from proxies import compute_divergence_proxies  # noqa: E402

_REPO_ROOT = _HERE.parent.parent
_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "divergence_proxies.schema.json"
)


def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _read_sector_stream_records(jsonl_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_stats(stats_path: Path) -> dict[str, Any]:
    """Parse ``stats.csv`` produced alongside the NVBit trace.

    The file has a single header row followed by one data row per
    kernel launch in the trace. For the R8 single-launch trace the
    second row is the only data row. Every column header is stripped
    of leading/trailing whitespace before lookup so the simbench
    writer's two-space-after-comma format does not require special
    handling at the call site.
    """
    with stats_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"stats.csv at {stats_path} is empty")
    header = [h.strip() for h in rows[0]]
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(
            f"stats.csv at {stats_path} has no data rows"
        )
    # Take the first data row; the R8 contract pins this to the
    # single-launch trace shape.
    cells = [c.strip() for c in data_rows[0]]
    return dict(zip(header, cells))


def _kernel_unique_id_from_stats(stats: dict[str, Any]) -> int:
    """Parse the ``kernel id`` cell into an integer.

    The simbench writer emits the cell as ``kernel-<id>.trace`` (e.g.
    ``kernel-1.trace``). Strip the prefix / suffix and parse the
    remaining digits. A malformed cell raises ``ValueError`` rather
    than silently substituting 0, so a mis-formatted trace does not
    hide behind an admitted bundle.
    """
    raw = stats.get("kernel id", "")
    if not raw:
        raise ValueError(f"stats.csv missing 'kernel id' cell")
    s = raw
    if s.startswith("kernel-"):
        s = s[len("kernel-"):]
    if s.endswith(".trace"):
        s = s[: -len(".trace")]
    return int(s)


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


def consume_trace_dir(
    trace_dir: Path,
    device: str,
    manifest_pairing_index: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Read the trace directory and return a status dict.

    The status dict carries the produced bundle, the number of
    sector-stream records consumed, the strict-schema admit result,
    and the path on disk.

    ``manifest_pairing_index`` carries the canonical 5-field
    bundle-provenance shape (``slug``, ``shape_point``,
    ``binary_sha256``, ``source_commit``,
    ``kernel_signature_sha256``). When omitted, a zero-fingerprint
    placeholder is used so the standalone CLI keeps working for
    ad-hoc inspection; the canonical ``run_suite.py`` path always
    passes a real value derived from live ``DeviceProps`` plus
    repo state.
    """
    sector_stream_jsonl = (
        trace_dir / "sector_stream" / "sector_stream.jsonl"
    )
    stats_csv = trace_dir / "stats.csv"

    if not sector_stream_jsonl.is_file():
        raise FileNotFoundError(
            f"Expected sector_stream.jsonl at {sector_stream_jsonl}; "
            f"run tools/sector_stream/consume_trace.py first."
        )
    if not stats_csv.is_file():
        raise FileNotFoundError(
            f"Expected stats.csv at {stats_csv}; the simbench writer "
            f"emits this file alongside the trace."
        )

    sector_stream_records = _read_sector_stream_records(
        sector_stream_jsonl
    )
    stats = _read_stats(stats_csv)
    kernel_unique_id = _kernel_unique_id_from_stats(stats)
    total_instruction_count = _total_instruction_count_from_stats(stats)

    effective_mpi: dict[str, str] = (
        dict(manifest_pairing_index)
        if manifest_pairing_index is not None
        else dict(_PLACEHOLDER_MANIFEST_PAIRING_INDEX)
    )
    bundle = compute_divergence_proxies(
        manifest_pairing_index=effective_mpi,
        kernel_unique_id=kernel_unique_id,
        device=device,
        sector_stream_records=sector_stream_records,
        total_instruction_count=total_instruction_count,
    )

    out_dir = trace_dir / "divergence_proxies"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "divergence_proxies.json"
    out_path.write_text(
        json.dumps(bundle, indent=2) + "\n", encoding="utf-8"
    )

    # R7 lesson: validate the bytes the producer wrote, not the
    # in-process bundle. Re-read the on-disk file before strict
    # validation so a discrepancy between in-memory shape and
    # serialized shape cannot recur silently.
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
        "kernel_unique_id": kernel_unique_id,
        "device": device,
        "sector_stream_records_consumed": len(sector_stream_records),
        "total_instruction_count": total_instruction_count,
        "schema_admits": schema_admits,
        "schema_first_failure": first_failure,
        "proxy_summary": _summarize_proxies(bundle),
    }


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/divergence_proxies/bundle_from_trace.py",
        description=(
            "Read a simbench NVBit trace directory's "
            "sector_stream.jsonl and stats.csv and emit the "
            "divergence-proxies bundle."
        ),
    )
    parser.add_argument(
        "--trace-dir", required=True, metavar="PATH",
        help=(
            "Path to the trace directory containing "
            "sector_stream/sector_stream.jsonl and stats.csv."
        ),
    )
    parser.add_argument(
        "--device", default="sm_120-laptop",
        choices=("sm_86", "sm_120-laptop"),
        help=(
            "Compute capability tag for the device that produced "
            "the trace. Default sm_120-laptop (the R8 live-evidence "
            "device)."
        ),
    )
    args = parser.parse_args(argv)

    trace_dir = Path(args.trace_dir).resolve()
    status = consume_trace_dir(trace_dir, args.device)
    sys.stdout.write(json.dumps(status, indent=2) + "\n")
    return 0 if status["schema_admits"] else 2


if __name__ == "__main__":
    sys.exit(main())
