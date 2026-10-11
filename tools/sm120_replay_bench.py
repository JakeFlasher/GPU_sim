#!/usr/bin/env python3
"""Run the SM120 alpha replay path and join it with local capture evidence.

This is intentionally a small, dependency-free driver for the cloud capture
mirror.  It does not recapture a workload, invoke Nsight Compute, or infer
proprietary scheduling parameters.  Each record joins four independently
stored surfaces:

* ``captures/<name>`` -- NVBit trace, ``stats.csv``, and the workload's
  ``program.stdout`` (the latter contains the tracer-run ``Avg runtime``);
* ``profiles/<name>/raw.csv`` -- NCU kernel rows, including
  ``gpu__time_duration.sum`` and cycle counters;
* ``captures/<name>/extra_info/native_static_metadata.json`` -- the SM120
  static/control metadata emitted by the tracer;
* ``sim-run`` output -- the simulator's model frontier and counters.

The NCU run is an instrumented observation and the NVBit program stdout is a
tracer-run host timing.  They are retained as separate fields instead of being
silently treated as the same ground truth.  Likewise simulator cycles are not
wall-clock time: when an NCU effective SM frequency is available the script
emits a clearly named *derived* ``sim_time_ms_at_ncu_sm_clock`` field.  The
unambiguous process-wall-clock replay time is ``sim_wall_ms``.

The current alpha tree does not yet admit the opaque SM120 static metadata as
the legacy ``ControlSidecar``.  Therefore the default driver is useful in
report mode immediately and records a typed ``control_status``.  With
``--synthesize-control`` it materializes a temporary legacy-shaped sidecar
from each native static table so a future SM120 parser can consume it; the
sidecar is marked derived and is never presented as a measured Blackwell
control-layout claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_WORK_ROOT = Path("/dev/shm/sm120_rtx_5090_work")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM = REPO_ROOT / "sim/target/release/sim-run"
DEFAULT_CARD = REPO_ROOT / "arch_profiles/cards/sm_120-rtx-5090.json"

RUNTIME_RE = re.compile(
    r"Avg\s+runtime\s*:?[ \t]*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)\s*ms",
    re.IGNORECASE,
)
GFLOPS_RE = re.compile(
    r"(?:GFLOPS|TFLOPS)\s*:?\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)",
    re.IGNORECASE,
)


def finite_float(value: Any) -> float | None:
    """Parse one finite numeric CSV/JSON value; blank/non-numeric => None."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def duration_ms_from_ncu(row: dict[str, Any]) -> float | None:
    """Convert the raw NCU duration sum to milliseconds using its unit row."""

    value = finite_float(row.get("gpu_time_duration_sum"))
    if value is None:
        return None
    unit = str(row.get("gpu_time_duration_sum_unit", "us")).strip().lower()
    if unit in {"us", "µs", "microsecond", "microseconds"}:
        return value / 1_000.0
    if unit in {"ns", "nanosecond", "nanoseconds"}:
        return value / 1_000_000.0
    if unit in {"ms", "millisecond", "milliseconds"}:
        return value
    if unit in {"s", "sec", "second", "seconds"}:
        return value * 1_000.0
    # Keep unknown NCU units visible, but do not silently claim a converted
    # duration.  Existing SM120 exports use `us`.
    return None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL objects one at a time.

    Sidecars can contain hundreds of thousands of synchronization records.  A
    list materialization made the old driver both memory-hungry and unable to
    preserve launch attribution cheaply.  Keep parsing streaming and aggregate
    only the small typed summaries needed by the report.
    """

    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # Keep a typed parse marker in the ordinary application
                # summary.  The alpha lane has no separate validation gate,
                # but it must not silently turn malformed observations into
                # zeros or drop them from the accounting surface.
                yield {"_sidecar_parse_error": line_number}
                continue
            if isinstance(value, dict):
                yield value


def _sidecar_bucket() -> dict[str, Any]:
    return {
        "tma_records": 0,
        "sync_records": 0,
        "tma_byte_count_sum": 0,
        "tma_opcode_classes": set(),
        "mbarrier_addresses": set(),
        "descriptor_byte_observation_records": 0,
        "zero_byte_records": 0,
        "predicated_off_records": 0,
        "parse_errors": 0,
    }


def _sidecar_kernel_key(value: Any) -> str:
    """Normalize sidecar kernel IDs while retaining an explicit unknown key."""

    if value is None or str(value).strip() == "":
        return "unknown"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _add_tma_sidecar_row(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    if row.get("_sidecar_parse_error") is not None:
        bucket["parse_errors"] += 1
        return
    bucket["tma_records"] += 1
    try:
        count = int(row.get("byte_count", 0) or 0)
    except (TypeError, ValueError):
        count = 0
        bucket["parse_errors"] += 1
    bucket["tma_byte_count_sum"] += count
    opcode = row.get("opcode")
    if opcode:
        bucket["tma_opcode_classes"].add(str(opcode))
    if count > 0:
        # Descriptor byte counts describe sidecar metadata only.  They are not
        # a consumer payload or destination-visibility witness.
        bucket["descriptor_byte_observation_records"] += 1
    else:
        bucket["zero_byte_records"] += 1
    if int(row.get("active_mask", 0) or 0) == 0 or int(row.get("predicate_mask", 0) or 0) == 0:
        bucket["predicated_off_records"] += 1


def _add_sync_sidecar_row(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    if row.get("_sidecar_parse_error") is not None:
        bucket["parse_errors"] += 1
        return
    bucket["sync_records"] += 1
    if row.get("barrier_address_present") and row.get("barrier_address_u32") is not None:
        try:
            bucket["mbarrier_addresses"].add(int(row["barrier_address_u32"]))
        except (TypeError, ValueError):
            pass


def _finish_sidecar_bucket(
    bucket: dict[str, Any],
    *,
    capture: Path,
    kernel_id: int | str | None,
) -> dict[str, Any]:
    tma_records = int(bucket["tma_records"])
    sync_records = int(bucket["sync_records"])
    return {
        "status": "descriptor_and_sync_observation_only"
        if (tma_records or sync_records)
        else "unavailable",
        "kernel_id_scope": _sidecar_kernel_key(kernel_id) if kernel_id is not None else "all",
        "tma_records": tma_records,
        "sync_records": sync_records,
        "tma_byte_count_sum": int(bucket["tma_byte_count_sum"]),
        "tma_opcode_classes": sorted(bucket["tma_opcode_classes"]),
        "mbarrier_addresses": sorted(bucket["mbarrier_addresses"]),
        "descriptor_bytes_observed": int(bucket["descriptor_byte_observation_records"]),
        "descriptor_bytes_scope": "sidecar_descriptor_metadata_only",
        "zero_byte_records": int(bucket["zero_byte_records"]),
        "predicated_off_records": int(bucket["predicated_off_records"]),
        "parse_errors": int(bucket["parse_errors"]),
        "destination_visibility": "unobserved_without_consumer_payload_litmus",
        "completion_timestamp": "unobserved",
        "raw_sidecars": {
            "tma_trace": str(capture / "extra_info/tma_trace.jsonl"),
            "sync_trace": str(capture / "extra_info/sync_trace.jsonl"),
        },
    }


def async_sidecar_observation(
    capture: Path, kernel_id: int | str | None = None
) -> dict[str, Any]:
    """Stream and summarize TMA/sync sidecars with launch attribution.

    The tracer sidecars carry ``kernel_id`` even when they do not carry the
    simulator's launch index.  The old implementation reduced the whole
    capture into one total; callers then had no safe way to tell which TMA or
    barrier records belonged to each launch.  We now retain a compact
    ``by_kernel_id`` map and optionally return one filtered bucket.  Raw JSONL
    remains the authoritative artifact and is never copied into every launch.
    """

    tma_path = capture / "extra_info/tma_trace.jsonl"
    sync_path = capture / "extra_info/sync_trace.jsonl"
    aggregate = _sidecar_bucket()
    per_kernel: dict[str, dict[str, Any]] = {}
    wanted = _sidecar_kernel_key(kernel_id) if kernel_id is not None else None

    for row in iter_jsonl(tma_path):
        key = _sidecar_kernel_key(row.get("kernel_id"))
        _add_tma_sidecar_row(aggregate, row)
        bucket = per_kernel.setdefault(key, _sidecar_bucket())
        _add_tma_sidecar_row(bucket, row)
    for row in iter_jsonl(sync_path):
        key = _sidecar_kernel_key(row.get("kernel_id"))
        _add_sync_sidecar_row(aggregate, row)
        bucket = per_kernel.setdefault(key, _sidecar_bucket())
        _add_sync_sidecar_row(bucket, row)

    selected_bucket = aggregate if wanted is None else per_kernel.get(wanted, _sidecar_bucket())
    result = _finish_sidecar_bucket(selected_bucket, capture=capture, kernel_id=kernel_id)
    if wanted is None:
        result["by_kernel_id"] = {
            key: _finish_sidecar_bucket(bucket, capture=capture, kernel_id=key)
            for key, bucket in sorted(per_kernel.items())
        }
        result["kernel_ids_observed"] = sorted(per_kernel)
    return result


def artifact_descriptor(path: Path) -> dict[str, Any]:
    """Return non-cryptographic provenance for one retained artifact.

    The alpha campaign deliberately does not hash or validate files.  The
    descriptor only records path, presence, and byte size so a downstream
    report can distinguish an unavailable observation from an empty one.
    """

    try:
        present = path.is_file()
        return {
            "path": str(path),
            "present": present,
            "size_bytes": path.stat().st_size if present else None,
        }
    except OSError as exc:
        return {"path": str(path), "present": False, "size_bytes": None, "error": str(exc)}


def evidence_contract(
    capture: Path,
    profile: Path,
    stats: list[dict[str, Any]],
    ncu: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    hardware: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe the four non-substitutable alpha evidence lanes."""

    return {
        "schema_version": 2,
        "capture_name": capture.name,
        "evidence_root": str(capture.parents[1]) if len(capture.parents) > 1 else str(capture.parent),
        "lanes": {
            "structure": {
                "status": "observed" if stats or metadata else "unavailable",
                "source": "NVBit dynamic stats plus native static metadata",
                "timing_claim": False,
            },
            "counters": {
                "status": "observed" if ncu else "unavailable",
                "source": "NCU raw CSV; units/domain/pass retained in row payload",
                "timing_claim": False,
            },
            "baremetal_timing": {
                "status": "observed_uninstrumented" if hardware else "unavailable",
                "source": "external CUDA-event/host timing contract",
                "timing_claim": bool(hardware),
            },
            "simulator": {
                "status": "model_relative",
                "source": "sim-run replay/process timers",
                "timing_claim": False,
            },
        },
        "scope_contract": {
            "strict_join_key": "device,stream,launch,kernel,function,pc,dynamic_index",
            "timing_denominator": "uninstrumented_cuda_event_only",
            "ncu_duration_role": "profiler_perturbed_observation",
            "ordered_fallback_role": "diagnostic_only",
            "cache_geometry_fit": False,
            "physical_topology_fit": False,
        },
        "launch_population": {
            "trace_stats_rows": len(stats),
            "ncu_rows": len(ncu),
            "static_kernel_count": len(metadata.get("kernels", [])) if metadata else 0,
            "hardware_timing_declared": bool(hardware),
        },
        "artifacts": {
            "dynamic_trace": artifact_descriptor(capture / "dynamic_trace.pb"),
            "stats_csv": artifact_descriptor(capture / "stats.csv"),
            "program_stdout": artifact_descriptor(capture / "program.stdout"),
            "native_static_metadata": artifact_descriptor(
                capture / "extra_info/native_static_metadata.json"
            ),
            "tma_trace_sidecar": artifact_descriptor(capture / "extra_info/tma_trace.jsonl"),
            "sync_trace_sidecar": artifact_descriptor(capture / "extra_info/sync_trace.jsonl"),
            "ncu_raw_csv": artifact_descriptor(profile / "raw.csv"),
            "ncu_effective_frequency": artifact_descriptor(profile / "effective_frequency_summary.csv"),
        },
    }


def parse_stats(path: Path) -> list[dict[str, Any]]:
    """Read NVBit ``stats.csv`` and keep one dictionary per kernel row."""

    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return []
    header = rows[0]
    output: list[dict[str, Any]] = []
    for row in rows[1:]:
        if not row or row[0].strip().lower() in {"device_id", ""}:
            continue
        values = row + [""] * max(0, len(header) - len(row))
        data = dict(zip(header, values))
        try:
            kernel_index = int(str(data.get("kernel id", "")).split("-", 1)[-1].split(".", 1)[0])
        except ValueError:
            kernel_index = len(output) + 1
        numeric: dict[str, Any] = {}
        for key in ("device_id", "stream_id", "#blocks", "#threads", "total_insts", "total_reported_insts"):
            raw = data.get(key, "").strip()
            try:
                numeric[key] = int(raw)
            except ValueError:
                numeric[key] = raw
        numeric["trace_kernel_index"] = kernel_index
        numeric["kernel_name"] = data.get("kernel mangled name", "")
        numeric["grid"] = [
            int(data.get("grid_dimX", "0") or 0),
            int(data.get("grid_dimY", "0") or 0),
            int(data.get("grid_dimZ", "0") or 0),
        ]
        numeric["block"] = [
            int(data.get("block_dimX", "0") or 0),
            int(data.get("block_dimY", "0") or 0),
            int(data.get("block_dimZ", "0") or 0),
        ]
        output.append(numeric)
    return output


def parse_ncu_raw(path: Path) -> list[dict[str, Any]]:
    """Extract per-kernel rows from an NCU raw CSV export.

    Nsight Compute writes a header row, a units row, then one row per launch.
    The parser deliberately retains only stable identity/duration/cycle
    columns; all other counters stay in the original NCU export.
    """

    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    header_start = next(
        (i for i, row in enumerate(rows) if row and row[0] == "ID" and "Kernel Name" in row),
        None,
    )
    if header_start is None:
        return []
    rows = rows[header_start:]
    if len(rows) < 2:
        return []
    header = rows[0]
    if not header:
        return []
    units = rows[1] if len(rows) > 1 else []
    data_start = 2
    # NCU exports normally include a units row whose ID cell is blank.  Keep
    # compatibility with compact exports that omit that row.
    if len(rows) > 1 and rows[1] and rows[1][0].strip().isdigit():
        units = []
        data_start = 1
    index = {name: i for i, name in enumerate(header)}

    def cell(row: list[str], name: str) -> str:
        position = index.get(name)
        return row[position] if position is not None and position < len(row) else ""

    output: list[dict[str, Any]] = []
    # Keep a compact counter surface for the alpha comparison table.  The raw
    # CSV remains the authoritative artifact; this subset only carries metrics
    # that can be joined to simulator native_metrics without unit conversion.
    metric_prefixes = (
        "gpu__time_duration",
        "gpc__cycles_",
        "sm__cycles_",
        "smsp__inst_",
        "smsp__warps_",
        "l1tex__t_sectors",
        "lts__t_sectors_",
        "dram__sectors_",
    )
    for row in rows[data_start:]:
        raw_id = cell(row, "ID").strip()
        if not raw_id.isdigit():
            continue
        record: dict[str, Any] = {"ncu_launch_id": int(raw_id)}
        for name, key in (
            ("Kernel Name", "kernel_name"),
            ("Process Name", "process_name"),
            ("Block Size", "block_size"),
            ("Grid Size", "grid_size"),
            ("Stream", "stream"),
            ("Context", "context"),
            ("CC", "compute_capability"),
        ):
            record[key] = cell(row, name).strip()
        for metric, key in (
            ("gpu__time_duration.sum", "gpu_time_duration_sum"),
            ("gpu__time_duration.avg", "gpu_time_duration_avg"),
            ("gpc__cycles_elapsed.sum", "gpc_cycles_elapsed_sum"),
            ("gpc__cycles_elapsed.avg", "gpc_cycles_elapsed_avg"),
            ("sm__cycles_elapsed.sum", "sm_cycles_elapsed_sum"),
            ("sm__cycles_elapsed.avg", "sm_cycles_elapsed_avg"),
            ("dram__cycles_elapsed.sum", "dram_cycles_elapsed_sum"),
        ):
            value = finite_float(cell(row, metric))
            if value is not None:
                record[key] = value
            unit_index = index.get(metric)
            if unit_index is not None and unit_index < len(units):
                record[f"{key}_unit"] = units[unit_index].strip()
        compact_metrics: dict[str, dict[str, Any]] = {}
        for metric, position in index.items():
            if not metric.startswith(metric_prefixes):
                continue
            numeric = finite_float(row[position] if position < len(row) else "")
            if numeric is None:
                continue
            unit = units[position].strip() if position < len(units) else ""
            # Keep only explicit scalar counter units; rates/percentages have
            # different domains and are not compared by this first alpha lane.
            if unit not in {"inst", "cycle", "warp", "sector", "request", "thread", "us", "ns"}:
                continue
            compact_metrics[metric] = {"value": numeric, "unit": unit}
        record["metrics"] = compact_metrics
        output.append(record)
    return output


def parse_effective_sm_frequency(path: Path) -> float | None:
    """Return NCU effective SM frequency in GHz, if exported."""

    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        for row in rows:
            metric = str(row.get("metric", "")).strip()
            if metric != "sm_cycles_elapsed_avg_per_second":
                continue
            return finite_float(row.get("median")) or finite_float(row.get("mean"))
    return None


def parse_program_observation(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    runtimes = [float(match.group(1)) for match in RUNTIME_RE.finditer(text)]
    gflops = [float(match.group(1)) for match in GFLOPS_RE.finditer(text)]
    return {
        "runtime_ms_values": runtimes,
        "gflops_or_tflops_values": gflops,
        "runtime_ms_first": runtimes[0] if runtimes else None,
        "runtime_ms_last": runtimes[-1] if runtimes else None,
    }


def parse_hardware_timing(path: Path | None, capture_name: str) -> dict[str, Any] | None:
    """Read an optional uninstrumented CUDA-event/host timing record.

    The alpha timing input is intentionally a tiny JSON contract and is kept
    outside the NVBit/NCU capture.  Accepted forms are either a single object
    (``{"hardware_elapsed_ns": ..., "coverage": "full"}``) or a mapping with
    a ``captures`` object keyed by capture directory name.  A directory may
    contain ``<capture>.json`` or ``<capture>.timing.json``.
    """

    if path is None:
        return None
    candidate = path
    if path.is_dir():
        options = [path / f"{capture_name}.json", path / f"{capture_name}.timing.json"]
        candidate = next((entry for entry in options if entry.is_file()), Path())
    if not candidate.is_file():
        return None
    try:
        value = read_json(candidate)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(value, dict) and isinstance(value.get("captures"), dict):
        value = value["captures"].get(capture_name)
    if not isinstance(value, dict):
        return None
    elapsed = value.get("hardware_elapsed_ns", value.get("elapsed_ns"))
    elapsed_value = finite_float(elapsed)
    if elapsed_value is None or elapsed_value <= 0:
        return None
    coverage = str(value.get("coverage", "unknown")).strip().lower()
    declared_full = value.get("coverage_full")
    coverage_full = declared_full is True or (declared_full is None and coverage == "full")
    return {
        "path": str(candidate),
        "hardware_elapsed_ns": elapsed_value,
        "hardware_timing_source": str(value.get("source", "unspecified")),
        "coverage": coverage,
        "coverage_full": coverage_full,
        "scope": str(value.get("scope", "unspecified")),
        "notes": value.get("notes"),
    }


def load_static_metadata(capture: Path) -> dict[str, Any] | None:
    path = capture / "extra_info/native_static_metadata.json"
    if not path.is_file():
        return None
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def existing_control_candidates(
    capture: Path, controls_root: Path | None, relative_name: str | None = None
) -> list[Path]:
    name = capture.name
    candidates = [
        capture / "extra_info" / f"{name}.control_bits.json",
        capture / "extra_info" / "control_bits.json",
        capture / "control" / f"{name}.control_bits.json",
    ]
    if controls_root is not None:
        key = relative_name or name
        candidates.extend(
            [
                controls_root / f"{key}.control_bits.json",
                controls_root / key / f"{name}.control_bits.json",
                controls_root / f"{key}.json",
            ]
        )
    return [path for path in candidates if path.is_file()]


def _wire_control(high_word: int) -> dict[str, int]:
    wire = (high_word >> 41) & ((1 << 23) - 1)
    return {
        "stall": wire & 0xF,
        "yield_flag": (wire >> 4) & 0x1,
        "write_barrier": (wire >> 5) & 0x7,
        "read_barrier": (wire >> 8) & 0x7,
        "wait_mask": (wire >> 11) & 0x3F,
        "reuse": (wire >> 17) & 0xF,
    }


def synthesize_legacy_sidecar(
    capture: Path, metadata: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Materialize a future-alpha sidecar from native SM120 metadata.

    This conversion preserves the raw words and uses the currently documented
    23-bit extraction only as a transport shape.  It is *not* evidence that
    Blackwell shares the SM86 control layout.  The generated document carries
    an explicit derived marker and native metadata provenance.
    """

    command_text = ""
    command_path = capture / "command.txt"
    if command_path.is_file():
        command_text = command_path.read_text(encoding="utf-8", errors="replace").strip()
    binary = shlex.split(command_text)[0] if command_text else metadata.get("benchmark_name", "sm120_capture")
    functions: dict[str, list[dict[str, Any]]] = {}
    summary_functions: dict[str, dict[str, int]] = {}
    for kernel in metadata.get("kernels", []):
        raw_name = str(kernel.get("kernel_name", ""))
        name = re.sub(r"___[0-9]+$", "", raw_name)
        records: list[dict[str, Any]] = []
        for inst in kernel.get("instructions", []):
            pc = int(inst.get("pc_num_dec", 0))
            high_text = str(inst.get("control_word_hex") or inst.get("word1_hex") or "0")
            low_text = str(inst.get("word0_hex") or "0")
            high_hex = high_text if high_text.lower().startswith("0x") else f"0x{high_text}"
            low_hex = low_text if low_text.lower().startswith("0x") else f"0x{low_text}"
            try:
                high_word = int(high_hex, 16)
            except ValueError:
                high_word = 0
            sass = str(inst.get("sass") or inst.get("opcode") or "UNKNOWN").strip().rstrip(";").strip()
            predicate = None
            if sass.startswith("@"):
                predicate, _, sass = sass.partition(" ")
                sass = sass.strip()
            records.append(
                {
                    "pc": pc,
                    "predicate": predicate,
                    "opcode_text": sass,
                    "low_word": low_hex,
                    "high_word": high_hex,
                    "status": "decoded",
                    "control": _wire_control(high_word),
                }
            )
        functions[name] = records
        summary_functions[name] = {
            "instructions": len(records),
            "decoded": len(records),
            "unknown_encoding": 0,
            "stall_cycles_static_sum": sum(int(x.get("control", {}).get("stall", 0)) for x in records),
        }
    document: dict[str, Any] = {
        "arch": "sm_120",
        "binary": binary,
        "functions": functions,
        "parser_contract": {
            "supported_toolkit_versions": ["NVBit-native-sm120-static-v3"],
            "supported_arches": ["sm_120"],
            "control_layout": {
                "source": "high_word >> 41 (23 bits)",
                "stall": "bits 0:3",
                "yield_flag": "bit 4",
                "write_barrier": "bits 5:7 (7 = none)",
                "read_barrier": "bits 8:10 (7 = none)",
                "wait_mask": "bits 11:16",
                "reuse": "bits 17:20",
                "reserved_must_be_zero": "bits 21:22",
            },
        },
        "toolkit_version": "NVBit-native-sm120-static-v3",
        "binary_source": "derived_from_extra_info/native_static_metadata.json",
        "summary": {
            "functions": summary_functions,
            "total_decoded": sum(x["decoded"] for x in summary_functions.values()),
            "total_unknown_encoding": 0,
        },
        "sm120_alpha_provenance": {
            "status": "derived_transport_only",
            "source": str(capture / "extra_info/native_static_metadata.json"),
            "control_latency_claims": bool(metadata.get("control_latency_claims", False)),
            "note": "Do not interpret transport extraction as a validated Blackwell control-layout model.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def discover_captures(work_root: Path, selected: Iterable[str] | None) -> list[Path]:
    captures = work_root / "captures"
    if not captures.is_dir():
        return []
    wanted = set(selected or ())
    output = []
    # Stop descending once a trace root is found. Grouping directories such
    # as flashinfer_attention/ and sol_l1/ are not themselves trace roots.
    for directory, subdirs, files in os.walk(captures):
        if "dynamic_trace.pb" not in files:
            continue
        subdirs[:] = []
        path = Path(directory)
        identity = path.relative_to(captures).as_posix()
        if not wanted or identity in wanted:
            output.append(path)
    return sorted(output)


def extract_sim_cycles(sim_json: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for launch in sim_json.get("launches", []):
        timing = launch.get("timing") or {}
        frontier = timing.get("completion_frontier") or {}
        value = frontier.get("lower_bound_cycles")
        if value is None:
            value = timing.get("terminal_modeled_event_time")
        numeric = finite_float(value)
        if numeric is not None:
            values.append(numeric)
    # The SM120 fast stage path intentionally returns `fast_frontier` rather
    # than full timing launches.  Keep its model-time endpoint on the same
    # per-launch surface so the runtime metric can compare the real executable
    # fast path against the uninstrumented CUDA-event numerator.
    if not values:
        for launch in sim_json.get("fast_frontier", []):
            if not isinstance(launch, dict):
                continue
            summary = launch.get("summary") or {}
            numeric = finite_float(
                summary.get("maximum_summarized_final_cycle")
                or summary.get("maximum_exact_final_cycle")
            )
            if numeric is not None:
                values.append(numeric)
    return values


def launch_feature_attribution(
    launch: dict[str, Any], sidecar: dict[str, Any]
) -> dict[str, Any]:
    """Emit only features owned by one simulator launch.

    ``trace_metrics`` is already computed per launch by ``sim-run``.  The
    sidecar summary is filtered by the launch's tracer ``kernel_id`` before it
    reaches this function.  This prevents capture-wide TMA/synchronization
    totals from being copied into every launch row.
    """

    trace_metrics = launch.get("trace_metrics") or {}
    instructions = trace_metrics.get("instructions") or {}
    by_opcode = trace_metrics.get("by_opcode") or {}
    opcode_counts: dict[str, int] = {}
    async_counts: dict[str, int] = {}
    sync_counts: dict[str, int] = {}
    for opcode, activity in by_opcode.items():
        if not isinstance(activity, dict):
            continue
        count = int(activity.get("warp_instructions", 0) or 0)
        name = str(opcode)
        opcode_counts[name] = count
        upper = name.upper()
        if upper.startswith(("UTMA", "UBLK", "LDGSTS")):
            async_counts[name] = count
        if upper.startswith(("SYNCS", "ARRIVES", "UTCBAR", "BAR", "DEPBAR")):
            sync_counts[name] = count
    memory = trace_metrics.get("memory") or {}
    memory_requests = sum(
        int(value.get("requests", 0) or 0)
        for value in memory.values()
        if isinstance(value, dict)
    )
    return {
        "scope": "single_simulator_launch_and_matching_tracer_kernel_id",
        "trace": {
            "warp_instructions": int(instructions.get("warp_instructions", 0) or 0),
            "active_thread_instructions": int(
                instructions.get("active_thread_instructions", 0) or 0
            ),
            "predicate_on_thread_instructions": int(
                instructions.get("predicate_on_thread_instructions", 0) or 0
            ),
            "memory_requests": memory_requests,
            "sectors_decoded": int(trace_metrics.get("sectors_decoded", 0) or 0),
            "opcode_warp_instructions": opcode_counts,
            "async_opcode_warp_instructions": async_counts,
            "sync_opcode_warp_instructions": sync_counts,
        },
        "sidecar": {
            "kernel_id_scope": sidecar.get("kernel_id_scope"),
            "tma_records": sidecar.get("tma_records", 0),
            "sync_records": sidecar.get("sync_records", 0),
            "tma_byte_count_sum": sidecar.get("tma_byte_count_sum", 0),
            "tma_opcode_classes": sidecar.get("tma_opcode_classes", []),
            "mbarrier_addresses": sidecar.get("mbarrier_addresses", []),
            "descriptor_bytes_observed": sidecar.get("descriptor_bytes_observed", 0),
            "descriptor_bytes_scope": sidecar.get(
                "descriptor_bytes_scope", "sidecar_descriptor_metadata_only"
            ),
            "predicated_off_records": sidecar.get("predicated_off_records", 0),
            "parse_errors": sidecar.get("parse_errors", 0),
            "destination_visibility": sidecar.get(
                "destination_visibility", "unobserved_without_consumer_payload_litmus"
            ),
            "completion_timestamp": sidecar.get("completion_timestamp", "unobserved"),
        },
        "source_contract": {
            "trace": "simulator result.launches[i].trace_metrics",
            "sidecar": "captures/<name>/extra_info/*jsonl filtered by kernel_id",
            "capture_totals_copied": False,
        },
    }


def normalized_kernel_name(name: str) -> str:
    return re.sub(r"___[0-9]+$", "", name or "")


def nested_numeric(document: dict[str, Any], *keys: str) -> float | None:
    """Find the first finite timing field in the direct-run result."""

    scopes: list[Any] = [document]
    for key in ("timing", "performance", "report", "alpha"):
        value = document.get(key)
        if isinstance(value, dict):
            scopes.append(value)
    for scope in scopes:
        for key in keys:
            value = finite_float(scope.get(key)) if isinstance(scope, dict) else None
            if value is not None and value > 0:
                return value
    return None


def join_ncu_rows(
    sim_launches: list[dict[str, Any]], ncu_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Associate independent launches without using measured timing values.

    TraceDir reports (device, stream, launch) order, whereas NCU uses launch
    chronology. With equal populations and unique ids on one device, rank
    the tracer kernel ids and NCU ids separately. This ordinal association is
    still a diagnostic, not shared run identity. Never use shape alone as
    kernel identity or force a known-incompatible grid/block pair.
    """
    def shape(value: Any) -> tuple[int, ...] | None:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return tuple(int(item) for item in value)
        values = re.findall(r"-?\d+", str(value or ""))
        return tuple(int(item) for item in values) if len(values) == 3 else None

    def compatible(launch: dict[str, Any], row: dict[str, Any]) -> bool:
        for sim_key, ncu_key in (("grid", "grid_size"), ("block", "block_size")):
            left, right = shape(launch.get(sim_key)), shape(row.get(ncu_key))
            if left is not None and right is not None and left != right:
                return False
        return True

    ordinal = {}
    sim_ids = [launch.get("kernel_id") for launch in sim_launches]
    ncu_ids = [row.get("ncu_launch_id") for row in ncu_rows]
    if (len(sim_launches) == len(ncu_rows)
        and all(isinstance(i, int) for i in sim_ids + ncu_ids)
        and len(set(sim_ids)) == len(sim_ids)
        and len(set(ncu_ids)) == len(ncu_ids)
        and len({launch.get("device_id") for launch in sim_launches}) <= 1
        and len({row.get("context") for row in ncu_rows}) <= 1):
        ordinal = dict(zip(
            sorted(range(len(sim_ids)), key=lambda i: sim_ids[i]),
            sorted(range(len(ncu_ids)), key=lambda i: ncu_ids[i]),
        ))

    names = [normalized_kernel_name(str(row.get("kernel_name", ""))) for row in ncu_rows]
    unused = set(range(len(ncu_rows)))
    joined: list[dict[str, Any]] = []
    for index, launch in enumerate(sim_launches):
        sim_name = str(launch.get("kernel_name", ""))
        sim_norm = normalized_kernel_name(sim_name)
        exact = [candidate for candidate in unused
                 if names[candidate] == sim_norm and compatible(launch, ncu_rows[candidate])]
        candidate = ordinal.get(index, index)
        if candidate in unused and compatible(launch, ncu_rows[candidate]) and (
            not exact or candidate in exact
        ):
            status = "identity" if len(exact) == 1 else "ordered_fallback"
        elif len(exact) == 1:
            candidate = exact[0]
            status = "identity"
        else:
            candidate = None
            status = "unmatched"
        if candidate is not None:
            unused.discard(candidate)
            ncu = ncu_rows[candidate]
            joined.append(
                {
                    "launch_index": index,
                    "kernel_id": launch.get("kernel_id"),
                    "kernel_name": sim_name,
                    "join_status": status,
                    "order_basis": "kernel_id_rank" if ordinal else "reported_launch_order",
                    "ncu_row_index": candidate,
                    "ncu_launch_id": ncu.get("ncu_launch_id"),
                    "ncu_kernel_name": ncu.get("kernel_name"),
                    "ncu_gpu_duration_us": ncu.get("gpu_time_duration_sum"),
                }
            )
        else:
            joined.append(
                {
                    "launch_index": index,
                    "kernel_id": launch.get("kernel_id"),
                    "kernel_name": sim_name,
                    "join_status": status,
                    "ncu_row_index": None,
                    "ncu_launch_id": None,
                    "ncu_kernel_name": None,
                    "ncu_gpu_duration_us": None,
                }
            )
    for candidate in sorted(unused):
        ncu = ncu_rows[candidate]
        joined.append(
            {
                "launch_index": None,
                "kernel_id": None,
                "kernel_name": None,
                "join_status": "ncu_only",
                "ncu_row_index": candidate,
                "ncu_launch_id": ncu.get("ncu_launch_id"),
                "ncu_kernel_name": ncu.get("kernel_name"),
                "ncu_gpu_duration_us": ncu.get("gpu_time_duration_sum"),
            }
        )
    return joined


def compare_native_metrics_to_ncu(
    sim_launches: list[dict[str, Any]],
    ncu_rows: list[dict[str, Any]],
    launch_join: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Emit a small unit-aware simulator-vs-NCU error table.

    This intentionally compares only scalar rows carrying the same explicit
    unit.  Rates, percentages, cache metrics, and profiler-duration rows that
    lack a simulator value remain visible as ``unavailable``/``unit_mismatch``
    rather than being coerced into a score.
    """

    errors: list[dict[str, Any]] = []
    for index, launch in enumerate(sim_launches):
        if index >= len(launch_join):
            continue
        joined = launch_join[index]
        row_index = joined.get("ncu_row_index")
        if not isinstance(row_index, int) or row_index < 0 or row_index >= len(ncu_rows):
            continue
        ncu_metrics = ncu_rows[row_index].get("metrics", {})
        native = launch.get("native_metrics", {})
        metric_map = native.get("metrics", {}) if isinstance(native, dict) else {}
        metric_map = dict(metric_map) if isinstance(metric_map, dict) else {}
        counters = launch.get("counters", {})
        existing_ncu_metrics = {
            str(value.get("ncu_metric"))
            for value in metric_map.values()
            if isinstance(value, dict) and value.get("ncu_metric")
        }
        if (
            isinstance(counters, dict)
            and finite_float(counters.get("instructions_issued")) is not None
            and "smsp__inst_executed.sum" not in existing_ncu_metrics
        ):
            metric_map["instructions_issued"] = {
                "ncu_metric": "smsp__inst_executed.sum",
                "value": counters.get("instructions_issued"),
                "unit": "inst",
            }
        for key, metric in metric_map.items():
            if not isinstance(metric, dict):
                continue
            ncu_name = metric.get("ncu_metric")
            sim_value = finite_float(metric.get("value"))
            sim_unit = str(metric.get("unit", ""))
            if not ncu_name or sim_value is None:
                continue
            observed = ncu_metrics.get(ncu_name)
            base = {
                "launch_index": index,
                "kernel_name": launch.get("kernel_name"),
                "metric_key": key,
                "ncu_metric": ncu_name,
                "simulator_value": sim_value,
                "simulator_unit": sim_unit,
                "join_status": joined.get("join_status"),
            }
            if not isinstance(observed, dict) or finite_float(observed.get("value")) is None:
                base["status"] = "unavailable_ncu_metric"
                errors.append(base)
                continue
            ncu_value = float(observed["value"])
            ncu_unit = str(observed.get("unit", ""))
            base["ncu_value"] = ncu_value
            base["ncu_unit"] = ncu_unit
            if sim_unit and ncu_unit and sim_unit != ncu_unit:
                base["status"] = "unit_mismatch"
                errors.append(base)
                continue
            absolute = sim_value - ncu_value
            base["absolute_error"] = absolute
            base["relative_error"] = absolute / ncu_value if ncu_value != 0 else None
            base["absolute_relative_error"] = (
                abs(absolute) / abs(ncu_value) if ncu_value != 0 else None
            )
            base["reference_status"] = "zero_reference" if ncu_value == 0 else "nonzero_reference"
            # Do not hide a mismatch behind one generic "compared" state. A
            # future alpha tolerance may be added as a declared field, but
            # this first lane reports exact equality versus non-equality.
            base["status"] = (
                "zero_reference" if ncu_value == 0 else "match" if absolute == 0.0 else "mismatch"
            ) if joined.get("join_status") == "identity" else "diagnostic_ordered_join"
            errors.append(base)
    return errors


def build_traffic_observation(
    sim_launches: list[dict[str, Any]],
    ncu_rows: list[dict[str, Any]],
    launch_join: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep simulator trace traffic and raw NCU sectors side by side.

    SM120 L1/L2 geometry, route policy, and TMA packetization are not admitted
    by the alpha lane.  Consequently this function deliberately does not
    compute hit/miss errors.  It preserves the two observation domains for the
    next cache milestone, so a future join cannot accidentally fit a geometry
    from a metric with a different replay/clock/cache scope.
    """

    rows: list[dict[str, Any]] = []
    for index, launch in enumerate(sim_launches):
        joined = launch_join[index] if index < len(launch_join) else None
        ncu_index = joined.get("ncu_row_index") if joined else None
        ncu = ncu_rows[ncu_index] if isinstance(ncu_index, int) and 0 <= ncu_index < len(ncu_rows) else None
        raw_ncu = {}
        if ncu is not None:
            for name, value in (ncu.get("metrics") or {}).items():
                if not isinstance(value, dict):
                    continue
                if any(token in name for token in ("l1tex__t_sectors", "lts__t_sectors", "dram__sectors")):
                    raw_ncu[name] = value
        trace = launch.get("trace_metrics") or {}
        rows.append(
            {
                "launch_index": index,
                "kernel_name": launch.get("kernel_name"),
                "join_status": joined.get("join_status") if joined else "unmatched",
                "status": "observation_domains_separate_no_geometry_join",
                "simulator_trace_memory": trace.get("memory", {}),
                "simulator_trace_coherence": trace.get("coherence", {}),
                "ncu_sector_metrics": raw_ncu,
                "service_envelope": {
                    "status": "observation_only_no_geometry_fit",
                    "candidate_axes": [
                        "l1_route",
                        "l1_residency",
                        "l2_residency",
                        "outstanding_demand",
                        "dram_read_write_service",
                    ],
                    "simulator_request_domain": trace.get("memory", {}),
                    "hardware_counter_domain": raw_ncu,
                    "logical_partition_model": "not_selected",
                    "partition_mapping_status": "typed_gap",
                    "mshr_count": "typed_gap",
                    "dram_queue_policy": "typed_gap",
                },
            }
        )
    return rows


def alpha_approximation_labels(sim_json: dict[str, Any]) -> dict[str, Any] | None:
    """Expose the flat alpha's unsupported completion/timing families.

    These are report labels only: the wrapper never alters the simulator
    result or substitutes values.  A successful trace replay and the model's
    internal ``complete`` frontier must not masquerade as physical TMA/
    cp.async/cluster completion when those operations use generic alpha
    arithmetic.
    """

    if not sim_json.get("sm120_alpha"):
        return None
    launches = []
    consumed_config = sim_json.get("sm120_alpha_config")
    if not isinstance(consumed_config, dict):
        consumed_config = {}
    profile_id = str(
        sim_json.get("sm120_alpha_profile_id")
        or "sm120_alpha_config_without_profile_id"
    )
    all_kinds: set[str] = set()
    native_issue_envelope: dict[str, Any] | None = None
    native_service_envelope: dict[str, Any] | None = None
    native_completion_ledger: dict[str, Any] | None = None
    for index, launch in enumerate(sim_json.get("launches", [])):
        if not isinstance(launch, dict):
            continue
        by_opcode = (launch.get("trace_metrics") or {}).get("by_opcode", {})
        kinds: set[str] = set()
        counts: dict[str, int] = {}
        for opcode, metrics in by_opcode.items():
            if not isinstance(metrics, dict):
                continue
            count = int(metrics.get("warp_instructions", 0) or 0)
            if count == 0:
                continue
            if str(opcode).startswith("UTMA"):
                kinds.add("tma_completion_unmodeled")
                counts[str(opcode)] = count
            elif str(opcode).startswith("LDGSTS"):
                kinds.add("cp_async_completion_approximated")
                counts[str(opcode)] = count
            elif str(opcode) in {"QMMA", "OMMA", "HMMA", "IMMA", "BMMA"}:
                kinds.add("tensor_pipeline_timing_approximated")
                counts[str(opcode)] = count
            elif str(opcode) in {
                "MBAR",
                "UTMABAR",
                "UTMACCTL",
                "UTMACMDFLUSH",
                "DEPBAR",
            }:
                kinds.add("mbarrier_completion_approximated")
                counts[str(opcode)] = count
        all_kinds.update(kinds)
        timing = launch.get("timing") or {}
        timing_alpha = timing.get("alpha_approximation") or {}
        if isinstance(timing_alpha.get("issue_envelope"), dict):
            native_issue_envelope = timing_alpha["issue_envelope"]
        if isinstance(timing_alpha.get("service_envelope"), dict):
            native_service_envelope = timing_alpha["service_envelope"]
        if isinstance(timing_alpha.get("completion_ledger"), dict):
            native_completion_ledger = timing_alpha["completion_ledger"]
        readiness = timing.get("readiness") or {}
        async_groups = readiness.get("async_groups") or {}
        launches.append(
            {
                "launch_index": index,
                "kernel_id": launch.get("kernel_id"),
                "kernel_name": launch.get("kernel_name"),
                "approximation_kinds": sorted(kinds),
                "affected_opcode_counts": counts,
                "completion_status_semantics": "model_internal_not_physical_completion",
                "completion_ledger": {
                    "issue": "observed_in_simulator_trace",
                    "source_transfer": "model_relative_flat_demand",
                    "transaction_complete": "unobserved_sm120",
                    "barrier_arrival": "model_relative_if_recorded",
                    "wait_or_test": "model_relative_if_recorded",
                    "destination_visibility": "unobserved_without_consumer_payload_observation",
                    "groups_committed": async_groups.get("groups_committed", 0),
                    "groups_waited": async_groups.get("groups_waited", 0),
                    "payload_observation": "not_present_in_alpha_capture_contract",
                },
            }
        )
    return {
        "status": "provisional_flat_alpha",
        "profile_id": profile_id,
        "timing_scope": "model_relative_not_hardware_timing",
        "cache_and_dram_accuracy_available": False,
        "issue_envelope": native_issue_envelope or {
            "status": "configured_alpha_inputs",
            "profile_id": profile_id,
            "issue_ports_configured": consumed_config.get("issue_ports"),
            "dual_issue_configured": consumed_config.get("dual_issue_supported"),
            "scalar_latency_configured_cycles": consumed_config.get("scalar_latency_cycles"),
            "scalar_initiation_configured_cycles": consumed_config.get(
                "scalar_initiation_interval_cycles"
            ),
            "max_resident_warps_configured": consumed_config.get("max_resident_warps_per_sm"),
            "max_resident_blocks_configured": consumed_config.get("max_resident_blocks_per_sm"),
            "configuration_source": "sim_json.sm120_alpha_config consumed by executor",
            "non_claim": "scenario inputs only; not a confidence interval or physical scheduler topology",
        },
        "service_envelope": native_service_envelope or {
            "status": "observation_only_no_geometry_fit",
            "source_domains": ["trace_memory_coherence", "ncu_raw_l1_l2_dram_sectors"],
            "route_policy": "unidentified",
            "residency": "effective_working_set_bracket_unavailable_in_flat_replay",
            "outstanding_demand": "typed_gap",
            "dram_service": "typed_gap",
            "geometry_status": "typed_gap",
            "partition_mapping_status": "typed_gap",
            "mshr_count": "typed_gap",
            "queue_policy": "typed_gap",
            "non_claim": "logical service envelope only; no physical cache/DRAM topology",
        },
        "completion_ledger": native_completion_ledger or {
            "status": "typed_alpha_completion_ledger",
            "issue": "observed_in_trace",
            "transaction_complete": "unobserved_sm120",
            "barrier_arrival": "model_relative_if_decoded",
            "wait_or_test": "model_relative_if_decoded",
            "destination_visibility": "unobserved_without_consumer_payload_observation",
            "payload_observation": "not_present_in_alpha_trace_contract",
            "affected_async_records": 0,
            "non_claim": "semantic ledger only; no native completion timing",
        },
        "generic_scheduler_assumptions": {
            "status": "executor_assumptions_not_hardware_facts",
            "resource_mapping": "alpha_executor_defined",
            "dependency_barrier_domain": "alpha_control_sidecar_domain",
            "source": "sm120 alpha implementation",
        },
        "approximation_kinds": sorted(all_kinds),
        "launches": launches,
    }


def fast_envelope_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract stage features for the fast hardware-in-the-loop evaluator."""

    rows: list[dict[str, Any]] = []
    for record in records:
        sim = record.get("simulator") or {}
        result = sim.get("result") or {}
        for launch_index, launch in enumerate(result.get("launches", []) if isinstance(result, dict) else []):
            metrics = launch.get("trace_metrics") or {}
            memory = metrics.get("memory") or {}
            readiness = (launch.get("timing") or {}).get("readiness") or {}
            async_groups = readiness.get("async_groups") or {}
            sidecar_all = record.get("async_sidecar_observation") or {}
            sidecar_by_kernel = sidecar_all.get("by_kernel_id") or {}
            sidecar = sidecar_by_kernel.get(
                _sidecar_kernel_key(launch.get("kernel_id")),
                {
                    "tma_records": 0,
                    "sync_records": 0,
                    "tma_byte_count_sum": 0,
                },
            )
            rows.append(
                {
                    "capture_name": record.get("capture_name"),
                    "launch_index": launch_index,
                    "kernel_id": launch.get("kernel_id"),
                    "stage_features": {
                        "issue_cycles": (launch.get("timing") or {}).get("diagnostic_last_issue_cycles"),
                        "instructions_issued": (launch.get("counters") or {}).get("instructions_issued"),
                        "memory_requests": sum(
                            int(value.get("requests", 0) or 0)
                            for value in memory.values()
                            if isinstance(value, dict)
                        ),
                        "decoded_sectors": metrics.get("sectors_decoded"),
                        "async_source_cycles": async_groups.get("source_cycles", 0),
                        "async_groups_waited": async_groups.get("groups_waited", 0),
                        "tma_sidecar_bytes": sidecar.get("tma_byte_count_sum", 0),
                        "tma_sidecar_records": sidecar.get("tma_records", 0),
                        "sync_sidecar_records": sidecar.get("sync_records", 0),
                        "barrier_records": readiness.get("barrier_pipeline", {}).get("arrivals", 0),
                    },
                    "prediction_status": "feature_vector_only_until_pc_sampling_calibration",
                }
            )
    return rows


def _score_error_values(records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        for error in (record.get("simulator") or {}).get("ncu_metric_errors", []) or []:
            value = finite_float(error.get("relative_error"))
            if value is not None:
                values.append(abs(value))
    return values


def build_campaign_scorecard(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Emit descriptive development accounting for the retained alpha corpus.

    The retained 12-capture corpus was not declared before fitting and is not
    an untouched holdout.  Keep aggregate diagnostics useful, but do not turn
    alphabetical ordering into a calibration/holdout claim or copy the
    driver's replay-speed metric into an evaluator scorecard.
    """

    ordered = sorted(records, key=lambda item: str(item.get("capture_name", "")))
    diagnostic_errors = _score_error_values(ordered)
    fast_rows = fast_envelope_rows(records)
    return {
        "schema_version": 2,
        "status": "development_corpus_accounting_no_holdout_claim",
        "development_scope": {
            "capture_count": len(ordered),
            "capture_names": [x.get("capture_name") for x in ordered],
            "launch_count": sum(
                len(((x.get("simulator") or {}).get("result") or {}).get("launches", []))
                for x in ordered
            ),
            "split_status": "not_declared_before_observation",
            "note": "Retained captures are development data; no calibration or untouched holdout is asserted.",
        },
        "external_cycle_simulator_comparison": {
            "status": "permanently_banned",
            "reason": "external detailed replay is thousands of times slower and is not an alpha development metric",
            "replacement": "fast_hardware_in_loop_stage_envelope",
        },
        "gcom": {
            "status": "optional_fast_analytical_reference",
            "repository": "https://github.com/yonsei-hpcp/gcom",
            "commit": "ce033b1",
            "score_status": "feature-level reference only; no external replay comparison",
        },
        "fast_hardware_in_loop_envelope": {
            "status": "feature_vectors_emitted",
            "row_count": len(fast_rows),
            "rows": fast_rows,
            "next_calibration_source": "CUPTI PC sampling stall vectors plus uninstrumented CUDA events",
            "model_form": "stage-centric issue_memory_async_quantile_envelope",
        },
        "alpha_ncu_diagnostic": {
            "development_rows": len(diagnostic_errors),
            "development_median_abs_relative_error": statistics.median(diagnostic_errors)
            if diagnostic_errors
            else None,
            "development_max_abs_relative_error": max(diagnostic_errors)
            if diagnostic_errors
            else None,
            "interpretation": "descriptive alpha instruction projection; not a holdout or generalization score",
        },
        "replay_speed": {
            "status": "reported_by_replay_driver_only",
            "source": "simulator.replay_speedup",
            "metric_scope": "not_aggregated_by_development_scorecard",
        },
        "non_claim": "no holdout accuracy, evaluator speed, or external cycle-simulator comparison is asserted",
    }


def run_simulator(
    sim_bin: Path,
    trace: Path,
    control: Path,
    card: Path,
    extra_args: list[str],
) -> tuple[int, float, str, str, dict[str, Any] | None, list[str]]:
    command = [
        str(sim_bin),
        "--trace",
        str(trace.resolve()),
        "--control",
        str(control),
        "--arch-card",
        str(card),
        *extra_args,
    ]
    started = time.perf_counter_ns()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    parsed: dict[str, Any] | None = None
    try:
        value = json.loads(completed.stdout)
        if isinstance(value, dict):
            parsed = value
    except json.JSONDecodeError:
        pass
    for line in reversed(completed.stdout.splitlines()):
        if parsed is not None:
            break
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    return completed.returncode, elapsed_ms, completed.stdout, completed.stderr, parsed, command


def join_one(
    capture: Path,
    work_root: Path,
    sim_bin: Path,
    card: Path,
    controls_root: Path | None,
    synthesize_dir: Path | None,
    run: bool,
    sim_extra_args: list[str],
    baremetal_runtime_index: int,
    hardware_timing: Path | None,
) -> dict[str, Any]:
    capture_started_ns = time.perf_counter_ns()
    relative_name = capture.relative_to(work_root / "captures").as_posix()
    profile = work_root / "profiles" / relative_name
    stats = parse_stats(capture / "stats.csv")
    ncu = parse_ncu_raw(profile / "raw.csv")
    observation = parse_program_observation(capture / "program.stdout")
    hardware = parse_hardware_timing(hardware_timing, relative_name)
    sm_frequency_ghz = parse_effective_sm_frequency(profile / "effective_frequency_summary.csv")
    metadata = load_static_metadata(capture)
    async_observation = async_sidecar_observation(capture)
    input_ready_ns = time.perf_counter_ns()
    candidates = existing_control_candidates(capture, controls_root, relative_name)
    control_path = candidates[0] if candidates else None
    control_status = "existing" if control_path else "missing_legacy_control_sidecar"
    generated_path: Path | None = None
    if control_path is None and synthesize_dir is not None and metadata is not None:
        generated_path = synthesize_dir / f"{relative_name}.control_bits.json"
        synthesize_legacy_sidecar(capture, metadata, generated_path)
        control_path = generated_path
        control_status = "derived_transport_sidecar"

    record: dict[str, Any] = {
        "schema_version": 1,
        "evidence_schema_version": 2,
        "capture_name": relative_name,
        "capture_dir": str(capture),
        "profile_dir": str(profile) if profile.is_dir() else None,
        "trace": {
            "dynamic_trace": str(capture / "dynamic_trace.pb"),
            "stats_csv": str(capture / "stats.csv"),
            "stats_rows": stats,
            "native_static_metadata": str(capture / "extra_info/native_static_metadata.json")
            if metadata is not None
            else None,
            "kernel_count": len(metadata.get("kernels", [])) if metadata else 0,
        },
        "baremetal_observation": observation,
        "hardware_timing": hardware,
        "async_sidecar_observation": async_observation,
        "evidence_contract": evidence_contract(
            capture=capture,
            profile=profile,
            stats=stats,
            ncu=ncu,
            metadata=metadata,
            hardware=hardware,
        ),
        # program.stdout is emitted while NVBit is active.  It is retained as
        # a tracer-run host observation, not silently promoted to uninstrumented
        # bare-metal CUDA-event ground truth.
        "tracer_host_timing": {
            "status": "tracer_run_host_observation"
            if observation.get("runtime_ms_values")
            else "unavailable",
            "source": "captures/*/program.stdout (NVBit tracer active)"
            if observation.get("runtime_ms_values")
            else None,
            "uninstrumented_cuda_event_ms": None,
        },
        "trace_coverage": {
            "status": "full_declared_by_hardware_timing"
            if hardware and hardware.get("coverage_full")
            else "unknown_without_sampling_sidecar",
            "speed_metric_eligible": bool(hardware and hardware.get("coverage_full")),
            "note": "A traced-SM replay is not a full-launch runtime unless a complete-capture declaration is supplied.",
        },
        "ncu": {
            "raw_csv": str(profile / "raw.csv") if (profile / "raw.csv").is_file() else None,
            "rows": ncu,
            "effective_sm_frequency_ghz": sm_frequency_ghz,
        },
        "control": {
            "status": control_status,
            "path": str(control_path) if control_path else None,
            "generated_path": str(generated_path) if generated_path else None,
        },
        "simulator": {"requested": bool(run), "binary": str(sim_bin), "arch_card": str(card)},
    }

    if run and control_path is not None:
        code, wall_ms, stdout, stderr, parsed, command = run_simulator(
            sim_bin, capture, control_path, card, sim_extra_args
        )
        sim_record: dict[str, Any] = {
            "returncode": code,
            "command": command,
            "wall_ms": wall_ms,
            "stdout": stdout,
            "stderr": stderr,
            "result": parsed,
        }
        sim_record["status"] = "completed" if code == 0 and parsed is not None else "failed_or_unparsed"
        # Keep the capture aggregate for provenance, but expose the compact
        # per-kernel map separately.  Per-launch rows below select exactly one
        # kernel_id bucket; they never inherit capture totals.
        sim_record["completion_witness"] = async_observation
        sim_record["completion_witness_by_kernel_id"] = async_observation.get(
            "by_kernel_id", {}
        )
        cycles: list[float] = []
        if parsed is not None:
            labels = alpha_approximation_labels(parsed)
            if labels is not None:
                sim_record["alpha_approximation"] = labels
            cycles = extract_sim_cycles(parsed)
            sim_record["completion_cycles_by_launch"] = cycles
            sim_record["completion_cycles_total"] = sum(cycles) if cycles else None
            if sm_frequency_ghz and cycles:
                sim_record["sim_time_ms_at_ncu_sm_clock"] = sum(cycles) / (sm_frequency_ghz * 1_000_000.0)
        selected_runtime = None
        values = observation.get("runtime_ms_values", [])
        if values:
            index = baremetal_runtime_index if 0 <= baremetal_runtime_index < len(values) else 0
            selected_runtime = values[index]
        sim_record["baremetal_runtime_ms_selected"] = selected_runtime
        successful_replay = code == 0 and parsed is not None
        if selected_runtime and wall_ms > 0 and successful_replay:
            sim_record["tracer_observation_over_sim_wall_ratio"] = selected_runtime / wall_ms
        if hardware and hardware.get("coverage_full") and successful_replay:
            hardware_elapsed_ns = float(hardware["hardware_elapsed_ns"])
            reported_total_ns = nested_numeric(
                parsed or {}, "sim_total_wall_ns", "sim_wall_ns_total", "wall_ns_total", "total_wall_ns"
            )
            reported_replay_ns = nested_numeric(
                parsed or {},
                "sim_wall_ns_replay",
                "sim_replay_wall_ns",
                "replay_wall_ns",
                "replay_ns",
            )
            fast_only = bool(parsed and parsed.get("fast_frontier_only"))
            sim_record["sim_wall_ns_total"] = reported_total_ns or wall_ms * 1_000_000.0
            # Fast-frontier reports deliberately do not carry the ordinary
            # full-replay timer fields.  Their process wall boundary is the
            # executable replay phase and is therefore the correct denominator
            # for the requested speed metric.
            sim_record["sim_wall_ns_replay"] = (
                reported_replay_ns
                if reported_replay_ns is not None
                else wall_ms * 1_000_000.0
                if fast_only
                else None
            )
            sim_record["sim_wall_timing_source"] = (
                "simulator_report"
                if reported_replay_ns or reported_total_ns
                else "driver_process_elapsed_fast_frontier"
                if fast_only
                else "driver_process_elapsed_fallback"
            )
            sim_record["process_speedup"] = hardware_elapsed_ns / sim_record["sim_wall_ns_total"]
            sim_record["process_speedup_status"] = "available_full_coverage_hardware_timing"
            replay_denominator = sim_record.get("sim_wall_ns_replay")
            sim_record["replay_speedup"] = (
                hardware_elapsed_ns / float(replay_denominator)
                if replay_denominator is not None and float(replay_denominator) > 0
                else None
            )
            sim_record["replay_speedup_numerator"] = "uninstrumented_hardware_elapsed_ns"
            sim_record["replay_speedup_denominator"] = "simulator_replay_wall_ns"
            sim_record["replay_speedup_status"] = (
                "available_full_coverage_hardware_timing"
                if reported_replay_ns or fast_only
                else "unavailable_replay_phase_timer_absent_use_process_speedup"
            )
        else:
            sim_record["replay_speedup"] = None
            sim_record["replay_speedup_status"] = (
                "unavailable_requires_full_coverage_hardware_timing"
            )
        modeled = sim_record.get("sim_time_ms_at_ncu_sm_clock")
        if selected_runtime and modeled and modeled > 0 and successful_replay:
            sim_record["tracer_runtime_over_modeled_clock_ratio"] = selected_runtime / modeled
            sim_record["modeled_clock_speedup_status"] = "tracer_run_observation_over_derived_clock_time"
        ncu_durations_ms = [
            duration_ms_from_ncu(row) for row in ncu
        ]
        ncu_total_ms = sum(value for value in ncu_durations_ms if value is not None)
        sim_record["ncu_gpu_duration_total_ms"] = ncu_total_ms or None
        launch_join: list[dict[str, Any]] = []
        if parsed is not None:
            comparison_launches = [item for item in (parsed.get("launches") or parsed.get("fast_frontier") or []) if isinstance(item, dict)]
            launch_join = join_ncu_rows(
                comparison_launches, ncu
            )
            sim_record["ncu_join"] = launch_join
            sim_record["ncu_metric_errors"] = compare_native_metrics_to_ncu(
                [item for item in parsed.get("launches", []) if isinstance(item, dict)],
                ncu,
                launch_join,
            )
            sim_record["traffic_observation"] = build_traffic_observation(
                [item for item in parsed.get("launches", []) if isinstance(item, dict)],
                ncu,
                launch_join,
            )
        if successful_replay and cycles:
            per_launch: list[dict[str, Any]] = []
            parsed_launches = parsed.get("launches", []) if isinstance(parsed, dict) else []
            fast_launches = parsed.get("fast_frontier", []) if isinstance(parsed, dict) else []
            for index, cycles_value in enumerate(cycles):
                launch = parsed_launches[index] if index < len(parsed_launches) else {}
                joined = launch_join[index] if index < len(launch_join) else None
                row = ncu[joined["ncu_row_index"]] if joined and joined.get("ncu_row_index") is not None else None
                if not launch and index < len(fast_launches) and isinstance(fast_launches[index], dict):
                    launch = fast_launches[index]
                launch_kernel_id = launch.get("kernel_id") if isinstance(launch, dict) else None
                launch_sidecar = async_observation.get("by_kernel_id", {}).get(
                    _sidecar_kernel_key(launch_kernel_id),
                    _finish_sidecar_bucket(
                        _sidecar_bucket(), capture=capture, kernel_id=launch_kernel_id
                    ),
                )
                item: dict[str, Any] = {
                    "launch_index": index,
                    "kernel_id": launch_kernel_id,
                    "completion_cycles": cycles_value,
                    "join_status": joined.get("join_status") if joined else "unmatched",
                    "kernel_name": joined.get("kernel_name") if joined else None,
                    "ncu_launch_id": joined.get("ncu_launch_id") if joined else None,
                    "ncu_gpu_duration_us": joined.get("ncu_gpu_duration_us") if joined else None,
                    "ncu_gpu_duration_ms": duration_ms_from_ncu(row) if row else None,
                    "ncu_sm_elapsed_cycles_avg": row.get("sm_cycles_elapsed_avg") if row else None,
                    "ncu_metrics": (row.get("metrics") if row else {}) or {},
                    "comparison_status": "identity_joined_diagnostic"
                    if joined and joined.get("join_status") == "identity"
                    else "ordered_join_diagnostic",
                    "feature_attribution": launch_feature_attribution(
                        launch if isinstance(launch, dict) and "trace_metrics" in launch else {},
                        launch_sidecar,
                    )
                    if isinstance(launch, dict) and "trace_metrics" in launch
                    else {
                        "scope": "fast_frontier_launch",
                        "trace": {},
                        "sidecar": {},
                        "source_contract": {"fast_frontier_summary": True},
                    },
                }
                ncu_cycles = item["ncu_sm_elapsed_cycles_avg"]
                if isinstance(ncu_cycles, (int, float)) and ncu_cycles > 0:
                    item["cycle_relative_error"] = (cycles_value - ncu_cycles) / ncu_cycles
                    item["cycle_absolute_relative_error"] = abs(cycles_value - ncu_cycles) / ncu_cycles
                if sm_frequency_ghz:
                    modeled_ms = cycles_value / (sm_frequency_ghz * 1_000_000.0)
                    item["sim_time_ms_at_ncu_sm_clock"] = modeled_ms
                    ncu_ms = duration_ms_from_ncu(row) if row else None
                    if ncu_ms and ncu_ms > 0:
                        item["sim_over_ncu_duration_ratio"] = modeled_ms / ncu_ms
                per_launch.append(item)
            sim_record["per_launch"] = per_launch
        if modeled and ncu_total_ms and successful_replay:
            sim_record["sim_over_ncu_duration_ratio"] = modeled / ncu_total_ms
            sim_record["ncu_over_sim_speedup"] = ncu_total_ms / modeled
            sim_record["ncu_duration_status"] = "profiler_perturbed_observation_not_ground_truth"
            sim_record["ncu_duration_join_status"] = (
                "identity_joined_diagnostic"
                if launch_join and all(x.get("join_status") == "identity" for x in launch_join)
                else "ordered_or_unmatched_diagnostic_not_accuracy_score"
            )
        record["simulator"] = sim_record
    elif run:
        record["simulator"]["status"] = "not_run_missing_control_sidecar"
    record["driver_phase_timing"] = {
        "input_prepare_ms": (input_ready_ns - capture_started_ns) / 1_000_000.0,
        "simulator_process_ms": float((record.get("simulator") or {}).get("wall_ms", 0.0))
        if run
        else None,
        "capture_record_assembly_ms": (time.perf_counter_ns() - input_ready_ns) / 1_000_000.0,
        "scope": "one_capture_driver_boundary; output serialization is measured by main",
    }
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--sim-bin", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--arch-card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument(
        "--hardware-timing",
        type=Path,
        help="optional uninstrumented CUDA-event/host timing JSON file or directory",
    )
    parser.add_argument("--capture", action="append", dest="captures", metavar="NAME")
    parser.add_argument("--out", type=Path, default=Path("sm120_replay_bench.jsonl"))
    parser.add_argument(
        "--scorecard-out",
        type=Path,
        help="optional campaign-level common-support/holdout scorecard JSON",
    )
    parser.add_argument("--run", action="store_true", help="invoke sim-run, not just join local evidence")
    parser.add_argument(
        "--sm120-alpha",
        action="store_true",
        help="pass the explicit --sm120-alpha --mode flat seam to sim-run",
    )
    parser.add_argument(
        "--fast-frontier",
        action="store_true",
        help="use the executable SM120 stage/event fast frontier instead of full timing replay",
    )
    parser.add_argument(
        "--synthesize-control",
        type=Path,
        metavar="DIR",
        help="write derived transport sidecars here when no legacy sidecar exists",
    )
    parser.add_argument(
        "--baremetal-runtime-index",
        type=int,
        default=0,
        help="tracer-run runtime sample used only for a diagnostic ratio (default: first)",
    )
    parser.add_argument(
        "--sim-arg",
        action="append",
        default=[],
        help="extra argument passed to sim-run (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    captures = discover_captures(args.work_root, args.captures)
    if not captures:
        parser.error(f"no NVBit captures found below {args.work_root / 'captures'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sim_args = list(args.sim_arg)
    if args.sm120_alpha and "--sm120-alpha" not in sim_args:
        sim_args.append("--sm120-alpha")
    if args.sm120_alpha and "--mode" not in sim_args:
        sim_args.extend(["--mode", "flat"])
    if args.fast_frontier and "--fast-frontier-only" not in sim_args:
        sim_args.append("--fast-frontier-only")
    records: list[dict[str, Any]] = []
    with args.out.open("w", encoding="utf-8") as stream:
        for capture in captures:
            record = join_one(
                capture=capture,
                work_root=args.work_root,
                sim_bin=args.sim_bin,
                card=args.arch_card,
                controls_root=args.control_root,
                synthesize_dir=args.synthesize_control,
                run=args.run,
                sim_extra_args=sim_args,
                baremetal_runtime_index=args.baremetal_runtime_index,
                hardware_timing=args.hardware_timing,
            )
            records.append(record)
            serialization_started_ns = time.perf_counter_ns()
            encoded = json.dumps(record, sort_keys=True)
            serialization_ms = (time.perf_counter_ns() - serialization_started_ns) / 1_000_000.0
            record.setdefault("driver_phase_timing", {})["jsonl_serialization_ms"] = serialization_ms
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print(
                f"{record['capture_name']}: control={record['control']['status']} "
                f"ncu_rows={len(record['ncu']['rows'])} "
                f"runtime_ms={record['baremetal_observation']['runtime_ms_values']}",
                file=sys.stderr,
            )
    scorecard_path = args.scorecard_out or args.out.with_suffix(args.out.suffix + ".scorecard.json")
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_path.write_text(
        json.dumps(build_campaign_scorecard(records), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
