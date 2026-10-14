#!/usr/bin/env python3
"""Small real-corpus adapters shared by the full replay and NCU join tools.

This module discovers capture roots, translates native SM120 static metadata
into the control shape consumed by sim-run, and joins independent NCU rows
after replay. NCU observations remain outside the replay input path.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_ROOT = Path("/root/sm120_rtx_5090_capture")
DEFAULT_SIM = REPO_ROOT / "sim/target/release/sim-run"
DEFAULT_CARD = REPO_ROOT / "arch_profiles/cards/sm_120-rtx-5090.json"


def discover_captures(work_root: Path, selected: Iterable[str] | None) -> list[Path]:
    captures = work_root / "captures"
    if not captures.is_dir():
        return []
    wanted = set(selected or ())
    found: list[Path] = []
    for directory, subdirs, files in os.walk(captures):
        if "dynamic_trace.pb" not in files:
            continue
        subdirs[:] = []
        path = Path(directory)
        relative = path.relative_to(captures).as_posix()
        if not wanted or relative in wanted or path.name in wanted:
            found.append(path)
    return sorted(found)


def load_static_metadata(capture: Path) -> dict[str, Any] | None:
    path = capture / "extra_info/native_static_metadata.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


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


def synthesize_static_sidecar(
    capture: Path, metadata: dict[str, Any], output: Path
) -> dict[str, Any]:
    command_path = capture / "command.txt"
    command = command_path.read_text(encoding="utf-8", errors="replace").strip() if command_path.is_file() else ""
    binary = shlex.split(command)[0] if command else metadata.get("benchmark_name", "sm120_capture")
    functions: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, int]] = {}
    for kernel in metadata.get("kernels", []):
        raw_name = str(kernel.get("kernel_name", ""))
        name = re.sub(r"___[0-9]+$", "", raw_name)
        records: list[dict[str, Any]] = []
        for inst in kernel.get("instructions", []):
            pc = int(inst.get("pc_num_dec", 0) or 0)
            high_text = str(inst.get("control_word_hex") or inst.get("word1_hex") or "0")
            low_text = str(inst.get("word0_hex") or "0")
            try:
                high_word = int(high_text, 16)
            except ValueError:
                high_word = 0
            sass = str(inst.get("sass") or inst.get("opcode") or "UNKNOWN").strip().rstrip(";").strip()
            predicate = None
            if sass.startswith("@"):
                predicate, _, sass = sass.partition(" ")
                sass = sass.strip()
            records.append({
                "pc": pc,
                "predicate": predicate,
                "opcode_text": sass,
                "low_word": low_text if low_text.lower().startswith("0x") else f"0x{low_text}",
                "high_word": high_text if high_text.lower().startswith("0x") else f"0x{high_text}",
                "status": "decoded",
                "control": _wire_control(high_word),
            })
        functions[name] = records
        summaries[name] = {
            "instructions": len(records),
            "decoded": len(records),
            "unknown_encoding": 0,
            "stall_cycles_static_sum": sum(int(x["control"]["stall"]) for x in records),
        }
    document = {
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
        "binary_source": "extra_info/native_static_metadata.json",
        "summary": {
            "functions": summaries,
            "total_decoded": sum(item["decoded"] for item in summaries.values()),
            "total_unknown_encoding": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "inf", "-inf"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def parse_ncu_raw(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
        rows = list(csv.reader(stream))
    header_index = next((i for i, row in enumerate(rows) if row and row[0] == "ID" and "Kernel Name" in row), None)
    if header_index is None or header_index + 2 >= len(rows):
        return []
    header = rows[header_index]
    units = rows[header_index + 1]
    index = {name: i for i, name in enumerate(header)}
    output: list[dict[str, Any]] = []
    identity = {"ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC"}
    for row in rows[header_index + 2 :]:
        if not row or not row[0].strip().isdigit():
            continue
        metrics: dict[str, dict[str, Any]] = {}
        for name, i in index.items():
            if name in identity or i >= len(row):
                continue
            numeric = _number(row[i])
            if numeric is not None:
                metrics[name] = {"value": numeric, "unit": units[i] if i < len(units) else ""}
        def field(name: str) -> str:
            i = index.get(name, -1)
            return row[i] if 0 <= i < len(row) else ""
        output.append({
            "ncu_launch_id": int(row[0]),
            "kernel_name": field("Kernel Name"),
            "grid_size": field("Grid Size"),
            "block_size": field("Block Size"),
            "context": field("Context"),
            "stream": field("Stream"),
            "metrics": metrics,
        })
    return output


def _shape(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(item) for item in value)
    values = re.findall(r"-?\d+", str(value or ""))
    return tuple(int(value) for value in values) if len(values) == 3 else None


def _normalized_kernel_name(value: str) -> str:
    return re.sub(r"___[0-9]+$", "", value or "")


def join_ncu_rows(sim_launches: list[dict[str, Any]], ncu_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def compatible(launch: dict[str, Any], row: dict[str, Any]) -> bool:
        return all(
            _shape(launch.get(left)) is None
            or _shape(row.get(right)) is None
            or _shape(launch.get(left)) == _shape(row.get(right))
            for left, right in (("grid", "grid_size"), ("block", "block_size"))
        )
    sim_ids = [launch.get("kernel_id") for launch in sim_launches]
    ncu_ids = [row.get("ncu_launch_id") for row in ncu_rows]
    ordinal: dict[int, int] = {}
    if (
        len(sim_launches) == len(ncu_rows)
        and all(isinstance(value, int) for value in sim_ids + ncu_ids)
        and len(set(sim_ids)) == len(sim_ids)
        and len(set(ncu_ids)) == len(ncu_ids)
    ):
        ordinal = dict(zip(
            sorted(range(len(sim_ids)), key=sim_ids.__getitem__),
            sorted(range(len(ncu_ids)), key=ncu_ids.__getitem__),
        ))
    names = [_normalized_kernel_name(str(row.get("kernel_name", ""))) for row in ncu_rows]
    unused = set(range(len(ncu_rows)))
    joined: list[dict[str, Any]] = []
    for index, launch in enumerate(sim_launches):
        sim_name = str(launch.get("kernel_name", ""))
        exact = [candidate for candidate in unused if names[candidate] == _normalized_kernel_name(sim_name) and compatible(launch, ncu_rows[candidate])]
        candidate = ordinal.get(index, index)
        if candidate in unused and compatible(launch, ncu_rows[candidate]) and (not exact or candidate in exact):
            status = "identity" if len(exact) == 1 else "ordered_fallback"
        elif len(exact) == 1:
            candidate, status = exact[0], "identity"
        else:
            candidate, status = None, "unmatched"
        if candidate is None:
            joined.append({"launch_index": index, "kernel_id": launch.get("kernel_id"), "kernel_name": sim_name, "join_status": status, "order_basis": "reported_launch_order"})
            continue
        unused.discard(candidate)
        target = ncu_rows[candidate]
        joined.append({
            "launch_index": index,
            "kernel_id": launch.get("kernel_id"),
            "kernel_name": sim_name,
            "join_status": status,
            "order_basis": "kernel_id_rank" if ordinal else "reported_launch_order",
            "ncu_row_index": candidate,
            "ncu_launch_id": target.get("ncu_launch_id"),
            "ncu_kernel_name": target.get("kernel_name"),
            "ncu_gpu_duration_us": (target.get("metrics", {}).get("gpu__time_duration.sum") or {}).get("value"),
        })
    return joined
