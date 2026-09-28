#!/usr/bin/env python3
"""Parse the bounded NVBit ``record_reg_vals`` text experiment.

The upstream NVBit example prints register snapshots as human-readable lines.
This adapter makes that evidence machine-readable while preserving its limits:
the text contains CTA/warp/opcode identity but no function id, PC, destination
writeback state, memory return value, or fence/atomic result. Consequently the
default output is a value-capture artifact with ``identity.status=partial``;
an exact sequence-keyed dynamic sidecar can prove the function/PC join, but it
is still not native SASS regeneration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_REGISTER_LINES = 8
MAX_LANES = 32

HEADER = re.compile(
    r"^CTA (?P<x>-?\d+),(?P<y>-?\d+),(?P<z>-?\d+) - "
    r"warp (?P<warp>\d+) - (?P<opcode>.*):$"
)
VALUE = re.compile(r"Reg(?P<reg>\d+)_T(?P<lane>\d+): 0x(?P<value>[0-9a-fA-F]{8})")


class Refusal(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)


def _read(path: Path) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise Refusal("input_missing", f"cannot read {path}: {exc}") from exc
    if len(data) > MAX_INPUT_BYTES:
        raise Refusal("input_limit_exceeded", f"input exceeds {MAX_INPUT_BYTES} bytes")
    return data


def parse(
    text: str,
    static_index: dict[str, list[dict[str, int]]] | None = None,
    dynamic_index: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_values: dict[int, dict[int, int]] = {}

    def finish() -> None:
        nonlocal current, current_values
        if current is None:
            return
        if not current_values:
            raise Refusal("record_without_values", "register record has no value rows")
        operands = []
        for register, lanes in sorted(current_values.items()):
            if len(lanes) > MAX_LANES:
                raise Refusal("lane_limit_exceeded", f"register {register} has too many lanes")
            operands.append(
                {
                    "register_ordinal": register,
                    "lanes": [
                        {"lane": lane, "value_u32": value}
                        for lane, value in sorted(lanes.items())
                    ],
                }
            )
        current["operands"] = operands
        if dynamic_index is not None:
            identity = dynamic_index.get(current["sequence"])
            if identity is None:
                raise Refusal(
                    "dynamic_identity_missing",
                    f"capture sequence {current['sequence']} has no exact dynamic identity",
                )
            if identity["opcode"] != current["opcode"]:
                raise Refusal(
                    "dynamic_identity_opcode_mismatch",
                    f"sequence {current['sequence']} has capture opcode {current['opcode']!r}, identity opcode {identity['opcode']!r}",
                )
            current["dynamic_identity"] = identity
        if static_index is not None:
            matches = static_index.get(current["opcode"], [])
            if len(matches) == 1:
                current["static_identity"] = {
                    "status": "unique",
                    **matches[0],
                }
            elif len(matches) > 1:
                current["static_identity"] = {
                    "status": "ambiguous",
                    "candidate_count": len(matches),
                }
            else:
                current["static_identity"] = {"status": "missing"}
        records.append(current)
        if len(records) > MAX_RECORDS:
            raise Refusal("record_limit_exceeded", f"more than {MAX_RECORDS} records")
        current = None
        current_values = {}

    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        header = HEADER.match(line)
        if header:
            finish()
            current = {
                "sequence": len(records),
                "cta": [
                    int(header.group("x")),
                    int(header.group("y")),
                    int(header.group("z")),
                ],
                "warp": int(header.group("warp")),
                "opcode": header.group("opcode"),
            }
            continue
        if not line.startswith("*"):
            raise Refusal("unexpected_line", f"line {line_number}: {line!r}")
        if current is None:
            raise Refusal("value_before_header", f"line {line_number} has no record header")
        values = list(VALUE.finditer(line))
        if not values:
            raise Refusal("malformed_value_row", f"line {line_number}: {line!r}")
        for value in values:
            register = int(value.group("reg"))
            lane = int(value.group("lane"))
            if register >= MAX_REGISTER_LINES:
                raise Refusal("register_limit_exceeded", f"register ordinal {register} is outside 0..7")
            if lane >= MAX_LANES:
                raise Refusal("lane_out_of_range", f"lane {lane} is outside 0..31")
            lanes = current_values.setdefault(register, {})
            if lane in lanes:
                raise Refusal("duplicate_lane_value", f"register {register} lane {lane} repeats")
            lanes[lane] = int(value.group("value"), 16)
    finish()
    if not records:
        raise Refusal("empty_capture", "capture contains no register records")
    joined = [
        record.get("static_identity", {}).get("status")
        for record in records
        if "static_identity" in record
    ]
    dynamic_joined = dynamic_index is not None
    return {
        "schema": "nvbit_value_trace_capture_v1" if dynamic_joined else "nvbit_register_value_capture_v1",
        "records": records,
        "identity": {
            "status": "dynamic_joined" if dynamic_joined else "partial",
            "dynamic_join": {
                "requested": dynamic_joined,
                "joined": len(records) if dynamic_joined else 0,
            },
            "static_join": {
                "requested": static_index is not None,
                "unique": joined.count("unique"),
                "ambiguous": joined.count("ambiguous"),
                "missing": joined.count("missing"),
            },
            "missing": [
                "function_unique_id",
                "pc",
                "destination_writeback_values",
                "memory_return_values",
                "atomic_results",
                "fence_scope_and_order",
            ],
        },
        "claims": {
            "value_capture": True,
            "value_dependent_dynamic_trace": dynamic_joined,
            "native_sass_regeneration": False,
            "timing": False,
        },
    }


def _load_static_index(path: Path) -> tuple[dict[str, list[dict[str, int]]], bytes]:
    data = _read(path)
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Refusal("static_index_invalid", str(exc)) from exc
    entries = raw if isinstance(raw, list) else raw.get("instructions") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise Refusal("static_index_invalid", "expected an instructions array")
    index: dict[str, list[dict[str, int]]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("opcode_text"), str):
            raise Refusal("static_index_invalid", "each entry needs opcode_text")
        if isinstance(entry.get("pc"), bool) or not isinstance(entry.get("pc"), int) or entry["pc"] < 0:
            raise Refusal("static_index_invalid", "each entry needs a nonnegative integer pc")
        identity = {"pc": entry["pc"]}
        if isinstance(entry.get("function_unique_id"), int):
            identity["function_unique_id"] = entry["function_unique_id"]
        index.setdefault(entry["opcode_text"], []).append(identity)
    return index, data


def _load_dynamic_index(path: Path) -> tuple[dict[int, dict[str, Any]], bytes]:
    data = _read(path)
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Refusal("dynamic_index_invalid", str(exc)) from exc
    entries = raw if isinstance(raw, list) else raw.get("records") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise Refusal("dynamic_index_invalid", "expected a non-empty records array")
    index: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise Refusal("dynamic_index_invalid", "each dynamic record must be an object")
        sequence = entry.get("sequence")
        function_id = entry.get("function_unique_id")
        pc = entry.get("pc")
        opcode = entry.get("opcode")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or sequence in index
            or isinstance(function_id, bool)
            or not isinstance(function_id, int)
            or function_id < 0
            or isinstance(pc, bool)
            or not isinstance(pc, int)
            or pc < 0
            or pc > 0xFFFF_FFFF
            or not isinstance(opcode, str)
            or not opcode.strip()
        ):
            raise Refusal("dynamic_index_invalid", "sequence/function_unique_id/pc/opcode are invalid or duplicated")
        index[sequence] = {
            "sequence": sequence,
            "function_unique_id": function_id,
            "pc": pc,
            "opcode": opcode.strip(),
        }
    return index, data


def adapt(
    path: Path,
    static_index_path: Path | None = None,
    dynamic_index_path: Path | None = None,
    bound: bool = False,
) -> dict[str, Any]:
    data = _read(path)
    static_index = None
    static_index_data = None
    dynamic_index = None
    dynamic_index_data = None
    if static_index_path is not None:
        static_index, static_index_data = _load_static_index(static_index_path)
    if dynamic_index_path is not None:
        dynamic_index, dynamic_index_data = _load_dynamic_index(dynamic_index_path)
    try:
        result = parse(data.decode("utf-8"), static_index, dynamic_index)
    except UnicodeDecodeError as exc:
        raise Refusal("utf8_invalid", str(exc)) from exc
    result["development_alpha"] = not bound
    result["input"] = {"path": str(path), "size_bytes": len(data)}
    if bound:
        result["input"]["sha256"] = hashlib.sha256(data).hexdigest()
    if static_index_path is not None and static_index_data is not None:
        result["input"]["static_index"] = {
            "path": str(static_index_path),
            "size_bytes": len(static_index_data),
        }
        if bound:
            result["input"]["static_index"]["sha256"] = hashlib.sha256(static_index_data).hexdigest()
    if dynamic_index_path is not None and dynamic_index_data is not None:
        result["input"]["dynamic_index"] = {
            "path": str(dynamic_index_path),
            "size_bytes": len(dynamic_index_data),
        }
        if bound:
            result["input"]["dynamic_index"]["sha256"] = hashlib.sha256(dynamic_index_data).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--static-index", type=Path)
    parser.add_argument("--dynamic-index", type=Path)
    parser.add_argument("--bound", action="store_true", help="opt into content-bound identities")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = adapt(args.input, args.static_index, args.dynamic_index, args.bound)
    except (Refusal, UnicodeDecodeError) as exc:
        code = exc.code if isinstance(exc, Refusal) else "utf8_invalid"
        detail = exc.detail if isinstance(exc, Refusal) else str(exc)
        print(json.dumps({"schema": "nvbit_value_trace_capture_v1", "status": "refused", "code": code, "detail": detail}, sort_keys=True))
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            print(json.dumps({"status": "refused", "code": "output_exists"}, sort_keys=True))
            return 2
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
