#!/usr/bin/env python3
"""Convert the native one-warp memory capture into a strict payload sidecar.

The sidecar carries known store bytes and load-return values with explicit
allocation/base/line/sector identity. Capture v2 additionally retains full
addresses and producer metadata. It is an input contract for future
value-bearing replay; it does not infer cache ownership, eviction, fences, or
host-visible flush completion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LINE_BYTES = 128
SECTOR_BYTES = 32


class PayloadRefusal(RuntimeError):
    pass


def _line(document: str, prefix: str) -> dict[str, Any]:
    for raw in document.splitlines():
        if raw.startswith(prefix):
            return json.loads(raw[len(prefix) :])
    raise PayloadRefusal(f"missing {prefix.strip()}")


def convert(path: Path) -> dict[str, Any]:
    document = path.read_text(encoding="utf-8")
    capture = _line(document, "CAPTURE ")
    result = _line(document, "RESULT ")
    if capture.get("overflow") != 0 or capture.get("memory_overflow") != 0:
        raise PayloadRefusal("capture overflow prevents payload admission")
    if result.get("cc") != "8.6" or len(result.get("inputs", [])) != 32:
        raise PayloadRefusal("only the complete SM86 one-warp fixture is admitted")
    input_base = int(result["input_address"])
    output_base = int(result["output_address"])
    capture_v2 = capture.get("schema") == "native_value_capture_v2"
    payloads: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    lanes = capture.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 32:
        raise PayloadRefusal("capture must contain exactly 32 lanes")
    for lane, entry in enumerate(lanes):
        memory = entry.get("memory")
        if entry.get("lane") != lane or not isinstance(memory, list):
            raise PayloadRefusal(f"lane {lane} identity/memory records invalid")
        expected_input = input_base + lane * 4
        expected_output = output_base + lane * 4
        for record in memory:
            address = int(record.get("address_u64", record.get("address")))
            kind = int(record["kind"])
            value = int(record["value"]) & 0xFFFF_FFFF
            if kind == 0:
                expected = expected_input if capture_v2 else expected_input & 0xFFFF_FFFF
                if address != expected:
                    raise PayloadRefusal(f"lane {lane} load address {address:#x} != expected {expected:#x}")
                observations.append({"lane": lane, "kind": "load_return", "address_u64": address,
                                     "address_u32": address & 0xFFFF_FFFF, "value_u32": value,
                                     "pc": record.get("pc"), "guard": record.get("guard"),
                                     "occurrence": record.get("occurrence")})
                continue
            if kind != 1:
                raise PayloadRefusal(f"lane {lane} has unknown memory kind {kind}")
            expected = expected_output if capture_v2 else expected_output & 0xFFFF_FFFF
            if address != expected:
                raise PayloadRefusal(f"lane {lane} store address {address:#x} != expected {expected:#x}")
            byte_offset = address % SECTOR_BYTES
            if byte_offset + 4 > SECTOR_BYTES:
                raise PayloadRefusal("store crosses a sector boundary")
            line_id = address // LINE_BYTES
            sector = (address % LINE_BYTES) // SECTOR_BYTES
            bytes_le = value.to_bytes(4, "little")
            known_mask = ((1 << 4) - 1) << byte_offset
            payloads.append({
                "allocation_id": 2,
                "allocation": "output",
                "generation": 1,
                "line_id": line_id,
                "sector": sector,
                "byte_offset": byte_offset,
                "bytes_hex": bytes_le.hex(),
                "known_mask": known_mask,
                "address_u64": address,
                "address_u32": address & 0xFFFF_FFFF,
                "lane": lane,
                "pc": record.get("pc"),
                "guard": record.get("guard"),
                "occurrence": record.get("occurrence", len(payloads)),
                "launch_id": capture.get("launch_id", 0),
                "function_name": capture.get("function_name", "value_kernel"),
                "function_unique_id": capture.get("function_unique_id"),
                "cta": [0, 0, 0],
                "warp": 0,
                "sequence": len(payloads) + 1,
            })
    if len(observations) != 32 or len(payloads) != 32:
        raise PayloadRefusal(f"expected 32 loads and 32 stores, got {len(observations)} and {len(payloads)}")
    return {
        "schema": "native_value_payload_sidecar_v2" if capture_v2 else "native_value_payload_sidecar_v1",
        "status": "ok",
        "scoring": False,
        "device": {key: result[key] for key in ("name", "pci", "cc", "driver_api", "uuid_hex")},
        "contract": {
            "sector_bytes": SECTOR_BYTES,
            "line_bytes": LINE_BYTES,
            "payloads_known": "store bytes only",
            "initial_memory_known": False,
            "coherence_ownership": False,
            "flush_completion": False,
            "address_width_bits": 64 if capture_v2 else 32,
            "join_key": ["launch_id", "function_name", "cta", "warp", "lane", "pc", "occurrence"] if capture_v2 else None,
        },
        "baseline": {
            "input_address_u64": input_base,
            "output_address_u64": output_base,
            "input_words_u32": result.get("inputs"),
            "output_initial_word_u32": 0xA5A5A5A5,
            "output_initial_bytes_known": True,
        },
        "observations": observations,
        "payloads": payloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = convert(args.capture)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, PayloadRefusal) as exc:
        print(json.dumps({"schema": "native_value_payload_sidecar_v2", "status": "refused", "detail": str(exc)}, sort_keys=True))
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
