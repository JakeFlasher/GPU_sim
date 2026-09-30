#!/usr/bin/env python3
"""Summarize SM86 PMA sample density after transport-packet reassembly.

The format follows tinygrad/extra/nv_pma/decode.py, not an NVIDIA wire-format
specification. Sample density is not dynamic instruction or physical route
coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time


def decode_ampere_samples(blob: bytes) -> tuple[list[int], dict]:
    # Each 32-byte transport packet has a four-byte header at offset 4.
    # Sample payloads continue across packets, independently per TPC stream.
    if len(blob) % 32:
        raise ValueError(f"truncated Ampere PMA transport packet: {len(blob)} bytes")
    pending: dict[int, bytearray] = {}
    samples: list[int] = []
    drops = discarded = payload_bytes = 0
    for offset in range(0, len(blob), 32):
        packet = blob[offset:offset + 32]
        header = int.from_bytes(packet[4:8], "little")
        count = header & 31
        if count > 28:
            raise ValueError(f"PMA packet at byte {offset}: payload count {count} exceeds 28")
        tpc = ((header >> 8) & 255) | (((header >> 25) & 7) << 8)
        stream = pending.setdefault(tpc, bytearray())
        if header & (1 << 28):
            drops += 1
            discarded += len(stream)
            stream.clear()
        stream.extend((packet[:4] + packet[8:])[:count])
        payload_bytes += count
        while len(stream) >= 8:
            samples.append(int.from_bytes(stream[:8], "little"))
            del stream[:8]
    return samples, {
        "transport_packets": len(blob) // 32,
        "payload_bytes": payload_bytes,
        "drop_marked_packets": drops,
        "discarded_partial_bytes": discarded,
        "trailing_partial_bytes_by_tpc": {str(k): len(v) for k, v in pending.items() if v},
        "stream_ids": sorted(pending),
    }


def one(directory: Path, shader: Path | None = None, pma: Path | None = None) -> dict:
    shader = shader or directory / "nak_shader.bin"
    if pma is None:
        candidates = sorted(directory.glob("pma_*.bin"))
        if not candidates:
            raise RuntimeError(f"no PMA blob in {directory}")
        pma = candidates[0]
    blob = pma.read_bytes()
    shader_bytes = shader.read_bytes()
    code_bytes = len(shader_bytes) - 728
    if code_bytes <= 0 or code_bytes % 16:
        raise ValueError(f"{shader}: expected a 728-byte info header followed by SM86 instruction words")
    raw, framing = decode_ampere_samples(blob)
    records = len(raw)
    return {
        "directory": str(directory),
        "shader": str(shader),
        "pma": str(pma),
        "shader_bytes": len(shader_bytes),
        "code_bytes_after_info_header": code_bytes,
        "static_instruction_words": code_bytes // 16,
        "pma_records": records,
        "raw_8byte_chunks_including_transport": len(blob) // 8,
        "framing": framing,
        "records_per_static_instruction": records / max(code_bytes // 16, 1),
        "unique_raw_records": len(set(raw)),
        "unique_raw_record_fraction": len(set(raw)) / max(records, 1),
        "unassigned_high_bit_one_counts": {
            str(bit): sum((sample >> bit) & 1 for sample in raw)
            for bit in (56, 57, 58, 59, 60, 61, 63)
        },
    }


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    if args.shader and len(args.shader) != len(args.capture):
        raise ValueError("--shader count must match --capture count")
    if args.pma and len(args.pma) != len(args.capture):
        raise ValueError("--pma count must match --capture count")
    samples = [
        one(path, (args.shader[index] if args.shader else None),
            (args.pma[index] if args.pma else None))
        for index, path in enumerate(args.capture)
    ]
    densities = [item["records_per_static_instruction"] for item in samples]
    report = {
        "schema": "sm86_nak_pma_coverage_model_v2",
        "format_source": "tinygrad/extra/nv_pma/decode.py: Ampere 32-byte packets and 8-byte samples",
        "samples": samples,
        "coverage_summary": {
            "median_records_per_static_instruction": statistics.median(densities),
            "min_records_per_static_instruction": min(densities),
            "max_records_per_static_instruction": max(densities),
        },
        "runtime_seconds": time.perf_counter() - started,
        "contract": "coverage-density observation only; no physical pipe/slot/issue inference",
        "limitations": [
            "raw PMA record encoding is not interpreted as a physical route witness",
            "v1 counted transport headers/padding as samples; use v2 decoded counts",
            "reverse-engineered field names and constant bits do not prove absence of hidden fields",
            "decoded sample density is not dynamic instruction coverage",
            "sample density depends on body duration, PMA interval, warp population, and driver state",
            "three bodies do not establish universal sampling law",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--shader", type=Path, action="append")
    parser.add_argument("--pma", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"coverage_summary": report["coverage_summary"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
