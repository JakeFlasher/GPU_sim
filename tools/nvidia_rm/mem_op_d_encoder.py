#!/usr/bin/env python3
"""Encode the documented Ampere MEM_OP_A..D dirty-L2-flush method stream.

This utility intentionally remains a dry-run encoder. A separate, explicitly
authorized current-driver probe submitted the resulting words through an
RM-owned C56F channel; its source/output are retained under
``results/alpha/rm_mem_op_d_flush_dirty_3090ti_20260928/``. Keeping this
command non-submitting prevents an ordinary encoding invocation from changing
GPU state.
"""

from __future__ import annotations

import argparse
import json


METHODS = (0x28, 0x2C, 0x30, 0x34)
SEC_OP_INC_METHOD = 0x1
OP_L2_FLUSH_DIRTY = 0x10
DEFAULT_PUSHBUFFER_VA = 0x200506000


def method_header(offset: int, count: int = 1, subchannel: int = 0) -> int:
    if offset % 4 or not 0 <= offset // 4 <= 0xFFF:
        raise ValueError(f"invalid method offset: {offset:#x}")
    if not 1 <= count <= 0x1FFF:
        raise ValueError(f"invalid method count: {count}")
    if not 0 <= subchannel <= 7:
        raise ValueError(f"invalid subchannel: {subchannel}")
    return (
        (SEC_OP_INC_METHOD << 29)
        | (count << 16)
        | (subchannel << 13)
        | (offset // 4)
    )


def gpfifo_entry(pushbuffer_va: int, dword_count: int, sync: bool = False) -> int:
    """Pack an Ampere C56F GP entry using the published field positions."""
    if pushbuffer_va & 0x3:
        raise ValueError("pushbuffer VA must be 4-byte aligned")
    if not 1 <= dword_count <= 0x1FFFFF:
        raise ValueError("GP entry dword count is outside C56F LENGTH")
    # GP_ENTRY0.GET is bits 31:2; GET_HI is bits 7:0 of word 1.
    low = ((pushbuffer_va >> 2) & 0x3FFFFFFF) << 2
    high = ((pushbuffer_va >> 32) & 0xFF) | ((dword_count & 0x1FFFFF) << 10)
    if sync:
        high |= 1 << 31
    return low | (high << 32)


def encode() -> dict[str, object]:
    # All unrelated MEM_OP fields are explicitly zero as required by the
    # Ampere PBDMA reference; operation 0x10 occupies MEM_OP_D[31:27].
    words = []
    for index, offset in enumerate(METHODS):
        words.extend([method_header(offset), 0x80000000 if index == 3 else 0])
    entry = gpfifo_entry(DEFAULT_PUSHBUFFER_VA, len(words))
    grouped_words = [method_header(METHODS[0], len(METHODS)), 0, 0, 0, 0x80000000]
    return {
        "schema": "ampere_mem_op_d_flush_dirty_dry_run_v1",
        "channel_class_candidates": ["AMPERE_CHANNEL_GPFIFO_A_C56F", "AMPERE_CHANNEL_GPFIFO_B_C76F"],
        "methods": ["MEM_OP_A", "MEM_OP_B", "MEM_OP_C", "MEM_OP_D"],
        "method_offsets": [hex(x) for x in METHODS],
        "operation": "L2_FLUSH_DIRTY",
        "operation_value": OP_L2_FLUSH_DIRTY,
        "pushbuffer_dwords_hex": [f"0x{x:08x}" for x in words],
        "grouped_pushbuffer_dwords_hex": [f"0x{x:08x}" for x in grouped_words],
        "example_pushbuffer_va": hex(DEFAULT_PUSHBUFFER_VA),
        "example_gpfifo_entry_hex": f"0x{entry:016x}",
        "submitted": False,
        "limitations": [
            "dry-run encoding only",
            "requires RM-owned GPFIFO channel and driver kickoff",
            "no CUDA/CUPTI submission API",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    value = encode()
    text = json.dumps(value, indent=2) + "\n"
    if args.output:
        with open(args.output, "x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
