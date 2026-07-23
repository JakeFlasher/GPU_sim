"""Independent NumPy reference for CUTLASS successor correctness receipts.

This module intentionally has no CUDA, CUTLASS, or witness-builder imports.  It
implements the fixed seed/layout contract used by the three bounded successor
runs so the receipt producer and the executable are separate implementations.
"""
from __future__ import annotations

import struct
from typing import Literal

import numpy as np


DTypeContract = Literal[
    "bf16_inputs_fp32_accumulator",
    "fp32_inputs_fp32_accumulator",
]


def _next_u32(state: int) -> int:
    return (state * 1103515245 + 12345) & 0xFFFFFFFF


def _sample_values(count: int, seed: int) -> np.ndarray:
    state = (seed * 2654435761 + 1) & 0xFFFFFFFF
    values = np.empty(count, dtype=np.float32)
    for index in range(count):
        state = _next_u32(state)
        values[index] = np.float32(-0.5 + state / 4294967296.0)
    return values


def _bf16_round(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", rounded & 0xFFFF0000))[0]


def reference_matrix(
    *,
    dtype_contract: DTypeContract,
    batch_count: int,
    m: int,
    n: int,
    k: int,
    seed: int,
) -> np.ndarray:
    """Return FP32 C tensors with row-major A/column-major B semantics."""
    if batch_count < 1 or min(m, n, k) < 1:
        raise ValueError("matrix dimensions and batch_count must be positive")
    a = _sample_values(batch_count * m * k, seed).reshape(batch_count, m, k)
    b_source = _sample_values(batch_count * k * n, seed + 1)
    b = np.empty((batch_count, k, n), dtype=np.float32)
    cursor = 0
    for batch in range(batch_count):
        for row in range(k):
            for column in range(n):
                value = float(b_source[cursor])
                cursor += 1
                if dtype_contract == "bf16_inputs_fp32_accumulator":
                    value = _bf16_round(value)
                b[batch, row, column] = np.float32(value)
    if dtype_contract == "bf16_inputs_fp32_accumulator":
        a = np.array(
            [_bf16_round(float(value)) for value in a.flat], dtype=np.float32
        ).reshape(a.shape)
    elif dtype_contract != "fp32_inputs_fp32_accumulator":
        raise ValueError(f"unknown dtype contract: {dtype_contract}")
    return np.matmul(a, b)
