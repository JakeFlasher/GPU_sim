"""Correctness equivalence checker for perturbation binaries.

Implements Section 4 of the task14 analysis artifact: builds the
canonical `correctness_delta` object per `comparison_mode`. The R30
patch fixed the per-mode `oneOf` shape across Sections 4, 6, 9; this
module emits records that conform to the schema's per-mode required
arrays.

Three modes:
- `abs_and_rel`: FP dtypes (`fp32`, `fp16`, `bf16`); element fails when
  `abs(actual - reference) > abs_tol + rel_tol * abs(reference)`.
- `exact_match`: INT dtypes (`int8`, `int32`); element fails when
  `actual != reference` (encoded as `max_abs_delta != 0`).
- `sha256_f32le`: digest comparison for large outputs; fails when
  `actual_sha256 != reference_sha256`. The digest input is the
  little-endian IEEE-754 FP32 byte stream of the logical output tensor
  in row-major order.

Pinned per-dtype tolerances (Section 4 table):
- fp32 abs=1e-5 rel=1e-4
- fp16 abs=1e-3 rel=1e-2
- bf16 abs=1e-2 rel=5e-2
- int8 / int32 exact-match (abs=0 implicit)
"""
from __future__ import annotations

import hashlib
import struct
from typing import Any, Iterable, Sequence

DTYPE_TOLERANCES: dict[str, dict[str, Any]] = {
    "fp32": {"abs_tol": 1e-5, "rel_tol": 1e-4, "comparison_mode": "abs_and_rel"},
    "fp16": {"abs_tol": 1e-3, "rel_tol": 1e-2, "comparison_mode": "abs_and_rel"},
    "bf16": {"abs_tol": 1e-2, "rel_tol": 5e-2, "comparison_mode": "abs_and_rel"},
    "int8": {"abs_tol": 0, "comparison_mode": "exact_match"},
    "int32": {"abs_tol": 0, "comparison_mode": "exact_match"},
}


def flatten(nested: Any) -> list[float]:
    """Flatten a nested JSON list into a flat list. The simbench
    output is `[warps][accesses][stride_classes]` of float; this
    helper produces the row-major flat sequence the per-element
    comparison walks."""
    out: list[float] = []
    if isinstance(nested, (list, tuple)):
        for item in nested:
            out.extend(flatten(item))
    else:
        out.append(float(nested))
    return out


def compute_abs_and_rel_delta(
    actual: Sequence[float], reference: Sequence[float], *,
    dtype: str,
    reference_path: str, actual_output_path: str,
) -> dict[str, Any]:
    """Compute the canonical `correctness_delta` for `abs_and_rel`
    mode. Walks both flat sequences in lockstep; the first element
    exceeding tolerance pins `mismatch_index` and the
    `first_mismatch_*` values; max-abs and max-rel deltas track the
    largest seen across the whole sequence.
    """
    tols = DTYPE_TOLERANCES[dtype]
    abs_tol = tols["abs_tol"]
    rel_tol = tols["rel_tol"]
    if len(actual) != len(reference):
        raise ValueError(
            f"correctness_check: actual length {len(actual)} != "
            f"reference length {len(reference)}"
        )
    max_abs_delta = 0.0
    max_rel_delta = 0.0
    mismatch_index = 0
    first_mismatch_actual = float(actual[0]) if actual else 0.0
    first_mismatch_ref = float(reference[0]) if reference else 0.0
    found_mismatch = False
    for i, (a, r) in enumerate(zip(actual, reference)):
        diff = abs(a - r)
        if diff > max_abs_delta:
            max_abs_delta = diff
        if abs(r) > 0:
            rel = diff / abs(r)
            if rel > max_rel_delta:
                max_rel_delta = rel
        if not found_mismatch and diff > abs_tol + rel_tol * abs(r):
            mismatch_index = i
            first_mismatch_actual = float(a)
            first_mismatch_ref = float(r)
            found_mismatch = True
    return {
        "dtype": dtype,
        "comparison_mode": "abs_and_rel",
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "max_abs_delta": float(max_abs_delta),
        "max_rel_delta": float(max_rel_delta),
        "mismatch_index": mismatch_index,
        "first_mismatch_actual_value": first_mismatch_actual,
        "first_mismatch_reference_value": first_mismatch_ref,
        "element_count": len(actual),
        "reference_path": reference_path,
        "actual_output_path": actual_output_path,
    }


def compute_exact_match_delta(
    actual: Sequence[int], reference: Sequence[int], *,
    dtype: str,
    reference_path: str, actual_output_path: str,
) -> dict[str, Any]:
    """Compute the canonical `correctness_delta` for `exact_match`
    mode (INT dtypes). `max_abs_delta != 0` implies failure; the
    schema's `exact_match` arm requires `abs_tol == 0`.
    """
    if dtype not in {"int8", "int32"}:
        raise ValueError(
            f"correctness_check: exact_match mode is for INT dtypes, "
            f"got {dtype!r}"
        )
    if len(actual) != len(reference):
        raise ValueError(
            f"correctness_check: actual length {len(actual)} != "
            f"reference length {len(reference)}"
        )
    max_abs_delta = 0
    mismatch_index = 0
    first_mismatch_actual = int(actual[0]) if actual else 0
    first_mismatch_ref = int(reference[0]) if reference else 0
    found_mismatch = False
    for i, (a, r) in enumerate(zip(actual, reference)):
        diff = abs(int(a) - int(r))
        if diff > max_abs_delta:
            max_abs_delta = diff
        if not found_mismatch and diff != 0:
            mismatch_index = i
            first_mismatch_actual = int(a)
            first_mismatch_ref = int(r)
            found_mismatch = True
    return {
        "dtype": dtype,
        "comparison_mode": "exact_match",
        "abs_tol": 0,
        "max_abs_delta": int(max_abs_delta),
        "mismatch_index": mismatch_index,
        "first_mismatch_actual_value": first_mismatch_actual,
        "first_mismatch_reference_value": first_mismatch_ref,
        "element_count": len(actual),
        "reference_path": reference_path,
        "actual_output_path": actual_output_path,
    }


def fp32_le_digest(values: Iterable[float]) -> str:
    """sha256 over the little-endian IEEE-754 FP32 byte stream of the
    given sequence, in row-major order. Mirrors the digest input
    format Section 4 pins for `sha256_f32le` mode."""
    h = hashlib.sha256()
    for v in values:
        h.update(struct.pack("<f", float(v)))
    return h.hexdigest()


def compute_sha256_f32le_delta(
    actual: Sequence[float], reference: Sequence[float], *,
    dtype: str,
    reference_path: str, actual_output_path: str,
) -> dict[str, Any]:
    """Compute the canonical `correctness_delta` for `sha256_f32le`
    mode. Both sequences are digested via the same byte layout; the
    record carries both digests so the verifier can recompute the
    pass/fail decision from the digests alone (no per-element
    walk required).
    """
    if dtype not in {"fp32", "fp16", "bf16"}:
        raise ValueError(
            f"correctness_check: sha256_f32le mode is for FP dtypes, "
            f"got {dtype!r}"
        )
    if len(actual) != len(reference):
        raise ValueError(
            f"correctness_check: actual length {len(actual)} != "
            f"reference length {len(reference)}"
        )
    tols = DTYPE_TOLERANCES[dtype]
    return {
        "dtype": dtype,
        "comparison_mode": "sha256_f32le",
        "abs_tol": tols["abs_tol"],
        "rel_tol": tols["rel_tol"],
        "actual_sha256": fp32_le_digest(actual),
        "reference_sha256": fp32_le_digest(reference),
        "element_count": len(actual),
        "reference_path": reference_path,
        "actual_output_path": actual_output_path,
    }


def delta_exceeds_tolerance(delta: dict[str, Any]) -> bool:
    """Per-mode dispatch over `delta["comparison_mode"]` mirroring the
    Section 9 `_correctness_delta_exceeds_tolerance` helper. Returns
    True iff the delta crosses tolerance under the active mode."""
    mode = delta.get("comparison_mode")
    if mode == "abs_and_rel":
        max_abs = float(delta["max_abs_delta"])
        abs_tol = float(delta["abs_tol"])
        rel_tol = float(delta["rel_tol"])
        ref = float(delta["first_mismatch_reference_value"])
        return max_abs > abs_tol + rel_tol * abs(ref)
    if mode == "exact_match":
        return float(delta["max_abs_delta"]) != 0
    if mode == "sha256_f32le":
        return delta["actual_sha256"] != delta["reference_sha256"]
    raise ValueError(
        f"correctness_check: unknown comparison_mode {mode!r}"
    )


__all__ = [
    "DTYPE_TOLERANCES",
    "flatten",
    "compute_abs_and_rel_delta",
    "compute_exact_match_delta",
    "fp32_le_digest",
    "compute_sha256_f32le_delta",
    "delta_exceeds_tolerance",
]
