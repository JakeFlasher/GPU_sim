"""Shared helpers for SimBench-BW reference-family modules.

`compare_with_tolerance` is the canonical comparison used by every family's
`compare()`. It resolves a (dtype, reduction_kind, kernel_family) triplet
against tolerances.toml and applies a tolerance-aware allclose without
requiring NumPy at import time.

L4 checkpoint digest helpers (`sha256_bf16_flat`, `sha256_float32_flat`,
`sha256_int32_flat`, `fp32_to_bf16_bytes`) pack tensors into canonical
byte streams matching the schema-declared dtype (BF16 = high-16-bits
truncation of FP32; INT32 / FP32 = little-endian `struct` pack) and
return SHA-256 digests. L4 pipeline reference modules (R30+) use
these to emit `build_l4_checkpoint` output that validates against
`schemas/l4_checkpoints/<slug>.schema.json` without depending on NumPy.
"""

from __future__ import annotations

import hashlib
import math
import struct
import tomllib
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parent.parent


def _load_tolerances() -> dict[tuple[str, str, str], dict[str, Any]]:
    with open(SUITE_ROOT / "tolerances.toml", "rb") as f:
        data = tomllib.load(f)
    return {
        (e["dtype"], e["reduction_kind"], e["kernel_family"]): e
        for e in data["entries"]
    }


_TOLERANCE_CACHE: dict[tuple[str, str, str], dict[str, Any]] | None = None


def resolve_tolerance(dtype: str, reduction_kind: str, kernel_family: str) -> dict[str, Any]:
    global _TOLERANCE_CACHE
    if _TOLERANCE_CACHE is None:
        _TOLERANCE_CACHE = _load_tolerances()
    key = (dtype, reduction_kind, kernel_family)
    if key in _TOLERANCE_CACHE:
        return _TOLERANCE_CACHE[key]
    raise KeyError(f"tolerance_unmapped: no entry for {key}")


def _flatten(x: Any) -> list[float]:
    """Flatten an arbitrary nested list / tuple / scalar into a list of floats."""
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        out: list[float] = []
        for v in x:
            out.extend(_flatten(v))
        return out
    # NumPy / torch arrays expose tolist(); fall back to iteration
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return _flatten(tolist())
    raise TypeError(f"unsupported reference/actual type: {type(x).__name__}")


def compare_with_tolerance(
    actual: Any,
    reference: Any,
    tolerance: dict[str, Any] | None = None,
    *,
    dtype: str | None = None,
    reduction_kind: str | None = None,
    kernel_family: str | None = None,
) -> bool:
    """Tolerance-aware allclose. Either pass a tolerance dict directly, or the
    (dtype, reduction_kind, kernel_family) triplet to resolve from tolerances.toml.

    Returns True if every element pair satisfies
        |a - r| <= atol + rtol * |r|
    under the (rtol, atol) from the tolerance entry. Shape mismatches fail.
    """
    if tolerance is None:
        if dtype is None or reduction_kind is None or kernel_family is None:
            raise ValueError("compare_with_tolerance needs either tolerance or (dtype, reduction_kind, kernel_family)")
        tolerance = resolve_tolerance(dtype, reduction_kind, kernel_family)
    rtol = float(tolerance["rtol"])
    atol = float(tolerance["atol"])
    a = _flatten(actual)
    r = _flatten(reference)
    if len(a) != len(r):
        return False
    for av, rv in zip(a, r):
        if math.isnan(av) or math.isnan(rv):
            return math.isnan(av) and math.isnan(rv)
        if math.isinf(av) or math.isinf(rv):
            if av != rv:
                return False
            continue
        if abs(av - rv) > atol + rtol * abs(rv):
            return False
    return True


def deterministic_seeded_tensor(shape: tuple[int, ...], seed: int, low: float = -1.0, high: float = 1.0) -> list:
    """Produce a deterministic nested Python list with the given shape + seed.

    Uses a multiply-with-carry PRNG so the output is reproducible without
    depending on NumPy. Values lie in [low, high).
    """
    total = 1
    for d in shape:
        total *= d
    state = (seed * 2654435761 + 1) & 0xFFFFFFFF

    def next_u32() -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        return state

    flat = [low + (high - low) * (next_u32() / 2**32) for _ in range(total)]
    return _reshape(flat, list(shape))


def _reshape(flat: list, shape: list[int]) -> Any:
    if not shape:
        return flat[0]
    n = shape[0]
    rest = shape[1:]
    block = 1
    for d in rest:
        block *= d
    return [_reshape(flat[i * block : (i + 1) * block], rest) for i in range(n)]


# -----------------------------------------------------------------------------
# L4 checkpoint digest helpers (R30+; shared across pipeline families)
# -----------------------------------------------------------------------------


def fp32_to_bf16_bytes(v: float) -> bytes:
    """Return the canonical 2-byte BF16 representation of `v` in little-
    endian order. BF16 = the high 16 bits of the FP32 bit pattern
    (truncation, not rounding). A compliant kernel reproduces this byte
    pattern exactly by casting FP32 → BF16 via `__float2bfloat16` with
    round-to-zero / truncation semantics. struct.pack('<f', v) lays out
    the FP32 little-endian; the high 16 bits are the last two bytes."""
    return struct.pack("<f", float(v))[2:4]


def sha256_bf16_flat(flat: list[float]) -> str:
    """SHA-256 digest of a flat float list packed as canonical BF16 bytes."""
    h = hashlib.sha256()
    for v in flat:
        h.update(fp32_to_bf16_bytes(v))
    return h.hexdigest()


def sha256_float32_flat(flat: list[float]) -> str:
    """SHA-256 digest of a flat float list packed as little-endian FP32."""
    h = hashlib.sha256()
    for v in flat:
        h.update(struct.pack("<f", float(v)))
    return h.hexdigest()


def sha256_int32_flat(flat: list[int]) -> str:
    """SHA-256 digest of a flat integer list packed as little-endian INT32."""
    h = hashlib.sha256()
    for v in flat:
        h.update(struct.pack("<i", int(v)))
    return h.hexdigest()


def flatten_nested(x: Any) -> list[float]:
    """Flatten an arbitrarily-nested list/tuple into a flat list of floats.
    Shares logic with `_flatten` but exposes it publicly so L4 checkpoint
    builders can digest nested tensor shapes without re-implementing the
    flattening recursion."""
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        out: list[float] = []
        for v in x:
            out.extend(flatten_nested(v))
        return out
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return flatten_nested(tolist())
    raise TypeError(f"unsupported type in flatten_nested: {type(x).__name__}")
