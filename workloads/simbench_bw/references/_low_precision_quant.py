"""Deterministic low-precision quantizers for SimBench-BW R3+ work.

R3 mainline objective: provide a single canonical FP32 → FP-of-N-bits
quantizer that the host driver (in C/CUDA) and the Python reference
family (`mma_microbench.py`) can both consume so they agree on the
exact numerical contract for the sm_120a-only Blackwell rows
(`fp4_e2m1_mma_dep_chain`, `fp6_e2m3_saturation`, `fp6_e3m2_mma_dep_chain`,
`sparse_2to4_fp8_gemm`).

The kind::f8f6f4 universal-fragment encoding stores each FP4/FP6/FP8
value in a single byte; the bit position WITHIN that byte depends on
the dtype, per PTX ISA
`references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md:18459-18467`:

  - `.e2m1` (FP4): central 4 bits [5:2]; padding zeros in [7:6] and [1:0].
  - `.e3m2` / `.e2m3` (FP6): lower 6 bits [5:0]; padding zeros in [7:6].
  - `.e4m3` / `.e5m2` (FP8): all 8 bits [7:0] used.

This matches the per-thread fragment shape (4 b32 = 16 bytes per
thread, 16 elements per thread distributed across m=16 rows ×
k={32,64} cols on the standard mma.kind::f8f6f4 path). The byte
contract is codified in `BL-20260425-kind-f8f6f4-byte-placement`
(introduced in R4 after R3 wrongly assumed "low 4 bits"
for e2m1; Codex's R3 review caught the mismatch).

Implemented dtypes: FP4 E2M1 (Rounds 3-4), FP6 E2M3 (R5), FP6
E3M2 (R6), FP8 E4M3 (R7). Each follows the same
POSITIVE_VALUES / _quantize_one / _dequantize_one /
quantize_to_packed_bytes structure (FP8 uses a full 256-entry value
table with NaN encodings handled instead of a positive-only table),
parameterized by the dtype's bit-width and byte position within the
8-bit container.
"""

from __future__ import annotations

import math


# -----------------------------------------------------------------------------
# FP4 E2M1
# -----------------------------------------------------------------------------
#
# E2M1 = 1 sign bit + 2 exponent bits + 1 mantissa bit, exponent bias = 1.
# 16 representable values (the `±0` pair is bit-distinct but numerically
# equal). Value table in unsigned-nibble bit-pattern order:
#
#   nibble | sign | exp | mant | value
#   -------+------+-----+------+--------
#   0x0    |  0   |  00 |  0   |  +0.0
#   0x1    |  0   |  00 |  1   |  +0.5  (denormal: (-1)^0 * 0.M * 2^(1-1) = 0.5)
#   0x2    |  0   |  01 |  0   |  +1.0
#   0x3    |  0   |  01 |  1   |  +1.5
#   0x4    |  0   |  10 |  0   |  +2.0
#   0x5    |  0   |  10 |  1   |  +3.0
#   0x6    |  0   |  11 |  0   |  +4.0
#   0x7    |  0   |  11 |  1   |  +6.0
#   0x8    |  1   |  00 |  0   |  -0.0
#   0x9    |  1   |  00 |  1   |  -0.5
#   0xA    |  1   |  01 |  0   |  -1.0
#   0xB    |  1   |  01 |  1   |  -1.5
#   0xC    |  1   |  10 |  0   |  -2.0
#   0xD    |  1   |  10 |  1   |  -3.0
#   0xE    |  1   |  11 |  0   |  -4.0
#   0xF    |  1   |  11 |  1   |  -6.0
#
# E2M1 has no NaN / Inf encoding. Out-of-range FP32 values clamp to the
# nearest signed maximum (+/- 6.0).

# Positive representable magnitudes in bit-pattern order (nibbles 0..7).
# The negative side is the same magnitudes with the sign bit set
# (nibbles 8..F).
FP4_E2M1_POSITIVE_VALUES: tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
)

FP4_E2M1_MAX = FP4_E2M1_POSITIVE_VALUES[-1]  # +6.0


def _quantize_one_fp4_e2m1(value: float) -> int:
    """Quantize a single FP32 value to its FP4 E2M1 nibble encoding.

    Algorithm: round-half-to-even toward the nearest representable
    magnitude; preserve sign; clamp to ±6.0; map ±NaN to +0 (E2M1 has
    no NaN encoding); map ±Inf to ±6.0 (saturation).

    Returns the 4-bit nibble in [0x0, 0xF].
    """
    if math.isnan(value):
        return 0x0
    sign_bit = 1 if (math.copysign(1.0, value) < 0.0 or value < 0.0) else 0
    abs_v = abs(float(value))
    if math.isinf(abs_v):
        idx = 7  # +/- 6.0 (max representable)
    else:
        # Saturating round-to-nearest-even toward an entry of FP4_E2M1_POSITIVE_VALUES.
        # The spacing is non-uniform, so a linear search is unavoidable.
        # Tie-breaking: round toward the value with the lower nibble bit
        # being even (banker's rounding); for E2M1 every adjacent pair
        # differs by a factor of <= 2, so ties are rare and the choice
        # affects only one half-step boundary per pair.
        idx = 7  # default to max if no closer
        best_dist = abs_v - FP4_E2M1_POSITIVE_VALUES[7]
        if best_dist < 0:  # the value is below +6.0; search for nearest
            best_dist = float("inf")
            best_idx = 0
            for i, candidate in enumerate(FP4_E2M1_POSITIVE_VALUES):
                d = abs(abs_v - candidate)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                elif d == best_dist:
                    # Banker's rounding: pick the candidate whose nibble
                    # has the LSB == 0 (i.e., the "even" of the two).
                    if (i & 0x1) == 0:
                        best_idx = i
            idx = best_idx
    if sign_bit and idx == 0:
        # Both +0 (0x0) and -0 (0x8) encode 0.0; prefer +0 to keep the
        # round-trip canonical. Nothing else uses nibble 0x0.
        return 0x0
    return (sign_bit << 3) | (idx & 0x7)


def _dequantize_one_fp4_e2m1(nibble: int) -> float:
    """Inverse of `_quantize_one_fp4_e2m1`: nibble in [0x0, 0xF] → FP32 value."""
    if not 0 <= nibble <= 0xF:
        raise ValueError(f"FP4 nibble must be in [0x0, 0xF], got {nibble:#x}")
    sign = -1.0 if (nibble & 0x8) else 1.0
    return sign * FP4_E2M1_POSITIVE_VALUES[nibble & 0x7]


def quantize_fp4_e2m1(values: list[float]) -> list[int]:
    """Quantize a flat list of FP32 values to FP4 E2M1 nibbles.

    Each output element is in [0x0, 0xF]. Length = `len(values)`.
    """
    return [_quantize_one_fp4_e2m1(v) for v in values]


def dequantize_fp4_e2m1(nibbles: list[int]) -> list[float]:
    """Inverse: convert FP4 E2M1 nibbles back to FP32 representable values."""
    return [_dequantize_one_fp4_e2m1(n) for n in nibbles]


def quantize_fp4_e2m1_to_packed_bytes(values: list[float]) -> bytes:
    """Quantize FP32 values and pack into the kind::f8f6f4 byte layout.

    PTX ISA reference (locally mirrored at
    `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md:18459-18467`,
    repeated at `:20141-20148` for `.kind::mxf8f6f4` — R37
    correction: the prior "sparse path" label was wrong; both citations
    are about the byte-packing rule for `.kind::f8f6f4` and
    `.kind::mxf8f6f4`, not sparse-mma-specific):

        "When `.kind` is either of `.kind::mxf8f6f4` or `.kind::f8f6f4`,
        the individual 4-bit and 6-bit floating point type elements
        must be packed in an 8-bit container. The matrix element of
        type `.e2m1` resides in central 4 bits of the 8-bit container
        with padding in the upper 2 bits and lower 2 bits of the
        container."

    So the byte layout is:

        bits [7:6]: padding (zero)
        bits [5:2]: the FP4 nibble (sign | exp[1:0] | mantissa)
        bits [1:0]: padding (zero)

    Encoding is `byte = (nibble & 0xF) << 2`. Decoding is
    `(byte >> 2) & 0xF`. The C-side host driver in
    `l1/fp4_e2m1_mma_dep_chain/main.cu` MUST produce bit-identical
    bytes — the byte-by-byte equality is the host-vs-Python contract
    for FP4 numerical agreement. The R4 lesson
    `BL-20260425-kind-f8f6f4-byte-placement` codifies this rule for
    all kind::f8f6f4 dtype variants (e2m1 → central 4 bits;
    e3m2/e2m3 → low 6 bits with upper 2 padding; e4m3/e5m2 → all 8 bits).
    """
    nibbles = quantize_fp4_e2m1(values)
    return bytes((n & 0xF) << 2 for n in nibbles)


def dequantize_fp4_e2m1_from_packed_bytes(packed: bytes | bytearray | memoryview) -> list[float]:
    """Inverse: read packed bytes (1 element per byte; central 4 bits
    [5:2] hold the FP4 nibble; bits [7:6] and [1:0] are padding zeros)
    and return FP32 representable values."""
    return [_dequantize_one_fp4_e2m1((b >> 2) & 0xF) for b in packed]


def quantize_fp4_e2m1_to_fp32_representable(values: list[float]) -> list[float]:
    """Convenience: quantize FP32 then immediately dequantize. The result
    is each input value rounded to the nearest FP4 E2M1 representable
    value (returned as FP32). Useful for the Python reference family
    when computing the FP32 matmul of the values the kernel actually
    sees on hardware."""
    return dequantize_fp4_e2m1(quantize_fp4_e2m1(values))


# -----------------------------------------------------------------------------
# FP6 E2M3 (R5)
# -----------------------------------------------------------------------------
#
# E2M3 = 1 sign bit + 2 exponent bits + 3 mantissa bits, exponent bias = 1.
# 64 representable values (the `±0` pair is bit-distinct but numerically
# equal). Decoding per IEEE-style rules:
#
#   - exp = 00 (denormal):  value = (-1)^s * (M / 8) * 2^(1-1)
#                                 = (-1)^s * M / 8       for M in 0..7
#                                 → 0/8, 1/8, 2/8, ..., 7/8
#   - exp = 01:             value = (-1)^s * (1 + M/8) * 2^(1-1)
#                                 = (-1)^s * (1 + M/8)
#                                 → 1.0, 1.125, 1.25, ..., 1.875
#   - exp = 10:             value = (-1)^s * (1 + M/8) * 2
#                                 → 2.0, 2.25, 2.5, ..., 3.75
#   - exp = 11:             value = (-1)^s * (1 + M/8) * 4
#                                 → 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5
#
# E2M3 has no NaN / Inf encoding. Out-of-range FP32 values clamp to the
# nearest signed maximum (±7.5).

# Positive representable magnitudes in bit-pattern order (codes
# 0x00..0x1F). Negative side is the same magnitudes with the sign bit
# set (codes 0x20..0x3F).
FP6_E2M3_POSITIVE_VALUES: tuple[float, ...] = tuple(
    # exp=0 (denormal): 8 values
    [m / 8.0 for m in range(8)]
    # exp=1: 8 values
    + [1.0 + m / 8.0 for m in range(8)]
    # exp=2: 8 values
    + [(1.0 + m / 8.0) * 2.0 for m in range(8)]
    # exp=3: 8 values
    + [(1.0 + m / 8.0) * 4.0 for m in range(8)]
)
assert len(FP6_E2M3_POSITIVE_VALUES) == 32

FP6_E2M3_MAX = FP6_E2M3_POSITIVE_VALUES[-1]  # +7.5


def _quantize_one_fp6_e2m3(value: float) -> int:
    """Quantize a single FP32 value to its FP6 E2M3 6-bit code.

    Algorithm: round-half-to-even toward the nearest representable
    magnitude; preserve sign; clamp to ±7.5; map ±NaN to +0 (E2M3 has
    no NaN encoding); map ±Inf to ±7.5 (saturation).

    Returns the 6-bit code in [0x00, 0x3F]: bit 5 is the sign, bits
    [4:0] are the magnitude index into `FP6_E2M3_POSITIVE_VALUES`.
    """
    if math.isnan(value):
        return 0x00
    sign_bit = 1 if (math.copysign(1.0, value) < 0.0 or value < 0.0) else 0
    abs_v = abs(float(value))
    if math.isinf(abs_v):
        idx = 31  # ±7.5 (max representable)
    else:
        # Saturating round-to-nearest-even toward an entry of FP6_E2M3_POSITIVE_VALUES.
        # The spacing is non-uniform across exponent bands, so a linear
        # search is unavoidable. Tie-breaking: round toward the
        # candidate whose code LSB is even (banker's rounding).
        idx = 31  # default to max if no closer
        best_dist = abs_v - FP6_E2M3_POSITIVE_VALUES[31]
        if best_dist < 0:  # the value is below +7.5; search for nearest
            best_dist = float("inf")
            best_idx = 0
            for i, candidate in enumerate(FP6_E2M3_POSITIVE_VALUES):
                d = abs(abs_v - candidate)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                elif d == best_dist:
                    if (i & 0x1) == 0:
                        best_idx = i
            idx = best_idx
    if sign_bit and idx == 0:
        # -0 collapses to code 0x00 (+0); keeps the round-trip canonical.
        return 0x00
    return (sign_bit << 5) | (idx & 0x1F)


def _dequantize_one_fp6_e2m3(code: int) -> float:
    """Inverse of `_quantize_one_fp6_e2m3`: code in [0x00, 0x3F] → FP32."""
    if not 0 <= code <= 0x3F:
        raise ValueError(f"FP6 E2M3 code must be in [0x00, 0x3F], got {code:#x}")
    sign = -1.0 if (code & 0x20) else 1.0
    return sign * FP6_E2M3_POSITIVE_VALUES[code & 0x1F]


def quantize_fp6_e2m3(values: list[float]) -> list[int]:
    """Quantize a flat list of FP32 values to FP6 E2M3 6-bit codes.

    Each output element is in [0x00, 0x3F]. Length = `len(values)`.
    """
    return [_quantize_one_fp6_e2m3(v) for v in values]


def dequantize_fp6_e2m3(codes: list[int]) -> list[float]:
    """Inverse: convert FP6 E2M3 codes back to FP32 representable values."""
    return [_dequantize_one_fp6_e2m3(c) for c in codes]


def quantize_fp6_e2m3_to_packed_bytes(values: list[float]) -> bytes:
    """Quantize FP32 values and pack into the kind::f8f6f4 byte layout.

    Per PTX ISA `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md
    :18459-18467` (the same byte-packing rule is repeated at
    `:20141-20148` for `.kind::mxf8f6f4`; R37 correction: the prior
    "sparse path mirror" label was wrong — both citations are byte-
    packing, not sparse-specific), for
    `.e3m2` and `.e2m3` "the matrix element resides in the lower 6
    bits of the 8-bit container with padding in the upper 2 bits."

    So the byte layout is:

        bits [7:6]: padding (zero)
        bits [5:0]: the FP6 code (sign | exp[1:0] | mantissa[2:0])

    Encoding is `byte = code & 0x3F`. Decoding is `byte & 0x3F`. The
    C-side host driver in `l1/fp6_e2m3_saturation/main.cu` MUST
    produce bit-identical bytes — the byte-by-byte equality is the
    host-vs-Python contract for FP6 numerical agreement.
    `BL-20260425-kind-f8f6f4-byte-placement` codifies this rule for
    all kind::f8f6f4 dtype variants.
    """
    codes = quantize_fp6_e2m3(values)
    return bytes(c & 0x3F for c in codes)


def dequantize_fp6_e2m3_from_packed_bytes(packed: bytes | bytearray | memoryview) -> list[float]:
    """Inverse: read packed bytes (1 element per byte; lower 6 bits
    [5:0] hold the FP6 code; bits [7:6] are padding zeros) and return
    FP32 representable values."""
    return [_dequantize_one_fp6_e2m3(b & 0x3F) for b in packed]


def quantize_fp6_e2m3_to_fp32_representable(values: list[float]) -> list[float]:
    """Convenience: quantize FP32 then immediately dequantize. The result
    is each input value rounded to the nearest FP6 E2M3 representable
    value (returned as FP32). Useful for the Python reference family
    when computing the FP32 matmul of the values the kernel actually
    sees on hardware."""
    return dequantize_fp6_e2m3(quantize_fp6_e2m3(values))


# -----------------------------------------------------------------------------
# FP6 E3M2 (R6)
# -----------------------------------------------------------------------------
#
# E3M2 = 1 sign bit + 3 exponent bits + 2 mantissa bits, exponent bias = 3.
# 64 representable values (the `±0` pair is bit-distinct but numerically
# equal). Decoding per IEEE-style rules:
#
#   - exp = 000 (denormal): value = (-1)^s * (M / 4) * 2^(1-3)
#                                 = (-1)^s * M / 16    for M in 0..3
#                                 → 0/16, 1/16, 2/16, 3/16
#   - exp = eee > 0:        value = (-1)^s * (1 + M/4) * 2^(eee-3)
#                                 → exp=1: 0.25, 0.3125, 0.375, 0.4375
#                                 → exp=2: 0.5, 0.625, 0.75, 0.875
#                                 → exp=3: 1.0, 1.25, 1.5, 1.75
#                                 → exp=4: 2.0, 2.5, 3.0, 3.5
#                                 → exp=5: 4.0, 5.0, 6.0, 7.0
#                                 → exp=6: 8.0, 10.0, 12.0, 14.0
#                                 → exp=7: 16.0, 20.0, 24.0, 28.0
#
# E3M2 has no NaN / Inf encoding. Out-of-range FP32 values clamp to the
# nearest signed maximum (±28.0).
#
# Byte layout (kind::f8f6f4 .e3m2): identical to .e2m3 — lower 6 bits
# [5:0] hold the code, padding mask 0xC0. Per PTX ISA :18459-18467.

# Positive representable magnitudes in bit-pattern order (codes
# 0x00..0x1F). Negative side is the same magnitudes with the sign bit
# set (codes 0x20..0x3F).
FP6_E3M2_POSITIVE_VALUES: tuple[float, ...] = tuple(
    # exp=0 (denormal): m / 16 for m in 0..3
    [m / 16.0 for m in range(4)]
    # exp=eee in 1..7 (normal): (1 + m/4) * 2^(eee - 3) for m in 0..3
    + [(1.0 + m / 4.0) * (2.0 ** (eee - 3))
       for eee in range(1, 8)
       for m in range(4)]
)
assert len(FP6_E3M2_POSITIVE_VALUES) == 32

FP6_E3M2_MAX = FP6_E3M2_POSITIVE_VALUES[-1]  # +28.0


def _quantize_one_fp6_e3m2(value: float) -> int:
    """Quantize a single FP32 value to its FP6 E3M2 6-bit code.

    Algorithm: round-half-to-even toward the nearest representable
    magnitude; preserve sign; clamp to ±28.0; map ±NaN to +0 (E3M2 has
    no NaN encoding); map ±Inf to ±28.0 (saturation).

    Returns the 6-bit code in [0x00, 0x3F]: bit 5 is the sign, bits
    [4:0] are the magnitude index into `FP6_E3M2_POSITIVE_VALUES`.
    """
    if math.isnan(value):
        return 0x00
    sign_bit = 1 if (math.copysign(1.0, value) < 0.0 or value < 0.0) else 0
    abs_v = abs(float(value))
    if math.isinf(abs_v):
        idx = 31  # ±28.0 (max representable)
    else:
        idx = 31  # default to max if no closer
        best_dist = abs_v - FP6_E3M2_POSITIVE_VALUES[31]
        if best_dist < 0:  # the value is below +28.0; search for nearest
            best_dist = float("inf")
            best_idx = 0
            for i, candidate in enumerate(FP6_E3M2_POSITIVE_VALUES):
                d = abs(abs_v - candidate)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                elif d == best_dist:
                    if (i & 0x1) == 0:
                        best_idx = i
            idx = best_idx
    if sign_bit and idx == 0:
        # -0 collapses to code 0x00 (+0); keeps the round-trip canonical.
        return 0x00
    return (sign_bit << 5) | (idx & 0x1F)


def _dequantize_one_fp6_e3m2(code: int) -> float:
    """Inverse of `_quantize_one_fp6_e3m2`: code in [0x00, 0x3F] → FP32."""
    if not 0 <= code <= 0x3F:
        raise ValueError(f"FP6 E3M2 code must be in [0x00, 0x3F], got {code:#x}")
    sign = -1.0 if (code & 0x20) else 1.0
    return sign * FP6_E3M2_POSITIVE_VALUES[code & 0x1F]


def quantize_fp6_e3m2(values: list[float]) -> list[int]:
    """Quantize a flat list of FP32 values to FP6 E3M2 6-bit codes.

    Each output element is in [0x00, 0x3F]. Length = `len(values)`.
    """
    return [_quantize_one_fp6_e3m2(v) for v in values]


def dequantize_fp6_e3m2(codes: list[int]) -> list[float]:
    """Inverse: convert FP6 E3M2 codes back to FP32 representable values."""
    return [_dequantize_one_fp6_e3m2(c) for c in codes]


def quantize_fp6_e3m2_to_packed_bytes(values: list[float]) -> bytes:
    """Quantize FP32 values and pack into the kind::f8f6f4 byte layout.

    Per PTX ISA `:18459-18467`, `.e3m2` shares the same byte layout as
    `.e2m3`: the matrix element resides in the lower 6 bits of the
    8-bit container with padding in the upper 2 bits. So:

        bits [7:6]: padding (zero)
        bits [5:0]: the FP6 code (sign | exp[2:0] | mantissa[1:0])

    Encoding `byte = code & 0x3F`. Decoding `byte & 0x3F`. The C-side
    host driver in `l1/fp6_e3m2_mma_dep_chain/main.cu` MUST produce
    bit-identical bytes. `BL-20260425-kind-f8f6f4-byte-placement`
    codifies this rule for all kind::f8f6f4 dtype variants.
    """
    codes = quantize_fp6_e3m2(values)
    return bytes(c & 0x3F for c in codes)


def dequantize_fp6_e3m2_from_packed_bytes(packed: bytes | bytearray | memoryview) -> list[float]:
    """Inverse: read packed bytes (1 element per byte; lower 6 bits
    [5:0] hold the FP6 code; bits [7:6] are padding zeros) and return
    FP32 representable values."""
    return [_dequantize_one_fp6_e3m2(b & 0x3F) for b in packed]


def quantize_fp6_e3m2_to_fp32_representable(values: list[float]) -> list[float]:
    """Convenience: quantize FP32 then immediately dequantize. The result
    is each input value rounded to the nearest FP6 E3M2 representable
    value (returned as FP32). Useful for the Python reference family
    when computing the FP32 matmul of the values the kernel actually
    sees on hardware."""
    return dequantize_fp6_e3m2(quantize_fp6_e3m2(values))


# -----------------------------------------------------------------------------
# FP8 E4M3 (R7)
# -----------------------------------------------------------------------------
#
# E4M3 = 1 sign bit + 4 exponent bits + 3 mantissa bits, exponent bias = 7.
# 256 codes. The OCP / NVIDIA E4M3 variant has special encodings:
#
#   - Bytes 0x7F and 0xFF (S.1111.111) → NaN (no Inf encoding).
#   - Max representable normal: byte 0x7E (= +448.0); 0xFE (= -448.0).
#   - Denormal (exp=0000): val = (-1)^s * (M / 8) * 2^(1-7)
#                              = (-1)^s * M / 512    for M in 0..7
#                              → 0/512, 1/512, ..., 7/512
#   - Normal (1 ≤ exp ≤ 15, excluding the NaN codes at exp=15, M=7):
#                              val = (-1)^s * (1 + M/8) * 2^(exp-7)
#                              → e.g. byte 0x38 = +1.0; byte 0x40 = +2.0;
#                                byte 0x78 = +256.0; byte 0x7E = +448.0.
#
# Byte layout (kind::f8f6f4 .e4m3): all 8 bits [7:0] used; no padding
# mask. Per PTX ISA :18459-18467, FP8 elements are stored directly in
# the 8-bit container. This is the SIMPLEST byte layout of the
# kind::f8f6f4 dtypes: encoding/decoding are identity on the byte.

# Saturation: out-of-range FP32 inputs clamp to ±448 (max normal).
# This matches the `satfinite` rounding mode that NVIDIA's MMA
# pipeline uses by default for E4M3.
FP8_E4M3_MAX = 448.0


def _build_fp8_e4m3_value_table() -> tuple[float, ...]:
    """Programmatically generate the 256-entry FP8 E4M3 value table.

    Index = byte; value = FP32 decoding of that byte. NaN bytes
    (0x7F, 0xFF) decode to `float('nan')`."""
    table: list[float] = []
    for byte in range(256):
        s = (byte >> 7) & 1
        e = (byte >> 3) & 0xF
        m = byte & 0x7
        if e == 15 and m == 7:
            table.append(float("nan"))
            continue
        sign = -1.0 if s else 1.0
        if e == 0:
            # Denormal: (M/8) * 2^(1-bias) = M/8 * 2^-6 = M/512
            value = sign * (m / 8.0) * (2.0 ** (1 - 7))
        else:
            # Normal: (1 + M/8) * 2^(e-bias)
            value = sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))
        table.append(value)
    return tuple(table)


_FP8_E4M3_VALUE_TABLE: tuple[float, ...] = _build_fp8_e4m3_value_table()
assert len(_FP8_E4M3_VALUE_TABLE) == 256
assert _FP8_E4M3_VALUE_TABLE[0x7E] == 448.0
assert _FP8_E4M3_VALUE_TABLE[0xFE] == -448.0
assert math.isnan(_FP8_E4M3_VALUE_TABLE[0x7F])
assert math.isnan(_FP8_E4M3_VALUE_TABLE[0xFF])

# Indices of representable (non-NaN) bytes for the encoder's nearest-search.
_FP8_E4M3_REPRESENTABLE_BYTES: tuple[int, ...] = tuple(
    b for b in range(256) if not math.isnan(_FP8_E4M3_VALUE_TABLE[b])
)
assert len(_FP8_E4M3_REPRESENTABLE_BYTES) == 254  # 256 - 2 NaN encodings


def _quantize_one_fp8_e4m3(value: float) -> int:
    """Quantize a single FP32 value to its FP8 E4M3 byte encoding.

    Algorithm: round-half-to-even toward the nearest representable
    magnitude; preserve sign (including for -0.0 → 0x80); clamp to
    ±448 (satfinite — clips ±Inf to max finite, since E4M3 has no
    Inf encoding); FP32 NaN → byte 0x7F (canonical positive E4M3
    NaN encoding).

    Special-value behavior matches `cuda_fp8.h::__nv_fp8_e4m3`
    byte-for-byte (Codex Round-29-review host-program probe; closed
    in R31): `__nv_fp8_e4m3(+0.0f) → 0x00`, `(-0.0f) → 0x80`,
    `(+nan/-nan/negative-payload-nan) → 0x7F`, `(+Inf) → 0x7E`,
    `(-Inf) → 0xFE`. Pre-Round-31 the helper canonicalized -0.0 →
    0x00 and NaN → 0x00 (a R7 design choice for portability);
    the seeded `[-0.5, 0.5)` correctness paths used by
    `fp8_e4m3_qmma_dep_chain`, `wgmma_fp8_saturation`, and
    `sparse_2to4_fp8_gemm` were unaffected (no -0.0 or NaN inputs
    in seed) but the byte-contract surface diverged from CUDA at
    edge-case inputs, blocking byte-level host-vs-Python equivalence.
    R31 aligns the helper using the same R26 template that
    closed the FP8 E5M2 sibling alignment.

    Returns the 8-bit byte in [0x00, 0xFF]. Outputs include 0x80 for
    -0.0 and 0x7F for any FP32 NaN (positive, negative, payload —
    all collapse to 0x7F per the Round-29-review host probe); the
    encoder excludes only 0xFF (negative NaN encoding) since the
    encoder collapses NaN sign upstream.
    """
    if math.isnan(value):
        # Canonical positive E4M3 NaN encoding. Per Codex Round-29-
        # review CUDA 13 host-program probe: __nv_fp8_e4m3(+nan),
        # __nv_fp8_e4m3(-nan), and a negative payload NaN ALL store
        # 0x7F. There's no negative-NaN cast behavior to mirror —
        # CUDA collapses NaN sign upstream of the late `res |= sign`
        # OR in cuda_fp8.hpp. Source-reading would naively predict
        # 0xFF for negative NaN; the host probe rules that out.
        # Tests in test_low_precision_quant.py lock this in across
        # NaN variants (positive, negative, negative payload) per
        # the R27 E5M2 pattern adapted to E4M3 in R31.
        return 0x7F
    sign_bit = 1 if (math.copysign(1.0, value) < 0.0 or value < 0.0) else 0
    abs_v = abs(float(value))
    if math.isinf(abs_v) or abs_v >= FP8_E4M3_MAX:
        # Saturate to ±448 (max normal). Byte: sign | exp=15 | mant=6.
        return 0xFE if sign_bit else 0x7E
    # Search the positive half of the value table (bytes 0x00..0x7E,
    # which exclude the +NaN at 0x7F) for the nearest |v|. Then OR in
    # the sign bit. This handles ±0 cleanly (±0.0 both compare True
    # to `>= 0` in Python, which would break a same-sign filter on the
    # full table). The general path produces -0.0 → 0x80 because the
    # nearest positive byte is 0x00 and the sign bit OR's in directly.
    best_idx = 0
    best_dist = float("inf")
    for b in range(0x7F):  # 0x00..0x7E (127 positive bytes including +0)
        cand = _FP8_E4M3_VALUE_TABLE[b]
        d = abs(abs_v - cand)
        if d < best_dist:
            best_dist = d
            best_idx = b
        elif d == best_dist:
            # Banker's rounding: pick the candidate whose byte LSB is even.
            if (b & 0x1) == 0:
                best_idx = b
    return best_idx | (sign_bit << 7)


def _dequantize_one_fp8_e4m3(byte: int) -> float:
    """Inverse of `_quantize_one_fp8_e4m3`: byte in [0x00, 0xFF] → FP32.

    Returns `float('nan')` for the two NaN encodings (0x7F, 0xFF)."""
    if not 0 <= byte <= 0xFF:
        raise ValueError(f"FP8 E4M3 byte must be in [0x00, 0xFF], got {byte:#x}")
    return _FP8_E4M3_VALUE_TABLE[byte]


def quantize_fp8_e4m3(values: list[float]) -> list[int]:
    """Quantize a flat list of FP32 values to FP8 E4M3 bytes.

    Each output element is in [0x00, 0xFF]. Outputs may include 0x7F
    (any FP32 NaN input encodes as the canonical positive E4M3 NaN
    byte 0x7F per R31 alignment with `cuda_fp8.h::__nv_fp8_e4m3`)
    and 0x80 (-0.0 inputs encode as sign-preserved zero). Outputs
    exclude 0xFF (the negative NaN encoding) since the encoder
    collapses NaN sign upstream. Length = `len(values)`.
    """
    return [_quantize_one_fp8_e4m3(v) for v in values]


def dequantize_fp8_e4m3(bytes_in: list[int]) -> list[float]:
    """Inverse: convert FP8 E4M3 bytes back to FP32 representable values.

    NaN bytes (0x7F, 0xFF) decode to `float('nan')`."""
    return [_dequantize_one_fp8_e4m3(b) for b in bytes_in]


def quantize_fp8_e4m3_to_packed_bytes(values: list[float]) -> bytes:
    """Quantize FP32 values and pack into the kind::f8f6f4 byte layout.

    Per PTX ISA `:18459-18467`, `.e4m3` uses **all 8 bits** of the
    8-bit container with **no padding mask**. So the byte layout is
    simply the 8-bit FP8 code; encoding is `byte = code`. Decoding is
    `code = byte`. The C-side host driver in
    `l1/sparse_2to4_fp8_gemm/main.cu` MUST produce bit-identical
    bytes. `BL-20260425-kind-f8f6f4-byte-placement` codifies this
    rule for all kind::f8f6f4 dtype variants.
    """
    return bytes(quantize_fp8_e4m3(values))


def dequantize_fp8_e4m3_from_packed_bytes(packed: bytes | bytearray | memoryview) -> list[float]:
    """Inverse: read packed bytes (1 element per byte; all 8 bits used;
    no padding) and return FP32 representable values."""
    return [_dequantize_one_fp8_e4m3(b) for b in packed]


def quantize_fp8_e4m3_to_fp32_representable(values: list[float]) -> list[float]:
    """Convenience: quantize FP32 then immediately dequantize. The result
    is each input value rounded to the nearest FP8 E4M3 representable
    value (returned as FP32). Useful for the Python reference family
    when computing the FP32 matmul of the values the kernel actually
    sees on hardware.

    Special-value behavior (post-Round-31 alignment with
    `cuda_fp8.h::__nv_fp8_e4m3`): FP32 NaN inputs round-trip back to
    NaN (encoded as byte 0x7F, decoded as `float('nan')`); -0.0
    inputs round-trip back to -0.0 (encoded as byte 0x80, decoded as
    -0.0); ±Inf inputs satfinite-clamp to ±448 (encoded as bytes
    0x7E/0xFE, since E4M3 has no Inf encoding). The seeded
    `[-0.5, 0.5)` correctness paths used by FP8 E4M3 rows are
    unaffected (no -0.0 or NaN inputs in seed range)."""
    return dequantize_fp8_e4m3(quantize_fp8_e4m3(values))


# =============================================================================
# FP8 E5M2: IEEE-style binary8 with 1 sign + 5 exp + 2 mant bits, bias=15.
# R20 addition (per Codex Round-19-review blocking finding #2 — the
# fp8_e5m2_saturation row's host driver casts FP32 → FP8 E5M2 via
# `__nv_fp8_e5m2` before H2D, but mma_microbench.py had no E5M2 dispatch
# elif and _low_precision_quant.py had no E5M2 quantizer — same defect
# class as R13 sparse INT8 fix and R19 dense FP8 E4M3 fix).
# Structurally similar to FP8 E4M3 (R7) but with IEEE conventions:
# e=31 reserved for ±Inf (m=0) and NaN (m≠0), max finite at e=30 m=3.
# =============================================================================

# Saturation: out-of-range FP32 inputs and ±Inf clamp to ±57344 (max
# finite normal). Matches NVIDIA's `satfinite` rounding mode for E5M2
# (which is the default for the cuda_fp8.h `__nv_fp8_e5m2` cast).
FP8_E5M2_MAX = 57344.0  # (1 + 3/4) * 2^(30-15) = 1.75 * 32768


def _build_fp8_e5m2_value_table() -> tuple[float, ...]:
    """Programmatically generate the 256-entry FP8 E5M2 value table.

    Index = byte; value = FP32 decoding of that byte. Special encodings
    follow IEEE 754 binary8 conventions:
      - byte e=31 m=0 → ±Inf (0x7C = +Inf, 0xFC = -Inf)
      - byte e=31 m≠0 → NaN (0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF)
    """
    table: list[float] = []
    for byte in range(256):
        s = (byte >> 7) & 1
        e = (byte >> 2) & 0x1F
        m = byte & 0x3
        sign = -1.0 if s else 1.0
        if e == 31:
            if m == 0:
                table.append(sign * float("inf"))
            else:
                table.append(float("nan"))
            continue
        if e == 0:
            # Denormal: (M/4) * 2^(1-bias) = M/4 * 2^-14 = M/65536
            value = sign * (m / 4.0) * (2.0 ** (1 - 15))
        else:
            # Normal: (1 + M/4) * 2^(e-bias)
            value = sign * (1.0 + m / 4.0) * (2.0 ** (e - 15))
        table.append(value)
    return tuple(table)


_FP8_E5M2_VALUE_TABLE: tuple[float, ...] = _build_fp8_e5m2_value_table()
assert len(_FP8_E5M2_VALUE_TABLE) == 256
assert _FP8_E5M2_VALUE_TABLE[0x7B] == 57344.0  # max finite positive
assert _FP8_E5M2_VALUE_TABLE[0xFB] == -57344.0  # max finite negative
assert _FP8_E5M2_VALUE_TABLE[0x7C] == float("inf")
assert _FP8_E5M2_VALUE_TABLE[0xFC] == float("-inf")
assert math.isnan(_FP8_E5M2_VALUE_TABLE[0x7D])
assert math.isnan(_FP8_E5M2_VALUE_TABLE[0x7E])
assert math.isnan(_FP8_E5M2_VALUE_TABLE[0x7F])
assert math.isnan(_FP8_E5M2_VALUE_TABLE[0xFD])
assert math.isnan(_FP8_E5M2_VALUE_TABLE[0xFE])
assert math.isnan(_FP8_E5M2_VALUE_TABLE[0xFF])

# Indices of representable (finite, non-NaN, non-Inf) bytes for the
# encoder's nearest-search. Excludes 8 special encodings (2 Inf + 6 NaN).
_FP8_E5M2_REPRESENTABLE_BYTES: tuple[int, ...] = tuple(
    b for b in range(256)
    if not (math.isnan(_FP8_E5M2_VALUE_TABLE[b]) or math.isinf(_FP8_E5M2_VALUE_TABLE[b]))
)
assert len(_FP8_E5M2_REPRESENTABLE_BYTES) == 248  # 256 - 2 Inf - 6 NaN


def _quantize_one_fp8_e5m2(value: float) -> int:
    """Quantize a single FP32 value to its FP8 E5M2 byte encoding.

    Algorithm: round-half-to-even toward the nearest representable
    magnitude; preserve sign (including for -0.0 → 0x80); clamp to
    ±57344 (satfinite — clips ±Inf to max finite, NOT to the ±Inf
    encoding); FP32 NaN → byte 0x7F (canonical positive E5M2 NaN
    encoding).

    Special-value behavior matches `cuda_fp8.h::__nv_fp8_e5m2`
    byte-for-byte (Codex Round-20-through-Round-25 recurring queued
    finding, closed in R26): a local CUDA-host program audit
    confirmed `__nv_fp8_e5m2(-0.0f) → 0x80`, `__nv_fp8_e5m2(NaN) →
    0x7F`, and `__nv_fp8_e5m2(INFINITY) → 0x7B` (satfinite). Pre-
    Round-26 the helper canonicalized -0.0 → 0x00 and NaN → 0x00; the
    seeded `[-0.5, 0.5)` correctness path was unaffected (no -0.0 or
    NaN inputs in seed) but the byte-contract surface diverged from
    CUDA, blocking byte-level host-vs-Python equivalence at edge-case
    inputs. R26 aligns the helper.

    Returns the 8-bit byte in [0x00, 0xFF]. Outputs include 0x80 for
    -0.0 and 0x7F for NaN; outputs exclude Inf encodings (0x7C, 0xFC)
    and the other 5 NaN encodings (0x7D, 0x7E, 0xFD-0xFF) since the
    encoder only ever emits a single canonical NaN byte.
    """
    if math.isnan(value):
        # Canonical positive E5M2 NaN encoding. Per Codex Round-26-
        # review CUDA 13 host-program audit: __nv_fp8_e5m2(+nan),
        # __nv_fp8_e5m2(-nan), and a negative payload NaN ALL store
        # 0x7F. There's no negative-NaN cast behavior to mirror — CUDA
        # always picks 0x7F regardless of the FP32 NaN's sign bit or
        # mantissa payload. Tests in test_low_precision_quant.py lock
        # this in across NaN variants.
        return 0x7F
    sign_bit = 1 if (math.copysign(1.0, value) < 0.0 or value < 0.0) else 0
    abs_v = abs(float(value))
    if math.isinf(abs_v) or abs_v >= FP8_E5M2_MAX:
        # Saturate to ±57344 (max finite). Byte: sign | exp=30 | mant=3.
        return 0xFB if sign_bit else 0x7B
    # Search the positive finite half of the value table (bytes
    # 0x00..0x7B, which exclude +Inf at 0x7C and the 3 +NaN bytes at
    # 0x7D-0x7F) for the nearest |v|. Then OR in the sign bit. This
    # handles ±0 cleanly (±0.0 both compare True to `>= 0` in Python,
    # which would break a same-sign filter on the full table).
    best_idx = 0
    best_dist = float("inf")
    for b in range(0x7C):  # 0x00..0x7B (124 positive finite bytes including +0)
        cand = _FP8_E5M2_VALUE_TABLE[b]
        d = abs(abs_v - cand)
        if d < best_dist:
            best_dist = d
            best_idx = b
        elif d == best_dist:
            # Banker's rounding: pick the candidate whose byte LSB is even.
            if (b & 0x1) == 0:
                best_idx = b
    return best_idx | (sign_bit << 7)


def _dequantize_one_fp8_e5m2(byte: int) -> float:
    """Inverse of `_quantize_one_fp8_e5m2`: byte in [0x00, 0xFF] → FP32.

    Returns ±Inf for the Inf encodings (0x7C, 0xFC) and `float('nan')`
    for the 6 NaN encodings (0x7D-0x7F, 0xFD-0xFF)."""
    if not 0 <= byte <= 0xFF:
        raise ValueError(f"FP8 E5M2 byte must be in [0x00, 0xFF], got {byte:#x}")
    return _FP8_E5M2_VALUE_TABLE[byte]


def quantize_fp8_e5m2(values: list[float]) -> list[int]:
    """Quantize a flat list of FP32 values to FP8 E5M2 bytes.

    Each output element is in [0x00, 0xFF]. Outputs may include 0x7F
    (canonical positive NaN encoding emitted for any FP32 NaN input,
    regardless of sign or payload — matches `cuda_fp8.h::__nv_fp8_e5m2`
    per R26) and 0x80 (sign-preserving for -0.0). Outputs exclude
    the other 5 NaN encodings (0x7D, 0x7E, 0xFD-0xFF) and the 2 Inf
    encodings (0x7C, 0xFC) — the encoder satfinite-clamps ±Inf to
    ±57344 (0x7B/0xFB) and only ever emits the single canonical NaN
    byte. Length = `len(values)`.
    """
    return [_quantize_one_fp8_e5m2(v) for v in values]


def dequantize_fp8_e5m2(bytes_in: list[int]) -> list[float]:
    """Inverse: convert FP8 E5M2 bytes back to FP32 representable values.

    Inf bytes (0x7C, 0xFC) decode to ±Inf; NaN bytes (0x7D-0x7F,
    0xFD-0xFF) decode to `float('nan')`."""
    return [_dequantize_one_fp8_e5m2(b) for b in bytes_in]


def quantize_fp8_e5m2_to_packed_bytes(values: list[float]) -> bytes:
    """Quantize FP32 values and pack into the kind::f8f6f4 byte layout.

    Per PTX ISA `:18459-18467`, `.e5m2` uses **all 8 bits** of the
    8-bit container with **no padding mask** (same byte layout as
    `.e4m3`; only the value table differs). So encoding is
    `byte = code` and decoding is `code = byte`. Same byte-layout
    contract `BL-20260425-kind-f8f6f4-byte-placement` codifies for
    `.e4m3`.
    """
    return bytes(quantize_fp8_e5m2(values))


def dequantize_fp8_e5m2_from_packed_bytes(packed: bytes | bytearray | memoryview) -> list[float]:
    """Inverse: read packed bytes (1 element per byte; all 8 bits used;
    no padding) and return FP32 representable values."""
    return [_dequantize_one_fp8_e5m2(b) for b in packed]


def quantize_fp8_e5m2_to_fp32_representable(values: list[float]) -> list[float]:
    """Convenience: quantize FP32 then immediately dequantize. The result
    is each input value rounded to the nearest FP8 E5M2 representable
    value (returned as FP32). NaN inputs encode as byte 0x7F and decode
    back to `float('nan')` (NaN-ness preserved through the round-trip
    per R26 alignment with `cuda_fp8.h::__nv_fp8_e5m2`); -0.0
    inputs encode as byte 0x80 and decode back to -0.0 (sign preserved);
    ±Inf inputs satfinite-clamp to ±57344."""
    return dequantize_fp8_e5m2(quantize_fp8_e5m2(values))


# -----------------------------------------------------------------------------
# All five low-precision dtypes used by SimBench-BW are now implemented:
# FP4 E2M1 (Rounds 3-4), FP6 E2M3 (R5), FP6 E3M2 (R6),
# FP8 E4M3 (R7), FP8 E5M2 (R20). Future rounds extending the
# suite to other low-precision paths (e.g. NVFP4 block-scaled) will
# follow the same POSITIVE_VALUES / _quantize_one / _dequantize_one /
# quantize_to_packed_bytes / host-independent contract-test pattern.
# -----------------------------------------------------------------------------
#
# Per-dtype byte placement within the 8-bit container per PTX ISA
# `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md:18459-18467`:
#
#   .e2m1 (FP4): central 4 bits [5:2]; padding mask 0xC3.
#   .e3m2 / .e2m3 (FP6): lower 6 bits [5:0]; padding mask 0xC0.
#   .e4m3 / .e5m2 (FP8): all 8 bits [7:0]; no padding mask.
#
# `BL-20260425-kind-f8f6f4-byte-placement` codifies this rule and
# enumerates the per-dtype host-independent contract-test pattern.
