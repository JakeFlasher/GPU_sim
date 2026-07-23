// Compiled-helper enforcement library (R46 addition; the FOURTH
// and final byte-contract C mirror; closes Codex's recurring queued
// #2 across all 4 surfaces — FP8 E4M3 [R43], FP4 E2M1 [R44],
// FP6 E2M3 + FP6 E3M2 [R46]).
//
// This file is an instance of the now-codified pattern in
// `BL-20260427-compiled-helper-enforcement-template-for-byte-contract-
// rows` (codified in this same R46 after the 4-of-4 surfaces
// reached completion). The BL is the canonical reference for the
// 5-component template: helpers_lib.cu + Makefile target + ctypes
// test + module fixture + pre-submission mutation verification.
//
// This file exports an `extern "C"` version of the canonical Round-6
// FP6 E3M2 byte-contract function so a Python test can load this
// library via `ctypes.CDLL` and verify that the COMPILED C-side
// codes match the canonical Python helper byte-for-byte.
//
// Why a separate file instead of exposing main.cu's `_quantize_fp6_e3m2_code`:
//   - main.cu's function is `static` (translation-unit-local) for the
//     kernel runner binary's needs.
//   - This file duplicates the function with `extern "C"` linkage so it
//     can be loaded as a shared library.
//   - The R33-34 static-source-analysis guards
//     (`tests/test_fp6_e3m2_byte_contract_static_guard.py`)
//     enforce main.cu's correctness; the R46 compiled-helper test
//     enforces THIS file's correctness via runtime code comparison.
//
// FP6 E3M2 vs FP6 E2M3 (the sibling row at l1/fp6_e2m3_saturation):
// identical algorithm + same code layout (sign bit at position 5,
// magnitude in low 5 bits, no NaN/Inf encoding, -0 canonicalizes to
// 0x00, banker's rounding). The ONLY difference is the value table:
//   - E2M3 (sibling): 3 mantissa bits, exponent bias=1, max=+7.5,
//     8 denormals + 8 each for exp=1/2/3.
//   - E3M2 (this file): 2 mantissa bits, exponent bias=3, max=+28.0,
//     4 denormals + 4 each for exp=1..7.
//
// FP6 E3M2 vs FP8 E4M3 / FP4 E2M1 contract differences:
//   - 6-bit code (0x00..0x3F) in the low 6 bits of a `uint8_t`.
//   - FP6 has NO NaN encoding: FP32 NaN → code 0x00.
//   - FP6 has NO Inf encoding: FP32 ±Inf → code 0x1F / 0x3F (saturate ±28.0).
//   - -0.0 canonicalizes to code 0x00 (NOT 0x20). Same as FP4.
//
// Build via the row's Makefile target `helpers_lib`:
//   make -C l1/fp6_e3m2_mma_dep_chain helpers_lib
// Produces `helpers_lib.so` in the row directory.

#include <math.h>
#include <stdint.h>

// FP6 E3M2 representable positive magnitudes (mirrors Python module's
// `references/_low_precision_quant.py::FP6_E3M2_POSITIVE_VALUES` and
// main.cu's `FP6_E3M2_POSITIVE_VALUES`). 32 entries: 4 denormals
// (exp=0) + 4 each for exp=1..7. Bias = 3.
static const float FP6_E3M2_POSITIVE_VALUES[32] = {
    // exp=0 (denormal): m / 16
    0.0f,    0.0625f, 0.125f,  0.1875f,
    // exp=1: (1 + m/4) * 2^(-2)
    0.25f,   0.3125f, 0.375f,  0.4375f,
    // exp=2: (1 + m/4) * 2^(-1)
    0.5f,    0.625f,  0.75f,   0.875f,
    // exp=3: (1 + m/4) * 2^0
    1.0f,    1.25f,   1.5f,    1.75f,
    // exp=4: (1 + m/4) * 2
    2.0f,    2.5f,    3.0f,    3.5f,
    // exp=5: (1 + m/4) * 4
    4.0f,    5.0f,    6.0f,    7.0f,
    // exp=6: (1 + m/4) * 8
    8.0f,    10.0f,   12.0f,   14.0f,
    // exp=7: (1 + m/4) * 16
    16.0f,   20.0f,   24.0f,   28.0f,
};

// Canonical Round-6 FP6 E3M2 quantizer (must match Python
// `references/_low_precision_quant.py::_quantize_one_fp6_e3m2` AND
// main.cu's `_quantize_fp6_e3m2_code` byte-for-byte).
//
// Algorithm:
//   - FP32 NaN → code 0x00 (FP6 has no NaN encoding).
//   - ±Inf → code 0x1F / 0x3F (saturate to ±28.0).
//   - |v| >= +28.0 → saturate to ±28.0.
//   - Otherwise: linear search nearest-representable + banker's tie-break.
//   - -0.0 + idx==0 → canonicalize to code 0x00.
//
// `extern "C"` linkage so the function is loadable via Python ctypes
// for the R46 compiled-helper enforcement test.
extern "C" uint8_t simbench_quantize_fp6_e3m2_code(float value) {
    if (isnan(value)) return 0x00;
    int sign_bit = (signbit(value) || value < 0.0f) ? 1 : 0;
    float abs_v = fabsf(value);
    int idx;
    if (isinf(abs_v)) {
        idx = 31;  // ±28.0 (max representable)
    } else if (abs_v - FP6_E3M2_POSITIVE_VALUES[31] >= 0.0f) {
        idx = 31;
    } else {
        float best_dist = INFINITY;
        int best_idx = 0;
        for (int i = 0; i < 32; ++i) {
            float d = fabsf(abs_v - FP6_E3M2_POSITIVE_VALUES[i]);
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
            } else if (d == best_dist) {
                if ((i & 0x1) == 0) best_idx = i;
            }
        }
        idx = best_idx;
    }
    if (sign_bit && idx == 0) {
        return 0x00;
    }
    return (uint8_t)((sign_bit << 5) | (idx & 0x1F));
}
