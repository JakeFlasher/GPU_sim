// Compiled-helper enforcement library (R46 addition; extends the
// R43-44 architecture from FP8 E4M3 + FP4 E2M1 to FP6 E2M3,
// completing 3 of 4 byte-contract surfaces — the FP6 E3M2 sibling
// in `l1/fp6_e3m2_mma_dep_chain/helpers_lib.cu` is the 4th. This
// closes Codex's recurring queued #2 across all 4 surfaces).
//
// This file is an instance of the now-codified pattern in
// `BL-20260427-compiled-helper-enforcement-template-for-byte-contract-
// rows` (codified in this same R46 after FP8 E4M3 [R43] +
// FP4 E2M1 [R44] + FP6 E2M3/E3M2 [R46] reached 4 of 4 byte-
// contract surfaces). The BL is the canonical reference for the
// 5-component template: helpers_lib.cu + Makefile target + ctypes
// test + module fixture + pre-submission mutation verification.
//
// This file exports an `extern "C"` version of the canonical Round-5
// FP6 E2M3 byte-contract function so a Python test can load this
// library via `ctypes.CDLL` and verify that the COMPILED C-side
// codes match the canonical Python helper byte-for-byte.
//
// Why a separate file instead of exposing main.cu's `_quantize_fp6_e2m3_code`:
//   - main.cu's function is `static` (translation-unit-local) for the
//     kernel runner binary's needs.
//   - This file duplicates the function with `extern "C"` linkage so it
//     can be loaded as a shared library.
//   - The R33-34 static-source-analysis guards
//     (`tests/test_fp6_e2m3_byte_contract_static_guard.py`)
//     enforce main.cu's correctness; the R46 compiled-helper test
//     enforces THIS file's correctness via runtime code comparison.
//   - Together: both copies must be canonical Round-5 form.
//
// FP6 E2M3 vs FP8 E4M3 / FP4 E2M1 contract differences:
//   - 6-bit code (0x00..0x3F) in the low 6 bits of a `uint8_t`. Sign
//     bit is at position 5; magnitude index is in low 5 bits.
//   - FP6 has NO NaN encoding: FP32 NaN (any sign / payload) →
//     code 0x00 (canonical positive zero). Same as FP4 E2M1.
//   - FP6 has NO Inf encoding: FP32 ±Inf → code 0x1F / 0x3F
//     (saturate to ±7.5; max representable magnitude). Same shape as FP4.
//   - FP32 -0.0 canonicalizes to code 0x00 (NOT 0x20). Both +0 and
//     -0 collapse to the positive-zero code. This differs from FP8
//     E4M3 where -0 → 0x80 (sign-preserving).
//   - 32 representable positive magnitudes (8 denormals + 8 each for
//     exp 1/2/3; max 7.5 with bias=1).
//   - Banker's rounding on tie (LSB == 0 wins) — same convention.
//
// FP6 E2M3 vs FP6 E3M2 (the sibling): identical algorithm + same code
// layout (sign bit at position 5, magnitude in low 5 bits, no NaN/Inf
// encoding, -0 canonicalizes to 0x00, banker's rounding). The ONLY
// difference is the value table: E2M3 has bias=1 + 3 mantissa bits +
// max=7.5; E3M2 has bias=3 + 2 mantissa bits + max=28.0.
//
// Build via the row's Makefile target `helpers_lib`:
//   make -C l1/fp6_e2m3_saturation helpers_lib
// Produces `helpers_lib.so` in the row directory.

#include <math.h>
#include <stdint.h>

// FP6 E2M3 representable positive magnitudes (mirrors Python module's
// `references/_low_precision_quant.py::FP6_E2M3_POSITIVE_VALUES` and
// main.cu's `FP6_E2M3_POSITIVE_VALUES`). 32 entries: 8 denormals
// (exp=0) + 8 each for exp=1, exp=2, exp=3.
static const float FP6_E2M3_POSITIVE_VALUES[32] = {
    // exp=0 (denormal): m / 8
    0.0f,    0.125f,  0.25f,   0.375f,  0.5f,    0.625f,  0.75f,   0.875f,
    // exp=1: 1.0 + m/8
    1.0f,    1.125f,  1.25f,   1.375f,  1.5f,    1.625f,  1.75f,   1.875f,
    // exp=2: (1.0 + m/8) * 2
    2.0f,    2.25f,   2.5f,    2.75f,   3.0f,    3.25f,   3.5f,    3.75f,
    // exp=3: (1.0 + m/8) * 4
    4.0f,    4.5f,    5.0f,    5.5f,    6.0f,    6.5f,    7.0f,    7.5f,
};

// Canonical Round-5 FP6 E2M3 quantizer (must match Python
// `references/_low_precision_quant.py::_quantize_one_fp6_e2m3` AND
// main.cu's `_quantize_fp6_e2m3_code` byte-for-byte).
//
// Algorithm:
//   - FP32 NaN → code 0x00 (FP6 has no NaN encoding).
//   - ±Inf → code 0x1F / 0x3F (saturate to ±7.5).
//   - |v| >= +7.5 → saturate to ±7.5 (code 0x1F / 0x3F).
//   - Otherwise: linear search nearest-representable + banker's tie-break.
//   - -0.0 + idx==0 → canonicalize to code 0x00.
//
// `extern "C"` linkage so the function is loadable via Python ctypes
// for the R46 compiled-helper enforcement test.
extern "C" uint8_t simbench_quantize_fp6_e2m3_code(float value) {
    if (isnan(value)) return 0x00;
    int sign_bit = (signbit(value) || value < 0.0f) ? 1 : 0;
    float abs_v = fabsf(value);
    int idx;
    if (isinf(abs_v)) {
        idx = 31;  // ±7.5 (max representable)
    } else if (abs_v - FP6_E2M3_POSITIVE_VALUES[31] >= 0.0f) {
        idx = 31;
    } else {
        float best_dist = INFINITY;
        int best_idx = 0;
        for (int i = 0; i < 32; ++i) {
            float d = fabsf(abs_v - FP6_E2M3_POSITIVE_VALUES[i]);
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
