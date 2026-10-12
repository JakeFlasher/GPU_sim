// Compiled-helper enforcement library (R44 addition; extends the
// R43 architecture from FP8 E4M3 to FP4 E2M1, per the user's
// Round-42-end authorization to start blocking issues directly +
// promote queued #2 (compiled-helper contract) across all four
// byte-contract C mirrors).
//
// This file is an instance of the now-codified pattern in
// `BL-20260427-compiled-helper-enforcement-template-for-byte-contract-
// rows` (codified in R46 after FP8 E4M3 [R43] + FP4 E2M1
// [R44] + FP6 E2M3/E3M2 [R46] reached 4 of 4 byte-contract
// surfaces). The BL is the canonical reference for the 5-component
// template: helpers_lib.cu + Makefile target + ctypes test + module
// fixture + pre-submission mutation verification.
//
// This file exports `extern "C"` versions of the canonical Round-4
// FP4 E2M1 byte-contract function so a capture tooling can load this
// library via `ctypes.CDLL` and verify that the COMPILED C-side
// nibbles match the canonical Python helper byte-for-byte.
//
// Why a separate file instead of exposing main.cu's `_quantize_fp4_e2m1_nibble`:
//   - main.cu's function is `static` (translation-unit-local) for the
//     kernel runner binary's needs.
//   - This file duplicates the function with `extern "C"` linkage so it
//     can be loaded as a shared library.
//   - The R33 + R34 static-source-analysis guards
//     (`the retained workload contract`; R33
//     added the initial 5 guards; R34 tightened them to exact
//     byte-contract semantics + added forbidden-pattern guards per
//     Codex Round-33-review)
//     enforce main.cu's correctness; the R44 compiled-helper path
//     enforces THIS file's correctness via runtime nibble comparison.
//   - Together: both copies must be canonical Round-4 form.
//
// FP4 E2M1 vs FP8 E4M3 differences (relative to R43's
// helpers_lib.cu in `l1/sparse_2to4_fp8_gemm/`):
//   - Return value is a NIBBLE (low 4 bits of a `uint8_t`), not a byte.
//   - FP4 E2M1 has NO NaN encoding: FP32 NaN (any sign / payload) →
//     nibble 0x0 (canonical positive zero), NOT a special pattern.
//   - FP4 E2M1 has NO Inf encoding: FP32 ±Inf → nibble 0x7 / 0xF
//     (saturate to ±6.0; the max representable magnitude).
//   - FP32 -0.0 canonicalizes to nibble 0x0 (NOT 0x8). Both +0 and -0
//     collapse to the positive-zero nibble. This differs from FP8 E4M3
//     where -0 → 0x80 (sign-preserving).
//   - 8 representable positive magnitudes (vs 128 for FP8 E4M3):
//     {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
//   - Banker's rounding on tie (LSB == 0 wins) — same convention.
//
// Build via the row's Makefile target `helpers_lib`:
//   make -C l1/fp4_e2m1_mma_dep_chain helpers_lib
// Produces `helpers_lib.so` in the row directory.

#include <math.h>
#include <stdint.h>

// FP4 E2M1 representable positive magnitudes (mirrors Python module's
// `references/_low_precision_quant.py::FP4_E2M1_POSITIVE_VALUES` and
// main.cu's `FP4_E2M1_POSITIVE_VALUES`). Index ↔ nibble mapping:
//   0x0 ↔ +0.0   0x1 ↔ +0.5   0x2 ↔ +1.0   0x3 ↔ +1.5
//   0x4 ↔ +2.0   0x5 ↔ +3.0   0x6 ↔ +4.0   0x7 ↔ +6.0
// Negative magnitudes use the same indices with sign bit (bit 3) set:
//   0x8 ↔ -0.0 (canonicalizes to 0x0 on quantize)
//   0x9 ↔ -0.5   0xA ↔ -1.0   0xB ↔ -1.5
//   0xC ↔ -2.0   0xD ↔ -3.0   0xE ↔ -4.0   0xF ↔ -6.0
static const float FP4_E2M1_POSITIVE_VALUES[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

// Canonical Round-4 FP4 E2M1 quantizer (must match Python
// `references/_low_precision_quant.py::_quantize_one_fp4_e2m1` AND
// main.cu's `_quantize_fp4_e2m1_nibble` byte-for-byte).
//
// Algorithm:
//   - FP32 NaN → nibble 0x0 (FP4 has no NaN encoding).
//   - ±Inf → nibble 0x7 / 0xF (saturate to ±6.0; FP4 has no Inf encoding).
//   - |v| >= +6.0 → saturate to ±6.0 (nibble 0x7 / 0xF).
//   - Otherwise: linear search for nearest representable positive
//     magnitude with banker's tie-break (on equal distance, even-nibble
//     index wins).
//   - -0.0 + idx==0 → canonicalize to nibble 0x0 (positive zero).
//
// `extern "C"` linkage so the function is loadable via Python ctypes
// for the R44 compiled-helper contract test.
extern "C" uint8_t simbench_quantize_fp4_e2m1_nibble(float value) {
    if (isnan(value)) return 0x0;
    int sign_bit = (signbit(value) || value < 0.0f) ? 1 : 0;
    float abs_v = fabsf(value);
    int idx;
    if (isinf(abs_v)) {
        idx = 7;  // ±6.0 (max representable)
    } else if (abs_v - FP4_E2M1_POSITIVE_VALUES[7] >= 0.0f) {
        // Saturation: |v| >= +6.0
        idx = 7;
    } else {
        // Linear search for nearest representable. Banker's rounding:
        // on tie (equal distance), pick the candidate whose nibble has
        // the LSB == 0 (even).
        float best_dist = INFINITY;
        int best_idx = 0;
        for (int i = 0; i < 8; ++i) {
            float d = fabsf(abs_v - FP4_E2M1_POSITIVE_VALUES[i]);
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
        // Negative-zero canonicalizes to nibble 0x0 (+0.0).
        return 0x0;
    }
    return (uint8_t)((sign_bit << 3) | (idx & 0x7));
}
