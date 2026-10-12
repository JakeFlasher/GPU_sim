// Compiled-helper enforcement library (R43 addition per user-
// authorized Round-42-end promotion of Codex's recurring queued #2
// "compiled-helper contract for low-precision byte-contract checks").
//
// This file is an instance of the now-codified pattern in
// `BL-20260427-compiled-helper-enforcement-template-for-byte-contract-
// rows` (codified in R46 after FP8 E4M3 [R43] + FP4 E2M1
// [R44] + FP6 E2M3/E3M2 [R46] reached 4 of 4 byte-contract
// surfaces). The BL is the canonical reference for the 5-component
// template: helpers_lib.cu + Makefile target + ctypes test + module
// fixture + pre-submission mutation verification.
//
// This file exports `extern "C"` versions of the canonical Round-31
// FP8 E4M3 byte-contract functions so a capture tooling can load this
// library via `ctypes.CDLL` and verify that the COMPILED C-side bytes
// match the canonical Python helper byte-for-byte.
//
// Why a separate file instead of exposing main.cu's `_quantize_fp8_e4m3_byte`:
//   - main.cu's function is `static` (translation-unit-local) for the
//     kernel runner binary's needs.
//   - This file duplicates the function with `extern "C"` linkage so it
//     can be loaded as a shared library.
//   - The retained byte and sparse-layout workload contracts must remain
//     aligned with main.cu; the compiled helper is kept for the capture build
//     path and runtime byte comparison.
//   - Together: both copies must be canonical Round-31 form.
//
// If the two copies ever diverge, the R32 + R38-41 static
// guards would still pass (they only check main.cu) but the R43
// compiled-helper path would catch the divergence (since it compares
// THIS file's bytes to the Python helper, which itself matches main.cu
// via the existing host-independent C/Python contract test).
//
// Build via the row's Makefile target `helpers_lib`:
//   make -C l1/sparse_2to4_fp8_gemm helpers_lib
// Produces `helpers_lib.so` in the row directory.

#include <math.h>
#include <stdint.h>

// FP8 E4M3 saturation max (OCP/NVIDIA E4M3 max normal = +448).
static const float FP8_E4M3_MAX = 448.0f;

// FP8 E4M3 representable value table generator (mirrors Python module's
// `_build_fp8_e4m3_value_table` and main.cu's `_init_fp8_e4m3_table`).
// 256 entries; bytes 0x7F and 0xFF are NaN. Filled lazily once at startup.
static float FP8_E4M3_VALUE_TABLE[256];
static int FP8_E4M3_TABLE_INITIALIZED = 0;

static void _init_fp8_e4m3_table(void) {
    for (int b = 0; b < 256; ++b) {
        int s = (b >> 7) & 1;
        int e = (b >> 3) & 0xF;
        int m = b & 0x7;
        if (e == 15 && m == 7) {
            FP8_E4M3_VALUE_TABLE[b] = NAN;
            continue;
        }
        float sign = s ? -1.0f : 1.0f;
        float val;
        if (e == 0) {
            // Denormal: (m/8) * 2^(1-7) = m / 512
            val = sign * ((float)m / 8.0f) * powf(2.0f, 1 - 7);
        } else {
            // Normal: (1 + m/8) * 2^(e-7)
            val = sign * (1.0f + (float)m / 8.0f) * powf(2.0f, e - 7);
        }
        FP8_E4M3_VALUE_TABLE[b] = val;
    }
    FP8_E4M3_TABLE_INITIALIZED = 1;
}

// Canonical Round-31 FP8 E4M3 quantizer (must match Python
// `references/_low_precision_quant.py::_quantize_one_fp8_e4m3` AND
// main.cu's `_quantize_fp8_e4m3_byte` byte-for-byte).
//
// R31 alignment with `cuda_fp8.h::__nv_fp8_e4m3` per Codex
// Round-29-review CUDA 13 host probe:
//   - FP32 NaN → byte 0x7F (canonical positive E4M3 NaN; all NaN sign
//     collapsed).
//   - -0.0 → byte 0x80 (sign-preserving via the general
//     `best_idx | (sign_bit << 7)` path).
//   - ±Inf → byte 0x7E / 0xFE (satfinite to ±448; E4M3 has no Inf encoding).
//   - Other values: nearest-representable + sign-preserving.
//
// `extern "C"` linkage so the function is loadable via Python ctypes
// for the R43 compiled-helper contract test.
extern "C" uint8_t simbench_quantize_fp8_e4m3_byte(float value) {
    if (!FP8_E4M3_TABLE_INITIALIZED) _init_fp8_e4m3_table();
    if (isnan(value)) return 0x7F;
    int sign_bit = (signbit(value) || value < 0.0f) ? 1 : 0;
    float abs_v = fabsf(value);
    if (isinf(abs_v) || abs_v >= FP8_E4M3_MAX) {
        return sign_bit ? 0xFE : 0x7E;
    }
    int best_idx = 0;
    float best_dist = INFINITY;
    for (int b = 0; b < 0x7F; ++b) {  // 0x00..0x7E (positive bytes)
        float cand = FP8_E4M3_VALUE_TABLE[b];
        float d = fabsf(abs_v - cand);
        if (d < best_dist) {
            best_dist = d;
            best_idx = b;
        } else if (d == best_dist) {
            if ((b & 0x1) == 0) best_idx = b;
        }
    }
    return (uint8_t)(best_idx | (sign_bit << 7));
}
