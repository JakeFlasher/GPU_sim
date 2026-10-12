"""SimBench-BW Control reference family.

Vendor-library realism control rows (cuBLASLt / cuBLAS). The reference is
a plain FP32 matmul `C = A @ B`; the vendor-library kernel rows
(`vendor_fp8_matmul_cublaslt` + `vendor_gemm_cublas` +
`vendor_sgemm_strided_batched_cublas`) are compared against this reference
under a precision-appropriate tolerance band. The strided-batched row emits
only its batch-0 matrix (under the canonical `output` field), so the same
single-matmul reference applies (batch 0 uses the row's seeded inputs).

Control rows are tier=control in `roster.toml` and excluded from
headline scoring; they act as provenance checks that the suite's
non-vendor rows agree with the vendor implementations at the same
shape.

Seeded inputs (same MWC PRNG as host drivers via
`_common.deterministic_seeded_tensor`):
    A = seed        : (m, k)
    B = seed + 1    : (k, n)

Tolerance entries in `tolerances.toml`:
    `(FP8_E4M3, fma_chain, control)` for `vendor_fp8_matmul_cublaslt`
        (wide band: rtol=atol=5e-2 — E4M3 accumulation has a wide floor).
    `(BF16, fma_chain, control)`     for `vendor_gemm_cublas`
        (rtol=atol=1e-2 — the BF16 accumulate floor).
    `(FP32, fma_chain, control)`     for `vendor_sgemm_strided_batched_cublas`
        (rtol=atol=1e-3 — the FP32 step of the control-family precision
        progression; deterministic FFMA SIMT path under CUBLAS_PEDANTIC_MATH).
"""

from __future__ import annotations

from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "control"

# Declares that every slug in this family uses a pure-Python CPU
# reference path. the runtime reference path
# reads this attribute and refuses roster drift — a row declaring
# `reference_type = "fp32_gpu"` for a slug this family covers would
# fail the check. R28-review closes the R27-review blocking
# side issue by aligning roster metadata and module runtime.
REFERENCE_BACKEND: str = "cpu"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "vendor_fp8_matmul_cublaslt",
    "vendor_gemm_cublas",
    "vendor_sgemm_strided_batched_cublas",
)


def _matmul(A: list, B: list, m: int, k: int, n: int) -> list:
    """FP32 three-loop matmul; C[i, j] = sum_k A[i, k] * B[k, j]."""
    C: list = [[0.0] * n for _ in range(m)]
    for i in range(m):
        A_row = A[i]
        C_row = C[i]
        for kk in range(k):
            a_val = A_row[kk]
            B_row = B[kk]
            for j in range(n):
                C_row[j] += a_val * B_row[j]
    return C


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    """FP32 matmul reference for the two vendor-library control rows.
    `m`, `n`, `k` axes come from the shape point; both vendor rows use the
    same matmul shape convention. The reference is always FP32 even for
    the FP8_E4M3 row — the tolerance band absorbs the kernel-side
    quantization."""
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    m = int(axes["m"])
    n = int(axes["n"])
    k = int(axes["k"])

    A = deterministic_seeded_tensor((m, k), seed,     low=-0.5, high=0.5)
    B = deterministic_seeded_tensor((k, n), seed + 1, low=-0.5, high=0.5)
    return _matmul(A, B, m, k, n)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    """Tolerance-aware allclose resolved via the row's tolerances.toml entry.
    `(FP8_E4M3, fma_chain, control)` for cuBLASLt; `(BF16, fma_chain,
    control)` for cuBLAS."""
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
