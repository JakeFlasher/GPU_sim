# `workloads/portable_lcd/flash_attn/` Layout

This directory vendors Tri Dao's FlashAttention v2 source. The
source is split into two subtrees with different roles:

## `upstream/` — verbatim Tri Dao source (third-party vendored)

Files copied byte-for-byte from
`github.com/Dao-AILab/flash-attention` at tag `v2.7.4.post1`
(commit SHA `5231d95fe13733fb534c01895f7ea88c6a6c7793`; see
`./VERSION` for the exact upstream-path → local-path map). The
license is BSD-3-Clause; the upstream `LICENSE` is mirrored at
`upstream/LICENSE` and the upstream `AUTHORS` is at
`upstream/AUTHORS`.

These files are **not compiled today** by any LCD-portable
build target. The upstream forward and backward kernel
templates depend on CUTLASS / CUTE layouts that this repo
vendors separately under `workloads/portable_lcd/cutlass/` (a
later round). Once that CUTLASS subset is present, a future
revision will replace the `lcd_glue/fwd_sm80.cuh` LCD-portable
wrapper with a thin shim that consumes the verbatim upstream
sources directly.

The full FA-local include closure is vendored — every quoted
`#include` in `upstream/*.h` and `upstream/*.cuh` resolves to
another file in `upstream/`, except for the external CUTLASS /
CUTE / CUDA-runtime headers. The
`workloads/portable_lcd/tests/test_portable_lcd_flash_attn_upstream_closure.py`
regression test enforces this invariant.

Per the per-kernel hand-written source-tree footprint cap,
third-party vendored headers are excluded. The `upstream/`
subtree is the conformant home for that exclusion.

## `lcd_glue/` — local LCD-portable glue (handwritten)

`lcd_glue/fwd_sm80.cuh` is a handwritten LCD-portable
FlashAttention-2 forward kernel that bridges the
`compute_80`-PTX-only LCD baseline (no CUTLASS today) and the
slug at `workloads/portable_lcd/flash_attn_fwd_bf16_sm80/`.
The algorithm is structurally faithful to upstream — row-blocked
Q, online softmax over K/V tiles, FP32 accumulator rescaling
per iteration, double-buffered `cp.async`, warp-level
`mma.sync.aligned.m16n8k16` BF16 GEMM — but uses the same
inline-PTX surface that the rest of the suite uses, so it
compiles cleanly without CUTLASS / CUTE.

`lcd_glue/` IS counted toward the per-kernel hand-written
source-tree footprint cap until it is replaced by the
verbatim-upstream wrapper.
