# `workloads/portable_lcd/cutlass/` — vendored CUTLASS subset

This directory holds the minimal CUTLASS subset that the
`portable_lcd` suite depends on. It is the CUTLASS sister of the
existing `flash_attn/` vendoring.

## Layout

```
cutlass/
├── LICENSE       # verbatim CUTLASS BSD-3-Clause LICENSE
├── VERSION       # tag, full commit SHA, upstream URL, bucket counts
├── NOTES.md      # this file
├── tools/
│   └── recompute_subset.py  # authoritative closure-regeneration tool
└── upstream/     # verbatim third-party source from CUTLASS v3.8.0
    ├── FILES.txt # enumerated file list (sorted, 315 paths)
    ├── cute/
    └── cutlass/
```

## What is in `upstream/`

A 315-file subset of `<CUTLASS_v3.8.0>/include/{cute, cutlass}/`,
copied verbatim. Every file's local path matches its upstream path,
just rooted at `cutlass/upstream/` instead of `<CUTLASS>/include/`.

The subset is the **include-closure** of these seed entries:

* `cutlass/cutlass.h`, `cutlass/array.h`, `cutlass/coord.h`,
  `cutlass/numeric_types.h` (top-level core types)
* `cutlass/arch/{mma.h, mma_sm80.h}` (warp-level MMA atoms)
* `cutlass/gemm/collective/sm80_mma_multistage.hpp` (sm_80 GEMM
  multistage mainloop)
* `cutlass/bfloat16.h`, `cutlass/half.h` (numeric type wrappers)
* `cutlass/conv/device/implicit_gemm_convolution.h` (Conv2dFprop
  device entry, used by future implicit-GEMM convolution kernels)
* `cutlass/conv/kernel/default_conv2d_fprop.h` (Conv2dFprop sm_80
  kernel entry)

The closure is computed by `tools/recompute_subset.py`, which walks
BOTH quoted (`#include "cutlass/..."`, `#include "cute/..."`) AND
angle-bracket (`#include <cutlass/...>`, `#include <cute/...>`)
include directives from those seeds. CUTLASS sources mix both forms
for local dependencies; an earlier quoted-only parser missed many
angle-bracket local targets such as `<cute/config.hpp>` and
`<cute/atom/mma_traits_sm80.hpp>`. The repaired both-syntax parser
yields the 315-file closure recorded here.

System / CUDA toolkit / external angle-bracket includes
(`<cuda_runtime.h>`, `<cinttypes>`, `<utility>`, etc.) are
recognised by the absence of a `cutlass/` or `cute/` prefix and are
not vendored.

## Authoritative regeneration / verification

The single source of truth for the seed list, parser rules, and
on-disk manifest is `tools/recompute_subset.py`. It supports two
modes:

```bash
# Verify the on-disk subset matches the closure under a fresh
# upstream snapshot (any drift exits non-zero with a typed
# diagnostic):
python3 workloads/portable_lcd/cutlass/tools/recompute_subset.py \
    --source-include-root <CUTLASS-v3.8.0>/include \
    --dest-upstream workloads/portable_lcd/cutlass/upstream \
    --mode check

# Rebuild the subset after bumping the pinned tag:
python3 workloads/portable_lcd/cutlass/tools/recompute_subset.py \
    --source-include-root <CUTLASS-vX.Y>/include \
    --dest-upstream workloads/portable_lcd/cutlass/upstream \
    --mode sync

# Self-consistency check (no fresh upstream snapshot needed):
python3 workloads/portable_lcd/cutlass/tools/recompute_subset.py \
    --source-include-root workloads/portable_lcd/cutlass/upstream \
    --dest-upstream workloads/portable_lcd/cutlass/upstream \
    --mode check
```

The tool's parser logic is unit-tested by
`tests/test_portable_lcd_cutlass_recompute_subset.py` (synthetic
include tree with quoted + sibling-relative + local-angle + external
angle exercises, plus the self-consistency mode against the real
vendored tree, plus a negative test that drives the
"missing required header" diagnostic).

## What IS in the closure (architecture-specialized CuTe headers)

The vendored subset is the include-closure under the documented
parser, not a hand-curated minimality claim. CUTLASS / CuTe
aggregate headers (notably `cute/atom/mma_atom.hpp` and
`cute/atom/copy_atom.hpp`) include their architecture-specialized
trait siblings unconditionally:

```cpp
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/mma_traits_sm90.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
```

Consequently the closure pulls in headers covering Volta through
Datacenter-Blackwell architectures, including:

* `cute/arch/mma_sm70.hpp`, `mma_sm75.hpp`, `mma_sm80.hpp`,
  `mma_sm90.hpp`, `mma_sm90_gmma.hpp`, `mma_sm90_gmma_ext.hpp`,
  `mma_sm90_gmma_sparse.hpp`, `mma_sm90_gmma_sparse_ext.hpp`,
  `mma_sm100.hpp`, `mma_sm100_umma.hpp`
* `cute/arch/copy_sm80.hpp`, `copy_sm90.hpp`, `copy_sm90_tma.hpp`,
  `copy_sm100.hpp`, `copy_sm100_tma.hpp`,
  `cluster_sm90.hpp`
* `cute/atom/copy_traits_sm90.hpp`, `copy_traits_sm90_tma.hpp`,
  `copy_traits_sm100.hpp`, `copy_traits_sm100_tma.hpp`
* `cute/atom/mma_traits_sm90.hpp`, `mma_traits_sm90_gmma.hpp`,
  `mma_traits_sm100.hpp`
* `cutlass/arch/mma_sm90.h`

These files contain GMMA / TMA / sm_100 / UMMA material. They are
vendored because the preprocessor needs them on disk to parse the
aggregate CuTe headers; **the consuming kernel does not emit any
of those instructions**. Architectural-LCD enforcement happens at
three orthogonal layers:

1. **Build-flag layer** — every consuming kernel compiles with
   `compute_80` PTX plus `sm_86 / sm_90 / sm_120` base-real-arch
   flags only. No `sm_90a` / `sm_120a` / `sm_120f`. The
   `compute_80` PTX path forces ptxas to emit Sm80 ISA, so any
   guarded Hopper / Blackwell CuTe code path is dead-stripped at
   compile time.
2. **Source-instantiation layer** — the consuming kernel only
   instantiates the Sm80 mainloop / Conv2dFprop entry points
   (e.g., `cutlass::gemm::collective::CollectiveMma<sm_80, ...>`,
   `cutlass::conv::kernel::DefaultConv2dFprop<...>`); the
   Hopper / Blackwell-only template specialisations are never
   reached.
3. **SASS-verification layer** — the fatbin-slice verifier
   scans the compiled sm_120 SASS for forbidden mnemonics
   (`UTC*MMA`, `LDT`, `STT`) and fails the build gate if any
   leak through. The static guard pytest forbids any CUPTI /
   NCU / privileged-`nvidia-smi` surface in active source.

Until a future round actually consumes CUTLASS in a compiled
kernel (i.e., lands `conv2d_implicit_gemm_bf16_sm80`), the
vendored subset is parse-only and emits no SASS.

## What is NOT vendored

* No CUTLASS profiler, examples, tests, or tools.
* No CUTE example kernels.
* No FP4 / FP6 / NVFP4 / FP8 QMMA paths beyond what aggregate
  CuTe / CUTLASS headers transitively pull in (those that ARE
  pulled in are dead-code under the `compute_80` build flag).
* No `lcd_glue/` directory yet. Future LCD-portable bridges between
  the vendored subset and the inline-PTX surface used by the rest
  of `portable_lcd/` will live there. (FA's
  `flash_attn/lcd_glue/{fwd_sm80.cuh, bwd_sm80.cuh}` is the precedent.)

## Vendoring discipline

* Files under `upstream/` are byte-identical to their CUTLASS
  v3.8.0 counterparts. Do not edit them. If a future round needs to
  patch CUTLASS behaviour, copy the file into `lcd_glue/` first.
* The `upstream/` subtree is excluded from per-kernel <=600 LOC
  caps, exactly as `flash_attn/upstream/` already is.
* Every CUTLASS-local include in `upstream/` (whether quoted or
  angle-bracketed) must resolve to a sibling under `upstream/`. The
  closure regression test
  `tests/test_portable_lcd_cutlass_upstream_closure.py` enforces
  this contract for both syntaxes. A separate preprocessor smoke
  test (`tests/test_portable_lcd_cutlass_preprocessor_smoke.py`)
  invokes `g++ -fsyntax-only -I cutlass/upstream` over a small set
  of representative seed headers when a host C++ toolchain is
  available. The committed regeneration tool's check mode is the
  third enforcement layer.

## License attribution

`LICENSE` is the verbatim CUTLASS v3.8.0 BSD-3-Clause license. CUTLASS
is © 2017-2025 NVIDIA CORPORATION & AFFILIATES. The vendored subset
inherits the upstream license verbatim; downstream consumers must
preserve the LICENSE file when redistributing.

## Why CUTLASS v3.8.0 specifically

The vendoring policy allows v3.7.0 or v3.8.0; v3.8.0 was selected because:
1. v3.8.0 retains the pure Sm80 mainloop unchanged from v3.7.0.
2. v3.8.0 fixes a profiler regression (#2124) in the upstream HEAD.
3. Both tags pre-date the v4.x betas that introduced cluster_dim
   leakage into Sm80 paths.
