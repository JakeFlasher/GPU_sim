# SimBench-BW Portability Policy

This file canonicalizes the portability vocabulary used throughout the suite
(`roster.toml`, `platform_matrix.toml`, `common.mk`, `verify_coverage.py`,
`run_suite.py`). It is the authoritative reference for the meaning of
`tier=P` and `tier=P-strict`; disagreements between tooling and this document
are bugs in the tooling.

## The Two Measurement Platforms

SimBench-BW reports numbers on two platforms:

| Platform              | Arch family     | Target flag                        | Role                                                              |
|-----------------------|-----------------|------------------------------------|-------------------------------------------------------------------|
| NVIDIA H100           | Hopper sm_90a   | `-gencode arch=compute_90a,code=sm_90a`       | Secondary cross-generation measurement target when available. |
| RTX PRO 6000 Blackwell| Blackwell sm_120| `-gencode arch=compute_120,code=sm_120`       | Primary Blackwell measurement target; cloud-rented, with local sm_86 smoke testing only. |

See `references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md` for
the authoritative feature-surface record of the Blackwell target. RTX 3090 Ti
(`sm_86`, GA102, Ampere) is the local smoke-test host on this branch; it is
not a reporting platform and cannot run any sm_120-specific kernels — only
sm_86-portable kernels execute locally.

## Portability Tiers

Every roster row declares exactly one tier. The tier drives:

1. Which compile flags `common.mk` selects for the row's Makefile.
2. Which platforms the row's result appears on in `suite_results.json`.
3. Whether the row counts toward subsystem coverage and release-gate floors.

### `tier=P` — Portable (default)

A `tier=P` row compiles on both `sm_90a` and `sm_120` under its declared
`arch_gate` and produces numerically-equivalent results (within the tolerance
declared in `tolerances.toml`) on both platforms.

**Branching policy.** `tier=P` permits compile-time branching inside a single
`.cu` file if and only if:

- Both branches compile under the declared `arch_gate` list.
- The branch predicate is the standard `__CUDA_ARCH__` preprocessor macro,
  `#if defined(__NVCC__)`-style guards, or a `#pragma nv_exec_check_disable`-class
  directive that preserves single-source semantics.
- The branches do not diverge on which reference family they target — both
  branches must map to the same `reference_family` and the same shape file.
- The output comparison passes under the row's tolerance entry on each platform.

A `tier=P` row with `#if __CUDA_ARCH__ >= 900` and `#if __CUDA_ARCH__ >= 1200`
branches that both compile and produce equivalent outputs is legal. A row whose
Hopper branch uses `wgmma` and whose Blackwell branch uses warp-level `mma`
is legal iff the numerical difference is within tolerance and both branches
have their `arch_gate` listed.

`tier=P` is the default for any row whose underlying operator has a clean
mapping to both architectures, even when that mapping requires different
intrinsics.

### `tier=P-strict` — Branchless portable (optional)

A `tier=P-strict` row is identical to `tier=P` except that it MUST NOT contain
any `__CUDA_ARCH__` branch. It is a stricter opt-in subset that surfaces
genuinely one-path-both-platforms kernels. `tier=P-strict` rows are verifiable
statically: `verify_coverage.py --check-portability` greps for `__CUDA_ARCH__`
and rejects the row if a branch is present.

`tier=P-strict` is not required. Most portable rows remain `tier=P`; declaring
`tier=P-strict` is a quality signal, not a gate.

### `tier=H` — Hopper-only

A `tier=H` row compiles only under `sm_90a`. It uses a Hopper-exclusive
instruction class (e.g. `wgmma.*`, `setmaxnreg`, Hopper-only TMA multicast).
These rows appear in `suite_results.json` with `platform=h100`; on
`rtx_pro_6000_blackwell` they appear as `status=unsupported` with
`reason=sm_90a_only`.

### `tier=B` — Blackwell-only

A `tier=B` row requires Blackwell-conditional hardware features (dense FP4 /
FP6 / NVFP4 block-scaled MMA via `sm_120f`, or sparse mxf4 via `sm_120a`, or
other `sm_120*`-conditional intrinsics). It does not build on `sm_90a`. These
rows appear with `platform=rtx_pro_6000_blackwell` when `nvcc` +
correctness both pass; on `h100` they are `status=unsupported` with
`reason=sm_120a_only` (or similar).

### `tier=control` — Vendor-library demoted

A `tier=control` row wraps a vendor library call (cuBLAS, cuBLASLt, cuDNN).
It is retained as a realism control but excluded from subsystem coverage and
from headline scoring. Control rows appear under the `vendor_control_only`
section of `suite_results.json`.

### `tier=unsupported` — Reserved

Reserved for rows that cannot be built on any current target platform. Such
rows remain in `roster.toml` with `status=unsupported` for provenance but
do not appear in scoring.

## Feature Availability Gating

SimBench-BW does not operate a runtime probe runner. Feature availability is
enforced by two orthogonal mechanisms:

1. **`nvcc` compile-time rejection.** A row declaring `arch_gate = sm_90a`
   fails to build on an `sm_120`-only system with
   `Unsupported gpu architecture 'sm_90a'`. A row declaring `arch_gate = sm_120a`
   and using a warp-level FP4 `mma` fails to build on `sm_90a` with a ptxas
   target mismatch. No partial binary is ever produced.
2. **Per-kernel correctness gate.** A row that compiles on a platform but
   produces incorrect results (e.g. because the arch-conditional branch went
   down a wrong path) fails its correctness check and enters
   `suite_results.json` with `status=correctness_failed`.

Under these two mechanisms, the `arch_gate ↔ platform_matrix` consistency
check (enforced by `verify_coverage.py --check-arch-gate-platform-consistency`)
is sufficient: if a row declares `tier=P` and its `arch_gate` is not a superset
that covers both `sm_90a` and `sm_120`, the check flags the row as
`tier_vs_arch_gate_mismatch`.

## Excluded Targets

These architectures are explicitly NOT validation targets for SimBench-BW:

- `sm_100` / `sm_100a` / `sm_100f` (B200 datacenter Blackwell). The repo has
  no B200 hardware. Rows requiring `tcgen05.*`, TMEM, or `.block16` /
  `.block32` block-scaled MMA variants are out of scope.
- Older Ampere / Turing generations (`sm_80`, `sm_86`, `sm_89`, `sm_75`,
  `sm_70`). The suite targets the 2025–2026 LLM inference surface, which is
  defined by features introduced in Hopper and later.

Support for additional targets is a future-work item; adding one requires
extending the `ARCH_GATE` enum in `common.mk`, the `arch_gate` and `tier`
enums in `roster.schema.json`, and the `platforms` enum in
`platform_matrix.schema.json`, then re-running
`verify_coverage.py --strict`.
