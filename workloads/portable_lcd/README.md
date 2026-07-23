# `workloads/portable_lcd/` — LCD-Portable Tracer-Debug Suite

A small, curated workload suite of real-world AI/HPC kernels whose
architectural feature set is the **lowest common denominator** across
NVIDIA Ampere `sm_86` (the local RTX 3090 Ti smoke-test host on this
branch), Hopper `sm_90a` (the cloud H100 target), and workstation
Blackwell `sm_120` (the cloud RTX PRO 6000 Blackwell target).

Single-source kernels compile via a single `-arch=compute_80` PTX
baseline plus three real-arch SASS slices (`sm_86` + `sm_90` +
`sm_120`), so the identical fatbin runs on the local sm_86 host and
on cloud sm_90a / sm_120 rentals. A fourth `compute_80` PTX section
embeds the LCD virtual arch as a forward-compatibility JIT fallback.

## Source of Truth

This README is authored against repository HEAD on the active
development branch (`5060_3090ti_cross_dev` at the time of writing).
Documented commands assume live CLI shapes of the scripts and
Makefiles they invoke. For re-reads without changing the worktree,
prefer `git show <branch>:workloads/portable_lcd/README.md` over
`git checkout`. The pattern matches
`BL-20260504-runbook-self-reference`: source-of-truth and
verification-target are decoupled so the operator never re-reads
the runbook from a stale post-checkout worktree.

## Quick Start

```bash
# Verify the preserved authority and all 15 local binary pins without writing:
.venv/bin/python tools/local_labels/build_preserved_binaries.py --check
.venv/bin/python tools/local_labels/gate_preserved_binaries_local.py

# Run the standalone contract verifier across the roster
# (skips rows whose status is "pending"):
python3 workloads/portable_lcd/verify_kernel_contracts.py

# Narrower verified-contract command when tracer_tool.so is unbuilt
# (skips the per-row NVBit trace verifier; arch-slice + SASS coverage
# verifiers still run):
python3 workloads/portable_lcd/verify_kernel_contracts.py --skip-traces
```

The verifier reads `roster.toml` from the standalone subtree by default;
`--roster <PATH>` overrides the location. Active rows must satisfy
`tools/build/verify_lcd_arch_slices.sh` (fatbin slice discipline),
`tools/build/verify_lcd_sass_coverage.sh` (SASS mnemonic coverage), and
(unless `--skip-traces` is passed) `tools/build/verify_lcd_nvbit_trace.sh`
(NVBit trace artifact contract).

## What this suite IS

* **Tracer-debug + protocol-validation infrastructure.** The local
  RTX 3090 Ti host on the `non_root_simbench_profile` branch has no
  runnable workload from the existing 74-row SimBench-BW roster
  (every row gates on `sm_90a` / `sm_120` / `sm_120a` / `sm_120f`).
  This suite fills the gap so the NVBit tracer + phase-injection
  foundation under `tools/nvbit_tracer/` can be exercised end-to-end
  on real instruction-rich SASS. Live CUPTI Activity collection was
  retired by D8; committed historical sidecars remain valid only
  against `workloads/simbench_bw/schemas/historical/cupti_activity_sidecar.v1.schema.json`.
* **A regression workload corpus** for the phase-injection
  substring-match bugs surfaced in the prior tracer review (the
  `is_bar` / `is_branch` / `is_ret` matchers in
  `tools/nvbit_tracer/tracer_tool/tracer_tool.cu`). The kernel
  roster collectively emits TRUE `BAR.SYNC` (CTA-level
  `__syncthreads()` lowering), `MEMBAR` / `ERRBAR` / `DEPBAR` /
  `LDGDEPBAR` (memory fences, NOT sync barriers), `BSSY` / `BSYNC`
  / `EXIT` / `RET`, and at least one of `WARPSYNC` / `CALL` / `JMX`.

## What this suite is NOT

* **NOT a published-quality benchmark.** Tile sizes are chosen for
  "reasonable for sm_80," not autotuned per arch.
* **NOT part of the SimBench-BW v1.0.0 release scope.**
* **NOT a calibration data source.** This suite does NOT restore
  the **retracted RTX 5060 Laptop calibration evidence** (the
  130560 ns dual-source CUPTI tolerance value derived from the
  5060 host, retracted prior to the 3090 Ti migration). D8 retired
  that live CUPTI calibration workflow in favor of NCU-only timing
  truth. The retained
  `workloads/simbench_bw/golden/cupti_dual_source_tolerance.json`
  is historical validator input, not authority for new collection.
* **NOT a tracer bug fix.** The phase-injection substring-match
  bugs and the `phase_records_buffer` saturation hang remain
  present after this suite lands; this suite produces the SASS
  regression material a future fix will need.

## Architectural Discipline

The LCD virtual arch is **`compute_80`**. The three real-arch
cubin slices are **`sm_86`, `sm_90`, `sm_120`** — the BASE
architectures of each device family. The architecture-conditional
variants `sm_90a`, `sm_120a`, and `sm_120f` are **explicitly
forbidden as build targets** because they enable Hopper-specific
(`wgmma`, `setmaxnreg`, `cp.async.bulk.tensor`) or
Blackwell-architecture-conditional (`mma.sync.kind::f8f6f4`,
`tcgen05`) feature surfaces outside the LCD discipline.

This restriction is enforced empirically by the AC-2 fatbin-slice
verifier under `tools/build/verify_lcd_arch_slices.sh`, which runs
`cuobjdump --list-elf` on every produced binary and rejects:

* Any binary missing one of `sm_86`, `sm_90`, `sm_120` slices
  (typed diagnostic: `lcd_arch_slice_missing: <arch>`).
* Any binary embedding `sm_90a`, `sm_120a`, or `sm_120f` slices
  (typed diagnostic: `lcd_arch_conditional_slice_forbidden:
  <arch>`).
* Any binary missing the `compute_80` PTX fallback section.

Build-time guard: `workloads/portable_lcd/common.mk` rejects the
forbidden architecture-conditional tokens at make-expansion time
with a typed `$(error ...)` message naming the LCD discipline
violation, so `ARCH_GATE = sm_90a` (or similar) cannot accidentally
slip into a kernel Makefile.

The forward-compatibility intent is **device-family based**: the
`sm_90` cubin is intended to run on H100-class devices (CC 9.0,
including `sm_90a` variants) without using any `sm_90a`-only
features; the `sm_120` cubin is intended to run on RTX PRO 6000
Blackwell (CC 12.0, including `sm_120a` / `sm_120f` variants)
without using any architecture-conditional features. This suite
does not claim "forward compatibility" beyond device-family intent.

## Roster Decisions (DEC-1 … DEC-6, resolved 2026-04-29)

| ID | Decision | Resolution |
|----|----------|------------|
| DEC-1 | Architectural target set | `sm_86 + sm_90a + sm_120` (project-aligned). Build slices target BASE archs `sm_86 + sm_90 + sm_120`. `sm_100` (B200) explicitly out of scope per `CLAUDE.md`. |
| DEC-2 | Wave 2 scope | Wave 2 IN PLAN as Lower bound (REQUIRED). All 15 kernels (six first-wave + nine wave-2) required for plan acceptance. |
| DEC-3 | Suite directory location | `workloads/portable_lcd/` (standalone subtree, NOT nested under `simbench_bw/`). |
| DEC-4 | CUTLASS dependency policy | Vendor minimal subset (~50-80 files) under `cutlass/`, pinned to CUTLASS v3.7.0 or v3.8.0. Tag + SHA recorded in `cutlass/VERSION`. |
| DEC-5 | FlashAttention sm_80 source | Vendor full Tri Dao FA v2 (forward + backward) under `flash_attn/`, BSD-3 LICENSE attribution + tag/SHA in `flash_attn/VERSION`. |
| DEC-6 | Sparse 2:4 metadata format | Use the sm_80 metadata layout uniformly across all three real-arch slices. |

## Roster

The full kernel roster lives at `workloads/portable_lcd/roster.toml`
with all 15 rows. Each row declares
`arch_gate = ["sm_86", "sm_90", "sm_120"]` and `tier = "P"` (LCD
portable). Wave-2 rows whose source has not been landed yet carry
`status = "pending"`; rows whose `<slug>_run` binary is buildable
carry `status = "active"`. The standalone subtree gate at
`workloads/portable_lcd/verify_kernel_contracts.py` reads the
roster and skips pending rows automatically.

The roster admits rows in two waves:

### First Wave (Milestone 2)

1. `gemm_hmma_bf16_sm80` — HMMA BF16 + cp.async + double-buffered shared mem.
2. `rmsnorm_fwd_bf16_sm80` — port from existing SimBench-BW slug.
3. `softmax_safe_bf16_sm80` — two-pass safe softmax, online normalization.
4. `swiglu_fwd_bf16_sm80` — Llama-style FFN gate, element-wise + multiply-add.
5. `embedding_gather_int32_sm80` — integer scatter-gather, atomic-free indexing.
6. `cross_entropy_fwd_fp32_sm80` — softmax + log + one-hot gather, reduction-heavy.

### Wave 2 (Milestone 5)

7. `gemm_hmma_fp16_sm80` — HMMA FP16 (paired with #1 for tensor-core opcode-mix coverage).
8. `gemm_imma_int8_sm80` — IMMA INT8, integer accumulator.
9. `gemm_sparse_2to4_bf16_sm80` — `mma.sp` BF16 sparse 2:4 (sm_80 metadata layout).
10. `flash_attn_fwd_bf16_sm80` — vendored Tri Dao FA v2 forward.
11. `flash_attn_bwd_bf16_sm80` — vendored Tri Dao FA v2 backward.
12. `rmsnorm_bwd_bf16_sm80` — RMSNorm backward (variance reuse via shared mem).
13. `layernorm_fwd_bf16_sm80` — LayerNorm forward (Welford reduction).
14. `silu_fwd_bf16_sm80` — SiLU forward (element-wise activation).
15. `conv2d_implicit_gemm_bf16_sm80` — CUTLASS Conv2dFprop on sm_80.

Wave 2 admission is gated by AC-1.1: the six first-wave kernels must
satisfy AC-1 (build cleanly + correctness shape exits 0) before any
Wave 2 kernel enters the build.

## Building

### Preserved binary boundary

The 15 non-ac4 binaries listed in
`workloads/portable_lcd/preserved_binaries_manifest.json` are preserved,
hash-bound inputs while the source delta is under review. The JSON authority
and `PRESERVED_BINARIES.md` are deterministic projections of the SHA-verified
frozen roster; their generator does not inspect local binary bytes. Check the
tracked projections without writing via:

```bash
.venv/bin/python tools/local_labels/build_preserved_binaries.py --check
.venv/bin/python tools/local_labels/gate_preserved_binaries_local.py
```

The first command proves authority regeneration. The second independently
checks the 15 local binary bytes. The shared ac4 binary is excluded from that
15-row projection because its two cells and fixed SHA-256 are documented and
gated separately in `ac4_mixed_long_sm80/PRESERVED_BINARY.md`. Do not rebuild,
clean, replace, or repin any of these bytes while the preserved-state contract
is active.

From the repository root:

The commands below describe the ordinary source-build interface after an
explicit preservation disposition. They are not authorized while the current
preserved-state contract is active.

```bash
# Build all kernel slugs in this subtree:
make -C workloads/portable_lcd

# Build a single kernel:
make -C workloads/portable_lcd/gemm_hmma_bf16_sm80

# Clean every slug:
make -C workloads/portable_lcd clean
```

Each per-kernel binary is named `<slug>_run` and lives under
`workloads/portable_lcd/<slug>/`. Three verifiers consume the
produced binaries and trace artifacts:

* `tools/build/verify_lcd_arch_slices.sh <slug>_run` — fatbin
  slice and forbidden-opcode discipline (sm_86 / sm_90 / sm_120
  base-arch slices present, architecture-conditional variants
  absent, compute_80 PTX fallback present, sm_120 SASS contains
  no tcgen05/TMEM-family opcodes).
* `tools/build/verify_lcd_sass_coverage.sh <slug>_run [<more>...]`
  — SASS mnemonic coverage across the suite (BAR.SYNC + memory
  fence family, plus BSSY / BSYNC / EXIT / RET and a warp-branch).
* `tools/build/verify_lcd_nvbit_trace.sh <slug>/traces` — NVBit
  trace artifact contract (non-empty manifest, parseable trace,
  per-CTA stream files present, non-zero traced instruction count).

## Running

Each `<slug>_run` exposes the AC-4 uniform CLI surface:

```
<slug>_run \
  [--correctness_shape | --timing_shape] \
  [--seed N] [--warmup N] [--reps N] \
  [--output_dir PATH] \
  [--allow-any-device]
```

The former `--enable-cupti-activity` collection path is retired by
D8. New timing truth is collected through the NCU label toolchain;
committed historical CUPTI evidence is validated against
`workloads/simbench_bw/schemas/historical/cupti_activity_sidecar.v1.schema.json`.

`--correctness_shape` runs the smallest shape-point that exercises
the kernel's instruction mix; the runner emits a per-row measurement
manifest under `<output_dir>/<slug>/` validated by
`workloads/simbench_bw/schemas/measurement_manifest.schema.json`
(schema_version 2). `--seed 42` produces bit-identical reduction
outputs across two clean process launches on the same host
(deterministic reduction discipline, AC-4 positive test).

## NVBit Tracing

The canonical tracer is a preserved artifact while H3 is active. Verify its
full SHA-256 and source the pinned environment in verify/export-only mode; do
not build or clean it in a Git worktree:

```bash
printf '%s  %s\n' \
  de47b3af6a18ae69287f61ff149dbdb03b4baf6ca8e33f03d2a2221a0482ee5d \
  tools/nvbit_tracer/tracer_tool/tracer_tool.so | sha256sum -c -
source tools/nvbit_tracer/pinned_build_env.sh
```

Then exercise the tracing tools end-to-end against a preserved portable LCD
binary:

```bash
PB_LIB="$PWD/.toolchain/pb/lib"
OUT="$PWD/workloads/portable_lcd/<slug>/traces"
mkdir -p "$OUT"
LD_LIBRARY_PATH="$PB_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  CUDA_INJECTION64_PATH="$PWD/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
  USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$OUT" \
  workloads/portable_lcd/<slug>/<slug>_run --correctness_shape --reps 1
```

The tracer emits the following layout under
`workloads/portable_lcd/<slug>/traces/`:

| Path | Purpose |
|------|---------|
| `dynamic_trace.pb` | Top-level trace manifest (small; per-launch metadata that `trace_printer` reads first). |
| `threadblocks/device_<D>/stream_<S>/kernel_<K>/d_<D>_s_<S>_k_<K>_<x,y,z>.pb` | Per-CTA instruction-stream protobuf, one file per launched CTA. |
| `extra_info/enhanced_execution_info.json` | Static SASS / control-flow analysis written by the trace_printer's enhancer pass. |
| `stats.csv` | One row per traced kernel launch with grid / block / thread counts and `total_insts` (the number of dynamic SASS instructions traced). |

The trace can be inspected interactively via
`tools/nvbit_tracer/tracer_tool/trace_printer
workloads/portable_lcd/<slug>/traces` and validated programmatically
first via `.venv/bin/python tools/nvbit_tracer/trace_admission.py
workloads/portable_lcd/<slug>/traces`, then via
`tools/build/verify_lcd_nvbit_trace.sh workloads/portable_lcd/<slug>/traces`.
The admission gate parses the root and instruction records instead of trusting
presence; the layout verifier additionally requires:
non-empty `dynamic_trace.pb`; `trace_printer` parses cleanly; at
least one non-empty per-CTA `.pb` under the nested `threadblocks/`
tree; `stats.csv` reports a non-zero `total_insts` summed across
all traced kernel rows. Typed diagnostics on failure:
`nvbit_dynamic_trace_missing`, `nvbit_dynamic_trace_empty`,
`nvbit_dynamic_trace_unparseable`, `nvbit_per_cta_pb_missing`,
`nvbit_per_cta_pb_layout_invalid` (a `d_*.pb` exists under
`threadblocks/` but not at the documented
`device_*/stream_*/kernel_*/d_*.pb` exact-depth path),
`nvbit_per_cta_pb_empty`, `nvbit_stats_csv_missing`,
`nvbit_zero_instruction_count`.

## Adding a New LCD-Portable Kernel

1. **Create the slug directory.** `mkdir
   workloads/portable_lcd/<new_slug>/`.
2. **Copy the per-slug Makefile template.** `cp Makefile.template
   <new_slug>/Makefile`, then replace `REPLACE_WITH_SLUG_NAME` with
   the new slug name.
3. **Author the kernel source.** `<new_slug>/<new_slug>.cu` holds
   the kernel + launch wrapper. Use only LCD-portable PTX:
   `mma.sync` (HMMA / IMMA / sparse `mma.sp`), `cp.async`,
   `mbarrier`, cooperative groups, `__shfl_sync` /
   `__ballot_sync` / `__activemask`, `__nv_bfloat16`, `__half2`,
   `%clock64` / `%globaltimer` / `%smid` / `%warpid` / `%laneid`.
4. **Author the runner.** `<new_slug>/main.cu` holds the AC-4
   uniform CLI + AC-3 manifest emission + correctness reference
   loop + timing loop.
5. **Verify the build.** `make -C workloads/portable_lcd/<new_slug>`
   should produce `<new_slug>_run`. Run
   `tools/build/verify_lcd_arch_slices.sh
   workloads/portable_lcd/<new_slug>/<new_slug>_run` to confirm the
   produced fatbin embeds `sm_86 + sm_90 + sm_120` ELF slices and
   a `compute_80` PTX section, with no `sm_90a` / `sm_120a` /
   `sm_120f` slices leaking in. Then smoke-trace the binary:
   the pinned-runtime invocation above with `<slug>` replaced by `<new_slug>`,
   then run
   `tools/build/verify_lcd_nvbit_trace.sh
   workloads/portable_lcd/<new_slug>/traces` to confirm the trace
   contract is met.
6. **Update the row in `roster.toml`.** The roster already
   contains all 15 LCD rows. When you land a new kernel, flip
   its row's `status` from `pending` to `active`. The standalone
   contract gate at
   `workloads/portable_lcd/verify_kernel_contracts.py` will then
   include the new row in its per-row + suite-level checks. Run
   `python3 workloads/portable_lcd/verify_kernel_contracts.py`
   from the repo root to confirm the row passes.

### Forbidden Surfaces

The repo-wide static guard at
`workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`
scans every active code path; the `common.mk` arch-token check
catches forbidden architecture-conditional build flags. Together
they reject:

* FP8 QMMA (`mma.sync.kind::e4m3` / `e5m2`), FP4 / FP6 / NVFP4
  MMA (`mma.sync.kind::f8f6f4` / `mxf4`).
* Thread-block clusters (`cluster_dim`, `__cluster_block_idx`),
  DSMEM, TMA (`cp.async.bulk` / `cp.async.bulk.tensor`).
* Hopper warpgroup MMA (`wgmma.*`), `setmaxnreg`, `tcgen05.*` /
  TMEM.
* NCU (`ncu` / `nv-nsight-cu-cli` / `.ncu-rep` / `.ncu-repz`).
* CUPTI Events / Metrics / Range Profiling / PM Sampling APIs
  (`cupti_events.h`, `cupti_metrics.h`, `cupti_profiler*`,
  `cupti_range_profiler*`, `cupti_pm_sampling*`).
* `nvidia-smi` clock / driver-mutation flags (`-lgc`, `-lmc`,
  `-rgc`, `-rmc`, `-pm`, `-pl`, `-c EXCLUSIVE_PROCESS`, `-mig`,
  `-ac`, `--lock-gpu-clocks`, `--lock-memory-clocks`).
* Kernel-module / driver-install operations (`modprobe nvidia`,
  `mokutil`, `dkms`, `NVreg_RestrictProfilingToAdminUsers`).
* Architecture-conditional build flags (`code=sm_90a`,
  `code=sm_120a`, `code=sm_120f`).

## References

* Plan: `plans/research/portable_lcd_workload_suite/plan.md`.
* CLAUDE.md: project-wide rules (no NCU, no GPU-driver-root, no
  privileged CUPTI surfaces).
* `workloads/simbench_bw/l1/bf16_hmma_gemm_traceable/` — the 85-LOC
  reference HMMA BF16 GEMM (raw inline `mma.sync` PTX, no CUTLASS
  dependency) that informed the first-wave HMMA kernel pattern.
* `workloads/simbench_bw/schemas/measurement_manifest.schema.json`
  — the schema-v2 manifest contract every runner satisfies.
* `tools/nvbit_tracer/README.md` — NVBit 1.8 host-build provenance
  + sm_120 binary-version workaround + concurrency model.
* `.claude/knowledge/INDEX.md` — repo knowledge router (first stop
  for CUDA / PTX / SASS / Blackwell / measurement questions).
