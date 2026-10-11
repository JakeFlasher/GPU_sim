# NVIDIA Manuals Problem Playbook

Use this file to jump from an engineering problem to the most relevant
NVIDIA manual subfolder under `references/NVIDIA_Manuals/`. Each subfolder is a
locally-crawled mirror of the official NVIDIA documentation page named
in its frontmatter.

**The mirror is the authoritative source of record** for any question
about CUDA semantics, PTX/SASS behavior, NCU metric definitions, or
SM120 (Blackwell) specifics. Use prior training knowledge only to
generate search terms or to summarize what the manual says — not as
the source of record.

## How to use this file

1. Find the row in sections 1–12 below that matches your problem.
2. Open the listed landing file first. **Caveat: `00_index.md` is not
   always a real top-level TOC.** Known cases:
   - `cuda_guide/` → `00_index.md` is the real top-level TOC, but the
     folder has ≈653 files, so grep is almost always faster.
   - `cutlass_cute/` → `00_index.md` is the CUTLASS landing, but the
     folder has ≈716 files. Grep by term, not by browse.
   - `cuda_tile_ir/` → same story; ≈295 files.
   - All other subfolders: `00_index.md` is the real TOC and is cheap
     to read end-to-end.
3. Grep within the folder for the specific keyword (instruction name,
   metric name, flag, register, etc.).
4. Read the matching numbered section file in full before answering.

Most small subfolders use the convention `NN_section.md` where `NN`
matches the chapter number in the official docs.

---

## 1. PTX Instruction Semantics, Encoding, Or Behavior

Use this when you need to know what a PTX instruction does, what its
operands mean, what state spaces it can address, or what the memory
consistency model says about it.

- `references/NVIDIA_Manuals/ptx_isa/` — authoritative PTX ISA reference
  - `09_instruction-set.md` — every PTX instruction with semantics
    (this file is very large; grep for the mnemonic)
  - `08_memory-consistency-model.md` — fences, acquire/release, scopes
  - `05_state-spaces-types-and-variables.md` — `.global`, `.shared`, `.local`, `.param`, `.const`, `.tex`
  - `06_instruction-operands.md` — operand modes, addressing
  - `10_special-registers.md` — `%tid`, `%ctaid`, `%smid`, `%clock*`, `%laneid`, etc.
  - `03_ptx-machine-model.md` — the abstract machine the ISA assumes
  - `13_release-notes.md` — PTX ISA version history; use this to find
    when a feature (e.g. `cp.async.bulk`, `wgmma`, `tensormap`) landed

Pair with `cuda_inline_ptx/01_using-inline-ptx-assembly-in-cuda.md`
only when you need to write or read inline `asm("...")` blocks inside
`.cu` sources (e.g., the D6 tensor-core microbenchmarks under
`workloads/d6_microbenchmarks/`).

## 2. SASS Disassembly, Cubin Inspection, Or Binary Layout

Use this when you need to disassemble a `.cubin`/`.fatbin`, inspect
section headers, or look up a SASS opcode.

- `references/NVIDIA_Manuals/cuda_binary_utility/`
  - `02_cuobjdump.md` — extract cubins/PTX from fat binaries
  - `03_nvdisasm.md` — disassemble cubins to SASS, control-flow graphs
  - `04_instruction-set-reference.md` — **the SASS instruction reference
    per arch** (the canonical source for mnemonics surfaced by the NVBit
    tracer in `nvbit_tracer/`)
  - `05_cu-filt.md` — demangling
  - `06_nvprune.md` — strip unused arches from a fat binary

## 3. NCU Metric Definitions, Profiling Workflow, Or Report Internals

Use this when validating the simulator against hardware. Validation is
NCU-only and metric-surface-aware (see `CLAUDE.md` rule 7).

The local `ncu_guide/` mirror has been removed (2026-05-12) because
follow-up experiments will not consult it. For NCU/Nsight Compute
documentation, fetch from <https://docs.nvidia.com/nsight-compute/>
directly (Profiling Guide, Nsight Compute CLI, Python Report Interface,
Occupancy Calculator Python Interface).

When a metric name comes back from NCU and you do not know what it
counts:

1. Search <https://docs.nvidia.com/nsight-compute/ProfilingGuide/> for
   the **full metric name** first (e.g. `sm__cycles_active.sum`).
2. If no hit, search the **base name only** (e.g. `sm__cycles_active`).
   NCU metrics have base names plus rollup suffixes (`.sum`, `.avg`,
   `.min`, `.max`, `.pct_of_peak_*`, etc.), and the docs usually
   describe the base only.
3. If still no hit, read the "Metrics Structure" / metric naming
   discussion in the Profiling Guide to understand how the name is
   composed (`unit__op_suffix`) and re-derive the base.
4. Cross-check against `tools/ncu/read_ncu_report.py` (canonical metric
   surface source) and `docs/validation-metric-manifest.toml` (frozen
   manifest) to see whether the metric is already in the pipeline.

Note the semantic pitfalls already documented in `CLAUDE.md`:
`l1_hits + l1_misses` is a **sector** aggregate, not a request count;
`active_cycles` is issue-driven, not the same as `sm__cycles_active.avg`.

## 4. SM120 / Blackwell Behavior (Cloud GeForce RTX 5090 Blackwell, The Validation Target)

**Blackwell semantics live in more than the `blackwell_*` guides.**
Route by question type — the `blackwell_tuning/` and
`blackwell_compatibility/` files are very short (one real content file
each) and mostly cover tuning guidance and forward-compat rules, not
the full programming or instruction model.

**Tuning, throughput, occupancy, L1/shared partitioning, compat rules,
forward-compat / JIT behavior:**
- `references/NVIDIA_Manuals/blackwell_tuning/01_nvidia-blackwell-tuning-guide.md`
- `references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md`

**Blackwell-era programming-model features** (thread block clusters,
distributed shared memory, TMA / tensor memory access, async copy,
`mbarrier`, cluster launch APIs) — see **section 5** below for the
full routing, but in short:
- `references/NVIDIA_Manuals/cuda_guide/` for the CUDA C++ surface
- `references/NVIDIA_Manuals/ptx_isa/` for the instruction-level surface

**Blackwell PTX / SASS instruction semantics** (`cp.async.*`,
`ldmatrix`, `stmatrix`, `mbarrier.*`, `st.async`, `wgmma`, `tensormap`,
etc.):
- `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md` — authoritative
  per-instruction semantics; grep for the mnemonic
- `references/NVIDIA_Manuals/ptx_isa/08_memory-consistency-model.md` — fences,
  acquire/release scopes (critical for async copy / `mbarrier` reasoning)
- `references/NVIDIA_Manuals/ptx_isa/13_release-notes.md` — when a given PTX
  feature first appeared, which tells you the earliest SM the
  instruction is valid on
- `references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md`
  — the SASS side

If a Blackwell question is not answered by any of the above, treat it
as genuinely unknown and say so. Do not fall back on Ampere/Hopper
assumptions.

## 5. Clusters, Distributed Shared Memory, TMA, Async Copy, mbarrier

These are the Blackwell-era primitives the simulator has to model
correctly, and `simulator/README.md` calls several of them out as
currently-unsupported opcodes. They do not have a single home — the
CUDA C++ surface, the PTX surface, and the tuning guides each cover
different parts.

**CUDA C++ programming model:**
- `references/NVIDIA_Manuals/cuda_guide/02-01-10_intro-to-cuda-cpp-thread-block-clusters.md`
- `references/NVIDIA_Manuals/cuda_guide/05-04-03-01_thread-block-cluster.md`
- `references/NVIDIA_Manuals/cuda_guide/02-02-03-08_distributed-shared-memory.md`
- Additional `cuda_guide/` grep terms: `cluster`,
  `cooperative_groups::cluster`, `cudaLaunchKernelEx`, `cp.async`,
  `mbarrier`, `cuda::barrier`, `memcpy_async`, `TMA`, `tensormap`

**PTX instruction semantics (the layer the simulator decodes):**
- `ptx_isa/09_instruction-set.md` — grep for: `cp.async`,
  `cp.async.bulk`, `cp.async.bulk.tensor`, `mbarrier.`, `st.async`,
  `tensormap`, `ldmatrix`, `stmatrix`, `wgmma`, `barrier.cluster`
- `ptx_isa/08_memory-consistency-model.md` — fence/scope semantics
  (required to reason about async-copy completion, `mbarrier` waits,
  and cross-cluster ordering)

**SASS side (what the NVBit tracer surfaces):**
- `cuda_binary_utility/04_instruction-set-reference.md`

**Arch-level framing (what lowers where, why, and at what cost):**
- `blackwell_tuning/01_nvidia-blackwell-tuning-guide.md`

When in doubt, start at the PTX layer — the simulator models PTX
semantics, not the higher-level C++ abstractions.

## 6. Trace / NVBit Tracer / Unsupported SASS Opcode / Decode Failures

Use this when:
- The NVBit tracer under `nvbit_tracer/` emits an opcode the Rust
  simulator does not know.
- `nvasim --trace-audit` fails or flags an instruction.
- You need to understand what a SASS mnemonic lowers from, or what it
  does.
- You need to inspect a cubin/fatbin to confirm what PTX/SASS the
  compiler actually generated.

First stops, in order:

1. Cross-reference `simulator/README.md` (and any `KNOWN_UNSUPPORTED`
   list in the simulator workspace) to see if the opcode is already
   on the known-unsupported list. If so, this is a modeling question,
   not a documentation question.
2. `references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md`
   — per-arch SASS mnemonic reference. Grep for the mnemonic.
3. `references/NVIDIA_Manuals/cuda_binary_utility/03_nvdisasm.md` — disassemble
   the affected cubin to SASS and confirm the mnemonic in context.
4. `references/NVIDIA_Manuals/cuda_binary_utility/02_cuobjdump.md` — extract
   cubins/PTX from a fatbin when you need to diff against what the
   tracer saw.
5. `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md` — PTX-side semantics
   of the instruction (what state spaces it touches, what operands it
   takes, what fence/scope it implies).
6. If the opcode is Blackwell-only:
   `blackwell_compatibility/01_blackwell-architecture-compatibility.md`
   for PTX/SASS forward-compat and JIT implications, plus
   `blackwell_tuning/01_nvidia-blackwell-tuning-guide.md` for
   throughput context.
7. If the instruction is part of the async-copy / `mbarrier` / cluster
   / TMA family: also consult **section 5** above.

Pair with `references/papers/knowledge_base/problem_playbook.md` section 1
("Simulator Is Wrong Or Mismatched Against Hardware") when the modeling
gap is not just a decode gap but a semantic-model gap.

## 7. Ampere (SM80) Reference For Cross-Arch Comparison

Use this when modeling decisions reference Ampere — most academic GPU
simulator papers, including those in `references/papers/knowledge_base/active/`,
target SM75/SM80, and the `tools/ncu/` pipeline was originally tuned
against an Ampere baseline.

- `references/NVIDIA_Manuals/ampere_tuning/01_nvidia-ampere-gpu-architecture-tuning-guide.md`
- `references/NVIDIA_Manuals/ampere_compatibility/01_nvidia-ampere-gpu-architecture-compatibility.md`

## 8. CUDA Programming Model Semantics (general, non-Blackwell-specific)

Use this for questions about kernels, blocks, warps, streams, unified
memory, atomics, cooperative groups, dynamic parallelism, graph APIs,
or anything in the CUDA C++ surface that is not tied to a specific
Blackwell-era primitive (those go in section 5).

- `references/NVIDIA_Manuals/cuda_guide/` — full CUDA Programming Guide (≈834 files)
  - `00_index.md` — top-level TOC; skim this, then grep
  - Useful grep entry points:
    - thread / block / cluster / warp model → search `01-02-02_*`,
      `02-02-*` or grep for `warp`, `block`, `cluster`
    - memory spaces and caches → `02-02-03-*`, `02-02-04-*` or grep
      for `shared memory`, `constant memory`, `local memory`
    - synchronization, fences, atomics → grep `cuda_guide/` for
      `__syncthreads`, `__threadfence`, `atomic`
    - streams, events, graphs → grep `cuda_guide/` for `cudaStream`,
      `cudaGraph`, `cudaEvent`
    - unified memory → `01-02-03-03_unified-memory.md`,
      `02-01-03-01_intro-to-cuda-cpp-unified-memory.md`

For questions that are really about performance tuning on top of the
programming model, jump to section 9.

## 9. CUDA Performance Tuning, Memory Coalescing, Occupancy, Bank Conflicts

Use this when interpreting NCU bottlenecks, choosing block sizes, or
deciding which optimization a workload most needs.

- `references/NVIDIA_Manuals/best_practices/`
  - `09_performance-metrics.md` — what the basic metrics mean
  - `10_memory-optimizations.md` — coalescing, shared memory bank conflicts
  - `11_execution-configuration-optimizations.md` — occupancy, block size
  - `12_instruction-optimization.md` — throughput, special functions
  - `13_control-flow.md` — branch divergence, predication
  - `04_application-profiling.md` — profiling methodology
  - `08_optimizing-cuda-applications.md` — APOD loop

Pair with `references/papers/knowledge_base/problem_playbook.md` section 1
("Simulator Is Wrong Or Mismatched Against Hardware") when an NCU
number disagrees with the simulator and you need both NVIDIA's stated
semantics and the academic modeling perspective.

## 10. nvcc Build Options, Compilation Phases, Separate Compilation

Use this when building the workloads under `workloads/` or the
validation binaries under `simulator/validation/`.

- `references/NVIDIA_Manuals/nvcc_guide/`
  - `04_nvcc-command-options.md` — every `-X*`, `-arch=`, `-code=`,
    `-gencode`, `-Xptxas`, `-Xnvlink`
  - `05_gpu-compilation.md` — virtual vs real arch, fatbin assembly
  - `06_using-separate-compilation-in-cuda.md` — relocatable device code
  - `02_compilation-phases.md` — `.cu` → PTX → cubin pipeline
  - `03_the-cuda-compilation-trajectory.md` — full trajectory diagram

The single-CTA validation build script
`simulator/validation/build_validation_kernels.sh` (which compiles
selected ai-kernels with `-arch=sm_120 -O2`) and the cluster /
D6 microbenchmark Makefiles under `workloads/cluster_microbench/`
and `workloads/d6_microbenchmarks/` are where most nvcc-flag
questions originate.

## 11. CUTLASS / CuTe (GEMM, Tensor Core Patterns)

Use this when looking at the `gemm_cublas` workload or any tensor-core
related modeling question that needs concrete kernel layouts as a
reference.

- `references/NVIDIA_Manuals/cutlass_cute/` (≈716 files — grep, do not browse)
  - Glossary / terminology: `0347_cutlass-terminology.md`
  - Device-wide GEMM API landing: `0629_device-wide-gemm-api.md`
  - **High-hit grep terms** (prefer these over `mma::SM90_*` /
    `mma::SM100_*`, which are low-coverage):
    - `cute::Layout` — the core layout type
    - `GemmUniversalAdapter` — the device-wide GEMM entry point
    - `Collective Mainloops` — the mainloop abstraction
    - `Blackwell SM100 GEMMs` — SM100-specific GEMM kernels
    - `TMA` — tensor memory access
    - `tiled_copy`, `tiled_mma` — tile-level operations
    - `ThreadblockSwizzle`, `EpilogueFunctor` — swizzle / epilogue
  - Start from `00_index.md` only to *confirm the term lives in this
    folder*, then grep.

## 12. Rarely-Needed Subfolders

These are crawled but unlikely to come up in this repo's day-to-day
work. Listed here so the agent knows they exist and can rule them out
quickly — do NOT open them unless the problem explicitly demands it.

- `references/NVIDIA_Manuals/ptx_writer_guide/` — only for hand-writing PTX with
  the ABI in mind. Not relevant unless writing a PTX-only kernel.
- `references/NVIDIA_Manuals/ptx_compier_api/` — `libnvptxcompiler` JIT API. No
  current consumer in this repo.
- `references/NVIDIA_Manuals/nvvm_ir/` — NVVM IR spec, lower-level than PTX. Only
  relevant if working below PTX, which `nvasim` does not.
- `references/NVIDIA_Manuals/cuda_tile_ir/` (≈295 files) — CUDA Tile IR. New
  abstraction; no current consumer in this repo.
- `references/NVIDIA_Manuals/cutile_python/` — cuTile Python bindings. No current
  consumer in this repo.

If you find yourself wanting to open these, double-check the question
is not better answered by `cuda_guide/`, `ptx_isa/`, `cutlass_cute/`,
or `cuda_binary_utility/` first.

---

## First Reading Order For A New Agent

If you are new to this repo and have never used the NVIDIA manuals
mirror, skim in this order to build a mental map:

1. `references/NVIDIA_Manuals/blackwell_tuning/01_nvidia-blackwell-tuning-guide.md`
   — the target architecture
2. `references/NVIDIA_Manuals/ptx_isa/03_ptx-machine-model.md` and
   `09_instruction-set.md` (header/skim, not full read) — the ISA you
   are modeling
3. <https://docs.nvidia.com/nsight-compute/ProfilingGuide/> — the
   validation tool (local `ncu_guide/` mirror was removed 2026-05-12)
4. `references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md`
   (skim) — the SASS surface the NVBit tracer captures
5. `references/NVIDIA_Manuals/best_practices/09_performance-metrics.md` — the
   vocabulary for performance discussions
6. `references/NVIDIA_Manuals/cuda_guide/02-01-10_intro-to-cuda-cpp-thread-block-clusters.md`
   and `02-02-03-08_distributed-shared-memory.md` — the Blackwell-era
   primitives that dominate current modeling gaps

After this, return here and use sections 1–12 as a lookup table.

---

## When This Playbook Is Wrong

If you discover a topic that is not routed correctly above, or a folder
that turned out to be more (or less) useful than this file says, update
this playbook in the same commit as the work that exposed the gap.
Treat stale routing entries as bugs.
