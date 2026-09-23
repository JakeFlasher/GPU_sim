---
name: navigating-project-kb
description: Routes the agent to the correct distilled knowledge card before answering CUDA, PTX, SASS, NVBit, CUPTI, Blackwell, Hopper, NVIDIA-architecture, measurement, or simulator-prior-art questions in this GPU simulator project. Use as the meta-router when a prompt could match multiple specialty skills, when the agent is unsure which subcorpus owns the topic, or when a question explicitly cites the project knowledge base, INDEX.md, or one of the specialty skills. Triggers also include phrases like "where is X documented", "which card explains Y", "find the right knowledge card", and "look this up in the KB". Always prefer this skill over generic training priors when the prompt touches project GPU/NVIDIA topics.
---

# Navigating the Project Knowledge Base

This is the meta-router. It exists so the agent can pick the right specialty skill *or* the right card directly, without re-deriving the routing every time.

## When to invoke

- Prompt mentions GPU/CUDA/PTX/SASS/Blackwell/Hopper/NVIDIA, but the specific specialty (PTX vs SASS vs measurement vs patents) is unclear.
- User explicitly asks "where is X documented" / "which card explains Y" / "look this up in the KB".
- An earlier specialty skill turned out to be the wrong owner.

## What this skill does

1. Names the specialty skills and what each owns.
2. Tells the agent how to pick one.
3. Points to the smallest useful source set.

## The specialty skills

| Skill | Owns | Trigger keywords |
|---|---|---|
| `interpreting-ptx-isa` | PTX instruction semantics, mnemonics, inline PTX | `mma`, `wgmma`, `tcgen05`, `cp.async`, `mbarrier`, `ldmatrix`, `bar.sync`, `cvt`, `setp`, `shfl.sync`, `ptxas` |
| `analyzing-sass-nvbit-traces` | SASS opcodes, tooling (cuobjdump/nvdisasm/cu++filt/nvprune), NVBit instrumentation | `SASS`, `nvbit`, `cuobjdump`, `nvdisasm`, `unknown opcode`, `binary instrumentation` |
| `tuning-blackwell-sm120-kernels` | sm_120 / sm_100 / Blackwell-specific tuning, cluster ceilings, TMEM, FP4/MX | `sm_120`, `sm_120a`, `sm_100`, `Blackwell`, `RTX PRO 6000`, `B200`, `GB202`, `TMEM`, `tcgen05.mma`, `NVFP4`, `MX format` |
| `validating-nonroot-gpu-measurement` | RETIRED (historical no-NCU/no-root contract; kept for archaeology only — do NOT apply its rules) | — |
| `optimizing-cuda-kernels` | Coalescing, occupancy, register file, warp scheduling/divergence, sync, memory hierarchy | `coalesc`, `occupancy`, `__launch_bounds__`, `register pressure`, `bank conflict`, `warp scheduler`, `cluster.sync`, `DSMEM`, `L2 persistence` |
| `compiling-cuda-artifacts` | nvcc flags, fatbin, PTX/cubin trajectory, separate compilation | `nvcc`, `fatbin`, `cubin`, `-gencode`, `-arch`, `-Xptxas`, `-keep`, `-lineinfo`, `JIT` |
| `reasoning-about-nvidia-architecture-history` | Tesla→Blackwell evolution, patent provenance, three rewrites, competitor IP | `patent`, `Fermi rewrite`, `Volta rewrite`, `Hopper rewrite`, `evolution`, `historical`, `architectural codename`, `tcgen05` history |
| `designing-gpu-simulator` | ACTIVE design skill — building the event-driven Rust simulator and its NCU comparison path; NVArchSim cards are prior-art background | `simulator`, `event-driven`, `nvasim`, `modeling paper`, `NVArchSim`, `GPGPU-Sim`, `Accel-Sim`, `sim/`, `sim-trace`, `sim-engine`, `sim-sm`, `sim-mem`, `prost`, `event queue`, `control bits sidecar`, `L1`, `MSHR`, `cache-op`, `L2`, `slice hash`, `DRAM`, `MemRequest/MemResponse`, `livelock`, `request/response inversion`, `Rust workspace` |

## How to pick one

1. **Specific instruction or mnemonic?** → `interpreting-ptx-isa` (or `analyzing-sass-nvbit-traces` if SASS layer).
2. **Tuning a specific kernel for SM120/SM100 Blackwell?** → `tuning-blackwell-sm120-kernels`.
3. **Measurement and label questions?** Start with root `CLAUDE.md`, the active
   `tools/local_labels/` runbook/schemas, and any cited NVBit tracing docs. The
   `measurement/` cards and `validating-nonroot-gpu-measurement` skill are retained
   historical references; they do not define current permission or capture policy.
4. **General CUDA performance (occupancy, coalescing, divergence, sync, memory)?** → `optimizing-cuda-kernels`.
5. **Build / compilation / fatbin / -gencode question?** → `compiling-cuda-artifacts`.
6. **"Why does feature X exist", "which patent introduced Y", "Tesla → Blackwell history"?** → `reasoning-about-nvidia-architecture-history`.
7. **Designing/implementing the ACTIVE Rust simulator (sim/) or consulting simulator prior art?** → `designing-gpu-simulator`.
8. **None of the above match cleanly?** Read `.claude/knowledge/INDEX.md` for the full inventory.

## Scope boundary

This router is not a measurement, privilege, or simulator permission system.
Root `CLAUDE.md` governs agent workflow. The stable scientific boundaries are:

1. **Targets-never-features:** NCU and other hardware values are labels/validation
   targets only; they never enter simulator inputs, architecture constants, or
   same-cell calibration.
2. Capture and label work uses the relevant tools, schemas, and experiment
   metadata. A missing optional historical artifact does not stop unrelated work.
3. Architecture-local facts come from the selected architecture card or direct
   evidence. A research gap is unavailable/unsupported, never an sm_86 fallback.

If a card or imported skill still says "NCU is forbidden" or "no clock locking",
treat it as stale historical text and do not apply it.

## When in doubt

`.claude/knowledge/INDEX.md` is the comprehensive topic-router inventory.
`.claude/knowledge/patents/INDEX.md` is the 24-card sub-router for architectural-history questions.
`.claude/knowledge/patents/problem-playbook.md` is the question-driven sister router for that subcorpus.

## Mechanical lookup

If the prompt cites a specific card path like `.claude/knowledge/X/Y.md`, just open it directly.
If it cites a referenced source like `references/NVIDIA_Manuals/...`, use the corresponding skill above based on subdirectory.
