---
name: navigating-project-kb
description: Route GPU/CUDA/PTX/SASS/NVBit/simulator questions to the smallest tracked project source or optional knowledge card. Use when ownership is unclear; prefer a specialty skill when the mnemonic or workflow is specific.
---

# Project knowledge router

Read root `AGENTS.md`, then `docs/PROJECT_MEMORY.md` and `docs/STATUS.md` for
the current dual-lane project state (SM86 baseline and SM120/RTX 5090 alpha). The optional `.claude/knowledge/` corpus is not
present in this checkout; do not loop on missing cards. Use tracked source,
`docs/`, `plans/`, `references/`, and owner runbooks instead.

## Route by question

| Question | Skill/source |
| --- | --- |
| Native Rust simulator, replay, cards, event queue, L1/L2/DRAM | `designing-gpu-simulator`, `docs/ARCHITECTURE.md`, `docs/experimental_diagnostics.md` |
| SASS/cubin/NVBit/opcode/trace | `analyzing-sass-nvbit-traces`, `tools/nvbit_tracer/README.md` |
| PTX mnemonic or memory ordering | `interpreting-ptx-isa`, `references/NVIDIA_Manuals/ptx_isa/` |
| nvcc/fatbin/cubin/JIT trajectory | `compiling-cuda-artifacts`, `references/NVIDIA_Manuals/nvcc_guide/` |
| CUDA occupancy/coalescing/scheduling/synchronization | `optimizing-cuda-kernels`, `references/NVIDIA_Manuals/` |
| NVIDIA architectural history/patents | `reasoning-about-nvidia-architecture-history`, `references/NVIDIA_patents/` |
| SOL/campaign baseline | opt-in `run-sol-campaign` / `implementing-sol-baselines`; preserve retained data |

## Non-negotiable project boundaries

NCU/CUPTI values are independent labels, never same-cell replay inputs.
Architecture cards do not inherit missing values from another architecture.
Fixed SASS replay supports trace-safe resource changes only; value/address/CFG,
atomic-order, or participant changes need a functional frontend or recapture.
Literature numbers and external simulator overlap are not local NCU ground
truth. Published Accel-Sim aggregates are active accuracy-ceiling and
throughput/speed-floor targets recorded in
`docs/accelsim_literature_lower_bound_manifest_20260928.json`;
executable overlap remains parser/direct-cycle evidence only.
Root `AGENTS.md` controls permissions and workflow; this file only routes
knowledge and does not create gates, review loops, or tool bans.
