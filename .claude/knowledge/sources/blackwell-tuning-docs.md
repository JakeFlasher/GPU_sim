---
Use when: looking up Blackwell (sm_100 / sm_100a) tuning guidance for TCGen05 / TMEM, NVFP4 block-scaling, cluster / CGA, DSMEM peer access, TMA box descriptors, 5th-generation tensor cores, or HBM3e bandwidth budgeting on the B200 scoring target.
Primary raw source: references/NVIDIA_Manuals/blackwell_tuning/00_index.md
Supporting sources: references/NVIDIA_Manuals/blackwell_compatibility/; references/NVIDIA_Manuals/cuda_guide/; references/NVIDIA_Manuals/ptx_isa/
Repo touchpoints: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; CLAUDE.md
Patents anchored here: None
Out of scope: Pre-Blackwell architectures (Hopper / Ampere / Ada / Volta) are covered by `references/NVIDIA_Manuals/cuda_guide/` and `references/NVIDIA_Manuals/ampere_tuning/`. For sm_120 (Blackwell consumer; RTX 5060) vs sm_100a (Blackwell datacenter; B200) divergence specifics per pattern, see the body of each `patterns/<name>.md` card.
Version family: blackwell-sm_100a (B200) / blackwell-sm_120 (consumer)
Default skill: tuning-blackwell-sm120-kernels
Also relevant skills: implementing-sol-baselines
Routing tags: blackwell; sm_100; sm_100a; sm_120; tcgen05; tmem; nvfp4; dsmem; tma; cluster; b200; hbm3e
Latency authority: paper
---

# Blackwell Tuning (Citation Stub)

## Use when

Looking up Blackwell tuning notes for: TCGen05 instructions / TMEM allocations, NVFP4 / FP8 block-scaling layouts, cluster-launch attributes, DSMEM peer access, TMA box descriptors, 5th-generation tensor-core opcodes (`tcgen05.mma`, `tcgen05.cp`), HBM3e bandwidth budgeting.

## Pointer

Local mirror: `references/NVIDIA_Manuals/blackwell_tuning/` (refreshed by `references/NVIDIA_Manuals/scripts/crawler_multiTOC.py`).

This is a citation gateway; for per-pattern arch-portability discipline (sm_120 ↔ sm_100a) see each `patterns/<name>.md` and `problems/<slug>.md` card.

## SM family map

- **sm_100 / sm_100a** -- Blackwell datacenter (GB100 die; B200, B100). Scoring target. Bandwidth per the paper: 8 TB/s HBM3e.
- **sm_120** -- Blackwell consumer (GB206 die; RTX 5060). Local compile / smoke target only.
- The `sm_100a` suffix is architecture-conditional -- opt-in for TCGen05 / TMEM instructions via `-gencode arch=compute_100a,code=sm_100a`. The B200 problem packager (`sol-execbench/src/sol_execbench/driver/problem_packager.py:120`) auto-injects this flag when `target_hardware: ["B200"]`.

## Cross-references

- `CLAUDE.md` § Architecture Portability Discipline -- binding rules for cross-arch kernel design.
- `patterns/` -- per-pattern Blackwell specialisations.
