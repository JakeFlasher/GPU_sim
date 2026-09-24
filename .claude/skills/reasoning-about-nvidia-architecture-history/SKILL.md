---
name: reasoning-about-nvidia-architecture-history
description: Patent-grounded NVIDIA GPU architectural-evolution context from Tesla through Blackwell and competitor tensor-near-storage IP. Use for historical feature origins, generation deltas, or patent questions; current simulator operations still follow root AGENTS.md and active project docs.
---

# Reasoning About NVIDIA Architecture History

This skill owns the 24-card patent-and-evolution corpus distilled from
`references/NVIDIA_patents/NVIDIA_patentd.md`. It is **historical context only**;
root `AGENTS.md` and the active evidence/host authorities govern current
operations.

## Two routers, pick one

- **Topic-driven**: `.claude/knowledge/patents/INDEX.md` (organized by GPU generation and thematic axis).
- **Question-driven**: `.claude/knowledge/patents/problem-playbook.md` (organized by question type — "what changed between X and Y?", "why does feature X exist?", "did Microsoft/Intel/Meta do something similar?").

## Cards

### Per-generation (11 cards)

| Generation | Year | Card |
|---|---|---|
| Tesla (G80) | 2006 | `arch-tesla.md` |
| Fermi (GF100) | 2010 | `arch-fermi.md` |
| Kepler (GK110/210) | 2012 | `arch-kepler.md` |
| Maxwell (GM204) | 2014 | `arch-maxwell.md` |
| Pascal (GP100) | 2016 | `arch-pascal.md` |
| Volta (GV100) | 2017 | `arch-volta.md` |
| Turing (TU102) | 2018 | `arch-turing.md` |
| Ampere (GA100/GA102) | 2020 | `arch-ampere.md` |
| Ada Lovelace (AD102) | 2022 | `arch-ada.md` |
| Hopper (GH100) | 2022 | `arch-hopper.md` |
| Blackwell (GB200/GB202) | 2024 | `arch-blackwell.md` |

### Thematic / cross-generation (7 cards)

| Theme | Card |
|---|---|
| Three-rewrites thesis (Fermi/Volta/Hopper) | `three-rewrites.md` |
| Execution model (warp lockstep → ITS → cluster → single-thread issue) | `evolution-execution-model.md` |
| Data supply (RF/collector → cp.async → TMA → TMEM) | `evolution-data-supply.md` |
| Memory side (passive cache → tunable domain → DSM → TMEM) | `evolution-memory-side.md` |
| Synchronization (busy-bit → scoreboard → mbarrier → transaction barrier) | `evolution-synchronization.md` |
| Precision (FP32 → FP16 → BF16/TF32 → FP8 → FP4/MX) | `evolution-precision.md` |
| Interconnect (crossbar → NVLink → NVSwitch → NV-HBI / NVL72) | `evolution-interconnect.md` |

### Auxiliary indexes (5 cards)

| Card | Purpose |
|---|---|
| `INDEX.md` | Topic-driven router (sub-router for this corpus) |
| `problem-playbook.md` | Question-driven sister router |
| `glossary.md` | One-line definitions for `BSSY`, `DSMEM`, `HMMA`, `TMEM`, `WGMMA`, `NVFP4`, `mbarrier.complete_tx`, etc. |
| `patent-family-map.md` | Every cited patent (NVIDIA + competitor) with dates, anchored generation |
| `sass-evolution-timeline.md` | SASS opcode introduction + retirement per generation |

### Competitor IP (1 card)

| Card | Purpose |
|---|---|
| `competitor-tensor-storage.md` | Microsoft (US10372456B2), Intel (EP4020173B1), Meta (US11922306B2), Micron (US10956315B2/US12229044B2), AMD (US11609785B2) — same tensor-near-storage pressure, different names |

## The three rewrites (corpus thesis)

| Patent | Generation | Year | What it rewrites |
|---|---|---|---|
| `US9639479B2` | Fermi | 2010 | **Memory side** becomes a software-observable, software-tunable hardware domain |
| `US10338919B2` | Volta | 2017 | **Execution path**: HMMA datapath peels out of the general FPU |
| `US20230289189A1` | Hopper | 2022 | **Cooperation boundary**: shared-memory-style CTA cooperation extends to a cluster (DSM) |

Blackwell extends rewrite #3 (TMEM, dual-die unified GPU, FP4 block-scaling) — it is **NOT** a fourth rewrite at the same level.

## How to use this corpus

1. **"Why does X exist?"** — open `problem-playbook.md` § "Why does feature X exist?" → routed card.
2. **"Which patent introduced X?"** — open `patent-family-map.md` (alphabetical patent index + by-generation lookup).
3. **"What's the corpus thesis?"** — read `three-rewrites.md`.
4. **"Compare Tesla→Blackwell on axis Y"** — open the relevant `evolution-Y.md` card.
5. **"Tell me about generation N"** — open `arch-N.md`.
6. **"Did competitors do similar?"** — open `competitor-tensor-storage.md`.

## Source-of-truth

- Translated narrative: `references/NVIDIA_patents/NVIDIA_patentd.md` (513 lines, 16 inline figure images).
- Per-generation Chinese SM specs (supporting source): `references/NVIDIA_patents/gpu-arch-wiki/user/{1.fermi,1.kepler,1.maxwell,1.pascal,2.volta,3.Turing,4.ampere,4.ga102,5.ada,6.hopper,7.blackwell}.md` plus `0.table.md` and `NV微架构梳理.md`.
- Master patent index: `patent-family-map.md` (every cited patent verified against Google Patents 2026-05-02).

## Hard rules

- This corpus is **historical/architectural-context only**. Root `AGENTS.md`, active
  schemas, and owner runbooks govern current privilege, evidence, and
  targets-never-features rules; this corpus cannot authorize operations. Mentions
  of measurement or privileged primitives explain historical design context and
  are not instructions for capture, driver, or host changes.
- The "three rewrites" thesis is the **source document's own analytic synthesis**, not an officially declared NVIDIA roadmap. Cite it as such.
- Patents listed are from the snapshot 2026-05-02. New filings or continuations are out of scope until added to `patent-family-map.md`.

## Cross-references

- Current PTX-level realization → `interpreting-ptx-isa`.
- Current SASS-level realization → `analyzing-sass-nvbit-traces`.
- Blackwell-specific tuning → `tuning-blackwell-sm120-kernels`.
- Generic CUDA optimization (the post-rewrite programming model) → `optimizing-cuda-kernels`.

## Cite consulted files

End every reply with `## Knowledge Consulted`.
