# NVIDIA GPU Architecture & Patent Knowledge — Index

**This file is the first stop for any Claude session reasoning about NVIDIA
GPU architectural evolution, the patents that bound each generation, and
how those decisions show up in PTX/SASS, the simulator, or modern CUDA
programming.**

It is a router, not a knowledge base. It maps a question type to the
shortest authoritative card. Open the listed card in full, fall through
to the deep-dive sources only when the card does not answer your
question.

## What this corpus is

Two source documents, distilled into 24 cards:

- `NVIDIA_patentd.md` — patent-grounded narrative of architectural
  evolution from Tesla (2006) to Blackwell (2024). Why each generation
  rewrote what; which patents anchor each rewrite.
- `gpu-arch-wiki/user/*.md` — per-generation SM specifications: core
  counts, GPC/TPC layout, register file / shared memory / L1 / L2
  capacities, memory interface.

Each card combines both views so a future agent gets both the
"what hardware existed" and the "why it was built that way."

## Routing

### Per-generation deep dives (10 cards)

When you need facts about a specific GPU generation — SM organization,
launch year, key patents filed for that era, what changed vs. the prior
generation:

- Tesla (2006, G80) — `arch-tesla.md`
- Fermi (2010, GF100) — `arch-fermi.md`
- Kepler (2012, GK110/210) — `arch-kepler.md`
- Maxwell (2014, GM204) — `arch-maxwell.md`
- Pascal (2016, GP100) — `arch-pascal.md`
- Volta (2017, GV100) — `arch-volta.md`
- Turing (2018, TU102) — `arch-turing.md`
- Ampere (2020, GA100/GA102) — `arch-ampere.md`
- Ada Lovelace (2022, AD102) — `arch-ada.md`
- Hopper (2022, GH100) — `arch-hopper.md`
- Blackwell (2024, GB200/GB202) — `arch-blackwell.md`

### Thematic / cross-generation cards (10 cards)

When the question crosses generations:

- The three-rewrites thesis (Fermi/Volta/Hopper) — `three-rewrites.md`
- All cited patents in one searchable index — `patent-family-map.md`
- Execution model: warp lockstep → ITS → cluster — `evolution-execution-model.md`
- Data-supply chain: RF/collector → cp.async → TMA → TMEM — `evolution-data-supply.md`
- Memory-side: passive cache → tunable domain → DSM → TMEM — `evolution-memory-side.md`
- Precision: FP32 → FP16 → BF16/TF32 → FP8 → FP4/MX — `evolution-precision.md`
- Synchronization: busy-bit → scoreboard → mbarrier → transaction barrier — `evolution-synchronization.md`
- Interconnect: crossbar → NVLink → NVSwitch → NV-HBI/NVL72 — `evolution-interconnect.md`
- SASS instruction history per generation — `sass-evolution-timeline.md`
- Tensor-near storage in non-NVIDIA patents (MS/Intel/Meta/Micron/AMD) — `competitor-tensor-storage.md`

### Auxiliary cards (3)

- **Question-driven router** — `problem-playbook.md` (sister to this
  INDEX; organized by problem rather than topic)
- **One-line definitions for every term** — `glossary.md`
- **This file** — `INDEX.md` (topic-driven router)

## Routing by question type

| If your question is about… | Open first | Then maybe |
|---|---|---|
| "What patent introduced X?" | `patent-family-map.md` | per-gen card for that era |
| "What did SM look like in generation X?" | `arch-{gen}.md` | `evolution-data-supply.md` |
| "Why did NVIDIA add cluster / DSM / TMA?" | `arch-hopper.md` | `evolution-execution-model.md`, `evolution-data-supply.md` |
| "What changed Volta → Turing → Ampere?" | per-gen cards in sequence | `evolution-precision.md` |
| "What is TMEM and why?" | `arch-blackwell.md` | `evolution-data-supply.md`, `competitor-tensor-storage.md` |
| "Which SASS opcodes are new in Blackwell?" | `sass-evolution-timeline.md` | `arch-blackwell.md` |
| "Three big rewrites of GPU architecture?" | `three-rewrites.md` | linked patents |
| "How did synchronization evolve?" | `evolution-synchronization.md` | `arch-maxwell.md`, `arch-volta.md`, `arch-hopper.md` |
| "Did Microsoft / Intel / AMD do something similar to TMEM?" | `competitor-tensor-storage.md` | `arch-blackwell.md` |
| "Pascal vs Volta numerical types?" | `evolution-precision.md` | `arch-pascal.md`, `arch-volta.md` |

## Reading order for a cold-start agent

If you have never used this corpus and want to build a mental map:

1. `three-rewrites.md` — the thesis statement; gives you the spine.
2. `patent-family-map.md` — meet the cast of patents.
3. `evolution-execution-model.md` and `evolution-data-supply.md` — the
   two technical axes the rewrites operate on.
4. Skim per-generation cards in chronological order (Tesla → Blackwell)
   for ~3 minutes each.
5. Return here as a lookup table.

After that you should be able to answer "where in the corpus does X
live?" without re-reading the full source.

## Source-of-truth pointers

Each card cites:

- **Primary raw source**: the single section in `NVIDIA_patentd.md`
  (translated, in `references/NVIDIA_patents/NVIDIA_patentd.md`)
  the card distills.
- **Supporting sources**: usually the matching `gpu-arch-wiki/user/*.md`
  for SM specs, plus any `references/NVIDIA_Manuals/` page that anchors
  a SASS / PTX claim.
- **Patent IDs**: every patent reference in this corpus has been
  verified against Google Patents (priority dates and titles confirmed).
  See `patent-family-map.md` for the master list.

## What this corpus does NOT cover

- CUDA programming-model ergonomics, kernel-launch APIs, occupancy
  calculation — those live in `.claude/knowledge/cuda/` and
  `.claude/knowledge/best-practices/`.
- NCU metric definitions and profiling workflow — ACTIVE in this project
  (NCU metrics are the simulator's prediction targets), but they live in
  `tools/local_labels/metric_registry.json` and
  `.claude/knowledge/measurement/pinned-protocol-v21.md`, not in this corpus.
- Per-instruction PTX / SASS semantics — `.claude/knowledge/ptx/`
  and `.claude/knowledge/sass/`. This corpus references those
  cards but does not duplicate them.
- Historical simulator implementation context — `.claude/knowledge/simulator/`.
- Future generations (Rubin and beyond) — not yet enough public material
  to write evidence-backed cards. `gpu-arch-wiki/user/8.rubin.md` has a
  forward-looking summary if you need a placeholder.

## When this index is wrong

If a card link is stale, a section feels mis-routed, or a deeper entry
turned out to be the right first stop, update this file in the same
commit as the work that exposed the gap. Treat stale routing entries
as bugs.
