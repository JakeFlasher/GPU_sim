# Knowledge Base & Auto-Routing

This suite has a **best-effort knowledge router**: when a future Claude session hits
a GPU/CUDA/PTX/SASS/NVBit/measurement/SOL-ExecBench question, the prompt hook
may surface relevant distilled cards. The hook can fall back to a reminder when the
ignored manifest is absent or no match is found; it does not replace deliberate
source review or the active project authorities. This page documents the routing
implementation and its maintenance boundaries.

Two layers, both now local:

1. **353 non-index knowledge cards** across 14 subcorpora under
   `.claude/knowledge/`, plus two physical index/router files; `_template.md` is
   excluded from the routable scan. The generated manifest therefore contains
   **355 rows** (see
   [`../.claude/knowledge/INDEX.md`](../.claude/knowledge/INDEX.md)).
2. **~458 MB of source material** under `references/` — full PDF papers, the NVIDIA docs
   mirror (`NVIDIA_Manuals/`), translated patent narratives (`NVIDIA_patents/`), the
   235 SOL-ExecBench problem specs, and the Rust/kernel-pilot reference material.
   A historical closure snapshot recorded **299 of 300** cited paths resolving
   locally; revalidate that count before treating it as current.

Each card is a *distilled index* whose frontmatter points at source material. Those
sources are documentation/provenance references for the cited topic, not project
policy authority, simulator features, or silicon validation truth; root
`CLAUDE.md`, accepted ADRs, and owning schemas/runbooks govern those concerns. The
router connects a prompt → a candidate card → its cited source when the local
manifest and hook are available.

---

## The auto-routing flow

```
   a future Claude session types a prompt
        e.g. "decode this Blackwell OMMA SASS opcode"
                          │
                          ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ .claude/settings.json  →  UserPromptSubmit hook                │
   │   python3 scripts/kb/route_query.py --root $DIR --top 3        │
   └───────────────────────────────┬───────────────────────────────┘
                                    │ reads the prompt from the event JSON
                                    ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ scripts/kb/route_query.py  (the merged SUPERSET router)        │
   │   1. FTS5/BM25 search over .claude/kb_manifest.sqlite          │
   │   2. SKILL_ANCHORS boost  (sass, nvbit, measurement, nvcc,     │
   │      ptx, blackwell, patents, simulator, +SOL: gemm/rope/moe/  │
   │      attention/fp8/problem_NNN → patterns,problems)            │
   │   3. SUBCORPUS_TO_SKILL  (14 subcorpora + root → 10 destinations)     │
   │      9 active; one retained historical measurement mapping             │
   │   4. relevance gate (_relevance.py) + meta-router for KB-about │
   └───────────────────────────────┬───────────────────────────────┘
                                    │ injects into the session context:
                                    ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  [KB] Skill: `analyzing-sass-nvbit-traces`. Open + cite:       │
   │    - .claude/knowledge/sass/sass-instruction-reference.md      │
   │    - .claude/knowledge/sass/sass-tooling.md                    │
   │    - .claude/knowledge/nvcc/nvcc-flags.md                      │
   └───────────────────────────────┬───────────────────────────────┘
                                    │ the card's frontmatter:
                                    │   Primary raw source: references/NVIDIA_Manuals/…
                                    ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  INTENDED local source material (documentation authority):              │
   │    references/NVIDIA_Manuals/cuda_binary_utility/…            │
   │    references/papers/*.pdf   references/NVIDIA_patents/…      │
   └───────────────────────────────────────────────────────────────┘

   ── on Stop ──────────────────────────────────────────────────────
   scripts/kb/check_kb_usage.py --mode warning  →  appends to
   .claude/metrics/kb_usage.jsonl and reminds if a KB-relevant prompt
   was answered WITHOUT consulting .claude/knowledge/ or references/.
```

The fallback: if the manifest is missing or nothing matches, the hook defers to
`.claude/hooks/kb_reminder.sh`, which prints a short "consult the KB" reminder.

---

## What was merged (routing reconciliation)

The four source trees carried **two** routers. They are reconciled so a single wired
router covers the whole unified KB:

| Piece | Resolution |
|---|---|
| Wired router | `scripts/kb/route_query.py` (codex) — **extended** with solbench's 4 subcorpus mappings (`sources`/`patterns`/`problems`/`cutlass_constructs` → `implementing-sol-baselines`) and its SOL anchors (`gemm`/`rope`/`moe`/`attention`/`fp8`/`problem_NNN`). It keeps codex's profiler anchors (sass/nvbit/measurement/nvcc/ptx/blackwell/patents) intact. |
| Manifest builder | `scripts/kb/build_manifest.py` — recursively scans `.claude/knowledge/**/*.md`, so it indexes all 14 subcorpora automatically. |
| SOL card-gen tooling | kept at `tools/sol_kb/` (`generate_dossiers.py`, `generate_patterns.py`, `lint_cards.py`) — its duplicate router is superseded by `scripts/kb/`; the retained authoring tools import the shared `scripts/kb/routing_contract.py`. |
| Sources | `references/` contains the migrated corpus intended for local closure; the historical path-accounting snapshot records one unresolved citation and must be revalidated before being treated as current. |

A verification run performed with a freshly generated manifest found that these
representative prompts route correctly; regenerate the manifest and repeat the
probes before treating the results as current:

| Prompt | → Skill | → Top card |
|---|---|---|
| "decode a Blackwell OMMA SASS opcode with cuobjdump" | `analyzing-sass-nvbit-traces` | `sass/sass-instruction-reference.md` |
| "implement a flash attention CUTLASS baseline for problem_012" | `implementing-sol-baselines` | `problems/problem_012_…md` |
| "where is the current local-label capture contract documented" | `validating-nonroot-gpu-measurement` (legacy pointer; consult root policy) | `measurement/pinned-protocol-v21.md` → `tools/local_labels/SWEEP_RUNBOOK.md` |
| "which patent introduced TMA and TMEM on Hopper/Blackwell" | `reasoning-about-nvidia-architecture-history` | `patents/sass-evolution-timeline.md` |

---

## Skills and routing ownership

The filesystem contains 11 scoped `SKILL.md` files. The shared routing contract
maps 14 non-root subcorpora plus the root entry to 10 destinations: 9 active
router destinations and one retained historical measurement mapping.
`kernel-knowledge` is a separately scoped imported reference outside that
vocabulary. The measurement mapping still points to the retired skill for
historical card routing; its output is only a pointer, and current policy comes
from root `CLAUDE.md`, active schemas, tools, and accepted ADRs.

`.claude/skills/` — task-shaped guidance and scoped historical references:

| Skill | Owns subcorpora | Raw sources |
|---|---|---|
| `navigating-project-kb` | meta-router (INDEX + `patents/INDEX.md`) | — |
| `validating-nonroot-gpu-measurement` | `measurement/` (historical only) | Historical CUPTI/NVBit/NVML references |
| `interpreting-ptx-isa` | `ptx/`, `inline-ptx/` | `references/NVIDIA_Manuals/ptx_isa/` |
| `analyzing-sass-nvbit-traces` | `sass/` | `references/NVIDIA_Manuals/cuda_binary_utility/` |
| `tuning-blackwell-sm120-kernels` | `blackwell/` | Blackwell tuning docs |
| `optimizing-cuda-kernels` | `cuda/`, `best-practices/` | `references/NVIDIA_Manuals/cuda_guide/` |
| `compiling-cuda-artifacts` | `nvcc/` | nvcc docs |
| `reasoning-about-nvidia-architecture-history` | `patents/` | `references/NVIDIA_patents/` |
| `designing-gpu-simulator` | `simulator/` (active Rust simulator design) | Rust/DES refs |
| `implementing-sol-baselines` | `sources/`, `patterns/`, `problems/`, `cutlass_constructs/` | `references/NVIDIA_Manuals/solbench_problems/`, `references/papers/{AVO,SOL-Execbench}.pdf` |
| `kernel-knowledge` | trimmed kernel-pilot pack | `references/kernel_pilot_imported/` |

---

## Maintenance

```bash
# After adding/editing any card, rebuild the FTS5 manifest so the router sees it:
python3 scripts/kb/build_manifest.py --root .

# Sanity-check routing for a prompt (simulates the UserPromptSubmit hook):
echo '{"prompt":"tcgen05.mma NVFP4 block scaling on sm_120a"}' \
    | python3 scripts/kb/route_query.py --root . --top 3

# The manifest (.claude/kb_manifest.sqlite) is gitignored — always regenerate it
# on a fresh clone before relying on auto-routing.
```

**Adding a source card:** create `.claude/knowledge/<subcorpus>/<slug>.md` following
[`../.claude/knowledge/_template.md`](../.claude/knowledge/_template.md) (frontmatter with
`Primary raw source:` pointing at the local `references/…` path), then rebuild the
manifest. If the subcorpus is new, add its ownership entry to
`scripts/kb/routing_contract.py`; `route_query.py` and the linter import the shared
contract.

---

## Why this matters

The point of migrating the full sources in-tree (rather than leaving them in the
original trees) is **intended closure**: when the manifest and hook are
available, a future session should be able to move from a distilled card to its
cited source without leaving the tree. Local closure remains subject to source
revalidation, and routing is advisory; root `CLAUDE.md`, accepted ADRs, evidence
schemas, and architecture-local source contracts remain authoritative.
