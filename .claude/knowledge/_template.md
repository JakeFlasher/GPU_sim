---
Use when: <one-line trigger phrase a developer or agent would actually type. This field is FTS-indexed with the highest BM25 weight (6.0); put your routing anchor tokens here. Required, non-blank.>
Primary raw source: <ONE canonical authority path or URL. For problems/ this MUST resolve under `references/NVIDIA_Manuals/solbench_problems/problem_NNN_*.md`. Required, non-blank.>
Supporting sources: <semicolon- or bullet-separated paths/URLs that complement Primary raw source. Required field; may use `None` if no supporting sources apply.>
Repo touchpoints: <repo-relative paths whose behaviour this card explains or governs (e.g. `scripts/kb/route_query.py`, `cutlass/include/cutlass/gemm/`). Required, non-blank.>
Patents anchored here: <relevant NVIDIA patent IDs, or the canonical placeholder `None` if not patent-anchored. Required field; `None` is allowed.>
Out of scope: <required for sources/ stubs (may be a single line) AND required for patterns/ and problems/ cards where it MUST enumerate ALL 8 forbidden-surface rule classes from CLAUDE.md — see block below. NEVER blank, NEVER `None`.>
Version family: <e.g. `sol-execbench-2026`, `cutlass-4.5.x`, `cuda-13.1`. Required, non-blank.>
Default skill: <optional. Only used to override the SUBCORPUS_TO_SKILL default. Must be in the canonical skill list — see scripts/kb/routing_contract.py (the shared ownership contract).>
Also relevant skills: <optional. Comma- or semicolon-separated list of other canonical skills that should be hinted in the routing reminder.>
Routing tags: <optional. Semicolon-separated free-form tokens to boost FTS5 matching. Indexed in the manifest's v2 `cards_fts` routing_tags column; legacy v1 manifests omit this column.>
Latency authority: <optional. Required ONLY if the card body cites numeric latencies / bandwidths / SOL scores. Must be exactly one of: `paper` | `leaderboard` | `local-checkout`.>
---

# <Card title — should contain at least one routing anchor token so cards_fts indexes it under the `title` column>

## Use when

<Restate the trigger conditions in prose. Body content is NOT FTS-indexed for routing-anchor purposes — anchors must live in the indexed fields above (name / title / Use when / Routing tags).>

## Approach / Content

<Card-specific body. For patterns/ cards: the canonical CUTLASS / CuTe recipe with B200 / sm_100a specialisation. For problems/ dossiers: a one-paragraph workload summary plus a link to the matching pattern card. For sources/ stubs: a short pointer to the cited source.>

## Architecture portability (REQUIRED for `patterns/` and `problems/` cards; OMIT for `sources/`)

`sm_120` (local RTX 5060) is a compile / smoke target only; `sm_100a` (cloud B200) is the scoring target.

This card respects `CLAUDE.md § Architecture Portability Discipline`. Cards under `patterns/` and `problems/` MUST call out divergences across: TCGen05 / TMEM codegen, TMA shapes, async-copy semantics, cluster behaviour, DSMEM, register / occupancy, L2 partitioning, HBM3e bandwidth, CUTLASS schedule selection, NVFP4 block-scaling layout, FP8 e4m3 / e5m2 layout.

Evidence-label vocabulary (`Compile-checked-sm_120`, `Ran-local-sm_120`, `Scored-B200-sm_100a`, `Not-B200-validated`) lives in `baselines/<category>/<slug>/README.md` per `CLAUDE.md:210-213` — these labels MUST NOT appear in `problems/<slug>.md` dossier bodies (the status-field rule forbids them).

## Forbidden-surface annotation block (REQUIRED in the `Out of scope` frontmatter for `patterns/` and `problems/` cards)

Copy this exact block into the `Out of scope:` frontmatter field. All 8 rule classes must be enumerated.

```
Out of scope:
- Python / Triton / CuTe DSL Python / cuTile Python / cudnn_frontend Python / torch.compile submissions (CLAUDE.md Rule 5).
- NCU / `nv-nsight-cu-cli` / `.ncu-rep`/`.ncu-repz` artefacts on any device, including local 5060 (CLAUDE.md Rule 1).
- `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` / persistence / power / MIG / compute-mode mutation (CLAUDE.md Rule 2).
- Privileged CUPTI: Events, Metrics, Range Profiling, PM Sampling, PerfWorks (CLAUDE.md Rule 3; CUPTI Activity remains allowed).
- Host-driver work: NVIDIA driver install, DKMS, MOK, `modprobe nvidia`, kernel-module parameter toggles, reboot workflows (CLAUDE.md Rule 4).
- User CUDA streams, background host threads, `torch.jit.fork`, null-stream side effects, CUDA Graphs without a per-problem note (CLAUDE.md Rule 7).
- Evaluator-state exploits: `data_ptr` caching, lazy-eval `__eq__` on `FakeTensor`, monkey-patched `Event.elapsed_time` / `do_bench` (CLAUDE.md Rule 6).
- Precision-contract violations: precision downgrade (e.g. FP16 compute upcast to FP32 output) to pass tolerance (CLAUDE.md Rule 8).
```

## Latency-authority rule (REQUIRED if any numeric latency / bandwidth / SOL Score is cited in the card)

When the body cites numeric values that match `lint_cards.py`'s two-pass detector — unit-suffixed regex (`0.020511 ms`, `8 TB/s`, `S=0.732`) or markdown-table-header context (`| Baseline (ms) | SOL (ms) |` with bare decimals like `0.0221`) — the frontmatter MUST set `Latency authority` to exactly one of: `paper`, `leaderboard`, `local-checkout`.

Three timing authorities documented in `CLAUDE.md § Active Measurement Stack`:
- `paper` — values from `SOL-Execbench.pdf` or `AVO.pdf`.
- `leaderboard` — values from the official online judge.
- `local-checkout` — values from this checkout's `sol-execbench/src/sol_execbench/core/bench/timing.py` or local runs.
