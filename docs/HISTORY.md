# Project history index

**Updated 2026-09-30.** This file is an index, not an execution plan. The
current interpretation is in [`STATUS`](STATUS.md); continuation context is in
[`PROJECT_MEMORY`](PROJECT_MEMORY.md). The former 700-line session diary and
all dated Markdown reports were superseded to reduce default context. Their
content remains under [`archive/research_20260930`](archive/research_20260930/)
and in Git history.

## Current consolidation

The 2026-09-30 maintenance pass:

- moved 122 dated research, probe, review, and campaign reports out of the
  active `docs/` surface;
- replaced the long memory/status/history handoffs with compact, corrected
  entry points;
- retained machine-readable data, raw campaign locations, source code, cards,
  schemas, tests, and historical experiments in their existing roles;
- marked all plans as historical context in `plans/README.md` while retaining
  their paths for source/data ancestry;
- corrected stale claims about projection speed, cohort independence, CUPTI
  semantics, clock witnesses, interval statistics, and unmatched-shape OOD;
- recorded that the four physical boundaries remain open and that metric
  surrogates are alternatives, not closure evidence.

No tests, hashing, or verification workflows were run for this documentation
pass. This line does not waive the normal validation contract for future code
changes.

## Evidence timeline

| Period | What was established | Current interpretation |
| --- | --- | --- |
| 2026-09-05--12 | Early replay regeneration, event-skipping, semantic and custom probes | Historical implementation context; not a universal cycle claim |
| 2026-09-23--26 | Native SM86 trace/capture seams, custom functional paths, memory/coherence and pipe notes, prior-art review | Contracts and negative results retained; detailed reports archived |
| 2026-09-27 | Bounded holdouts, PhaseFuse/fast-path prototypes, external-baseline and SOTA positioning | Local NCU remains the detailed target; published literature is a separate floor |
| 2026-09-28 | Adversarial review and interpretation corrections | Related-cohort evidence, no broad OOD or end-to-end speed proof |
| 2026-09-29 | Shape/archetype, cache-phase, endpoint, clock and NAK/ELF alternatives | Candidate projections with explicit fallback; physical mechanisms still unknown |
| 2026-09-30 | CUPTI/NAK/PMA follow-ups, endpoint committees, temporal matrix, scope-corrected intervals | Descriptive independent observations; no unified endpoint or blocker closure |

## Current evidence lanes

- **Native replay:** NVBit/SASS -> protobuf -> Rust trace/SM/memory/event
  models. Keep issue, modeled-event, completion, launch-envelope, and typed-tail
  frontiers distinct.
- **Fast projection:** frontier/archetype reference plus non-promoted lattice,
  tile, PM, CUPTI, clock, cache, NAK, and endpoint committees. NCU labels are
  score-only.
- **Custom functional:** `simt_u32_v1` and `simt_its_v1`; not native timing.
- **Independent observations:** NAK-generated bodies, CUDA-ELF templates,
  CUPTI PC sampling, PMA words, public RM controls, CUDA events, and host clock
  probes. Each has a distinct domain and cannot be silently fused.

## Archive navigation

- `archive/research_20260930/sm86_*` — current SM86 campaign and blocker work.
- `archive/research_20260930/custom_*`, `accelsim_*`, `gpu_simulator_*`,
  `curry_*`, `huerta_*` — architecture, prior-art, and custom probes.
- `archive/research_20260930/adversarial_*`, `remaining_*`, `*_audit_*` —
  reviews and corrections; they do not create workflow gates.
- `archive/vortex/` — earlier Vortex adapter artifacts.
- `plans/` — retained plan paths with historical notices; `plans/README.md`
  is the only plan index to consult by default.

## Supersession rules

1. A dated report records what was observed or hypothesized at that time. It
   cannot override current source, config, `STATUS`, or `AGENTS.md`.
2. A historical plan is context, not an automatic task list, approval system,
   hash gate, or completion certificate.
3. New claims must name the input cohort, endpoint, holdout, and whether labels
   influenced fitting. If those are unavailable, state the result as a
   hypothesis or diagnostic.
4. Update this short index only for a new durable boundary or correction; do
   not append a transcript.
