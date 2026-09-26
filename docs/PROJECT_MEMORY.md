# Continuation memory

Updated 2026-09-26. Read this first; [STATUS](STATUS.md) owns current capability
claims, [runbook](runbook_host.md) owns commands, [HISTORY](HISTORY.md) owns dated
records. Do not load every linked report. Root [AGENTS.md](../AGENTS.md) is policy.

## Current state

- Branch `GPUSim_main`; inspect `git status` before editing. Native hardware
  target is SM86/RTX 3090 Ti. No AMD/SM120 validation is established here.
- Native replay consumes fixed NVBit/SASS instructions, predicates, addresses,
  participants, cards, and control sidecars. NCU is post-replay comparison only.
- `simt_u32_v1`/`sim-sm::custom`: fixed functional ISA with dependency/resource
  scheduling, resident groups, shared memory, bounded barriers, atomics and
  collectives. It does not share native cache timing.
- `simt_its_v1`: single-group functional per-lane PCs plus indirect u32 memory;
  `custom_simt_its SPEC PROGRAM --trace` emits `custom_its_dynamic_v1` JSON.
  Trace cap is 100k records plus a 32 MiB **estimate**, not an RSS/JSON-byte cap.
- Native `schema_version=1`, `metric_surface_version=2`; the metric map is
  additive. Dirty-writeback/coherence scenario rows are postprocessing formulas.
  `min_selected_then_lexical_v1` balances candidate counts in extraction order,
  not scheduler pressure or dynamic hardware routing. See [diagnostics](experimental_diagnostics.md).

## Do not repeat earlier overclaims

- The four native gaps are **not closed**: scheduled writeback/coherence,
  timed physical-pipe selection, value-dependent SASS regeneration, and a
  general ISA/compiler. The user accepted bounded models and requested compiler
  alternatives; current heuristics are not full models of those mechanisms.
- The eight-row September-26 table reuses a September-1 development corpus.
  Its 3.46% GPC-frontier discrepancy is not held-out elapsed-time accuracy.
  TEX-only simulator versus all-source NCU L2 totals are not a valid L2 score.
- Stable module/library files can be identified. JIT/opaque origins remain
  unavailable; active graph launches are rejected, not successfully captured.
- Full Accel-Sim/MGPUSim comparison has not run; confirm scope before resuming.
  Old “all future tasks closed” session/ledger statements are superseded.

## Next concrete seams

Finish mechanism-backed memory/pipeline execution if requested; tighten the ITS
trace schema/budget and reconvergence-successor semantics; use same-domain,
content-bound independent captures for evaluation. Strict Clippy has known
pre-existing debt. Do not turn these into automatic review/workflow loops.

## Operational anchors

- Root Cargo flag: `--manifest-path sim/Cargo.toml`.
- Package `sim-run` = `sim/direct_run`, library `sim_direct_run`;
  `sim-run-campaign` = `sim/run`, library `sim_run`.
- L1/L1-L2 modes require `--pchase-card`; flat mode does not.
- Build each intended binary explicitly; `--bin custom_simt_its` does **not**
  also rebuild `sim-run` just because both packages are listed.
- `.claude/knowledge/` is absent here; use tracked source/docs/`references/`.
- [History](HISTORY.md) records this session's validation and consolidation.
  Project-local Codex memory mirrors point here; raw session logs stay intact.
