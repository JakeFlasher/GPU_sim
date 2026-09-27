# Continuation memory

Updated 2026-09-27. Read this first; [STATUS](STATUS.md) owns current capability
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
  `custom_simt_its SPEC PROGRAM --trace` emits `custom_its_dynamic_v2` JSON.
  Trace cap is 100k records plus 32 MiB of accumulated serialized bytes for the
  individual JSON records (the result envelope is not counted), not an RSS cap.
  Records include grouped successor masks and explicit BSSY/BSYNC
  join/resume PCs; this remains a custom functional CFG artifact.
- Native `schema_version=1`, `metric_surface_version=2`; the metric map is
  additive. Dirty-writeback/coherence scenario rows are postprocessing formulas.
  `min_selected_then_lexical_v1` balances candidate counts in extraction order,
  not scheduler pressure or dynamic hardware routing. See [diagnostics](experimental_diagnostics.md).
- `sim-run --fast-frontier-only` is an opt-in trace-derived PhaseFuse lane. It
  skips native timing replay, certifies only narrow scalar regions, and reports
  refusal reasons; its output is not a timing prediction or NCU replacement.
- `ptx_subset_compile` lowers a small text subset into `simt_its_v1`; it is a
  functional compiler/assembler seam, not arbitrary PTX, CUDA ABI or SASS.
- Custom `operation_resources` may be a scalar or an alternative list. The
  custom event pipeline selects a feasible resource using modeled latency,
  initiation interval and in-flight capacity, and reports the concrete choice;
  this is not native SM86 physical-pipe attribution.
- `sim-mem::writeback` provides an opt-in dirty-sector queue primitive with
  transactional capacity/timing preflight, fixed-point shared read/write
  service, same-line fetch hazards and tokenized completion. Native launch
  replay enables it only through explicit `--dirty-writeback-capacity-sectors`;
  coherence ownership/visibility and resident dirty flush remain tails.
- `sim-mem::coherence` provides an opt-in directory-style service selected by
  `--coherence-service-cycles N --coherence-capacity-sectors N`. It models only
  ordered, fully decoded L2-request topology; L1-hidden accesses and protocol
  fence/ownership visibility remain explicit limits.

## Do not repeat earlier overclaims

- The four native gaps are **not closed**: full coherence/resident dirty flush,
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

Use the PhaseFuse fast lane to gather refusal/certificate populations, the
bounded PTX subset for value-dependent functional studies, and the custom route
contract for controlled scheduling experiments. Extend the opt-in writeback and
directory queues only after native event/config evidence is independently
validated; then add
state-complete summaries only for one mechanism at a time; extend the ITS
trace schema and reconvergence-successor semantics with further independent
input-varying cases; use same-domain, content-bound independent captures for
evaluation. Strict Clippy has known
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
