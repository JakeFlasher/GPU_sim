# Continuation memory

Updated 2026-09-28. Read this file first, then [STATUS](STATUS.md) and
[HISTORY](HISTORY.md). Load dated reports only for the narrow question at hand.
`AGENTS.md` is the governing repository policy.

## Current scope

- Repository: `/home/jakeshea/GPU_simulator_rebase`, branch `GPUSim_main`.
- Native target: SM86 / RTX 3090 Ti. Native replay is
  NVBit/SASS -> enhanced protobuf -> `sim-trace` -> `sim-sm`/`sim-mem`/
  `sim-engine` -> `sim-run`.
- NCU is the detailed local measurement lane and is never used to select
  simulator parameters. CUPTI is independent launch/resource observation.
- Alpha development is intentionally unbound: no artifact hashes, freshness
  gates, verification manifests, correctness gates, or scoring gates are
  required. Historical/bound forms remain explicit `--bound` promotion paths.

## Implemented bounded seams

- Native replay: fixed captured instructions, predicates, addresses,
  participants, cards, and control sidecars; issue/event/completion/launch
  frontiers remain separate.
- Custom `simt_u32_v1`: fixed functional ISA with dependency/resource
  scheduling, resident groups, shared memory, barriers, atomics, and
  collectives. Custom `simt_its_v1` is a separate single-group functional
  executor, not native SASS or a timing backend.
- `custom_its_dynamic_v3`: bounded per-lane PCs, predicates, successor masks,
  explicit BSSY/BSYNC join/resume PCs, and optional register/predicate snapshots;
  limits are 100k records and 32 MiB of serialized record bytes.
- `ptx_subset_v2`: bounded functional frontend including deterministic indirect
  atomic-add; it is not arbitrary PTX, CUDA ABI, SASS, or a timing compiler.
- Native writeback/coherence experiments are opt-in:
  `--dirty-writeback-capacity-sectors`, `--coherence-service-cycles` plus
  `--coherence-capacity-sectors`, `--l1-visibility-protocol`, and
  `--l1-resident-writeback`. They provide typed queues, ownership observations,
  invalidation events, terminal dirty gauges, and transactional service; they do
  not establish proprietary cache ownership, eviction, payload, or host-flush
  semantics.
- `--native-payload-sidecar` admits bounded value-bearing sidecar records but
  does not alter default replay or complete a native value/cache protocol.
- PhaseFuse `--fast-frontier-only` is conservative trace analysis. It refuses
  sampled/multi-launch/non-full-population/unknown-state cases and is not a
  timing prediction.
- The `.traceg` probe is an ALU-only v2 representation contract. Its default
  alpha output is path/size-only; `--bound` adds historical digest fields.
- Alpha compiler/value adapters and register capture tools are unbound by
  default. They remain finite static/value mechanisms, not general compiler or
  ISA generation.

## Latest development evidence

- Full alpha campaign: `results/alpha/full_3090ti_20260928_r3/performance.csv`
  (ignored local output, 15/15 workloads completed).
- Companion campaigns: `results/alpha/metric_domains_*_20260928` and
  `results/alpha/writeback_*_20260928` (ignored local outputs).
- Aggregate full-campaign medians: native CUDA-event timing 0.009445 ms;
  NCU duration 10,144 ns; NCU GPC cycles 13,614; simulator modeled-event
  cycles 11,780; simulator GPC envelope 13,596; replay wall time 0.665 s.
  Instruction populations matched all 15 rows. These are development
  observations, not an elapsed-time accuracy score; clocks were not locked.
- Read-only clock observation saw target SM/memory values but no drift-free
  active-window lock admission. The probe did not mutate device state.
- The literature lane is registered in
  `docs/accelsim_literature_lower_bound_manifest_20260928.json`; those are
  aggregate ceilings/floors, not per-kernel ground truth.

## Unresolved closure blockers

Do not relabel these as complete:

1. Proprietary per-instruction physical-pipe/issue witness.
2. Unrestricted NVIDIA ISA/compiler generation and CUDA ABI contract.
3. Native cache ownership, eviction, payload, and host-flush semantics.
4. Drift-free locked-clock NCU timing with active-window evidence.
5. Fully matched external timing domains or same-card executable score.
6. General native value-dependent SASS regeneration, calls, atomics, fences,
   races, and complete memory-order reconstruction.

## Development rules and anchors

- Preserve separate observation/model/frontier domains; never collapse a
  diagnostic formula into native completion or timing accuracy.
- Keep alpha campaign outputs unbound and independent; do not feed NCU labels
  into model selection.
- Inspect focused diffs before edits and preserve unrelated work. Build-only
  checks are acceptable when needed; do not launch test suites unless the user
  explicitly changes the alpha policy.
- Active review ledger: [code adversarial review](code_adversarial_review_20260928.md).
- Active limitation audit: [remaining limitations](remaining_limitations_audit_20260928.md).
- Active experiment plan: [SM86 campaign plan](../plans/sm86_next_experiment_campaign_20260927.md).
- Host commands: [runbook](runbook_host.md). Historical reports and plans are
  context, not automatic queues.
