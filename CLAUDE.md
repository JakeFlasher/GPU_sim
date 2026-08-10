# CLAUDE.md — GPU simulator project context

## Mission

Build a fast, event-driven, trace-driven simulator of modern NVIDIA GPUs in
Rust. Bring up Ampere `sm_86` first, then Blackwell `sm_120`. NVBit/SASS traces
and architecture cards are simulator inputs; silicon NCU vectors are validation
targets only.

The live dataflow is:

`NVBit tracer_tool` → `traces_enhanced` protobuf → committed Prost bindings in
`sim/trace` → `sim-sm` replay → `sim-mem` → `sim-engine` → NCU-shaped outputs
from `sim_run::emit` inside `sim-run`.

M-SIM-1 binds accuracy to `gpc__cycles_elapsed.max`, the sole
publication-binding headline: MAPE ≤10% meets the engineering target, >10% and
≤15% is floor-only, and >15% is publication failure. Cycles emission is
deliberately `required_unimplemented` until a checked-in aggregation-semantics
study validates the single-launch mapping and selects a multi-launch composition
rule from admitted silicon evidence. `KernelReplayResult`'s
`diagnostic_last_issue_cycles` and `terminal_modeled_event_time` fields are
separate directional diagnostics, not admissible primary predictions. The seven current
`sim_run::emit` descriptors are secondary, implemented, reported, and ungated.

## Binding rules

1. **Targets, never features.** NCU/hardware-counter values may label and score
   predictions. They must never enter trace features, simulator configuration,
   architecture constants, or calibration inputs for the same scored cells.
2. **NCU-only timing.** The former dual-source duration path is retired. Active
   timing labels travel as v5 `unit_tagged_metrics`, notably
   `gpc__cycles_elapsed.max` (`cycles`) and `gpu__time_duration.sum` (`ns`).
3. **Silicon-only accuracy claims.** Another simulator may be an evaluation
   comparator, never ground truth. Report per metric and kernel family; do not
   hide failures in aggregate-only summaries.

   **Current scope narrows this permission:** no external simulator comparator is
   active. NCU supplies the ground-truth labels, and runtime/profile evidence is
   recorded only for this repository's `sim-run`. Accel-Sim and GPGPU-Sim are
   inactive preserved references—not baselines, dependencies, gates, or current
   work—unless the user explicitly reauthorizes them.
4. **Typed, fail-closed evidence.** Missing authority, unknown versions, dirty
   fixtures, hash mismatches, unsupported architecture paths, and absent
   hardware evidence are explicit failures or typed exclusions—never silent
   passes.
5. **Architecture-local facts.** Latency, geometry, scheduler, SASS, register,
   cache, clock, and thermal behavior are per-architecture/per-SKU facts.
   Consumers must not substitute a different architecture's value for a
   `research_gap`.
6. **Additive producer contracts.** Proto enum changes append only; new fields
   are capability-gated; producer and consumer validation land in lockstep.
   Schema-version bumps are reserved for breaking wire changes.
7. **Evidence retention.** Do not destroy possibly unique evidence. Historical
   ledger rows remain byte-exact. Historical path relocation resolves through
   the versioned alias table with mandatory SHA-256 equality.
8. **No history rewrite.** Commit hashes are evidence anchors. Cleanup
   deletions are HEAD-only and remain recoverable from Git history.
9. **Privilege boundary.** Never run a sudo command. Give the exact command to
   the operator. Capture tooling must record relevant driver, CUDA, NCU, clock,
   cache, replay, and device state.
10. **Escalation.** `AGENTS.md` is binding. A Fable-5 consultation is permitted
    only when the user explicitly requests one. Otherwise exhaust safe local
    evidence and ask the user directly for any required authority or unavailable
    external fact; do not guess or weaken a gate.
11. **Subagent concurrency.** Never run more than 10 subagents concurrently.
    Count known live subagents before spawning, resume interrupted agents rather
    than duplicating them, and serialize or queue all work beyond the ceiling.

Retired rules must not be revived: the NCU ban/quarantine, the no-driver-root
thesis, dual-source timing, Rule 5 “no simulator revival,” or the former
iterative orchestration loop. Archived loop state is migration input, not
active process state.

## Active authorities

- The three-part v5 label authority is
  `workloads/simbench_bw/schemas/local_label_sidecar.schema.json`,
  `workloads/simbench_bw/schemas/local_label_admitted.schema.json`, and
  `tools/local_labels/metric_registry.json` (61 metric entries).
- `workloads/simbench_bw/schemas/measurement_manifest.schema.json` remains the
  active v2 per-measurement provenance contract. It is not a v5 label schema.
- Retired duration evidence dispatches explicitly to
  `workloads/simbench_bw/schemas/historical/duration_admitted.v1.schema.json`;
  its content hash is frozen. Missing/unknown versions fail closed.
- `arch_profiles/` and `data/pchase/` hold architecture inputs.
- `tools/nvbit_tracer/`, `tools/sass_control/`, `tools/cache_replay/`,
  `tools/cache_predictor/`, and `tools/trace_sampling/` are active simulator
  instrumentation or calibration inputs.
- `tools/local_labels/` is the active NCU capture/label factory.
- `data/local_labels/campaign/`, `data/annex/`, and the external capture and
  preparation-failure archives are frozen evidence and undeletable in this
  cleanup.

## Hosts and architecture order

- Current primary host: RTX 3090 Ti (GA102, `sm_86`). Recorded toolchain:
  driver 610.43.02, NCU 2026.1.0, CUDA 13.3 default and pinned CUDA 13.2 where
  the capture/build contract requires it.
- Second target: RTX 5060 laptop (GB206, `sm_120`).
- Controlled same-ISA cross-SKU target: cloud RTX PRO 6000 (GB202, `sm_120`).

Evidence status is device-specific: `Offline-verified`, `Verified-live-sm_86`,
`Verified-live-sm_120-gb206`, `Verified-live-sm_120-gb202`, or
`Cross-arch-verified`. A result on one device never implies verification on
another.

## Current execution state

M0–M3 are complete. M4 is in progress. M5 has not started; M6/M7 are
blocked on upstream milestones. The canonical authority contains 344 cells: 122
runnable and 222 excluded. Current defensible canonical credit is 1/122
runnable. The stopped 49-row revision and the Task7/H3 capture-defect archive
are frozen, zero-credit, and reuse-forbidden. The interrupted 53/201 SOL
runnability attempt is separately archived and canonical v1 is restored.
Supplemental launch-set v3 is a separate cohort and remains live-disabled until
a pinned canary passes.

ADR 0001 authorizes Task7/H3 continuation outside RLCR as
`CONTINUE_PRESERVE_PINNED`; ADR 0002 fixes publication scope as
`CYCLES_ONLY_BINDING`. ADRs 0003–0006 record later capture-envelope decisions.
Neither the accepted decisions nor any later decision admits historical evidence
or weakens a campaign, provenance, or hardware gate.

Authoritative plans:

- `plans/micro2027_ncu_sim_execution_plan.md`
- `plans/sm86_comprehensive_live_capture_campaign_plan.md`

Authoritative project documentation lives under `docs/`: start with
[`docs/README.md`](docs/README.md), then use the architecture, component
inventory, status, host runbook, decisions, testing policy, and durable lessons.

## Default verification

From the repository root:

```bash
(cd sim && cargo fmt --all -- --check)
(cd sim && cargo test --workspace --all-targets)
.venv/bin/python -m pytest -q
.venv/bin/python workloads/simbench_bw/validate.py --all
```

The default Python suite is self-contained and deselects `campaign_evidence`
and `hardware_evidence`. Their owner gates must fail when required external
evidence is absent; absence must never be reported as a pass.
