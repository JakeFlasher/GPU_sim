# General software GPU simulator roadmap

Updated 2026-09-24. Consolidates the earlier milestone prose without changing
historical measurements. Current implementation details live in
[STATUS](../docs/STATUS.md), not in duplicated task checklists.

## Research objective

Build a configurable software architectural simulator that lets users study
GPU timing/resource choices and demonstrate useful accuracy–speed trade-offs
against independent profiler observations on available hardware. The native
anchor is SM86; a bounded custom functional/resource frontend complements it.

A falsifiable paper thesis is one GPU-specific timing mechanism that changes a
real architecture decision while improving measured simulator turnaround at a
declared accuracy/detail point. Configuration files, Rust, event queues, and
native tracing alone are established prior art. The project is not yet a
universal, globally SOTA, or paper-ready simulator.

## Milestone ledger

| Milestone | Current result | Remaining boundary / evidence |
| --- | --- | --- |
| M0: workload and claim | One SM86 GEMM launch with recorded device, tools, geometry, clock/cache/replay scope | Development evidence, not an untouched final evaluation set |
| M1: NCU mapping | Instruction count matches at 197,120 for one launch and three NCU repeats | GPC/SM cycle crosswalk remains provisional; missing retirement/coherence tails |
| M2: production resource choice | Scalar and `trace_sass_v1` profile issue-width overrides reach direct/prepared replay; card resources are consumed | Only this bounded resource axis; fixed SASS semantics cannot change |
| M3: simulator speed | Event/dense wall-time probe and bounded L1 preflight optimization recorded | Local engineering result, not broad or cross-simulator speed superiority |
| M4: coverage | Inventory expanded to 18 retained rows; typed gaps/tails retained | Static module origin unresolved; no universal feature support claim |
| M5: custom architecture | Resource pipeline, resident groups, shared operations, barriers, atomics, broadcast/reduction, plus separate ITS example | Fixed bounded semantics; no arbitrary ISA/compiler, general memory model, or vendor timing |
| M6: fair baselines | Accel-Sim/Huerta executable-overlap records and explicit non-scoring matrix | **Deferred by user request**; hardware/trace/observable semantics do not align sufficiently |
| M7: design conclusion | Negative issue-width study and custom resource sweep retained | Not a complete paper evaluation or demonstrated SOTA advantage |

Evidence: [measurement](../docs/sm86_measurement_boundary_20260924.md),
[speed](../docs/sm86_speed_probe_20260923.md),
[coverage](../docs/sm86_feature_coverage_20260923.md),
[design study](../docs/sm86_issue_width_design_study_20260923.md),
[custom backend](../docs/custom_program_pipeline_20260924.md),
[baseline matrix](../docs/baseline_matrix_20260924.json).

The original three implementation blockers have bounded evidence: count
mapping, a production-consumable profile, and basic pipeline overlap. This does
**not** close elapsed-time accuracy, all configurability, or all future work.

## Keep the evidence lanes distinct

| Lane | Inputs | Permitted conclusion |
| --- | --- | --- |
| Functional | Bounded program + architecture spec | Architectural-state behavior under the stated ISA |
| Model-relative timing | Program/trace + declared resource model | Timing and design trends within that model |
| Hardware comparison | NVBit trace + architecture card + independent NCU row | Error for a declared workload, launch, and observable |
| Simulator performance | Same model/input with recorded host/build | Replay/process time, throughput, RSS, and preprocessing cost |

A functional result is not hardware validation. NCU values from scored
workloads must not tune model inputs, implementation selection, or training.
Keep issue, event, completion, launch, and typed tails separately observable.

## Fixed-trace intervention contract

- `trace_safe`: changes timing/resource service without changing represented
  instructions, addresses, predicates, participants, or values. The current
  issue-width profile is in this lane for the studied inputs.
- `trace_bounded`: changes a declared effect only within an explicitly stated
  observable bound; unsupported cases remain visible.
- `requires_execution`: may regenerate control/data flow, addresses, atomic
  order, participation, or values. Requires functional execution or recapture;
  fixed SASS replay alone cannot answer it.

[Profile contracts](../docs/archprofile.md) describe the narrow production
bridge. [Custom programs](../arch_profiles/programs/README.md) describe the
separate execution frontend. Neither silently changes vendor-card identity.

## Next research choices, not automatic tasks

Choose one concrete workload/mechanism question before expanding code:

1. If pursuing cycle accuracy, identify the first missing modeled mechanism or
   profiler boundary and use a mechanism-control workload. Keep development
   measurements separate from a final evaluation set; do not fit a scored row.
2. If pursuing configurability, add only the semantic/resource change needed by
   a design question, together with functional behavior, dependency lowering,
   output observability, and explicit unsupported cases. Unknown latency stays
   unknown. Do not add unrelated ISAs or a compiler generator as prerequisites.
3. If pursuing speed, profile a representative release replay, separate trace
   loading from simulation, and retain negative results. Compare declared
   observables without collapsing independent frontiers into one score.
4. Resume M6 only on explicit request and a defensible common comparison cell.
   Literature numbers remain context, not a numerical ranking.

A useful evaluation reports per-workload errors and outliers, wall time/RSS,
trace cost, host threads, configurations, versions, and an ablation tied to an
architecture conclusion. The current results are feasibility/development
observations, not a guarantee of ISCA/MICRO/HPCA acceptance.

## Exclusions and retained history

RTL, gate/FPGA/physical design, proprietary decoder internals, silicon-equivalent
state reconstruction, and hardware the team cannot measure are not completion
requirements. Decoding a documented/custom ISA and using a vendor-neutral IR
are feasible but different from universal proprietary binary compatibility.

[Research positioning](../docs/gpu_simulator_research_positioning_20260923.md)
and [extensibility research](../docs/architecture_extensibility_research_20260924.md)
record the prior-art basis. Older plans remain historical context; root
[AGENTS.md](../AGENTS.md) governs execution. SOL/campaign source, environments,
data, schemas, and experiments remain retained. No new workflow gates arise
from this roadmap.
