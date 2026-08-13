# P8 target-free cycles-semantics boundary study

> Retained campaign reference. This document records the historical strict
> campaign interpretation of cycle metrics. It is not a repository-wide gate,
> does not restrict direct simulator or NCU/NVBit research, and applies only
> when the opt-in SOL/campaign skill is deliberately invoked.

**Status:** Target-free boundary record only. This document does not select cycle semantics, does not satisfy the `AggregationSemanticsStudy` blocker, and does not promote cycle emission.

For the retained campaign path, the applicable metric definitions are in the
[metric registry](metric-registry.md). This study uses only checked-in code,
architecture metadata, and vendor documentation. It neither reads nor uses
silicon target values.

## D1 — NCU observable contract

### Target-free findings

The active [metric registry](../tools/local_labels/metric_registry.json) identifies `exec_cycles_gpc_max` as the target-only `gpc__cycles_elapsed.max` observable, with unit `cycles`, counting unit `GPC`, and roll-up `max`. NVIDIA's checked-in [Nsight Compute Profiling Guide](../references/NVIDIA_Manuals/ncu_guide/02_profiling-guide.md#metrics-entities) defines `.max` as the maximum counter value across all unit instances, and its [cycle-metrics section](../references/NVIDIA_Manuals/ncu_guide/02_profiling-guide.md#cycle-metrics) says a cycles counter reports cycles in that unit's clock domain. These facts define the label-side observable; they do not define a simulator-to-NCU mapping.

### Hypotheses only

A future mapping from a modeled temporal observable to the GPC-scoped NCU observable is a hypothesis to test against separately admitted silicon evidence. No mapping, temporal frontier, or modeled-event endpoint is named or selected here.

### Silicon-blocked decisions

The single-launch correspondence between simulator time and `gpc__cycles_elapsed.max`, including the NCU measurement envelope needed to interpret elapsed GPC cycles, is undecided. Target-free source inspection cannot select that correspondence.

### Promotion forbidden until

Promotion is forbidden until admitted, launch-identity-bound silicon evidence validates a single-launch mapping under the recorded operating and capture state, and an explicit accepted decision binds that mapping. A checked-in target-free document alone is insufficient.

## D2 — Simulator diagnostic candidates

### Target-free findings

The campaign simulator exposes `diagnostic_last_issue_cycles` and
`terminal_modeled_event_time` per launch. Its retained result format treats
both as nonbinding diagnostics, and [launch aggregation](../sim/sm/src/replay/aggregation.rs)
preserves them in `launch_timing_diagnostics` without composing a launch-set
timing scalar. The [cycle emitter](../sim/run/src/emit.rs) receives a counter
snapshot that deliberately excludes both diagnostics.

### Hypotheses only

The two named diagnostics may be evaluated only as nonbinding, directional hypotheses. This study selects neither diagnostic, no function of either diagnostic, and no other simulator frontier as the binding cycle prediction.

### Silicon-blocked decisions

Whether either diagnostic has a stable single-launch relationship to the NCU observable remains undecided. Their names and units do not establish semantic equivalence, and target-free replay behavior cannot resolve that question.

### Promotion forbidden until

Promotion is forbidden until admitted silicon evidence distinguishes the candidate semantics, an accepted decision selects one lawful mapping, and a separate reviewed implementation changes the emission boundary. Diagnostics must remain per-launch and nonmetric until then.

## D3 — SM/GPC topology and spatial mapping

### Target-free findings

The admitted [sm_86 architecture card](../arch_profiles/cards/sm_86.json) provides total `sm_count`; it provides no admitted `gpc_count`, GPC roster, or SM-to-GPC map. That absence is a topology research gap, not evidence from which a GPC topology may be inferred. The registry's GPC counting unit and `max` roll-up therefore cannot be reconstructed from architecture metadata currently consumed by the simulator.

### Hypotheses only

An architecture-local spatial mapping may be studied after an authoritative topology source exists. This target-free study names no grouping, assigns no SM to a GPC, and invents no GPC count.

### Silicon-blocked decisions

The spatial reduction from modeled SM activity to per-GPC elapsed-cycle instances is undecided. Neither total SM count nor unrelated cache-slice geometry supplies the missing GPC topology.

### Promotion forbidden until

Promotion is forbidden until architecture-local authority for the exact SKU admits the topology and mapping required by the chosen semantics, and tests fail closed when that authority is absent. Evidence for one architecture or SKU must not be substituted for another.

## D4 — Launch composition and emission boundary

### Target-free findings

[Full launch-set aggregation](../sim/sm/src/replay/aggregation.rs) sums checked counters while retaining one timing-diagnostic record per launch in canonical roster order; it exposes no scalar temporal reducer. The real [`sim_run::run_manifest`](../sim/run/src/lib.rs) consumer serializes those diagnostics under each launch, marks aggregate timing composition `required_unimplemented_pending_aggregation_semantics_study`, and calls [`sim_run::emit::emit_all`](../sim/run/src/emit.rs). The aggregate `exec_cycles_gpc_max` row remains exactly `RequiredUnimplemented(AggregationSemanticsStudy)` and has no numeric binding value.

### Hypotheses only

A future multi-launch composition rule is a hypothesis class only. No reducer is named, selected, or encoded by this study, and per-launch preservation must not be mistaken for composition.

### Silicon-blocked decisions

How multiple launches compose at the result-emission boundary is undecided. Target-free evidence cannot establish launch overlap, serialization, observation scope, or a lawful scalar composition rule.

### Promotion forbidden until

Promotion is forbidden until admitted silicon evidence selects both the single-launch semantics and a multi-launch composition rule, policy accepts them, and a separate code change updates emitter and real-consumer regressions together. Completion of this target-free study does not itself promote emission.
