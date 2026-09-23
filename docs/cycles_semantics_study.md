# P8 target-free cycles-semantics boundary study

> Retained campaign reference. This document records the historical strict
> campaign interpretation of cycle metrics. It is not a repository-wide gate,
> does not restrict direct simulator or NCU/NVBit research, and applies only
> when the opt-in SOL/campaign skill is deliberately invoked.

**Status:** Target-free boundary record with a model-relative reducer now
implemented in `sim-run`. This document does not promote the reducer to an
NCU/GPC or silicon-equivalent cycle mapping; hardware promotion remains
blocked.

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
`terminal_modeled_event_time` per launch. [Launch aggregation](../sim/sm/src/replay/aggregation.rs)
preserves them in `launch_timing_diagnostics`; the [cycle emitter](../sim/run/src/emit.rs)
uses a checked simulator-scope max reducer over issue, terminal-event-plus-one,
and modeled-completion frontiers. The emitted value is typed
`BindingCycles`, not an NCU prediction.

### Hypotheses only

The named diagnostics remain nonbinding diagnostics; the reducer is a
mechanism-local, model-relative hypothesis. This study selects neither diagnostic
as a silicon mapping and selects no GPC envelope semantics.

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

[Full launch-set aggregation](../sim/sm/src/replay/aggregation.rs) sums checked counters while retaining one timing-diagnostic record per launch in canonical roster order. The emitter now applies an explicit model-relative reducer: for each launch it takes the maximum of issue cycles, terminal-event timestamp plus one, and modeled completion lower bound; across launches it takes the maximum again. The aggregate `exec_cycles_gpc_max` row is a typed `BindingCycles` value, never a sum or silicon measurement.

### Hypotheses only

The reducer is deliberately mechanism-local and model-relative. It does not claim the profiler's GPC envelope or establish launch overlap on hardware.

### Silicon-blocked decisions

Hardware launch overlap, serialization, and GPC observation scope remain undecided; the simulator rule is not a substitute for those measurements.

### Promotion forbidden until

Hardware-validated promotion remains forbidden until silicon evidence selects the corresponding launch-envelope semantics. The implemented reducer is limited to simulator-scope output.
