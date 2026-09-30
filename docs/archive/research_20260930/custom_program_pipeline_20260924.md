# Production custom-program pipeline probe — 2026-09-24

`sim-sm` now exposes a bounded custom-program backend that lowers declarative
SIMT programs into the shared dependency/resource event engine. This is
separate from native NVBit/SASS replay and remains model-relative, but it is no
longer only a standalone `sim-arch` toy timing binary.

## Command

```bash
cargo build --manifest-path sim/Cargo.toml -p sim-sm --bin custom-program-run
sim/target/debug/custom-program-run \
  arch_profiles/specs/simt_u32_pipeline2.json \
  arch_profiles/programs/simt_u32_pipeline_overlap.json \
  arch_profiles/programs/simt_u32_lane_store_300.json
```

## Observed result

The run admitted two resident groups concurrently and emitted:

```text
backend: sim_sm_custom_event_pipeline
groups: 2
group 0: admission 0, retirement 45
group 1: admission 0, retirement 37
final_cycle: 45
completion_events: 10
```

The JSON result now also reports one row per declared resource. For this probe
the ALU resource carries 5 tasks/5 service cycles and the memory resource
carries 3 tasks/24 service cycles; the completion-based peak in-flight
populations are 2 for ALU and 3 for memory.
Utilization is reported against the model's final-cycle/resource-width
capacity; for resources with an initiation interval above one, the denominator
now counts only the corresponding initiation slots (`ceil(final_cycle / II) *
issue_width`). The report also exposes this denominator as
`ideal_work_capacity`. It is a design-space diagnostic, not a silicon
utilization counter. `peak_in_flight` follows scheduled completion rather than
the final initiation beat, so it describes the same outstanding-work domain as
the resource's `max_in_flight` limit.

The result additionally emits `schedule_diagnostics`, a causal decomposition
for architecture-design sweeps. The two-group probe reports a completion
frontier of 45 cycles, a 32-cycle explicit dependency-path bound, a 24-cycle
serialized memory-work bound, and a 13-cycle residual gap. `bottleneck_resource`
is `memory`. These are conservative bounds and a residual interval within the
declared event model; they are not elapsed-silicon or NCU counters, and the
gap is deliberately not attributed to a proprietary hardware cause.

The backend rejects global conflicts and same-phase shared conflicts; a
matching `group_barrier` allows ordered shared producer/consumer accesses and
propagates values through the shared map. It does not invent memory consistency
from host iteration order. Initial global/shared words are visible to every
group in this bounded run; final global maps remain per-group because
cross-group global writes are rejected. It currently supports the bounded
`simt_u32_v1` semantics;
arbitrary user-defined instruction semantics, vendor SASS, and multi-GPU
behavior remain separate frontends.

Matching `group_barrier` generations are now supported as cross-group ordering
dependencies with separated shared-memory producer/consumer propagation; see
`custom_group_barrier_20260924.md`.
