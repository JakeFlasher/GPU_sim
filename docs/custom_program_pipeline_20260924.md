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
