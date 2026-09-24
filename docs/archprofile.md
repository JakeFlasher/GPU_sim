# Architecture profiles

Architecture cards live in `arch_profiles/cards` and are the only source for
architecture-specific geometry and timing defaults.

```text
arch token -> profile loader -> ArchProfile -> sim-arch/sim-sm/sim-mem
```

The active cards are `sm_86` and exploratory `sm_120`. A card carries device
identity, compute capability, SM/cache geometry, latency values, and explicit
research-gap markers for unsupported behavior. The loader rejects unknown or
ambiguous tokens instead of borrowing another architecture's values.

To add a card:

1. Add the JSON card and update the schema.
2. Keep values that are not measured as typed research gaps.
3. Add the card to the loader's profile list.
4. Run a release build and replay a representative trace.

The simulator reads cards at launch; workload scripts must not embed card
constants.

The SM86 scheduler-resource block (`smsp`, `occupancy`, `dep_barriers`,
`barrier_latencies`, and `dual_issue`) is an architecture-bound input surface,
not proof that every value was independently microbenchmarked. In particular,
the `25`/`8` barrier latencies and occupancy ceilings retain their
architecture-local reviewed-hypothesis status; card binding prevents identity
mixing but does not upgrade provenance to silicon measurement. The result
wire's `architecture_card` label means source binding, while the simulator's
model-relative timing and NCU independence rules still apply.

Experimental user-defined shapes are separate from cards under
`arch_profiles/specs`. They are consumed by `sim-arch::spec` and, for the
strict `trace_sass_v1` issue-width profile only, by `sim-run --architecture-spec`.
They carry an explicit evidence status and cannot silently provide SM86/SM120
production defaults. Latency, initiation, in-flight, extra-resource, and
operation-binding fields are rejected by that native bridge when unused. See
the [custom spec guide](../arch_profiles/specs/README.md) and the
[current roadmap](../plans/cross_architecture_gpu_simulator_research_plan_20260923.md).

The dated host profile under
`workloads/research_probes/sm86_ncu_demo_benchmark/host_profile.json` records
the 2026-09-24 RTX 3090 Ti/driver/clock observation. It is not a replacement
for architecture-card provenance or an assertion that future hosts match it.
