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
