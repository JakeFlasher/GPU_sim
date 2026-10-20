# Architecture

## Native trace replay

```text
NVBit -> enhanced protobuf + control sidecars
      -> sim-trace -> sim-sm <-> sim-mem
                         \-> sim-engine event ordering
                         \-> sim-run JSON -> independent NCU comparison
```

`sim-trace` reads trace evidence; `sim-sm` handles warp scheduling,
dependencies, barriers, and asynchronous operations; `sim-mem` models cache
and memory service; and `sim-engine` supplies deterministic event ordering.
Native replay does not execute the original CUDA program again.

Architecture cards select the supported geometry and timing assumptions.
Direct and prepared replay consume the card's scheduler-resource blocks.
Resource-less legacy trace input has an explicit compatibility path; partial
blocks are rejected. An optional scalar/profile override changes issue width,
is labeled hypothetical, and cannot alter traced semantics.

Package `sim-run` (`sim/direct_run`) provides the direct CLI and `sim_direct_run`
library, including the native metric emitter. NCU joins add observations after
replay, never model inputs. See [components](COMPONENTS.md) and
[profiles](archprofile.md).

## Custom functional/resource model

```text
ArchitectureSpec + SimtProgram
  -> sim-arch::simt (functional state)
  -> sim-arch::simt_pipeline (dependencies and resource tasks)
  -> sim-sm::custom (resident groups and bounded shared barriers)
  -> sim-engine::pipeline (issue/completion events)
  -> custom-program-run JSON (functional state + model-relative timing)
```

The custom frontend shares the event engine, not the native SM86 scoreboard or
`sim-mem` cache hierarchy. It supports in-order per-group issue with independent
resource overlap, latency/initiation/in-flight limits, and bounded residency.
The `simt_u32_v1` instruction set is fixed; resource bindings do not define new
instruction semantics. [Program contracts](../arch_profiles/programs/README.md)
state address-space, barrier, atomic, and collective limits.

The separate `simt_its_v1` executor demonstrates per-lane control flow and
reconvergence without a timing backend. Neither custom frontend promises
arbitrary binary/ABI compatibility, vendor-equivalent timing, or RTL behavior.

## Counter attribution and locality analysis

`sim/direct_run/src/metrics.rs` produces versioned native metrics after replay;
counter-attribution reports do not add timing events. The locality observers in
`sim/sm/src/metrics.rs` operate over declared trace order and remain separate
from the live cache and timing models. The separate ITS dynamic JSON trace is
not input to `sim-trace`. See
[diagnostic contract](experimental_diagnostics.md) and
[terminology](TERMINOLOGY.md).
