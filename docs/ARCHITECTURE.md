# Architecture

## Native trace replay

```text
NVBit -> enhanced protobuf + control sidecars
      -> sim-trace -> sim-sm <-> sim-mem
                         \-> sim-engine event ordering
                         \-> sim-run JSON -> independent NCU comparison
```

`sim-trace` reads trace evidence; `sim-sm` handles scheduling, dependencies,
barriers, and async operations; `sim-mem` models cache and memory service.
`sim-engine` supplies deterministic event ordering. Native replay does not
execute the original CUDA program again.

Native architecture cards select supported geometry/timing. Direct and prepared
replay consume the carded scheduler-resource blocks. Resource-less legacy
fixtures have an explicit compatibility path; partial blocks reject. An
optional scalar/profile override changes issue width, labels it hypothetical,
and disables independent dual-issue widening. It cannot alter traced semantics.

Package `sim-run` (`sim/direct_run`) provides the direct CLI and `sim_direct_run`
library. Package `sim-run-campaign` (`sim/run`, library `sim_run`) retains the
campaign CLI and shared emitters. NCU joins add observations after replay, never
model inputs. See [components](COMPONENTS.md) and [profiles](archprofile.md).

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
