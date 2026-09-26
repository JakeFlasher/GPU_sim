# Rust simulator workspace

The workspace contains two deliberately separate execution paths.

## Native path

- `sim-trace`: validated NVBit protobuf/control-sidecar readers and semantic adapter.
- `sim-sm`: native SM86 warp/CTA issue, dependency, barrier, and async replay.
- `sim-mem`: L1/TEX, L2, and memory-service models.
- `sim-engine`: deterministic event ordering.
- `sim-run` (`sim/direct_run`): direct trace replay CLI/library.
- `sim-run-campaign` (`sim/run`): retained manifest/campaign CLI/library.

Architecture cards are evidence-bound inputs. NCU joins are observations added
after replay and never alter simulator parameters.

## Bounded custom path

- `sim-arch::spec`: validated hypothetical resources and operation bindings.
- `sim-arch::simt`: fixed `simt_u32_v1` functional/timing semantics.
- `sim-arch::its`: separate single-group `simt_its_v1` functional example
  with optional `custom_its_dynamic_v1` execution records.
- `sim-arch::simt_pipeline`: register/memory dependency lowering.
- `sim-sm::custom` and `custom-program-run`: resident-group scheduling through
  the shared dependency/resource engine.

The custom backend supports model-relative width/latency/initiation/in-flight
experiments, shared memory, bounded CTA barriers, deterministic shared atomic
add, lane-0 broadcast, and wrapped-u32 reduction. It is not native SASS,
`sim-mem` cache timing, an arbitrary ISA/compiler, or silicon timing.

## Build and run

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
cargo build --manifest-path sim/Cargo.toml -p sim-sm --bin custom-program-run
cargo test --manifest-path sim/Cargo.toml --workspace --all-targets
```

For native commands see [`docs/runbook_host.md`](../docs/runbook_host.md). For
custom inputs see [`arch_profiles/programs/README.md`](../arch_profiles/programs/README.md).
Current limitations and the compact continuation record live in
[`docs/STATUS.md`](../docs/STATUS.md) and [`docs/PROJECT_MEMORY.md`](../docs/PROJECT_MEMORY.md);
dated reports are indexed by [`docs/HISTORY.md`](../docs/HISTORY.md).
