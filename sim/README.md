# Rust simulator workspace

The workspace contains two deliberately separate execution paths.

## Native path

- `sim-trace`: validated NVBit protobuf/control-sidecar readers and semantic adapter.
- `sim-sm`: native SM86 warp/CTA issue, dependency, barrier, and async replay.
- `sim-mem`: L1/TEX, L2, and memory-service models.
- `sim-engine`: deterministic event ordering.
- `sim-run` (`sim/direct_run`): direct trace replay CLI/library.
- `sim-run-campaign` (`sim/run`): retained manifest/campaign CLI/library.
- `sim-semantics::latent`: opt-in unknown-preserving route/compiler/cache/
  timing candidate domains and endpoint/frontier contract; it does not change
  the native replay path or claim vendor-equivalent hardware behavior.

Architecture cards are evidence-bound inputs. NCU joins are observations added
after replay and never alter simulator parameters.

`sim-run --fast-frontier-only` is an opt-in PhaseFuse analysis lane. It reads
the trace/control/card contract, skips native timing replay, and emits only
conservative scalar-region certificate/refusal evidence. It is not a timing
replacement; see [`docs/sm86_fast_exact_prototype_20260927.md`](../docs/sm86_fast_exact_prototype_20260927.md).

`sim-run --latent-envelope` is a separate opt-in diagnostic projection. It
adds unknown-preserving route/class bounds and simulator-only frontier/endpoint
metadata; ordinary output is unchanged when the flag is absent.

## Bounded custom path

- `sim-arch::spec`: validated hypothetical resources and operation bindings.
- `sim-arch::simt`: fixed `simt_u32_v1` functional/timing semantics.
- `sim-arch::ptx_subset`: bounded `ptx_subset_v2` text-to-ITS functional
  frontend, including deterministic indirect atomic-add while explicitly
  rejecting unsupported ABI, fence/memory-model and ISA forms.
- `sim-arch::its`: separate single-group `simt_its_v1` functional example
  with optional `custom_its_dynamic_v3` execution records.
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
Current limitations, the predictive replacement contract, and the compact continuation record live in
[`docs/STATUS.md`](../docs/STATUS.md), [`docs/PREDICTIVE_CLOSURE.md`](../docs/PREDICTIVE_CLOSURE.md),
and [`docs/PROJECT_MEMORY.md`](../docs/PROJECT_MEMORY.md); dated reports are indexed by
the active docs map. The third-round corpus result and claim boundary are in
[`docs/review_round3_20261002/README.md`](../docs/review_round3_20261002/README.md)
and [`plans/third_round_rebuttal_20261002.md`](../plans/third_round_rebuttal_20261002.md).
