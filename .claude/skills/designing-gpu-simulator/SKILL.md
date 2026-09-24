---
name: designing-gpu-simulator
description: Design or implement the trace-driven Rust GPU simulator, native SM86 replay, bounded custom SIMT/resource models, architecture cards, NCU-shaped output, or simulator-performance probes.
---

# Designing the GPU simulator

Read root `AGENTS.md`, `docs/STATUS.md`, and `docs/PROJECT_MEMORY.md` before
changing simulator code. The active path is software architectural simulation:

```text
NVBit/SASS trace -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run -> NCU comparison
custom spec/program -> sim-arch -> sim-sm::custom -> sim-engine -> custom-program-run
```

## Source ownership

- `sim/trace`: NVBit protobuf, sidecars, coverage, semantic adapter.
- `sim/arch`: evidence-bound cards plus bounded `simt_u32_v1` and `simt_its_v1` examples.
- `sim/arch/src/simt_pipeline.rs`: fixed custom ISA dependency/resource lowering.
- `sim/sm`: native warp/CTA replay; `sim/sm::custom`: bounded resident groups/barriers.
- `sim/mem`: native L1/L2/DRAM service; `sim/engine`: deterministic events and generic pipeline.
- `sim/direct_run`: package `sim-run`, including the narrow `trace_sass_v1` profile bridge.
- `sim/run`: retained package `sim-run-campaign` and emitters.
- `arch_profiles`: cards, specs, programs, and JSON schemas.
- `tools/custom_simt_differential.py`: independent bounded oracle, not silicon truth.

## Scientific boundaries

- NCU/CUPTI measurements are independent labels; never feed scored values into
  features, tuning, or implementation choices.
- Keep issue, modeled-event, completion, launch, and unmodeled-tail frontiers
  separate; do not collapse them into a selection score.
- A fixed SASS trace supports only `trace_safe` interventions. Changes that can
  alter control flow, addresses, values, atomics, or participants require a
  functional frontend or recapture.
- The custom frontend is fixed-ISA/model-relative. It is not a compiler, ABI,
  vendor-neutral proprietary decoder, cache-equivalent SM, RTL model, or
  silicon-equivalent timing reconstruction.
- Fair Accel-Sim/MGPUSim numerical comparison is deferred until a common
  hardware/trace/observable/semantics cell exists.

## Implementation loop

Use checked arithmetic and typed errors. Run the smallest executable path after
each change. For custom changes, use the relevant program and
`tools/custom_simt_differential.py`; for native changes, replay a retained
canary or representative trace. Then run focused package tests and only the
workspace checks needed to support the change. A test suite is engineering
evidence, not a workflow gate.

## Useful commands

```bash
cargo test --manifest-path sim/Cargo.toml -p sim-arch --lib
cargo test --manifest-path sim/Cargo.toml -p sim-sm --lib
cargo test --manifest-path sim/Cargo.toml -p sim-run --lib
cargo build --manifest-path sim/Cargo.toml -p sim-sm --bin custom-program-run
```

Do not add approval, hash/seal, receipt, audit-loop, or reviewer-swarm
machinery. Historical SOL/campaign artifacts remain retained but are not
active requirements.
