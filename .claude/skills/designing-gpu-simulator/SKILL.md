---
name: designing-gpu-simulator
description: Design and implement the event-driven, trace-driven Rust GPU simulator in sim/. Use for the event engine, SM/warp issue model, memory hierarchy, trace replay, architecture cards, direct runner, NCU-shaped outputs, calibration, or simulator performance work.
---

# Design the event-driven GPU simulator

The active project builds a fast Rust simulator for Ampere `sm_86`, followed
by architecture-appropriate Blackwell `sm_120` support. Work directly from
code, traces, architecture cards, focused experiments, and tests. Do not create
audit companions, admission ceremonies, frozen candidates, or review swarms.

## Current components

- `sim/engine`: deterministic time-ordered event queue.
- `sim/trace`: enhanced NVBit protobuf reader and trace semantics.
- `sim/sm`: control-bit join, warp issue, event-driven replay, and SM state.
- `sim/mem`: sectored L1, shared L2, MSHRs, and DRAM timing.
- `sim/arch`: architecture/SKU cards and explicit research gaps.
- `sim/direct_run`: default `sim-run` CLI over ordinary input paths.
- `sim/run`: retained optional `sim-run-campaign` manifest interface.

The direct replay API is `sim_sm::replay_trace_kernel`. The historical
campaign adapter is compiled only with the `sim-sm/campaign` feature.

## Design sources

- Founding note:
  `references/papers/notes/draft_new_gpu_simulator.md`
- Execution plan:
  `plans/micro2027_ncu_sim_execution_plan.md`
- Progress log:
  `plans/phase2_sim_core_progress.md`
- Simulator cards:
  `.claude/knowledge/simulator/`
- Prior art:
  `references/papers/knowledge_base/`

## Ground rules

- Batch local actor progress; avoid global per-cycle scans and one queued event
  per instruction.
- Read SASS control bits from the sidecar; do not invent scoreboard state when
  the trace already supplies the dependency information.
- Keep one typed memory-mode selector: flat, L1, or L1+shared-L2.
- NCU observations are comparison targets, never same-cell simulator inputs.
- Architecture constants come from vendor documentation or focused
  microbenchmarks, not per-workload label fitting.
- Preserve unknown microarchitecture as an explicit hypothesis or missing
  model.
- Use checked arithmetic and typed errors on count/time boundaries.

## Direct commands

```bash
cd sim
cargo test -p sim-engine
cargo test -p sim-trace
cargo test -p sim-sm
cargo test -p sim-mem
cargo test -p sim-run

cargo run -p sim-run -- \
  --trace TRACE_DIR \
  --control CONTROL_BITS.json \
  --arch-card ARCH_CARD.json \
  --mode flat
```

Use `cargo test -p sim-run-campaign` only when the user explicitly asks for
the retained campaign interface.

## Related skills

- PTX semantics: `interpreting-ptx-isa`
- SASS and tracing: `analyzing-sass-nvbit-traces`
- Blackwell kernels: `tuning-blackwell-sm120-kernels`
- Optional SOL/campaign operation: `run-sol-campaign`
