---
name: designing-gpu-simulator
description: "Design or implement native SM86 Rust replay, bounded custom SIMT execution, architecture inputs, metric projections, or simulator performance experiments."
---

# Simulator engineering routes

Root `AGENTS.md` owns policy. Resume from `docs/PROJECT_MEMORY.md` and use
`docs/STATUS.md` for claims; do not reload all dated research.

| Seam | Source |
| --- | --- |
| Protobuf/sidecars/coverage | `sim/trace` |
| Cards/capabilities/specs | `sim/arch`, `arch_profiles/README.md` |
| Native issue/dependencies/async | `sim/sm` |
| L1/L2 and shared-system memory | `sim/mem`, `sim/sm/src/system.rs` |
| Custom resource/dependency execution | `sim/arch/src/simt_pipeline.rs`, `sim/sm/src/custom.rs`, `sim/engine/src/pipeline.rs` |
| Functional per-lane ITS | `sim/arch/src/its.rs` |
| Direct/prepared CLI | `sim/direct_run` (package `sim-run`, library `sim_direct_run`) |
| Retained campaign emitter | `sim/run` (package `sim-run-campaign`) |

Native metrics' completion scenarios and count-balanced pipe choices are
observers, not timing mechanisms. The custom ITS JSON trace is not native SASS;
see `docs/experimental_diagnostics.md`. Keep NCU labels, functional results,
model timing and host performance separate.

Use `--manifest-path sim/Cargo.toml` from the repository root. First run the
smallest affected executable, then focused tests. Commands: `docs/testing.md`
and `docs/runbook_host.md`. For prior art use `docs/HISTORY.md` only as needed.
