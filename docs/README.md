# Documentation

## Current contracts

- [Status](STATUS.md) — implemented paths, measured evidence, open boundaries.
- [Project memory](PROJECT_MEMORY.md) — compact continuation state for Codex and agents.
- [Architecture](ARCHITECTURE.md) and [components](COMPONENTS.md) — executable data flow.
- [Host runbook](runbook_host.md) — capture, NCU, and replay commands.
- [Architecture profiles](archprofile.md) — evidence-bound cards versus custom specs.
- [Custom program/spec inputs](../arch_profiles/programs/README.md) and [specs](../arch_profiles/specs/README.md).
- [Testing](testing.md) — bounded engineering checks, not a scientific gate.

## Research and evidence

- [Research positioning](gpu_simulator_research_positioning_20260923.md) — prior art and claim boundary.
- [Extensibility research](architecture_extensibility_research_20260924.md) — ISA/resource/compiler limits.
- [Roadmap review](roadmap_adversarial_review_20260923.md) — dated findings and dispositions.
- [Active roadmap](../plans/cross_architecture_gpu_simulator_research_plan_20260923.md).
- [Adversarial synthesis](adversarial_review_and_research_20260924.md) — latest implementation review.
- [SM86 measurement boundary](sm86_measurement_boundary_20260924.md) — 197,120 instruction-count mapping.
- [SM86 speed probe](sm86_speed_probe_20260923.md) and [feature coverage](sm86_feature_coverage_20260923.md).
- [Baseline matrix](baseline_matrix_20260924.json) — explicitly non-scoring overlap cells.

## Custom-model probes

- [Resource sweep](custom_architecture_design_sweep_20260923.md)
- [Pipeline overlap](custom_pipeline_overlap_20260924.md)
- [Production custom pipeline](custom_program_pipeline_20260924.md)
- [Shared memory](custom_shared_memory_20260924.md)
- [Group barrier](custom_group_barrier_20260924.md)
- [Shared atomic](custom_atomic_shared_20260924.md)
- [Broadcast](custom_collective_20260924.md) and [reduction](custom_reduce_20260924.md)
- [Independent differential](custom_simt_differential_20260923.md)
- [Curry-inspired ITS study](curry_gpu_its_study_20260923.md)

## Reference and history

The [Accel-Sim](accelsim_executable_overlap_probe_20260923.md), [derived
SM86](accelsim_derived_sm86_overlap_20260924.md), and [Huerta](huerta_micro25_executable_overlap_probe_20260923.md)
records are executable overlap evidence, not numerical rankings. Vortex,
SimX/RTL, ELF, and other superseded experiments remain under
[archive](archive/README.md); they are not active goals, dependencies, or
validation gates. Metric, predictor, tracing, and SOL documentation remains
available from the repository map but does not change the current scope.
