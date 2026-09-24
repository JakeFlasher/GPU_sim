# Documentation

- [Research and positioning memo](gpu_simulator_research_positioning_20260923.md) — web-researched scope, prior art, claim boundary, and paper agenda.
- [Adversarial roadmap review](roadmap_adversarial_review_20260923.md) — concrete claim gaps, prior-art collisions, and stop/fallback rules.
- [SM86 measurement boundary](sm86_measurement_boundary_20260923.md) — first paired NCU/NVBit/simulator probe and current mapping blocker.
- [SM86 speed and override probe](sm86_speed_probe_20260923.md) — first production issue-width override and event/dense wall-time measurement.
- [SM86 feature coverage](sm86_feature_coverage_20260923.md) — first opcode-family, trace-gap, and trace-safety inventory.
- [SM86 issue-width design study](sm86_issue_width_design_study_20260923.md) — first production resource intervention and negative speed result.
- [Baseline feasibility](baseline_feasibility_20260923.md) — first M6 build/overlap probe; no invalid numeric comparison reported.
- [Baseline matrix](baseline_matrix_20260924.json) — machine-readable valid/unsupported M6 cells with no forced ranking.
- [Accel-Sim executable overlap](accelsim_executable_overlap_probe_20260923.md) — native-trace build/replay evidence with explicit RTX3070/trace-contract limits.
- [Huerta MICRO'25 executable overlap](huerta_micro25_executable_overlap_probe_20260923.md) — same-enhanced-trace probe with explicit RTX3090/protobuf limits.
- [CurryGPU functional-architecture study](curry_gpu_its_study_20260923.md) — local primary-source analysis and the bounded functional ITS seam.
- [General software GPU simulator roadmap](../plans/cross_architecture_gpu_simulator_research_plan_20260923.md) — active implementation direction and NCU/speed completion bar.
- [Custom architecture design sweep](custom_architecture_design_sweep_20260923.md) — model-relative SIMT/async resource experiment.
- [Custom SIMT differential](custom_simt_differential_20260923.md) — bounded functional/timing feasibility probe.
- [Architecture](ARCHITECTURE.md) — executable components and data flow.
- [Components](COMPONENTS.md) — crate and tool responsibilities.
- [Status](STATUS.md) — current SM86 path, custom-model boundary, and next evidence.
- [Host runbook](runbook_host.md) — build, capture, and replay commands.
- [NVBit tracing](nvbit-tracing.md) — trace production and replay inputs.
- [Metric registry](metric-registry.md) — metric names and units.
- [Architecture profiles](archprofile.md) — device-card inputs.
- [Predictors](predictors.md) — optional comparison models.
- [SOL-ExecBench](sol-execbench.md) — opt-in benchmark integration.

Historical Vortex SimX/rtlsim, ELF, and cross-model experiments are retained
under [docs/archive/vortex](archive/vortex/); they are not active goals,
validation gates, or required dependencies.
