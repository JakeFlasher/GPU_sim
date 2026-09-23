# Documentation

- [General GPU simulator roadmap](../plans/cross_architecture_gpu_simulator_research_plan_20260923.md) — current direction, SOTA context, user-defined architecture path and completion evidence.
- [Custom architecture design sweep](custom_architecture_design_sweep_20260923.md) — reproducible model-relative SIMT/async resource experiment.
- [Custom SIMT differential](custom_simt_differential_20260923.md) — independent Python functional/timing oracle.
- [Vortex configuration adapter](vortex_config_adapter_20260923.md) — structural open-GPU config import with explicit unvalidated limits.
- [Vortex SimX/RTL differential](vortex_vecadd_simx_rtlsim_20260923.md) — independent functional parity result.
- [Vortex regression timing differential](vortex_vecadd_regression_timing_20260923.md) — repeated native SimX/RTL cycle comparison.
- [Vortex ELF summary](vortex_elf_summary_20260923.md) — static RISC-V/open-ISA ingestion with explicit unknowns.
- [Vortex ELF vecadd adapter](vortex_elf_vecadd_adapter_20260923.md) — narrow real-binary pattern adapter with fail-closed recognition.
- [Vortex adapted differential](vortex_elf_adapted_differential_20260923.json) — real-ELF-adapted program vs independent oracle.
- [Vortex shared semantic probe](vortex_elf_shared_semantic_probe_20260923.md) — narrow real-ELF pattern projected through Rust `sim-semantics`.
- [Adversarial code review](adversarial_code_review_20260923.md) — concrete fixes, validation commands, and bounded residuals for the cross-architecture seams.

- [Architecture](ARCHITECTURE.md) — executable components and data flow.
- [Components](COMPONENTS.md) — crate and tool responsibilities.
- [Status](STATUS.md) — current SM86 state, evidence, and next work.
- [Replay regeneration](sm86_replay_regeneration_20260905.md) — latest full small/large replay and regression results.
- [Post-merge campaign comparison](postmerge_campaign_comparison_20260909.md) — clock-qualified 8-small/7-large campaign, NCU observation joins, causal probe, and same-input non-regression.
- [SM86 semantic round-trip](sm86_semantic_roundtrip_20260912.md) — semantic-event adapter and cycle-accurate replay equivalence results.
- [SM86 replay determinism](sm86_replay_determinism_20260918.md) — repeated event/dense process outputs for the retained canary.
- [SM86 current-source replay](sm86_current_source_replay_20260923.md) — retained-trace non-regression check against the active binary.
- [SM86 replay result](sm86_current_source_replay_20260923.json) — machine-readable release replay comparison.
- [Adversarial audit](adversarial_audit_20260905.md) — confirmed trace-input fixes, replay evidence, and open probes.
- [Host runbook](runbook_host.md) — build, capture, and replay commands.
- [NVBit tracing](nvbit-tracing.md) — trace production and replay inputs.
- [Metric registry](metric-registry.md) — metric names and units.
- [Architecture profiles](archprofile.md) — device-card inputs.
- [Predictors](predictors.md) — optional comparison models.
- [SOL-ExecBench](sol-execbench.md) — opt-in benchmark integration.
