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
- [SM86 metric expansion](sm86_metric_expansion_20260924.md) — trace-derived activity, occupancy, and expanded NCU observations.
- [SM86 metric comparison CSV](sm86_metric_comparison_20260924.csv) — paired SM86 simulator/NCU values for the fresh three-workload cohort.
- [SM86 metric-domain experiment](sm86_metric_domain_experiment_20260924.csv) — small/large read, ATOM, RED, and control workload comparisons.
- [SM86 writeback experiment](sm86_writeback_experiment_20260924.csv) — deterministic store streams separating dirty-demand topology from NCU DRAM-write interval counts.
- [SM86 coherence domain](sm86_coherence_domain_20260925.md) — bounded cross-SM global-line and 32-byte-sector contention domains without claiming invalidation/writeback completion.
- [SM86 memory topology](sm86_memory_topology_20260925.md) — distinct decoded sectors/lines and repeated ATOM/RED span aliases across the retained holdout.
- [Rebuilt tracer follow-up](sm86_rebuilt_tracer_followup_20260925.md) — fresh rebuilt-NVBit read/ATOM traces with independent NCU count checks.
- [SM86 module-origin follow-up](sm86_module_origin_followup_20260926.md) — module/library enumeration and hashed accepted-launch provenance.
- [SM86 read-bandwidth projection](sm86_read_bandwidth_projection_20260926.md) — read-only modeled throughput estimates kept separate from total NCU DRAM throughput.
- [SM86 direct DRAM-rate capture](sm86_dram_rate_capture_20260926.md) — direct GA102 NCU read-rate/throughput labels and Rust round-trip.
- [SM86 speed cohort](sm86_speed_cohort_20260927.md) — phase-separated event-driven release replay across GEMM, RMSNorm, and SiLU.
- [NCU/HyFiSS profiler crosswalk](nvprof_hyfiss_crosswalk_20260926.md) — MICRO'24 profiler vocabulary, implementation crosswalk, and claim boundary.
- [SM86 profiler crosswalk holdout](sm86_profiler_crosswalk_holdout_20260926.csv) — fresh three-workload NCU observations for the newly retained profiler families.
- [SM86 branch-domain experiment](sm86_branch_domain_experiment_20260925.csv) — divergent and uniform branch-target counts against NCU.
- [SM86 branch holdout](sm86_branch_holdout_20260925.csv) — branch/CBU counts on retained GEMM, RMSNorm, and SiLU workloads.
- [SM86 scheduler-stall holdout](sm86_scheduler_stall_holdout_20260925.csv) — independent NCU scheduler samples retained beside simulator causal readiness.
- [SM86 full metric holdout](sm86_full_metric_holdout_20260925.csv) — pipelines, memory requests/sectors, shared wavefronts, L2/DRAM, and branch counts across retained workloads.
- [SM86 physical-pipeline probe](sm86_physical_pipeline_probe_20260925.csv) — independent GA102 NCU ADU/CBU/ALU/FMA/FP64/LSU/Tensor/Uniform/XU rows with explicit native attribution status.
- [SM86 pipeline-category probe](sm86_pipeline_category_probe_20260925.md) — independent SHF/SHFL/control/conversion category characterization.
- [SM86 trace-structure holdout](sm86_trace_structure_holdout_20260925.csv) — CFG, coherence-domain, and physical-pipeline-confidence summaries across retained workloads.
- [Host runbook](runbook_host.md) — CUPTI resource-only capture for driver-selected shared-memory configuration.
- [NVBit build blocker resolution](nvbit_build_blocker_resolution_20260925.md) — pinned CUDA 13.2/GCC 14/protobuf build and runtime validation.
- [Baseline matrix](baseline_matrix_20260924.json) — explicitly non-scoring overlap cells.
- [GPU simulator SOTA matrix](gpu_simulator_sota_matrix_20260924.md) — public numerical results, normalized metric families, and project capability gaps.
- [GPU simulator SOTA matrix CSV](gpu_simulator_sota_matrix_20260924.csv) — machine-readable extraction of the published and local numerical rows.

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
