# Non-Root Measurement Paper Playbook

Use this file to jump from a current research problem to concrete PDFs in
`references/papers/knowledge_base/active/`. Every recommendation below must be
read under the project contract: no NCU collection, no GPU hardware counters,
no clock locking, no privileged driver changes, and no simulator revival.

## 1. Building The NVBit Trace Surface

Open these when the question is what dynamic SASS, memory, launch, stream, or
phase evidence can be captured from user space.

- [NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs](active/2019__nvbit_a_dynamic_binary_instrumentation_framework_for_nvidia_gpus.pdf) — primary reference for dynamic binary instrumentation on NVIDIA GPUs.
- [Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs](active/2019__low_overhead_instruction_latency_characterization_for_nvidia_gpgpus.pdf) — useful for low-overhead instruction timing methodology.
- [A Hybrid Framework for Fast and Accurate GPU Performance Estimation through Source-Level Analysis and Trace-Based Simulation](active/2019__a_hybrid_framework_for_fast_and_accurate_gpu_performance_estimation_through_source_level_analysi.pdf) — useful for combining static/source features with dynamic trace evidence.
- [Integrating Per-Stream Stat Tracking into Accel-Sim](active/2023__integrating_per_stream_stat_tracking_into_accel_sim.pdf) — prior art for per-stream attribution; use the idea without adopting the simulator objective.

## 2. Predicting Wall-Clock Time Without Counters

Use these when the target is kernel or application elapsed time from allowed
signals such as CUDA events, CUPTI Activity timestamps, host timing, launch
metadata, static SASS features, and NVBit traces.

- [A Performance Prediction Model for the CUDA GPGPU Platform](active/2009__a_performance_prediction_model_for_the_cuda_gpgpu_platform.pdf) — foundational runtime-prediction structure.
- [An Analytical Model for a GPU Architecture with Memory-Level and Thread-Level Parallelism Awareness](active/2009__an_analytical_model_for_a_gpu_architecture_with_memory_level_and_thread_level_parallelism_awaren.pdf) — early MLP/TLP-aware timing model.
- [A Quantitative Performance Analysis Model for GPU Architectures](active/2011__a_quantitative_performance_analysis_model_for_gpu_architectures.pdf) — bottleneck-oriented prediction method.
- [GPUMech: GPU Performance Modeling Technique Based on Interval Analysis](active/2014__gpumech_gpu_performance_modeling_technique_based_on_interval_analysis.pdf) — interval-analysis model for runtime prediction.
- [GCoM: A Detailed GPU Core Model for Accurate Analytical Modeling of Modern GPUs](active/2022__gcom_a_detailed_gpu_core_model_for_accurate_analytical_modeling_of_modern_gpus.pdf) — high-signal analytical core model; mine it for features and constraints, not for simulator construction.
- [SynPerf: A Hybrid Analytical-ML Framework for GPU Performance Prediction](active/2026__synperf_a_hybrid_analytical_ml_framework_for_gpu_performance_prediction.pdf) — current hybrid analytical/ML direction.

## 3. Inferring Cache Hit Ratio Or Cache-Behavior Proxies

Open these when predicting cache hit ratio, miss tendency, sector pressure, or
reuse behavior from traces and static features rather than hardware counters.

- [A Detailed GPU Cache Model Based on Reuse Distance Theory](active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf) — primary prior art for cache-hit/miss prediction without direct counters.
- [Dissecting GPU Memory Hierarchy through Microbenchmarking](active/2015__dissecting_gpu_memory_hierarchy_through_microbenchmarking.pdf) — memory-hierarchy behavior and microbenchmark structure.
- [MDM: The GPU Memory Divergence Model](active/2020__mdm_the_gpu_memory_divergence_model.pdf) — memory divergence and irregular-access performance behavior.
- [DeLTA: GPU Performance Model for Deep Learning Applications with In-Depth Memory System Traffic Analysis](active/2019__delta_gpu_performance_model_for_deep_learning_applications_with_in_depth_memory_system_traffic_a.pdf) — memory-traffic modeling ideas for DL kernels.
- [Exploring Modern GPU Memory System Design Challenges through Accurate Modeling](active/2018__exploring_modern_gpu_memory_system_design_challenges_through_accurate_modeling.pdf) — prior-art warning about memory-model assumptions; use as validation caution only.

## 4. Understanding Modern NVIDIA Behavior Through Microbenchmarks

Use these to interpret observed timing and trace features on specific
architectures. They are references for public behavior, not permission to rely
on privileged counters.

- [Demystifying the Nvidia Ampere Architecture through Microbenchmarking](active/2022__demystifying_the_nvidia_ampere_architecture_through_microbenchmarking.pdf) — SM80 baseline behavior.
- [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](active/2024__benchmarking_and_dissecting_the_nvidia_hopper_gpu_architecture.pdf) — SM90 behavior including newer execution and memory features.
- [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis](active/2025__dissecting_the_nvidia_hopper_architecture_through_microbenchmarking_and_multiple_level_analysis.pdf) — second Hopper view.
- [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](active/2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks.pdf) — Blackwell behavior for current-generation interpretation.
- [Microbenchmarking NVIDIA's Blackwell Architecture](active/2025__microbenchmarking_nvidia_s_blackwell_architecture.pdf) — additional Blackwell microbenchmark evidence.
- [Analyzing Modern NVIDIA GPU Cores](active/2025__analyzing_modern_nvidia_gpu_cores.pdf) — useful for visible pipeline assumptions behind timing models.

## 5. Designing Calibration And Error Analysis

Open these when deciding train/test splits, calibration protocols, sampling,
error bounds, and trust metrics for predictors.

- [Cross-Architecture Performance Prediction (XAPP)](active/2015__cross_architecture_performance_prediction_xapp.pdf) — cross-architecture prediction framing.
- [A Simple Model for Portable and Fast Prediction of Execution Time and Power Consumption of GPU Kernels](active/2021__a_simple_model_for_portable_and_fast_prediction_of_execution_time_and_power_consumption_of_gpu_k.pdf) — lightweight portable prediction approach.
- [Analytical Performance Estimation during Code Generation on GPUs](active/2022__analytical_performance_estimation_during_code_generation_on_gpus.pdf) — fast estimation during configuration search.
- [Prediction of Performance and Power Consumption of GPGPU Applications](active/2023__prediction_of_performance_and_power_consumption_of_gpgpu_applications_thesis.pdf) — survey-style support for prediction methodology.
- [NeuroScalar: A Deep Learning Framework for Fast, Accurate Cycle-Level Performance Prediction](active/2025__neuroscalar_a_deep_learning_framework_for_fast_accurate_cycle_level_performance_prediction.pdf) — learned-surrogate reference; useful when evaluating ML error controls.
- [STEM: Swift and Trustworthy Large-Scale GPU Simulation with Fine-Grained Error Modeling and Hierarchical Clustering](active/2025__stem_swift_and_trustworthy_large_scale_gpu_simulation_with_fine_grained_error_modeling_and_hiera.pdf) — keep the error-modeling and sampling ideas, not the simulator goal.

## 6. Legacy Simulator Literature

These PDFs are retained only as background for trace schemas, validation
pitfalls, and the history of GPU performance modeling. Do not use this section
as an implementation plan.

- [GPGPU-Sim](active/2009__analyzing_cuda_workloads_using_a_detailed_gpu_simulator_gpgpu_sim.pdf)
- [Accel-Sim](active/2020__accel_sim_an_extensible_simulation_framework_for_validated_gpu_modeling.pdf)
- [NVArchSim](active/2021__need_for_speed_experiences_building_a_trustworthy_system_level_gpu_simulator_nvarchsim.pdf)
- [MGPUSim](active/2019__mgpusim_enabling_multi_gpu_performance_modeling_and_optimization.pdf)
- [MAccel-Sim](active/2024__maccel_sim_a_multi_gpu_simulator_for_architectural_exploration.pdf)

## When The Playbook Is Wrong

If the papers require hardware counters, NCU, root access, fixed clocks, or a
simulator as a validation target, mark that limitation explicitly in the local
work and choose a different method for the active project.
