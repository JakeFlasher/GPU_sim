# Selected Index

This index highlights the papers most relevant to the active non-root
measurement and prediction project. It is intentionally smaller than the full
PDF corpus.

## Instrumentation And Trace Capture

- [NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs](active/2019__nvbit_a_dynamic_binary_instrumentation_framework_for_nvidia_gpus.pdf) | `2019` | `MICRO`
  Summary: Dynamic binary instrumentation for NVIDIA GPUs; primary prior art
  for collecting SASS and memory-access evidence without NCU.
- [Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs](active/2019__low_overhead_instruction_latency_characterization_for_nvidia_gpgpus.pdf) | `2019` | `—`
  Summary: Methodology for instruction-latency experiments with lower
  perturbation.
- [A Hybrid Framework for Fast and Accurate GPU Performance Estimation through Source-Level Analysis and Trace-Based Simulation](active/2019__a_hybrid_framework_for_fast_and_accurate_gpu_performance_estimation_through_source_level_analysi.pdf) | `2019` | `HPCA`
  Summary: Combines static/source features and trace evidence for fast
  performance estimation.
- [Integrating Per-Stream Stat Tracking into Accel-Sim](active/2023__integrating_per_stream_stat_tracking_into_accel_sim.pdf) | `2023` | `—`
  Summary: Prior art for stream-level attribution; useful as a trace-accounting
  idea only.

## Wall-Clock Prediction

- [A Performance Prediction Model for the CUDA GPGPU Platform](active/2009__a_performance_prediction_model_for_the_cuda_gpgpu_platform.pdf) | `2009` | `HiPC`
  Summary: Early CUDA runtime-prediction model.
- [An Analytical Model for a GPU Architecture with Memory-Level and Thread-Level Parallelism Awareness](active/2009__an_analytical_model_for_a_gpu_architecture_with_memory_level_and_thread_level_parallelism_awaren.pdf) | `2009` | `ISCA`
  Summary: MLP/TLP-aware analytical model for execution time.
- [A Quantitative Performance Analysis Model for GPU Architectures](active/2011__a_quantitative_performance_analysis_model_for_gpu_architectures.pdf) | `2011` | `HPCA`
  Summary: Bottleneck-oriented model useful for choosing predictor features.
- [GPUMech: GPU Performance Modeling Technique Based on Interval Analysis](active/2014__gpumech_gpu_performance_modeling_technique_based_on_interval_analysis.pdf) | `2014` | `MICRO`
  Summary: Interval-analysis approach for runtime estimation.
- [GCoM: A Detailed GPU Core Model for Accurate Analytical Modeling of Modern GPUs](active/2022__gcom_a_detailed_gpu_core_model_for_accurate_analytical_modeling_of_modern_gpus.pdf) | `2022` | `ISCA`
  Summary: Modern analytical core model; mine it for measurable features and
  stall constraints.
- [SynPerf: A Hybrid Analytical-ML Framework for GPU Performance Prediction](active/2026__synperf_a_hybrid_analytical_ml_framework_for_gpu_performance_prediction.pdf) | `2026` | `—`
  Summary: Hybrid analytical/ML prediction reference.

## Cache And Memory Behavior

- [A Detailed GPU Cache Model Based on Reuse Distance Theory](active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf) | `2014` | `HPCA`
  Summary: Primary prior art for cache-hit/miss prediction from reuse behavior.
- [Dissecting GPU Memory Hierarchy through Microbenchmarking](active/2015__dissecting_gpu_memory_hierarchy_through_microbenchmarking.pdf) | `2015` | `—`
  Summary: Public memory-hierarchy behavior and microbenchmark structure.
- [DeLTA: GPU Performance Model for Deep Learning Applications with In-Depth Memory System Traffic Analysis](active/2019__delta_gpu_performance_model_for_deep_learning_applications_with_in_depth_memory_system_traffic_a.pdf) | `2019` | `ISPASS`
  Summary: Memory-traffic model for deep-learning workloads.
- [MDM: The GPU Memory Divergence Model](active/2020__mdm_the_gpu_memory_divergence_model.pdf) | `2020` | `MICRO`
  Summary: Memory-divergence modeling for irregular, memory-bound behavior.

## Architecture Behavior For Interpretation

- [Demystifying the Nvidia Ampere Architecture through Microbenchmarking](active/2022__demystifying_the_nvidia_ampere_architecture_through_microbenchmarking.pdf) | `2022` | `—`
  Summary: Ampere microbenchmark evidence.
- [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](active/2024__benchmarking_and_dissecting_the_nvidia_hopper_gpu_architecture.pdf) | `2024` | `—`
  Summary: Hopper behavior, including newer execution and memory features.
- [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](active/2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks.pdf) | `2025` | `—`
  Summary: Blackwell public microbenchmark evidence.
- [Microbenchmarking NVIDIA's Blackwell Architecture](active/2025__microbenchmarking_nvidia_s_blackwell_architecture.pdf) | `2025` | `—`
  Summary: Additional Blackwell microbenchmark evidence.
- [Analyzing Modern NVIDIA GPU Cores](active/2025__analyzing_modern_nvidia_gpu_cores.pdf) | `2025` | `—`
  Summary: Reverse-engineered core behavior useful for interpreting trace and
  timing features.

## Calibration And Error Analysis

- [Cross-Architecture Performance Prediction (XAPP)](active/2015__cross_architecture_performance_prediction_xapp.pdf) | `2015` | `MICRO`
  Summary: Cross-architecture prediction framing.
- [A Simple Model for Portable and Fast Prediction of Execution Time and Power Consumption of GPU Kernels](active/2021__a_simple_model_for_portable_and_fast_prediction_of_execution_time_and_power_consumption_of_gpu_k.pdf) | `2021` | `ACM TACO`
  Summary: Lightweight portable prediction using architecture-independent
  features.
- [Analytical Performance Estimation during Code Generation on GPUs](active/2022__analytical_performance_estimation_during_code_generation_on_gpus.pdf) | `2022` | `—`
  Summary: Fast estimation during code-generation search.
- [Prediction of Performance and Power Consumption of GPGPU Applications](active/2023__prediction_of_performance_and_power_consumption_of_gpgpu_applications_thesis.pdf) | `2023` | `—`
  Summary: Survey and methods for performance/power prediction.
- [NeuroScalar: A Deep Learning Framework for Fast, Accurate Cycle-Level Performance Prediction](active/2025__neuroscalar_a_deep_learning_framework_for_fast_accurate_cycle_level_performance_prediction.pdf) | `2025` | `—`
  Summary: Learned-surrogate reference for model design and error handling.

## Simulator Prior Art Only

The following PDFs remain useful background for trace schemas, validation
discipline, sampling, and historical comparisons. They are not active project
direction.

- [GPGPU-Sim](active/2009__analyzing_cuda_workloads_using_a_detailed_gpu_simulator_gpgpu_sim.pdf)
- [Accel-Sim](active/2020__accel_sim_an_extensible_simulation_framework_for_validated_gpu_modeling.pdf)
- [Need for Speed / NVArchSim](active/2021__need_for_speed_experiences_building_a_trustworthy_system_level_gpu_simulator_nvarchsim.pdf)
- [Parallelizing a Modern GPU Simulator](active/2025__parallelizing_a_modern_gpu_simulator.pdf)
- [STEM](active/2025__stem_swift_and_trustworthy_large_scale_gpu_simulation_with_fine_grained_error_modeling_and_hiera.pdf)
