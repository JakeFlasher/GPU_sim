#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = ROOT / "papers" / "knowledge_base"
MANIFEST_PATH = LIBRARY_DIR / "manifest.json"
PDF_DIR = LIBRARY_DIR / "active"
ARCHIVE_DIR = LIBRARY_DIR / "archive"
CURATION_DATE = "2026-04-11"
FOCUS_NAME = "gpu_architectural_simulation_and_analytical_modeling"


AREA_DEFINITIONS = {
    "core_simulator": "Directly proposes or extends a GPU architectural simulator used for design-space exploration.",
    "simulator_modeling": "Improves simulator fidelity or exposes undocumented behavior needed to calibrate modern GPU models.",
    "microbenchmarking": "Reverse-engineers modern GPU behavior to parameterize or validate simulators and analytical models.",
    "simulation_acceleration": "Targets faster yet trustworthy GPU simulation or cycle-level surrogate modeling.",
    "analytical_model": "Proposes an architecture-level analytical or hybrid performance model for GPU performance evaluation.",
    "simulation_tooling": "Provides trace capture or validation support that is directly useful for GPU simulation workflows.",
}


SELECTED_PAPERS = {
    "2020__accel_sim_an_extensible_simulation_framework_for_validated_gpu_modeling": "core_simulator",
    "2012__multi2sim_a_simulation_framework_for_cpu_gpu_computing": "core_simulator",
    "2017__multi2sim_kepler_a_detailed_architectural_gpu_simulator": "core_simulator",
    "2025__adding_mfma_support_to_gem5": "core_simulator",
    "2009__analyzing_cuda_workloads_using_a_detailed_gpu_simulator_gpgpu_sim": "core_simulator",
    "2014__gem5_gpu_a_heterogeneous_cpu_gpu_simulator": "core_simulator",
    "2023__integrating_per_stream_stat_tracking_into_accel_sim": "core_simulator",
    "2024__maccel_sim_a_multi_gpu_simulator_for_architectural_exploration": "core_simulator",
    "2018__mgsim_mgmark_a_framework_for_multi_gpu_system_research": "core_simulator",
    "2019__mgpusim_enabling_multi_gpu_performance_modeling_and_optimization": "core_simulator",
    "2022__navisim_a_highly_accurate_gpu_simulator_for_amd_rdna_gpus": "core_simulator",
    "2021__need_for_speed_experiences_building_a_trustworthy_system_level_gpu_simulator_nvarchsim": "core_simulator",
    "2022__vulkan_sim_a_gpu_architecture_simulator_for_ray_tracing": "core_simulator",
    "2025__accurate_models_of_nvidia_tensor_cores": "simulator_modeling",
    "2024__analyzing_and_improving_hardware_modeling_of_accel_sim": "simulator_modeling",
    "2018__analyzing_machine_learning_workloads_using_a_detailed_gpu_simulator": "simulator_modeling",
    "2025__analyzing_modern_nvidia_gpu_cores": "simulator_modeling",
    "2024__control_flow_management_in_modern_gpus": "simulator_modeling",
    "2025__dissecting_and_modeling_the_architecture_of_modern_gpu_cores": "simulator_modeling",
    "2018__exploring_modern_gpu_memory_system_design_challenges_through_accurate_modeling": "simulator_modeling",
    "2018__modeling_deep_learning_accelerator_enabled_gpus_tensor_core_in_gpgpu_sim": "simulator_modeling",
    "2024__benchmarking_and_dissecting_the_nvidia_hopper_gpu_architecture": "microbenchmarking",
    "2018__performance_characterisation_and_simulation_of_intel_s_integrated_gpu_architecture": "core_simulator",
    "2022__demystifying_the_nvidia_ampere_architecture_through_microbenchmarking": "microbenchmarking",
    "2015__dissecting_gpu_memory_hierarchy_through_microbenchmarking": "microbenchmarking",
    "2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks": "microbenchmarking",
    "2025__dissecting_the_nvidia_hopper_architecture_through_microbenchmarking_and_multiple_level_analysis": "microbenchmarking",
    "2018__dissecting_the_nvidia_volta_gpu_architecture_via_microbenchmarking": "microbenchmarking",
    "2019__low_overhead_instruction_latency_characterization_for_nvidia_gpgpus": "microbenchmarking",
    "2025__microbenchmarking_nvidia_s_blackwell_architecture": "microbenchmarking",
    "2018__nvidia_tensor_core_programmability_performance_and_precision": "microbenchmarking",
    "2026__toward_a_universal_gpu_isa_cross_vendor_analysis_of_hardware_invariant_computational_primitives": "microbenchmarking",
    "2024__accelerating_computer_architecture_simulation_through_machine_learning": "simulation_acceleration",
    "2013__accelerating_gpgpu_architecture_simulation": "simulation_acceleration",
    "2017__gatsim_abstract_timing_simulation_of_gpus": "simulation_acceleration",
    "2015__cross_architecture_performance_prediction_xapp": "simulation_acceleration",
    "2026__gcl_sampler_discovering_kernel_similarity_for_sampled_gpu_simulation_via_graph_contrastive_learn": "simulation_acceleration",
    "2024__gpu_scale_model_simulation": "simulation_acceleration",
    "2025__neuroscalar_a_deep_learning_framework_for_fast_accurate_cycle_level_performance_prediction": "simulation_acceleration",
    "2013__parallel_gpu_architecture_simulation_framework_exploiting_work_allocation_unit_parallelism": "simulation_acceleration",
    "2025__parallelizing_a_modern_gpu_simulator": "simulation_acceleration",
    "2023__photon_a_fine_grained_sampled_simulation_methodology_for_gpu_workloads": "simulation_acceleration",
    "2021__principal_kernel_analysis_a_tractable_methodology_to_simulate_scaled_gpu_workloads": "simulation_acceleration",
    "2023__sieve_stratified_gpu_compute_workload_sampling": "simulation_acceleration",
    "2025__stem_swift_and_trustworthy_large_scale_gpu_simulation_with_fine_grained_error_modeling_and_hiera": "simulation_acceleration",
    "2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory": "analytical_model",
    "2010__an_adaptive_performance_modeling_tool_for_gpu_architectures": "analytical_model",
    "2021__a_simple_model_for_portable_and_fast_prediction_of_execution_time_and_power_consumption_of_gpu_k": "analytical_model",
    "2019__a_hybrid_framework_for_fast_and_accurate_gpu_performance_estimation_through_source_level_analysi": "analytical_model",
    "2009__a_performance_prediction_model_for_the_cuda_gpgpu_platform": "analytical_model",
    "2011__a_quantitative_performance_analysis_model_for_gpu_architectures": "analytical_model",
    "2009__an_analytical_model_for_a_gpu_architecture_with_memory_level_and_thread_level_parallelism_awaren": "analytical_model",
    "2022__analytical_performance_estimation_during_code_generation_on_gpus": "analytical_model",
    "2019__delta_gpu_performance_model_for_deep_learning_applications_with_in_depth_memory_system_traffic_a": "analytical_model",
    "2014__gputejas_a_parallel_simulator_for_gpu_architectures": "simulation_acceleration",
    "2022__gcom_a_detailed_gpu_core_model_for_accurate_analytical_modeling_of_modern_gpus": "analytical_model",
    "2014__gpumech_gpu_performance_modeling_technique_based_on_interval_analysis": "analytical_model",
    "2020__mdm_the_gpu_memory_divergence_model": "analytical_model",
    "2023__prediction_of_performance_and_power_consumption_of_gpgpu_applications_thesis": "analytical_model",
    "2026__synperf_a_hybrid_analytical_ml_framework_for_gpu_performance_prediction": "analytical_model",
    "2019__nvbit_a_dynamic_binary_instrumentation_framework_for_nvidia_gpus": "simulation_tooling",
    "2025__clustersim_modeling_thread_block_clusters_in_hopper_gpus": "simulator_modeling",
}


SECTION_ARCHIVE_BUCKETS = {
    "Category F: LLM/AI Workload-Focused Architecture Exploration & Simulation": (
        "llm-system-simulation",
        "Focused on LLM system or serving studies rather than GPU architectural simulation or architecture-level analytical modeling.",
    ),
    "Category G: Distributed Training / Multi-GPU System Simulation": (
        "distributed-training-simulation",
        "Targets distributed training or cluster-scale system modeling instead of single-GPU architectural simulation or architecture-level analytical models.",
    ),
    "Category H: AI Workload Benchmark Generation": (
        "benchmark-generation",
        "Generates or studies AI benchmark suites rather than GPU simulator or analytical-model internals.",
    ),
    "Category I: GPU Power & Energy Modeling": (
        "power-and-energy",
        "Centers on power or energy estimation rather than the requested performance-evaluation focus.",
    ),
    "5. GPU Power & Energy Modeling": (
        "power-and-energy",
        "Centers on power or energy estimation rather than the requested performance-evaluation focus.",
    ),
    "2. GPU Power Modeling": (
        "power-and-energy",
        "Centers on power or energy estimation rather than the requested performance-evaluation focus.",
    ),
    "6. GPU Architecture Proposals Evaluated via Simulation": (
        "architecture-proposals",
        "Uses simulation to evaluate a GPU architecture proposal, but the paper itself is not about simulator design or analytical modeling methodology.",
    ),
    "6. GPU Memory System & Address Translation": (
        "architecture-proposals",
        "Studies a GPU memory architecture mechanism rather than simulator design or analytical modeling methodology.",
    ),
    "5. GPU Warp Scheduling & Core Microarchitecture": (
        "architecture-proposals",
        "Focuses on a GPU microarchitectural proposal, not on simulator construction or architecture-level performance modeling.",
    ),
    "8. GPU Preemption & Multi-Tasking": (
        "resource-management",
        "Targets runtime sharing, preemption, or multitasking policy instead of architectural simulation or analytical modeling.",
    ),
    "10. GPU Scheduling & Multi-Tenancy Simulation": (
        "resource-management",
        "Targets runtime scheduling or multi-tenancy management instead of architectural simulation or analytical modeling.",
    ),
    "11. GPU Synchronization & Locking": (
        "resource-management",
        "Targets synchronization policy rather than simulator construction or architecture-level performance modeling.",
    ),
    "7. GPU Cluster / Training Simulators": (
        "llm-system-simulation",
        "Targets cluster-scale AI service simulation instead of GPU architectural simulation or architecture-level analytical modeling.",
    ),
    "7. Multi-Chip / Chiplet GPU Architectures": (
        "architecture-proposals",
        "Focuses on chiplet architecture proposals rather than simulator design or analytical modeling methodology.",
    ),
    "9. GPU Ray Tracing Architecture": (
        "graphics-ray-tracing",
        "Ray-tracing-specific architecture work is outside the narrower simulator and analytical-modeling focus.",
    ),
    "8. GPU Instrumentation & Profiling Tools": (
        "tooling-and-compilers",
        "Provides tooling or profiling support that is not central to simulator construction or architecture-level analytical modeling.",
    ),
    "11. Design Space Exploration & GPU Evolution": (
        "surveys-and-background",
        "Serves as survey or background material rather than a direct simulator or analytical-model contribution.",
    ),
    "9. Roofline Analysis & Performance Characterization": (
        "high-level-performance-models",
        "Uses higher-level performance characterization rather than architecture-level GPU analytical modeling.",
    ),
}


TITLE_ARCHIVE_OVERRIDES = {
    "2024__a_statically_and_dynamically_scalable_soft_gpgpu_fpga": (
        "open-gpu-designs",
        "Presents a soft GPU design platform rather than a simulator or architecture-level analytical model.",
    ),
    "2023__a_survey_on_deep_learning_hardware_accelerators_for_heterogeneous_hpc_platforms": (
        "surveys-and-background",
        "Survey material is useful background, but outside the narrow simulator and analytical-modeling database focus.",
    ),
    "2021__daisen_a_framework_for_visualizing_detailed_gpu_execution": (
        "tooling-and-compilers",
        "Visualization support is adjacent, but not a direct simulator or analytical-model contribution.",
    ),
    "2025__decoupled_control_flow_and_data_access_in_risc_v_gpgpus": (
        "open-gpu-designs",
        "Presents a RISC-V GPU architectural proposal rather than a simulator or analytical-modeling method.",
    ),
    "2025__e_gpu_an_open_source_and_configurable_risc_v_gpu_for_tinyai_applications": (
        "open-gpu-designs",
        "Presents an open GPU design rather than a simulator or architecture-level analytical model.",
    ),
    "2019__emerald_graphics_modeling_for_soc_systems": (
        "graphics-ray-tracing",
        "Graphics SoC modeling is outside the narrower GPGPU simulator and analytical-modeling focus.",
    ),
    "2021__g_gpu_a_fully_automated_generator_of_gpu_like_asic_accelerators": (
        "open-gpu-designs",
        "Focuses on generating GPU-like accelerators instead of simulator or analytical-model methodology.",
    ),
    "2020__halcone_a_hardware_level_timestamp_based_cache_coherence_scheme_for_multi_gpu_systems": (
        "architecture-proposals",
        "Uses a simulator to study a memory-system proposal instead of advancing simulator or analytical-model methodology.",
    ),
    "2025__hardware_vs_software_implementation_of_warp_level_features_in_vortex_risc_v_gpu": (
        "open-gpu-designs",
        "Studies a RISC-V GPU implementation tradeoff rather than simulator or analytical-model methodology.",
    ),
    "2024__improving_multi_instance_gpu_efficiency_via_sub_entry_sharing_tlb_design": (
        "architecture-proposals",
        "Uses simulation to evaluate a TLB design rather than advancing simulator or analytical-model methodology.",
    ),
    "2021__habitat_a_runtime_based_computational_performance_predictor_for_dnn_training": (
        "high-level-performance-models",
        "Targets runtime prediction for DNN training workloads rather than architecture-level GPU modeling.",
    ),
    "2024__llm_inference_unveiled_survey_and_roofline_model_insights": (
        "high-level-performance-models",
        "Focuses on LLM roofline analysis rather than GPU architectural simulation or architecture-level analytical modeling.",
    ),
    "2025__forecasting_gpu_performance_for_deep_learning_training_and_inference_neusight": (
        "high-level-performance-models",
        "Targets workload-level DL performance prediction instead of architecture-level GPU modeling.",
    ),
    "2025__omniwise_predicting_gpu_kernels_performance_with_llms": (
        "high-level-performance-models",
        "Uses LLMs for kernel performance prediction rather than architecture-level analytical modeling.",
    ),
    "2019__optimizing_gpu_cache_policies_for_mi_workloads_gem5_based": (
        "architecture-proposals",
        "Uses gem5-based simulation to study a cache-policy optimization rather than simulator or analytical-model methodology.",
    ),
    "2024__opal_a_modular_framework_for_optimizing_performance_using_analytics_and_llms": (
        "high-level-performance-models",
        "Optimization tooling is outside the narrower simulator and analytical-modeling methodology focus.",
    ),
    "2020__the_gem5_simulator_version_20_0": (
        "surveys-and-background",
        "General gem5 infrastructure is useful background, but not a GPU-specific simulator or analytical-model paper.",
    ),
    "2020__vortex_opencl_compatible_risc_v_gpgpu": (
        "open-gpu-designs",
        "Presents an open GPU architecture platform rather than a simulator or analytical-modeling method.",
    ),
    "2021__vortex_extending_the_risc_v_isa_for_gpgpu_and_3d_graphics_research": (
        "open-gpu-designs",
        "Presents an open GPU architecture platform rather than a simulator or analytical-modeling method.",
    ),
    "2023__skybox_open_source_graphic_rendering_on_programmable_risc_v_gpus": (
        "graphics-ray-tracing",
        "Targets graphics rendering infrastructure rather than the requested simulator/modeling focus.",
    ),
}


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_whitespace(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


SUMMARY_OVERRIDES = {
    "2020__accel_sim_an_extensible_simulation_framework_for_validated_gpu_modeling": "Introduces a validated GPU simulation framework with SASS trace-driven and PTX execution-driven modes, updated performance models, and automated correlation against real NVIDIA GPUs.",
    "2012__multi2sim_a_simulation_framework_for_cpu_gpu_computing": "Introduces Multi2Sim's heterogeneous CPU-GPU framework, enabling joint simulation of CPU cores and AMD-style GPU execution for system-level studies.",
    "2017__multi2sim_kepler_a_detailed_architectural_gpu_simulator": "Extends Multi2Sim with a detailed Kepler-era GPU model to support more realistic NVIDIA-oriented architectural studies.",
    "2025__adding_mfma_support_to_gem5": "Extends gem5's AMD GPU model with MFMA matrix-core support so modern ML workloads and matrix instructions can be simulated more faithfully.",
    "2009__analyzing_cuda_workloads_using_a_detailed_gpu_simulator_gpgpu_sim": "Presents GPGPU-Sim, an early detailed CUDA GPU simulator used to study how real workloads stress cores, memory, and scheduling.",
    "2014__gem5_gpu_a_heterogeneous_cpu_gpu_simulator": "Combines gem5 and GPGPU-Sim to model integrated CPU-GPU systems with unified memory behavior, coherence, and system-level interactions.",
    "2023__integrating_per_stream_stat_tracking_into_accel_sim": "Adds per-stream statistics to Accel-Sim so overlapping CUDA streams and multi-stream behavior can be analyzed with finer attribution.",
    "2024__maccel_sim_a_multi_gpu_simulator_for_architectural_exploration": "Extends architectural simulation to multi-GPU settings so inter-GPU communication and scaling effects can be explored with more fidelity.",
    "2018__mgsim_mgmark_a_framework_for_multi_gpu_system_research": "Provides a multi-GPU research framework and benchmark suite for studying system bottlenecks and optimizations across several GPUs.",
    "2019__mgpusim_enabling_multi_gpu_performance_modeling_and_optimization": "Introduces an AMD-focused multi-GPU simulator for performance modeling, software optimization, and architectural exploration.",
    "2022__navisim_a_highly_accurate_gpu_simulator_for_amd_rdna_gpus": "Builds a highly accurate AMD RDNA simulator and validates it with microbenchmarks plus full applications to support modern architecture studies.",
    "2021__need_for_speed_experiences_building_a_trustworthy_system_level_gpu_simulator_nvarchsim": "Shares the methodology, pitfalls, and validation lessons behind building a trustworthy modern GPU system-level simulator.",
    "2022__vulkan_sim_a_gpu_architecture_simulator_for_ray_tracing": "Introduces a ray-tracing-oriented GPU simulator that models Vulkan workloads and dedicated ray-tracing hardware for architectural studies.",
    "2025__accurate_models_of_nvidia_tensor_cores": "Constructs bit-accurate software models of NVIDIA tensor cores across several GPU generations to support faithful simulation and numerical analysis.",
    "2024__analyzing_and_improving_hardware_modeling_of_accel_sim": "Audits Accel-Sim's hardware assumptions and improves modeling choices that materially affect cycle accuracy on modern GPUs.",
    "2018__analyzing_machine_learning_workloads_using_a_detailed_gpu_simulator": "Extends detailed GPU simulation to cuDNN-backed deep learning workloads so ML kernels can be studied with architectural visibility.",
    "2025__analyzing_modern_nvidia_gpu_cores": "Reverse-engineers modern NVIDIA core pipelines and updates GPU modeling assumptions so simulators better reflect current microarchitectures.",
    "2024__control_flow_management_in_modern_gpus": "Dissects modern GPU control-flow handling and identifies the mechanisms a faithful core model must capture for branch and divergence behavior.",
    "2025__dissecting_and_modeling_the_architecture_of_modern_gpu_cores": "Combines reverse engineering and modeling to replace outdated GPU core assumptions with a more accurate modern core model.",
    "2018__exploring_modern_gpu_memory_system_design_challenges_through_accurate_modeling": "Shows that inaccurate memory-system models can mislead design decisions and argues for tighter simulator validation against modern GPUs.",
    "2018__modeling_deep_learning_accelerator_enabled_gpus_tensor_core_in_gpgpu_sim": "Adds Tensor Core support to GPGPU-Sim so Volta-class deep learning acceleration can be studied with detailed simulation.",
    "2024__benchmarking_and_dissecting_the_nvidia_hopper_gpu_architecture": "Uses microbenchmarks to characterize Hopper core, memory, and accelerator behavior for modeling and optimization.",
    "2018__performance_characterisation_and_simulation_of_intel_s_integrated_gpu_architecture": "Characterizes Intel integrated GPUs and extends MacSim-style modeling so their execution and memory behavior can be simulated with architectural fidelity.",
    "2022__demystifying_the_nvidia_ampere_architecture_through_microbenchmarking": "Uses microbenchmarks to expose Ampere microarchitectural behavior beyond vendor documentation.",
    "2015__dissecting_gpu_memory_hierarchy_through_microbenchmarking": "Reverse-engineers cache, shared-memory, and latency behavior across multiple NVIDIA GPU generations with fine-grained microbenchmarks.",
    "2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks": "Characterizes NVIDIA Blackwell with microbenchmarks to recover modeling-relevant details of the compute and memory subsystems.",
    "2025__dissecting_the_nvidia_hopper_architecture_through_microbenchmarking_and_multiple_level_analysis": "Studies Hopper through microbenchmarks and multi-level analysis to expose behaviors useful for simulator calibration.",
    "2018__dissecting_the_nvidia_volta_gpu_architecture_via_microbenchmarking": "Reverse-engineers Volta execution, memory, and tensor-core behavior to support accurate modeling.",
    "2019__low_overhead_instruction_latency_characterization_for_nvidia_gpgpus": "Presents a lower-overhead method for measuring instruction latencies on NVIDIA GPUs, providing calibration data for detailed models.",
    "2025__microbenchmarking_nvidia_s_blackwell_architecture": "Maps Blackwell architectural changes with microbenchmarks to guide performance analysis and future simulator updates.",
    "2018__nvidia_tensor_core_programmability_performance_and_precision": "Characterizes Volta tensor cores from programmability, throughput, and precision perspectives relevant to model builders.",
    "2026__toward_a_universal_gpu_isa_cross_vendor_analysis_of_hardware_invariant_computational_primitives": "Compares GPU ISAs across vendors to identify stable computational primitives that can support more portable analytical and simulation models.",
    "2024__accelerating_computer_architecture_simulation_through_machine_learning": "Explores machine-learning techniques for reducing simulator cost while preserving useful architectural accuracy.",
    "2013__accelerating_gpgpu_architecture_simulation": "Speeds up GPGPU architectural simulation by selecting representative kernels or phases instead of simulating every dynamic region in full.",
    "2017__gatsim_abstract_timing_simulation_of_gpus": "Introduces abstract timing simulation for GPUs, combining native functional execution with timing abstraction to accelerate early architectural evaluation.",
    "2015__cross_architecture_performance_prediction_xapp": "Predicts GPU performance across architectures so code and design choices can be screened before full porting or detailed simulation.",
    "2026__gcl_sampler_discovering_kernel_similarity_for_sampled_gpu_simulation_via_graph_contrastive_learn": "Learns similarity between GPU kernels from trace graphs to drive representative sampling for much faster simulation.",
    "2024__gpu_scale_model_simulation": "Proposes scale-model simulation techniques for projecting behavior of larger GPU designs from smaller simulation targets.",
    "2025__neuroscalar_a_deep_learning_framework_for_fast_accurate_cycle_level_performance_prediction": "Uses a learned surrogate model to approximate cycle-level performance quickly, aiming to make detailed architectural evaluation far faster.",
    "2025__parallelizing_a_modern_gpu_simulator": "Parallelizes Accel-Sim with deterministic OpenMP-based execution to reduce turnaround time without changing simulated results.",
    "2023__photon_a_fine_grained_sampled_simulation_methodology_for_gpu_workloads": "Introduces fine-grained sampled simulation for GPU workloads to cut runtime substantially while keeping sampling error bounded.",
    "2021__principal_kernel_analysis_a_tractable_methodology_to_simulate_scaled_gpu_workloads": "Selects representative kernels from scaled workloads so GPU simulations remain tractable as applications grow.",
    "2023__sieve_stratified_gpu_compute_workload_sampling": "Applies stratified sampling to GPU-compute workloads to preserve behavior diversity while shortening simulation campaigns.",
    "2025__stem_swift_and_trustworthy_large_scale_gpu_simulation_with_fine_grained_error_modeling_and_hiera": "Combines sampling with fine-grained error modeling to accelerate large-scale GPU simulation while keeping trust in the final estimates.",
    "2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory": "Extends reuse-distance theory to GPU caches, modeling warp-level behavior and cache effects to predict miss rates faster than full simulation.",
    "2010__an_adaptive_performance_modeling_tool_for_gpu_architectures": "Builds an adaptive analytical performance model for GPUs to predict execution behavior and expose optimization bottlenecks across kernels.",
    "2021__a_simple_model_for_portable_and_fast_prediction_of_execution_time_and_power_consumption_of_gpu_k": "Builds a lightweight cross-GPU model for kernel runtime and power prediction using hardware-independent features instead of full detailed simulation.",
    "2019__a_hybrid_framework_for_fast_and_accurate_gpu_performance_estimation_through_source_level_analysi": "Combines source-level analysis with trace-based simulation to estimate GPU performance much faster while retaining useful accuracy.",
    "2009__a_performance_prediction_model_for_the_cuda_gpgpu_platform": "Presents an early CUDA GPU performance model that captures scheduling, memory hierarchy, and pipelining effects to estimate runtime.",
    "2011__a_quantitative_performance_analysis_model_for_gpu_architectures": "Builds a microbenchmark-driven performance model for NVIDIA GPUs that identifies bottlenecks and estimates optimization benefit.",
    "2009__an_analytical_model_for_a_gpu_architecture_with_memory_level_and_thread_level_parallelism_awaren": "Introduces an early analytical GPU model based on memory- and computation-warp parallelism to estimate execution time.",
    "2022__analytical_performance_estimation_during_code_generation_on_gpus": "Uses analytical performance estimates during code generation to prune configuration choices without exhaustive autotuning.",
    "2019__delta_gpu_performance_model_for_deep_learning_applications_with_in_depth_memory_system_traffic_a": "Models deep-learning GPU performance with explicit attention to memory-system traffic so scaling tradeoffs are predicted more accurately.",
    "2022__gcom_a_detailed_gpu_core_model_for_accurate_analytical_modeling_of_modern_gpus": "Presents a detailed analytical GPU core model that captures modern core-side stalls and greatly improves accuracy over prior analytical models.",
    "2014__gpumech_gpu_performance_modeling_technique_based_on_interval_analysis": "Uses interval analysis to approximate GPU execution behavior, offering much faster prediction than cycle-accurate simulation.",
    "2020__mdm_the_gpu_memory_divergence_model": "Introduces a memory-divergence analytical model for GPUs to better predict performance of irregular, memory-bound workloads.",
    "2023__prediction_of_performance_and_power_consumption_of_gpgpu_applications_thesis": "PhD thesis surveying and developing methods for predicting GPGPU performance and power, retained as supporting material for analytical-model work.",
    "2026__synperf_a_hybrid_analytical_ml_framework_for_gpu_performance_prediction": "Combines analytical structure with learned components to predict GPU performance more quickly while retaining architectural interpretability.",
    "2013__parallel_gpu_architecture_simulation_framework_exploiting_work_allocation_unit_parallelism": "Parallelizes GPU architectural simulation by exploiting work-allocation-unit parallelism to reduce simulator runtime.",
    "2014__gputejas_a_parallel_simulator_for_gpu_architectures": "Introduces a fast trace-driven parallel GPU simulator that trades limited error for much higher simulation throughput.",
    "2019__nvbit_a_dynamic_binary_instrumentation_framework_for_nvidia_gpus": "Provides dynamic binary instrumentation for NVIDIA GPUs, enabling trace capture and measurement workflows used by modern simulators such as Accel-Sim.",
    "2025__clustersim_modeling_thread_block_clusters_in_hopper_gpus": "Introduces ClusterSim, a cycle-level model for Hopper H100 thread-block clusters that characterizes DSMEM access latency, cluster barrier synchronization costs, and GPC-local scheduling constraints via microbenchmarks and validates against NCU on real hardware.",
}


def extract_text(pdf_path: Path, layout: bool = False) -> str:
    command = ["pdftotext"]
    if layout:
        command.append("-layout")
    command.extend(["-f", "1", "-l", "2", str(pdf_path), "-"])
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def extract_abstract_from_text(text: str) -> str | None:
    patterns = [
        r"(?ims)^\s*abstract\b[\s:.-]*(.*?)(?=^\s*(?:\d+(?:\.\d+)?|[ivxlcdm]+\.?)\s*introduction\b)",
        r"(?ims)^\s*abstract\b[\s:.-]*(.*?)(?=^\s*(?:ccs concepts|keywords|index terms|acm reference format)\b)",
        r"(?ims)^\s*abstract\b[\s:.-]*(.*?)(?=^\s*[A-Z][A-Z ]{4,}\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            abstract = clean_whitespace(match.group(1))
            if abstract:
                return abstract
    return None


def extract_abstract(pdf_path: Path) -> str | None:
    candidates: list[str] = []
    for layout in (False, True):
        try:
            text = extract_text(pdf_path, layout=layout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        abstract = extract_abstract_from_text(text)
        if abstract:
            candidates.append(abstract)

    if not candidates:
        return None

    def score(candidate: str) -> tuple[int, int]:
        lowered = candidate.lower()
        penalty = 0
        if "@" in candidate or "university" in lowered or "www." in lowered:
            penalty += 2
        if len(candidate) < 80:
            penalty += 2
        if lowered.startswith("1.") or lowered.startswith("i."):
            penalty += 2
        keyword_hits = sum(
            1
            for keyword in (
                "propose",
                "present",
                "introduce",
                "develop",
                "extend",
                "model",
                "simulate",
                "predict",
                "estimate",
                "parallelize",
            )
            if keyword in lowered
        )
        return (keyword_hits - penalty, -abs(len(candidate) - 700))

    return max(candidates, key=score)


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if part.strip()]


def is_noise_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if not sentence or len(sentence) < 20:
        return True
    if "@" in sentence or "http" in lowered or "www." in lowered:
        return True
    if "permission to make" in lowered or "acm reference format" in lowered:
        return True
    if "university" in lowered and len(sentence.split()) < 16:
        return True
    if re.fullmatch(r"[0-9ivx.\- ]+", sentence.strip(), flags=re.IGNORECASE):
        return True
    return False


def select_summary_sentences(abstract: str) -> list[str]:
    sentences = [s.lstrip("—- ").strip() for s in sentence_split(abstract)]
    clean_sentences = [s for s in sentences if not is_noise_sentence(s)]
    if not clean_sentences:
        return []

    contribution_keywords = (
        "propose",
        "present",
        "introduce",
        "develop",
        "extend",
        "model",
        "simulate",
        "predict",
        "estimate",
        "parallelize",
        "demonstrate",
        "enable",
    )
    result_keywords = (
        "error",
        "accuracy",
        "speed-up",
        "speedup",
        "faster",
        "validate",
        "achieve",
        "mean absolute error",
        "cycle error",
        "mips",
    )

    anchor_index = 0
    for index, sentence in enumerate(clean_sentences):
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in contribution_keywords):
            anchor_index = index
            break

    selected = [clean_sentences[anchor_index]]
    for sentence in clean_sentences[anchor_index + 1 :]:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in result_keywords):
            if len(" ".join(selected + [sentence])) <= 360:
                selected.append(sentence)
            break

    return selected


def build_summary(record: dict, area: str) -> str:
    override = SUMMARY_OVERRIDES.get(record["canonical_id"])
    if override:
        return override

    pdf_path = Path(record["local_pdf"]) if record.get("local_pdf") else None
    if pdf_path and pdf_path.exists():
        abstract = extract_abstract(pdf_path)
    else:
        abstract = None

    if abstract:
        selected = select_summary_sentences(abstract)
        if selected:
            summary = " ".join(selected)
            return summary[:360].rstrip()

    title = record.get("title", "Paper")
    fallback = {
        "core_simulator": f"{title} contributes directly to GPU architectural simulation methodology and design-space exploration.",
        "simulator_modeling": f"{title} informs simulator fidelity by modeling or exposing microarchitectural GPU behavior that needs to be represented accurately.",
        "microbenchmarking": f"{title} reverse-engineers GPU behavior that can be used to validate or parameterize simulators and analytical models.",
        "simulation_acceleration": f"{title} targets faster GPU performance evaluation without giving up simulator trustworthiness.",
        "analytical_model": f"{title} proposes an architecture-level analytical or hybrid model for GPU performance estimation.",
        "simulation_tooling": f"{title} supports simulator trace capture or validation workflows for GPU performance studies.",
    }
    return fallback[area]


def determine_archive_reason(record: dict) -> tuple[str, str]:
    canonical_id = record["canonical_id"]
    if canonical_id in TITLE_ARCHIVE_OVERRIDES:
        return TITLE_ARCHIVE_OVERRIDES[canonical_id]

    section = ""
    occurrences = record.get("occurrences") or []
    if occurrences:
        section = occurrences[0].get("section", "")

    if section in SECTION_ARCHIVE_BUCKETS:
        return SECTION_ARCHIVE_BUCKETS[section]

    return (
        "miscellaneous",
        "This paper falls outside the narrowed focus on GPU architectural simulators and architecture-level analytical modeling.",
    )


def archive_pdf(record: dict, bucket: str) -> str | None:
    current = record.get("local_pdf")
    if not current:
        return None

    source = Path(current)
    if not source.exists():
        return str(source)

    if ARCHIVE_DIR in source.parents:
        return str(source.resolve())

    destination_dir = ARCHIVE_DIR / bucket / record["canonical_id"]
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name

    if source.resolve() != destination.resolve():
        if destination.exists():
            source.unlink()
        else:
            shutil.move(str(source), str(destination))

    return str(destination.resolve())


def curate_manifest(records: list[dict]) -> tuple[int, int]:
    selected_count = 0
    archived_count = 0

    for record in records:
        canonical_id = record["canonical_id"]
        curation = {
            "focus": FOCUS_NAME,
            "curated_on": CURATION_DATE,
        }

        if canonical_id in SELECTED_PAPERS:
            area = SELECTED_PAPERS[canonical_id]
            curation.update(
                {
                    "status": "selected",
                    "area": area,
                    "selection_reason": AREA_DEFINITIONS[area],
                    "summary": build_summary(record, area),
                }
            )
            selected_count += 1
        else:
            bucket, reason = determine_archive_reason(record)
            archived_path = archive_pdf(record, bucket)
            if archived_path:
                record["local_pdf"] = archived_path
            curation.update(
                {
                    "status": "archived",
                    "archive_bucket": bucket,
                    "archive_reason": reason,
                }
            )
            archived_count += 1

        record["focus_curation"] = curation

    return selected_count, archived_count


def write_manifest(records: list[dict]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_report(records: list[dict], selected_count: int, archived_count: int) -> None:
    bucket_counts: dict[str, int] = {}
    area_counts: dict[str, int] = {}
    for record in records:
        curation = record["focus_curation"]
        if curation["status"] == "selected":
            area = curation["area"]
            area_counts[area] = area_counts.get(area, 0) + 1
        else:
            bucket = curation["archive_bucket"]
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    lines = [
        "# Knowledge Base Curation Report",
        "",
        f"- Focus: `{FOCUS_NAME}`",
        f"- Curated on: `{CURATION_DATE}`",
        f"- Selected papers: {selected_count}",
        f"- Archived papers: {archived_count}",
        "",
        "## Selected by Area",
        "",
    ]
    for area, count in sorted(area_counts.items()):
        lines.append(f"- `{area}`: {count}")
    lines.extend(["", "## Archived by Bucket", ""])
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"- `{bucket}`: {count}")
    lines.append("")

    report_path = LIBRARY_DIR / "focus_curation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


AREA_LABELS = {
    "core_simulator": "Core Simulators",
    "simulator_modeling": "Simulator Modeling",
    "microbenchmarking": "Microbenchmarking",
    "simulation_acceleration": "Simulation Acceleration",
    "analytical_model": "Analytical Models",
    "simulation_tooling": "Simulation Tooling",
}


AREA_ORDER = [
    "core_simulator",
    "simulator_modeling",
    "microbenchmarking",
    "simulation_acceleration",
    "analytical_model",
    "simulation_tooling",
]


def write_selected_index(records: list[dict]) -> None:
    selected = [
        record
        for record in records
        if record.get("focus_curation", {}).get("status") == "selected"
    ]
    selected.sort(
        key=lambda record: (
            AREA_ORDER.index(record["focus_curation"]["area"]),
            int(record.get("year") or 0),
            record["title"].lower(),
        )
    )

    lines = [
        "# Selected Index",
        "",
        "This index is the fastest entry point into the active paper knowledge base. Start here before reading full PDFs.",
        "",
    ]

    for area in AREA_ORDER:
        area_records = [
            record for record in selected if record["focus_curation"]["area"] == area
        ]
        if not area_records:
            continue
        lines.extend([f"## {AREA_LABELS[area]}", ""])
        for record in area_records:
            pdf_name = Path(record["local_pdf"]).name
            venue = record.get("venue") or "—"
            lines.append(
                f"- [{record['title']}](active/{pdf_name}) | `{record['year']}` | `{venue}`"
            )
            lines.append(f"  Summary: {record['focus_curation']['summary']}")
        lines.append("")

    (LIBRARY_DIR / "selected_index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = load_manifest()
    selected_count, archived_count = curate_manifest(records)
    write_manifest(records)
    write_report(records, selected_count, archived_count)
    write_selected_index(records)
    print(
        f"[curation] selected={selected_count} archived={archived_count} manifest={MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
