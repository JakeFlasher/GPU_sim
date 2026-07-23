# Research Draft: Accurate GPU Execution Time Modeling from Architectural Simulators

## Research Direction
Fast, Accurate GPU Architecture Simulation for Modern AI Workloads

## Phase 1 Focus
How to accurately model GPU execution time on a real kernel from architectural simulators?

## Problem Statement
Cycle-level GPU architectural simulators (Accel-Sim, GPGPU-Sim, and the MICRO 2025 modern GPU simulator) are the primary tools for GPU architecture research. However, they exhibit significant execution time prediction errors compared to real hardware, especially for modern AI workloads (GEMM, attention, convolution kernels running on Ampere/Turing GPUs). Understanding and reducing these errors is the foundational step for all subsequent GPU simulation research.

## Research Questions

### RQ1: Sources of Simulation Error
What are the primary sources of execution time prediction error in state-of-the-art GPU simulators when running AI kernels? Categories include:
- Frontend modeling errors (instruction fetch, decode, issue scheduling)
- Memory system modeling errors (cache hierarchy, coalescing, bank conflicts, TLB)
- Compute pipeline errors (tensor core timing, FP32/FP16 throughput, special function units)
- Control flow errors (warp divergence, reconvergence mechanisms)
- Missing microarchitectural features (register file caching, descriptor-based memory access)

### RQ2: MICRO 2025 Simulator Architecture
How does the MICRO 2025 simulator ("Dissecting and Modeling the Architecture of Modern GPU Cores" by Huerta et al.) model modern NVIDIA GPU cores? Key aspects:
- Sub-core architecture modeling (processing blocks, private L0 instruction cache)
- Issue scheduler policy and hardware-compiler co-design
- Register file structure and register file cache
- Memory pipeline modeling improvements over vanilla Accel-Sim
- What is the achieved MAPE (13.98%) and how does it vary by kernel type?

### RQ3: Validation Methodology
How to validate simulator output against real hardware profiling (Nsight Compute)?
- Which Nsight Compute metrics are most informative for simulator validation? (IPC, warp stall reasons, memory throughput, SM occupancy)
- How to set up a systematic validation pipeline: trace collection (NVBit) -> simulation -> correlation with hardware counters
- What benchmarks and AI kernels should be used for validation?
- Statistical methodology for error reporting (MAPE, correlation coefficient, per-kernel breakdown)

### RQ4: Missing Microarchitectural Features
What microarchitectural features are missing or inaccurately modeled that cause execution time prediction errors specifically for AI kernels?
- Tensor Core timing models (mma.sync latency, wgmma pipeline, FP8/BF16 throughput)
- Async copy engine (cp.async, TMA) modeling
- L2 cache partitioning and persistence
- DRAM bandwidth modeling under contention
- Warp scheduling policy accuracy (GTO vs round-robin vs hardware-specific)

## Available Resources

### Source Code
- `modern-gpu-simulator-micro-2025/`: SOTA GPU simulator from MICRO 2025 paper, built on Accel-Sim/GPGPU-Sim 4.x
- Accel-Sim framework (GitHub: accel-sim/accel-sim-framework)
- NVBit for trace generation (GitHub: NVlabs/NVBit)

### Documentation
- `NVIDIA_Manuals/ptx_isa/`: Complete PTX ISA reference for instruction-level simulation
- `NVIDIA_Manuals/ncu_guide/`: Nsight Compute profiling guide for validation
- `NVIDIA_Manuals/ampere_tuning/`: Ampere architecture parameters
- `NVIDIA_Manuals/blackwell_tuning/`: Blackwell architecture parameters
- `NVIDIA_Manuals/.claude/knowledge/ptx/`: Pre-distilled PTX knowledge files (tensor cores, async copy, memory, barriers)
- `NVIDIA_Manuals/cutlass_cute/`: CUTLASS/CuTe for understanding AI kernel patterns

### Literature
- 65 curated papers in `papers/FINAL_curated_paper_list.md` covering simulators, accuracy improvements, microbenchmarking, performance models, and AI workload simulation
- Key accuracy papers: Huerta et al. (MICRO 2025, CAMS 2024, arXiv 2025), Khairy et al. (ISCA 2020, 2018)

## Hardware Available
- NVIDIA GPU(s) for running real workloads and collecting Nsight Compute profiles
- CPU machine for running simulations

## Expected Outcomes
1. A systematic understanding of WHERE and WHY GPU simulators produce execution time errors for AI kernels
2. A documented validation pipeline: trace -> simulate -> compare -> analyze
3. Quantified error breakdown by microarchitectural component (frontend, memory, compute, control flow)
4. Prioritized list of simulator improvements that would most reduce AI kernel simulation error
5. Baseline accuracy measurements for the MICRO 2025 simulator on a representative set of AI kernels

## Constraints
- This is Phase 1 of a multi-phase research project; do not attempt to solve everything
- Focus on understanding the problem before proposing solutions
- Use existing simulator code (modern-gpu-simulator-micro-2025) as the starting point, not building from scratch
- Validate on Ampere (sm_80/sm_86) and Turing (sm_75) architectures first
