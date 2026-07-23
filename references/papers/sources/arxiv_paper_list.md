# GPU Architectural Simulator Papers — Comprehensive ArXiv List

Compiled: 2026-04-08

---

## 1. Core GPU Architectural Simulators

### 1.1 GPGPU-Sim / Accel-Sim Family

| # | Title | Authors | Year | ArXiv ID | Venue |
|---|-------|---------|------|----------|-------|
| 1 | Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling | M. Khairy, Z. Shen, T.M. Aamodt, T.G. Rogers | 2020 | N/A (conf) | ISCA 2020 |
| 2 | Analyzing and Improving Hardware Modeling of Accel-Sim | R. Huerta, A. Gonzalez | 2024 | [2401.10082](https://arxiv.org/abs/2401.10082) | CAMS 2023 |
| 3 | Parallelizing a Modern GPU Simulator | R. Huerta, A. Gonzalez | 2025 | [2502.14691](https://arxiv.org/abs/2502.14691) | |
| 4 | Integrating Per-Stream Stat Tracking into Accel-Sim | — | 2023 | [2304.11136](https://arxiv.org/abs/2304.11136) | |
| 5 | Analyzing Machine Learning Workloads Using a Detailed GPU Simulator | A. Ariel, G. Tyson | 2018 | [1811.08933](https://arxiv.org/abs/1811.08933) | |
| 6 | Exploring Modern GPU Memory System Design Challenges through Accurate Modeling | M. Khairy, A. Jain, T. Aamodt, T.G. Rogers | 2018 | [1810.07269](https://arxiv.org/abs/1810.07269) | |
| 7 | Control Flow Management in Modern GPUs | R. Huerta, A. Gonzalez | 2024 | [2407.02944](https://arxiv.org/abs/2407.02944) | |
| 8 | Analyzing Modern NVIDIA GPU Cores | R. Huerta et al. | 2025 | [2503.20481](https://arxiv.org/abs/2503.20481) | |
| 9 | Modeling Deep Learning Accelerator Enabled GPUs (Tensor Core model in GPGPU-Sim) | M.A. Raihan, N. Goli, T.M. Aamodt | 2018 | [1811.08309](https://arxiv.org/abs/1811.08309) | |

### 1.2 MGPUSim / MGSim (AMD GCN)

| # | Title | Authors | Year | ArXiv ID | Venue |
|---|-------|---------|------|----------|-------|
| 10 | MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization | Y. Sun et al. | 2019 | N/A (conf) | ISCA 2019 |
| 11 | MGSim + MGMark: A Framework for Multi-GPU System Research | Y. Sun et al. | 2018 | [1811.02884](https://arxiv.org/abs/1811.02884) | |
| 12 | Daisen: A Framework for Visualizing Detailed GPU Execution | Y. Sun et al. | 2021 | [2104.00828](https://arxiv.org/abs/2104.00828) | |
| 13 | HALCONE: A Hardware-Level Timestamp-based Cache Coherence Scheme for Multi-GPU Systems | — | 2020 | [2007.04292](https://arxiv.org/abs/2007.04292) | |
| 14 | Improving Multi-Instance GPU Efficiency via Sub-Entry Sharing TLB Design | — | 2024 | [2404.18361](https://arxiv.org/abs/2404.18361) | |

### 1.3 gem5 GPU Model

| # | Title | Authors | Year | ArXiv ID | Venue |
|---|-------|---------|------|----------|-------|
| 15 | The gem5 Simulator: Version 20.0+ | J. Lowe-Power et al. | 2020 | [2007.03152](https://arxiv.org/abs/2007.03152) | |
| 16 | Adding MFMA Support to gem5 | M. Kurzynski, M.D. Sinclair | 2025 | [2501.18113](https://arxiv.org/abs/2501.18113) | |
| 17 | Optimizing GPU Cache Policies for MI Workloads (gem5-based) | — | 2019 | [1910.00134](https://arxiv.org/abs/1910.00134) | |

### 1.4 Vortex (RISC-V GPU)

| # | Title | Authors | Year | ArXiv ID | Venue |
|---|-------|---------|------|----------|-------|
| 18 | Vortex: Extending the RISC-V ISA for GPGPU and 3D-Graphics Research | B. Tine et al. | 2021 | [2110.10857](https://arxiv.org/abs/2110.10857) | |
| 19 | Vortex: OpenCL Compatible RISC-V GPGPU | B. Tine et al. | 2020 | [2002.12151](https://arxiv.org/abs/2002.12151) | |
| 20 | Hardware vs. Software Implementation of Warp-Level Features in Vortex RISC-V GPU | — | 2025 | [2505.03102](https://arxiv.org/abs/2505.03102) | |
| 21 | Decoupled Control Flow and Data Access in RISC-V GPGPUs | — | 2025 | [2512.00032](https://arxiv.org/abs/2512.00032) | |

### 1.5 Other Open-Source / RTL GPU Architectures

| # | Title | Authors | Year | ArXiv ID | Venue |
|---|-------|---------|------|----------|-------|
| 22 | e-GPU: An Open-Source and Configurable RISC-V GPU for TinyAI Applications | — | 2025 | [2505.08421](https://arxiv.org/abs/2505.08421) | |
| 23 | G-GPU: A Fully-Automated Generator of GPU-like ASIC Accelerators | — | 2021 | [2111.06166](https://arxiv.org/abs/2111.06166) | |
| 24 | A Statically and Dynamically Scalable Soft GPGPU (FPGA) | — | 2024 | [2401.04261](https://arxiv.org/abs/2401.04261) | |

### 1.6 NVIDIA NVArchSim / NaviSim (Papers You Already Have)

| # | Title | Authors | Year | ArXiv ID | Venue |
|---|-------|---------|------|----------|-------|
| 25 | NVArchSim (NVIDIA Architectural Simulator) | O. Villa et al. | 2021 | N/A | HPCA 2021 |
| 26 | NaviSim: A Highly Accurate GPU Simulator for AMD RDNA GPUs | T. Baruah, J. Kim, A. Joshi, D. Kaeli | 2022 | N/A | PACT 2022 |
| 27 | Dissecting and Modeling the Architecture of Modern GPU Cores | — | 2025 | N/A | MICRO 2025 |

---

## 2. GPU Microarchitecture Reverse Engineering & Microbenchmarking

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 28 | Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking | Z. Jia et al. | 2018 | [1804.06826](https://arxiv.org/abs/1804.06826) |
| 29 | Dissecting GPU Memory Hierarchy through Microbenchmarking | X. Mei, X. Chu | 2015 | [1509.02308](https://arxiv.org/abs/1509.02308) |
| 30 | Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis | — | 2022 | [2208.11174](https://arxiv.org/abs/2208.11174) |
| 31 | Benchmarking and Dissecting the Nvidia Hopper GPU Architecture | Q. Wang et al. | 2024 | [2402.13499](https://arxiv.org/abs/2402.13499) |
| 32 | Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis | W. Luo et al. | 2025 | [2501.12084](https://arxiv.org/abs/2501.12084) |
| 33 | Microbenchmarking NVIDIA's Blackwell Architecture | — | 2025 | [2512.02189](https://arxiv.org/abs/2512.02189) |
| 34 | Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks | A. Jarmusch et al. | 2025 | [2507.10789](https://arxiv.org/abs/2507.10789) |
| 35 | Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs | — | 2019 | [1905.08778](https://arxiv.org/abs/1905.08778) |
| 36 | Accurate Models of NVIDIA Tensor Cores | F.A. Khattak, M. Mikaitis | 2025 | [2512.07004](https://arxiv.org/abs/2512.07004) |
| 37 | NVIDIA Tensor Core Programmability, Performance & Precision | — | 2018 | [1803.04014](https://arxiv.org/abs/1803.04014) |
| 38 | Toward a Universal GPU ISA: Cross-Vendor Analysis of Hardware-Invariant Computational Primitives | — | 2026 | [2603.28793](https://arxiv.org/abs/2603.28793) |

---

## 3. GPU Performance Modeling & Prediction (Analytical / ML-based)

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 39 | SynPerf: A Hybrid Analytical-ML Framework for GPU Performance Prediction | — | 2026 | [2601.14910](https://arxiv.org/abs/2601.14910) |
| 40 | NeuSight: Forecasting GPU Performance for Deep Learning Training and Inference | — | 2024 | [2407.13853](https://arxiv.org/abs/2407.13853) |
| 41 | Accelerating Computer Architecture Simulation through Machine Learning | — | 2024 | [2402.18746](https://arxiv.org/abs/2402.18746) |
| 42 | Omniwise: Predicting GPU Kernels Performance with LLMs | — | 2025 | [2506.20886](https://arxiv.org/abs/2506.20886) |
| 43 | Prediction of Performance and Power Consumption of GPGPU Applications (Thesis) | — | 2023 | [2305.01886](https://arxiv.org/abs/2305.01886) |
| 44 | Analytical Performance Estimation during Code Generation on GPUs | — | 2022 | [2204.14242](https://arxiv.org/abs/2204.14242) |
| 45 | LLMCompass: A Hardware Evaluation Framework for Large Language Model Inference | H. Zhang et al. | 2023 | [2312.03134](https://arxiv.org/abs/2312.03134) |
| 46 | LUMINA: LLM-Guided GPU Architecture Exploration via Bottleneck Analysis | — | 2026 | [2603.05904](https://arxiv.org/abs/2603.05904) |
| 47 | TokenSim: Enabling Hardware and Software Exploration for LLM Inference Systems | — | 2025 | [2503.08415](https://arxiv.org/abs/2503.08415) |

---

## 4. GPU Simulation Acceleration & Workload Sampling

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 48 | GCL-Sampler: Discovering Kernel Similarity for Sampled GPU Simulation via Graph Contrastive Learning | — | 2026 | [2603.00551](https://arxiv.org/abs/2603.00551) |
| 49 | NeuroScalar: A Deep Learning Framework for Fast, Accurate, and In-the-Wild Cycle-Level Performance Prediction | — | 2025 | [2509.22410](https://arxiv.org/abs/2509.22410) |

---

## 5. GPU Power & Energy Modeling

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 50 | Methodology for Fine-Grain GPU Power Visibility and Insights (FinGraV) | — | 2024 | [2412.12426](https://arxiv.org/abs/2412.12426) |
| 51 | Power and Energy-efficiency Roofline Model for GPUs | — | 2018 | [1809.09206](https://arxiv.org/abs/1809.09206) |
| 52 | Dynamic GPU Energy Optimization for ML Training Workloads (GPOEO) | — | 2022 | [2201.01684](https://arxiv.org/abs/2201.01684) |
| 53 | Automating Energy-Efficient GPU Kernel Generation (references AccelWattch) | — | 2024 | [2411.18873](https://arxiv.org/abs/2411.18873) |
| 54 | Automated PMC-based Power Modeling Methodology for Modern Mobile GPUs | — | 2024 | [2408.04886](https://arxiv.org/abs/2408.04886) |

---

## 6. GPU Architecture Proposals Evaluated via Simulation

### 6.1 Memory System & Cache

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 55 | Irregular Accesses Reorder Unit: Improving GPGPU Memory Coalescing | — | 2020 | [2007.07131](https://arxiv.org/abs/2007.07131) |
| 56 | GPUVM: GPU-driven Unified Virtual Memory | — | 2024 | [2411.05309](https://arxiv.org/abs/2411.05309) |
| 57 | Enabling Large-Reach TLBs for High-Throughput GPUs (MESC) | — | 2021 | [2110.08613](https://arxiv.org/abs/2110.08613) |
| 58 | A GPU Register File using Static Data Compression | — | 2020 | [2006.05693](https://arxiv.org/abs/2006.05693) |
| 59 | RegDem: Increasing GPU Performance via Shared Memory Register Spilling | — | 2019 | [1907.02894](https://arxiv.org/abs/1907.02894) |

### 6.2 Warp Scheduling & Core Architecture

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 60 | RLWS: A Reinforcement Learning based GPU Warp Scheduler | — | 2017 | [1712.04303](https://arxiv.org/abs/1712.04303) |
| 61 | WaSP: Warp Scheduling to Mimic Prefetching in Graphics Workloads (uses TEAPOT simulator) | — | 2024 | [2404.06156](https://arxiv.org/abs/2404.06156) |
| 62 | KHEPRI: Design of a GPU with Heterogeneous Cores for Graphics (uses TEAPOT simulator) | — | 2026 | [2601.22862](https://arxiv.org/abs/2601.22862) |
| 63 | Virgo: Cluster-level Matrix Unit Integration in GPUs for Scalability and Energy Efficiency | H. Kim et al. | 2024 | [2408.12073](https://arxiv.org/abs/2408.12073) |
| 64 | Resource Sharing on GPUs (register & scratchpad sharing) | — | 2015 | [1503.05694](https://arxiv.org/abs/1503.05694) |

### 6.3 Multi-Chip / Chiplet GPU Architectures

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 65 | COPA-GPU: GPU Domain Specialization via Composable On-Package Architecture | — | 2021 | [2104.02188](https://arxiv.org/abs/2104.02188) |
| 66 | MuchiSim: A Simulation Framework for Design Exploration of Multi-Chip Manycore Systems | — | 2023 | [2312.10244](https://arxiv.org/abs/2312.10244) |
| 67 | CHIPSIM: A Co-Simulation Framework for Deep Learning on Chiplet-Based Systems | — | 2025 | [2510.25958](https://arxiv.org/abs/2510.25958) |

### 6.4 Homomorphic Encryption / Domain-Specific GPU Extensions

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 68 | GME: GPU-based Microarchitectural Extensions to Accelerate Homomorphic Encryption (uses NaviSim) | — | 2023 | [2309.11001](https://arxiv.org/abs/2309.11001) |

---

## 7. GPU Cluster / Training Simulators

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 69 | Phantora: Live GPU Cluster Simulation for ML System Performance Estimation | — | 2025 | [2505.01616](https://arxiv.org/abs/2505.01616) |
| 70 | Simulating LLM Training Workloads for Heterogeneous Configurations (SimAI) | — | 2025 | [2508.05370](https://arxiv.org/abs/2508.05370) |
| 71 | AnalySIM: A Simulation Platform for Multi-tenant ML Services on Thousands of GPUs | — | 2022 | [2201.03175](https://arxiv.org/abs/2201.03175) |
| 72 | Towards Universal Performance Modeling for ML Training on Multi-GPU Platforms | — | 2024 | [2404.12674](https://arxiv.org/abs/2404.12674) |
| 73 | Comprehensive Performance Modeling and System Design Insights for Foundation Models | — | 2024 | [2410.00273](https://arxiv.org/abs/2410.00273) |
| 74 | LIFE: Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling | — | 2025 | [2508.00904](https://arxiv.org/abs/2508.00904) |
| 75 | Quantifying the Energy Consumption and Carbon Emissions of LLM Inference via Simulations | — | 2025 | [2507.11417](https://arxiv.org/abs/2507.11417) |
| 76 | MIST: End-to-End Modeling and Optimization of Multi-Stage LLM Serving | — | 2025 | [2504.09775](https://arxiv.org/abs/2504.09775) |

---

## 8. GPU Instrumentation & Profiling Tools

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 77 | GPA: A GPU Performance Advisor Based on Instruction Sampling | K. Zhou et al. | 2020 | [2009.04061](https://arxiv.org/abs/2009.04061) |
| 78 | Understanding GPU Resource Interference One Level Deeper | P. Elvinger et al. | 2025 | [2501.16909](https://arxiv.org/abs/2501.16909) |
| 79 | cuThermo: Understanding GPU Memory Inefficiencies with Heat Map Profiling | — | 2025 | [2507.18729](https://arxiv.org/abs/2507.18729) |
| 80 | KPerfIR: Towards an Open and Compiler-centric Ecosystem for GPU Kernel Performance Tooling | — | 2025 | [2505.21661](https://arxiv.org/abs/2505.21661) |
| 81 | HetGPU: The Pursuit of Making Binary Compatibility Towards GPUs | — | 2025 | [2506.15993](https://arxiv.org/abs/2506.15993) |
| 82 | CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark | — | 2025 | [2505.16968](https://arxiv.org/abs/2505.16968) |
| 83 | Inside VOLT: Designing an Open-Source GPU Compiler | — | 2025 | [2511.13751](https://arxiv.org/abs/2511.13751) |

---

## 9. Roofline Analysis & Performance Characterization

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 84 | Hierarchical Roofline Performance Analysis for Deep Learning Applications | — | 2020 | [2009.05257](https://arxiv.org/abs/2009.05257) |
| 85 | Time-Based Roofline for Deep Learning Performance Analysis | — | 2020 | [2009.04598](https://arxiv.org/abs/2009.04598) |
| 86 | LLM Inference Unveiled: Survey and Roofline Model Insights | Z. Yuan et al. | 2024 | [2402.16363](https://arxiv.org/abs/2402.16363) |
| 87 | Opal: A Modular Framework for Optimizing Performance using Analytics and LLMs | — | 2025 | [2510.00932](https://arxiv.org/abs/2510.00932) |
| 88 | Modeling Utilization to Identify Shared-memory Atomic Bottlenecks | R. Dong | 2025 | [2503.17893](https://arxiv.org/abs/2503.17893) |

---

## 10. GPU Scheduling & Multi-Tenancy Simulation

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 89 | GFS: A Preemption-aware Scheduling Framework for GPU Clusters | — | 2025 | [2509.11134](https://arxiv.org/abs/2509.11134) |
| 90 | Unleashing the Power of Preemptive Priority-based Scheduling for Real-Time GPU Tasks | — | 2024 | [2401.16529](https://arxiv.org/abs/2401.16529) |
| 91 | GCAPS: GPU Context-Aware Preemptive Priority-based Scheduling | — | 2024 | [2406.05221](https://arxiv.org/abs/2406.05221) |
| 92 | Scheduler-Driven Job Atomization (MIG) | — | 2025 | [2509.19086](https://arxiv.org/abs/2509.19086) |

---

## 11. Design Space Exploration & GPU Evolution

| # | Title | Authors | Year | ArXiv ID |
|---|-------|---------|------|----------|
| 93 | MLDSE: Scaling Design Space Exploration Infrastructure for Multi-Level Hardware | — | 2025 | [2503.21297](https://arxiv.org/abs/2503.21297) |
| 94 | NVIDIA GPU Evolution Study (101 GPUs analyzed) | — | 2026 | [2601.20115](https://arxiv.org/abs/2601.20115) |
| 95 | A Survey on Deep Learning Hardware Accelerators for Heterogeneous HPC Platforms | — | 2023 | [2306.15552](https://arxiv.org/abs/2306.15552) |
| 96 | AI Accelerators for Large Language Model Inference | — | 2025 | [2506.00008](https://arxiv.org/abs/2506.00008) |

---

## Notes

- Papers marked "N/A (conf)" were published at conferences without an ArXiv preprint but are foundational works referenced extensively in the ArXiv literature.
- The three papers already in `papers/` (NVArchSim HPCA 2021, NaviSim PACT 2022, MICRO 2025) are included in Section 1.6 for completeness.
- This list focuses on **GPU architectural simulators and closely related work** (microbenchmarking for simulator validation, performance modeling that replaces simulation, simulation acceleration techniques, and architectural proposals evaluated via GPU simulators).
