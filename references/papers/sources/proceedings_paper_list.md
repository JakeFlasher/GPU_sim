# GPU Architectural Simulator Papers — Conference Proceedings (ISCA/MICRO/HPCA/ASPLOS)

Compiled: 2026-04-08
Deduplicated against: arxiv_paper_list.md

---

## 1. Core GPU Simulators & Frameworks

| # | Title | Authors | Venue | Year | Notes |
|---|-------|---------|-------|------|-------|
| P1 | Analyzing CUDA Workloads Using a Detailed GPU Simulator (GPGPU-Sim) | A. Bakhoda, G. Yuan, W. Fung, H. Wong, T.M. Aamodt | ISPASS 2009 | 2009 | Foundational GPGPU-Sim paper |
| P2 | Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling | M. Khairy, Z. Shen, T.M. Aamodt, T.G. Rogers | ISCA 2020 | 2020 | *Already in arxiv list (#1)* |
| P3 | MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization | Y. Sun et al. | ISCA 2019 | 2019 | *Already in arxiv list (#10)* |
| P4 | Emerald: Graphics Modeling for SoC Systems | A.A. Gubran, T.M. Aamodt | ISCA 2019 | 2019 | Graphics on GPGPU-Sim + gem5 |
| P5 | Skybox: Open-Source Graphic Rendering on Programmable RISC-V GPUs | B. Tine et al. | ASPLOS 2023 | 2023 | Full-stack Vulkan GPU on RISC-V |
| P6 | Vulkan-Sim: A GPU Architecture Simulator for Ray Tracing | M. Saed, Y.H. Chou, L. Liu, T. Nowicki, T.M. Aamodt | MICRO 2022 | 2022 | Ray tracing on GPGPU-Sim + Mesa |
| P7 | NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs | O. Villa, M. Stephenson, D. Nellans, S.W. Keckler | MICRO 2019 | 2019 | Enables trace-driven simulation |
| P8 | Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator (NVArchSim) | O. Villa et al. | HPCA 2021 | 2021 | *Already in arxiv list (#25)* |
| P9 | gem5-gpu: A Heterogeneous CPU-GPU Simulator | J. Power, J. Hestness, M.S. Orr, M.D. Hill, D.A. Wood | Computer Architecture Letters 2014 | 2014 | Integrates GPGPU-Sim with gem5 |

---

## 2. GPU Power Modeling

| # | Title | Authors | Venue | Year |
|---|-------|---------|-------|------|
| P10 | GPUWattch: Enabling Energy Optimizations in GPGPUs | J. Leng, T. Hetherington, A. ElTantawy et al. | ISCA 2013 | 2013 |
| P11 | AccelWattch: A Power Modeling Framework for Modern GPUs | V. Kandiah, S. Peverelle, M. Khairy et al. | MICRO 2021 | 2021 |
| P12 | GPGPU Performance and Power Estimation Using Machine Learning | G. Wu et al. | HPCA 2015 | 2015 |

---

## 3. GPU Analytical / Performance Models

| # | Title | Authors | Venue | Year |
|---|-------|---------|-------|------|
| P13 | GCoM: A Detailed GPU Core Model for Accurate Analytical Modeling of Modern GPUs | J. Lee, Y. Ha, S. Lee et al. | ISCA 2022 | 2022 |
| P14 | GPUMech: GPU Performance Modeling Technique Based on Interval Analysis | J. Kim et al. | MICRO 2014 | 2014 |
| P15 | A Quantitative Performance Analysis Model for GPU Architectures | Y. Zhang, D. Owens | HPCA 2011 | 2011 |
| P16 | GPU Scale-Model Simulation | L. Eeckhout et al. | HPCA 2024 | 2024 |
| P17 | LLMCompass: A Hardware Evaluation Framework for LLM Inference | H. Zhang et al. | ISCA 2024 | 2024 | *Already in arxiv list (#45)* |

---

## 4. GPU Simulation Acceleration & Workload Sampling

| # | Title | Authors | Venue | Year |
|---|-------|---------|-------|------|
| P18 | Principal Kernel Analysis: A Tractable Methodology to Simulate Scaled GPU Workloads | M. Khairy et al. | MICRO 2021 | 2021 |
| P19 | STEM: Swift and Trustworthy Large-Scale GPU Simulation with Fine-Grained Error Modeling | — | MICRO 2025 | 2025 |
| P20 | Accelerating GPGPU Architecture Simulation | — | SIGMETRICS 2014 | 2014 | Miniature representative kernels |
| P21 | Cross-Architecture Performance Prediction (XAPP) | — | MICRO 2015 | 2015 | CPU code to predict GPU perf |

---

## 5. GPU Warp Scheduling & Core Microarchitecture

| # | Title | Authors | Venue | Year |
|---|-------|---------|-------|------|
| P22 | Dynamic Warp Formation and Scheduling for Efficient GPU Control Flow | W. Fung, I. Sham, G. Yuan, T.M. Aamodt | MICRO 2007 | 2007 |
| P23 | GhOST: A GPU Out-of-Order Scheduling Technique for Stall Reduction | I. Chaturvedi et al. | ISCA 2024 | 2024 |
| P24 | Orchestrated Scheduling and Prefetching for GPGPUs | A. Jog et al. | ISCA 2013 | 2013 |
| P25 | CAWA: Coordinated Warp Scheduling and Cache Prioritization for Critical Warp Acceleration | S. Lee et al. | ISCA 2015 | 2015 |
| P26 | RegMutex: Inter-Warp GPU Register Time-Sharing | F. Khorasani et al. | ISCA 2018 | 2018 |
| P27 | APRES: Improving Cache Efficiency by Exploiting Load Characteristics on GPUs | S. Oh et al. | ISCA 2016 | 2016 |

---

## 6. GPU Memory System & Address Translation

| # | Title | Authors | Venue | Year |
|---|-------|---------|-------|------|
| P28 | Architectural Support for Address Translation on GPUs | B. Pichai, L. Hsu, A. Bhattacharjee | ASPLOS 2014 | 2014 |
| P29 | Supporting x86-64 Address Translation for 100s of GPU Lanes | J. Power, M.D. Hill, D.A. Wood | HPCA 2014 | 2014 |
| P30 | Neighborhood-Aware Address Translation for Irregular GPU Applications | M. LeBeane et al. | MICRO 2018 | 2018 |
| P31 | MASK: Redesigning the GPU Memory Hierarchy to Support Multi-Application Concurrency | R. Ausavarungnirun et al. | ASPLOS 2018 | 2018 |
| P32 | MGvm: Designing Virtual Memory System of MCM GPUs | P.B. et al. | MICRO 2022 | 2022 |

---

## 7. Multi-Chip / Chiplet GPU Architectures

| # | Title | Authors | Venue | Year |
|---|-------|---------|-------|------|
| P33 | MCM-GPU: Multi-Chip-Module GPUs for Continued Performance Scalability | A. Arunkumar et al. | ISCA 2017 | 2017 |
| P34 | CPElide: Efficient Multi-Chiplet GPU Implicit Synchronization | P. Dalmia et al. | MICRO 2024 | 2024 |
| P35 | Heterogeneous Die-to-Die Interfaces for Chiplet Interconnection | B. Yin et al. | MICRO 2023 | 2023 |

---

## 8. GPU Preemption & Multi-Tasking

| # | Title | Authors | Venue | Year |
|---|-------|---------|-------|------|
| P36 | Chimera: Collaborative Preemption for Multitasking on a Shared GPU | J. Park, M. Park, S. Mahlke | ASPLOS 2015 | 2015 |
| P37 | Enabling Preemptive Multiprogramming on GPUs | I. Tanasic et al. | ISCA 2014 | 2014 |
| P38 | FLEP: Enabling Flexible and Efficient Preemption on GPUs | B. Wu et al. | ASPLOS 2017 | 2017 |
| P39 | Simultaneous Multikernel GPU: Multi-Tasking Throughput Processors via Fine-Grained Sharing | Z. Wang et al. | HPCA 2016 | 2016 |
| P40 | Warped-Slicer: Efficient Intra-SM Slicing through Dynamic Resource Partitioning | Q. Xu et al. | ISCA 2016 | 2016 |
| P41 | Quality of Service Support for Fine-Grained Sharing on GPUs | Z. Wang et al. | ISCA 2017 | 2017 |
| P42 | HSM: A Hybrid Slowdown Model for Multitasking GPUs | Q. Zhao, J. Jahre, L. Eeckhout | ASPLOS 2020 | 2020 |

---

## 9. GPU Ray Tracing Architecture

| # | Title | Authors | Venue | Year |
|---|-------|---------|-------|------|
| P43 | Extending GPU Ray-Tracing Units for Hierarchical Search Acceleration | A. Barnes, F. Shen, T. Rogers | MICRO 2024 | 2024 |
| P44 | Generalizing Ray Tracing Accelerators for Tree Traversals on GPUs | D. Ha et al. | MICRO 2024 | 2024 |
| P45 | Treelet Prefetching for Ray Tracing | Y.H. Chou, T. Nowicki, T.M. Aamodt | MICRO 2023 | 2023 |
| P46 | Treelet Accelerated Ray Tracing on GPUs | Y.H. Chou et al. | ASPLOS 2025 | 2025 |
| P47 | CoopRT: Accelerating BVH Traversal for Ray Tracing via Cooperative Threads | — | ISCA 2025 | 2025 |
| P48 | Heliostat: Harnessing Ray Tracing Accelerators for Page Table Walks | Y. Feng et al. | ISCA 2025 | 2025 |

---

## 10. Dissecting & Modeling Modern GPU Cores (Conference Versions)

| # | Title | Authors | Venue | Year |
|---|-------|---------|-------|------|
| P49 | Dissecting and Modeling the Architecture of Modern GPU Cores | R. Huerta et al. | MICRO 2025 | 2025 | *Already in arxiv list (#27)* |

---

## 11. GPU Synchronization & Locking

| # | Title | Authors | Venue | Year |
|---|-------|---------|-------|------|
| P50 | Warp Scheduling for Fine-Grained Synchronization | A. ElTantawy, T.M. Aamodt | HPCA 2018 | 2018 |

---

## Notes

- Entries marked *Already in arxiv list (#N)* are duplicates with the arxiv_paper_list.md and are included here only for cross-reference.
- **New unique papers not in arxiv_paper_list.md**: P1, P4, P5, P6, P7, P9, P10, P11, P12, P13, P14, P15, P16, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45, P46, P47, P48, P50 (45 new unique papers)
