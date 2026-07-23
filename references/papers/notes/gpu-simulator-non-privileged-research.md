# Modern GPU Architectural Simulator Under Non-Privileged Driver Constraints

A comprehensive research report on building an advanced, fast, accurate GPU architectural simulator on a cloud GPU server with root inside a container but non-root on the host driver — ruling out NCU profiling and `nvidia-smi` frequency locking. Workaround relies on CUPTI + NVBit binary instrumentation. Includes Codex independent expert review and revised proposal.

Date: 2026-04-29

---

## Table of Contents

1. [TL;DR](#tldr)
2. [The Constraint Landscape](#1-the-constraint-landscape)
3. [State of the Art in GPU Architectural Simulators (2024-2026)](#2-state-of-the-art-in-gpu-architectural-simulators-2024-2026)
4. [The Three Pillars of Unprivileged Measurement](#3-the-three-pillars-of-unprivileged-measurement)
5. [NCU-Accurate Metrics Without NCU — Reconstruction Recipes](#4-ncu-accurate-metrics-without-ncu--reconstruction-recipes)
6. [Wall-Clock and Frequency Specifically](#5-wall-clock-and-frequency-specifically)
7. [Novel SOTA Approaches and Research Frontiers](#6-novel-sota-approaches-and-research-frontiers)
8. [Original Recommended Architecture](#7-original-recommended-architecture)
9. [Open Research Questions](#8-open-research-questions)
10. [Codex Independent Expert Review](#9-codex-independent-expert-review)
11. [Revised Solution: Counter-Free Unprivileged GPU Simulator Calibration](#10-revised-solution-counter-free-unprivileged-gpu-simulator-calibration)
12. [Sources](#sources)

---

## TL;DR

1. **The constraint isn't a blocker — it's your novelty axis.** No major published simulator has rigorously tackled "trustworthy modern-GPU simulation calibrated *without* NCU/Nsight Compute." Recent work (Hopper/Blackwell microbenchmarks at HKUST/HIT, Hopper dissection 2025) explicitly *abstains* from privileged tools and computes metrics from PTX `%clock64` / wall-time and SASS-level NVBit traces. That methodology is the seed of your contribution.

2. **The SOTA simulator landscape (2024-2026) is split four ways**: (a) cycle-accurate trace-driven (**Accel-Sim** + the **MICRO 2025 "Modern GPU Cores"** redesign), (b) hybrid-fidelity stall-aware (**HyFiSS** MICRO 2024), (c) sampled / fast (**Photon** MICRO 2023, **GCL-Sampler** 2026, **STEM/ROOT** MICRO 2025), and (d) analytical/ML-based (**GCoM**, **AMALI** ISCA 2025, **NeuSight** ASPLOS 2025, **TrioSim** ISCA 2025). All but the analytical ones depend on NVBit. None of them shows a fully self-contained calibration-without-NCU pipeline.

3. **CUPTI 13.0 (2025) replaces the privileged-only legacy Event/Metric API with four new APIs** — Range Profiling, PM Sampling, PC Sampling, and SASS Metric. CUPTI Activity API + Callback API + CUDA Events have always been usable without driver-level privilege; the *new* APIs partially overlap the historical "needs NCU" zone but the privilege wall (`ERR_NVGPUCTRPERM`) remains for hardware-counter-backed metrics. This means your usable surface is broader than two years ago, but the hardest metrics (warp stall reasons, sector-level cache hits, real DRAM read/write bytes) are still gated.

4. **For "near-NCU-accurate wall-clocks," the proven recipe is**: NVBit-injected `clock64`/`%globaltimer` reads at kernel/CTA boundaries → derive per-kernel cycle counts and SM-clock estimate → cross-check with `cudaEventElapsedTime` (drift-free wall-time, no privilege) + read-only NVML clock queries to get DVFS-aware per-kernel frequency → publish *both cycles and time* with statistical confidence intervals over N≥30 runs and outlier filtering by NVML throttle-reason signal.

5. **A defensible novel system** combines: NVBit SASS+address tracing → augmented with on-device clock injection → CUPTI 13 PC-Sampling / SASS-Metric for stall hints → fed into a hybrid-fidelity simulator (cycle-accurate cores + analytical memory + ML-corrected residuals) → calibrated by *comparing simulator-emitted cycles* (not time) against *NVBit-measured cycles* on microbenchmarks, with a residual-correction model trained per architecture. This is the gap nobody has filled.

> **NOTE:** Section 10 below contains the **revised proposal** after Codex's critical review. Several of the original claims above were softened or replaced (notably: the ML residual head, the silent control kernel, and "NCU-accurate" framing). Read sections 1-8 as the research base; section 9 as the critique; section 10 as the working plan.

---

## 1. The Constraint Landscape

### 1.1 What "non-root on the driver" rules out

| Capability | Privilege required | Available to you? |
|---|---|---|
| `nvidia-smi --lock-gpu-clocks` (DVFS lock) | Host root, modifies driver state | ❌ |
| `NVreg_RestrictProfilingToAdminUsers=0` (the gate that NCU/CUPTI Event API checks) | Host root + `modprobe.d/profile.conf` + driver reload | ❌ |
| **NCU / Nsight Compute** in any meaningful sense | Above gate must be off OR run as root | ❌ |
| **CUPTI legacy Event/Metric API** | Same gate | ❌ (dropped in CUPTI 13.0 anyway) |
| **CUPTI Range Profiler / PM Sampling / SASS Metric** | Same gate for *hardware-counter-backed* metrics; many *derived* ones still work | ⚠ partial |
| **CUPTI PC Sampling** | Same gate for stall-reason data | ⚠ partial |
| **CUPTI Activity API** (kernel start/end, memcpy, runtime API records) | None | ✅ |
| **CUPTI Callback API** | None | ✅ |
| **CUDA Events** (`cudaEventElapsedTime`) | None | ✅ |
| **NVML read-only queries** (`nvmlDeviceGetClockInfo`, throttle reasons, power, temperature) | None | ✅ |
| **NVBit binary instrumentation** | Runs in process context; no driver privilege required | ✅ |
| **PTX `clock`, `clock64`, `%clock`, `%globaltimer` registers in kernels** | None | ✅ |
| **`nvdisasm` SASS dump from cubin** | None | ✅ |
| **DCGM hardware metrics** | Profiling fields require `SYS_ADMIN` capability | ⚠ host-dependent |

**Key insight:** Everything that NVBit does runs in your process address space; everything CUPTI Activity does goes through the user-mode CUDA driver. Both pre-date the ERR_NVGPUCTRPERM gate and remain unaffected. The gate is specifically over hardware performance counters that share state with security-sensitive features (timing side channels, fingerprinting).

### 1.2 What you actually want from NCU and what's recoverable elsewhere

NCU's value comes from ~150 hardware counters per kernel. For a simulator, the small subset that actually feeds calibration is: kernel duration, dynamic SASS instruction count, IPC, achieved occupancy, L1/L2 hit rates, DRAM read/write bytes, warp execution efficiency, branch efficiency, stall reasons (`smsp__pcsamp_warps_issue_stalled_*`), tensor pipe utilization. The next sections show which of these you can recover, with what fidelity, from unprivileged tooling alone.

---

## 2. State of the Art in GPU Architectural Simulators (2024-2026)

### 2.1 The four families and how they partition the design space

| Family | Speed | Cycle-accuracy | Modern-arch coverage | Validation pipeline | Representative |
|---|---|---|---|---|---|
| Trace-driven cycle-accurate | Slow (days) | High | Volta→Hopper; Blackwell partial | NVBit traces + NCU calibration | Accel-Sim 1.3, MICRO'25 GPU-cores rework |
| Hybrid-fidelity / sampled cycle | Medium | Medium-High | Volta-class | NVBit traces + sampling | HyFiSS, Photon, STEM/ROOT, GCL-Sampler |
| Analytical | Fast (seconds) | Low-Medium MAPE | Up to Hopper | Microbench + NCU | GCoM, AMALI, MDM, GPUMech |
| ML-based / data-driven | Fast | Variable | Generalizes to unseen | Profiler traces (Habitat: NCU; NeuSight: PyTorch profiler) | Habitat ATC'21, Daydream ATC'20, NeuSight ASPLOS'25, Forecasting GPU Performance |

### 2.2 Per-simulator deep dives

**Accel-Sim** is the de-facto academic baseline. v1.3.0 was released Feb 2025. It claims Volta–Blackwell coverage in marketing copy, but in practice:

- Volta–Ampere are well validated against real cards via the bundled correlator tool.
- Hopper is supported in trace generation via NVBit ≥1.7.4 (sm_90).
- Blackwell (sm_120 / RTX 5090) is broken — there's an open issue (#385, Feb 2025) showing both the cuobjdump and the runtime path failing on CUDA 12.4/12.8.
- Calibration in the original paper used per-NCU-counter correlation (Khairy et al. ISCA'20).

The **MICRO 2025 paper "Dissecting and Modeling the Architecture of Modern GPU Cores"** (Huerta, Abaie Shoushtary, Cruz, González at UPC) is the most important 2025 simulator paper for your purposes. It re-architects the SM front-end (sub-core pipeline, separate fetch/decode, L0 i-cache, instruction prefetcher), the result-bus write-back, and the memory pipeline (per-sub-core, address-calc/coalescing split). The companion paper "Parallelizing a modern GPU simulator" (arXiv:2502.14691) makes Accel-Sim 5.8× faster with OpenMP at zero accuracy cost. The simulator is open-source (upc-arco/modern-gpu-simulator-micro-2025) and includes Turing/Ampere/Blackwell example traces. Crucially it parses **control bits** from SASS — the Hopper/Blackwell scheduler hints encoded by ptxas — which most prior simulators ignore.

**HyFiSS** (MICRO 2024) is the hybrid-fidelity reference. It explicitly tracks stall events (compute structural, compute data, memory structural, memory data, sync, control) and reports MAPE comparable to Accel-Sim while being 12.8× faster and using 3.2× less disk for traces. Validated on Volta QV100 against 1,784 kernels from 43 apps. **Why this matters for you:** it shows a path to predicting NCU-style stall taxonomies without NCU, by simulating the stall events from traces.

**GPGPU-Sim** itself (the underlying engine) is functionally subsumed by Accel-Sim now. The standalone repo's last meaningful activity moved into Accel-Sim's fork. v4.2.1 was released February 12, 2025; there is no public GPGPU-Sim 5.0.

**MGPUSim / Akita** is AMD-focused (GCN3 ISA). Sun's group at William & Mary continues active work; **TrioSim** (ISCA 2025) is their 2025 contribution — a lightweight DNN-multi-GPU simulator with 2.91-6.82% error vs. real hardware, predicting from PyTorch operator-level traces (no NVBit, no NCU). HDPAT and QuCo are HPCA 2026. AkitaRTM (MICRO 2024) is a real-time visualizer for simulations.

**PPT-GPU** (Performance Prediction Toolkit) takes the "high-level model + traces" approach and is interesting because it doesn't depend on NCU for execution but does for calibration; the SC'21 hybrid scalable version is the most recent.

**GCoM** (PPoPP 2022) is a detailed analytical model that captures sub-core structural stalls, L1 sectored-cache effects, and bank conflicts; achieves 10% MAE vs Accel-Sim and 24.4% MAE vs RTX 2060 hardware. **AMALI** (ISCA 2025) extends GCoM's lineage for LLM inference specifically — it adds tensor-core math-pipe throttle modeling, constant/instruction cache modeling, and multi-warp distribution; reduces MAPE from 127% to 23.6% vs GCoM. AMALI is the SOTA analytical model for modern arch as of ISCA 2025.

**NeuSight** (ASPLOS 2025) pioneers "tile-decomposed" ML prediction — each operator decomposed to tile-level subproblems, predicted by an MLP whose features include hardware spec sheets. Reduces GPT-3 forecasting error on H100 from 121.4% (training) and 30.8% (inference) to 2.3%. Critically, it generalizes to *unseen* GPUs and unseen models. This is where NeuSight beats Habitat (ATC'21), which had 2067% max error on out-of-distribution GPUs.

**Daydream** (ATC'20) builds a fine-grained dependency graph from CUPTI traces and replays it under hypothetical optimizations to predict speedup. Pure-CUPTI; needs no NCU at all. DAG-based, so granularity is operator-level not warp-level.

**Photon** (MICRO 2023) introduces fine-grained sampled simulation for GPU workloads — chooses representative warp-level segments via basic-block-vector clustering. **GCL-Sampler** (Feb 2026) replaces hand-crafted features with graph contrastive learning over RGCN-encoded SASS dependency graphs from NVBit traces, achieving 258× speedup at 0.37% error using **HyFiSS** as the simulation backend. **STEM/ROOT** (MICRO 2025) statistically estimates simulation error from kernel execution-time distributions and uses optimization to minimize sample size — very relevant if you intend to publish on simulation-time reduction.

**MAccel-sim** (IISWC 2024 poster from Tim Rogers' group at Purdue) extends Accel-Sim to multi-GPU. Less mature than TrioSim for ML workloads but cycle-accurate.

**Need-for-Speed / NVArchSim** (HPCA 2021, NVIDIA internal) is the gold standard for "trustworthy *system-level*" GPU simulation; the paper itself is a useful template for *how to argue trust*.

**GainSight** (Stanford) is an Accel-Sim fork that dumps verbose SM↔cache-interface traces; useful as a starting reference for cache analysis modules.

**GCStack + GCScaler** (Yonsei, ISCA 2025): Fine-grained stall cycle accounting + interval analysis-based scaling. Captures concurrently occurring stall events; enables cycle-stack characterization. **32.7x faster than repetitive timing simulation** for 1,000-design exploration with **6.37% MAPE**.

**Swift-Sim** (Beihang, DATE 2025): Modular hybrid GPU simulator: cycle-accurate for components of interest, analytical elsewhere. **82.6x and 211.2x speedup** for two simplification levels with acceptable accuracy degradation.

**ClusterSim** (TU Hamburg, IISWC 2025): Specifically targets **Hopper Thread Block Clusters** and **distributed shared memory (DSMEM)** modeling — features missing from current public simulators.

**SimAI** (Alibaba, NSDI 2025): End-to-end LLM training simulator: framework + collective communication + network. **98.1% alignment** with real-world results.

**Multiverse** (Tsinghua, NSDI 2025): GPU-accelerated AI training simulator. **<3% discrepancy on 54,000-GPU clusters**, **43-73x speedup** over CPU-based simulators.

### 2.3 Recent benchmarking-focused papers — most relevant for your methodology

The Hopper-and-Blackwell microbenchmarking literature is the closest precedent to what you're trying to do, because all of it operated *without* relying on NCU for timing. Three papers stand out:

**"Benchmarking and Dissecting the Nvidia Hopper GPU Architecture"** (arXiv:2402.13499, Luo et al., HKUST-GZ + HIT-SZ, 2024). Methodology highlights:

- Pure PTX-level microbenchmarks with `clock64()` for latency.
- Throughput computed as `Total_OPS / Duration` — they **explicitly avoid total-clock-cycles for throughput** because GPU frequencies vary across instruction mixes ("we abstain from utilizing total clock cycles to compute throughput due to potential variations in GPU frequencies").
- SASS disassembly to verify no compiler reordering broke timing.
- Document and report power throttling (H800 hitting 350W cap) rather than hide it.

**"Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis"** (arXiv:2501.12084, Luo et al., Jan 2025). Same group, deeper. Key novel techniques:

- **Fine-grained pointer-chase + K-means clustering** to discover partitioned L2 cache (4 distinct latency groups on H800, missed by traditional p-chase).
- TMA latency including async issuance + Mbarrier arrive/wait.
- DSM (Distributed Shared Memory) latency via `%smid` register tracking.
- Random vs zero matrix initialization to expose frequency degradation under power throttle.
- 1024 iterations per measurement.

**"Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks"** (arXiv:2507.10789, Jarmusch et al., Univ Delaware, 2025). Methodology: PTX `%clock64`, no NCU, no NVBit, only nvidia-smi (read-only) + cuBLASLt + TensorRT for application-level. 1024 reps per config. Validated 5th-gen Tensor Cores including FP4/FP6/FP8.

These three papers together effectively *prove* that NCU is unnecessary for high-quality micro-benchmarking on modern GPUs. Your job is to extend this from microbenchmarking to a *simulator's calibration loop*.

---

## 3. The Three Pillars of Unprivileged Measurement

### 3.1 NVBit (binary instrumentation, runs in user space)

**Latest:** v1.8 (Apr 2024). Adds **TMA support (alpha) for Hopper and Blackwell** — host-side APIs to generate the global/shared addresses each TMA instruction will touch. v1.7.7 (Feb 2024) added tmem address parsing. CUDA 13.2 headers in v1.8.

**What NVBit gives you (unprivileged):**

| Capture | Mechanism | Useful for |
|---|---|---|
| Dynamic SASS instr count (per warp/CTA/kernel) | `nvbit_insert_call` before each instr | IPC reconstruction, occupancy proxy |
| Memory addresses per load/store | `nvbit_add_call_arg_mref_addr` | Trace-driven cache simulation, coalescing analysis |
| Active thread mask | predicate inspection | Branch divergence, warp efficiency |
| Register reads/writes (operand-level) | operand iteration API | RF pressure analysis |
| TMA descriptors and addresses (v1.8) | new TMA APIs | Hopper/Blackwell DMA modeling |
| Custom on-device cycle reads | inject your own kernel that calls `clock64()` | **wall-clock cycles per kernel** |
| SASS opcode/predicate | static analysis | Tensor-core / FP8/FP16 / wgmma counting |

**Limitation:** instrumentation adds 3-100× slowdown depending on density (Accel-Sim's mem_trace tool runs ≈30× slower; lighter-weight cycle-stamp injections can be sub-2×). The NVBit paper (MICRO 2019) characterized 14% overhead for warp-uniform tools and 85% for divergent ones. **You will distort timing** when measuring traced runs — solve by running the *same workload* both instrumented (for traces) and uninstrumented (for wall-clock + cycle counts via `clock64`-only injection at kernel boundaries, which is near-zero overhead).

**Hopper/Blackwell caveat:** `nvbit_get_line_info()` is broken on sm_90+; use `nvbit_dump_cubin()` + nvdisasm for source mapping.

**NVBit does NOT expose control bits.** The MICRO 2025 paper recovers them by parsing the cubin offline with `cuobjdump`/`nvdisasm` and joining on PC.

### 3.2 CUPTI (CUDA Profiling Tools Interface)

CUPTI 13.0 (shipped with CUDA 13.0) is the most important update for your project. The reorganization:

| API | Purpose | Privilege | Status |
|---|---|---|---|
| **Activity API** | Async records: kernels, memcpys, runtime/driver API calls, NVTX, CUDA graph nodes | None | ✅ Live |
| **Callback API** | Synchronous notifications around CUDA API entry/exit | None | ✅ Live |
| **Range Profiling API** (CUPTI 12.6+) | Per-range hardware metrics, replaces legacy Profiling API | Counter access (gated) for HW metrics | ⚠ Partial |
| **PM Sampling API** (CUPTI 12.6+) | Periodic GPU performance-monitor sampling — Turing+ | Counter access (gated) | ⚠ Partial |
| **PC Sampling API** | Warp PC + scheduler-state sampling, stall reasons | Counter access (gated) | ⚠ Partial |
| **SASS Metric API** (CUPTI 12.2+) | Source-level SASS metrics via SASS patching | Counter access (gated for HW-backed metrics; pure SASS counts may not be) | ⚠ Partial |
| **HW Trace** (`cuptiActivityEnableHWTrace`, Blackwell) | Hardware Event System tracing | Documented as not requiring counter privilege; **probe before assuming gated** | ⚠ Probe |
| Legacy Event API + Metric API | Old hardware counter access | Was privileged; **dropped in CUPTI 13** | ❌ Removed |

**Critical insight: the privilege gate is per-metric, not per-API.** For Range Profiling and SASS Metric, queries that compute from *Activity-API-equivalent* data succeed; queries that touch hardware perf counters return `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`. You'll need to enumerate and probe — there's no published "non-privileged metric list."

### 3.3 CUDA Events + on-device clocks (zero-privilege, near-zero overhead)

| Source | Granularity | Frequency-coupled? |
|---|---|---|
| `cudaEventRecord` + `cudaEventElapsedTime` | per-kernel or sub-kernel boundary | wall-time, frequency-independent for *time* |
| PTX `%clock` (32-bit, per-SM cycle counter, increments on SM clock) | per-instruction, per-thread | yes — gives cycles, decouples from frequency |
| PTX `%clock64` (64-bit version of `%clock`) | same | same |
| PTX `%globaltimer` (nanoseconds, single global counter) | per-instruction | wall-time, frequency-independent |
| PTX `%smid` | identifies which SM ran a thread | none |
| `clock()`, `clock64()` device-side functions | wrappers around above | same |

The MICRO 2024 / 2025 Hopper papers all use `clock64` exactly this way: read at start, read at end, subtract. The trick is preventing the compiler from reordering — the established pattern is to make the post-read depend on a result of the timed code. Use `__threadfence()` or insert the clock read in inline PTX to ensure correct ordering.

**Per-SM clock skew:** different SMs do *not* run on identical clocks but they're coupled to the same SM-clock domain; offsets between SMs are small and bounded. The Hopper paper handles this by doing inter-SM cycle differences only after recording `%smid`.

> **Codex caveat:** `clock64()` is a per-thread elapsed cycles counter on its SM, **including time-slicing**. It does NOT give kernel-wide `cycles_active`. See Section 9 / 10.

---

## 4. NCU-Accurate Metrics Without NCU — Reconstruction Recipes

> **Note:** Section 10 below revises the framing of these reconstructions from "NCU-accurate" to "counter-free modeled observables with uncertainty intervals." The recipes below are technically correct but the fidelity claims are softened in section 10.

### 4.1 Wall-clock cycles (the hardest single thing because of frequency)

**Recommended approach (composite):**

1. **NVBit injection** of a `clock64()` read at every kernel entry and exit (CTA-level if you want sub-kernel detail).
2. **Concurrently** record `cudaEventRecord` immediately before launch and immediately after `cudaDeviceSynchronize`.
3. From these two: `cycles = clock64_end - clock64_start`, `time = cudaEventElapsedTime`. Implied frequency `f = cycles / time`.
4. **Cross-check** with `nvmlDeviceGetClock(NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT)` polled in a sidecar thread at 100 Hz during execution.
5. **Discard** runs where `nvmlDeviceGetCurrentClocksThrottleReasons` returns non-`GPU_IDLE`.

This gives you both **cycles** and **time**. Your simulator should *output cycles*. Convert to time only at the boundary, using the per-kernel measured `f`.

### 4.2 Dynamic SASS instruction count and IPC

NVBit `count_instrs` tool pattern, customized to also bin by opcode class. IPC = total instructions / measured cycles from §4.1.

### 4.3 Achieved occupancy

Hybrid static + dynamic. Static occupancy = `cudaOccupancyMaxActiveBlocksPerMultiprocessor()`. Achieved occupancy: inject a sampling kernel via NVBit that periodically reads `%nactive_threads_in_warp` and tracks issue cycles.

### 4.4 L1/L2 cache hit rate

Trace-driven simulation: NVBit captures all addresses; your simulator's L1/L2 model produces hit/miss counts.

**Workaround (novel):** Calibrate your cache model with **microbenchmarks where the answer is analytically known** (P-chase patterns where every access is guaranteed L1, L2, or DRAM). Run them, measure cycles via §4.1, validate your model reproduces those cycles. This converts "hit-rate calibration" into "latency calibration."

### 4.5 DRAM read/write bytes

Trace-driven from NVBit memory traces. Aggregate every global-memory load/store address; apply coalescing logic → 32-byte sectors; subtract anything that hits in L2.

### 4.6 Stall reasons (the holy grail of NCU output, hardest to recover)

NCU's `smsp__pcsamp_warps_issue_stalled_*` family is the most valuable single signal in NCU.

**Path 1 (if accessible):** Try CUPTI 13 PC Sampling first.

**Path 2 (simulate them):** This is what HyFiSS does. Build the stall taxonomy into your timing model.

**Path 3 (statistical inference):** Treat stall reasons as latent variables. Train an ML head on top of NVBit features (instruction mix, address-trace properties, register pressure) → predict NCU stall fractions.

### 4.7 Tensor Core / wgmma utilization

Pure NVBit — count dynamic HMMA, IMMA, BMMA, GMMA, IGMMA, QGMMA, FMMA-FP4/FP8 instructions.

### 4.8 Branch divergence / warp efficiency

NVBit-side branch predicate iteration captures the active mask per instruction.

### 4.9 The summary matrix (original)

| NCU metric class | Recoverable? | Recommended source | Expected error vs NCU |
|---|---|---|---|
| `gpu__time_duration` | ✅ | CUDA Events | <2% |
| `sm__cycles_active.avg` | ✅ | NVBit clock64 injection | <5% |
| `sm__inst_executed.sum` | ✅ | NVBit instruction count | 0% |
| `sm__warps_active.avg.pct_of_peak_sustained_active` (occupancy) | ⚠ | Static + NVBit sampling | 5-10% |
| `sm__pipe_tensor_op_*` (TC ops) | ✅ | NVBit opcode count | 0% |
| `smsp__thread_inst_executed_per_inst_executed.ratio` (warp eff) | ✅ | NVBit predicate capture | 0% |
| `lts__t_sectors_lookup_hit_rate` (L2 hit rate) | ⚠ | Trace-driven sim | 5-15% |
| `l1tex__t_sectors_lookup_hit_rate` (L1 hit rate) | ⚠ | Trace-driven sim | 5-15% |
| `dram__bytes_read.sum` | ⚠ | Trace-driven sim | 5-15% |
| `smsp__pcsamp_warps_issue_stalled_*` (stall reasons) | ⚠/❌ | CUPTI 13 PC Sampling probe; else simulate | 10-30% if simulated |
| Frequency / SM clock | ✅ | clock64 + cudaEvent + NVML | <3% |
| Power | ⚠ | NVML telemetry, sampled | 5-10% |
| Energy/op | ⚠ | NVML × time | 10-20% |

> **Codex caveat:** "0% error" entries are too strong (NCU definition ≠ NVBit definition for predicated/replayed/library instrs). 5-15% cache/DRAM is aspirational without privileged validation. See Section 10 revised matrix.

---

## 5. Wall-Clock and Frequency Specifically

The bottom line: **don't fight DVFS, model it.**

### 5.1 The double-clock approach

For every kernel of interest, instrument with NVBit to capture *both* `%clock64` start/end (cycles) and `%globaltimer` start/end (nanoseconds). `%globaltimer` is frequency-invariant. So `(globaltimer_end - globaltimer_start) / (clock64_end - clock64_start) = ns/cycle = 1/f_SM` in real time.

### 5.2 Statistical hardening

Run each calibration kernel N=30+ times with cool-downs, drop the first M=3 (warmup), tag each with `nvmlDeviceGetCurrentClocksThrottleReasons`, filter out runs flagged HW slowdown / thermal / power, and report median + IQR.

### 5.3 The "frequency control without locking" trick (ORIGINAL — REVISED IN §10)

> **Codex flagged this as bad.** The silent control kernel contends for SMs, scheduler bandwidth, cache, DRAM bandwidth, power headroom, and thermal budget — confounding the kernel-of-interest. Replaced in §10 with sequential warmup/preconditioning.

### 5.4 The simulator output convention

Your simulator should output **cycles** as primary and **time** as a derived quantity computed from the user's calibrated `f`. Cycle-first is the only honest convention.

---

## 6. Novel SOTA Approaches and Research Frontiers

### 6.1 Sampling and fast simulation

- **Photon** (MICRO 2023) — basic-block-vector clustering, fine-grained warp sampling.
- **GCL-Sampler** (2026) — RGCN over SASS dependency graphs, 258× speedup at 0.37% error using HyFiSS backend.
- **STEM/ROOT** (MICRO 2025) — statistically minimum sample size to bound simulation error.
- **PKA / TBPoint** — older but foundational.

### 6.2 ML-augmented simulation

- **Habitat** (ATC'21) — runtime predictor, uses MLPs + roofline. Now superseded.
- **NeuSight** (ASPLOS'25) — tile decomposition, predicts unseen GPUs. Best generalization.
- **Forecasting GPU Performance** (arXiv:2407.13853v3).
- **Daydream** (ATC'20) — CUPTI dependency graphs.

### 6.3 Analytical models with stall awareness

- **GCoM** (PPoPP'22) — first to capture sub-core structural and sectored-cache stalls.
- **AMALI** (ISCA'25) — extends to LLM inference, 23% MAPE.
- **MDM** — older roofline-derivative.
- **GPUMech** (MICRO'14).

### 6.4 Validation methodology innovations

- **ShadowScope** (arXiv:2509.00300) — side-channel-based validation.
- **AkitaRTM** (MICRO'24) — real-time monitoring of architecture simulations.

### 6.5 The original novelty pitch (REVISED IN §10)

Original three claims:

1. Trustworthy modern-arch (Hopper/Blackwell) calibration *without privileged tooling*.
2. A residual-correction model trained per architecture.
3. Hybrid-fidelity speed coupled with sampling.

> **Codex flagged claim 2 as weak** (cross-system NCU data carries driver/toolkit/compiler/clock/power version bias) and the "first trustworthy Blackwell simulator" framing as risky (Huerta already claims Blackwell). See §10 for revised framing.

---

## 7. Original Recommended Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     CALIBRATION PIPELINE (offline)                  │
│                                                                      │
│  Microbenchmark suite (you write these)                             │
│    ├─ P-chase L1/L2/DRAM latency (Hopper-paper style)               │
│    ├─ wgmma/MMA throughput per precision                            │
│    ├─ TMA descriptor latency                                        │
│    ├─ Frequency probe (clock64+globaltimer)                         │
│    └─ Stall-pattern microkernels                                    │
│                                                                      │
│  Run each on real hardware with:                                    │
│    NVBit (clock64 injection, instruction count, address trace)      │
│    + CUDA Events (wall time)                                        │
│    + NVML poll (frequency, throttle reasons)                        │
│    + CUPTI Activity (kernel/memcpy boundaries)                      │
│    + CUPTI Range Profiler / SASS Metric (probe what's accessible)   │
│    + Silent control kernel for frequency stability  [REMOVED §10]   │
│                                                                      │
│  Output: per-arch calibration JSON                                  │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                       TRACE PIPELINE (per workload)                 │
│                                                                      │
│  Target CUDA application → NVBit tracer (Accel-Sim style or own)    │
│    captures: SASS instr stream, addresses, predicates, control bits │
│  CUPTI Activity: kernel boundaries, dependencies, streams           │
│  Lightweight per-kernel cycle/time stamps (clock64+cudaEvent)       │
│                                                                      │
│  Output: protobuf-compressed traces (HyFiSS-style 3.2× smaller)     │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                    SIMULATION ENGINE (parallel)                     │
│                                                                      │
│  Sub-core SM model    (MICRO'25 Modern-GPU-Cores style)             │
│  Memory pipeline      (per-sub-core, addr-calc/coalesce split)      │
│  Sectored L1/L2 cache (GCoM / Accel-Sim style)                      │
│  MSHR + interconnect                                                │
│  HBM3/HBM3e DRAM model (banks, channels, refresh)                   │
│  Tensor cores: HMMA / wgmma / qgmma async pipelines                 │
│  TMA engine (shared-memory async DMA)                               │
│  Stall taxonomy tracked per-warp-per-cycle (HyFiSS style)           │
│  Sampled execution (GCL-Sampler / Photon optional layer)            │
│  Parallel via OpenMP (Huerta-González 2025 technique, 5.8×)         │
│                                                                      │
│  Outputs (cycle-first): cycles, IPC, per-pipe utilization, stalls,  │
│  cache hits, DRAM bytes, achieved occupancy, branch eff             │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│              ML RESIDUAL-CORRECTION HEAD  [REPLACED §10]            │
│                                                                      │
│  Input: simulator outputs + NVBit-derived static features +         │
│         CUPTI 13 unprivileged features                              │
│  Output: NCU-equivalent metric corrections                          │
│  Training data: open-published NCU profiles for canonical kernels   │
│  Architecture: small transformer or GNN over kernel dependency graph│
└────────────────────────────────────────────────────────────────────┘
```

### 7.1 Implementation phasing (original — REVISED IN §10)

- **Phase 0 (weeks 1-3):** Reproduce Hopper-paper microbenchmarks.
- **Phase 1 (weeks 4-8):** NVBit tracer with `clock64` injection.
- **Phase 2 (months 3-5):** Adopt the Accel-Sim MICRO'25 SM/memory pipeline as your starting simulator core.
- **Phase 3 (months 5-8):** Add HyFiSS-style stall taxonomy.
- **Phase 4 (months 8-12):** Sampling layer + parallelization.
- **Phase 5 (months 12-15):** ML residual head.

### 7.2 Pitfalls to avoid

- **Do not trust `cudaDeviceProp.clockRate`** — it reports the maximum boost clock, not the running one.
- **Do not measure timing on a kernel that's also being NVBit-instrumented for memory traces** in the same run.
- **Don't naively use total clock cycles for throughput** — Hopper paper warns against this.
- **Don't assume NVML clock readings are instantaneous** — they are typically averaged.
- **Watch for kernel concurrency confounding `clock64`** — if multiple kernels overlap, your boundary cycles aren't disjoint.

---

## 8. Open Research Questions

1. **Can CUPTI 13.0 SASS Metric API leak enough HW-counter-derived info to reach NCU stall-reason fidelity unprivileged?**
2. **Can a small per-architecture residual head bridge the simulated-vs-NCU gap to <5% MAPE on stall reasons?**
3. **Is per-kernel SM frequency more predictable than current methodology assumes?**
4. **For Blackwell wgmma + 2nd-gen TMA**, what's the right cycle-accurate pipeline model?
5. **Does the "silent control kernel" frequency-pinning trick distort other SMs' resources?** (Codex says yes; see §10.)

---

## 9. Codex Independent Expert Review

Codex reviewed the original proposal (sections 1-8) on 2026-04-29 with `gpt-5.5:high`. Verbatim findings:

### Bottom Line

> The proposal is directionally interesting, but several claims are too strong. The publishable core is not "NCU without NCU." It is "counter-free, unprivileged calibration and uncertainty-aware simulation for modern NVIDIA GPUs." Do not claim NCU-equivalent metrics or trustworthy Blackwell cycle accuracy until you have privileged validation somewhere or a very strong timing-only inverse-validation story.

### Technical Correctness Flags

1. **`clock64`/`globaltimer` per-kernel frequency is not robust as stated.** `clock64()` is a per-SM counter, not a global kernel cycle counter. NVIDIA documents it as measuring elapsed cycles for the executing thread, including time-slicing, not "cycles_active." `%globaltimer` is a target-specific tool-oriented register whose behavior may change. Paired reads from the same thread can estimate that SM's average clock over that thread's residency, but min/max across CTAs or SMs is not a valid kernel-wide cycle count. Concurrent kernels make this worse.

2. **`cycles_active` from NVBit `clock64` injection is mislabeled.** You can get per-thread/per-CTA elapsed SM cycles. You cannot reconstruct NCU `sm__cycles_active` without PM counters or a very intrusive residency sampler.

3. **"0% error" for `inst_executed`, tensor ops, warp efficiency is too strong.** NVBit can count your instrumented SASS definition. That is not automatically equal to NCU's metric definition, especially with predication, replays, WGMMA/HMMA variants, compiler-generated code, async tensor/TMA paths, and instrumentation perturbation.

4. **L1/L2 hit rate and DRAM bytes at 5-15% are aspirational.** Trace-driven modeling can estimate them, but modern cache behavior includes sectoring, cache operators, compression/eviction effects, L2 partition hashing, TMA/cp.async paths, atomics, and library kernels. Without counters, you cannot honestly validate those numbers broadly.

5. **Stall reasons at 20-30% are probably not defensible.** NCU stall reasons are scheduler-sampled symptoms, not clean causal labels. HyFiSS/GCStack-style accounting may produce useful modeled stall classes, but calling them NCU-equivalent is a category error.

6. **The silent control kernel is a bad calibration primitive.** It can pin boost sometimes, but it consumes SM slots, scheduler bandwidth, cache/L2/DRAM bandwidth, power headroom, and thermal budget. It directly changes the resource availability of the kernel under test. Use warmup/preconditioning before measurement, not concurrent load, unless the paper is explicitly about co-run interference.

7. **Published-NCU-profile ML residual correction is weak.** Cross-system NCU data bakes in driver/toolkit/compiler versions, clock policy, power limits, benchmark shapes, library versions, and counter-definition drift. Use published data only as priors or sanity checks. Train calibration on local timing microbenchmarks and controlled SASS transformations.

8. **Accel-Sim Blackwell support risk is real.** Issue #385 shows `sm_120`/CUDA 12.8 tracing problems. More importantly, Huerta et al. already claim Blackwell modeling with 17.41% MAPE, but their table is RTX 5070 Ti, not necessarily B200/GB200/H100-class data-center behavior. Do not generalize "Blackwell validated" across SKUs.

### Suggested Replacements

- Use CUDA Events or CUPTI Activity timestamps as the primary timing source. On Blackwell, explicitly probe CUPTI Hardware Event System tracing; CUPTI 13 documents HES for kernel timestamps, with limitations such as no MPS, MIG, vGPU, WSL, or confidential compute.
- Use `clock64/globaltimer` only for controlled microbenchmarks and paired same-thread calibration, reported as "effective SM clock distribution," not kernel cycles.
- Replace "NCU-equivalent reconstruction" with "counter-free modeled observables with uncertainty intervals." That is more honest and stronger scientifically.
- Replace ML residuals with Bayesian or optimization-based parameter inference from local experiments: vary occupancy, unroll, cache operator, stride, CTA shape, dependency distance, SASS control bits, and tensor instruction mix; fit simulator parameters to timing deltas.
- Treat Huerta/UPC's MICRO 2025 model as the base core model, not something to casually combine with HyFiSS. Mixing a cycle-accurate core with a hybrid stall taxonomy can double-count stalls unless you define one source of truth.

### Better Novel Angles

The strongest underexplored angle is **differential unprivileged calibration**:

- Generate controlled SASS variants with CUAssembler or equivalent.
- Perturb one suspected bottleneck at a time: dependency distance, reuse bits, register bank pressure, cache operators, stride/locality, CTA residency, tensor pipe pressure.
- Measure timing deltas only.
- Infer hidden structural parameters and stall attribution through inverse modeling.

Other useful angles:

- CUPTI Activity HES on Blackwell for low-overhead timing.
- CUPTI SASS Metric API probing. It uses SASS patching and is separate from Range/PM profiling; do not assume it is counter-gated until tested.
- Sentinel microbenchmarks before/after target kernels to infer cache/frequency state, not concurrent with the target.
- Uncertainty-aware simulator output: cycles, time, and confidence bounds per modeled subsystem.

### Infeasible Or Weak Under These Constraints

- **eBPF/GPUprobe-style host observation** is likely infeasible in this cloud setup. eBPF normally needs host privileges/capabilities and would only expose driver/ioctl behavior, not SM counters.
- **Reverse-engineering counters from CUPTI Activity timestamps** is not realistic. Activity timestamps can improve timing/dependency reconstruction; they do not contain hidden PMU information.
- **"First trustworthy Blackwell simulator"** is risky because Huerta et al. already make a Blackwell accuracy claim, and without NCU you cannot validate cache/stall internals. A better claim is: "first unprivileged, counter-free calibration methodology for modern NVIDIA GPU simulation under cloud restrictions," with a Blackwell case study if you actually validate on the target SKU.

### CUPTI 13 Answer

- Range Profiling and PM Sampling remain hardware-performance-metric paths and should be expected to hit `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` when performance counters are restricted. CUDA 13 also removed the legacy Event/Metric APIs.
- SASS Metric API is different enough to probe. CUPTI documents it as SASS-level metric collection via patching, with GPU/SM/SMSP granularity. It may still fail in some environments, but **do not pre-classify it with Range Profiling until you test it**.

### Risk-Ranked Time Budget

- **1-2 months:** reliable unprivileged timing/tracing harness for CUDA Events, CUPTI Activity, NVBit traces, cubin parsing, NVML logging.
- **3-6 months:** credible Hopper/Ampere simulator calibration for timing/ranking on a bounded benchmark suite.
- **6-12 months:** Blackwell SASS/control-bit/tracer bring-up for common kernels, likely still timing-only validated.
- **12-18+ months:** defensible Blackwell cycle-accuracy on one named SKU with broad workloads.
- **18+ months or infeasible:** NCU-equivalent stall/cache/DRAM metrics without any privileged validation path.

---

## 10. Revised Solution: Counter-Free Unprivileged GPU Simulator Calibration

### What changes after Codex

| Codex flag | Original (wrong/weak) | Revised |
|---|---|---|
| Reframe the claim | "NCU-accurate metrics without NCU" | **"Counter-free modeled observables with uncertainty intervals"** — every output ships with confidence bounds, never claims NCU equivalence |
| `clock64`/`globaltimer` semantics | Treated as kernel-wide cycle counter; "implied frequency f = cycles/time" | `clock64` is per-thread elapsed cycles on its SM, includes time-slicing. Use only for **paired same-thread** measurements on **microbenchmarks**; report **effective SM-clock distribution**, never kernel `cycles_active` |
| "0% error" claims | inst_executed, TC ops, warp eff = 0% | **NCU metric definitions ≠ NVBit definitions.** Predication, replays, async TMA paths, library kernels, instrumentation perturbation all introduce gaps. Report as "NVBit-defined SASS dynamic counts" with a documented mapping (and divergence) from NCU |
| Silent control kernel | Recommended for frequency pinning | **REMOVED.** It contends for SMs, scheduler bandwidth, cache, DRAM bandwidth, power headroom, and thermal budget — confounding the kernel-of-interest. Replaced with **warmup/preconditioning** runs before measurement |
| ML residual head on others' NCU profiles | Primary calibration mechanism | **Demoted to priors / sanity-checks.** Cross-system NCU data carries driver/toolkit/compiler/clock/power version bias. Primary calibration is local differential SASS perturbation (see novel angle below) |
| Combining HyFiSS stalls + UPC cycle-accurate core | Stacked freely | Risk of **double-counting stalls.** Pick one source of truth: either the cycle-accurate model emits stall taxonomy directly, or use HyFiSS-style classification on top of an analytical core — never both at once |

### The novel angle (Codex-suggested, replaces ML residual head)

**Differential unprivileged calibration.** This is the strongest publishable contribution.

1. Generate controlled SASS variants of a calibration kernel using **CUAssembler** (or equivalent SASS-level rewriting). One variant = one perturbed parameter.
2. Perturb a single suspected bottleneck at a time: dependency distance, instruction reuse-cache hint bits, register bank pressure, cache operators (`.ca`/`.cg`/`.cv`), stride/locality, CTA residency, tensor-pipe pressure, control-bit stall counts.
3. Measure **timing deltas only** (CUDA Events / CUPTI Activity).
4. Fit simulator parameters by **inverse modeling**: minimize discrepancy between simulator-predicted and measured timing across the perturbation grid.

This converts the calibration problem from "match counters" to "match timing under controlled SASS perturbations." It's both novel and provably executable in the unprivileged regime.

### Revised wall-clock and frequency recipe

| Source | Use for | Caveat |
|---|---|---|
| `cudaEventElapsedTime` | Primary kernel duration (always-on) | Sub-microsecond; no privilege issues |
| CUPTI Activity (`CUpti_ActivityKernel*`) | Kernel/memcpy boundary, dependencies, streams | Probe **CUPTI HES** (`cuptiActivityEnableHWTrace`) on Blackwell — low-overhead hardware-derived timestamps; documented as not requiring counter privilege; verify on your image |
| `cudaEventRecord` straddling a kernel | Confirmation of CUPTI Activity | Cross-validation only |
| `clock64`/`%globaltimer` paired in **same thread** in **microbenchmarks** | Effective SM-clock distribution | Per-thread, includes time-slicing; not a kernel-wide cycle metric |
| NVML `nvmlDeviceGetClockInfo` (sidecar) | Kernel-side frequency snapshot at start/end | ~50–100 Hz sampling resolution; useful only for kernels >100 ms |
| NVML `nvmlDeviceGetCurrentClocksEventReasons` | Throttle filter | Discard runs flagged HW_SLOWDOWN / thermal / power_brake / sw_power_cap |

**Frequency stabilization (revised):** 100+ warmup iterations *before* measurement, **never concurrent** with the measured kernel. Pre-warm drives the GPU to P0 and pins the boost clock until thermals push it down. Statistical handling: N≥30 trials post-warmup, throttle-bitmask filter, CV<3% acceptance criterion, report median + min + std + measured-frequency mean and std.

### Revised metric reconstruction matrix

Every entry now states the **modeled observable**, the **fidelity claim**, and the **uncertainty source**. None claim NCU equivalence.

| Modeled observable | Source | Honest fidelity claim | Uncertainty source |
|---|---|---|---|
| Kernel wall-time | CUDA Events / CUPTI Activity / HES probe | <2% deviation from internal reproducibility (not vs NCU) | DVFS, scheduler queue |
| Effective SM-clock distribution (per-microbench) | `clock64`/`globaltimer` paired same-thread | 1–3% on calibration kernels | Per-SM clock variation; time-slicing |
| Dynamic SASS instruction counts (NVBit-definition) | NVBit instr count tool | Exact under NVBit's definition | NVBit ≠ NCU definition for predicated/replayed/library instrs |
| Dynamic Tensor-pipe instruction counts (HMMA/wgmma/QGMMA/etc.) | NVBit opcode count | Exact for opcode subset | Opcode taxonomy maintenance, async TMA paths |
| Active-mask average per warp instruction | NVBit predicate capture | Exact under NVBit's sampling | NVBit can perturb scheduler order |
| L1/L2 modeled hit rate | Trace-driven simulation + reuse-distance | Quoted as **modeled**, not measured; bracket with uncertainty bands | Sectoring, L2 partitioning, cache operators, atomics, library paths — *not* validated without privileged counters |
| Modeled DRAM bytes | Trace-driven simulation | Modeled with uncertainty bands | Same as above + replacement-policy assumption |
| Modeled stall classes | Cycle-accurate model emits taxonomy directly (no double-count) | Modeled, not equivalent to NCU PC-sampled stall reasons | Different definition; reported as model output |
| Modeled occupancy timeline | Simulator scheduler model | Modeled output | Scheduler-policy mismatch |
| Static occupancy bound | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | Exact (analytic) | None |

### Revised architecture

- **One source of truth for stalls.** Adopt the UPC MICRO 2025 cycle-accurate sub-core / memory pipeline as the core; let *it* emit the stall taxonomy. Do not stack HyFiSS on top. (HyFiSS-style fast modes remain a separate fast-mode build.)
- **Trace pipeline unchanged at the inputs:** NVBit SASS+addresses+predicates, control bits parsed from cubin via cuobjdump (UPC paper's approach), CUPTI Activity for kernel boundaries.
- **Probe** CUPTI 13.0 SASS Metric API and Hardware Event System (HES) on the target image *before* assuming they're gated.
- **Sampling layer kept** (Photon → GCL-Sampler).
- **OpenMP parallelization kept** (Huerta-González 2025).
- **Calibration: differential SASS perturbation grid, not ML residual on others' NCU.** Use published NCU data as priors/sanity-checks.
- **Output convention kept:** simulator emits cycles + uncertainty bounds as primary; time is derived. Now reinforced by Codex's "uncertainty-aware output" recommendation.

### Revised publishable framing

**Drop:** "First trustworthy Blackwell simulator." Huerta et al. already make a Blackwell accuracy claim and you cannot validate cache/stall internals without NCU.

**Adopt:** **"First unprivileged, counter-free calibration methodology for modern NVIDIA GPU simulation under cloud restrictions, validated by differential SASS perturbation, with a Blackwell case study on a named SKU."**

This is sharper, defensible, and the methodology contribution stands even if the Blackwell case study takes longer than expected.

### Revised time budget (Codex-grounded)

| Months | Milestone | Confidence |
|---|---|---|
| 1–2 | Unprivileged harness: CUDA Events, CUPTI Activity, NVBit tracer, cubin parser, NVML logging — all working in your container | High |
| 3–6 | Hopper/Ampere timing-only calibration on bounded benchmark suite using differential SASS perturbation | High |
| 6–12 | Blackwell SASS/control-bit/tracer bring-up for common kernels (cuBLAS GEMM, FlashAttention-3, LayerNorm); timing-only validation | Medium — Accel-Sim issue #385 needs resolution |
| 12–18+ | Defensible Blackwell cycle-accuracy on **one named SKU** with a broad workload set | Medium-low; depends on tracer maturity |
| 18+ or infeasible | NCU-equivalent stall/cache/DRAM metrics without any privileged validation path | **Codex flags this as possibly infeasible** — frame as future work, don't promise |

### Things to drop entirely (Codex)

- **eBPF / GPUprobe-style host observation.** Needs host privilege, won't help here.
- **Reverse-engineering counters from CUPTI Activity timestamps.** Activity carries no PMU information.
- **Silent concurrent control kernel.** Replaced with sequential warmup.

### Things to probe before assuming gated (Codex)

- **CUPTI 13.0 Hardware Event System** (`cuptiActivityEnableHWTrace`) on Blackwell — documented as low-overhead hardware-derived kernel timestamps not requiring counter privilege; limitations: not on MPS, MIG, vGPU, WSL, or confidential compute.
- **CUPTI 13.0 SASS Metric API** — uses SASS patching, semantically distinct from Range/PM profiling. May or may not be counter-gated; test before classifying.

---

## Sources

### Simulators (papers, repos)

- [Accel-Sim Framework site](https://accel-sim.github.io/)
- [Accel-Sim ISCA 2020 paper](https://mkhairy.github.io/Docs/Accel-Sim.pdf) / [IEEE Xplore](https://ieeexplore.ieee.org/document/9138922/)
- [Accel-Sim GitHub](https://github.com/accel-sim/accel-sim-framework)
- [Accel-Sim Blackwell issue #385](https://github.com/accel-sim/accel-sim-framework/issues/385)
- [Analyzing and Improving Hardware Modeling of Accel-Sim (arXiv:2401.10082)](https://arxiv.org/html/2401.10082v1)
- [Parallelizing a modern GPU simulator (arXiv:2502.14691)](https://arxiv.org/abs/2502.14691)
- [MICRO 2025: Dissecting and Modeling Modern GPU Cores (GitHub)](https://github.com/upc-arco/modern-gpu-simulator-micro-2025)
- [Dissecting and Modeling Modern GPU Cores (MICRO 2025)](https://dl.acm.org/doi/full/10.1145/3725843.3756041)
- [Analyzing Modern NVIDIA GPU cores (arXiv:2503.20481)](https://arxiv.org/abs/2503.20481)
- [HyFiSS MICRO 2024 (DOI)](https://dl.acm.org/doi/10.1109/MICRO61859.2024.00022) / [GitHub](https://github.com/ConvolutedDog/HyFiSS)
- [PPT-GPU GitHub](https://github.com/NMSU-PEARL/PPT-GPU)
- [PPT-GPU SC'21 paper (DOI)](https://dl.acm.org/doi/10.1145/3458817.3476221)
- [GCoM PPoPP'22 paper](https://dl.acm.org/doi/10.1145/3470496.3527384)
- [AMALI ISCA 2025](https://dl.acm.org/doi/10.1145/3695053.3731064)
- [TrioSim ISCA 2025](https://sarchlab.org/triosim_isca2025.pdf) / [DOI](https://dl.acm.org/doi/10.1145/3695053.3731082)
- [GCStack+GCScaler ISCA 2025](https://dl.acm.org/doi/10.1145/3695053.3731068) / [GitHub](https://github.com/yonsei-hpcp/gcstack_gcscaler)
- [STEM+ROOT MICRO 2025](https://dl.acm.org/doi/10.1145/3725843.3757107) / [PDF](https://seonjinna.github.io/assets/pdf/MICRO25_stem_root.pdf)
- [Photon MICRO 2023](https://dl.acm.org/doi/10.1145/3613424.3623773) / [Repo](https://github.com/nus-comparch/photon)
- [GCL-Sampler arXiv:2603.00551](https://arxiv.org/abs/2603.00551)
- [Swift-Sim DATE 2025](https://ieeexplore.ieee.org/document/10992801/) / [Repo](https://github.com/xurongxiang/Swift-GPUSim)
- [ClusterSim IISWC 2025](https://ieeexplore.ieee.org/document/11241997/)
- [SimAI NSDI 2025](https://github.com/aliyun/SimAI)
- [Multiverse NSDI 2025](https://github.com/Harnets/multiverse)
- [Lumos arXiv](https://arxiv.org/abs/2504.09307)
- [MGPUSim / Akita](https://gitlab.com/akita/mgpusim) / [SARC Lab](https://sarchlab.org/)
- [MAccel-sim IISWC 2024 poster](https://engineering.purdue.edu/tgrogers/publication/bose-iiswc-poster-2024/bose-iiswc-poster-2024.pdf)
- [Need-for-Speed / NVArchSim HPCA 2021](https://d1qx31qr3h6wln.cloudfront.net/publications/HPCA_2021_NVArchSim.pdf)
- [GainSight (Stanford)](https://gainsight.stanford.edu/backend/gpu-sim.html)
- [Habitat ATC'21](https://www.usenix.org/system/files/atc21-yu.pdf)
- [NeuSight ASPLOS 2025](https://dl.acm.org/doi/10.1145/3669940.3707265) / [arXiv:2407.13853](https://arxiv.org/abs/2407.13853) / [Repo](https://github.com/sitar-lab/NeuSight)
- [Daydream ATC 2020](https://arxiv.org/abs/2006.03318)
- [GhOST ISCA 2024](https://liberty.cs.princeton.edu/Publications/isca24_ghost.pdf)
- [GPU-FPX repo](https://github.com/LLNL/GPU-FPX)
- [PyTorchSim MICRO 2025](https://dl.acm.org/doi/10.1145/3725843.3756045) / [Repo](https://github.com/PSAL-POSTECH/PyTorchSim)
- [GPEmu VLDB 2024](https://vldb.org/pvldb/vol18/p1919-wang.pdf) / [Repo](https://github.com/mengwanguc/gpemu)
- [Vortex GPU](https://vortex.cc.gatech.edu/) / [Repo](https://github.com/vortexgpgpu/vortex)
- [Vulkan-Sim repo](https://github.com/ubc-aamodt-group/vulkan-sim)
- [gem5 ISCA 2024 GPU tutorial](https://www.gem5.org/assets/files/isca2024-tutorial/05-gpu.pdf)
- [NaviSim PACT 2022](https://bu-icsg.github.io/publications/2022/navisim_pact_2022.pdf)
- [GPU Scale-Model HPCA 2024](https://users.elis.ugent.be/~leeckhou/papers/hpca2024.pdf)
- [ASTRA-sim repo](https://github.com/astra-sim/astra-sim) / [Site](https://astra-sim.github.io/)

### NVBit + CUPTI + profiling

- [NVBit GitHub](https://github.com/NVlabs/NVBit) / [Releases](https://github.com/NVlabs/NVBit/releases)
- [NVBit MICRO 2019 paper](https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2019_NVBit.pdf) / [ACM DOI](https://dl.acm.org/doi/10.1145/3352460.3358307)
- [NVBitFI GitHub](https://github.com/NVlabs/nvbitfi)
- [Accel-Sim NVBit tracer README](https://github.com/accel-sim/accel-sim-framework/blob/release/util/tracer_nvbit/README.md)
- [CUPTI 13.0 release notes](https://docs.nvidia.com/cupti/13.0.1/release-notes/release-notes.html)
- [CUPTI 12.8 release notes](https://docs.nvidia.com/cupti/12.8/release-notes/release-notes.html)
- [CUPTI 13.0 docs](https://docs.nvidia.com/cupti/)
- [CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)
- [CUPTI Range Profiling API](https://docs.nvidia.com/cupti/api/group__CUPTI__RANGE__PROFILER__API.html)
- [CUPTI PC Sampling API](https://docs.nvidia.com/cupti/api/group__CUPTI__PCSAMPLING__API.html)
- [CUPTI PM Sampling API](https://docs.nvidia.com/cupti/api/group__CUPTI__PM__SAMPLING__API.html)
- [ERR_NVGPUCTRPERM solutions for CUPTI](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti)
- [ERR_NVGPUCTRPERM solutions for NCU](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute)
- [CUPTI fakeroot Singularity forum](https://forums.developer.nvidia.com/t/access-cupti-performance-counter-as-fakeroot-inside-singularity-container/273542)
- [Restricted vs. unrestricted profiling forum](https://forums.developer.nvidia.com/t/restricted-vs-unrestricted-profiling/358215)
- [Eunomia: 2025 Accelerator Profiling Toolkit Review](https://eunomia.dev/blog/2025/04/11/the-accelerator-toolkit-a-review-of-profiling-and-tracing-for-gpus-and-other-co-processor/)
- [Eunomia: GPU Profiling Under the Hood](https://eunomia.dev/blog/2025/04/21/gpu-profiling-under-the-hood-an-implementation-focused-survey-of-modern-accelerator-tracing-tools/)
- [Polar Signals: Continuous NVIDIA CUDA Profiling In Production](https://www.polarsignals.com/blog/posts/2025/10/22/gpu-profiling)

### Microbenchmarking / measurement methodology

- [Benchmarking and Dissecting the Nvidia Hopper Architecture (arXiv:2402.13499)](https://arxiv.org/abs/2402.13499)
- [Dissecting the Hopper Architecture (arXiv:2501.12084)](https://arxiv.org/html/2501.12084v1)
- [Dissecting the Blackwell Architecture (arXiv:2507.10789)](https://arxiv.org/html/2507.10789v2)
- [Microbenchmarking Blackwell in-depth (arXiv:2512.02189)](https://arxiv.org/abs/2512.02189)
- [Demystifying Ampere through Microbenchmarking (arXiv:2208.11174)](https://ar5iv.labs.arxiv.org/html/2208.11174)
- [Methodology for GPU Frequency Switching Latency (arXiv:2502.20075)](https://arxiv.org/html/2502.20075v1)
- [Architectural Trade-offs (H100 vs H200 power-capping, arXiv:2604.11391)](https://arxiv.org/html/2604.11391)
- [Methodology for Fine-Grain GPU Power Visibility (arXiv:2412.12426)](https://arxiv.org/html/2412.12426v1)
- [cuThermo: Heat Map Profiling with NVBit (arXiv:2507.18729)](https://arxiv.org/abs/2507.18729)
- [GainSight (arXiv:2504.14866)](https://arxiv.org/abs/2504.14866)
- [Forecasting LLM Inference Performance (LIFE, arXiv:2508.00904)](https://arxiv.org/html/2508.00904v1)
- [SynPerf (arXiv:2601.14910)](https://arxiv.org/html/2601.14910)
- [HetGPU (arXiv:2506.15993)](https://arxiv.org/html/2506.15993v1)
- [MSched (arXiv:2512.24637)](https://arxiv.org/html/2512.24637)
- [Low Overhead Instruction Latency Characterization (arXiv:1905.08778)](https://arxiv.org/pdf/1905.08778)
- [Dissecting GPU Memory Hierarchy (arXiv:1509.02308)](https://arxiv.org/pdf/1509.02308)

### Profiling alternatives + analytical models

- [Omniwise (arXiv:2506.20886)](https://arxiv.org/abs/2506.20886)
- [Privacy-Preserving Performance Profiling (arXiv:2509.21762)](https://arxiv.org/abs/2509.21762)
- [Neutrino OSDI '25](https://www.usenix.org/conference/osdi25/presentation/huang-songlin) / [Site](https://open-neutrino.github.io/)
- [GPA: GPU Performance Advisor (arXiv:2009.04061)](https://arxiv.org/abs/2009.04061)
- [DrGPU SPEC ICPE 2023](https://research.spec.org/icpe_proceedings/2023/proceedings/p43.pdf)
- [HPCToolkit](https://hpctoolkit.org/)
- [HPCToolkit GPU PC Sampling - ALCF](https://www.alcf.anl.gov/sites/default/files/2024-11/HPCToolkit-ALCF-2024-10.pdf)
- [Solving Nsight Compute Permission Error (Hung-Yueh Chiang)](https://hychiang.info/blog/2024/nsight-compute-permission-error/)
- [Detailed GPU Cache Model Based on Reuse Distance Theory - Nugteren 2014](https://cnugteren.github.io/downloads/Nugteren2014b.pdf)

### NVML / DCGM / containers / clouds

- [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)
- [NVML Clocks Throttle Reasons](https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksThrottleReasons.html)
- [DCGM GitHub](https://github.com/NVIDIA/DCGM)
- [AWS: Optimize GPU settings on EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/optimize_gpu.html)
- [Robust benchmarking in noisy environments (Edelman, MIT)](https://math.mit.edu/~edelman/publications/robust_benchmarking.pdf)
- [In Pursuit of High-Fidelity GPU Kernel Benchmarking (standardkernel.com)](https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/)
- [How we (try to) benchmark GPU kernels accurately (jan.ai)](https://www.jan.ai/post/how-we-benchmark-kernels)
- [Forum: Allow non-root users to change application clocks](https://forums.developer.nvidia.com/t/allow-non-root-users-to-change-application-clocks/244709)
- [Forum: Using NVML on HPC cluster without root](https://forums.developer.nvidia.com/t/using-nvml-api-on-hpc-cluster-without-root/219775)

---

*Generated by Claude Opus 4.7 (1M context) on 2026-04-29. Independent expert review by Codex `gpt-5.5:high`.*
