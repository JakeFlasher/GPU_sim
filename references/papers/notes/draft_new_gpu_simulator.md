> **Retired research draft (2026-09-23).** This document proposes a cloud
> SM120/RTL-oriented direction and is not an active project plan. The current
> scope is local-machine SM86 software simulation judged by NCU accuracy and
> simulator wall time; see the current roadmap and positioning memo.

# Research Draft: Event-Driven GPU Architectural Simulator (NVArchSim-Inspired)

## Research Direction
Build an extremely fast GPU architectural simulator from scratch, mimicking the design pattern presented in NVIDIA's NVArchSim (HPCA 2021, Villa et al.) and surpassing it with open-source availability and modern GPU support.

## Problem Statement
Current SOTA simulators (GPGPU-Sim, Accel-Sim, MICRO 2025 enhanced simulator) are all based on cycle-by-cycle ticking of every component every cycle. This is fundamentally slow (480,000x slower than hardware per GPGPU-Sim benchmarks; 93% of time spent in SM cycles per Huerta 2025 profiling). They also ignore too many architectural details of modern GPUs (Blackwell, Hopper) and use an architectural design based on very old NVIDIA GPU details (Tesla/Fermi era, 2006). This makes them extremely slow AND inaccurate for modern AI workloads.

NVArchSim (NVIDIA's internal simulator) achieves 40-250x speedup over GPGPU-Sim through event-based simulation, but it is proprietary and unavailable to the research community. No Blackwell-capable event-driven simulator exists in open-source.

## Target Hardware
Cloud RTX PRO 6000 Blackwell (SM120/Blackwell workstation, GB202, compute capability 12.0). Local smoke-test host on this branch is RTX 3090 Ti (GA102/Ampere consumer, sm_86, CC 8.6, 24GB GDDR6X) — used only for non-root protocol smoke testing of sm_86-portable kernels; sm_120-specific paths require the cloud rental. Validation against real hardware via Nsight Compute profiling.

Critical distinction: SM120 (consumer Blackwell) does NOT have tcgen05 or TMEM. It uses extended mma.sync (Ampere-style warp-synchronous) with FP8/FP6/FP4 + block_scale support. The simulator should model sm_120 tensor execution as extended warp-synchronous mma.sync, NOT assuming SM100-style tcgen05/TMEM semantics.

## Core Design Principles (from NVArchSim HPCA 2021)

1. **Event-based simulation engine**: Components only tick on cycles they have work, not every cycle. Skip idle cycles entirely.
2. **No retry logic**: Flow-controlled queues prevent wasted simulation effort. Components signal "not full" to upstream before they send.
3. **Single-threaded** for reproducibility and simplicity. NVArchSim dispatches "several millions of events per second" single-threaded.
4. **Variable fidelity per component**: Only add detail where it improves system-level accuracy.
5. **Hybrid trace- and execution-driven**: SASS traces for deterministic regions, partial functional execution for non-deterministic regions (spinloops, atomic contention).
6. **Abstract channels** with bandwidth and latency for interconnects (NVArchSim showed <1% accuracy impact vs detailed interconnect).

## Proposed Architecture

### Event Engine
- Start with binary heap priority queue, upgrade to hybrid timing wheel + min-heap after profiling
- Tagged enum event dispatch (no virtual dispatch on hot path)
- Pure next-event clock advancement (skip idle cycles)
- Event record: compact POD {time, seq, target_id, kind, payload_ref}
- Target throughput: 3-10M events/sec initially, 10-30M optimized

### Programming Language
Rust recommended (tagged enums for zero-cost dispatch, memory safety, cargo build system, future parallelization). C++ interop layer for NVBit trace parsing (protobuf).

Alternative: C++ for v1 if team prefers (Codex GPT-5.4 recommended C++ as default, Rust as strong alternative).

### Component Hierarchy
- **GPU-level**: CTA/cluster distributor, kernel queue manager
- **Per-SM actors** (one per SM, event-driven): Each SM is an independent actor containing 4 sub-cores with warp schedulers, register file (banked), execution units (FP32/INT32/FP64/SFU/TC/Uniform), L0 icache, L1 data cache (sectored, banked), shared memory, async copy engine
- **SM internal model**: NOT per-event for warp scheduler picks. SM wakes, runs internal mini-loop consuming ready work for several issue cycles, schedules next self-event. This avoids flooding the event queue with per-instruction events.
- **L2 slice actors** (per memory partition)
- **DRAM controller actors** (timing-accurate, GDDR7 parameters)
- **Abstract bandwidth+latency channels** for interconnect (no routing simulation)

### Fidelity Levels (following NVArchSim)
- SM frontend: LOW (trace-driven, stream buffer prefetch)
- Warp scheduler: MEDIUM (CGTY policy + control bits from trace)
- Execution units: MEDIUM (throughput + latency queues)
- Tensor Core (sm_120): MEDIUM (mma.sync latency model, not tcgen05)
- L1 cache: MEDIUM (tags + banks + sectors + MSHR)
- Interconnect: LOW (abstract channels)
- L2 cache: MEDIUM (partitioned, hit/miss)
- DRAM: HIGH (timing-accurate, GDDR7-specific)

Note from Codex review: "RF cache", "uniform unit", and "L0 icache with stream-buffer prefetcher" are not solid public architectural facts for sm_120. These should be treated as tunable hypotheses, not baseline spec.

### Trace Format
SASS-level via NVBit (compatible with existing MICRO 2025 enhanced tracer format), extended with:
- mbarrier identities and arrive/wait tracking
- Cluster metadata (rank, dimensions, DSMEM accesses)
- Async copy group boundaries
- Control bits (stall counter, yield, dependency counters, reuse bits)
- For non-determinism: detect divergence at dynamic trace level (control flow, memory addresses, predicate outcomes, barrier counts, mbarrier state transitions). "Unstable PCs across runs" alone is insufficient.

### Validation Strategy
- Primary metric: sm__cycles_elapsed.avg from Nsight Compute
- Must include secondary counters: issued instructions by pipe, tensor utilization, L1/L2 hit rates, DRAM bytes/latency, occupancy, stall reasons
- Cross-validate against MICRO 2025 simulator on same traces
- Target MAPE: less than 15%
- Hardware measurement protocol: locked clocks, 10 repeats, warmup discard, thermal monitoring
- Laptop-specific: monitor clock/power variability during validation

### Missing/Under-modeled sm_120 Features to Investigate
- Explicit static warp-to-scheduler assignment
- Thread block clusters + DSMEM
- clusterlaunchcontrol.try_cancel
- mbarrier semantics
- ldmatrix/stmatrix variants used by tensor paths
- Shared/L1 carveout behavior
- Laptop-specific clock/power variability

## Anti-Patterns to Avoid (from Codex review)
1. Per-cycle global ticking hidden inside "event-driven" marketing
2. Actor over-decomposition (don't make sub-cores separate actors unless profiling demands it)
3. Assuming undocumented microarchitecture as fact (treat reverse-engineered details as tunable hypotheses)
4. Validating only against one top-line cycle metric (need multi-level counter validation)
5. Coupling high-fidelity DRAM model to overly abstract interconnect with no contention calibration

## Key References
- NVArchSim HPCA 2021: Villa et al., "Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator"
- MICRO 2025: Huerta et al., "Dissecting and Modeling the Architecture of Modern GPU Cores"
- NaviSim PACT 2022: Bao et al., "NaviSim: A Highly Accurate GPU Simulator for AMD RDNA GPUs"
- 65 curated papers in papers/FINAL_curated_paper_list.md
- Blackwell microbenchmarking: arxiv 2512.02189, arxiv 2507.10789
- SM120 MMA atoms: CUTLASS cute/arch/mma_sm120.hpp
- Huerta 2025: "Parallelizing a Modern GPU Simulator" (93% time in SM cycles)

## Available Resources
- modern-gpu-simulator-micro-2025/: Existing SOTA open-source GPU simulator (reference architecture)
- NVIDIA_Manuals/: Complete PTX ISA, tuning guides, Nsight Compute guide, CUTLASS/CuTe docs
- papers/: 65 curated papers covering all relevant domains
- Local RTX 3090 Ti hardware for sm_86-portable smoke testing; cloud RTX PRO 6000 Blackwell rental for sm_120 validation
- NVBit 1.8 tracer (working on SM120, verified in Phase 0)
- Docker container with CUDA 12.8 + g++11 for existing simulator builds

## Implementation Phases
- Phase A: Event engine + minimal SM (binary heap, 1 SM, 4 sub-cores, round-robin scheduler)
- Phase B: Memory hierarchy (L1/shared, L2, DRAM with GDDR7 timing)
- Phase C: Trace integration (NVBit protobuf parser, SASS opcode mapping, control bits)
- Phase D: Modern features (tensor core model, async copy, mbarrier, clusters/DSMEM)
- Phase E: Accuracy tuning + validation (sweep benchmarks, per-kernel error attribution, fidelity tuning)

## Constraints
- All simulator development and testing can run on host (no GPU needed for simulation)
- Local trace collection and sm_86-portable smoke testing run on bare-metal host with RTX 3090 Ti; sm_120 trace collection and the cloud-target hardware profiling require the cloud RTX PRO 6000 Blackwell rental
- Must be compatible with existing NVBit trace format for reuse of collected traces
- Open-source from day one
