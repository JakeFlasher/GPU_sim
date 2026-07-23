# Fable-5 design prompt #1 — sm_86 calibration microbench designs (the sim's accuracy walls)

> Copy everything below the line and send it to Fable 5. Self-contained. HIGH priority.

---

You are a GPU microarchitecture and microbenchmark-design expert. Design **five counter-free (clock64-based) calibration microbenches** for an NVIDIA **RTX 3090 Ti (GA102, sm_86, CC 8.6)** whose measured outputs become **INPUTS to an event-driven GPU simulator** (they fill per-architecture "card" constants). I need concrete, implementable probe designs — kernels, access patterns, sweep axes, what `clock64()` measures, the expected signature, the confounds, and the fallback when inconclusive — not a survey.

## Hard constraints (these shape every probe)

- **Rule 1 — the measured value must be counter-free (clock64/latency/throughput-derived), NEVER fitted from NCU/hardware-counter values.** NCU is the simulator's prediction TARGET and may be used only as an out-of-band *cross-check* of a probe's interpretation, never as the fitted source. So each probe must recover its quantity from timing/throughput signatures, not from reading a hit-rate/traffic counter.
- **Pinned protocol v2.1:** the host is clock-locked — graphics/SM clock 1350 MHz, memory clock 10501 MHz (full-rate GDDR6X), persistence on. All probes run under this lock; report values in SM-cycles or bytes/SM-cycle at this operating point.
- Each probe must emit either a **measured value** OR a **typed bounded gap** carrying its evidence curve — never a silent guess.

## GA102 / RTX 3090 Ti facts

- 84 SMs; 4 warp schedulers/SM; up to 48 warps (1536 threads)/SM.
- L1: 128 KB unified L1/shared per SM, carveout-configurable; **sectored** (32 B sectors, 128 B lines); finite MSHR pool (count unknown — probe (d)).
- L2: **6 MiB**, sliced (GA102 whitepaper: 512 KB per 32-bit memory controller × 12); 128 B lines / 32 B sectors.
- DRAM: GDDR6X, 384-bit, 21 Gbps/pin → **~1008 GB/s** vendor peak.
- **Measured latencies already in hand (clock64, v2.1, dependent-load pointer-chase):** L1-hit **33.0** cy, shared-load **23.0** cy, L2-hit **243.6** cy, DRAM **422.6** cy. These are stable/validated.
- Existing harness: a single `.cu` with `chase` / `shared_chase` / `mshr` / `conflict` pointer-chase probes driven by clock64 + read-only NVML covariates. It is extensible with new standalone `.cu` probes + a Python sweep driver.

## The accuracy walls these probes must close (why they matter)

The simulator currently mispredicts on sm_86: **L2 unified sector-hit-rate MAE ≈ 13.96 pp**, and **cycles run ~2.8× too fast** (predicted/measured cycle ratio geomean ≈ 0.358). The sim's own error attribution says the cycles undershoot is **upstream of aggregate DRAM bandwidth** (per-SM port width / MSHR / async-copy timing bind first), and the L2 residual is bracketed by an **L2↔DRAM fetch-granularity** effect (37.5% modeled with 32 B misses / 62.5% if 64 B sector-pairs / ~55% measured by NCU).

## The five probes (design each; then RANK them by load-bearing value)

**(a) L2↔DRAM fetch granularity (32 B vs 64 B).** Does an L2 miss fetch a single 32 B sector or a 64 B sector-pair (or a full 128 B line) from DRAM? Public literature is CONTESTED (V100-era reports 64 B L2↔DRAM; A100 descriptions say all L2 fills are 32 B; GA102/GDDR6X has no published value), so a microbench is the ONLY admissible source. Design a sector-stride latency/throughput probe that reveals the DRAM-fetch granule (e.g. striding accesses so consecutive-vs-alternating-sector patterns expose whether an adjacent sector rides in "for free").

**(b) Per-SM L1↔XBAR port width.** The per-SM bytes-per-cycle bound from L1 to the L2/crossbar — a ~32 B/cycle-class per-SM limit that saturates before the global DRAM aggregate. Design a per-SM streaming/saturation microbench (single-SM, many warps) that measures achieved L1-miss bytes/SM-cycle to the L2 and exposes the per-SM port ceiling.

**(c) Achievable (not peak) DRAM bandwidth.** The achievable GDDR6X streaming bandwidth under the lock (vendor peak 1008 GB/s; one workload showed hardware ≈ 668 B/cy ≈ 90% of peak). Design an all-SM streaming read/write/copy microbench that measures achieved sustained DRAM bytes/SM-cycle to supersede the vendor-peak default.

**(d) MSHR / outstanding-miss capacity (per SM).** The single-thread `mshr` probe (k independent DRAM chains interleaved in ONE thread) found NO saturation knee through k=32 — it can't saturate a per-SM MSHR pool. Design a **multi-warp saturation** probe that pins the per-SM outstanding-miss capacity (the MLP limit), disentangling MSHR count from scheduler/port limits.

**(e) L2 associativity / slice-hash / replacement policy.** Currently documented-default (16-way, LRU, `line_mod` slice hash). The `conflict` pointer-chase probe is L1-only and came back FLAT at 33 cy (index hashing defeats it). Design an L2-targeted associativity/slice-hash probe (an sm_120 NCU-in-loop associativity probe exists to port — but re-found it clock64-first for Rule 1).

## What I need (be concrete and implementable)

1. **Per probe:** the kernel (access pattern, pointer-chase vs streaming, stride schedule), the thread/warp/block/occupancy configuration, exactly what `clock64()` (or a throughput timer) measures, the confound-isolation trick, the expected numeric signature/knee, and the typed **bounded-gap fallback** when the signature is ambiguous.
2. **Rank the five** by load-bearing value for closing the L2-rate and cycles walls, with rationale. (A prior reviewer's guess was: L2 fetch granularity → per-SM XBAR port width → MSHR saturation → achievable DRAM bandwidth → L2 associativity/slice-hash — validate or reorder.)
3. **The disentanglement method:** how to design each probe so it measures a REAL hardware limit and cannot be confounded by a simulator compensating error (the core risk — a plausible-but-useless "bounded gap").
4. **Confounds per probe** and how to defeat them: index/set hashing, DVFS/thermal (mitigated by the lock but verify via NVML), L2 compression/ECC, sector prefetch/coalescing, cache-warmth, the compressed L2:DRAM latency contrast under the lock (already known to weaken step detectors), and cross-SM interference.
5. **Which probes genuinely need multi-warp / occupancy sweeps**, and **which (if any) truly require NCU-in-the-loop** (the calibration carve-out — NCU as validation, still never a fitted input) vs pure clock64.
6. **How to record each result** so the simulator consumes a measured value or a typed bounded gap (evidence curve + operating point + confound verdicts), never a guess.

Deliver: five concrete probe designs, the ranked load-bearing order, the HW-limit-vs-sim-compensating-error disentanglement method, the per-probe confound table, and the record/provenance shape.
