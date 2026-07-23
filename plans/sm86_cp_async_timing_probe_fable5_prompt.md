# Fable-5 design prompt #4 (follow-up) — the 6th calibration probe: cp.async/LDGSTS timing

> Copy everything below the line and send it to Fable 5. Self-contained. This is a FOLLOW-UP to the
> calibration probe suite you already designed (probes a–e) — you flagged a missing 6th probe; design it fully.

---

You are a GPU microarchitecture and microbenchmark-design expert. In an earlier consultation you designed a **five-probe clock64-only calibration suite** for an **RTX 3090 Ti (GA102, sm_86)** whose measured values become INPUTS to an event-driven GPU simulator: (a) L2↔DRAM fetch granularity, (b) per-SM L1↔XBAR port width, (c) achievable DRAM bandwidth, (d) per-SM MSHR/MLP capacity, (e) L2 associativity/slice-hash. You explicitly flagged a **missing 6th probe**: *"your attribution names async-copy (cp.async/LDGSTS) timing among first-binders, and no probe here measures it. A sixth probe (dependent cp.async→cp.async.wait_group→consume latency, plus a (b)-style cp.async streaming plateau) is cheap to bolt onto (b)'s harness; without it, part of the 2.8× [cycles undershoot] may stay unattributed."* **Design that 6th probe fully now.**

## Why this matters (the unattributed part of the cycles wall)

The simulator runs ~2.8× too fast (predicted/measured cycle ratio geomean ≈ 0.358). On sm_86 the tensor-core-feeding kernels (GEMM, conv2d, flash-attention) move their global loads through **`LDGSTS` (cp.async / `cp.async.cg`/`.ca`)** — an asynchronous global→shared copy that bypasses the register file and lands data in shared memory, committed in groups and awaited via `cp.async.wait_group` / `cp.async.commit_group`. The simulator currently prices `LDGSTS` crudely (as a bypass load straight to L2). The real cost — issue cost, in-flight async-copy capacity, the global→shared completion latency, and the effective streaming bandwidth of the async path vs the demand-load path — is uncarded. This is likely a first-order binder for exactly the tensor-core cells where the sim is most wrong.

## Hard constraints (same as the suite)

- **Rule 1 — counter-free (clock64/throughput-derived), NEVER fitted from NCU.** NCU may only cross-CHECK a probe's interpretation as an out-of-band verdict, never be the fitted source.
- **Pinned protocol v2.1:** SM clock 1350 MHz, mem clock 10501 MHz, persistence on; report in SM-cycles or bytes/SM-cycle at this lock.
- Emit a **measured value** OR a **typed bounded gap** (your gap taxonomy: `RANGE` / `CENSORED` / `MIXTURE` / `UNRESOLVED_STRUCTURE` / `SURFACE_GAP`) — never a silent guess.
- Every card constant must be **overdetermined by ≥2 independent readouts** and never tuned against the simulator's end-to-end output (your disentanglement doctrine).

## GA102 / sm_86 facts

- 84 SMs; 4 warp schedulers/SM; up to 48 warps (1536 threads)/SM; 128 KB unified L1/shared per SM.
- Measured latencies in hand (clock64, v2.1): L1-hit 33.0, shared-load 23.0, L2-hit 243.6, DRAM 422.6 SM-cycles.
- `LDGSTS` is Ampere's async global→shared copy; on sm_86 the global source goes through L1/L2 (cache modifier `.cg`/`.ca`) and lands in shared memory; `cp.async.commit_group` + `cp.async.wait_group N` gate completion; there is a hardware limit on outstanding async-copy groups/bytes in flight.
- Your suite already measures the demand-load port (b) and the demand-miss MSHR/MLP capacity (d) — the async path may share or bypass those structures; the 6th probe must say which.

## Design the probe (concrete, implementable)

1. **Latency sub-probe.** A dependent chain that issues a single `cp.async` of one element, `commit_group`, `wait_group 0`, a `__syncthreads()` (or bar), then a shared-memory read whose result addresses the next iteration — so `clock64` around the round trip measures the **global→shared async-copy completion latency**. Sweep: source residency (L2-resident 3 MB footprint vs DRAM-resident 384 MB), cache modifier (`.cg` vs `.ca`), and copy size (4/8/16 B per thread). Compare against the demand-load latency (L2-hit 243.6 / DRAM 422.6) to isolate any async-path overhead (commit/wait/bar cost) beyond the underlying memory latency.
2. **Throughput/plateau sub-probe** (a (b)-style streaming plateau). One block on one SM, many warps streaming `cp.async` copies of a large footprint into a shared-memory ring (double/multi-buffered so `wait_group` doesn't serialize), measuring achieved **B/SM-cycle** of the async path; sweep warp count and in-flight group depth (`wait_group N` for N ∈ {0,1,2,4,…}). This gives the async-copy streaming ceiling and the **in-flight async-group/byte capacity** (the knee).
3. **What clock64 measures** per sub-probe; the expected signature (e.g. does async latency ≈ demand latency + a fixed commit/wait overhead? does the throughput plateau match the (b) demand port width, or is there a separate async port/queue?).
4. **Disentanglement (your doctrine):** how to tell whether the async path shares the demand-load port (b) and MSHR pool (d) or has its own — via RTT contrast + Little's-law triangulation across the latency and throughput sub-probes, so the result is overdetermined. What card constant(s) does this yield (async-copy issue cost, completion latency, in-flight capacity, streaming bandwidth) and how do they feed a sim `LDGSTS` timing model?
5. **Confounds & defeats:** shared-memory bank conflicts in the sink, `wait_group` over-serialization masking the true in-flight capacity, `bar`/`__syncthreads` cost contaminating the latency read, the L2/DRAM residency of the source, compiler eliding the async copy, and the v2.1 lock (verify via NVML). Give the defeat + verification signal for each (matching your consolidated confound-table style).
6. **Typed-gap fallback** if the plateau/knee is ambiguous, and the **provenance record shape** (matching your probe record schema: `probe`, `quantity`, `decision`, `value|gap`, `conditioning`, `operating_point`, `evidence`, `confound_verdicts`, `cross_checks`, `consumer_binding`).
7. **Ranking:** where does this 6th probe sit in your load-bearing order (you had b→a→d→c→e) given it targets the tensor-core cells' cycles error specifically?

Deliver: the two sub-probe designs (latency + throughput) with kernels/sweeps/signatures, the disentanglement from (b)/(d), the card constants produced + how they feed the sim LDGSTS model, the confound table, the typed-gap fallback + record shape, and the ranking placement.
