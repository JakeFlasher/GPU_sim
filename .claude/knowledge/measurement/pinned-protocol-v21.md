# Pinned Measurement Protocol v2.1 (sm_86 reference host)

Use when: capturing or interpreting ANY NCU label, pchase card, or noise-floor
number in this repo; migrating the protocol to the 5060 laptop / GB202 cloud.
Companion evidence: `plans/phase1_full_tier_results.md`,
`plans/host_state_and_migration_log.md` (sudo history + replay checklist).

## The pin (v2.1, 2026-07-06, RTX 3090 Ti, driver 610.43.02)

| layer | setting |
|---|---|
| host (root, does NOT survive reboot) | `nvidia-smi -pm 1; -lgc 1350; -lmc 10501` (core at the card's `sm_clock_ref_mhz`; mem at the single full-rate GDDR6X point) |
| NCU (explicit, never defaulted) | `--clock-control none --cache-control all --replay-mode kernel` (NCU default changed base→boost in 2026.1 — release-note verified) |
| sidecar | `measurement_protocol` block records clock/cache/replay pins + one POST-capture host sample (persistence, SM/mem clock, power, temp, throttle bitmask, GPU index) |
| bundle | `freeze_label_bundle.py` VERIFIES every sidecar's block against the pin — it never stamps the protocol on trust |

Protocol history: v1 (legacy, driver 595, no explicit control) → v2 (core lock
only — memory clock free-ran and wobbled uncore latencies) → **v2.1 (both
locks; fixed core:mem ratio 1350:10501)**. NEVER mix timing labels across
protocol versions in one accuracy claim: v2→v2.1 shifted cycles −2.4/−3.0%
and L2 hit rate +3pt absolute while counts stayed bit-identical.

## Why the pin exists (measured, full 17-cell roster)

- Cycles shift 13–19% ACROSS clock-control modes while staying <1% repeatable
  within each; "DVFS-stable" cycles depend on the core:mem ratio.
- Uncore pchase latencies in SM cycles are ratio-dependent (L2 250→256→243.6,
  DRAM 527→436→422.6 across the three regimes) while core-domain latencies are
  EXACT constants (L1 33.0016, shared 23.0016 — bit-identical across drivers
  and clock regimes; matches Abdelkhalik et al. HPEC'22 A100 numbers exactly).
- NCU pct-of-sustained-peak metrics can exceed 100% (NVIDIA-documented:
  multi-pass replay on non-steady-state kernels <~20 µs); empirically they
  overflowed ONLY under NCU-delegated clocking, never under the host lock.

## Determinism classes = accuracy-gate denominators (v2.1 reference arm)

`determinism_class_report.py` classifies every registry metric from the
5-repeat reference arm: **deterministic** (CV=0; all T1 counts — also
protocol-INVARIANT) / **quasi_stable** (≤0.5%) / **clock_noise** (≤2%; cycles
median 0.89%) / **noisy** (>2%; SOL pcts, occupancy, and BOTH cache hit rates
by max — L2 max 2.94%, L1tex max 6.31%, driven by the cp.async
composition-variance cells; medians 0.75%/0.13%). Caveats the report itself
stamps: floors are right-censored at the registry admission gate (CV 0.05 /
stddev 0.03); count metrics use range-based `relative_tolerance`
(conservative); hit-rate stddevs are fraction-scale at the aggregator and
percent-scale in the matrix summary (scale bug fixed 2026-07-06 — CVs were
once understated 100×).

## Known label-instability findings (typed, not noise)

- `flash_attn_bwd_bf16_sm80`: BIMODAL L2 residency (hit rate flips ~0.88↔~0.10
  across repeats under v2.1) — its L2 target is a distribution, not a scalar.
- cp.async HMMA GEMMs: run-to-run L1 hit/miss COMPOSITION variance (±1.5pt hit
  rate, ±2–3% miss sectors) under the fixed v2.1 ratio; requests stay exact.
- `gemm_imma_int8_sm80`: recurring single-repeat ~0.88-vs-0.95 layout-flip
  outlier in both protocol arms.

## Operational traps (each cost real time — do not rediscover)

1. **Silent NVBit injection failure**: the tracer .so links the pinned
   protobuf runtime; without `LD_LIBRARY_PATH=<repo>/.toolchain/pb/lib`,
   `CUDA_INJECTION64_PATH` dlopen fails SILENTLY (app runs untraced, zero
   stderr). `activation_trace_orchestrator._tracer_env()` exports + ldd-
   preflights this for every consumer; a missing NVBit banner = failure.
2. **Never `git commit` mid-sweep**: the driver stamps `source_commit` per
   repeat; a mid-sweep commit splits cell provenance and the aggregator
   fail-closes (`label_provenance_duplicate`).
3. **NVBit 1.8 driver window**: README pins driver ≤575.xx; host 610.43.02 is
   out-of-support (works, but re-run the M0 trace smoke on every driver bump).
4. **Protobuf version regimes**: C++ AND Python gencode require an EXACT
   gencode==runtime match (7.35 gencode on 7.34 runtime fails the version
   check). The Rust side (prost) is EXEMPT from the runtime check — prost
   implements the wire format itself; its rules are instead: regenerate
   `sim/trace/src/pb/dynamic_trace.rs` via `sim/trace/regen_pb.sh` (pinned
   protoc 34.1, committed pbgen/Cargo.lock) in the SAME commit as any .proto
   change (drift guard: `sim/trace/tests/proto_schema_pin.rs`).

## 5060 / GB202 replay

Follow `plans/host_state_and_migration_log.md` §checklist: pick the TARGET
card's own reference clocks from `SUPPORTED_CLOCKS` (never reuse 1350/10501),
verify locks via `clocks_event_reasons.active`, rebuild with the pinned 13.2
toolkit, then M0 smoke → pchase → matrix → determinism classes → freeze.
