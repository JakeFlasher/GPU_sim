# Phase-1 Full-Tier Results (2026-07-06, RTX 3090 Ti, host-locked 1350 MHz, driver 610.43.02)

Scale-up of `plans/phase1_noise_floor_first_results.md` from 2 kernels to the full
portable_lcd tier, plus the 1D latency card fill and 1E control-bits roster sweep.
Evidence (machine-local annex): `data/annex/repeatability_matrix_full_20260706/`
(`matrix_summary.json`, 3,333 rows; `determinism_classes.json`),
`data/annex/control_bits_sm86_20260706/`, `data/pchase/sm_86_static_card.json` (tracked).

## 1C — 510-capture repeatability matrix, 17 cells x 6 protocols x 5 repeats

Roster: all 15 portable_lcd kernels + both unique `ac4_mixed_long` cases
(`dram_streaming_dot_product`, `global_atomic_histogram` — the only atom/red-pipe
exercisers), each at its timing shape, with the FULL registry metric set
(53 sm_86-eligible unit metrics + the driver's always-collected six).
499/510 captures admitted; 11 rejected by the parser's fail-closed range gate
(finding 4 below). Zero infrastructure failures.

### Per-metric determinism classes (reference protocol: host -lgc 1350 + --clock-control none + --cache-control all)

Classifier: `tools/local_labels/determinism_class_report.py`. CV semantics are
repeatability_rule-aware (coefficient_of_variation / relative_tolerance dispersions
are already relative; stddev_abs is normalized by the cell mean).

> **REVIEW-GATE CORRECTION (2026-07-06 late night).** The first published
> version of this table understated BOTH cache-hit-rate CVs by exactly 100x
> (percent-scale mean paired with fraction-scale stddev in the summarizer;
> caught by the pre-Phase-2 review gate, verified against raw sidecars,
> fixed in `repeatability_matrix_runner.summarize`). The table and gate
> consequences below are the CORRECTED (v2, then v2.1-refreshed) numbers.
> Additional scope caveat: rows reflect ADMITTED evidence only, so floors
> above the registry admission gate (CV/rel-tol 0.05, stddev 0.03) are
> right-censored — `noisy` is a (2%, admission-gate] band; count metrics use
> the range-based `relative_tolerance` statistic (conservative vs CV).

| class | n (v2.1 ref) | definition (max CV over cells) | members (abridged) |
|---|---|---|---|
| deterministic | 21 | CV == 0 in every cell | ALL l1tex request/sector counts (ld/st/atom/red/local/surface/texture pipes), `smsp__inst_executed.sum`, exact-or-zero stall ratios |
| quasi_stable | 3 | CV <= 0.5% | dram clock rate, `stalled_wait`, xbar write sectors |
| clock_noise | 7 | CV <= 2% | `gpc__cycles_elapsed.max` (median 0.89%, max 1.93%), `gpu__time_duration.sum`, `dram__sectors_read.sum`, `stalled_lg_throttle` |
| noisy | 24 | CV > 2% somewhere | BOTH cache hit rates by max (L2 max 2.94% / median 0.75%; L1tex max 6.31% / median 0.13% — the maxima are exactly the cp.async GEMMs + flash_attn_fwd composition-variance cells; all other cells are sub-1%), all SOL `pct_of_peak_sustained_*`, occupancy, `dram__sectors_write.sum`, short-kernel stall-ratio tails |

Gate consequences for M2/M3 accuracy denominators (corrected):
- Count-class targets (T1) gate at ~0 tolerance — the hardware counts them exactly.
- Cache hit-rate targets gate PER-CELL: <=0.3 percentage points absolute for
  non-async kernels (median stddev ~0.1pt), but ~2-3pt for cp.async
  streaming kernels whose hit/miss composition genuinely varies run-to-run
  — still 2-20x tighter than the frozen v5 head's 6.4% geomean error.
- Cycles/duration gate against a ~1% floor (D5's <=10% target is 10x the floor).
- SOL/occupancy/stall-tail targets gate against per-metric floors of 2-5%,
  right-censored at the 5% admission gate.

### Cross-protocol sensitivity (median relative shift vs reference, over 17 cells)

NCU-boost delegation shifts (vs host-locked 1350):
`gpc__cycles_elapsed.avg.per_second` +39.7% (that IS the clock), cycles +15.7%,
`sm__cycles_active` +14.3%, duration -17.0%, `stalled_long_scoreboard` +22.7%,
`stalled_imc_miss` +13.4%, `dram__bytes.sum.per_second` +23.0%,
`compute_memory_request_throughput` +15.5%, `sm__memory_throughput` -12.3%.
Cache-none (warm->cold) shifts: `stalled_no_instruction` -21.9% (icache warmth),
L2 hit rate +1.0pt median. Counts: 0.0 shift under every protocol — counts are
protocol-INVARIANT, not merely repeatable.

### Finding 4 (new): NCU occupancy pct overflows >100% under NCU-delegated clocks

All 11 capture rejects are one phenomenon: `sm__warps_active.avg.pct_of_peak_sustained_active`
= 105-125% on the two shortest elementwise kernels (silu, swiglu) under
{base, boost} x cache-none — and NEVER under host-locked reference (which stays
in-range and merely `noisy`). Same class as the sm_120 `lts` pct overcount (S7).
Mechanism attribution (web-verified): NVIDIA's Kernel Profiling Guide documents
that pct-of-SUSTAINED-peak metrics "can occasionally exceed 100%" and that ratio
metrics go out of range on multi-pass replay when the kernel does not reach
steady state (generally <~20 us) — numerator/denominator land in different
replay passes. Our EMPIRICAL addition on top of the documented mechanism: on
these kernels the overflow manifests only under NCU-delegated clock control
(base/boost re-lock transients between passes) and never under the externally
locked host reference. Third pillar for the pinned protocol: percentage metrics
stayed bounded, across the full roster, only under host-locked clocks.

## 1D — latency card filled from locked-clock pchase (sm_86)

`tools/pchase/run_pchase_card.py --arch sm_86` re-run under the lock on driver
610.43.02; transcribed into `arch_profiles/cards/sm_86.json#latencies_cycles` by
`tools/pchase/transcribe_latencies_to_arch_card.py`; lockstep tests pin card ==
pchase evidence (`TestSm86LatenciesLockstep`, 4 tests).

| latency (cycles/dependent load) | legacy 595.71.05, unlocked (boost ~2040) | fresh 610.43.02, locked 1350 | reading |
|---|---|---|---|
| shared | 23.0016 | 23.0016 | bit-identical — core-domain, clock-invariant |
| L1 hit | 33.0016 | 33.0016 | bit-identical — core-domain, clock-invariant |
| L2 hit | 250.01 | 256.09 | +2.4% — uncore, core:mem-ratio-dependent |
| DRAM | 527.03 | 436.40 | -17.2% — uncore; SM-cycle count shrinks at lower core clock over fixed-time DRAM access |

This is the pchase analog of the 13-19% cycles cross-protocol spread: core-domain
latencies are exact cycle constants; uncore latencies must be calibrated under the
SAME pinned protocol as the labels the sim is judged on. The locked-1350 values are
the sim/calib inputs. (Rule 1 note: pchase constants are clock64-derived microbench
inputs, never NCU counter values.) Literature cross-check (web-verified): our
L1-hit 33 and shared 23 cycles match Abdelkhalik et al., "Demystifying the Nvidia
Ampere Architecture" (HPEC'22, A100: L1 33, shared-load 23) EXACTLY; L2/DRAM
exceed A100's published 200/~290 but GA102 has no published cycle-accurate
numbers and Turing global was 434 — plausible. (Citation note: the Ampere
dissection paper is Abdelkhalik et al., NOT Jia et al. — Jia/Citadel published
the Volta and Turing dissections.)

Typed research_gaps kept (probe-design limits, not DVFS noise): MSHR count (no
saturation knee through k=32 — sim fidelity knob), L1 associativity (conflict grid
flat at 33c through 128KB strides — index hashing suspected), replacement policy.
Known detector artifact: the L2 capacity knee reads 3MiB effective under lock
(5MiB legacy) because the L2:DRAM latency contrast compresses from 2.1x to 1.7x,
weakening the step detector; ArchProfile geometry stays vendor 6MiB.

## Addendum — protocol v2.1 (memory clock locked)

After this report's campaigns, the operator locked the memory clock
(`sudo nvidia-smi -lmc 10501`), upgrading the pinned protocol to **v2.1**
(core 1350 + mem 10501, fixed ratio). Consequences, measured same evening:

- pchase v2.1: carveout-100 resolved at 32768 (mem-lock removed the L2-plateau
  wobble that broke its confidence gate under the core-only lock); L2 hit
  243.62 / DRAM 422.65 cycles (transcribed to the ArchProfile — these
  supersede the v2 values); L1 33.0016 / shared 23.0016 bit-identical across
  ALL THREE clock regimes measured today.
- v2→v2.1 label spot-check (reference config): cycles −2.4/−3.0%, L2 hit rate
  +3pt absolute on rmsnorm, uncore SOL metrics 2–9% — systematic protocol bias
  well above the 1% noise floor. T1 counts: unchanged, exact.
- Disposition: this report's determinism CLASSES and cross-protocol
  sensitivity tables stand (protocol-internal / protocol-relative quantities);
  reference-config absolute means were re-captured under v2.1 into
  `data/annex/repeatability_matrix_v21_ref_20260706/` (85/85 captures, zero
  failures, zero pct-overflow rejects) and THOSE are the M2/M3 gate
  denominators. v2 and v2.1 timing labels must never be mixed in one
  accuracy claim. Host-state history: `plans/host_state_and_migration_log.md`.
- v2.1 floors (17 cells x 5 repeats): cycles maxCV 1.69% / median 0.86%
  (vs v2 1.87%/0.99%); L2 hit rate maxCV 0.029% (vs 0.053%). Class structure
  stable (21 deterministic / 6 quasi_stable / 8 clock_noise / 20 noisy) with
  one REAL microarchitectural movement: both HMMA GEMMs (the two cp.async
  streaming kernels) show run-to-run L1 hit/miss COMPOSITION variance under
  v2.1 (`l1tex..ld_hit_rate.pct` ±1.5pt around 16.3%; xbar miss sectors
  ±2-3%) that was exactly deterministic under v2 — the locked full-rate mem
  clock shifts LDGSTS-vs-demand arrival races. L1 requests stay exact; only
  the hit/miss split moves. Gate consequence: L1 sector-hit-rate tolerance
  for async-copy kernels is ~2-3% relative, not 0; all other kernels' L1 hit
  rates remain deterministic-class.

## 1E — control-bits sidecars, full roster

`emit_control_bits_sidecar.py` over all 16 binaries: **7,336/7,336 instructions
decoded, 0 unknown_encoding** (largest: ac4 1,800 across 5 kernels; conv2d 1,088;
flash_attn_bwd 1,376). Sidecars in `data/annex/control_bits_sm86_20260706/`.
The sim/sm dependence-stall input surface is complete for the sm_86 tier.

## M1 — traces captured + label bundle FROZEN (v2.1)

- **NVBit traces: 17/17 roster cells captured and contract-valid** (AC4
  per-instruction contract via `validate_trace_dir`), zero record drops.
  Volumes: 15 standard cells 167 MB total; ac4 dot-product 0.9 GB (78M
  records); ac4 histogram 3.5 GB (411M records). Runner:
  `tools/local_labels/roster_trace_capture.py` — exports the pinned protobuf
  runtime on LD_LIBRARY_PATH and preflights `ldd tracer_tool.so` fail-closed
  (first attempt failed 17/17 SILENTLY: CUDA_INJECTION64_PATH dlopen swallows
  unresolved-dep errors and runs the app untraced).
- **Frozen bundle: `eval/frozen_labels/sm86_m1_bundle_v21.json`** (tracked) —
  15 admitted cells x 5 v2.1-protocol label sidecars (sha256-pinned) +
  headline admitted means + trace identity (pb/stats sha256, record counts,
  binary_sha256) + protocol/driver/commit stamps. 2 typed exclusions kept
  honest by the aggregator's 0.03 L2-stddev gate:
  - `flash_attn_bwd_bf16_sm80` — **BIMODAL L2 residency**: per-repeat L2 hit
    rate [0.12, 0.82, 0.89, 0.09, 0.87] under v2.1 (one high outlier already
    under v2). Two stable regimes, not noise — likely run-to-run VA-layout /
    L2-set-hash interaction in the attention-bwd workspace. Simulating this
    cell's L2 means predicting a DISTRIBUTION, not a scalar; deferred with
    typed reason.
  - `gemm_imma_int8_sm80` — recurring single-repeat outlier (~0.88 vs ~0.95
    in BOTH protocol arms; stddev 0.0325 vs 0.03 gate) — same layout-flip
    class, marginal magnitude.
- Provenance lesson (cost one recapture): committing mid-sweep splits
  `source_commit` across a cell's manifests and the driver/aggregator
  fail-close on it (`label_provenance_duplicate`). Never commit while a
  capture campaign is running.

## Phase-1 exit status toward M1

- 1A capture protocol v2: DONE (commit 255c0017)
- 1B registry 61 metrics: DONE (255c0017)
- 1C noise floors: DONE at full tier (this report) — per-metric determinism
  classes published as accuracy-gate denominators
- 1D latency card: DONE for the 4 core latencies + typed gaps (this report)
- 1E control bits: DONE roster-wide (this report)
- Remaining for M1 label freeze: NVBit trace capture per roster cell (the sim's
  INPUT side) + frozen label bundle with the v2 protocol sidecars. Then Phase 2
  (Rust sim core v0) consumes traces + control bits + latency card, gated by the
  determinism-class denominators above.
