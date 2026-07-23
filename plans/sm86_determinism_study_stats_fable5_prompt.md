# Fable-5 design prompt #3 — Uncensored NCU determinism-study statistics

> Copy everything below the line and send it to Fable 5. Self-contained. MEDIUM priority.

---

You are a measurement-statistics and reproducibility expert. Design the **statistical methodology for an uncensored NCU counter-determinism study** on an **RTX 3090 Ti (GA102, sm_86)**. The output is (a) per-metric noise floors that become a simulator's accuracy-gate DENOMINATORS and (b) a standalone study (the missing GPU analog of the Weaver/McKee CPU hardware-counter determinism work). I need a concrete statistical design — repeat count, stopping rule, class taxonomy with thresholds, modality detection, and out-of-range treatment — not a survey.

## The measurement

A cross-protocol repeatability matrix captures NCU metrics per cell over: **{clock: none / base / boost} × {cache: all / none} × N repeats**, under a pinned protocol (host clock-locked 1350/10501 MHz; the reference config is `clock=none` (host-locked) × `cache=all`). ~55 sm_86-available metrics per cell across ~229–250 cells. Metrics span classes: deterministic counts (instructions, requests, sectors), cache/traffic rates, cycles/duration, occupancy, SOL%/throughput, and warp-stall families.

## The problems to fix (why the current classifier is unsound)

1. **Right-censoring.** The current classifier reads only ADMITTED evidence (cells that passed an admission gate: CV/rel-tol ≤ 0.05, stddev ≤ 0.03). Cells that trip the gate are dropped BEFORE the study sees them, so the top of the dispersion tail is truncated and the `noisy` class is a censored `(2%, gate]` band. The raw per-repeat sidecars persist on disk EVEN for excluded cells, so an uncensored pass is possible — but it needs a sound statistical treatment.
2. **Multiple dispersion rules.** Metrics carry different `repeatability_rule`s: `coefficient_of_variation`, `relative_tolerance`, and `stddev_abs` (normalized by the cell mean). The study must classify consistently across these without mixing scales (a prior bug paired a percent-scale mean with a fraction-scale stddev and understated cache-rate CVs by 100×).
3. **Out-of-range / vendor-defect values.** Percentage metrics (occupancy `pct_of_peak_sustained`, `lts__t_sector_hit_rate.pct`) occasionally EXCEED 100% — empirically ONLY under NCU-delegated clock control (base/boost re-lock transients between multi-pass replay), never under the host-locked reference. NCU refuses to emit an in-range value for these, so they are hard-censored at the very top.
4. **Genuine multimodality vs jitter.** Some cells are bimodal, not noisy: e.g. an attention-backward L2 hit rate reads `[0.877, 0.869, 0.851, 0.987, 0.878]` across 5 repeats — two stable VA-layout/L2-set-hash regimes, not clock jitter. A scalar CV collapses this to "noisy" and hides that predicting it means predicting a DISTRIBUTION, not a scalar.

## What I need (concrete)

1. **Repeat count N + stopping rule.** How many repeats per (cell, config) to (a) estimate a stable per-metric noise floor and (b) have any power to detect bimodality — honestly bounding what small N can and cannot certify (a sequential/stop-when-stable rule vs a fixed N; the cost is ~55 metrics × 6 configs × N × ~240 cells).
2. **The full class taxonomy with thresholds.** Extend {`deterministic` (CV=0), `quasi_stable` (≤0.5%), `clock_noise` (≤2%), `noisy` (>2%)} with a `vendor_defect`/`out_of_range` class (values > registry `valid_max`) and a `multimodal` class — giving exact dispersion thresholds per `repeatability_rule` type (`coefficient_of_variation` / `relative_tolerance` / `stddev_abs`), and a rule to keep the study's dispersion statistic in LOCKSTEP with the label-admission statistic (import the same dispersion function, never re-derive).
3. **A modality detector appropriate for SMALL N** (e.g. N=5–20). Dip test? gap statistic? a robust 2-regime separation with a minimum-separation + minimum-support rule? — and how to avoid false-positive bimodality from a single outlier, while catching the real 2-regime cells.
4. **Out-of-range treatment.** How to retain-but-exclude pct-overflow rows with a typed reason, how to bound the noise floor for a metric whose top is hard-censored (NCU won't emit the value), and how to report the floor honestly as a lower bound.
5. **Denominator derivation.** How the uncensored per-metric floors feed the simulator's accuracy-gate denominators WITHOUT re-censoring (e.g. per-metric-class floors, per-cell vs pooled, how to treat multimodal and out-of-range metrics as gate denominators), and the honest statement of residual hard-censoring.
6. **Cross-protocol sensitivity.** How to report each metric's shift under base/boost/cache-none vs the host-locked reference as part of the determinism class (protocol-sensitive vs protocol-invariant), given counts are protocol-invariant and uncore/timing metrics are not.

Deliver: the N + stopping rule (with an honest small-N caveat), the complete class taxonomy with per-rule thresholds, the small-N modality detector, the out-of-range/vendor-defect treatment, the denominator-derivation rule, and the cross-protocol sensitivity reporting. Flag any place where small-N repeats cannot support a claim the study would want to make.
