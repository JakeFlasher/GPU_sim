# Uncensored NCU Determinism-Study Statistics — Fable-5 design (sm_86)

> Provenance: Fable 5 answer (2026-07-08) to `plans/sm86_determinism_study_stats_fable5_prompt.md`. Verbatim design portion extracted from `results.md` (thinking-block stripped). Referenced by `plans/sm86_comprehensive_live_capture_campaign_plan.md`.

---

Below is a complete design. I'll fix the architecture invariants first, because three of your four problems (censoring, scale-mixing, lockstep) are solved structurally, not statistically.

## 0. Design invariants (anti-censoring architecture)

**The accounting identity.** Every `(cell × config × metric × repeat)` row from the raw sidecars appears in the analysis table exactly once, either with a class contribution or a *typed* exclusion reason. Assert `collected = classified + excluded_typed` at build time. The admission gate is demoted from a filter to a *reporting flag*: recompute the historical gate verbatim per cell and store `would_have_been_admitted ∈ {true, false}` so you can (a) audit that your shared dispersion function reproduces the historical admission set bit-for-bit, and (b) quantify exactly what the censored study missed. This is the audit that proves the uncensored pass is actually uncensored.

**One dispersion module, two exports.** A single module (the registry's) exports `dispersion_gate(rule, values)` — the admission statistic, verbatim, used only for the flag above — and `sigma_rel(rule, values)` — a σ-equivalent transform used for classification and floors. `sigma_rel` maps all three rules onto one scale: for `coefficient_of_variation` and `stddev_abs` it is \(s/|\bar{x}|\) (after unit assertion); for `relative_tolerance` (range-based) it is \((\max-\min)/(d_2(N)\,|\bar{x}|\)), where \(d_2(N)\) is the expected range of \(N\) normal samples in σ units (\(d_2\) = 2.326, 3.078, 3.735, 4.086 at N = 5, 10, 20, 30). Without the \(d_2\) normalization, a fixed threshold on a range statistic silently tightens as N grows — a second-order version of your problem 2. Add two contract tests: (i) golden-vector equality between `dispersion_gate` and the historical admission decisions; (ii) a scale-invariance property test — multiplying inputs by 100 must leave relative statistics unchanged — which is precisely the regression test for the 100× percent/fraction bug.

**Run-order design.** Collect repeats as R shuffled sweeps over all cells (block-randomized), not back-to-back, one fresh process per profile invocation. Back-to-back repeats measure best-case short-term repeatability; shuffled sweeps make repeats exchangeable with respect to drift (thermal, driver state), which is the dispersion a simulator's evaluator actually faces. Fresh processes matter for problem 4 specifically: VA-layout/L2-set-hash regimes resample per process; same-process repeats freeze one regime and hide the bimodality you're trying to detect. Record per-repeat sidecars: sweep index, timestamp, SM/mem clock (NVML poll), temperature, and NCU replay pass count — these are your attribution covariates later.

## 1. Repeat count N and stopping rule

**Reject adaptive "stop-when-CV-stable."** Optional stopping keyed on the statistic under study is a censoring mechanism: sequences that happen to look quiet early stop early, so the floor estimator is biased *down* — the exact failure mode this study exists to eliminate. Any stopping rule conditioned on observed dispersion re-creates problem 1 in sequential form.

**Use fixed tiers with pre-registered escalation** (escalation only *adds* repeats, all data analyzed jointly, so it is information-monotone and doesn't bias floors):

- **Tier 1:** N = 20 for the reference config (`clock=none` × `cache=all`), N = 10 for the five non-reference configs. The reference gets more because it alone feeds denominators. Cost: 240 × (20 + 5×10) ≈ 16,800 profiles, plus escalations.
- **Tier 2:** escalate the *reference* measurement of a cell to N = 30 total when any pre-registered trigger fires: (i) modality flag or single-point outlier flag (§3); (ii) `sigma_rel` within a factor of 2 of a class boundary; (iii) any out-of-range repeat; (iv) discrete metric with ≥2 distinct values. Expect ~5–15% of cells to escalate.

**Why these N are enough — and what they can't do.** Per-cell σ estimates are individually mediocre: the one-sided 95% upper confidence factor on σ, \(\sqrt{(N-1)/\chi^2_{0.05,N-1}}\), is 2.37 at N=5, 1.65 at N=10, 1.37 at N=20, 1.28 at N=30. So a *per-cell* floor at N=10 is uncertain by ±65% one-sided — useless as a lone denominator. The design is rescued by pooling: each per-metric floor is a quantile over ~240 per-cell estimates, and its uncertainty is set by the cell count, not per-cell N (bootstrap over cells, §5). Correct the small-N bias of \(s\) with \(c_4(N)\) (0.973, 0.987, 0.991 at N = 10, 20, 30) in the floor layer only — never inside the lockstep gate statistic.

**Honest bounds at these N:** a cell that never varies in 20 repeats certifies, by the rule of three, only that its per-repeat flip probability is ≤ 3/20 = 15% at 95% confidence (≤10% at N=30). The metric-level claim is much stronger if *no* cell ever flips: ≤ 3/4800 ≈ 0.06% under cross-cell exchangeability — report both, since exchangeability of flip probability across cells is itself an assumption.

## 2. Class taxonomy and per-rule thresholds

Classes are assigned per `(cell × metric × config)` in strict priority order — validity before shape before dispersion — because each earlier condition invalidates the statistic used by the later one:

| Priority | Class | Condition (on in-range repeats unless noted) |
|---|---|---|
| 1 | `out_of_range` / `vendor_defect` | any repeat outside registry `[valid_min, valid_max]`; typed sub-reason (e.g. `pct_overflow_delegated_clock`) |
| 2 | `insufficient_evidence` | in-range repeats < 5 after exclusions |
| 3 | `multimodal` | certified by §3 detector (tier 2); `suspected_multimodal` if flagged but uncertified |
| 4 | `deterministic` | all repeats exactly equal at emitted precision (range = 0; for counts, bit-identical) |
| 5 | `quasi_stable` | \(0 < \sigma_{rel} \le 0.005\) |
| 6 | `clock_noise` | \(0.005 < \sigma_{rel} \le 0.02\) |
| 7 | `noisy` | \(0.02 < \sigma_{rel} \le 0.10\) |
| 8 | `unstable` | \(\sigma_{rel} > 0.10\) (new — the uncensored tail needs a home; report the empirical P95 and max per metric) |

Thresholds per `repeatability_rule`, all equivalent to the same underlying \(\sigma_{rel}\) bands {0.5%, 2%, 10%}:

| Rule | Classification statistic | quasi_stable | clock_noise | unstable > |
|---|---|---|---|---|
| `coefficient_of_variation` | \(s/|\bar x|\) | ≤ 0.5% | ≤ 2% | 10% |
| `stddev_abs` (mean-normalized) | \(s_{abs}/|\bar x|\), unit-asserted | ≤ 0.5% | ≤ 2% | 10% |
| `relative_tolerance` (raw range), N=10 | \((\max-\min)/|\bar x|\) | ≤ 1.5% | ≤ 6.2% | 30.8% |
| `relative_tolerance` (raw range), N=20 | " | ≤ 1.9% | ≤ 7.5% | 37.4% |
| `relative_tolerance` (raw range), N=30 | " | ≤ 2.0% | ≤ 8.2% | 40.9% |

(The range rows are just the σ-bands multiplied by \(d_2(N)\); implement via `sigma_rel` so only one threshold set exists in code.)

**Near-zero means:** if \(|\bar x| < \varepsilon_m\) (registry-declared: e.g. 1 count, 0.01 pct-point), relative statistics are undefined-in-practice; classify on absolute spread against a per-metric absolute band, mirroring whatever the gate does — this must live in the shared module too. All-zero cells are `deterministic`.

**Label uncertainty:** attach a CI to \(\sigma_{rel}\) (χ² for s-based rules, bootstrap for range). A cell whose CI straddles a boundary carries `uncertain_boundary`; wherever downstream code needs one label, take the noisier class. That conservatism is directionally correct for denominators.

## 3. Small-N modality detector

Two-stage, matching the tiers: a cheap high-recall **flag** at N=10–20, a certifying **test** at N=30. Never certify from tier 1.

**Discrete branch** (integer counts, or quantized at resolution \(q_m\)): multimodal-flag if ≥2 distinct values each with support ≥2 and pairwise separation exceeding the quasi_stable band. Certify at tier 2 with support ≥3 per value. (Dip tests misbehave under heavy ties; don't use them here. Two distinct instruction counts = two code paths, and this branch catches it exactly.)

**Continuous branch — largest-gap rule (tier-1 flag).** Sort values, find the largest inter-order-statistic gap \(g^*\), split there into L and R. Flag iff all of: support ≥ 2 on each side; separation \(g^* \ge 4 \times \max(s_L, s_R, q_m)\) (the \(q_m\) floor prevents 0/0 when regimes are internally bit-identical, which they often are); relative materiality \(g^*/|\bar x| \ge 0.5\%\) (sub-band bimodality is operationally quasi_stable — note it in a flag column, don't class on it); gap dominance \(g^* \ge 2.5 \times\) the second-largest gap (rejects smooth heavy tails).

**Single-outlier discipline.** If the best split leaves one point alone, that is *never* `multimodal` — it's `outlier_flag` → escalate. The point stays in the dispersion computation (uncensored ethos: excursions are data; only OOR values are excluded). Your example makes the case: `[0.877, 0.869, 0.851, 0.987, 0.878]` has gaps {0.018, 0.008, 0.001, 0.109} — \(g^*\) is 6× dominant but the 0.987 side has support 1, so at N=5 this is *formally indistinguishable from a transient*. It escalates; at N=30 with fresh processes the upper regime recurs (VA-layout regimes typically have 20–50% prevalence) and certifies. Note also: this cell's CV is 6.05% — it tripped your 5% admission gate, which is why the old pipeline never saw the bimodality at all. It's problems 1 and 4 in one cell.

**Tier-2 certification (N=30):** require the gap rule with support ≥3 per regime *and* one of: Hartigan's dip test p < 0.05, or ΔBIC > 6 for a 2- vs 1-component Gaussian mixture with variance floored at \(q_m^2\) (the floor prevents degenerate zero-variance spikes). Correct dip p-values with Benjamini–Hochberg at FDR 5% within each metric family — across ~79k cell-tests, even 1% false positives is 800 bogus labels, so flags are free but certification is gated. If the gap rule passes and both tests fail: `suspected_multimodal`, and denominators fall back to total-σ (conservative).

**Regime-vs-artifact check:** test regime membership against sweep index/timestamp (runs test or point-biserial). Temporally clustered regimes are drift or re-lock transients (especially under base/boost), not layout regimes — relabel toward `clock`-attributed noise and say so.

**What N cannot certify:** dip has essentially no power below N≈15 at any realistic separation. The probability of even *seeing* a regime with support ≥2 at N=10 is 26% at 10% prevalence and 85% at 30%; support ≥3 at N=30 is 59% at 10% prevalence, 96% at 20%. State it plainly: absence of a `multimodal` label certifies nothing below ~15–20% prevalence, and every floor is a lower bound with respect to regimes rarer than roughly 1/N.

## 4. Out-of-range / vendor-defect treatment

**Retain, type, never repair.** Keep every OOR row with `validity = out_of_range_high`, reason `ncu_pct_overflow`, plus config, replay pass count, and clock telemetry. Do not clamp to 100 (fabricates a value the instrument refused to stand behind), do not impute, do not delete (censoring). The emitted 100%+ number is not a measurement of the quantity — it is a basis mismatch, consistent with ratio numerator and denominator counters being collected in different replay passes at different clocks during base/boost re-lock transients; your telemetry sidecar lets you report this as attribution rather than speculation.

**Why survival-analysis machinery does *not* apply:** Kaplan–Meier-style right-censoring requires the censoring point to be a true lower bound on the actual value. Here the true occupancy of that repeat was presumably ≤100% and simply *unknown* — the reading isn't "at least 103%", it's "invalid." So the honest treatment is exclusion-with-typed-reason plus rate reporting, not censored-likelihood estimation.

**Cell and floor consequences.** The `(cell, metric, config)` gets the `vendor_defect` label for that config; dispersion is computed on the in-range subset if ≥5 repeats survive, else `insufficient_evidence`. For any `(metric, config)` with OOR rate \(r > 0\), the floor is reported as a **lower bound**: "≥ \(\hat\sigma_{rel}\) on in-range repeats; upper tail hard-censored by instrument fault at rate r." Report r per metric × config. The structural good news you've already observed — r = 0 under the host-locked reference — means the *denominator-feeding* floors are uncensored; say this loudly, and file the defect itself as a cross-protocol finding (§6): the metric cannot be certified under delegated-clock protocols.

## 5. Denominator derivation (no re-censoring)

**Pooled floors with per-cell override, from the reference config only.**

1. Per cell: \(\hat\sigma_{rel}\) via `sigma_rel`, \(c_4\)-corrected, on in-range repeats, unimodal cells.
2. Per metric: **floor\(_m\) = the 90% upper bootstrap bound (resampling cells, the exchangeable unit) of the P90 across cells of \(\hat\sigma_{rel}\)**, computed over *all* classified cells including `noisy` and `unstable` — excluding only typed-OOR values. Excluding noisy cells from the floor would be re-censoring; that rule is the whole point.
3. Denominator: \(\mathrm{denom}(m,c) = \max\big(q_m,\ \mathrm{floor}_m,\ \hat\sigma^{UB}_{rel}(m,c)\ \text{if cell } c \text{ is above P90}\big)\), where the per-cell override uses that cell's own χ² upper bound. Rationale: per-cell denominators everywhere would be a lottery at N=20 (±37%); pooled-only denominators punish simulators on the ≤10% of cells where hardware itself is noisier than the floor. P90-plus-override is the compromise; note that per-cell estimator noise inflates the cross-cell spread and hence P90 slightly — conservative in the safe direction.
4. Gate criterion (policy layer, separate from floors): pass iff \(|sim - ref| \le z \cdot \mathrm{denom} \cdot |ref|\) with z = 2 fixed by policy.

**Per-class rules:** `deterministic` metrics get an absolute denominator \(q_m\) (1 count / 1 LSB) and an exact-match gate — never a zero denominator. `multimodal` cells gate against the *nearest regime mean* with the pooled *within-regime* σ as denominator; a simulator predicting the grand mean must fail, because that is a value the hardware never produces. Regime *weights* are never gated: a 6/30 split has a 95% CI of roughly ±14 percentage points — unquantifiable at this N; report weights with binomial CIs only. `vendor_defect` metrics get reference-derived denominators plus an annotation; against base/boost-collected data they support only a one-sided, provisional gate (a lower-bound floor can flag gross failure but cannot certify a pass). `insufficient_evidence`: no gate, listed in the ledger. Pool within metric, not across metrics; a class-level pooled floor is a fallback only when a metric has <30 valid cells.

**Residual censoring statement (publish verbatim with the floors):** floors remain right-censored by (i) instrument faults under delegated-clock configs at reported rate r; (ii) cells with <5 in-range repeats; (iii) finite-N rare-event truncation — an excursion with per-repeat probability 5% is missed entirely in 20 repeats with probability \(0.95^{20} \approx 36\%\)… and observed at all with probability only 64%, so floors are lower bounds with respect to events rarer than ~1/N; (iv) session scope — these are within-session, single-card, single-driver floors. Run a small replication arm (reference config, 2–3 extra sweeps on separate days, ~500–700 profiles) and publish within-session vs across-session floors separately; flag any metric where the ratio exceeds 2, because a simulator validated against archived data faces the across-session floor.

## 6. Cross-protocol sensitivity

**The two axes are semantically different — never pool them.** The clock axis (none/base/boost) varies the *instrument*; sensitivity there is measurement artifact. The cache axis (all/none) varies the *system state under measurement*; sensitivity there is genuine microarchitectural response, and its *absence* is what's suspicious (an L2 hit rate invariant to cache flushing is a red flag, as is a clock-sensitive instruction count — use these expected patterns as validity checks on the harness itself).

Per metric and config: paired per-cell shifts \(\Delta_{rel}(m,c) = (\bar x_{config} - \bar x_{ref})/|\bar x_{ref}|\), summarized by the cross-cell median with a cell-bootstrap CI. Classify with **equivalence-test logic**, not significance tests — a failed t-test is absence of evidence, not invariance:

- `protocol_invariant(axis)`: CI entirely inside ±band, band = \(\max(0.5\%,\ 2\times\mathrm{floor}_m)\);
- `protocol_sensitive(axis)`: CI entirely outside the band (report signed magnitude);
- `undetermined(axis)`: otherwise — expected to be common at N=10 for noisy metrics; say so rather than forcing a label.

The final published label is a tuple: *(determinism class @ reference) × (clock sensitivity) × (cache sensitivity) × (defect annotation)* — sensitivity is an attribute orthogonal to the reference-config class, not a class override. Add a class-migration table (e.g. `quasi_stable`@reference → `noisy`@boost) and the OOR incidence per config, which is where the vendor-defect finding lives as a protocol result. Apply BH-FDR within metric families to sensitivity *claims*; keep the primary framing estimation-with-CIs.

## Claims small N cannot support — the explicit flag list

- **Unimodality is never certified.** N=20 with no flag means "no regime with ≳20% prevalence and ≳4σ separation detected," nothing more.
- **Determinism is a bounded claim:** zero variation in N repeats bounds per-repeat variation probability at 3/N (15% at N=20 per cell); only the pooled all-cells statement is strong.
- **Regime weights are not estimable** to useful precision below N≈100; report with binomial CIs, never gate.
- **Per-cell tail quantiles are meaningless** at N≤30; only pooled cross-cell quantiles are reported.
- **Boundary labels are soft:** cells whose \(\sigma_{rel}\) CI straddles a threshold are formally ambiguous; the conservative-label rule is a policy choice, disclosed as such.
- **Delegated-clock floors are lower bounds, period** — no analysis at any N can recover the OOR-censored tail, because the instrument, not the sample size, is the censor.

One implementation warning that cuts across everything: the lockstep requirement means the *study* must be able to reproduce every historical admission decision from the raw sidecars using the imported gate function before you trust anything downstream. If that reconciliation fails on even one cell, you have a second scale bug, and the reconciliation — not the taxonomy — is where it will surface.
