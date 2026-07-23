# Trace-Sampling Strategy — Fable-5 design (sm_86 / GA102 NVBit campaign)

Provenance: authored by Fable 5 (2026-07-08) in response to `plans/sm86_trace_sampling_strategy_fable5_prompt.md`,
which was seeded by a Codex-vetted ladder + the repo's sim-replay constraints. This is the DEC-1 detailed
design referenced by `plans/sm86_comprehensive_live_capture_campaign_plan.md` (AC-9, task13/15/16/17).
Repo alignment cross-checked against `.claude/knowledge/simulator/rust-des-patterns.md` (typed tagged-enum
guard fits the sim's fail-closed error taxonomy) and `.claude/knowledge/measurement/nvbit-injection-playbook.md`
(two-pass counting/dump capture + the silent-injection trap).

> **task13 ADJUDICATION (binding on task15/16/17):** this design was adjudicated
> against the sim's actual replay semantics in
> `plans/sm86_trace_sampling_strategy_fable5_design_adjudication.md` — central
> thesis ACCEPTED; SIX binding revisions (R1 completeness in the replay path, not
> only the reader; R2 invariant-owner split; R3 per-segment sub-trace replay +
> ghost-differencing; R4 explicit `segments[]{block_ids}`; R5 `wave_gate`≥16-wave
> unit fix; R6 whole-block retention forced by `sm_id`/`warpid_sm`). Read the
> adjudication's ACCEPTED thresholds — where it revises a value below, the
> adjudication wins.

---

## Executive summary — where this differs from the vetted ladder

The ladder's ORDER is correct, but its center of gravity is wrong. **Tier 2 (block-subset) is admissible for far fewer cells than assumed; tier 1 (compression) + tier 3 (reduced-shape) carry almost all the load.** The two named pathological cells illustrate it: the DRAM-streaming cell compresses ~10–20× and never needs sampling; the atomic-histogram cell (256 blocks ≈ 1.5–3 occupancy waves on 84 SMs) has **too few concurrency units to sample at all** — it fails structurally (the `wave_gate`) before it fails statistically. The cells that breach budget after the atomic/LDGSTS tracer change are disproportionately the same cells where subsetting is unsound (data-dependent addresses compress worst AND couple blocks through contention). Independent per-block sampling is replaced by **wave-segment cluster sampling**: the sim replays traced placement literally, so thinning blocks independently silently dilutes co-residency and corrupts L1/MSHR/L2 contention even when every retained record is bit-exact.

## 1. Ranked, refined ladder

1. **Tier 1 — lossless compression + retention tiers + a compressibility probe.** zstd `-19 --long=27`, per-block `.pb.zst` (preserves lazy per-block reads), optional per-cell trained dictionary for the ~14 MB block files. Contract hashes are over DECOMPRESSED bytes. Expected ratios: strided streaming 8–20×; data-dependent atomic/hash addresses 1.5–3×. Probe: compress 10 blocks, extrapolate; low compressibility is itself a red flag for subset sampling.
2. **Tier 1.5 (new) — admissibility gate.** A deterministic classifier routes each still-breaching cell to exactly one of: subset / reduced-shape / temporal / unavailable. No cell reaches tier 2 by default; it must pass the gates.
3. **Tier 2 — wave-segment stratified subset** (§2–4). Admissible only for large homogeneous grids (≥ ~12 reconstructable waves), atomic density below threshold, no grid-wide sync, bounded block skew. Targets: additive sector/traffic counts and hit-rate ratios. NEVER cycles as a primary output.
4. **Tier 3 — matched reduced-shape.** Strictly better than subset when (a) the metric is defined by cross-block coupling (atomics/histograms), (b) the grid has < ~8–12 waves, (c) cooperative launch / grid-wide sync, (d) block skew so heavy the certainty stratum alone breaches budget, (e) you need a TIMING label at all. Answers a DIFFERENT question (a different point in launch-config space — valid for calibration coverage, not full-shape prediction). Two variants, provenance-distinguished: **grid_stride_footprint_preserving** (preserves total footprint / DRAM working set; changes concurrency — preferred) vs **dataset_shrunk** (preserves per-block behavior; changes working-set-to-L2 ratio — FORBID when the full-shape working set exceeds the 6 MB L2 but the reduced one wouldn't, which silently inflates hit rates).
5. **Tier 4 — temporal windows**: narrowly, for MRC-stationary kernels and cache-rate summaries only (K=3 / ≤0.02 ledger). Window defined over launch-index order; replayed with the §3 warm-state protocol.
6. **Tier 5 — typed `unavailable`** (or `budget_limited_wide_ci` — a labeled wide interval refused for point scoring).

Classes that must NEVER be block-subset-sampled → route to tier 3 or 5: cooperative/grid-sync kernels (hard fail); global-atomic contention kernels above the atomic gate (histogram); persistent/grid-stride kernels with blocks ≈ resident slots; any grid with < 8 reconstructable waves; producer-consumer through global memory across blocks; cells where top-1% blocks carry ≥ ~15–20% of records and the certainty set breaks budget. Per-record reservoir AND deterministic every-k decimation are both inadmissible — it is the ordered replay semantics, not the randomness, that breaks.

## 2. Selector: wave-segment stratified sampling

The primary sampling unit (PSU) is a full concurrency slice, not a block. **Pseudo-wave reconstruction:** per SM, order blocks by start order; assign block `i` on SM `s` pseudo-wave index `g = floor(seq_s(i) / r_s)` where `r_s` = resident blocks per SM (occupancy calculator, cross-checked against `warpid_sm` slot disjointness — overlapping slot sets ⇒ provably non-co-resident; greedy slot-disjoint packing when ambiguous). A **global pseudo-wave** = union over SMs of generation `g`; a **segment** = `L` consecutive pseudo-waves (default `L=3`: 1 ghost warm-up + 2 measured). Segments are the PSUs.

**Two-pass capture.** Pass 1 = counting-only tracer (no addresses): per-block record counts, op-class mix (global ld/st, atomic, LDGSTS), active-lane averages, unique-sector sketch (HLL), within-block sector-reuse ratio (small Bloom). Pass 2 = re-run dumping only selected blocks; keep counting active for ALL blocks in pass 2 (exact same-run auxiliary totals for calibration + a free bias alarm). Because placement is nondeterministic across runs, wave membership is decided ONLINE in pass 2 (per-SM sequence counters); blockIdx-keyed force-includes are decided offline from pass 1.

**Gates** (run before selection; thresholds provenance-recorded): `coop_gate` (cooperative/grid-sync → forbid); `wave_gate` (reconstructable waves G < 8 → forbid — kills the histogram cell); `atomic_gate` (atomic/total > 0.5% AND top-256 sectors > 10% of atomic traffic → forbid for L2/cycles/atomic metrics); `skew_gate` (top-1% blocks ≥ 15% of records or Gini > 0.5 → heavy blocks become certainty includes; if the certainty set > 40% of the cell byte budget → tier 3); `homogeneity` (CV of per-SM block counts + per-wave record totals; change-point detection → phases as strata).

**Strata & allocation.** Stratify SEGMENTS by (launch-phase tercile) × (detected phase). Certainty strata: segment 0 (cold start), the final segment (tail/partial wave), and any segment containing a force-include block (boundary corners, epilogue, skew heavies — expanded to the enclosing segment). Allocation: Neyman `n_h ∝ N_h·S_h` (S_h = pilot std-dev of per-segment sector traffic), floor `n_h ≥ 2`; within stratum SRS (seeded) or systematic-with-random-start (recorded).

## 3. Full-shape estimator

Per sampled segment: replay the first wave as a GHOST (metrics muted, warms L2/DRAM row state), then measure the remaining `L-1` waves. Intra-wave co-residency / L1 / MSHR / intra-wave L2 sharing / within-wave DRAM contention are exact by construction; L2 history deeper than one wave and inter-segment effects remain approximate.

- **(a) Additive counts** (sectors, L1/L2 misses, DRAM sectors, instructions, records): Horvitz–Thompson `T̂ = Σ_{s∈C} t_s + Σ_h (N_h/n_h) Σ_{s∈S_h} t_s`, then CALIBRATE with the exact same-run per-segment record counts `x_s` (combined ratio estimator `T̂_cal = T̂ · X/X̂`). The record-count estimator is an exact self-check: if `X̂` deviates from `X` beyond its CI, the design is biased — ABORT, don't calibrate it away.
- **(b) Ratio metrics** (sector hit rates): ratio of CALIBRATED HT totals `R̂ = T̂_hits / T̂_sectors`. NEVER average per-block/per-segment rates — NCU's metric is a ratio of sums.
- **(c) Cycles/timing — NOT soundly recoverable.** The only surrogate is wave-serial composition; biases have indeterminate sign (ignores inter-wave overlap slack → over-estimate; shallow L2 warmth → over-estimate; misses cross-segment DRAM interference + grid phase alignment → under-estimate). Ship only as `cycles_estimate_experimental`, gated on §7; default the field to `not_recoverable`. Also not recoverable: atomic serialization with cross-wave hot-line residency; cross-wave reuse deeper than ghost depth; launch/DVFS transients; SOL-% timing family; tail beyond the force-included final segment.

**CI:** stratified cluster bootstrap (resample segments within strata, recompute the WHOLE pipeline incl. calibration, B=1000, 95% percentile). Require ≥ 8 sampled segments and `n_h ≥ 2`; below that, merge strata for variance or refuse.

## 4. Sample-size / stopping rule

Pilot = certainty segments + 2 per stratum; then sequential doubling (farthest-point infill in launch order) until converged, ≤ 3 doublings or budget:
- Rates: 95% CI half-width ≤ **0.02 absolute**.
- Counts: CI half-width ≤ **2–5% relative** (absolute thresholds on counts are meaningless — a likely typo in the seed prompt).
- Floors: ≥ 8 sampled segments; every active SM present with full resident cohort in every measured wave (else flag underfill); ≥ 1 segment per phase; calibration self-check `|X̂−X|/X` within CI.
- Warm-state check: recompute L2 hit-rate at ghost depth 0 vs 1 (and 2 on one segment); if still moving > 0.01 at 1→2, cross-wave reuse exceeds the protocol → escalate to tier 3/5 for L2 metrics.
- Not converged at budget → `budget_limited_wide_ci` (interval only) or strict `unavailable`. Never ship the point estimate.

## 5. Provenance sidecar (`trace_sampling.json`, closed schema)

One per cell, `additionalProperties:false`, hash-pinned in the manifest. Fields: `schema_version`; `cell_id`, `kernel{mangled_name,args_hash}`; `method ∈ {none, lossless_compressed_full, threadblock_stratified_subset, reduced_shape_matched_label, temporal_window}`; `reason ∈ {byte_budget, walltime_budget, gate_forced}`; `shape{original_grid, original_block, captured_grid, captured_block}`; `reduced_shape_variant ∈ {grid_stride_footprint_preserving, dataset_shrunk, null}`; `gates{coop, wave_count, atomic_density, hotline_share, skew_top1pct_share, verdict}`; `wave_model{reconstruction, resident_blocks_per_sm, n_waves, segment_len, ghost_depth}`; `strata[]{id, segment_ids, N_h, n_h, pi}`; `certainty_segments[]`, `force_include_blocks[]`; `selected_block_ids_sha256`, `per_sm_counts{orig,sampled}`; `rng{algo, seed}`; `estimator{mode ∈ {literal_partial_replay, weighted_full_shape_estimate, matched_reduced_shape}, form, aux, ci{method,B,alpha}, targets[], not_recoverable[], convergence{status ∈ {converged, budget_limited_wide_ci, unavailable}, per_target}}`; `stationarity{ledger, max_quantile_delta, verdict}`; `label_pairing ∈ {full_shape_label, matched_reduced_shape_label, no_comparable_label}`; `tracer{version, features{atomic_addr, ldgsts_addr}, stats_pass_version}`; `bytes{raw, compressed, codec, level, long, dict_id}`; `records{expected_full, captured}`; `hashes{per_block_raw_sha256_manifest, sidecar_sha256}`; `full_shape_equivalent`. Pin `tracer.version` + `stats_pass_version` + estimator version together (the atomic/LDGSTS address change alters record counts + strata stats; never mix pre/post-change stats within a cell).

## 6. Simulator guard (defense in depth)

1. **Reader completeness check (closes the current hole).** The contract doesn't require all blocks present — add it at the READER: the launch record carries `gridDim`; reader computes coverage; `< 100%` WITHOUT a sidecar → hard error `E_INCOMPLETE_TRACE`. Corollary: a subset NEVER rewrites `gridDim` (that would masquerade as reduced-shape and pass completeness); the reader cross-checks the sidecar's `shape.original_grid` against the launch record.
2. **Typed results** (idiomatic with the sim's tagged-enum/typed-error taxonomy):
```rust
enum TraceScope { FullShape, SampledSubset(SamplingManifest), ReducedShape(ShapeInfo) }
enum SimOutput {
    Exact { metrics: MetricVec },
    Estimate { point: MetricVec, ci: MetricIntervals, unrecoverable: Vec<MetricId>, provenance: Digest },
    ReducedExact { metrics: MetricVec, shape: ShapeInfo },
}
fn replay(t: Trace, mode: ReplayMode) -> Result<SimOutput, Refusal>; // default mode refuses sampled input
fn score_full_shape(o: &ExactOutput, l: &FullShapeLabel) -> Score;    // only compiles for Exact
fn score_estimator(o: &EstimateOutput, l: &FullShapeLabel, opt_in: EstimatorAck) -> EstimatorScore;
fn score_reduced(o: &ReducedExact, l: &ReducedShapeLabel) -> Score;   // shapes must match exactly
```
3. **Label-side stamping.** NCU labels carry `shape_scope`; the join requires scope equality or an explicit `EstimatorAck`. Invariants: `issued == records_seen` (over the subset), plus new `records_seen == sidecar.records.captured`, plus the calibration check `X̂ ≈ X` as a runtime assertion.

## 7. Fidelity validation

Golden set: 8–12 cells small enough for full capture, spanning classes (streaming, stencil boundary-heavy, tiled GEMM-like, transpose, reduction, small histogram, persistent grid-stride). Full replay → reference metrics; sampler at fractions {1/16, 1/8, 1/4} × ≥ 5 seeds. Acceptance (all required):
- **Accuracy:** p95 over seeds×cells of `|R̂ − R_full|` ≤ 0.02 for rates; ≤ 5% relative for counts.
- **CI calibration:** nominal 95% intervals cover the full value in ≥ 90% of trials.
- **End-to-end error budget:** sampling-induced change in sim-vs-NCU error ≤ 0.3 × the model's own MAE (so sampling noise cannot reorder model-quality conclusions).
- **Negative controls:** the histogram + few-wave cells MUST fail (gates fire or estimator misses with honest CI failure). If negative controls pass, the test is too weak — the single most important check.
- **Ablation:** wave-segment vs independent-block on the same cells, documenting the co-residency effect.
Re-run on any change to tracer / stats-pass / estimator version (all pinned in the sidecar).

## 8. Top silent-bias failure modes + detection

1. Co-residency thinning (per-block) → hit rates biased up. Prevented by wave-segment PSUs + §7 ablation.
2. L2 cold-start per segment → hit-rate biased down. Caught by ghost-depth sweep convergence.
3. Heavy-tail blocks outside sample → traffic under-estimated with deceptively tight CIs. Caught by exact same-run record totals (`X̂` vs `X` aborts); skew gate force-includes heads.
4. Subset masquerading as reduced-shape via rewritten `gridDim`. Prevented by never-rewrite + reader cross-check.
5. Cross-wave atomic hot-line contention missed. Caught by `atomic_gate`; re-gate after the tracer atomic-address change.
6. Phase/launch-index aliasing. Caught by random-start selection, change-point phase strata, slice-occupancy histogram comparison.
7. Cycles laundering into full-shape scoring. Prevented by `not_recoverable` + typed `Estimate.unrecoverable` refusal.
8. Cross-run placement drift. Prevented by keying offline decisions to `blockIdx` + online wave membership + same-run calibration.
9. Stats-version skew across the tracer upgrade. Prevented by version pinning + refusal to mix within a cell.
10. Small-cluster bootstrap over-confidence (< 8 segments). Structural floor + `wave_gate` forbid.
11. Reduced-shape working set slipping under the 6 MB L2. Caught by the footprint check that forbids `dataset_shrunk` across the L2 boundary.
12. Compression byte corruption from any "slimming". Permitted only with round-trip raw-hash equality; plain zstd needs no exemption.

**Bottom line:** implement tier 1 + the compressibility probe immediately (likely clears the streaming half of the roster); build the gate classifier BEFORE any sampler code; implement tier 2 as wave-segment cluster sampling with the calibrated HT estimator — expecting it to apply to perhaps a quarter of breaching cells, with the atomic-heavy remainder going to matched reduced-shape or honest `unavailable`.
