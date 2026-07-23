# Trace-Sampling Design — task13 finalized adjudication (AC-9)

Adjudicates `plans/sm86_trace_sampling_strategy_fable5_design.md` against the
simulator's ACTUAL replay semantics, grounded in `sim/sm/src/replay.rs` and
`sim/trace/src/reader.rs` (read directly, not via the design's paraphrase). This is
the task13 deliverable: the ACCEPTED thresholds/estimator/guards + the binding
revisions task15/task16/task17 must honor. Analyze-lane adjudication (independent
reviewer, 2026-07-09); no NCU metric enters the design (Rule 1 — this is about
trace sampling, not model inputs).

## Verdict: the design's central thesis is SOUND against the real replay.
Literal `sm_id`/`warpid_sm` placement ("the traced placement IS the model",
`replay.rs:8-11`) + per-SM L1 + one shared L2 (`replay.rs:317-321,348-356`) make
**wave-segment cluster sampling of WHOLE blocks** the correct primary sampling
unit; the exact `records_seen` count the sim already computes (`replay.rs:168`,
invariant-checked `:263-275`) makes the `X̂≈X` calibration abort genuinely
powerful; cycles are correctly `not_recoverable` (`max_cycles` is a whole-launch
max, `replay.rs:122,265`); and per-record / every-k decimation is inadmissible for
the RIGHT reason — `classify_with_l1` dedups sectors per-instruction in first-seen
order (`replay.rs:433-447`), so ordered replay (not randomness) is what breaks.
All gaps are implementation-level; none reject the design.

## Per-item verdict

| Item | Verdict | Grounding |
|---|---|---|
| A. Literal placement ⇒ wave-segment PSU | **ACCEPT** | Block SM = `inst.sm_id` (`replay.rs:187-198,239-241`); independent block thinning changes co-residency/L1/shared-L2; a whole-block wave segment preserves exactly what replay consumes within a segment (cross-segment L2 warmth is the design's own §3 approximation). |
| B. Five gates + thresholds | **ACCEPT-WITH-REVISION** | Replay self-detects NONE of grid-sync/few-waves/atomic-hotline/skew/phases — it just places and prices records — so every gate is necessary and the thresholds defensible; `atomic_gate` correctly presupposes the post-flip address-carrying contract. REVISION R5 (unit). |
| C. HT estimator + calibration + stopping rule | **ACCEPT-WITH-REVISION** | HT + combined-ratio `T̂·X/X̂` + `X̂≈X` abort leverage the exact `records_seen`; ratio-of-totals matches the sim's `L1Counters` sum→ratio (`replay.rs:266-268`). REVISION R3 (per-segment measurement). |
| D. Reader guards (task16) | **ACCEPT-WITH-REVISION** | The coverage hole is real — `threadblock_paths`/`read_threadblocks_in_launch_order`/`block_launch_index` (`reader.rs:208-324`) only range-check + dedup, never assert all `grid_total` indices present; replay is likewise blind. REVISIONS R1 (replay-path), R2 (owner split). |
| E. Closed schema + inadmissible list | **ACCEPT-WITH-REVISION** | Inadmissibility rationale matches the ordered first-seen replay exactly. REVISION R4 (segment membership). |
| F. Revisions the sim FORCES | see R6 | Placement-identity records make whole-block retention mandatory. |

## Accepted thresholds (task15 binding — provenance-recorded)
- `coop_gate`: cooperative/grid-sync launch → **forbid** subset (route tier 3/5). ACCEPT.
- `wave_gate`: **≥ 8 MEASURED waves required** — i.e. `n_waves ≥ 8·(L − ghost)` with default `L=3, ghost=1` ⇒ **≥ 16 reconstructable waves** (revised from the design's bare "G < 8" to reconcile with the ≥8-segment estimator floor; see R5). Below → forbid (kills the histogram cell: 256 blocks ≈ 1.5–3 waves). ACCEPT-WITH-REVISION.
- `atomic_gate`: `atomic/total > 0.5%` **AND** top-256 sectors > 10% of atomic traffic → forbid for L2/cycles/atomic metrics. ACCEPT (only computable post-flip — atomics now carry addresses).
- `skew_gate`: top-1% blocks ≥ 15% of records **OR** Gini > 0.5 → heavy blocks become certainty includes; if certainty set > 40% of the cell byte budget → tier 3. ACCEPT.
- `homogeneity`: CV of per-SM block counts + per-wave record totals; change-point detection → phases become strata. ACCEPT.

## Accepted estimator + stopping rule (task15 binding)
- Additive counts: HT `T̂ = Σ_C t_s + Σ_h (N_h/n_h) Σ_{S_h} t_s`, calibrated `T̂_cal = T̂·X/X̂` where `X`,`X̂` are exact same-run record counts; `|X̂−X|/X` outside its CI → **ABORT** (do not calibrate away). ACCEPT.
- Rates: ratio of CALIBRATED HT totals `R̂ = T̂_hits/T̂_sectors` (never a mean of per-segment rates). ACCEPT.
- Cycles: `not_recoverable` (ship only `cycles_estimate_experimental`, §7-gated). ACCEPT.
- CI: stratified cluster bootstrap, B=1000, 95% percentile; floors `≥ 8 sampled segments`, `n_h ≥ 2`. ACCEPT.
- Stop: rates 95% CI half-width ≤ **0.02 absolute**; counts CI half-width ≤ **2–5% relative**; calibration self-check within CI; ghost-depth 1→2 movement ≤ 0.01 for L2 rates. Not converged at budget → `budget_limited_wide_ci` or `unavailable` (never the point estimate). ACCEPT.

## Reader/guard invariants (task16 binding)
- `E_INCOMPLETE_TRACE`: coverage < 100% of `gridDim` WITHOUT a `trace_sampling.json` sidecar → hard error. **Enforced where replay consumes blocks** (R1).
- Never rewrite `gridDim`; cross-check the launch record's `grid_dim` against `sidecar.shape.original_grid` (a rewrite would trip the existing `BlockIdOutOfRange`, `reader.rs:313-321` — the rule matches the reader's behavior). Reader-owned.
- `records_seen == sidecar.records.captured`. Reader-owned.
- `issued == records_seen` over the subset — already enforced (`CountMismatch`, `replay.rs:270-275`); orthogonal to coverage.

## Closed-schema sidecar (task15 binding)
`trace_sampling.json`, `additionalProperties:false`, hash-pinned. All §5 fields, **plus R4**: an explicit `segments[] { id, block_ids[] }` membership map (and per-SM slot map), because the sim has no arrival/start-order and sorts per-SM blocks by `launch_index` (`replay.rs:247-249`) — arrival-order pseudo-waves are NOT re-derivable at sim time. `not_recoverable` defaults to `[cycles, SOL-%timing, tail-beyond-final-segment]`; inadmissible methods (per-record reservoir, deterministic every-k) are schema-forbidden.

## Binding REVISIONS (task15 / task16 must honor)
- **R1 (task16, HARD — a real conflict).** The completeness / `E_INCOMPLETE_TRACE` check must live in (or be shared into) `extract_per_sm_blocks` (`replay.rs:140-251`, which already builds `seen_indices: BTreeSet<i64>` at `:149` and has `grid_dim` via `block_launch_index`) — NOT only in `read_threadblocks_in_launch_order`, because replay iterates `td.threadblock_paths()` directly (`replay.rs:152-154`) and never calls that helper. The design's §6.1 "add at the reader" alone would leave the replay path unguarded.
- **R2 (task16).** Split the invariants by owner: `records_seen==sidecar.records.captured` + `gridDim==sidecar.shape.original_grid` at the reader; MOVE the `X̂≈X` self-check to task15 as an estimator abort — the reader cannot compute `X̂` (it needs strata/HT weights). The design's §6.3 misplaces it.
- **R3 (task15).** The sim's finest output is per-SM (`KernelReplayResult`, `replay.rs:118-135`) with a whole-launch shared L2 and cold caches — there is NO per-wave/per-segment attribution. Realize per-segment `t_s` + ghost warm-up via **per-segment sub-trace replay with ghost-differencing** (replay ghost-only, replay ghost+measured, subtract). The design assumes an attribution the sim does not provide.
- **R4 (task15/schema).** Freeze block→segment membership explicitly (the `segments[]` addition above) OR redefine `wave_model.reconstruction` on `launch_index` — the sim has no start-order concept.
- **R5 (task15).** Reconcile the `wave_gate` unit: the design states waves (`G<8`) but the estimator floor is ≥8 SEGMENTS (a segment = `L=3` waves), so a cell with 8–23 waves would pass the gate yet fail the floor. Require ≥ `8·(L−ghost)` measured waves (≥16 at defaults). Adopted in the accepted thresholds above.
- **R6 (reinforcement).** Whole-block retention is mandatory for a second, code-forced reason the design omits: `extract_per_sm_blocks` requires a consistent `sm_id` per block and `warpid_sm` per record-bearing warp (`SmIdMissing`/`SmIdInconsistent` `:187-198,239-241`; `WarpSlotMissing`/`WarpSlotInconsistent` `:200-232`). Any future sub-block "slimming" that strips placement-identity records trips these typed errors. Wave-segment sampling already retains whole blocks — state it so no slimming is ever added.

## Bottom line for the AC-9 build order
Tier 1 (compression + probe) + the gate classifier are unaffected and can be built
first. Tier 2 (wave-segment subset + calibrated HT) is sound but must be built to
R3/R4/R5. task16's reader/replay guards must be built to R1/R2. R6 is a standing
invariant. No conceptual redesign is required.
