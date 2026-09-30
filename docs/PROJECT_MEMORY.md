# Continuation memory

**Updated 2026-09-30.** This is the short resume note. Read it, then
[`STATUS`](STATUS.md), and only then the one archived report needed for a
specific question. `AGENTS.md` is the governing repository policy. Dated
reports are retained under [`archive/research_20260930`](archive/research_20260930/)
and are not an automatic queue.

## Scope and completion bar

- Repository: `/home/jakeshea/GPU_simulator_rebase`, branch `GPUSim_main`.
- Hardware anchor: NVIDIA RTX 3090 Ti, Ampere SM86. The native path is
  `NVBit/SASS -> enhanced protobuf -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run`.
- Custom paths are separate: `simt_u32_v1` through `sim-sm::custom`, and the
  functional-only single-group `simt_its_v1` executor. Neither is native SASS
  timing.
- NCU is the detailed comparison target. CUPTI, CUDA events, PMA, and replay
  counters are independent observations; none silently becomes an NCU label or
  a physical truth.
- The useful near-term product is a fast, uncertainty-aware NCU metric
  projection plus a conservative native replay. It is **not** a claim of
  cycle-identical silicon simulation. The previous 1,500--2,050x figure is
  withdrawn: it compared post-processing time for one cohort with replay time
  for many workloads and excluded acquisition and replay.

## Current evidence

- The retained frontier/archetype projector (`tools/sm86_ncu_projection_surrogate.py`)
  supplies a candidate point and interval for every current score row. Its
  strongest results are related-cohort observations, not independent
  architecture-wide generalization. The archetype/shape path is a reference
  projector, not a universal deployment decision.
- Fixed committees cover alternative explanations: frontier ratios, shape and
  tile envelopes, PM/CUPTI semantic surfaces, clock elasticity, native-event
  crosswalks, cache/reuse scenarios, and the NAK stall-prior projection. They
  are mechanism-agnostic approximations and remain non-promoted.
- The four-cell temporal matrix (`results/alpha/sm86_regime_matrix_goal_20260930/matrix.json`)
  retains 15/15 rows per cell. Neutral low/unknown cold and warm routes score
  8.36%/8.08% and 8.02%/7.90% GPC/duration MAPE; independently clock-witnessed
  boost routes score 9.88%/9.67% and 9.36%/9.76%. These are development
  observations on related cohorts, not a universal accuracy guarantee.
- The corrected interaction-envelope reports use opposite-cohort calibration:
  cold coverage is 13/15 GPC and 15/15 duration; warm coverage is 15/15 GPC
  and 13/15 duration. Widths are empirical scenario spreads, not confidence
  intervals. The corrected paths are recorded in the refined model config.
- Native CUPTI PC campaigns collected 470,506 samples/21,161 records (cold)
  and 920,753/42,500 records including warmups (warm), with zero reported
  drops. Decoder v2 separates all-cycle states from no-issue subsets. This is
  descriptive scheduler/resource evidence, not cache ownership or endpoint
  identity. `imc_miss` means immediate-constant-cache miss.
- Independent Mesa NAK SM86 operation-family bodies, a CUDA-ELF template bridge,
  direct PMA telemetry, repeated CUPTI sampling, and public RM flush controls
  are retained. They expand the observable feature surface but do not expose
  proprietary scheduling, cache read-from, or unified timing state.

## Open physical boundaries

These remain explicit unknowns in the model and config:

1. exact per-instruction physical pipe, sub-unit, and issue-slot identity;
2. unrestricted independent SM86 SASS/scheduler/control-word/CUDA-ABI
   generation;
3. cache-line ownership, read-from, dirty writeback, and complete DRAM ordering;
4. one synchronized endpoint for NCU, CUPTI, CUDA events, device timers, and
   simulator cycles.

They are not logical prerequisites for useful metric approximation. Surrogates,
intervals, latent factors, matched interventions, and conservative fallbacks
can model their effects, but must be labelled as approximations rather than
renaming the unknown mechanisms as solved.

## Scientific invariants

- Keep issue, modeled-event, completion, launch-envelope, and typed-tail
  frontiers separate. Never collapse them into a scalar route score.
- Traces and architecture cards are inputs. Scored NCU observations are
  comparison labels only; they cannot be features, tuning inputs, or route
  selection criteria for the same workload.
- Base and large are related cohorts. “15/15 covered” means a prediction row
  was emitted, not that every possible shape is in-distribution.
- An unmatched shape receives `candidate_fallback` with a widened interval and
  an out-of-hull diagnostic. This avoids an OOD refusal for the current input
  contract, but it does not promise valid accuracy for arbitrary future
  workloads.
- Native replay semantics and custom functional semantics stay separate. A
  passing self-authored probe is implementation evidence, not silicon
  validation.
- Projection wall time is reported as post-processing only unless trace
  acquisition, replay, and native measurement are included in the same cost.

## Active implementation seams

- `sim/mem/src/payload.rs` accepts capture schema v1/v2 and preserves optional
  launch/function/CTA/warp/PC metadata without changing the replay primitive.
- `workloads/research_probes/sm86_value_differential/` uses 64-bit addresses,
  schema-v2 capture, metadata joining, and explicit validation helpers.
- `workloads/research_probes/sm86_ncu_demo_benchmark/alpha_campaign.py` has
  bounded process-wait fallback, launch-window clock context, and aggregate
  NCU/CUDA-event fields. It does not make a physical pipe claim.
- New tools under `tools/sm86_*` are small research alternatives. Their
  current status, source inputs, and result locations are summarized in
  `docs/STATUS.md` and the JSON config; do not infer promotion from filename.

## Resume procedure

1. Read `docs/STATUS.md` and inspect the narrow source seam.
2. Decide whether the task is native replay, custom functional execution, or
   NCU projection; do not mix their contracts.
3. Use retained result paths only as unbound development evidence. State the
   cohort, endpoint, intervention, and whether labels influenced fitting.
4. Prefer a family-disjoint or counter-free holdout and a complete cost
   comparison before changing the reference projector.
5. Update the compact entry points. Add a dated report only when it preserves
   a genuinely new observation; place it directly in the archive and add one
   line to `HISTORY.md`.

The current research goal remains blocked at the four external boundaries; it
is not marked complete by this documentation consolidation. No tests, hashing,
or verification workflows were run for the 2026-09-30 documentation/commit
pass.
