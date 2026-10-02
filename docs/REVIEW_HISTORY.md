# Review and refactor history

This is a narrative research history, not an approval, audit, provenance, or
execution gate. It records what the supplied review identified and what the
current worktree now demonstrates.

## Initial review — 2026-10-01

The review rejected five overclaims:

- descriptor recurrence equality was not native causal equivalence;
- service demand did not reach an observer endpoint;
- unknown completion tails were being given finite upper bounds;
- multi-warp order and completion multiplicity were synthetic/count-only;
- cache topology did not establish ownership, read-from, coherence, or DRAM
  causality.

The defensible contribution was reframed as unknown-preserving,
state-complete selective reification with exact fallback when the boundary is
not closed.

## Implemented refactor milestones

1. **Model-local frontier contract** — service composition, elapsed endpoint
   arithmetic, unknown-tail propagation, and explicit model-local vocabulary.
2. **Native boundary fallback** — typed region contracts, exact projection
   comparison, opt-in reference decisions, and structured reification reasons.
3. **Order/completion/coverage** — multi-warp refusal, independent dependency
   evidence, completion-multiplicity fallback, suffix entry-state refusal, and
   full-trace coverage counters.
4. **Service/observer composition** — capacity-aware probe profiles, v2
   identifiable probe modes, explicit admission/service overlap, and observer
   propagation.
5. **Cache witness separation** — warp-local instrumentation order only where
   supplied; fence, generation, owner/read-from, and hardware completion stay
   unknown; unresolved cache history widens a separate frontier.
6. **Staged evaluation** — service-capacity variants, three-family
   calibration/development/held-out prediction rows, post-prediction NCU join,
   interval/refusal/runtime/memory reporting, and model-local compression probe.

## Current evidence

- Three current SM86 families were captured and predicted before the NCU join:
  atomic histogram, tensor async GEMM, and the mixed-memory canary.
- NCU `gpc__cycles_elapsed.max` values were 8,811, 4,530, and 5,270 cycles;
  all open lower-bound intervals covered their target.
- Lower-bound-proxy APEs were 72.36%, 77.40%, and 91.80%; these are comparison
  results, not calibration or selection features.
- A witnessed single-warp descriptor region reduced 16 exact model steps to
  one summary step. Real memory-heavy traces remained exact/refused.
- Full Rust workspace and focused reification/evaluation Python suites pass;
  the unrelated PTX subset compiler test still observes a pre-existing
  `ptx_subset_compiler_alpha_v1` versus `...artifact_v1` binary/schema mismatch.

## Second-round R1 hardening — 2026-10-01 (complete; R2 active)

The second-round review found several concrete soundness holes rather than
asking for more report volume. The active first stage now addresses those
holes before any executable macro work:

- unresolved/unclassified service demand produces an open upper, and optional
  zero-demand pools contribute no startup latency to the lower bound;
- explicit endpoint projections propagate every active lower-only completion
  frontier instead of checking only typed/cache tails;
- transform intervals separate event time from optional insertion-sequence
  order, so an earlier timestamp inserted later is not rejected by a synthetic
  monotonicity edge;
- design/evaluation summaries compare the declared timing projection, report
  finite interval coverage separately from open refusal, and expose partial
  order ties rather than lexical rankings;
- prediction/observation documents use a v2 target contract and reject metric,
  endpoint, clock, cache, launch, repetition, GPU, or family-split mismatches;
- record coverage is named separately from exact/summarized execution-step
  reduction; no record ratio is presented as compression.

These changes are still model/evaluation evidence. R2 is now complete at the
model-executor level; native `SmCore` remains untouched and no naming change is
treated as native completion.

## Second-round R2 execution seam — 2026-10-01 (complete; R3 active)

The descriptor analyzer now exposes an explicit model executor rather than only
returning report telemetry. `ModelExecutor::execute_macro` derives a
nonidentity scalar `MacroTransition` from the closed-form interval certificate,
installs its issue/completion effects and pending completion vector, and then
accepts a detailed continuation. The model-local probe observes one macro
dispatch for the 16-instruction witnessed region; a pending predecessor forces
`IncomingPipelineWork` fallback. Native `SmCore` still does not consume this
transition, so no native compression claim follows.
The fast-frontier output carries a `macro_dispatches` counter and the schema
requires it. R3 is now active: independently derive eligibility witnesses and
close or refuse global order and same-cycle effects.

## Second-round R3 witness/order closure — 2026-10-01 (complete; R4 active)

Eligibility now requires contiguous `CausalFrontierWitness` facts from an
allowed static/independent-trace origin. The witness is checked against every
dependency, control, side-effect, and completion-multiplicity field; stale
facts and producer-premise-only ranges reify. The compact native path no longer
creates evidence from its own candidate segment, so real traces remain exact or
refused until a separate witness source is supplied. Multi-warp order remains
an explicit `CrossWarpOrderUnknown` refusal, while same-cycle effects retain
the engine's time/phase/insertion-order check. This closes the descriptor
evidence boundary, not a hardware scheduler proof. R4 is now active on service
queue/assignment semantics.

## Second-round review questions

- Can a native typed macro exit be produced for a nontrivial region without
  leaving future-relevant state unknown?
- Can a larger, architecture-compatible trace corpus provide independent
  family splits and a finite-width interval rather than only open bounds?
- Does selective reification change an architectural conclusion, rather than
  only reducing model work, while preserving exact fallback?
- Which rankings remain stable when independent service and cache uncertainty
  intervals are swept?

These are research questions, not prerequisites enforced by repository
machinery.

## Second-round R4 service semantics — 2026-10-01 (complete; R5 active)

Service composition now carries an explicit role: residual closure,
replacement timing, or independent post-replay. Because current trace demand
counts the same full instruction population already represented by detailed
replay, the live path uses `independent_post_replay` and leaves the observer
frontier unchanged. A separate `ServiceQueueProjection` reports serial
admission, queue depth, open completion uppers, and unresolved marginal pool
 assignment. The v2 fitter now records occupancy/contention as unmodeled rather
than pretending their presence identifies a parameter, and retains a
lower/upper initiation-interval envelope. Grouped assignment envelopes are
preserved when supplied; the native metrics path remains marginal without a
per-record route witness. R5 is now active on native state refinement.

## Second-round R5 native refinement — 2026-10-01 (active)

The typed region validator now checks internal effect payload spans, and the
trace-aware native comparator checks effect/label sequences in addition to
entry/exit state. Comparator decisions are converted into an explicit
typed-macro versus exact-replay plan in the opt-in native writer. This improves
fallback soundness but does not yet consume a nonidentity native macro; the
retained SM86 canary remains exact-fallback evidence. A pure-scalar SM86 probe
now supplies a complete typed pre/post boundary pair; its cold front-end
transition is intentionally rejected by the strict typed scalar macro executor
instead of being claimed as preserved.
