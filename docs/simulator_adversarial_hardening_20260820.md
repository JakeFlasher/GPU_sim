# Completed one-time adversarial hardening: batching and compact loading

**Status:** complete 2026-08-20  
**Reviewed milestones:**
[same-cycle event batching](simulator_same_cycle_event_batch_milestone.md) and
[compact trace lifetime](simulator_compact_trace_lifetime_milestone.md)

## Independent review result

Three fresh reviews independently examined event ordering/L2 mutation,
trace-loader parity, and end-to-end campaign integration. No release-blocking
simulator correctness defect was found:

- a batch contains only a consecutive same-SM/same-time emission run;
- its STORE/RED members are reply-less and execute the pre-existing L2
  transaction in exact order;
- cross-SM queue order and timing-active feedback remain separate;
- a later member error cannot commit the kernel-private L2 snapshot; and
- compact loading completes the entire established direct validation path
  before releasing only raw threadblock bytes. Replay consumes the retained
  validated blocks, while raw and campaign loaders remain unchanged.

Independent feature-binary comparisons also produced byte-identical normal
JSON for all batching-active cases and candidate-bearing paths.

## Problems found and resolved

The review did find material evidence and integration defects:

1. The matched harness could silently run an ordinary binary where feature-only
   environment controls were compiled out. It now performs a fail-closed live
   probe for both phase and event diagnostics, accepts an explicit binary, and
   verifies that binary's file identity remains unchanged through capture.
2. A/B artifacts retained only a claimed equality boolean. Each case now keeps
   its canonical normal JSON bytes; reducers require the file, decode it,
   verify its declared length and case binding, and reconcile every repeat's
   result length.
3. Feedback-labeled throughput cases could execute a different candidate.
   Cases now require schemas 10/11/13 and IDs 12/202/214 respectively on every
   launch.
4. Ratio-of-condition-medians ignored the adjacent paired design. Headline
   time/RSS changes are now medians of per-repeat paired changes, with the
   paired range and ratio-of-medians retained separately.
5. Phase fractions summed independent medians and could exceed one. Fractions
   are now computed within each run and then reduced; every retained run is at
   or below one.
6. Event/phase collectors could not reproduce the legacy condition and
   accepted duplicate, negative, or duplicate phase records. Both expose
   `--disable-batching`, validate exact nonnegative fields and launch counts,
   reject duplicate phase lines, and recheck the binary after capture.
7. Replay reuse accepted empty counters and malformed nested timing values.
   It now checks the complete counter tree, completion-frontier variants,
   readiness base shape and numeric domains, exact candidates, launch identity,
   strict dependency freshness, and twin bytes. Base retirement/arrival/
   admission reducers and write-completion predecessors use the same check.
8. The active-corpus validator ignored surplus replay directories. It now
   rejects extras. Twenty old calibration pairs and 33 outputs from a stopped
   over-broad command were removed; all were generated data recoverable from
   their candidate commands.
9. Batch payload metadata grew with historical batch count. Each event now
   directly owns its boxed request slice, bounding storage by live queue depth.
   The event engine reserves the represented logical sequence span, preserving
   FIFO and sequence-exhaustion semantics.
10. Compact callers lacked an ordered validated-block API. The new
    `validated_threadblocks_in_launch_order` works without raw bytes, and the
    raw ordered API documents its loader requirement.

## Regenerated evidence

An isolated `performance-counters` build regenerated both 12-case, seven-repeat
matched comparisons, exact legacy/batched event counts, both phase campaigns,
and the current clean baseline. The ordinary release binary was independently
confirmed to fail the feature probe.

- Batching retains: deep queue depth falls 96.8%, targeted paired time changes
  are -1.9% to -12.9%, and the worst cohort regression is 0.48%.
- Compact loading retains: every case saves memory, the range is 5.5–29.5%,
  and the worst paired timing regression is 0.35%.
- Every matched condition emits byte-identical normal JSON.

The final default binary then regenerated the full active matrix. The terminal
validation result is exactly 5,519 fresh, nested-schema/configuration-matched,
byte-identical pairs and zero inactive replay directories.

Validation completed with the full Rust workspace, explicit performance and
campaign feature suites, formatting, Python compilation, 60 affected Python
tests, all predecessor/successor reducers, and the active-corpus validator.
No sudo command or new GPU/NCU/NVBit capture was required for this host-side
hardening pass.

