# Reuse-distance / reuse-interval consultation record

**Status:** external research and design advice only; no reuse model has been
implemented or promoted. The live cache/timing model is unchanged.

**Date:** 2026-10-18

## Request and evidence supplied to Gemini

The read-only brain was invoked by subagents with:

```text
agy --dangerously-skip-permissions --model gemini-4-argon-high \
  --effort high --mode plan
```

The compiled context is in `consultation_prompt.txt`. It includes the source
scope rules, the 339-capture SM120 DRAM-enabled lane, the 286-successful-capture
SM86 lane, twelve-worker full replay only, current metric residuals, observer
seams, trace limitations, and the prior rejected pending-read experiment.
Gemini was explicitly told not to edit code/files, run replay, or use NCU
values to fit or select a model.

Latest complete replay evidence supplied to the consultation:

- SM120: 339/339 captures `ok`, 11,558 launches; approximately 58.4% unified
  L1-hit APE, 39.4% global-L1 lookup-hit APE, 24.3% generic L2-TEX lookup-hit
  APE after the trace-semantic store/TMA observers, 14.9% operation-read L2
  hit APE, 49.8% DRAM-read APE, 9.45% active-IPC APE, and 13.6% elapsed-IPC
  APE.
- SM86: 356 attempted captures, 286 `ok`, 62 `control_error`, and 8
  `replay_error`, 12,392 launches; approximately 49.3% unified-L1-hit APE,
  27.6% global-L1 hit APE, 8.5% generic-L2 hit APE, 4.3% operation-read L2
  hit APE, 24.8% DRAM-read APE, 18.6% active-IPC APE, and 11.9% elapsed-IPC
  APE.

These are independent post-replay NCU joins. NCU is not replay input.

## Literature findings

The literature supports the following narrowly defined claims:

1. Ding and Zhong's whole-program locality work uses approximate LRU/reuse
   distance, pattern recognition, and distance-based sampling to make locality
   predictions at manageable cost. It does not justify fitting a simulator to
   profiler counters.
2. The Rochester reuse-interval thesis describes measuring reuse intervals and
   predicting cache-line-granularity MRCs, including complex control flow and
   writebacks. Our traces lack a compiler dependence graph, so any interval
   model here must be trace-derived and explicitly approximate.
3. The measurement-theory paper distinguishes reuse time/interval, reuse
   distance, footprint, working set, and miss ratio. A reuse-distance histogram
   directly produces a fully-associative LRU MRC; reuse time alone does not
   uniquely determine reuse distance. Cold starts, finite-trace boundaries,
   and the chosen interleaved reference stream must be represented.
4. Set-associative/non-LRU GPU caches are not solved by a fully-associative
   MRC. A set-local stream is needed when set/hash/replacement information is
   known; otherwise the result is a diagnostic bound/sensitivity curve, not a
   hardware prediction.

Primary/official references:

- [Ding & Zhong, PLDI 2003](https://doi.org/10.1145/780822.781159)
- [Rochester reuse-interval thesis](http://hdl.handle.net/1802/35043)
- [A Measurement Theory of Locality](https://arxiv.org/abs/1802.01254)
- [Chen Ding's research page](https://cs.rochester.edu/~cding/Research/)
- [Chen Ding faculty profile](https://cs.rochester.edu/people/faculty/ding_chen/index.html)

## Gemini advice (consolidated from two successful consultations)

Gemini's two memos agree on the following design, with the important caveat
that every hardware-specific assertion must be independently verified against
the repository cards/traces before coding:

### Reference-stream contract

- Coalesce active lanes within each warp instruction first. Use a 128-byte
  line key with a four-bit mask for its 32-byte sectors; do not count duplicate
  lane references as multiple cache references.
- Keep separate streams for private/per-SM L1 and the shared/partitioned L2.
  Do not fabricate a total cross-warp or cross-SM time order. Preserve the
  existing within-warp order and expose unavailable interleaving as a typed
  diagnostic.
- Keep launch-cold and stream-persistent L2 scopes separate. Reset/retain only
  when the replay contract and capture metadata establish that boundary.
- Classify global loads, stores, atomics/reductions, local traffic, and TMA in
  operation-qualified lanes. TMA payloads and store allocation/bypass rules
  must remain separate until trace/card evidence proves their route.

### Observer and MRC

- Start with a side-effect-free `DingLocalityObserver` attached to the existing
  `TraceMetrics` memory observation path. It must not mutate live cache state,
  DRAM demand, issue timing, or completion frontiers.
- Record cold and finite reuse intervals in bounded log-linear histograms and
  produce a report-only footprint/MRC curve. Where an exact order-statistics
  structure is affordable, use it as a verifier; otherwise keep the interval
  result explicitly approximate.
- For a sector-aware MRC, maintain line residency and four-sector validity
  separately. A line/tag reuse is not automatically a sector hit.
- Pending-fill merges must be separated from settled SRAM hits only when the
  simulator has an evidence-backed ready/completion frontier. A guessed global
  instruction-window constant is not acceptable, and the rejected pending-read
  promotion must not be revived by assumption.
- L2 demand should be derived from an explicitly documented L1-filtered stream,
  while an unfiltered diagnostic stream is retained for conservation checks.
- Any set-associative projection must use a documented, measured geometry/hash/
  replacement policy. SM120's card marks those fields as research gaps, so its
  curve must remain a bound/sensitivity result unless new metadata resolves it.

### Complexity and safety targets

Gemini proposed an O(1)-update reuse-interval histogram plus optional exact
set-sampled verifier, bounded per-worker state, and deterministic aggregation.
The exact table size, set count, cache capacities, write policy, TMA route,
MSHR horizon, and any 1/16 sampling ratio are proposals—not established
hardware facts. Do not hard-code them from the memo. Measure their replay cost
before considering promotion.

### Validation gates

Before any observer can affect scored outputs, require all of the following:

- sector conservation; `hits + misses == requests`; no underflow/overflow;
  monotone MRC; explicit cold-start and finite-trace endpoint checks;
- deterministic replay and unchanged SM120/SM86 status profiles;
- no NCU values in fitting, model selection, or per-workload lookup tables;
- family-stratified held-out captures plus the full enabled lanes;
- report sector-weighted WAPE, percentage-point hit-rate error, mean/median/P90
  APE, and signed bias; do not let near-zero targets dominate;
- no regression in existing generic-L2 or IPC metrics, and a measured replay
  overhead/RSS budget acceptable for twelve workers;
- only promote if the held-out and full-corpus gates improve the intended
  endpoint without violating any non-regression guard. Otherwise retain the
  MRC as diagnostic evidence and report the scope/metadata limitation.

Gemini specifically warned that reuse modelling cannot by itself repair NCU
endpoint/join mismatches, missing SMEM/L1 carveout metadata, unknown SM120
slice/hash/replacement behavior, DRAM burst/promotion, driver pollution, or
unmatched SM86 launch joins. Those are not legitimate reasons to force a
sub-10% number.

## Independent adversarial review of the advice

The memo's equations and hardware examples are hypotheses, not repository
facts. In particular, the cards explicitly mark SM120 L1 geometry, L2 slice
mapping, replacement, and timing as research gaps. The following must be
validated before any implementation:

- whether the current trace contains a usable SM/CTA assignment and an ordering
  sufficient for an L1 stream;
- whether cache line/sector IDs are physical-address-stable across captures;
- whether the existing issue model exposes a completion/ready cycle suitable
  for a pending-merge split;
- the actual store, atomic, TMA, and cache-modifier routes for each architecture;
- the exact replay scope corresponding to each NCU profile;
- whether any proposed set/hash or capacity tier is card evidence or merely an
  assumption.

The proposal therefore remains **shadow-only**. The next execution step is an
independent code/evidence audit of those seams, followed by a minimal report
field and a smoke/full replay comparison. No active cache/timing change is
authorized by this consultation record.

## Focused gate consultation and implementation boundary

A second focused `agy` consultation was requested before implementation. Its
partial memo is `/tmp/gemini_reuse_gate.txt`. It sharpened the boundary:

* `TraceMetrics::observe_with_mask_presence_arch` runs while extraction walks
  warp streams and therefore cannot provide true inter-warp or cross-SM arrival
  order. A physically meaningful L1 observer would need a read-only hook in
  the cycle-interleaved SM issue path, and a physically meaningful L2 observer
  would need the timestamp-ordered system event path.
* The first safe implementation is consequently a per-warp, report-only RI
  census. It must not be promoted to L1/L2 predictions merely because its
  proxy curve is monotone. Lines within one memory record share a logical tick;
  their order is unavailable.
* Any RI-derived curve must be labelled an interval/sensitivity proxy. Exact
  fully-associative LRU MRC requires reuse distance; set-associative/non-LRU
  and SM120 geometry/hash/replacement behavior remain unidentifiable from the
  current contract. Right-censored final intervals and missing/empty spans
  are explicit counters rather than fabricated misses.

The executioner implemented only this shadow phase in `TraceMetrics`, added
focused unit tests, and added `tools/validate_reuse_mrc.py`. The observer does
not alter existing predictions. Full enabled-SM120 and SM86 replay validation
with twelve workers is required before considering any later, separately
consulted cycle-interleaved design.

## Follow-up correction to the locality equations

The focused correction memo `/tmp/gemini_reuse_correction.txt` supersedes any
tempting stronger interpretation of the earlier design sketch. A timestamp
gap `Δt` is reuse interval/time; exact LRU reuse distance is the number of
**distinct** intervening items. Sector intervals likewise do not prove sector
residency after tag eviction. Therefore this implementation emits only the
bounded interval histogram and survival sensitivity proxy; it does not emit
MRC hit rates or feed those values into the simulator. A future exact sampled
RD observer would need deterministic SHARDS-style sampling plus an
order-statistics/Fenwick structure, and would still require true issue/L2
arrival streams. The correction's go/no-go invariants include reference
conservation, coalescing idempotence, `Δt - 1 >= RD` for sampled witnesses,
line/sector cardinality bounds, and bit-identical timing with the observer
disabled.

## Raw consultation artifacts

- `/tmp/gemini_mrc_consultation_retry.txt` — concise successful memo.
- `/tmp/gemini_reuse_design3.txt` — detailed successful memo (39,631 bytes).
- `/tmp/gemini_reuse_compiled.txt` — compiled all-findings prompt attempt;
  `agy` timed out after three minutes while inspecting the repository and did
  not return a design. The two successful subagent memos are the authoritative
  brain advice for this turn.
