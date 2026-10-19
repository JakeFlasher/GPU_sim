# Full-replay results

The retained verification campaign used the copied real corpora and the
release binary with twelve workers. Full replay only; no one-worker replay was
used.

| Campaign | Captures | Statuses | Launches |
| --- | ---: | --- | ---: |
| SM120, DRAM disabled | 339 | 339 `ok` | 11,558 |
| SM120, carded DRAM enabled | 339 | 339 `ok` | 11,558 |
| SM86 / RTX 3090 Ti | 356 attempted | 286 `ok`, 62 `control_error`, 8 `replay_error` | 12,392 |

The runner now enforces these complete-corpus status profiles (not merely the
set of allowed status names), and the launch totals match the retained baseline
result trees. The
comparison report (`FULL_NUMERIC_COMPARISON.json`) records zero missing or
extra rows, zero required-metadata differences, zero dropped baseline numeric
fields, and zero common numeric-field differences for all three campaigns.
“Zero” does not mean byte-identical JSON:
wall-clock fields are nondeterministic and the unified merge adds typed
observer fields.

The full output directories are intentionally ignored because they contain
large generated artifacts. To reproduce the check, follow [`REPLAY.md`](REPLAY.md)
and pass the baseline/result paths explicitly. `FULL_REPLAY_AUDIT.md` retains
the root-level campaign summary and root-cause list.

## Correctness fixes carried by the merge

* L1 lookup and store-hit projections dispatch by architecture.
* Shared-wavefront aggregation no longer increments the merged counter twice.
* Branch/control opcode accounting uses the lane-specific set.
* SM120 global-load metrics retain the broad logical-memory scope.
* SM86 shared-wavefront and width-16 observers use an exact compatibility
  projection rather than SM120 phase-local corrections.
* Memory candidate documents cannot cross the SM86/SM120 boundary.

## NCU accuracy campaign

The latest full-only replay used the release binary and `--workers 12` for
both architecture lanes. SM120 completed 339/339 captures in each DRAM mode
(11,558 launches); SM86 completed 356 attempted captures with the expected
286 `ok`, 62 `control_error`, and 8 `replay_error` rows (12,392 launches).
The complete artifacts are `full_replay_runs/priority4_sm120_disabled/`,
`full_replay_runs/priority5_sm120_enabled/`, and
`full_replay_runs/priority4_sm86/`.

`FULL_NUMERIC_COMPARISON_ACCURACY.json` compares the enabled SM120 and SM86
rows with the retained baseline trees. Row identities, statuses, required
metadata, and dropped-field checks remain clean; numeric differences are the
intentional observer changes (SM120 packet/TMA scope and SM86 LDG.CONSTANT
scope), not replay failures. The SM120 DRAM-disabled variant is not part of
the current required replay scope.

The report joins NCU only after replay. The current residuals are not all
replay correctness failures:

* SM86 LDG opcode scope now includes traced `LDG.CONSTANT` forms and improved
  from 44.8% to 1.36% APE; LTC128B and LDGSTS remain separate scope lanes.
* SM120 operation-read L2 misses and most sector-volume rows are close. The
  trace-semantic store-packet and TMA-store-cold-miss observers reduce generic
  L2 hit APE to about **24.3%** (from about 57%), while unified-L1 hit,
  DRAM-client totals, and some timing endpoints remain high-residual
  scope/effective-model gaps.
* SM86 shared-memory carveout values are absent from trace sidecars, so NCU's
  per-launch occupancy/configuration endpoints cannot be reconstructed from
  the current replay inputs without an external runtime-resource source.

The remaining generic SM120 L2 residual is retained explicitly rather than
corrected by count balancing: trace-semantic coalescing now removes repeated
store packets and TMA stores are treated as cold misses, but ordinary
operation-read/client scope and resident-state differences remain. See
[`consultation_prompt.txt`](../consultation_prompt.txt) for the evidence and
Gemini consultation context. These limitations prevent an honest claim that
every major NCU metric is below 10% APE yet; no NCU value is used as a replay
or model-selection input.

## Reuse-interval observer (shadow-only)

Following the Gemini/Chen-Ding consultation, the unified replay now emits a
`trace_metrics.reuse_interval` object on each full-replay launch. It coalesces
decoded 32-byte sectors into 128-byte line keys with four-bit sector masks and
tracks cold/reused lines, sector intervals, bounded histograms, and a
monotone record-gap survival proxy. The observer is reset at each warp and
uses only the trace's within-warp order; lines in one memory record share a
logical tick, and cross-warp/SM arrival order is explicitly unavailable.

This first phase is diagnostic only. It does not change L1/L2 state, DRAM
demand, issue timing, completion frontiers, or scored NCU projections. The
proxy capacities are logical record-gap points, not physical cache geometry.
SM120 set/hash/replacement and per-launch carveout fields remain research
gaps, so no hardware MRC or <10% accuracy claim is made from this observer.
The standalone full-only invariant checker is
`tools/validate_reuse_mrc.py`; it checks conservation, histogram mass,
monotonicity, and zero-capacity behavior without reading NCU data.

The complete shadow campaign is recorded in
`FULL_NUMERIC_COMPARISON_REUSE_SHADOW.json`: SM120 has 339/339 `ok` captures
and 11,558 launches; SM86 has 286 `ok`, 62 `control_error`, and 8
`replay_error` captures across 12,392 launches. Both lanes have zero missing
or extra rows, zero required-metadata differences, and zero common numeric
differences versus the retained full-replay baselines.

The recorded driver wall time was 59.21 s for the shadow SM120 lane versus
58.02 s for its baseline (about 2.1% overhead), and 45.17 s for SM86 versus
44.44 s (about 1.6%). Compressed output grew by about 3% in each lane. These
measurements remain below the consultation's provisional 5% replay-overhead
budget, while no resident cache/timing state was changed.

The final Gemini gate rejected promotion of the interval proxy (and any
unvalidated sampled fully-associative RD curve) into scored L1/L2 predictions:
the live cycle-interleaved L1 and event-driven L2 models are more faithful,
while cross-warp chronology, TMA timing, carveout metadata, and SM120
set/hash/replacement behavior remain unresolved. Thus this locality phase is
complete as a validated diagnostic, but it does not claim to close the global
<10% APE objective or alter existing NCU accuracy projections.

## Async-client L1 scope refinement

The full-replay audit found a trace-proven SM120 `ldgsts_async_copy` scope
distinction: older-pending async merges are not ordinary global-load `op_ld`
tag hits, although physical L1 state and timing retain them. A report-only
exclusion of that older-pending class was added; ordinary pending merges,
constant loads, and all SM86 paths are unchanged.

The required full-only validation completed with twelve workers: SM120
339/339 `ok` captures and 11,558 launches; SM86 286 `ok`, 62 `control_error`,
8 `replay_error`, and 12,392 launches. Reuse invariants passed in both lanes;
SM86 had zero common numeric differences versus baseline. SM120 differences
are limited to the intentional async L1 observer fields in
`FULL_NUMERIC_COMPARISON_ASYNC_SCOPE.json`.

The full SM120 evaluation shows a small improvement in unified/global L1 hit
APE (approximately 58.40%→58.01% and 39.36%→39.32%); L2, DRAM, and IPC are
unchanged. LTC128B and L2 additive hit-volume residuals remain observer-only
research gaps; Gemini rejected active sibling-sector promotion.

## Typed L2 client-scope shadow

The committed L2 event seam now emits passive client-scope accounting for
ordinary LDG, constant LDG, LTC128B, LDGSTS cache/bypass, local, store,
atomic, reduction, and unknown requests. Read requests are partitioned into
resident, pending-merge, and fill-miss masks; write-class requests retain
request/resident observations without read pending/fill arms. LTC128B potential sibling
promotion is reported only as diagnostic volume; no sibling sector is inserted
into physical cache or DRAM state.

The resident/pending/fill partition is a read-scope identity. Write-class
requests still contribute request and resident counts, while their pending and
fill arms are structurally zero under the `WriteAtomic` event class.

The final scope-tag refinement assigns write, atomic, reduction, and async
bypass classes at the request-emission seam (ordinary unqualified bypasses
remain `Unknown`). Full-only validation with twelve workers passed for both
lanes: SM120 339/339 `ok` and 11,558 launches; SM86 286/62/8 and 12,392
launches. The reuse invariant checker passed. The machine-readable
`FULL_NUMERIC_COMPARISON_L2_SHADOW3.json` shows no physical-cache, DRAM, or
timing regression: its 896 SM120 numeric differences are only the already
approved async L1 scope projection, and SM86 has zero common numeric
differences. The client-scope arrays and LTC128B sibling volume remain
diagnostic only and do not claim a new <10% L2 prediction.

Gemini's final closure gate therefore freezes active L2/DRAM changes. The
remaining major-metric residuals require trace-unobservable SM120 slice/set
geometry, replacement/fill-buffer behavior, TMA chronology, carveout/runtime
state, or NCU endpoint scope; fitting or count-balancing those values would
not be a defensible replay improvement. The <10% objective remains open, but
this shadow phase has no safe trace-only promotion left.

## Rejected follow-up projections

Two trace-only candidates were audited after the typed L2 shadow. Replacing
raw TMA payload demand with address-shadow L2 misses worsened the gated DRAM
APE (39.56% to 67.62%), so the raw payload arm remains. A pure-
`global_load_ltc128b` L1 candidate using trace resident hits plus older
pending merges improved 62 nonzero-target rows (54.4% to 4.38%), but the full
pure-LTC audit contains 300 zero-target rows and two false positives. Because
the older-pending counter's sector-unit partition is not yet proven, this
candidate remains shadow-only; no scored L1 correction was promoted.

A later report-only unified-L1 numerator-consistency correction was accepted
for the SM86 native-metrics path only. It reuses the validated SM86
form-qualified global lookup endpoint and leaves SM120's alpha path unchanged.
Full status profiles passed; on 107 joined SM86 NCU rows, unified-L1 mean APE
improved from 29.04% to 12.15%, with no other joined metric changing. It
remains above 10%, so the goal is not complete.

The proposed SM120 native-L2 precedence correction was separately rejected:
the SM120 alpha native-metrics branch exits before the later SM86-only L2
native block, so no SM120 rank-0 native-L2 override exists. Adding TMA-shadow
hits directly to retained SM120 L2 estimates also worsened mean APE.

Gemini's definitive stop gate freezes further active IPC/L1/L2 projection or
model changes. Resuming accuracy work requires TMA descriptor/coordinate and
launch-cache metadata, disaggregated tag/MSHR/fill/stall PMU counters, or
isolated LTC128B/TMA microbenchmarks; cache-hint subtraction, fixed coalescing
windows, constant-pipeline re-scoping, IPC scaling, and further TMA/LTC
projections are not defensible under the current trace contract.
