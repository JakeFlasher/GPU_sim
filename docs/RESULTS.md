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
