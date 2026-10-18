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

`FULL_NUMERIC_COMPARISON_ACCURACY.json` compares these rows with the retained
baseline trees: both SM120 cases have zero row/metadata/numeric differences;
SM86 has the expected opcode-field differences from the LDG.CONSTANT fix and
no status, row-identity, or dropped-field differences.

The report joins NCU only after replay. The current residuals are not all
replay correctness failures:

* SM86 LDG opcode scope now includes traced `LDG.CONSTANT` forms and improved
  from 44.8% to 1.36% APE; LTC128B and LDGSTS remain separate scope lanes.
* SM120 operation-read L2 misses and most sector-volume rows are close, while
  generic L2 hit, unified-L1 hit, DRAM-client totals, and some timing endpoints
  remain high-residual scope/effective-model gaps.
* SM86 shared-memory carveout values are absent from trace sidecars, so NCU's
  per-launch occupancy/configuration endpoints cannot be reconstructed from
  the current replay inputs without an external runtime-resource source.

The generic SM120 L2 discrepancy is retained as an explicit residual rather
than corrected by count balancing: a representative mixed launch emits 128
simulator store lookups (96 hits) while its NCU generic source-TEX row reports
32 store sectors (3 hits). The operation-read endpoint is independently
conserved. See [`consultation_prompt.txt`](../consultation_prompt.txt) for the
brain-review context and required evidence for a future upstream coalescer
slice. These limitations prevent an honest claim that every major NCU metric
is below 10% APE yet; no NCU value is used as a replay or model-selection
input.
