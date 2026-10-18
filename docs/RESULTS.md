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
