# Full-replay audit

Canonical current results are maintained in [`docs/RESULTS.md`](docs/RESULTS.md);
this root file is a compatibility summary for existing tooling.

All complete-corpus invocations used the twelve-worker simulator process pool;
fast replay and one-worker replay were not used.

| Lane/mode | Captures | Statuses | Launches |
| --- | ---: | --- | ---: |
| SM120, DRAM disabled | 339 | 339 `ok` | 11,558 |
| SM120, carded DRAM enabled | 339 | 339 `ok` | 11,558 |
| SM86 / RTX 3090 Ti | 356 attempted | 286 `ok`, 62 `control_error`, 8 `replay_error` | 12,392 |

The runner enforces the complete-corpus status profiles, and these totals match
the retained baseline result trees. Large replay outputs are
ignored; regenerate them with [`docs/REPLAY.md`](docs/REPLAY.md).

`FULL_NUMERIC_COMPARISON.json` is the authoritative machine-readable result.
It checks duplicate/missing/extra IDs, status, architecture, memory mode,
cache scope, launch count, return code, finite values, and common numeric
fields. Wall-clock fields and additive observer fields are intentionally
excluded. Historical SM120 capture IDs are compared with the explicit `/` to
`__` identifier bridge; collisions fail rather than overwrite rows.

The loop fixed architecture-specific L1/store projections, duplicate shared
wavefront accounting, branch-list dispatch, SM120 global-load scope, and the
SM86 shared-wavefront compatibility projection. No unexplained numeric
discrepancy remains.
