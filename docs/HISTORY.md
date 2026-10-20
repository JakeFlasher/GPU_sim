# Project History

This file records architectural changes and accepted evidence. It is not an
execution procedure; current commands and boundaries are in `CONTRACT.md`,
`REPLAY.md`, and `STATUS.md`.

## Repository merge

The repository combines the SM86/RTX 3090 Ti and SM120/RTX 5090 lanes in one
card-selected Rust replay path:

```text
trace + sidecars -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run
```

The merge separated architecture-specific L1/L2 attribution, shared-memory
wavefront accounting, opcode scope, and candidate documents. Complete replays
use the integrated `l1-l2` path, compressed JSONL, and twelve workers.

## Accepted simulator changes

The following changes were accepted only after full-corpus comparison:

1. architecture-specific L1 and store lookup attribution;
2. exact pending-sector and already-requested merge accounting;
3. sub-warp shared-memory wavefront aggregation;
4. per-SM SM86 write-tag observation;
5. SM120 write-tag and `GlobalUntagged` operation attribution;
6. pre-packet L2 write-tag accounting and same-instruction store grouping;
7. report-only sector-aware locality observers;
8. the SM120 conditional store-packet interval for the mixed-client TEX-L2
   endpoint.

These observers preserve cache state, DRAM demand, issue timing, and
completion timestamps unless a change is explicitly described as behavioral.

## Current release evidence

The current closure artifacts are:

- SM120: `full_replay_runs/cache_locality_m7closure_sm120/`
  (339 captures, 11,558 launches, all `ok`);
- SM86: `full_replay_runs/cache_locality_m7closure_sm86/`
  (286 `ok`, 62 `control_error`, 8 `replay_error`, 12,392 launches).

The cache-locality plan is closed under the project's effective-model
criterion. The final point and interval metrics, join populations, and known
scope limitations are in `RESULTS.md` and `RESIDUAL_FRONTIER.md`.

## Historical experiments

Rejected alternatives—including TMA-shadow DRAM substitution, pure-LTC
promotion, large-store packet suppression, and NCU-derived count balancing—
remain in Git history and dated research notes. They are not active model
behavior and must not be revived without a new mechanism and a complete
regression study.

## Academic release pass (2026-10-20)

The public vocabulary, metric definitions, architecture description, status
contract, and persistent session notes were aligned with standard
computer-architecture usage. Internal pending-bound terminology was renamed in
the event scheduler without changing serialized replay semantics. Full
12-worker replay parity against the M7 closure artifacts was confirmed for
both lanes before shipping the release wording.
