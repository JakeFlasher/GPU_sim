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

## Academic documentation and source modularization (2026-10-20)

The large Rust facades were split into named Rust child modules while
retaining their original parent scope and public paths. The split covers SM
issue/replay/system state, trace reader state, semantic boundaries, L1/L2
models, pipeline/locality metrics, and direct-run reports. This layout change
has no serialized-schema or timing behavior change. The authoritative prose is
now in `RESULTS.md`, `RESIDUAL_FRONTIER.md`, `STATUS.md`, `ARCHITECTURE.md`,
and `COMPONENTS.md`; this file retains the chronology and rejected alternatives.

Fresh full 12-worker replays after modularization produced the required SM120
and SM86 status profiles. `FULL_NUMERIC_COMPARISON_MODULARIZATION.json` reports
zero missing/extra rows, zero required metadata differences, and zero common
numeric differences against the M7 closure artifacts. Both locality validators
reported zero errors.

## Genuine Rust module conversion (2026-10-20)

The earlier lexical/include layout was replaced with real Rust child modules.
The ten parent facades now declare `mod` children with explicit path attributes
where direct per-file rustfmt requires them, and public APIs are preserved by
re-exports. Large inherent implementations are split across focused child
modules; trait implementations remain intact in one module to avoid conflicting
trait impls. No `include!` macros or `part_*.rs` files remain.

A fresh SM120/SM86 12-worker full replay after this conversion matched the M7
closure artifacts with zero missing/extra rows, zero metadata differences, and
zero common numeric differences. Both reuse validators reported zero errors.
`FULL_NUMERIC_COMPARISON_REAL_MODULARIZATION.json` and
`FULL_NCU_SUMMARY_REAL_MODULARIZATION.json` record the parity evidence.

## Descriptive module hierarchy closure (2026-10-21)

The remaining suffix-split and pass-through files were replaced with cohesive
names: `l1/cache_state.rs` and `cache_ops.rs`, `system/event_loop.rs` and
`event_helpers.rs`, `issue/request_types.rs` and `core_state.rs`, and named
SM86 pipeline projection modules. Parent facades now declare direct children;
there are no `_a.rs`, `_b.rs`, `part_*.rs`, or textual `include!` fragments.

The release workspace passes direct per-file `rustfmt` and
`RUSTFLAGS=-D warnings cargo check --workspace --all-targets`. Fresh 12-worker
full `l1-l2` replays over 339 SM120 captures (11,558 launches) and 356 SM86
captures (12,392 launches) match the retained closure artifacts with zero
common numeric differences; reuse/MRC validation reports zero errors.
The detailed parity record is `FULL_NUMERIC_COMPARISON_DESCRIPTIVE_REFACTOR.json`.
