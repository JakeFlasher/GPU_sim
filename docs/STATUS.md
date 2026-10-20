# Unified simulator status

## Authoritative state

The merged repository supports two independent architecture lanes:

| Lane | Card | Full-corpus status |
|---|---|---|
| SM120 / RTX 5090 | `arch_profiles/cards/sm_120-rtx-5090.json` | 339/339 captures `ok`, 11,558 launches |
| SM86 / RTX 3090 Ti | `arch_profiles/cards/sm_86.json` | 286 `ok`, 62 `control_error`, 8 `replay_error`, 12,392 launches of 356 attempted |

Both lanes use full integrated `l1-l2` replay with twelve workers and compressed
`.jsonl.zst` artifacts. SM120 DRAM-disabled replay is historical and is not a
required acceptance lane. Fast replay is diagnostic only.

## Verified closure

Milestone 7 and `CACHE_LOCALITY_EXECUTION_PLAN.md` are **CLOSED / COMPLETE**
under the 2026-10-14 effective-model criterion. The closure artifacts are
`full_replay_runs/cache_locality_m7closure_sm120/` and
`full_replay_runs/cache_locality_m7closure_sm86/`.

SM120 final counter accuracy: global L1 lookup hit 1.5567% Mean APE (0 FPs),
global L1 lookup miss 0.5211% (0), unified L1 1.6386% (7), operation-read L2
6.6859% (5), and generic TEX-L2 17.7364% (51, within the <=55 conditional
point-estimate gate). The store-packet interval covers 99.81% of all and
99.46% of nonzero compared launches with 0.4530% interval-distance Mean APE.

SM86 final counter accuracy: operation-read L2 4.3416% Mean APE (2 FPs),
generic L2 8.5239% (0), and unified L1 9.87% on exact identity joins (the
ordered/all-scope value is explicitly not identity-safe). HOTL-vs-Fenwick WAPE
is 0.8767%/0.4462% for SM120 and 1.0223%/0.5324% for SM86, with zero
validator errors.

## Engineering state

The Rust workspace is modularized behind API-preserving facades. Large
mechanisms are stored as named Rust child modules so private fields,
serialized names, and impl visibility are unchanged. No `sim/**/*.rs` file
exceeds 2,500 lines. `docs/ARCHITECTURE.md`, `docs/COMPONENTS.md`, and
`docs/TERMINOLOGY.md` define the public codemap and vocabulary.

NCU values are joined after replay only. Effective models and conditional
intervals are not proprietary RTL recovery and never use NCU fitting. Trace
inputs and architecture cards remain the sole model inputs.

## Required checks

* `cargo fmt --manifest-path sim/Cargo.toml --all -- --check`
* `cargo build --release --manifest-path sim/Cargo.toml --workspace`
* `cargo test --release --manifest-path sim/Cargo.toml --workspace`
* `tools/validate_reuse_mrc.py` on both closure lanes
* `compare_numeric_full.py` against both closure artifacts

The release rule is strict: reject any change with a non-random regression in
any scored metric, status disposition, row identity, or independent-target
separation. Historical diaries and rejected experiments are in `HISTORY.md`.

The release is ready for source review and paper integration. Any change to a
cache observer, timing endpoint, card, or serializer must update the codemap,
rerun the full corpus, and preserve the current tables.
