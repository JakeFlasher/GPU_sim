# Unified simulator status

## Supported lanes

| Lane | Card | Runtime status |
| --- | --- | --- |
| SM86 / RTX 3090 Ti | `arch_profiles/cards/sm_86.json` | Full compatibility replay with card-backed memory inputs and scoped SM86 candidates. |
| SM120 / RTX 5090 | `arch_profiles/cards/sm_120-rtx-5090.json` | Full effective replay with typed geometry gaps, TMA/async sidecars, and optional carded DRAM. |

Both lanes use the architecture-neutral `tools/sm120_corpus_replay.py` entry
point and the same Rust workspace. Full-corpus results and expected
dispositions are in [`RESULTS.md`](RESULTS.md).

## Boundaries

NCU data is joined after replay and never selects a model. SM86 controls and
substitutes are not relabeled as SM120 features. Effective SM120 values are
conditional scenarios, not proprietary topology recovery. Unsupported behavior
remains an explicit scope limitation. Full replay must use `l1-l2`,
compressed `.jsonl.zst` outputs, and twelve workers; core-only fixed-latency
diagnostics are separate.

## Maintenance

Build/test commands and the portable replay recipe are in the root
[`README.md`](../README.md) and [`REPLAY.md`](REPLAY.md). This page is the
current status contract; historical experiment notes are not execution policy.
Public vocabulary and the release rewrite are maintained in
[`TERMINOLOGY.md`](TERMINOLOGY.md) and
[`ACADEMIC_RELEASE_PLAN.md`](ACADEMIC_RELEASE_PLAN.md).

## Cache-locality execution plan status (2026-10-19)

The plan was executed through a final fresh full replay. M1, M2A/M2B, M3,
and the report-only portions of M4 are implemented. Final artifacts are
`full_replay_runs/cache_locality_final2_sm120/` and
`full_replay_runs/cache_locality_final2_sm86/`; both use 12 workers and pass
`tools/validate_reuse_mrc.py`.

The exact `sector_mrc_observer_v1` Class-A oracle remains exact and the new
Tier-2/Tier-3 observer schemas are emitted without changing physical cache,
DRAM, IPC, or timing. The interval-derived HOTL approximation is **not** a
<1.5% oracle substitute (SM120 line/sector WAPE 20.6283%/19.1585%; SM86
16.7574%/17.8661%). This is an information-boundary residual, not a reason to
fit NCU or fabricate cross-warp order. The four ordered-fallback SM120
zero-target global-L1 rows likewise remain explicitly unresolved. Details and
commands are in [`CACHE_LOCALITY_EXECUTION_REPORT.md`](CACHE_LOCALITY_EXECUTION_REPORT.md).

A follow-up M2A/M2C correction is recorded in
`CACHE_LOCALITY_EXECUTION_REPORT.md`: SM120 unified-L1 endpoint APE is ~1.63%
and operation-read L2 is ~6.69% after using all active write-tag observations
and trace-qualified store-to-load forwarding. Generic source-TEX L2 remains a
scope residual (~23.97%); a large-store shape rule was tested across the full
corpus and rejected because it regressed generic-L2 WAPE.

## M6 replay status (2026-10-20)

Fresh full-only twelve-worker replays are in `full_replay_runs/cache_locality_m6r_sm120/` and `full_replay_runs/cache_locality_m6r_sm86/`. Both pass the reuse/MRC validator. HOTL WAPE is 0.8767%/0.4462% (SM120 line/sector) and 1.0223%/0.5324% (SM86 line/sector). CCTL-qualified SM120 Global-L1 false positives fell from four to zero. The generic TEX-L2 M6 store-packet/ST.E seam remains an evidenced blocker: full replay is 724.92% APE with 6,159 zero-target FPs versus final4's 23.97% and 117. No NCU fitting or kernel exception was added.

The observer-only ST.E follow-up keeps physical L2 class/timing unchanged while
using `preview_mask` for effective operation attribution. Its complete SM120
replay (`full_replay_runs/cache_locality_m6final_sm120/`) still reports
724.72% generic TEX-L2 APE / 6,158 zero-target FPs, so the client-scope blocker
is confirmed rather than hidden.

## M7 closure replay (2026-10-20)

M7's report-only store-packet split is implemented and verified with fresh
12-worker full replays in `full_replay_runs/cache_locality_m7closure_sm120/`
and `full_replay_runs/cache_locality_m7closure_sm86/`. SM120 is 339/339 `ok`
(11,558 launches); SM86 retains the expected 286 `ok`, 62 `control_error`, and
8 `replay_error` dispositions (12,392 launches). Both reuse validators are
clean, and the release workspace tests pass.

The SM120 Generic TEX-L2 point estimate is 17.7364% Mean APE with 51
zero-target FPs; the 724.72% M6 regression is fixed and the reconciled <=55
gate is satisfied. The conditional packet/L2 interval report covers 99.81%
of all compared launches and 99.46% of nonzero launches with 0.4530%
interval-distance Mean APE. No NCU fitting, recapture, or kernel exception
was used. See `docs/CACHE_LOCALITY_EXECUTION_REPORT.md` for the formal closure
evidence.

## Final cache-locality closure (2026-10-20)

**CLOSED / COMPLETE.** Milestone 7 and the entire cache-locality execution
plan are formally closed under the 2026-10-14 Modeling Closure Criterion.
The closure uses the existing verified 12-worker full replay artifacts only.
SM120 is 339/339 `ok` with 11,558 launches; SM86 is 286 `ok`, 62
`control_error`, and 8 `replay_error` with 12,392 launches.

SM120 final metrics are Global L1 hit 1.5567% APE (0 FPs), Global L1 miss
0.5211% (0), Unified L1 1.6386% (7), operation-read L2 6.6859% (5), and
Generic TEX-L2 17.7364% (51 <=55). Its conditional packet/L2 envelope covers
99.81% all and 99.46% nonzero launches with 0.4530% interval-distance APE.
SM86 operation-read L2 is 4.3416% (2 FPs), Generic L2 is 8.5239% (0 FPs), and
Unified L1 is 9.87% on exact identity joins (24.6597% across all joins).

Gemini's Regime-B audit found 567/568 (99.82%) positive write-tag launches
have NCU > 0; the one zero-target row is covered by `[0,16]`. The cold-store
fallback was rejected for worsening corpus accuracy. No NCU fitting, workload
exception, recapture, or additional replay was used.

## Academic release terminology pass (2026-10-20)

The simulator's human-facing terminology, metric definitions, history, and
persistent continuation notes were rewritten using conventional computer
architecture language. See `TERMINOLOGY.md`, `ACADEMIC_STYLE_REVIEW.md`, and
`ACADEMIC_RELEASE_PLAN.md`.

The refactor was verified with fresh full integrated replays using twelve
workers: SM120 339/339 `ok` and 11,558 launches; SM86 286 `ok`, 62
`control_error`, 8 `replay_error`, and 12,392 launches. Both reuse validators
pass, and `compare_numeric_full.py` reports zero common numeric differences
against the M7 closure artifacts. No performance metric or timing output
changed.
