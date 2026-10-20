# Unified GPU simulator: current evaluation

This is the authoritative evaluation record for the merged simulator. Superseded
campaigns and implementation diaries belong in [`HISTORY.md`](HISTORY.md), not
in this result table.

## Corpus and replay method

| Lane | Capture disposition | Launches | Replay configuration |
|---|---:|---:|---|
| SM120 / RTX 5090 | 339/339 `ok` | 11,558 | full integrated `l1-l2`, carded DRAM, 12 workers |
| SM86 / RTX 3090 Ti | 286 `ok`, 62 `control_error`, 8 `replay_error` of 356 attempted | 12,392 | full integrated `l1-l2`, card-selected inputs, 12 workers |

Only compressed `.jsonl.zst` full-replay artifacts are used for accuracy claims.
The retained closure artifacts are
`full_replay_runs/cache_locality_m7closure_sm120/full.jsonl.zst` and
`full_replay_runs/cache_locality_m7closure_sm86/full.jsonl.zst`. A second
12-worker academic-release replay produced the same row identities, metadata,
and numeric fields: zero missing rows, zero extra rows, zero metadata
 differences, and zero common numeric differences in either lane. Wall-clock
process times are not comparison fields.

NCU profiles are independent post-replay targets. They are joined by exact
capture/launch identity when available and by a deterministic ordered fallback
only where the capture contract has no identity. NCU values never select a
model, tune a parameter, or enter a replay input.

## Hardware-counter validation

Mean APE is the arithmetic mean of `abs(prediction-target)/abs(target)` over
nonzero targets. Zero-target false positives are reported separately. WAPE is
reported for the locality estimators below; the NCU comparison tool publishes
Mean APE as its primary hardware-counter error statistic.

| Lane / metric | NCU counter | Join population | Mean APE | WAPE | Zero-target FP |
|---|---|---|---:|---:|---:|
| SM120 global L1 lookup hit | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` | all 11,558 matched launches | 1.5567% | not reported | 0 |
| SM120 global L1 lookup miss | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum` | all 11,558 matched launches | 0.5211% | not reported | 0 |
| SM120 unified L1 sector hit rate | `l1tex__t_sector_hit_rate.pct` | all 11,520 target-bearing launches | 1.6386% | not reported | 7 |
| SM120 operation-read L2 lookup hit | `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum` | scope-compatible joins | 6.6859% | not reported | 5 |
| SM120 generic TEX-L2 lookup hit | `lts__t_sectors_srcunit_tex_lookup_hit.sum` | point estimate, scope-compatible joins | 17.7364% | 7.77% (closure report) | 51 (<=55 gate) |
| SM86 operation-read L2 lookup hit | `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum` | scope-established joins | 4.3416% | not reported | 2 |
| SM86 generic TEX-L2 lookup hit | `lts__t_sectors_srcunit_tex_lookup_hit.sum` | scope-established joins | 8.5239% | not reported | 0 |
| SM86 unified L1 sector hit rate | `l1tex__t_sector_hit_rate.pct` | exact identity joins | 9.87% | not reported | 24.6597% when all scope-compatible joins are shown |
| SM120 active IPC | `sm__inst_executed.avg.per_cycle_active` | ordered/exact matched launches | 9.45% | not reported | — |
| SM120 elapsed IPC | `sm__inst_executed.avg.per_cycle_elapsed` | ordered/exact matched launches | 13.60% | not reported | — |
| SM86 active IPC | `sm__inst_executed.avg.per_cycle_active` | ordered/exact matched launches | 18.56% | not reported | — |
| SM86 elapsed IPC | `sm__inst_executed.avg.per_cycle_elapsed` | ordered/exact matched launches | 11.88% | not reported | — |
| SM120 DRAM read sectors | `dram__sectors_op_read.sum` | all matched launches | 49.75% | not reported | 0 |
| SM86 DRAM read sectors | `dram__sectors_read.sum` | 165 nonzero-target joins | 24.83% | not reported | 0 |

The SM120 generic-L2 point estimate is accompanied by a conditional,
trace-derived store-packet/L2 interval. It covers 11,483/11,505 compared
launches (99.81%) and 4,082/4,104 nonzero-target launches (99.46%); mean
interval-distance error is 0.4530%. This interval is an uncertainty report,
not a claim of recovered proprietary cache geometry.

## Trace-locality and HOTL validation

The exact reference is `sector_mrc_observer_v1`, a within-warp Fenwick
stack-distance/Mattson miss-ratio curve. `reuse_interval_observer_v1` applies
Ding--Denning/Xiang footprint inversion to the same reference streams. Both
are report-only and do not mutate L1, L2, DRAM, or timing state.

| Lane | Line WAPE | Sector WAPE | Invariant violations | Same-tick global envelope width |
|---|---:|---:|---:|---:|
| SM120 | 0.8767% | 0.4462% | 0 | 3.07 percentage points |
| SM86 | 1.0223% | 0.5324% | 0 | 2.52 percentage points |

WAPE is `sum(abs(estimate-reference))/sum(reference)` over finite MRC points;
zero-reference points are excluded from the denominator. The validator also
checks monotonicity, finite-boundary tail accounting, and line/sector
conservation.

## Mechanisms and scope boundaries

* **M1:** L1 lookup accounting distinguishes same-request merges from older
  pending-sector merges and records both without changing service timing.
* **M2:** SM120 uses a per-SM write-tag lookup observer and a CCTL-qualified
  store-to-load forwarding projection; the observer is report-only.
* **M3/M7:** L2 snapshots the pre-packet 128-byte tag and separately reports
  resident-sector coalescing. The conditional packet envelope exposes the
  unresolved client/interleaving range without changing physical L2 state.
* **M4/M6:** HOTL footprint inversion and coupled sector-bottleneck intervals
  estimate trace-local MRCs; Tier-2/Tier-3 observers remain shadow-only.
* Timing, residency, cache mutation, DRAM service, and IPC remain in the
  active event model. Post-replay attribution never feeds NCU values back into
  those mechanisms.

## Reproduction and release status

Run `cargo build --release --manifest-path sim/Cargo.toml --workspace` and
`cargo test --release --manifest-path sim/Cargo.toml --workspace`. The portable
replay recipe is in [`REPLAY.md`](REPLAY.md); use `--mode full --workers 12`.
Run `tools/validate_reuse_mrc.py` on both closure artifacts. The academic
release modularization preserves the closure numerics exactly; no metric was
altered by documentation or source-layout changes.

## Join and error conventions

`Exact Identity` means the capture and NCU row share the recorded launch
identity. `Ordered Fallback` is used only when that identity is absent and the
kernel/grid/launch order is deterministic; it is never mixed silently with
identity rows. `All Matched` is an explicitly labelled union used for corpus
coverage, not a claim that every row has the same physical scope.

For a nonzero target `y`, the row error is `|p-y|/|y|`. A zero target is not
placed in that denominator: `p != 0` is a false positive and `p == 0` is an
exact zero. Counter units are sectors, cycles, instructions, bytes, or
percentages as named in the NCU column; no ratio is converted to a count by
using an unrelated row.

## Determinism and artifact checks

The parity comparison checks status, capture identity, launch index, kernel
metadata, and every numeric field shared by the baseline and release outputs.
It intentionally ignores wall-clock duration fields. The reuse validator
checks MRC monotonicity, finite-boundary tails, line/sector conservation, and
zero structural errors. These checks are implementation checks, not sources
of hardware truth.

The Rust modularization is a source-layout change only: lexical child sections
are included in their original parent scope, so private fields, checked
arithmetic, serde names, and error paths remain unchanged. The release build
and test commands above are run after formatting. Any future behavioral change
must repeat both full lanes before it is accepted.

## Interpretation

The low-error L1 and operation-read rows are direct counter projections. The
17.7364% generic TEX-L2 point estimate is retained because it is the best
non-fitting arm under the no-regression rule; its interval is the scientifically
valid statement where client scope is unidentifiable. The HOTL table is a
trace-local estimator validation, not an NCU cache-hit claim.
