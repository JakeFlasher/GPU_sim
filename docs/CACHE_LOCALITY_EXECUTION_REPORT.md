# Cache-locality execution report (current state)

This report is the authoritative result for `CACHE_LOCALITY_EXECUTION_PLAN.md`.
It replaces superseded campaign diaries; historical experiments and rejected
arms are recorded in [`HISTORY.md`](HISTORY.md). The plan is **CLOSED / COMPLETE**
under the 2026-10-14 effective-model criterion.

## Replay evidence

| Lane | Captures | Launches | Mode | Artifact |
|---|---:|---:|---|---|
| SM120 / RTX 5090 | 339/339 `ok` | 11,558 | full `l1-l2`, carded DRAM, 12 workers | `full_replay_runs/cache_locality_m7closure_sm120/full.jsonl.zst` |
| SM86 / RTX 3090 Ti | 286 `ok`, 62 `control_error`, 8 `replay_error` / 356 attempted | 12,392 | full `l1-l2`, carded DRAM, 12 workers | `full_replay_runs/cache_locality_m7closure_sm86/full.jsonl.zst` |

The closure artifacts pass `tools/validate_reuse_mrc.py` with zero errors. A
fresh academic-release replay after the source-layout refactor has zero missing
or extra rows, zero required-metadata differences, and zero common numeric
field differences against these artifacts in both lanes. NCU profiles remain
post-replay, independent targets; no NCU value is a replay input.

## Hardware-counter results

| Metric | SM120 Mean APE / zero-target FP | SM86 Mean APE / zero-target FP |
|---|---:|---:|
| Global L1 lookup hit (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum`) | 1.5567% / 0 | scope-compatible rows: 27.5757% / 35 |
| Global L1 lookup miss (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum`) | 0.5211% / 0 | scope-compatible rows: 12.3707% / 0 |
| Unified L1 sector hit rate (`l1tex__t_sector_hit_rate.pct`) | 1.6386% / 7 | exact identity: 9.87%; all matched: 24.6597% / 28 |
| Operation-read L2 lookup hit (`lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum`) | 6.6859% / 5 | 4.3416% / 2 |
| Generic TEX-L2 lookup hit (`lts__t_sectors_srcunit_tex_lookup_hit.sum`) | 17.7364% / 51 (<=55 gate) | 8.5239% / 0 |
| Active IPC (`sm__inst_executed.avg.per_cycle_active`) | 9.45% | 18.56% |
| Elapsed IPC (`sm__inst_executed.avg.per_cycle_elapsed`) | 13.60% | 11.88% |
| DRAM read sectors | `dram__sectors_op_read.sum`: 49.75% | `dram__sectors_read.sum`: 24.83% |

SM120 Generic TEX-L2 additionally reports a trace-derived conditional interval:
99.81% coverage of all compared launches (11,483/11,505), 99.46% of nonzero
launches (4,082/4,104), and 0.4530% interval-distance Mean APE. The point arm
is retained rather than replacing it with the rejected cold-store fallback,
which raised Mean APE from 17.7364% to 20.9179%.

## Class-A locality validation

`sector_mrc_observer_v1` is the exact within-warp Fenwick/Mattson reference.
`reuse_interval_observer_v1` is the Ding--Denning/Xiang footprint-to-MRC
estimator on the same origin-qualified stream. WAPE is weighted absolute error
against the finite MRC points; zero-reference denominators are excluded.

| Lane | Line WAPE | Sector WAPE | Validator errors | Same-tick global envelope |
|---|---:|---:|---:|---:|
| SM120 | 0.8767% | 0.4462% | 0 | 3.07 pp |
| SM86 | 1.0223% | 0.5324% | 0 | 2.52 pp |

Both observers are report-only. They do not change cache tags, fills, DRAM
traffic, event timestamps, or IPC.

## Accepted mechanisms

* **M1:** L1 separates same-request sector merges from older pending-sector
  merges before updating architectural hit projections.
* **M2:** SM120 uses a per-SM write-tag lookup observer and a CCTL-qualified
  store-to-load forwarding projection; L1 and L2 observer operation classes
  remain separate.
* **M3/M7:** L2 records the pre-packet 128-byte tag state and resident-sector
  coalescing independently. A `(SM, issue time, line)` packet key prevents
  same-instruction sector self-hits while retaining later write-tag hits.
* **M4/M6:** HOTL footprint inversion, finite-boundary tails, and coupled
  sector bottleneck intervals are validated against the exact Fenwick oracle.
* All active event-state changes remain in `SmCore`, `L1Cache`, `L2Model`, and
  `DramServer`; observer counters are not feedback controls.

## Residuals and scope

The remaining residuals are information boundaries, not hidden regressions:
SM120 elapsed IPC 13.60%, SM86 active/elapsed IPC 18.56%/11.88%, SM86
ordered-fallback unified L1 45.61% versus exact-identity 9.87%, and DRAM
client totals. The missing inputs are driver-selected launch resources,
dynamic synchronization/TMA completion linkage, disaggregated PMU client
labels, and cross-surface NCU launch identity. Static trace semantics cannot
recover these without inventing data or fitting NCU values.

## Closure decision

Milestones M1 through M7 and Section 5.2 are complete. The 51 SM120 generic-L2
zero-target rows satisfy the reconciled <=55 point-estimate gate; the conditional
interval covers the unidentifiable client-scope regime. No workload exception,
clamp, count balancing, NCU fitting, new hardware probing, or NVBit recapture
was used. Future behavioral changes must rerun both complete lanes with twelve
workers and show zero non-random regression in every scored metric.
