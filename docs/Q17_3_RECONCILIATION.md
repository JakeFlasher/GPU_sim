# Q17.3 reconciliation

**Updated:** 2026-10-11  
**Branch:** `GPUSim_main`  
**Simulator:** `/root/GPU_simulator_rebase`  
**Canonical corpus:** `/root/sm120_rtx_5090_capture`

## Scope and evidence rule

The SM120 alpha lane is evaluated on the 327 `recaptured_zero_exit` rows in
the final campaign audit. Five superseded duplicates are excluded and twelve
extra controls remain outside the scored population. Full replay is primary;
the fast frontier is complementary. NCU rows are joined after replay and are
never replay or model inputs. No fixture, hash, checking, validation,
verification, admission, manifest, or synthetic acceptance workflow is used.

## Q17.3.1–Q17.2 population closure

Canonical discovery is relative-path based and recursive. It includes the 37
nested FlashInfer/SOL/Quant rows omitted by the earlier direct-child run. The
canonical queue contains 327 captures, 339 recursive trace roots including
controls, and 11,529 launches.

## Q17.3.3 typed sidecar join

`sim/sm/src/replay.rs` streams `extra_info/tma_trace.jsonl` and
`extra_info/sync_trace.jsonl`, filtering by device and kernel and joining the
producer records by CTA, warp, instruction ordinal, PC, and function identity.
The enclosing trace launch supplies stream identity; the sidecar schema does
not carry a stream column. Process-local maps avoid rescanning a sidecar for
each launch in one `sim-run` process.

The TMA join uses the observed `byte_count` for the logical async completion
token. A missing record produces zero logical payload work and an explicit
missing counter; it does not fabricate the former universal 8-KiB payload.
The output separates loads, stores, zero-byte transfers, and
`oob_or_empty_transfer` rows. Sync rows are observation-only counters for
mbarrier address/register operands, before/after phase, arrivals, phase
checks, TRYWAIT waits, exchanges, incomplete values, and distinct barrier
addresses. Neither sidecar supplies a native engine completion timestamp or a
destination-visibility instant.

The latest canonical typed-sidecar full run observed:

| observation | count |
|---|---:|
| TMA records | 1,932 |
| TMA loads / stores | 675 / 1,257 |
| TMA payload bytes | 11,242,496 |
| TMA zero-byte/OOB rows | 336 / 336 |
| sync rows | 402,208 |
| sync before / after rows | 272,024 / 130,184 |
| sync arrivals / phase checks / waits | 5,458 / 390,552 / 202,056 |
| distinct barrier addresses | 136,062 |
| missing TMA or sync instruction joins | 0 / 0 |

These counters are serialized in `TraceMetrics` and are deliberately separate
from NCU comparison and timing-domain metrics.

## Q17.3.4 canonical replay

The current full replay is being regenerated after the sidecar cache and
replay-cost instrumentation change. The prior typed-sidecar run completed
327/327 captures and 11,529 launches. The matching fast replay is run only
after that full run completes. Raw rows are retained under `/dev/shm` as
`sm120_alpha_q17f_canonical_{full,fast}.jsonl`.

## Q17.3.5–Q17.3.6 NCU and timing scope

The canonical NCU comparison has one row per capture profile and zero grid or
block-shape mismatches. Launch identity is exact for 97 rows; the remaining
11,432 joins use an explicitly labelled ordered/shape fallback. No rows are
unmatched. Full and fast timing domains remain model-relative frontiers and
are not substituted for bare-metal CUDA-event timing. The independent NCU
comparison is recorded at
`/dev/shm/rust_research/q17e_canonical_ncu_comparison.json`.

## Q17.3.7 model decision

The independent endpoint control envelope separates a launch/intercept domain
from body-dependent GPC/SM-active deltas, but it does not identify a portable
SM120 route, service, or scheduler coefficient. No NCU-derived coefficient or
post-hoc offset is admitted. The correct result for this slice is to retain
the existing model and its typed residuals rather than fit a scored workload.

## Q17.3.8 replay cost

`sim-run` exposes total and replay wall time; the performance-counters feature
also reports setup, trace decode, execution metadata, control, schedule,
replay, serialization, and process wall phases. `TraceMetrics` records
`sidecar_join_wall_ns` as a host-side diagnostic. `/usr/bin/time` measurements
retain RSS separately from simulator timing. The sidecar map cache is a
bounded process-local optimization: it reduces repeated JSONL scans within one
capture, while the largest observed non-canonical grouped-GEMM sidecars used
approximately 280 MiB peak RSS. No model timing or physical interpretation is
changed by this optimization.

## Remaining genuine gaps

The following are not silently relabelled as complete: native TMA/mbarrier
completion timestamps, destination visibility instants, proprietary SM120
issue-port/warp-selector wiring, cache hash/geometry/MSHR/DRAM policy, and
bare-metal CUDA-event numerators for captures that were not collected with an
uninstrumented event scope. Future cache/scheduler/async observations must be
new full-workload evidence and must not use NCU targets as tuning inputs.
