# SM86 terminal-EXIT store-drain slice — H71 — 2026-10-16

## Independent RTX 3090 Ti evidence

The matched `store_ack_frontier_h71` binary was built with CUDA 13.2 and run on
 the locked RTX 3090 Ti. The no-fence arm (`store_none`) and explicit device
fence arm (`store_device`) used the same 84-block/depth-16/active-32,
line-unique workload. NCU reported:

| arm | `sm__cycles_active.max` | `smsp__cycles_active.max` | `smsp__inst_executed.sum` | `...stalled_drain.sum` | `...stalled_membar.sum` | DRAM read/write sectors |
|---|---:|---:|---:|---:|---:|---:|
| no fence / terminal EXIT | 10,345 | 10,327 | 80,808 | 253,415 | 0 | 160 / 81,992 |
| explicit `MEMBAR.GL` | 11,730 | 11,712 | 81,060 | 1,915 | 488,950 | 184 / 80,532 |

This isolates a stall-domain distinction: no-fence stores are not evidence that
stores are replyless at CTA `EXIT`; the hardware attributes the wait to the
drain domain, while an explicit fence attributes it to MEMBAR. The independent
store-ack matrix also shows shape-dependent no-fence tails (approximately
3,500 cycles at depth 16 and 11,400 at depth 64 for 84 blocks). These are NCU
observations only, never replay inputs or fitted targets.

## Bounded implementation

The SM86 memory-admission schema now accepts an explicit
`exit_waits_for_service_completion` flag. When enabled by an experimental
candidate, only a **terminal dynamic `EXIT`** is blocked while that warp owns an
unpriced STORE admission or a typed write acknowledgement. The existing event
queue wakes the actor after acknowledgement processing; no deferred-retirement
state, DRAM service, dirty-writeback, or coherence timestamp is invented.

Two causality safeguards were required by the real corpus:

* feedback acknowledgement service start is clamped to the L2 pricing event
  timestamp, preventing an acknowledgement from being scheduled in the past;
* a stale actor-local `SmWake` is rearmed at `max(local_wake, queue_now)` when
  a cross-SM event has already advanced the shared event clock.

The candidate remains **store-only**; no RED-only EXIT law is claimed. The
repository card keeps the field explicitly disabled (`false`) after the
full-corpus result below. The implementation is retained as an opt-in seam for
a future endpoint-matched experiment.

## Full-first real-corpus evaluation

Release `sim-run` was rebuilt and the complete 356-root corpus was replayed
with exactly twelve workers, full before fast, and compressed `.jsonl.zst`
artifacts.

* Experimental EXIT arm enabled: `../sm86_replay_results/full_corpus_h71_exit_drain_20261016_v2/`
  (286 `ok`, 62 `control_error`, 8 `replay_error`, 12,392 full launches).
* Default card with EXIT arm disabled: `../sm86_replay_results/full_corpus_h71_exit_drain_20261016_v3/`
  (same 286/62/8 coverage and 12,392 full launches).
* Reports: `../sm86_replay_results/metric_report_h71_exit_drain_20261016_v2/` and
  `../sm86_replay_results/metric_report_h71_exit_drain_20261016_v3/`.

The enabled-arm secondary scope-compatible mean APE changed relative to H70 as
follows:

| metric | H70 baseline | H71 arm enabled | delta |
|---|---:|---:|---:|
| `gpc__cycles_elapsed.max` | 12.9008% | 12.8836% | -0.017 pp |
| `sm__cycles_active.avg` | 18.0202% | 18.1389% | +0.119 pp |
| `sm__cycles_active.max` | 17.8377% | 17.8763% | +0.039 pp |
| `smsp__cycles_active.avg` | 17.5137% | 17.6580% | +0.144 pp |
| `sm__inst_executed.avg.per_cycle_active` | 14.4865% | 14.5529% | +0.066 pp |

The default v3 replay and report are bit-for-bit metric-baseline equivalent
to H70 for these scored surfaces (including the existing 12.9008% GPC,
18.0202% SM-active, 17.5137% SMsp-active, and 14.4865% IPC-projection APEs).
The H71 arm therefore does not meet the accuracy-improvement bar and is not
active in default full replay.

## Decision and next evidence

H71 is a **falsifiable negative for promotion**, not a claim that terminal
EXIT drain is absent. The independent stall-domain relation is real, but the
current typed acknowledgement/service frontier is too weak and endpoint
mismatched to reproduce the corpus SM-active residual without harming IPC.
Default full replay remains the comprehensive H70 model plus the causal queue
safeguards, with aggregate carded DRAM and typed dirty/coherence tails
unchanged.

A future promotion requires a same-binary NVBit crosswalk that maps terminal
EXIT/drain events to the exact store acknowledgement population, separates
STORE from RED, and supplies a held-out no-fence/fence workload matrix. NCU
profiles remain post-replay comparison targets only.
