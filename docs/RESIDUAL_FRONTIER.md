# Accuracy residual frontier

This document records the current post-replay frontier after the accepted
SM86 per-SM write-tag observer correction. It is evidence, not a tuning plan.
NCU values remain independent comparison targets and never enter replay or
model selection.

## Complete replay evidence

All runs below are full `l1-l2` replay with twelve workers and carded DRAM.

| Lane | Result | Mean APE examples |
| --- | --- | --- |
| SM86, 286 successful captures | 286 `ok`, 62 `control_error`, 8 `replay_error`, 12,392 launches | unified L1 **24.66%**; exact joins **9.87%**; ordered fallback **45.61%**; active IPC **18.56%**; elapsed IPC **11.88%**; DRAM sectors **24.83%**; generic L2 **8.52%**; read L2 **4.34%** |
| SM120, 339 captures | 339 `ok`, 11,558 launches | unified L1 **58.01%**; global L1 **39.32%**; generic L2 **24.33%**; read L2 **14.89%**; active IPC **9.45%**; elapsed IPC **13.60%** |

The SM86 per-SM observer change is report-only and leaves IPC, L2, DRAM, and
timing numerics unchanged. A second complete SM86 replay matched all numeric
fields except wall-clock fields; a fresh SM120 replay after the change was
339/339 `ok` and unchanged. Machine-readable comparisons are
`FULL_NUMERIC_COMPARISON_PER_SM.json` (repeat determinism) and
`FULL_NUMERIC_COMPARISON_PER_SM_BASELINE.json` (intentional observer deltas).

## Occupancy metadata audit

Gemini proposed using the captured `driver_occupancy_limit_blocks` as an
NCU-independent per-launch CTA ceiling. The trace contract already carries
this driver-computed field, so a bounded experiment was implemented and run
on the complete SM86 corpus. The experiment preserved the required status
profile, but was a timing no-op: across all launches, the baseline traced
`peak_live_warps` never exceeded
`driver_occupancy_limit_blocks * ceil(block_threads / 32)`. The full candidate
therefore produced bit-identical IPC, elapsed/active frontiers, L1/L2, and DRAM
metrics. This falsifies occupancy-cap admission as an available improvement
under the current trace placement: the trace already records a schedule whose
observed peak respects the driver ceiling.

The check is reproducible with `tools/audit_driver_occupancy.py`; the latest
run reports 12,392 launches, zero observed peaks above the driver limit, and
`no_op: true`.

## Remaining information barrier

The residual metrics require state absent from the current contract rather than
an untested scalar:

* SM120 L1/L2: per-launch carveout/effective capacity, set/slice hash,
  replacement and fill-buffer policy, and cross-SM/TMA chronology.
* SM120 elapsed IPC and SM86 IPC: launch envelope/tail endpoint semantics,
  exact CTA residency and memory-stall frontiers, including runtime resource
  state not represented by traced issue order.
* DRAM-client totals: disaggregated non-SM/TMA/partition client scope and
  promotion/writeback behavior.
* SM86 ordered-fallback rows: NCU profiles intentionally contain one profiled
  launch while captures contain multiple launches; the current first-launch
  shape match is deterministic, but no cross-surface kernel identity is
  retained for all rows.

Changing these endpoints from existing trace bytes would require inventing
metadata or fitting NCU labels. The smallest unblockers are a launch-resource
sidecar with driver-selected cache/occupancy state, dynamic TMA coordinate and
issue-order records, disaggregated tag/MSHR/fill/stall PMU counters, and a
cross-surface NCU launch identity. Until those exist, the remaining <10%
objective is not honestly closable without violating the no-fitting and
report-only boundaries.

## Newly verified SM120 synchronization gap

The permitted SM120 capture metadata contains a detailed synchronization/TMA
audit (`metadata/adversarial_sync_tma_gap_20261008.md` in the read-only source
checkout). It confirms that dynamic `SYNCS` records retain only PC, masks, and
identity: barrier addresses, transaction counts, phase/state values, and
TRYWAIT results are absent. TMA records likewise lack explicit completion-
barrier linkage. The audit traces this to the producer's `SYNCS` raw-SASS
fallback, which clears operand instrumentation. Consequently, the simulator
cannot reconstruct mbarrier/TMA completion chronology from current inputs;
changing SM120 elapsed IPC, TMA DRAM demand, or cache state from static opcode
names would invent timing/state. The minimum unblocker is a typed `sync_trace`
sidecar carrying runtime barrier address, count/transaction count, phase/state
encoding, TRYWAIT result, memory scope/order, and deterministic per-launch
sequence, plus completion linkage in TMA records.
