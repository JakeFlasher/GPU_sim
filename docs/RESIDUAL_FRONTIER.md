# Accuracy residuals and information boundaries

This page is the single post-M7 residual summary. It is not a chronological
work log; superseded measurements are retained in [`HISTORY.md`](HISTORY.md).
NCU remains an independent comparison target.

## Current residual table

| Lane / endpoint | Current Mean APE | Interpretation |
|---|---:|---|
| SM120 elapsed IPC | 13.60% | launch-envelope and completion-tail semantics remain partially observed |
| SM86 active IPC | 18.56% | runtime occupancy and memory-stall endpoints are incomplete |
| SM86 elapsed IPC | 11.88% | launch-envelope and retirement-tail endpoint are incomplete |
| SM86 unified L1, ordered fallback | 45.61% | join scope is not identity-safe across multi-launch captures |
| SM86 unified L1, exact identity | 9.87% | valid identity subset; all-scope value is not comparable |
| SM120 DRAM read-sector total | 49.75% | NCU client attribution is broader than trace-visible read service |
| SM86 DRAM read-sector total | 24.83% | same client-scope limitation |

The main cache endpoints are otherwise stable: SM120 global L1 hit/miss are
1.5567%/0.5211%, unified L1 is 1.6386%, operation-read L2 is 6.6859%, and
generic TEX-L2 is 17.7364% with its conditional interval. SM86 operation-read
and generic L2 are 4.3416% and 8.5239%.

## Four physical information barriers

1. **Launch-resource sidecar.** The traces do not retain the driver-selected
   carveout/effective cache capacity and all occupancy inputs for every launch.
   A typed sidecar is required; fitting NCU values is not a substitute.
2. **Dynamic synchronization/TMA linkage.** `SYNCS` records expose PC, masks,
   and identity but not barrier address, transaction count, phase, TRYWAIT
   result, or TMA completion linkage. Static opcode names cannot determine the
   completion chronology.
3. **Disaggregated PMU clients.** NCU DRAM and generic L2 totals combine
   clients, promotions, writeback, and partition traffic that the trace does
   not label independently.
4. **Cross-surface launch identity.** Several SM86 profiles contain one NCU
   launch while a capture contains multiple launches. Ordered fallback is
   deterministic bookkeeping, not physical identity.

## Scope and acceptable next evidence

The effective-model closure criterion accepts conditional, uncertainty-aware
projections when the trace contract cannot identify a proprietary mechanism.
No residual may be reduced by clamping, count balancing, kernel-name rules, or
NCU fitting. A future release can narrow these rows only with the four typed
sidecars/crosswalks above, followed by a complete 12-worker full replay and a
no-regression comparison against this page.

Report-only HOTL/MRC observers are not a replacement for physical cache state.
They are valid because their reference stream, units, boundary conditions, and
error against the exact Fenwick reference are explicit (SM120 0.8767% line /
0.4462% sector WAPE; SM86 1.0223% / 0.5324%).

## Reproduction

Use the closure artifacts and `tools/validate_reuse_mrc.py`. Compare any future
release with `compare_numeric_full.py`; require zero missing/extra rows, zero
metadata differences, and zero common numeric differences before interpreting
an accuracy change. Full replay uses compressed `.jsonl.zst` output and twelve
workers; fast replay is not an acceptance result.

## Evidence discipline

The residual table distinguishes a missing physical input from an inaccurate
implementation. A future sidecar must identify its launch, units, and scope;
otherwise a new value cannot be compared to the current baseline. Static opcode
classification does not provide a completion timestamp, client partition, or
replacement decision.

The current result therefore supports the claim that the simulator is
reproducible and accurate on the identified endpoints while making the
unidentified endpoints explicit. It does not claim full proprietary hardware
recovery, and it does not hide ordered-fallback rows behind an all-scope mean.
