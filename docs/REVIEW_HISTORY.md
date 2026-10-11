# Current implementation history

This file is a compact live history, not an approval, audit, provenance, or
execution gate. Detailed evidence belongs beside the corresponding run output.

## 2026-10-10 — SM120 alpha execution slice

- Replaced the prior workstation-SKU card narrative with an explicit RTX 5090
  `sm_120` card and aliases.
- Added SM120 trace-contract/address handling, compact sector metadata, and
  cache-operator retention without treating cache operators as timing policy.
- Added the full-replay SM120 alpha lane and a separate fast-frontier diagnostic;
  outputs identify behavioral approximations and typed gaps.
- Added launch attribution, CUDA-event/replay-wall timing, independent NCU
  joins, direct address/locality features, logical memory service, async state,
  and dependency/front-end features.

## 2026-10-11 — fresh evidence and reconciliation

- Corrected compressed-address/MRC denominators and added lane-aware fixed
  register RAW/WAW readiness to full replay.
- Collected fresh route, store, atomic, RED, cp.async, scheduler, cluster, and
  TMA observations on the RTX 5090. Full replay remained primary; fast replay
  remained complementary.
- Reconciled Q10–Q15: placement and multiwarp slices are behavioral evidence;
  RED/control and `.ca`/`.cg` residuals did not justify new physical terms;
  process-start NCU collection corrected the earlier profiler-range mismatch.
- Observed an 8,192-byte payload after synchronization and cluster ranks 0/1,
  while retaining native completion timestamps as unavailable.

## State reconciliation commit

The state-reconciliation commit records the implementation, card/configuration updates, live
entrypoint rewrite, and removal of superseded narrative/control files from the
checkout. Raw evidence and retained measurement sources were left in place.

## Current continuation

Q16.1 is active: split TMA load/store payload observations from mbarrier phase
observations in replay output. The next updates must preserve the distinction
between source observation, model ledger, and native completion claim.
