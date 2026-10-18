# SM86 unresolved mechanism gap tracker — 2026-10-16

This ledger is intentionally not a closure checklist. It records remaining
uncertainties after H59 and the evidence needed to reduce them.

## Accuracy gaps

- **L1 raw/all-row:** global-load sectors 100.96% mean APE, dominated by a
  form/scope outlier; secondary scope-compatible view 0.21%. U16/LTC128B
  lookup semantics remain unresolved.
- **L2 raw/all-row:** source sectors 28.24%, lookup hits 35.04%, lookup misses
  27.00%. Secondary scope-compatible misses are exact (0%); hits remain 11.77%
  and concentrated-route joins are sparse.
- **DRAM raw/all-row:** read sectors 40.91%; secondary scope-compatible 26.84%
  over 58 joins. Demand intervals bracket only 58.6% of these targets.
- **Cycles/IPC:** GPC 12.90%, SM-active 18.02% on exact joins. Route/tail
  strata are confounded and high-grid envelope groups are sparse.

## Typed tails and scope gaps

- Dirty writeback and coherence tails remain unmodeled on the cycle/DRAM
  surfaces; the bounded writeback scenario is mixed and remains opt-in.
- 125 scope warnings and 67 scope-mismatch exclusions are retained in
  compressed reports. Raw/all-row residuals remain authoritative; exclusions
  are unresolved scope/model gaps, not completed mechanisms.
- Complete corpus coverage is 356 roots (286 `ok`, 62 controls, 8 failures),
  12,392 launches, 107 exact NCU identity joins; ordered fallback joins are
  metadata only.

## Required next evidence

1. Scope-matched NCU metric sets for LDG forms, L2 source sectors, and DRAM
   reads/writes.
2. Locked form-specific LDG plain/U16/LTC128B probes with reuse/warp sweeps,
   including a valid LTC128B control.
3. Locked DRAM demand probes varying one factor at a time with writeback and
   coherence completion accounted independently.
4. Larger independent route/queue probes before any partitioned timing arm.

No current gap is silently excluded, renamed, or declared physically recovered.
