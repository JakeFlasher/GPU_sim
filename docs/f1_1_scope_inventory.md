# F1.1 — launch-scope inventory

**Status:** complete (scope reporting seam)  
**Output:** `/dev/shm/sm120_rtx_5090_work/f1_scope_inventory.json`

The inventory reads the retained `stats.csv`, NCU row counts, tracer stdout,
and the independent uninstrumented timing contract. It records 12 captures and
29 launch rows. Eleven captures have a `coverage_full=true` timing declaration
with scope `kernel_cuda_event_sum_over_reported_launches`; the smoke capture
has no corresponding timing record and remains `unknown`. Initialization,
warm-up, and epilogue phases remain unknown unless a source-side marker exists.

The table is reporting-only. It is not passed to the predictor, and it does
not convert tracer stdout or NCU duration into bare-metal timing.
