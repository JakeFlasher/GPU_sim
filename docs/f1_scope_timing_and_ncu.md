# F1 — scope-aligned timing and NCU comparison seam

**State:** F1.1–F1.5 implemented in the alpha driver seam.  
**Scope inventory:** `/dev/shm/sm120_rtx_5090_work/f1_scope_inventory.json`  
**Real replay example:** `/dev/shm/sm120_rtx_5090_work/f1_scope/replay.jsonl`

The scope inventory reports 12 retained captures / 29 launches, with 11
captures covered by the independent `m11_hardware_timing.json` contract and
the smoke capture explicitly `unknown`.  The driver now records per-capture
input preparation, simulator process, record assembly and JSONL serialization
boundaries. The simulator's own `sim_wall_ns_replay` remains the denominator
when present; the fast-frontier process boundary is used only when its report
has no internal replay timer.

NCU joins continue to prefer kernel identity plus launch ID, with ordered
fallback labelled diagnostic. Metric rows retain raw units and now expose
`absolute_relative_error`, zero-reference status, and missing/unit-mismatch
states. NCU duration remains profiler-perturbed observation and is never
injected into trace-only prediction.

Initialization, warm-up and epilogue labels remain unknown when the retained
source command has no explicit region marker. Tracer stdout is retained as a
separate instrumented observation; it is not used as the bare-metal numerator.
