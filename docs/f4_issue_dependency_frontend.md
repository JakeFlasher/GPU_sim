# F4 — issue/dependency/front-end behavioral surface

**Output:** `/dev/shm/sm120_rtx_5090_work/f4_dependency_features_all.json`

`tools/sm120_dependency_features.py` parses the retained native static tables,
counts conservative RAW/WAW/WAR candidates, control stalls and dependency
barrier masks, and reports tensor/memory/branch/synchronization/scalar classes
plus static instruction-footprint density. The 12-capture campaign completes
without using NCU targets. RFU-unavailable operands are counted as ambiguous;
the result is not a scoreboard or physical issue-port claim.

The Rust alpha scheduler already consumes explicit resource classes,
control-stall fields and asynchronous group state. The new
`sm120_fast_stage_event_v1` path prices those classes for a direct trace
frontier; full replay remains the detailed fallback. Dependency candidates and
front-end footprint are exposed as model inputs for future microbench sweeps,
not substituted for a hardware scheduler trace.
