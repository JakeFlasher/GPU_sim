# SM86 DRAM demand interval audit — 2026-10-15

H46 audits the existing trace-derived DRAM demand interval (L2-missing
sectors, compulsory instruction sectors, constant-domain candidates, and source
envelopes) against H42 scope-compatible independent NCU DRAM read rows.

Of 58 scope-compatible exact joins, the interval brackets 34 targets (58.6%).
The interval's median low/high bounds are 262/392 sectors, while the target
median is 392. High residuals frequently lie above the interval, but demand,
workload family, and NCU operation scope remain confounded; the largest outlier
is adjacent to a previously flagged scope mismatch.

No multiplier, bound expansion, or timing change is promoted. The interval is
retained as a falsifiable demand observer; a new source-domain capture must
separate DRAM read demand from writeback/coherence and profiler operation scope
before expanding it.
