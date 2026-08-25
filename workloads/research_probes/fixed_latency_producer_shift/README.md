# sm_86 dense producer-shift localization

This campaign reuses the validated fixed d4/d5 and global d5/d6 cubin pairs
from `fixed_latency_result_bypass` and sweeps every warp count W4–W16. Physical
warp slots map to SMSPs by `slot % 4`, so W5/W9/W13 add the second/third/fourth
warp to SMSP0 and the next three widths fill SMSP1–3.

The matrix distinguishes a true all-partition occupancy threshold from effects
that begin on one partition and propagate as the cohort fills. It also tests
whether the same transition transfers between a memory-free fixed loop and a
prewarmed global-load loop. Valid pair work, selected stall sums, values, and
host oracles remain unchanged.

The completed matrix contains 52 cells, 364 correctness executions, 160 NCU
profiles, 32 SourceCounters profiles, 52 NVBit captures, and 264,940 replay
records. The phase begins at W5 with only one two-warp SMSP and changes sign by
instruction family across later cohorts. Ten reference-inspired IBuffer/fetch
candidates were scored; the best reduces treatment-delta MAE substantially but
does not transfer across W9–W12 and W13–W16, so no front-end timing is active.
