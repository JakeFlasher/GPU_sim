# SM86 INT8 IMMA issue probe

The timed loop contains only warp-uniform
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` instructions.  The dependent
arm feeds one accumulator tuple back into the next operation; the independent
arm rotates eight tuples.  A numerical per-lane oracle checks every accumulator
against the exact INT8 dot-product result.  The probe is an initiation-rate
observation, not a result-latency or completion measurement and is not fitted
to the blocked SimBench row.
