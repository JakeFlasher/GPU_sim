# sm_86 branch-frequency/static-footprint factorial

This probe follows the dense fixed-result producer-shift campaign.  It keeps
the fixed d4/d5 and global d5/d6 treatment pairs and total dynamic instruction
work constant while compiling 1, 2, 4, or 8 measured bodies between backward
branches.  The resulting U1/U2/U4/U8 kernels separate branch-redirection cost
from instruction-buffer refill and sequential prefetch effects.

The long kernels execute 1,024 generations; NVBit trace kernels execute 64.
Both counts are divisible by every unroll factor.  A compiler-shape audit must
confirm exact repeated bodies, one loop branch per unrolled group, invariant
register/resource usage within each treatment pair, and control-only pair
differences before live measurements are admitted. Register count may differ
across U because U is the deliberate static-footprint treatment; fixed U2/U4/U8
all use 14 registers, while the global path grows from 16 to 24.

The completed matrix covers U1/U2/U4/U8 at W4/W5/W8/W9/W13/W16: 96 cells,
672 correctness executions, 320 general NCU profiles, 96 direct front-end
profiles, 32 SourceCounters profiles, 96 NVBit captures, and 381,480 replay
records. Direct counters identify three cycles per uniform backward target.
The best IBuffer candidate conflicts with the reference capacity and misses
held-out phases, so only branch-target resolution is promoted; IBuffer/fetch
queue timing remains inactive. See
`results/captures/sm86_fixed_latency_front_end_footprint_20260817_v1/analysis/README.md`.
