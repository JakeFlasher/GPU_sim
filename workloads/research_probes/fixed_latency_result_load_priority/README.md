# sm_86 fixed-result versus L1TEX writeback-priority probe

This probe isolates the priority of a fixed-latency HADD2 result and a
prewarmed global-load result that target one regular-register bank. The timed
load is issued every generation from the lane's already warmed address and is
consumed by an instruction carrying its write-barrier wait mask. A separate,
later consumer checks the HADD2 result without putting HADD2 on the load's RAW
path.

The compiler allocates HADD2 to R9 and the timed load to R11, both bank 1. A
complete two-instruction live-range rename changes only the load destination
and its sole consumer from R11 to compiler-declared, otherwise unused R14,
creating the opposite-bank control. R15 is present in native SASS, so R14 is
inside the compiler's declared 18-register allocation.

All variants schedule the load consumer at relative cycle 29 and preserve a
static selected-stall sum of 46:

- `same_noncollision`: HADD2 issues at cycle 22 and reaches bank 1 before the
  load;
- `same_collision`: HADD2 issues at cycle 23, making the load wait one cycle
  when fixed-result priority is active;
- `opposite_noncollision` and `opposite_shifted`: repeat those issue schedules
  with the load result in bank 0.

The load consumer stalls 15 cycles before the fixed consumer. This keeps every
W1/W4/W8/W12/W16 oracle safe even when multiwarp arbitration delays completion.
The long cubins run 1,024 generations; NVBit cubins run 64. The host checks one
half-precision increment per generation plus the even-parity load checksum.

Build with `make`. Run the reproducible campaign through `campaign.py`; the
default result root is
`results/captures/sm86_fixed_latency_result_load_priority_20260817_v1`.

The completed campaign contains 20 cells, 140 correctness executions, 60 NCU
profiles, eight SourceCounters profiles, 20 NVBit captures, and 88,396 replay
records. W1/W4 same-bank collision-minus-noncollision is exactly +1.000
cycles/generation in the driver, while both opposite-bank controls are exactly
zero. Every NCU profile reports 4,096 timed L1 hits and eight setup misses per
warp. The promoted narrow replay rule reduces five-width same-pair delta MAE
from 1.058 to 0.305 cycles/generation; all 234 predecessor and 12 production
outputs remain byte-identical. See the campaign analysis for the W8+ generic
producer-shift residual and the exact activation boundary.
