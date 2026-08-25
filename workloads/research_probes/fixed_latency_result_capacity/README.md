# sm_86 fixed-result reservation-capacity feasibility probe

The MICRO'25 reference model reserves one regular result-queue slot when a
fixed-latency instruction issues and frees it only after writeback. Its
published sm_86 configuration uses eight slots and at most one pop per cycle.
This probe tests whether that boundary is observable rather than importing it.

One native function contains a tensor anchor followed by a statically audited
run of seventeen regular fixed-result instructions. All variants have equal
instructions, controls, values, and baseline terminal issue time. A checked
patch:

- gives the preceding BAR a common 15-cycle drain;
- canonicalizes the reservation run to stall 1, yield 0, reuse 0; and
- moves one identical 15-cycle stall after reservation 7, 8, or 9.

Only a monotonic b9-only delay is evidence for an eight-slot issue-reservation
boundary. A b8-only or non-monotonic effect is classified as scheduler/pipe
phase. W1 is identification; W4 and W8 test subcore and multiwarp transfer.

The completed campaign is under
`results/captures/sm86_fixed_latency_result_capacity_20260817_v1`. W1/W4 are
null; W8 measures `b8−b7=+0.164` and `b9−b8=+1.126` cycles/generation.
Production predicts the b9 step with the wrong sign. A capacity-8/pop-1
candidate changed none of 18 outputs and was removed. Physical capacity and
pop width remain unidentified; the W8 b9 cell is retained as a canary. The
campaign contains nine cells, 63 correctness executions, 63 NCU profiles, six
SourceCounters profiles, nine NVBit captures, and 61,269 replay records.
