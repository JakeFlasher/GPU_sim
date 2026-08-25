# sm_86 fixed-result bypass-frontier probe

This probe reproduces the consumer-availability distinction in Listing 3 of
Huerta et al., MICRO 2025, and turns the valid side of that boundary into a
profileable campaign.

The fixed path uses a loop-carried LOP3 producer and dependent LOP3 consumer.
Distance 3 fails the checksum, distance 4 is correct through the fixed-result
bypass, and distance 5 is the post-retirement control. The global canary builds
a 64-bit address from an IADD3-produced high half. Distance 4 causes
`CUDA_ERROR_ILLEGAL_ADDRESS`; distance 5 loads correctly. The long global loop
uses the compiler's equivalent IMAD.IADD address producer at safe distances 5
and 6.

Within each valid pair, one stall cycle moves from the producer/spacing region
to a later checksum instruction. Instruction work and selected static stall
sums remain equal. Long cubins run 1,025 generations; NVBit cubins run 65 so
the alternating fixed checksum retains an unambiguous oracle.

Build with `make`; use `campaign.py` for calibration, correctness, NCU,
SourceCounters, NVBit, static audit, and replay. The default result root is
`results/captures/sm86_fixed_latency_result_bypass_20260817_v1`.

The completed campaign contains 20 cells, 140 correctness executions, 60 NCU
profiles, eight SourceCounters profiles, 20 NVBit captures, and 83,558 replay
records. Maximum active-cycle CV is 1.545%. W1/W4 valid controls are null;
W8+ re-enters the independent producer-shift phase. Exact `IMAD.IADD` operand
projection is retained, but no bypass timing or scoreboard is added: the
semantic boundary belongs to compiler-scheduled correctness, and replay does
not execute register values.
