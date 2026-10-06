# sm_86 same-bank fixed-result producer-order probe

This compiler-native pair extends the admitted opposite-bank order experiment
without patching any register field. Both functions are spill-free, have 48
instructions and 16 registers, and use the same relative loop layout:

- `hf`: HADD2 R4 then FFMA R8, aligning five-/four-cycle completion;
- `fh`: FFMA R4 then HADD2 R8, separating completion by two cycles.

R4 and R8 both target bank 0. HADD2 and FFMA have identical per-opcode source
bank demand across the pair, and the two timed LOP3 consumers are byte-identical.
Padding that equalizes resource declarations executes before `clock64`; the one
commutative oracle operand-order difference executes after the stop clock.

A bounded patch clears only the first producer yield bit in each function,
giving both `(stall,yield,reuse)` schedules `(1,0,0)` then `(2,0,0)`. W4/W8
identify same-bank bypass/result buffering; W12/W16 are held out, with W16 also
serving as the global phase sentinel discovered by the opposite-bank campaign.

The completed campaign is under
`results/captures/sm86_fixed_latency_result_same_bank_20260817_v1`. It contains
eight cells, 56 correctness executions, 56 NCU profiles, eight SourceCounters
profiles, eight NVBit captures, and 37,920 replay records. W4/W8 are null to
within 0.001 cycles/generation, identifying timing-neutral two-result same-bank
buffering/bypass. W12 adds a bank-specific 0.742 cycles/generation beyond the
opposite-bank pair; queue capacity versus bank retirement remains unresolved.

## C7-S33AX live same-binary refresh (2026-10-04)

The active Makefile/campaign interface now uses one 8,192-generation cubin for
correctness, NVBit, replay, and NCU; the historical 64-generation trace twin
is not an active input. The driver now parses numeric arguments strictly,
initializes elapsed/sink buffers with sentinels, and reports elapsed/sink
validity separately from the functional oracle. The selected W8 HF cell is the
current live vertical target; no physical bank mapping or latency is inferred
from this single contrast.
