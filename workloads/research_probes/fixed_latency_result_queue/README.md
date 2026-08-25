# sm_86 fixed-latency result/writeback probe

The first admitted pair uses four HADD2 and four FFMA producers with dependent
LOP3 consumers per generation. Runtime inputs come from distinct addresses.
The cubins remain byte-identical except for the four HADD2 controls. Both
treatments canonicalize yield and reuse to zero because CUDA 13.2 suppresses
HADD2's textual `.reuse` operands when yield is clear; this keeps the encoded
flags, disassembly, exact operand projection, and hardware interpretation in
agreement. The only treatment variable is the stall field:

- `collision`: stall=1/yield=0 creates three direct HADD2→FFMA neighborhoods
  where the four-cycle FFMA may complete with the five-cycle HADD2; the fourth
  HADD2 supplies a matched pacing treatment around an intervening predicate.
- `separated`: stall=3/yield=0 separates their completion by two cycles.

Every opcode, register, dependence control, and resource declaration is
otherwise unchanged, and both treatments have identical reuse flags.
Destination-bank treatments remain deferred until a safe non-coalesced output
allocation is generated.

The completed pilot campaign is under
`results/captures/sm86_fixed_latency_result_queue_20260817_v1`. It contains
eight W4/W8/W12/W16 cells, seven correctness trials and seven base-clock NCU
profiles per cell, four SourceCounters profiles, eight NVBit traces, exact
static sidecars, and deterministic replay. Reduce it with:

```bash
../../../.venv/bin/python analyze.py
```

The reduction rejects active result-queue timing: every current HADD2/FFMA
destination pair is opposite-bank, and the multiwarp treatment crosses over
through math-pipe/scheduler pressure. The next retained slice must balance
producer order and destination bank while keeping controls and issue pacing
equal.
