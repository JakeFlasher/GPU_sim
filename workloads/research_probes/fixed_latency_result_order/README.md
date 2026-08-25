# sm_86 balanced fixed-result producer-order probe

This slice isolates completion coincidence without changing issue pacing. Both
spill-free functions have 48 instructions, 15 registers, identical controls,
and identical code outside PCs `0x100` and `0x110`:

- `hf`: HADD2 R11 followed by FFMA R4. With fixed latencies five and four and
  one-cycle issue spacing, the opposite-bank results complete together.
- `fh`: FFMA R11 followed by HADD2 R4. The results complete two cycles apart.

The destinations at each PC are fixed, consumers are byte-identical, and each
producer result is XORed twice so its contribution cancels exactly. Runtime
zero/one operands and host-checked accumulators prevent value or dead-code
differences. A bounded patch changes only the two producer yield bits per
function, giving both orders the same `(stall, yield, reuse)` schedule of
`(1,0,0)` then `(2,0,0)`.

This is an opposite-bank null/control surface. It does not identify same-bank
retirement, queue depth, or writeback priority.

The frozen identification widths are W4/W8, with W12/W16 held out. After W16
exposed a completion-order effect absent through W12, the bounded adaptive
threshold slice adds W13/W14/W15: these place four warps on one, two, and three
of the four modeled scheduler partitions respectively, while W16 places four
warps on all four.

The completed campaign is under
`results/captures/sm86_fixed_latency_result_order_20260817_v1`. It contains 14
cells, 98 correctness executions, 110 NCU profiles, eight SourceCounters
profiles, 14 NVBit captures, and 77,900 replay records. W4/W8 form an exact
opposite-bank null. W16 enters a global slow regime in 8/11 HF profiles and is
retained as a phase sentinel rather than queue-depth evidence.
