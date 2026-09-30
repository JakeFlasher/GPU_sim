# NAK bridge operation-family endpoint envelope

The retained six-body NAK-to-CUDA-ELF family matrix now has a conservative
aggregate envelope:

- duration: 2.304--2.496 us
- GPC cycles: 3,054--3,314
- instructions: 14--18
- ALU aggregate: 6--10
- LSU aggregate: 3
- XU aggregate: 0--8
- PM long-scoreboard average: 2.214--2.783

The mixed body is 2.336 us / 3,085 cycles with 26 instructions, ALU/LSU/XU
18/3/0. It lies inside the endpoint envelope despite being outside the family
instruction-count range. This supports using aggregate resource intervals for
ABI-generated bodies, but does not identify per-PC routes, physical issue slots,
or general CUDA ABI metadata generation.

Machine-readable output:
`results/alpha/sm86_bridge_operation_envelope_goal_20260929/envelope.json`.
