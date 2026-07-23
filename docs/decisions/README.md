# Binding decisions

Accepted decisions override older planning text that still describes them as
open. The active operator rulings are:

- [ADR 0001](0001-task7-h3-continuation.md):
  `TASK7_H3=CONTINUE_PRESERVE_PINNED`.
- [ADR 0002](0002-publication-metric-scope.md):
  `PUBLICATION_SCOPE=CYCLES_ONLY_BINDING`.
- [ADR 0003](0003-atomic-histogram-capture-envelope.md):
  the atomic-histogram end-to-end capture envelope is 5,400 seconds.
- [ADR 0004](0004-sol-problem-027-capture-envelope.md):
  the SOL problem-027 end-to-end capture envelope is 10,800 seconds.
- [ADR 0005](0005-sol-problem-039-capture-envelope.md):
  the SOL problem-039 end-to-end capture envelope is 21,600 seconds.
- [ADR 0006](0006-sol-problem-040-capture-envelope.md):
  the SOL problem-040 end-to-end capture envelope is 18,000 seconds.

The 61-entry metric registry remains the capture ceiling. Only
`exec_cycles_gpc_max` / `gpc__cycles_elapsed.max` is publication-binding: ≤10%
MAPE meets the target, >10% and ≤15% is floor-only, and >15% is publication
failure. The seven currently implemented secondary outputs remain reported but
ungated and cannot replace a failed cycles headline.
