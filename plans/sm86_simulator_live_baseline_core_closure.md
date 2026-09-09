# SM86 baseline closure slice

This slice covers the live Rust path from trace loading through SM issue and
memory completion. The working tree already contains changes in those seams.

## Acceptance bar

A representative SM86 trace must build, replay, terminate, and emit a result
without changing modeled traffic unexpectedly. Record cycle frontiers and
queue/event counts when they explain a change.

## Execution order

1. Build `sim-run` in release mode.
2. Replay one current retained trace and inspect stderr and result JSON.
3. Compare the relevant workload row with the current campaign output.
4. Expand only after the representative replay is stable.

## Active limits

The simulator cannot infer host range timing from a single SASS trace. Dirty
writeback and some RED/STORE retirement tails remain typed incomplete state.
SM120 timing remains a separate future slice.
