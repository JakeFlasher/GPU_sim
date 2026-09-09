# SM86 live capture plan

Capture one workload at a time with the same executable, launch geometry, and
architecture profile used by replay. Keep NCU measurements as independent
comparison data.

## Capture path

1. Build the CUDA workload and NVBit tracer.
2. Capture the dynamic protobuf trace and SASS control sidecar.
3. Replay the trace with `sim/target/release/sim-run`.
4. Collect a matching NCU profile when the GPU is available.
5. Compare numeric JSON with `eval.direct_score` or the workload report.

## Current inputs

The latest SM86 NCU inputs are the small and large v10 roots listed in
`docs/STATUS.md`; the current simulator campaign is v11. New captures should
use a new scenario directory and keep only the latest usable root for that
scenario.

## Boundaries

Do not pass NCU values into replay or tuning. Keep raw issue, modeled events,
completion, and launch observations distinct. If an endpoint is not represented
by the trace, report it as incomplete instead of adding a guessed tail.
