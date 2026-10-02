# Full retained-trace simulator cost campaign (2026-10-02)

This is a simulator utility measurement over the retained 15-workload SM86
corpus. It does not use NCU values and does not claim native speedup.

## Protocol

For each identical trace/control/card input, the current release `sim-run`
was run twice:

1. detailed native replay (`--mode flat --cache-scope launch`);
2. `--fast-frontier-only` analyzer.

Wall time and peak RSS include process startup, trace parsing, replay/planning,
effect materialization, comparison, and JSON output. Both runs were bounded
with a 180-second timeout. The compact summary is
[`full_trace_cost_summary.json`](full_trace_cost_summary.json); raw per-run
stdout/stderr remain under the ignored `results/captures/sm86_reification_cost_20261002/` tree.

## Result

- 15/15 workloads completed both paths.
- 5,130,860 trace records were covered; 0 candidate records were accepted;
  all 5,130,860 were uncovered/refused by the conservative analyzer.
- Native macro dispatches: 0.
- Total detailed wall time: 7,287.9 ms; analyzer wall time: 4,463.9 ms.
- The analyzer is not a native replacement engine here, so the wall-time ratio
  is **not** reported as speedup. It measures two different executable paths
  with zero accepted macro regions.

## Interpretation

This campaign supplies real full-corpus parser/replay/analyzer cost and refusal
coverage. It does not establish useful native acceleration, a production macro
path, or a hardware-causal prediction. The remaining positive falsifier is the
same as R9: a complete independently derived region must be accepted through
the real production event loop before any compression or speedup claim.
