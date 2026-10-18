# SM86 post-replay metric-unit crosswalk — H78 — 2026-10-16

## Defect

The reporter receives native prediction units (`%`, `byte`, `request`,
`thread/block`, `block/SM`, and `register/thread`) but its compatibility table
only recognized internal aliases such as `pct` and `count`. The H77 report
therefore marked 24 dimensionally compatible metrics as `unit_mismatch`, so
they were excluded from APE despite having finite predictions and targets.

## Bounded repair

`tools/sm120_metric_report.py` now performs a name-qualified, post-replay unit
crosswalk. It accepts percentage aliases, empty-unit request/launch endpoints,
branch counts, launch geometry/occupancy aliases, and register/resource units.
For `dram__bytes_read.sum` and explicitly named shared-memory/configuration
metrics it scales NCU `Kbyte`/`Mbyte` values to the simulator's byte domain.
The conversion is applied only while comparing report rows; it never feeds NCU
values into replay, prediction selection, cache state, or timing. GPU duration
(`gpu__time_duration.sum`, ns versus us/ms) remains intentionally unavailable
because no clock crosswalk is established.

## Real-corpus evidence

The H77 full-first artifact was re-reported without replay mutation:

* source artifact: `../sm86_replay_results/full_corpus_h77_adapter_20261016_v2/`
* report: `../sm86_replay_results/metric_report_h78_units_20261016_v3/`
* full joins: 107 identity rows; coverage remains 286 `ok`, 62 controls, 8
  replay errors, and 12,392 full launches from twelve workers.
* compared metrics increased from 40 to 63; the only remaining unit mismatch is
  `gpu__time_duration.sum` (107 rows), which is deliberately not converted.

Newly scored examples are exact for launch block/grid geometry, occupancy
limits, global-store requests, branch-target divergence, and static/dynamic
shared-memory byte rows. Percentage rows are now scored as percentages; their
residuals are reported honestly (for example, global-load lookup-hit rate
remains 30.71% mean APE). DRAM bytes now compare in byte units and retain the
same 40.91% raw/all-row residual as the sector-domain read metric.

## Boundary

This slice changes reporting dimensionality only. It does not claim an IPC,
L1/L2, DRAM, or timing accuracy improvement, and it does not relax the
currently-unavailable presentation rule. The next accuracy work must still
target the L1 lookup, DRAM multi-client demand, and cycle/IPC residuals with
independent endpoint evidence.
