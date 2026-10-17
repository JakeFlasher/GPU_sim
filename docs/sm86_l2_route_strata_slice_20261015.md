# SM86 L2 route strata — 2026-10-15

H34 converts the continuous H33 concentration observer into deterministic
route-local strata: `incomplete` when fewer than six effective slices receive
requests, `balanced` when all six are active and concentration is below
250,000 ppm, and `concentrated` otherwise. The strata are descriptive only;
they do not select queue timing or use NCU data.

Complete full-first replay passed 356 roots (286 `ok`, 62 mixed-code controls,
8 replay/input failures), 12,392 launches in 40.574 s with 12 workers:
`../sm86_replay_results/full_corpus_h34_route_strata_20261015_v1/full.jsonl.zst`.
Report:
`../sm86_replay_results/metric_report_h34_route_strata_20261015_v1/`.

Full-corpus route strata: 3,206 incomplete, 9,135 balanced, and 51
concentrated launches. H34 is timing/traffic neutral versus H33 for all
L1/L2/DRAM/IPC/frontier outputs. The L2 probe evidence still supports pressure
observation but no stable transferable service latency.
