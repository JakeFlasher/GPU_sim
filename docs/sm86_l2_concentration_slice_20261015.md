# SM86 L2 route concentration observer — 2026-10-15

H33 adds a normalized sum-of-squares concentration metric over effective
six-slice request demand. It distinguishes evenly distributed traffic from
single-slice concentration without assuming a physical GA102 hash or queue
service latency.

Complete full-first replay passed 356 roots (286 `ok`, 62 mixed-code controls,
8 replay/input failures), 12,392 launches in 39.868 s with 12 workers:
`../sm86_replay_results/full_corpus_h33_l2_concentration_20261015_v1/full.jsonl.zst`.
Report:
`../sm86_replay_results/metric_report_h33_l2_concentration_20261015_v1/`.

Concentration statistics: min 166,667 ppm, median 168,405 ppm, p90 500,000
ppm, max 1,000,000 ppm. The preexisting imbalance ratio remains min 1.0,
median 1.133, p90 3.0, max 38.0. H33 is timing/traffic neutral versus H32;
no NCU-derived or queue-timing arm is enabled.
