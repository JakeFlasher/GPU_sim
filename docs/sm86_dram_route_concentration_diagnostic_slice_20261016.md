# SM86 DRAM route-concentration diagnostic — H82 — 2026-10-16

## Implementation

Full replay now emits normalized maximum-demand fractions (ppm) for the modulo
partition, XOR partition, and modulo L2-slice candidate arrays. It also emits a
joint observer row pairing the maximum candidate concentration with the
modeled GPC observation frontier. These are report-only covariates; they do not
select a hash, add a queue, alter `DramBw::Carded`, or change any L2/DRAM/SM
frontier.

## Full-first evidence

Replay/report:

* `../sm86_replay_results/full_corpus_h82_route_concentration_20261016_v1/`
* `../sm86_replay_results/metric_report_h82_route_concentration_20261016_v1/`

Coverage remains 286 `ok`, 62 controls, 8 replay errors, and 12,392 full
launches with twelve workers. The route diagnostic is present on 286 full
launches; its median maximum candidate concentration is 175,631 ppm, P90
500,000 ppm, and 23 launches exceed 500,000 ppm. GPC-cycle and SM-active APE
remain exactly unchanged from H81 (12.9008% and 18.0202%).

## Independent adversarial decision

The partition review finds no default-safe timing arm. H35 concentrated-route
residuals have only 4–8 exact joins; H43's 58 scope-compatible DRAM joins are
all balanced; pressure quartiles are non-monotonic; H45/H59 show GPC/SM error
is not explained by route concentration or grid size; H64's queue probe changes
selected NCU request-stall endpoints by only about 4% and is confounded by
issue/occupancy/overlap. Therefore no L2 hash, finite queue, partition service,
or GPC/SM timing correction is promoted. A locked one-factor partition/demand
probe with exact critical-path joins remains the required evidence gate.
