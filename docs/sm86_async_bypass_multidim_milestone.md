# SM86 multidimensional bypass topology milestone

**Status:** complete 2026-08-30; schema-15 observation promoted, timing unchanged

**Architecture:** `sm_86` / NVIDIA GeForce RTX 3090 Ti

**Evidence root:** `results/captures/sm86_async_bypass_multidim_20260830_v1/`

**Machine report:** `analysis/report.json`

## Goal and construction

The three-layout projection assay proved that neither the cache-all bank
projection nor NCU's `.cg` destination-wavefront counter is a sufficient
scalar bypass completion coordinate. This milestone adds an orthogonal
96-byte `skew` destination stride and retains the underlying topology per
copy, without changing the simulator's wait timing.

All layouts use one warp, 32-byte source lane spacing, 1 KiB source-position
spacing, 4 KiB destination-position spacing, and the same 32 KiB shared
allocation:

| layout | destination lane stride | line fanout / source cohort | bank multiplicity / cohort |
|---|---:|---:|---:|
| compact | 32 B | 1 | 1 |
| conflict | 64 B | 2 | 2 |
| skew | 96 B | 3 | 1 |
| severe | 128 B | 4 | 4 |

For bypass observation, each flexible 128-byte source cohort derives

`W = min(active transfers, destination-line fanout × bank multiplicity)`.

This yields per-cohort `W=1/4/3/4` for compact/conflict/skew/severe. The
calculation uses trace addresses only; NCU is an independent comparison target.

## Implementation

`sim-trace` now exposes `LdgstsDestinationTopology` with source-cohort count,
destination-line fanout total/max, cache-all bank-wavefront total/max, and
bypass `W` total/max. Its retained per-instruction form remains 16 bytes by
compacting the bounded 32-lane dimensions into bytes.

The compact async-copy token carries bypass payload work plus cohort count,
line fanout, maximum bank multiplicity, and maximum bypass wavefronts. Direct
source replies remain paired with their exact per-copy record.

Readiness schema 15 adds the separated topology and a timing-neutral local
candidate:

`P(W,B) = 4 × (W−1) + 2 × (B−1)` cycles per copy.

The candidate frontier is `max_i(source_i + payload_i + P_i)`. It is reported
alongside the existing group aggregate, including later/earlier deltas, but it
does not gate issue, responses, waits, terminal time, or tail state.

## Evidence matrix

The final root contains:

- 180 direct live-GPU executions: nine repeats of L2 compact/conflict/skew/
  severe at depths 1/2/4/8, plus the depth-4 cache-all control;
- eight source-only NCU profiles;
- eight complete marker-scoped NVBit captures;
- 16 deterministic simulator replay outputs; and
- eight deterministic predecessor-regression pairs.

Static evidence covers 48 phase functions (4,096 decoded instructions) and 60
store-free functions (2,720 decoded), with zero unknown encodings. Phase
resources and clock controls are equal across layouts. Source-only resources
and instruction counts are equal; severe carries one extra static stall, which
is retained explicitly and is not treated as destination-topology work.

## Live and trace result

All paired issue deltas are exactly zero. The median L2 wait deltas versus
compact, in cycles per group, are:

| depth | conflict | skew | severe |
|---:|---:|---:|---:|
| 1 | 12.011 | 11.942 | 14.060 |
| 2 | 11.409 | 10.562 | 13.449 |
| 4 | 14.437 | 7.734 | 18.448 |
| 8 | 11.724 | 12.987 | 16.609 |

Every value is positive in all nine repeats. At depth 4, the cache-all control
reports compact/conflict/skew/severe wait times of
83.045/147.041/84.045/276.034 cycles per group.

All NCU cells execute exactly 256 LDGSTS instructions and 8,192 source sectors.
Destination wavefronts are:

| route | compact | conflict | skew | severe |
|---|---:|---:|---:|---:|
| cache-all `.ca` | 2,048 | 4,096 | 2,048 | 8,192 |
| L2-hit `.cg` | 2,048 | 8,192 | 6,144 | 8,192 |

The trace-derived cache-all and bypass projections match all eight NCU counts.
Small 128-sector LTS setup asymmetries appear in individual uncontrolled-cache
profiles and are retained; instruction and source-sector work remain exact.

## Simulator result and decision

All four L2 traces report schema 15, 256 per-copy observations, 2,048 source
cohorts, and exact line/bank/bypass topology. For example:

| layout | destination lines | cache-bank upper bound | bypass `W` | candidate penalty sum |
|---|---:|---:|---:|---:|
| compact | 2,048 | 2,048 | 2,048 | 0 |
| conflict | 4,096 | 4,096 | 8,192 | 3,584 |
| skew | 6,144 | 2,048 | 6,144 | 2,048 |
| severe | 8,192 | 8,192 | 8,192 | 4,608 |

The local candidate fits depth 4 with 0.384 cycles/group MAE, including the
untouched skew layout. It fails the untouched depths: holdout MAE is 3.137 and
the largest error is 4.987 cycles/group for depth-8 skew. A latency-only
per-copy rule is therefore incomplete; initiation interval, response spacing,
or a pipelined destination server remains uncharacterized.

No timing is promoted and `async_copy_shared_visibility` remains set. Cache-all
outputs are byte-identical to the predecessor. Every bypass output becomes
predecessor-identical after removing only the schema-15 observation and
restoring the prior schema number.

The next meaningful experiment, if pursued, is an arrival-spacing assay that
identifies a per-copy/cohort destination-server initiation interval. Do not fit
another local latency to these consumed layouts or activate the reported
candidate as production timing.

## Validation

```bash
cargo test --manifest-path sim/Cargo.toml --workspace --lib
cargo build --release --manifest-path sim/Cargo.toml -p sim-run

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_multidim.py \
  --root results/captures/sm86_async_bypass_multidim_20260830_v1 \
  --baseline-root results/captures/sm86_async_source_visibility_20260830_v2 \
  --output results/captures/sm86_async_bypass_multidim_20260830_v1/analysis/report.json
```

The report status is
`pass_multidimensional_observation_no_timing_promotion`.

## Content identities

- Release `sim-run`: `9f7448b41cb920dd304ba5731b195698299b0b0660b50254da50c98b45594387`
- Phase binary: `d0997c46c656417d5bb5adf770516b95093eb287b444df523ec99ddf190dd2d2`
- Store-free binary: `c18dbe060f6c31117a05f058bc179ca545dd337caed6afab114b1cd79aeeaca7`
- Phase control sidecar: `5716a0bb67d6c58eb32975c14b1c11781de506030a90aaf6a1cf9dc68e68335c`
- Store-free control sidecar: `b90c086986bc966f847002a10ddf88ad47e7794a815c631042bde3ceb21caae2`
- Machine report: `b19f4560ed5cb557c35df91616849b1a33678470810b3c31dd88e97928684d22`
