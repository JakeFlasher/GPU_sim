# SM86 bypass per-copy observation milestone

**Status:** complete 2026-08-30; structural observability added, timing unchanged

**Architecture:** `sm_86`

**Evidence root:** `results/captures/sm86_async_copy_per_copy_observation_20260830_v2/`

**Machine report:** `analysis/report.json`

## Purpose

The phase assay localized the bypass layout effect after LDGSTS issue and
showed that its L2-hit differential is nearly depth-independent. The existing
simulator cannot represent that shape because one async group retains only the
maximum source-response timestamp and summed payload work. This milestone adds
the missing event identity without changing scheduler readiness or declaring a
bypass timing constant.

## Implementation

Destination-bearing bypass records now carry two independently typed values in
the compact async-copy token:

- the established payload work that continues to control timing; and
- the trace-derived destination wavefront count, marked observational only.

Inside shared-L2 replay, every bypass LDGSTS instruction has a cold per-copy
record keyed by its opaque async-group identity and dynamic instruction index.
Each direct source reply updates both that copy and the unchanged legacy group
aggregate. The copy record retains pending source units, its maximum source
completion, payload work, and projected destination wavefronts, then is removed
with the consumed wait prefix.

Readiness schema 14 exposes one boxed, bypass-only diagnostic:

- `copies_observed`;
- `destination_wavefronts_observed`;
- `copy_source_spread_cycles`;
- `per_copy_completion_cycles`; and
- `group_aggregate_excess_cycles`.

The box keeps disabled storage outside the hot inline readiness object; the
existing size guard still passes. None of these fields gates issue, source
responses, destination service, wait readiness, terminal event time, or typed
tail state. Cache-all and legacy payload-only paths do not allocate or
serialize the diagnostic.

## Retained-trace result

The current release runner replayed all eight fixed-source visibility traces
twice. Cache-all outputs are byte-for-byte identical to the predecessor. For
bypass outputs, removing only `bypass_async_copies` and restoring the prior
schema number produces an object exactly equal to the predecessor; traffic,
readiness, terminal time, completion frontier, and
`async_copy_shared_visibility` are unchanged.

| route/layout | depth | copies | projected destination wavefronts | source spread sum | per-copy completion sum | group aggregate excess |
|---|---:|---:|---:|---:|---:|---:|
| DRAM compact | 4 | 256 | 2,048 | 1,024 | 27,840 | 3,072 |
| DRAM conflict | 4 | 256 | 4,096 | 1,024 | 27,840 | 3,072 |
| DRAM compact | 8 | 512 | 4,096 | 2,304 | 27,840 | 7,168 |
| DRAM conflict | 8 | 512 | 8,192 | 2,304 | 27,840 | 7,168 |

The group aggregate serializes 48 extra payload cycles per waited depth-4
group and 112 per depth-8 group relative to independent per-copy payload
completion. That is an ordering observation, not evidence that the independent
frontier is complete: compact and conflict payload frontiers remain identical
until destination work is priced.

The trace projection distinguishes layouts by 2×, while the independent live
`.cg` NCU metric distinguished the same pair by 4×. Therefore this milestone
does not promote the cache-all projection or any per-wavefront timing to the
bypass route. The typed visibility tail remains correct.

## Validation

```bash
cargo test --manifest-path sim/Cargo.toml -p sim-mem -p sim-sm --lib
cargo build --release --manifest-path sim/Cargo.toml -p sim-run

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_per_copy.py \
  --root results/captures/sm86_async_copy_per_copy_observation_20260830_v2 \
  --baseline-root results/captures/sm86_async_source_visibility_20260830_v2 \
  --output results/captures/sm86_async_copy_per_copy_observation_20260830_v2/analysis/report.json
```

The focused Rust run passes 114 `sim-mem` and 347 `sim-sm` tests. Replay twins
are byte-identical in all eight cells, all non-observational output is
unchanged, and the report status is
`pass_observational_per_copy_no_timing_change`.

## Decision and next slice

Per-copy event ordering is now representable and inspectable, but bypass
destination projection remains underidentified. The next narrow discriminator
is one additional fixed-source destination geometry that separates the
cache-all source-cohort projection from the observed `.cg` wavefront
multiplier. Only after that projection transfers should a per-copy completion
rate be evaluated against the L2 depth holdouts. JADE/HETR, CPU-GPU
co-simulation, `sm_120`, and scored timing defaults remain unchanged.

## Content identities

- Release `sim-run`: `8e41dd19ea4a376e81e25969664f6bde9094a4caa1e06c11cd7f278588a28cc7`
- Machine report: `6344103a5ddfbe1e3dbe7eade81ebe7e28209707d418721017d20d21256181f4`
