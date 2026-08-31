# SM86 asynchronous-copy source-line ownership discriminator

## Outcome

The prospective line-ownership smoke is complete as:

```text
complete_line_ownership_structural_no_go
```

The new probe preserves its source/preload vectors, destination address
multiset, and exact early/late line-ownership permutation. Current schema 18
and every production fingerprint tie between variants as required. The frozen
wavefront contract does not hold: each 64-group NCU row reports 1,280 shared
LDGSTS wavefronts rather than 2,048. The reducer therefore stops before opening
even the smoke wait delta, and the full two-orientation campaign does not run.

The tracked aggregate report is
[sm86_async_line_ownership_report.json](../data/local_labels/sm86_async_line_ownership_report.json),
with content digest
`1165e3829dd938b47591d61481939c72975fcb097afcfe3a2f8606cd79ad801a`.

No simulator schema, timing, card, scheduler, traffic, readiness, or tail state
changes. `async_copy_shared_visibility` remains required.

## Fresh ownership geometry

One warp issues one fixed-lead LDGSTS per group. Four aligned source cohorts
occupy one physical line; four split cohorts occupy two. A scalar `.cg` load
preloads every aligned sector and the first line of every split cohort. Its
result remains unconsumed until after the post-wait clock.

Each aligned destination cohort uses offsets `0/32/128/160`. Each split cohort
uses the fixed multiset `0/128/256/288`. `early_fanout` assigns the preloaded
source-line lanes to the two-line `0/128` pair; `late_fanout` assigns those
same addresses to the late source-line lanes. The source vector, preload
vector, and destination address multiset are byte-for-byte equal between
variants.

The fresh NVBit traces prove all 64 LDGSTS records per endpoint have the exact
lane mapping. Unsupported, unmapped, and cross-allocation sidecars are empty.

## The wavefront distinction

The smoke makes three previously conflated quantities explicit:

| Coordinate | Per copy | 64-group total |
|---|---:|---:|
| NCU shared LDGSTS wavefront metric | 20 | 1,280 |
| Simulator `destination_wavefronts_observed` | 24 | 1,536 |
| Trace-projected bypass wavefronts | 32 | 2,048 |
| Unique destination lines | 20 | 1,280 |

The plan froze 32 payload/shared wavefront observations and required the NCU
shared metric to report 2,048. The hardware instead reports 1,280, and the
simulator's existing destination-wavefront observation reports 1,536. The
bypass projection alone reports 2,048. These are different mechanisms and
must not be relabeled into agreement.

This is a useful structural result: the NCU counter follows the 20-wavefront
shared destination service coordinate for this mapping, while the current
simulator observation and bypass projection expose different aggregates.

## Retained smoke evidence

The canonical evidence root is:

```text
results/captures/sm86_async_line_ownership_20260831_v1/
```

It was captured entirely under the repository's pinned `.venv`, including
`protobuf==7.34.1`, so the static control-tool identity and per-lane trace
decoder share one interpreter contract.

The admitted smoke contains:

- two SM86 orientation specializations, 328 decoded instructions, zero unknown
  controls, 40 registers, 4 KiB shared memory, and zero stack/local/spills;
- one correct direct AB pair retained only for structural/issue availability;
  its wait fields are not read by the reducer;
- one seven-family paired NCU bundle, seven one-pass reports, and 14 ordered
  rows;
- two marker-scoped NVBit endpoint traces; and
- two byte-identical schema-18 outputs per endpoint.

Every static, direct, NVBit, NCU profile/import, simulator-build, and replay
invocation has a validated command/timeout/timestamp/status and exact
input/output identity manifest.

### NCU result

All non-wavefront gates pass exactly in both arms:

```text
scalar LDG instructions = 64
LDGSTS instructions     = 64
local-device LTS total  = 3,584
LTS hits                = 1,536
LTS misses              = 2,048
hit/miss closure error  = 0
paired hit/miss delta   = 0
shared wavefronts       = 1,280 (frozen requirement: 2,048)
```

Raw DRAM remains a recorded, non-gating device-background sentinel; its value
is not used to override the exact attributable LTS evidence.

### Replay result

Both variants report the same 179-cycle response envelope per copy, zero
prearrival hiding, 64 stream-hidden topology cycles per copy, and 14 residual
cycles/group. Schema-18 observations and production fingerprints are exact
paired ties. The simulator aggregate independently fails only its frozen
32-wavefront coordinate: 24/copy are observed.

## Decision

Do not read the direct wait delta, run the second-line-early holdout, add a
line-local schema, or modify production timing under this plan.

A future source-line ownership plan must prospectively distinguish and freeze
all four quantities in the table: NCU shared wavefronts 20, simulator
destination wavefronts 24, bypass projection 32, and destination lines 20. It
may then reuse the ownership mechanism with a fresh binary and fresh labels;
the present smoke timing is consumed only as an unread structural artifact.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  workloads/research_probes/async_source_only/capture_line_ownership_execution.py \
  --smoke --phase all --timeout 1800

CUDA_VISIBLE_DEVICES=0 LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python \
  workloads/research_probes/async_source_only/capture_line_ownership_ncu.py \
  --root results/captures/sm86_async_line_ownership_20260831_v1 \
  --smoke --timeout 900

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  workloads/research_probes/async_source_only/analyze_line_ownership.py \
  --smoke-only \
  --root results/captures/sm86_async_line_ownership_20260831_v1 \
  --output results/captures/sm86_async_line_ownership_20260831_v1/analysis/report.json \
  --output-md results/captures/sm86_async_line_ownership_20260831_v1/analysis/report.md \
  --aggregate-output data/local_labels/sm86_async_line_ownership_report.json
```

The analyzer exits zero only for the completed scientific no-go. Missing,
malformed, stale, mixed-tracer, manifest-invalid, or replay-inconsistent
evidence remains nonzero. Both capture CLIs also reject every non-smoke,
subset, resume, or validation request unless an identity-matched tracked
aggregate records an explicit smoke pass; this no-go root cannot create any
full-campaign output.
