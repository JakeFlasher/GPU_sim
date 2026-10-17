# SM86 constant-domain / DRAM-demand slice — 2026-10-15

## Scope and correction

H2 retains the real trace producer's constant-bank fields without injecting
them into L1/L2 state, DRAM service, warp readiness, or IPC. Field 15 is a
vector of active-and-predicate-on **bank-local effective CBANK offsets**;
field 16 identifies the bank and field 17 is one scalar metadata offset. A
field-17-only record is therefore counted as `metadata_only` and never
fabricated into traffic.

The emitted 32-B packet count, launch-wide 32-B offset-domain cardinality, and
64-B candidate-line cardinality are source-domain observations. They are not
GCC requests, DRAM sectors, cache fills, or mathematical DRAM lower/upper
bounds. Constant `LDG.*.CONSTANT` instructions remain on the ordinary traced
global/L1/L2 path; attributing their GCC return traffic requires a separate
static-opcode/source-class arm.

## Representative full replay

Four real roots completed integrated full replay **4/4** with carded SM86
DRAM and twelve workers:

| Root | Active offset refs | Metadata-only records | 32-B packets | Unique 32-B offsets | Unique 64-B candidates | L2+instruction demand |
|---|---:|---:|---:|---:|---:|---:|
| scalar 32-bit copy | 33 | 1 | 33 | 4 | 3 | 60 |
| L2 prefetch hint sweep | 810 | 6 | 810 | 5 | 4 | 137 |
| local/register spill | 350 | 6 | 350 | 5 | 4 | 797 |
| RMSNorm BF16 | 6,912 | 0 | 6,912 | 4 | 3 | 1,093 |

Artifact: `../sm86_replay_results/h2_constant_demand_20261015_v5/full.jsonl.zst`.

The corresponding independent NCU observations are not replay inputs:

| Root | DRAM sectors read | GCC constant requests |
|---|---:|---:|
| scalar 32-bit copy | 140 | 3 |
| L2 prefetch hint sweep | 316 | 16 |
| local/register spill | 912 | 16 |
| RMSNorm BF16 | 1,220 | 384 |

The different scopes are the result, not a parameter-fitting opportunity:
field-15 references (33/810/350/6,912) are CBANK operand observations, while
GCC counters are L1.5 requests and DRAM counters are post-cache service
observations. The 64-B launch-wide union hides CTA/warp replication and is
therefore intentionally not scored as a GCC prediction.

## Complete-corpus full-first evidence

The corrected H2 path completed the complete real corpus:

| Field | Result |
|---|---:|
| Roots discovered | 356 |
| `ok` roots | 286 |
| Mixed-code control dispositions | 62 |
| Replay/input failures | 8 |
| Successful launches | 12,392 |
| Driver wall time | 34.263 s |
| Workers | 12 |
| NCU used as replay input | No |

Artifact: `../sm86_replay_results/full_corpus_h2_constant_20261015_v2/full.jsonl.zst`.

Across the 107 exact-identity NCU rows, active constant-offset vectors were
observed on all 107 rows (2,845,702 active-offset references); 3,840
metadata-only records were retained separately. The post-replay reducer still
joins 107 exact rows and excludes ordered/shape fallback rows from primary APE.

## Exact-identity residual deltas

H2 is deliberately demand/source-only, so the scored timing/cache rows are
unchanged from S2. Full-corpus exact-identity means are:

| Metric | S2 mean APE | H2 mean APE | H2 median | H2 P90 |
|---|---:|---:|---:|---:|
| L2 lookup-hit sectors | 35.04% | 35.04% | 0% | 42.86% |
| L2 lookup-miss sectors | 27.00% | 27.00% | 0% | 0% |
| L2 source-total sectors | 28.24% | 28.24% | 0% | 28.57% |
| DRAM read sectors | 40.91% | 40.91% | 40.25% | 70.08% |
| SM active IPC | 14.57% | 14.57% | 12.82% | 29.61% |
| SM elapsed IPC | 12.41% | 12.41% | 11.76% | 21.97% |

Report artifact: `../sm86_replay_results/metric_report_h2_constant_20261015_v3/`.

## Verdict and next arm

H2 closes bounded constant-offset observability and prevents an invalid
field-15-to-DRAM overclaim, but it is insufficient to improve the requested
IPC/DRAM APE. The next default-full-replay arm is H3: a conditional SM86
fixed-latency register-read/RFC envelope, using real SASS operand/reuse fields
and Huerta et al. MICRO'25 Ampere evidence. It must remain transactional,
typed-gap aware, and independent of the cache/DRAM path; complete-corpus
full-first replay follows representative invariants.
