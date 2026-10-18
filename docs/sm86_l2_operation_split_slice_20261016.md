# SM86 L2 operation-split source observers — H74 — 2026-10-16

## Implementation

Full replay now emits NCU-shaped operation-split L2 source observers directly
from the existing `L2SourceSplit` event counters:

* `lts__t_sectors_srcunit_tex_op_read.sum` plus lookup hit/miss;
* `..._op_write.sum` plus lookup hit/miss;
* `..._op_atom_dot_alu.sum` plus lookup hit/miss;
* `..._op_red.sum` plus lookup hit/miss.

Derived misses use checked `requests - hits` subtraction. This is an
observer-only surface: no lookup policy, write coalescing, L2 state, DRAM
service, or completion frontier changed.

## Full-first result

Release replay ran full before fast with exactly twelve workers and compressed
artifacts at:

`../sm86_replay_results/full_corpus_h74_l2_operation_split_20261016_v1/`

Coverage remains 356 roots (286 `ok`, 62 controls, 8 replay errors) and 12,392
full launches. Report:

`../sm86_replay_results/metric_report_h74_l2_operation_split_20261016_v2/`

The new operation endpoints score as follows in the secondary scope-compatible
view:

| endpoint | compared | mean APE | exact |
|---|---:|---:|---:|
| TEX read sectors | 104 | 1.90% | 99 |
| TEX read lookup hits | 104 | 5.44% | 97 |
| TEX read lookup misses | 104 | 0.00% | 104 |
| TEX write sectors | 107 | 39.86% | 78 |
| TEX write lookup hits | 107 | 39.88% | 78 |
| TEX write lookup misses | 107 | zero-target/exact | 107 |
| TEX atom-dot-alu sectors/hits/misses | 107 each | 0.00% | 106/106/107 |
| TEX reduction sectors/hits/misses | 107 each | 0.00% | 107/107/107 |

The existing aggregate L2/TEX and DRAM metrics are unchanged. The reporter
retains raw/all-row authority and secondary scope exclusions; operation-split
rows are now machine-readable and directly comparable rather than unavailable.

## Decision

Promote the operation-split observer surface into default full replay, but do
not promote a P3 write-coalescing or write-lookup timing arm. Read/atom/reduction
rows are endpoint-compatible; write sectors/hits remain a large residual and
need a same-binary L1TEX-to-XBAR coalescing crosswalk before any state/timing
change. Aggregate carded DRAM, typed dirty/coherence tails, and all existing
lookup policy remain unchanged.
