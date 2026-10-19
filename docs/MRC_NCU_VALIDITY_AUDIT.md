# MRC versus NCU validity audit

Date: 2026-10-19  
Scope: allowed reference outputs under `/home/jakeshea/5090_branch/final_results/`

## Evidence

The retained NCU comparison report (`final_results/metric_report_sm120_enabled_full_only/summary.md`)
contains realized single-operating-point counters such as
`l1tex__t_sector_hit_rate.pct`, `lts__t_sectors_srcunit_tex_lookup_hit.sum`,
and operation-qualified lookup misses. It contains no capacity-indexed MRC,
Mattson stack-distance histogram, or NCU reuse-distance/reuse-interval target.
The same absence holds for the SM86 final-results comparison surface. A
read-only search of the permitted `final_results/` tree finds no MRC or stack-
distance result artifact.

The trace contract also lacks a total hardware arrival order across warps and
SMs, and does not expose proprietary set/slice hashing, replacement state,
MSHR horizons, or TMA completion linkage. The simulator's exact MRC therefore
has a valid Class-A reference only for its declared within-warp, origin-
qualified stream.

## Consequence

There is no scientifically valid universal Class-B MRC APE against NCU for
these corpora. Comparing a within-warp curve at a nominal card capacity with
one realized NCU counter would conflate chronology, associativity, client
scope, pending fills, and launch persistence. Such a score would be invalid,
even if its numerical error were below ten percent.

The valid result is the exact Class-A MRC and its same-tick interval envelope,
validated against structural invariants and deterministic replay. A future
Class-B endpoint may be scored only after an independent capacity sweep or a
scope-compatible NCU target is supplied; zero-target rows require exact-zero
classification rather than relative APE.

This is a documented observability boundary, not a simulator failure and not
permission to fit NCU labels.
