# SM86 DRAM dirty-tail residual ledger — 2026-10-15

H44 joins the 58 scope-compatible exact DRAM rows from H42 with simulator dirty
residency, dirty-eviction, and typed tail flags. All 58 rows carry both
`dirty_writeback` and `coherence` unmodeled-tail flags. No row has dirty L2
eviction sectors; the varying demand is terminal L2 dirty residency.

Dirty-residency stratification is non-monotonic:

| partition | n | mean DRAM APE |
|---|---:|---:|
| low dirty-residency quartile | 13 | 21.29% |
| middle dirty-residency quartile | 31 | 35.82% |
| high dirty-residency quartile | 14 | 12.10% |

The high-residency rows include several near-exact predictions, while other
rows at the same tail state retain large residuals. This is confounded by
read-demand volume, workload family, and launch shape; it does not identify a
writeback bandwidth, queue capacity, or coherence service law.

Decision: retain the typed dirty/coherence tails and existing optional
writeback scenario as explicit research surfaces. Do not enable default
writeback timing or infer hardware completion from terminal gauges.
