# SM86 L2 line/sector residual ledger — 2026-10-16

H53 joins H48 modeled/effective L2 line-miss/sector-fill counters, source lanes,
pending/new-fill ownership, and route strata to H42 scope-aware exact L2 lookup
rows. There are 104 joins with both L2 lookup-hit and lookup-miss targets
remaining after scope exclusions.

Lookup-miss APE is exactly 0% across all 104 joins. Lookup-hit mean APE is
11.77% overall: 12.50% balanced (85 rows), 1.02% incomplete (15), and 36.77%
concentrated (4). Line-miss/sector-fill and pending/new-fill pressure
partitions are non-monotonic; no source-local correction is stable.

Decision: retain current modeled/effective line/sector state semantics and
source observers; do not add an L2 hit multiplier or route timing arm. The
remaining hit residual is concentrated in a sparse route class and needs
independent scope-matched probe evidence.
