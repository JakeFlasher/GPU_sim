# Custom lane-reduction probe — 2026-09-24

The bounded custom SIMT frontend now includes `reduce_add`, which computes a
deterministic wrapped-u32 sum across active lanes and broadcasts the result to
each lane’s destination register. It exercises a reduction semantic category,
not hardware reduction throughput or weak-memory behavior.

The independent functional/timing differential reports `functional_match=true`
and `timing_match=true`; for eight lane IDs the final value is `33` (sum
`0..7` plus `5`).
