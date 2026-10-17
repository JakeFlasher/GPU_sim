# SM86 L1 lookup retry hypothesis — negative result — 2026-10-15

## Question

Modern Ampere models expose finite L1 memory queues/PRT and retry stalled
misses. H39 tested whether the simulator's L1 lookup-sector denominator should
count MSHR-full miss attempts again, using the H38 occupancy ledger and
independent post-replay NCU joins. NCU values were not used as replay inputs.

## Evidence

The H38 complete full-first artifact covers 12,392 launches, with 107 exact
identity NCU joins. Across those joins, Pearson correlations between
MSHR-full-stall samples and absolute percentage error are:

| NCU metric | correlation |
|---|---:|
| global-load sectors | −0.015 |
| global-load lookup hits | −0.041 |
| global-load lookup misses | −0.018 |
| GPC elapsed cycles | 0.046 |
| SM active cycles | −0.032 |

The high-stall quartile does not show a consistent lookup residual direction;
some zero-stall launches have larger lookup errors than the streaming roots.
The modern retry/PRT prior therefore does not justify adding stalled attempts to
the L1 lookup denominator or altering timing.

## Decision

No correction was implemented. H38's observer-only MSHR pressure rows remain
the active model surface, and the H38 full/fast compressed artifacts and report
remain the retained evidence. A future retry model needs a dedicated locked
probe with request-level NCU semantics, not a corpus-wide count adjustment.
