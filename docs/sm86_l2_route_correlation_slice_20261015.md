# SM86 L2 route-strata correlation — 2026-10-15

H35 correlates H34 route strata with independent NCU comparison rows without
using NCU values as model inputs. The correlation is a residual ledger, not a
parameter fit.

## Corpus strata

H34 contains 3,206 incomplete, 9,135 balanced, and 51 concentrated launches.
The concentrated class is intentionally sparse; it is not sufficient to
identify a queue/service law.

## Joined residual evidence

On exact post-replay joins, the concentrated class has higher cycle residuals
(GPC mean APE about 64%, SM-active about 31%, elapsed-IPC about 22%), but only
4–8 joined rows per metric. Incomplete-route rows show high DRAM mean APE
(about 65%) while L2 source-sector APE is near zero on the small joined subset.
Balanced rows dominate the joins and retain the established L2/DRAM/IPC
residuals. Family splits (LCD, SimBench, and SM86 modern traces) do not remove
the sparse-join confound.

Conclusion: route concentration is useful for stratified residual reporting but
does not supply a stable service timing differential. No L2 queue timing or NCU
fit is promoted; the active observer remains the scientifically valid endpoint.
