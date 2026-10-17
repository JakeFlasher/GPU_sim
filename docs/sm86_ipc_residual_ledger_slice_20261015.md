# SM86 IPC/cycle residual ledger — 2026-10-15

H45 joins exact H42 cycle predictions to causal frontiers, H37 L2 route/fill
pressure, H38 MSHR pressure, and typed unmodeled tails. NCU values remain
post-replay targets only.

For 107 exact joins, GPC mean APE is 12.90% and SM-active mean APE 18.02%.
Route partitions show a sparse but directionally higher residual:

| route | n | GPC APE | SM-active APE |
|---|---:|---:|---:|
| balanced | 88 | 11.67% | 15.07% |
| incomplete | 15 | 16.08% | 31.76% |
| concentrated | 4 | 28.03% | 31.32% |

All but four exact rows carry dirty-writeback/coherence tails. Issue-terminal
frontier gap and pending-merge pressure partitions are non-monotonic; high
pressure can coincide with either lower or higher residuals. The concentrated
class is too sparse for a service/timing law, and incomplete routes mix tail and
launch-shape effects.

Decision: no IPC/cycle timing correction is promoted. Preserve separate issue,
SM-active, GPC, and tail frontiers; require a larger independent route/queue
probe before any conditional timing arm.
