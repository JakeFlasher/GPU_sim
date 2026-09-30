# SM86 PM-sampling shape response

Independent paired-warm PM-sampling profiles at `Sk=64,96,192,256` are retained
under `results/alpha/sm86_flash_bwd_pmsampling_shape_goal_20260929/`. Each has
335--666 samples at a 2,048-cycle interval without buffer overflow.

A degree-2 interpolation over independent anchors predicts the score-only
`Sk=128` stall mixture. Against the fresh target PM profile:

| stall population | prediction APE |
|---|---:|
| wait | 0.02% |
| short scoreboard | 0.005% |
| branch resolving | ~0% |
| barrier | 0.32% |
| math-pipe throttle | 1.60% |
| long scoreboard | 22.6% |

Two fresh target repeats are retained at
`results/alpha/sm86_flash_bwd_pmsampling_target_repeats_20260929/`. Both report
long-scoreboard 980.70/980.76, matching the independent-anchor prediction
980.995. The earlier 1,267.65 target observation is therefore a
campaign/sampling-state outlier rather than a stable deterministic law. The
other populations also interpolate closely, but no metric exposes exact
dynamic pipe or issue-slot identity.
