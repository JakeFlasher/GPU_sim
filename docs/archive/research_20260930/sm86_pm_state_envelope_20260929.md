# SM86 PM-state phase envelope (2026-09-29)

## Research basis

The public Accel-Sim GPUVision path demonstrates that CUPTI PM Sampling can
provide timestamped, cycle-level aggregate counter streams for simulator
correlation, while the CUPTI tutorial exposes fixed GPU-sysclk sampling and
decoded sample records.  GPUMech's interval analysis supplies the complementary
idea of modeling a sequence as sustained issue intervals plus explicit stall
events.  The PM-state envelope combines those ideas without treating a sampled
warp as a per-instruction physical route witness.

References: [Accel-Sim/GPUVision](https://github.com/accel-sim/accel-sim-framework),
[CUPTI PM sampling](https://docs.nvidia.com/cupti/tutorial/tutorial.html), and
[GPUMech](https://hsienhsinlee.github.io/MARS/pub/micro14.pdf).

## Model

Independent paired-warm PM anchors are converted into a state vector of:

```text
long-scoreboard, wait, short-scoreboard, branch, barrier, math-throttle shares
+ total sampled stalls + log executed instructions
```

A ridge envelope predicts GPC cycles and duration independently.  The target
PM labels are not used in calibration.  Rows without the exact shape use the
independent frontier fallback, so no workload is discarded.

Executable: `tools/sm86_pm_state_envelope.py`.

## Fresh score-only result

The warm flash target is covered by four independent `Sk=64/96/192/256`
anchors.  The predicted aggregate PM state is approximately:

| state | share |
|---|---:|
| long scoreboard | 56.58% |
| wait | 23.10% |
| short scoreboard | 11.03% |
| branch | 6.88% |
| barrier | 1.60% |
| math throttle | 0.81% |

The score-only prediction is `183,942` GPC cycles / `136,293` ns versus fresh
NCU `175,094` / `129,760` (5.05% / 5.03% APE).  Its intervals are
`175,020--192,864` cycles and `129,675--142,911` ns, containing the measured
values.  The full large holdout MAPE is 11.23%/11.25% because unmatched rows
use the generic fallback; this is a negative result against the promoted
archetype path, so the model is not promoted.

The value of this path is phase interpretability and an interval that can be
conditioned on future PM time series.  It does not identify a physical pipe,
issue slot, cache owner, read-from source, ABI metadata, or unified timer.

