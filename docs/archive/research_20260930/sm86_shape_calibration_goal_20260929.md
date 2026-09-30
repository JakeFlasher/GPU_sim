# Live BF16 HMMA shape calibration

Direct NCU profiling of the existing SM86 BF16 HMMA runner at independent
shape variants, without test/correctness/hash/verification stages.

| shape | NCU duration | GPC cycles | instructions | tensor aggregate | ALU aggregate |
|---|---:|---:|---:|---:|---:|
| 64x64x64 | 4.128 us | 5,532 | 4,448 | 128 | 768 |
| 128x128x128 | 5.888 us | 7,872 | 28,288 | 1,024 | 3,840 |
| 192x192x192 | 8.256 us | 11,107 | 87,264 | 3,456 | 10,368 |
| target 256x256x256 | 9.952 us | 13,374 | 197,120 | 8,192 | 21,504 |
| 384x384x384 | 18.080 us | 24,343 | 632,448 | 27,648 | 62,208 |
| 512x512x512 | 42.304 us | 57,119 | 1,460,224 | 65,536 | 135,168 |
| 768x768x768 | 127.520 us | 172,172 | 4,796,928 | 221,184 | 414,720 |

The independent 192/384 points bracket the target. Their conservative target
intervals are 8,256–18,080 ns and 11,107–24,343 GPC cycles; the observed 256
target lies inside both. Instruction and tensor populations scale predictably,
but GPC cycles do not follow a single linear or log-linear law. The next
analytical path should emit intervals/quantiles with occupancy state instead of
forcing a scalar timing multiplier. Raw NCU profiles and `shape_matrix.csv`
are under `results/alpha/sm86_shape_calibration_goal_20260929/`.

The executable interval helper is
`tools/sm86_shape_interval_envelope.py`. For shape 256 it emits
`target_256_interval.json`: 8,256–18,080 ns and 11,107–24,343 GPC cycles from
the independent 192/384 anchors. The target NCU measurement is not an input to
that envelope.
