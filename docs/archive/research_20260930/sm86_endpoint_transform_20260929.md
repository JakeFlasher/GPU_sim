# SM86 CUDA-event to NCU endpoint transform

A live unprofiled CUDA-event campaign was collected for independent warm shape
anchors `Sk=64,96,192,256`, then aligned with their paired-warm NCU profiles.
The target `Sk=128` event timing and NCU values were held out from fitting.

## Independent response

| Sk | CUDA event ns | NCU duration ns | NCU GPC cycles |
|---:|---:|---:|---:|
| 64 | 72,759 | 66,560 | 89,797 |
| 96 | 104,677 | 98,432 | 132,846 |
| 192 | 199,854 | 193,408 | 261,059 |
| 256 | 263,269 | 257,088 | 347,047 |

The endpoint transform fits a degree-2 surface from CUDA-event duration to NCU
duration/GPC metrics. On the score-only target, the direct CUDA event is 135,890
ns and the transform predicts 129,495 ns / 174,774 cycles. Fresh NCU measures
129,600 ns / 174,937 cycles: **0.081% duration / 0.093% GPC APE**.

The transform is retained at
`results/alpha/sm86_endpoint_transform_goal_20260929/transform.json`.
It is a local measurement transform for the paired-warm cache phase, not a
unified clock model: NCU and CUDA events retain different start/end semantics.
