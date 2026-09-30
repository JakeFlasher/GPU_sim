# SM86 bounded independent control generation — cache-operator body

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09, locked 1350/10501 MHz.

## Result

A pointer/pointer/u32 current-ABI-v8 body with descriptor-form global load/store
(the ptxas lowering of an inline cache-operator body) was independently
re-encoded and control-scheduled. Special memory templates use the corrected
SM86 dependency fields, including the LDG wait and store wait mask.

The no-template image executes through the Driver API and returns
`1,2,3,4,5,6,7,8` on all five runs:

| run | CUDA-event envelope |
|---:|---:|
| 1 | 1.860608 ms |
| 2 | 1.863680 ms |
| 3 | 1.960960 ms |
| 4 | 1.892352 ms |
| 5 | 1.895424 ms |

Median: **1.892352 ms per 1,000 launches**.

Artifacts:
`results/alpha/sm86_abi_v8_independent_control_codegen_20260930/cache_operator/`.

## Boundary

This extends independent control/codegen coverage to a cache-sensitive global
memory operation family. It does not expose cache ownership/read-from/DRAM
completion and does not establish unrestricted SM86 generation.
