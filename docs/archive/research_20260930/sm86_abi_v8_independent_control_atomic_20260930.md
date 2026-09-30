# SM86 bounded independent control generation — reduction/atomic body

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09, locked 1350/10501 MHz.

## Result

A current-ABI-v8 pointer/pointer/u32 body exercises a distinct reduction/atomic
family: VOTEU, UFLO, REDUX, and predicated `RED.E.ADD.STRONG.GPU`. The
instruction words were independently re-encoded through the SM86 CuAssembler
repository; control fields were synthesized from corrected SM86 operation
classes and dependency templates without copying ptxas control words.

Five Driver runs return the expected atomic reduction output `32896000`:

| run | CUDA-event envelope |
|---:|---:|
| 1 | 1.935296 ms |
| 2 | 1.873760 ms |
| 3 | 1.867776 ms |
| 4 | 1.943392 ms |
| 5 | 1.885184 ms |

Median: **1.885184 ms per 1,000 launches**.

Artifacts:
`results/alpha/sm86_abi_v8_independent_control_codegen_20260930/atomic_reduction/`.

## Boundary

This broadens independent control/codegen coverage to reduction and atomic
operation families. It does not expose cache ownership/read-from/DRAM
completion and does not establish unrestricted SM86 compiler/linker closure.
