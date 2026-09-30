# SM86 bounded independent control-word generation — branch body

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09, locked 1350/10501 MHz.

## Result

The branch/predicate current-ABI-v8 body was regenerated with CuAssembler SM86
instruction words and the corrected SM70/SM86 control-field synthesizer. No
ptxas control words were copied into the generated image. The body includes
pointer/pointer/u32 parameters, ISETP, predicated EXIT, LOP3, IADD3, descriptor
LDG/STG, and multiple branch forms.

The image parses with nvdisasm, runs through the Driver API, and returns
`7,4,9,10,11,16,13,22` on all five runs:

| run | CUDA-event envelope |
|---:|---:|
| 1 | 1.957888 ms |
| 2 | 1.879040 ms |
| 3 | 1.949344 ms |
| 4 | 1.896320 ms |
| 5 | 1.957888 ms |

Median: **1.949344 ms per 1,000 launches**. The retained ptxas-control
reference median is 1.936384 ms.

Artifacts:
`results/alpha/sm86_abi_v8_independent_control_codegen_20260930/branch/`.

## Boundary

This broadens independent control-word generation from the structured loop to a
branch/predicate/descriptor-memory family. It remains finite operation-family
coverage; unrestricted SM86 scheduling, modifiers, resources, relocations, and
linker semantics remain open.
