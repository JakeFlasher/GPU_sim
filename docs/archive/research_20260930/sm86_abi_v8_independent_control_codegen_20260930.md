# SM86 bounded independent control-word generation

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09, locked 1350/10501 MHz.

## Result

A 56-row structured-control loop was regenerated with:

1. CuAssembler SM86 instruction words for the operation/branch forms;
2. an independent control-word synthesizer using the corrected SM70/SM86 field
   layout (delay, yield, write barrier, read barrier, wait mask, reuse mask);
3. opcode-class templates and explicit dataflow rules for the initial S2R wait,
   LDG consumer wait-mask, and backward-loop reuse.

The generated image did **not** copy ptxas control words. It loads through the
Driver API, disassembles successfully with `nvdisasm`, and returns
`0,2,5,9,4,6,9,13` on all five runs:

| run | CUDA-event envelope |
|---:|---:|
| 1 | 2.028544 ms |
| 2 | 2.019264 ms |
| 3 | 2.092032 ms |
| 4 | 2.095104 ms |
| 5 | 2.112512 ms |

Median: **2.092032 ms per 1,000 launches**. The retained ptxas-control image
median is 2.025472 ms under the same locked-clock launcher.

Artifacts:
`results/alpha/sm86_abi_v8_independent_control_codegen_20260930/structured_loop/`.

## Boundary

This is a genuine independent control-word generation point for one structured
SM86 body and materially strengthens the ABI/codegen ledger. It is not
unrestricted generation: control templates remain operation-family-specific,
full scheduler optimization is not synthesized, and arbitrary SM86 modifiers,
resources, relocations, device functions, and linker semantics remain open.
