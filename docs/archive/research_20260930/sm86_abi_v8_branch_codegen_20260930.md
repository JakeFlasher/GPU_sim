# SM86 current ABI-v8 branch/control code-generation point

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09, locked 1350/10501 MHz.

## New bounded capability

A separate SM86 branch kernel broadens the no-template current-ABI surface beyond
the earlier straight-line vector body. The body contains:

* pointer/pointer/u32 parameters, with a 20-byte `PARAM_CBANK` extent;
* `ISETP.GE`, predicated `EXIT`, `LOP3`, `IMAD`, `IADD3`;
* descriptor-form `LDG.E` and `STG.E`;
* two exit offsets (`0x30` and `0xf0`);
* a 512-byte text image with unreachable tail NOPs.

The SASS surface was disassembled from an SM86 ptxas image, then re-encoded
through the public CuAssembler SM86 repository. The descriptor memory forms
were selected from its SM86 instruction repository; ptxas control words are
retained for this bounded bridge, so this is not independent scheduler/control
synthesis. The image uses the repaired current-ABI metadata and minimal valid
DWARF64 CIE/FDE shape.

The resulting no-template image is accepted by `cuobjdump --dump-elf`,
`cuobjdump --dump-sass`, and `nvdisasm`, and launches through the CUDA Driver
API. Five 1,000-launch runs returned the branch-sensitive prefix
`7,4,9,10,11,16,13,22`:

| run | CUDA-event envelope |
|---:|---:|
| 1 | 1.867776 ms |
| 2 | 1.953504 ms |
| 3 | 1.899264 ms |
| 4 | 1.937408 ms |
| 5 | 1.936384 ms |

Median: **1.936384 ms per 1,000 launches**.

Artifacts:
`results/alpha/sm86_abi_v8_branch_codegen_20260930/`.

## Boundary interpretation

This is a genuine expansion of the finite ABI/operation ledger to branch,
predicate, multiple exits, and descriptor-memory forms. It does not establish
a general CFG compiler, independent control-word scheduler, complete SM86
modifier/opcode coverage, dynamic resource/shared/texture/device-function ABI,
relocation/linker semantics, or unrestricted generation.
