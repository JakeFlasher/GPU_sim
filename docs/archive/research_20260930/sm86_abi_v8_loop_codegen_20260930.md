# SM86 current ABI-v8 structured-control/loop code-generation point

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09, locked 1350/10501 MHz.

## New bounded capability

A third no-template current-ABI-v8 body exercises real structured control rather
than only predicated exit:

* pointer/pointer/u32 parameters with a 20-byte parameter block;
* `BSSY`/`BSYNC` structured barriers;
* multiple predicated `BRA` paths and a loop;
* `PLOP3`, `LOP3`, `ISETP`, `IMAD.IADD`, and `IADD3`;
* descriptor-form `LDG.E` and `STG.E`;
* exit offsets `0x30` and `0x290`.

All 56 instruction rows were canonicalized and re-encoded through the public
CuAssembler SM86 repository. Ptxas control words are retained for this bounded
bridge, so independent scheduler/control synthesis is not claimed. The
repaired current-ABI metadata and minimal DWARF64 CIE/FDE are used.

The image is accepted by `cuobjdump --dump-elf`, `cuobjdump --dump-sass`, and
`nvdisasm`. Five locked-clock Driver runs return the loop-sensitive prefix
`0,2,5,9,4,6,9,13`:

| run | CUDA-event envelope |
|---:|---:|
| 1 | 2.113376 ms |
| 2 | 2.016000 ms |
| 3 | 2.114368 ms |
| 4 | 2.033664 ms |
| 5 | 1.992672 ms |

Median: **2.033664 ms per 1,000 launches**.

Artifacts:
`results/alpha/sm86_abi_v8_loop_codegen_20260930/`.

## Boundary interpretation

This materially expands the finite ABI/operation ledger to structured control,
real branches, barriers, and loops. It still does not establish independent
scheduler/control-word generation, complete SM86 modifiers/opcodes, dynamic
resource/shared/texture/device-function ABI, relocations/linking, or
unrestricted generation.
