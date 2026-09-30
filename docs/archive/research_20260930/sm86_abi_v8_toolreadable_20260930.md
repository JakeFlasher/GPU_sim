# SM86 ABI-v8 tool-readable vector point — debug-frame closure for the narrow image

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09, clocks 1350/10501 MHz.

## Result

The prior metadata repair made the independent pointer/uint4/pointer image
readable by `cuobjdump`, but `nvdisasm` still timed out on the copied
`.debug_frame` relocation. A final narrow metadata variant retains the valid
CUDA 13.x SM86 three-PHDR image, adds the missing text section symbol and
section links/flags, and emits a valid DWARF64 CIE/FDE shape whose
`initial_location` is zero and whose debug relocation section is empty. This is
sufficient for all three installed binary tools:

* `cuobjdump --dump-elf`: success;
* `cuobjdump --dump-sass`: success;
* `nvdisasm`: success.

The CIE/FDE is intentionally minimal and does not claim a complete source-level
or relocatable debug/linker implementation. It only makes the admitted final
image parseable while preserving the same 14-operation-family body.

The CUDA Driver API launches the body and consumes all four vector components,
returning `0,3,6,9`:

| run | CUDA-event envelope |
|---:|---:|
| 1 | 1.721344 ms |
| 2 | 1.777664 ms |
| 3 | 1.740640 ms |
| 4 | 1.729536 ms |
| 5 | 1.793024 ms |

Median: **1.740640 ms per 1,000 launches**.

Artifacts:
`results/alpha/sm86_abi_v8_param_campaign_20260930/uint4_toolreadable_consumed/`.

## Boundary interpretation

This closes the observed image-format/tool-readability defects for one current
ABI-v8 body. It does **not** close unrestricted generation: arbitrary section
and program-header synthesis, real debug/relocation semantics, resource and
shared/texture/device-function ABI, all SM86 instruction/modifier/control
forms, scheduler generation, and linker semantics remain outside the admitted
capability. The compiler/ISA blocker is narrowed to those general seams rather
than declared solved.
