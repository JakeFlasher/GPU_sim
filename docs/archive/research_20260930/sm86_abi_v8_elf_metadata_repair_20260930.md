# SM86 ABI-v8 ELF metadata repair and tool-readable vector point

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09, clocks 1350/10501 MHz.

## Repair

A field-level comparison against a real CUDA 13.x SM86 image showed that the
hand emitter had three concrete ELF metadata defects:

* `.nv.info` and `.nv.callgraph` did not link to `.symtab` (`sh_link=3`);
* `.nv.constant0.<kernel>` lacked `SHF_INFO_LINK` (`flags=0x42`);
* `.symtab` omitted the `.text.<kernel>` section symbol, leaving the global
  kernel symbol at the wrong index.

The three-program-header layout is retained because it is the layout emitted by
CUDA 13.x for the observed SM86 final images; the six-header layout in the
public reverse-engineering reference is not universal across architectures or
CUDA releases.

With only those metadata repairs, the no-template pointer/uint4/pointer image:

* loads and runs through the CUDA Driver API;
* returns `0,3,6,9` after consuming all four vector components;
* is accepted by `cuobjdump --dump-elf`;
* is accepted by `cuobjdump --dump-sass`.

The retained `nvdisasm` path still times out while decoding the hand-emitted
`.debug_frame`/relocation combination. This is a remaining tool-compatibility
limit, not hidden success.

## Locked-clock performance

Five 1,000-launch Driver API runs returned:

| run | CUDA-event envelope |
|---:|---:|
| 1 | 1.736704 ms |
| 2 | 1.727488 ms |
| 3 | 1.776640 ms |
| 4 | 1.800928 ms |
| 5 | 1.767424 ms |

Median: **1.767424 ms per 1,000 launches**.

Artifacts:
`results/alpha/sm86_abi_v8_param_campaign_20260930/uint4_toolfix_consumed/`.

## Boundary

This converts the former “invalid ELF” observation into a more precise result:
a finite current-ABI-v8 image can be made readable by `cuobjdump` and accepted
by the driver for the admitted body. It does not provide general phdr/section
synthesis, debug/relocation correctness, all instruction/control forms,
resource/shared/texture/device-function ABI, or linker semantics. The
unrestricted compiler/ISA blocker therefore remains open, but its next seam is
now concrete and measurable rather than speculative.

Reference consulted for the SM86/CUDA 13.x header and section shape:
<https://gh.evko.io/crucible-notes/ptxas/reference/data/elf-header-notes.html>
