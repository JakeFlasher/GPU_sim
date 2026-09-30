# SM86 independent ELF/codegen v2 — current ABI-v8 narrow launch

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09, clocks 1350/10501 MHz.

## New result

A second from-scratch ELF emitter was built outside the repository. It emits:

* CuAssembler-generated SM86 words for the retained 14-instruction NAK body;
* current CUDA ABI-v8 header fields (OSABI 0x41, ABI 8, flags 0x06005604);
* CUDA 13.2 toolkit/cubin note records;
* current .nv.info language/API/parameter/Mercury/exit attributes;
* .nv.callgraph, .nv.rel.action, .rel.debug_frame, constant-bank and
  load-segment metadata;
* a 24-instruction text section padded with independently encoded EXIT words.

No NVIDIA cubin template is read by the emitter. The generated file loaded and
executed through the CUDA Driver API on the target card. Five runs of 1,000
launches with an 8-thread block returned 0,3,6,9:

| run | CUDA-event envelope |
|---:|---:|
| 1 | 1.727488 ms |
| 2 | 1.715040 ms |
| 3 | 1.642208 ms |
| 4 | 1.638240 ms |
| 5 | 1.733632 ms |

Median: **1.715040 ms per 1,000 launches**.

Artifacts are under
results/alpha/sm86_independent_elf_codegen_20260930/v8_current_abi/.

## Boundary

This closes a materially stronger but still narrow compiler capability:

* independent SM86 instruction encoding for the admitted body;
* an independent current-ABI-v8 ELF/container/parameter path for this body;
* native Driver API launch without an NVIDIA cubin template.

It does **not** close unrestricted generation. Remaining fields include:

* general ABI-v8 metadata/relocation synthesis;
* callgraph and debug/rel-action generation for arbitrary kernels;
* complete SM86 instruction/modifier/control-word coverage;
* dynamic shared memory, texture/surface, cooperative and device-function ABI;
* independent scheduler/control-word synthesis;
* current ABI behavior across parameter layouts and kernel families.

The original generated file was accepted by the CUDA Driver API but reported
`Invalid ELF` from `cuobjdump --dump-elf`. A later field-level repair added the
missing text section symbol, corrected `.nv.info`/`.nv.callgraph` links, and set
`.nv.constant0` flags; the repaired image is readable by `cuobjdump --dump-elf`
and `--dump-sass`. The three-program-header layout is normal for the observed
CUDA 13.x SM86 image. A subsequent minimal DWARF64 CIE/FDE variant with no debug relocation makes
`cuobjdump --dump-elf`, `cuobjdump --dump-sass`, and `nvdisasm` all complete for
the admitted image. This is tool-readability closure for one body, not a
complete debug/relocation implementation; general relocatable/final
 distinctions and linker/resource records remain explicit ABI limits. See
`sm86_abi_v8_toolreadable_20260930.md`.


## Three-parameter ABI expansion attempt

A temporary v8 emitter variant changed the parameter CBANK size from 16 to 24
bytes and added a third 64-bit KPARAM_INFO record, while leaving the admitted
body's first two pointers unchanged. A Driver API launcher supplied three
pointers. The generated file loaded, but launch returned
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES on every attempted launch.

This negative narrows the remaining ABI boundary: current v8 metadata is not
generalized by simply appending a KPARAM_INFO record and enlarging the
parameter block. The failure is retained under
results/alpha/sm86_independent_elf_codegen_20260930/v8_three_param_attempt/
and is not treated as a test/verification gate.


## Correction: parameter-bank extent

The initial three-pointer attempt advertised a 24-byte KPARAM list but retained a
16-byte PARAM_CBANK extent; its launch failure was therefore an emitter defect,
not an ABI impossibility. After correcting both the PARAM_CBANK extent and the
constant-bank allocation to 24 bytes, the same no-template ABI-v8 emitter
launched the three-pointer body successfully through the Driver API:

    launches=1000 params=3 event_ms=1.722368 first_values=0,3,6,9

Artifact:
results/alpha/sm86_abi_v8_param_campaign_20260930/extent_fix/.

This closes a second narrow current-ABI parameter-layout point (three
64-bit parameters, with the body consuming the first two). It still does not
generalize arbitrary layouts, relocations, resources, or all operation forms.
The earlier failure is retained as a corrected intermediate observation and
must not be cited as an external ABI blocker.


A four-pointer ABI-v8 variant was then emitted with a 32-byte PARAM_CBANK
extent and four KPARAM_INFO records. A Driver API launcher supplied all four
pointers; five 1,000-launch runs returned 0,3,6,9 with event envelopes
1.705760, 1.731552, 1.705952, 1.679072, and 1.732608 ms (median
1.705952 ms). Artifact:
results/alpha/sm86_abi_v8_param_campaign_20260930/four_param/.

This expands the observed current-ABI parameter seam to three and four
64-bit pointer layouts. It remains a finite capability result, not an
unrestricted parameter ABI or complete linker.


A mixed pointer/scalar/pointer ABI-v8 variant was then generated. The body was
independently re-encoded so its input address uses c0[0x170]/c0[0x174], while
the output remains c0[0x160]/c0[0x164]. Metadata matches the current ptxas
shape: 24-byte parameter block, reverse KPARAM_INFO records with 64-bit/32-bit/
64-bit size flags, and offsets 0/8/16. A Driver API launcher supplied
(pointer, u32, pointer) and returned 0,3,6,9 over 1,000 launches at
**1.725152 ms**.

Artifact:
results/alpha/sm86_abi_v8_param_campaign_20260930/mixed_pointer_u32_pointer/.

This materially broadens the current-ABI narrow capability to heterogeneous
pointer/scalar parameter layouts, but remains finite operation/body coverage;
it is not an unrestricted compiler/linker.


A stronger mixed ABI-v8 variant consumes the 32-bit scalar at c0[0x168]:
the independently re-encoded IMAD uses that scalar as its multiplier, while
the third pointer is read from c0[0x170] and output remains c0[0x160]. With
scalar n=3, 1,000 Driver API launches returned 0,3,6,9 at 1.722368 ms.

Artifact:
results/alpha/sm86_abi_v8_param_campaign_20260930/mixed_scalar_consumed/.

This confirms that the heterogeneous parameter metadata is not merely ignored
by the body; it remains a narrow ABI/codegen capability result.


A current ABI-v8 uint2 parameter variant now also launches without a template.
The body consumes the first 32-bit vector component as an IMAD multiplier while
using the third pointer as input; the second vector component remains in the
declared 8-byte parameter record. Three 1,000-launch runs returned 0,3,6,9 at
1.697792, 1.675104, and 1.631232 ms.

Artifact:
results/alpha/sm86_abi_v8_param_campaign_20260930/uint2_consumed/.

This extends the exercised ABI family to an actually consumed vector parameter,
but does not establish arbitrary vector widths, structs, relocations, or
general device-function ABI.


A correction to the ABI-v8 uint4 (16-byte vector) point is recorded in
`sm86_abi_v8_uint4_flagfix_20260930.md`. The earlier emitter used the correct
40-byte layout but the 64-bit-pointer KPARAM size word (`0x21f000`) for the
vector; its output was `0,3,0,0` and is not promoted as a successful vector
result. The corrected emitter uses the ptxas-derived 16-byte vector word
`0x41f000`, with offsets 0/16/32 and a 40-byte parameter block. Five
1,000-launch Driver API runs consumed all four vector components and returned
0,3,6,9 at 1.762304, 1.747968, 1.763328, 1.793024, and 1.752064 ms
(median **1.762304 ms**).

Artifact:
`results/alpha/sm86_abi_v8_param_campaign_20260930/uint4_flagfix_consumed/`.

This removes an emitter defect and extends the narrow current-ABI seam to a
semantically consumed 16-byte vector parameter. Arbitrary structs/relocations,
resource and control synthesis, linker semantics, and complete SM86 coverage
remain unresolved.
