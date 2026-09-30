# SM86 independent encoding + minimal ELF emission (2026-09-30)

## Scope

This is a bounded compiler/ABI research observation. It does **not** claim
unrestricted CUDA ABI or SASS generation, and it adds no repository tests,
hashes, or verification workflow. The experiment asks whether the retained
NAK-generated body can execute after both instruction encoding and ELF metadata
are emitted without copying an NVIDIA cubin template.

## Inputs and method

- Source instruction body: the 14-instruction SM86 NAK `value_kernel` body
  already retained under `results/alpha/sm86_nak_cuda_elf_bridge_20260929/`.
- Encoder: the public CuAssembler SM86 repository (`DefaultInsAsmRepos.sm_86`),
  loaded in an isolated `uv` environment. All 14 instruction forms in this
  body (`S2R`, `SHF`, `PRMT`, `LEA`, `LEA.HI.X`, `LDG.E.128`, `IMAD.U32`,
  `STG.E.128`, `EXIT`) returned an encoding from that repository.
- ELF emitter: `emit_independent_two_param.py`. It constructs an ELF64 CUDA
  cubin from scratch: section string/symbol tables, `.nv.info`, per-kernel
  parameter metadata, `.nv.constant0.value_kernel`, `.text.value_kernel`,
  three program headers, and explicit `SHI_REGISTERS=19`. No NVIDIA-generated
  cubin is read or used as an ELF template.
- Launch ABI: two 64-bit pointer parameters at `c[0][0x160]` and
  `c[0][0x168]`, with high halves at `0x164`/`0x16c`; 32-lane launch adapter
  uses an 8-thread block only to bound the timing observation.

The emitted file is the older SM86 CUDA ELF ABI-v7 form (`EI_OSABI=0x33`,
`EI_ABIVERSION=7`, legacy SM86 flags `0x00560556`). Current CUDA 13.2/13.4
ptxas cubins use a newer ELF ABI/flag layout (`e_flags=0x06005604`, OSABI
`0x41`, ABI version 8). `nvdisasm` therefore warns that the independent file
uses the old format. This is deliberate evidence of the remaining ABI-version
boundary, not a claim that the current ABI was reconstructed.

## Device observation

The independent file loaded and executed on the RTX 3090 Ti (SM86) through the
CUDA Driver API. Five bounded runs, each 1,000 launches at locked 1350 MHz SM /
10501 MHz memory, returned:

```
launches=1000 block=8 event_ms=1.664000 first_values=0,3,6,9
launches=1000 block=8 event_ms=1.732608 first_values=0,3,6,9
launches=1000 block=8 event_ms=1.650688 first_values=0,3,6,9
launches=1000 block=8 event_ms=1.656768 first_values=0,3,6,9
launches=1000 block=8 event_ms=1.650688 first_values=0,3,6,9
```

Median CUDA-event envelope is **1.656768 ms per 1,000 launches** (range
1.650688--1.732608 ms). The returned values are a launch observation of the
existing body, not a correctness claim or a new campaign label.

## What this closes

1. **Instruction encoding for this operation family:** CuAssembler can emit the
   exact words needed by the retained NAK body, and the words execute in an
   independently built file.
2. **Minimal loadable ELF path:** section/symbol tables, `.nv.info`, parameter
   records, constant-bank placement, resource count, and load segments are
   sufficient for this narrow two-pointer kernel under the legacy SM86 format.
3. **Template dependence for this body:** the new cubin is not produced by
   replacing `.text` in an NVIDIA cubin; its ELF container is emitted by the
   local script.

## What remains open

- **Current CUDA ABI v8:** the ptxas 13.x layout (`OSABI=0x41`, ABI=8,
  `e_flags=0x06005604`) and associated `.note.nv.tkinfo`/`.note.nv.cuinfo`
  records are not reproduced. An attempt to switch only the header fields
  produced `CUDA_ERROR_INVALID_IMAGE`; therefore legacy-format execution must
  not be relabelled as current-ABI closure.
- **General metadata:** no general parameter-layout solver, relocation writer,
  callgraph/rel-action builder, texture/surface descriptor handling, dynamic
  shared-memory metadata, cooperative launch metadata, or device-function ABI
  exists in this emitter.
- **Instruction coverage:** 425 CuAssembler SM86 forms are available in the
  2023 default repository, but it is explicitly known to have poor SM86
  coverage. Unsupported/new forms, hidden control bits, and combinations
  outside the learned affine spans remain reject-on-unsupported.
- **Scheduling/control words:** the body reuses encoded control words learned
  by CuAssembler; there is no independent scheduler or physical issue/pipe
  witness.

Thus the result changes the next action from “template bridge only” to a
three-tier compiler contract: (a) independent legacy-ABI micro-kernels are
now executable, (b) current ABI-v8 generation remains a distinct unknown, and
(c) unrestricted all-form SASS/ABI generation remains unresolved.

## Public source anchors

- [CuAssembler](https://github.com/cloudcores/CuAssembler): affine/linear
  encoding repository, with explicit warning that SM86 coverage is incomplete
  and metadata/ABI are not a general linker.
- [DocumentSASS](https://github.com/reoLantern/nvsass-disassembler): extracts
  semantic operation/pipeline descriptions from `nvdisasm`; current CUDA 13.x
  interception exposed operation sets but no complete SM86 architecture file.
- [OpenPTXas](https://github.com/garrick247/openptxas): demonstrates an open
  PTX-to-cubin pipeline for SM120 and a bounded SM89 emitter, but does not
  provide SM86/current-ABI-v8 generation.
- [NVIDIA CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/):
  official disassembly/ELF tooling, not a public SASS assembler or ABI spec.

## Conclusion

This is a real new execution seam and materially strengthens the compiler/ISA
model, but it does **not** resolve the unrestricted compiler/ISA blocker. The
remaining blocker is now field-specific: current ABI-v8 container semantics,
general metadata/relocations, complete instruction-form coverage, and an
independent scheduler/control-word generator.
