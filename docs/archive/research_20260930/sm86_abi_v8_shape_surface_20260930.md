# SM86 current ABI-v8 parameter-shape surface

**Date:** 2026-09-30  
**Toolchain:** CUDA 13.2, ptxas sm_86.  
**Scope:** metadata source study; no launch or test workflow.

A temporary cubin contains six ptxas-generated kernels:

* pointer + u32;
* pointer + f32;
* pointer/u32/pointer;
* pointer/u32/u32/pointer;
* pointer/uint2/pointer;
* pointer/uint2/uint4/pointer.

The per-kernel nv.info records show:

* PARAM_CBANK base c[0][0x160];
* reverse-order KPARAM_INFO records;
* scalar size flags for 32-bit values;
* 8-byte pointer records;
* 16-byte vector records;
* natural alignment/padding (for example, pointer/u32/u32/pointer uses
  offsets 0, 8, 12, 16 and an 0x18-byte block; pointer/uint2/uint4/pointer
  uses offsets 0, 8, 16, 32 and an 0x28-byte block);
* exit-offset lists and current ABI-v8 metadata.

Artifacts: results/alpha/sm86_abi_v8_shape_surface_20260930/.

This source surface materially expands the ABI ledger, but it does not
provide a general linker, relocation solver, SASS scheduler/control generator,
or complete SM86 encoding coverage. The shape matrix is metadata evidence,
not a full compiler closure.
