# SM86 independent code-generation capability matrix

**Date:** 2026-09-30  
**Target:** RTX 3090 Ti / SM86, driver 615.71.09, locked 1350/10501 MHz.

## Observed no-template current-ABI-v8 families

| Family | ABI shape | Independent instruction words | Independent control synthesis | Tool-readable | Consumed output | Median / 1,000 launches |
|---|---|---|---|---|---|---:|
| straight-line vector | pointer/uint4/pointer | yes, bounded CuAssembler forms | bounded templates | yes | `0,3,6,9` | 1.740640 ms |
| branch/predicate | pointer/pointer/u32 | yes | bounded templates | yes | `7,4,9,10,11,16,13,22` | 1.949344 ms |
| structured loop | pointer/pointer/u32 | yes | corrected-field rule generator | yes | `0,2,5,9,4,6,9,13` | 2.092032 ms |
| cache-sensitive global memory | pointer/pointer/u32 | yes | descriptor-memory templates | yes | `1,2,3,4,5,6,7,8` | 1.892352 ms |
| reduction/atomic | pointer/pointer/u32 | yes | reduction/atomic templates | yes | `32896000` | 1.885184 ms |

The branch, loop, cache-sensitive, and reduction/atomic rows use no ptxas
control words in the generated images. The loop and branch generators use the
correct SM70+/SM86 dependency layout; the cache and atomic families add
operation-specific wait templates.

Artifacts are under:

* `results/alpha/sm86_abi_v8_independent_control_codegen_20260930/`
* `results/alpha/sm86_abi_v8_branch_codegen_20260930/`
* `results/alpha/sm86_abi_v8_loop_codegen_20260930/`

## Explicitly outside the admitted capability

* complete SM86 opcode/modifier/control coverage;
* global scheduler optimization and occupancy-aware control generation;
* tensor-core, texture/surface, dynamic shared-memory, cooperative/device-
  function, and device-linker ABI;
* general relocation, callgraph, debug, resource, and final-linker semantics;
* arbitrary parameter structs, alignment classes, and address-space forms;
* cross-kernel ABI and CUDA runtime integration.

The matrix is a capability ledger, not an unrestricted compiler claim. Unsupported
forms must remain rejected or widened in the model rather than silently mapped
to one of these finite operation families.
