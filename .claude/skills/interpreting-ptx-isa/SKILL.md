---
name: interpreting-ptx-isa
description: "Resolve PTX instruction, memory-ordering, cache-modifier, lane-participation or inline-assembly semantics for the explicitly selected GPU target."
---

# PTX semantics

Use the smallest relevant section under `references/NVIDIA_Manuals/ptx_isa/`;
inline constraints/clobbers live under `references/NVIDIA_Manuals/cuda_inline_ptx/`.
Optional `.claude/knowledge/ptx/` cards may be absent. Search by the exact
mnemonic, qualifier and target, rather than loading the whole instruction set.

Distinguish PTX virtual semantics from compiled SASS and simulator resource
hypotheses. Check target/version requirements and participant masks explicitly.
For synchronization inspect scope, memory ordering, address space, arrival and
wait semantics together; cache operators alone do not provide synchronization.
For async copies, separate source completion, destination visibility, and
barrier participation. Do not extrapolate Hopper/Blackwell primitives to SM86.

Use `cuobjdump` for emitted SASS and a bounded executable probe when semantics
are uncertain. Native trace replay and `simt_its_v1` are not general PTX
interpreters; see `docs/STATUS.md` for the implemented subset boundaries.
