---
name: interpreting-ptx-isa
description: Routes PTX instruction-semantics questions to distilled PTX cards in `.claude/knowledge/ptx/` before opening raw NVIDIA manuals. Use when prompts mention PTX mnemonics (mma, wgmma, tcgen05, ldmatrix, cp.async, mbarrier, bar.sync, fence, cvt, setp, shfl.sync, ballot.sync), state spaces (.global/.shared/.local/.const/.param), cache modifiers, FP4/FP6/FP8/BF16/TF32/NVFP4 data types, per-architecture deltas (sm_70 ITS through sm_120a tcgen05/TMEM/FP4), or inline-PTX `asm` blocks inside .cu files. Prefer distilled cards over raw `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md`.
---

# Interpreting PTX ISA

This skill owns PTX instruction semantics — the largest specialty subcorpus (13 ptx/ cards + 1 inline-ptx/ card).

## Cards (open the relevant card BEFORE the raw manual)

| Topic | Card |
|---|---|
| Tensor cores: `mma.sync` / `wgmma.mma_async` / `tcgen05.mma` / `ldmatrix` / `stmatrix` | `.claude/knowledge/ptx/ptx-isa-tensor-cores.md` |
| Async copy: `cp.async` / `cp.async.bulk` / `cp.async.bulk.tensor` (TMA) | `.claude/knowledge/ptx/ptx-isa-async-copy.md` |
| Barriers: `bar.sync` / `mbarrier` / `mbarrier.complete_tx` / `barrier.cluster` / `wgmma.commit_group` / `tcgen05.commit` | `.claude/knowledge/ptx/ptx-isa-barriers.md` |
| Load/store/atomic/reduction: `ld` / `st` / `atom` / `red` qualifiers | `.claude/knowledge/ptx/ptx-isa-load-store.md` |
| Memory spaces & fences: `.global` / `.shared` / `.local` / `.const` / `.param` / `fence` / `membar` | `.claude/knowledge/ptx/ptx-isa-memory-spaces.md` |
| Cache modifiers: `.ca` / `.cg` / `.cs` / `.cv` / `.lu` / `.wt` / `.wb` | `.claude/knowledge/ptx/ptx-isa-cache-hints.md` |
| Data types: FP4 / FP6 / FP8 / FP16 / BF16 / TF32 / NVFP4 / MX block-scaling | `.claude/knowledge/ptx/ptx-isa-data-types.md` |
| Arithmetic: `add` / `sub` / `mul` / `mad` / `fma` / `div` / `rcp` / `sqrt` / rounding modes | `.claude/knowledge/ptx/ptx-isa-arithmetic.md` |
| Control flow & predicated execution: `bra` / `setp` / `@p` | `.claude/knowledge/ptx/ptx-isa-control-flow.md` |
| Warp ops: `shfl.sync` / `vote.sync` / `match.sync` / `ballot.sync` / `activemask` | `.claude/knowledge/ptx/ptx-isa-warp-ops.md` |
| Misc: `prmt` / `popc` / `clz` / `brev` / `bfe` / `bfi` / `nanosleep` / `setmaxnreg` | `.claude/knowledge/ptx/ptx-isa-misc.md` |
| sm_90 Hopper deltas | `.claude/knowledge/ptx/ptx-isa-sm90-hopper.md` |
| sm_100 / sm_120 Blackwell deltas | `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md` |
| Inline PTX from `.cu` (asm constraint letters, clobbers) | `.claude/knowledge/inline-ptx/inline-ptx-from-cu.md` |

## Routing decision tree

1. Is it about a Hopper-specific instruction (`wgmma`, `cp.async.bulk.tensor`, `barrier.cluster`)? → `ptx-isa-sm90-hopper.md`.
2. Is it about a Blackwell-specific instruction (`tcgen05.mma`, TMEM, FP4 block-scaled MMA)? → `ptx-isa-sm100-blackwell.md`.
3. Otherwise classify by mnemonic family from the table above.

## When to escalate

The 13 cards are the fast path. **If the card doesn't answer the question, open `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md`** at the §9.7.x subsection cited in the card's `Primary raw source:` frontmatter line. Do NOT browse the raw manual blindly — it is ~26,000 lines.

The PTX ISA chapter map (verified 2026-05):

| § | Topic |
|---|---|
| 9.7.1 | Integer Arithmetic (also bit ops 9.7.1.14–22: popc, clz, brev, bfe, bfi, prmt) |
| 9.7.2 | Extended-Precision Integer Arithmetic |
| 9.7.3 | Floating-Point |
| 9.7.4 | Half Precision Floating-Point |
| 9.7.5 | Mixed Precision Floating-Point |
| 9.7.6 | Comparison and Selection |
| 9.7.7 | Half Precision Comparison |
| 9.7.8 | Logic and Shift |
| 9.7.9 | Data Movement and Conversion (ld, st, cp.async, cp.async.bulk, cvt, cache operators, ldmatrix, stmatrix) |
| 9.7.10 | Texture |
| 9.7.11 | Surface |
| 9.7.12 | Control Flow |
| 9.7.13 | Parallel Synchronization and Communication (bar.sync, mbarrier, fence, shfl.sync, vote.sync, ballot.sync) |
| 9.7.14 | Warp-Level Matrix Multiply-Accumulate (mma.sync, ldmatrix, stmatrix) |
| 9.7.15 | Asynchronous Warpgroup Level Matrix Multiply-Accumulate (wgmma.mma_async) |
| 9.7.16 | TensorCore 5th Generation Family (tcgen05.mma) |
| 9.7.17 | Stack Manipulation |
| 9.7.18 | Video |
| 9.7.19 | Miscellaneous (nanosleep, trap, brkpt, pmevent, setmaxnreg) |

## Compute-capability surface (for "does this run on my GPU?")

| Feature | Min CC |
|---|---|
| `_sync` warp intrinsics (mask required) | sm_70 (Volta) |
| `mma.sync` (1st-gen Tensor Core) | sm_70 |
| `ldmatrix`, INT8/INT4 MMA | sm_75 (Turing) |
| `cp.async`, BF16, TF32, structured 2:4 sparsity, FP64 Tensor Core | sm_80 (Ampere) |
| FP8 (E4M3/E5M2) | sm_89 (Ada) / sm_90 (Hopper) |
| `cp.async.bulk`, `cp.async.bulk.tensor` (TMA), `wgmma.mma_async`, `mbarrier.complete_tx`, cluster + DSMEM | sm_90 (Hopper) |
| `tcgen05.mma`, TMEM, OMMA/QMMA, FP4/FP6/MX block-scaled | sm_100 / sm_120 (Blackwell) |
| Architecture-conditional features (FP4 NVFP4 specifics) | `sm_120a` opt-in |

## Cross-references

- Hopper architecture context → `tuning-blackwell-sm120-kernels` (not strictly Blackwell — but for SM120 it's the active skill); patent provenance → `reasoning-about-nvidia-architecture-history`.
- SASS-level realization of these PTX instructions → `analyzing-sass-nvbit-traces`.
- nvcc/ptxas compilation context → `compiling-cuda-artifacts`.
- Synchronization model formal axioms → `references/NVIDIA_Manuals/ptx_isa/08_memory-consistency-model.md`.

## Cite consulted files

End every reply with `## Knowledge Consulted` listing every card / raw source read.
