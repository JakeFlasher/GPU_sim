---
Use when: identifying which generation introduced or removed a SASS opcode; reasoning about whether a given trace is sm_70 vs sm_80 vs sm_90 vs sm_100/120; understanding the SASS-level evidence for the architectural rewrites; preparing instruction-level evidence for a hardware-modeling claim.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §4 (ISA-level evolution).
Supporting sources: references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md (per-arch SASS catalogs); references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md (PTX equivalents).
Patents anchored here: each SASS group corresponds to one or more patents already cataloged in `patent-family-map.md`.
Out of scope: per-instruction syntactic detail (covered in `.claude/knowledge/sass/sass-instruction-reference.md` and the manual mirror); CUDA-side feature semantics.
Version family: Maxwell through Blackwell. Mirror snapshot: 2026-05.
---

# SASS Evolution — what each generation makes a first-class instruction

SASS instruction history is **not** a sparse "list of newly added
opcodes." It's another generation-level evidence trail closer to
hardware-boundary changes than patents alone. When a hardware object
becomes important enough to need its own SASS encoding, it gets one.
When an old protocol is replaced, its old encoding retires.

## Why SASS evolution is good evidence

An execution protocol that is purely a microarchitectural detail
typically does not appear stably in SASS readings. Once it becomes
`BSSY/BSYNC`, `DEPBAR`, `LDGSTS`, `UTMA*`, `UCGABAR_*`, `LDT/LDTM`,
or similar, the compiler, assembler, scheduler, and analysis tools
**must** handle it head-on. So SASS deltas tell you what NVIDIA
considers a stable, externally visible mechanism.

## Reading order

The five physical-architecture lines (`evolution-execution-model`,
`evolution-data-supply`, `evolution-memory-side`, `evolution-precision`,
`evolution-synchronization`, `evolution-interconnect`) describe what
the hardware does. The SASS layer projects those changes into machine
instructions.

A useful framing: when a group of SASS opcodes appears or retires
together, **what hardware boundary moved**?

## Per-generation deltas (Maxwell → Blackwell)

### Maxwell / Pascal

**Dominant SASS body**: traditional SIMT, classic load/store,
texture/surface, old-style reconvergence, integer/floating-point mixed
arithmetic.

**Reading**: ISA still serves the "general-purpose, lockstep, single-
`SM`-as-primary-boundary" machine. The Maxwell-era patents (software
scoreboard, convergence barrier) introduce new compiler-emitted
opcodes (e.g., `DEPBAR`, `BSSY/BSYNC` skeleton) but the *default
execution model* hasn't flipped yet.

### Volta — first real major rewrite at SASS layer

**Appearing**:

| New opcode | What it surfaces |
|---|---|
| `BSSY` | Barrier set convergence sync point |
| `BSYNC` | Synchronize threads on convergence barrier |
| `WARPSYNC` | Synchronize threads in warp |
| `DEPBAR` | Dependency barrier (software scoreboard wait) |
| `HMMA` | Half-precision matrix multiply-accumulate (Tensor Core) |
| `IMMA` | Integer matrix multiply-accumulate |
| `FADD32I` | FP32 add with 32-bit immediate |
| `FFMA32I` | FP32 fused multiply-add with 32-bit immediate |
| `IADD32I` | Integer add with 32-bit immediate |
| `MOV32I` | Move 32-bit immediate |

**Retiring**:

| Old opcode | What it was |
|---|---|
| `SSY` | Old-style single-step sync |
| `SYNC` | Pre-Volta reconvergence sync |
| `XMAD` | Pre-Volta integer multiply-add (16-bit chunked) |

**Reading**: the new `BSSY/BSYNC/WARPSYNC/DEPBAR` mark the rewrite of
execution sync objects; `HMMA/IMMA` mark matrix becoming a SASS
object; the `*32I` immediate forms reveal the new datapath
organization re-encoding immediates. Old objects retiring shows that
Volta's machine **doesn't need** the old execution protocol or the
old integer-MAC habit.

### Turing — uniform datapath lands officially

**Appearing** (uniform-path family — all `U*`):

| New opcode | What it surfaces |
|---|---|
| `R2UR` | Move from vector register to uniform register |
| `S2UR` | Move special register to uniform register |
| `UIADD3` | Uniform integer add with 3 sources |
| `UIMAD` | Uniform integer multiply-add |
| `UISETP` | Integer compare to set uniform predicate |
| `ULDC` | Uniform load from constant memory |
| `ULEA` | Uniform load effective address |
| `USEL` | Uniform select |
| `USHF` | Uniform funnel shift |
| `VOTEU` | Voting with uniform destination |

Plus matrix-side:

| New opcode | What it surfaces |
|---|---|
| `LDSM` | Load matrix from shared memory |
| `BMMA` | Bit (1-bit) matrix multiply-accumulate |

**Reading**: warp-uniform values now have explicit data paths and an
operation set. `LDSM` improves matrix data-supply; `BMMA` extends the
matrix path to binary precision (continuing the inference-fork from
INT8/INT4).

### Ampere / Ada — async staging hits the ISA

**Appearing**:

| New opcode | What it surfaces |
|---|---|
| `LDGSTS` | Asynchronous global-to-shared memcopy (the SASS form of `cp.async`) |
| `LDGDEPBAR` | Global load dependency barrier |
| `DMMA` | Double-precision matrix multiply-accumulate (Tensor Core FP64) |
| `F2IP` | FP32 down-convert to integer with pack |
| `I2FP` | Integer to FP32 convert with pack |
| `UF2FP` | Uniform FP32 down-convert and pack |

**Reading**: async staging and wait protocols enter the instruction
layer (`LDGSTS` + `LDGDEPBAR`); matrix path adds FP64 (`DMMA`); format
conversion gets routine forms.

### Hopper — cluster, TMA, warp group, bulk async

**Appearing**:

| New opcode | What it surfaces |
|---|---|
| `UTMACMDFLUSH` | TMA command flush |
| `UTMALDG` | Tensor load global → shared (TMA) |
| `UTMASTG` | Tensor store shared → global (TMA) |
| `UTMAREDG` | Tensor store with reduction (TMA) |
| `UTMACCTL` | TMA cache control |
| `UTMAPF` | Tensor prefetch (TMA) |
| `UCGABAR_ARV` | CGA barrier sync (arrive) |
| `UCGABAR_WAIT` | CGA barrier sync (wait) |
| `WARPGROUP` | Warpgroup synchronization |
| `WARPGROUPSET` | Set warpgroup counters |
| `UBLKCP / UBLKPF / UBLKRED` | Bulk data copy / prefetch / reduction |
| `ENDCOLLECTIVE` | Reset MCOLLECTIVE mask |
| `SYNCS` | Sync unit |
| `CGAERRBAR` | CGA error barrier |
| `PREEXIT` | Dependent task launch hint |
| `HGMMA / IGMMA / QGMMA / BGMMA` | Warp-group MMA family (FP16/INT/FP8/binary) |

**Reading**: wait objects, transport objects, and cooperation objects
are no longer confined to a single warp or a single SM. The `*GMMA`
family is the SASS realization of `wgmma.mma_async`. `UTMA*` is the
SASS form of `cp.async.bulk.tensor`. `UCGABAR_*` is cluster barrier.
Each opcode group corresponds to one of the Hopper rewrite's five
interlocking interfaces.

### Blackwell — TMEM, FP4 MMA, single-thread issue

**Appearing**:

| New opcode | What it surfaces |
|---|---|
| `OMMA` | FP4 matrix multiply-accumulate across a warp |
| `QMMA` | FP8 matrix multiply-accumulate across a warp |
| `LDT / LDTM` | Load matrix from Tensor Memory to register file |
| `STT / STTM` | Store matrix from register file to Tensor Memory |
| `UTCCP` | Async copy from shared memory to Tensor Memory |
| `UTCSHIFT` | Shift elements in Tensor Memory |
| `UTCBAR` | Tensor Core barrier |
| `UTCHMMA / UTCOMMA / UTCQMMA` | Uniform forms of MMA |
| `UGETNEXTWORKID` | Uniform get-next-work-id |
| `UVIRTCOUNT` | Virtual resource management |
| Many more `UF*` / `UI*` variants | Continued uniform-path expansion |

**Retiring** (per public SASS history):

| Old opcode | What it was |
|---|---|
| `WARPGROUP` | Hopper warpgroup synchronization (replaced by tcgen05 protocol) |
| `WARPGROUPSET` | Hopper warpgroup counter set |
| Parts of `*GMMA` | Hopper warp-group MMA encoding |

**Reading**: package, fabric, tensor, and uniform directions are
continuing to expand at the ISA layer. The encoding and execution
protocol for matrix tasks **converges** rather than just adding more
opcodes — matrix tasks descend from the surface of warpgroup
instructions into the new interface set of `TMEM + descriptor +
tcgen05`. So writing the Blackwell delta as "WGMMA SASS forms keep
getting strengthened" would be wrong; it's "Hopper-style WGMMA
encoding is replaced by `tcgen05` + TMEM."

## Cross-architecture quick-reference

What a SASS opcode tells you about minimum compute capability:

| Opcode | Min CC |
|---|---|
| `XMAD` | ≤ 6.x (retired in Volta) |
| `HMMA`, `IMMA`, `BSSY`, `BSYNC`, `WARPSYNC`, `DEPBAR` | 7.0 (Volta) |
| `R2UR`, `S2UR`, `UIADD3`, `LDSM`, `BMMA` | 7.5 (Turing) |
| `LDGSTS`, `LDGDEPBAR`, `DMMA` | 8.0 (Ampere) |
| `UTMA*`, `UCGABAR_*`, `WARPGROUP`, `*GMMA` | 9.0 (Hopper) |
| `OMMA`, `QMMA`, `LDT/LDTM`, `STT/STTM`, `UTCCP`, `UTCSHIFT`, `UTC*` | 10.0 / 12.0 (Blackwell) |

## Why a future agent cares

- A trace dump containing `WARPGROUP` is **Hopper** — not Blackwell;
  Blackwell removes those.
- A trace containing `OMMA` or `LDT` is **Blackwell**.
- A trace with `XMAD` is **pre-Volta**. Modern compilers won't emit it.
- The presence of `LDGSTS` without any `UTMA*` is **Ampere/Ada**, not
  Hopper.
- "Uniform-path" `U*` instructions appear in batches — if you see
  many `U*` variants, the floor is Turing+.

## Opcode → minimum-arch reverse lookup (alphabetical)

For "I have an unfamiliar SASS opcode in a trace; which generation
introduced it?" Reverse-lookup table for the opcodes this card cites.
For an opcode not in this table, fall through to
`references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md`.

| Opcode | Min architecture | Family |
|---|---|---|
| `BGMMA` | sm_90 (Hopper) | Warpgroup MMA (binary) |
| `BMMA` | sm_75 (Turing) | Warp-level binary MMA |
| `BSSY` | sm_70 (Volta) | Convergence barrier — set sync point |
| `BSYNC` | sm_70 (Volta) | Convergence barrier — sync threads |
| `CGAERRBAR` | sm_90 (Hopper) | CGA error barrier |
| `DEPBAR` | sm_70 (Volta) | Software-scoreboard dependency barrier |
| `DMMA` | sm_80 (Ampere) | Tensor Core FP64 MMA |
| `ENDCOLLECTIVE` | sm_90 (Hopper) | Reset MCOLLECTIVE mask |
| `F2IP` | sm_80 (Ampere) | FP32 down-convert and pack |
| `FADD32I` | sm_70 (Volta) | FP32 add with 32-bit immediate |
| `FFMA32I` | sm_70 (Volta) | FP32 FMA with 32-bit immediate |
| `HGMMA` | sm_90 (Hopper) | Warpgroup MMA (FP16) |
| `HMMA` | sm_70 (Volta) | Warp-level FP16 MMA — first Tensor Core SASS |
| `I2FP` | sm_80 (Ampere) | Integer to FP32 convert and pack |
| `IADD32I` | sm_70 (Volta) | Integer add with 32-bit immediate |
| `IGMMA` | sm_90 (Hopper) | Warpgroup MMA (integer) |
| `IMMA` | sm_70 (Volta) | Warp-level integer MMA |
| `LDGDEPBAR` | sm_80 (Ampere) | Global load dependency barrier |
| `LDGSTS` | sm_80 (Ampere) | Async global-to-shared memcopy (cp.async SASS) |
| `LDSM` | sm_75 (Turing) | Load matrix from shared memory |
| `LDT / LDTM` | sm_100 / sm_120 (Blackwell) | Load matrix from Tensor Memory |
| `MOV32I` | sm_70 (Volta) | Move 32-bit immediate |
| `OMMA` | sm_100 / sm_120 (Blackwell) | Warp-level FP4 MMA |
| `PREEXIT` | sm_90 (Hopper) | Dependent-task launch hint |
| `QGMMA` | sm_90 (Hopper) | Warpgroup MMA (FP8) |
| `QMMA` | sm_100 / sm_120 (Blackwell) | Warp-level FP8 MMA |
| `R2UR` | sm_75 (Turing) | Move vector → uniform register |
| `S2UR` | sm_75 (Turing) | Move special → uniform register |
| `SSY` | ≤ sm_60 (retired in Volta) | Old-style single-step sync |
| `STT / STTM` | sm_100 / sm_120 (Blackwell) | Store matrix to Tensor Memory |
| `SYNC` | ≤ sm_60 (retired in Volta) | Pre-Volta reconvergence sync |
| `SYNCS` | sm_90 (Hopper) | Sync unit |
| `UBLKCP / UBLKPF / UBLKRED` | sm_90 (Hopper) | Bulk data copy / prefetch / reduction |
| `UCGABAR_ARV / UCGABAR_WAIT` | sm_90 (Hopper) | CGA barrier sync |
| `UF2FP` | sm_80 (Ampere) | Uniform FP32 down-convert and pack |
| `UGETNEXTWORKID` | sm_100 / sm_120 (Blackwell) | Uniform get-next-work-id |
| `UIADD3 / UIMAD / UISETP` | sm_75 (Turing) | Uniform integer ops |
| `ULDC / ULEA / USEL / USHF` | sm_75 (Turing) | Uniform load/LEA/select/shift |
| `UTCBAR` | sm_100 / sm_120 (Blackwell) | Tensor Core barrier |
| `UTCCP` | sm_100 / sm_120 (Blackwell) | Async copy SMEM → Tensor Memory |
| `UTCHMMA / UTCOMMA / UTCQMMA` | sm_100 / sm_120 (Blackwell) | Uniform MMA forms |
| `UTCSHIFT` | sm_100 / sm_120 (Blackwell) | Shift elements in Tensor Memory |
| `UTMACCTL / UTMACMDFLUSH / UTMALDG / UTMAPF / UTMAREDG / UTMASTG` | sm_90 (Hopper) | TMA family |
| `UVIRTCOUNT` | sm_100 / sm_120 (Blackwell) | Virtual resource management |
| `VOTEU` | sm_75 (Turing) | Voting with uniform destination |
| `WARPGROUP / WARPGROUPSET` | sm_90 (Hopper); **removed** in Blackwell | Warpgroup synchronization |
| `WARPSYNC` | sm_70 (Volta) | Synchronize threads in warp |
| `XMAD` | ≤ sm_60 (retired in Volta) | Pre-Volta integer multiply-add |

## See also

- `three-rewrites.md` — SASS deltas as evidence for the rewrites.
- `arch-volta.md`, `arch-turing.md`, `arch-ampere.md`, `arch-hopper.md`,
  `arch-blackwell.md` — generation-specific deltas.
- `evolution-execution-model.md`, `evolution-data-supply.md`,
  `evolution-precision.md`, `evolution-synchronization.md` — the five
  physical lines projected onto SASS.
- `.claude/knowledge/sass/sass-instruction-reference.md` —
  per-instruction syntax / encoding.
- `.claude/knowledge/sass/sass-tooling.md` — tooling
  (cuobjdump / nvdisasm / nvcc -keep).

## Escalate to raw when

- You need exact opcode byte-encoding — `references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md`
  is the per-arch catalog.
- You need a PTX → SASS lowering example for a specific instruction —
  compile a kernel with `nvcc -keep -arch=sm_NN` and inspect the
  `.cubin` via `cuobjdump --dump-sass`.
- You need archived per-instruction docs — NVBit's instruction
  metadata (`tools/nvbit_tracer/` if in this repo) contains a complete
  per-arch opcode catalog.
