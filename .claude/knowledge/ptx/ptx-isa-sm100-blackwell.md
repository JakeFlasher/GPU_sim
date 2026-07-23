---
Use when: reasoning about Blackwell-specific PTX (sm_100 / sm_120 / sm_120a); placing `tcgen05.mma` (single-thread issue) and `OMMA`/`QMMA`; understanding Tensor Memory (TMEM) operations `tcgen05.cp`/`tcgen05.shift`/`tcgen05.alloc`/`tcgen05.commit`/`tcgen05.wait`; placing block-scaled MMA (`.kind::mxf4`/`.kind::mxf8f6f4`); CTA-pair semantics; FP4/FP6/NVFP4/MX numerical formats; the SM100 vs SM120 split (datacenter B200 CC 10.0 vs consumer GB202/GB206 CC 12.0).
Primary raw source: references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md (§9.7.16 TensorCore 5th Generation Family Instructions).
Supporting sources: references/NVIDIA_Manuals/blackwell_tuning/01_nvidia-blackwell-tuning-guide.md; references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md; references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md (Blackwell SASS); .claude/knowledge/blackwell/blackwell-tuning.md; .claude/knowledge/blackwell/blackwell-tensor-perf.md; .claude/knowledge/blackwell/rtx_pro_6000_blackwell_capabilities.md; .claude/knowledge/sass/sass-instruction-reference.md (LDT/LDTM/STT/STTM/UTCCP/UTCSHIFT); .claude/knowledge/patents/arch-blackwell.md (TMEM as ISA-visible object).
Repo touchpoints: workloads/simbench_bw/ (current measurement workloads); tools/nvbit_tracer/ (dynamic SASS traces).
Out of scope: Hopper `wgmma.mma_async` (in `ptx-isa-sm90-hopper.md`); pre-Hopper tensor-core `mma.sync` (in `ptx-isa-tensor-cores.md`).
Version family: PTX ISA 9.2; CUDA 13.x; sm_100/sm_120/sm_100a/sm_120a. Mirror snapshot: 2026-04-29.
---

<!-- PTX ISA 9.2 -->

# Blackwell -- sm_100 (Datacenter) & sm_120 (Consumer) Deltas

This card covers the **two PTX-visible Blackwell families**:
- **sm_100 / sm_100a / sm_100f** — datacenter Blackwell (B200, B100). Full tcgen05 / TMEM path.
- **sm_120 / sm_120a / sm_120f** — consumer/workstation Blackwell (RTX PRO 6000 Blackwell / GB202 — the cloud target on this branch; RTX 5090, RTX 50-series). **No tcgen05 / TMEM path**; FP4 / FP6 MMA goes through warp-level `mma.sync.kind::*` (see `ptx-isa-tensor-cores.md`).

The repo's primary cloud target is **sm_120 (GB202, RTX PRO 6000 Blackwell)**. Per the root `CLAUDE.md` (re-founded 2026-07-02), the project builds an event-driven Rust simulator predicting NCU metric vectors; NCU is the first-class ground-truth instrument, captured under pinned protocol v2.1 (`.claude/knowledge/measurement/pinned-protocol-v21.md`). Use sm_100 entries here only when targeting datacenter parts or reading datacenter-Blackwell papers.

## Target-suffix taxonomy

| Suffix | Meaning |
|--------|---------|
| (none) | Virtual arch, JIT-compatible across the family |
| `a` | **Architecture-conditional**; kernel binds to that exact `sm_X` and is *not* forward-compat |
| `f` | **Family-conditional** (PTX 8.8+); kernel runs on any sm_X+ device that exposes the same family-level features |

## sm_100 family target differences

| Target | Features enabled |
|--------|-----------------|
| `sm_100` | Virtual arch, no tcgen05 |
| `sm_100a` | All tcgen05, `.kind::i8`, `.kind::mxf4nvf4`, `.scale_vec::1X/2X/4X`, `scale-input-d` |
| `sm_100f` | Most tcgen05 (not `.kind::i8` alone, not `.scale_vec::NX`), `.block16/.block32`, `setmaxnreg`, introduced PTX 8.8 |

All tcgen05 instructions in a kernel **must** use the same `.cta_group` value.

## sm_120 family target differences (consumer Blackwell)

| Target | Features enabled |
|--------|-----------------|
| `sm_120` | Virtual arch. FP4/FP6/FP8 MMA via warp-level `mma.sync` only (no tcgen05). |
| `sm_120a` | Adds: `mma`/`mma.sp` with `.kind::f8f6f4`, `.kind::mxf4`, `.kind::mxf4nvf4`, `.block_scale`, `.scale_vec_size`; `ldmatrix`/`stmatrix` `.b8` variants (m16n16/m8n16/m16n8) with `.src_fmt`/`.dst_fmt` decompression; `multimem.{ld_reduce,st,red}` on `.e4m3*`/`.e5m2*`; `.acc::f16` qualifier; `clusterlaunchcontrol.try_cancel.multicast::cluster::all`; `tensormap.replace.swizzle_atomicity`. PTX 8.6+. |
| `sm_120f` | Same family features promoted under PTX 8.8+ (sm_120f / sm_121f umbrella). PTX 9.1 also adds `.f16x2` / `.bf16x2 → .e2m1x2` / etc. cvt under sm_120f. |
| `sm_121` / `sm_121a` / `sm_121f` | Sibling consumer-Blackwell die; same feature surface as sm_120 family (release notes treat them together for FP4/FP6 mma and multimem). |

**Critical for the repo (sm_120):** there is **no `tcgen05.*`, no TMEM
(`tmem`), no `wgmma`, no Hopper-style cluster.`block.*` umbrella from
the sm_100 family.** A trace from cloud RTX PRO 6000 Blackwell will not
contain any `tcgen05` SASS (`LDT`/`STT`/`UTC*MMA`); FP4/FP8 MMA appears
as warp-level SASS `OMMA` / `QMMA` instead. Cross-reference
`sass-instruction-reference.md` §"Mnemonic families". (Local RTX 3090 Ti
smoke-test traces are sm_86 Ampere SASS — no Blackwell mnemonics at all.)

## .blocksareclusters Directive

## .blocksareclusters Directive

### Syntax
```ptx
.blocksareclusters
```
### Constraints
- Introduced PTX ISA 9.0.
- Specifies that CUDA thread blocks are mapped to clusters.
- Kernel-level directive.

## Tensor Memory (TMEM)

- 512 columns x 128 lanes (rows) per CTA, each cell 32 bits.
- Address: bits[31:16] = lane, bits[15:0] = column.
- Allocation unit: 32 columns, power of 2, range [32, 512].
- Divided into 4 chunks: warp N in warpgroup accesses lanes `[32*N, 32*N+31]`.

## tcgen05.alloc / dealloc / relinquish_alloc_permit

### Syntax
```ptx
tcgen05.alloc.cta_group.sync.aligned{.shared::cta}.b32 [dst], nCols;
tcgen05.dealloc.cta_group.sync.aligned.b32               taddr, nCols;
tcgen05.relinquish_alloc_permit.cta_group.sync.aligned;
.cta_group = { .cta_group::1, .cta_group::2 }
```
### Constraints
- `nCols` in [32, 512], power of 2. Warp-level collective. Must dealloc before kernel exit.
- `.cta_group::2`: one warp from each peer CTA collectively; may block.

## tcgen05.mma

### Syntax
```ptx
// Dense, no block scaling:
tcgen05.mma.cta_group.kind [d-tmem], a-desc, b-desc, idesc,
    {disable-output-lane}, enable-input-d {, scale-input-d};
tcgen05.mma.cta_group.kind [d-tmem], [a-tmem], b-desc, idesc,
    {disable-output-lane}, enable-input-d {, scale-input-d};

// With block scaling (mx kinds):
tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
    [d-tmem], a-desc, b-desc, idesc,
    [scale-A-tmem], [scale-B-tmem], enable-input-d;

.kind     = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8,
              .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
.cta_group = { .cta_group::1, .cta_group::2 }
```
### Variants
- `tcgen05.mma.sp` -- sparse A matrix (adds `[sp-meta-tmem]` operand).
- `tcgen05.mma.ws` -- weight stationary (only `.cta_group::1`).
- `tcgen05.mma.ws.sp` -- weight stationary + sparse A.
- `.collector::a::{fill,use,lastuse,discard}` (activation stationary, A buffer).
- `.collector::bN::{fill,use,lastuse,discard}` (weight stationary, N=0-3).
- `.ashift` -- shifts A rows down by 1 in TMEM (M=128 or 256 only).
- `scale-input-d` -- `D = A*B + D * 2^(-scale)`, scale in [0,15], `.kind::f16`/`.kind::tf32` only (`sm_100a`).

### Shape/Type Summary (cta_group::1, dense, no .ws)

| `.kind` | dtype | atype/btype | M | N | K |
|---------|-------|-------------|---|---|---|
| `f16` | f16/f32 | f16, bf16 | 64, 128 | 8..256 step 8 | 16 |
| `tf32` | f32 | tf32 | 64, 128 | 8..256 step 8 | 8 |
| `f8f6f4` | f16/f32 | e4m3,e5m2,e2m3,e3m2,e2m1 | 64, 128 | 8..256 step 8 | 32 |
| `i8` | s32 | s8, u8 | 64, 128 | 8,16,24,32,48..256 step 16 | 32 |
| `mxf8f6f4` | f32 | above x ue8m0 | 128 | 8..256 step 8 | 32 |
| `mxf4` | f32 | e2m1 x ue8m0 | 128 | 8..256 step 8 | 64 |
| `mxf4nvf4` | f32 | e2m1 x ue8m0/ue4m3 | 128 | 8..256 step 8 | 64 |

**cta_group::2**: M doubles (128/256), N steps become 16.
**ws shapes** (cta_group::1 only): M={32,64,128}, N={64,128,256}.

### Instruction Descriptor (idesc, 32-bit register)

| Bits | Field | Encoding |
|------|-------|----------|
| 0-1 | Sparsity selector | 0-3 |
| 2 | Sparse flag | 0=dense, 1=sparse |
| 3 | Saturate (i8 only) | 0/1 |
| 4-5 | dtype | f16=0, f32=1, s32=2 |
| 7-9 | atype | kind-dependent |
| 10-12 | btype | kind-dependent |
| 13 | Negate A | 0/1 |
| 14 | Negate B | 0/1 |
| 15 | Transpose A | 0/1 |
| 16 | Transpose B | 0/1 |
| 17-22 | N >> 3 | |
| 24-28 | M >> 4 | |
| 30-31 | Max shift (.ws B-reuse) | 0=none, 1=8, 2=16, 3=32 |

### Block Scaling (.scale_vectorsize)

| Qualifier | Alias for | Applies to |
|-----------|-----------|------------|
| `.scale_vec::1X` | `.block32` (mxf8f6f4) | `sm_100a` |
| `.scale_vec::2X` | `.block32` (mxf4, mxf4nvf4) | `sm_100a` |
| `.scale_vec::4X` | `.block16` (mxf4nvf4) | `sm_100a` |
| `.block16` | -- | `sm_100f`, `sm_110f` |
| `.block32` | -- | `sm_100f`, `sm_110f` |

### Sparse Matrices

| `.kind` | Sparsity pattern |
|---------|-----------------|
| `tf32` | 1:2 |
| `f16/f8f6f4/mxf8f6f4/i8` | 2:4 |
| `mxf4/mxf4nvf4` | 4:8 pairwise structured |

### Example
```ptx
tcgen05.mma.cta_group::1.kind::tf32 [taddr0], adesc, bdesc, idesc, {m0,m1,m2,m3}, p;
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale
    [taddr2], [taddr1], bdesc, idesc, [sf_a], [sf_b], p;
tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use
    [taddr2], [taddr1], bdesc, idesc, p;
```

## tcgen05.cp -- Shared Memory to TMEM

### Syntax
```ptx
tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc;
.shape     = { .128x256b, .4x256b, .128x128b, .64x128b, .32x128b }
.multicast = { .warpx2::02_13, .warpx2::01_23, .warpx4 }
.src_fmt   = { .b6x16_p32, .b4x16_p64 }
.dst_fmt   = { .b8x16 }
```
### Constraints
- `.64x128b` requires `.warpx2::02_13` or `.warpx2::01_23`.
- `.32x128b` requires `.warpx4`.
- Decompression: 4-bit->8-bit (`.b4x16_p64`->`.b8x16`), 6-bit->8-bit (`.b6x16_p32`->`.b8x16`).

### Example
```ptx
tcgen05.cp.cta_group::1.128x256b [taddr], sdesc;
tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [taddr], sdesc;
```

## tcgen05.ld / tcgen05.st

### Syntax
```ptx
tcgen05.ld.sync.aligned.shape.num{.pack::16b}.b32   r, [taddr];
tcgen05.st.sync.aligned.shape.num{.unpack::16b}.b32  [taddr], r;
.shape = { .16x64b, .16x128b, .16x256b, .32x32b, .16x32bx2 }
.num   = { .x1, .x2, .x4, .x8, .x16, .x32, .x64, .x128 }
```
### Variants
- `tcgen05.ld.red` -- load with `.min`/`.max` reduction (`.32x32b` or `.16x32bx2`, `.x2` minimum).
- `.16x32bx2` takes additional `immHalfSplitoff` immediate operand.

### Register count per .num

| .num | .32x32b/.16x64b/.16x32bx2 | .16x128b | .16x256b |
|------|---------------------------|----------|----------|
| .x1 | 1 | 2 | 4 |
| .x2 | 2 | 4 | 8 |
| .x4 | 4 | 8 | 16 |
| .x8 | 8 | 16 | 32 |
| .x16 | 16 | 32 | 64 |
| .x32 | 32 | 64 | 128 |
| .x64 | 64 | 128 | N/A |
| .x128 | 128 | N/A | N/A |

## tcgen05.shift

### Syntax
```ptx
tcgen05.shift.cta_group.down [taddr];
.cta_group = { .cta_group::1, .cta_group::2 }
```
### Constraints
- Shifts 32-byte elements down by one row (all rows except last). Lane of `taddr` must be aligned to 32.

## tcgen05.fence

### Syntax
```ptx
tcgen05.fence::before_thread_sync ;
tcgen05.fence::after_thread_sync  ;
```
### Constraints
- `before_thread_sync`: orders prior async tcgen05 ops before subsequent sync/execution ops.
- `after_thread_sync`: orders subsequent async tcgen05 ops after prior sync/execution ops.

## tcgen05.commit

### Syntax
```ptx
tcgen05.commit.cta_group.mbarrier::arrive::one{.shared::cluster}{.multicast::cluster}.b64
    [mbar] {, ctaMask};
.cta_group = { .cta_group::1, .cta_group::2 }
```
### Constraints
- Tracks completion of prior async tcgen05 ops (mma/cp/shift) from current thread.
- Triggers arrive-on with count=1 at cluster scope. Optional `.multicast::cluster` with 16-bit `ctaMask`.

## tcgen05.wait

### Syntax
```ptx
tcgen05.wait::ld.sync.aligned;
tcgen05.wait::st.sync.aligned;
```
### Constraints
- Blocks until all prior `tcgen05.ld` (or `.st`) from executing thread have completed.

## 2CTA / CTA Pair Mode

- **CTA pair**: two CTAs in a cluster whose `%cluster_ctarank` differs only in bit 0.
- `.cta_group::2`: tcgen05 ops access TMEM of both CTAs in the pair.
- `.cta_group::1`: operate on current CTA's TMEM only.

### Issue Granularity

| Operation | cta_group::1 | cta_group::2 |
|-----------|-------------|-------------|
| mma, cp, shift, commit | 1 thread | 1 thread from CTA pair |
| alloc, dealloc, relinquish | 1 warp | 1 warp from each peer CTA (blocking) |
| ld, st, wait | 1 warp (N/A) | N/A |
| fence | 1 thread (N/A) | N/A |

### Example (dealloc with 2CTA)
```ptx
// Both CTA0 and CTA1 warps must participate:
barrier.cluster.arrive;
barrier.cluster.wait;
tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, 32;
exit;
```

## Shared Memory Descriptor (64-bit)

| Bits | Field |
|------|-------|
| 0-13 | Matrix start addr `(addr & 0x3FFFF) >> 4` |
| 16-29 | Leading dim byte offset/addr (encoded same way) |
| 32-45 | Stride dim byte offset |
| 46-48 | Fixed `0b001` |
| 49-51 | Matrix base offset |
| 52 | Leading dim mode: 0=relative, 1=absolute |
| 61-63 | Swizzle: 0=none, 1=128B+32B atom, 2=128B, 4=64B, 6=32B |

## Pipelined Instruction Pairs

| Producer -> Consumer | Same cta_group, additional constraints |
|---------------------|-----------------------------------------|
| `mma -> mma` | Same accumulator and shape |
| `cp -> mma` | Same cta_group |
| `shift -> mma` | Same cta_group |
| `mma -> shift` | Same cta_group |
| `shift -> cp.4x256b` | Same cta_group |
| `mma/cp/shift -> commit` | Implicit pipeline |
| `ld -> wait::ld` | Implicit pipeline |
| `st -> wait::st` | Implicit pipeline |
