---
title: "15.6. Warp-level Operations"
section: "15.6"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#warp-level-operations"
---

## [15.6. Warp-level Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#warp-level-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#warp-level-operations "Permalink to this headline")

### [15.6.1. Barrier Synchronization](https://docs.nvidia.com/cuda/nvvm-ir-spec#barrier-synchronization)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#barrier-synchronization "Permalink to this headline")

The following intrinsic performs a barrier synchronization among a subset of threads in a warp.

```text
declare void @llvm.nvvm.bar.warp.sync(i32 %membermask)
```

This intrinsic causes executing thread to wait until all threads corresponding to `%membermask` have executed the same intrinsic with the same `%membermask` value before resuming execution.

The argument `%membership` is a 32bit mask, with each bit corresponding to a lane in the warp. 1 means the thread is in the subset.

The behavior of this intrinsic is undefined if the executing thread is not in the `%membermask`.

### [15.6.2. Data Movement](https://docs.nvidia.com/cuda/nvvm-ir-spec#data-movement)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#data-movement "Permalink to this headline")

The following intrinsics synchronize a subset of threads in a warp and then perform data movement among these threads.
Note that the old intrinsic with the `%mode` parameter is deprecated on the modern NVVM IR dialect.

```text
; New intrinsics for the modern dialect
declare {i32, i1} @llvm.nvvm.shfl.sync.idx.i32p(i32 %membermask, i32 %a, i32 %b, i32 %c)
declare {i32, i1} @llvm.nvvm.shfl.sync.up.i32p(i32 %membermask, i32 %a, i32 %b, i32 %c)
declare {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 %membermask, i32 %a, i32 %b, i32 %c)
declare {i32, i1} @llvm.nvvm.shfl.sync.bfly.i32p(i32 %membermask, i32 %a, i32 %b, i32 %c)

; Old intrinsic for the LLVM 7 dialect
declare {i32, i1} @llvm.nvvm.shfl.sync.i32(i32 %membermask, i32 %mode, i32 %a, i32 %b, i32 %c)
```

This intrinsic causes executing thread to wait until all threads corresponding to `%membermask` have executed the same intrinsic with the same `%membermask` value before reading data from other threads in the same warp.

The argument `%membership` is a 32bit mask, with each bit corresponding to a lane in the warp. 1 means the thread is in the subset.

Each thread in the currently executing warp will compute a source lane index j based on input arguments `%b`, `%c`, and the shuffle mode (implicit for new intrinsics, explicit via %mode for the old intrinsic). If the computed source lane index j is in range, the returned `i32` value will be the value of `%a` from lane j; otherwise, it will be the the value of `%a` from the current thread. If the thread corresponding to lane j is inactive, then the returned `i32` value is undefined. The returned `i1` value is set to 1 if the source lane j is in range, and otherwise set to 0.

In the LLVM 7 dialect intrinsic, the argument `%mode` must be a constant and its encoding is specified in the following table.

| Encoding | Meaning | Corresponding Suffix |
| --- | --- | --- |
| 0 | IDX | .idx |
| 1 | UP | .up |
| 2 | DOWN | .down |
| 3 | BFLY | .bfly |

Argument `%b` specifies a source lane or source lane offset, depending on `%mode`.

Argument `%c` contains two packed values specifying a mask for logically splitting warps into sub-segments and an upper bound for clamping the source lane index.

The following pseudo code illustrates the semantics of this intrinsic. The `%mode` in the switch would be determined by the intrinsic suffix in the modern dialect intrinsics
and by the `%mode` parameter in the LLVM 7 dialect intrinsic.

```text
wait until all threads in %membermask have arrived;

%lane[4:0] = current_lane_id; // position of thread in warp
%bval[4:0] = %b[4:0]; // source lane or lane offset (0..31)
%cval[4:0] = %c[4:0]; // clamp value
%mask[4:0] = %c[12:8];

%maxLane = (%lane[4:0] & %mask[4:0]) | (%cval[4:0] & ~%mask[4:0]);
%minLane = (%lane[4:0] & %mask[4:0]);
switch (%mode) {
case UP: %j = %lane - %bval; %pval = (%j >= %maxLane); break;
case DOWN: %j = %lane + %bval; %pval = (%j <= %maxLane); break;
case BFLY: %j = %lane ^ %bval; %pval = (%j <= %maxLane); break;
case IDX: %j = %minLane | (%bval[4:0] & ~%mask[4:0]); %pval = (%j <= %maxLane); break;
}
if (!%pval) %j = %lane; // copy from own lane
if (thread at lane %j is active)
   %d = %a from lane %j
else
   %d = undef
return {%d, %pval}
```

Note that the return values are undefined if the thread at the source lane is not in `%membermask`.

The behavior of this intrinsic is undefined if the executing thread is not in the `%membermask`.

### [15.6.3. Vote](https://docs.nvidia.com/cuda/nvvm-ir-spec#vote)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#vote "Permalink to this headline")

The following intrinsic synchronizes a subset of threads in a warp and then performs a reduce-and-broadcast of a predicate over all threads in the subset.

```text
declare {i32, i1} @llvm.nvvm.vote.sync(i32 %membermask, i32 %mode, i1 %predicate)
```

This intrinsic causes executing thread to wait until all threads corresponding to `%membermask` have executed the same intrinsic with the same `%membermask` value before performing a reduce-and-broadcast of a predicate over all threads in the subset.

The argument `%membermask` is a 32-bit mask, with each bit corresponding to a lane in the warp. 1 means the thread is in the subset.

`@llvm.nvvm.vote.sync()` performs a reduction of the source `%predicate` across all threads in `%membermask` after the synchronization. The return value is the same across all threads in the `%membermask`. The element in the returned aggregate that holds the return value depends on `%mode`.

The argument `%mode` must be a constant and its encoding is specified in the following table.

| Encoding | Meaning | return value |
| --- | --- | --- |
| 0 | ALL | `i1:`1 if the source predicates is 1 for all thread in `%membermask`, 0 otherwise |
| 1 | ANY | `i1:`1 if the source predicate is 1 for any thread in `%membermask`, 0 otherwise |
| 2 | EQ | `i1:`1 if the source predicates are the same for all thread in `%membermask`, 0 otherwise |
| 3 | BALLOT | `i32:`ballot data, containing the `%predicate` value from each thread in `%membermask` |

For the `BALLOT` mode, the `i32` value represents the ballot data, which contains the `%predicate` value from each thread in `%membermask` in the bit position corresponding to the thread’s land id. The bit value corresponding to a thread not in `%membermask` is 0.

Note that the return values are undefined if the thread at the source lane is not in `%membermask`.

The behavior of this intrinsic is undefined if the executing thread is not in the `%membermask`.

### [15.6.4. Match](https://docs.nvidia.com/cuda/nvvm-ir-spec#match)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#match "Permalink to this headline")

The following intrinsics synchronize a subset of threads in a warp and then broadcast and compare a value across threads in the subset.

```text
declare i32 @llvm.nvvm.match.any.sync.i32(i32 %membermask, i32 %value)
declare i32 @llvm.nvvm.match.any.sync.i64(i32 %membermask, i64 %value)
declare {i32, i1} @llvm.nvvm.match.all.sync.i32(i32 %membermask, i32 %value)
declare {i32, i1} @llvm.nvvm.match.all.sync.i64(i32 %membermask, i64 %value)
```

These intrinsics cause executing thread to wait until all threads corresponding to `%membermask` have executed the same intrinsic with the same `%membermask` value before performing broadcast and compare of operand `%value` across all threads in the subset.

The argument `%membership` is a 32bit mask, with each bit corresponding to a lane in the warp. 1 means the thread is in the subset.

The `i32` return value is a 32-bit mask where bit position in mask corresponds to thread’s laneid.

In the `any` version, the `i32` return value is set to the mask of active threads in `%membermask` that have same value as operand `%value`.

In the `all` version, if all active threads in `%membermask` have same value as operand `%value`, the `i32` return value is set to `%membermask`, and the `i1` value is set to 1. Otherwise, the `i32` return value is set to 0 and the `i1` return value is also set to 0.

The behavior of this intrinsic is undefined if the executing thread is not in the `%membermask`.

### [15.6.5. Matrix Operation](https://docs.nvidia.com/cuda/nvvm-ir-spec#matrix-operation)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#matrix-operation "Permalink to this headline")

THIS IS PREVIEW FEATURE. SUPPORT MAY BE REMOVED IN FUTURE RELEASES.

NVVM provides warp-level intrinsics for matrix multiply operations. The core operation is a matrix multiply and accumulate of the form:

```text
D = A*B + C, or
C = A*B + C
```

where `A` is an `MxK` matrix, `B` is a `KxN` matrix, while `C` and `D` are `MxN` matrices. `C` and `D` are also called accumulators. The element type of the `A` and `B` matrices is 16-bit floating point. The element type of the accumulators can be either 32-bit floating point or 16-bit floating point.

All threads in a warp will collectively hold the contents of each matrix `A`, `B`, `C` and `D`. Each thread will hold only a fragment of matrix `A`, a fragment of matrix `B`, a fragment of matrix `C`, and a fragment of the result matrix `D`. How the elements of a matrix are distributed among the fragments is opaque to the user and is different for matrix `A`, `B` and the accumulator.

A fragment is represented by a sequence of element values. For fp32 matrices, the element type is `float`. For fp16 matrices, the element type is `i32` (each `i32` value holds two fp16 values). The number of elements varies with the shape of the matrix.

#### [15.6.5.1. Load Fragments](https://docs.nvidia.com/cuda/nvvm-ir-spec#load-fragments)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#load-fragments "Permalink to this headline")

The following intrinsics synchronize all threads in a warp and then load a fragment of a matrix for each thread.

```text
; load fragment A
declare {i32, i32, i32, i32, i32, i32, i32, i32} @llvm.nvvm.hmma.m16n16k16.ld.a.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);
declare {i32, i32, i32, i32, i32, i32, i32, i32} @llvm.nvvm.hmma.m32n8k16.ld.a.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);
declare {i32, i32, i32, i32, i32, i32, i32, i32} @llvm.nvvm.hmma.m8n32k16.ld.a.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);

; load fragment B
declare {i32, i32, i32, i32, i32, i32, i32, i32} @llvm.nvvm.hmma.m16n16k16.ld.b.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);
declare {i32, i32, i32, i32, i32, i32, i32, i32} @llvm.nvvm.hmma.m32n8k16.ld.b.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);
declare {i32, i32, i32, i32, i32, i32, i32, i32} @llvm.nvvm.hmma.m8n32k16.ld.b.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);

; load fragment C
declare {float, float, float, float, float, float, float, float} @llvm.nvvm.hmma.m16n16k16.ld.c.f32.p<n>f32(float addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);
declare {float, float, float, float, float, float, float, float} @llvm.nvvm.hmma.m32n8k16.ld.c.f32.p<n>f32(float addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);
declare {float, float, float, float, float, float, float, float} @llvm.nvvm.hmma.m8n32k16.ld.c.f32.p<n>f32(float addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);

; load fragment C
declare {i32, i32, i32, i32} @llvm.nvvm.hmma.m16n16k16.ld.c.f16.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);
declare {i32, i32, i32, i32} @llvm.nvvm.hmma.m32n8k16.ld.c.f16.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);
declare {i32, i32, i32, i32} @llvm.nvvm.hmma.m8n32k16.ld.c.f16.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol);
```

These intrinsics load and return a matrix fragment from memory at location `%ptr`. The matrix in memory must be in a canonical matrix layout with leading dimension `%ldm`. `%rowcol` specifies which the matrix in memory is row-major (0) or column-major (1). `%rowcol` must be a constant value.

The returned sequence of values represent the fragment held by the calling thread. How the elements of a matrix are distributed among the fragments is opaque to the user and is different for matrix `A`, `B` and the accumulator. Therefore, three variants (i.e. `ld.a`, `ld.b`, and `ld.c`) are provided.

These intrinsics are overloaded based on the address spaces. The address space number `<n>` must be either 0 (generic), 1 (global) or 3 (shared).

The behavior of this intrinsic is undefined if any thread in the warp has exited.

#### [15.6.5.2. Store Fragments](https://docs.nvidia.com/cuda/nvvm-ir-spec#store-fragments)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#store-fragments "Permalink to this headline")

The following intrinsics synchronize all threads in a warp and then store a fragment of a matrix for each thread.

```text
; The last 8 arguments are the elements of the C fragment
declare void @llvm.nvvm.hmma.m16n16k16.st.c.f32.p<n>float(float addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol, float, float, float, float, float, float, float, float);
declare void @llvm.nvvm.hmma.m32n8k16.st.c.f32.p<n>float(float addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol, float, float, float, float, float, float, float, float);
declare void @llvm.nvvm.hmma.m8n32k16.st.c.f32.p<n>float(float addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol, float, float, float, float, float, float, float, float);

; The last 4 arguments are the elements of the C fragment
declare void @llvm.nvvm.hmma.m16n16k16.st.c.f16.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol, i32, i32, i32, i32);
declare void @llvm.nvvm.hmma.m32n8k16.st.c.f16.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol, i32, i32, i32, i32);
declare void @llvm.nvvm.hmma.m8n32k16.st.c.f16.p<n>i32(i32 addrspace(<n>)* %ptr, i32 %ldm, i32 %rowcol, i32, i32, i32, i32);
```

These intrinsics store an accumulator fragment to memory at location `%ptr`. The matrix in memory must be in a canonical matrix layout with leading dimension `%ldm`. `%rowcol` specifies which the matrix in memory is row-major (0) or column-major (1). `%rowcol` must be a constant value.

These intrinsics are overloaded based on the address spaces. The address space number `<n>` must be either 0 (generic), 1 (global) or 3 (shared).

The behavior of this intrinsic is undefined if any thread in the warp has exited.

#### [15.6.5.3. Matrix Multiply-and-Accumulate](https://docs.nvidia.com/cuda/nvvm-ir-spec#matrix-multiply-and-accumulate)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#matrix-multiply-and-accumulate "Permalink to this headline")

The following intrinsics synchronize all threads in a warp and then perform a matrix multiply-and-accumulate operation.

```text
declare {i32, i32, i32, i32} @llvm.nvvm.hmma.m16n16k16.mma.f16.f16(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, i32 %c0, i32 %c1, i32 %c2, i32 %c3);
declare {i32, i32, i32, i32} @llvm.nvvm.hmma.m32n8k16.mma.f16.f16(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, i32 %c0, i32 %c1, i32 %c2, i32 %c3);
declare {i32, i32, i32, i32} @llvm.nvvm.hmma.m8n32k16.mma.f16.f16(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, i32 %c0, i32 %c1, i32 %c2, i32 %c3);

declare {float, float, float, float, float, float, float, float} @llvm.nvvm.hmma.m16n16k16.mma.f32.f16(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, i32 %c0, i32 %c1, i32 %c2, i32 %c3);
declare {float, float, float, float, float, float, float, float} @llvm.nvvm.hmma.m32n8k16.mma.f32.f16(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, i32 %c0, i32 %c1, i32 %c2, i32 %c3);
declare {float, float, float, float, float, float, float, float} @llvm.nvvm.hmma.m8n32k16.mma.f32.f16(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, i32 %c0, i32 %c1, i32 %c2, i32 %c3);

declare {float, float, float, float, float, float, float, float} @llvm.nvvm.hmma.m16n16k16.mma.f32.f32(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, float %c0, float %c1, float %c2, float %c3, float %c4, float %c5, float %c6, float %c7);
declare {float, float, float, float, float, float, float, float} @llvm.nvvm.hmma.m32n8k16.mma.f32.f32(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, float %c0, float %c1, float %c2, float %c3, float %c4, float %c5, float %c6, float %c7);
declare {float, float, float, float, float, float, float, float} @llvm.nvvm.hmma.m8n32k16.mma.f32.f32(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, float %c0, float %c1, float %c2, float %c3, float %c4, float %c5, float %c6, float %c7);

declare {i32, i32, i32, i32} @llvm.nvvm.hmma.m16n16k16.mma.f16.f32(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, float %c0, float %c1, float %c2, float %c3, float %c4, float %c5, float %c6, float %c7);
declare {i32, i32, i32, i32} @llvm.nvvm.hmma.m32n8k16.mma.f16.f32(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, float %c0, float %c1, float %c2, float %c3, float %c4, float %c5, float %c6, float %c7);
declare {i32, i32, i32, i32} @llvm.nvvm.hmma.m8n32k16.mma.f16.f32(i32 %rowcol, i32 %satf, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %b0, i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6, i32 %b7, float %c0, float %c1, float %c2, float %c3, float %c4, float %c5, float %c6, float %c7);
```

These intrinsics perform a matrix multiply-and-accumulate operation. `%rowcol` specifies the layout of `A` and `B` fragments. It must be a constant value, which can have the following values and semantics.

| Encoding | Meaning |
| --- | --- |
| 0 | A fragment is row-major, B fragment is row-major |
| 1 | A fragment is row-major, B fragment is column-major |
| 2 | A fragment is column-major, B fragment is row-major |
| 3 | A fragment is column-major, B fragment is column-major |

Support for `%satf` has been removed and this operand must be a constant zero.

The behavior of these intrinsics are undefined if any thread in the warp has exited.
