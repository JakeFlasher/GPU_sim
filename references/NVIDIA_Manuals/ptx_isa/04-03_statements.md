---
title: "4.3. Statements"
section: "4.3"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#statements"
---

## [4.3. Statements](https://docs.nvidia.com/cuda/parallel-thread-execution#statements)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#statements "Permalink to this headline")

A PTX statement is either a directive or an instruction. Statements begin with an optional label and
end with a semicolon.

**Examples**

```text
        .reg     .b32 r1, r2;
        .global  .f32  array[N];

start:  mov.b32   r1, %tid.x;
        shl.b32   r1, r1, 2;          // shift thread id by 2 bits
        ld.global.b32 r2, array[r1];  // thread[tid] gets array[tid]
        add.f32   r2, r2, 0.5;        // add 1/2
```

### [4.3.1. Directive Statements](https://docs.nvidia.com/cuda/parallel-thread-execution#directive-statements)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#directive-statements "Permalink to this headline")

Directive keywords begin with a dot, so no conflict is possible with user-defined identifiers. The
directives in PTX are listed in [Table 1](https://docs.nvidia.com/cuda/parallel-thread-execution/#directive-statements-ptx-directives) and
described in [State Spaces, Types, and Variables](https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces-types-and-variables)
and [Directives](https://docs.nvidia.com/cuda/parallel-thread-execution/#directives).

| `.address_size` | `.explicitcluster` | `.maxnreg` | `.section` |
| --- | --- | --- | --- |
| `.alias` | `.extern` | `.maxntid` | `.shared` |
| `.align` | `.file` | `.minnctapersm` | `.sreg` |
| `.branchtargets` | `.func` | `.noreturn` | `.target` |
| `.callprototype` | `.global` | `.param` | `.tex` |
| `.calltargets` | `.loc` | `.pragma` | `.version` |
| `.common` | `.local` | `.reg` | `.visible` |
| `.const` | `.maxclusterrank` | `.reqnctapercluster` | `.weak` |
| `.entry` | `.maxnctapersm` | `.reqntid` |  |

### [4.3.2. Instruction Statements](https://docs.nvidia.com/cuda/parallel-thread-execution#instruction-statements)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#instruction-statements "Permalink to this headline")

Instructions are formed from an instruction opcode followed by a comma-separated list of zero or
more operands, and terminated with a semicolon. Operands may be register variables, constant
expressions, address expressions, or label names. Instructions have an optional guard predicate
which controls conditional execution. The guard predicate follows the optional label and precedes
the opcode, and is written as `@p`, where `p` is a predicate register. The guard predicate may
be optionally negated, written as `@!p`.

The destination operand is first, followed by source operands.

Instruction keywords are listed in
[Table 2](https://docs.nvidia.com/cuda/parallel-thread-execution/#instruction-statements-reserved-instruction-keywords-new). All instruction keywords are
reserved tokens in PTX.

| `abs` | `cvta` | `membar` | `setp` | `vabsdiff` |
| --- | --- | --- | --- | --- |
| `activemask` | `discard` | `min` | `shf` | `vabsdiff2` |
| `add` | `div` | `mma` | `shfl` | `vabsdiff4` |
| `addc` | `dp2a` | `mov` | `shl` | `vadd` |
| `alloca` | `dp4a` | `movmatrix` | `shr` | `vadd2` |
| `and` | `elect` | `mul` | `sin` | `vadd4` |
| `applypriority` | `ex2` | `mul24` | `slct` | `vavrg2` |
| `atom` | `exit` | `multimem` | `sqrt` | `vavrg4` |
| `bar` | `fence` | `nanosleep` | `st` | `vmad` |
| `barrier` | `fma` | `neg` | `stackrestore` | `vmax` |
| `bfe` | `fns` | `not` | `stacksave` | `vmax2` |
| `bfi` | `getctarank` | `or` | `stmatrix` | `vmax4` |
| `bfind` | `griddepcontrol` | `pmevent` | `sub` | `vmin` |
| `bmsk` | `isspacep` | `popc` | `subc` | `vmin2` |
| `bra` | `istypep` | `prefetch` | `suld` | `vmin4` |
| `brev` | `ld` | `prefetchu` | `suq` | `vote` |
| `brkpt` | `ldmatrix` | `prmt` | `sured` | `vset` |
| `brx` | `ldu` | `rcp` | `sust` | `vset2` |
| `call` | `lg2` | `red` | `szext` | `vset4` |
| `clz` | `lop3` | `redux` | `tanh` | `vshl` |
| `cnot` | `mad` | `rem` | `tcgen05` | `vshr` |
| `copysign` | `mad24` | `ret` | `tensormap` | `vsub` |
| `cos` | `madc` | `rsqrt` | `testp` | `vsub2` |
| `clusterlaunchcontrol` | `mapa` | `sad` | `tex` | `vsub4` |
| `cp` | `match` | `selp` | `tld4` | `wgmma` |
| `createpolicy` | `max` | `set` | `trap` | `wmma` |
| `cvt` | `mbarrier` | `setmaxnreg` | `txq` | `xor` |
