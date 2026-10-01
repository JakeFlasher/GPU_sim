# SM86 finite value-trace SHF probe — 2026-10-01

## Scope

This is a bounded functional-regeneration check for `tools/nvbit_tracer/
sm86_value_trace.py`. It is not a compiler, cubin generator, or timing
validation.

## Independent static probe

CUDA Toolkit `ptxas` 13.4.92 was invoked offline with `-arch=sm_86 -O0` on
PTX containing:

```ptx
shf.l.clamp.b32 r0, a, b, n;
shf.r.clamp.b32 r1, a, b, n;
```

The resulting SM86 disassembly was:

```sass
SHF.L.U32.HI R8, R0, R3, R2;
SHF.R.U32    R0, R0, R3, R2;
```

The emitted operand order is therefore `D,A,B,C`; the final operand is the
shift amount. The interpreter's admitted forms use this same order and map it
to the official PTX funnel-shift semantics:

```text
SHF.L: d = (B << n) | (A >> (32-n))
SHF.R: d = (B << (32-n)) | (A >> n)
n = C & 0x1f
```

For the independent values `A=0x11223344`, `B=0x55667788`, `n=8`, the
expected results are `0x66778811` and `0x88112233`; the finite interpreter
matches both. The official reference is
<https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-shf>.

## Validation boundary

The Python suite now contains deterministic randomized checks over both
directions and wrapped shift amounts, plus malformed-form refusals. These
tests establish implementation evidence for the finite subset only. They do
not establish arbitrary value-dependent SASS regeneration, compiler/ABI
generation, hardware pipe routing, or timing equivalence.
