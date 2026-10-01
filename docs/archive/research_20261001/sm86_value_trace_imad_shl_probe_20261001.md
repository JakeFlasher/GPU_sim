# SM86 finite value-trace IMAD.SHL probe — 2026-10-01

The retained SM86 control-bit corpora contain frequent forms such as
`IMAD.SHL.U32 R24, R0, 0x40, RZ` and `IMAD.SHL.U32 R201, R216, 0x2, RZ`.
An offline CUDA Toolkit 13.4.92 `ptxas -arch=sm_86 -O0` probe compiling
`mul.lo.u32 r0, x, 0x10; add.u32 r0, r0, y;` and a second `0x400` multiply
emitted:

```sass
IMAD.SHL.U32 R3, R0, 0x10, RZ;
IADD3 R4, R3, R2, RZ;
IMAD.SHL.U32 R0, R0, 0x400, RZ;
```

This independently confirms the bounded immediate-multiplier form. Ampere
SASS discussion likewise characterizes the idiom as an integer multiply/add
datapath operation using a power-of-two multiplier, equivalent to
`(A * immediate) + C` modulo 32 bits, rather than a general compiler or
arbitrary SASS-generation contract:

<https://forums.developer.nvidia.com/t/how-to-understand-the-sass-code-on-rtx3080gpu-and-cuda11-8/276295>

The finite interpreter admits only the bounded four-operand
`IMAD.SHL.U32 D,A,immediate,C` form, refuses register/zero/negative
multipliers, and now has direct tests for value regeneration and refusal.
This is implementation evidence for one operation family only; it does not
establish unrestricted IMAD variants, SASS encoding, ABI generation, or
hardware timing.

The same finite lane now admits the separately observed multiplier-1
`IMAD.IADD D,A,0x1,C` address idiom as `A+C` modulo 32 bits. Other
`IMAD.IADD` multipliers remain explicitly refused.
