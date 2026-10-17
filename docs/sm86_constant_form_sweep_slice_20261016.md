# SM86 constant-form reuse/warp sweep — 2026-10-16

H50 extends the notes probe with 24 real RTX 3090 Ti points: plain 32-bit
`__ldg` versus uint16 `__ldg`, 1/4/8 resident warps, and reuse distances of
1/4/32/256 lines. Each pair uses identical addresses, iterations, and
functional work. All points pass checksums; SASS remains
`LDG.E.CONSTANT` versus `LDG.E.U16.CONSTANT`.

Clock64 ranges increase with reuse distance and pressure, as expected, but
plain/U16 differences are small and non-monotonic (for example, W1/reuse1
maxima 9,257 vs 9,258 cycles; W4/reuse32 23,089 vs 21,574; W8/reuse256
39,123 vs 39,108). These are mechanism observations, not a stable form-latency
law.

Decision: retain H48 form counters and defer lookup/timing correction. A future
NCU campaign may use this matrix as independent form-specific scope evidence,
but no NCU target is consumed by replay or parameter selection.
