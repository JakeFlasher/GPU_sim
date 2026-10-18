# SM86 constant-form independent NCU probe — 2026-10-16

H54 profiled the rebuilt notes probe with NCU independently (no replay or
parameter fitting). Plain 32-bit `__ldg` and uint16 `__ldg` used identical work
and produced identical operation domains at two points:

| point | global-load inst/requests | sectors | lookup hits/misses |
|---|---:|---:|---:|
| 1 warp, 64 iterations, reuse 1 | 64/64 | 256 | 252/4 |
| 8 warps, 64 iterations, reuse 256 | 512/512 | 2,048 | 1,764/284 |

Both variants retain distinct SASS (`LDG.E.CONSTANT` vs
`LDG.E.U16.CONSTANT`) but no independent lookup-domain differential appears in
these two controls. This is mechanism evidence only: it does not prove all
forms/general workloads equivalent and is not consumed by the simulator.

Decision: retain H48 form observers and defer form-specific lookup/timing
correction pending a broader controlled sweep including LTC128B.
