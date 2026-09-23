# Vortex regression timing differential — 2026-09-23

The Vortex regression `vecadd` was run three times through the native SimX and
Verilator RTL backends using the same compiled `kernel.vxbin` and configuration.
All runs were deterministic:

| Backend | Instructions | Cycles | IPC | Functional result |
|---|---:|---:|---:|---|
| SimX | 112 | 455 | 0.246 | PASS |
| rtlsim | 112 | 397 | 0.282 | PASS |

The 14.6% cycle difference is a useful model-differential finding, not a bug
claim and not a silicon-accuracy result. It identifies a concrete future
parity investigation: SimX and RTL agree functionally but do not yet expose
the same timing. The machine-readable record is
`vortex_vecadd_regression_timing_20260923.json`.
