# SM86 atomic conflict-service prior — 2026-10-15

## Scope

H13 evaluates whether a timing-active atomic conflict arm can be admitted.
Independent local RTX 3090 Ti microbenchmarks (warm L2 `atomicAdd`, 100k
iterations, stable repeats) show same-word conflict overhead of about 10–18%
versus distinct addresses across 1–4 warps, with distinct sectors/lines also
having separate service behavior. These are mechanism priors, not NCU fits and
not direct DRAM measurements.

The simulator therefore keeps H12's exact conflict observer and does **not**
apply a per-alias multiplier: traced request construction loses the per-lane
conflict multiplicity before the shared system, and atomic/reduction queue
context differs from the microbench.

## Evidence and residuals

The complete H12 full corpus (default 825 GB/s scenario) remains the relevant
holdout: 356 roots, 286 `ok`, 62 controls, 8 failures, 12,392 launches, and
35.578 s. H12 observed 58 conflict-bearing atomic instructions, 622 distinct
instructions, and 198 alias sectors. Exact DRAM/L1/L2/IPC residuals remain
unchanged (DRAM 40.91%, active IPC 18.19%, elapsed IPC 12.42%).

A timing-active atomic service arm is **not promoted** until a locked local
probe supplies per-transaction semantics and the trace carries a lossless
conflict identity through the system request.
