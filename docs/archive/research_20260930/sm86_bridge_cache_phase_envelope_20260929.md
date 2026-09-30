# NAK bridge cache-phase transfer envelope

Six independent bridged operation-family profiles provide warm/cold endpoint
ratios:

- duration ratio: **0.9268–1.0000**, median **0.95485**
- GPC ratio: **0.9330–1.0028**, median **0.95105**

Applying the independent median ratios to the mixed body’s cold profile predicts
2.353 us / 3,105 cycles warm. Fresh warm NCU measures 2.336 us / 3,085 cycles,
for **0.717% duration / 0.654% GPC APE**.

The transfer is body-local and cache-phase-specific. It does not establish
cache ownership, read-from source, eviction, or general ABI behavior.
