# SM86 GPC observation-envelope residual — 2026-10-16

H59 audits the measured SM86 GPC observation envelope (`base 1776 + 8·floor(log2(grid))`)
against 107 exact independent NCU joins. Mean GPC APE is 12.90% and is
non-monotonic by grid log₂: log₂ 0/2/3/8/10 groups show 12.11%/9.56%/17.51%/
21.76%/29.19%, with small high-grid samples. Active-SM partitions are likewise
non-monotonic.

The envelope was measured on independent unscored launch controls and remains
separate from the causal replay frontier. No grid term, active-SM multiplier,
or NCU-derived correction is promoted; high-grid residuals require broader
independent envelope controls.
