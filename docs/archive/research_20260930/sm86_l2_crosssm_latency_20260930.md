# SM86 cross-SM L2 latency matrix

**Date:** 2026-09-30  
**GPU:** RTX 3090 Ti / GA102 / SM86, driver 615.71.09, clocks 1350/10501.

A cooperative 84-block kernel (one block per enabled SM under the observed
launch) warms one 128-byte-spaced target line from block 0 using
ld.global.cg.u32, synchronizes the grid, and then measures the same target
from lane zero of every block. The experiment repeats 32 target addresses and
records each block's %smid and clock64 span. The warm load is consumed before
the grid barrier; each measured load is consumed by a dependent store.

The resulting per-SM median latency envelope is **286--361 cycles**, with
per-SM ranges stable across the 32 target addresses. Representative groups
include SM 0 at 286 cycles, SMs 74--79 around 290--292 cycles, SMs 32--41
around 299--313 cycles, and SMs 1--3/10--13/42--55 around 340--361 cycles.
The full matrix and per-SM summary are retained in the result directory.

This is new local evidence for nonuniform L2/NoC/DRAM path latency and gives a
stronger cache-topology prior than a single scalar L2 latency. It does not
identify which cache level supplied each read, a physical owner, a directory
read-from source, compression state, or DRAM completion. ld.global.cg and
the warm/measure intervention constrain the scenario but are not an oracle.

The result is therefore an interval/topology observation for the cache-phase
model, not closure of native cache ownership.

Artifacts: results/alpha/sm86_l2_crosssm_latency_goal_20260930/.

DGNA provides related methodology on A100/H100, but its topology conclusions
are not transferred to GA102 (<https://arxiv.org/abs/2607.19922>).


The corrected run uses inline ld.global.cg rather than the CUDA __ldcg()
intrinsic, whose behavior DGNA warns can introduce misleading extra effects.
The warm load is consumed into a volatile sink before grid synchronization.
For SM IDs other than the warmup block's SM 0, the measurement is therefore a
two-SM warm/read path; SM 0 is retained as a self-access control and excluded
from the remote-topology interpretation.
