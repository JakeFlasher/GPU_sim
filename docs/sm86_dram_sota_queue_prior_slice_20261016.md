# SM86 DRAM SOTA queue prior — 2026-10-16

H57 reviewed retained GPGPU-Sim/Accel-Sim source and local SOTA notes. Generic
modern simulators model per-partition private/shared credits, finite
L2→DRAM/DRAM→L2/L2→interconnect queues, and round-robin subpartition arbitration
(`gpgpu-sim/l2cache.cc`, `l2cache.h`, `dram_sched.cc`). Huerta's Ampere model
adds finite memory-subcore/PRT/coalescer structures.

These mechanisms are source-inspection priors only; no Accel-Sim binary/runtime
or imported constant is used. They motivate the existing SM86 route/pressure
observers, but do
not identify GA102's physical partition hash, queue capacities, row policy, or
completion ownership. H43 pressure residuals are confounded and H47
writeback A/B is mixed; no generic queue law transfers to 3090 Ti.

Decision: retain aggregate carded DRAM timing plus observer-only modulo/XOR
pressure and typed writeback/coherence tails. Do not import generic queue
capacities or activate partitioned timing without a locked SM86 probe isolating
one demand factor.
