# SM86 simulator accuracy-recovery goal

## Objective

Evolve the current RTX 3090 Ti trace-driven simulator from the retained
eight-row failure baseline into a repeated, hardware-validated result whose
critical metric families each meet a 20% aggregate error ceiling. The existing
eight rows are evaluation-only. Their NCU values must not become simulator
features, parameters, per-workload corrections, or calibration targets.

## Repeated measurement contract

- Final hardware ground truth uses seven independent NCU application-replay
  processes per workload, one matched target launch per process.
- Final simulator evidence uses seven independent release `sim-run` processes
  per workload. Predicted JSON values must be byte-identical after removing
  process-wall observations; process wall uses the median and retains every
  sample, minimum, maximum, and relative range.
- Smoke uses at least three NCU and three simulator processes per workload.
- Tables report target launches per process, NCU sample count, simulator sample
  count, and metric coverage separately. A deterministic simulator repeat is
  still a real process/timing sample; it is not mislabeled as a new modeled
  launch.

## Accuracy gates

The candidate can be promoted for user review only when all gates pass on the
evaluation matrix, with raw values and unavailable reasons retained:

1. Terminal modeled-event/runtime-equivalent MAPE: at most 20%, coverage 8/8.
2. Issue-frontier/instruction-throughput-equivalent MAPE: at most 20%, coverage
   8/8.
3. Unified L1 hit-rate MAPE: at most 20% on positive NCU denominators, with
   absolute percentage-point coverage 8/8 and mean pp error at most 10.
4. Unified L2 hit-rate MAPE: at most 20%, coverage 8/8, with mean pp error at
   most 10.
5. Instructions/global-load/global-store/DRAM-read counter MAPE: at most 20%,
   full 32/32 metric-launch coverage.
6. No gate may be passed by clamping, target-derived offsets, changing an
   architecture card from scored NCU values, dropping a failing row, or
   collapsing issue and terminal frontiers into one scalar.

## Calibration and implementation boundary

- Build independent microbenchmarks for cache line/sector allocation, L1-to-L2
  routing, MSHR merges, write policy, dirty eviction, DRAM service, and
  completion. Freeze candidate parameters before reopening the eight-row gate.
- Use architecture documentation, targeted device timers, trace invariants,
  P-chase cards, and unscored microbenchmarks for identification. NCU may
  validate those probes but remains outside simulator inputs.
- Compare implementation structure against primary-source simulators and
  papers, including Accel-Sim/GPGPU-Sim's banked sectored cache and explicit
  MSHR/miss-queue/writeback paths and MGPUSim/Akita's finite writeback cache
  resources. Reuse concepts, not unverified architecture constants.
- Preserve current default behavior behind opt-in candidates until an
  independent calibration plus untouched holdout transfers.

## Work sequence

1. Repair campaign repeat semantics and add runtime/throughput/cache gates.
2. Establish independent cache-accounting probes and decompose current L1/L2
   request, hit, miss, MSHR-merge, fill, write, and DRAM streams.
3. Fix one causally identified cache/accounting mechanism at a time, run its
   independent holdout, then reopen the evaluation gate.
4. Address terminal/runtime error through independently identified memory and
   completion services while retaining typed dirty-writeback/coherence tails
   until each mechanism is actually drained.
5. Produce portable smoke/final evidence and present the passing candidate to
   the user. The persistent goal remains active until manual approval.

## Primary references

- https://github.com/accel-sim/accel-sim-framework
- https://github.com/gpgpu-sim/gpgpu-sim_distribution
- https://github.com/sarchlab/mgpusim
- https://people.ece.ubc.ca/~aamodt/publications/papers/accelsim.isca2020.pdf
- https://arxiv.org/abs/1810.07269
