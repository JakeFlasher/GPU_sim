# SM86 L2 queue-frontier timing arm — negative decision H64 — 2026-10-16

## Independent probe

The notes workspace probe
`/home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/l2_queue_frontier/`
was rebuilt/ran on the independently identified host: GeForce RTX 3090 Ti,
UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6,
driver 615.71.09, device 0. The probe keeps one `.cg` SASS family and varies
resident warps, burst depth, line stride, and reuse; it has no scored-workload
or simulator/NCU fitting input.

The broad matrix contains 288 points (`unique`/`reuse`, 1/2/4/8 warps,
1/4/8/16/32/64 loads, strides 1/8/32, reuse 1/4/32 where applicable). All
288 functional/timestamp/checksum checks passed. Raw points:
`.../l2_queue_frontier/results/h64_clock_sweep_20261016_v1/points.jsonl`;
repeat summary:
`.../results/h64_clock_repeat_20261016_v1.jsonl` and
`.../results/h64_clock_sweep_20261016_v1/analysis_summary.json`.

Twenty-repeat clock64 samples show a robust stride differential at 8 warps and
64 independent loads: median maximum cycles are 20,765 (stride 1), 28,137
(stride 8), and 32,334 (stride 32). The corresponding population standard
deviations are 40, 495, and 506 cycles. A reuse/stride-8 point is 30,589 cycles
median; a 4-warp/16-load/stride-32 point is 5,317 cycles median.
In unique mode, those stride settings also change the allocation footprint from
roughly 5.5 MiB to 44 MiB and 176 MiB; the kernel issues independent `.cg`
loads, so the clock interval is not an isolated L2 latency measurement.

## NCU isolation attempt

Application-replay NCU observations were collected for six selected points with
identical launch identity and the same rebuilt binary. The NCU reports are under
`.../results/h64_ncu_20261016_v1/`; imported CSV summaries are retained. At the
8-warp/64-load points, `sm__cycles_active.max` is 42,981 cycles (stride 1) and
44,588 cycles (stride 32), while `lts__cycles_active.sum` changes from
1,803,541 to 1,875,139 (~4%), while request-XBAR activity changes only ~0.16%.
`lts__xbar2lts_cycles_active.sum` and `lts__lts2xbar_cycles_active.sum` remain
about 49.3k and 176.0k. Both
`l1tex__m_l1tex2xbar_req_cycles_stalled.sum` and
`l1tex__texin_sm2tex_req_cycles_stalled.sum` are zero for these selected points.
NCU values remain independent observations, never simulator inputs.

## Decision

Do **not** add a finite L2 queue capacity, partition/hash selector, stride delay,
or one-cycle-per-merged-sector drain law. The clock64 differential mixes LSU
issue, occupancy, address/partition effects, overlap, and return visibility;
the selected NCU interfaces do not isolate a matching L2 request-stall or
frontier law. H31/H35/H37 full-corpus pressure/route evidence is likewise
non-monotonic and sparse; the prior merged-sector timing arm changed 1,670
frontiers, improved GPC mean APE by only ~0.004 percentage points, and worsened
SM-active APE before removal.

H30–H37 route/slice/fill/pressure counters remain observer-only. A future
partition/queue timing arm may be promoted only if a locked one-factor DRAM/L2
probe produces a monotonic independent frontier/stall delta and the captured
NVBit trace replays with a held-out real-trace benefit. The next probe must
separate read demand from terminal dirty writeback/coherence completion before
any `sim/sm/src/system.rs` service change.

No simulator source or full-mode behavior changed in H64; therefore no new
full-corpus replay is claimed. Existing raw/all-row H41 APE remains authoritative
and no scope-compatible secondary value is relabeled as closure.
