# L1TEX admission/backpressure probe

This prospective `sm_86` probe separates three intervals that elapsed-kernel
time alone collapses:

- CLOCK-to-CLOCK issue/admission time for a memory burst;
- the matched-control `MEMBAR.SC.GPU; ERRBAR; CCTL.IVALL` fence sequence after
  the final burst instruction; and
- the external NCU GPC/SM-active launch envelope.

It varies STORE, non-returning RED, returning ATOMG, unique versus reused line
topology, same-word aliases, cross-warp 256-byte regions, burst depth, compute
spacing, active warp count, and sparse SMSP occupancy. Inactive warps retain
the same instruction loop but issue no target memory operation.

Build and run a pilot:

```bash
make -C workloads/research_probes/l1tex_admission_backpressure
workloads/research_probes/l1tex_admission_backpressure/l1tex_admission_backpressure \
  store line_unique 84 8 32 8 0 all 3
```

Arguments are
`TAIL LAYOUT BLOCKS WARPS DEPTH ACTIVE_LANES GAP_STEPS PATTERN TRIALS`.
Every target uses a GPU-scope memory barrier after its measured issue region;
nvcc lowers it to a composite barrier/ERRBAR/CCTL sequence. The control executes
the same CLOCK, loop, fence sequence, and output structure. The second interval
must not be interpreted as a pure retirement latency.

The completed campaign is under
`results/captures/sm86_l1tex_admission_backpressure_20260818_v1`. Actions are
`manifest`, `correctness`, `native`, `ncu`, `source`, `nvbit`, `static`,
`simulator`, `memory-service-candidates`, `memory-admission-candidates`,
`memory-admission-feedback`, and `memory-admission-feedback-service-sweep`.
Run `analyze.py` for the base campaign and `analyze_feedback_service_sweep.py`
for the rate/width successor. `memory-write-completion-feedback` and
`analyze_write_completion_feedback.py` run the schema-11 successor;
`analyze_write_completion_latency_sweep.py` reduces the schema-12 fixed-tail
sweep. `red-rmw-completion-sweep` and
`analyze_red_rmw_completion_sweep.py` run the schema-13 exact request/region
work and bounded-lane successor. The retained result contains 100 cells, 384 NCU
profiles, 16 SourceCounters reports, 100 NVBit captures, and deterministic
ordinary/candidate replay pairs. Sweep selection uses five shallow cells and
advances every width at calibration-Pareto rates to the remaining 20 cells.

## C7-S33AR live refresh (2026-10-04)

The active Makefile/campaign interface now builds and invokes only
`l1tex_admission_backpressure`; the historical trace-only executable is no
longer a required input. A fresh `store, line_unique, blocks=84, warps=8,
depth=32, active_lanes=8, gap=0, pattern=all` cell is captured with the same
rebuilt binary through correctness, NVBit, replay, and five application-replay
NCU runs. The retained artifact records the exact binary/control/trace binding,
instruction crosswalk, lower-bound comparison, and open completion tails.

The follow-up same-shape `control` versus `store` differential also uses that
single rebuilt binary. Five NCU medians are 16,967 and 29,790 cycles, whereas
the simulator lower-bound delta is 556 cycles (12,349 to 12,905). This is a
mechanism-local mismatch, not a fitted store/L1TEX latency. Evidence:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_pair_live_v1.json`.
