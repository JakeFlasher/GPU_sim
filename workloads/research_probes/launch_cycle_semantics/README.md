# Launch-cycle and drain-semantics probe

This `sm_86` probe distinguishes three timing boundaries that must not be
silently conflated:

- the final instruction issue on an SM;
- completion of outstanding modeled memory work at kernel retirement; and
- the profiler's launch or multi-launch range envelope.

Every CTA contains one warp.  All lanes execute the same runtime-sized
dependent IMAD chain, rendezvous, and then execute one compile-time-specialized
tail: no memory operation, a discarded or consumed global load, a global
store, a non-returning reduction, or a returning atomic whose value is stored.
The tail addresses are spaced by 128 bytes so one lane cannot hide behind
another lane's cache line.

Build and run a correctness smoke:

```bash
make -C workloads/research_probes/launch_cycle_semantics
workloads/research_probes/launch_cycle_semantics/launch_cycle_semantics \
  red 84 256 4 3
```

Arguments are `TAIL BLOCKS ITERATIONS CHAIN_LENGTH TRIALS`.  The executable
places `cudaProfilerStart/Stop` around the exact same-stream launch chain.
Ordinary NCU application replay therefore reports one row per kernel, whereas
NCU Range Replay reports one row for the complete chain.  NVBit captures the
same explicit range and retains every launch for direct replay.

The completed campaign is under
`results/captures/sm86_launch_cycle_semantics_20260818_v1`. Reproduce or resume
its stages with:

```bash
.venv/bin/python workloads/research_probes/launch_cycle_semantics/campaign.py manifest
.venv/bin/python workloads/research_probes/launch_cycle_semantics/campaign.py correctness
.venv/bin/python workloads/research_probes/launch_cycle_semantics/campaign.py ncu-kernel
.venv/bin/python workloads/research_probes/launch_cycle_semantics/campaign.py ncu-range
.venv/bin/python workloads/research_probes/launch_cycle_semantics/campaign.py ncu-app-range
.venv/bin/python workloads/research_probes/launch_cycle_semantics/campaign.py nvbit
.venv/bin/python workloads/research_probes/launch_cycle_semantics/campaign.py static
.venv/bin/python workloads/research_probes/launch_cycle_semantics/campaign.py simulator
.venv/bin/python workloads/research_probes/launch_cycle_semantics/campaign.py simulator-carded-dram
.venv/bin/python workloads/research_probes/launch_cycle_semantics/analyze.py
```

The final result promotes a typed complete/incomplete modeled-retirement
frontier, but rejects numerical `gpc__cycles_elapsed.max` emission and generic
multi-launch range composition.
