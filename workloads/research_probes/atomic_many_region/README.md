# sm_86 many-region atomic scheduling probe

This probe holds one RED cubin, 84 one-warp CTAs, 1,024 generations, and a
fixed 32 MiB device allocation constant while varying only the runtime-selected
set of aligned 256-byte regions.

```bash
.venv/bin/python workloads/research_probes/atomic_many_region/campaign.py manifest
.venv/bin/python workloads/research_probes/atomic_many_region/campaign.py correctness
.venv/bin/python workloads/research_probes/atomic_many_region/campaign.py ncu-pairs
.venv/bin/python workloads/research_probes/atomic_many_region/campaign.py ncu
.venv/bin/python workloads/research_probes/atomic_many_region/campaign.py source
.venv/bin/python workloads/research_probes/atomic_many_region/campaign.py trace-correctness
.venv/bin/python workloads/research_probes/atomic_many_region/campaign.py nvbit
.venv/bin/python workloads/research_probes/atomic_many_region/campaign.py static
.venv/bin/python workloads/research_probes/atomic_many_region/campaign.py simulator --force
.venv/bin/python workloads/research_probes/atomic_many_region/analyze.py
```

The pair lane collects only `lts__t_sectors_op_red.sum`; NCU expands it to
per-LTS average/minimum/maximum/sum reductions. The triplet lane tests whether
maximum LTS-slice occupancy explains the previously unresolved many-region
timing classes. Hardware counters remain evaluation evidence, never simulator
inputs.

