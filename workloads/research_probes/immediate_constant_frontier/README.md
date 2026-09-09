# SM86 Immediate Constant Cache (IMC) probe

This is an unscored hardware-characterization probe. The canonical
`valid-imc-probe` target uses hand-authored PTX (`imc_kernel.ptx`) so four
parameter loads remain inside the timed loop and ptxas lowers them to explicit
SASS `IADD3 ... c[0x0][offset]` operands without local-memory spills. The older
`immediate_constant_frontier.cu` target is retained as historical setup
coverage; its timestamp bookkeeping spills to local memory, so its `Sites`
sweep must not be used to infer IMC latency.

Build and inspect the emitted SASS:

```bash
make -C workloads/research_probes/immediate_constant_frontier valid-imc-probe
# PTXAS_FLAGS=--opt-level=1 is pinned by the Makefile to keep the emitted
# SASS schedule comparable to the retained no-spill holdout capture.
/usr/local/cuda-13.2/bin/cuobjdump --dump-sass \
  workloads/research_probes/immediate_constant_frontier/imc_kernel.cubin
workloads/research_probes/immediate_constant_frontier/imc_valid_driver 84 256

# Historical setup-only probe (do not use for latency fitting):
/usr/local/cuda-13.2/bin/cuobjdump --dump-sass \
  workloads/research_probes/immediate_constant_frontier/immediate_constant_frontier
workloads/research_probes/immediate_constant_frontier/immediate_constant_frontier 84 4 1024 1
```

Nsight Compute identifies IMC as a per-SMSP cache for `c[bank][offset]`/`cx`
references. A miss issues the instruction without dispatching it and re-issues
after the miss returns. The released modern-core SM86 model uses a `4:64:2`
L0 constant cache, 32 MSHRs, an 11-cycle SM-side structure, and a pipelined
one-cycle initiation interval. Record the GPU UUID, clock protocol, exact
SASS, direct-clock outputs, and any NCU
`smsp__warps_issue_stalled_imc_miss` profile before promoting a replay change.
The simulator readiness JSON reports line-probe attempts and pending merges
separately from NCU's overlapping warp-cycle stall metric.

With the pinned `--opt-level=1` build, the local cubin's dumped SASS is
identical to the retained no-spill holdout SASS (the ELF/cubin hashes can
still differ because of toolchain metadata). A fresh qualitative smoke run
from the probe directory produced `min_cycles=7218` and `max_cycles=7266` for
`84` blocks and `256` iterations. The run used the live, unmodified clock
state and is not inserted into the scored cohort; the retained base-clock NCU
capture remains the quantitative holdout reference.
