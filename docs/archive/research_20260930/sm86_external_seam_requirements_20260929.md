# SM86 external seam requirements

This is a continuation plan, not a completion claim. It records the minimum
new external evidence required to move each remaining mechanism from a bounded
latent model to a mechanism-level model. Existing public campaigns have already
exercised the available local seams.

## 1. Physical pipe and issue identity

Required external record, per issued instruction:

```text
GPU UUID, GPC/TPC/SM, warp/lane mask, kernel launch ordinal,
SASS PC, dynamic issue ordinal/cycle, selected scheduler,
physical sub-unit/pipe, stall/dependency reason
```

A PMA/PC-sampling stream without physical pipe and issue ordinal is
insufficient. Aggregate `smsp__inst_executed_pipe_*` counters, isolated class
correspondence, and source pages remain useful priors but cannot close mixed
kernels.

## 2. Independent CUDA ABI/ISA/control generation

Required external/compiler surface:

- complete SM86 SASS encoding plus control words;
- relocatable CUDA ELF sections and relocations;
- QMD/parameter ABI generation for arbitrary signatures;
- scheduler-safe barrier/register metadata;
- launchable Driver API images produced without reusing NVIDIA metadata.

Mesa NAK, NVIDIA PTX compilation, `nvJitLink`, `denvdis`, and the bounded
NAK-to-CUDA-ELF bridge are retained as partial producers. The shifted-template
experiment demonstrates that metadata layout changes warm timing even when
SASS/resource populations are unchanged.

## 3. Cache ownership/read-from/DRAM completion

Required external telemetry or oracle:

- line address plus owner/cache level and coherence state;
- read-from source for every load (L1/L2/DRAM/peer);
- dirty payload residency and eviction/writeback event;
- fence/flush completion tied to the specific line;
- proof that a terminal dirty line reached DRAM or a declared bypass consumer.

The current cache-phase model uses cold/warm transfer intervals and sector/L2
observations. Public full-cache controls and CPU-visible consumer coupling do
not provide the missing line-level ownership/read-from trace.

## 4. Unified timing endpoint

Required synchronized endpoint:

- common start marker and end marker;
- inclusion/exclusion rules for launch setup, front-end scheduling, final
  memory completion, and semaphore/WaitForIdle work;
- per-launch mapping across NCU, CUPTI, CUDA event/device timer, simulator
  completion, and host replay;
- clock/cache/serialization state captured with the endpoint.

The CUDA-event→NCU transform is accurate for the paired-warm flash anchors, but
it is a local crosswalk, not proof that the endpoints are semantically equal.
The clock-sampled follow-up at
`results/alpha/full_3090ti_20260929_stage_clock_sampled_live2/` records an
external SM-clock trace (1,905 MHz active median) and confirms that static NCU
device attributes do not substitute for per-launch clock state. It still lacks
the common start/end markers and endpoint inclusion rules required here.
The opt-in clock-regime adapter reduces duration error under this observed
boost regime, but it is only dimensional normalization; it does not supply the
missing synchronized endpoint or GPC-clock relation.
The subsequent NCU `--clock-control base` campaign still observed a 2,040 MHz
active median externally, so profiler clock-control selection is recorded as a
request, not accepted as proof of the physical clock regime.
The interleaved probe at `results/alpha/sm86_profile_clock_context_20260930/`
adds per-kernel NCU SM/GPC frequency counters, but those counters still do not
provide the common simulator/CUDA-event/NCU start-end relation required for
unified endpoint closure.

## Current stopping rule

When a new public endpoint supplies one of the records above, rerun the
corresponding bounded campaign and update the latent model. Until then, keep
these fields typed as `unknown`, interval, candidate set, or partial order;
do not promote a formula, count-balancing heuristic, or template bridge into
mechanism closure.
