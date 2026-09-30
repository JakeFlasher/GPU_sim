# SM86 scheduler/SMSP contention witness (2026-09-30)

## Purpose

This is a fresh, performance-only attack on the exact SM86 pipe/issue witness
boundary.  It asks a narrower falsifiable question than the unavailable
per-instruction route tag: do logical warp groups partition into independent
issue paths on the local RTX 3090 Ti, and can that partition be observed under
controlled one-CTA-per-SM pressure?

The experiment does **not** claim that a CUDA `%warpid` value is a stable SMSP
ABI or that a scheduler path is an ALU/FMA/LSU sub-pipe.  Those distinctions are
important for the remaining blocker.

## Reproducible local setup

- GPU: NVIDIA GeForce RTX 3090 Ti (GA102, SM86), 84 enabled SMs
- UUID: `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`
- Driver: 615.71.09
- Clocks at capture: graphics/SM 1350 MHz, memory 10501 MHz, P0
- Toolchain: CUDA `nvcc`, `-O3 -std=c++17 -arch=sm_86`
- Source snapshots and raw output: `results/alpha/sm86_smsps_scheduler_contention_goal_20260930/`

The temporary source is adapted from the public
[`rtx5090-smsp-badapple`](https://github.com/Loong0x00/rtx5090-smsp-badapple)
allocation probe.  It launches a cooperative grid with 84 CTAs and uses the
maximum opt-in dynamic shared-memory allocation to limit residency to one CTA
per SM.  Every selected warp executes eight independent FFMA accumulators, so
the measurement is issue-pressure dominated rather than dependency-latency
dominated.  CUDA events report the elapsed time for each mask; no repository
source or test was changed.

## Placement witness

A separate minimal kernel reads the documented volatile `%smid` and `%warpid`
special registers from lane zero of each of the four warps in every CTA.  The
single launch reported:

```
GPU=NVIDIA GeForce RTX 3090 Ti SMs=84 unique_smid=84 range_smid=0..83 unique_warpid=4 range_warpid=0..3
same_sm_per_cta_bad=0
```

Every one of the 84 enabled SMs was covered by one CTA; each CTA's four warps
reported warp IDs 0, 1, 2, and 3.  This establishes placement and the
existence of four observable warp slots in this occupancy envelope.  It does
not itself name the internal scheduler/SMSP or guarantee placement in a later
launch.

The separate 512-thread placement follow-up reports 84 unique SM IDs, physical warp IDs
0 through 15 in logical-warp order in every CTA, and zero within-CTA SM-ID
disagreements.  This supports the plausibility of a logical-to-physical
mapping in the corresponding occupancy envelope, but placement was not read
inside the timed contention launch.  It therefore cannot prove that mapping
for the actual sixteen-warp contention samples.

## Contention witness

At 100,000 loop iterations and five repeats (three identical campaign runs),
CUDA-event times were:

| Selected logical warps | Mask(s) | Mean time (ms) | Interpretation |
|---|---:|---:|---|
| one of 0,1,2,3 | `0x1`, `0x2`, `0x4`, `0x8` | 0.910 | one active warp path |
| any pair among 0..3 | `0x3`, `0x5`, `0x9`, `0x6`, `0xa`, `0xc` | 0.910 (one run had 0.966 for two pairs) | no measurable contention |
| same low-two-bit pair | `0x11`, `0x22`, `0x44`, `0x88` | 1.484 | two warps share one path |
| four stride-4 warps | `0x1111`, `0x2222`, `0x4444`, `0x8888` | 2.964 | four warps share one path |
| all first 16 warps | `0xffff` | 2.964 | four groups run in parallel |

A direct 100,000-iteration run with a 512-thread block produced 1.485--1.487
ms for masks `0x11/0x22/0x44/0x88` and 2.966--2.970 ms for
`0x1111/0x2222/0x4444/0x8888/0xffff`.  Thus the approximately 2x slowdown
from two to four same-path warps is stable, while adding four independent
low-two-bit paths does not add another 4x.  This is strong local evidence for
four independently productive resource paths and for a same-low-two-bit
logical-warp contention partition on this SM86 stack.

The public probe's original interpretation says `SMSP = physical_warpid & 3`
for its tested GB202.  On this GA102 run we retain the weaker claim:
`logical_warp_index & 3` predicts the contention partition in this exact
one-CTA/SM envelope.  Separate placement runs reported matching physical warp
IDs, but they are not same-launch identity witnesses.  No physical SMSP
number or PTX affinity guarantee is established.

The unmodified external allocation probe also evaluates its own timing-ratio
predicate at process exit.  That bundled pass/fail interpretation was run; it
is not independent silicon validation and is not a repository test suite.
The observations above come from the printed timings and placement values,
not that predicate.

## Aggregate NCU cross-check

A separate one-kernel launcher was profiled with Nsight Compute 2026.1.0 using
aggregate metrics only (`smsp__inst_executed_pipe_fma.sum`,
`smsp__inst_executed_pipe_alu.sum`, `smsp__inst_executed.sum`,
`smsp__cycles_active.sum`, and `gpu__time_duration.sum`).  For 10,000 loop
iterations:

| Mask | FMA instructions | ALU instructions | all instructions | SMSP active cycles | NCU duration |
|---:|---:|---:|---:|---:|---:|
| `0x11` (warps 0,4) | 13,442,856 | 849,408 | 15,141,000 | 17,128,129 | 108.03 us |
| `0x1111` (0,4,8,12) | 26,885,712 | 1,690,752 | 30,267,216 | 33,917,890 | 213.54 us |
| `0xffff` (16 warps) | 107,542,848 | 6,738,816 | 121,024,512 | 134,745,890 | 213.82 us |

Instruction counts scale with the number of active warps.  Active cycles scale
approximately 2x from `0x11` to `0x1111` and approximately 8x from `0x11` to
`0xffff`, while the latter two masks have nearly identical elapsed duration:
the four same-path warps serialize, whereas four independent paths execute in
parallel.  These metrics are aggregate rollups; no per-instruction dynamic
pipe tag or globally identifiable SMSP coordinate is present.

The Nsight Compute metric catalog itself describes
`sass__inst_executed_per_opcode_pipeline` as an **estimated** SASS opcode
pipeline and explicitly says dynamic assignment is not taken into account.
The raw report therefore supports a four-path resource model only; it does
not turn `smsp__inst_executed_pipe_fma` into a physical route witness.
The default Nsight Compute GA102 metric query in
`ga102_ncu_sampling_metric_query.txt` returned no `warpidsamp` or `pcsamp`
metric names.  That query omits the separate source-metric collection and is
**not** evidence of missing GA102 sampling support; the retained source-metric
query and successful local PC-sampling captures take precedence.

## External corroboration and limits

- CUPTI's current PC-sampling guide states that each SM sampler selects a
  random active warp and that the scheduler may select a different warp in the
  same cycle; collected samples lack time resolution.  See
  <https://docs.nvidia.com/cupti/main/main.html#cupti-pc-sampling-api>.
- NVIDIA open-kernel-modules issue [#1351](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1351), opened 2026-09-10, is an independent user's feature request.
  Its author reports not finding a supported globally identifiable physical
  SM/SMSP sampling interface.  It corroborates the research question, but is
  not an authoritative NVIDIA statement of support or impossibility.
- The public allocation probe's README documents the same-low-two-bit
  contention experiment on GB202 and explicitly labels the formula empirical,
  not a PTX ABI: <https://github.com/Loong0x00/rtx5090-smsp-badapple>.
- Nsight Compute's current metric semantics describe
  `math_pipe_throttle` as waiting for an execution pipe, but do not expose the
  selected physical sub-pipe or issue slot:
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>.

## Closure status

**New evidence:** local SM86 logical-warp groups are independently productive,
and same-low-two-bit groups contend with a stable ~2x slope.  This is
resource-model corroboration, not exact scheduler/SMSP identity or closure.
It gives a conservative hypothesis for model resource envelopes: four parallel
paths with a latent path key derived from the logical warp index only inside
this occupancy/control envelope.

**Still open:** exact per-instruction physical pipe/sub-unit and issue-slot
identity.  PMA, CUPTI PC samples, Nsight aggregate counters, and this contention
experiment do not expose the dynamic route chosen for a particular SASS
instruction.  The exact hardware pipe/issue witness blocker therefore remains
open; the safe model update is a set-valued four-path constraint, not a solved
tag.
