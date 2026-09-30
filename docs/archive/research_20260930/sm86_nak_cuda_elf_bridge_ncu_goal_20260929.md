# NAK-to-CUDA-ELF bridge: live NCU observation

This is a live performance/execution observation, not a test, hash, or
verification workflow. It profiles two already-retained NAK-generated bodies
inserted into the bounded two-parameter NVIDIA CUDA-ELF template. NCU is an
independent measurement endpoint and was not used to choose the bridge or its
metadata.

## Setup

* RTX 3090 Ti / SM86 / driver 615.71.09 / GPU index 0.
* CUDA Driver API launcher compiled from the retained
  `launch_two_param.cc` source.
* NCU `--clock-control none`, `--cache-control all`, one application-replay
  launch, and metrics for duration, GPC cycles, total instructions, and
  aggregate ALU/LSU counters.
* Raw reports and profiler outputs are retained under
  `results/alpha/sm86_nak_cuda_elf_bridge_ncu_goal_20260929/`.

## Results

| body | Driver API output | NCU duration | NCU GPC cycles | SASS instructions | ALU aggregate | LSU aggregate |
|---|---|---:|---:|---:|---:|---:|
| `03_mul.cubin` | `0,3,6,9` | 2.400 us | 3,226 | 14 | 6 | 3 |
| `06_shift.cubin` | `0,2,4,6` | 2.432 us | 3,233 | 14 | 10 | 3 |

The expanded family matrix adds four more retained bodies:

| body | Driver API output | NCU duration | NCU GPC cycles | SASS instructions | ALU aggregate | LSU aggregate |
|---|---|---:|---:|---:|---:|---:|
| `04_fma_like.cubin` | `5,8,11,14` | 2.464 us | 3,260 | 18 | 10 | 3 |
| `05_reciprocal.cubin` | raw FP values | 2.496 us | 3,325 | 18 | 6 | 3 |
| `07_xor.cubin` | `3,2,1,0` | 2.432 us | 3,239 | 14 | 10 | 3 |
| `08_maximum.cubin` | `5,5,5,5` | 2.624 us | 3,478 | 14 | 10 | 3 |

Warm-cache NCU profiles (`--cache-control none`) are retained under `warm/`.
Across the six bodies, the median warm/cold duration ratio is 0.95485 and the
median warm/cold GPC-cycle ratio is 0.95105. Duration ratios range from
0.9268–1.0000 and GPC-cycle ratios from 0.9330–1.0028. SASS instruction and
aggregate pipe counts remain unchanged. This adds cache-domain endpoint
sensitivity for the bridge, not a cache-owner, read-from, or DRAM-completion
witness.

The profiles establish that the bounded bridged bodies execute through the
CUDA Driver API and are visible to NCU. They do not expose a dynamic physical
pipe/issue-slot tag, cache ownership/read-from state, complete DRAM writeback,
or general relocatable CUDA ABI generation. The metadata and parameter layout
remain reused from NVIDIA's template, so the compiler/ABI blocker is narrowed
but not closed.
