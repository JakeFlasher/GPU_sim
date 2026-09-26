# Rebuilt NVBit + fresh SM86 follow-up — 2026-09-25

This is a bounded post-review regression, not a replacement for the retained
three-workload holdout. The NVBit producer was rebuilt with the repository
contract (`CUDA 13.2`, `/usr/bin/g++-14`, protobuf/protoc 34.1), then used to
capture fresh one-warp and 168-block SM86 memory-domain workloads. NCU was run
independently after capture; no NCU value was used to configure replay.

## Host and binary identity

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090 Ti |
| UUID | `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c` |
| Compute capability | 8.6 |
| Driver | 615.71.09 |
| NVCC | CUDA 13.2.51 |
| Host compiler | GCC 14.3.1 |
| Protobuf | protoc 34.1 / runtime 34.1.0 |
| `sim-run` SHA-256 | `620faa8531ed6dc92c752f539b5f5f337feeb5b643aeae0c1b45be93a7399700` |
| `tracer_tool.so` SHA-256 | `5191b4b6f701c897438175f7cc2c423c790148a05ecc41dce3244ce9f6fc4a74` |

Full input/output identities and commands are retained under the ignored local
directory `results/captures/sm86_review_followup_20260925_z7tv/`.

## Independent count cross-check

| Case | Blocks × iterations | Instructions | Global load req/sectors | Global store req/sectors | ATOM sectors | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| contiguous read | 1 × 8 | 55 / 55 | 1 / 4 | 1 / 4 | 0 / 0 | exact |
| same-word ATOM | 1 × 8 | 67 / 67 | 0 / 0 | 1 / 4 | 256 / 256 | exact |
| same-word ATOM | 168 × 32 | 31,416 / 31,416 | 0 / 0 | 168 / 672 | 172,032 / 172,032 | exact |

The pairs in each cell are simulator / NCU. The 168-block case exercises the
large-grid trace path and the driver occupancy metadata (`16 blocks/SM`). The
same-word ATOM cases also expose the topology split: 256 serialized sector
entries from only 8 unique sectors in the one-warp case, and 172,032 entries
from 5,376 unique sectors in the larger case.

## Profiler crosswalk round trip

A separate NCU capture requested the newly crosswalked families on the fresh
read kernel. The raw export had 304 columns and the rebuilt `sim-run` binary
round-tripped these observations without using them as replay inputs:

| NCU observation | Value |
| --- | ---: |
| `l1tex__t_requests.sum` | 2 requests |
| `lts__t_requests_srcunit_tex.sum` | 2 requests |
| L1 M-stage read/write sectors | 4 / 4 |
| DRAM bytes/s | 1,405,405,405.41 |
| compute-memory access/request throughput | 1.49% / 0.49% |
| SM memory-pipe throughput | 0.00% |
| GPC/DRAM cycles per second | 1,976,532,335.91 / 10,054,054,054.05 |
| kernel duration sum | 2,368 ns |

The parser stores these under `ncu_observation`; native replay metrics remain
separate and do not consume them. The capture command, raw CSV, replay JSON,
and identity checksums are retained under the local follow-up directory.

## Timing boundary

The simulator still reports separate issue, modeled-event, GPC-envelope, and
completion-tail frontiers. For example, the 168-block ATOM replay's modeled
GPC observation frontier is 11,547 cycles, while dirty writeback and coherence
remain typed completion tails. These fresh count matches do not establish
elapsed-cycle accuracy.
