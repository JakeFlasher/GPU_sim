# NVIDIA Architecture Reference

> **SOLBENCH NOTICE.** Solbench targets B200 (sm_100a). The sm89 (Ada Lovelace) section below is preserved as **cross-architecture background context** for understanding what came before Blackwell. The standalone `sm89-optimization-guide.md` file was NOT imported into solbench (Ada is not a relevant tuning target). sm120 is solbench's local dev proxy only; sm_100a (B200) is the scoring target.

Based on the sm89, sm90, sm100, sm103, and sm120 architecture summaries from kernel-pilot. sm89 / sm120 content remains here as historical / cross-arch reference; only sm_100a/sm_100 (B200) is a solbench tuning target.

Read this file before interpreting `nsys` results or static-introspection output (cuobjdump/nvdisasm/ptxas -v), or changing tile sizes, stage count, memory movement, tensor-core path, or occupancy assumptions. (Original kernel-pilot text referenced `ncu`; solbench is counter-free per CLAUDE.md Rule 1.)

## sm89: Ada Lovelace

Examples: L40S, RTX 4090, RTX 6000 Ada.

Key traits:

- 48 warps/SM, 1536 threads/SM, up to 24 blocks/SM
- about 100 KB max shared memory per SM
- large L2 on high-end devices, but GDDR bandwidth is far below HBM
- no TMA or thread-block clusters
- supports `cp.async`
- FP8 tensor cores exist, but custom kernels usually use FP16/BF16 with FP32 accumulation

Tuning bias:

- vectorize memory access aggressively
- rely on L2 reuse and fusion to compensate for lower memory bandwidth
- prefer 256 or 512 thread blocks; avoid 1024-thread blocks when occupancy matters
- use `cp.async` for suitable global-to-shared staging

## sm90: Hopper

Examples: H100, H200.

Key traits:

- 64 warps/SM, 2048 threads/SM
- about 192 KB max shared memory per SM
- HBM bandwidth and TMA support
- thread-block clusters and distributed shared memory
- WGMMA and warpgroup programming patterns

Tuning bias:

- use larger shared-memory tiles than Ada when they improve reuse
- inspect TMA overlap and warpgroup behavior
- use `nsys` for overlap and static introspection (ptxas -v) for occupancy, register pressure, tensor-core utilization, and stall reasons

## sm100: Blackwell Datacenter

Examples: B200, GB200.

Key traits:

- 64 warps/SM, 2048 threads/SM, up to 32 blocks/SM
- about 228 KB max shared memory per SM
- very large L2 and HBM bandwidth
- enhanced TMA, multicast, clusters, and distributed shared memory
- tcgen05 tensor core instruction family
- TMEM separates tensor operands from the register file
- native FP4/FP6/FP8 paths

Tuning bias:

- consider larger tiles than Hopper when shared-memory budget allows
- use TMA for regular bulk movement
- use tcgen05/TMEM-capable library or DSL paths for GEMM-like kernels
- profile tensor-core utilization, TMA throughput, L2 hit rate, and cross-die effects

## sm103: Blackwell Ultra

Examples: B300, GB300.

Key traits:

- same SM microarchitecture as sm100
- more SMs, larger L2, more HBM capacity, and PCIe Gen6
- sm100 kernels generally port directly

Tuning bias:

- query SM count at runtime and size grids for 160 SMs when applicable
- exploit larger L2 with persistence hints for hot inference data such as KV cache
- watch wave quantization: grid sizes just over a full wave can reduce average utilization

## sm120: Blackwell Desktop or Workstation

Examples: RTX 5090, RTX PRO 6000.

Key traits:

- closer to Ada than datacenter Blackwell for occupancy and execution model
- 48 warps/SM, 1536 threads/SM, up to 24 blocks/SM
- about 128 KB shared memory per SM
- large L2 and GDDR7 bandwidth, but still far below B200 HBM
- supports TMA for regular bulk movement
- no clusters, no TMEM, no datacenter tcgen05 path
- tensor-core access is closer to WMMA/register-fragment style

Tuning bias:

- fuse kernels aggressively because memory bandwidth is the limiter
- watch register pressure because there is no TMEM to hold tensor operands
- reduce sm100 tile sizes and remove cluster assumptions when porting
- prefer 256-thread blocks; avoid 1024-thread blocks when occupancy matters

## Cross-Architecture Rules

- Query SM count and device properties at runtime.
- Do not hardcode shared-memory budgets or SM counts.
- Keep tile choices tied to target architecture.
- Validate on every architecture claimed by the code path.
- Treat performance conclusions as architecture-specific until measured elsewhere.
