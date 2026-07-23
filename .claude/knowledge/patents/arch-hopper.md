---
Use when: explaining Thread Block Clusters; placing DSMEM; reasoning about TMA descriptors and `cp.async.bulk.tensor`; understanding the transaction barrier (`mbarrier.complete_tx`); placing WGMMA; understanding the FP8 + Transformer Engine numerical contract; placing the third architectural rewrite of the corpus.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.4.
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/6.hopper.md (GH100 SM card); references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (memory hierarchy row); references/NVIDIA_Manuals/cuda_guide/01-02-02-01-01_thread-block-clusters.md; references/NVIDIA_Manuals/cuda_guide/02-02-03-08_distributed-shared-memory.md; references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md (wgmma, tcgen05).
Patents anchored here: US20230289189A1 (Distributed Shared Memory, filed 2022-03-10), US12141082B2 (TMA / Tensor Memory Access Unit, filed 2022-03-10), US20230289242A1 (transaction barrier / mbarrier, filed 2022-03-10).
Out of scope: Blackwell's TMEM / tcgen05 / FP4 (`arch-blackwell.md`); generic CUDA cluster API ergonomics (covered in `.claude/knowledge/cuda/cuda-clusters-dsmem.md`).
Version family: Hopper architecture, 2022. Representative product: GH100 (H100 SXM5, 2022-09). Compute capability 9.0.
---

# Hopper (2022, GH100) — rewrite #3: cluster as a hardware execution domain

Hopper is **rewrite #3** of the three-rewrites thesis (`three-rewrites.md`).
The breakpoint is **the cooperation boundary**: shared-memory-style CTA
cooperation crosses the single-`SM` boundary for the first time, via
**Thread Block Clusters + Distributed Shared Memory + TMA + transaction
barriers + WGMMA**.

Looking only at the H100 SM diagram is misleading. Yes, Hopper
strengthens per-`SM` Tensor Core / FP32/INT / LD/ST / L1/SMEM /
scheduler / register resources — but the real architectural change
is that the full execution domain extends *beyond* the single SM.

## SM organization (GH100)

| Component | Count |
|---|---|
| Warp Scheduler | 4 |
| Dispatch Unit | 4 |
| FP32 Core | **128** (per SM, 2× the doubled-FP32 of GA102/AD102) |
| INT32 Core | 64 |
| FP64 Core | 64 |
| **Tensor Core (4th gen)** | **4** (per SM) |
| LD/ST Unit | 32 |
| SFU | 16 |
| Register file / SM | 64 KB |
| Shared memory + L1 (combined) | **256 KB / SM** (max 228 KB usable for SMEM) |
| L2 cache (whole chip, GH100) | up to 60 MB; H100 ships with 50 MB |
| Memory | HBM3, 5 stacks × 10 × 512-bit memory controllers, 80 GB |
| SM count (H100 SXM5) | 132 (out of full GH100) |

**Notes:**

- H100 SXM5: 132 SMs (66 TPCs × 2 SMs/TPC distributed across 8 GPCs;
  full GH100 has 72 TPCs / 144 SMs).
- Per-SM L1+SMEM goes up to 256 KB (Volta was 128, Ampere 192).

## Performance numbers (H100 SXM5, NVIDIA marketing)

Sparse-mode peak (NVIDIA quotes sparse; dense is half):

- FP16 Tensor Core: ~989 TFLOPS dense, ~1979 TFLOPS sparse.
- BF16 Tensor Core: ~989 TFLOPS dense, ~1979 TFLOPS sparse.
- FP8 Tensor Core: ~1979 TFLOPS dense, ~3958 TFLOPS sparse.
- HBM3 bandwidth: ~3.35 TB/s.
- **NVLink 4.0**: ~900 GB/s aggregate per GPU.
- MIG 2.0.

**Caveat**: throughput numbers are nominal peak per Tensor Core × SM
count × clock; achieving them in real kernels requires correctly-
shaped tile descriptors and full pipeline saturation. Source:
NVIDIA H100 product page and H100 datasheet (not in
NVIDIA_patentd.md, which doesn't quote per-precision TFLOPS).

## Five interlocking interfaces forming a closed loop

Hopper's core cannot be reduced to a feature checklist. Reading any
single one out of context misreads it as "module X got stronger." The
right framing is that **five interfaces close a loop**:

1. **Numerical contract**: FP8 (E4M3 / E5M2) + Transformer Engine
   bind low-bit format choice, scale management, and matrix operators
   together.
2. **Consumer granularity**: `WGMMA` lifts the matrix consumer side
   from one warp to a **warp group** (128 threads, 4 warps).
3. **Transport path**: TMA moves address generation and transport for
   large multi-dimensional tensor tiles off the threads onto a
   **dedicated hardware path**.
4. **Shared scope**: Thread Block Clusters + DSMEM extend
   shared-memory-style cooperation from **a single SM to a group of
   co-scheduled SMs in one GPC**.
5. **Async + barriers**: transaction-aware barriers wait not only on
   thread arrival but also on data-transport transaction arrival.

Anchoring patents:
- DSMEM → US20230289189A1.
- TMA → US12141082B2.
- Transaction barrier → US20230289242A1.
- WGMMA → no dedicated patent in the corpus, but the warp-group
  collective issue model is described in PTX 9.0 `wgmma.mma_async`.

## Thread Block Clusters

- An **optional third level of grouping** above thread block. Clusters
  are 1-D, 2-D, or 3-D sets of thread blocks.
- All blocks in a cluster execute simultaneously **on SMs in a single
  GPC**.
- Grid dim must be a multiple of cluster dim in every axis.
- Portable cluster size: **8**. Nonportable opt-in (B200): up to **16**
  via `cudaFuncAttributeNonPortableClusterSizeAllowed`.
- `cluster.sync()` guarantees all blocks in the cluster are resident
  and is required before any DSMEM access of another block's shared
  memory.
- Block cannot move between GPCs mid-execution.

## Distributed Shared Memory (DSMEM) — US20230289189A1

- **Filed** 2022-03-10, NVIDIA. Inventors: Bangalore Prabhakar, Hirota,
  Krashinsky, Long, Pharris, Dash, Tuckey, Duluk Jr., Shah, Durant,
  Choquette, Werness, Govil, Patel, Deb, Navada, Edmondson, Palmer,
  Gandhi, Manyam, Parle, Giroux, Gadre, Heinrich.
- **Title**: "Distributed Shared Memory."
- **Claim**: blocks of memory distributed across a processor (a GPU).
  Threads on one processing core can access the memory block local to
  a different processing core. New programming and hardware execution
  model for tighter coordination at scopes larger than a single SM.

### How DSMEM works

Each thread block still owns its own per-block shared-memory
partition. Capacity declaration and lifetime stay scoped to the block.
What DSMEM does is **map all per-block shared-memory partitions in
one cluster into a single address space accessible across blocks**.

- `Block 0` writes a tile into its own SMEM.
- After `cluster.sync()`, `Block 1` reads that tile via DSMEM.
- Block 1 can also do atomics on Block 0's SMEM.
- If Block 0 has exited, peers must not access its SMEM (lifetime
  rule).

**Key reading**: DSMEM is **distributed sharing**, NOT a physical
merge. Each block still owns its piece. Treating "DSMEM" as
"all blocks share one big un-owned shared memory" is wrong.

### Performance numbers (per microbenchmark studies)
- SM-to-SM network latency: **180 cycles** (~32% reduction vs L2).
- Peak DSMEM throughput: **3.27 TB/s** with cluster size 2 (drops to
  2.65 TB/s with cluster size 4 due to cross-cluster contention).
- Up to **7× reduction** in inter-block transfer overhead vs
  global-memory exchange (per NVIDIA's documentation).

### CUDA-side surface
```cpp
auto cluster = cooperative_groups::this_cluster();
cluster.sync();
// SMEM* in another block
unsigned* peer_smem = cluster.map_shared_rank(my_smem, /*rank=*/1);
```
PTX side: `mapa.shared::cluster` for the address mapping.

## TMA — US12141082B2

- **Filed** 2022-03-10, NVIDIA.
- **Title**: "Method and apparatus for efficient access to
  multidimensional data structures and/or other large data blocks."
- **Internal naming**: TMAU (Tensor Memory Access Unit). Externally:
  TMA.

### What TMA does
- A single thread submits a **descriptor**-driven bulk async copy of a
  tensor tile.
- Descriptor encodes dimensions, strides, bounds, layout (up to 5-D).
- Hardware then handles **address generation** and **data movement**
  end to end. The thread is free to continue.
- Single-threaded programming model: one thread launches the copy;
  synchronization (mbarrier) tracks completion for the consuming
  group.
- Supports **element-wise reductions** (add/min/max, bitwise and/or)
  on global writes.

### Why it matters vs `cp.async`
Ampere's `cp.async` already let global→shared be async — but the
threads still owned address generation, loop tiling, and copy
choreography. TMA pulls that choreography off thread cycles entirely
and gives the matrix mainloop a clear "tile arrived" event to
schedule against.

PTX side: `cp.async.bulk.tensor.{1d|2d|3d|4d|5d}.shared::cta.global`
with a `tensor_map` argument. Modern CUTLASS Hopper mainloops are
TMA-based.

## Transaction barrier — US20230289242A1

- **Filed** 2022-03-10, NVIDIA.
- **Title**: "Hardware accelerated synchronization with asynchronous
  transaction support."
- **Claim**: extends or replaces shared-memory-backed arrive-wait
  barriers with **transaction-aware** barriers that count both thread
  arrivals and async-transaction arrivals.

### Why ordinary barriers aren't enough
A traditional barrier answers "have all threads arrived?" It cannot
answer "has the data they were supposed to produce *actually
arrived*?" When the producer is a TMA copy, those are two different
events.

### How it works
- The barrier object accepts an **expected transaction count** in
  bytes.
- Producer (TMA copy) signals `mbarrier.complete_tx` for each
  completed transaction.
- Consumer waits on the barrier; release happens when **arrival count
  AND transaction-count expectation are both met**.

### Compute-capability surface
- Available on sm_90+ (Hopper).
- PTX: `mbarrier.arrive.expect_tx`, `mbarrier.complete_tx::bytes`.
- CUDA C++: `cuda::barrier` with `barrier_arrive_tx` /
  `barrier_expect_tx`.

## WGMMA — warp-group MMA

- PTX: `wgmma.mma_async.sync.aligned.{shape}.{layout}.{dtypes}`.
- All threads in a **warp group (4 warps × 32 threads = 128 threads)**
  must execute the same `.aligned` instruction.
- Operands: `D` is distributed across participating threads as
  registers; `A` and `B` come from SMEM (or registers in some forms).
- Issue is **collective**: all 4 warps must collaborate on issue.
- Completion via `wgmma.commit_group` + `wgmma.wait_group`.

This is the **consumer side** of the producer-consumer chain.

## The full Hopper main-thread chain

```text
global / L2 tensor tile
  -> TMA descriptor-driven bulk copy
  -> local SMEM or DSM resident tile
  -> transaction-aware barrier confirms data arrival
  -> scheduler releases the waiting consumer group
  -> warp-group / WGMMA consumes the prepared matrix
```

WGMMA-readiness condition compresses to:

```text
WGMMA-ready = resident tile + transaction completion + visibility + group release
```

All four must hold. Reading any single component as the headline
misses the architectural rewrite.

## Other Hopper features

- **DPX instructions**: hardware-accelerated dynamic-programming
  primitives (Smith-Waterman, etc.) — up to **13× speedup** on those
  workloads.
- **Transformer Engine**: hardware/software co-design that
  dynamically selects between FP8 and FP16 per layer to preserve
  training accuracy. The numerical-contract surface that binds FP8
  format selection, scale management, and matrix ops together.
- **Memory synchronization domains**: explicit scope reduction for
  fences/barriers — a fence can target a memory effect class /
  observer class instead of "the whole memory world."

## Two open uncertainties

1. **FP8 / Transformer Engine internal interface**: the corpus has no
   direct patent evidence pinning down where scale is stored, the
   rounding boundary, or the internal interface between Transformer
   Engine and the WGMMA datapath.
2. **`SMEM/DSM → WGMMA` last hop**: public material describes the
   contract among tile residency, barrier completion, and warp-group
   consumption, but cannot pin down the operand-collector or pipeline
   structure used.

Both are written-down limits in the source document. A future agent
should not invent details on these axes.

## Reference reading

- "NVIDIA TMA patent analysis" —
  https://zhuanlan.zhihu.com/p/1999510078827544585
- "NVIDIA Mbarrier patent analysis" —
  https://zhuanlan.zhihu.com/p/1996698248342152937

## Why a future agent cares

- "Cluster" / "DSMEM" / "TMA" / "WGMMA" / "transaction barrier" all
  require **sm_90 or higher**. Forward-port to Ada (sm_89) won't
  work. Forward-port to Blackwell (sm_100/sm_120) requires further
  attention because Blackwell **replaces** WGMMA semantics with
  `tcgen05.mma` (see `arch-blackwell.md`).
- The five-interfaces framing is the right model. Treating any one as
  isolated mis-reads what Hopper does.
- DSMEM is "distributed sharing," not a unified pool. Lifetime rules
  matter: you cannot read the SMEM of an exited block.
- Transaction barriers exist precisely to bridge "the threads
  arrived" and "the data arrived." They're not redundant with
  ordinary `bar.sync`.

## See also

- `three-rewrites.md` — Hopper as rewrite #3.
- `arch-blackwell.md` — Blackwell replaces WGMMA encoding with
  tcgen05 + TMEM.
- `arch-ampere.md` — `cp.async` is the primitive TMA generalizes.
- `arch-ada.md` — consumer Hopper-era part *without* clusters / DSMEM
  / TMA / WGMMA.
- `evolution-execution-model.md` — cluster as the latest layer of the
  warp/CTA hierarchy.
- `evolution-data-supply.md` — TMA and the staging chain.
- `evolution-synchronization.md` — transaction barrier in the sync
  timeline.
- `evolution-precision.md` — FP8 (E4M3/E5M2) entry.
- `.claude/knowledge/cuda/cuda-clusters-dsmem.md` — full CUDA
  programming-model surface for clusters.
- `patent-family-map.md` — full patent index.

## Escalate to raw when

- You need exact PTX `wgmma.mma_async` syntax — see
  `.claude/knowledge/ptx/ptx-isa-tensor-cores.md` and
  `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md`.
- You need exact PTX `cp.async.bulk.tensor` syntax — see
  `.claude/knowledge/ptx/ptx-isa-async-copy.md`.
- You need full PTX `mbarrier` semantics — see
  `.claude/knowledge/ptx/ptx-isa-barriers.md`.
- You need claim-level patent reading — Google Patents
  US20230289189A1 / US12141082B2 / US20230289242A1.
- You need GH100 vs H100 SXM5 vs PCIe deltas — NVIDIA's H100
  whitepaper is canonical.
