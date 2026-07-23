---
Use when: defining or resolving NVIDIA GPU terminology / abbreviations / jargon (active_mask, BSSY, BSYNC, collector unit, convergence barrier, CTA, CGA, DSM, DSMEM, HMMA, OMMA, QMMA, mbarrier, NVFP4, NV-HBI, RF cache, scoreboard, SubCore, TMA, TMAU, TMEM, transaction barrier, WGMMA, etc.); needing a one-line definition before reading a deeper card; confirming whether two terms refer to the same object across generations.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md (terms scattered through the document); per-arch cards.
Supporting sources: INDEX.md, the per-arch cards, the evolution-* cards.
Patents anchored here: (router card; full patent index is in `patent-family-map.md`).
Out of scope: full conceptual treatment of any term — that lives in the relevant card.
Version family: corpus snapshot 2026-05-02.
---

# Glossary — one-line definitions for corpus terms

A future agent encountering an unfamiliar term should be able to land
here, get a one-line definition, and find the right card to deep-dive.

Alphabetical.

## Architectural concepts

| Term | One-line definition | Primary card |
|---|---|---|
| `active mask` | Pre-Volta hardware bitmask of which threads in a warp are at the warp's PC; replaced by per-thread PC + convergence barriers in Volta. | `evolution-execution-model.md` |
| `BSSY / BSYNC` | SASS instructions for setting / waiting on a convergence barrier. Compiler-inserted; not visible in CUDA C++ or PTX. Volta+. | `arch-maxwell.md`, `sass-evolution-timeline.md` |
| `busy-bit` | Per-register hardware flag: write-pending. Pre-Maxwell synchronization primitive. | `evolution-synchronization.md` |
| `Cluster` (Thread Block Cluster) | Hopper-introduced grouping of thread blocks co-scheduled on SMs in a single GPC; supports DSMEM and `cluster.sync()`. | `arch-hopper.md` |
| `collector` (operand collector) | Per-bank or per-datapath unit that gathers source operands across cycles and forwards to an execution unit when complete. Origins: Tesla US7339592B2. | `arch-tesla.md`, `evolution-data-supply.md` |
| `convergence barrier` | Volta-default mechanism for joining divergent warp paths. Decoupled from scheduling; YIELD provides forward-progress guarantee. Patent: US10067768B2. | `arch-maxwell.md`, `arch-volta.md` |
| `cp.async` | Ampere-introduced async global→shared copy primitive. Bypasses register stage. SASS form: `LDGSTS`. | `arch-ampere.md`, `evolution-data-supply.md` |
| `cp.async.bulk.tensor` | Hopper-introduced bulk descriptor-driven async copy of a tensor tile (1-D up to 5-D). The PTX surface for TMA. | `arch-hopper.md` |
| `crossbar` (off-core) | On-chip switching fabric routing requests between SMs/GPCs and partitions/L2/memory interface. Tesla US8539130B2 formalizes virtual channels on it. | `arch-tesla.md` |
| `CTA` (Cooperative Thread Array) | PTX/hardware name for a thread block. | (every CUDA card) |
| `CTA-pair` | Blackwell `tcgen05`-introduced concept: even/odd CTA pairing for some matrix operations. | `arch-blackwell.md` |
| `DEPBAR` | SASS dependency barrier; software-visible scoreboard wait. Maxwell US9612836B2. | `arch-maxwell.md`, `sass-evolution-timeline.md` |
| `DSMEM` (Distributed Shared Memory) | Hopper-introduced address space mapping all per-block shared-memory partitions in a cluster into one accessible space. NOT a unified pool. Patent: US20230289189A1. | `arch-hopper.md` |
| `FB DRAM interface` | Frame-buffer interface from a memory partition out to external DRAM. Fermi US9639479B2 fixes it as a hardware boundary. | `arch-fermi.md` |
| `Fermi rewrite` | Architectural rewrite #1: memory side becomes a tunable hardware domain. Patent: US9639479B2. | `three-rewrites.md`, `arch-fermi.md` |
| `GPC` (Graphics Processing Cluster) | Mid-level grouping above SM in chip layout. Renamed from Tesla's TPC. Cluster (Hopper+) co-schedules within a single GPC. | `arch-fermi.md` |
| `head-of-line blocking` | Queue stall pattern: head-of-queue request blocks the queue even though items behind it could route to free destinations. Solved by virtual channels at crossbar. | `arch-tesla.md` |
| `HMMA` | Half-precision Matrix Multiply-Accumulate (Volta-introduced Tensor Core SASS instruction). | `arch-volta.md` |
| `HMMA datapath` | Volta-dedicated matrix path: `RF → crossbar → operand collectors → HMMA datapath → result queue → RF writeback`. Patent: US10338919B2. | `arch-volta.md`, `evolution-data-supply.md` |
| `Hopper rewrite` | Architectural rewrite #3: cooperation boundary crosses the SM. Patent: US20230289189A1. | `three-rewrites.md`, `arch-hopper.md` |
| `Independent Thread Scheduling` (ITS) | Volta-default execution model: per-thread PC, divergence/reconvergence via barriers, `_sync` family on the user surface. | `arch-volta.md`, `evolution-execution-model.md` |
| `L2 slice` | Per-address-range cache slice within a partition. Fermi US9639479B2 fixes it. | `arch-fermi.md` |
| `LDGSTS` | SASS form of `cp.async` (Ampere). | `sass-evolution-timeline.md` |
| `LDT / LDTM` | Blackwell SASS: load matrix from Tensor Memory to register file. | `arch-blackwell.md`, `sass-evolution-timeline.md` |
| `mbarrier` | Software-managed shared-memory barrier object (Ampere+). Hopper extends it with transaction-count tracking. | `arch-ampere.md`, `arch-hopper.md` |
| `mbarrier.complete_tx` | Hopper PTX instruction: signal completion of a byte-counted async transaction (e.g., a TMA copy). | `arch-hopper.md`, `evolution-synchronization.md` |
| `Memory synchronization domain` | Hopper-introduced fence/barrier scope-reduction mechanism: target a class of memory effects/observers instead of the whole memory world. | `arch-hopper.md`, `evolution-memory-side.md` |
| `MIG` (Multi-Instance GPU) | A100-introduced hardware GPU partitioning into up to 7 isolated GPU instances. Datacenter-only. | `arch-ampere.md` |
| `MMA` | Matrix Multiply-Accumulate. Generic name; hardware-specific names are HMMA / IMMA / DMMA / etc. | `arch-volta.md` |
| `NV-HBI` (NVIDIA High-Bandwidth Interconnect) | Blackwell B200 in-package die-to-die fabric, 10 TB/s. | `arch-blackwell.md`, `evolution-interconnect.md` |
| `NVL72` | Blackwell rack-scale system: 72 GPUs in one NVLink fabric. | `arch-blackwell.md`, `evolution-interconnect.md` |
| `NVLink` | NVIDIA proprietary high-bandwidth GPU↔GPU interconnect. Generations 1 (Pascal) through 5 (Blackwell). | `evolution-interconnect.md` |
| `NVSwitch` | Node-level switch fabric for NVLink. Ampere first. | `arch-ampere.md`, `evolution-interconnect.md` |
| `NVFP4` | NVIDIA's specific FP4 recipe: micro-block scaling + E4M3 scale factor + Hadamard reshape + 2D block quantization + stochastic rounding. Blackwell. | `arch-blackwell.md`, `evolution-precision.md` |
| `OMMA` | Blackwell SASS: FP4 Matrix Multiply-Accumulate across a warp. | `arch-blackwell.md`, `sass-evolution-timeline.md` |
| `Operand collector` | See `collector`. | `evolution-data-supply.md` |
| `partition unit` | Memory-side processing domain at L2/partition. Fermi US9639479B2. | `arch-fermi.md` |
| `PB` (Processing Block) | Within an SM, a sub-partition with its own warp scheduler, instruction buffer, and register file. 4 PBs per SM since Maxwell (with 2 PBs in Pascal as exception). | `arch-maxwell.md` |
| `QMMA` | Blackwell SASS: FP8 Matrix Multiply-Accumulate across a warp. | `arch-blackwell.md`, `sass-evolution-timeline.md` |
| `Reticle limit` | Maximum die size achievable in a single photolithography exposure. Blackwell B200 = two reticle-limited dies in one package. | `arch-blackwell.md` |
| `result queue` | Volta `HMMA datapath`-introduced buffer for arbitrating writeback to RF. | `arch-volta.md` |
| `Rewrite #1, #2, #3` | The three architectural rewrites: Fermi (memory side), Volta (matrix path + ITS), Hopper (cluster). | `three-rewrites.md` |
| `Scoreboard` | Hardware tracker for pending writes / completed instructions used by issue-ready logic. Pre-Maxwell: hardware-implicit; Maxwell+: software-controllable. | `evolution-synchronization.md` |
| `SHARP` | Scalable Hierarchical Aggregation and Reduction Protocol. Collective reductions executed in the NVSwitch fabric, not at endpoints. Hopper+. | `arch-hopper.md`, `evolution-interconnect.md` |
| `Shard scheduling` | Patent-side inference (Ampere-era) of finer-grained warp-internal scheduling post-divergence. NOT a public main feature. | `arch-ampere.md` |
| `SIMT` | Single Instruction, Multiple Threads. NVIDIA's execution model since Tesla. | `arch-tesla.md` |
| `SM` | Streaming Multiprocessor. The unit that contains warp schedulers, register files, shared memory, ALUs/Tensor Cores. | (every architecture card) |
| `SMM / SMX` | Maxwell's / Kepler's name for the SM. (Earlier and later generations just call it SM.) | `arch-kepler.md`, `arch-maxwell.md` |
| `SSY / SYNC` | Pre-Volta reconvergence SASS instructions. Retired in Volta. | `sass-evolution-timeline.md` |
| `STT / STTM` | Blackwell SASS: store matrix from RF to Tensor Memory. | `arch-blackwell.md` |
| `tcgen05.mma` | Blackwell PTX 5th-generation Tensor Core matrix instruction. Single-thread issue semantics; operands TMEM-resident or shared-mem descriptor. | `arch-blackwell.md`, `evolution-execution-model.md` |
| `Tensor Core` | Dedicated matrix-multiply-accumulate hardware unit. 1st gen: Volta (FP16+FP32). 5th gen: Blackwell (FP4/MX). | `evolution-precision.md` |
| `TF32` | TensorFloat-32 format: 8-bit exponent + 10-bit mantissa. NOT IEEE; range of FP32, precision of FP16. Default Tensor Core type for FP32 inputs on A100+. | `arch-ampere.md`, `evolution-precision.md` |
| `Three rewrites` | The corpus's central thesis. Fermi (memory side), Volta (matrix datapath + ITS), Hopper (cooperation boundary). | `three-rewrites.md` |
| `TMA` (Tensor Memory Accelerator) | Hopper-introduced descriptor-driven bulk async copy unit. PTX: `cp.async.bulk.tensor`. Internal name: TMAU. Patent: US12141082B2. | `arch-hopper.md`, `evolution-data-supply.md` |
| `TMEM` (Tensor Memory) | Blackwell-introduced on-SM dedicated tensor storage; ISA-visible via `LDT/LDTM` / `STT/STTM` / `tcgen05.mma`. | `arch-blackwell.md`, `evolution-data-supply.md` |
| `TPC` (Texture Processing Cluster) | Pre-Fermi grouping above SP. Renamed to GPC in Fermi; reappears in modern layouts. | `arch-tesla.md` |
| `Transaction barrier` | Hopper `mbarrier` extension that tracks both thread arrivals and async-transaction byte counts. Patent: US20230289242A1. | `arch-hopper.md`, `evolution-synchronization.md` |
| `Transformer Engine` | Hopper hardware/software co-design dynamically choosing FP8 (E4M3 / E5M2) vs FP16 per layer for training accuracy. Gen2 in Blackwell. | `arch-hopper.md` |
| `URF / UDP` | Uniform register file / uniform datapath. Stores warp-uniform values, broadcast-readable. ISA-visible from Turing+. | `arch-turing.md` |
| `Virtual channels` | Logical traffic-class queues sharing one physical crossbar. Tesla US8539130B2. | `arch-tesla.md`, `evolution-interconnect.md` |
| `Volta rewrite` | Architectural rewrite #2: matrix path peels out of the FPU + ITS becomes default execution model. Patent: US10338919B2. | `three-rewrites.md`, `arch-volta.md` |
| `WARPGROUP` | Hopper SASS: warpgroup synchronization. Removed in Blackwell (encoding replaced by `tcgen05`). | `sass-evolution-timeline.md` |
| `Warp group` | 4 warps × 32 threads = 128 threads cooperating on a `wgmma.mma_async` (Hopper). | `arch-hopper.md` |
| `Warp lockstep` | Pre-Volta default: all active threads in a warp share one PC. Replaced by ITS in Volta. | `arch-tesla.md`, `evolution-execution-model.md` |
| `WGMMA` | Hopper warp-group matrix multiply-accumulate. PTX: `wgmma.mma_async`. Replaced by `tcgen05.mma` in Blackwell. | `arch-hopper.md`, `arch-blackwell.md` |
| `YIELD` | Forward-progress mechanism in convergence-barrier patent (US10067768B2). Guarantees no thread can indefinitely block another. Required for ITS correctness. | `arch-maxwell.md` |

## Cache modifiers (PTX-level, all introduced in Fermi US9639479B2)

| Modifier | Semantics |
|---|---|
| `.cg` | Cache at global / L2 level; do not preferentially occupy each SM's L1. |
| `.cs` | Streaming — even if cached, prefer eviction. |
| `.cv` | For system-memory-related accesses, prefer refetch over reusing possibly-stale lines. |
| `.wt` | Write-through-style store; push to memory side / system memory. |
| `.lu` | Last-use hint (PTX 8.1+). Performance hint, not a consistency primitive. |

(Not consistency primitives. Performance hints only. See
`arch-fermi.md` for the caveat.)

## Format names

| Format | Bits | Layout | Min CC for Tensor Core | Notes |
|---|---|---|---|---|
| FP64 | 64 | IEEE 754 | 8.0 (Tensor Core) | FPU since always |
| FP32 | 32 | IEEE 754 | — | FPU since always |
| TF32 | 19 (in 32-bit storage) | 8e + 10m + 1s | 8.0 | NOT IEEE; FP32 range, FP16 precision |
| BF16 | 16 | 8e + 7m + 1s | 8.0 | Drop-in for FP32 in many ML settings |
| FP16 | 16 | IEEE 754 | 7.0 | Tensor Core: Volta+; FPU: Pascal+ |
| FP8 E4M3 | 8 | 4e + 3m + 1s | 8.9 / 9.0 | Activations / weights |
| FP8 E5M2 | 8 | 5e + 2m + 1s | 8.9 / 9.0 | Gradients (more range) |
| FP6 E3M2 | 6 | 3e + 2m + 1s | 10.0 / 12.0 | Blackwell |
| FP6 E2M3 | 6 | 2e + 3m + 1s | 10.0 / 12.0 | Blackwell |
| FP4 E2M1 | 4 | 2e + 1m + 1s | 10.0 / 12.0 | Blackwell base FP4 |
| NVFP4 | 4 + scale | E2M1 + E4M3 scale + Hadamard reshape + 2D block quant + stochastic rounding | 12.0 (sm_120a) | NVIDIA FP4 recipe |
| MX (microscaling) | varies | Block of N values + shared scale | 12.0 (sm_120a) | OCP MX spec |
| INT8 | 8 | Signed/unsigned int | 7.5 | Inference |
| INT4 | 4 | Signed/unsigned int | 7.5 | Inference |
| Binary | 1 | Bit (`xor.popc` reduction) | 7.5 | Research / extreme quantization |

## Generation acronyms

| Generation | Code | Year | Compute Capability |
|---|---|---|---|
| Tesla | G80, GT200 | 2006–2009 | 1.x |
| Fermi | GF100, GF110 | 2010 | 2.x |
| Kepler | GK104, GK110, GK210 | 2012–2014 | 3.x |
| Maxwell | GM107, GM200, GM204 | 2014–2016 | 5.x |
| Pascal | GP100, GP102, GP104 | 2016 | 6.x |
| Volta | GV100 | 2017 | 7.0 |
| Turing | TU102, TU104, TU106 | 2018 | 7.5 |
| Ampere | GA100, GA102, GA104 | 2020 | 8.0, 8.6, 8.7 |
| Ada Lovelace | AD102, AD103, AD104 | 2022 | 8.9 |
| Hopper | GH100, GH200 | 2022 | 9.0 |
| Blackwell datacenter | GB200, B200 | 2024 | 10.0 |
| Blackwell consumer | GB202, GB203, GB205, GB206, GB207 | 2024–2025 | 12.0 |

## See also

- `INDEX.md` — corpus router.
- `problem-playbook.md` — question-driven router.
- `patent-family-map.md` — every patent in the corpus.
