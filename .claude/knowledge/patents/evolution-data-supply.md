---
Use when: explaining how data reaches the operator entry generation by generation; reasoning about why operand collectors exist; placing `cp.async`, TMA, and TMEM on one timeline; understanding the `RF → crossbar → collector → datapath` chain.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §3.2.
Supporting sources: per-arch cards (Tesla, Kepler, Volta, Ampere, Hopper, Blackwell); references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md (cp.async, cp.async.bulk.tensor, tcgen05).
Patents anchored here: US7339592B2 (collector unit), US10338919B2 (HMMA datapath), US12141082B2 (TMA).
Out of scope: precision (`evolution-precision.md`); execution model (`evolution-execution-model.md`).
Version family: Tesla through Blackwell. Mirror snapshot: 2026-05.
---

# Data Supply — from `RF/collector` to `cp.async / TMA`-driven staging to TMEM

While the execution-progress unit grows outward (warp → cluster), the
data-supply chain grows in a different direction: **closer to the
operator entry**, with **larger asynchronous transactions**.

## The original problem

As long as the execution core is just a general `ALU/FPU`, the
near-end data-supply chain only has to answer "how do we get register
values to the execution port in time?" Once a dedicated matrix
datapath exists, that chain immediately comes under pressure — matrix
ops want **block-shaped**, **highly reused**, **well-staged** data, and
classical RF ports can't deliver.

## Timeline

| Generation | Data-supply primitive | What's new |
|---|---|---|
| Tesla | `RF → collector → ALU` | Collector unit (US7339592B2) gathers source operands across cycles |
| Fermi | Same | (rewrite is on memory side, not data supply) |
| Kepler | `banked RF → crossbar → collector → ALU` | Loop completed; collector becomes a stable per-SM hardware feature |
| Maxwell | Same | (rewrite is on synchronization) |
| Pascal | Same + FP16 in CUDA cores | Format precedes datapath |
| **Volta** | `RF → crossbar → operand collectors → HMMA datapath → result queue → RF writeback` | US10338919B2 introduces dedicated matrix datapath |
| Turing | Same; INT8/INT4/Binary added | Precisions extend the matrix datapath |
| **Ampere** | Same + `cp.async` | First peeling-off of staging from the LSU |
| Hopper | Same + **TMA** + **DSMEM** | TMA descriptor-driven bulk copy; DSM extends shared-memory scope |
| Blackwell | Same + **TMEM** | Tensor Memory becomes an explicit ISA object |

## Each step in detail

### Step 1: Tesla collector (US7339592B2)

Multi-ported register files are expensive. Tesla solves it by
emulating multi-port behavior with a banked single-port store + a
**collector unit** that gathers source operands across cycles and
forwards them to an execution unit when complete.

```text
RF banks
  -> bank request arbitration
  -> collector unit (gathers operands across cycles)
  -> execution unit
```

The **collector** is the load-bearing object. Every later data-supply
generation either:

- Extends it (Kepler: add crossbar route)
- Specializes it for a new datapath (Volta: HMMA datapath)
- Adds asynchronous staging in front (Ampere: `cp.async`)
- Replaces it with explicit storage (Blackwell: TMEM)

### Step 2: Kepler — RF→crossbar→collector→ALU

Kepler completes the data-supply loop on the SM side: banked RF
reads operands distributed across banks, the crossbar routes them to
the right collector, the collector accumulates and reuses operands
near the operator entry. ALU consumes from collector output.

The collector is more like an **operand staging / reuse layer** than
a second register file — it buffers operands so they can be replayed
across pipeline stages without re-reading from RF.

### Step 3: Volta — HMMA datapath (US10338919B2)

```text
RF banks
  -> crossbar
  -> operand collectors
  -> HMMA datapath           <-- NEW dedicated matrix path
  -> result queue            <-- NEW write-arbitration buffer
  -> RF writeback
```

The matrix path:

- Adds a **dedicated `HMMA datapath`** between collectors and result.
- Adds a **result queue** that buffers writes; writeback is arbitrated.
- Lets one set of operands be reused across multiple dot-product
  passes without re-reading from RF/crossbar.

Operand collectors are the **vector staging layer** near the datapath
entry. The patent describes them as a structure that can preload
operands across multiple clock cycles and then feed the datapath in
parallel during one MMA execution.

`Fig. 9` of US10338919B2 shows HMMA datapath 930 sharing operand
collectors and result queue with FP64 datapath 940. So Volta
**specializes** the front+middle of the matrix dot-product but does
**not** rebuild data supply, completion, and writeback from scratch.

### Step 4: Ampere — `cp.async` peels staging off the LSU

```ptx
cp.async.cg.shared.global  [smem_dst], [global_src], 16;
cp.async.commit_group;
cp.async.wait_group  N;
```

Three changes:

- **Async**: load doesn't block the issuing thread.
- **Group-aware**: `commit_group`/`wait_group` track batches; one
  wait can release multiple copies.
- **Bypass registers**: data goes directly global → shared, never
  through a thread-private register.

This is the first time staging is **explicitly peeled off** the LSU
as its own protocol.

SASS-level: `LDGSTS` (asynchronous global-to-shared memcopy) +
`LDGDEPBAR` for fence/wait.

### Step 5: Hopper — TMA (US12141082B2)

Ampere's `cp.async` is per-thread (one address per thread). For tile
sizes typical in Hopper-era GEMMs, hundreds of threads spend cycles on
address-generation and loop choreography.

TMA replaces that with **descriptor-driven bulk async copy**:

- Descriptor encodes dimensions, strides, bounds, layout (up to 5-D).
- One thread submits the copy.
- Hardware handles **all address generation and data movement** end
  to end.
- Element-wise reductions (add/min/max, bitwise and/or) supported on
  global writes.

PTX: `cp.async.bulk.tensor.{1d|2d|3d|4d|5d}.shared::cta.global` with a
`tensor_map` argument. Modern CUTLASS Hopper mainloops are TMA-based.

Together with **DSMEM** (US20230289189A1), the supply chain extends to
**SM-to-SM** data sharing within a cluster:

```text
global / L2 tensor tile
  -> TMA descriptor-driven bulk copy
  -> local SMEM or DSM resident tile        <-- across cluster boundary
  -> (transaction-aware barrier confirms data arrival)
  -> warp-group / WGMMA consumes
```

See `arch-hopper.md` for the full Hopper main-thread chain.

### Step 6: Blackwell — Tensor Memory (TMEM)

The new layer: an **on-SM dedicated tensor storage** that's first-
class in the ISA.

| Object | What it is |
|---|---|
| `tmem[URX]` | Tensor Memory operand location |
| `LDT/LDTM` | Load matrix from TMEM into RF |
| `STT/STTM` | Store matrix from RF to TMEM |
| `UTCCP` | Async copy shared-memory → TMEM |
| `UTCSHIFT` | Shift elements within TMEM |

`tcgen05.mma` is the new MMA instruction:

- `D` lives in TMEM (always)
- `A` can live in TMEM or shared
- `B` lives in shared
- Operands as TMEM addresses or shared-memory descriptors

The data-supply problem has continued from "staging protocol" (Ampere)
to "dedicated near-Tensor-Core storage and movement objects" (Hopper +
Blackwell). From here on, how to get data to the operator entry is
**both** a protocol-level **and** a dedicated-storage-level problem.

## The unifying axis

What's growing across the timeline is **the data path's independence
from threads**:

- Tesla: threads explicitly coordinate operand fetch.
- Kepler: threads explicitly coordinate, but a stable RF→crossbar→
  collector loop reduces redundant reads.
- Volta: matrix path has its own stages; threads issue at instruction
  granularity.
- Ampere: staging has its own protocol; threads issue at copy-batch
  granularity (`cp.async`).
- Hopper: tile transport has its own hardware unit; **one thread**
  starts a tile copy.
- Blackwell: tile **storage** has its own hardware location; **one
  thread** starts a matrix op (single-thread issue).

The trend: **threads spend less and less of their cycles on data
movement**, and more on dispatching descriptors that the dedicated
machinery executes.

## Why a future agent cares

- A modern GEMM mainloop on Blackwell uses TMA → SMEM → TMEM →
  `tcgen05.mma`. **None** of those primitives exist pre-Hopper for
  the first three; pre-Blackwell for TMEM.
- "Threads loading data" is the bottleneck in Ampere-era code that
  modernized Hopper code rewrites with TMA.
- An RTX 5060 (SM120) inference kernel using TMEM is binary-
  incompatible with H100 (sm_90).
- The single-thread issue model (Blackwell) does not mean "one thread
  does the work" — it means one thread descriptors the operation,
  and the matrix machinery does the work. A future agent reading
  pseudocode with one thread "doing" `tcgen05.mma` should not infer
  serial execution.

## See also

- `arch-tesla.md` — collector origin.
- `arch-kepler.md` — RF/crossbar/collector loop completed.
- `arch-volta.md` — HMMA datapath.
- `arch-ampere.md` — `cp.async` peels off staging.
- `arch-hopper.md` — TMA + DSMEM.
- `arch-blackwell.md` — TMEM as an ISA object.
- `evolution-precision.md` — formats as parallel axis to data supply.
- `competitor-tensor-storage.md` — non-NVIDIA tensor-near storage.
- `.claude/knowledge/ptx/ptx-isa-async-copy.md` — current-PTX
  `cp.async.bulk.tensor` semantics.
- `patent-family-map.md` — full patent index.

## Escalate to raw when

- You need detailed operand-collector micro-architecture — academic
  microbenchmark papers ("CORF: Coalescing Operand Register File for
  GPUs," ASPLOS '19) are the canonical sources.
- You need exact TMA descriptor encoding — PTX
  09_instruction-set.md and CUTLASS TMA documentation.
- You need TMEM bank organization / port behavior — Blackwell
  whitepaper + microbenchmark studies; not in this corpus.
