---
Use when: explaining Kepler-era SM organization (SMX, 192 cores); reasoning about why the data-supply chain has a "collector" stage near the operator entry; placing Dynamic Parallelism, Hyper-Q, and GPUDirect on the timeline.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.2 (Kepler is discussed as the post-Fermi continuation along the RF→crossbar→collector axis).
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/1.kepler.md (SM config); references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (memory hierarchy row).
Patents anchored here: builds on US7339592B2 (collector unit, Tesla-era); the data-supply line continues here without a single new defining patent in the corpus.
Out of scope: Maxwell's synchronization rewrite (`arch-maxwell.md`); FP16 (Pascal); Tensor Core (Volta).
Version family: Kepler architecture, 2012–2014. Representative products: GK104 (GTX 680, 2012-03), GK110/GK210 (Tesla K20/K80). Compute capability 3.x.
---

# Kepler (2012, GK110/GK210) — RF→crossbar→collector loop completed

Kepler is **not a breakpoint** — it is the post-Fermi continuation phase.
What it does is **close the data-supply loop on the SM side**: take
the collector concept that existed in Tesla-era patents and elevate it
to a clean `RF -> crossbar -> collector` chain near the datapath
entry.

## SM organization (GK210 — "SMX")

| Component | Count |
|---|---|
| Warp Scheduler | 4 |
| Dispatch Unit | 8 |
| INT32+FP32 Core | **192** (4 × 3 × 16, each lane is 16-wide) |
| Independent FP64 Unit | 64 |
| LD/ST Unit | 32 |
| SFU | 32 |
| Register file / SM | 64 KB |
| Shared memory / SM | 64 KB |
| L1 cache / SM | 64 KB |
| L2 cache (whole chip) | 1.5 MB |
| Read-only data cache | **48 KB** (Kepler-only feature) |
| Memory | GDDR5 |

**Notes:**

- Kepler removes the TPC/GPC layer of Fermi and just makes SMs huge,
  renamed **SMX**.
- 192 CUDA cores per SM is **6× Fermi's 32**. This is the high-water
  mark for cores per SM in this lineage; Maxwell will reduce it.
- **Independent 64-bit FP64 units** (no longer borrowing FP32 cores) —
  Kepler has the strongest double-precision performance of its era.
- **Warp Scheduler : Dispatch Unit ratio is 4:8 (1:2)**. Each scheduler
  can dispatch two independent instructions per cycle when ILP allows.
- Kepler's **48 KB read-only data cache** is a Kepler-unique feature
  for caching read-only inputs at higher bandwidth than the
  shared/L1 path.
- Per-cycle bandwidth: each SM can read 256 B/cycle; each LD/ST unit
  reads 4 B/cycle (per "Dissecting the NVIDIA Volta GPU Architecture
  via Microbenchmarking", arXiv:1804.06826).

## What Kepler completes (the data-supply chain)

The architectural axis Fermi pushed onto the **memory side**. Kepler
takes the **SM side** and closes the local data-supply loop:

```text
banked RF
  -> crossbar
  -> collector   (operand staging / reuse near operator entry)
  -> ALU
```

- `banked RF` reads source operands distributed across different
  register banks.
- `crossbar` routes to the appropriate collector.
- `collector` accumulates and reuses operands close to the datapath,
  so the operator entry doesn't have to re-read from RF every time.

**Important**: the collector is more like an **operand staging /
reuse layer** than a second register file. It buffers operands so they
can be replayed across pipeline stages or sub-computations without
hitting RF→crossbar bandwidth on every step.

This is exactly the structure Volta later specializes for matrix
multiply-accumulate. See `arch-volta.md` and `evolution-data-supply.md`.

## Kepler's other notable features

These are mainly programming-model and system-level features, not
architectural rewrites:

### Dynamic Parallelism + Grid Management Unit
- GPU itself can launch kernels onto itself, not just the CPU.
- Enables recursive algorithms, adaptive refinement.
- A new **Grid Management Unit** handles host vs device grid queues.

### Hyper-Q
- Multiple hardware task queues — increases from **1 (Fermi) to 32**.
- Each CUDA stream maps to a hardware queue, eliminating false
  dependencies between unrelated streams.
- This is what makes a Kepler GPU genuinely useful for multi-CPU-core
  hosts feeding work concurrently.

### GPUDirect — RDMA support
- Direct GPU↔GPU and GPU↔NIC transfers without round-tripping host
  memory.
- Foundation for the multi-GPU clustering software stack.

## Patents on the Kepler line

The corpus does not anchor Kepler to a single new patent. It builds on:

- **US7339592B2** — the collector-unit patent from the Tesla era; what
  Kepler does is elevate the collector to a stable per-SM hardware
  feature.
- **US9639479B2** (Fermi's L2/partition patent) — Kepler still uses
  Fermi-era memory-side semantics, no rewrite.

The next patent on the timeline that **does** matter for Kepler/Maxwell
is the Maxwell-era synchronization patent family — see `arch-maxwell.md`.

## Why a future agent cares

- If you see "192 cores per SM" in trace replay or analysis, you are
  on Kepler. Maxwell drops to 128, Pascal to 64.
- The Kepler 4:8 scheduler:dispatch ratio is unique. **Volta returns to
  1:1**. Don't infer Volta's scheduler discipline from Kepler.
- The 48 KB read-only data cache is a Kepler-only object. It does
  not exist on Maxwell or later in the same form (later generations
  unify under L1/texture/shared).
- Independent FP64 units make Kepler unusually strong at double
  precision — workloads tuned for Kepler FP64 will see degradation
  going forward (Maxwell removes them; later generations have FP64 at
  reduced ratios).

## See also

- `arch-fermi.md` — the prior generation and the memory-side rewrite.
- `arch-maxwell.md` — the next post-Fermi consolidation step,
  rewriting synchronization.
- `evolution-data-supply.md` — Tesla collector → Kepler RF/crossbar/
  collector loop → Volta HMMA datapath → Hopper TMA → Blackwell TMEM.
- `arch-tesla.md` — Tesla collector (US7339592B2) origin.
- `patent-family-map.md` — full patent index.

## Escalate to raw when

- You need exact GK104 vs GK110 vs GK210 deltas — gpu-arch-wiki only
  has the GK210 card; consult NVIDIA's Kepler whitepaper.
- You need register-file bank organization details — modern PTX/SASS
  dumps and academic microbenchmark papers (Mei + Chu, "Dissecting
  the NVIDIA Volta…", arXiv:1804.06826) are the canonical sources.
- You need Hyper-Q internals — NVIDIA's Kepler GK110 whitepaper
  describes the queue / WDDM interaction.
