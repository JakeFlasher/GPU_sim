---
Use when: explaining how the GPU interconnect grows from on-chip crossbar to rack-scale fabric; placing NVLink generations (1.0 through 5.0), NVSwitch, NVL72, NV-HBI die-to-die fabric; reasoning about GPU-to-GPU bandwidth, latency, topology, partition mapping, multi-GPU communication overhead, all-reduce / collective primitives; understanding why interconnect is "part of the cooperation model," not just transport.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §3.5 (interconnect evolution) and §2.5 (Blackwell die-to-die + NVLink 5).
Supporting sources: per-arch cards (Tesla, Pascal, Hopper, Blackwell); references/NVIDIA_Manuals/blackwell_tuning/01_nvidia-blackwell-tuning-guide.md §1.4.3.
Patents anchored here: US8539130B2 (virtual channels for crossbar, Tesla).
Out of scope: NVLink PHY-level details (proprietary, not in this corpus); per-link bandwidth tables (see NVIDIA whitepapers).
Version family: Tesla through Blackwell. Mirror snapshot: 2026-05.
---

# Interconnect — from on-chip `crossbar` to cross-`GPU` and cross-package cooperation

The GPU interconnect grows along two axes simultaneously: **bandwidth**
(more bits per second) and **semantics** (interconnect carries a
larger share of the cooperation contract).

This card walks the timeline along both axes.

## The original problem (Tesla)

Early interconnect is a routing problem: how do `SM`-side requests
reach `partition / memory interface`? Easy to dismiss as background
plumbing — but the very first NVIDIA GPU patent on this is
**US8539130B2** (priority 2009-09-24), formalizing **virtual channels
for crossbar**.

The pressure that made on-chip traffic organization a problem from
day one:

- Multiple `SM/GPC`s concurrently contend for shared `memory-side`
  resources.
- Compute, graphics, and memory traffic share the switching plane.
- Without traffic classification, head-of-line blocking on a shared
  FIFO stalls everything.

For details: `arch-tesla.md`.

## On-chip crossbar — the foundational object

The on-chip crossbar (or fabric) is the layer where:

```text
request sources         on-chip fabric              memory-side destinations
SM0   --\                                                                 
SM1   ---+--> [ crossbar / fabric ] --> [ partition0 ] -> [L2_0] -> DRAM0
GPC_A ---+--> [ crossbar / fabric ] --> [ partition1 ] -> [L2_1] -> DRAM1
GPC_B --/               |                                                 
busy destination <------+----- backpressure / queue state upstream        
```

Virtual channels split logical traffic on the same physical wires:

```text
+------------------------------------------------------------+
| one physical crossbar/link, multiple logical channels      |
| VC-compute  --\                                            |
| VC-memory   ---+--> [ arbitration / shared switch ] -> out |
| VC-graphics --/                                            |
+------------------------------------------------------------+
```

This same VC abstraction recurs at every later interconnect layer
(NVLink, NVSwitch, NV-HBI) — different traffic classes get logically
isolated channels even if they share physical links.

## Pascal — NVLink 1.0 (2016)

Bandwidth: ~160 GB/s bidirectional between two GP100 GPUs.

The first proprietary NVIDIA high-bandwidth GPU↔GPU interconnect.
Replaces PCIe as the primary data path for multi-GPU compute.

**Semantic shift**: interconnect is no longer just "route the bits."
It now carries unified-memory page migration traffic, and the
multi-GPU coherence story starts depending on it.

For details: `arch-pascal.md`.

## Volta — NVLink 2.0 (2017)

Higher bandwidth, more links per GPU. Supports cache coherence
between GPUs (with limitations).

## Ampere — NVLink 3.0 (2020)

A100: 600 GB/s aggregate. Pairs with first-gen NVSwitch to enable
all-to-all 8/16-GPU systems.

NVSwitch is the first node-level switch fabric for NVLink — turns
GPU↔GPU into a switched topology rather than a fixed point-to-point
mesh.

## Hopper — NVLink 4.0 (2022)

H100: 900 GB/s aggregate. Second-gen NVSwitch.

Critically, this is when interconnect starts carrying **cluster-scope
cooperation traffic** intra-node — though clusters live within a
single GPU's GPC, the cluster-level data-transport patterns motivate
a corresponding NVLink upgrade for inter-GPU collectives.

NVLink + SHARP (Scalable Hierarchical Aggregation and Reduction
Protocol) — collective reductions are computed in the switch fabric
itself, not at endpoints. Reduces all-reduce time materially.

## Blackwell — NVLink 5.0 + NV-HBI + NVL72 (2024)

This is the generation where interconnect crosses **multiple new
physical scopes** simultaneously.

### NV-HBI — die-to-die in-package
- B200 datacenter SKU is **two reticle-limited dies** with **10 TB/s
  chip-to-chip interconnect**.
- Software still sees one CUDA GPU.
- New layer in the hierarchy: **die-to-die in-package** is now an
  intermediate scope between intra-die (crossbar / GPC mesh) and
  inter-GPU (NVLink).

### NVLink 5.0
- Per-GPU bidirectional bandwidth ~1.8 TB/s.
- Continues the bandwidth-doubling cadence.

### NVLink Switch + NVL72
- Rack-scale system: **72 GPUs in one NVLink fabric**.
- All-reduce with SHARP becomes more efficient at scale.
- The cooperation boundary now scales to a rack.

For details: `arch-blackwell.md`.

## Two-axis growth

```
                |  intra-GPU      | inter-GPU         | rack-scale
  Tesla         |  crossbar (VC)  | -                 | -
  Pascal        |  crossbar       | NVLink 1          | -
  Volta         |  crossbar       | NVLink 2          | -
  Ampere        |  crossbar       | NVLink 3 + NVSwitch | -
  Hopper        |  crossbar+cluster (DSMEM, sm-to-sm) | NVLink 4 + NVSwitch + SHARP | -
  Blackwell     |  crossbar+cluster + NV-HBI (die-to-die) | NVLink 5 | NVL72 (72 GPUs)
```

What changes per generation:

- Tesla–Volta: intra-GPU stays the same VC-style crossbar; inter-GPU
  scales bandwidth.
- Hopper: intra-GPU adds a new **cluster-scope** semantic layer (the
  SM-to-SM fabric backing DSMEM is essentially intra-GPC NVLink-style
  routing).
- Blackwell: intra-GPU **adds** a die-to-die layer (NV-HBI); rack
  becomes a real cooperation scope (NVL72).

## Interconnect as part of the cooperation model

The architectural insight from §3.5: **after Pascal, interconnect no
longer just answers whether bandwidth is sufficient.** It starts
answering:

- **How far does the cooperation boundary extend?** (single-SM →
  cluster → die-pair → rack)
- **How are coherence and visibility guaranteed across that scope?**
- **What new sync primitives correspond to that scope?** (cluster
  barriers; NVLink+SHARP collective primitives.)

So interconnect goes from "transport layer" to "part of the
cooperation model." This is the same trajectory as memory side
(Fermi → Hopper) and execution model (Volta → Hopper) — increasingly
**software-visible semantic** at scope boundaries that used to be
purely hardware-internal.

## Why a future agent cares

- "More NVLink bandwidth" alone doesn't capture what's new in a
  generation. SHARP (Hopper) and rack-scale NVL72 (Blackwell) change
  the **semantics**, not just the rate.
- A B200 is **one CUDA GPU** even though it's two physical dies.
  Software cannot opt to see two; the unification is below the CUDA
  programming model.
- A Blackwell-rack workload uses NVLink 5 + NVL72 + SHARP + NV-HBI +
  cluster + DSM together — the cooperation contract is layered. Don't
  treat any single layer as the whole story.
- DSMEM (intra-cluster) and NVLink (inter-GPU) are different scopes;
  the cooperation primitives at each scope differ.

## See also

- `arch-tesla.md` — virtual-channels patent.
- `arch-pascal.md` — NVLink 1.0 entry.
- `arch-hopper.md` — cluster as intra-GPU "interconnect" layer.
- `arch-blackwell.md` — NV-HBI + NVL72.
- `evolution-execution-model.md` — cooperation boundary expanding.
- `evolution-memory-side.md` — memory-side semantics descending in
  parallel.
- `patent-family-map.md` — full patent index.

## Escalate to raw when

- You need PHY-level NVLink details — proprietary; partial coverage in
  NVIDIA's NVLink whitepaper.
- You need NVSwitch internals — NVIDIA's NVSwitch whitepaper or
  Hot Chips talks.
- You need SHARP collective semantics — Mellanox/NVIDIA SHARP
  documentation.
- You need NV-HBI bandwidth / latency numbers — NVIDIA's Blackwell
  whitepaper; not in this corpus.
- You need exact NVL72 topology / bisection bandwidth — NVIDIA's
  GB200 NVL72 system documentation.
