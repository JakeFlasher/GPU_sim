---
Use when: reasoning about the baseline GPGPU machine — the assumptions every later generation rewrites; explaining where SIMT, off-chip crossbar, and per-SM shared memory came from; placing an early NVIDIA patent on the architectural timeline.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.1.
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (Tesla row of memory-hierarchy table; G80 has no dedicated SM card in the wiki because it predates Fermi's "complete GPU" claim).
Patents anchored here: US8539130B2 (virtual channels for crossbar, priority 2009-09-24), US7339592B2 (multi-port RF emulation via banks + collector, filed 2004-07-13), US7490208B1 (compact multi-ported register file with phase-multiplexed single-port RAM, filed 2004-10-05).
Out of scope: graphics-only Tesla features (texture sampling, ROP); Fermi's L2/partition rewrite (`arch-fermi.md`); any feature added after 2009.
Version family: Tesla architecture, 2006–2009. Representative product: G80 (GeForce 8800 GTX, 2006-11). Compute capability 1.x.
---

# Tesla (2006, G80) — the baseline machine

Tesla is the first NVIDIA architecture designed for general-purpose
compute. Everything later generations break away from is what Tesla
made the default.

## What Tesla fixed as the default

| Axis | Tesla default |
|---|---|
| Execution | `warp lockstep` advancement; one PC per warp |
| Data supply | `RF -> collector -> ALU` near-end chain |
| Synchronization | register dependencies, busy-bit / scoreboard ready, in-`SM` issue stall and restart |
| Cooperation | single `SM` is the boundary for hardware cooperation and shared state |
| Memory side | `L2 / partition / DRAM` exists but does not yet carry strong software-visible semantics |
| Programming model | SIMT + warp; per-`SM` shared memory + scoreboard + execution aggregate |

Every later breakpoint (Fermi, Volta, Hopper) breaks at least one of
these.

## SM organization (G80, indicative)

The Tesla generation predates the "modern" SM card layout. G80 had:

- 16 SMs, each with 8 streaming processors (SPs)
- 8 KB register file per SM (the smallest in the lineage)
- 16 KB shared memory per SM
- No L1 / L2 data cache — only texture caches and constant cache
- GDDR3 memory interface

Compare against Fermi (`arch-fermi.md`): Fermi 4× the registers, 3× the
shared memory, and adds L1+L2 for the first time.

## Patents anchored to this era

### US8539130B2 — virtual channels for crossbar
- **Priority** 2009-09-24, **filed** 2010-08-31, NVIDIA.
- **Claim**: a crossbar unit with multiple **virtual channels** as
  logical data flows; arbitration logic selects a VC per request based
  on source-client type and request type. Higher-priority traffic uses
  VCs designed to avoid deadlocks and head-of-line blocking; lower
  priority traffic uses VCs that may stall.
- **Why it matters**: at the latest by Tesla, NVIDIA had formalized
  off-core crossbar + virtual channels as an architectural problem.
  This is evidence that on-chip traffic organization was a foundational
  problem from the very first general-purpose GPU. Caveat: priority is
  2009 while G80 shipped in 2006, so the patent is not direct evidence
  that G80 itself implemented this exact form.

### US7339592B2 — multi-port RF emulation
- **Filed** 2004-07-13 (priority same), NVIDIA.
- **Claim**: simulate a multi-ported memory using lower-port-count
  memories as banks. A **collector unit** per bank gathers source
  operands as they emerge and forwards them to an execution unit when
  all operands for one instruction are present.
- **Why it matters**: introduces the **collector** abstraction that
  later generations elevate (Kepler routes via crossbar to collectors;
  Volta makes collectors the staging layer for `HMMA datapath`).

### US7490208B1 — compact multi-ported register file
- **Filed** 2004-10-05, NVIDIA.
- **Claim**: a register file using single-port RAM with phase
  multiplexing — N read + M write operations are serialized across
  (N+M) clock phases generated from a clock. Achieves ≥50% area
  reduction vs a true multi-ported memory.
- **Why it matters**: an alternative implementation strategy for the
  multi-ported RF problem. Together with US7339592B2 it shows two
  early answers to the same pressure: bank+collector vs phase-multiplexed
  single-port RAM. Both were available; later generations converge on
  banked-RF + collector style.

## On-chip traffic organization (the core insight)

```text
+---------------------------------------------------------------------------+
| request sources -> on-chip fabric -> memory-side destinations             |
| SM0   --\                                                                 |
| SM1   ---+--> [ crossbar / fabric ] --> [ partition0 ] -> [L2_0] -> DRAM0 |
| GPC_A ---+--> [ crossbar / fabric ] --> [ partition1 ] -> [L2_1] -> DRAM1 |
| GPC_B --/               |                                                 |
| busy destination <------+----- backpressure / queue state upstream        |
+---------------------------------------------------------------------------+
```

Multiple SMs / GPCs concurrently contend for shared memory-side
resources. Without traffic classification (virtual channels), one
class of traffic can stall everything via head-of-line blocking on a
shared FIFO.

The "virtual channels" abstraction at this layer is:

```text
+------------------------------------------------------------+
| one physical crossbar/link, multiple logical channels      |
| VC-compute  --\                                            |
| VC-memory   ---+--> [ arbitration / shared switch ] -> out |
| VC-graphics --/                                            |
+------------------------------------------------------------+
```

Same physical wires, multiple logical queues, traffic-class arbitration.

## Why a future agent cares

- A Tesla-era code base or trace-replay reasoning question will hit
  the **single PC per warp** default. Anything written assuming Volta's
  per-thread PC will break against a Tesla baseline.
- The "memory side has no strong software-visible semantics yet" is the
  pressure that Fermi resolves. If an analysis suggests Fermi-style
  cache modifiers are available pre-Fermi, it is wrong.
- The collector unit in US7339592B2 is the same conceptual object that
  Volta's `HMMA datapath` patent (US10338919B2) elevates — when reading
  Volta material that mentions "operand collectors," recognize the
  Tesla-era origin.

## See also

- `arch-fermi.md` — Tesla's memory-side limitations get rewritten.
- `evolution-execution-model.md` — warp lockstep starts here, ends at
  Volta + Hopper cluster.
- `evolution-data-supply.md` — RF/collector here; Kepler completes the
  loop; Volta specializes it.
- `patent-family-map.md` — full patent index.

## Escalate to raw when

- You need precise G80 SP / SFU / scheduling counts — `gpu-arch-wiki/`
  does not have a Tesla SM card; consult NVIDIA's G80 whitepaper or
  the David Kanter Tesla architecture analysis.
- You need claim-level reading of US7339592B2 / US7490208B1 — open the
  patents on Google Patents; the patent abstract here is condensed.
- You need to verify whether G80 actually shipped virtual channels (the
  patent priority is later than the launch) — check independent
  microarchitecture analyses.
