# SM86 PhaseFuse: fast exact-region prototype and SOTA study

Updated 2026-09-27. This note records a bounded prototype, not a claim that
the four native hardware/trace gaps are solved. It intentionally does **not**
run full Accel-Sim, GPGPU-Sim, MGPUSim, Vortex, or other large external
workloads.

## Executive decision

The most credible path to a research-grade fast simulator is a **three-lane
contract**, not a single approximate timing number:

1. **Exact trace lane.** Decode the immutable NVBit/SASS trace once and retain
   exact instruction, predicate, address/sector, launch, pipeline-domain and
   NCU-shaped counter observations. These values remain independent of NCU
   targets.
2. **Sparse event lane.** Use typed next-event advancement for memory and
   inactive components. The current `sim-sm` event path already skips idle
   intervals while preserving response horizons, L2 ordering, DRAM service and
   same-time sequence order.
3. **PhaseFuse certificate lane.** Summarize only state-complete, homogeneous,
   control-clean, dependency-free scalar regions with no memory or external
   side effect. Every other region is explicitly reified/refused. The new
   `--fast-frontier-only` path exposes this lane without pretending it is native
   timing.

The research contribution is therefore a **fidelity contract with exact
fallback**, not “skip GPU cycles.” A successful certificate means equivalence
to the declared recurrence and boundary state; it does not mean equivalence to
undocumented SM86 scheduling or silicon.

## Primary-source study

Fresh shallow clones were inspected outside the repository at
`/tmp/gpu-simulator-sota-20260927`:

| Project | Revision | Relevant source evidence |
| --- | --- | --- |
| [Accel-Sim](https://github.com/accel-sim/accel-sim-framework) | `d930ad6` | SASS trace frontend plus PTX execution frontend; `trace_driven.cc` explicitly uses trace masks and disables a SIMT stack in trace mode. `gpgpu-sim4.md` models sector caches, write-validate/write-back behavior, DRAM queues and configurable execution units. |
| [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) | `a4ce3fe` | Detailed cycle loop, shader issue/writeback, interconnect, cache and DRAM state. It is a useful dense reference, not a fast-path dependency. |
| [MGPUSim](https://github.com/sarchlab/mgpusim) | `adf7c16` | Separate functional emulation and timing paths over Akita. The driver has explicit cache `FlushCommand`; control middleware drains, flushes and invalidates caches before completion. Current README says NVIDIA support is unfinished. |
| [Vortex](https://github.com/vortexgpgpu/vortex) | `ed4098d` | Open RISC-V SIMT stack with configurable warps, issue width and units; LLVM-based VOLT toolchain. SimX has IPDOM scheduling, barriers, sector caches, write-back LLC rules and an explicit multi-level `flush_caches()` walk. |
| [EventQueue](https://github.com/cucapra/EventQueue) | `d4b4dd5` | MLIR EQueue dialect separates structure/control from a generic timed event engine and compiler passes. It is a strong extensibility pattern, not NVIDIA SASS compatibility. |
| [PIMSIM-NN](https://github.com/wangxy-2000/pimsim-nn) | `3e3442b` | Compiler-generated instruction stream + JSON architecture/NoC configuration + event-driven SystemC simulator. It demonstrates an ISA/compiler/simulator contract for a new accelerator, not arbitrary GPU ISA generation. |
| [FlashGPU-Sim](https://github.com/FlashGPU-Sim/FlashGPU-Sim) | `d20ce45` | Execution-driven modern CUDA simulator with TMA, `mbarrier`, MMA/WGMMA and TritonTrace/NCU validation. Its public configurations target newer GPUs; it is prior art for feature coverage, not local SM86 validation. |

The corresponding papers were read from primary PDFs/arXiv:

- [Accel-Sim, ISCA 2020](https://engineering.purdue.edu/tgrogers/papers/khairy.isca2020.pdf)
  explicitly separates native mISA trace replay from vISA execution and notes
  that data-dependent control and global synchronization are difficult under a
  fixed trace.
- [MGPUSim, ISCA 2019](https://arxiv.org/abs/1811.02884) combines functional
  emulation, timing components and a discrete-event engine; its “no busy
  ticking” principle applies to closed component contracts, not arbitrary SM
  instruction regions.
- [Vortex](https://arxiv.org/abs/2110.10857) demonstrates why an open ISA plus
  compiler/runtime/RTL/simulator stack is the practical answer to arbitrary
  ISA generation; it is not a route to NVIDIA SASS compatibility.
- [Accel-Sim 2.0](https://arxiv.org/abs/2608.22602) adds Hopper/Blackwell
  asynchronous primitives, chiplet/NUMA modeling and fine-grained GPUVision
  validation. Its paper also documents the trace problem: mbarrier polling
  iteration counts are timing-dependent, and unsupported operations may be
  mapped or replaced rather than reconstructed from missing state.
- [Parallelizing a modern GPU simulator](https://arxiv.org/abs/2502.14691)
  reports deterministic OpenMP acceleration of Accel-Sim. Parallel local
  clocks are complementary to, not a substitute for, PhaseFuse certificates.
- [Compiler-Driven Simulation / EQueue](https://arxiv.org/abs/2202.00739)
  supplies the strongest design pattern for configurable structure and
  compiler lowering through an IR.
- [PIMSIM-NN](https://arxiv.org/abs/2402.18089) shows a concrete
  compiler-to-ISA-to-event-simulator workflow with JSON architecture/NoC
  configuration.
- [OpenVADL](https://openvadl.github.io/openvadl/) and LLVM TableGen document
  realistic generated-ISA stages (ISS, encoder/decoder, assembler and LLVM
  backend) while explicitly leaving a cycle-approximate simulator generator as
  future work. The local subset therefore stops at a functional executor and
  assembler-like lowering.

## What the study rules out

* **Writeback/coherence completion:** Accel-Sim/GPGPU-Sim and Vortex/MGPUSim
  contain explicit cache/flush or memory-system machinery, but their mechanisms
  do not transfer to the fixed SM86 trace contract. A post-replay service
  formula cannot recover dirty-line ordering, visibility scope, ownership, or
  profiler interval boundaries.
* **Dynamic physical-pipe selection:** configurable unit maps and scheduler
  models exist in Accel-Sim, MGPUSim and Vortex, but an SM86 trace record does
  not expose the physical unit selected at issue. The existing
  `min_selected_then_lexical_v1` policy remains a diagnostic count-balancer.
* **Value-dependent CFG regeneration:** PTX/Vortex/GCN execution frontends can
  compute new paths. Fixed NVBit/SASS replay cannot regenerate a path after a
  hypothetical timing or memory intervention without a functional executor or
  recapture.
* **Arbitrary ISA/compiler generation:** EQueue/PIMSIM/Vortex demonstrate
  bounded compiler/IR contracts. A JSON instruction allow-list is not a
  compiler, assembler, ABI, linker, SIMT memory model or native SASS generator.

## Implemented PhaseFuse contract

`sim-sm::fast_frontier_for_trace` now builds a conservative report directly
from a real compact trace and its admitted control schedule. It:

1. validates the trace architecture and traced placement;
2. finds contiguous scalar, non-memory, `OpClass::Other` regions;
3. joins each record to the SM86 schedule and control bits;
4. invokes the existing state-complete causal certificate with the complete
   block resident-warp roster;
5. reports candidate, certified, reified and collapsed populations plus refusal
   reasons; and
6. never changes native replay state or consumes NCU values.

The executable `sim-run --fast-frontier-only` performs only this bounded
trace-derived analysis. It emits `contract=sm86_fast_exact_scalar_regions_v1`,
empty native `launches`, and explicit fast-frontier launch reports. It is a
research/profiling lane, not a replacement for ordinary `flat`, `l1`, or
`l1-l2` replay. The envelope is described by
[`fast_frontier.v1.schema.json`](../sim/run/schema/fast_frontier.v1.schema.json).

The certificate deliberately refuses mixed timing, dependencies, barriers,
memory roles, non-scalar pipelines, incomplete residency and unsupported
control. Zero certified regions is a valid result: the fast path must not turn
an absence of proof into an approximation silently.

## Bounded evidence

No external simulator was built or run. The local release fast lane was run on
retained inputs only:

| Input | Fast-only wall time | Result |
| --- | ---: | --- |
| SM86 address-carry canary, 54 records | ~2 ms | 1 block / 1 warp; scalar candidates refused by control guards |
| September development GEMM, 197,120 instructions / 512 blocks | ~0.20 s | 23,040 scalar candidate instructions; 8,704 regions refused; no native timing replay |
| September development LayerNorm | ~0.30 s | 165,852 scalar candidate instructions; 53,292 regions refused; no native timing replay |

The existing event-vs-dense local comparison remains the timing baseline: the
retained GEMM event lane was about 1.7–1.8× faster than the dense reference with
matching modeled frontiers. That is a same-model engineering result, not a
comparison to Accel-Sim or silicon. The fast-only lane is faster because it does
not run native timing; it must not be advertised as timing accuracy.

## Research acceptance bar

Before claiming a publishable breakthrough, PhaseFuse needs:

* zero false certificates in adversarial control, residency, memory, barrier,
  async-copy and same-cycle-order tests;
* a producer that supplies complete boundary state at every candidate edge;
* planner-inclusive wall time and peak memory, not only collapsed event counts;
* same-domain independent NCU validation of deterministic counters and timing
  envelopes, with NCU never entering replay decisions;
* architecture-decision stability across cache, issue, occupancy and memory
  interventions; and
* explicit fallback rates. A high refusal rate is preferable to a false exact
  claim and identifies the next mechanism to model.

The four native gaps therefore remain open. PhaseFuse resolves a real,
executable fast-analysis seam and creates a path to close portions of those
gaps only when the missing state contracts become available.

## Bounded custom routing seam

The custom architecture contract now admits alternative resources per
operation. Its event scheduler evaluates modeled service completion, issue
interval and in-flight capacity, records the selected resource, and uses a
canonical resource-index tie break. The route example is executable and
reproducible, but it is intentionally not a claim that SM86 physical pipe IDs
can be recovered from NVBit traces.

## Bounded dirty writeback seam

`sim-mem::writeback::WritebackQueue` consumes the exact dirty/valid/pending
sector masks already exposed by `L2Eviction`. With explicit
`--dirty-writeback-capacity-sectors N`, native launch replay performs
transactional queue-capacity checks, shared fixed-point byte-server timing for
read fetches and dirty writebacks, same-line fetch-hazard tracking and explicit
token completions. The default carded path is unchanged. This is the smallest
executable mechanism that can be validated without inventing a coherence
protocol; ownership, visibility and resident dirty flush completion remain
open.

## Bounded coherence/visibility seam

The opt-in `CoherenceQueue` is a directory-style model over the ordered,
fully decoded L2-request stream. It gives read/read sharing no service cost and
charges a configured interval for cross-SM owner transfer or write
invalidation, with finite in-flight capacity and explicit completion events.
Unknown SM/sector topology refuses the run. L1-hidden accesses, fences,
protocol messages, resident flushes and actual hardware ownership remain out of
contract, so this is a bounded mechanism experiment rather than coherence
reconstruction.

## Bounded functional compiler seam

`ptx_subset_compile` lowers `ptx_subset_v1` text into a validated
`simt_its_v1` JSON program. It supports lane ID, integer add/multiply,
less-than predicates, direct labels, indirect u32 loads/stores and exit. It
rejects calls, ABI sequences, negated predication, atomics, vector types,
unknown memory semantics and arbitrary PTX. This makes value-dependent CFG
experiments reproducible without implying CUDA/PTX/SASS compatibility.
