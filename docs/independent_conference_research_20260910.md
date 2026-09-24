# Historical research direction — 2026-09-10

> **Superseded scope notice (2026-09-24).** This memo is retained for prior-art
> context only. Its heterogeneous SM86+CDNA3/MI300X, chiplet/fabric, and
> held-out-silicon agenda is not the active project plan. The current scope is
> software-only trace-driven architectural simulation anchored on SM86, with
> NCU/NVBit measurements used only as independent machine observations. Use
> `docs/gpu_simulator_research_positioning_20260923.md` and
> `plans/cross_architecture_gpu_simulator_research_plan_20260923.md` as the
> current direction; do not revive RTL, Verilator, FPGA, or unavailable-AMD
> requirements from this historical memo.

## Executive verdict

The current repository is a credible, carefully validated SM86 replay tool, but
its present contribution is not yet a sufficient ISCA, MICRO, or HPCA paper.
The causal-frontier/event-skipping mechanism is useful engineering and can be a
component of a paper; it is not, by itself, a new architectural insight. The
strongest path is to turn the project into an evidence-preserving,
multi-fidelity simulator that spans NVIDIA SM86 and AMD CDNA3/MI300X while
retaining enough software-stack context to answer system questions. The paper
must then demonstrate a falsifiable hardware or runtime design conclusion on
held-out workloads, rather than only report replay speed or correctness.

The recommended thesis is:

> A typed, architecture-neutral event representation and partial-order
> envelope lets one reuse functional evidence across heterogeneous GPU front
> ends, choose detailed timing only where dependencies make it necessary, and
> explore chiplet, memory, and collective designs without leaking silicon
> measurements into the simulated input.

This thesis is materially different from “we skip causally irrelevant events.”
It makes three claims that can be tested independently: semantic equivalence
across front ends, soundness of adaptive fidelity, and usefulness for a design
choice that existing single-architecture simulators cannot answer.

## What is currently present

The checked-in project is centered on NVIDIA Ampere SM86 (RTX 3090 Ti), with
SM120 support explicitly exploratory. Its path is NVBit/SASS trace, enhanced
protobuf, Rust trace reader, SM scheduling and memory models, a deterministic
event engine, and NCU-shaped output. NCU observations are kept as comparison
data rather than simulator input. The retained post-merge campaign covers eight
small and seven large SM86 workloads; the latest report records the causal
probe, raw issue and terminal frontiers, and the largest comparison-only DRAM
gaps. These are strong reproducibility assets, but they do not constitute AMD
coverage or a validated cross-ISA model.

The current repository has a particularly useful scientific discipline: it
separates simulator state from live NCU observations, keeps typed frontiers
(issue, modeled event, completion, and launch), and preserves independent trace
and control inputs. That discipline should be promoted into the new paper's
central method instead of being replaced by a single fitted performance score.

The current worktree had two uncommitted walker changes when this study began:
`tools/cache_predictor/l1_walker.py` and its focused test. They were preserved;
this report does not modify or stage them.

## What the AMD reference project contributes

The bounded review inspected `zhaosiying12138/AMDGPU-CDNA4-HALFFullSYS-Simulator`
at commit `4f194a419823273b6fbcef81abad1fb9065d28ea`; that temporary checkout
is no longer retained locally. The repository is a large gem5-based
HALF-FullSYS system. Its README claims an AF_UNIX bridge, sealed-memfd shared
device memory, a host-native ROCr/HIP/Triton/PyTorch/aiter/SGLang/vLLM stack,
gem5 VEGA/gfx950 execution, multiple gem5 instances, and a second CCL path. It
also reports Qwen3.5-0.8B TP2 and Qwen3.5-9B TP4 functional runs, token golden
checks, and large wall-clock improvements from functional-fast, direct-copy,
and hybrid layers. Those results are useful project-reported evidence, not
independent silicon validation.

The implementation has several ideas worth reusing:

* A stable bridge keeps runtime and model-specific concerns out of the GPU
  execution core. `docs/framework-runtime-layering.md` explicitly separates
  ROCr/KFD semantics, a fixed-width runtime ABI, collective communication, and
  generic HSACO/kernarg/dispatch/memory semantics.
* The functional-fast/hybrid/timing split is explicit. The README correctly
  warns that `simTicks` from functional-fast and hybrid modes are not timing
  claims, and that functional workgroup stepping is serial. This is an
  important precedent for reporting simulated time separately from verification
  throughput.
* The gem5 patch set contains real execution fixes, including memory-response
  lane attribution, repeated-address bookkeeping, outstanding-memory CU wakeup,
  BF16 MFMA, DS group/swizzle behavior, masked destinations, and configurable
  CU residency. These are evidence that wavefront and memory semantics, rather
  than a superficial AMD label, determine useful CDNA support.
* The CCL path and multi-instance bridge expose a system-level research seam:
  queue admission, rank rendezvous, collective ordering, device memory, and
  compute timing can interact. That seam is absent from the current SM86
  kernel-centric simulator.

There are also limitations that prevent the repository from being treated as a
ready MI300X timing model. The README identifies serial hybrid execution, no
performance optimization for TP greater than one CCL, a layer-gate out-of-memory
case, and the non-timing status of functional-fast/hybrid ticks. Its naming also
needs care: MI300X is CDNA3/gfx942, while gfx950 is associated with newer AMD
targets in the cloned project's terminology. A paper must not call a gfx950
model “MI300X” without an explicit device mapping.

## What the primary sources establish

AMD's own documentation identifies MI300X as CDNA3 with eight XCDs, 304 active
compute units, 192 GiB HBM3, and a chiplet hierarchy. AMD's MI300 microarchitecture
documentation describes 40 CUs per XCD (38 active plus two disabled), a 4 MiB
XCD L2, ACE schedulers, and up to eight XCD/HBM3 stacks connected through
Infinity Fabric.[1][2] ROCm's architecture table gives MI300X target spellings
gfx941/gfx942, wavefront size 64, 64 KiB LDS per CU, and the relevant vector,
scalar, instruction-cache, VGPR, and SGPR capacities.[3] These facts are a
starting card, not enough to infer undocumented bank conflicts, queue latencies,
fabric arbitration, or per-access HBM timing.

The public gem5 Full System AMD model already lists MI300X/gfx942 as a
configuration and explains that the GPU is modeled at gfx9 ISA level with a
native ROCm stack. It also states that the provided configurations are generally
minimally sized and are not intended to predict real hardware measurements.[4]
Thus “add MI300X” is not novel by itself. The contribution must explain what is
measured, what is abstracted, how uncertainty is represented, and what new
architectural decision becomes possible.

The Accel-Sim ISCA paper established the value of native-machine-ISA front ends,
a configurable detailed timing model, and counter-by-counter validation. Its
paper claims a large error reduction across many workloads and uses validation
to expose incorrect baseline assumptions.[5] The later Accel-Sim analysis
shows that front-end, result-bus, and memory-pipeline detail can materially
change conclusions.[6] MGPUSim established a validated AMD GCN3 multi-GPU
simulator and showed that memory management and inter-GPU communication are
research questions in their own right.[7][8] These works make a generic
“multi-GPU simulator” or a new trace reader an insufficient novelty claim.

Recent work raises the bar further. Sim-FA reports an asynchronous
TMA/WGMMA-aware pipeline and 5.7% MAPE on H800 for its defined scope while
explicitly limiting claims to timing-visible structures.[9] A recent chiplet
locality simulator models per-chiplet L2 and local/remote HBM traffic for LLM
GEMM design-space exploration.[10] Another recent preprint claims a
cycle-level asynchronous/distributed GPU framework with H100 validation and
chiplet case studies.[11] These papers imply that the project must provide a
clear methodological guarantee or a cross-vendor system insight; adding a
second ISA without such a result will look incremental.

### Fresh event-driven comparison

The 2026 ACALSim preprint makes the same distinction explicit: its framework
provides event-driven fast-forward and a deterministic two-phase parallel
execution model, but assigns timing-model accuracy to the simulator developer.
That separation supports this project's contract design rather than replacing
it: a fast event engine is infrastructure, while cycle preservation still
requires state-complete transition rules and independent differential evidence.
The paper also reports that thread-management strategy and lock contention are
major performance variables, so future speed claims must include planner,
queue, and synchronization overhead rather than only virtual cycles skipped.

The current upstream MGPUSim issue tracker provides an adversarial warning:
issue #32 reports that a large timing run can produce different estimated
execution times across repeated runs. This reinforces the need to keep the
rebase's `(time, sequence)` ordering, same-cycle phase, and event/dense
boundary comparisons as correctness requirements rather than optional
diagnostics. Accel-Sim's hardware-modeling analysis likewise identifies the
front end and memory pipeline as conclusion-changing structures; those are now
part of the typed SM86 boundary instead of being hidden in a formatted debug
projection.

For a source-level comparison, I also cloned `twhpcedu/acalsim` at shallow
commit `45f4b7456ecad63a2d867da2d79545825318ee2a` into
`/tmp/research_acalsim_20260918`. Its developer documentation and
`include/sim/ThreadManager*` sources make the two-phase contract explicit:
workers execute private simulator work in phase one, while the control thread
applies buffered inter-simulator effects, updates task priorities, and chooses
the next time in phase two. The source therefore supports the same conclusion
as the paper: fast-forward is safe only when the next-event frontier and all
phase-visible effects are represented; it is not a substitute for a complete
GPU timing model.

The current SM86 address-carry canary was also run twice in each mode as a
separate-process determinism probe; both event and dense stdout/stderr pairs
were byte-identical. This is a local reproducibility result, not evidence that
all future parallel extensions will be deterministic without the same ordering
contract.

## Novelty matrix

| Capability | Current SM86 project | AMD HALF-FullSYS reference | What a publishable version must add |
|---|---|---|---|
| Native instruction evidence | NVIDIA SASS/NVBit traces | AMD HSACO/gfx9 execution through gem5 | A common semantic event IR with loss reports and native front ends |
| Detailed timing | SM86 scheduler, memory, event queue | gem5 GPU timing plus functional-fast/hybrid paths | One timing contract that marks exactly which structures are modeled and which are skipped |
| Fidelity control | Causal frontier prototype | Functional-fast/hybrid/timing modes | A dependency-aware selector with a sound partial-order envelope and ablations |
| AMD coverage | None validated | GCN/VEGA/gfx950-oriented gem5 stack | Independently validated CDNA3/gfx942 wave, LDS, MFMA, cache, HBM, and fabric model |
| Chiplet/system model | No XCD/HBM/fabric | Multiple instances and CCL bridge | MI300X XCD/AID/HBM topology, placement, local/remote traffic, and collective contention |
| Custom hardware | JSON architecture cards for current path | gem5 script/config parameters | Versioned schema that generates legal CU/WGP, cache, fabric, and memory variants |
| Software stack | CUDA/NVBit capture path | Host ROCr/HIP/Triton/PyTorch/engines | Shared launch/queue/memory/collective contract across CUDA and ROCr |
| Validation | NCU comparison and replay invariants | Functional golden outputs and project campaigns | Hold-out silicon counters and latency data on both vendors, with no target leakage |
| Architectural result | SM86 replay and causal diagnostics | Functional layer speedups and stability | A falsifiable design rule or Pareto result that changes a hardware/runtime choice |

## Recommended flagship contribution

The most defensible paper is an **evidence-preserving heterogeneous GPU
simulator**. The implementation should have four layers.

### 1. A typed semantic event IR

Define events independently of ISA mnemonics: wave/warp launch and retire,
lane mask, scalar/vector/matrix operation, LDS/shared-memory operation, cache
request and response, barrier/signal, atomic, queue/collective operation, and
host-visible completion. Each event carries an architecture-neutral identity,
an optional native instruction span, address/sector information when available,
scope (lane, wave, CU/SM, XCD, device, peer), and an explicit “unknown or
abstracted” field. A front end must emit a loss report when it cannot recover
an attribute. This prevents a silent lowest-common-denominator model.

For NVIDIA, the existing enhanced protobuf and SM86 trace reader become one
front end. For AMD, start from HSACO disassembly and gem5/ROCr bridge records;
add wave64 lane masks, SGPR/VGPR/AGPR use, LDS, VM/LGKM/EXP waits, MFMA, DS,
and scope-aware atomics. The semantic IR is the scientific object that can be
compared across vendors; the native traces remain available for debugging.

### 2. A partial-order fidelity envelope

Replace a scalar “skipped events” count with typed bounds. For every skipped
interval, record the events that may be reordered, the predecessor/successor
frontiers that are fixed, the resources whose occupancy is conserved, and the
conditions that force detailed replay. A fast path is admissible only when it
preserves memory visibility, barrier/atomic order, queue ordering, and resource
capacity. The report should expose issue, modeled-event, memory-return,
completion, and launch frontiers separately.

The selector should be mechanism-local: choose detailed mode for intervals
that cross a dependency, a cache/fabric contention boundary, a collective,
or an unknown semantic field; choose functional mode for independent compute
regions. It must never use the scored hardware target to choose the mode. The
paper needs a theorem-like invariant (for example, preservation of all
cross-interval happens-before edges and resource lower bounds) plus falsifying
counterexamples, not a broad claim that skipping is “safe.”

### 3. A common chiplet and memory fabric

Model MI300X as eight XCDs grouped under four AIDs/IODs, eight HBM stacks, and
explicit local versus remote paths. The minimum timing graph should include CU
or WGP admission, per-XCD L1/L2, AID/IOD routing, HBM channel/bank queues,
Infinity Fabric links, and inter-device collective links. A synthetic card
must permit changing XCD count, CU count, LDS, cache capacity, HBM stack count,
link bandwidth/latency, partition mode (CPX/NPS4), and collective placement.

The model must distinguish facts from assumptions. The public card supplies
topology and capacities. Microbenchmarks and `rocprof`/Omniperf counters supply
observable latency/bandwidth. Arbitration, bank mapping, and undocumented
queue depths remain typed unknowns until measured. Report parameter intervals
or sensitivity bands instead of inventing exact values.

### 4. A host/runtime time-domain contract

The bridge should expose command enqueue, queue doorbell, dispatch admission,
DMA/memory visibility, signal, and collective events in a host time domain
separate from GPU simulated time. A host-native functional run can supply
software semantics and addresses while the detailed model supplies GPU timing.
The output must state whether a number is host wall time, simulated cycles,
or a verification throughput measurement. This makes it possible to study
launch overhead, queue overlap, CCL ordering, and TTFT/TPOT without pretending
that functional-fast `simTicks` are silicon latency.

## MI300X implementation plan

Use the following staged scope.

1. **Card and ISA contract.** Add a CDNA3/gfx942 card with wave64, CU/LDS,
   cache, register, matrix-core, and HBM capacities from AMD documentation.
   Add explicit target aliases and reject ambiguous gfx950/MI300X spellings.
   Parse HSACO metadata and disassembly into the semantic IR; fail closed when
   wave, wait, or memory-scope semantics are unknown.
2. **Single-XCD functional parity.** Reproduce vector, LDS/DS, barrier, atomic,
   MFMA, and wait-counter behavior on small HIP kernels. Compare outputs and
   event identities against gem5/ROCr or hardware traces. Do not claim timing
   accuracy yet.
3. **Single-XCD timing.** Add CU occupancy, wave scheduling, LDS and L1/L2
   contention, MFMA/vector pipelines, and memory-return attribution. Validate
   instruction mix, wave occupancy, cache sectors, memory transactions, and
   kernel latency on held-out microbenchmarks.
4. **Eight-XCD topology.** Add local/remote HBM, AID/IOD routing, XCD placement,
   and link contention. Validate with address-placement and partition-mode
   probes before running LLM kernels.
5. **Collectives and host bridge.** Translate RCCL/CCL operations to the same
   queue/fabric events and model overlap with compute and memory. Reproduce
   AllReduce, AllGather, ReduceScatter, and barrier behavior for at least
   world sizes 2, 4, and 8.
6. **Custom cards and DSE.** Generate legal variants from a schema, run each
   variant with the same semantic trace, and report Pareto fronts for latency,
   bandwidth, remote traffic, and (if implemented) energy. Keep software
   schedule fixed when claiming architecture effects; when schedule changes,
   regenerate the trace and label the experiment accordingly.

The first MI300X milestone should be one XCD and a small set of kernels. A
full-chip model with guessed fabric parameters is a publication liability.
The kill criterion is clear: if independent probes cannot identify a stable
mapping for a claimed metric, narrow the claim to functional semantics or
locality rather than fitting the result.

## Experimental design that can survive review

Use three disjoint data roles: calibration microbenchmarks, development
workloads, and held-out evaluation workloads. Hardware counters and timings for
the held-out set must never enter architecture cards, selector thresholds,
training data, or trace features. Record GPU model, UUID, driver, ROCm/CUDA
version, clock policy, workload, seed, compiler, trace revision, and card
revision for every run.

The workload matrix should include:

* vector and LDS/DS kernels with barriers and atomics;
* FP16/BF16/FP8 MFMA/GEMM at several tile shapes;
* FlashAttention or another asynchronous pipeline;
* memory-streaming and cache-reuse probes with controlled local/remote HBM;
* AllReduce, AllGather, ReduceScatter, and overlapping compute/collective
  kernels;
* LLM prefill and decode with multiple batch sizes and sequence lengths.

Compare at least: the current detailed SM86 path; the new adaptive path; a
functional-only path; fixed-size event skipping; Accel-Sim/GPGPU-Sim where the
ISA permits; MGPUSim or gem5 gfx942 for AMD baselines; and the AMD reference
project's functional-fast/hybrid modes as an engineering comparison. Use
latency/cycle error distributions, counter error by mechanism, rank correlation
of design choices, replay speed, peak memory, invariant failures, remote-HBM
traffic, collective overlap, and TTFT/TPOT. A single average percentage error
is not enough.

Required ablations are: no semantic IR, no fidelity envelope, no criticality
selector, no chiplet topology, no host-time contract, and no per-frontier
validation. Each ablation should show either accuracy loss, invalid ordering,
lost design ranking, or runtime cost. Include negative controls where the
adaptive path is deliberately given an unknown instruction or a barrier edge;
it must fall back to detailed mode rather than silently produce a result.

The final case study must be a hardware or runtime conclusion, such as a
placement/partition policy that reduces remote HBM traffic under a defined
collective mix, or a cache/fabric tradeoff that changes the best XCD topology.
The simulator is useful only if the conclusion remains consistent when tested
on held-out shapes and when the model is perturbed within its uncertainty
interval.

## Reviewer risks and direct answers

**“Causal skipping is old.”** Agree. Present skipping as the optimization
inside a new typed cross-architecture semantic and certification method. Show
counterexamples where naive skipping fails and the envelope forces detailed
replay.

**“This is only an integration project.”** Include one falsifiable design
insight, a baseline, an ablation, and a held-out validation set. Bridge and
ROCm work are infrastructure for the result, not the result itself.

**“The MI300X numbers are guessed.”** Publish a parameter ledger that labels
each field as documented, measured, inferred, or unknown. Sweep unknowns and
avoid claims whose ranking changes across the plausible interval.

**“Your speedups are wall-clock tricks.”** Report wall time only as simulator
throughput. Report modeled cycles and hardware latency separately, and never
use functional-fast or hybrid `simTicks` as timing evidence.

**“NVIDIA and AMD semantics cannot be unified.”** Preserve native front ends
and make the IR explicitly partial. A loss report and typed unknowns are more
credible than pretending all instructions have the same meaning.

**“The model is too broad to finish.”** Stage the paper around one-XCD CDNA3,
SM86, and one chiplet/collective case study. Add full MI300X and custom cards
only after the single-XCD validation gate passes.

## Suggested paper milestones

* **Month 1:** freeze the novelty claim, IR schema, source ledger, and
  SM86-to-IR adapter; demonstrate byte-identical replay through the old path.
* **Months 2–3:** implement CDNA3/gfx942 functional front end and one-XCD
  semantic tests; publish a public microbenchmark corpus.
* **Months 4–5:** implement single-XCD timing and independent ROCm profiling;
  decide which metrics are identifiable. Kill or narrow unidentifiable claims.
* **Months 6–7:** add XCD/HBM/fabric topology, CCL events, and custom-card
  generation; run controlled locality and collective experiments.
* **Months 8–9:** run held-out cross-vendor validation and all ablations;
  reproduce the design conclusion under parameter perturbations.
* **Month 10:** release code, cards, traces, scripts, and a concise artifact
  guide; write the paper around the method and the design insight.

The acceptance bar is a reproducible cross-vendor semantic path, a bounded and
auditable fidelity mechanism, independent validation on both vendors, and one
architectural conclusion that survives ablation and uncertainty analysis. If
only the first two are complete, target a systems/tool paper rather than
claiming an ISCA/MICRO/HPCA architecture contribution.

## Requested-source availability

The GitHub repository was reachable and cloned successfully. All seven supplied
Zhihu URLs were attempted on 2026-09-10, but none yielded article content
through the available browser: `p/2080711700358099548` was rejected as unsafe,
`p/2066191335330145691` and `p/2044711939746566963` timed out, and
`p/2056027770405270973`, `p/2048501401848903199`, `p/2047985048213780000`, and
`p/2046857919606661244` returned HTTP 403. Search by those IDs did not expose
the article text. Their conclusions are therefore not used as evidence here.

## Sources

1. AMD, “AMD CDNA Architecture,” https://www.amd.com/en/technologies/cdna.html.
2. AMD Instinct documentation, “AMD Instinct MI300 series microarchitecture,” https://instinct.docs.amd.com/develop/gpu-arch/mi300.html.
3. ROCm documentation, “GPU architecture hardware specifications,” https://rocm.docs.amd.com/en/docs-6.0.2/reference/gpu-arch/gpu-arch-spec-overview.html.
4. gem5 documentation, “Full System AMD GPU model,” https://www.gem5.org/documentation/general_docs/gpu_models/gpufs.
5. M. Khairy et al., “Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling,” ISCA 2020, https://ieeexplore.ieee.org/document/9138922.
6. R. Huerta et al., “Analyzing and Improving Hardware Modeling of Accel-Sim,” arXiv:2401.10082, https://arxiv.org/abs/2401.10082.
7. Y. Sun et al., “MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization,” ISCA 2019, https://doi.org/10.1145/3307650.3322230.
8. Y. Sun et al., “MGSim + MGMark: A Framework for Multi-GPU System Research,” arXiv:1811.02884, https://arxiv.org/abs/1811.02884.
9. “Sim-FA: A GPGPU Simulator Framework for Fine-Grained Asynchronous Pipeline Analysis,” arXiv:2605.00555, https://arxiv.org/abs/2605.00555.
10. E. Chung and H. Kim, “A Fast Locality Simulator for GEMM Design-Space Exploration on Multi-Chiplet GPUs,” arXiv:2606.11716, https://arxiv.org/abs/2606.11716.
11. J. Pan et al., “Architecting the Next Generation of Asynchronous, Distributed GPUs for the AI Era,” arXiv:2608.22602, https://arxiv.org/abs/2608.22602.
12. AMD, “AMD Instinct MI300X Platform Data Sheet,” https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf.
13. `zhaosiying12138/AMDGPU-CDNA4-HALFFullSYS-Simulator`, inspected at commit `4f194a419823273b6fbcef81abad1fb9065d28ea`, https://github.com/zhaosiying12138/AMDGPU-CDNA4-HALFFullSYS-Simulator.
14. Local source inspection of the reviewed AMD reference checkout at commit
    `4f194a419823273b6fbcef81abad1fb9065d28ea`, covering
    `docs/framework-runtime-layering.md`, `docs/host-native-architecture.md`,
    `patches/gem5/`, `README_EN.md`, and `SOURCE_LOCK.json`; that temporary
    checkout is not retained locally.
15. Local retained project evidence: `docs/STATUS.md`, `docs/postmerge_campaign_comparison_20260909.md`, `docs/prototype_causal_frontier.md`, and `docs/sm86_replay_regeneration_20260905.md`.
