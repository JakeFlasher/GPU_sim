# Heterogeneous GPU simulator novelty plan

> **Continuation, 2026-09-23:** the [general GPU simulator roadmap](cross_architecture_gpu_simulator_research_plan_20260923.md)
> now controls development order. Preserve this earlier proposal as context.
> Its SM86 semantic adapter work remains reusable; the next generality target
> is an open/custom functional architecture, not hardware-dependent MI300X
> timing. Historical SM86-only scope statements below describe the prior slice.

## Decision

Build one publishable vertical slice around **typed cross-vendor event
semantics plus certified adaptive fidelity**. AMD CDNA3/MI300X is the first
new target, and chiplet/HBM/collective exploration is the first architectural
case study. The existing SM86 replay remains the regression and validation
anchor.

This scope is deliberately narrower than “support every modern GPU.” A second
ISA, a faster event queue, or a working gem5 bridge alone is not a top-tier
architecture contribution. The paper needs a method that changes what can be
measured and a design conclusion that survives held-out validation.

**Current execution scope:** implementation work is restricted to SM86
validation and cycle-accurate simulation. The CDNA3/MI300X portions below are
research context and a deferred follow-on; they are not enabled in the current
code path and must not be started until an AMD machine or retained AMD captures
are supplied.

## Evidence that shapes the plan

The repository already has the right foundations: a Rust event queue, typed
issue/completion frontiers, architecture-bound cards, per-lane memory evidence,
and a fail-closed trace reader. The card schema is currently closed to `sm_86`
and `sm_120`; all latency and cache hypotheses are architecture-local. The
causal analyzer currently summarizes explicit SM86-style schedules and refuses
dirty control, dependencies, external side effects, multi-occupancy, or
unknown timing.

The nearby literature makes the required differentiation clear:

* NVArchSim argues for adding fidelity only where it improves system-level
  accuracy and reports high throughput for large workloads.[1]
* Accel-Sim and its follow-up show that native instruction front ends,
  sub-core pipelines, result buses, and memory detail strongly affect
  conclusions.[2][3]
* NaviSim already provides an open, configurable, validated AMD RDNA simulator
  with multi-GPU support and roughly 9% average error in its stated scope.[4]
  Therefore “AMD support” is not a sufficient claim.
* The MICRO 2025 simulator already exposes control-bit-aware modern NVIDIA
  traces, sub-core modeling, and parallel execution.[5] The current project
  should reuse compatible traces and differentiate on semantics, uncertainty,
  and cross-vendor system questions.
* KPerfIR demonstrates the value of compiler-centric intermediate
  representation and observability for modern AI kernels.[6] A simulator IR
  should complement this by preserving timing-visible wave, memory, and
  synchronization semantics rather than replacing compiler IR.
* Sim-FA and recent chiplet simulators show that asynchronous pipeline detail
  and local/remote HBM traffic are active research baselines.[7][8] A new
  contribution must expose a general method or a stronger validated result.
* A recent cross-vendor ISA study proposes hardware-invariant primitives.[9]
  The project should avoid claiming a universal ISA; its novelty is a
  partial, typed timing representation with explicit semantic loss and
  architecture-specific execution backends.

## Target claim and non-claims

### Claim

For a trace whose semantics are represented in a typed event IR, an adaptive
simulator can replace independent intervals with functional summaries while
preserving all cross-interval happens-before edges, resource lower bounds, and
observable completion order. The same IR can feed SM86 and CDNA3 front ends,
making a controlled MI300X XCD/HBM/collective design sweep possible without
silicon measurements entering simulator inputs.

### Non-claims

Do not claim that causal skipping is new, that functional-fast ticks are
hardware time, or that a public MI300X card has undocumented queue,
arbitration, or bank parameters. Do not claim full ROCm compatibility or full
MI300X cycle accuracy until those surfaces are independently measured.

## Code and data seams

| Research surface | Existing seam | Planned change |
|---|---|---|
| Architecture identity | `sim/arch/src/lib.rs`, `arch_profiles/schema/arch_profile.schema.json` | Add `cdna3_gfx942` as an explicit identity; keep target aliases, SKU, and ISA family separate. |
| Card parameters | `sim/arch/src/card.rs`, `arch_profiles/cards/*.json` | Add documented topology/capacity fields and typed `research_gap` values for unknown CDNA3 timing. |
| Dynamic trace input | `sim/trace/src/pb`, `sim/trace/src/reader.rs` | Keep NVBit protobuf compatibility; add a separate semantic adapter API instead of making protobuf fields vendor-specific. |
| Semantic events | new `sim/semantics` crate or `sim/trace/src/semantic.rs` | Define wave/warp, lane mask, memory scope, barrier/signal, atomic, matrix, queue, and collective events plus loss reasons. |
| Native front ends | current SM86 join in `sim/sm/src/replay.rs`; AMD HSACO/ROCr or gem5 bridge | Emit the same IR from SASS and CDNA3 records; preserve native IDs for diagnosis. |
| Fidelity control | `sim/sm/src/causal_frontier.rs`, `sim/engine/src/lib.rs` | Generalize the analyzer to typed interval envelopes and a mechanism-local detailed/summary selector. |
| Memory/fabric | `sim/mem/src/l1.rs`, `l2.rs`; new topology module | Add XCD/AID/HBM placement and local/remote service paths after single-XCD timing works. |
| Results | `sim/run/src/emit.rs`, campaign manifests | Emit simulated cycles, frontiers, skipped interval certificates, unknown fields, and hardware-observation joins as separate objects. |
| Experiments | `results/captures`, `tools/local_labels`, workload probes | Add CDNA3 microbenchmarks and held-out validation partitions; never use scored counters as features. |

## Phases and completion bars

### Phase 0 — Freeze the scientific contract (1 week)

Write the semantic vocabulary and a parameter ledger before changing the
executor. Each field must be labeled `documented`, `measured`, `inferred`, or
`unknown`. Define event scopes (lane, wave, CU/SM, XCD, device, peer) and the
frontiers that can be compared.

Completion evidence:

* a versioned event schema document and two example streams (SM86 and CDNA3);
* a list of events that the current SM86 path cannot represent;
* a mapping table from existing `ReplayInst`, memory roles, barriers, and
  frontiers to the proposed types;
* no code-path change to current SM86 replay.

Stop condition: if the proposed IR collapses vendor distinctions or requires
silicon counters to fill mandatory fields, narrow it before implementation.

### Phase 1 — Semantic IR and SM86 adapter (2–3 weeks)

Implement a small Rust semantic module. Keep it independent of `sim-sm` so the
existing replay remains unchanged. The adapter should consume the current
validated trace records and emit events for instruction issue, active lanes,
global/shared/atomic/async-copy memory, barriers, and completion. Every event
must retain native launch/warp/PC identity and a recoverability/loss enum.

First vertical slice:

1. Convert one retained SM86 HMMA or memory trace to the IR.
2. Run the existing replay and the IR-backed replay with the same card.
3. Compare instruction count, sector masks, issue order, memory origin
   partitions, and terminal frontiers.
4. Deliberately inject an unsupported instruction and prove that the adapter
   reports a typed loss instead of silently mapping it to generic ALU.

Completion bar: all retained SM86 canary outputs remain unchanged, and the IR
adapter emits no unknowns for the chosen canary except fields intentionally
outside its scope.

### Phase 2 — CDNA3/gfx942 functional semantics (3–5 weeks)

Add the architecture card using AMD's documented MI300X topology and ROCm
target spellings. Add a HSACO disassembly/metadata ingestion path or a minimal
bridge export that emits wave64 execution records. Start with vector, LDS/DS,
barrier, atomics, MFMA, waits, and memory scopes.

Use small HIP kernels that isolate one semantic feature at a time. Compare
functional outputs and event identities against the AMD HALF-FullSYS/gem5
reference and, where available, hardware/ROCm traces. Keep timing fields typed
unknown until probes identify them.

Completion bar: vector/LDS/barrier/atomic/MFMA probes execute with correct
outputs, wave64 masks and address scopes are preserved, and unsupported CDNA3
features fail closed. No full-chip latency claim is made.

Stop condition: if HSACO records cannot provide stable lane, wait, or memory
scope semantics, publish the limitation and restrict Phase 2 to functional
locality traces rather than inventing timing.

### Phase 3 — Certified adaptive fidelity on one XCD/SM (3–4 weeks)

Generalize `causal_frontier` into an interval certificate:

* predecessor and successor frontier for each collapsed interval;
* conserved resource demand and occupancy bounds;
* memory visibility and synchronization edges;
* native events represented by the summary;
* explicit unknowns that force detailed mode.

Implement a selector based only on trace semantics, card parameters, and
current simulated state. It may choose detailed mode at dependency, barrier,
atomic, cache/fabric contention, collective, or unknown boundaries. It may
summarize independent compute spans. Do not tune the selector on scored
hardware targets.

Completion bar:

* exact and adaptive modes produce identical completion order and agreed
  frontiers on all eligible SM86 probes;
* adversarial barriers, aliasing, repeated addresses, and occupied SMSPs force
  reification;
* a certificate checker can reject a deliberately corrupted summary;
* replay throughput and peak memory are measured as engineering metrics.

This is the first point at which the project has a defensible method claim.

### Phase 4 — Single-XCD CDNA3 timing (4–6 weeks)

Model CU/wave admission, LDS, L1/L2, vector and MFMA pipelines, wait counters,
and memory returns for one XCD. Use independent microbenchmarks for latency,
throughput, occupancy, and contention. Keep each parameter in the ledger with
uncertainty rather than fitting a workload-specific correction.

Validation set:

* vector and MFMA issue-rate kernels;
* LDS bank/conflict and barrier kernels;
* streaming, reuse, and outstanding-memory kernels;
* atomics and wait-counter kernels;
* two or three small real application kernels held out from calibration.

Completion bar: report per-mechanism error distributions and rank correlation
for design changes. A single average latency number is insufficient.

### Phase 5 — XCD/HBM/fabric and collectives (4–6 weeks)

Add eight-XCD topology as a separate module. The first model should include
local/remote HBM placement, XCD/AID routing, cache ownership, link bandwidth,
and collective traffic. Add CPX/NPS4-like partition variants only when their
logical visibility and memory placement are represented explicitly.

Run controlled address-placement and collective microbenchmarks before LLM
workloads. Then evaluate GEMM, attention, prefill/decode, and AllReduce/
AllGather/ReduceScatter overlap. Keep host wall time, simulated cycles, and
functional verification time in separate output fields.

Completion bar: the simulator predicts held-out local/remote traffic and
collective ordering, and a topology/placement conclusion remains stable across
the documented uncertainty interval.

### Phase 6 — Custom-card design study and paper package (3–4 weeks)

Extend the card schema with legal ranges for XCD count, CU count, cache/LDS,
HBM stacks, link bandwidth, and queue capacity. Generate variants without
rewriting the semantic trace. Run fixed-software and regenerated-software
experiments separately. Report Pareto fronts for latency, remote traffic,
throughput, and energy only where an energy model is independently justified.

Required ablations:

* no semantic IR;
* fixed all-detailed replay;
* naive fixed-size skipping;
* no frontier certificate;
* no chiplet topology;
* no host/collective events;
* no uncertainty sweep.

Completion bar: at least one design recommendation changes under a baseline
or an ablation, and the recommended choice still holds for held-out shapes.

## Measurement and data protocol

Use disjoint calibration, development, and held-out sets. Hardware profiling
with NCU/rocprof/Omniperf is validation evidence only. Trace features, card
values, selector thresholds, and model choices must be fixed before scoring a
held-out workload. Record device identity, driver/runtime version, clock
policy, compiler, launch shape, seed, trace revision, and card revision.

The primary reported metrics are:

* event-representation loss rate and certificate rejection rate;
* per-mechanism cycle/latency error and rank correlation;
* local versus remote HBM traffic and collective overlap;
* adaptive versus detailed replay time and memory;
* TTFT/TPOT only when host/runtime and GPU simulated time are explicitly
  separated;
* sensitivity to every parameter labeled inferred or unknown.

## Risks and kill criteria

* **IR becomes a lowest-common-denominator format.** Keep native IDs and
  explicit loss; reject claims that depend on erased semantics.
* **MI300X timing is underdetermined.** Narrow to functional locality or
  single-XCD measured mechanisms; do not fit a full-chip black box.
* **Bridge engineering dominates.** Use the bridge only to obtain launch,
  memory, and collective events; the paper result must come from the IR,
  certificate, or design study.
* **The adaptive path changes output ordering.** Treat any certificate
  violation as a correctness failure and reify the interval.
* **A design ranking is unstable.** Report the uncertainty band and avoid a
  headline recommendation until the ranking is robust.

If Phase 3 cannot preserve order and resource bounds on SM86, stop the
cross-vendor expansion and fix the method first. If Phase 4 cannot validate a
small CDNA3 mechanism set, do not claim MI300X timing; publish the functional
semantic and locality result as a narrower tool contribution.

## Immediate next implementation slice

The next code change should be limited to a new semantic event type and an
SM86 adapter for one retained trace. It should touch only the new semantic
module, its adapter, and a focused replay comparison command. Do not modify the
current walker files or broaden the architecture card yet. Once that slice
shows unchanged SM86 output and a useful loss report, add the CDNA3 identity and
functional probe card in a separate change. For the active scope, continue with
the SM86 timing-frontier producer and cycle-accurate validation slice instead.

## Implemented in the first slice

The first slice is now present in the worktree:

* `sim-semantics` is a pure architecture-neutral crate defining instruction,
  lane-mask, memory operand, native-identity, loss, and partial-order
  certificate types; it has no trace-reader or executor dependency.
* `sim-trace` owns the SM86 NVBit adapter and
  `semantic-probe TRACE_ROOT [CONTROL_SIDECAR_JSON]` builds a semantic summary
  from the immutable validated trace snapshot, optionally joining canonical
  static opcode text by `(function_unique_id, pc)`.
  The adapter preserves byte addresses and alias multiplicity, while sector
  counts remain an explicitly named 32-byte locality projection. Canonical
  control-sidecar joins also retain stall, yield, read/write barrier,
  wait-mask, and reuse fields as timing-visible evidence.
* `sim-sm` uses the shared certificate decision in its causal-frontier region
  admission, so the old collapse result is preserved while the decision now
  carries explicit reification reasons and resource bounds.
* All eight retained small and seven retained large SM86 workloads have
  semantic instruction counts equal to replay `instructions_issued`. The
  sidecar classifier leaves no opcode unresolved. Static `LDC/ULDC`
  constant-cache records have explicit dynamic memory-class gaps because the
  producer omits that tag; the adapter preserves them as static-only events. A
  serialized semantic event round-trip followed by native replay matches
  counters, timing, and coverage for all 15 rows; the causal probe remains
  `26` exact events, `19` summarized events, and equal final cycle `9`.

The active scope is now SM86 validation and cycle-accurate simulation only.
The remaining boundary is intentional: the adapter preserves exact memory
addresses and lane masks, but it cannot invent dependency or completion facts
that are absent from the trace. CDNA3/MI300X work is deferred until an
accessible machine or retained AMD captures are provided.

## Sources

1. O. Villa et al., “Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator,” HPCA 2021, https://research.nvidia.com/publication/2021-02_need-speed-experiences-building-trustworthy-system-level-gpu-simulator.
2. M. Khairy et al., “Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling,” ISCA 2020, https://ieeexplore.ieee.org/document/9138922.
3. R. Huerta et al., “Analyzing and Improving Hardware Modeling of Accel-Sim,” arXiv:2401.10082, https://arxiv.org/abs/2401.10082.
4. Y. Bao et al., “NaviSim: A Highly Accurate GPU Simulator for AMD RDNA GPUs,” PACT 2022, https://sarchlab.org/navisim_pact_2022.pdf.
5. Huerta et al., `upc-arco/modern-gpu-simulator-micro-2025`, https://github.com/upc-arco/modern-gpu-simulator-micro-2025.
6. J. Guan et al., “KPerfIR: Towards an Open and Compiler-centric Ecosystem for GPU Kernel Performance Tooling on Modern AI Workloads,” OSDI 2025, https://www.usenix.org/system/files/osdi25-guan.pdf.
7. “Sim-FA: A GPGPU Simulator Framework for Fine-Grained Asynchronous Pipeline Analysis,” arXiv:2605.00555, https://arxiv.org/abs/2605.00555.
8. E. Chung and H. Kim, “A Fast Locality Simulator for GEMM Design-Space Exploration on Multi-Chiplet GPUs,” arXiv:2606.11716, https://arxiv.org/abs/2606.11716.
9. O. Abraham and O. Okoli, “Toward a Universal GPU Instruction Set Architecture,” arXiv:2603.28793, https://arxiv.org/abs/2603.28793.
10. AMD, “AMD Instinct MI300 series microarchitecture,” https://instinct.docs.amd.com/develop/gpu-arch/mi300.html.
11. ROCm, “GPU architecture hardware specifications,” https://rocm.docs.amd.com/en/docs-6.0.2/reference/gpu-arch/gpu-arch-spec-overview.html.
