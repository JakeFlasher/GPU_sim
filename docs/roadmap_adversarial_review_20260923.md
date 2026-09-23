# Adversarial roadmap review

Updated 2026-09-23 after a source/code review and a fresh primary-source
comparison. This is an engineering review, not a process gate.

## Verdict

The repository is worth continuing, but its current roadmap overstates
generality relative to the executable seam. The immediate risk is not missing
RTL or another toy ISA. It is that the project could accumulate functional
examples and event counters without producing a comparable NCU accuracy/speed
result or a production timing mechanism.

The defensible paper target remains:

> one GPU-specific timing mechanism that changes the production SM86 replay,
> improves a measured accuracy/speed point, and enables one consequential
> architecture study.

Everything else is supporting evidence.

## Claim audit

| Intended claim | What exists now | Adversarial objection | Required closure |
| --- | --- | --- | --- |
| NCU-grounded timing accuracy | NCU CSV join and model-relative BindingCycles exist. | No accepted same-launch mapping from BindingCycles to the NCU observable; the retained ~9.7% result is not elapsed-time MAPE. | Select one launch-level NCU metric, run a paired NVBit/NCU capture, document clock/cache/replay/launch scope, and either establish a mapping or explicitly block the accuracy claim. |
| User-defined architectures | ArchitectureSpec, simt_u32_v1, and simt_its_v1 execute bounded functional examples. | The spec is not consumed by production sim-sm; changing a field cannot yet answer a real timing design question. | Route one named scheduler/resource parameter into production replay, with no silent card fallback, then show a same-trace differential. |
| Faster simulation | Event/dense step counts and idle jumps are recorded. | Internal event reduction is not wall-clock speed; Swift-Sim and LPGSim already establish hybrid/parallel speed points. | Measure optimized wall time, throughput definition, peak RSS, and trace/preprocessing cost at equal modeled detail. |
| General SIMT semantics | Curry-inspired per-lane PC/reconvergence slice exists. | It has no load, collective, async, or multiple-group semantics and is not a compiler/frontend. | Keep it explicitly functional; add only the next semantic needed by a real architecture question. Do not claim arbitrary ISA support. |
| Modern GPU coverage | SM86 trace path and selected async/cache mechanisms exist. | Accel-Sim 2.0 already covers Hopper TMA/WGMMA/mbarrier/clusters and 34,000+ H100 kernels. | Publish a feature/workload coverage matrix and mark unsupported cells rather than extrapolating SM86. |
| SOTA comparison | Literature table names current systems. | MGPUSim, Accel-Sim 2.0, Huerta, FlashGPU-sim, Swift-Sim, and LPGSim use different hardware, traces, metrics, and detail. | Compare only overlapping cells with identical observable definitions; otherwise report context, not rank. |

## Findings

### F1 — P0: the first observable is undefined

BindingCycles is a simulator reducer, not a hardware metric. The plan asks
for a mapping but does not specify the candidate NCU metric, kernel boundary,
clock domain, replay mode, cache policy, or launch overlap rule. A reviewer can
reject every later percentage until this is closed.

**Action:** make measurement closure the first milestone. The output is a
small human-readable record and raw NCU/simulator rows, not a hash/seal system.

### F2 — P0: the proposed novelty is currently a bundle of prior art

Event-driven execution, selective fidelity, hybrid analytical models, trace
compression, and parallel simulation are established:

* NVArchSim uses selective fidelity for speed
  (HPCA 2021: https://research.nvidia.com/publication/2021-02_need-speed-experiences-building-trustworthy-system-level-gpu-simulator).
* Swift-Sim reports 82.6x and 211.2x geometric-mean speedups using modular
  cycle/analytical hybrids
  (DATE 2025: https://doi.org/10.23919/DATE64628.2025.10992801).
* LPGSim reports 9.97x single-thread and 197.4x parallel speedups, with
  21–23% errors
  (https://doi.org/10.1145/3805642).
* Huerta et al. report deterministic 5.8x average and up to 14x parallel
  Accel-Sim speedups
  (https://arxiv.org/abs/2502.14691).
* Accel-Sim 2.0 reports 34,000+ H100 kernels, 99% Pearson correlation, and
  13.4% mean cycle error
  (official release: https://accel-sim.github.io/).

**Action:** choose one mechanism that is technically distinct and measure it
against at least one of these dimensions. “Rust + event queue + schema” is
not sufficient.

### F3 — P0: production configurability is absent

ArchitectureSpec and the Curry-inspired executor are isolated from the
ArchCard/sim-sm path. A user can change JSON but cannot yet alter the
production replay.

**Action:** add one explicit override seam for one resource family (issue
transaction width, scheduler policy, or memory-service capacity). Keep vendor
card admission separate. Unknown or incompatible values must fail closed.

### F4 — P1: fixed-trace counterfactual limits are not operationalized

A fixed SASS trace preserves executed instructions and addresses. It cannot
regenerate data-dependent control flow, atomics, synchronization order, or
warp participation after arbitrary ISA/microarchitecture changes. The current
plan says this in prose but does not classify interventions.

**Action:** maintain a three-way intervention contract:

1. trace_safe: changes only timing/resource parameters that cannot alter
   represented facts;
2. trace_bounded: changes a declared effect with a conservative observable
   limit; and
3. requires_execution: requires a new functional frontend or recapture.

Every configurable knob must be assigned one class before use.

### F5 — P1: modern baseline coverage is asymmetric

Accel-Sim 2.0 now models Hopper TMA, WGMMA, mbarrier, clusters, chiplet
memory, compressed traces, GPUVision, and modern LLM tracing. FlashGPU-sim
reports modern async/Triton coverage and 5.24% cycle MAPE on 131 configurations
(paper: https://arxiv.org/abs/2609.15311). An SM86-only result cannot claim
to defeat these systems globally.

**Action:** create a workload/feature overlap table. A missing cell is
unsupported, not an implicit extrapolation.

### F6 — P1: speed evidence is not yet independent

The current event and dense lanes share substantial implementation state.
Equality is useful for model-relative consistency but is not an independent
oracle. Event counts and virtual-cycle skips do not establish wall-clock
speed.

**Action:** measure real process wall time and RSS in release builds, separate
trace loading from replay, use fixed host/thread settings, and report both
detailed and accelerated lanes. Treat dense/event equality as a correctness
control, not as the speed result.

### F7 — P1: workload and label leakage are easy

The retained campaign has already influenced implementation choices. NCU
values must remain labels, not features, and development rows cannot silently
become final evidence.

**Action:** label each workload as development, mechanism-control, or final
measurement. Do not tune parameters from final rows. If a held-out split is
not yet practical, call the result development-only rather than inventing a
formal gate.

### F8 — P1: the paper contribution needs a design decision

A simulator that accepts configuration files is a tool contribution only if it
changes an architecture conclusion. The current roadmap does not choose the
first decision study.

**Action:** select one mechanism with a measurable decision, for example:
scheduler/issue-port organization, L1 admission policy, or async completion
capacity. Define the baseline, intervention, predicted metric, and NCU
observable before implementation.

### F9 — P2: Curry functional scope is correctly bounded but incomplete

The local CurryGPU sources define a functional ISS, per-lane PCs,
reconvergence barriers, separate address spaces, CTA barriers, and future
tensor/async concepts. The current simt_its_v1 diamond is a useful seam but
not an ISA, compiler, or timing model.

**Action:** keep it as a functional contract example. Add semantics only when
a production timing experiment needs them; do not expand it into a general
compiler project.

### F10 — P2: error reporting needs a first-class unsupported path

Unsupported opcodes, incomplete memory effects, and trace-semantic gaps must
be visible in final tables. Silent generic ALU/memory fallbacks can improve
aggregate error while invalidating the claim.

**Action:** every run emits coverage counts and a typed unsupported reason list;
baseline comparisons include unsupported cells.

## Required paper evidence

A reviewer should be able to find, for one declared SM86 workload family:

* exact device/driver/compute capability and NCU settings;
* matching NVBit trace/control inputs;
* simulator binary/configuration and wall-time measurement;
* per-workload error against one declared NCU observable;
* speed/RSS/trace-cost measurements;
* an ablation of the new mechanism;
* one architecture decision whose conclusion changes or becomes affordable;
* explicit unsupported cases and trace-safety classification.

No RTL, FPGA, silicon-internal, hash/seal, receipt, or reviewer workflow is
needed for this evidence.

## Stop/fallback rules

* If no stable NCU-to-simulator mapping is found, keep the simulator as a
  model-relative design tool and remove hardware-accuracy language.
* If the production override changes no replay observable, stop expanding the
  schema and fix the integration seam.
* If acceleration does not improve wall time after trace-load separation,
  report a negative result and do not market event skipping.
* If a final workload requires regenerated control/address behavior, classify
  it as requires_execution instead of scoring fixed-trace accuracy.
* If no baseline shares a valid workload/metric cell, do not report a numerical
  SOTA comparison.

## Review conclusion

The project should proceed, but the next work must be evidence-led and
vertical: close one measurement boundary, integrate one real parameter, measure
one speed mechanism, and use it in one architecture study. Broader ISA and
cross-vendor work is deferred until those four obligations succeed.
