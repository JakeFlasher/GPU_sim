# Architecture-extensibility boundary research — 2026-09-24

## Finding

“User-defined architecture” has three different deliverables that should not
be conflated:

1. a declarative **resource/timing model**;
2. a functional **ISA/semantic frontend**; and
3. a compiler/assembler/ABI toolchain that produces runnable binaries.

This repository now has (1) for a bounded SIMT model and a trace-safe SM86
resource profile, plus a deliberately small fixed (2) frontend. It does not
claim (3), and adding an ad-hoc compiler would be the wrong next step.

## Primary evidence

[Accel-Sim ISCA 2020](https://people.ece.ubc.ca/~aamodt/publications/papers/accelsim.isca2020.pdf)
separates native SASS trace evidence from execution-driven PTX semantics. Its
trace representation preserves PC/active mask, register/functional-unit
information, and memory addresses, but trace replay cannot answer interventions
that regenerate value-dependent control flow, values, or synchronization.
That is the correct boundary for this project’s `trace_safe` contract.

[OpenVADL](https://openvadl.github.io/openvadl/) demonstrates the much larger
compiler/ISA-generator problem: one description separates ISA and
microarchitecture and can generate an ISS, assembler, linker, LLVM backend,
and other artifacts. This is strong prior art, not a small feature that should
be recreated inside the SM86 timing code.

[TAIDL (MICRO 2025)](https://doi.org/10.1145/3725843.3756075) shows a related
ISA-definition direction for tensor accelerators, including generated scalable
functional test oracles. It reinforces that a language-level contribution
requires a real semantic specification and generated oracle, not merely a JSON
allow-list.

For the available NVIDIA boundary, the [NVBit project](https://github.com/NVlabs/NVBit)
describes dynamic SASS instrumentation and tracing, while the [Nsight Compute
Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
documents profiler metric collection and replay behavior. These public tools
are evidence-producing adapters/labels; neither exposes a universal proprietary
decoder or internal completion-timing state.

The CurryGPU material is best used as a functional-SIMT semantic reference:
per-lane PCs, predicates, reconvergence tokens, address spaces, and barriers.
It does not imply that this repository should implement a Curry compiler or
RTL model.

## Current implementation boundary

* `sim-arch::spec` validates architecture/resource shape, operation-resource
  bindings, issue width, initiation interval, latency, and in-flight capacity.
* `sim-arch::simt_pipeline` lowers the fixed `simt_u32_v1` operations into a
  dependency/resource DAG with RAW/WAW and memory-order dependencies. Global
  and shared word address spaces are separate dependency domains.
* `sim-engine::pipeline` provides deterministic issue/completion events and
  resident-group admission.
* `sim-sm::custom` executes multiple groups through that engine. Matching
  `group_barrier` generations add cross-group ordering dependencies and allow
  separated shared-memory stores/loads to propagate through the shared map.
* `sim-run --architecture-spec` exposes only a `trace_sass_v1` SM86 issue-width
  profile; it rejects functional-only ISAs and every unused timing/resource
  knob.
* `sim-trace` already exposes an SM86 semantic adapter with explicit losses for
  unclassified opcodes, missing dynamic memory classes, unavailable address
  spaces, and absent addresses. Those losses are the correct native-trace
  extensibility surface; silently inventing values is not.

## Adversarial residuals

The following are intentionally still unsupported rather than hidden:

* arbitrary instruction semantics in JSON;
* generated assembly/compiler/ABI support;
* general cross-lane collectives and a general memory-consistency model (the
  bounded slice now has lane-0 broadcast, deterministic lane reduction, and
  shared atomic RMW, but excludes inter-group atomics, fences, weak-memory
  outcomes, and partial participation);
* native trace regeneration after value-dependent interventions (a new
  functional/frontend capture is required);
* a universal vendor-independent binary decoder (native adapters are added only
  for available, evidence-producing trace formats);
* silicon-equivalent timing internals. NCU/CUPTI observations remain external
  validation labels, not inputs or a promise to reconstruct proprietary
  hardware state.

The last two items are **not project completion obligations**. Academic
researchers generally cannot obtain proprietary vendor decoder internals or
silicon completion state; the defensible method is to consume available native
traces and compare declared observables against profiler ground truth.

The next safe extension is therefore a **semantic operation contract** for one
additional bounded operation family only when a concrete design study requires
it. It must add:

1. a typed instruction/semantic record;
2. functional state transition;
3. dependency/resource lowering;
4. unsupported/fixed-trace classification; and
5. an independent differential oracle.

Do not add a compiler generator, RTL backend, or new vendor ISA before such an
operation is required by a real experiment.
