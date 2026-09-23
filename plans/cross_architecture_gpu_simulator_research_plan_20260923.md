# General software GPU simulator roadmap

Updated 2026-09-23 after a scope and literature review. This is the active
roadmap for the alpha branch. It deliberately targets **architectural software
simulation**, not RTL, FPGA, physical design, or a replacement for Nsight
Compute.

## Decision

Continue the repository. Its useful foundation is the runnable

```text
CUDA -> NVBit SASS/protobuf trace -> sim-trace -> sim-sm -> sim-mem
     -> sim-engine -> sim-run -> NCU comparison
```

plus an isolated `sim-arch` specification/functional pilot for user-defined
SIMT shapes. The project is a credible substrate for a configurable GPU timing
simulator, but it is **not yet SOTA, universally architecture-neutral, or
paper-ready**. The next result must be a measured accuracy/speed point on the
available GPU, followed by a configuration that changes the production timing
path and answers a real architecture question.

## Research question and claim boundary

> Can a composable, demand-driven GPU timing model expose meaningful
> scheduler, cache, memory, and asynchronous-resource choices while improving
> simulator wall time without losing agreement with independent NCU
> observations?

The hypothesis is unproven. A JSON schema, a Rust rewrite, event skipping, or a
straight-line eight-lane example is not a contribution by itself. A strong
paper needs one mechanism that (a) changes real production behavior, (b) has a
measurable accuracy/speed trade-off, and (c) enables a design conclusion that
would be difficult to obtain with the current baselines.

### In scope

* NVIDIA SM86 as the current hardware anchor; SM120 only when an input trace
  and machine are actually available.
* NVBit/SASS traces, trace metadata, the existing event-driven SM and memory
  models, and architecture cards.
* Software-configurable issue width, scheduler policy, pipeline/resource
  capacities, cache geometry/policy, memory-service parameters, and explicit
  asynchronous mechanisms.
* A small custom-SIMT functional/timing pilot as a model-relative design
  exploration path. It must remain separate from the vendor trace contract
  until its semantics are genuinely integrated.
* Final comparison against NCU measurements collected on machines we control,
  and simulator wall time/KIPS (or an explicitly defined equivalent).

### Explicitly out of scope

* Verilog/SystemVerilog, RTL simulation, Verilator, FPGA implementation,
  physical-design or signoff modeling, and silicon-internal simulators.
* Claims of access to or prediction of unobservable transistor/RTL behavior.
* MI300X/ROCm timing work until an AMD device and reproducible inputs exist.
* A universal ISA/compiler generator, full-system CPU co-simulation, graphics,
  multi-GPU fabric, or a claim of full CurryGPU ISA/compiler compatibility.
  The locally read CurryGPU design is used as a functional-SIMT reference only.
* Hash/seal/receipt/admission workflows, documentation-policing tests, or a
  broad self-authored test pyramid. Keep only focused smoke checks that make a
  changed executable runnable; the research result is the final NCU comparison
  and simulator-speed measurement.

## What the current code actually provides

* The SM86 replay path reconstructs SASS issue, dependencies, synchronization,
  caches, memory service, and typed diagnostics from NVBit input.
* `sim-arch::spec` validates an opaque architecture/SIMT description. The
  original `simt_u32_v1` executor and the Curry-inspired `simt_its_v1` executor
  are bounded functional feasibility probes, not production timing backends or
  arbitrary-ISA claims.
* Event/dense agreement and the retained approximately 9.7% campaign maximum
  are model-relative development observations. They are **not** held-out
  elapsed-time MAPE and must not be advertised as NCU accuracy.
* `BindingCycles` is a model-relative output. The NCU-to-simulator elapsed
  cycle mapping still needs an explicit, same-launch measurement on the
  available GPU.
* The current source contains real scheduler/cache/memory fixes. Do not revert
  the large cross-architecture commit wholesale merely to remove the unrelated
  Vortex experiments.

## Competitive boundary (primary sources)

| System | Established capability | Consequence for this project |
| --- | --- | --- |
| [GPGPU-Sim](https://gpgpu-sim.org/manual/index.php/Main_Page) | Configurable functional/performance simulation with long-standing CUDA/PTX support. | Configuration knobs are prior art; they are not novelty. |
| [MGPUSim](https://github.com/sarchlab/mgpusim), [ISCA'19 paper](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf) | Modular, parallel execution-driven AMD/GCN3 and multi-GPU modeling; the paper reports 5.5% average error on R9 Nano. | Treat it as AMD/multi-GPU prior art, not as a claim we can reproduce on an NVIDIA-only host. |
| [Accel-Sim ISCA'20](https://ieeexplore.ieee.org/document/9138922) | NVBit native-SASS trace frontend plus PTX execution and a validated, configurable GPGPU-Sim model. | The trace/replay seam and counter-wise validation are established baselines. |
| [Accel-Sim 2.0](https://github.com/accel-sim/accel-sim-framework) | Modern Hopper async features, compressed traces, GPUVision correlation, and a public 13.4% H100 mean cycle error claim. | Compare against the current release, not only the 2020 code. |
| [Huerta et al. MICRO'25 artifact](https://github.com/upc-arco/modern-gpu-simulator-micro-2025) | Modern sub-core/control-bit and parallel simulation close to this repository’s trace seam. | Inherited control-bit/protobuf ideas are prior art; show a distinct mechanism. |
| [FlashGPU-sim](https://arxiv.org/abs/2609.15311) | Execution-driven modern async/Triton modeling; authors report 5.24% cycle MAPE on 131 configurations and 7.86x at 16 host threads. | It is a fresh comparison point where workload and hardware overlap; numbers are not locally reproduced. |
| [NVArchSim, HPCA'21](https://research.nvidia.com/publication/2021-02_need-speed-experiences-building-trustworthy-system-level-gpu-simulator) | Hybrid trace/execution and selective fidelity for speed. | Selective detail is prior art; justify any new demand-driven mechanism empirically. |
| [VADL](https://arxiv.org/abs/2402.09087), [TAIDL](https://doi.org/10.1145/3725843.3756075) | User-defined ISA/microarchitecture specifications and generated functional tools. | “Users can define an architecture” alone is not novel; the GPU-specific timing/NCU path is the differentiator to test. |

Reported percentages above use different devices, workload sets, observables,
and aggregation rules. They are context, not a winner table.

## Staged implementation

### 0. Close the measurement boundary

Define one launch-level observable that the simulator emits and that NCU can
measure on the same SM86 machine. Record device identity, driver/tool version,
clock policy, cache/replay mode, launch scope, and the exact NVBit trace. Keep
issue, modeled-event, completion, launch, and unmodeled-tail diagnostics
separate; do not collapse them into a selection score.

### 1. Make configuration real

Route one architecture-spec family into the production replay, starting with a
single consequential choice (for example issue-port width/scheduler policy or
an L1/L2 resource). A changed parameter must alter scheduling/resource state,
not merely survive JSON parsing. Unsupported trace semantics fail closed.

### 2. Measure the speed mechanism

Benchmark detailed and demand-driven lanes on the same trace and optimized
binary. Report simulator wall time, modeled instructions/events, host thread
budget, and peak memory. Event reduction is only an internal explanation;
fewer events do not imply a hardware speedup.

### 3. Add one user-defined model

Keep the bounded custom-SIMT executors (the original simt_u32_v1 slice and
Curry-inspired simt_its_v1 per-lane-PC slice) as functional feasibility
oracles, then connect a real user-editable scheduler/resource/cache model to
the timing path.
Use a second meaningful configuration and one architecture design question;
do not expand into every ISA or a compiler generator.

### 4. Run a fair baseline comparison

Where hardware, trace format, and workload support overlap, run the smallest
reproducible comparison against Accel-Sim 2.0, the MICRO'25 model, and
FlashGPU-sim. Treat MGPUSim/NaviSim as AMD prior art unless a matching AMD
machine/input set exists. Report unsupported cells rather than silently
excluding them.

### 5. Produce the paper result

Use the final machine-grounded NCU set for the claim. If a later paper study
reserves a held-out family, treat that as an experimental-design choice rather
than a repository gate. Report per-workload NCU error, signed/absolute
distributions, simulator wall time, throughput, memory, trace/preprocessing
cost, and an ablation of the new mechanism. Finish with one consequential
scheduler/cache/async-resource design study. Do not claim “SOTA” until the
same observable, hardware scope, and baseline detail are genuinely comparable.

## Measurement rules

NCU is the independent ground-truth target; NVBit observations are simulator
inputs. Never feed scored NCU values into trace features, tuning, or model
selection. Nsight Compute can replay kernels, flush caches, serialize launches,
and control clocks, so final runs must pin those settings and compare the same
launch boundary. See the [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

The only headline simulator metrics are:

1. accuracy/error against the declared NCU observable; and
2. simulator speed (wall time, with KIPS or another precisely defined rate).

Functional smoke runs, parser checks, and focused unit tests are engineering
signals only. They do not replace the final machine-grounded result.

## Historical material and superseded memory

The Vortex SimX/RTL logs, ELF adapters, and cross-model JSON were exploratory
alpha artifacts. They are archived under `docs/archive/vortex/` and are not
active evidence or completion gates. The external memory file
`/home/jakeshea/GPU_sims_notes/gpu_simulator_research_plan_20260923.md` and the
`gpu_sim_breakthrough_research/` proposal contain a larger CounterTrace/RTL
program; those documents are historical context, not instructions for this
repository. This roadmap supersedes their RTL, Verilator, silicon-validation,
hash/seal, and solver-gate requirements.

## Immediate next executable

Build `sim-run` in release mode, replay one retained SM86 trace, collect the
matching NCU launch observation under fixed clock/cache/replay settings, and
measure simulator wall time. Then wire one architecture-spec resource into
that same replay before adding another custom ISA or architecture adapter.
