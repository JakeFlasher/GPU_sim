# GPU simulator research and positioning memo

Date: 2026-09-23
Scope: software architectural simulation only

## Bottom line

Continue this project, but do not call it a current SOTA or universal GPU
simulator. The repository has a credible NVIDIA SM86 seam and a bounded custom
resource frontend: NVBit-generated SASS/protobuf traces feed the native Rust
SM, memory, and event engine, while custom SIMT programs feed a separate shared
dependency/resource pipeline. Neither path supports an arbitrary
user-defined-architecture claim.

The strongest defensible research target is:

> A composable, GPU-specific timing simulator that combines native SASS trace
> replay with user-editable SIMT/resource models and improves the accuracy–speed
> trade-off measured on available GPUs.

RTL, Verilator, FPGA, physical-design, and silicon-internal simulation are not
needed for that target and are outside this repository’s scope.

## Primary-source landscape

| System | What it establishes | What this project must do differently |
| --- | --- | --- |
| [GPGPU-Sim manual](https://gpgpu-sim.org/manual/index.php/Main_Page) | Long-standing configurable PTX functional/performance simulation. | Configuration fields alone are prior art. |
| [MGPUSim (ISCA 2019)](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf) and [project](https://github.com/sarchlab/mgpusim) | Modular, parallel AMD GCN3/multi-GPU execution-driven modeling; paper reports 5.5% average R9 Nano error and high simulation throughput. | Use it as AMD/multi-GPU prior art. Do not claim MI300X accuracy without an AMD machine and inputs. |
| [Accel-Sim (ISCA 2020)](https://ieeexplore.ieee.org/document/9138922) | Native NVIDIA SASS traces through NVBit plus PTX execution, detailed configurable timing, and counter-wise hardware validation. | Native trace replay and validation are baseline capabilities, not novelty. |
| [Accel-Sim 2.0](https://github.com/accel-sim/accel-sim-framework) | Public modern Hopper support (TMA, WGMMA, mbarrier, clusters, compressed traces, GPUVision) and a release claim of 99% Pearson/13.4% mean cycle error on H100. | Compare against this current baseline where hardware/workloads overlap. |
| [Huerta et al. MICRO 2025 artifact](https://github.com/upc-arco/modern-gpu-simulator-micro-2025) | Modern subcore/control-bit/dependency modeling and parallel execution using a trace path close to ours. | Treat inherited control-bit/protobuf mechanisms as prior art. |
| [FlashGPU-sim (MICRO 2026)](https://arxiv.org/abs/2609.15311) | Execution-driven modern async/Triton modeling; reports 5.24% cycle MAPE over 131 RTX 5090/H100/B200 configurations and 7.86x at 16 host threads. | It is a direct contemporary comparison point, but the paper values are not local replications. |
| [NVArchSim (HPCA 2021)](https://research.nvidia.com/publication/2021-02_need-speed-experiences-building-trustworthy-system-level-gpu-simulator) | Hybrid trace/execution and selective component fidelity for large speedups. | Event-driven/selective fidelity must be improved or specialized, not merely renamed. |
| [NaviSim (PACT 2022)](https://sarchlab.org/navisim_pact_2022.pdf) | AMD RDNA execution-driven simulation and feature-isolation studies. | “General AMD support” is already prior art; defer without AMD hardware. |
| [VADL](https://arxiv.org/abs/2402.09087) / [OpenVADL](https://github.com/OpenVADL/openvadl) and [TAIDL (MICRO 2025)](https://doi.org/10.1145/3725843.3756075) | User-defined ISA/microarchitecture descriptions and generated functional tools. | A schema or user-defined ISA alone is not a top-tier contribution; the GPU SIMT timing/NCU path must carry the novelty. |

The reported error and speed numbers are not directly comparable: they use
different hardware, traces, clocks, workload populations, observables, and
aggregation. This memo uses them to set the comparison boundary, not to rank
systems.

## Evidence from this repository

* README.md, sim/trace, sim/sm, sim/mem, and sim/engine describe a real
  NVBit-to-Rust replay path anchored on Ampere SM86.
* sim-arch::spec and sim-arch::simt expose bounded simt_u32_v1 and
  Curry-inspired simt_its_v1 examples. The simt_u32_v1 subset reaches the
  shared sim-sm dependency/resource event backend with resident groups,
  barriers, shared operations, atomics, broadcast, and reduction; it remains
  model-relative and is not native SASS/cache timing or arbitrary ISA support.
* The retained approximately 9.7% maximum is a development counter diagnostic,
  not held-out elapsed-time MAPE. Event/dense equality is model-relative
  equivalence, not hardware accuracy.
* BindingCycles is explicitly model-relative. A final paper result requires an
  NCU observable with the same launch scope and clock/cache/replay conditions.
* The Vortex SimX/rtlsim, ELF, and cross-model artifacts were exploratory
  software/open-GPU probes. They are now archived and are not evidence gates.

## Research agenda for an ISCA/MICRO/HPCA-quality result

1. **Measurement boundary:** choose one launch-level NCU observable, pin device
   identity/driver/tool versions and clock/cache/replay settings, and collect a
   matching NVBit trace. Keep issue, event, completion, launch, and tail
   frontiers separate.
2. **Production configurability:** retain the implemented trace-safe issue-width
   profile and demonstrate its effect on real replay without changing the
   traced instruction stream. Do not broaden this bridge to functional custom
   ISAs or unconsumed resource fields.
3. **Mechanism-level speed:** compare detailed and demand-driven execution on
   identical traces; report wall time, instruction/event counts, host threads,
   and peak memory. Fewer modeled events are not a hardware-speed claim.
4. **User model:** add one meaningful user-editable scheduler/cache/resource
   organization and one architecture design question. Keep custom functional
   execution separate from trace semantics unless a real frontend is present.
5. **Fair baselines:** use Accel-Sim 2.0, Huerta MICRO'25, and FlashGPU-sim only
   where target/workload support overlaps. Treat MGPUSim/NaviSim as AMD
   references unless matching hardware is available.
6. **Final evidence:** use the final machine-grounded NCU set for the claim.
   Report per-workload NCU error distributions and simulator wall time; finish
   with an ablation and a consequential design conclusion.

## Measurement constraint

Nsight Compute itself may replay kernels in multiple passes, flush caches,
serialize launches, and control clocks. Its guide recommends stable capture
settings and externally locked clocks when comparing runs:
[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
NCU measurements remain independent ground truth; do not feed scored NCU values
into trace features, tuning, or implementation choices.

## Curry GPU note

The seven article snapshots supplied by the user are now available locally and
were read in full. They define a functional, bit-exact per-lane-PC ISS with
reconvergence barriers, distinct memory spaces, CTA synchronization, and
future async/TMEM/MMA effects, while explicitly separating timing and RTL.
The repository adds only a bounded Curry-inspired ITS functional example; it
makes no claim of Curry ISA/compiler compatibility. See
[curry_gpu_its_study_20260923.md](curry_gpu_its_study_20260923.md).

## Scope decision

The project is viable and worth continuing. The path to a strong submission is
not a universal hardware framework; it is one measured, falsifiable software
mechanism that changes a real GPU timing resource and supports a declared
accuracy/speed/design claim. The active roadmap records that decision. A fair
numerical baseline comparison remains deferred until hardware, trace,
observable, and semantics overlap.
