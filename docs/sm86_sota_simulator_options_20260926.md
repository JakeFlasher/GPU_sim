# GPU simulator and compiler alternatives — 2026-09-26

A dated primary-source options summary, not a local ranking. No full external
simulator execution was performed for this report. Current implementation and
measurement limits are in [STATUS](STATUS.md); the broader literature
extraction remains in the [dated matrix](gpu_simulator_sota_matrix_20260924.md).
The Accel-Sim aggregate values are registered as accuracy error ceilings and
throughput/speedup floors in
[`accelsim_literature_lower_bound_manifest_20260928.json`](accelsim_literature_lower_bound_manifest_20260928.json),
not as local ground truth; NCU on the RTX 3090 Ti supplies that truth.

The newer source/paper review and the locally executable PhaseFuse prototype are
recorded in [the 2026-09-27 study](sm86_fast_exact_prototype_20260927.md).

## Shortlist by question

| Need | Option | What must not be assumed |
| --- | --- | --- |
| Native NVIDIA SASS timing reference | [Accel-Sim/GPGPU-Sim](https://github.com/accel-sim/accel-sim-framework) | Published aggregates register the external floor lane; the pinned executable is parser smoke unless a direct frontend/endpoint match is established. New Hopper support is not local NCU ground truth. |
| Value-dependent functional execution | Accel-Sim PTX mode / [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) | PTX is virtual ISA, not the exact compiled SASS. Check the selected version's opcode/runtime support, not an old Fermi manual. |
| Modern CUDA/Triton execution-driven research | [FlashGPU-Sim](https://github.com/FlashGPU-Sim/FlashGPU-Sim) | Public workflow focuses on newer GPU configurations; reported results are not local SM86 evidence. Porting costs and support are separate from paper claims. |
| Hybrid-fidelity/stall research | [HyFiSS](https://github.com/ConvolutedDog/HyFiSS), [GATSim](https://slam.ece.utexas.edu/pubs/date17.GATSim.pdf) | Different abstraction levels and validation GPUs; GATSim is older prior art, not a newly released SM86 baseline. |
| Heterogeneous CPU/GPU trace research | [MacSim](https://github.com/gthparch/macsim) | Its heterogeneous frontends do not imply a validated SM86 card. |
| Tile-level LLM exploration | [GPU-Tile-Sim](https://arxiv.org/abs/2607.11262) | Tile/warp graph abstraction is not instruction-level SASS or general value-driven execution; preprint claims remain unreplicated here. |
| Open GPU ISA plus compiler stack | [Vortex/VOLT](https://github.com/vortexgpgpu/vortex) | Open RISC-V GPU/LLVM stack, not NVIDIA binary or timing compatibility. ISA changes still need semantics and compiler work. |
| Generate tools from ISA/ABI descriptions | [OpenVADL](https://openvadl.github.io/openvadl/) | Current implementation generates ISS/LLVM backend/assembler/linker with feature limits; language goals are not universal GPU/SIMT/RTL generator support. |

[MGPUSim](https://github.com/sarchlab/mgpusim) / [Akita](https://github.com/sarchlab/akita)
and [gem5 GPU](https://www.gem5.org/documentation/general_docs/gpu_models/vega)
are useful modular/full-system prior art but not a ready SM86 replacement;
no AMD/MI300X work is proposed by this report.

## Recommendation (engineering inference)

Keep the native SM86 path while closing its concrete event/metric contracts.
For value-dependent experiments, select a bounded PTX/semantic frontend and
retain separate functional versus timing outputs. For an actual new ISA/ABI,
start from Vortex/VOLT or evaluate OpenVADL rather than claiming a JSON allow-list
is a compiler generator. Decide the language/ABI and required operations first.
None of these choices closes native writeback/coherence or hardware-pipe
observability by itself; a useful model requires explicit state and executable
interventions, not just more diagnostics.
