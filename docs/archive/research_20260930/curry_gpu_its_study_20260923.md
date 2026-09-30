> Dated context; current capabilities and open work are in [STATUS](STATUS.md).
> Historical completion claims and next-step prose are not active instructions.

# CurryGPU functional-architecture study

Updated 2026-09-23 from the locally cloned CurryGPU article corpus.

## Sources read

The seven primary article snapshots are retained under
/home/jakeshea/GPU_sims_notes/currygpu_series:

1. 现代GPGPU设计（一）ISA
   https://zhuanlan.zhihu.com/p/2044711939746566963
2. 现代GPGPU设计（二）ISS
   https://zhuanlan.zhihu.com/p/2046857919606661244
3. 现代GPGPU设计（三）独立线程调度
   https://zhuanlan.zhihu.com/p/2047985048213780000
4. 现代GPGPU设计（四）内存同步
   https://zhuanlan.zhihu.com/p/2048501401848903199
5. 现代GPGPU设计（五）MMA
   https://zhuanlan.zhihu.com/p/2056027770405270973
6. 现代GPGPU设计（六）架构
   https://zhuanlan.zhihu.com/p/2066191335330145691
7. 现代GPGPU设计（七）软件栈
   https://zhuanlan.zhihu.com/p/2080711700358099548

The Curry design site also exposes a trace viewer at
https://curryfromuestc.github.io/curryGPU/, but no public GitHub source tree
was available. The article snapshots are therefore the primary design
specification available here, not an independently validated implementation.

## What CurryGPU actually proposes

### Functional ISS boundary

The ISS article explicitly defines a behavior-level SIMT interpreter:

* bit-exact functional state, no timing;
* declarative instruction/operand schema as the source for generated decoding
  and assembly helpers;
* per-lane registers and per-lane program counters;
* warp-local reconvergence barriers (BSSY/BSYNC/BREAK/EXIT);
* separated global/shared/local/constant memory spaces;
* CTA barriers, atomics, and deterministic read-compute-commit memory steps;
* later MMA/TMEM, async copy, TMA, and mbarrier effects;
* timing and RTL deliberately outside the ISS boundary.

This is exactly the separation needed here: functional semantics may support a
user-defined architecture, while timing remains a separate model with its own
resource contract.

### ISA and microarchitecture ideas

The ISA article describes a 128-bit fixed-width custom instruction with a
21-bit control section (stall, yield, read/write dependency barriers, wait
mask, operand reuse), separate vector/uniform/predicate/special/descriptor/
tensor-memory operands, and 32-lane SIMT semantics. The later architecture
article describes:

* four subcores plus shared SM resources;
* per-subcore instruction caches, instruction buffers, schedulers, register
  files, scoreboards, and execution units;
* shared LSU/TMA/mbarrier/tensor resources;
* explicit credits and completion lanes;
* a splinter table for per-lane PC groups;
* shared/L1 partitioning, L2 slices, a NoC, and DRAM.

These are design hypotheses and educational architecture choices, not public
NVIDIA facts and not silicon validation.

### Memory and tensor boundary

The memory article's functional model uses sparse 4 KiB byte blocks, distinct
address spaces, 32-lane gather/scatter, per-thread CTA barrier arrival,
deterministic lane-ordered atomic RMW, and a sequentially consistent
data-race-free baseline. It intentionally does not model cache timing, bank
conflicts, retries, scoreboard timing, or weak-memory outcome enumeration.

The MMA article separates sparse metadata, TMEM-resident accumulators, named
tile shapes, and block scaling (MXFP4/NVFP4). These are appropriate future
functional/timing capability categories, but not a reason to add RTL.

### Software stack boundary

The software article makes the frontend boundary explicit:

    CUDA/C++ -> PTX -> currycc backend -> custom 128-bit machine code
              -> module image (.cgm) -> launch/CTA admission -> SM execution

Uniform analysis, register allocation, static control-bit scheduling,
warp-specialization budgets, TMA descriptors, and resource/occupancy metadata
are compiler responsibilities. A fixed NVBit SASS trace cannot prove behavior
after changing arbitrary ISA/control semantics; that requires a new frontend or
functional execution path.

## Consequence for this repository

CurryGPU gives us a precise generality ladder:

1. **Functional architecture:** per-lane control/data semantics, address spaces,
   barriers, collectives, async/tensor effects.
2. **Timing architecture:** named resources, queues, issue/latency/credit
   rules, cache/memory service, and configurable scheduler policy.
3. **Compiler/frontend:** only required when a user changes instruction meaning,
   control semantics, or binary generation; it is not implied by trace replay.

The active implementation now adds only the first bounded step: an
ITS-functional example with per-lane PCs and reconvergence barriers. It has no
cycle timing, compiler, RTL, or silicon claim. The existing simt_u32_v1
functional/timing probe now also includes explicit global/shared address-space
operations, separate dependency domains, and a bounded whole-CTA
`group_barrier` slice in `sim-sm::custom`; ordered shared producer/consumer
values propagate, while inter-group atomics, arbitrary reductions, and
weak-memory outcomes remain outside the functional contract. The bounded slice
also covers deterministic shared atomic RMW, lane-0 broadcast, and one
deterministic lane reduction, but not general collectives or hardware
throughput.

The runnable smoke command is:

    cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin custom_simt_its -- \
      arch_profiles/specs/simt_its8.json \
      arch_profiles/programs/simt_its8_diamond.json

The retained diamond produces eight stores [11, 12, 13, 14, 25, 26, 27, 28]
with all lanes retired. This is functional semantics evidence only.

The bounded ITS frontend now also has an opt-in `--trace` output mode. It
records executed PCs, active masks, lane addresses, and observed successors
after each group step. `simt_its_dynamic.json` exercises `set_lane_id ->
mul_imm -> load_indirect -> store_indirect`, proving that values computed by
the functional executor regenerate addresses rather than replaying fixed
addresses. The result remains custom-ISA evidence: it is not accepted by the
native SM86 SASS path and carries no hardware timing claim.

The trace-safe SM86 issue-width profile and bounded custom resource/event path
are now implemented. The next semantic step should be selected by a concrete
design question; do not expand into a compiler or arbitrary ISA generator.
