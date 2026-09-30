# AMALi source study for SM86 abstractions (2026-09-30)

Inspected shallow clone: `/tmp/research_amali_20260930`.

## Relevant code paths

* `ISA_parser/sass_parser.py::parse` maps SASS forms to semantic units,
  records dependency references, and emits instruction latency/task metadata.
  Its unit labels are an analytical abstraction, not a physical NVIDIA pipe
  witness.
* `src/memory_model.py::get_hit_rate_analytical` maps reuse-distance
  probabilities to cache hit rates using capacity, line size, and associativity.
  It is a piecewise cache-response law, not ownership/read-from reconstruction.
* `src/kernels.py::process_GCoM` separates selected, fixed-dependency wait,
  long/short scoreboard, functional-unit, memory, and idle terms; it composes
  them with max-plus bottleneck logic.
* `src/accelerators.py` stores occupancy limits, cache/DRAM/NoC bandwidth,
  latency parameters, and launch slopes as architecture-card inputs.
  The explicit `max_active_blocks_per_SM`, warp, register, shared-memory, and
  launch-overhead terms are useful semantic features for the SM86 stage model.

## Transfer to this repository

The transferable mechanism is separation, not constants:

1. semantic instruction demand and dependency latency;
2. reuse/capacity response intervals;
3. occupancy and active-SM envelope;
4. max-plus stage composition;
5. explicit launch/tail overhead.

The SM86 implementation already uses analogous semantic stages and replay
frontiers. Any future extension should import the structure while calibrating
rates only from independent cohorts. It must not import AMALi's architectural
constants, assume its scheduler is GA102, or relabel semantic units as physical
pipe tags.

## Non-transferable boundaries

AMALi assumes a complete trace/ISA/architecture-card contract and analytical
cache parameters. This repository lacks public GA102 line ownership,
read-from, complete dirty persistence, unrestricted CUDA ABI generation, and a
unified NCU/CUDA-event/simulator endpoint. Those remain explicit unknowns or
intervals.

