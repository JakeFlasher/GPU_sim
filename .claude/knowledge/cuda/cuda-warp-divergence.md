---
Use when: reasoning about branch divergence, active masks, reconvergence, or control-flow-derived features for non-root performance prediction.
Primary raw source: references/papers/knowledge_base/active/2024__control_flow_management_in_modern_gpus.pdf.
Supporting sources:
- references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md
- references/NVIDIA_Manuals/best_practices/13_control-flow.md
- .claude/knowledge/best-practices/perf-instruction-controlflow.md
Repo touchpoints:
- tools/nvbit_tracer/
- workloads/simbench_bw/
Out of scope: NCU divergence metrics and active simulator divergence modeling.
Version family: Pre-Volta SIMT stack through Blackwell convergence barriers.
---

# Warp Divergence and Reconvergence

Branch divergence changes dynamic instruction count and lane utilization. In
this fork it is a feature source for wall-clock/cache-behavior prediction, not
an NCU metric to read.

## Key Concepts

- A warp has 32 lanes.
- Divergent branches execute paths serially for the active lane subsets.
- Short branches may be converted to predicated instructions.
- Volta+ independent thread scheduling requires explicit warp synchronization
  when code assumes reconvergence.
- `__syncwarp()` is the programmer-visible primitive for warp-scope
  convergence.

## Allowed Evidence

- Static PTX/SASS control-flow structure.
- Branch, predicate, and barrier instruction counts.
- NVBit active-mask or branch-target traces where bounded.
- CUDA-event or CUPTI Activity elapsed time under branch-shape variants.
- Source-level structure when the workload source is controlled.

NCU `SourceCounters`, `WarpStateStats`, and hardware-counter-derived
branch-efficiency metrics are prediction targets / validation labels only
(Rule 1 targets-never-features) — never simulator input features.

## Prior Art

The control-flow-management paper is retained to explain how reconvergence
mechanisms changed across generations. It should guide feature design and
interpretation; the ACTIVE simulator build (re-founded 2026-07-02) is the
primary consumer of this card for its divergence/reconvergence model.

## See Also

- `.claude/knowledge/best-practices/perf-instruction-controlflow.md`
- `.claude/knowledge/cuda/cuda-warp-scheduling.md`
- `.claude/knowledge/ptx/ptx-isa-control-flow.md`
- `.claude/knowledge/measurement/nvbit-injection-playbook.md`

## Escalate to Raw When

- A claim depends on exact Volta+ convergence-barrier behavior.
- A compiler version changes predication or branch lowering.
