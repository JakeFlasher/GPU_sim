# Historical adversarial roadmap review

Updated 2026-09-24. This is a dated engineering record, not a workflow gate.
The consolidated current state is [STATUS](STATUS.md),
[PROJECT_MEMORY](PROJECT_MEMORY.md), and the
[2026-09-24 adversarial synthesis](adversarial_review_and_research_20260924.md).

## Original verdict

The repository was worth continuing, but the earlier roadmap overstated
universality. A credible paper target required one GPU-specific mechanism that
changed production SM86 replay, a declared profiler boundary, honest speed
measurements, and one architecture decision. RTL, another toy ISA, or a global
SOTA claim would not close that gap.

## Findings and dispositions

| Finding | Disposition |
| --- | --- |
| BindingCycles had no declared hardware observable | One same-launch instruction-count mapping now matches `smsp__inst_executed.sum` to `instructions_issued` at 197,120. Cycle mapping remains provisional because retirement/coherence tails are incomplete. |
| ArchitectureSpec did not reach production | `--architecture-spec` now consumes only strict matching `trace_sass_v1` SM86 issue-width profiles; functional/custom specs remain separate. |
| Direct replay could ignore card scheduler resources | Direct and prepared replay use carded resources when present; resource-less legacy fixtures use an explicit fallback and partial blocks fail closed. |
| Hypothetical issue width could retain dual-issue widening | Overrides disable independent dual-issue admission and retain the original card for validation. |
| Custom timing serialized independent resources | The shared pipeline now models register/memory dependencies, resource overlap, initiation interval, in-flight limits, resident groups, completion events, and bounded cross-group barriers. |
| Custom functional state lacked address-space/collective coverage | Shared memory, deterministic shared atomic add, lane-0 broadcast, wrapped-u32 reduction, and bounded barriers are implemented with explicit limits. |
| Unsupported behavior could be hidden by generic fallbacks | Coverage, unmodeled tails, producer unsupported-memory paths, and unmapped ranges remain typed; no opcode-based support claim is fabricated. |
| Historical scope drift advertised RTL/MI300X/Vortex as active | Older records and skills now carry historical/superseded scope; current docs explicitly exclude those obligations. |

## Scientific boundaries

- NCU/CUPTI observations are independent labels, never simulator inputs.
- Fixed SASS replay may vary only `trace_safe` resource service unless a new
  functional frontend or recapture is used.
- The custom path is fixed-ISA/model-relative, not an arbitrary ISA/compiler,
  native SASS/cache model, vendor-neutral proprietary decoder, RTL model, or
  silicon-equivalent timing reconstruction.
- Fair numerical Accel-Sim/MGPUSim comparison is intentionally deferred because
  the retained overlap cells do not share enough hardware, trace, observable,
  and semantic identity.

## Prior-art conclusion

Accel-Sim establishes NVBit/native-SASS tracing plus detailed timing and
hardware-counter correlation. MGPUSim establishes modular AMD GCN3/multi-GPU
modeling. NVArchSim, Swift-Sim, Huerta, FlashGPU-sim, VADL/OpenVADL, and TAIDL
cover selective fidelity, parallel/hybrid speed, generated architecture
semantics, or modern GPU scope. The current project must state a mechanism-local
contribution rather than “Rust + event queue + schema” or “universal GPU
simulator.” See [research positioning](gpu_simulator_research_positioning_20260923.md)
for links to those sources.

## Retained evidence

The original issue-width, speed, coverage, baseline-overlap, and Curry notes
remain in their dated files. They are development/context evidence. They do
not create a completion gate, approval ceremony, hash/seal requirement, or
numerical SOTA ranking.
