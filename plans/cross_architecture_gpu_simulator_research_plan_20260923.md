> **Superseded context (2026-09-30).** This file is retained for historical source/data ancestry, not as an active plan or task queue. The compact capability boundary is [`docs/STATUS.md`](../docs/STATUS.md); root `AGENTS.md` controls execution.

# SM86 software simulator research roadmap

Updated 2026-09-26; original filename retained for links. [STATUS](../docs/STATUS.md)
is the capability record, not this plan. [History](../docs/HISTORY.md) retains
measurements and previous milestone claims.

## Objective

A configurable software GPU simulator with useful, independently evaluated
accuracy/detail/turnaround trade-offs. Native SM86 replay and custom functional
frontends are separate evidence lanes. Neither a JSON resource schema nor an
arbitrary postprocessing formula establishes a novel hardware mechanism.

## Research choices

| Question | Existing seam | Evidence still needed |
| --- | --- | --- |
| Memory completion | Shared-L2 dirty state, event queue, diagnostic service formulas | Scheduled reads/writes, visibility/order contract, same-endpoint validation |
| Physical pipe routing | Opcode candidate domains and count-balancing observer | SMSP-local availability, capacities, dependencies and timing effects |
| Value-dependent behavior | Single-group ITS functional execution and custom trace | Complete successor contract, stronger tests; separate native/PTX frontend if required |
| Accuracy | Retained SM86 trace/NCU pairs | Same-source-domain counters, content-bound independent holdout and tail mapping |
| Speed | Event/dense path and phase counters | Repeated phase/RSS measurements on unchanged semantics, before/after evidence |
| Fast research prototype | `sim-run --fast-frontier-only` / `../docs/archive/research_20260930/sm86_fast_exact_prototype_20260927.md` | Certificate rate, refusal reasons, planner-inclusive wall time and boundary-state differential |
| Architecture contribution | Width override and custom resource scheduling | A real mechanism whose intervention changes a defensible design decision |
| Toolchain | PTX, Vortex/VOLT and OpenVADL options researched | User-selected input/output ABI and concrete semantic/compiler subset |
| External baseline | Two separate lanes: published Accel-Sim aggregate accuracy-ceiling/throughput-speed-floor corpus plus optional Accel-Sim/Huerta executable/parser overlap | Local NCU is ground truth; evaluate the registered literature targets, while direct executable timing still requires an explicitly matched frontend/workload/observable |

Fixed replay can vary timing/resources only when instructions, addresses,
predicates and participants remain valid. A value/CFG/atomic-order intervention
requires execution or recapture. Keep model-relative, functional, silicon and
simulator-performance results separate; do not select a frontier by its NCU fit.

The current prototype direction is PhaseFuse: exact trace counters plus sparse
event timing plus state-complete scalar certificates with detailed fallback.
It is intentionally not a claim that native writeback/coherence, physical-pipe
routing, SASS CFG regeneration, or general compiler generation are closed.

Earlier full-completion and “consequential tail study” claims were withdrawn:
scenario arithmetic is not completion of the mechanism. Older plans are
historical context, not revived obligations. Choose the next seam from the
user's actual question; retain SOL/campaign and reference material intact.
