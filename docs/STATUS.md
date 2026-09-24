# Current status

Updated 2026-09-24. This is the current implementation/evidence summary, not a
claim that every future research task or a SOTA submission is complete.

## Implemented paths

| Path | What is runnable | Boundary |
| --- | --- | --- |
| Native SM86 | NVBit/SASS trace -> `sim-trace` -> SM/cache/memory/event replay -> `sim-run` JSON | Fixed instructions, addresses, predicates, and participants; not value-dependent trace regeneration |
| Trace-safe resource profile | `--architecture-spec` or `--hypothetical-issue-ports` changes issue width in ordinary/prepared replay | Hypothetical override, not a new vendor card or ISA |
| Custom `simt_u32_v1` | `custom-program-run` schedules bounded resident groups through `sim-engine::pipeline` | Fixed functional ISA and resource model; not native SASS or `sim-mem` cache timing |
| Custom `simt_its_v1` | Per-lane PCs, predicate branches, BSSY/BSYNC/BREAK/EXIT, stores | Single-group functional example; no timing/compiler claim |

Custom resources have operation bindings, width, latency, initiation interval,
and in-flight limits. Register/memory dependencies permit independent overlap.
Global/shared address spaces, deterministic shared atomic add, lane-0 broadcast,
u32 reduction, and bounded whole-CTA barriers are implemented. See
[specs](../arch_profiles/specs/README.md) and
[programs](../arch_profiles/programs/README.md) for precise limits.

Barriers require equal program lengths, matching IDs at matching instruction
positions, one use per ID, and all participating groups resident. Same-phase
shared conflicts and all inter-group global read/write conflicts reject.
Initial global/shared words are visible to all groups; shared final state is
common, while final global maps remain per-group. This is not a general GPU
memory-consistency model.

Direct replay now consumes carded scheduler resources; resource-less legacy
fixtures retain an explicit fallback. Issue-width overrides disable independent
dual-issue widening. The ITS executor rejects unarmed `BSYNC`.

## Evidence and its limits

| Evidence | Observation | What it does not establish |
| --- | --- | --- |
| [2026-09-24 NCU/NVBit pair](sm86_measurement_boundary_20260924.md) | Three NCU `smsp__inst_executed.sum` rows and simulator `instructions_issued` all equal 197,120 | Elapsed-cycle accuracy, general workload accuracy, or a speed/accuracy frontier |
| [Timing crosswalk](sm86_measurement_boundary_20260924.md) | NCU GPC 13,475–13,529; raw BindingCycles 11,102; model GPC frontier 12,949 | Complete retirement/coherence/writeback timing |
| [2026-09-23 speed probe](sm86_speed_probe_20260923.md) | Event replay faster than dense on one GEMM; local L1 preflight optimization reduced median replay 782.036 -> 755.453 ms | Universal speedup or independent hardware correctness |
| [Issue-width study](sm86_issue_width_design_study_20260923.md) | Widths 2/4 change model frontiers but are slower in the retained study | Hardware speedup; retain the negative result |
| [Feature inventory](sm86_feature_coverage_20260923.md) | 18 rows; explicit producer gaps and typed replay tails | Arbitrary opcode support; static module origin remains unresolved |
| [Custom pipeline](custom_program_pipeline_20260924.md) / [barrier](custom_group_barrier_20260924.md) | Two-group cycles 45; producer/consumer cycles 90 and value 100 | A validated commercial GPU or arbitrary user-defined ISA |

The older approximately 9.7% maximum is a development counter diagnostic,
not held-out elapsed-time MAPE. Python/Rust differentials and focused tests are
implementation evidence against the declared model, not independent silicon
validation. Dated wall times refer to the recorded builds and hosts, not every
subsequent revision.

## Remaining work versus deliberate exclusions

- **Open research:** cycle-observable mapping and missing modeled tails;
  static-module identity; broader independent accuracy/speed evaluation; a
  consequential architecture study with a demonstrated novel mechanism.
- **Deferred by request:** fair numerical Accel-Sim/MGPUSim comparison. Existing
  [baseline cells](baseline_matrix_20260924.json) are non-scoring; executable
  overlap alone does not make them comparable.
- **Unsupported in the custom slice:** arbitrary instruction definitions,
  compiler/ABI generation, native trace regeneration, partial barriers,
  inter-group atomic ordering, weak memory, and general collectives. These are
  not automatically queued implementation obligations.
- **Outside scope:** RTL/Verilator/FPGA, physical design, internal-silicon
  reconstruction, and unavailable-AMD validation. SM120 work needs matching
  local hardware/inputs. Universal binary decoding and silicon-equivalent
  timing are not academic simulator prerequisites.

NCU labels remain independent. Issue, modeled-event, completion, launch, and
unmodeled-tail frontiers stay separate. Available profiling can support useful
cycle prediction without exposing proprietary internals; no exact-equivalence
promise follows.

Use the [roadmap](../plans/cross_architecture_gpu_simulator_research_plan_20260923.md)
for future experiments and [project memory](PROJECT_MEMORY.md) to resume local
work. [Historical records](archive/README.md) and SOL/campaign trees remain
available without becoming active review or approval machinery.
