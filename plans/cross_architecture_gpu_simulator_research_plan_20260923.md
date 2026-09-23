# General software GPU simulator roadmap

Updated 2026-09-23 after the adversarial roadmap review. This is an
implementation plan for the alpha branch, not a publication guarantee.

## 1. Outcome and falsifiable thesis

### North-star outcome

Produce one measured, reproducible software-simulation result on the available
SM86 host:

* one declared NCU observable;
* one production timing/resource mechanism that users can change;
* one measurable accuracy/speed trade-off;
* one architecture design conclusion enabled by that mechanism.

The project is not yet SOTA, universal, or paper-ready.

### Falsifiable thesis

> A composable, demand-driven GPU timing mechanism can expose a meaningful
> scheduler, memory, or asynchronous-resource choice while improving simulator
> wall time without materially worsening the declared NCU comparison.

The thesis fails if the parameter does not affect production replay, the speed
does not improve after trace-load separation, or the accuracy boundary cannot
be defined.

## 2. Scope

### Active

* NVIDIA SM86 as the hardware anchor.
* NVBit/SASS/protobuf input and the existing trace -> SM -> memory -> runner
  path.
* Architecture cards for evidence-bound vendor replay.
* Explicit software overrides for hypothetical timing/resource studies.
* Curry-inspired functional SIMT examples, kept separate from vendor timing.
* Final NCU accuracy and simulator wall time/RSS.

### Not active

* RTL, Verilog/SystemVerilog, Verilator, FPGA, physical design, or
  silicon-internal modeling.
* MI300X/ROCm until matching hardware and inputs exist.
* Full CurryGPU ISA/compiler compatibility.
* A universal compiler generator, graphics, full-system CPU coupling, or
  multi-GPU fabric.
* Process gates, hash/seal/receipt workflows, broad test pyramids, or
  documentation-policing checks.

## 3. Claim lanes

Every result must name its lane:

| Lane | Permitted input | Permitted claim |
| --- | --- | --- |
| Functional | custom program + architecture spec | architectural-state behavior of the bounded program |
| Model-relative timing | trace/program + hypothetical resource profile | timing relationship inside the declared model |
| Hardware comparison | NVBit trace + evidence-bound card + independent NCU row | error against the declared NCU observable |
| Simulator performance | same input/configuration and host | process wall time, throughput, RSS, and trace cost |

Do not use a functional or model-relative result as hardware validation.

## 4. Trace-safety contract

Before a new parameter is used, classify it:

1. trace_safe — changes timing/resource service only and cannot alter represented
   instructions, addresses, predicates, synchronization participants, or values;
2. trace_bounded — changes a declared effect with a conservative observable
   limit and an explicit unsupported boundary; or
3. requires_execution — changing it can regenerate control flow, addresses,
   atomics, barriers, or values, so a new functional frontend/trace is needed.

Fixed SASS replay cannot silently answer a requires_execution intervention.

## 5. Milestones

Each milestone has a goal, concrete steps, an output, an exit criterion, and a
fallback. A failed criterion produces a scoped negative result; it does not
trigger a new process gate.

### M0 — Claim and workload closure

**Goal:** choose the first experiment rather than expanding the framework.

**Steps**

0.1 Select one retained SM86 kernel/trace with complete NVBit control metadata
and a corresponding NCU launch row.

0.2 Record the exact GPU name/UUID, compute capability, driver, CUDA/NCU
versions, clock policy, cache/replay mode, launch scope, and simulator revision.

0.3 Select one headline NCU observable and one simulator field that purports to
predict it. Keep issue, modeled-event, completion, launch, and tail frontiers
as separate columns.

0.4 Label each input as development, mechanism-control, or final measurement.
Do not tune from final rows.

**Output**

docs/sm86_measurement_boundary_20260923.md containing the command lines,
observable definition, scope, and current uncertainty.

**Exit**

The pair can be rerun and either has a justified mapping or is explicitly
marked hardware-mapping-blocked.

**Fallback**

If no mapping is stable, remove hardware-accuracy wording and continue only
with model-relative design and simulator-speed work.

### M1 — NCU/simulator measurement closure

**Goal:** obtain the first comparable accuracy row.

**Steps**

1.1 Run the selected workload under fixed NCU clock/cache/replay settings.

1.2 Run the matching NVBit capture and preserve the exact trace/control inputs.

1.3 Replay the same launch with sim-run and emit the candidate timing fields.

1.4 Compare only the same launch boundary and clock domain. Do not sum
independent frontiers into a scalar score.

1.5 Repeat one control workload to expose launch, cache, or clock instability.

**Output**

A small CSV/JSON table with NCU value, simulator value, signed error,
absolute error, and all capture settings.

**Exit**

One metric has a documented same-launch interpretation, or the mapping is
blocked with the first concrete cause.

**Fallback**

Use the metric only as an independent label and keep BindingCycles explicitly
model-relative.

### M2 — Production resource override

**Goal:** make one architecture choice affect real sim-sm behavior.

**Chosen first axis**

Start with one issue-resource axis: issue transaction width or an explicitly
named scheduler policy. Do not add scheduler families, cache policies, and
async engines simultaneously.

**Steps**

2.1 Add a separate hypothetical timing-profile input; do not weaken ArchCard
identity or silently replace card values.

2.2 Parse and validate one field with bounded range and a typed error for
unsupported combinations.

2.3 Apply it in the IssueModelParams construction used by sim-run and prepared
replay. Keep the source label visible in output.

2.4 Run the same trace at baseline and at one changed value.

2.5 Inspect issue, completion, event, and launch frontiers plus unsupported
counts. Confirm the changed field affects at least one declared observable.

**Output**

A timing-profile example, two result JSON files, and a short differential note.

**Exit**

The production replay changes as expected, the override is explicitly
hypothetical, and no path silently falls back to a card/default.

**Fallback**

If the field cannot be integrated without violating trace safety, choose a
different resource axis and document the unsupported dependency.

### M3 — Speed mechanism

**Goal:** establish a real wall-clock speed result.

**Candidate mechanism**

Measure and, only if necessary, improve demand-driven idle-gap advancement or
event batching. The candidate must be distinguished from generic event-driven
simulation and from existing parallel/hybrid prior art.

**Steps**

3.1 Build optimized release binaries.

3.2 Measure trace-load time, replay-only time, total process time, peak RSS,
host thread count, modeled instructions/events, and virtual cycles.

3.3 Run detailed and accelerated modes on identical inputs/configuration.

3.4 Profile the first hot path before changing code.

3.5 Implement one bounded optimization and rerun the same measurement.

3.6 Check that the accelerated lane preserves the declared model-relative
observables and reports unsupported/degraded regions.

**Output**

A wall-time/RSS table and one before/after profile note.

**Exit**

The optimization improves replay-only wall time at a declared accuracy/detail
point, or the negative result is recorded and the candidate is retired.

**Fallback**

Keep the detailed lane as reference and choose one component-local hybrid
abstraction; do not claim speed from event counts alone.

### M4 — Feature and unsupported coverage

**Goal:** prevent aggregate results from hiding missing modern behavior.

**Steps**

4.1 Inventory retained traces by opcode family, control primitive, memory
family, async effect, tensor operation, and cache mode.

4.2 Map each feature to implemented, approximated, model-relative, or
unsupported.

4.3 Add run-time coverage counters and a typed unsupported-reason list to the
result output.

4.4 Build an overlap table for SM86 versus Accel-Sim 2.0, Huerta MICRO'25, and
FlashGPU-sim. Mark non-overlap as unsupported.

4.5 Choose one mechanism-control workload that exercises the M2 override and
one workload that must be excluded.

**Output**

docs/sm86_feature_coverage_20260923.md and a machine-readable coverage table.

**Exit**

Every final row has a coverage classification; no unsupported feature is
silently mapped to generic ALU or memory behavior.

**Fallback**

Shrink the claimed workload population rather than broadening the fallback
model.

### M5 — User-defined functional architecture

**Goal:** demonstrate useful semantic extensibility without claiming a
compiler or arbitrary ISA.

**Current slice**

simt_its_v1 provides per-lane PCs, predicate branching, BSSY/BSYNC, BREAK,
EXIT, and lane-specific stores. It is functional-only.

**Steps**

5.1 Keep the declarative ArchitectureSpec separate from production ArchCard.

5.2 Add only the next semantic required by an identified architecture question
(for example a load/address-space effect or one collective). Do not add every
CurryGPU instruction family.

5.3 Run a positive divergent diamond and one fail-closed unsupported/malformed
program.

5.4 Record per-lane state, barrier state, memory effects, step count, and
termination status.

5.5 State which semantics would require a regenerated trace or compiler.

**Output**

A bounded program/spec pair and a concise semantic note.

**Exit**

A user can edit the bounded functional program/spec and obtain deterministic
architectural state without affecting the SM86 path.

**Fallback**

Keep the existing slice unchanged and treat additional ISA work as deferred.

### M6 — Fair baseline comparison

**Goal:** compare only valid cells, not headline numbers from incompatible
experiments.

**Steps**

6.1 Select one workload/metric cell supported by this project and at least one
modern baseline.

6.2 Use identical workload scope, trace/program semantics, architecture detail,
host budget, and output observable where possible.

6.3 Measure baseline wall time separately from trace generation and preprocessing.

6.4 Report unsupported cells, version/commit, device, and metric definition.

6.5 Do not compare NVIDIA results numerically to AMD MGPUSim/NaviSim results;
use those as architectural prior art unless matching AMD hardware is available.

**Output**

A baseline matrix with valid/unsupported cells and no forced ranking.

**Exit**

At least one fair comparison exists, or the overlap is too small and the paper
claim is limited accordingly.

**Fallback**

Use literature results only as context and remove SOTA language.

### M7 — Architecture design study and paper package

**Goal:** show that the simulator changes an architecture decision.

**Steps**

7.1 Choose one question tied to M2 (scheduler/issue width, memory admission,
or async completion capacity).

7.2 Define baseline, intervention, workload family, metric, and trace-safety
class before running the sweep.

7.3 Run the sweep on development/control inputs first.

7.4 Run the final NCU comparison set without using its values for tuning.

7.5 Report per-workload error, wall time, RSS, trace cost, unsupported cases,
and an ablation of M3.

7.6 State uncertainty and the exact hardware/model boundary.

**Output**

One design-study report, figures/tables, commands, and a clean artifact
description.

**Exit**

The design conclusion is stable under the declared inputs and the result
supports a mechanism-local accuracy/speed claim.

**Fallback**

Publish/retain a negative or model-relative result; do not claim SOTA.

## 6. Current progress

### M0/M1 probe — measurement boundary

The retained `gemm_hmma_bf16_sm80` row has full NVBit coverage and stable
seven-repeat NCU labels. After the user locked external SM/memory clocks, three
fresh NVBit/NCU pairs were collected; the current-source architecture-scoped
GPC observation frontier is within 0.32–3.65% of the fresh GPC labels, while
raw `BindingCycles` remains a lower bound. Dirty-writeback/coherence tails
remain unmodeled. The detailed rows and uncertainty are in
`docs/sm86_measurement_boundary_20260923.md`. This is a candidate crosswalk,
not a scored accuracy claim.

### M2/M3 probe — production override and speed

`sim-run --hypothetical-issue-ports N` now routes a bounded issue-width
override through the real replay and labels it in output. On the retained
eight-row small cohort, widths 2 and 4 changed frontiers but increased replay
wall time relative to width 1. A fresh locked-clock GEMM process measured
0.910 s event-driven, 1.541 s dense-reference, and 1.015 s at issue width 2;
peak RSS was 57.6–58.1 MB. These are engineering observations; the negative
issue-width speed result is retained rather than tuned away. See
`docs/sm86_speed_probe_20260923.md`.

The next discriminating work is a second-cohort crosswalk check, then an issue
loop profile and one demand-driven optimization only if it improves measured
replay wall time without weakening the declared model-relative observables.

The first architecture design artifact is now the negative issue-width study in
`docs/sm86_issue_width_design_study_20260923.md`: widths 2/4 alter modeled
frontiers but are slower than width 1. The result is retained as a design
finding, not tuned away or promoted to hardware behavior. The three-row timing
holdout reproduces the same neutral/slower conclusion.

The M4 inventory is now expanded to 18 rows (8 small + 3 timing holdout + 7
regenerated large): 6,392 static records, 130 distinct opcode spellings in
the largest cohort, zero producer-side unsupported memory-path entries, and
zero unmapped memory ranges. Static module origin remains unresolved on all
rows; this is an explicit provenance limitation.

The first M6 baseline feasibility probe is documented in
`docs/baseline_feasibility_20260923.md`, with the Accel-Sim executable overlap
details in `docs/accelsim_executable_overlap_probe_20260923.md`. Accel-Sim 2.0
now builds with a local CUDA 12.8 toolchain and completes three native-trace
replays. Its RTX3070 card and legacy `.traceg` contract do not align with this
project's RTX3090 Ti/enhanced-protobuf boundary, so the probe remains
non-scoring and no baseline ranking is reported.
The Huerta MICRO'25 source also completes the same enhanced-protobuf GEMM with
its native RTX3090 card; two successful runs are recorded in
`docs/huerta_micro25_executable_overlap_probe_20260923.md`. Its differing card,
counter semantics, and protobuf-map runtime instability keep that overlap
non-scoring as well.

## 7. Measurement and reporting rules

NCU measurements are independent targets; NVBit observations and architecture
cards are simulator inputs. Never feed scored NCU values into features,
calibration, implementation selection, or training.

Nsight Compute can replay kernels, flush caches, serialize launches, and alter
clocks. Pin those settings and compare the same launch scope.

Headline metrics remain:

1. error against the declared NCU observable; and
2. simulator wall time, with a precisely defined throughput and peak RSS.

Functional smoke commands and focused build checks are engineering evidence,
not hardware validation.

## 8. Current next action

Next: resolve the M6 alignment questions (3090 Ti-compatible card, stable
protobuf/runtime, and a common trace/observable boundary) without weakening the
local NCU contract. Keep both executable overlap probes as non-scoring
evidence and unsupported cells explicit; do not add another ISA, vendor, or
system layer while that boundary is being resolved.
