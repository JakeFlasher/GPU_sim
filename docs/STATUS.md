# Current status

Updated 2026-09-24.

## Scope

This repository is a software architectural timing simulator. The production
path consumes NVBit/SASS traces and architecture inputs, then emits simulator
metrics for comparison with NCU on available machines.

* Current hardware anchor: NVIDIA Ampere SM86 (RTX 3090 Ti).
* SM120 is exploratory only when a local trace/card and machine exist.
* NCU observations are independent comparison targets; they never become
  simulator inputs for a scored run.
* RTL, Verilator, FPGA, physical-design, internal-silicon, and MI300X/ROCm work
  are outside the active scope.

## Runnable path

    CUDA workload -> NVBit trace -> sim-trace -> sim-sm -> sim-mem
                  -> sim-engine -> sim-run
                                  \-> NCU comparison in eval/

The Rust workspace contains the trace reader, SM scheduler/dependency model,
cache and memory service, event engine, architecture cards, and JSON result
emitter. The current source includes scheduler, cache, memory-completion, and
trace fixes; they remain the live implementation.

## Current evidence

The retained SM86 campaign contains 15 workload shapes. Its maximum
approximately 9.7% scored error is a development counter diagnostic, not
held-out elapsed-time MAPE. Event/dense replay agreement establishes only
model-relative equivalence. The emitted BindingCycles field is likewise a
model-relative diagnostic until a same-launch NCU mapping is measured.

The custom architecture slice is intentionally separate from production
SM86 replay:

* sim-arch::spec parses a bounded, fail-closed architecture/SIMT description.
* sim-arch::simt executes one matching simt_u32_v1 program with lane
  arithmetic and stores. The separate simt_its_v1 slice executes a bounded
  per-lane-PC diamond with BSSY/BSYNC reconvergence and lane-specific stores.
* The retained width/issue-width sweep is a hypothetical design probe, not
  silicon accuracy or a universal architecture claim.
* `--hypothetical-issue-ports N` is now a bounded production timing control
  for the trace-safe issue-resource lane; it remains explicitly hypothetical
  and does not alter vendor card identity.

## Measurement boundary

The final evidence bar is deliberately small:

1. one declared launch-level NCU observable on the available GPU;
2. the matching NVBit trace and architecture card;
3. fixed device identity, driver/tool versions, clocks, cache/replay mode, and
   launch scope;
4. simulator error against that NCU observable; and
5. simulator wall time (and a precisely defined throughput rate if useful).

NCU may replay kernels, flush caches, serialize launches, and change clocks.
Keep those settings explicit and stable. Keep issue, modeled-event,
completion, launch, and unmodeled-tail frontiers separate rather than summing
them into a selection score.

## Immediate work

M0/M1 now has fresh locked-clock paired rows documented in
[sm86_measurement_boundary_20260923.md](sm86_measurement_boundary_20260923.md):
the candidate GPC observation frontier is within 0.32–3.65% of three fresh NCU
GPC rows, while raw BindingCycles remains a lower bound. Dirty-writeback/
coherence tails remain explicit, so this is a candidate crosswalk rather than
promoted hardware accuracy.
M2/M3 now have an executable probe documented in
[sm86_speed_probe_20260923.md](sm86_speed_probe_20260923.md): event-driven replay
is faster than the dense reference on one retained trace, while issue-width 2/4
is currently slower. This is model-relative speed evidence, not a final claim.
The first M4 coverage inventory is in
[sm86_feature_coverage_20260923.md](sm86_feature_coverage_20260923.md): eight
full traces, 2,000 static records, 100 opcode spellings, empty producer-side
unsupported arrays, and a remaining static-module provenance gap.
The first architecture design artifact is the
[SM86 issue-width study](sm86_issue_width_design_study_20260923.md): widths 2/4
change modeled frontiers but are slower than the baseline. It is a negative,
model-relative result and does not claim hardware behavior.
The M3 L1-preflight optimization is recorded in
[sm86_speed_probe_20260923.md](sm86_speed_probe_20260923.md): aggregating
per-origin counter budgets reduced median replay phase from 782.036 ms to
755.453 ms on the fresh GEMM while preserving the reported counters and
frontiers.
The M6 baseline feasibility notes and machine-readable matrix are in
[baseline_feasibility_20260923.md](baseline_feasibility_20260923.md) and
[baseline_matrix_20260924.json](baseline_matrix_20260924.json). The
executable overlap record is in
[accelsim_executable_overlap_probe_20260923.md](accelsim_executable_overlap_probe_20260923.md).
Accel-Sim 2.0 now builds and completes a native-trace replay, but its native
SM86 card is RTX 3070 and its legacy trace contract differs from the enhanced
protobuf capture. The probe is therefore executable overlap evidence only; no
numeric baseline ranking is reported.
The Huerta MICRO'25 source has a same-enhanced-trace overlap record in
[huerta_micro25_executable_overlap_probe_20260923.md](huerta_micro25_executable_overlap_probe_20260923.md);
its RTX 3090 card, non-NCU counter, and protobuf-map runtime instability keep
that cell non-scoring.
The derived Accel-Sim SM86 profile probe reduces only device-scale mismatch and
completes three stable replays, but inherited RTX3070 model fields keep it
non-scoring; see
[accelsim_derived_sm86_overlap_20260924.md](accelsim_derived_sm86_overlap_20260924.md).

Follow the fine-grained milestones in the
[adversarial roadmap review](roadmap_adversarial_review_20260923.md):

1. close one same-launch NCU/simulator measurement boundary;
2. route one explicit hypothetical resource override into production replay;
3. measure replay-only wall time/RSS before and after one speed mechanism;
4. publish feature/unsupported coverage; and
5. use the mechanism in one architecture design study.

Do not add another ISA or system layer before the first three items produce
executable evidence.

## Historical material

The Vortex SimX/rtlsim, ELF, and cross-model probes are archived in
docs/archive/vortex. They remain readable as dated alpha notes, but they are
not active evidence, dependencies, or completion gates.

Long campaign and semantic-roundtrip reports under docs/ retain historical
measurements. They should not be read as a current SOTA claim or as a
requirement to add hashing, receipts, review gates, broad tests, RTL parity, or
silicon validation.
