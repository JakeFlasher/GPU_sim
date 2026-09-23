# Current status

Updated 2026-09-23.

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
* A custom parameter is not yet a production timing control; the next
  implementation must route one scheduler/resource choice into sim-sm.

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

1. Run a release sim-run replay and the matching NCU launch observation under
   the same declared conditions.
2. Measure simulator wall time on the same trace and host.
3. Route one architecture-spec scheduler, issue, cache, or memory-resource
   parameter into the production replay.
4. Compare the resulting accuracy/speed point with applicable Accel-Sim 2.0,
   Huerta MICRO'25, and FlashGPU-sim runs when their workload/hardware support
   overlaps.
5. Use one consequential configurable design study before expanding the custom
   ISA surface.

## Historical material

The Vortex SimX/rtlsim, ELF, and cross-model probes are archived in
docs/archive/vortex. They remain readable as dated alpha notes, but they are
not active evidence, dependencies, or completion gates.

Long campaign and semantic-roundtrip reports under docs/ retain historical
measurements. They should not be read as a current SOTA claim or as a
requirement to add hashing, receipts, review gates, broad tests, RTL parity, or
silicon validation.
