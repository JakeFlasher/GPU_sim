# Event-driven simulator execution roadmap

## Objective

Build a useful, fast `sm_86` trace-driven simulator, compare it with real NCU
measurements, then extend it to `sm_120`.

## Technical constraints

- NCU values are evaluation targets, not replay inputs.
- Keep the simulator event-driven and batch local actor progress.
- Keep architecture-dependent values in architecture profiles.
- Treat unknown behavior as a research question, not an inherited constant.
- Use direct GPU/toolchain access and focused tests throughout.
- Milestones organize work; they never prevent starting another sound task.

## Work sequence

### 1. Direct experiment path

- Provide direct NCU and NVBit commands.
- Make simulator inputs ordinary paths with useful validation errors.
- Remove legacy capture lifecycle wrappers and duplicated schemas.
- Produce one end-to-end example from CUDA workload to trace to prediction.

### 2. Cycle semantics

- Define launch-level and workload-level meanings for
  `gpc__cycles_elapsed.max`.
- Implement last-issue, drain, and terminal-event accounting.
- Compare alternate aggregation rules on repeated `sm_86` measurements.
- Add positive and negative unit tests for the chosen semantics.

### 3. SM and memory fidelity

- Validate scoreboard, barrier, replay, async-copy, atomic, and tensor paths.
- Calibrate L1/L2/DRAM geometry and latency from targeted microbenchmarks.
- Keep hypotheses exposed as configurable parameters until evidence supports a
  default.

### 4. Evaluation

- Join simulator outputs and NCU observations by explicit experiment keys.
- Report MAE, RMSE, MAPE where defined, per-workload residuals, and coverage.
- Investigate residual structure rather than fitting per-workload constants.
- Target cycle MAPE at or below 10%; report the achieved value honestly.

### 5. Performance

- Benchmark events per second, trace decode throughput, and memory use.
- Profile hot paths and reduce allocation, dispatch, and queue overhead.
- Preserve numerical behavior with focused regression tests.

### 6. Blackwell

- Capture `sm_120` traces and microbenchmarks directly.
- Add control decoding, cache policy, latency, and scheduling behavior from
  Blackwell-specific evidence.
- Reuse generic algorithms, not Ampere parameter values.

## Definition of useful progress

A change counts when it implements simulator behavior, enables an experiment,
fixes a demonstrated defect, improves measured speed, or explains a scientific
result. Producing more workflow documentation about whether work may proceed
does not count.
