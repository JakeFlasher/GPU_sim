# sm_86 direct capture roadmap

## Goal

Collect a representative RTX 3090 Ti cohort with matched NCU observations,
NVBit traces, static SASS/control data, and workload metadata for simulator
development and evaluation.

## Tools

NCU, NVBit, CUDA, GPU clock controls, and administrative commands are available.
Use the direct commands in `docs/runbook_host.md`.

## Per-workload procedure

1. Build the workload and run its ordinary correctness check.
2. Record workload name, arguments, shape, GPU, and relevant tool versions.
3. Run NCU for the selected metric set with enough repeats to estimate variance.
4. Run NVBit separately with the same workload and launch configuration.
5. Extract static SASS/control information.
6. Run the simulator and compare predicted values with the NCU observations.
7. Record failures as technical issues and continue with other runnable cells.

A retry may overwrite or replace a failed experiment output when the researcher
chooses. No approval artifact is required.

## Initial cohort

Start with a small mix that exercises:

- coalesced and strided memory access;
- compute-heavy FP32 and tensor-core kernels;
- shared-memory and barrier behavior;
- atomics;
- async copy where supported;
- short and long kernels;
- multiple grids and block shapes.

Expand only after the direct path works end to end. This keeps debugging
bounded without turning the initial set into a permission boundary.

## Output layout

```text
results/captures/<experiment>/
  metadata.json
  ncu-report.ncu-rep
  ncu.csv
  nvbit/
  sass.txt
  simulator.json
  comparison.json
```

Metadata may include tool versions and descriptive identifiers. No checksum,
receipt, seal, reviewer signature, or admission status is needed.

## Completion criteria

The cohort is useful when its runnable cells have parseable inputs, repeated
measurements with visible variance, simulator outputs, and numerical comparison
reports. Excluded workloads should have a short human-readable reason.
