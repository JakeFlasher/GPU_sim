# Paper Knowledge Base

This corpus supports the current non-root GPU measurement and prediction
project. Use it to reason about instrumentation, runtime timing, cache-behavior
inference, and prediction methodology without relying on NCU, GPU hardware
counters, privileged driver configuration, or simulator validation as a goal.

## Read First

1. `problem_playbook.md`
   Map the current measurement or prediction problem to a small set of papers.
2. `selected_index.md`
   Shortlist high-signal PDFs for the active research direction.
3. `manifest.json`
   Use only as historical metadata when programmatic filtering is needed.
4. `active/*.pdf`
   Read the source paper after the shortlist is clear.

## Search Policy

- Prefer instrumentation, microbenchmarking, and analytical/predictive-model
  papers before simulator papers.
- Treat simulator papers as prior art for trace representation, validation
  discipline, workload sampling, and error analysis. They are not instructions
  to build or revive a simulator.
- Prefer methods that can operate from user-space observations: NVBit dynamic
  traces, CUPTI Activity records, CUDA events, host wall-clock timing, Nsight
  Systems timelines, static SASS features, and read-only device metadata.
- Reject methods that require NCU metric collection, privileged performance
  counters, root-only profiling policy changes, or `nvidia-smi` clock/power/MIG
  mutation.

## Workflow

1. Define the target observable: wall-clock time, launch timing, L1/L2/cache
   hit-ratio proxy, memory traffic proxy, or per-kernel phase behavior.
2. Identify what the repo can observe in a non-root run.
3. Pick 2-5 papers from `problem_playbook.md` that match the observable and
   method class.
4. Extract assumptions, required inputs, validation method, and reported error.
5. Translate the idea into repository actions: a trace feature, a manifest
   field, a calibration protocol, a prediction target, or a static guard.
6. Keep paper evidence separate from local inference in comments and docs.

## Maintenance

Drop newly recovered PDFs into `inbox/` and move them manually into `active/`
when they are confirmed relevant. Historical metadata rebuild scripts and raw
paper-source lists are retained under `references/papers/scripts/` and
`references/papers/sources/` for provenance; treat them as maintenance aids,
not as active simulator-project direction.
