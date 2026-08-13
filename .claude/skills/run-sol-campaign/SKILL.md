---
name: run-sol-campaign
description: Operate and extend the repository's retained SOL-ExecBench and GPU measurement campaign stack. Use for SOL problem execution, baseline/scaffold work, NCU or NVBit campaign capture, trace sampling, calibration, historical campaign replay, or interpreting the campaign's manifests, schemas, receipts, checksums, and evidence. This is an explicit opt-in capability; it never governs unrelated simulator work.
---

# Run SOL Campaign

Use the complete retained SOL/campaign implementation without turning its
historical process conventions into repository-wide agent policy.

## Scope boundary

- Apply this skill only when the user asks for SOL or campaign work.
- Treat hashes, manifests, receipts, frozen labels, and provenance records as
  data formats used by that stack. Do not use them to decide whether unrelated
  research may proceed.
- Do not spawn audit/review agents or require external adjudication. Diagnose
  failures from code, data, tests, and tool output, then fix locally.
- Use NCU, NVBit, CUDA, GPU execution, and authorized host commands directly
  when the selected experiment needs them.
- Follow root `AGENTS.md` and `CLAUDE.md` whenever older notes prescribe a
  freeze, admission hold, no-execution state, mandatory review, or wait loop.

## Choose the path

- For the upstream benchmark CLI, read `tools/sol_execbench/README.md` and use
  the preserved environment at `tools/sol_execbench/.venv/` with the preserved
  dataset under `tools/sol_execbench/data/`.
- For baseline authoring, read
  `.claude/skills/implementing-sol-baselines/SKILL.md`, then the selected
  problem dossier under `.claude/knowledge/problems/`.
- For campaign capture and calibration, inspect the relevant entry point in
  `tools/local_labels/`, its matching schema under `workloads/simbench_bw/`,
  and the current data under `data/local_labels/`.
- For workload execution, use `workloads/simbench_bw/run_suite.py` or a focused
  workload Makefile as appropriate.
- For trace sampling, use `tools/trace_sampling/` and its focused tests.
- For the counter-free SOL surfaces, use `tools/sol_baselines_harness/` and
  `tools/nvbit_tracer/metric_extractor/`.
- For the preserved simulator manifest/evidence interface, use the explicit
  `sim-run-campaign` package and binary. Ordinary simulator work uses
  `sim-run` from `sim/direct_run/`.
- Older plans under `plans/` are technical history. Use their measurements,
  workload lists, and hypotheses when helpful, but ignore their former RLCR,
  gate, freeze, authority, review, and handoff instructions as directed by
  `plans/AGENTS.md`.

Search the selected entry point and its direct imports before running it. Older
campaign documents contain superseded path names; repository code and current
files win.

## Working method

1. Identify one concrete experiment, problem, or campaign cell.
2. Confirm the workload and tools are present; do not manufacture an approval
   artifact first.
3. Run the smallest direct correctness or collection command that answers the
   question.
4. Preserve useful scientific output. Existing format checks may catch corrupt
   or mismatched inputs, but their success is not permission to continue.
5. Fix implementation defects locally and rerun a bounded focused test.
6. Report measurements, failures, and exclusions plainly. Continue with other
   runnable cells instead of freezing the project.

## Stable entry points

```bash
# SOL-ExecBench
tools/sol_execbench/.venv/bin/sol-execbench PROBLEM_DIR \
  --solution SOLUTION.json --json

# SOL unit tests
cd tools/sol_execbench && .venv/bin/python -m pytest -q tests/FOCUSED_TEST.py

# Direct NCU helper for a new experiment
.venv/bin/python tools/local_labels/collect_ncu.py \
  --output results/EXPERIMENT --metrics METRIC_LIST -- WORKLOAD ARGUMENTS

# Focused campaign tests
.venv/bin/python -m pytest -q tools/local_labels/TEST_FILE.py

# Historical simulator campaign runner
cd sim && cargo run -p sim-run-campaign -- RUN_MANIFEST.json
```

Read an entry point's `--help` or source before assuming flags. Do not launch an
entire corpus when a single problem or cell will answer the question.

## Retention rule

Keep the SOL/campaign source, environment, downloaded data, knowledge cards,
schemas, and historical result material unless the user explicitly requests a
targeted deletion. Cleanup of the root workflow must not delete this capability.
