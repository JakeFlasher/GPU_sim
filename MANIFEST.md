# MANIFEST — Active Repository Scope

> **CURRENT STATUS — HISTORICAL DUPLICATE; NON-OPERATIVE (2026-07-18).**
> For the active repository scope use
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
> [`docs/STATUS.md`](docs/STATUS.md),
> [`docs/COMPONENTS.md`](docs/COMPONENTS.md), and the accepted
> [`ADRs`](docs/decisions/README.md), especially ADR 0001 and ADR 0002. All text
> below that presents retired predictor/no-root/no-simulator-revival policy,
> claims cycles are DVFS-stable, or describes a then-current repository state is
> historical and non-operative. It is not an active deletion manifest or scope
> authority. The historical body is intentionally preserved.

This repository was cleaned on May 3, 2026 to remove historical simulator
project material. The active scope is now non-root GPU measurement and
prediction for wall-clock behavior and cache hit ratio.

## Active Contents

### `tools/`

- `tools/nvbit_tracer/` — NVBit 1.8 tracer source and support code for dynamic
  SASS/address evidence. NVBit is the primary user-space instrumentation path.
- `tools/cloud/` — cloud-container bootstrap, preflight, provenance, capability
  checks, and trace-budget helpers. These scripts are restricted to non-root
  GPU-driver behavior.
- `tools/local_labels/` — local-only labeling driver that invokes Nsight
  Compute on local devices (RTX 3090 Ti sm_86, RTX 5060 laptop sm_120) and
  emits per-launch label sidecars into the `data/local_labels/` quarantine
  path. This subtree is the ONLY place in the repository where NCU
  invocation is permitted; it is bounded by the local-label exception in
  `CLAUDE.md` Rule 1 and the scoped-allowlist test in
  `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`.

### `data/`

- `data/local_labels/` — quarantine directory for NCU-derived ground-truth
  labels emitted by `tools/local_labels/`. Contains per-launch JSON sidecars
  and `.ncu-rep*` artifacts. Excluded from the no-NCU static guard scan;
  no other path in the repository may write into this directory.

### `workloads/`

- `workloads/simbench_bw/` — active benchmark and evidence harness. It carries
  the kernel roster, shape files, correctness references, CUPTI Activity
  collector, NVBit adapters, dual-run manifest logic, validation schemas, and
  tests.
- `workloads/portable_lcd/` — second active benchmark registry with
  portable-LCD-discipline kernels (sm_86 / sm_90 / sm_120 baseline; no
  architecture-conditional `sm_90a` / `sm_120a` / `sm_120f` build targets).
  Carries its own roster, shape files, makefile preflight, kernel-contract
  verifier, and tests. Reintroduced after the May 2026 cleanup.

### `references/`

- `references/NVIDIA_Manuals/` — local CUDA/PTX/SASS/NVIDIA documentation
  mirror. Nsight Compute documentation is retained only to understand metric
  definitions that this project aims to replace without using NCU.
- `references/papers/` — prior-art paper corpus for performance prediction,
  microbenchmarking, instrumentation, modeling, and simulator literature.
  Simulator papers are prior art only; they are not the project direction.
- `references/Rust_refs/` — retained source-material references from the fork
  history. They are not active project direction.

### `.claude/`

- `.claude/knowledge/` — retained pre-built knowledge-card layer with links
  back to source material under `references/`.
- `.claude/hooks/`, `.claude/skills/`, and `.claude/settings.json` — retained
  Claude support files from the fork.
- `.claude/memory/` — intentionally removed.

## Removed During Cleanup

- `.claude/memory/` project memories.
- `.humanize/` archived RLCR outputs and cloud evidence logs.
- `plans/` roadmaps, research pillars, drafts, and simulator-validation step
  plans.
- `gpu-simulator-non-privileged-research.md`.
- `scripts/kb/` Claude knowledge-base routing helpers.
- `tools/perturbation_sweep/`, `tools/build/`, and the historical
  `tools/cloud/CLOUD_LCD_RUNBOOK.md`.
- Old generated `results/`.
- `.pytest_cache/`, `__pycache__/`, and other regeneratable tool state where
  encountered.

Note: `workloads/portable_lcd/` was removed during the May 2026 cleanup
and has since been reintroduced as an active benchmark registry. See the
`workloads/` entry in "Active Contents" above.

Note: `.humanize/` was re-populated after the cleanup as a host-local
workspace for the Humanize RLCR loop tooling. It now contains
`.humanize/bitlesson.md` (gitignored project-level lesson knowledge base
per `BL-20260507-tracked-doc-hygiene-after-substantive-work` and the
related lesson family; survives raw scp but NOT git fetch),
`.humanize/rlcr/<session>/` (gitignored per-session loop state; host-local
and MUST be deleted on the receiving host per CLAUDE.md "Session
Handoff"), `.humanize/ideas/` (idea drafts produced by `/humanize:gen-idea`
that feed `/humanize:gen-plan`), and `.humanize/skill/` (per-skill
output transcripts). `plans/` was likewise re-populated after the
cleanup with the active plan tree. The current active plan (as of
2026-06-02) is
`plans/sm86_submission_grade_evidence_expansion_plan.md` (the sm_86
submission-grade evidence expansion; tasks 1–6 committed, tasks 7–13
remaining), alongside its wall-clock timing leg
`plans/sm86_nonroot_wallclock_prediction_plan.md` and the research
dossier under `plans/research/`. A longer historical plan chain (the
m1/m2 lower-bound + label-stabilization plans and the v3/v6/v7/v8
development chain) is retained as durable RLCR history and documented
in `HANDOFF.md`'s historical-context sections; the `sm120_*`,
`cross_arch_*`, `tier2_rd_calibration_cross_arch*`, and
`arch_config_unification_merge_prep*` plans are retained cross-arch
design for the deferred sm_120 study (see CLAUDE.md "Two-Device Local
Development Model"). `scripts/kb/` was likewise re-populated as the
Claude KB routing layer wired to `UserPromptSubmit` (see
`.claude/knowledge/INDEX.md`). `HANDOFF.md` at the repo root is the
operative cross-host transfer document per CLAUDE.md "Session
Handoff"; its current state reflects the sm_86 submission work on the
RTX 3090 Ti host, with the cross-arch sm_120 study deferred until the
repo physically moves to the 5060-laptop host.

## Guardrails

The repo remains under the non-root GPU measurement contract for every
cloud or server collection script: no NCU on cloud/server GPUs, no
hardware counter dependency, no GPU clock locking, no driver-module
changes, and no host-driver installation assumptions.

A scoped local-label exception applies under `tools/local_labels/` and
`data/local_labels/`: NCU MAY be invoked there to produce ground-truth
labels on local devices (RTX 3090 Ti sm_86 and RTX 5060 laptop sm_120).
The exception is bounded by `CLAUDE.md` Rule 1 and enforced by the
static guard at
`workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`. Rules
2-5 (no driver-root, no privileged CUPTI, no host-driver work, no
simulator revival) remain absolute everywhere, including inside the
local-label subtree.
