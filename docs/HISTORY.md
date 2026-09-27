# Project history and stale-context map

Updated 2026-09-26. This is a compact index, not a second status file. Read
[PROJECT_MEMORY](PROJECT_MEMORY.md) and [STATUS](STATUS.md) for current work.
Raw traces, result trees, reference manuals, old plans, and old Codex
transcripts remain retained; this file tells a future session which material
to load and which claims not to revive.

## Current consolidation baseline

* Repository: `/home/jakeshea/GPU_simulator_rebase`, branch `GPUSim_main`.
* Code/evidence baseline before this consolidation: commit `4e090116`
  (`Add SM86 phase-separated speed cohort`); the working tree then contained
  the SM86 bounded-model and documentation changes under review.
* Active source of truth: `AGENTS.md` -> `docs/PROJECT_MEMORY.md` ->
  `docs/STATUS.md` -> the narrow source/runbook needed by the task.
* Current scope: software architectural replay anchored on SM86/RTX 3090 Ti,
  with bounded custom functional/resource frontends. NCU/CUPTI are independent
  labels; no full Accel-Sim/MGPUSim, AMD MI300X, or SM120 execution is implied.

## Current evidence lanes

| Lane | Load first | Do not infer |
| --- | --- | --- |
| Native identity/frontiers | `sm86_holdout_replay_20260926.md` and CSV | elapsed-time accuracy from a modeled frontier |
| Memory/tail diagnostics | `experimental_diagnostics.md`, writeback/coherence model notes | physical completion from a service scenario |
| Speed | `sm86_speed_cohort_20260927.md` | universal speed or simulator ranking |
| Custom execution | `curry_gpu_its_study_20260923.md`, `arch_profiles/programs/README.md` | native SASS/PTX/compiler compatibility |
| Options/prior art | `sm86_sota_simulator_options_20260926.md` | literature numbers as local measurements |
| Module provenance | `sm86_module_origin_followup_20260926.md` | stable identity for JIT/deleted/opaque/graph-owned code |

## Dated reports

Reports named with `YYYYMMDD` are retained observations. They may contain
development counters, old paths, or superseded interpretations. Prefer the
current summaries above; only open a dated report when its row is needed.

* `sm86_measurement_boundary*`, `sm86_metric_*`, `sm86_full_metric_holdout*`:
  SM86/NCU mappings and metric-domain evidence.
* `sm86_writeback_experiment*`, `sm86_coherence_domain*`,
  `sm86_memory_topology*`, `sm86_*completion_model*`:
  memory topology and explicitly incomplete tail scenarios.
* `sm86_speed_*`, `sm86_issue_width_design_study*`:
  replay turnaround and negative resource-width results.
* `sm86_branch_*`, `sm86_pipeline_*`, `sm86_physical_pipeline_*`:
  branch and candidate-domain observations, not dynamic hardware attribution.
* `sm86_module_origin_*`, `nvbit_build_blocker_*`, `nvprof_hyfiss_*`:
  producer/toolchain/provenance and profiler-crosswalk records.
* `adversarial_*`, `roadmap_adversarial_*`, `transition_soundness_*`,
  `sm86_replay_determinism*`: historical review/debug records, not active gates.

## Plans and historical context

`plans/cross_architecture_gpu_simulator_research_plan_20260923.md` is the only
active roadmap. The other plans are historical context for acceleration,
heterogeneous/AMD ideas, old baseline campaigns, or typed-L2/power proposals;
they do not create implementation obligations. `docs/archive/` and retained
SOL/campaign trees are preserved but inactive unless the user explicitly asks
for them.

## Session/context cleanup

The current Codex task is titled **SM86 documentation consolidation and
validated handoff**. Earlier GPU-simulator research sessions were renamed as
historical context and should not be loaded by default. Raw Codex session
records remain outside the repository and are not rewritten. The project-local
memory mirror is `~/.codex/memories/gpu_simulator_rebase.md`; it points here
and must stay shorter than this file. A future session should not use old
“complete”, “SOTA”, review-loop, receipt, hash-gate, or historical-branch
language as current instructions.

## Consolidation policy

We consolidate entry-point docs and point at dated evidence rather than delete
retained data. Configuration/skill files may mention optional knowledge cards;
when `.claude/knowledge/` is absent, tracked docs/source/references are the
fallback. No repository hook, approval gate, archive deletion, or session-log
mutation is required for this cleanup.

## 2026-09-26 consolidation pass

The active handoff was rebuilt from source and executable evidence. Entry-point
docs were shortened; diagnostic notes were merged; prior “closed” wording for
scenario formulas, candidate pipe balancing, the reused NCU table, and the
tail-sensitivity study was corrected. Historical GPU-focused Codex tasks were
archived through task management; raw session records remain available. The
project-local memory mirror was reduced to a compact summary and points back to
this repository's memory/status files. No global Codex config, hooks, schedules,
or unrelated sessions were changed.

Validation after consolidation:

```text
cargo test --manifest-path sim/Cargo.toml --workspace --quiet                   PASS
cargo test --manifest-path sim/Cargo.toml -p sim-sm --features campaign --quiet PASS
git diff --check                                                               PASS
```

The skill-creator validator could not run because the available Python
environments lack PyYAML; edited frontmatter was inspected manually. This is a
tool-environment limitation, not a validator pass claim.

## 2026-09-26 trace-contract refinement

The bounded `custom_its_dynamic_v1` output now reports the accumulated UTF-8
JSON byte length of its individual records, enforces that 32 MiB record budget
in addition to the 100,000-record cap, and preserves partially arrived barrier
lanes in `lane_blocked_by`. The schema, executable diagnostics note, project
memory and program guide were updated together. A focused reconvergence test
and a wide-store stress probe cover the blocked-lane and serialized-byte
failure paths; the result envelope remains outside the byte accounting.

Additional validation after the baseline commit:

```text
cargo test --manifest-path sim/Cargo.toml --workspace --quiet                   PASS
cargo test --manifest-path sim/Cargo.toml -p sim-sm --features campaign --quiet PASS
cargo build --release --manifest-path sim/Cargo.toml -p sim-run -p sim-arch   PASS
arch_profiles JSON/schema validation + custom ITS trace validation             PASS
retained SM86 address-carry canary replay (l1-l2, ncu-shaped metrics)          PASS
`sim-arch` strict Clippy                                                     PASS
```

Strict workspace Clippy remains red on pre-existing `sim-sm` findings such as
large error variants and high-arity replay helpers; the changed ITS package is
clean under `-D warnings`. This is recorded as lint debt, not hidden as a
validation pass.

The follow-up ITS regression runs one program against all-true and mixed loaded
predicate inputs, verifies different successor masks and lane results, and
leaves native SASS regeneration explicitly out of scope.
