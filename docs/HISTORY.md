# Project history and stale-context map

Updated 2026-09-27. This is a compact index, not a second status file. Read
[PROJECT_MEMORY](PROJECT_MEMORY.md) and [STATUS](STATUS.md) for current work.
Raw traces, result trees, reference manuals, old plans, and old Codex
transcripts remain retained; this file tells a future session which material
to load and which claims not to revive.

## Current consolidation baseline

* Repository: `/home/jakeshea/GPU_simulator_rebase`, branch `GPUSim_main`.
* Code/evidence baseline before this continuation: commit `2d761986`
  (`Add PhaseFuse fast SM86 frontier lane`); the working tree then contained
  the bounded route/compiler and documentation changes under review.
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

## 2026-09-27 SOTA review and PhaseFuse prototype

Fresh shallow clones of Accel-Sim, GPGPU-Sim, MGPUSim, Vortex, EventQueue and
PIMSIM-NN were inspected outside the repository, alongside primary PDFs for
Accel-Sim 2020, Accel-Sim 2.0, MGPUSim, Vortex, EQueue, PIMSIM-NN and deterministic
Accel-Sim parallelization. No external large-workload simulator was built or
run. The resulting design note is [PhaseFuse](sm86_fast_exact_prototype_20260927.md).

The repository now has an opt-in `sim-run --fast-frontier-only` path. It reads
the actual SM86 trace/control/card, skips native timing replay, and emits a
typed scalar-region certificate/refusal report. Retained canary/GEMM/LayerNorm
fast-only smoke runs completed successfully; the ordinary event replay remains
the timing reference. This closes the missing producer/CLI seam for the narrow
causal analyzer, not any native hardware gap.

Validation for this pass:

```text
cargo test --manifest-path sim/Cargo.toml --workspace --quiet                 PASS
cargo test --manifest-path sim/Cargo.toml -p sim-sm --features campaign --quiet PASS
cargo clippy --manifest-path sim/Cargo.toml -p sim-arch --all-targets -- -D warnings PASS
fast_frontier.v1 schema validation on retained canary output                 PASS
ordinary SM86 flat/l1-l2 canary replay (54 records each)                    PASS
```

Strict workspace Clippy remains blocked by existing `sim-sm` error-size and
high-arity findings; no new PhaseFuse-specific lint was observed. External
SOTA repositories were cloned and inspected only; no large external simulation
was run.

The follow-up continuation then bumped the opt-in trace contract to
`custom_its_dynamic_v2`, adding grouped `successor_masks` and explicit
BSSY/BSYNC join/resume PCs for functional CFG experiments. Native SASS
regeneration remains explicitly unavailable.

## 2026-09-27 bounded routing/compiler continuation

The custom architecture schema now accepts scalar or alternative resource
bindings. The shared event pipeline chooses a concrete allowed resource using
modeled latency, initiation interval and in-flight capacity, exposes the
selection in timing/resource reports, and preserves scalar JSON compatibility.
This closes a controlled custom scheduling seam only; native SM86 physical-pipe
identity remains unavailable in the trace contract.

The `ptx_subset_v1` frontend and `ptx_subset_compile` CLI now lower a bounded
PTX-like subset into validated `simt_its_v1` programs with explicit source,
instruction, memory and step limits. Mixed signed memory inputs exercise
value-dependent successor masks and output stores. Unsupported directives,
predication, calls, atomics, ABI forms and arbitrary PTX are rejected.

Fresh source inspection also covered FlashGPU-Sim and OpenVADL/LLVM generation
patterns. No full external simulator or large workload was run. Native
coherence/resident dirty-flush completion, physical-pipe reconstruction,
execution-driven SASS regeneration and arbitrary ISA/compiler generation remain
honest gaps. A
transactional `sim-mem::writeback::WritebackQueue` was added as a reusable
dirty-sector mechanism primitive. The follow-up adds an explicit direct
`sim-run --dirty-writeback-capacity-sectors N` launch path using one shared
read/write byte service and separate completion counters; coherence
ownership/visibility and resident dirty flush remain deliberately unclaimed.

Validation for this continuation:

```text
cargo test --manifest-path sim/Cargo.toml --workspace --quiet                  PASS
cargo test --manifest-path sim/Cargo.toml -p sim-sm --features campaign --quiet PASS
cargo clippy --manifest-path sim/Cargo.toml -p sim-arch --all-targets -- -D warnings PASS
cargo clippy --manifest-path sim/Cargo.toml -p sim-engine --all-targets -- -D warnings PASS
cargo clippy --manifest-path sim/Cargo.toml -p sim-mem --lib -- -D warnings PASS
PTX subset -> ITS v2 mixed-input execution and JSON-schema validation          PASS
alternative-resource route executable smoke and selected-resource checks       PASS
```

Strict `sim-sm` and workspace Clippy remain red on pre-existing error-size,
argument-count and complexity findings; the changed `sim-arch`, `sim-engine`
and writeback library targets are clean under their focused `-D warnings`
checks.

The follow-up native writeback checks additionally passed the synthetic
single-way dirty-eviction system case, direct `sim-run` carded read/write
equivalence on the retained SM86 canary, and release CLI execution with an
explicit writeback capacity. No large external simulator workload was run.

The next bounded memory slice adds `--coherence-service-cycles N` plus
`--coherence-capacity-sectors N`. It refuses flat or incomplete decoded
topology, tracks ordered L2-request directory transitions, waits response
service behind cross-SM transfers, and reports separate completion counters;
L1-hidden accesses, fences, resident flushes and protocol ownership remain
explicit limits.

Validation for this slice:

```text
cargo test --manifest-path sim/Cargo.toml --workspace --quiet                  PASS
cargo test --manifest-path sim/Cargo.toml -p sim-sm --features campaign --quiet PASS
cargo test --manifest-path sim/Cargo.toml -p sim-mem --lib --quiet            PASS
synthetic cross-SM owner-transfer directory completion                         PASS
release direct coherence/writeback CLI smoke and default read equivalence       PASS
```

The requirement-by-requirement follow-up campaign is recorded in
[`plans/sm86_next_experiment_campaign_20260927.md`](../plans/sm86_next_experiment_campaign_20260927.md).
It separates local litmus/refusal work from blocked RTX 3090 Ti/NCU holdouts
and matched external baselines, with explicit no-go conditions for unsupported
speed or accuracy claims.
