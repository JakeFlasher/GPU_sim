# Project history and stale-context map

Updated 2026-09-28. This is a compact index, not a second status file. Read
[PROJECT_MEMORY](PROJECT_MEMORY.md) and [STATUS](STATUS.md) for current work.
Raw traces, result trees, reference manuals, old plans, and old Codex
transcripts remain retained; this file tells a future session which material
to load and which claims not to revive.

## Current consolidation baseline

* Repository: `/home/jakeshea/GPU_simulator_rebase`, branch `GPUSim_main`.
* The bounded-campaign continuation is represented by the earlier commits
  `9614f62b`, `e9d750f6`, `506c6d6e`, `5112ae23`, `0731a292`, and `32785a6e`;
  inspect `HEAD` for the current tip. The latest consolidation commit carries
  the current alpha mechanisms and handoff docs. Commit
  `2d761986` is the earlier PhaseFuse starting point, not the current evidence
  baseline.
* Active source of truth: `AGENTS.md` -> `docs/PROJECT_MEMORY.md` ->
  `docs/STATUS.md` -> the narrow source/runbook needed by the task.
* Current scope: software architectural replay anchored on SM86/RTX 3090 Ti,
  with bounded custom functional/resource frontends. NCU/CUPTI are the local
  ground-truth measurements; CUPTI remains independent launch/resource
  observation; published/collected Accel-Sim aggregates are registered
  as accuracy ceilings and throughput/speedup floors to meet or surpass in
  `accelsim_literature_lower_bound_manifest_20260928.json`; no same-card
  executable Accel-Sim/MGPUSim cycle score, AMD MI300X, or SM120 execution is
  implied.

## Current evidence lanes

| Lane | Load first | Do not infer |
| --- | --- | --- |
| Native identity/frontiers | `sm86_holdout_replay_20260926.md` and CSV | elapsed-time accuracy from a modeled frontier |
| Memory/tail diagnostics | `experimental_diagnostics.md`, writeback/coherence model notes | physical completion from a service scenario |
| Speed | `sm86_speed_cohort_20260927.md` | universal speed or simulator ranking |
| Custom execution | `curry_gpu_its_study_20260923.md`, `arch_profiles/programs/README.md` | native SASS/PTX/compiler compatibility |
| Options/prior art | `sm86_sota_simulator_options_20260926.md` | literature numbers as local measurements |
| Module provenance | `sm86_module_origin_followup_20260926.md` | stable identity for JIT/deleted/opaque/graph-owned code |

## 2026-09-28 alpha campaign, adversarial review, and handoff trim

The alpha development lane completed the combined base/expanded SM86 campaign
on the local RTX 3090 Ti: 15/15 workloads completed through native CUDA-event
timing, NVBit capture, simulator replay, and repeated NCU collection. The
retained performance table is the ignored local file
`results/alpha/full_3090ti_20260928_r3/performance.csv`; its medians and domain
separation are summarized in `docs/PROJECT_MEMORY.md`. Companion metric-domain
and writeback campaigns remain under the ignored `results/alpha/` tree.

The adversarial source review fixed sequence-state poisoning in the memory-order
ledger, unbounded payload-sidecar reads, duplicate/mixed-generation payload
writeback, L2 dirty-flush set lookup, L1 visibility sector qualification,
overbroad fence classification, pipe-identifiability truncation, stale alpha
binary reuse, fragile NCU/control parsing, and conflated L1/coherence event
counters. The review ledger is `docs/code_adversarial_review_20260928.md`.

Alpha adapters, value capture, and campaign outputs are path/size-only by
default; historical content-bound forms require explicit `--bound`. No test
suite, verification gate, or hash-bound campaign was run for this handoff. The
Rust affected path received only a build-only compile check. The exact native
closure blockers remain unchanged: physical pipes, unrestricted compiler/ABI,
native cache ownership/flush semantics, drift-free locked-clock NCU timing, and
fully matched external timing domains.

## Dated reports

Reports named with `YYYYMMDD` are retained observations. They may contain
development counters, old paths, or superseded interpretations. Prefer the
current summaries above; only open a dated report when its row is needed.

* `sm86_measurement_boundary*`, `sm86_metric_*`, `sm86_full_metric_holdout*`,
  `sm86_independent_holdout_20260927.md`:
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

`plans/sm86_next_experiment_campaign_20260927.md` is the active bounded
experiment/acceptance plan; `plans/cross_architecture_gpu_simulator_research_plan_20260923.md`
is the broader design roadmap. The other plans are historical context for
acceleration, heterogeneous/AMD ideas, old baseline campaigns, or typed-L2/
power proposals; they do not create implementation obligations. `docs/archive/`
and retained SOL/campaign trees are preserved but inactive unless the user
explicitly asks for them.

## Session/context cleanup

The prior Codex task was titled **SM86 documentation consolidation and
validated handoff**. Earlier GPU-simulator research sessions were renamed as
historical context and should not be loaded by default. Raw Codex session
records remain outside the repository and are not rewritten. The project-local
memory mirror is `~/.codex/memories/gpu_simulator_rebase.md`; it points here
and must stay shorter than this file. A future session should not use old
“complete”, “SOTA”, review-loop, receipt, hash-gate, or historical-branch
language as current instructions.

## 2026-09-28 interpretation correction: Accel-Sim roles

Earlier dated baseline notes use “non-scoring” and “matched baseline” for the
optional pinned Accel-Sim executable/derived-card cycle probes. Those facts and
their numbers remain preserved, but that wording must not reject the separate
published-aggregate lane. The current contract is: NCU on the local RTX 3090 Ti
is detailed ground truth; the Accel-Sim literature manifest records aggregate
accuracy ceilings and throughput/speedup floors that the project must meet or
surpass; and the pinned executable is parser/representation smoke unless a
matched direct timing endpoint is independently established. Raw Codex session
logs and the session index remain archival; only the project-local memory mirror
is updated for this correction.

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
L1-hidden accesses (including same-launch L1-hit invalidation after a remote
writer), fences, resident flushes and protocol ownership remain
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

## 2026-09-27 bounded-campaign correctness follow-up

The first Phase-0 correctness slice is now implemented and tested. The
coherence queue validates its constructor-only positive configuration, keeps
directory ownership pending until the service-completion event, rejects a
same-line request while visibility is pending, and gives zero-service
read/read requests their own issue-cycle readiness instead of inheriting a
global service head-of-line delay. Native integration now maps the physical
trace `sm_id`, excludes `local_sector_mask` bits from the global directory,
keeps read/writeback service ordering orthogonal to the coherence lane, and
excludes writeback/coherence-only events from the SM-active terminal frontier.
Texture accesses are fail-closed as an unknown global-coherence domain rather
than being counted as ordinary global reads. Coherence replay rejects sampled
coverage and incomplete decoded/mask topology; the trace reader itself already
rejects missing lane masks before replay.

Bounded evidence from this slice:

```text
cargo test --manifest-path sim/Cargo.toml --workspace --quiet                  PASS
cargo test --manifest-path sim/Cargo.toml -p sim-sm --features campaign --quiet PASS
cargo test --manifest-path sim/Cargo.toml -p sim-mem --lib --quiet            PASS (156)
cargo test --manifest-path sim/Cargo.toml -p sim-run --lib --quiet            PASS (51)
coherence HOL/pending-visibility/texture/refusal litmus tests                  PASS
SM86 retained canary flat replay (54 records; 0.01 s shell time)               PASS
SM86 retained canary coherence-option replay (54 records; 0.00 s shell time)   PASS
SM86 fast-frontier-only canary (1 block, 1 warp; 0.00 s shell time)            PASS; 0 certificates
```

These are deterministic mechanism/refusal checks, not independent RTX 3090 Ti
or NCU accuracy measurements. The directory still starts cold and cannot
import hidden L1/L2 ownership, resident dirty flushes, fences, or a native
coherence protocol. PhaseFuse now requires static register projections for
RAW/WAW-free scalar certificates and gathers the complete traced SM roster;
unknown register provenance and slot reuse remain refusals. Full workspace
Clippy remains red only on pre-existing `sim-sm` high-arity/large-error lint
debt; the changed `sim-mem` target is clean under `-D warnings`.

The per-experiment denominators, refusal classes, output digests and blocked
T2/T3 gates are retained in
[`docs/sm86_next_experiment_results_20260927.md`](sm86_next_experiment_results_20260927.md).

The bounded E6 PhaseFuse scale probe additionally covers 112 synthetic process
descriptor cases (`N=1k..64k`, four region lengths, complete/unknown residency,
sorted/shuffled order) with one child per case. All 112 exited successfully and
preserved terminal state; complete residency collapsed only eligible regions,
unknown residency collapsed none, and `region_len=1` collapsed none. This is
descriptor/analyzer scaling with RSS, not native trace parsing or silicon
speedup.

## 2026-09-27 live holdout and PhaseFuse boundary correction

The host audit confirmed a live RTX 3090 Ti (UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, SM86, driver 615.71.09), and the
existing small metric-domain campaign was rerun as 13 one-block/one-warp
read/atomic/reduction/control/store cases with NVBit and NCU. The source/case
population is reused development material, not an independent holdout. Warp instruction,
global-store, atomic and reduction counter populations matched; clocks were not
locked and DRAM domains differed, so the result is non-scoring counter-contract
evidence. Details and content identities are in
[`docs/sm86_independent_holdout_20260927.md`](sm86_independent_holdout_20260927.md).
The locked-clock follow-up is a typed prerequisite blocker: `sudo -n true` is
unavailable and the attempted `nvidia-smi -rgc`/`-rmc` reset probes were denied
for this user. No clock state was changed or claimed restored.

The matched-baseline audit also ran a three-record one-CTA Accel-Sim smoke
(`gpu_tot_sim_cycle=5024`, `gpu_tot_sim_insn=96`, one CTA) after discovering
that legacy `.traceg` parsing requires an explicit `RZ`→`R255` normalization.
This is a T1 executable compatibility datum, not an SM86/NCU timing score; the
strict E10A–E10C adapter/refusal contract is in
[`docs/sm86_external_baseline_contract_20260927.md`](sm86_external_baseline_contract_20260927.md).
The strict `sm86_traceg_adapter_probe_v2` now round-trips a synthetic two-record
full-mask `IADD3` fixture with an actual dynamic `EXIT` and refuses the real
holdout's unsupported `IMAD.MOV.U32` at record 0. The probe is a self-parser
round trip, not independent legacy-parser validation.

An adversarial PhaseFuse review found and corrected a P0 false-certificate
path: sampled/multi-launch traces, non-full lane populations, multi-block or
multi-warp rosters, and suffix regions after discarded prefixes now refuse;
single-launch empty-boundary assumptions are explicit. The fast lane returns
typed/refusal summaries rather than treating those contexts as fresh pipeline
state.

The hardened PhaseFuse wrapper was rerun for five repeats across all 112 matrix
cases (560 rows, zero failures/timeouts/truncations); executable/source
identities are recorded by the wrapper and in the latest E6 ledger row. The
scaling result remains descriptor/analyzer evidence, not native trace timing.

The final adapter hardening narrowed E10A to a strict v2 ALU-only,
control-sidecar-driven representation probe. It requires an actual dynamic
`EXIT`, refuses memory/branches/predicates and EOF, bounds all inputs, refuses
output overwrite, and has 24 pinned-protobuf adversarial tests (15 default
tests plus 9 missing-runtime skips). Its self-parser result is not independent
legacy-parser validation.
