# HANDOFF.md — Cross-Host Session Transfer

> **CURRENT STATUS — HISTORICAL DUPLICATE; NON-OPERATIVE (2026-07-18).**
> This is not the current handoff or project-state authority. Use
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
> [`docs/STATUS.md`](docs/STATUS.md),
> [`docs/COMPONENTS.md`](docs/COMPONENTS.md),
> [`docs/runbook_host.md`](docs/runbook_host.md), and the accepted
> [`ADRs`](docs/decisions/README.md), especially ADR 0001 and ADR 0002. All text
> below that presents retired predictor/no-root/no-simulator-revival policy,
> claims cycles are DVFS-stable, or describes a then-current transfer/project
> state is historical and non-operative. The historical body is intentionally
> preserved.

> **Do not execute or treat any instruction below as current.** The historical
> body is retained for provenance only; its old `rm`, privilege, host-transfer,
> and project-state examples are not an operator runbook. Do not use it as a
> source of truth after a cross-host transfer. Start with root `CLAUDE.md`,
> `AGENTS.md` for consultation protocol only, `docs/README.md`,
> `docs/STATUS.md`, `docs/runbook_host.md`, and accepted ADRs instead.

> **READ FIRST — CURRENT STATE (2026-06-02).** The project is
> on the **RTX 3090 Ti (sm_86)** host, mid-flight in the
> **sm_86 submission-grade evidence expansion** plan
> (`plans/sm86_submission_grade_evidence_expansion_plan.md`).
> Tasks 1–6 are committed (gap audit, expansion roster, Codex
> roster review, candidate preflight, bundle-readiness
> recovery, and the task6 live-NCU admission ledgers — HEAD
> `30900b50` "half-way there"); **tasks 7–13 remain**
> (slug-disjoint calibration/validation/holdout split, timing
> group-heldout/LOSO, L2 refit under holdout, L1
> promotion-or-typed-non-promotion, baselines+ablations, Codex
> stats review, and the final evidence bundle + SHA-256
> manifest + compliance guards).
>
> The task6 sweep admitted **49 cells / 35 unique slugs**
> (above the AC-4 floor of 31/30), but the downstream
> timing/L1/L2 modeling heads have **not** been re-fit on that
> expanded cohort, so no metric passes its submission gate
> yet (timing AC-6, L1 promotion, and L2 robustness all still
> reflect the pre-expansion baseline).
>
> **TRAP:** the on-disk
> `data/sm_86_predictor_accuracy_recovery_v2/cohort_split.json`
> and `sealed_holdout_labels.json` are STALE prior-loop
> leftovers (frozen at commit `13777154`, splits
> cal=11/val=5/holdout=4 under old floors 10/4/3) — they are
> NOT a completed task7 and do not reconcile with the 49-cell
> cohort. task7 must regenerate the split from the task6
> cohort under AC-5 floors (cal≥15 / val≥8 / holdout≥8).
>
> **Cross-arch (sm_120) is DEFERRED** under DEC-1 until the
> repo physically moves to the 5060-laptop host. The sm_120
> study is out of the current submission claim, but **all
> cross-arch design is retained** (the `data/local_labels/sm_120/`
> trees, the `sm120_*` tooling, the `arch_capabilities`
> registry, and the `cross_arch_*` / `tier2_rd_calibration_cross_arch*`
> / `sm120_only_decoupled_framework*` /
> `arch_config_unification_merge_prep*` plans) to seed the
> later hardware/arch-config decoupling and generic
> NCU-replacement modeling.
>
> **Everything below is HISTORICAL CONTEXT.** The
> `## PRIOR TRANSFER: M1 → M2` section and all sections after
> it describe earlier cross-host transfers (M1→M2, then the
> v3/v6/v7/v8 chain) that brought the project to its current
> HEAD. They are reference for how the project arrived here,
> NOT the current operative transfer.

## PRIOR TRANSFER: M1 → M2 (Historical Context — 5060 → 3090 Ti)

- **Source host**: RTX 5060 laptop (sm_120, Blackwell
  consumer, GB206 die). The M1 lower-bound RLCR loop closed
  here on 2026-05-09.
- **Destination host**: RTX 3090 Ti (sm_86, Ampere desktop,
  GA102 die). The M2 RLCR loop will execute here.
- **Handoff prepared at**: HEAD of branch
  `5060_3090ti_cross_dev` at the M2 plan-landing commit (or
  a descendant). The M2 plan is at
  `plans/cache_predictor_m2_label_stabilization_plan.md`.
- **What just closed on the 5060 (M1 lower-bound loop)**:
  - 8 rounds (R0-R7) of the M1 lower-bound RLCR loop tracked
    against `plans/cache_predictor_m1_lower_bound_plan.md`;
    the loop fired the Mainline Drift Circuit Breaker at
    scope-closure stalemate after R7. Despite the
    circuit-breaker fire, the M1 lower-bound surfaces
    (comparison_runner.py library + CLI wrapper,
    verify_artifacts.py hook, comparison_bundle.schema.json,
    47+15 test methods) all landed and verified. The
    circuit-breaker fire was a review-heuristic-vs-plan-scope
    conflict, not a code defect.
  - Repository state to inherit:
    - `workloads/simbench_bw/comparison_runner.py` —
      M1-produced library + `__main__` CLI wrapper that
      loads `comparison_contract.toml`, joins admitted
      predictor bundles against admitted NCU label cells on
      the contract-pinned 7-field key, normalizes NCU
      percent to fraction, computes the relative-error
      formula, dispatches scalar/bounded/typed_unavailable
      arms, applies coverage gate, emits a deterministic
      schema-valid output bundle. Verified across R6
      (Verified-live-sm_120 evidence on this 5060 laptop —
      smoke-floor 5/5 cells admitted via synthetic
      fixtures; the runner's wire format is
      architecture-independent).
    - `workloads/simbench_bw/verify_artifacts.py`
      `_verify_cache_predictor_run_summaries` hook with 7
      typed defect tokens.
    - `workloads/simbench_bw/schemas/comparison_bundle.schema.json`
      — strict draft-2020-12 output schema with
      allOf[if/then] conditionals.
    - `data/local_labels/label_admitted.json` is
      `{schema_version: 1, cells: []}` — empty admitted set.
      Two prior failed cells live in
      `data/local_labels/label_excluded.json` with
      `typed_reason: label_repeatability_failed`. **AC-5
      (label stabilization) is the M2 work to populate
      this file.**
- **What M2 will close on the receiving 3090 Ti**:
  - The M2 plan binds AC-5 closure to a binary external
    criterion: `label_admitted.json` contains at least 5
    admitted cells per arch with full 7-field join key +
    profiler_version provenance.
  - **Per CLAUDE.md "Two-Device Local Development Model"**
    architecture-coupling discipline ("Codex review is
    one-arch-at-a-time. A round verified live on one arch
    is NOT verified on the other"), AC-5 is split:
    - **AC-5.1 (sm_86 evidence on 3090 Ti)**: REQUIRED for
      M2 lower-bound closure on this receiving session.
      Closes the round at `Verified-live-sm_86`.
    - **AC-5.2 (sm_120 evidence on 5060 laptop)**: QUEUED
      as a follow-on cross-host operator task back to the
      5060 laptop OR a separate M2-prime sm_120 loop.
      Combined AC-5.1 + AC-5.2 lifts to
      `Cross-arch-verified`.
  - The receiving 3090 Ti session MUST NOT claim AC-5
    closure (singular) or `Cross-arch-verified` evidence
    status — only AC-5.1 closure with `Verified-live-sm_86`
    evidence is reachable in this M2 round.
- **`.humanize/rlcr/` on the source 5060 host**: directory
  `.humanize/rlcr/2026-05-09_17-04-56/` carries the M1
  lower-bound loop's R0-R7 contracts/summaries/goal-tracker.
  **NOT portable.** Per CLAUDE.md "Session Handoff": the
  receiving 3090 Ti host MUST `rm -rf .humanize/rlcr/`
  before starting the M2 RLCR loop.
- **`.humanize/bitlesson.md` on the source 5060 host**:
  gitignored; survives raw scp but NOT git fetch. Carries
  the M1-loop-introduced lessons including
  `BL-20260509-strict-admission-contract-first` (test-first
  defect-token lockstep),
  `BL-20260509-rlcr-recovery-round-bounded-scope`
  (R(N+1) scope MUST be bounded BELOW R(N)'s rejected
  scope),
  `BL-20260509-hygiene-validation-grep-must-be-case-insensitive`
  (case-insensitive + multi-file grep validation). The M2
  plan's BitLesson Anchors list these as binding for the
  M2 round.
- **Auto-memory at `~/.claude/projects/.../memory/` on
  the source 5060 host**: machine-local; NOT carried by
  scp OR git fetch. The receiving 3090 Ti session starts
  with empty auto-memory. The M1-closure auto-memory
  entries (`project_loop_state.md` describing M1 closure
  + M2 plan handoff state; the recurring
  review-heuristic-vs-plan-scope conflict pattern observed
  across M1's R5-R7 STALLs) can be carried only via
  separate raw file scp of the memory directory itself
  (independent of repo transport).

### Mandatory First Actions on the Receiving 3090 Ti (M2)

Run these in order, every time, before the first M2 round
contract:

1. **Confirm the repo is at the M2 plan-landing HEAD.**
   ```sh
   git -C "$(pwd)" rev-parse HEAD
   git -C "$(pwd)" status --short    # should be clean
   git -C "$(pwd)" log -1 --oneline -- plans/cache_predictor_m2_label_stabilization_plan.md
                                     # confirms the M2
                                     # plan is tracked at
                                     # current HEAD or a
                                     # descendant
   git -C "$(pwd)" log -1 --oneline -- HANDOFF.md
                                     # confirms this M1→M2
                                     # HANDOFF.md update
                                     # is at HEAD
   ```

2. **Delete the source 5060's RLCR loop state.**
   ```sh
   rm -rf .humanize/rlcr/
   ```
   The M1 lower-bound loop's contracts and summaries are
   bound to the 5060's HEAD and capture artifacts; they
   are not portable.

3. **Verify the receiving host is the 3090 Ti (sm_86).**
   ```sh
   nvidia-smi                       # confirms NVIDIA GeForce
                                    # RTX 3090 Ti
   nvcc --version                   # CUDA toolkit version
   gcc --version                    # host gcc version
   ```
   Per CLAUDE.md "Two-Device Local Development Model":
   the M2 receiving session executes on sm_86; the AC-5.1
   live-sweep evidence is recorded with the receiving
   device + driver + CUDA + observed clocks/throttle
   state.

4. **Rebuild any host-specific binaries that came across
   via raw scp.** Per CLAUDE.md: "the built tracer at
   `tools/nvbit_tracer/tracer_tool/tracer_tool.so`, the
   built CUPTI Activity collector at
   `workloads/simbench_bw/libsimbench_cupti_activity_collector.so`,
   per-slug compiled binaries" do NOT survive cross-host
   transport cleanly — even when raw scp carries them,
   treat them as STALE host artifacts targeting the
   originating arch (sm_120). Rebuild on the 3090 Ti
   before any live evidence:
   ```sh
   # NVBit tracer (exact CUDA/protobuf/compiler pins):
   source tools/nvbit_tracer/pinned_build_env.sh
   make -C tools/nvbit_tracer ARCH=sm_86

   # CUPTI Activity collector library — pin COLLECTOR_ARCH
   # explicitly to sm_86 on this host:
   make -C workloads/simbench_bw cupti-collector-lib \
       COLLECTOR_ARCH=sm_86 \
       NVCC='nvcc -ccbin=/usr/bin/g++-14'

   # Per-slug binaries the M2 sweep will exercise (the M2
   # candidate cell set includes the prior failed cell
   # gemm_hmma_bf16_sm80/correctness_shape on sm_86 +
   # additional candidates the operator picks via the
   # sweep harness's --candidates flag):
   make -C workloads/simbench_bw/l2/gemm_hmma_bf16_sm80 clean all \
       NVCC='nvcc -ccbin=/usr/bin/g++-14'
   # ... and any other M2 candidate cells (see the M2
   # plan's task6 description for the operator's candidate
   # list).
   ```

   **Toolchain framework discipline (cross-host).** Both
   local devices (RTX 3090 Ti sm_86, RTX 5060 laptop
   sm_120) MUST use a native system `g++-14` binary at
   `/usr/bin/g++-14`. Do NOT provision conda / mamba /
   miniforge envs as a substitute. The 3090 Ti host ships
   gcc-15/16 by default, both too new for CUDA 13.2's
   nvcc — install gcc-14 from AUR if not present:
   ```sh
   yay -S gcc14         # or: paru -S gcc14
   ```
   Verify: `which g++-14` returns `/usr/bin/g++-14`.

   Add the export to your shell rc:
   ```sh
   # bash / zsh
   echo 'export NVCC_CCBIN=g++-14' >> ~/.bashrc

   # fish
   echo 'set -x NVCC_CCBIN g++-14' >> ~/.config/fish/config.fish
   ```

5. **Run the host-only baseline sweep.** Confirms
   nothing arch-coupled regressed during scp:
   ```sh
   python -m pytest tools/ workloads/simbench_bw/tests/ -q
   python tools/test_plan_token_scanner.py            # MUST be 23 passed
   ```
   Treat any regression as a triage item before the first
   M2 round contract.

6. **Read the M2 plan in full.**
   `plans/cache_predictor_m2_label_stabilization_plan.md`
   — especially the `## Acceptance Criteria` section's AC-5
   per-arch split (AC-5.1 sm_86 REQUIRED; AC-5.2 sm_120
   QUEUED), the `## Path Boundaries` section's lower bound
   (AC-1 through AC-4 + AC-7 + AC-5.1), and the
   `## Dependencies and Sequence` section's "Cross-Host
   Transfer Prelude" (this exact set of mandatory first
   actions, mirrored in the M2 plan for the receiving
   session's first-round contract reference).

7. **Read `tools/BLOCKED_ON_3090TI.md`** to confirm no
   sm_86-blocked items contradict the M2 candidate cell
   set. The M2 candidate set (4-6 cells per arch including
   the 2 prior failed cells) is operator-chosen via
   `sweep_repeatability_matrix.py --candidates`; if any
   candidate kernel is sm_86-blocked, pick a different
   candidate.

8. **Start the fresh M2 RLCR loop.** Recommended
   invocation:
   ```sh
   /humanize:start-rlcr-loop --max 60 --full-review-round 10 --privacy --skip-quiz --track-plan-file plans/cache_predictor_m2_label_stabilization_plan.md
   ```
   Round 0 contract should re-read the M2 plan + this
   `HANDOFF.md` + `tools/BLOCKED_ON_3090TI.md` +
   `.humanize/bitlesson.md`.

### M2 Punch List on the Receiving 3090 Ti

1. **Stages A through E (architecture-independent
   tooling)**: AC-1 metric-name alignment + AC-2 sweep
   harness + AC-3 provenance enrichment + AC-4 admitted-cell
   schema + AC-7 pre-live runbook. All five stages run
   slice-by-AC test-first per
   `BL-20260509-strict-admission-contract-first` (~10
   commits). Tests use fake-NCU shims; no live GPU
   required for stages A-E.
2. **Stage F1 (AC-5.1 sm_86 live evidence)**: operator
   runs the sweep on the 3090 Ti against an expanded
   candidate cell set (4-6 cells per arch, including the
   prior failed cell `gemm_hmma_bf16_sm80/correctness_shape`).
   Aggregator populates `data/local_labels/label_admitted.json`
   with at least 5 cells where `arch == "sm_86"`. Round
   summary records the architecture-coupling discipline
   metadata (device, driver, CUDA toolkit, observed
   clocks/throttle state) per
   `BL-20260508-post-commit-recapture-discipline`.
3. **Stage F2 (AC-5.2 sm_120 follow-on, NOT in M2 round)**:
   queued for a follow-on cross-host transfer back to the
   5060 laptop OR a separate M2-prime sm_120 loop. The M2
   round on the 3090 Ti CANNOT close AC-5.2 — only AC-5.1.

---

## Prior v8 Handoff (Historical Context — 3090 Ti → 5060)

The sections below document the PRIOR v8 cross-arch
handoff (3090 Ti → 5060) that brought the project to its
current state. The M1 lower-bound RLCR loop ran on the
5060 after this v8 transfer. The receiving 3090 Ti
session executing M2 should treat these sections as
historical reference, NOT as operative direction.

## Prior v8 Source → Destination (Historical)

- **Source host**: RTX 3090 Ti (sm_86, Ampere desktop,
  GA102 die).
- **Destination host**: RTX 5060 laptop (sm_120, Blackwell
  consumer, GB206 die).
- **Handoff prepared at**: HEAD `3c02fca` (the third
  doc-sync commit `fold Codex pass-2 review into doc-sync
  state`, which folded the Codex pass-2 findings on top
  of `5912ab6`. The doc-sync history reaching this HEAD
  is `423375f` (the initial v8-plan + HANDOFF sync;
  Codex pass-1) -> `5912ab6` (the pass-1 fixes +
  remaining tracked-doc sync) -> `3c02fca` (the pass-2
  path/API + provenance fixes). The v8 plan
  `plans/ac16_dram_stream_cupti_hook_sm120_plan.md` is
  tracked at this HEAD), branch `5060_3090ti_cross_dev`,
  2026-05-08. This handoff covers the v3 + v6 + v7 chain
  that closed since the original v3 R26 handoff prep at
  substantive parent `38a353d`. The v6+v7 substantive code
  chain landed at HEAD `5ea52f3` BEFORE any doc-sync
  commit: `9f5a2ca` (v6 R0 dlsym + cudaEvent hook)
  + `14e37bc` (v6 R0 TOML repair) + `eaf5eee` (v6 R0
  regression tests) + `30ae840` (v6 R0 text-contract
  cleanup) + `5ea52f3` (v7 R1 AC-7 hardening). The
  doc-sync commits `423375f`, `5912ab6`, `3c02fca` are
  doc-only (no behavioral change). The git range
  `38a353d..3c02fca` also includes earlier v3-era doc
  commits (`ea1e22b`, `f95c22f`, `85b5691`, `a6547ac`,
  `e65ffe2`, `2fdc563`, etc.) plus v6-era plan commits
  (`1635434` v6 init, `039e663` v3-era plan revert,
  `1050428` + `461493c` v7 plan-only alignment edits) —
  the substantive v6+v7 chain is just the 5 commits
  named above.
- **Loop just closed**: v7 RLCR loop on the 3090 Ti host,
  Rounds 0-4 (the structural-stall chain that fired the
  Mainline Drift Circuit Breaker after 3 consecutive
  STALLED verdicts in Rounds 2/3/4). All 10 plan-scoped
  ACs verified across v7 Rounds 0-1 per Codex Round-1
  ADVANCED verdict (`ACs: 10/10 addressed for the
  plan-scoped sm_86 objective | Forgotten items: 0 |
  Unjustified deferrals: 0`). Subsequent STALLs were
  structural — the loop's stop-hook treated the
  explicitly-deferred 5060-laptop sm_120 replay as
  incomplete despite the deferral being correct per
  CLAUDE.md "Two-Device Local Development Model" + the
  v7 plan IMMUTABLE. **The structural-stall pattern is
  documented as the auto-memory entry
  `~/.claude/projects/.../memory/project_rlcr_structural_stall_pattern.md`
  (machine-local; survives raw scp but is NOT carried by
  git fetch and NOT carried by repo scp either, per
  CLAUDE.md). The pre-v7 v6 RLCR loop closed via
  `/humanize:cancel-rlcr-loop` after introducing the
  dram_stream_stride_mix dlsym hook chain. The pre-v6 v3
  RLCR loop closed at Rounds 0-26 covering the original
  rmsnorm_fwd_bf16 sm_86 work** (see `### v3 R0-R25
  Compact History` below). The v7 plan at
  `plans/ac16_dram_stream_cupti_hook_plan.md` and the v3
  plan at
  `plans/nonroot_gpu_cache_observability_continuation_plan_v3.md`
  are both operative contracts at this handoff: v7 covers
  the dram_stream_stride_mix second-workload AC-16 sm_86
  closure; v3 covers the broader Track B sm_86 closure
  list. The new v8 plan at
  `plans/ac16_dram_stream_cupti_hook_sm120_plan.md` is the
  laptop receiving session's narrowest first cross-arch
  loop (dram_stream_stride_mix/timing_00 sm_120).
- **`.humanize/rlcr/` on the source host**: directory
  `.humanize/rlcr/2026-05-08_08-57-02/` carries the v7
  Round 0-4 contracts/summaries/goal-tracker;
  `.humanize/rlcr/2026-05-07_02-01-13/` carries the v3
  Round 0-26 directory. **Both are NOT portable.** Per
  CLAUDE.md "Session Handoff": delete on the receiving
  host before starting a fresh loop. The v7 plan, v3 plan,
  v8 plan (after commit), `.humanize/bitlesson.md` (gitignored;
  survives raw scp but NOT git fetch),
  `tools/BLOCKED_ON_3090TI.md`, and the live evidence
  artifacts (under `workloads/simbench_bw/results/...`,
  `workloads/portable_lcd/.../traces/...`,
  `workloads/simbench_bw/l1/.../traces/...`,
  `data/local_labels/...`) all DO survive the handoff via
  git, scp, or both — see CLAUDE.md "Session Handoff" for
  the full survival-semantics matrix. **Auto-memory at
  `~/.claude/projects/.../memory/` is machine-local and is
  NOT carried by repo scp OR git fetch** (per CLAUDE.md:
  "Memory at `~/.claude/projects/<project>/memory/` is
  machine-local and is NOT carried by scp or git"). The
  receiving session starts with empty auto-memory; the
  v7-R3 structural-stall lesson
  (`project_rlcr_structural_stall_pattern.md`) and the
  v7-close `project_loop_state.md` / `project_progress.md`
  entries can be carried to the laptop only via separate
  raw file scp of the memory directory itself (independent
  of repo transport).
- **Receiving host's repo**: must `scp` the entire repo
  (or `git fetch && git checkout 5060_3090ti_cross_dev` if
  the receiving host is on the same git remote), then
  **manually delete `.humanize/rlcr/`** before starting a
  fresh loop.
  - **`.humanize/bitlesson.md` (gitignored)** survives raw
    `scp` but NOT `git fetch`. The 9 lessons currently
    captured (BL-20260504 runbook-self-reference,
    BL-20260505 tools-subdir-module-name-collision,
    BL-20260505 cross-arch-sass-mnemonic-divergence,
    BL-20260504 runbook-command-verification,
    BL-20260506 schema-validator-probe-driven-tightening,
    BL-20260507 arch-token-multi-surface-lockstep,
    BL-20260508 post-commit-recapture-discipline,
    BL-20260509 strict-admission-contract-first,
    BL-20260507 tracked-doc-hygiene-after-substantive-work) include
    cross-arch-relevant ones (especially BL-20260505
    SASS mnemonic divergence and BL-20260507 arch-token
    multi-surface lockstep) — preserve via raw scp, OR
    manually copy from the source host after a git
    fetch. Starting fresh is acceptable but the
    recurrence-extraction discipline that produced these
    lessons resets.

## Three-Fold Completion Gate (introduced R25, recorded here R26)

The v3 plan's loop reaches `COMPLETE` only when ALL THREE
of the following paths close concurrently. None is
individually sufficient:

- **Path 1 — Operator sanction of the R18/R19
  prior-round-summary amendment-method process blocker —
  CLOSED via operator sanction granted outside the v3 loop
  (referenced by the v7 plan body's "Path 1 (R18/R19
  amendment-method bypass) was just operator-sanctioned"
  note).** The carried blocker no longer gates loop
  closure. The sanctioned future amendment-path policy
  is captured in the project memory and applied: the v7
  loop's R4 attempted a cross-round summary edit, which
  the loop-edit-validator hook correctly blocked, and the
  workaround was to record the correction in the next
  round's surfaces (the v7 R4 round summary + goal-tracker
  Plan Evolution Log). Future RLCR loops should follow
  the same pattern: NEVER use Python `open(...).write(...)`
  to bypass the hook on prior-round summaries; record
  corrections in the current round's surfaces only.
- **Path 2 — sm_120 replay legs** (THE RECEIVING
  SESSION'S MAINLINE) lifting the 17 producer-side and
  tracker-side surfaces promoted to `Verified-live-sm_86`
  / `Offline-verified` on the 3090 Ti to
  `Verified-live-sm_120`. The surface count grew from 16
  (v3 R0-R25) to 17 with the v6+v7 addition of AC-16
  second-workload-eligibility for
  `dram_stream_stride_mix/timing_00`
  (`Verified-live-sm_86` at v7 HEAD `5ea52f3`). Once both
  arches have matching evidence, the surfaces promote to
  `Cross-arch-verified` per CLAUDE.md "Two-Device Local
  Development Model". This is the receiving session's
  primary reason for existing. **The narrowest first
  cross-arch loop on the laptop is covered by the v8 plan
  at `plans/ac16_dram_stream_cupti_hook_sm120_plan.md`
  (dram_stream_stride_mix/timing_00 sm_120 only).**
  Subsequent laptop loops follow this HANDOFF.md's
  "Cross-Arch Sweep Punch List" for additional sm_120
  legs (rmsnorm_fwd_bf16 AC-15/AC-16, sector-stream,
  divergence-proxies, HRD profiler, etc.).
- **Path 3 — Server/cloud AC closure**: AC-12 (server
  collection bootstrap), AC-16 (server-architecture
  NVML), AC-18 (cloud `tools/cloud/RUNBOOK.md`), plus
  the queued preflight `12.*` prose/device-policy drift
  reconciliation. Explicitly deferred per IMMUTABLE plan
  section ("Server work is out of scope until BOTH local
  devices reach `Cross-arch-verified` on Track B");
  revisit trigger requires path 2 closing first.

The receiving session's `COMPLETE` criterion is therefore:
path 2 closure + scoped server/cloud work (path 3). Path 1
is CLOSED via prior operator sanction (see "Path 1" above
and "Active Side Issues / Carried Blockers" below). What
this HANDOFF.md provides is the path-2 receiving-session
prelude (the documentation precondition CLAUDE.md
mandates the receiving session read first). Path 3
remains explicitly deferred per IMMUTABLE plan ("Server
work is out of scope until BOTH local devices reach
`Cross-arch-verified` on Track B"); its revisit trigger
fires only after path 2 closes via the receiving session.

## v3 R0-R25 Compact History

The 3090 Ti session closed 16 surfaces at
`Verified-live-sm_86` / `Offline-verified`:

- **R0** — sm_86 NVBit tracer rebuild + live trace
  contract sweep (869 passed, 3 skipped, 12 subtests in
  66.72s); C-AC-4A partial promotion.
- **R1-R3** — C-AC-4A (AC-4 sm_86) FULL
  `Verified-live-sm_86` via dram_stream_stride_mix
  workload-backed `--phase-inject-ab` capture; sm_86
  added to common.mk + roster + Makefile + .cu +
  run_suite.py `_ARCH_TOKEN_TO_GENCODE`; arch-token
  multi-surface lockstep regression test added
  (BL-20260507).
- **R4** — C-AC-2A3A (AC-2 / AC-3 sm_86 local-label
  hardware acceptance) closed via typed-exclusion
  verdict per `tools/local_labels/TASK7_RUNBOOK.md`.
- **R5** — C-AC-F2 sector-stream sm_86 evidence
  (769/769 records admit, GLOBAL memory_op_class).
- **R6-R8** — C-AC-G2 divergence-proxies sm_86
  evidence; R7 5-surface manifest_pairing_index
  widening (schema + producer + CLI consumer +
  controller + verifier + tests); R8 post-commit
  recapture closing the carried temporal-lockstep
  gap (BL-20260508).
- **R9** — C-AC-H2 trace-budget sm_86 verifier via
  correctly-does-not-produce arm (slowdown 10.40 < 200,
  drop_rate 0.0 < 0.05).
- **R10** — AC-6 HRD profiler sm_86 (13 bins; 8 ok +
  5 typed-unavailable).
- **R11** — AC-7 sharing-locality sm_86.
- **R12-R13** — AC-8 order-projections sm_86 (6
  projections); R13 added per-sidecar canonical
  regression coverage.
- **R14** — AC-10 perturbation harness sm_86; multi-
  surface widening from sm_120-laptop-only to
  sm_86 + sm_120-laptop (5 lockstep surfaces); 10
  perturbations admit including cache_modifier_cs.
- **R15-R16** — AC-11 adaptive sampler sm_86; R16
  reconciled 4 source-of-truth surfaces with v3 R15
  sm_86 evidence.
- **R17** — AC-14 microbench suite sm_86 across 7
  slugs (9 R51-contract canonical sm_86 shape points).
- **R18** — C-AC-I2 host-only suite re-run on sm_86
  (broad sweep 2022 passed, 2 skipped, 678 subtests).
- **R19** — task11 ledger reconciliation; R18 amendment
  method retraction; carried R18/R19 operator-sanction
  request surfaced.
- **R20-R23** — AC-15/AC-16 producer-side
  `Verified-live-sm_86` (Track D); preflight admission
  widened to sm_86; CUPTI dual-source live admission
  helper introduced and iteratively tightened across
  4 review cycles to a contract-first design with full
  identity contract (4 surfaces × 5 axes = 20 typed
  defect tokens; bidirectional structural lockstep);
  scenario-backed token coverage map introduced
  (BL-20260509). Recalibrated dual-source tolerance
  from 100-sample to 1000-sample pool. Final state at
  R23 commit `38a353d` (this handoff's substantive
  parent): AC-15 combined pairing index `len=25,
  registries={portable_lcd, simbench_bw}, at_head=25/25,
  zero=0`; AC-16 helper admits `ok=True 30/30 CUPTI
  pass, 0 timing failures, 0 defects` for
  `rmsnorm_fwd_bf16/timing_00`.
- **R24** — stand-down round (no autonomously-closeable
  mainline; recorded the four Codex-prescribed
  next-paths).
- **R25** — wording-retraction round closing Codex R24
  review's mainline gap #3 (R24 "sole remaining
  loop-completion gate" overclaim retraction); recorded
  the corrected three-fold completion gate framework.
- **R26** — original HANDOFF.md update (autonomously-
  closeable prelude to the 5060-laptop sm_120 receiving
  session).

The full per-round contracts/summaries/reviews live in
the host-local `.humanize/rlcr/2026-05-07_02-01-13/`
directory which the receiving session deletes per the
mandatory first action below.

## v6 + v7 Compact History (since v3 R26)

After v3 closed at R26, two further loops ran on the
3090 Ti host before this updated handoff:

**v6 RLCR loop (cancelled, dram_stream_stride_mix
dlsym hook chain):**

- Loop session directory `.humanize/rlcr/<v6-session-id>/`
  was deleted before the v7 loop started; the substantive
  commits introduced by v6 R0 are tracked in git:
  - `9f5a2ca` — `dram_stream_stride_mix: add cudaEvent
    wall-clock + dlsym CUPTI collector hook` (instruments
    `workloads/simbench_bw/l1/dram_stream_stride_mix/main.cu`
    with the dlsym + cudaEvent contract that the v3 R23
    `cupti_dual_source_live_admission.py` strict-admission
    helper consumes; mirrors the rmsnorm_fwd_bf16 v3 R20-R23
    pattern)
  - `14e37bc` — `dram_stream_stride_mix.toml: repair
    timing_00 launch_config to match launcher` (corrects
    grid `[1,1,1]` block `[32,1,1]`)
  - `eaf5eee` — `add regression tests for the
    dram_stream_stride_mix dlsym hook` (adds
    `test_dram_stream_stride_mix_hook_static.py` 4
    substring assertions + `CuptiDualSourceAdmissionSecondWorkloadParameterizedTest`
    3 cases)
  - `30ae840` — `clean up two text-contract issues from
    Codex R0 review`
- Loop closed via `/humanize:cancel-rlcr-loop` (the v6
  loop reached plan-scoped completion but did not emit
  COMPLETE; the operator cancelled cleanly).

**v7 RLCR loop (Mainline Drift Circuit Breaker fired):**

- Loop session directory
  `.humanize/rlcr/2026-05-08_08-57-02/` carries the v7
  Round 0-4 contracts/summaries/goal-tracker. **NOT
  portable.** Receiving host deletes it.
- Plan: `plans/ac16_dram_stream_cupti_hook_plan.md` (v7
  plan, 524 lines). Mainline objective: AC-16
  second-workload-eligibility for
  `dram_stream_stride_mix/timing_00` on sm_86, with all
  10 plan ACs covering source instrumentation, build,
  backward-compat, fresh sm_86 capture, strict-admission
  helper round-trip, regression coverage, broad-sweep
  no-regression, status wording, and TOML repair.
- **R0** — verification round. All 10 ACs verified per
  Codex Round-0 ADVANCED (with one mainline gap on AC-7:
  static test was bare-substring `wall_time_ms`; needed
  exact-literal-with-cardinality assertion).
- **R1** — AC-7 hardening: replaced
  `assertIn("wall_time_ms", ...)` with
  `self.text.count(r'printf("  \"wall_time_ms\": %.9f,\n",
  wall_time_ms);') == 2`. Single coding commit `5ea52f3`.
  Codex Round-1 ADVANCED with `ACs: 10/10 addressed for
  the plan-scoped sm_86 objective | Forgotten items: 0 |
  Unjustified deferrals: 0`. **AC-16
  second-workload-eligibility for
  `dram_stream_stride_mix/timing_00` promoted to
  `Verified-live-sm_86`.**
- **R2** — holding round (no tracked changes). Codex
  Round-2 STALLED, explicitly named as STRUCTURAL: the
  loop's stop-hook treats the explicitly-deferred sm_120
  replay as incomplete despite the deferral being correct
  per CLAUDE.md.
- **R3** — holding round + project-memory entry created at
  `~/.claude/projects/.../memory/project_rlcr_structural_stall_pattern.md`
  (machine-local; documents the structural-stall pattern
  for future RLCR-loop scoping). Codex Round-3 STALLED.
- **R4** — drift-recovery round per the loop's
  drift-recovery-mode trigger. Added a substantive Plan
  Evolution Log entry to v7 goal-tracker MUTABLE
  diagnosing the structural deadlock. Mainline Drift
  Circuit Breaker fired after this round (3 consecutive
  STALLED verdicts), halting the loop.

**The v7 dram_stream_stride_mix sm_86 evidence (live
capture):** `workloads/simbench_bw/results/dram_stream_stride_mix/rtx_3090_ti/sm_86/`
directory contains 30 cupti_activity sidecars + 30
nvml_covariate sidecars + shape-level `timing_00/measurement_manifest.json`
+ parent `timing_timing_00.json`. Helper admits with
`ok=True 30/30 0-defect` for
`expected_slug="dram_stream_stride_mix"
expected_shape_point="timing_00"`. Wrong-slug sentinel
returns `ok=False 62 typed defects` across all 4
surfaces. The capture artifacts are gitignored; they
survive raw scp but NOT git fetch.

**Why the v7 loop went STALLED structurally and what it
means for the receiving session**: the v7 plan correctly
scoped itself to sm_86 (CLAUDE.md item 1 prohibits
cross-arch promotion without explicit live witness on
the second arch), but the loop's stop-hook treated every
deferred item as incomplete by default. The composition
created a deadlock: plan-scoped completion (achieved at
v7 R1) ≠ loop-global completion (gated on the deferred
sm_120 leg that this host cannot capture). The v8 plan
inverts this trap by being sm_120-only on the laptop with
no host-constrained deferrals; the v7 sm_86 evidence is
regression-locked baseline carried forward, NOT deferred
work in the v8 loop.

## Mandatory First Actions on the Receiving Host

Run these in order, every time, before the first round
contract:

1. **Confirm the repo is at the handoff HEAD.**
   ```sh
   git -C "$(pwd)" rev-parse HEAD   # should resolve to
                                    # the third doc-sync
                                    # commit 3c02fca (or
                                    # a descendant)
   git -C "$(pwd)" status --short   # should be clean
   git -C "$(pwd)" log -1 --oneline -- HANDOFF.md
                                    # confirms the most
                                    # recent HANDOFF.md
                                    # commit (expected:
                                    # 3c02fca or a later
                                    # doc-only descendant;
                                    # earlier doc-sync
                                    # parents: 423375f,
                                    # 5912ab6)
   git -C "$(pwd)" log --oneline    # the substantive v6+v7
                                    # commits 9f5a2ca,
                                    # 14e37bc, eaf5eee,
                                    # 30ae840, 5ea52f3
                                    # should all appear
                                    # in the chain along
                                    # with the v3-era
                                    # substantive parent
                                    # 38a353d below them
   ```

2. **Delete the source host's RLCR loop state.**
   ```sh
   rm -rf .humanize/rlcr/
   ```
   The previous loop's contracts and summaries are
   bound to the source host's HEADs and capture
   artifacts; they are not portable.

3. **Verify CUDA / NVBit / NVML environment + rebuild
   any built artifacts that came across.**
   ```sh
   nvidia-smi                       # confirms sm_120 device
   nvcc --version                   # CUDA toolkit version
   gcc --version                    # host gcc version (the
                                    # tracer build needs a
                                    # gcc compatible with
                                    # the local nvcc)
   ```
   Treat ALL built binaries / shared libraries that came
   across via raw scp as STALE host artifacts targeted at
   the originating arch (sm_86). Rebuild them on this host
   before any live evidence:
   ```sh
   # NVBit tracer (exact CUDA/protobuf/compiler pins):
   source tools/nvbit_tracer/pinned_build_env.sh
   make -C tools/nvbit_tracer ARCH=sm_120

   # CUPTI Activity collector library (workloads/simbench_bw/Makefile
   # uses NVCC ?= nvcc and does NOT honor NVCC_CCBIN; override NVCC
   # directly to inject -ccbin; on the 5060 laptop also pass
   # COLLECTOR_ARCH=sm_120 since the default is sm_120 but pin it
   # explicitly):
   make -C workloads/simbench_bw cupti-collector-lib \
       COLLECTOR_ARCH=sm_120 \
       NVCC='nvcc -ccbin=/usr/bin/g++-14'

   # Per-slug binaries — same pattern as the collector:
   # the per-kernel Makefile includes common.mk which has
   # NVCC ?= nvcc, so pass NVCC='nvcc -ccbin=...' here too.
   # Build only the ones the round actually exercises; active
   # portable_lcd binaries build for the local arch automatically:
   make -C workloads/portable_lcd/<active-slug> \
       NVCC='nvcc -ccbin=/usr/bin/g++-14'
   make -C workloads/simbench_bw/l1/<simbench-slug> clean all \
       NVCC='nvcc -ccbin=/usr/bin/g++-14'
   make -C workloads/simbench_bw/l2/rmsnorm_fwd_bf16 clean all \
       NVCC='nvcc -ccbin=/usr/bin/g++-14'
   ```

   **Toolchain-sanity-check alternative**: if the laptop's
   host-default `nvcc` already resolves `-ccbin` to
   `/usr/bin/g++-14` (verify via
   `nvcc -v -dryrun /tmp/empty.cu 2>&1 | grep -i ccbin`),
   the bare `make ... cupti-collector-lib COLLECTOR_ARCH=sm_120`
   and `make clean all` invocations work without the
   `NVCC=...` override. Default to the explicit override
   unless the operator has confirmed the host default.

   **Toolchain framework discipline (cross-host).**
   Both local devices (RTX 3090 Ti sm_86, RTX 5060 laptop
   sm_120) MUST use a native system `g++-14` binary at
   `/usr/bin/g++-14`. Do NOT provision conda / mamba /
   miniforge envs as a substitute — toolchain divergence
   between the two hosts breaks the cross-arch parity
   discipline (CLAUDE.md "Two-Device Local Development
   Model"). The canonical build is
   `source tools/nvbit_tracer/pinned_build_env.sh` followed by
   `make -C tools/nvbit_tracer ARCH=<host-arch>` on both machines.

   **If the receiving host has no `g++-14` yet,** install
   it from the OS native package manager. Per-OS recipes:

   - **Arch Linux** (the 3090 Ti host ships gcc-15/16 by
     default, both too new for CUDA 13.2's nvcc —
     libstdc++ 16's `__type_identity` templates make
     `helpers_lib.cu` compilation fail at parse-time).
     Install gcc-14 from AUR:
     ```sh
     yay -S gcc14         # or: paru -S gcc14
     # Manual fallback if no AUR helper:
     git clone https://aur.archlinux.org/gcc14.git ~/aur/gcc14
     cd ~/aur/gcc14 && makepkg -si
     ```
     Verify: `which g++-14` returns `/usr/bin/g++-14`.

   - **Debian / Ubuntu** (likely the 5060 laptop):
     ```sh
     sudo apt update && sudo apt install -y g++-14 gcc-14
     ```
     Verify: `which g++-14` returns `/usr/bin/g++-14`.

   **Persistent shell env (required for the loop's Codex
   review subshell to inherit the toolchain pin).** Add
   the export to your shell rc so every subshell —
   including `pytest`'s `subprocess.run`, the loop
   harness, and `codex exec` — sees it:

   ```sh
   # bash / zsh
   echo 'export NVCC_CCBIN=g++-14' >> ~/.bashrc   # or ~/.zshrc

   # fish
   echo 'set -x NVCC_CCBIN g++-14' >> ~/.config/fish/config.fish
   ```

   With g++-14 installed, source `tools/nvbit_tracer/pinned_build_env.sh`;
   then `make -C tools/nvbit_tracer`
   builds the live `tracer_tool.so`, the per-slug
   `helpers_lib` Makefile builds the FP4 / FP6 / FP8 /
   sparse-2:4 byte-contract `helpers_lib.so`, and
   `pytest tools/ workloads/simbench_bw/tests/ -q` admits
   the host-only baseline on this receiving host.
   CLAUDE.md Rule 4 is unaffected: native package install
   via `pacman` / `apt` requires `sudo` for the install
   step itself but does NOT touch the NVIDIA driver,
   DKMS, MOK, `modprobe`, or any GPU-driver-root surface;
   the compiled output is a per-user toolchain binary at
   `/usr/bin/g++-14`.

   **Codex sandbox bypass for GPU-gated tests.** If the
   receiving session uses Codex review and intends to run
   GPU-gated tests within the review subshell, set
   `HUMANIZE_CODEX_BYPASS_SANDBOX=true` in the shell rc
   so `codex exec` invokes
   `--dangerously-bypass-approvals-and-sandbox`.
   Otherwise the sandbox masks `/dev/nvidia*` and the
   GPU-gated tests silently skip in the review shell.

4. **Run the host-only baseline sweep.** Confirms
   nothing arch-coupled regressed during scp:
   ```sh
   python -m pytest tools/ workloads/simbench_bw/tests/ -q
   python workloads/simbench_bw/trace_size_estimator.py --validate-all
   python workloads/simbench_bw/validate.py --all
   python workloads/simbench_bw/verify_coverage.py --strict
   python -m pytest tools/test_microbench_ac14_coverage.py \
       tools/test_plan_token_scanner.py -q
   ```
   At the v3 R18 broad-sweep target the receiving session
   should expect 2022+ passed, 2 skipped, 678+ subtests
   (the sm_86 baseline at the substantive parent
   `38a353d`); the sm_120 figures may differ slightly
   because some tests are arch-gated. If the host-only
   baseline regresses substantially, triage before the
   first round contract.

5. **Read `tools/BLOCKED_ON_3090TI.md`.** This is the
   per-task ledger. After v3 R0-R25 most ledger entries
   that pinned sm_86-blocked items are marked closed via
   v3 sm_86 capture. The ledger records each task's
   sm_86 evidence pin. The receiving 5060 session's job
   is to add the matching sm_120 evidence to lift each
   surface to `Cross-arch-verified`.

6. **Open the v8 plan first**:
   `plans/ac16_dram_stream_cupti_hook_sm120_plan.md`.
   This is the recommended narrowest first cross-arch loop
   on the laptop — it covers the dram_stream_stride_mix/timing_00
   sm_120 leg only and pre-empts the v7 structural-stall
   trap (no host-constrained deferrals; sm_120 IS the
   mainline on the laptop). The 11 ACs cover bootstrap,
   stale-artifact cleanup, rebuild + cuobjdump, backward-
   compat with cross-arch digest equality, fresh sm_120
   capture, strict-admission helper round-trip, regression
   coverage, broad sweep, status wording, TOML, and
   cross-arch promotion.

   Subsequent laptop loops can follow the v3 plan
   (`plans/nonroot_gpu_cache_observability_continuation_plan_v3.md`)
   for the broader cross-arch sweep punch list (items 1-14
   in this HANDOFF.md), and the historical v2/v1 plans
   (`plans/nonroot_gpu_cache_observability_continuation_plan_v2.md`,
   `plans/nonroot_gpu_cache_observability_continuation_plan.md`)
   are historical reference only — do not modify any of
   them. The v7 plan
   (`plans/ac16_dram_stream_cupti_hook_plan.md`) is also
   historical reference (it produced the v6+v7 sm_86
   closure that the v8 plan inherits).

7. **Start the fresh RLCR loop** when the operator
   triggers `/humanize:start-rlcr-loop` or equivalent.
   Recommended first invocation:
   ```sh
   /humanize:start-rlcr-loop --max 60 --full-review-round 10 --privacy --skip-quiz --track-plan-file plans/ac16_dram_stream_cupti_hook_sm120_plan.md
   ```
   Round 0 contract should re-read the v8 plan + this
   `HANDOFF.md` + `tools/BLOCKED_ON_3090TI.md`.

## Compact AC Closure Table (post-v3-R25)

The receiving session's first round MUST re-validate the
table by running the host-only sweep on this hardware. The
"needs sm_120 live sweep?" column flags the items the new
loop should explicitly target.

| AC | Most recent verified status | Evidence path | Needs sm_120 live sweep? |
|----|------------------------------|---------------|--------------------------|
| C-AC-A2 | Offline-verified (v2 R3) | `tools/test_blocked_on_3090ti_hygiene.py` | No (host-only ledger schema) |
| C-AC-2A3A (AC-2 / AC-3 sm_86) | Verified via typed-exclusion (v3 R4) | `data/local_labels/<slug>/sm_86/`, `tools/local_labels/repeatability_aggregator.py` | **Yes** — re-run task7 NCU local labels on sm_120 per `tools/local_labels/TASK7_RUNBOOK.md` |
| C-AC-4A (AC-4 sm_86) | Verified-live-sm_86 (v3 R3) | `workloads/simbench_bw/l1/dram_stream_stride_mix/traces/...`, manifest with `compiler_flags` including `arch=compute_86,code=sm_86` | **Yes** — rebuild tracer + run `--phase-inject-ab` on sm_120 |
| C-AC-F2 (sector-stream) | Verified-live-sm_86 (v3 R5) | `.../sector_stream.jsonl`; 769 admitted records | **Yes** — re-capture sector_stream on sm_120 (expect different distribution) |
| C-AC-G2 (divergence-proxies) | Verified-live-sm_86 (v3 R8 post-commit recapture per BL-20260508) | `.../divergence_proxies.json`; `manifest_pairing_index.source_commit` pinned to v3 R8 HEAD | **Yes** — re-run live trace bundle on sm_120 |
| C-AC-H2 (trace-budget) | Verified-live-sm_86 via correctly-does-not-produce arm (v3 R9) | `.../trace_<shape>.json` slowdown/drop-rate sidecars | **Yes** — re-verify trace-budget bounds on sm_120 |
| AC-6 (HRD profiler) | Verified-live-sm_86 (v3 R10) | `.../hrd_profile/hrd_profile.json`; 13 bins (8 ok + 5 typed-unavailable) | **Yes** — capture sm_120 HRD bundle (bin distribution will differ) |
| AC-7 (sharing-locality) | Verified-live-sm_86 (v3 R11) | `.../sharing_locality.json` | **Yes** — capture sm_120 sharing locality |
| AC-8 (order-projections) | Verified-live-sm_86 (v3 R12+R13) | `.../order_projections/order_projections.json`; 6 projections; per-sidecar regression coverage | **Yes** — capture sm_120 order projections |
| AC-10 (perturbation harness) | Verified-live-sm_86 (v3 R14) | `.../perturbation_run_ledger/perturbation_run_ledger.json`; 10 perturbations including cache_modifier_cs | **Yes** — re-run perturbation harness on sm_120; cache-modifier SASS regex BL-20260505 already admits both Ampere `LDG.E.CA/CG` and Blackwell `LDG.E.STRONG.SM/GPU` |
| AC-11 (adaptive sampler) | Verified-live-sm_86 (v3 R15+R16) | adaptive sampler windows + `verify_artifacts.py --results`; `_SUPPORTED_ARCHES = ("sm_86", "sm_120-laptop")` | **Yes** — adaptive sampler sm_120 sweep |
| AC-14 (microbench suite) | Verified-live-sm_86 (v3 R17) | 9 R51-contract sm_86 shape points; per-shape feature sidecars; 45-way per-shape source-commit equality | **Yes** — re-capture 9 shape points on sm_120 |
| AC-15 (combined pairing index producer-side) | Verified-live-sm_86 (v3 R20, maintained R21+) | `workloads/simbench_bw/dual_workload_pairing_index.py`; combined pairing index `len=25, registries={portable_lcd, simbench_bw}, at_head=25/25, zero=0` at substantive parent `38a353d` | **Yes** — produce sm_120-side combined pairing index from sm_120-built portable_lcd binaries; verify dual-registry contribution |
| AC-16 (NVML covariate + CUPTI Activity producer-side, rmsnorm_fwd_bf16/timing_00) | Verified-live-sm_86 (v3 R23 invalid-JSON-guarded helper + scenario-backed token coverage; supersedes R20+R21+R22) | `workloads/simbench_bw/cupti_dual_source_live_admission.py` strict-admission helper; FULL sidecar identity contract across 4 surfaces × 5 axes = 20 typed `DEFECT_TOKENS`; `rmsnorm_fwd_bf16/timing_00` admits `ok=True 30/30, 0 failures, 0 defects` at substantive parent `38a353d` | **Yes** — produce a sm_120 preflight manifest, run `run_suite.py --mode timing --kernel rmsnorm_fwd_bf16 --shape-point timing_00 --enable-cupti-activity --preflight-manifest <path>` on sm_120, run the v3 R23 helper against the fresh sm_120 artifact |
| AC-16 second-workload-eligibility (dram_stream_stride_mix/timing_00) | **Verified-live-sm_86 (v6 R0 dlsym hook chain + v7 R0/R1 fresh capture + AC-7 hardening at HEAD `5ea52f3`)** | The same `cupti_dual_source_live_admission.py` strict-admission helper used for rmsnorm; `dram_stream_stride_mix/timing_00` admits `ok=True 30/30, 0 failures, 0 defects` at v7 HEAD `5ea52f3`; wrong-slug sentinel returns `ok=False 62 typed defects` across all 4 surfaces | **Yes** — covered by the new v8 plan at `plans/ac16_dram_stream_cupti_hook_sm120_plan.md` (the laptop session's narrowest first cross-arch loop). v8 is sm_120-only on the laptop; rebuild the binary + collector for sm_120, generate fresh sm_120 preflight, run timing-mode capture, round-trip the same helper, promote to `Verified-live-sm_120` then `Cross-arch-verified` |
| C-AC-I2 (host-only sm_86 re-run) | Offline-verified (v3 R18) | `pytest tools/ workloads/simbench_bw/tests/ -q` -> 2022 passed, 2 skipped, 678 subtests passed; R56-R60 host-only test counts | No (already cross-arch via host-only by design) — receiving session re-runs to confirm baseline parity |
| AC-12 (server collection bootstrap) | Explicitly Deferred | Path 3 of three-fold completion gate; revisit trigger requires path 2 closure | No (server-arch deferral) |
| AC-16 (server-architecture NVML) | Explicitly Deferred | Path 3 | No (server-arch deferral) |
| AC-18 (workload portion) | Offline-verified (v2 R0+R0-review) | `workloads/simbench_bw/README.md`, `workloads/portable_lcd/README.md`, `tools/test_runbook_command_verification.py` (fake-* command verification suite) | No (host-only doc + fake-* verification, no live device evidence required) |
| AC-18 (cloud) | Explicitly Deferred | Path 3 | No (cloud-arch deferral) |

## Why AC-15 / AC-16 are now Verified-live-sm_86 producer-side

The original v2-era HANDOFF.md retagged AC-15 and AC-16
from `Verified-live-sm_120` to `Offline-verified` per
Codex's v2 R59/R60 review discipline (the host-only suites
counted as Offline-verified, not Verified-live-anything).
The v3 RLCR loop's R20-R23 work moved AC-15 and AC-16 to
`Verified-live-sm_86` producer-side via genuine live-GPU
evidence:

- **AC-15 (R20)**: Combined dual-workload pairing index
  built on sm_86 with both registries (portable_lcd +
  simbench_bw) contributing 25 entries; source_commit
  equality at HEAD; LiveRosterSmokeTest admits with real
  binaries built (the typed-exception path is no longer
  the only admission route). Promotion to
  `Verified-live-sm_86` is justified because the
  producer-side capture pipeline ran end-to-end on sm_86
  hardware with built binaries.
- **AC-16 (R20-R23)**: NVML covariate + CUPTI Activity
  preflight admission widened to sm_86 (R20 commit
  `a0cabd4`), CUPTI dual-source tolerance recalibrated
  on sm_86 (R20+R21), strict-admission helper introduced
  (R21 commit `6affa89`) with iterative tightening across
  R22 (full 4-surface identity contract, commit
  `8066e0f`) and R23 (manifest-invalid-JSON path guard +
  scenario-backed token coverage, commit `38a353d`). The
  helper's positive admission of the live
  `rmsnorm_fwd_bf16/timing_00` artifact (`ok=True 30/30,
  0 failures, 0 defects`) is the substantive
  `Verified-live-sm_86` evidence. BL-20260509 captures
  the contract-first design pattern that emerged from
  the R20-R23 review chain.

The receiving 5060-laptop session's path 2 work for
AC-15/AC-16 is therefore **producer-side** — it needs to
add a matching `Verified-live-sm_120` producer-side leg
(not merely re-run the host-only suite, which still
counts as `Offline-verified`).

## Cross-Arch Sweep Punch List for the Receiving 5060 Session

Order roughly mirrors AC dependency. The receiving
session's first 5-10 rounds typically pick from the top
of this list. Each item is the sm_120 leg of a v3 R0-R25
sm_86 surface; closing the leg lifts the surface from
`Verified-live-sm_86` to `Cross-arch-verified`.

1. **sm_120 NVBit tracer rebuild** (closes the sm_120
   side of v3 R0; advances AC-4 sm_120 evidence).
   Follow `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md`:
   rebuild `tracer_tool.so` on this host, run the live
   trace contract sweep against `dram_stream_stride_mix`
   on sm_120, capture an admitted `dynamic_trace.pb` +
   matching `manifest.json`, run `python -m pytest
   workloads/simbench_bw/tests/test_nvbit_trace_contract_live.py`
   with `SIMBENCH_REQUIRE_LIVE_GPU=1` and confirm the
   helper-section resolver admits both Ampere and
   Blackwell layouts. Many later rounds depend on a
   working sm_120 tracer.

2. **sm_120 local-label hardware acceptance** (advances
   AC-2 / AC-3 sm_120 evidence).
   Follow `tools/local_labels/TASK7_RUNBOOK.md` to
   produce the sm_120 NCU-derived label sidecars under
   `data/local_labels/<slug>/sm_120/` and run the
   `tools/local_labels/repeatability_aggregator.py`
   N>=5 repeatability verdict. The local-label NCU
   exception in CLAUDE.md Rule 1 covers this work.

3. **C-AC-F2 sector-stream sm_120 re-capture** (lifts
   v3 R5 to Cross-arch-verified). Build the tracer, run
   `dram_stream_stride_mix` under NVBit on sm_120,
   consume the `sector_stream.jsonl`, schema-validate,
   compare record count + class distribution to the
   sm_86 baseline (769 admitted records on sm_86;
   sm_120 distribution may differ — that is expected,
   not a regression).

4. **C-AC-G2 divergence-proxies sm_120 re-capture**
   (lifts v3 R8 to Cross-arch-verified). Re-run the
   live trace bundle on sm_120; expect different
   `stride_entropy` distribution and possibly different
   `mshr_pressure_proxy` signal because of Blackwell's
   different MSHR depth. **Apply BL-20260508 post-commit
   recapture discipline** if any HEAD-derived bundle
   field changes during this work.

5. **C-AC-H2 trace-budget sm_120 re-verify** (lifts v3
   R9 to Cross-arch-verified). Slowdown bound and
   drop-rate bound are arch-dependent.

6. **AC-6 HRD profiler sm_120 re-capture** (lifts v3
   R10 to Cross-arch-verified). HRD bundle's bin
   distribution is L1/L2-cache-size-dependent; the bin
   layout is the same enum, but counts differ by arch.

7. **AC-7 sharing-locality sm_120 re-capture** (lifts
   v3 R11 to Cross-arch-verified). CTA-identity logic
   is arch-independent; admission counts differ.

8. **AC-8 order-projections sm_120 re-capture** (lifts
   v3 R12+R13 to Cross-arch-verified). Six projection
   admission counts should remain 6 on sm_120 if all
   projections are structurally available. The
   per-sidecar canonical regression tests added in v3
   R13 will exercise on sm_120 too.

9. **AC-10 perturbation harness sm_120 re-run** (lifts
   v3 R14 to Cross-arch-verified). The cache-modifier
   SASS regex per BL-20260505 already admits both
   Ampere `LDG.E.CA/CG` and Blackwell
   `LDG.E.STRONG.SM/GPU` shapes. Verify
   `cache_modifier_cs` admits on sm_120.

10. **AC-11 adaptive sampler sm_120 sweep** (lifts v3
    R15+R16 to Cross-arch-verified). The v3 R15 work
    pinned `_SUPPORTED_ARCHES = ("sm_86",
    "sm_120-laptop")`, so the sm_120 path is already
    structurally admitted; the live sweep just needs to
    run.

11. **AC-14 microbenchmark suite sm_120 cross-arch**
    (lifts v3 R17 to Cross-arch-verified). 9
    R51-contract shape points × 7 canonical result
    roots; expect the structural-only families to
    admit, the timing families to land different
    `wall_time_ms` / `cv_pct` numbers. The 45-way
    per-shape source-commit equality discipline applies.

12. **AC-15 combined dual-workload pairing index sm_120
    producer-side live evidence** (lifts v3 R20 to
    Cross-arch-verified). On sm_120 build the active
    portable_lcd binaries, then run
    `build_combined_pairing_index` against the committed
    rosters. Confirm dual-registry contribution
    (portable_lcd + simbench_bw); confirm 25/25 entries
    pin to current HEAD; confirm zero zero-fill source
    commits.

13. **AC-16 NVML covariate + CUPTI Activity sm_120
    producer-side live capture** (lifts v3 R23 to
    Cross-arch-verified). Produce a preflight manifest
    first (`bash tools/cloud/preflight.sh --out
    /tmp/simbench_preflight.json`) and run a
    `--mode timing --kernel rmsnorm_fwd_bf16
    --shape-point timing_00 --enable-cupti-activity
    --preflight-manifest /tmp/simbench_preflight.json`
    timing-shape capture on sm_120 — the
    `--preflight-manifest` is a mandatory companion of
    `--enable-cupti-activity` per `run_suite.py`'s
    `EXIT_CUPTI_ACTIVITY_PREFLIGHT_MANIFEST_REQUIRED`
    guard. Verify NVML sidecars are emitted; round-trip
    the v3 R23 strict-admission helper at
    `workloads/simbench_bw/cupti_dual_source_live_admission.py`
    against the fresh sm_120 artifact (expect
    `ok=True`, 30/30 dual-source pass at the contract-
    prescribed `timing_00` shape, 0 timing failures, 0
    defects). Confirm dual-source tolerance is
    appropriately recalibrated for sm_120 if needed
    (the R21 1000-sample pool was sm_86-DVFS-specific;
    sm_120 may need its own pool). Record device /
    driver / CUDA / NVBit / clocks in the round summary.

14. **AC-18 workload READMEs** — already
    `Offline-verified` from v2 R0; live evidence
    (operator running the documented commands on
    sm_120) remains the path to `Cross-arch-verified`
    for the workload portion.

15. **AC-16 second-workload-eligibility for
    dram_stream_stride_mix/timing_00 sm_120
    producer-side live capture** (NEW — lifts the v6+v7
    sm_86 work to Cross-arch-verified). **The narrowest
    first cross-arch loop on the laptop**: covered end-to-
    end by the v8 plan at
    `plans/ac16_dram_stream_cupti_hook_sm120_plan.md`. The
    plan has 11 ACs (AC-1..AC-11), 14 tasks
    (task1, task1b, task2..task13), and was reviewed by
    Codex on the 3090 Ti host with all 9 REQUIRED CHANGES +
    3 SUGGESTED IMPROVEMENTS folded in. Procedure:
    rebuild tracer + collector (sm_120) + dram binary on
    laptop (with explicit `NVCC='nvcc -ccbin=/usr/bin/g++-14'`
    override for collector + dram binary, `NVCC_CCBIN=g++-14`
    for tracer); standalone smoketest verifying the
    `output_sha256` digest equals the sm_86 baseline
    `786c10f82133609842e2be4226ff90154a5efde3c2ff001a504dd2367828e782`
    (cross-arch correctness invariance — the kernel
    output is identity `out[i] = in[i]` so digest equality
    is mathematically required); generate fresh sm_120
    preflight; run `run_suite.py --mode timing --kernel
    dram_stream_stride_mix --shape-point timing_00
    --enable-cupti-activity --preflight-manifest <fresh>`
    on sm_120; round-trip the strict admission helper
    (`ok=True 30/30 0-defect` expected, plus wrong-slug
    + forced-failure negative sentinels); two-step
    promotion: `Verified-live-sm_120` first (AC-9), then
    combine with carried v7 `Verified-live-sm_86` to
    record `Cross-arch-verified` (AC-11). Recommended as
    the laptop session's first loop because: (a) the
    substantive code is already at HEAD with no v8 code
    edits required; (b) the plan pre-empts the v7
    structural-stall trap by being sm_120-only with no
    host-constrained deferrals; (c) the helper is
    regression-locked by the existing `CuptiDualSourceAdmissionSecondWorkloadParameterizedTest`
    test class.

After path 2 closes (steps 1-14 above), the receiving
session is positioned to start path 3 (server/cloud)
per the IMMUTABLE plan ordering. Path 3 sub-list:

a. Reconcile the queued preflight `12.*`
   prose/device-policy drift before any server admission
   claim. The code at `tools/cloud/preflight.sh` and
   `tools/cloud/preflight_manifest.py` already admits
   every CC `12.*` device via
   `ADMITTED_CAP_PREFIXES = ("8.6", "12.")`. The prose
   drift is between this code admission and the prose
   that frames the policy as "consumer laptop only";
   the receiving session should align prose with code OR
   add a policy-level distinguishing field
   (e.g., `device_class = consumer | server | datacenter`)
   that gates server admission.

b. AC-12 server collection bootstrap without NCU,
   driver-root, privileged CUPTI Events/Metrics/Range/
   PM/PerfWorks, host-driver installation, or simulator
   revival.

c. AC-16 server NVML using read-only NVML covariates
   parallel to the local producer-side helper at
   `workloads/simbench_bw/nvml_metadata_verifier.py` +
   `workloads/simbench_bw/cupti_dual_source_live_admission.py`.

d. AC-18 cloud `tools/cloud/RUNBOOK.md` with
   fake-command verification (per BL-20260504-runbook-
   command-verification) and static guards proving the
   cloud path does not invoke forbidden profiler / root /
   driver operations.

## Active Side Issues / Carried Blockers (carried into the receiving session)

### Carried Blockers (none active)

- **R18/R19 prior-round-summary amendment-method process
  blocker — CLOSED.** Operator sanction was granted
  outside the v3 loop (referenced by the v7 plan body's
  "Path 1 (R18/R19 amendment-method bypass) was just
  operator-sanctioned" note). The sanctioned future
  amendment-path policy: NEVER use Python
  `open(...).write(...)` to bypass the loop-edit-validator
  hook on prior-round summaries; record corrections in the
  current round's surfaces (round summary + goal-tracker
  Plan Evolution Log mutable section). The v7 R4 round
  exercised this policy in practice: a cross-round summary
  edit attempt was correctly blocked by the hook, and the
  correction was instead recorded in the v7 R4 summary +
  goal-tracker MUTABLE Plan Evolution Log entry.

### Queued Side Issues (non-blocking, carried from v3 R0-R25 review chain)

- **Preflight `12.*` prose/device-policy drift**
  (R20 review queued). Revisit trigger is "before
  server/cloud admission policy depends on it" —
  effectively path 3 prerequisite, sub-step (a) above.
- **Manifest `driver_version` ambiguity** (R5 review).
  Manifest's `13.2` (CUDA toolkit) vs. preflight's
  `595.71.05` (NVIDIA driver) reported under the same
  field. Observational; not blocking any AC.
- **Per-source NVBit drop counters beyond residual
  expected-record mismatch** (carried from v2). Not
  blocking C-AC-H2's accepted correctly-does-not-produce
  arm.
- **Event-based async-free retirement for stream-ordered
  allocator hooks** (carried from v2). Not blocking
  C-AC-4A.
- **Workload-side `wall_time_ms` for
  `dram_stream_stride_mix`** (carried from v2). Not
  blocking C-AC-4A.
- **`tools/arch_capabilities.json` perturbation-harness
  arch-capability prose drift** (R14 review). The
  `build.py` entry says "nine perturbation binaries"
  while the current loop emits 10; the actual bundle
  truth at HEAD records 10. Not blocking AC-10.
- **R21 retroactive-helper wording caution** (R23
  review). The R21 summary's framing of the helper as
  "retroactive" is technically imprecise; the helper is
  forward-looking and only exercises retroactively in
  test fixtures. Documentation-only.

### Documentation-only carries from earlier rounds

- HRD budget-exceeded producer status dict reports stale
  `bundle_underconfigured` (v2 R55 review queued). Touches
  `tools/hrd_profiler/bundle_from_trace.py`. Non-blocking.
- `hrd_profile.schema.json` `bundle_underconfigured`
  description drift (v2 R55 review queued). Documentation-
  only.
- `label_driver.sh` duplicate helper JSON on rejected
  arches (older queued). Non-blocking.
- `sector_stream.schema.json` description typo for the
  sectors-per-request invariant (older queued).
  Documentation-only.
- Slowdown/drop-rate sidecar schema field-description
  drift (older queued). Documentation-only.
- Task17 run_sampler command-test docstring illustrative
  arch drift (older queued). Documentation-only.

## Pitfalls to avoid

- **Do NOT trust prior `Verified-live-sm_86` tags as
  evidence of arch-independence.** Several ACs have
  arch-coupled timing or SASS evidence that will produce
  different numbers on sm_120. Read the per-task ledger
  entry's `(d)` and `(e)` sections in
  `tools/BLOCKED_ON_3090TI.md` for the prerequisites and
  the most-recent sm_86 evidence pin. CLAUDE.md
  "Two-Device Local Development Model" enumerates
  specific risks: broader sm_120 instruction-class set,
  TMA / async-copy semantics, clock / DVFS / throttle
  variance.
- **Do NOT promote AC-15 / AC-16 to `Verified-live-sm_120`
  by re-running the host-only suite.** That counts as
  `Offline-verified` again. Promotion requires a real
  live sweep that exercises the producer surface (build
  binaries on sm_120, run capture pipeline end-to-end,
  round-trip the strict-admission helper at
  `workloads/simbench_bw/cupti_dual_source_live_admission.py`
  against the fresh sm_120 artifact).
- **Do NOT modify the immutable section of the new
  plan's goal-tracker after Round 0.** Mutable section
  only. The receiving session opens a fresh
  `.humanize/rlcr/<new-session-id>/` directory and starts
  Round 0 with a fresh goal-tracker; the v3 sm_86
  goal-tracker stays in `.humanize/rlcr/2026-05-07_02-01-13/`
  on the source host or is deleted before the new loop.
- **Do NOT attempt to amend prior-round summary files
  via Python `open(...).write(...)` to bypass the loop-
  edit-validator hook.** R18 did this and the bypass is
  the carried R18/R19 operator-sanction blocker (path 1
  of three-fold gate). The disciplined alternative is
  R19's pattern: record corrections in the next round's
  surfaces and the goal-tracker mutable section, never
  via a hook bypass.
- **Do NOT use absolute language about loop-completion
  gates** (e.g., "sole remaining gate", "all that
  remains") in stand-down or process-integrity rounds.
  R24 fell into this; R25 had to retract. Use scope-
  bounded language: "the only autonomous-host-closeable
  mainline gap", "the carried operator-sanction request",
  etc.
- **Do NOT scp `.humanize/rlcr/` from the previous host
  AND keep it.** The directory survives raw scp but per
  CLAUDE.md must be deleted before the new loop. Round
  contracts/summaries are bound to the previous host's
  HEAD + capture artifacts and not portable.
- **Do NOT assume committed evidence files survive a
  cross-arch claim.** A trace JSON committed during v3
  sm_86 still admits on sm_120's offline validators (the
  schema is arch-independent), but the numbers it
  reports are sm_86-coupled and not a baseline for the
  receiving session's claims.

## Pointers

- **CLAUDE.md** — durable rules, two-device discipline,
  cross-host survival semantics.
- **`plans/ac16_dram_stream_cupti_hook_sm120_plan.md`** —
  **NEW v8 plan**: the laptop receiving session's
  narrowest first cross-arch loop (dram_stream_stride_mix/timing_00
  sm_120 only). Pre-empts the v7 structural-stall trap by
  being sm_120-only on the laptop. 11 ACs, 14 tasks,
  Codex-reviewed. **Recommended first loop on the laptop**.
- **`plans/ac16_dram_stream_cupti_hook_plan.md`** — v7 plan
  (closed Round 4 with circuit-breaker fired); the
  source-of-truth contract for the v6+v7 substantive
  commits at HEAD `5ea52f3`. Historical reference; do not
  modify.
- **`plans/nonroot_gpu_cache_observability_continuation_plan_v3.md`**
  — v3 plan (closed at R26); covers the broader Track B
  sm_86 closure work that landed before v6+v7. Historical
  reference for the v3-era cross-arch sweep punch list
  (items 1-13 above).
- **`plans/nonroot_gpu_cache_observability_continuation_plan_v2.md`**
  — original 3090 Ti / sm_86 contract; historical reference;
  do not modify.
- **`plans/nonroot_gpu_cache_observability_continuation_plan.md`**
  — historical Loop 2 (sm_120-side) contract; do not
  modify.
- **`tools/BLOCKED_ON_3090TI.md`** — per-task ledger of
  surfaces that pinned sm_86-blocked items; v3 R0-R25 +
  v6 R0 + v7 R0/R1 closed many; the receiving session
  adds the matching sm_120 pins.
- **`MANIFEST.md`** — repository scope and cleanup record.
- **`README.md`** — current project framing.
- **`.humanize/bitlesson.md`** (gitignored; preserve via
  raw scp) — 9 BitLesson entries capturing recurring
  problem-solution patterns. The v6+v7 work did not add
  any new BL entries (the structural-stall pattern from
  v7 R3 was captured as auto-memory rather than a
  BitLesson because it does not fit the BL template's
  "specific failure mode + root cause + exact fix"
  structure).
- **`~/.claude/projects/.../memory/project_rlcr_structural_stall_pattern.md`**
  (machine-local; NOT carried by any cross-host transport;
  preserve manually via raw file scp if desired) — auto-memory
  entry from v7 R3 documenting the structural-stall pattern;
  informs future RLCR-loop scoping decisions in this project.
  The v8 plan applies this lesson by being sm_120-only.
- **`~/.claude/projects/.../memory/project_loop_state.md`**
  (machine-local) — current loop-state snapshot at v7
  close.
- **`~/.claude/projects/.../memory/project_progress.md`**
  (machine-local) — AC progress snapshot at v7 close.
