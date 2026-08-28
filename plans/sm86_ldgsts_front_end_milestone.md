# SM86 LDGSTS front-end transfer plan

## Goal Description

Add one default-off, architecture-scoped Ampere `LDGSTS` front-end candidate
to the standalone GPU simulator, then decide it with CPU-only replay of the
retained RTX 3090 Ti campaigns. The candidate represents the missing finite
memory ingress before the existing L1/L2/DRAM response and LDGSTS destination
models. It is a paper-derived simplified composite candidate: five outstanding
entries per SMSP (four queued slots plus the dispatch latch), four-cycle local
initiation, two-cycle SM-shared acceptance, 13-cycle WAR visibility, and a
39-cycle source completion floor. It is not a line-for-line reproduction of
Huerta et al.'s multi-stage backpressured pipeline. In particular, their
four-slot queue plus dispatch latch and 4/2-cycle rates come from general
memory-instruction experiments, while 13/39 are separate LDGSTS latency
measurements. Combining them only for LDGSTS is this project's scoped
hypothesis, not a published LDGSTS relation.

The loop is deliberately small and terminal. It does not tune rates, run the
GPU, regenerate the full replay corpus, reopen either Jade restart track, or
make a production/validated-Ampere claim. A successful existing-data transfer
keeps the mechanism opt-in and nominates a fresh held-out live campaign; a
failed transfer records the no-go and removes any unused implementation
surface.

## Acceptance Criteria

- AC-1: The candidate has one strict, default-off configuration and is part of
  replay identity.
  - Positive Tests (expected to PASS):
    - An explicit `ampere_queue_v1` selection is accepted for full-coverage
      `sm_86` L1/L2 replay and appears in the resolved configuration.
    - Omitting the selection follows the existing path without allocating or
      consulting LDGSTS front-end state.
  - Negative Tests (expected to FAIL):
    - Unknown modes, duplicate settings, non-`sm_86` use, sampled traces, or
      incompatible replay modes return a typed input/capability error.

- AC-2: The finite queue implements the frozen paper-derived composite without
  a fitted parameter.
  - Positive Tests (expected to PASS):
    - Five effective LDGSTS instructions on one SMSP can issue consecutively;
      a sixth waits for the earliest accepted entry.
    - Independent SMSPs retain four-cycle local initiation while same-cycle
      candidates share one deterministic two-cycle SM acceptance frontier.
    - With issue attempts starting at cycles 0–4, the one-SMSP sixth/seventh/
      eighth issue cycles are 13/17/21. With all four SMSPs issuing together,
      their sixth issues are 13/15/17/19, seventh issues 21/23/25/27, and
      eighth issues 29/31/33/35 in SMSP order.
    - For issue cycle `t`, the model records
      `u=max(t,Ls)`, `Ls=u+4`, `v=u+13`, `a=max(v,G)`, `G=a+2`, and
      `delta=a-(t+13)` with checked arithmetic.
  - Negative Tests (expected to FAIL):
    - Predicated-off and zero-source records do not reserve an entry, and a
      projected overflow does not partially mutate queue or scheduler state.

- AC-3: Only excess front-end delay changes source readiness; existing
  hierarchy and destination work are not double-counted.
  - Positive Tests (expected to PASS):
    - LDGSTS WAR clears at `a`; source completion is
      `max(existing_response + delta, t + 39 + delta)` for L1 hits, L2/DRAM
      responses, direct bypass replies, fill waiters, and retries.
    - Existing payload work, exact destination wavefronts, and the SM
      destination server are unchanged and remain downstream of source ready.
    - Focused diagnostics report issued tokens, queue blocks, excess-delay
      cycles, and maximum occupancy without an instruction-sized event log.
  - Negative Tests (expected to FAIL):
    - The implementation never adds 13 or 39 to the full 34/244/423 card
      latency, never charges per sector/response/destination wavefront, and
      never lets a delayed response clear before its frozen floor.

- AC-4: A small CPU-only evaluator reaches a terminal transfer decision on
  frozen evidence.
  - Positive Tests (expected to PASS):
    - Mechanism sanity uses W4 `c10` gap 0/64 L1/L2/DRAM cells; transfer uses
      W2/W8 `c10/c11/c12` plus the independent 18-cell warm-L2 coupled
      campaign; the W1-W8 warmed-source table is a double-count guard.
    - The six production workloads are scored descriptively against the
      current 33.128% active-SM MAPE only after the probe transfer score is
      frozen; they are verdict-inert.
    - For every replay/label comparison, normalize the simulator's 64-group
      last-issue value and the live 16,384-group active-SM median by their own
      group counts. The transfer score is unweighted MAE across exactly 30
      cells: all 18 coupled-campaign cells plus the 12 v4 W2/W8 ×
      `c10/c11/c12` × L1/DRAM gap-0 cells. `retained_opt_in` requires this MAE
      to be strictly below the current baseline.
    - The warmed-source guard is unweighted MAE over the retained W1-W8
      cycles/group table and must be no greater than baseline. All six coupled
      L1/L2/DRAM orderings (W2/W4/W8 × `c10/c11`) must remain strict.
    - Sign gates are frozen as: all ten v4 corrected interactions
      `(c11-c10)-(c01-c00)` for W4 × L1/L2/DRAM × gap 0/64 and W2/W8 ×
      L1/DRAM × gap 0 must match their live nonzero sign; all nine coupled
      `c11-c10` L1/L2/DRAM contrasts must match their live positive sign.
      Traffic and instruction cardinality must remain exact. Otherwise the
      verdict is `not_retained` with the exact failed gates.
  - Negative Tests (expected to FAIL):
    - The evaluator rejects missing/mismatched cell keys and cannot select a
      parameter from the six production labels. It performs no live NCU/NVBit
      run and no hash comparison.

- AC-5: The milestone leaves a concise scientific handoff and a clean ordinary
  development path.
  - Positive Tests (expected to PASS):
    - One decision document records the model equation, evidence split,
      aggregate/per-cell results, approximation boundary, terminal verdict,
      and exact claim ceiling.
    - Focused Rust tests for the changed crates and the CPU-only evaluator run
      successfully; formatting is applied only to touched Rust code.
  - Negative Tests (expected to FAIL):
    - The decision does not call the result P3, validated Ampere, production
      cycle prediction, Jade M3 progress, M4/G1 authorization, or publication
      readiness, and it adds no broad workflow, full-corpus regeneration, or
      SHA-based gate.

## Path Boundaries

### Upper Bound (Maximum Scope)

The strict opt-in candidate, focused configuration plumbing, transactional
queue/source composition, compact diagnostics, CPU-only retained-data
evaluator, terminal result artifact, and concise status/roadmap update. If all
frozen gates pass, the candidate remains opt-in and the decision names one
fresh live holdout as the next milestone.

### Lower Bound (Minimum Scope)

The model can be selected explicitly, the first-five/finite-queue and source
floor behavior is exercised by focused tests, the retained W2/W4/W8 evidence
is replayed once, and a terminal retain/no-go decision is recorded. A failed
candidate is a complete result, not a reason to add a fitted correction.

### Allowed Choices

- Can use: existing `sim-arch`, `sim-sm`, replay/configuration plumbing,
  retained full-coverage traces and NCU labels, one compact Python evaluator,
  and the existing Humanize RLCR review loop.
- Cannot use: new GPU execution, NCU/NVBit recapture, workload-name constants,
  rate fitting, per-cell corrections, measured labels as replay inputs,
  absolute/relative address hashes, SHA comparisons, full-corpus regeneration,
  new Jade Track-A/Track-B work, or shared/coherent topology work.
- Preserve: standalone builds, exact instruction/traffic accounting, existing
  cache and destination ownership, and unchanged behavior when the candidate
  is absent.

## Feasibility Hints and Suggestions

### Conceptual Approach

Keep the queue state small: four per-SMSP release-time deques, four local
frontiers, and one shared frontier. Project all same-cycle tokens on a clone in
SMSP order before committing. Queue-full readiness should point to the oldest
acceptance rather than advancing one cycle at a time. Each token occupies one
of five effective outstanding slots (the modeled four-slot queue plus dispatch
latch) only while `now < a`; it is released at `a`. Do not describe this as a
physical five-entry queue.

The existing card latencies already include the uncontended front end. Carry
only `delta` and the issue-anchored floor through the async completion identity.
For cacheable fills, the fill target may retain the timing pair; for bypass,
the direct reply may retain it. This permits response handling to publish a
future source-ready timestamp without changing L2/cache event order. Document
that as an observational ordering approximation.

Prefer one explicit CLI mode over a general parameter file. Do not add the
11-cycle fixed-II comparator unless it falls out almost for free; the research
already rejects it as the primary mechanism.

### Relevant References

- Research decision:
  `/home/jakeshea/Jade_HETR/research/sm86_ldgsts_front_end_research_2026-08-28.md`
- Current LDGSTS group and destination state: `sim/sm/src/issue.rs`
- Static pipeline classification: `sim/sm/src/program.rs`
- Shared L2 response scheduling: `sim/sm/src/system.rs`
- Direct configuration boundary: `sim/direct_run/src/main.rs` and
  `sim/direct_run/src/lib.rs`
- Retained evidence:
  `results/captures/sm86_async_barrier_readiness_20260817_v4`,
  `results/captures/sm86_barrier_pipeline_coupled_20260817_v1`, and
  `results/captures/sm86_cycle_residual_20260817_v1`
- Huerta et al. paper/code:
  `GPU_simulator_rebase/.toolchain/modern-gpu-simulator-micro-2025`
- GASS negative comparator:
  `/home/jakeshea/Jade_HETR/research/external_code/gass_ldgsts_20260828`

## Dependencies and Sequence

### Milestones

1. Milestone M1 — Candidate contract and finite front end.
   - Add the strict opt-in mode, architecture capability, compact state, and
     transactional issue/readiness projection.
2. Milestone M2 — Async completion composition.
   - Carry only excess delay and the source floor through cacheable/bypass
     completion while leaving downstream work unchanged.
3. Milestone M3 — Frozen offline transfer.
   - Replay the retained probe/transfer cells once, reduce the frozen gates,
     then score the six production references descriptively.
4. Milestone M4 — Terminal handoff.
   - Record retain/no-go, update current GPU status/roadmap concisely, remove
     dead candidate surface if the implementation supplies no reusable value,
     and finish the Humanize review.

Dependencies: M2 depends on M1. M3 depends on M2 and uses no live hardware.
M4 depends on the M3 terminal result.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---|---|---|---|---|
| task1 | Add strict default-off `ampere_queue_v1` selection and configuration identity | AC-1 | coding | - |
| task2 | Implement checked per-SMSP capacity/local/shared projection and compact diagnostics | AC-2 | coding | task1 |
| task3 | Compose excess delay with WAR and async source completion across hit/fill/bypass/retry paths | AC-3 | coding | task2 |
| task4 | Add only the focused queue/floor/configuration tests | AC-1, AC-2, AC-3 | coding | task3 |
| task5 | Build the compact retained-data evaluator and run the frozen CPU-only decision | AC-4 | coding | task4 |
| task6 | Write the decision, update current guidance, and run focused checks | AC-5 | coding | task5 |

## Implementation Notes

- Use ordinary code names; do not embed `AC-*`, task, or milestone vocabulary
  into production APIs.
- Do not add generic workflow machinery or broad validation scripts.
- Do not compute or compare SHA256 values as part of this milestone.
- Do not rerun already retained evidence merely to prove that files exist.
- Do not change defaults based only on this retrospective cohort.
- Keep the worktree changes limited to the simulator, one compact evaluator,
  the result artifact, and concise documentation.
