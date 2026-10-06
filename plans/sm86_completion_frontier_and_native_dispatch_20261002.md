# SM86 completion and accuracy program

Revised 2026-10-06 on `GPUSim_main`. This is a technical milestone plan, not
an approval workflow. No `/goals`, reviewers, receipts, or execution gates.

## Current focused work — 2026-10-06 reassessment

The user requested independent reassessment of AO/AP, not acceptance of their
prior completion labels. The broader C0–C8 milestones below remain the research
program; capture completion does not close hardware accuracy or native dispatch.
Only the explicitly marked active phase is the current implementation target; completed negative branches remain evidence, not authority.

| Phase / milestone | Practical completion bar | State |
|---|---|---|
| M0 / compatibility and prior-source review | Read ff275f5c, 811ac667, 7227459b; check actual NVBit exports and live host; separate claims from raw runs | complete |
| M1 / C7-S33AO-R | Strict capacity CLI; per-trial initialized outputs and elapsed checks; rebuild; direct/sanitizer/NVBit/current replay/five-NCU on W8 b9 | complete |
| M2 / C7-S33AP-R | Actual modulo-u32 result oracle, correct thread/warp units, bounded CLI/module lookup; malformed-input rejection; same-cubin seven trials, sanitizer, native trace/replay, five NCU runs | complete |
| M3 / touched-probe regression and reconciliation | Check every changed driver/campaign, retain all failed runs, record actual replay mode/cards/output/runtime; remove inflated and stale completion claims | complete |
| M4 / remaining declared population | Continue each uncaptured family one at a time; source safety and oracle first; keep unsupported/failing rows in the denominator | complete — independently rechecked by M4-E |
| M5 / C1–C6 mechanisms, accuracy, C8 | Actual completion obligations, calibration-only models and untouched evaluation, real native continuation differential, synchronized claims | open — M5-O exact INT8 IMMA capability and M5-N4 residual audit complete; M5-N3 external-evidence residuals remain active |

## Current continuation route — 2026-10-06

| Phase | Practical milestone | State |
|---|---|---|
| Late L1/L2/Quant custom rows | Official input/criterion screen and one full representative native launch chain per row; blockers remain in the denominator | complete scoped — current C7 source/vertical closure reconciled; unresolved rows carry to M5-N |
| Remaining source population | Revisit earlier unexecuted L1 custom markers, then L2/Quant and CUTLASS/vendor paths; classify real source, not comment tokens | complete scoped — C7-S34CS reconciles all 14 CUTLASS rows; explicit blockers remain |
| Capture gaps | Resolve concrete mixed-module/parser and correctness failures where supported; source classification alone is not execution completion | open |
| Model mechanisms and accuracy | Independent calibration/held-out evaluations and real continuation differentials; keep SM-active, GPC, modeled completion, and physical tails distinct | open; not closed by capture coverage |

Use one row's source → official correctness/memcheck → native capture/replay →
independent observations before moving to the next row. Failed experiments stay
visible; no NCU-guided implementation or parameter selection.

### M4 continuation ledger — 2026-10-04

M4 is decomposed into four finite phases so a source review cannot be
mistaken for a live observation:

| M4 phase | Milestone and completion bar | State |
|---|---|---|
| M4-A | Reconcile every uncaptured Makefile-backed family and every source-only/analysis-only directory; identify the executable, bounded input, independent oracle, and an explicit unsupported disposition where no native oracle exists | complete |
| M4-B | Repair any unsafe host oracle/argument path found by M4-A; rebuild the affected source and run a direct correctness plus sanitizer smoke before profiling | complete |
| M4-C | For each admitted family, run one same-binary correctness/NVBit/replay/five-NCU vertical slice; retain failed or physically inapplicable rows with evidence | complete |
| M4-D | Update the inventory and historical milestone ledger, schema-validate all new records, reconcile stale prior artifacts, and close M4 only when every current directory has a live slice or an explicit non-native/unsupported disposition | complete |
| M4-E | Independently recheck the closed population: regenerate any missing declarative manifests, verify every live-record path and matrix population, rebuild all current Makefile targets, repair any host-toolchain blocker, and refresh a live vertical for a repaired family | complete — 42/42 directories reconciled, 37/37 Makefile targets build, 135 document/schema pairs pass, fresh SFU falsifier retained |

### M5 continuation ledger — executed after M4 (scoped phases)

M5 is intentionally not marked complete by the C7 captures. Its next phases
remain mechanism-local and must be completed in order:

| M5 phase | Milestone and completion bar | State |
|---|---|---|
| M5-A | Reconcile completion, launch-envelope, typed-tail, and native-dispatch frontiers from the 48 retained rows; keep every open/refused tail explicit and produce one falsifiable native-tail differential | complete — differential retained typed-tail falsifier |
| M5-B | Fit only calibration-probe observer/service models, materialize predictions before untouched NCU observations, and report interval coverage/width without using scored values as features | complete — scoped async observer holdout |
| M5-C | Obtain a real native continuation witness with complete front-end/pipeline state or preserve exact fallback after a bounded falsifier; no synthetic witness may close this bar | complete — negative branch, exact fallback preserved |
| M5-D | Synchronize C1–C8 claims, rerun affected held-out checks, and close only the mechanisms actually supported by independent evidence | complete — claims synchronized; underlying open surfaces retained |

The initial M5-A snapshot over the twelve new M4 records is negative but
actionable: native macro dispatches are `0/12`, all twelve retain dirty-
writeback and coherence tails, one retains async shared-visibility, and two
retain reduction-RMW tails. Ten simulator lowers are below their NCU medians;
the load-priority and corrected LDGSTS rows are lower-bound falsifiers. This
historical snapshot selected the first M5 falsifier; the scoped M5-A result is recorded below.

### M5-A focused differential — complete (historical execution record)

The first M5-A differential is the `launch_cycle_semantics` executable with
the same rebuilt binary, grid (`84` blocks × `32` threads), dependent-chain
work (`256` iterations), and one launch in both arms:

* `control` has no global tail after the barrier;
* `red` issues one global reduction per warp after the same barrier.

The falsifier is declared before capture: compare the simulator's
`red-control` lower-bound delta with the independent NCU median delta. A
negative or materially non-monotone measured delta, a lower-bound crossing, a
trace/instruction mismatch, or an unresolved typed-tail transition is evidence
against treating the reduction tail as a closed completion model. No NCU value
will be used to alter simulator parameters. Correctness, sanitizer, NVBit,
deterministic replay, and five independent marker-scoped NCU repetitions are
required for each arm; failed attempts remain retained.

### M5-A result — complete with typed-tail falsifier

The matched control/reduction capture is retained in
`docs/sm86_m5a_native_tail_differential_launch_cycle_v1.json`. Both arms use
the rebuilt launch-cycle executable at `84 × 32`, `256` dependent iterations,
and one launch. Seven host-oracle trials, Compute Sanitizer, NVBit, and two
byte-identical simulator replays passed for each arm; NCU executed five
independent application-replay repetitions per arm with the same binary and
arguments. Trace/NCU instruction counts match.

The simulator GPC lower-bound delta is `671` cycles (`10290 - 9619`), while
the independent NCU median delta is `740` cycles (`10618 - 9878`). The
positive direction is preserved, but the `69`-cycle delta difference is kept
as an observation rather than fitted. The control arm has a complete modeled
event frontier; the reduction arm explicitly remains incomplete with
reduction-RMW, dirty-writeback, and coherence tails. Native macro dispatch is
`0/2`. This closes the M5-A differential/reconciliation bar without claiming
native completion or continuation.

The inventory now retains 48 stage records (36 earlier rows plus the twelve
M4-C current rows), 37 current CUDA entries with live artifacts, five explicit
analysis-only dispositions, and no unassigned directory. Expanded matrix and
arm observations remain separate from the stage count; the reconciliation
record reports 420 observation units, 388 lower-bound-consistent units, and
21 lower-bound falsifiers. Historical rows do not all expose a GPC envelope,
so the new differential records that envelope separately rather than
retroactively rewriting older rows.

### M5-B result — complete for one declared calibration slice

`docs/sm86_m5b_calibration_holdout_reconciliation_v1.json` reconciles the
existing async observer holdout without broadening its claim. The prediction
was materialized before profiling the disjoint group-128 target, used only
group-16 and group-32 calibration samples, and explicitly consumed no
group-128 NCU values. The five target repeats were collected afterward; the
`37737..38655` interval covered every repeat, with a `38196` point estimate,
`38348` median, `918`-cycle width, and `0.396%` point APE. The simulator's
`168898`-cycle lower bound still has an open semantic upper, so this is a
finite observer interval rather than a completion bound. M5-B is complete at
this scoped calibration-only bar; family-wide C4/C5 accuracy remains partial.

### M5-C result — complete through the permitted negative branch

The current `sim-run --features reference` build was rerun on the bounded real
SM86 trace and retained under
`results/alpha/m5c_native_continuation_20261004/`; the compact result is
`docs/sm86_m5c_native_continuation_fallback_v1.json`. The run returned zero
with a footer after `10,033` records (`10,032` boundary records and `5,016`
pre/post pairs). The typed-boundary summarizer found zero complete boundaries,
zero complete transitions, and `10,032` exact-fallback boundaries, all for the
missing typed model-macro region contract. The footer and replay each report
zero native macro dispatches; no synthetic witness was used. The replay keeps
dirty-writeback and coherence tails open.

This satisfies M5-C's negative branch: exact fallback is preserved after a
bounded falsifier. It does not provide the positive real native continuation
witness, which remains an explicit open capability rather than being relabeled
as complete.

### M5-D result — complete as a claim-synchronization phase

`docs/sm86_m5d_claim_synchronization_v1.json` is the synchronized C1-C8
claim map. It keeps C2's observed-fact layer complete, C6 on exact fallback,
and C7's 48-stage population reconciliation, while leaving C1/C3/C4/C5/C7/C8
open surfaces explicit. No unfinished mechanism is promoted by a renamed
diagnostic, no synthetic witness is promoted to native, and no NCU value is a
model feature or fitted parameter.

Focused regression passed: `sim-run` 86 tests, `sim-sm` 493 tests (with only
the two machine-local integration tests ignored because their optional
sidecars were not supplied), 17 semantic/artifact Python tests, 123 matching
JSON-schema pairs, and `git diff --check`. The one known unrelated failure is
the pre-existing compile-manifest schema-name expectation
`sm86_static_sass_artifact_v1` versus emitted `sm86_static_sass_alpha_v1`.

M5-D therefore closes the synchronization work, not the physical completion
or broad hardware-accuracy program. The top M5 row remains open until those
independent obligations have evidence.

### C8 continuation ledger — after M5-D

| C8 phase | Milestone and completion bar | State |
|---|---|---|
| C8-A | Recheck active CLI/source surfaces, retired interfaces, inventory links, and schema coverage against the synchronized claim map; do not delete historical raw evidence | complete — active surfaces clean; historical references retained |
| C8-B | Rerun only corrected workload failures or record a concrete unsupported/OOM/OOD disposition; preserve all failed attempts and exact binary/argument identities | complete — six concrete blockers retained |
| C8-C | Update the runbook/status with supported claims and remaining open mechanisms; close C8 only if no unfinished mechanism is called complete | complete — runbook and claim map synchronized |

### C8-A result — active interface and link recheck

`docs/sm86_c8a_interface_reconciliation_v1.json` records successful help
checks for `sim-run` and `tools/run_sm86_workload.py`, with no retired
`--enable-cupti-activity` token in active simulator/workload source surfaces.
Historical SimBench README/schema/golden references are retained and listed as
historical material rather than silently removed. The inventory links resolve
for all 42 directory entries and 48 retained stages; the four M5 records also
resolve, and the 123 matching schema pairs remain valid. This closes C8-A but
does not close corrected workload failures or C8 as a whole.

### C8-B result — corrected-failure dispositions retained

`docs/sm86_c8b_corrected_failure_dispositions_v1.json` records six remaining
non-substituted blockers: one external-tracer contract blocker, two unsupported
SM86 tensor-pipeline rows (INT8 IMMA and TF32 HMMA), and three vendor cuBLAS
static-module blockers. Each retains its exact artifact, binary/argument
identity where available, trace/replay reason, and no-substitute disposition;
NCU values are not consumed by simulator or fitting. This closes the C8-B
disposition bar while keeping all rows in the coverage denominator.

### C8-C result — runbook and claim synchronization complete

`docs/sm86_c8c_runbook_synchronization_v1.json` links the M5/C8 records from
`docs/README.md` and confirms that `STATUS`, `PROJECT_MEMORY`, and this plan
retain the C1/C3/C4/C5/C6/C7/C8 open surfaces. No unfinished mechanism is
called complete, no positive native-dispatch claim is introduced, and
documentation does not clear semantic tails. C8 is therefore complete as a
scoped runbook/claim phase; the top M5 program remains open for physical
completion, broader accuracy, and a positive native continuation witness.

### C4-G result and C6-A/B/C/D/E/F/G/H branches — C6-H scoped negative branch complete

C1-R-B is complete as a negative reconciliation: the store/fence effects are
measured, while the system-scope crosswalk blocker is reproduced and retained
as not score-applicable. C3-A through C3-D then validated the model-local
acknowledgement, async-refusal, checked-arithmetic, tail-fact, bounded
perturbation, and exact-fallback seams. C3-E attempted the async reference
continuation and produced zero complete boundaries. C3-F preserved the
physical upper as unknown and selected the next independent accuracy/observer
seam; dirty-writeback/coherence and native continuation remain open, and NCU
values cannot select a model change. C4-A reconciled the existing async-group
and destination-layout observer
contracts; that historical seam led to the later C4-B through C4-I slices;
those observer results remain separate from completion-upper claims.

### C3-G / C6-I phase — opt-in completion-tail audit (complete scoped)

This bounded phase tested the existing model-local writeback/coherence service
path against the typed completion contract. It demonstrated model-local
writeback/visibility progress, but it cannot promote a silicon completion upper
or create a native witness from simulator options.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C3-G1 | Source-review TailFlags, explicit writeback/coherence queues, typed boundary projection, and macro eligibility; verify opt-in services cannot silently clear unknown tails | complete — source preserves `dirty_writeback`/`coherence` capability flags and refuses incomplete boundaries |
| C3-G2 | Replay one real store/device-fence trace with explicit bounded writeback/coherence service options; record frontier, tails, and termination without NCU fitting | complete — 512-sector writeback service observed; tails retained |
| C3-G3 | Compare default versus opt-in model-local outputs; if tails remain unknown, retain the refusal and identify the independently observable state needed next | complete — writeback arm lower unchanged; visibility arm moves model-local frontier +6 while SM-active stays separate; tails remain unknown |
| C3-G4 | Synchronize the C3-G/C6-I artifact, plan, status, and continuation memory; close only the model-local audit bar | complete — artifact and claim surfaces synchronized |

**C3-G/C6-I result.** The fresh one-block store/device-fence trace was replayed
with default, explicit carded-DRAM writeback/coherence, and explicit
L1-visibility/resident-writeback options. The writeback arm completed modeled
service for 512 sectors (16,384 bytes) at 11,121, while the visibility arm
moved the model-local completion frontier to 11,146; the separate SM-active
frontier stayed 11,140. Both arms retained `dirty_writeback` and `coherence`
as unknown, and the completion status remained
`incomplete_unmodeled_retirement_tails`. The typed boundary and macro guards
therefore continue to refuse incomplete completion state before native
mutation. This closes only the model-local audit; no physical upper, hardware
completion, NCU fit, or native scheduling claim is made. Evidence:
`docs/sm86_c3g_optin_writeback_coherence_tail_audit_v1.json`.

### C3-H / C6-J result — visibility protocol negative branch

The same real store/device-fence trace was replayed in reference mode with the
explicit L1 visibility/resident-writeback path. A 256-record bound first
returned the retained limit failure; a 4096-record run reached a footer with
1130 reconstructed boundaries (565 pre/post pairs), including 1024 visibility
accesses, two fences, two invalidations, and 32 L2 batches. Every boundary was
incomplete for `unmodeled_completion_tails`, exact replay was selected, and
native macro dispatch stayed zero. Fence scope/order are observable event
fields, but ownership/read-from, line generation, global order, and hardware
completion remain unknown. Evidence:
`docs/sm86_c3h_visibility_reference_boundary_v1.json`.

### C3-I / C6-K negative branch — complete; independent ownership/generation remains open

This completed phase reviewed the source and contract needed to turn a visibility
event prefix into a complete continuation. It either needed an independent
owner/generation/read-from witness or exact fallback; no queue counter or
same-trace self-derived witness was accepted.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C3-I1 | Enumerate owner/read-from, line-generation, fence-scope, and global-order fields across trace, typed boundary, and queue state; identify missing durable edges | complete — source review identifies all four as outside the closed macro contract |
| C3-I2 | Attempt one independently sourced owner/generation continuation on a real SM86 trace; reject self-derived or aggregate-only witnesses | complete — independent extractor found no owner/generation/read-from witness |
| C3-I3 | Preserve exact fallback and update the claim map if the witness is absent; close only the supported boundary | complete — 1130 visibility boundaries remain exact fallback; physical upper open |

**C3-I/C6-K result.** The independent raw-protobuf/static-SASS extractor was
run against the real store/device-fence capture without simulator or NCU
inputs. It confirmed 17 store records and `MEMBAR.SC.GPU`/`ERRBAR`, but
reported no owner/read-from, line-generation, global-order, fence-scope, or
hardware-completion relation. The reference visibility run therefore retains
exact replay for all 1130 boundaries and no native macro dispatch. This closes
the negative witness branch only; the next supported work is independent
ownership/generation evidence or continued exact fallback. Evidence:
`docs/sm86_c3i_owner_generation_negative_v1.json`.

### C3-J / C6-L result — detailed-effect pre-mutation guard

The narrow scalar actor now rejects any supplied detailed effect span whose
cardinality does not equal the collapsed instruction count or whose issue times
are non-monotone, before mutating warp, residency, counters, or queue state.
The model-local recurrence path with no supplied effects remains compatible for
its existing unit seam. A regression proves a truncated effect list returns
`detailed_effect_span_not_closed` with an unchanged boundary snapshot.

### C3 continuation ledger — after C1-R-B

| C3 phase | Completion bar | State |
|---|---|---|
| C3-A | Reconcile model-local acknowledgements, scoped waits, async refusals, checked completion arithmetic, and explicit tail propagation with focused regressions | complete — local obligations validated |
| C3-B | Repair per-warp completion readiness and SM-active projection if failing regressions confirm the source defects; replay retained native store/fence traces with a declared acknowledgement-latency perturbation, keeping terminal tails separate | complete — two local defects fixed and perturbation passed |
| C3-C | Independently extract raw store/fence tail facts from the same native traces; keep hardware completion, ownership, generation, and scope unknown | complete — observed facts retained |
| C3-D | Reconcile async-copy/visibility tail facts and one bounded copy perturbation without converting protocol facts into a completion upper | complete — protocol prefix retained; visibility tail open |
| C3-E | Attempt one independently witnessed completion/visibility continuation; if absent, preserve exact refusal and keep the physical upper unknown | complete — async reference branch exact-fallback negative |
| C3-F | Preserve the physical completion upper as unknown and select the next independent accuracy/observer seam without treating refusal as hardware completion | complete — physical upper remains unknown; C4-A selected |
| C3-G | Audit explicit model-local writeback/coherence and L1 visibility service against typed completion flags; preserve unknown tails | complete — writeback/visibility progress observed; tails remain unknown |
| C3-H | Run a real visibility-protocol reference boundary with a retained limit attempt and a footer-terminated run; preserve exact fallback when scope/ownership state is incomplete | complete — 1130 boundaries exact fallback; owner/generation/read-from remain unknown |
| C3-I | Obtain an independently sourced owner/generation/read-from continuation or preserve the exact-fallback boundary without self-certification | complete — independent witness absent; exact fallback preserved; physical upper remains open |
| C3-J | Guard supplied detailed effect spans before scalar actor mutation; preserve the model-local recurrence compatibility path | complete — truncated/non-monotone spans refuse pre-mutation with regression |

### C4-H/I scoped phases — independent-family observer protocol and domain sync complete

C3-I/C6-K closes the supported negative continuation branch, and C4-H/I
closes the independent-family observer slice. The scoped C5-C synthesis is
complete; any future holdout must remain predeclared, disjoint, and separate
from simulator execution while C3/C6 physical mechanisms remain open.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C4-H1 | Select one family from the reconciled M4 population whose observer feature and calibration/holdout split are explicit; no scored target values enter model selection | complete — constant-load dependent-chain family selected |
| C4-H2 | Materialize the target prediction before fresh NCU collection and retain simulator/observer frontiers separately | complete — 64-block interval written before target NCU |
| C4-H3 | Collect the disjoint target repeats, report coverage/width/OOD, and stop if the family boundary falsifies transfer | complete — 0/5 coverage; transfer falsifier retained |
| C4-H4 | Synchronize the C4-H artifact and leave physical completion/native dispatch claims unchanged | complete — reconciliation and claim surfaces synchronized |

**C4-H result.** The constant-load dependent-chain family used blocks 8/16/32
for calibration and an untouched 64-block target. The frozen affine observer
interval was `13805.3..13848.1` GPC cycles; five target repeats were
`14019, 13887, 13855, 13896, 13876`, so coverage was `0/5` and point APE
was `0.434%`. The miss is retained without refit or interval widening. The
target simulator lower bound is `62287` cycles with dirty-writeback/coherence
tails open, and no NCU value entered simulator execution. Evidence:
`docs/sm86_c4h_constant_load_reconciliation_v1.json`.

### C4-I result — observer-domain synchronization

C4-I reconciles the C4-G domain boundary with the independent constant-load
transfer falsifier. The scoped positive SFU slice remains valid only for its
dependent-chain regime; cross-family constant-load transfer and the earlier
quad interventions are rejected. Broad family transfer, physical completion,
and native dispatch remain open. Evidence:
`docs/sm86_c4i_observer_domain_reconciliation_v1.json`.

### C5-A result — scoped held-out evaluation

C5-A evaluates the frozen C4-H constant-load interval on the untouched
64-block target. The interval `13805.3..13848.1` covered zero of five target
repeats; point APE was `0.434%`, width was `42.86` cycles, and repeat
dispersion (stdev `64.67`) is retained. The simulator lower bound is `62287`
cycles with dirty-writeback/coherence tails open, so this is not a completion
upper or broad accuracy claim. Evidence:
`docs/sm86_c5a_constant_load_holdout_evaluation_v1.json`.

### C5-B scoped disjoint-family holdout — complete negative branch

C5-B used a new disjoint mode-1 family after the constant-load miss. It stopped
at a falsifiable boundary and did not widen the C4-H interval or use target NCU
values to alter replay.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C5-B1 | Select a disjoint family and calibration/holdout split with a declared observer feature and no target leakage | complete — mode-1 independent constant-load family selected (blocks 8/16 calibration, 32 target) |
| C5-B2 | Materialize the target prediction before collecting fresh NCU repeats; retain simulator frontiers separately | complete — mode-1 target interval written before NCU |
| C5-B3 | Evaluate coverage, width, signed error, dispersion, and OOD; retain misses and stop if transfer fails | complete — 0/5 coverage; 2.578% point APE; transfer miss retained |
| C5-B4 | Synchronize the C5 artifact and leave completion/native-dispatch claims unchanged | complete — reconciliation and claim surfaces synchronized |

**C5-B result.** The disjoint independent-load mode-1 family used blocks 8/16
for calibration and an untouched 32-block target. Its frozen interval
`11003..11293` missed all five target repeats (`11486,11368,11417,11449,11443`),
with `0/5` coverage and `2.578%` point APE. The miss is retained without
refit or widening; simulator lower/frontier outputs and semantic tails remain
separate. Evidence:
`docs/sm86_c5b_constant_load_mode1_reconciliation_v1.json`.

### C5-C result — cross-family boundary synthesis (complete scoped)

C5-C synthesizes the C4/C5 positive and negative slices into an explicit
observer-domain boundary. It closes only the scoped evaluation phase; it cannot
call broad accuracy, physical completion, or native dispatch complete.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C5-C1 | Reconcile C4-H/C5-A and C5-B intervals, misses, widths, signed errors, and repeat dispersion without target leakage | complete — both negative intervals and repeat statistics reconciled |
| C5-C2 | Define the supported observer partial order and explicit out-of-domain families; preserve all target misses | complete — SFU slice scoped; constant-load/quad transfer out of domain |
| C5-C3 | Synchronize plan/status/memory/runbook and leave broad accuracy open | complete — C5-C artifact and claim surfaces synchronized |

**C5-C result.** The cross-family synthesis preserves the scoped SFU positive
slice and retains both constant-load misses (`0/5` coverage at `0.434%` and
`2.578%` point APE) plus the earlier quad OOD/occupancy misses. Coverage,
width, signed error, semantic lower bounds, and completion tails remain separate
partial-order dimensions; no scalar score or post-target refit is introduced.
Broad accuracy, physical completion, and native dispatch remain open. Evidence:
`docs/sm86_c5c_cross_family_boundary_v1.json`.

### C4 continuation ledger — after M4-E

| C4 phase | Completion bar | State |
|---|---|---|
| C4-A | Reconcile existing prediction/holdout documents: calibration and holdout dimensions must be disjoint, predictions must precede target NCU collection, NCU targets must not feed simulator execution, and model-local endpoints must not be labeled physical completion | complete — `docs/sm86_c4a_observer_contract_reconciliation_v1.json`; semantic upper remains open |
| C4-B | Add one new disjoint calibration/holdout dimension with a declared observer interval and mechanism-local counters; keep target values post-prediction and report coverage/width without using them for model selection | complete — SFU warp-count holdout; one controlled slice |
| C4-C | Test observer transfer on a distinct mechanism/family or an explicit OOD intervention, with a predeclared miss/coverage rule and no semantic-upper promotion | complete — quad-chain OOD miss retained |
| C4-D | Refit only on a newly declared in-domain mechanism family after the OOD miss, then predict an untouched target before NCU; retain family-specific and cross-family results separately | complete — quad-family shape miss retained |
| C4-E | Investigate the observed quad-family nonlinearity with a predeclared mechanism-local feature/intervention, then run one fresh prediction/holdout without retrofitting the missed targets | complete — block-count intervention miss retained |
| C4-F | Select a new observer feature or stop with a falsifiable nonlinearity boundary; never reinterpret OOD misses as completion or fit the simulator to scored targets | complete — batch-feature interval has one retained miss |
| C4-G | Decide the supported observer domain from the retained positive/negative slices; preserve misses and stop claiming broad accuracy until an independent family protocol exists | complete — scoped domain boundary; broad accuracy remains open |
| C4-H | Run one independent-family pre-NCU observer/holdout protocol and preserve a transfer miss without refit | complete — constant-load 64-block target misses frozen interval |
| C4-I | Synchronize the observer-domain boundary with the new family falsifier; do not promote broad accuracy or completion | complete — scoped domain retained; broad transfer open |

**C4-F experiment (before its target NCU).** Replace the global linear
slope with `ceil(grid_blocks / 84)` as an empirical service-batch feature;
this is not the hardware occupancy/wave count. Fit the fresh 42/63/105/126-block calibration measurements (quad, W8, 512
iterations). The already collected 84-block row was previously scored in C4-D
and is excluded from this fit. Obtain separate interval-calibration medians at blocks 21/35/56/70/77/91/119/140/154 (five
repeats per shape). Predict block 210 before its NCU capture. Interval rank
is ceil((n+1)*0.9), using n=9 distinct shapes rather than 45 repeats; no
coverage guarantee is claimed for this designed extrapolation. Do not refit
or widen after target observation. Physical completion stays unknown.
`workloads/research_probes/sfu_frontier/observer.py` implements this numerical
model, not an execution gate. Full four-launch capture timed out at 180s;
a selected-start attempt returned zero launches, so it is inapplicable. The
source increments the launch counter only after accepted launches; start>1
cannot reach its requested counter. Preserve both attempts and use one
bounded 360s full-capture retry; fix the selection counter separately.

**C4-B result.** The SFU warp-count observer used five application-replay NCU
calibration repeats for warp counts 1, 2, and 4, then materialized its
`34205.4..34521.3` empirical interval before collecting the warp-8 target.
The five target values (`34374,34312,34319,34295,34282`) were all covered and
the point APE was `0.150%`. The target simulator lower bound is retained
separately (`38126` cycles); target NCU values never entered `sim-run` or
model selection. This closes C4-B as one controlled observer slice, not
family-wide accuracy or physical completion. Evidence:
`docs/sm86_c4b_sfu_warp_observer_reconciliation_v1.json`.

**C4-C result.** The frozen dependent-chain interval was applied unchanged to
an explicit quad-chain OOD target at the same warp count and launch protocol.
All five target values (`69123,69122,69187,69143,69127`) missed the
`34205.4..34521.3` interval (coverage `0.0`, point APE `50.289%`). The miss is
retained as a transfer falsifier; no refit, interval widening, simulator timing
change, completion claim, or native dispatch claim is allowed. Evidence:
`docs/sm86_c4c_sfu_ood_observer_reconciliation_v1.json`.

**C4-D result.** A new quad-family calibration at iterations `512` and warp
counts `1/2/4` produced the pre-target interval `21719.6..21922.9` for an
untouched warp-8/512 target. All five target values (`36173,36163,36158,36158,36170`)
missed it (coverage `0.0`, point APE `39.659%`). The family-specific miss is
retained; no target value was used to refit or widen the model. Evidence:
`docs/sm86_c4d_sfu_quad_observer_reconciliation_v1.json`.

**C4-E result.** A fixed warp-8 quad block-count observer used calibration
blocks 42/63/126 and materialized `80236.0..90313.8` before collecting an
untouched 168-block target. All five target values (`68904,68914,68893,68904,68903`)
missed it (coverage `0.0`, point APE `23.759%`). The occupancy intervention
miss is retained without refit or interval widening; target NCU values did not
enter simulator execution or model selection. Evidence:
`docs/sm86_c4e_sfu_quad_occupancy_reconciliation_v1.json`.

**C4-F result.** A three-feature empirical observer (intercept, block count,
and `ceil(blocks/84)` batch feature) was fit on fresh shapes 42/63/105/126,
with nine separate interval-calibration shapes and a 210-block target. The
prediction interval was `101574.5..101742.0`; four of five target repeats were
covered (`0.8` raw coverage), one value (`101754`) missed by 12 cycles, and the
point APE was `0.028%`. This is a designed extrapolation with no coverage
claim, not physical CTA-wave or occupancy evidence. The selected-launch
attempt was preserved as empty; the bounded full retry completed and supplied
the target trace. Evidence:
`docs/sm86_c4f_sfu_quad_wave_reconciliation_v1.json`.

**C4-G result.** The observer domain is now explicitly bounded by the C4-B
controlled positive slice and the C4-C/C4-D/C4-E/C4-F falsifiers. The project
retains the one dependent-chain observer interval and the empirical batch slice
as scoped results only; quad cross-family transfer, quad warp/shape transfer,
raw block-count occupancy extrapolation, and designed batch extrapolation are
not promoted. Physical completion and native dispatch remain unknown. Evidence:
`docs/sm86_c4g_observer_domain_boundary_v1.json`.

### Next active seam after C4-G — mechanism-local completion/native continuation

C4-G closes the observer-domain decision, not C4/C5 accuracy globally. The
positive dependent SFU slice is retained only as one controlled observer result;
quad transfers and the batch extrapolation remain rejected or partial. The
physical completion upper is still unknown, and no NCU target has entered the
simulator. The next implementation seam returns to C1-R/C3/C6: repair or
falsify the store/fence completion mismatch, preserve exact fallback, and make
bounded attempts at a real native continuation witness without synthetic
promotion. C6-A's fresh reference attempt reached its 10000-record limit
without a footer; C6-B preserves the prior footer-terminated fallback, and
C6-C closes a narrow no-front-end real dispatch; C6-D/E close a narrow
default-front-end snapshot transition; C6-F/G preserve the multi-warp boundary,
and C6-H now closes the scoped real store/fence negative branch; broader completion remains open.

### C6 continuation ledger — after C4-G

| C6 phase | Completion bar | State |
|---|---|---|
| C6-A | Make one bounded real-trace reference-feature attempt; preserve incomplete/limit failures and do not call them complete fallback | complete — `docs/sm86_c6a_native_continuation_bound_probe_v1.json`; no positive witness |
| C6-B | Obtain a footer-terminated complete typed witness or preserve a footer-terminated exact fallback; synthetic fixtures cannot close the native bar | complete — footer-terminated real exact fallback preserved |
| C6-C | Obtain a positive real native continuation witness with complete typed state/effects, or retain the explicit open blocker without relabeling fallback | complete — one real scalar trace dispatches in reference-only no-front-end mode; readiness diagnostics/front-end remain open |
| C6-D | Obtain a positive default-front-end real witness or preserve the explicit front-end refusal; no readiness mismatch may be silently treated as exact | complete — narrow default-front-end dispatch; readiness boundary retained |
| C6-E | Model and validate the typed front-end/cache transition for the narrow scalar probe, or preserve the explicit production boundary without claiming front-end continuation | complete — narrow cache/stream snapshot transition; broader state classes/readiness exactness remain open |
| C6-F | Extend beyond the one-warp terminal probe to a distinct real front-end continuation, or preserve the broader production blocker | complete — real two-warp refusal preserved |
| C6-G | Decide whether multi-warp/global state can be typed without self-certification; otherwise preserve the explicit production boundary and return to completion/native-tail seams | complete — multi-warp boundary preserved |
| C6-H | Return to C1-R/C3 completion tails and retain the one-warp continuation domain without broadening it by heuristic | complete — real store/device-fence branch exact-fallback negative; physical upper remains unknown |
| C6-M | Preserve the async-tail refusal boundary in the typed scalar seam; no live async state may cross before an independent visibility witness | complete — focused semantics regression added; async visibility remains open |

#### C6-H execution milestones

| Sub-milestone | Completion bar | State |
|---|---|---|
| C6-H1 | Review the typed boundary and actor source for store/fence completion state; enumerate every refusal path and add no heuristic promotion | complete — source review confirms tails/feedback/front-end state refuse before mutation |
| C6-H2 | Run a real store/fence NVBit trace through the reference boundary path with a bounded record budget; retain footer or limit outcome and exact refusal reasons | complete — one-block live trace footer, 34 reconstructed boundaries, all exact replay |
| C6-H3 | If a footer-terminated path is available, compare detailed versus any candidate continuation without using NCU; otherwise preserve the negative branch and identify the next independently observable state | complete — no complete typed boundary or witness; exact fallback retained |
| C6-H4 | Update the C6-H artifact, status/memory, and plan; close only the supported seam and carry unresolved physical completion/native scheduling forward | complete — C6-H artifact and claim surfaces synchronized |

**C6-B result.** The prior footer-terminated real SM86 reference run is now
reconciled as the permitted negative branch: `33290` typed boundary records,
`0` complete boundaries/transitions, and `33290` exact fallbacks. Native macro
dispatch is `0`; no synthetic witness is used. Evidence:
`docs/sm86_c6b_native_continuation_fallback_v1.json`. The default-front-end
refusal is recorded as the C6-D negative control; C6-D/E now have a narrow
positive snapshot transition, C6-F/G preserve the multi-warp boundary, and
C6-H now closes the scoped real store/fence negative branch; broader completion remains open.

**C6-C result.** The hand-authored no-memory SM86 cubin was captured through
NVBit and replayed with a complete typed boundary after explicitly disabling the
reference-only front end. A separate detailed replay supplied six issue effects
(`244,246,271,275,288,294`) and a terminal barrier/exit state. The witness run
performed one typed native macro dispatch with zero exact fallbacks and matched
the detailed run's instruction count, counters, completion frontier, terminal
event time, and SM activity. Readiness diagnostics differ because the
compressed path does not replay every issue-causal diagnostic increment; that
frontier is kept separate. With the default front end enabled, the same witness
falls back with zero dispatch. This closes only the narrow reference-only
transition seam. Evidence:
`docs/sm86_c6c_positive_real_scalar_dispatch_v1.json`.

**C6-D negative control.** The same real witness explicitly refuses with the default
front end enabled (`front_end_or_scalar_pipeline_transition_not_implemented`),
with zero dispatch and zero fallback counter increments before exact replay is
kept. This is the explicit no-snapshot front-end boundary; the positive
snapshot transition is recorded separately below. Evidence:
`docs/sm86_c6d_default_frontend_refusal_v1.json`.

**C6-D/E result.** The same hand-authored scalar probe was rerun with the
production/default front end enabled. The witness supplied the complete typed
instruction-cache/stream exit snapshot; one typed macro dispatch occurred with
zero fallbacks, and detailed and macro replays matched counters, cycles,
completion frontier, terminal event time, and SM activity. Readiness diagnostics
remain separate and differ because compressed execution skips issue-causal
counter accumulation. The prior no-snapshot run remains the negative control.
Evidence: `docs/sm86_c6d_positive_default_frontend_dispatch_v1.json`.

**C6-F result.** A real two-warp scalar cubin captured 12 instructions with
no memory records and a complete typed entry. The supplied one-warp witness
refused before mutation with `future_relevant_warp_or_response_state_not_closed`;
footer-terminated exact replay remained authoritative and native dispatch stayed
zero. Evidence: `docs/sm86_c6f_multiwarp_frontend_boundary_v1.json`.

**C6-G result.** The one-warp no-front-end and default-front-end snapshot
transitions remain the supported narrow domain. A real two-warp trace reaches a
complete typed entry but refuses before mutation with explicit multi-warp/global
state closure missing; exact replay remains authoritative. Evidence:
`docs/sm86_c6g_multiwarp_domain_boundary_v1.json`. C6-H records the scoped
store/fence negative branch without widening this domain heuristically.

**C6-H result.** A fresh one-block SM86 store/device-fence vertical passed the
host correctness oracle, NVBit capture, static SASS/control extraction, and
deterministic normal replay. The independent tail extractor observed 17 store
records and `MEMBAR.SC.GPU`/`ERRBAR`, while fence scope, ownership/read-from,
line generation, and hardware completion remained unknown. A reference-feature
run reached a footer with 34 reconstructed boundary records (17 pre/post
pairs); every boundary was incomplete for `unmodeled_completion_tails` and
selected exact replay, with zero native macro dispatches. This is the supported
negative completion/fence branch, not a physical completion upper or scheduling
claim. Evidence: `docs/sm86_c6h_store_fence_completion_boundary_v1.json`.

C3-B source inspection found two concrete suspected defects, independent of
NCU: completion feedback waits on *any* warp's unpriced write despite its
per-warp contract; and the SM-active observation projection reads the full
terminal frontier rather than the already separated SM-active frontier.
Regressions must fail before changing these paths. The live-trace check will
use the retained store-none/device-fence traces with acknowledgement latency
`0` versus `4096` simulator cycles, equal hit/miss service `1` cycle/sector,
and queue capacity `16`. These are deliberately artificial test inputs, not
fitted physical rates or new hardware predictions. Expected invariants:
every write receives one ack; no-fence post-exit acks extend completion but
not SM-active observation; a dependent fence keeps its delay in SM activity;
dirty-writeback/coherence tails remain open.

**C3-A/B result.** The first focused regression failed, exposing a real
per-warp bug: `completion_ready_at` rejected a warp's fence whenever *any*
other warp had an unpriced write. The fix scopes the refusal to the issuing
warp while retaining global unpriced state for terminal drain. A second
projection defect was then confirmed: `sm_active_observation_frontier` used
the full terminal event time, so a post-exit acknowledgement moved the
SM-active observation. It now uses the separate SM-active terminal frontier.
The retained store traces replayed with acknowledgement latency `0` and
`4096`: no-fence SM-active stayed unchanged while full completion moved by
`4096`; the device-fence arm moved both because its fence depended on the
acknowledgement. Evidence:
`docs/sm86_c3b_completion_feedback_live_perturbation_v1.json`.

**C3-C result.** Independent raw-protobuf/static-SASS extraction of the same
store-none/device-fence traces sees `1428` stores per arm and an explicit
`MEMBAR.SC.GPU` only in the device arm, but hardware completion tokens,
owner/read-from, line generation, global order, and closed fence-scope
semantics remain unknown. Evidence:
`docs/sm86_c3c_store_fence_tail_facts_v1.json`.

**C3-D result.** The async visibility trace contains `8192` lane copy
operations and `2048` committed/waited groups with no terminal unwaited group.
Uncarded versus carded-DRAM replay is identical because the model reports a
destination-local limit and zero destination service observations. This
negative perturbation does not close async shared visibility, writeback, or
coherence. Evidence:
`docs/sm86_c3d_async_visibility_tail_reconciliation_v1.json`.

**C3-E result.** The async trace was sent through the reference event loop.
The first bounded attempt reached its `10000`-record cap; the rerun completed
with a footer at `33290` boundary records and `16645` pre/post pairs. Complete
typed boundaries and complete transitions were both `0`; all `33290` records
required exact fallback for the missing typed model-macro region contract.
Evidence:
`docs/sm86_c3e_async_reference_continuation_v1.json`. This closes the
negative continuation branch, not physical async visibility or completion.

M4-A/B source review found seven admitted Makefile families: async-source
phase (using `async_visibility_phase`), GA102 atomic LTS mapping, fixed-result
load priority, launch-cycle semantics, LDGSTS burst release, plus the already
captured safe bypass and SMSP-contention arms. The review repaired unchecked
numeric parsing in the atomic-LTS, load-priority, and launch-cycle hosts,
strengthened every-trial sink/elapsed checks on the repaired fixed-result and
SMSP hosts, added a device-scope release/acquire litmus, and bounded the
LDGSTS marker contract. Direct smokes rejected malformed arguments and passed
the independent output/timestamp oracles before profiling.

**M4-C/D result.** Twelve new/repaired vertical records are now retained:
seven Makefile families plus branch, memory-domain, pipeline-category,
value-differential, and device-coherence source families. Each admitted row
has same-binary correctness, Compute Sanitizer, NVBit, deterministic replay,
and five independent NCU observations. Two lower-bound falsifiers are explicit:
load-priority (`70,215` simulator lower vs `65,485` NCU median) and corrected
LDGSTS (`32,695` vs `31,835`); they are not fitted. The async, atomic-LTS,
launch-cycle, source-family, bypass, and SMSP rows remain observational with
open completion tails. The LDGSTS process-start trace timeout and two wrong
NCU launch selections are retained under its v2 result root. The inventory is
`48_live_probe_stages_reconciled_m4c_verticals_complete`; every current
directory now has a live artifact or explicit analysis/source disposition.

**M0 findings.** ff275f5c changes only channel.hpp: byte reservation/publication
counters, device release/acquire publication, system-scope doorbell and host
atomic read/ack, plus atomic host thread state. `get_thread()` and old pthread
registration symbols remain; headers/core library/build system were not replaced.
The installed archive exports both old pthread functions. The upstream
[1.8.1 release](https://github.com/NVlabs/NVBit/releases/tag/v1.8.1)
confirms the separate API/CMake and opaque-library fixes. No Blackwell execution
claim is possible on this independently queried SM86 host.

811ac667 correctly removed the active capacity trace twin but left `atoi`,
last-trial-only sink checks and no elapsed validation; its elapsed-valid claim
was not implemented. Its XOR-canceling sink is only a liveness/output oracle,
not proof that every measured producer result is numerically correct.
7227459b changed only documents: the IMC driver still accepted malformed or
negative iteration strings, did not check the accumulator, and mislabeled
2,688 **thread** records as warps (there are 84 warps). It collected only one
NCU profile. Those v1 results are historical, not completed corrected stages.

Additional captures AQ–AZ made during this continuation are retained below as
bounded observations. They do not substitute for M1/M2. No timing/card
parameter has been fitted to these scored NCU measurements.

**M1 result.** Corrected capacity run: `docs/sm86_c7_research_probe_fixed_latency_result_capacity_live_v2.json`. The strict driver checks all seven trials and rejects malformed input; same b9 cubin replay/NCU match at 1,573,144 instructions, lower 590,830, NCU median 851,446. Capacity remains unidentified.

**M2 result.** Corrected IMC run: `docs/sm86_c7_research_probe_immediate_constant_frontier_live_v2.json`. The driver checks an independent modulo-u32 output oracle and timestamp bounds, reports 2,688 thread records/84 warps, resolves its cubin beside the executable, and runs seven correctness plus five NCU repetitions. Replay/NCU match at 151,872 instructions, lower 6,037, NCU median 10,315. This is observational only.

**M3 result.** The additional same-binary slices AQ–AZ are schema
validated and retained as bounded observations. The front-end IBuffer slice is
`docs/sm86_c7_research_probe_front_end_ibuffer_depth_live_v1.json` (W16
baseline_s1, 944 replay instructions, lower 818, NCU median 3,678). The
touched campaign analyzers now use the active long-generation populations;
historical short-trace names are no longer execution prerequisites. Inventory
latest-artifact links point to the corrected M1/M2 records and the new AQ–AZ
records. The subsequent M4-A through M4-E ledger below closed and independently
rechecked the remaining source families; no current directory is left without
a live vertical or an explicit non-native disposition.

## Outcome

A useful native GPU simulator must predict kernel performance, conserve the
trace's work, and validate on independent NCU measurements. Finishing a run,
printing a finite number, or covering a median with an arbitrarily wide
interval does not finish this program. Simulator runtime is recorded alone;
traditional simulators are not executed and no hardware/compression speedup
metric is introduced.

Three distinct questions must not be conflated:

1. **Model completion:** are modeled responses/acks/copy groups serviced and
   consumed exactly once, and are all unknown obligations retained?
2. **Hardware prediction:** how accurately does the model predict NCU cycles
   on unobserved workloads? A statistical prediction interval may be finite
   while the semantic completion upper remains unknown. Its probability,
   fitting population, width, and applicability must be explicit. It is not a
   worst-case silicon bound and must not clear semantic tail flags.
3. **Selective replay:** is a region transition equivalent to detailed native
   replay for future-relevant model state? This needs an independent detailed
   model differential, not a token exposing proprietary hardware cache state.
   Dispatch is not a prerequisite for accuracy of detailed replay.

## Corrections to the 2026-10-02 claims

- `close_explicit_hierarchy_tails` cleared all SM tail flags when one writeback
  timestamp existed or topology counts were zero. Neither implies complete
  ownership, ordering, initial state, or native completion. Remove this
  heuristic; keep the existing queue services and their measured model events.
- C1 async used a 64-group trace and a 16,384-group NCU executable. The apparent
  lower consistency is not a valid matched comparison. Profile the **same
  bounded executable** with the same launch arguments and fresh control data.
- C1 message passing used managed memory and spin-dependent execution. Its
  large timing dispersion cannot be attributed solely to coherence. Use
  resident device allocations and release/acquire semantics; detect divergent
  dynamic work across NVBit/NCU instead of declaring it equivalent.
- The C3 1,600-cycle padding was hand-chosen, not reproducibly fitted. Store
  pattern variants in one kernel are not family-held-out embedding workloads.
  Retain the numerical files as historical attempts but withdraw their finite
  upper/held-out validation claims. Do not enlarge the padding after a miss.
- A8 used old static controls with a freshly rebuilt binary. Rebuild/capture
  alone is insufficient; controls must be freshly extracted from that binary.
- The previous tail-witness report reads simulator aggregates and hardcodes
  unknown domains. It is not an independent extraction pass or evidence of a
  native witness. C0/C1/C2/C3 completion claims are reopened where applicable.

## Research basis

- NCU timestamps bracket grid execution with front-end wait-for-idle and
  semaphores. Software-patched counter passes do not supply workload duration.
  Stable replay/cache/clock settings and deterministic launch matching matter.
  [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#reproducibility)
- PTX cache modifiers are performance hints, not ordering operations. Global
  data coherence at L2 does not imply coherent private L1 caches or mandatory
  eviction of every dirty L2 line to DRAM on kernel exit. A generic directory
  experiment is not evidence of NVIDIA's actual coherence protocol.
  [PTX cache operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)
- Fences establish scoped ordering, not a public cycle-count upper bound.
  `cp.async.wait_group` closes the relevant copy groups, not unrelated stores.
  [CUDA fences](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#memory-fence-functions)
- Statistical intervals require separate model-fit and uncertainty-calibration
  samples. Marginal coverage under exchangeability is not a worst-case bound
  or a guarantee under workload-family shift. Repeats of one workload are not
  independent new workload-calibration examples.
  [Angelopoulos & Bates](https://arxiv.org/abs/2107.07511)

## Milestones — one active implementation phase

| Milestone | Concrete outcome and real completion bar | State |
|---|---|---|
| C0-R | Remove unsound flag/upper inference; rerun identical real store trace and preserve service events while upper stays open; correct prior claims | complete |
| C1-R | Matched resident-memory/async/fence probes: same binary/shape/launch, fresh control sidecar, full trace counts and NCU launch/count comparison; retain mismatches | active — C1-R-C refusal guard complete; async/visibility positive witness remains open |
| C2 | Independent **trace** extraction of stores, scopes, masks, allocations, async groups and synchronization. Separate observed facts, derived partial orders and unknowns; no invented global order | complete for observed-fact layer; native witness/effect relation open |
| C3 | Implement missing native completion obligations: per-request acknowledgements, scoped waits, copy completion, correct outstanding-work drain; demonstrate live memory/fence/copy cases plus perturbation differentials | active — negative ownership/generation branch and pre-mutation guard complete; physical upper remains unknown |
| C4 | Independently fit mechanism/observer models using new calibration-only probes, then materialize predictions on untouched cases. No ad-hoc padding or post-score model selection | partial — C4-H transfer falsifier and C4-I domain sync complete; broad transfer remains open |
| C5 | Live held-out evaluation of point error, signed bias, finite statistical interval coverage/width, repeat dispersion and OOD; keep semantic tails separate | partial — C5-A/B/C scoped slices complete; broad accuracy and family coverage remain open |
| C6 | Real scalar native region transition with full front-end/pipeline continuation differential and pre-mutation fallback; extend only after the first real positive case | active — C6-L effect-span and C6-M async-state guards complete; independent ownership/generation evidence remains open |
| C7 | Inventory all available workload families (portable, native probes, SOL/other rosters). Full live campaign over the declared supported population; failures remain in coverage denominator | complete scoped — all declared source families have current evidence or explicit blockers; C7-S34CS reconciles all 14 CUTLASS rows. Source/capture/replay blockers remain in the denominator; physical tails, broad accuracy, and native continuation carry to M5-N |
| C8 | Rerun corrected workload failures, remove obsolete interfaces, synchronize supported claims, results and runbook; no unfinished mechanism called complete | complete — scoped runbook/claim closeout; physical gaps retained |

C0-R/C1-R repair the previous work; they do not finish C3/C5/C6. C7 is not
limited to 15 workloads by fiat. Native/SOL populations and practical memory/
trace budgets must be explicitly inventoried before calling a campaign full.

## C1-R experiment specification

First recapture the bounded async binary on both NVBit and NCU. Use its own
static controls and select the target rather than prewarm. Then run a small
resident-memory suite with host-computed expected values, checked allocation/
launch/synchronization errors and fixed shapes:

- store-only and store+device-fence, reused and streaming addresses;
- load/store dependency, returning atomic and non-returning reduction;
- async-copy+wait+use with cache-all and bypass routes;
- deterministic single-CTA barrier and a valid inter-CTA release/acquire case.

Native, NVBit, and NCU must use the same binary/arguments. All simulator
outputs for the batch are saved before its first NCU capture. Check selected
mangled kernel, grid/block, shared memory, registers, and executed warp count;
when scheduling changes a polling loop, call it path-nondeterministic and do
not numerically score that trace against the other path. No hidden shape
shrinking, unit conversion, or profiler values in the simulator.

5 independent NCU processes per case; application replay, cache all, clock
none on the queried 1350/10501 MHz host. Explicit kernel selection, not an
unexamined global launch-skip. Kernel time is NCU time; trace/profiler process
wall time is overhead. Record actual process timeout/failure. A missing report
is not success even if the process returned zero. Native CUDA return status
alone is not correctness: check outputs and non-finite values too.

### C1-R live progress — matched negative slices

The rebuilt 64-group `async_barrier_overlap_trace` binary was used unchanged
for native, NVBit, simulator, and NCU. Fresh controls were extracted from
that binary; NCU selected launch 1 of the prewarm+target pair. Trace target
records and NCU `sm__inst_executed.sum` both equal 9,104. NCU GPC cycles are
`20937,20950,20937,20953,20946` (median 20,946), while simulator lower bound
is 84,866; lower-bound consistency is false. This is a real matched model
mismatch, not a coverage claim. See
`docs/sm86_completion_frontier_c1r_async_matched_v1.json`.

A new resident-device-memory message-passing binary removes managed-memory
migration from the experiment. Native, NVBit, simulator and five NCU runs use
the same binary and 16-iteration argument. NCU median is 4,321 cycles; the
simulator lower bound is 1,753 and remains open for writeback/coherence. See
`docs/sm86_completion_frontier_c1r_message_device_v1.json`. The model is
conservative on the lower bound for this case, but no finite upper or native
dispatch is claimed.

The current `memory_retirement_service` source then supplied a matched
resident-memory fence slice. Four `84 × 32`, depth-16, line-stream arms
(control/store × no/device fence) used one rebuilt binary for seven host
oracles, Compute Sanitizer, NVBit, deterministic replay, and five independent
NCU repetitions. The store-minus-control simulator GPC lower-bound deltas are
`2886` cycles for both fence scopes; NCU median deltas are `6080` (no fence)
and `6228` (device fence). The simulator's device-fence delta is only `12`
cycles, versus NCU deltas of `177` (control) and `325` (store), so the fence
effect is a retained model mismatch rather than a fitted constant. Store arms
retain dirty-writeback/coherence tails. Evidence:
`docs/sm86_c1r_store_fence_differential_live_v1.json`.

The two current-source system-fence arms were then captured in the same root.
Host oracles, Compute Sanitizer, NVBit, deterministic replay, and five NCU
repetitions passed, but the system-scope NCU instruction counts exceed replay
records by `756` for both arms (`93492` vs `92736` control; `132216` vs
`131460` store). No/device arms crosswalk exactly. System medians are `18067`
and `21662` cycles, but those rows are retained as not score-applicable rather
than treated as timing evidence. The scope reconciliation is
`docs/sm86_c1r_store_fence_scope_reconciliation_live_v1.json`.

### C1-R residual ledger

| C1-R phase | Completion bar | State |
|---|---|---|
| C1-R-A | Same-binary resident-memory control/store × no/device-fence slice with checked outputs, fresh controls, NVBit, replay, and five NCU repetitions | complete — fence mismatch retained |
| C1-R-B | Reconcile async/visibility and remaining fence effects without timing fitting; preserve open completion tails | complete — system-fence crosswalk blocker reproduced |
| C1-R-C | Add a focused typed-boundary regression proving a live async-copy state refuses scalar macro derivation before mutation | complete — structurally valid async state remains exact fallback; visibility tail open |

C1-R remains active for fence/async model repair and matched mechanism
validation; neither slice supplies a finite completion upper. A same-binary
`c00` control differential makes the async mismatch actionable: NCU median
delta `13760` cycles (`c11 - c00`) versus simulator lower-bound delta `78852`
cycles. This points to the async-group scheduling/completion model, not a
generic launch envelope. The differential is retained in
`docs/sm86_completion_frontier_c1r_async_control_differential_v1.json`.

The first C3 repair is conservative: when an SM86 trace contains async-copy
groups, `TimingReport::latent_frontiers` now widens issue/modeled/typed-tail
frontiers and refuses a prediction lower until the async completion model is
independently closed. The repaired matched async output has null predicted
lower/upper and is retained as
`docs/sm86_completion_frontier_c3_async_refusal_v1.json`. This fixes the
previous 84866-cycle false lower; it is not an accuracy result.

A matched one-warp `c00`/`c11` live differential separates this mechanism from
multi-warp destination arbitration. For the same `async_barrier_overlap_trace`
binary and `l1 0 1 1` launch, five fresh NCU medians are 7,194 cycles (c00)
and 14,668 cycles (c11), a 7,474-cycle increment; the repaired simulator is
5,995 and 17,394, an 11,399-cycle increment. Unprofiled native `clock64`
repeats give a 13,386-cycle increment. Thus the simulator is below the direct
clock differential but above the NCU GPC endpoint; neither endpoint alone
identifies the missing async mechanism. The result is retained in
`docs/sm86_completion_frontier_c3_async_endpoint_differential_warp1_v1.json`.

An independent cache-all source-only layout pair now tests the destination
projection without changing the binary, depth, source route, or launch shape.
Compact and conflict variants preserve 256 LDGSTS instructions and source
cycles, while destination wavefronts change from 2,048 to 4,096. The repaired
simulator predicts a 4,096-cycle layout delta; five fresh NCU medians differ by
4,070 cycles (26-cycle, 0.64% delta error). Absolute endpoints remain 4.91%
and 3.87% high and hardware completion tokens remain unknown. This is narrow
model-local differential evidence, not a global predictor or semantic hardware
upper. See `docs/sm86_completion_frontier_c3_async_destination_layout_pair_v1.json`.

C4/C5 now also has a declared layout holdout. Compact/conflict NCU repeats
were used as calibration for an OLS observer over trace-derived destination
wavefronts; the severe target prediction `24,476.2..24,902.2` (point
`24,689.2`) was materialized before any severe NCU capture. Five fresh severe
NCU values are `24,822,24,855,24,875,24,891,24,840` (median `24,855`), so the
interval covers the median and point APE is 0.667%. The prediction and joined
holdout are separated in
`docs/sm86_completion_frontier_c4_destination_layout_prediction_pre_ncu_v1.json`
and `docs/sm86_completion_frontier_c4_destination_layout_holdout_v1.json`.
This remains one controlled layout family; it is not a semantic completion
upper or a general hardware-accuracy result.

The first C7 population slice is complete for the declared 15 portable-LCD
workloads. The live campaign ran native timing, NVBit tracing, two simulator
replays, and five fresh NCU repetitions for every row in 389 seconds; all
15/15 completed with zero OOM/OOD/timeout/failed-stage outcomes. The raw
campaign is under `results/alpha/c7_full_live_20261003_run/` and its tracked
summary is `docs/sm86_c7_full_live_campaign_v1.json`. It is not yet the full
repository campaign: native probes, fixed-latency rosters, SOL/other suites,
and correctness-mode joins still need an explicit population inventory. All
portable replay rows retain open dirty-writeback/coherence tails and the
campaign did not request native macro dispatch.

The population inventory is now explicit in
`docs/sm86_c7_workload_population_inventory_v1.json`: 15 active portable-LCD
rows, 92 SimBench-BW rows (31 with an SM86 arch gate), 34 research-probe
directories, and 203 SOL baseline problem directories. The portable 15 and
six direct SimBench rows have completed bounded slices; the other populations
remain pending and must receive memory/shape/OOM dispositions before any
repository-wide claim.

The portable slice has now been executed live, not merely inventoried:
`docs/sm86_c7_full_live_campaign_v1.json` records 15/15 successful rows,
75 NCU captures, 15 NVBit traces, and 30 deterministic simulator replays with
zero OOM/OOD/timeout/failed-stage outcomes. It intentionally does not claim
correctness (this performance campaign uses timing shapes) or finite accuracy;
all replay tails remain explicit. C7 remains partial until the other inventoried
populations receive bounded campaign dispositions.

### C7-S1 — direct SimBench async-copy slice — complete

The second direct SM86 SimBench slice uses the rebuilt
`cp_async_shared_tile_copy` binary and the exact same arguments for host
correctness, NVBit, replay, and five fresh NCU application captures. Host
validation passed with zero mismatches after explicit launch/synchronize error
checks. The trace contains one complete block, 644 replay records, 32
`LDGSTS` copy instructions, and four committed/waited async groups. NCU
`gpc__cycles_elapsed.max` is `4247,4233,4232,4264,4229` (median `4233`);
the simulator completion lower bound is `2190`, so lower-bound consistency is
true but no point estimator or finite upper is claimed. Dirty-writeback and
coherence tails remain open, and the independent extractor observes no
hardware completion token, owner/read-from relation, or global instrumentation
order. The fast analyzer refuses all four warps for nonfull predicate
populations; native macro dispatch remains zero. The complete matched slice is
recorded in `docs/sm86_c7_simbench_cp_async_direct_live_v1.json` with its
schema. This closes only this slice; it does not close C3/C5/C6 or the C7
population campaign.

### C7-S2 — execution specification

Run one additional direct SM86 SimBench row from a different mechanism family
(prefer `global_atomic_histogram` or `global_atomic_and_return_used`) using
the same bounded pipeline and independent five-repeat NCU contract. Preserve
the current refusal rules: correctness must be checked by the workload, NCU
must use the exact traced binary/arguments, and open semantic tails or refused
dispatch remain explicit. If a candidate is blocked by OOM, unsupported SASS,
or a stale launcher, record a bounded disposition and move to the next row
without substituting a different shape silently.

### C7-S2 — direct SimBench global reduction slice — complete

`global_atomic_histogram` now has a direct SM86 timing-shape slice using one
rebuilt binary and identical `samples=1024,bins=64,seed=42` arguments for
correctness, NVBit, replay, and five fresh NCU application captures. The host
reference checks both the identity output and the complete histogram (zero
mismatches; sum `66,560`). SASS contains `RED.E.ADD.STRONG.GPU`, and the
independent extractor observes 2,080 reduction records plus one block barrier.
Replay has full one-block coverage and 42,843 records; NCU reports the same
42,843 warp instructions. NCU `gpc__cycles_elapsed.max` is
`516202,516413,516605,515743,516297` (median `516297`), while the simulator
completion lower bound is `394976`. Lower-bound consistency is true, but the
121,321-cycle gap is not a point error: reduction read-modify-write,
dirty-writeback, and coherence tails remain open, and no hardware completion
token, owner/read-from relation, or global instrumentation order is observed.
The fast analyzer refuses the one warp for a nonfull predicate population;
native macro dispatch remains zero. Evidence and schema:
`docs/sm86_c7_simbench_global_atomic_histogram_direct_live_v1.json` and
`docs/sm86_c7_simbench_global_atomic_histogram_direct_live_v1.schema.json`.
This closes only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S3 — execution specification

Run one additional direct SM86 SimBench row from a distinct scalar-memory or
tensor family (prefer `global_vectorized_128b_copy` or
`bf16_hmma_gemm_traceable`) with the same correctness, fresh NVBit, replay, and
five-repetition NCU contract. Preserve exact binary/argument identity and
record any OOM/OOD/unsupported-SASS disposition without silently shrinking
the shape.

### C7-S3 — direct SimBench vectorized-memory slice — complete

`global_vectorized_128b_copy` now has a direct SM86 timing-shape slice using
one rebuilt binary and identical `samples=1024,bins=64,seed=42` arguments for
correctness, NVBit, replay, and five fresh NCU application captures. The host
identity-output check passed with zero mismatches; the SASS contains
`LDG.E.128`/`STG.E.128`. Replay has one-block coverage and 2,977 records, and
NCU reports the same 2,977 warp instructions. NCU
`gpc__cycles_elapsed.max` is `74859,74473,74887,75352,75109` (median
`74887`); the simulator completion lower bound is `63246`, so lower-bound
consistency is true but no point estimator or finite upper is claimed. The
independent extractor observes 512 vector loads and 512 vector stores but no
hardware completion token, owner/read-from relation, or global instrumentation
order. Dirty-writeback and coherence tails remain open. The fast analyzer
refuses the one warp for a nonfull predicate population; native macro dispatch
remains zero. Evidence and schema:
`docs/sm86_c7_simbench_global_vectorized_128b_copy_direct_live_v1.json` and
`docs/sm86_c7_simbench_global_vectorized_128b_copy_direct_live_v1.schema.json`.
This closes only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S4 — execution specification

Run one additional direct SM86 SimBench row from a tensor or scalar-compute
family (prefer `bf16_hmma_gemm_traceable` or `fp16_hmma_gemm_traceable`) with
the same checked correctness, fresh NVBit, replay, and five-repetition NCU
contract. Preserve exact binary/argument identity and record any
OOM/OOD/unsupported-SASS disposition without silently shrinking the shape.

### C7-S4 — direct SimBench BF16 HMMA slice — complete

`bf16_hmma_gemm_traceable` now has a direct SM86 timing-shape slice using one
rebuilt binary and identical `m=32,n=16,k=64,seed=42` arguments for host
correctness, NVBit, replay, and five fresh NCU application captures. The host
BF16-cast FP32 reference check passed with zero mismatches and maximum absolute
error `3.58e-7`; SASS contains `HMMA.16816.F32.BF16`. Replay has full four-block
coverage and 404 records, and NCU reports the same 404 warp instructions plus
16 tensor-pipeline instructions. NCU `gpc__cycles_elapsed.max` is
`4896,5073,4923,4926,4885` (median `4923`); the simulator completion lower
bound is `3337`, so lower-bound consistency is true but no point estimator or
finite upper is claimed. The independent extractor observes 16 HMMA records
and 64 cross-SM line-topology events, but no hardware completion token,
owner/read-from relation, or global instrumentation order. Dirty-writeback and
coherence tails remain open. The fast analyzer refuses all four warps for
nonfull predicate populations; native macro dispatch remains zero. Evidence
and schema: `docs/sm86_c7_simbench_bf16_hmma_gemm_direct_live_v1.json` and
`docs/sm86_c7_simbench_bf16_hmma_gemm_direct_live_v1.schema.json`. This closes
only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S5 — execution specification

Run one additional direct SM86 SimBench row from a different memory or tensor
family (prefer `global_scalar_32b_copy` or `int8_imma_gemm_traceable`) with the
same checked correctness, fresh NVBit, replay, and five-repetition NCU
contract. Preserve exact binary/argument identity and record any OOM/OOD or
unsupported-SASS disposition without silently shrinking the shape.

### C7-S5 — direct SimBench scalar-width slice — complete

`global_scalar_32b_copy` now has a direct SM86 timing-shape slice using one
rebuilt binary and identical `samples=1024,bins=64,seed=42` arguments for host
correctness, NVBit, replay, and five fresh NCU application captures. The host
identity-output check passed with zero mismatches; SASS contains scalar
`LDG.E`/`STG.E` and no `.128` form. Replay has one-block coverage and 11,809
records, and NCU reports the same 11,809 warp instructions. NCU
`gpc__cycles_elapsed.max` is `249180,249114,248894,249782,249151` (median
`249151`); the simulator completion lower bound is `245262`, so lower-bound
consistency is true with a 3,889-cycle residual gap, but no point estimator or
finite upper is claimed. The independent extractor observes 2,048 scalar
loads and 2,048 scalar stores but no hardware completion token, owner/read-from
relation, or global instrumentation order. Dirty-writeback and coherence tails
remain open. The fast analyzer refuses the one warp for a nonfull predicate
population; native macro dispatch remains zero. Evidence and schema:
`docs/sm86_c7_simbench_global_scalar_32b_copy_direct_live_v1.json` and
`docs/sm86_c7_simbench_global_scalar_32b_copy_direct_live_v1.schema.json`.
This closes only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S6 — execution specification

Repair `global_atomic_return_used` before capturing it: its per-lane ordinary
stores currently overwrite shared histogram bins, and its host only prints
an unrelated identity copy. Store returned-value sums in disjoint per-lane
64-bit outputs; independently check the histogram and the order-independent
sum of returned old values. Keep the registered `trace_00` shape unchanged,
verify SASS still emits ATOM/ATOMG, then run native, NVBit, replay and five NCU
processes. Check non-power-of-two/tail shapes and invalid inputs. Do not tune
the simulator from these scored values. Related MAX/AND returning variants
have the same sink defect and are next after this first real repaired slice.
Atomicity is not synchronization for the old ordinary histogram stores:
[NVIDIA legacy atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#legacy-atomic-functions).

### C7-S6 — repaired returning-atomic live slice — complete

`global_atomic_return_used` was repaired before capture. The old host driver
only checked an identity copy while the kernel overwrote histogram bins with
ordinary per-lane stores; that was not a valid atomic-return workload. The
repaired kernel writes one disjoint 64-bit returned-old-value sum per lane and
the host checks the identity output, every histogram bin, and the
order-independent sum `count*(count-1)/2`. The trace shape (`samples=1024,
bins=64,seed=42`) and three additional shapes (`32x8`, `37x13`, `1x1`) pass.
Fresh SASS contains `ATOMG.E.ADD.STRONG.GPU`. Replay has one-block coverage
and 47,139 records; NCU reports the same 47,139 warp instructions. NCU
`gpc__cycles_elapsed.max` is `1059805,1059871,1060145,1060504,1061112`
(median `1060145`); the simulator completion lower bound is `917785`, so
lower-bound consistency is true but no point estimator or finite upper is
claimed. The independent extractor observes 2,080 ATOMG records, 2,049
stores, and one block barrier, but no hardware completion token, owner/read-from
relation, or global instrumentation order. Dirty-writeback and coherence tails
remain open. The fast analyzer refuses the one warp for a nonfull predicate
population; native macro dispatch remains zero. Evidence and schema:
`docs/sm86_c7_simbench_global_atomic_return_used_direct_live_v1.json` and
`docs/sm86_c7_simbench_global_atomic_return_used_direct_live_v1.schema.json`.
This closes only this repaired slice; C3/C5/C6 and the wider C7 campaign
remain open.

### C7-S7 — execution specification

Run one additional direct SM86 SimBench row from a different atomic or tensor
family (prefer `global_atomic_and_return_used` or
`int8_imma_gemm_traceable`). Apply the same rule: repair correctness and
resource safety before profiling, preserve exact binary/arguments, and retain
any OOM/OOD/unsupported-SASS disposition without silent shape substitution.

### C7-S7 — INT8 IMMA capability repair — complete scoped

The exact rebuilt int8_imma_gemm_traceable binary passes the host quantized
reference check, emits a fresh NVBit trace with 432 records and eight IMMA
instructions, and retains five same-binary NCU observations:
4526,4624,4545,4542,4567 cycles (median 4545). M5-O adds a corrected
no-memory IMMA.16832 probe with a numerical oracle and zero-error memcheck,
then admits only the exact SM86 opcode at a distinct 16-cycle initiation
interval. The retained 432-record trace now replays deterministically; dirty
writeback/coherence tails remain open and no native scheduling or physical
completion claim is made. The original blocked artifact remains historical;
current evidence is `docs/sm86_m5o_imma16832_pipeline_replay_v1.json`.

### C7-S8 — next active milestone

Run fp16_hmma_gemm_traceable as the next tensor-family complete slice. Reuse
the checked BF16-HMMA host safety/reference pattern, preserve exact binary and
trace arguments, and record any unsupported-SASS disposition separately from
the successful BF16 row. Do not infer INT8 accuracy from the BF16/FP32 rows.

### C7-S8 — direct SimBench FP16 HMMA slice — complete

The fp16_hmma_gemm_traceable row now has a direct SM86 timing-shape slice
using one rebuilt binary and identical m=32,n=16,k=64,seed=42 arguments for
host correctness, NVBit, replay, and five fresh NCU application captures. The
host FP16-cast FP32 reference check passed with zero mismatches and maximum
absolute error 9.54e-7; SASS contains HMMA.16816.F32. Replay has full
four-block coverage and 404 records, and NCU reports the same 404 warp
instructions plus 16 HMMA tensor-pipeline instructions. NCU
gpc__cycles_elapsed.max is 4934,4910,4990,4900,5096 (median 4934); the
simulator completion lower bound is 3337, so lower-bound consistency is true
but no point estimator or finite upper is claimed. The independent extractor
observes 16 HMMA records and 64 cross-SM line-topology events, but no hardware
completion token, owner/read-from relation, or global instrumentation order.
Dirty-writeback and coherence tails remain open. The fast analyzer refuses all
four warps for nonfull predicate populations; native macro dispatch remains
zero. Evidence and schema:
docs/sm86_c7_simbench_fp16_hmma_gemm_direct_live_v1.json and its schema. This
closes only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S9 — next active milestone

Run one additional direct SM86 SimBench row from the remaining memory, control,
or tensor populations (prefer global_vectorized_64b_copy,
shared_bank_conflict_stride, or tf32_hmma_gemm_traceable). Preserve the same
checked correctness, exact binary/arguments, fresh NVBit, replay, and
five-repetition NCU contract; retain any unsupported-SASS or OOM/OOD
disposition without silent shape substitution.

### C7-S9 — direct SimBench 64-bit vector slice — complete

The global_vectorized_64b_copy row now has a direct SM86 timing-shape slice
using one rebuilt binary and identical samples=1024,bins=64,seed=42 arguments
for host correctness, NVBit, replay, and five fresh NCU application captures.
The host identity-output check passed with zero mismatches; SASS contains
LDG.E.64/STG.E.64. Replay has one-block coverage and 5,921 records, and NCU
reports the same 5,921 warp instructions. NCU gpc cycles were
130097,129581,130029,129449,129721 (median 129721); simulator completion
lower bound is 123918, so lower-bound consistency is true but no point
estimator or finite upper is claimed. The independent extractor observes
1,024 vector loads and 1,024 vector stores, but no hardware completion token,
owner/read-from relation, or global instrumentation order. Dirty-writeback and
coherence tails remain open. The fast analyzer refuses the one warp for a
nonfull predicate population; native macro dispatch remains zero. Evidence:
docs/sm86_c7_simbench_global_vectorized_64b_copy_direct_live_v1.json and its
schema. This closes only this slice; C3/C5/C6 and the wider C7 campaign
remain open.

### C7-S10 — next active milestone

Run one additional direct SM86 SimBench row from the remaining memory, control,
or tensor populations (prefer shared_bank_conflict_stride or
tf32_hmma_gemm_traceable). Preserve the same checked correctness, exact
binary/arguments, fresh NVBit, replay, and five-repetition NCU contract;
retain any unsupported-SASS or OOM/OOD disposition without silent shape
substitution.

### C7-S10 — direct SimBench shared-bank slice — superseded

The original artifact `docs/sm86_c7_simbench_shared_bank_conflict_stride_direct_live_v1.json`
is superseded. Its default-limit refusal was not a valid silicon OOD result:
SM86 permits dynamic shared memory above 48 KiB with an explicit opt-in.
Retain it only as historical execution evidence.

### C7-S10r — corrected direct SimBench shared-bank slice — complete

The same `trace_00` shape (64x64x4, 65,536 dynamic shared bytes) now opts into
the device's 101,376-byte SM86 limit. Host identity/sink checks and memcheck
pass. Fresh NVBit/replay/five-NCU use the exact opted-in binary; replay has
full one-block coverage and 15,350 records, and NCU reports the same 15,350
warp instructions. NCU gpc cycles are 100467,100317,100939,99980,100758
(median 100467); simulator lower is 103225, so lower-bound consistency is
false by 2,758 cycles (2.75%). Shared LDS/STS traffic, open writeback/
coherence tails, and zero native dispatch remain explicit. Evidence and schema:
`docs/sm86_c7_simbench_shared_bank_conflict_stride_corrected_live_v1.json`.
This is the active corrected slice; C3/C5/C6 remain open.

### C7-S11 — execution specification

Run one additional direct SM86 SimBench row from the remaining tensor/control
populations (prefer tf32_hmma_gemm_traceable or l2_thrash_pointer_chase).
Preserve checked correctness, exact binary/arguments, fresh NVBit, replay, and
five-repetition NCU; retain every unsupported-SASS or OOM/OOD disposition.

### C7-S11 — TF32 HMMA blocked disposition — complete

The exact rebuilt tf32_hmma_gemm_traceable binary passes host TF32-rounded
reference correctness, produces a fresh NVBit trace with 588 records and 32
HMMA instructions, and completes five same-binary NCU captures:
6621,6328,6373,6360,6417 cycles (median 6373). The independent extractor
observes 112 cross-SM line-topology events but no hardware completion token or
global instrumentation order. Replay is explicitly blocked before any timing
frontier because sim-run reports that HMMA.1688.F32.TF32 has no SM86
execution-pipeline model. No simulator endpoint, fast-frontier result, native
dispatch, or accuracy comparison is claimed. Evidence:
docs/sm86_c7_simbench_tf32_hmma_gemm_blocked_live_v1.json and its schema.
This is an explicit unsupported-SASS disposition, not a silently substituted
shape or a passing pipeline row.

### C7-S12 — next active milestone

Run one additional direct SM86 SimBench row from the remaining control or
memory populations (prefer l2_thrash_pointer_chase or
global_atomic_and_return_used). Preserve checked correctness, exact
binary/arguments, fresh NVBit, replay, and five-repetition NCU; retain every
unsupported-SASS or OOM/OOD disposition.

### C7-S12 — direct SimBench pointer-chase slice — complete

The repaired l2_thrash_pointer_chase row passes host identity-output and
dependent-sink checks on its declared trace_00 shape
(chain_length=256,warps=32); unsigned bit hashing prevents the prior possible
negative-index OOB. The exact binary produces a fresh NVBit trace, full 32-block
replay coverage, and five NCU captures. Replay has 118,976 records and NCU
reports the same 118,976 warp instructions. NCU gpc cycles were
49573,49580,49739,49630,49523 (median 49580), while simulator completion lower
bound is 59545: lower-bound consistency is false by 9,965 cycles (20.1% above
the hardware median). This is a real falsifying mismatch, not a point-error
score or a reason to tune the model post hoc. The extractor reports 256
cross-SM line-topology events and 31 unknown address operands, with no hardware
completion token or global instrumentation order. Dirty-writeback and
coherence tails remain open; native dispatch remains zero. Evidence:
docs/sm86_c7_simbench_l2_thrash_pointer_chase_direct_live_v1.json and its
schema. This closes only this slice; C3/C5/C6 and the wider C7 campaign remain
open.

### C7-S13 — execution specification

Run one additional direct SM86 SimBench row from the remaining memory/control
population (prefer l2_persisting_window_sweep or warm_cold_pair_relaunch).
Preserve checked correctness, exact binary/arguments, fresh NVBit, replay, and
five-repetition NCU; retain every unsupported-SASS, OOM/OOD, or falsifying
disposition without parameter tuning.

The active SimBench wrapper still rejects the repository-contained NVBit
tracer because it requires an external split-v2 tracer root; see
`docs/sm86_c7_simbench_population_slice_v1.json`. The blocker is recorded,
not bypassed with a hidden tracer or substituted shape. It does not invalidate
the eight direct bounded slices, which use the repository tracer explicitly and
retain their own exact binary/trace/control identities.

The wrapper blocker did not prevent a direct bounded timing-shape check: the
same `fp32_fma_dep_chain` binary/arguments completed native execution, NVBit,
replay, and five same-binary NCU captures. The analyzer refused all candidates
for nonfull predicate populations, so native dispatch stayed zero; simulator
8152 versus NCU median 8488 is a 3.96% mismatch. See
`docs/sm86_c7_simbench_fp32_direct_live_v1.json`. This is one probe row, not a
repository-wide claim.

Historical C3 store/service artifacts were also normalized: their schemas now
use explicit `withdrawn_historical_attempt`/`withdrawn_do_not_use` statuses
instead of encoding finite completion as current state. Raw numerical files
remain available for review, but they cannot be mistaken for active accuracy
evidence.

C6 now has a real same-shape scalar negative slice. The fixed-latency
`w4_d4_split` trace uses the exact 64-generation trace cubin for correctness,
NVBit, replay, and five NCU captures. Correctness passes; the analyzer refuses
all candidates because active/predicate populations are nonfull, so native
macro dispatch remains zero. The simulator endpoint is 3,578 cycles versus
NCU median 6,812 (47.5% low). The old 8,192-generation NCU rows are excluded
from this comparison. Artifact:
`docs/sm86_completion_frontier_c6_fixed_latency_live_v1.json`.

The phase-assay probe itself was corrected before another capture: its
`correct` field now compares the returned lane-0 token against a host
recomputation from the deterministic source initialization, in addition to
checking interval-sum arithmetic. Three direct SM86 runs (cache-all compact,
cache-all conflict, and streaming severe) pass both checks. Existing phase
captures remain historical because the executable changed; no old capture is
retroactively relabeled.

The corrected phase binary then completed one fresh NVBit + replay + five-NCU
path. Its direct run independently reports the expected token and interval
sum; the simulator endpoint is 18,305 cycles versus NCU median 16,658 (9.89%
APE), with the result-store writeback/coherence tails open. Artifact:
`docs/sm86_completion_frontier_c3_async_phase_corrected_live_v1.json`.

The resident-device message trace also has a same-input service differential:
the explicit model-local writeback/coherence queues complete 3 sectors and 2
coherence transactions, but the terminal tail flags remain true and the
semantic upper remains open. See
`docs/sm86_completion_frontier_c3_message_service_differential_v1.json`.

The next source-side repair corrected a concrete granularity error in the
model-local LDGSTS return ledger. The old code charged the eight-cycle
architecture-local token once per returned 32-byte sector; the raw trace
represents each warp-level LDGSTS as one coalesced source instruction with a
sector span. The new code charges one token per coalesced instruction and
shares its service timestamp across that instruction's sectors, with a
transactional overflow test. On the same previously captured live sweep the
post-repair lower bounds are 7,442/13,090/24,386 cycles for groups
16/32/64 and 46,978 for the untouched group-128 holdout; NCU medians are
7,904/12,254/20,949/38,348. The controlled mismatch is reduced but not
closed, and all dirty-writeback/coherence tails remain open. Artifact:
`docs/sm86_completion_frontier_c3_async_source_token_repair_v1.json`.

### C2 — observed tail facts — fact layer complete

`tools/extract_sm86_tail_facts.py` is a separate raw-protobuf plus static-SASS
extraction pass. Fresh resident-device message and matched async traces now
report store/atomic/fence/async-opcode facts, masks, allocation IDs, payload
coverage and cross-SM topology while explicitly leaving line generation,
owner/read-from, fence semantics and hardware completion unknown. Reports:
`docs/sm86_c2_message_device_tail_facts_v1.json` and
`docs/sm86_c2_async_tail_facts_v1.json`. The extractor now uses the tracer's
static function-id/PC map when available, reports deterministic same-warp
program-order edges and protocol counts, and explicitly records that no
global instrumentation order was present in either capture. It now also
reconstructs a launch-scoped logical async prefix relation (copy, commit,
wait, empty-group, and terminal-unwaited counts) from raw instructions and
decoded masks, without using SmCore aggregates. This closes the observed-fact
and protocol-partial-order deliverable. It is still not an independent native
witness: owner/read-from/generation/fence-scope/hardware completion and a
native effect/continuation relation remain open for C3/C6.

### C4/C5 exploratory observer calibration — positive statistical slice

Using only the five NCU repeats from compile-time group counts 16 and 32, an
ordinary least-squares observer model was materialized **before** profiling a
new group-128 binary. Its predicted interval was `37737..38655` cycles. The
five group-128 NCU values were `38353,38339,38348,38338,38395`: raw coverage
1.0 and point error 0.396%. This is recorded in
`docs/sm86_async_observer_calibration_holdout_v1.json`.

This is the first finite empirical NCU interval in the corrected program, but
it is not a semantic completion upper: the simulator group-128 lower bound is
168898 and dirty-writeback/coherence remain open. The calibration covers one
controlled async-group dimension, not workload-family generalization. C4/C5
remain active for independent families, OOD behavior, repeat dispersion, and
mechanism counters.

## C2/C3 technical seams

`sim/sm/src/issue.rs`: response ownership, `memory_write_ack`, GPU MEMBAR,
async groups; `sim/sm/src/system.rs`: due events, ack delivery, drain and
modeled writeback/coherence queues; `sim/sm/src/replay.rs`: coverage and tail
propagation; `sim/direct_run/src/lib.rs`: separate endpoints and projections.

Use these existing seams instead of adding reporting frameworks. A missing
hardware ownership token does not prevent an approximate model, but it prevents
calling the approximation an exact physical history. Every modeled request
needs a stable model identity, one completion, and the right consumer/scope.
Never clear all tails based on totals, topology, or presence of a timestamp.
Do not force a terminal L2 flush to make completion true; evaluate whether it
belongs in the chosen endpoint and keep its cost separate otherwise.

## C4/C5 accuracy experiment (declare before scoring)

- Fit point model and mechanism parameters only on independent calibration
  probes; keep the supplied example profile unvalidated until this is done.
- Use a third, disjoint workload-level set for interval calibration; candidate
  model selection is completed before held-out targets are collected.
- Store point error for an actual point estimator, not a lower endpoint renamed
  as a point. Report semantic lower consistency separately.
- Record medians, IQR, all repeats, signed error, per-family median/P90 APE,
  interval coverage and width. Five repeats cannot establish 95% tail coverage.
- Initial research target: median APE <=15%, P90 <=30% and 90% empirical
  interval coverage, median width/observed <=50% on a declared in-domain set.
  These are targets, not achieved results or global guarantees. Family/OOD
  transfer is reported separately and must not silently widen the domain.
- A miss stays a miss; once inspected, that row is development evidence for
  subsequent changes, never a fresh holdout again.

## C6 selective replay

First use a real native scalar region without memory effects. The independent
reference is an ordinary detailed native replay; compare front-end cursor,
pipeline releases, queue identities/order, effects and suffix results. Preserve
external events rather than discarding global state. Do not require a full
silicon cache proof for a memory-free model transition. Do not call synthetic
fixtures or diagnostic counters completion of this milestone.

## Current record

2026-10-03: C0-R is complete. The code no longer clears tail flags from coarse
queue counters/topology. The identical real store-2 trace drains 36 modeled
sectors but retains dirty-writeback/coherence uncertainty and an open observer
upper. Prior finite-upper claims are withdrawn. C1-R is now active: recapture
matched async/fence cases with fresh controls and exact launch selection.

2026-10-03 continuation: C2's extractor is now independent of the replay
validator and reports compressed-address reconstruction, launch-scoped
topology, raw dynamic order edges, static opcode joins, and logical async
copy/commit/wait prefix facts. The cacheable LDGSTS source-return ledger was
repaired to charge one service token per coalesced warp instruction rather than
per sector. Fresh source-only compact/conflict/severe captures show a
4,070-cycle NCU layout delta versus a 4,096-cycle model delta; the severe
holdout interval covers its five NCU repeats. A corrected phase probe checks
its data token independently. A real scalar fixed-latency trace remains
negative (zero candidates/native dispatch; 47.5% low on matched NCU), while
the full 15-row portable C7 live pipeline completes with 75 NCU captures,
15 NVBit traces, and 30 simulator replays. C3/C4/C5 remain partial: tails,
absolute endpoint bias, native selective dispatch, and non-portable populations
are still open.
### C7-S13 — direct SimBench persisting-window slice — complete

The repaired l2_persisting_window_sweep row passes exact identity-output checks
at allocation=1MiB, effective persistence window=524288 bytes, accesses=64,
warps=32, and completes fresh NVBit, replay, and five NCU captures. Replay has
full 32-block coverage and 159,200 records; NCU reports the same 159,200 warp
instructions. NCU gpc cycles were 30197,29757,29747,29616,29965 (median
29757), while simulator completion lower bound is 31173: lower-bound
consistency is false by 1,416 cycles (4.76% above the hardware median). This
is a real lower-bound falsifier, not a point-error score or a post-score tuning
target. The independent extractor reports 256 cross-SM line-topology events
and 31 unknown address operands, with no hardware completion token or global
instrumentation order. Dirty-writeback and coherence tails remain open; native
dispatch remains zero. Evidence:
docs/sm86_c7_simbench_l2_persisting_window_sweep_direct_live_v1.json and its
schema. This closes only this slice; C3/C5/C6 and the wider C7 campaign remain
open.

### C7-S14 — next active milestone

Run one additional direct SM86 SimBench row from the remaining control/memory
population (prefer warm_cold_pair_relaunch or l2_prefetch_hint_sweep).
Preserve checked correctness, exact binary/arguments, fresh NVBit, replay, and
five-repetition NCU; retain every unsupported-SASS, OOM/OOD, or falsifying
disposition without tuning scored parameters.

### C7-S14 — direct SimBench prefetch-hint slice — complete

The repaired l2_prefetch_hint_sweep row passes host identity-output checks at
accesses=256,prefetch_distance=4,warps=32, produces a fresh NVBit trace, and
completes five same-binary NCU captures. The compiler emits no explicit
prefetch mnemonic in static SASS, so the hint remains advisory/unobserved; the
trace still captures the workload path. Replay has full 32-block coverage and
267,712 records, and NCU reports the same 267,712 warp instructions. NCU gpc
cycles were 40653,40590,40577,40598,40726 (median 40598), while simulator
completion lower bound is 46874: lower-bound consistency is false by 6,276
cycles (15.46% above the hardware median). This is a real lower-bound
falsifier, not a point-error score or a tuning target. The independent extractor
reports 39 cross-SM line-topology events and 31 unknown address operands, with
no hardware completion token or global instrumentation order. Dirty-writeback
and coherence tails remain open; native dispatch remains zero. Evidence:
docs/sm86_c7_simbench_l2_prefetch_hint_sweep_direct_live_v1.json and its schema.
This closes only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S15 — warm/cold multi-launch slice — complete

The corrected warm_cold_pair_relaunch binary uses separate first/repeat output
buffers and sinks, so the repeat pass cannot hide a first-pass error. The
uninterrupted sequence is cold pass, optional scratch eviction, and repeat pass;
host validation checks both outputs, both sinks with a floating-point error
bound, and every eviction element. Compute Sanitizer memcheck reports zero
errors. Fresh NVBit tracing records three launches with full coverage: 185,440
records for each warm/cold kernel and 44,224 for eviction. Detailed simulator
replay completes all three launches. Launch-scope lower bounds are 69,383 /
116,502 / 69,383 cycles; stream-L2 replay gives 69,383 / 116,502 / 52,557,
which demonstrates why cache scope must remain explicit. The fast frontier
refuses the multi-launch trace by its single-launch contract; this is not a
detailed replay failure.

Five application-replay NCU processes use `cache-control=none`, preserving the
application-managed cache sequence rather than flushing each replay. Instruction
counts match each launch. Cycle medians are 63,183 (first), 66,548 (eviction),
and 46,876 (repeat), with large repeat/eviction dispersion retained rather
than converted into a point score. This is cache-state observation and
multi-launch plumbing evidence, not a finite completion upper, point-accuracy,
or speedup claim. Evidence and schema:
`docs/sm86_c7_simbench_warm_cold_pair_relaunch_direct_live_v1.json` and its
schema.

### C7-S16a — persisting-window sink repair — complete

The corrected sink now includes all four C7-S13 Stage-1 anchor revisits; exact
sink/output checks pass on a fresh rebuild. C7-S10 dynamic shared-memory
opt-in and original-shape recapture are also complete. Do not change simulator
parameters from these scored observations.
The corrected live recapture is recorded in
`docs/sm86_c7_simbench_l2_persisting_window_sweep_corrected_live_v1.json`.

### C7-S16b — prefetch evidence correction — complete

The exact C7-S14 binary's static SASS contains five `CCTL.E.PF2` instructions.
The corrected artifact now reports that instruction as observed while keeping
cache effect, request completion, and hardware timing unobservable. The same
trace/replay/NCU measurements remain a lower-bound falsifier; no scored target
was used to alter the model. See
`docs/sm86_c7_simbench_l2_prefetch_hint_sweep_corrected_live_v1.json`.

### C7-S17 — corrected-artifact regression — complete

All 19 current live/blocked C7 artifacts validate against their sibling JSON
schemas; the inventory has 14 complete direct rows and two explicit replay
blockers. The Cargo workspace regression passes (38 test groups; no reported
failures). The corrected shared-bank and persisting-window artifacts are the
active records; their pre-opt-in/pre-sink artifacts remain superseded history.
No scored NCU target was used to change simulator parameters.

### C7-S18 — repaired atomic-and-return slice — complete

`global_atomic_and_return_used` was repaired before capture: ordinary stores
that raced with histogram bins were removed, disjoint return sinks were added,
and the host now checks the identity output plus deterministic final bitwise-AND
histogram. The exact trace shape passes; SASS contains
`ATOMG.E.AND.STRONG.GPU`. Replay has full one-block coverage and 46,882
records; NCU reports the same instruction count. NCU gpc cycles are
1060292,1060281,1060397,1061866,1060637 (median 1060397); simulator lower
916220 remains separated by open completion tails. Native dispatch is zero.
Evidence: `docs/sm86_c7_simbench_global_atomic_and_return_used_direct_live_v1.json`.

### C7-S19 — carveout sensitivity slice — complete

`carveout_sensitivity_probe` now validates its identity output after explicit
shared-memory/carveout attribute setup and launch-error checks. The exact
allocation=1MiB, shared=16KiB, carveout=25%, accesses=256, warps=32 binary
passes fresh NVBit, detailed replay, memcheck, and five NCU captures. Replay
has full 32-block coverage and 824,576 records; NCU reports the same
instruction count. NCU gpc cycles are 178537,177540,177927,178647,178521
(median 178521); simulator lower is 211002, a 32,481-cycle (18.2%) falsifier.
Open writeback/coherence tails and zero native dispatch remain explicit.
Evidence: `docs/sm86_c7_simbench_carveout_sensitivity_probe_direct_live_v1.json`.

### C7-S20 — vendor SGEMM blocked disposition — complete

The exact rebuilt cuBLAS FP32 binary passes host correctness and produces a
fresh NVBit trace of ampere_sgemm_128x32_nn. Five same-binary NCU captures
report 8,784 warp instructions and cycles 8904,8905,8894,8955,8898 (median
8904). Replay is explicitly blocked: the kernel is loaded from libcublasLt.so
with no stable static module/control sidecar, and the traced dynamic record
count (29,752) cannot be silently crosswalked to NCU's 8,784. No simulator
endpoint, native dispatch, or accuracy comparison is claimed. Evidence:
`docs/sm86_c7_simbench_vendor_sgemm_cublas_blocked_live_v1.json`.

### C7-S21 — vendor strided SGEMM blocked disposition — complete

The exact rebuilt strided-batched cuBLAS FP32 binary passes host batch-0
correctness and produces a fresh NVBit trace of ampere_sgemm_128x128_nn. Five
same-binary NCU captures report 102,176 warp instructions and cycles
15524,15492,15550,15511,15503 (median 15511). Replay is explicitly blocked:
the kernel is loaded from libcublasLt.so with no stable static module/control
sidecar, and the traced dynamic record count (367,168) cannot be silently
crosswalked to NCU. No simulator endpoint, native dispatch, or accuracy
comparison is claimed. Evidence:
`docs/sm86_c7_simbench_vendor_sgemm_strided_batched_cublas_blocked_live_v1.json`.

### C7-S22 — working-set capacity slice — complete

The repaired working_set_capacity_sweep binary passes anchor identity-output
checks at working_set=1MiB, accesses=256, warps=32, with full 32-block replay
coverage and Compute Sanitizer-safe execution. Replay has 789,888 records and
NCU reports the same 789,888 warp instructions. NCU gpc cycles are
129756,129914,129648,130077,129710 (median 129756), while simulator lower is
143555: lower-bound consistency is false by 13,799 cycles (10.63%). This is a
real memory-capacity falsifier; tails remain open and native dispatch is zero.
Evidence: `docs/sm86_c7_simbench_working_set_capacity_sweep_direct_live_v1.json`.

### C7-S23 — DRAM stream stride slice — complete

The repaired dram_stream_stride_mix binary passes host identity-output checks at
warps=32,accesses=128,stride_classes=2 and completes fresh NVBit, replay, and
five NCU captures. Replay has full 32-block coverage and 1,716,864 records;
NCU reports the same instruction count. NCU gpc cycles are
418778,418925,419105,419040,419638 (median 419040), while simulator lower is
550005: lower-bound consistency is false by 130,965 cycles (31.25%). This is a
real DRAM-stream falsifier, not a point-error score or tuning target. Tails
remain open and native dispatch is zero. Evidence:
`docs/sm86_c7_simbench_dram_stream_stride_mix_direct_live_v1.json`.

### C7-S24 — local dynamic-index slice — complete

The repaired local_dynamic_index_array binary passes host identity-output checks
at samples=1024,bins=64, exposes STL.128/LDL local traffic in SASS, and
completes fresh NVBit, replay, and five NCU captures. Replay has full one-block
coverage and 72,655 records; NCU reports the same 72,655 warp instructions.
NCU gpc cycles are 1243158,1244664,1244381,1243032,1243714 (median 1243714),
while simulator lower is 1443183: lower-bound consistency is false by 199,469
cycles (16.04%). This is a real local-memory falsifier; tails remain open and
native dispatch is zero. Evidence:
`docs/sm86_c7_simbench_local_dynamic_index_array_direct_live_v1.json`.

### C7-S25 — local volatile-array slice — complete

The repaired local_volatile_array binary passes host identity-output checks and
exposes STL local traffic in SASS. Replay has full one-block coverage and
29,017 records; NCU reports the same 29,017 warp instructions. NCU gpc cycles
are 421983,422645,423139,422474,422190 (median 422474), while simulator lower
is 524608: lower-bound consistency is false by 102,134 cycles (24.18%). This
is a real local-memory falsifier; tails remain open and native dispatch is zero.
Evidence: `docs/sm86_c7_simbench_local_volatile_array_direct_live_v1.json`.

### C7-S26 — register-spill local-memory slice — complete

The register_spill_local_mem build uses -maxrregcount=32 and reports STACK 424
bytes plus STL traffic. Host identity output passes with independent float-print
tolerance; replay has full 64-block coverage and 271,680 records, and NCU
reports the same instruction count. NCU gpc cycles are
23036,23073,23002,23206,23117 (median 23073), while simulator lower is 30010:
lower-bound consistency is false by 6,937 cycles (30.08%). This is a real
spill/local-memory falsifier; tails remain open and native dispatch is zero.
Evidence: `docs/sm86_c7_simbench_register_spill_local_mem_direct_live_v1.json`.

### C7-S27 — atomic max histogram slice — complete

The repaired global_atomic_max_histogram binary checks identity output and the
deterministic final atomicMax histogram; fresh SASS emits RED.MAX. Replay has
full one-block coverage and 42,874 records, and NCU reports the same
instructions. NCU gpc cycles are 568531,567884,568408,568964,568350 (median
568408), while simulator lower is 436096, so lower-bound consistency is true;
reduction RMW and dirty-writeback/coherence tails remain open and native dispatch
is zero. Evidence: docs/sm86_c7_simbench_global_atomic_max_histogram_direct_live_v1.json.

### C7-S28 — atomic max return slice — complete

The repaired global_atomic_max_return_used binary checks identity output and
final atomicMax histogram with disjoint finite return sinks; SASS contains
ATOMG.E.MAX. Replay has full one-block coverage and 46,882 records, and NCU
reports the same instruction count. NCU gpc cycles are
1060971,1060224,1060648,1060929,1060243 (median 1060648), while simulator
lower is 916220, so lower-bound consistency is true; tails remain open and
native dispatch is zero. Evidence: `docs/sm86_c7_simbench_global_atomic_max_return_used_direct_live_v1.json`.

### C7-S29 — full-shape DRAM streaming dot-product slice — complete

The hardened `dram_streaming_dot_product` timing-shape binary validates
`n=131072, passes=4, blocks=128, threads=256` with per-block floating-point
roundoff checks, launch/error checks, bounded allocation arithmetic, and zero
Compute Sanitizer errors. Fresh NVBit covers all 128 blocks; detailed replay
has 297,600 warp records and the five NCU rows report the same 297,600
`smsp__inst_executed.sum` instructions. NCU gpc cycles are
7714,7579,7548,7524,7562 (median 7562); simulator lower is 5345, so
lower-bound consistency is true. The completion upper remains open because
dirty-writeback and coherence are unobserved; fast-frontier candidates and
native macro dispatch remain zero. Evidence:
`docs/sm86_c7_simbench_dram_streaming_dot_product_direct_live_v1.json` and
its schema. This is not a point-accuracy or speedup claim.

### C7-S30 — full-shape RMSNorm BF16 slice — complete

The repaired `rmsnorm_fwd_bf16` driver now validates the actual BF16 inputs,
checks every output against a bounded BF16-relative tolerance, reports
allocation/argument/launch failures, and emits a `--summary` mode so a large
timing capture cannot be mistaken for an output dump. Shape metadata now
matches the implementation: one 256-thread block per row. The timing shape
`rows=1024, hidden=4096` passes host validation and Compute Sanitizer; fresh
NVBit covers all 1,024 blocks, replay has 4,235,264 warp records, and NCU
reports the same instruction count. Five application-replay NCU gpc cycles are
48453,48106,48931,50179,48481 (median 48481); simulator lower is 43214, so
lower-bound consistency is true. Dirty-writeback/coherence tails remain open,
and fast-frontier candidates/native dispatch remain zero. Evidence:
`docs/sm86_c7_simbench_rmsnorm_fwd_bf16_direct_live_v1.json` and its schema.
This is not a point-accuracy or speedup claim.

### C7-S31 — atomic-AND histogram slice — complete

The repaired `global_atomic_and_histogram` driver now uses checked integer
parsing, bounded index/allocation arithmetic, launch-error checks, host
allocation checks, and deterministic final histogram validation. Its former
billion-operation timing shape was reduced to `samples=16384,bins=256` in the
shape contract to avoid an OOM/impractical-trace path; the captured `trace_00`
row remains `samples=1024,bins=64`. Fresh host validation and Compute
Sanitizer pass; SASS contains `RED.E.AND`, NVBit/replay cover one block with
42,874 warp records, and NCU reports the same count. Five NCU gpc cycles are
568244,568570,568083,567453,568196 (median 568196); simulator lower is
436096, so lower-bound consistency is true. Dirty-writeback/coherence tails
remain open and native dispatch is zero. Evidence:
`docs/sm86_c7_simbench_global_atomic_and_histogram_direct_live_v1.json` and
its schema. This is not a point-accuracy or speedup claim.

### C7-S32 — vendor BF16 cuBLAS control disposition — complete

The repaired `vendor_gemm_cublas` driver passes the exact `32x32x32` BF16
reference check and Compute Sanitizer. Five same-binary application-replay NCU
captures observe the dynamic
`cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_32x1_nn_align8` kernel with
1,456 warp instructions and gpc cycles 5347,5360,5397,5245,5235 (median
5347). The repository binary's control sidecar has zero functions; NVBit fails
closed on the unsupported tensor-core shape before publishing a complete
trace, even with enhanced tracing disabled. Because the kernel originates in
cuBLASLt and no stable static control crosswalk exists, replay, simulator
timing, native dispatch, and accuracy comparison remain blocked. Evidence:
`docs/sm86_c7_simbench_vendor_gemm_cublas_blocked_live_v1.json` and its
schema. This is not a point-accuracy or speedup claim.

### C7-S33A — research-probe inventory correction — complete

The current tree contains 42 research-probe directories, not the stale 34-row
count: 31 have Makefiles for SM86 CUDA builds, six contain CUDA source or
preserved binaries without a Makefile, and five are analysis-only directories
with no native binary. The corrected inventory is
`docs/sm86_c7_research_probe_inventory_v1.json`; no build or live-capture
success is inferred from an existing executable. This historical 31/6 split
was superseded by M4-E/C7-S33BD after six source rows gained current Makefiles.

### C7-S33B — research-probe build matrix — complete

All 31 current Makefile probes build successfully with the pinned CUDA 13.2,
GCC 14, and `sm_86` toolchain. The machine-readable result is
`docs/sm86_c7_research_probe_build_matrix_v1.json`; generated binaries are
not treated as live evidence. This historical 31-row matrix was extended by
C7-S33BD to the current 37-row matrix; the five analysis-only rows remain
explicit.

### C7-S33C — issue-port vertical slice — complete

The matched `issue_port_control_trace` cell `w1_p10_l16` (dependent IADD,
length 16, one warp, 64 generations) passes seven host correctness trials,
fresh NVBit tracing, full replay, independent tail extraction, and five
same-binary application-replay NCU captures. Replay has 3,034 warp records;
NCU gpc cycles are 9977,9987,9983,9975,9969 (median 9977), while the
simulator completion lower is 7159, so lower-bound consistency is true. The
earlier long 8,192-generation binary measurement was excluded because it was
not the traced executable. The auxiliary enhanced SASS parser skipped its
large input, but the primary payload and static control sidecar are present.
Evidence: `docs/sm86_c7_research_probe_issue_port_control_live_v1.json` and
its schema. This is a probe-slice timing comparison, not a point-accuracy or
speedup claim.

### C7-S33D — matched issue-port 68-cell matrix — complete

The campaign interface was repaired so correctness, NVBit, replay, static
control, and NCU all use the same 64-generation trace executable; the legacy
8,192-generation binary is no longer mixed into scored rows. All 68 cells pass
seven correctness trials, produce full NVBit coverage, replay deterministically,
and have five NCU repetitions with instruction-count crosswalks. Aggregate
result: 68/68 complete, 65 lower-bound-consistent cells, and 3 real
lower-bound falsifiers; native macro dispatch remains zero and all completion
uppers remain open. Evidence:
`docs/sm86_c7_research_probe_issue_port_control_matrix_live_v1.json` and its
schema. NVBit active-thread totals remain separate from warp-record counts.

### C7-S33E — async barrier vertical slice — complete

The async-barrier campaign was repaired to use its bounded 64-group trace
binary for correctness, NCU, NVBit, and replay; the legacy 16,384-group binary
is no longer mixed into scored rows. Cell `w4_c11_l1_g0` passes seven host
correctness trials, fresh NVBit, full deterministic replay, independent async
tail extraction, and five NCU repetitions. Replay and NCU both report 9,104
warp instructions; NCU gpc cycles are 21278,21246,21262,21689,21254 (median
21262), while simulator lower is 24386, a real lower-bound falsifier. Async
copy protocol facts are present, but hardware completion/writeback/coherence
tails remain open. The simulator readiness schema check was updated from a
stale literal v1 requirement to positive-version validation. Evidence:
`docs/sm86_c7_research_probe_async_barrier_overlap_live_v1.json` and its
schema. This is not a point-accuracy or speedup claim.

### C7-S33F — matched async-barrier 36-cell matrix — complete

The repaired campaign uses the same 64-group trace binary for all stages and
accepts the simulator's current positive readiness schema instead of requiring
stale schema version 1. All 36 cells pass seven host correctness trials,
produce full NVBit coverage, replay deterministically, and have five NCU
repetitions with instruction-count crosswalks. Aggregate result: 36/36
complete, 23 lower-bound-consistent cells, and 13 lower-bound falsifiers.
Async-copy lane facts remain explicit per cell; hardware completion,
writeback, and coherence uppers remain open, and native macro dispatch is zero.
Evidence: `docs/sm86_c7_research_probe_async_barrier_overlap_matrix_live_v1.json`
and its schema. No fixed barrier latency or point-accuracy claim is made.

### C7-S33G — store-ack vertical slice — complete

The `store_ack_frontier` store-none arm (`b1_d4_l32_line_unique_g0`) passes
seven host correctness trials, fresh NVBit tracing, full replay, independent
tail extraction, and five same-binary NCU captures. Replay and NCU both report
662 warp instructions; NCU gpc cycles are 6963,6890,6881,6892,6912 (median
6892), while simulator lower is 5238, so lower-bound consistency is true.
Store acknowledgement, writeback, and coherence completion remain unobserved;
native dispatch is zero. Evidence:
`docs/sm86_c7_research_probe_store_ack_frontier_live_v1.json` and its schema.
This is not a point-accuracy or speedup claim.

### C7-S33H — matched store-ack 30-arm matrix — complete

All ten declared `store_ack_frontier` cells and three explicit arms were
captured with one-arm trace-only launches, full NVBit coverage, deterministic
replay, and five NCU repetitions per arm (30 arms, 150 NCU rows). Instruction
counts crosswalk per arm with no failures; all 30 simulator lower bounds remain
below their NCU medians. The write-ack candidate remains a model intervention,
not a hardware completion oracle; completion/writeback/coherence uppers and
native dispatch remain open. Evidence:
`docs/sm86_c7_research_probe_store_ack_frontier_matrix_live_v1.json` and its
schema.

### C7-S33I — memory-retirement vertical slice — complete

The matched `memory_retirement_service` store/none/line_stream cell
`b1_store_none_line_stream_d4_l32` passes seven host correctness trials, fresh
NVBit, full replay, independent tail extraction, and five NCU captures. Replay
and NCU both report 1,178 warp instructions; NCU gpc cycles are
11464,11391,11370,11363,11402 (median 11391), while simulator lower is 9064,
so lower-bound consistency is true. Memory retirement/writeback/coherence
completion remains open and native dispatch is zero. Evidence:
`docs/sm86_c7_research_probe_memory_retirement_service_live_v1.json` and its
schema.

### C7-S33J — memory-retirement 102-cell matrix — complete

All 102 declared cells pass host correctness, fresh NVBit capture, and five
NCU repetitions. Replay now completes all 102 cells after the simulator parks
`CCTL.IVALL` behind the next external L1-fill response horizon when a pending
fill has no local timestamp. Eight system-fence cells retain explicit NCU/replay
instruction-count mismatches (NCU reports nine additional warp instructions
per CTA); these are crosswalk failures, not silently scored rows. Lower-bound
consistency is reported only where instruction identity matches. Evidence:
`docs/sm86_c7_research_probe_memory_retirement_service_matrix_live_v1.json`
and its schema. Completion uppers and native dispatch remain open.

### C7-S33K — CCTL pending-fill repair — complete

The replay seam now refuses to mutate L1 on an untimed pending fill. It parks
the SM at the system-provided response horizon, then retries `CCTL.IVALL`
after the response is consumed; timed fills use the cache's explicit due-time
frontier. Focused sim-mem/sim-sm tests pass (199 + 493 tests), and all four
previous CCTL replay blockers now complete. No invented completion token or
upper bound was added.

### C7-S33L — system-fence crosswalk diagnostic — complete

Investigate the eight system-fence NCU/replay instruction-count mismatches.
Static/dynamic opcode populations and NCU pipe/op counters were compared on a
matched no-fence/system-fence pair. NVBit adds three visible SASS instructions
(`MEMBAR`, `ERRBAR`, `CCTL`), while NCU reports twelve additional warp
instructions and deltas across branch/ALU/LSU/texture/global-store counters.
This is an unresolved NCU-versus-trace metric-semantics/counter-replay
mismatch; simulator padding and hidden-instruction synthesis are explicitly
rejected. Diagnostic evidence:
`docs/sm86_c7_memory_retirement_system_fence_crosswalk_diagnostic_v1.json` and
its schema. The eight rows remain crosswalk exclusions until an independent
instruction-identity explanation is found.

### C7-S33M — source-counter crosswalk check — corrected

An independent NCU SourceCounters export was compared for matched no-fence and
system-fence binaries. The source rows report one warp execution for each
visible `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL` PC; their separate stall
sampling columns are zero. The full source-row sum is 1104 versus 1101, while
`smsp__inst_executed.sum` is 1113 versus 1101. The metric crosswalk remains
unresolved; no simulator padding or hidden-instruction synthesis is applied.
The corrected diagnostic is
`docs/sm86_c7_memory_retirement_system_fence_crosswalk_diagnostic_v1.json`.

### C7-S33N — barrier-pipeline vertical slice — complete

The matched 64-generation `barrier_pipeline` cell `w4_b10_s0` (one CTA
barrier, four warps, zero skew) passes seven host correctness trials, fresh
NVBit, full replay, independent tail extraction, and five NCU captures. Replay
and NCU both report 1,884 warp instructions; NCU gpc cycles are
5731,5732,5755,5728,5732 (median 5732), while simulator lower is 3545, so
lower-bound consistency is true. Barrier rendezvous/retirement tails remain
open and native dispatch is zero. Evidence:
`docs/sm86_c7_research_probe_barrier_pipeline_live_v1.json` and its schema.

### C7-S33O — matched barrier-pipeline 33-cell matrix — complete

All 33 declared `barrier_pipeline` cells use the matched 64-generation trace
binary for correctness, fresh NVBit, deterministic replay, and five NCU
repetitions. Every cell has an instruction-count crosswalk and remains
lower-bound-consistent; barrier protocol, warp width, and skew effects remain
mechanism-local observations with open completion tails. Native dispatch is
zero. Evidence:
`docs/sm86_c7_research_probe_barrier_pipeline_matrix_live_v1.json` and its
schema.

### C7-S33P — l1tex admission vertical slice — complete

The matched 64-generation `l1tex_admission_backpressure` store/line-unique
cell (`blocks=1, warps=4, depth=4, active_lanes=32, gap=0, all pattern`)
passes seven host correctness trials, fresh NVBit, full replay, independent
tail extraction, and five NCU captures. Replay and NCU both report 2,980 warp
instructions; NCU gpc cycles are 8898,8956,8887,8905,8976 (median 8905),
while simulator lower is 6524, so lower-bound consistency is true. The
post-burst MEMBAR/ERRBAR/CCTL sequence and writeback/coherence tails remain
open; native dispatch is zero. Evidence:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_live_v1.json` and
its schema.

### C7-S33Q — matched l1tex admission 100-cell matrix — complete

All 100 configured `l1tex_admission_backpressure` cells pass host correctness,
fresh NVBit capture, deterministic replay, and five NCU repetitions with
per-cell instruction crosswalks. Every replay-complete cell is lower-bound
consistent; the post-burst MEMBAR/ERRBAR/CCTL sequence and writeback/coherence
uppers remain open, and native dispatch is zero. Evidence:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_matrix_live_v1.json`
and its schema.

### C7-S33R — l2 queue-frontier vertical slice — superseded

The original two-launch `l2_queue_frontier` capture is retained as historical
v1 evidence. Its shape and replay observations remain useful, but its host
binary did not independently validate the checksum and allocation contract.
The corrected same-binary capture is recorded under C7-S33S.

### C7-S33S — L2 probe safety and evidence repair — complete

The L2 queue-frontier driver now rejects malformed/overflowing shapes, checks
the aggregate allocation against available device memory before `cudaMalloc`,
initializes output sentinels, independently recomputes the lane-zero checksum,
and fails on invalid timestamp bounds or checksum results. A valid slice and a
reuse slice pass; an oversized allocation is rejected before launch, and
Compute Sanitizer reports zero errors. The fresh artifact is
`docs/sm86_c7_research_probe_l2_queue_frontier_live_v2.json`; it uses the same
rebuilt binary for correctness, NVBit, replay, and five NCU repetitions. Target
replay/NCU both report 70,560 warp instructions; NCU gpc cycles are
7714,7784,7729,7666,7663 (median 7714), while the simulator lower is 5751.
The lower remains consistent, but dirty-writeback/coherence completion is
unknown and no point-accuracy or speedup claim is made.

### C7-S33T — L1TEX arrival/replay vertical slice — complete

The driver now uses strict integer parsing, checked allocation arithmetic, a
512-MiB/half-free-memory budget, and explicit dependent-work observability.
The stale separate trace binary was removed from the campaign interface;
correctness, NVBit, replay, and NCU all bind to the rebuilt executable. The
`store,line,84,8,8,32,0,1` cell passes seven host trials, a reduced
Compute-Sanitizer run, fresh 84-block NVBit capture, deterministic replay, and
five application-replay NCU captures. Replay and NCU both report 809,088 warp
instructions; NCU gpc cycles are 32085,31590,32333,31748,31518 (median
31748), while the simulator lower is 8486. Output-wavefront, request, L2,
DRAM-write, and issue counters are retained as independent observations, not
simulator inputs. Dirty-writeback/coherence completion remains unknown and no
point-accuracy or speedup claim is made. Evidence:
`docs/sm86_c7_research_probe_l1tex_arrival_replay_service_live_v1.json`.

### C7-S33U — SFU dependent-chain vertical slice — complete with falsifier

The `sfu_frontier` driver now uses strict integer parsing, a 262,144-iteration
bound, checked allocation arithmetic, a 64-MiB/half-free-memory budget, output
sentinels, and timestamp/checksum-presence checks. The dependent cell
`84,4,1024,dependent` passes seven host trials, reduced Compute Sanitizer,
four-launch NVBit capture, target-4 replay, and five launch-3-skipped NCU
captures. Replay and NCU both report 1,389,024 target warp instructions; NCU
gpc cycles are 34274,34282,34292,34297,34274 (median 34282), but the
simulator lower is 38099. This is a real lower-bound falsifier, not a tuning
opportunity; the cell is retained as negative evidence. No SFU latency is
fitted into a scored workload, and dirty-writeback/coherence completion
remains unknown. Evidence:
`docs/sm86_c7_research_probe_sfu_frontier_live_v1.json`.

### C7-S33V — result-queue collision vertical slice — complete

The campaign no longer treats 64-generation trace cubins as equivalent to the
8,192-generation NCU cubins. It now uses one patched collision or separated
cubin for correctness, NVBit, replay, and NCU; the stale trace artifacts were
removed. The CUDA-driver harness uses strict parsing, bounded allocations,
sentinel output checks, elapsed-sample checks, and the deterministic sink
oracle. The collision cell (`warps=4, generations=8192`) passes seven host
trials, Compute Sanitizer, full NVBit coverage, deterministic replay, and five
NCU runs. Replay and NCU both report 524,416 warp instructions; NCU gpc cycles
are 307642,307646,307655,307628,307679 (median 307646), while simulator lower
is 287635. The result-queue hypothesis remains exploratory; no latency is
activated or fitted and dirty-writeback/coherence completion remains unknown.
Evidence: `docs/sm86_c7_research_probe_fixed_latency_result_queue_live_v1.json`.

### C7-S33W — result-queue matched collision/separated differential — complete with falsifier

The separated treatment was captured at the identical width and generation
count with the same driver, patched-cubin contract, replay, clocks, cache
policy, and five-NCU protocol. Both arms pass the host sink oracle, Compute
Sanitizer, full one-CTA NVBit/replay coverage, and instruction-count
crosswalks. Collision NCU median is 307646 with simulator lower 287635;
separated median is 340424 with simulator lower 353171, so separated is a real
lower-bound falsifier. The NCU treatment delta is 32778 cycles while the
simulator lower-bound delta is 65536; this differential is mechanism-local
evidence, not a fixed latency. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_pair_live_v1.json`.

### C7-S33X — result-queue held-out width-8 pair — complete

The held-out `warps=8, generations=8192` collision/separated pair uses the
same one-cubin-per-treatment protocol. The live trace exposed a simulator
preflight ceiling rather than a workload failure: each block has 1,048,832
decoded instructions, so the finite wire-preflight limit was raised from
1,000,000 to 2,000,000 and the compact route index widened by one bit. The
same captured traces then replay successfully. Both arms pass seven host
trials, Compute Sanitizer, full NVBit capture, deterministic replay, and five
NCU runs with 1,048,832 matching target warp instructions. Collision NCU median
is 447092 with simulator lower 320466; separated median is 438935 with lower
369618. Both lowers are individually consistent, but hardware separated is
8157 cycles below collision while the simulator lower delta is +49152; the
separated repeats are high-dispersion. No result-queue latency is activated.
Evidence: `docs/sm86_c7_research_probe_fixed_latency_result_queue_width8_pair_live_v1.json`.

### C7-S33Y — result-queue width-12 pair — complete

The width-12 collision/separated pair uses the same 8,192-generation
single-cubin contract and the finite two-million-instruction decoder ceiling.
Each block has 1,573,248 decoded instructions and replays without truncation.
Both arms pass seven host trials, Compute Sanitizer, full NVBit capture,
deterministic replay, and five NCU runs with matching 1,573,248 target warp
instructions. Collision NCU median is 589202 with simulator lower 527551;
separated median is 561568 with lower 533490. Both lowers are individually
consistent, but hardware separated is 27634 cycles below collision while the
simulator lower delta is +5939; repeats remain dispersed and no latency is
fitted. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_width12_pair_live_v1.json`.

### C7-S33Z — result-queue width-16 pair — complete

The declared width-16 collision/separated pair contains 2,097,664 decoded
instructions per block. The prior two-million instruction/decoded-node
ceilings rejected the exact trace before replay, so the finite limits were
raised once to 3,000,000 instructions and 4,000,000 decoded nodes, with the
compact route index widened to 22 bits. Both arms then passed seven host
trials, Compute Sanitizer, full NVBit capture, deterministic replay, and five
NCU runs with matching 2,097,664 target warp instructions. Collision NCU
median is 708805 with simulator lower 579430; separated median is 693935 with
lower 654998. Both lowers are individually consistent, but hardware separated
is 14870 cycles below collision while the simulator lower delta is +75568; no
result-queue latency is fitted. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_width16_pair_live_v1.json`.

### C7-S33AA — completion-distance split vertical slice — complete

The completion-distance campaign now uses one patched 8,192-generation cubin
for correctness, NVBit, replay, and NCU; the stale 64-generation trace pair
was removed. The CUDA-driver harness has strict parsing, bounded allocations,
sentinel output checks, elapsed-sample checks, and the deterministic sink
oracle. The `distance=1, variant=split, warps=4` cell passes seven host trials,
Compute Sanitizer, full NVBit capture, deterministic replay, and five NCU runs.
Replay and NCU both report 622,832 target warp instructions; NCU median is
586172 and simulator lower is 566204, so lower-bound consistency is true.
Completion tails remain open; this is a narrow mechanism observation, not an
activated latency or point-accuracy result. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_completion_live_v1.json`.

### C7-S33AB — completion-distance bank3 matched arm — complete

The matched `distance=1, variant=bank3, warps=4` arm uses one patched
8,192-generation cubin across correctness, NVBit, replay, and NCU. It passes
seven host trials, Compute Sanitizer, full NVBit capture, deterministic replay,
and five NCU runs with 622,832 matching target warp instructions. Its simulator
lower is 566204 and NCU median is 586129. Against the fresh split arm (NCU
median 586107), bank3 differs by only +22 cycles; no completion latency is
activated. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_completion_pair_live_v1.json`.

### C7-S33AC — completion-distance held-out distance-2 pair — complete

The `distance=2, warps=4` split/bank3 pair uses one patched 8,192-generation
cubin per layout across correctness, NVBit, replay, and NCU. Both arms pass
seven host trials, Compute Sanitizer, full NVBit capture, deterministic replay,
and five NCU runs with 622,832 matching target warp instructions. The simulator
lower is 336828 for both arms; split NCU median is 340415 and bank3 median is
356791. Both lowers are consistent, while the 16,376-cycle bank3 differential
is retained as mechanism-local evidence and no latency is fitted. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_completion_distance2_pair_live_v1.json`.

### C7-S33AD — completion-distance held-out distance-4 pair — complete

The `distance=4, warps=4` split/bank3 pair uses one patched 8,192-generation
cubin per layout across all stages. Both arms pass seven host trials, Compute
Sanitizer, full NVBit capture, deterministic replay, and five NCU runs with
622,832 matching target warp instructions. Simulator lower is 336826 for both;
split NCU median is 348610 and bank3 median 414145. Both lowers are consistent,
while the 65,535-cycle bank3 differential is retained as mechanism-local
evidence and no latency is fitted. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_completion_distance4_pair_live_v1.json`.

### C7-S33AE — completion-distance held-out distance-8 pair — complete

The `distance=8, warps=4` split/bank3 pair uses one patched 8,192-generation
cubin per layout across all stages. Both arms pass seven host trials, Compute
Sanitizer, full NVBit capture, deterministic replay, and five NCU runs with
622,832 matching target warp instructions. Simulator lower is 336826 for both;
split NCU median is 356797 and bank3 median 414112. Both lowers are consistent,
while the 57,315-cycle bank3 differential is retained as mechanism-local
evidence and no latency is fitted. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_completion_distance8_pair_live_v1.json`.

### C7-S33AF — completion-distance family closure and inventory reconciliation — complete

All current completion/result-queue/L1TEX campaign manifests now use the one
rebuilt binary per treatment and 8,192-generation contract. Focused correctness
selection runs pass for the updated drivers; current static projection generation
works without the old predecessor-sidecar requirement. The stale 64-generation
paired-binary names are absent from the active campaign interfaces and the
analyzers now use the current three-cubin/one-pair layouts. The inventory now
records 18 live probe stages: the original six matrices plus L2, L1TEX arrival,
SFU, result-queue width arms, and completion distances 1/2/4/8. All negative
lower-bound, crosswalk, open-tail, and unsupported dispositions remain explicit.
Evidence: `docs/sm86_c7_research_probe_inventory_v1.json`.

### C7-S33AG — atomic completion serialization vertical slice — complete

The atomic campaign now uses one rebuilt 1,024-generation cubin per variant for
correctness, NVBit, replay, and NCU; the stale 64-generation trace cubins were
removed. The driver uses strict parsing, variant/layout consistency checks,
bounded allocations, output sentinels, elapsed-sample validation, returned-sum
and final-data oracles, and explicit cleanup. The `ret_lane_word, warps=4`
cell passes seven host trials, Compute Sanitizer, full NVBit capture,
deterministic replay, and five NCU runs. Replay and NCU both report 24,660
warp instructions; atomic request counters report 4,096, NCU median is 316109,
and simulator lower is 267561. Atomic serialization/coherence tails remain
unknown; no atomic timing model is activated. Evidence:
`docs/sm86_c7_research_probe_atomic_completion_serialization_live_v1.json`.

### C7-S33AH — multi-SM atomic service vertical slice — complete

The multi-SM atomic campaign now uses one rebuilt 1,024-generation cubin per
operation for correctness, NVBit, replay, and NCU; stale 64-generation trace
cubins are removed. The driver rejects non-1024 generation requests, bounds
allocation size, checks elapsed samples, validates data/returned sums, and
cleans up CUDA resources. The `ret32, blocks=4, spacing=0` cell passes seven
host trials, Compute Sanitizer, full NVBit capture, deterministic replay, and
five NCU runs. Replay/NCU report 24,692 target warp instructions; NCU observes
4,096 atomic requests and 131,072 LTS requests, median 355107, while simulator
lower is 268019. Routing/completion tails remain black-box; no timing model is
activated. Evidence:
`docs/sm86_c7_research_probe_atomic_multism_service_live_v1.json`.

### C7-S33AI — atomic alias adversarial vertical slice — complete

The alias-adversarial campaign now uses one rebuilt 1,024-generation cubin per
width/operation; stale 64-generation trace twins are removed. The driver
rejects stale generations, validates the pattern/alias/width contract, bounds
allocation size, checks elapsed samples, data/returned sums, and cleans up.
The ret32 dense alias-32 cell (`warps=4, stride=4`) passes seven host trials,
Compute Sanitizer, full NVBit capture, deterministic replay, and five NCU runs.
Replay/NCU report 24,684 target warp instructions; atomic requests are 4,096,
LTS requests 131,072, NCU median 382651, and simulator lower 267996. This
remains observational; no proprietary hash is reconstructed and no timing model
is activated. Evidence:
`docs/sm86_c7_research_probe_atomic_alias_adversarial_live_v1.json`.

### C7-S33AJ — CUDA VMM alias diagnostic — complete, simulator not applicable

The current VMM driver was rebuilt and its bounded correctness campaign passes
19/19 cells, including seven-trial exact VMM alias/data oracles. The selected
`alias_same_offset` offset-0 cell has five fresh NCU profiles and an NVBit
capture covering all seven VMM launches. NCU read-sector medians are retained,
but this probe is explicitly not a native sim-run timing row: VMM virtual
mappings and physical allocation handles are outside the ordinary trace replay
contract. No lower bound, hash, or timing model is fabricated. Evidence:
`docs/sm86_c7_research_probe_atomic_vmm_alias_live_v2.json`.

### C7-S33AK — task-done multi-kernel boundary slice — complete

The same `store_device` binary passes seven host correctness trials for a
16-MiB footprint, 16-MiB thrash, four blocks, and two warmups. NVBit captures
all nine launches with enhanced metadata enabled; sim-run replays all nine
launches with full coverage, and five application-replay NCU runs profile the
same nine-launch application. Per-kernel instruction counts match. Producer,
thrash, consumer, copy, and host endpoints remain separate; no fixed
retirement tail or scalar upper is inferred. Evidence:
`docs/sm86_c7_research_probe_task_done_boundary_live_v1.json`.

### C7-S33AL — live inventory reconciliation and stale-interface closure — complete

All 23 live probe stages have schema-validated artifacts, and the eight current
campaign manifests plus focused selection runs pass. Active paired-binary
campaigns no longer require stale trace twins or predecessor-sidecar
populations. Population entries now carry explicit live status/artifact links
for the completed probe families; 26 current research-probe entries remain
uncaptured or analysis-only and are not counted as complete. Unsupported
SimBench tensor/vendor rows, system-fence crosswalk exclusions,
lower-bound falsifiers, open semantic tails, and VMM physical-identity limits
remain explicit. Evidence: `docs/sm86_c7_research_probe_inventory_v1.json`.

### C7-S33AM — SMSP scheduler-contention vertical slice — complete

The scheduler campaign now uses one rebuilt 8,192-generation binary for
correctness, NVBit, replay, and NCU; its stale trace twin is removed. The
`q20, phase=0, warps=8` cell passes seven host trials, Compute Sanitizer, full
NVBit capture, deterministic replay, and five NCU runs. Its 69-MiB block
payload required the finite anchored-reader limit to rise from 64 MiB to 128
MiB; focused anchored tests pass. Replay and NCU match at 2,556,096
instructions; simulator lower is 1,123,534 and NCU median 1,271,033. Phase
labels remain experimental controls, not physical scheduler-slot truth; no
scheduler timing model is activated. Evidence:
`docs/sm86_c7_research_probe_smsp_scheduler_contention_live_v1.json`.

### C7-S33AN — fixed-result bypass safe-arm vertical slice — complete

The bypass campaign now uses one rebuilt 1,025-generation cubin per safe
variant; stale 65-generation trace twins are removed. The driver rejects stale
generation counts, validates elapsed/sink outputs, and cleans up. The safe
`fixed_retired, kind=fixed, warps=4` arm passes seven host trials, Compute
Sanitizer, metadata-enabled NVBit capture, deterministic replay, and five NCU
runs. Replay/NCU match at 28,788 instructions; simulator lower is 27505 and
NCU median 34930. The intentionally illegal-address global distance-4 canary
was not run. No bypass latency model is activated. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_result_bypass_live_v1.json`.

### C7-S33AO — original selection

Select the next uncaptured non-memory probe after source review, prioritizing
`fixed_latency_result_capacity` or `immediate_constant_frontier`. Preserve all
replay blockers and lower-bound falsifiers.

### C7-S33AO — original refresh — superseded; correction in M1

The capacity campaign no longer builds or invokes a 64-generation trace twin.
Its driver requires 8,192 generations, and the same patched b7/b8/b9 cubins are
used for correctness, NVBit, replay, and NCU. The W8/b9 arm passed seven live
correctness trials, one complete NVBit capture, deterministic replay, and five
independent NCU profiles. Replay and NCU agree at 1,573,144 target
instructions; simulator lower is 590,830 cycles and NCU median is 851,475.
This is a canary only: one arm cannot identify physical result capacity or pop
width, so no capacity timing model is enabled. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_result_capacity_live_v1.json`.

### C7-S33AP — original selection

Review and refresh `immediate_constant_frontier` as the next uncaptured
non-memory probe. Preserve its functional-only boundary unless a native
NVBit→sim-run trace and independent NCU anchor can be produced.

### C7-S33AP — original anchor — superseded; correction in M2

The hand-authored PTX cubin was run from its correct module directory. The
same cubin passed the direct-clock correctness run, metadata-enabled NVBit
capture, control-sidecar generation, deterministic sim-run replay, and an
independent base-clock NCU profile. The 84-block/256-iteration arm produced
2688 thread records (84 warps); replay/NCU agree at 151,872 instructions, simulator lower is
6,037 cycles, and NCU reports 10,709 GPC cycles. The driver ELF was correctly
excluded from cubin parsing; the kernel SASS still yielded 32 decoded
instructions. This remains observational and does not activate an IMC latency
model. Evidence: `docs/sm86_c7_research_probe_immediate_constant_frontier_live_v1.json`.

### C7-S33AQ — constant-load path vertical slice — complete with falsifier

The previously preserved `constant_load_path` source was not accepted as live
evidence. Its executable was rebuilt with strict integer parsing, bounded input
sizing, sentinel-initialized result buffers, and a separate functional-result
output so correctness does not depend on a clock sample. The same rebuilt
binary (`mode=0, blocks=84, threads=256, iterations=256, warmups=0`) passed
seven full-shape correctness runs, an independent result oracle, reduced
Compute Sanitizer, NVBit tracing, deterministic replay, and five independent
application-replay NCU profiles. Replay and NCU agree at 1,909,824 target warp
instructions. The simulator lower bound is 123,727 cycles while the NCU median
is 24,016, making this a mechanism-local lower-bound falsifier. No constant
load latency or cache parameter was fitted, and dirty-writeback/coherence
tails remain open. Evidence:
`docs/sm86_c7_research_probe_constant_load_path_live_v1.json`.

### C7-S33AR — L1TEX admission same-binary refresh — complete

The L1TEX campaign no longer builds or invokes a trace-only executable. Its
Makefile, campaign, static sidecar path, and simulator path now use one rebuilt
`l1tex_admission_backpressure` binary. The `store, line_unique, blocks=84,
warps=8, depth=32, active_lanes=8, gap=0, pattern=all` cell passed seven
correctness trials, one full-shape Compute Sanitizer trial, NVBit, deterministic
replay, and five application-replay NCU profiles. Replay and NCU match at
1,255,968 target warp instructions; simulator lower is 12,905 cycles and NCU
median is 29,879. This is a current same-binary plumbing observation, not a
held-out accuracy result; no L1TEX admission/cache parameter is fit and
dirty-writeback/coherence tails remain open. Evidence:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_live_v2.json`.

### C7-S33AS — same-binary control/store differential — complete with mismatch

The predeclared `control` versus `store` pair uses one rebuilt binary, one
control sidecar, identical shape/clock/cache/launch settings, seven correctness
trials, full-shape Compute Sanitizer, NVBit, deterministic replay, and five
application-replay NCU repetitions per arm. Both instruction crosswalks match
and both simulator lowers remain below their NCU observations. Hardware adds
12,823 cycles (16,967 to 29,790), while the simulator lower-bound delta is
556 cycles (12,349 to 12,905). This is a mechanism-local differential
mismatch; no store/L1TEX latency is fitted and completion/dirty-writeback/
coherence remain open. Evidence:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_pair_live_v1.json`.

### C7-S33AT — inventory/interface reconciliation — complete

The inventory now records the current same-binary L1TEX refresh and its
control/store differential, and removes the retired trace-only executable from
the active L1TEX and Allocate candidate interfaces. Historical matrix artifacts
remain labeled as historical; their rows are not silently promoted to current
same-binary evidence. The reconciled inventory is
`docs/sm86_c7_research_probe_inventory_v1.json`.

### C7-S33AU — fixed-Allocate same-binary vertical slice — complete

The Allocate Makefile/campaign now use one 8,192-generation cubin for every
active stage. The host driver was hardened with strict integer parsing,
sentinel-initialized elapsed/sink buffers, and separate elapsed/sink validity
fields. The `w8_r10_native` cell passed seven correctness trials, a full-shape
Compute Sanitizer trial, NVBit, deterministic replay, and five
application-replay NCU profiles. Replay and NCU match at 1,245,536 target warp
instructions; simulator lower is 336,374 cycles and NCU median is 593,194.
No register-bank/allocation timing is fitted, and dirty-writeback/coherence
tails remain open. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_allocate_live_v1.json`.

### C7-S33AV — front-end footprint same-binary vertical slice — complete

The front-end footprint Makefile/campaign now use one 1,024-generation U1
cubin for correctness, NVBit, replay, and NCU; the historical short trace pair
is not an active input. The driver now parses numeric arguments strictly,
initializes elapsed/sink buffers with sentinels, and reports elapsed/sink
validity separately from its functional oracle. The `u1_w4_fixed_retired` cell
passed seven correctness trials, a full-shape Compute Sanitizer trial, NVBit,
deterministic replay, and five application-replay NCU profiles. Replay and NCU
match at 32,852 target warp instructions; simulator lower is 32,403 cycles and
NCU median is 38,629. No branch-target/instruction-buffer timing is fitted and
dirty-writeback/coherence tails remain open. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_front_end_footprint_live_v1.json`.

### C7-S33AW — result-order same-binary vertical slice — complete

The result-order Makefile/campaign now use one 8,192-generation cubin for
correctness, NVBit, replay, and NCU; the historical 64-generation trace cubin
is not an active input. The host driver now parses numeric arguments strictly,
initializes elapsed/sink buffers with sentinels, and reports elapsed/sink
validity separately from its functional oracle. The `w8_hf` cell passed seven
correctness trials, a full-shape Compute Sanitizer trial, NVBit, deterministic
replay, and five application-replay NCU profiles. Replay and NCU match at
458,968 target warp instructions; simulator lower is 197,490 cycles and NCU
median is 257,751. No result-order/issue-queue timing is fitted and
dirty-writeback/coherence tails remain open. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_result_order_live_v1.json`.

### C7-S33AX — same-bank result-order vertical slice — complete

The same-bank Makefile/campaign now use one 8,192-generation cubin for
correctness, NVBit, replay, and NCU; the historical short trace twin is not an
active input. The host driver now parses numeric arguments strictly,
initializes elapsed/sink buffers with sentinels, and reports elapsed/sink
validity separately from the functional oracle. The `w8_hf` cell passed seven
correctness trials, a full-shape Compute Sanitizer trial, NVBit, deterministic
replay, and five application-replay NCU profiles. Replay and NCU match at
458,960 target warp instructions; simulator lower is 197,489 cycles and NCU
median is 233,169. No physical bank mapping or latency is inferred and
dirty-writeback/coherence tails remain open. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_result_same_bank_live_v1.json`.

### C7-S33AY — register-reuse same-binary vertical slice — complete

The reuse Makefile/campaign now use one 8,192-generation cubin per variant for
correctness, NVBit, replay, and NCU; historical short trace twins are not
active inputs. The host driver now parses numeric arguments strictly,
initializes elapsed/sink buffers with sentinels, and reports elapsed/sink
validity separately from its functional oracle. The `w8_u11_a` cell passed
seven correctness trials, a full-shape Compute Sanitizer trial, NVBit,
deterministic replay, and five application-replay NCU profiles. Replay and NCU
match at 1,245,824 target warp instructions; simulator lower is 336,953 cycles
and NCU median is 635,271. No reuse/scoreboard timing is fitted and
dirty-writeback/coherence tails remain open. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_reuse_live_v1.json`.

### C7-S33AZ — front-end IBuffer same-binary vertical slice — complete

The baseline_s1 cubin passed seven timeline/sink-oracle correctness trials,
full-shape Compute Sanitizer, NVBit, deterministic replay, and five
application-replay NCU profiles. Replay and NCU match at 944 target warp
instructions; simulator lower is 818 cycles and NCU median is 3,678. No
instruction-buffer depth is fitted and completion tails remain open. Evidence:
`docs/sm86_c7_research_probe_front_end_ibuffer_depth_live_v1.json`.

### C7-S33BB — remaining-source review and vertical capture — complete (M4-E rechecked)

The remaining population was source-reviewed one family at a time. Numeric
argument and result-oracle repairs landed before profiling. The admitted rows
then received same-binary correctness, Compute Sanitizer, NVBit, deterministic
replay, and five NCU observations. The source-only CUDA families were built
and captured rather than silently excluded; Python-only wrappers and host-only
benchmarks received explicit non-native dispositions in the inventory.

Evidence records:

- `sm86_c7_research_probe_async_source_only_live_v1.json`: 2,088 replay
  instructions, lower 30,905, NCU median 33,800; async visibility remains
  open.
- `sm86_c7_research_probe_atomic_lts_mapping_live_v1.json`: 6,171 replay
  instructions, lower 3,400, NCU median 9,413; no address hash promotion.
- `sm86_c7_research_probe_fixed_latency_result_load_priority_live_v1.json`:
  65,752 instructions, lower 70,215, NCU median 65,485; lower-bound
  falsifier retained.
- `sm86_c7_research_probe_launch_cycle_semantics_live_v1.json`: 87,780
  instructions, lower 8,466, NCU median 10,711; launch and retirement
  boundaries remain separate.
- `sm86_c7_research_probe_ldgsts_burst_release_live_v2.json`: 9,922
  instructions, lower 32,695, NCU median 31,835; corrected marker contract,
  process-start timeout and wrong-launch attempts retained.
- `sm86_c7_research_probe_fixed_latency_result_bypass_live_v2.json` and
  `sm86_c7_research_probe_smsp_scheduler_contention_live_v2.json`: repaired
  every-trial host oracles and fresh safe-arm/contention verticals.
- `sm86_c7_research_probe_sm86_branch_domains_live_v1.json`,
  `sm86_c7_research_probe_sm86_metric_domains_live_v1.json`,
  `sm86_c7_research_probe_sm86_pipeline_categories_live_v1.json`,
  `sm86_c7_research_probe_sm86_value_differential_live_v1.json`, and
  `sm86_c7_research_probe_sm86_coherence_litmus_live_v1.json`: source-only
  families built and captured with finite value/control oracles.

No scored NCU value was supplied to replay or fitting, and no timing/card
parameter was selected from these observations.

### C7-S33BC — M4 inventory/schema reconciliation — complete

`docs/sm86_c7_research_probe_inventory_v1.json` now reports
`48_live_probe_stages_reconciled_m4c_verticals_complete`. Every current
research-probe directory has either a live artifact link or an explicit
analysis-only/source-wrapper disposition; all twelve new schemas validate with
the retained records. M4 is closed. M5 remains open for completion-tail
semantics, calibration-only accuracy, and a real native continuation
differential; capture completion is not a native-dispatch or hardware-accuracy
claim.

### C7-S33BD — research-probe inventory/build-matrix synchronization — complete

The inventory/build audit found a stale retained build matrix: the current
inventory contains 42 directories and 37 Makefile probes, while the prior
matrix had only 31 rows. The six missing current Makefile families
(`constant_load_path`, `sm86_branch_domains`, `sm86_coherence_litmus`,
`sm86_metric_domains`, `sm86_pipeline_categories`, and
`sm86_value_differential`) received bounded make checks on 2026-10-05 with return code zero.
The matrix now has 37/37 passed rows, the inventory and matrix name sets are
equal, and the five analysis-only rows remain explicit. This repairs
bookkeeping drift only; it does not clear semantic tails or claim native
dispatch. Evidence: `docs/sm86_c7_research_probe_inventory_sync_v1.json`.

### C7-S34A — SOL baseline source population inventory — complete

The 203 SOL baseline problem directories were enumerated from the current
tree without execution or scoring: FIB 25, L1 94, L2 82, and Quant 2. All
203 contain CUDA source, host C++ source, and `solution.json`; 201 contain an
SM86/compute-86 source marker, two Quant rows do not, and no local cubin or
executable was present. A source marker is not proof of compilation or GPU
architecture. Evidence: `docs/sm86_c7_sol_population_source_inventory_v1.json`.

### C7-S34B — one SOL candidate source review — complete explicit blocker

Select one SM86-marked problem with a bounded host entry point and an explicit
correctness oracle before attempting any native build. The candidate is
`FIB/problem_001_fused_add_rmsnorm_h2048/cuda_cpp`, whose source declares
SM86 code generation, a torch extension entry point, BF16 shape checks, and
destination-passing output. The dependency check found CUDA 13.2/GCC 14 but no Torch in either local Python
environment. No build or native execution was attempted; the row remains an
explicit non-substituted blocker.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34B1 | Verify candidate source, build dependencies, shape/oracle contract, and exact SM86 compile path | complete — source contract valid; existing SOL Torch environment found |
| C7-S34B2 | Build and run a bounded correctness/sanitizer smoke without NCU or simulator fitting | complete — explicit Torch environment found; SM86 build/correctness/memcheck passed |
| C7-S34B3 | If the smoke is admissible, capture NVBit/replay/NCU as a separate vertical; otherwise retain an explicit blocker | complete — NVBit/control/replay and five target NCU vertical passed with parser/unit limitations retained |

**C7-S34B result (source/build smoke).** The selected FIB problem has a valid
SM86 source/shape/oracle contract. The default Python environments lack Torch,
but the existing `tools/sol_execbench/.venv` supplies Torch 2.12.0+cu130 and
SM86 CUDA; explicit `nvcc -ccbin=/usr/bin/g++-14` produced the extension. Batch
1/2 BF16 correctness passed with finite output and maximum absolute error below
0.015, and Compute Sanitizer memcheck reported zero errors. The native vertical result
is recorded immediately below. Evidence: `results/alpha/c7_s34b_sol_fib001_20261005/`.

**C7-S34B native vertical result.** The profiler-delimited candidate produced a
real NVBit launch, an independent 176-instruction SM86 control sidecar, and a
deterministic `l1-l2` replay with 3824 records. Five corrected application
replay NCU runs selected the fused RMSNorm kernel and measured a 5374-cycle
median with 3824 executed instructions, matching simulator records. The raw
NVBit stats count (11816) is retained as a separate unit; enhanced JIT SASS
parsing failed but did not prevent the independent sidecar. Dirty-writeback
and coherence tails remain open and native macro dispatch is zero. Evidence:
`docs/sm86_c7_sol_candidate_vertical_v1.json`.

### C7-S34C next active phase — environment disposition (superseded for selected candidate)

C7-S34C is superseded for the selected candidate: the existing
`tools/sol_execbench/.venv` Torch runtime is permitted and SM86-capable. The
broader SOL population still requires per-candidate dependency dispositions;
no package installation or substitution is authorized.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34C1 | Search existing repository environments/toolchain manifests for a compatible Torch runtime | complete — Torch 2.12.0+cu130 found in tools/sol_execbench/.venv |
| C7-S34C2 | If absent, record a reproducible environment blocker without external writes or substitution | complete scoped — default venv lacks Torch; selected SOL environment is usable |
| C7-S34C3 | Synchronize the population inventory and plan; leave native completion claims unchanged | complete — candidate plan updated; native completion claims unchanged |

### C7-S34D — second SOL candidate source review — active

Select `FIB/problem_002_fused_add_rmsnorm_h4096/cuda_cpp` as a distinct FIB
shape/launch family. Its source declares SM86 code generation, hidden size
4096, BF16 destination-passing output, and an explicit FP32 reference contract
with EPS `1e-5`. Reuse only the permitted Torch/GCC14 environment; do not
transfer timing or NCU parameters from problem 001.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34D1 | Build source and run bounded correctness/memcheck smoke with the independent shape/oracle contract | complete — SM86 build, batch 1/2 correctness, and memcheck passed |
| C7-S34D2 | Capture NVBit/control/replay and verify target selection/instruction identity | complete — one target launch, 176 decoded control instructions, deterministic 4562-record replay |
| C7-S34D3 | Collect five independent NCU repetitions after replay and retain tails/units separately | complete — target kernel selected, 7922-cycle median; raw tracer units and tails retained |

**C7-S34D result.** FIB/problem_002 (hidden size 4096) passed the independent
SM86 build/correctness/memcheck smoke, one profiler-delimited NVBit launch,
176-instruction control extraction, and deterministic `l1-l2` replay with
4562 records. Five corrected NCU repetitions selected the target kernel and
measured a 7922-cycle median with 4562 executed instructions, matching the
simulator records. The simulator lower bound is 6686 with dirty-writeback and
coherence tails open; native macro dispatch remains zero. Evidence:
`docs/sm86_c7_sol_candidate_fib002_vertical_v1.json`.

### C7-S34E — L1 pure-ATen candidate source review — active

Select `L1/problem_001_attention_softmax_dropout_value_matmul_backward/cuda_cpp`
as the next distinct SOL class. Its source is a Torch C++ extension using pure
ATen/cuBLAS dispatch (no custom device kernel), with an explicit GQA/BF16
shape and precision contract. The first bounded step is build and correctness
smoke only; multi-kernel tracing/replay may be refused if static module identity
is not stable.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34E1 | Build the pure-ATen extension for SM86 and compare outputs against the declared analytic reference | complete — exact analytic correctness smoke passed |
| C7-S34E2 | Run bounded sanitizer and decide whether a stable native trace contract exists | complete — memcheck passed; profiler-delimited NVBit attempt hung before trace emission |
| C7-S34E3 | If admissible, capture/replay/NCU; otherwise retain explicit pure-ATen/static-module blocker | complete — no native vertical admitted; blocker retained |

**C7-S34E result.** The L1 pure-ATen candidate built and matched its
analytic reference exactly, and Compute Sanitizer reported zero errors. Its
profiler-delimited NVBit attempt then hung before emitting a trace; the stats
file remained header-only after 116 seconds at high CPU and the agent
terminated its own bounded attempt. Because no stable multi-kernel static
crosswalk was produced, replay/NCU were not admitted. Evidence:
`docs/sm86_c7_sol_l1_candidate_blocked_v1.json`.

### C7-S34F — third FIB SOL candidate — complete

Select `FIB/problem_003_fused_add_rmsnorm_h7168/cuda_cpp` as a distinct custom
CUDA candidate. Its source declares SM86 code generation, hidden size 7168,
BF16 destination-passing output, and EPS `1e-6`; it will use the existing
Torch/GCC14 environment but no timing parameters from prior FIB rows.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34F1 | Build and run bounded correctness/memcheck smoke with the independent shape/oracle contract | complete — SM86 build, batch 1/2 correctness, and memcheck passed |
| C7-S34F2 | Capture NVBit/control/replay and verify target identity | complete — one launch, 176 decoded controls, deterministic 7202-record replay |
| C7-S34F3 | Collect five independent target NCU repetitions; retain open tails and unit distinctions | complete — target kernel selected, 10966-cycle median; tails retained |

**C7-S34F result.** FIB/problem_003 (hidden size 7168) passed the SM86
build/correctness/memcheck smoke, one profiler-delimited NVBit launch, 176
instruction control extraction, and deterministic `l1-l2` replay with 7202
records. Five corrected NCU repetitions selected the target kernel and measured
a 10966-cycle median with 7202 instructions matching replay. The simulator
lower bound is 9604 with dirty-writeback/coherence tails open; native macro
dispatch remains zero. Evidence:
`docs/sm86_c7_sol_candidate_fib003_vertical_v1.json`.

### C7-S34G — L2 pure-ATen source blocker — complete

The L2 candidate `problem_001_fused_vision_multihead_attention_with_norms`
was source-reviewed and found to be a pure ATen/cuBLAS chain: four 2D GEMM,
four strided-batched GEMM, and elementwise/reduction kernels, with no custom
CUDA kernel. Given the adjacent L1 pure-ATen bounded NVBit hang, no duplicate
unbounded trace attempt was made. The row remains an explicit source blocker;
no native replay/NCU/completion claim is made. Evidence:
`docs/sm86_c7_sol_l2_pure_aten_source_blocked_v1.json`.

### C7-S34H — CUTLASS TensorOp candidate — complete post-fix L1-L2 vertical

Select `FIB/problem_004_gemm_n128_k2048/cutlass` as a distinct CUTLASS-backed
SM86 candidate. The repository contains `third_party/cutlass`; the source
contract is fp16 `C=A@B^T` with `M` variable, `N=128`, `K=2048`, and SM80
TensorOp code intended to compile on SM86.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34H1 | Build with the pinned Torch/CUDA/GCC14/CUTLASS environment and run bounded GEMM correctness/memcheck smoke | complete — max absolute error 0.031; memcheck passed |
| C7-S34H2 | Capture NVBit/control/replay and verify target kernel identity | complete — NVBit/control capture passed; post-fix L1-L2 replay terminates twice with 31816 records |
| C7-S34H3 | Collect five target NCU repetitions after replay; retain tensor-pipeline/tail limitations | complete — five explicit profiler-gated NCU observations; median GPC 73790, SM-active 72149, 31812 NCU instructions versus 31816 simulator records; four-record unit difference retained |

**C7-S34H result.** The CUTLASS TensorOp candidate passed SM86 build, fp16
GEMM correctness, and memcheck, and produced a 1160-instruction decoded
control sidecar from a real NVBit capture. The zero-fill topology validator
fix now permits the sparse source-bearing cohort sidecar to be a lower-
dimensional component of the aggregate topology; L1-L2 replay terminates
twice. Five explicit profiler-gated NCU runs measured a 73790-cycle median,
72149 SM-active cycles, and 31812 NCU instructions versus 31816 simulator
records. The unit difference and async/writeback/coherence tails remain
explicit. Evidence: `docs/sm86_c7_sol_candidate_fib004_l1_l2_vertical_v2.json`;
the flat fallback and pre-fix blocker remain historical records.

### C7-S34I — SOL source-class census — complete

The remaining 203 SOL baseline rows were classified from source markers without
execution: the corrected census is 72 custom CUDA rows, 14 CUTLASS rows, and
117 pure ATen/cuBLAS-style rows; C7-S34P1 repaired one comment-only
`__global__` false positive. The census defines candidate classes but does not imply compilation,
traceability, replayability, completion, or timing. Evidence:
`docs/sm86_c7_sol_source_class_census_v1.json`.

### C7-S34J — FIB RMSNorm h128 candidate — active

Select `FIB/problem_021_rmsnorm_h128/cuda_cpp` as a small custom CUDA
RMSNorm candidate. Its SM86 contract is BF16 destination-passing RMSNorm at
hidden size 128 with EPS `1e-6`; use the existing Torch/GCC14 environment and
independent FP32 reference.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34J1 | Build and run bounded correctness/memcheck smoke | complete — batch 1/16 correctness and memcheck passed |
| C7-S34J2 | Capture NVBit/control/replay and verify target identity | complete — one launch, 128 decoded controls, deterministic 154-record replay |
| C7-S34J3 | Collect five NCU repetitions after replay; retain tails and units separately | complete — target kernel selected, 4341-cycle median; tails retained |

**C7-S34J result.** FIB/problem_021 (hidden size 128) passed the SM86
build/correctness/memcheck smoke, one profiler-delimited NVBit launch, 128
instruction control extraction, and deterministic `l1-l2` replay with 154
records. Five corrected NCU repetitions selected the target kernel and measured
a 4341-cycle median with 154 instructions matching replay. The simulator lower
bound is 2499 with dirty-writeback/coherence tails open; native macro dispatch
remains zero. Evidence: `docs/sm86_c7_sol_candidate_fib021_vertical_v1.json`.

### C7-S34K — FIB RMSNorm h512 candidate — complete

Select `FIB/problem_022_rmsnorm_h512/cuda_cpp` as a small custom CUDA RMSNorm
candidate with a distinct hidden size 512 and explicit BF16/FP32/EPS contract.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34K1 | Build and run bounded correctness/memcheck smoke | complete — batch 1/16 correctness and memcheck passed |
| C7-S34K2 | Capture NVBit/control/replay and verify target identity | complete — one launch, 128 decoded controls, deterministic 562-record replay |
| C7-S34K3 | Collect five NCU repetitions after replay; retain tails and units separately | complete — target kernel selected, 4433-cycle median; tails retained |

**C7-S34K result.** FIB/problem_022 (hidden size 512) passed the SM86
build/correctness/memcheck smoke, one profiler-delimited NVBit launch, 128
instruction control extraction, and deterministic `l1-l2` replay with 562
records. Five corrected NCU repetitions selected the target kernel and measured
a 4433-cycle median with 562 instructions matching replay. The simulator lower
bound is 2900 with dirty-writeback/coherence tails open; native macro dispatch
remains zero. Evidence: `docs/sm86_c7_sol_candidate_fib022_vertical_v1.json`.

### C7-S34L — FIB RMSNorm h1536 candidate — complete

Select `FIB/problem_023_rmsnorm_h1536/cuda_cpp` as the next distinct custom
CUDA RMSNorm shape. Its source declares SM86 code generation, hidden size 1536,
BF16 destination-passing output, and EPS `1e-6`; use the existing Torch/GCC14
environment and the source's independent FP32 reference.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34L1 | Build and run bounded correctness/memcheck smoke | complete — batch 1/16 correctness and memcheck passed |
| C7-S34L2 | Capture NVBit/control/replay and verify target identity | complete — one launch, 152 decoded controls, deterministic 1442-record replay |
| C7-S34L3 | Collect five NCU repetitions after replay; retain tails and units separately | complete — target kernel selected, 6365-cycle median; tails retained |

**C7-S34L result.** FIB/problem_023 (hidden size 1536) passed the SM86
build/correctness/memcheck smoke, one profiler-delimited NVBit launch, 152
instruction control extraction, and deterministic `l1-l2` replay with 1442
records. Five corrected NCU repetitions selected the target kernel and measured
a 6365-cycle median with 1442 instructions matching replay. The simulator lower
bound is 5130 with dirty-writeback and coherence tails open; raw tracer units,
the enhanced JIT parser limitation, and zero native macro dispatch remain
explicit. Evidence: `docs/sm86_c7_sol_candidate_fib023_vertical_v1.json`.

### C7-S34M — FIB RMSNorm h2048 candidate — complete

Source-review and, if admissible, vertically capture
`FIB/problem_024_rmsnorm_h2048/cuda_cpp`. This is a new hidden-size shape in
the same custom-CUDA class, not a transfer of timing or NCU values from prior
rows. The completion bar is source/shape/oracle verification, an SM86
build/correctness/memcheck smoke, then one profiler-delimited NVBit/control
capture, deterministic replay, and five target-kernel NCU repetitions. Any
unstable trace or replay refusal remains an explicit blocker/fallback and does
not count as a positive vertical.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34M1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — SM86 marker, hidden-size/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34M2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, batches 1/16 correctness, and memcheck passed with zero errors |
| C7-S34M3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one launch, 152 decoded controls, 1762-record replay, five target NCU repetitions |

**C7-S34M result.** FIB/problem_024 (hidden size 2048) passed source review,
SM86 build, batch-1/16 correctness, and Compute Sanitizer memcheck. The
profiler-delimited capture accepted one launch and the independent sidecar
decoded 152 static instructions. Deterministic `l1-l2` replay produced 1762
records/instructions with a 6090-cycle lower frontier. Five corrected target
NCU repetitions measured `[7329, 7413, 7350, 7448, 7503]` cycles (median
7413) and 1762 instructions in every run. Dirty-writeback and coherence tails,
the JIT parser limitation, raw tracer units, and zero native macro dispatch
remain explicit. Evidence: `docs/sm86_c7_sol_candidate_fib024_vertical_v1.json`.

### C7-S34N — FIB RMSNorm h4096 candidate — complete

Source-review and, if admissible, vertically capture
`FIB/problem_025_rmsnorm_h4096/cuda_cpp`. This is a new hidden-size shape in
the custom-CUDA class. It must establish its own source/shape/oracle contract,
then pass a fresh build/correctness/memcheck smoke before any bounded
NVBit/control/replay/NCU attempt. No timing or NCU value transfers from
earlier FIB rows; an unstable trace or replay refusal remains a blocker or
flat fallback rather than a positive vertical.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34N1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — SM86 marker, hidden-size/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34N2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, batches 1/16 correctness, and memcheck passed with zero errors |
| C7-S34N3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one launch, 152 decoded controls, 3042-record replay, five target NCU repetitions |

**C7-S34N result.** FIB/problem_025 (hidden size 4096) passed source review,
SM86 build, batch-1/16 correctness, and Compute Sanitizer memcheck. The
profiler-delimited capture accepted one launch and the independent sidecar
decoded 152 static instructions. Deterministic `l1-l2` replay produced 3042
records/instructions with a 9930-cycle lower frontier. Five corrected target
NCU repetitions measured `[11292, 11083, 11317, 11024, 11164]` cycles (median
11164) and 3042 instructions in every run. Dirty-writeback and coherence tails,
the JIT parser limitation, raw tracer units, and zero native macro dispatch
remain explicit. Evidence: `docs/sm86_c7_sol_candidate_fib025_vertical_v1.json`.

### C7-S34O — FIB RMSNorm h7168 candidate — complete

Source-review and, if admissible, vertically capture
`FIB/problem_026_rmsnorm_h7168/cuda_cpp`. This is the final listed FIB
RMSNorm shape in the source inventory and a new hidden-size class point. It
must establish its own source/shape/oracle contract and fresh build/sanitizer
smoke before any bounded NVBit/control/replay/NCU attempt. No timing or NCU
value transfers from earlier rows; an unstable trace or replay refusal remains
a blocker or flat fallback rather than a positive vertical.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34O1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — SM86 marker, hidden-size/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34O2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, batches 1/16 correctness, and memcheck passed with zero errors |
| C7-S34O3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one launch, 152 decoded controls, 5426-record replay, five target NCU repetitions |

**C7-S34O result.** FIB/problem_026 (hidden size 7168) passed source review,
SM86 build, batch-1/16 correctness, and Compute Sanitizer memcheck. The
profiler-delimited capture accepted one launch and the independent sidecar
decoded 152 static instructions. Deterministic `l1-l2` replay produced 5426
records/instructions with a 9056-cycle lower frontier. Five corrected target
NCU repetitions measured `[10718, 10531, 10487, 10749, 10537]` cycles (median
10537) and 5426 instructions in every run. Dirty-writeback and coherence tails,
the JIT parser limitation, raw tracer units, and zero native macro dispatch
remain explicit. Evidence: `docs/sm86_c7_sol_candidate_fib026_vertical_v1.json`.

### C7-S34P — L1 source-class correction — complete

The first selected L1 row,
`L1/problem_002_vae_conv3x3_groupnorm_silu_residual_fused/cuda_cpp`, was
source-reviewed before execution. Its `kernel.cu` is a stub translation unit;
the implementation delegates to ATen `conv2d`, `group_norm`, and `silu`, and
the only `__global__` token occurs in a comment (also copied into
`solution.json`). The prior marker-only census therefore over-counted custom
CUDA by one. This row is corrected to pure ATen and retained as a source
blocker; no duplicate unbounded NVBit attempt is made after the adjacent L1
pure-ATen blocker.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34P1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — source class corrected to pure ATen; shape/reference contract retained |
| C7-S34P2 | Build and run bounded correctness/memcheck smoke | complete scoped — not repeated; adjacent pure-ATen build/memcheck and trace blocker establish the bounded disposition |
| C7-S34P3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — no native vertical admitted; source blocker retained |

Evidence: `docs/sm86_c7_sol_l1_problem002_source_class_correction_v1.json`.

### C7-S34Q — L1 Hyena depthwise custom-CUDA candidate — complete

Select `L1/problem_006_hyena_depthwise_conv1d_split_gate/cuda_cpp`, whose
source contains an actual custom CUDA kernel rather than a marker-only ATen
stub. Verify its independent shape/oracle contract and permitted runtime before
building. No timing or NCU values transfer from FIB or prior L1 rows.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34Q1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — two actual custom kernels, shape/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34Q2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, batch/sequence correctness, and memcheck passed with zero errors |
| C7-S34Q3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — two launches, 144 decoded controls, 679936-record replay, five per-kernel NCU repetitions |

**C7-S34Q result.** L1/problem_006 Hyena depthwise split/gate passed source
review, SM86 build, representative `[1,512]`/`[4,2048]` correctness, and
Compute Sanitizer memcheck. The bounded capture retained both actual launches;
the independent sidecar decoded 72 instructions per kernel. Deterministic
`l1-l2` replay produced 438272 records for phase 1 and 241664 for phase 2
(679936 total). Five corrected NCU repetitions selected both kernels and
reported matching per-kernel instruction counts. Per-kernel lower frontiers
are 5242 and 4359; dirty-writeback/coherence tails, parser limitation, raw
tracer units, and zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p006_vertical_v1.json`.

### C7-S34R — L1 rotary-position custom-CUDA candidate — complete

Source-review and, if admissible, vertically capture
`L1/problem_011_rotary_position_embedding/cuda_cpp`. This row changes from the
two-launch Hyena family to a single custom RoPE kernel with custom position/
frequency inputs. Establish its own source/shape/oracle contract and fresh
build/sanitizer smoke before capture. No timing or NCU values transfer from
prior rows; any custom-input or trace-selection failure remains explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34R1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — custom RoPE kernel, custom-input contract, and permitted Torch/GCC14 runtime verified |
| C7-S34R2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, custom-input RoPE correctness, and memcheck passed with zero errors |
| C7-S34R3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one launch, 56 decoded controls, 22016-record replay, five target NCU repetitions |

**C7-S34R result.** L1/problem_011 RoPE passed source review, SM86 build,
custom-input correctness at `[1,256]`/`[4,1024]`, and Compute Sanitizer
memcheck. The profiler-delimited capture accepted one launch and the
independent sidecar decoded 56 static instructions. Deterministic `l1-l2`
replay produced 22016 records/instructions with a 2000-cycle lower frontier.
Five corrected target NCU repetitions measured `[4425, 4466, 4432, 4429,
4439]` cycles (median 4432) with 22016 instructions matching replay.
Dirty-writeback/coherence tails, parser limitation, raw tracer units, and zero
native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p011_vertical_v1.json`.

### C7-S34S — L1 fused cos/sin custom-CUDA candidate — complete

Source-review and, if admissible, vertically capture
`L1/problem_012_fused_cos_sin_embedding_generation/cuda_cpp`. This continues
the L1 custom-kernel population with a separate trigonometric embedding
kernel. Establish its own source/custom-input/oracle contract and fresh
build/sanitizer smoke before capture. No timing or NCU values transfer from
prior rows; any input or trace-selection failure remains explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34S1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — custom cos/sin kernel, shape/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34S2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, scaled cos/sin correctness, and memcheck passed with zero errors |
| C7-S34S3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one launch, 40 decoded controls, 61440-record replay, five target NCU repetitions |

**C7-S34S result.** L1/problem_012 fused cos/sin passed source review, SM86
build, scaled correctness at `[1,512]`/`[4,1024]`, and Compute Sanitizer
memcheck. The profiler-delimited capture accepted one launch and the
independent sidecar decoded 40 static instructions. Deterministic `l1-l2`
replay produced 61440 records/instructions with a 2023-cycle lower frontier.
Five corrected target NCU repetitions measured `[4200, 4160, 4158, 4172,
4176]` cycles (median 4172) with 61440 instructions matching replay.
Dirty-writeback/coherence tails, parser limitation, raw tracer units, and zero
native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p012_vertical_v1.json`.

### C7-S34T — L1 fused residual RMSNorm-backward custom-CUDA candidate — complete

Source-review and, if admissible, vertically capture
`L1/problem_013_fused_residual_rms_norm_backward/cuda_cpp`. This continues the
L1 custom-kernel population with a backward residual/RMSNorm workload. Establish
its own source/oracle contract and fresh build/sanitizer smoke before capture.
No timing or NCU values transfer from prior rows; custom-input or replay
failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34T1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — two custom kernels, backward shape/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34T2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, backward-gradient correctness, and memcheck passed with zero errors |
| C7-S34T3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — two launches, 872 decoded controls, 2457600-record replay, five per-kernel NCU repetitions |

**C7-S34T result.** L1/problem_013 fused residual RMSNorm backward passed
source review, SM86 build, representative `[1,256]`/`[2,512]` correctness,
and Compute Sanitizer memcheck. Both launches were retained; the independent
sidecar decoded 552 `grad_x` and 320 `grad_weight` instructions. Replay
produced 985600 and 1472000 records. Five NCU repetitions matched both
instruction counts. The `grad_weight` lower frontier (113069) exceeds its NCU
median (34972), so this is a retained lower-bound falsifier, not a fit; tails,
parser limitation, raw tracer units, and zero native macro dispatch remain
explicit. Evidence: `docs/sm86_c7_sol_l1p013_vertical_v1.json`.

### C7-S34U — L1 inverse-frequency custom-CUDA candidate — complete

Source-review and, if admissible, vertically capture
`L1/problem_016_rope_inverse_frequency_computation/cuda_cpp`. Establish its
own source/input/oracle contract and fresh build/sanitizer smoke before
capture. No timing or NCU values transfer from prior rows; custom-input or
replay failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34U1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — custom inverse-frequency kernel, scalar/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34U2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, seven scalar oracle values, and memcheck passed with zero errors |
| C7-S34U3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one launch, 24 decoded controls, 30-record replay, five target NCU repetitions |

**C7-S34U result.** L1/problem_016 inverse-frequency computation passed source
review, SM86 build, all seven scalar values, and Compute Sanitizer memcheck. The
profiler-delimited capture accepted one launch and the independent sidecar
decoded 24 static instructions. Deterministic replay produced 30
records/instructions with a 1163-cycle lower frontier. Five NCU repetitions
measured `[2975, 2999, 2995, 2982, 3119]` cycles (median 2995) with 30
instructions matching replay. Tails, parser limitation, raw tracer units, and
zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p016_vertical_v1.json`.

### C7-S34V — L1 fused RoPE/QK-norm/cache-update custom-CUDA candidate — complete

Source-review and, if admissible, vertically capture
`L1/problem_018_fused_rope_with_qk_norm_and_kv_cache_update/cuda_cpp`. Establish
its own source/input/oracle contract and fresh build/sanitizer smoke before
capture. No timing or NCU values transfer from prior rows; custom-input or
replay failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34V1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — four custom kernels, cache/shape/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34V2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, cache/RoPE correctness, and memcheck passed with zero errors |
| C7-S34V3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — four launches, 304 decoded controls, 48592-record replay, five per-launch NCU repetitions |

**C7-S34V result.** L1/problem_018 fused RoPE/QK-norm/cache update passed
source review, SM86 build, bounded `[B=1,S=1,cache_len=0]` correctness, and
Compute Sanitizer. The capture retained all four kernels; the sidecar decoded
304 instructions. Replay and NCU matched per-launch instructions at
42432/3536/1312/1312. The 1 GiB-class cache clone is explicit; per-launch
tails, parser limitation, raw tracer units, and zero native macro dispatch
remain open. Evidence: `docs/sm86_c7_sol_l1p018_vertical_v1.json`.

### C7-S34W — L1 Vision 3D RoPE-backward custom-CUDA candidate — complete

Source-review and, if admissible, vertically capture
`L1/problem_019_vision_3d_rotary_embedding_with_spatial_merge_indexing_backward/cuda_cpp`.
Establish its own source/input/oracle contract and fresh build/sanitizer smoke
before capture. No timing or NCU values transfer from prior rows; custom-input
or replay failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34W1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — two custom kernels, Vision RoPE backward contract, and permitted Torch/GCC14 runtime verified |
| C7-S34W2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, Vision RoPE backward correctness, and memcheck passed with zero errors |
| C7-S34W3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — three launches, 336 decoded controls, 764928-record replay, five per-launch NCU repetitions |

**C7-S34W result.** L1/problem_019 Vision 3D RoPE backward passed source
review, SM86 build, FP32 backward correctness at sequence lengths 128/449,
and Compute Sanitizer memcheck. Three launches were retained; the sidecar
decoded 336 instructions. Replay and NCU matched per-launch instruction counts
at 278528/278528/207872. Per-launch tails, parser limitation, raw tracer
units, and zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p019_vertical_v1.json`.

### C7-S34X — L1 Vision MLP/GELU CUTLASS candidate — complete scoped blocker

Source-review and, if admissible, vertically capture
`L1/problem_022_vision_mlp_gelu_backward/cuda_cpp`. Establish its own
source/input/oracle contract and fresh build/sanitizer smoke before capture.
No timing or NCU values transfer from prior rows; custom-input or replay
failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34X1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS/SM80 TensorOp source, MLP/GELU contract, and permitted Torch/GCC14 runtime verified |
| C7-S34X2 | Build and run bounded correctness/memcheck smoke | complete scoped — CUTLASS include path fixed; build and memcheck passed, but independent FP32 correctness failed materially |
| C7-S34X3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — no native vertical admitted; correctness blocker retained |

**C7-S34X result.** The CUTLASS row required the repository include path
`third_party/cutlass/include`, then built and ran without memcheck errors. The
independent FP32 smoke nevertheless failed materially on the TF32 TensorOp
GEMM-derived gradients (for example, grad-hidden max errors 3.26 and 2.79 on
S=64/449). No NVBit/replay/NCU vertical was admitted; this is a correctness
blocker, not a timing result. Evidence:
`docs/sm86_c7_sol_l1p022_smoke_v1.json`.

### C7-S34Y — L1 problem024 source-class correction — complete

Source-review and, if admissible, vertically capture
`L1/problem_024_vision_rotary_position_embedding_generation_backward/cuda_cpp`.
Establish its own source/input/oracle contract and fresh build/sanitizer smoke
before capture. No timing or NCU values transfer from prior rows; custom-input
or correctness failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34Y1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — marker-only custom-CUDA false positive corrected to pure ATen |
| C7-S34Y2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34Y3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem024_source_class_correction_v1.json`.

### C7-S34Z — L1 vectorized GELU custom-CUDA candidate — complete

Source-review and, if admissible, vertically capture
`L1/problem_025_video_latent_gelu_activation/cuda_cpp`. Establish its own
source/input/oracle contract and fresh build/sanitizer smoke before capture.
No timing or NCU values transfer from prior rows; failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34Z1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — vectorized/tail GELU kernels, shape/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34Z2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, vectorized/tail GELU correctness, and memcheck passed with zero errors |
| C7-S34Z3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one live vector launch, 88 decoded controls, 368640-record replay, five target NCU repetitions; tail kernel remains source-present/unreached |

**C7-S34Z result.** L1/problem_025 video-latent GELU passed source review,
SM86 build, vectorized/tail correctness smoke, and memcheck. The live vector
launch replayed 368640 records and matched five NCU instruction counts. The
scalar tail kernel was not reachable for fixed intermediate size 4096 and is
retained as an explicit uncaptured source path; tails, parser limitation, raw
tracer units, and zero native macro dispatch remain open. Evidence:
`docs/sm86_c7_sol_l1p025_vertical_v1.json`.

### C7-S34AA — L1 problem029 source-class correction — complete

Source-review and, if admissible, vertically capture
`L1/problem_029_mamba_conv1d_with_gating/cuda_cpp`. Establish its own
source/input/oracle contract and fresh build/sanitizer smoke before capture.
No timing or NCU values transfer from prior rows; failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AA1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — marker-only custom-CUDA false positive corrected to pure ATen |
| C7-S34AA2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34AA3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem029_source_class_correction_v1.json`.

### C7-S34AB — L1 post-norm residual custom-CUDA candidate — complete

Source-review and, if admissible, vertically capture
`L1/problem_033_post_norm_residual/cuda_cpp`. Establish its own source/input/
oracle contract and fresh build/sanitizer smoke before capture. No timing or
NCU values transfer from prior rows; failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AB1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — custom post-norm RMSNorm kernel, shape/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34AB2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, post-norm residual correctness, and memcheck passed with zero errors |
| C7-S34AB3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one launch, 160 decoded controls, 440576-record replay, five target NCU repetitions |

**C7-S34AB result.** L1/problem_033 post-norm residual passed source review,
SM86 build, BF16 correctness at `[1,128]`/`[2,512]`, and memcheck. The live
capture decoded 160 controls; replay and NCU matched at 440576 instructions.
Lower frontier remains open-tail; parser limitation, raw tracer units, and zero
native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p033_vertical_v1.json`.

### C7-S34AC — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_034_flux_multi_axis_rope_frequency_computation/cuda_cpp`. Establish
its own source/input/oracle contract and fresh build/sanitizer smoke before
capture. No timing or NCU values transfer from prior rows; failures remain
explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AC1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — actual custom kernel; FP32 16/56/56-axis repeat-interleave agrees with the retained upstream reference; 16 declared rows |
| C7-S34AC2 | Build and run bounded correctness/memcheck smoke | complete — all 16 declared shapes × 3 deterministic seeds (48 rows) passed at unchanged 1e-5 tolerances; memcheck passed |
| C7-S34AC3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — row-0 launch, 56 decoded controls, 160517-record replay, five target NCU repetitions |

**C7-S34AC result.** L1/problem_034 Flux multi-axis RoPE passed all-shape
screening (48 rows), memcheck, and a row-0 live vertical. Replay and NCU
matched at 160517 instructions; the lower frontier is 2107 cycles versus an
NCU median of 4659. Tails, parser limitation, raw tracer units, and zero native
macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p034_vertical_v1.json`.

### C7-S34AD — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_035_flux_ada_layer_norm_zero_modulation_extraction/cuda_cpp`.
Establish its own source/input/oracle contract and fresh build/sanitizer smoke
before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AD1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — custom AdaLN modulation kernel, 3072/18432 chunk contract, and permitted Torch/GCC14 runtime verified |
| C7-S34AD2 | Build and run bounded correctness/memcheck smoke | complete — SM86 build, AdaLN matmul/chunk correctness at B=1/16, and memcheck passed with zero errors |
| C7-S34AD3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one launch, 232 decoded controls, 8607744-record replay, five target NCU repetitions; unsupported-memory-path detail retained |

**C7-S34AD result.** L1/problem_035 AdaLN modulation passed all-shape
screening, memcheck, and a large B=1 live vertical. Replay and NCU matched at
8607744 instructions; the NCU median was 341471 cycles versus a 319277 lower
frontier. The tracer reports 18432 typed unsupported-memory-path records for
the indirect chunk store; this remains an explicit limitation. Evidence:
`docs/sm86_c7_sol_l1p035_vertical_v1.json`.

### C7-S34AE — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_036_flux_output_norm_projection_chain/cuda_cpp`. Establish its own
source/input/oracle contract and fresh build/sanitizer smoke before capture.
No timing or NCU values transfer from prior rows; failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AE1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — four-stage Flux norm/modulation/projection chain, shape/oracle contract, and permitted Torch/GCC14 runtime verified |
| C7-S34AE2 | Build and run bounded correctness/memcheck smoke | complete scoped — build and memcheck passed, but B=2/S=293 failed declared numerical tolerance |
| C7-S34AE3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — no native vertical admitted; correctness blocker retained |

**C7-S34AE result.** The Flux output norm/projection chain built and passed
memcheck, but the independent FP32 smoke failed the declared tolerance on
B=2/S=293 (max error `0.0045166` versus absolute `0.0031`). No NVBit/replay/
NCU vertical was admitted; this is a correctness blocker, not a timing result.
Evidence: `docs/sm86_c7_sol_l1p036_smoke_v1.json`.

### C7-S34AF — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_038_flux_multi_head_rmsnorm_qk/cuda_cpp`. Establish its own
source/input/oracle contract and fresh build/sanitizer smoke before capture.
No timing or NCU values transfer from prior rows; failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AF1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — custom per-head RMSNorm Q/K kernel, 48×128 contract, and permitted Torch/GCC14 runtime verified |
| C7-S34AF2 | Build and run bounded correctness/memcheck smoke | complete — five declared Q/K shape rows passed official 1e-5 tolerances; memcheck passed |
| C7-S34AF3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pinned CUDA 13.2/GCC14 two-launch capture, 64 target decoded controls, 5013504-record replay, five per-kernel NCU repetitions |

**C7-S34AF result.** L1/problem_038 per-head Q/K RMSNorm passed the full
declared shape screen, pinned CUDA 13.2/GCC14 rebuild, memcheck, and two-launch
live capture. Query/key replay and NCU matched at 2506752 instructions each;
NCU medians were 23102 and 22036 cycles. Completion tails, parser limitation,
raw tracer units, and zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p038_vertical_v1.json`.

### C7-S34AG — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_039_adaptive_instance_normalization_2d_backward/cuda_cpp`. Establish
its own source/input/oracle contract and fresh build/sanitizer smoke before
capture. No timing or NCU values transfer from prior rows; failures remain
explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AG1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — marker-only custom-CUDA false positive corrected to pure ATen |
| C7-S34AG2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34AG3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem039_source_class_correction_v1.json`.

### C7-S34AH — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_040_conv2d_residual_block/cuda_cpp`. Establish its own source/input/
oracle contract and fresh build/sanitizer smoke before capture. No timing or
NCU values transfer from prior rows; failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AH1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — two actual custom conv kernels, 3/32/3 residual contract, and permitted Torch/GCC14 runtime verified |
| C7-S34AH2 | Build and run bounded correctness/memcheck smoke | complete — five declared conv/residual shapes passed; memcheck passed |
| C7-S34AH3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pinned two-launch capture, 3024 decoded controls, 10155520-record replay, five per-kernel NCU repetitions |

**C7-S34AH result.** L1/problem_040 Conv2d residual passed all five declared
shapes and memcheck. The live capture retained both convolution launches; replay
and NCU matched per-launch instructions at 6112768/4042752. NCU medians were
29338/43676 cycles. Completion tails, parser limitation, raw tracer units, and
zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p040_vertical_v1.json`.

### C7-S34AI — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_041_strided_conv2d_downsampling/cuda_cpp`. Establish its own
source/input/oracle contract and fresh build/sanitizer smoke before capture.
No timing or NCU values transfer from prior rows; failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AI1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — marker-only custom-CUDA false positive corrected to pure ATen |
| C7-S34AI2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34AI3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem041_source_class_correction_v1.json`.

### C7-S34AJ — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_042_moe_expert_load_balancing_and_token_capacity_backward/cuda_cpp`.
Establish its own source/input/oracle contract and fresh build/sanitizer smoke
before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AJ1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — marker-only custom-CUDA false positive corrected to pure ATen |
| C7-S34AJ2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34AJ3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem042_source_class_correction_v1.json`.

### C7-S34AK — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_043_mla_fused_qkv_rope_split/cuda_cpp`. Establish its own
source/input/oracle contract and fresh build/sanitizer smoke before capture.
No timing or NCU values transfer from prior rows; failures remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AK1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — marker-only custom-CUDA false positive corrected to pure ATen |
| C7-S34AK2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34AK3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem043_source_class_correction_v1.json`.

### C7-S34AL — L1 custom-CUDA row — complete

Source-review and, if admissible, vertically capture
`L1/problem_046_attention_softmax_with_softcapping_and_dropout/cuda_cpp`.
Establish its own source/input/oracle contract and fresh build/sanitizer smoke
before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AL1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine custom kernel; BF16 softcap/softmax contract and 16 declared rows recorded |
| C7-S34AL2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 16 declared rows × 3 seeds passed; representative memcheck zero errors |
| C7-S34AL3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one profiler-gated launch, independent 248-instruction sidecar (232 target + 16 inert CUB helper), deterministic replay, and five NCU repetitions; instruction crosswalk exact; retirement tails retained |

Evidence: `docs/sm86_c7_sol_l1p046_source_v1.json`, `docs/sm86_c7_sol_l1p046_smoke_v1.json`, and `docs/sm86_c7_sol_l1p046_vertical_v1.json`.

### C7-S34AM — L1 custom-CUDA-plus-cuBLAS row — complete with bounded capture blocker

Source-review and, if admissible, vertically capture
`L1/problem_048_fused_gate_up_projection_with_swiglu/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AM1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine custom elementwise kernel plus two vendor matmul launches; BF16/GELU contract and 16 declared rows recorded |
| C7-S34AM2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 16 declared rows × 3 seeds passed after the tanh-tail numerical repair (multiply-only hypothesis refuted); representative memcheck zero errors |
| C7-S34AM3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — two bounded NVBit attempts fail before trace emission on vendor `HMNMX2.BF16_V2` static-SASS decode (including `SKIP_ENHANCED_TRACER=1`); no substitute vertical admitted |

Evidence: `docs/sm86_c7_sol_l1p048_source_v1.json`, `docs/sm86_c7_sol_l1p048_smoke_v1.json`, and `docs/sm86_c7_sol_l1p048_capture_blocked_v1.json`.

### C7-S34AN — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_052_altup_hidden_state_collapse_with_magnitude_normalization/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AN1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — pure-ATen source correction; no custom launcher or native vertical admitted |
| C7-S34AN2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34AN3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem052_source_class_correction_v1.json`.

### C7-S34AO — L1 custom-CUDA row — complete

Source-review and, if admissible, vertically capture
`L1/problem_053_gaussian_topk_sparse_activation/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AO1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine BF16x2 custom row kernel; FP32 statistics/host-ndtri contract and 12 declared rows recorded |
| C7-S34AO2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 12 declared rows × 3 seeds passed (36 rows; max absolute error 0.03125); representative memcheck zero errors |
| C7-S34AO3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one profiler-gated launch, independent 488-instruction sidecar (472 target + 16 inert CUB helper), byte-identical replay repeat, and five application-replay NCU repetitions; instruction crosswalk exact; retirement tails retained |

Evidence: `docs/sm86_c7_sol_l1p053_source_v1.json`, `docs/sm86_c7_sol_l1p053_smoke_v1.json`, and `docs/sm86_c7_sol_l1p053_vertical_v1.json`.

### C7-S34AP — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_056_mamba_ssm_dt_projection_softplus_clamp_backward/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AP1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine per-head custom backward kernel; BF16/FP32 clamp-sigmoid-sum contract and 16 declared rows recorded |
| C7-S34AP2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 16 declared rows × 3 seeds passed (48 rows; max grad_dt error 0.0009765625); representative memcheck zero errors |
| C7-S34AP3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one profiler-gated launch, independent 224-instruction sidecar (208 target + 16 inert CUB helper), byte-identical replay repeat, and five application-replay NCU repetitions; instruction crosswalk exact; retirement tails retained |

Evidence: `docs/sm86_c7_sol_l1p056_source_v1.json`, `docs/sm86_c7_sol_l1p056_smoke_v1.json`, and `docs/sm86_c7_sol_l1p056_vertical_v1.json`.

### C7-S34AQ — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_057_mtp_shifted_embedding_with_dual_rms_norm_fusion/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AQ1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — pure-ATen source correction; no custom launcher or native vertical admitted |
| C7-S34AQ2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34AQ3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem057_source_class_correction_v1.json`.

### C7-S34AR — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_058_moe_expert_token_radix_sort_with_prefix_sum/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AR1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — pure-ATen source correction; no custom launcher or native vertical admitted |
| C7-S34AR2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34AR3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem058_source_class_correction_v1.json`.

### C7-S34AS — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_059_moe_group_score_aggregation_and_masking/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AS1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine FP32 custom row kernel; top-2/top-4 grouping and -inf mask contract with 16 declared rows recorded |
| C7-S34AS2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 16 declared rows × 3 seeds passed (48 rows; negative-infinity exception explicit); representative memcheck zero errors |
| C7-S34AS3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — checked profiler-gated launch, independent 312-instruction sidecar (296 target + 16 inert CUB helper), byte-identical replay repeat, and five application-replay NCU repetitions; instruction crosswalk exact; retirement tails retained |

Evidence: `docs/sm86_c7_sol_l1p059_source_v1.json`, `docs/sm86_c7_sol_l1p059_smoke_v1.json`, and `docs/sm86_c7_sol_l1p059_vertical_v1.json`.

### C7-S34AT — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_061_tanh_gated_residual_add_backward/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AT1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine two-kernel BF16x2 custom backward path; FP32 tanh/reduction contract and 16 declared rows recorded |
| C7-S34AT2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 16 declared rows × 3 seeds passed (48 rows; max grad_hidden error 0.0625); representative memcheck zero errors |
| C7-S34AT3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — two custom launches after identity/scalar source repair, independent 400-instruction sidecar (216 + 168 target + 16 inert CUB helper), byte-identical replay repeat, and five application-replay NCU repetitions per kernel; instruction crosswalk exact; retirement tails retained |

Evidence: `docs/sm86_c7_sol_l1p061_source_v1.json`, `docs/sm86_c7_sol_l1p061_smoke_v1.json`, and `docs/sm86_c7_sol_l1p061_vertical_v1.json`.

### C7-S34AU — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_062_kv_cache_update_with_rope_backward/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AU1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — pure-ATen source correction; no custom launcher or native vertical admitted |
| C7-S34AU2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34AU3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem062_source_class_correction_v1.json`.

### C7-S34AV — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_064_latent_kv_expansion_with_split/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AV1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — pure-ATen source correction; no custom launcher or native vertical admitted |
| C7-S34AV2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34AV3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen source blocker retained; no native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1_problem064_source_class_correction_v1.json`.

### C7-S34AW — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_066_masked_softmax_with_attention_dropout_backward/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AW1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine FP32 custom row kernel; dropout/softmax backward contract and memory-bounded smoke subset recorded |
| C7-S34AW2 | Build and run bounded correctness/memcheck smoke | complete historical raw attempt — custom-input diagnostic retained; not the official SOL disposition |
| C7-S34AW3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | superseded by C7-S34BB — official criterion audit passes all 16 rows; corrected live vertical is now admissible |

Evidence: `docs/sm86_c7_sol_l1p066_source_v1.json`, `docs/sm86_c7_sol_l1p066_smoke_v1.json`, and `docs/sm86_c7_sol_l1p066_correctness_blocked_v1.json`.

### C7-S34AX — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_068_fused_rope_application_backward/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AX1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine two-kernel FP32 RoPE backward path; head/reduction contract and 14 declared rows recorded |
| C7-S34AX2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 14 declared rows × 3 seeds passed (48 rows; max absolute error 0.0000190735); representative memcheck zero errors |
| C7-S34AX3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — three custom launches, independent 768-instruction sidecar (40 + 712 target + 16 inert CUB helper), byte-identical replay repeat, and five application-replay NCU repetitions per launch; instruction crosswalk exact; retirement tails retained |

Evidence: `docs/sm86_c7_sol_l1p068_source_v1.json`, `docs/sm86_c7_sol_l1p068_smoke_v1.json`, and `docs/sm86_c7_sol_l1p068_vertical_v1.json`.

### C7-S34AY — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_069_rms_norm/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AY1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine 1024-thread BF16x2 RMSNorm custom row kernel; 16 declared rows recorded |
| C7-S34AY2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 16 declared rows × 3 seeds passed (48 rows; max absolute error 0.125); representative memcheck zero errors |
| C7-S34AY3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one custom launch, independent 192-instruction sidecar (176 target + 16 inert CUB helper), byte-identical replay repeat, and five application-replay NCU repetitions; instruction crosswalk exact; retirement tails retained |

Evidence: `docs/sm86_c7_sol_l1p069_source_v1.json`, `docs/sm86_c7_sol_l1p069_smoke_v1.json`, and `docs/sm86_c7_sol_l1p069_vertical_v1.json`.

### C7-S34AZ — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_071_kv_cache_update_with_rope/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34AZ1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine variable-head-dimension custom RoPE/cache update kernel; 16 declared rows recorded |
| C7-S34AZ2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 16 declared rows × 3 seeds passed (48 rows; exact output equality); representative memcheck zero errors |
| C7-S34AZ3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one custom launch, independent 152-instruction sidecar (136 target + 16 inert CUB helper), byte-identical replay repeat, and five application-replay NCU repetitions; instruction crosswalk exact; retirement tails retained |

Evidence: `docs/sm86_c7_sol_l1p071_source_v1.json`, `docs/sm86_c7_sol_l1p071_smoke_v1.json`, and `docs/sm86_c7_sol_l1p071_vertical_v1.json`.

### C7-S34BA — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BA1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — custom FP16 RMSNorm plus vendor projections/layout copies; 16 workload rows, packaged compile options, generator and 0.99 matched-ratio criterion inspected |
| C7-S34BA2 | Build and run bounded correctness/memcheck smoke | complete — full 16-row × 3-seed SOL generator/reference screen passed official `required_matched_ratio=0.99` criterion; strict all-element diagnostics show small mismatches; representative memcheck zero errors |
| C7-S34BA3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — full mixed-module capture fails before trace emission on vendor/ATen `DEPBAR.LE SB0 …`; no RMSNorm-only substitute or native vertical admitted |

Evidence: `docs/sm86_c7_sol_l1p073_source_v1.json`, `docs/sm86_c7_sol_l1p073_smoke_v1.json`, and `docs/sm86_c7_sol_l1p073_capture_blocked_v1.json`.

The full `run` chain is the target, not a custom-RMSNorm-only substitute.
Keep mixed module controls and any host/device-copy gap explicit. Use CUDA
13.2/GCC14 with the **packaged** compile options (no extra `--use_fast_math`).
The benchmark's `required_matched_ratio=0.99` and heuristic input generation
are load-bearing; an all-elements diagnostic is not its pass/fail rule.

After BA: **C7-S34BB** reconciles the concrete input/criterion mismatch in
the earlier problem066 blocker; **C7-S34BC** resumes remaining-population
source/execution work. Neither a source reclassification nor a recorded
failure closes an unexecuted workload or any physical completion mechanism.

### C7-S34BB — correctness-contract reconciliation — active

Reconcile the earlier problem066 disposition against the actual SOL criterion
and generator. The retained contract allows `required_matched_ratio=0.99`;
the raw screen's all-element interpretation was stricter than the benchmark
criterion. Recompute the disposition from the actual reference/generator,
without NCU inputs or tolerance widening, then run one focused differential if
the result changes.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BB1 | Recompute disposition from retained rows and actual criterion | complete — official generator/reference/criterion: 16/16 rows pass, minimum matched ratio 1.0 |
| C7-S34BB2 | Run one focused corrected differential if disposition changes | complete — no NCU input; audit itself corrected the prior disposition, so a simulator differential is not applicable |
| C7-S34BB3 | Synchronize blocker/vertical claims and select next population seam | complete — problem066 correctness blocker superseded and corrected one-launch vertical retained; next population seam selected |

Evidence: `docs/sm86_c7_sol_l1p066_correctness_recheck_v1.json`; corrected vertical: `docs/sm86_c7_sol_l1p066_vertical_v1.json`; superseded raw attempt remains `docs/sm86_c7_sol_l1p066_smoke_v1.json`.

### C7-S34BC — next population/source-execution seam — active

Continue the remaining L1 custom-CUDA/source population after problem073 and
the corrected problem066 vertical. Source-review the next declared custom row
one at a time; preserve pure-ATen corrections and bounded vendor/parser or
correctness blockers in the denominator. No timing or NCU values transfer.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BC1 | Reconcile next source-class and dependency contract | complete — problem075 is pure ATen; source-class correction retained, no build admitted |
| C7-S34BC2 | Build/correctness/memcheck or retain a concrete blocker | complete scoped — no build admitted after source correction |
| C7-S34BC3 | Admit full-chain capture/replay/NCU only after correctness | complete — pure-ATen blocker retained; next genuine custom row selected |

Evidence: `docs/sm86_c7_sol_l1_problem075_source_class_correction_v1.json`.

### C7-S34BD — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_078_group_norm_fusion/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BD1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine FP32 32-group custom GroupNorm kernel; 16 declared rows recorded |
| C7-S34BD2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 16 declared rows × 3 seeds passed (48 rows; max absolute error 2.861e-6); representative memcheck zero errors |
| C7-S34BD3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — one custom launch, independent 448-instruction sidecar (432 target + 16 inert CUB helper), byte-identical replay repeat, and five application-replay NCU repetitions; instruction crosswalk exact; retirement tails retained |

Evidence: `docs/sm86_c7_sol_l1p078_source_v1.json`, `docs/sm86_c7_sol_l1p078_smoke_v1.json`, and `docs/sm86_c7_sol_l1p078_vertical_v1.json`.

### C7-S34BE — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_080_adaptive_layernorm_continuous_with_modulation/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BE1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine two-kernel FP32 adaptive LayerNorm/modulation path; 16 declared rows recorded |
| C7-S34BE2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; 16 declared rows × 3 seeds passed (48 rows; max absolute error 0.00036621); representative memcheck zero errors |
| C7-S34BE3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — two custom launches, independent 576-instruction sidecar (216 + 344 target + 16 inert CUB helper), byte-identical replay repeat, and five application-replay NCU repetitions per kernel; instruction crosswalk exact; retirement tails retained |

Evidence: `docs/sm86_c7_sol_l1p080_source_v1.json`, `docs/sm86_c7_sol_l1p080_smoke_v1.json`, and `docs/sm86_c7_sol_l1p080_vertical_v1.json`.

### C7-S34BF — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_081_joint_attention_context_projection/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BF1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — problem081 is pure ATen; source-class correction retained, no build admitted |
| C7-S34BF2 | Build and run bounded correctness/memcheck smoke | complete scoped — no build admitted after source correction |
| C7-S34BF3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — pure-ATen blocker retained; next genuine custom row selected |

Evidence: `docs/sm86_c7_sol_l1_problem081_source_class_correction_v1.json`.

### C7-S34BG — next executable L1 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L1/problem_082_qk_norm_scaled_dot_product_attention/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BG1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — problem082 is pure ATen; source-class correction retained, no build admitted |
| C7-S34BG2 | Build/correctness/memcheck or retain a concrete blocker | complete scoped — no build admitted after source correction |
| C7-S34BG3 | Admit full-chain capture/replay/NCU only after correctness | complete — pure-ATen blocker retained; next genuine custom row selected |

Evidence: `docs/sm86_c7_sol_l1_problem082_source_class_correction_v1.json`.

### C7-S34BH — SiLU backward representative vertical — complete; timing falsifiers retained

Source-review and, if admissible, vertically capture
`L1/problem_084_silu_activation_backward/cuda_cpp`.
Establish its own source/input/oracle contract and fresh pinned build/sanitizer
smoke before capture. No timing or NCU values transfer from prior rows; failures
remain explicit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BH1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine two-path FP32 custom CUDA SiLU backward; 16 declared element counts, including non-multiple-of-four tail rows, reviewed against the independent reference |
| C7-S34BH2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; official generator/reference rows passed 48/48 with minimum matched ratio 1.0 and maximum absolute error 1.53e-5; representative non-multiple-of-four tail memcheck reported zero errors |
| C7-S34BH3 | If admissible, capture/control/replay and five target NCU repetitions; otherwise retain a bounded blocker | complete — same official rebuilt extension produced both profiler-gated custom launches, 80 target + 16 helper control instructions, replay/NCU instruction crosswalks matched, replay repeat was byte-identical, and five application-replay NCU repetitions ran per launch; retirement tails and the separate observation envelope remain open |

Evidence: `docs/sm86_c7_sol_l1p084_source_v1.json`, `docs/sm86_c7_sol_l1p084_smoke_v1.json`.

Evidence: `docs/sm86_c7_sol_l1p084_vertical_v1.json`.

The primary experiment uses the official SOL generator and explicitly GCC14-built
extension, not the earlier uniform-`sigmoid_x` exploratory probe. Its modeled
SM-active terminal lowers `2705/1838` exceed all five measured SM-active values
(medians `1926/1400`): retained lower-bound falsifiers. GPC envelope `4489/3614`
and measured GPC medians `3551/2976` stay separate; no model parameter changed.

### C7-S34BI — GEGLU representative vertical — complete; physical tails open

Source-review and, if admissible, vertically capture
`L1/problem_085_geglu_activation/cuda_cpp`. Establish its own GEGLU input,
GELU-tanh numerical, and SM86 build contract before profiling; the `tanhf`
plus `--use_fast_math` combination is a source-level numerical risk and must
be checked against the official tolerance rather than assumed safe. No timing
or NCU value transfers from problem084.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BI1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — genuine one-kernel FP32 GEGLU custom CUDA path; 16 declared shapes and the `tanhf`/`--use_fast_math` numerical risk reviewed |
| C7-S34BI2 | Build and run bounded correctness/memcheck smoke | complete — pinned CUDA 13.2/GCC14 build; official criterion passed 48/48 with minimum matched ratio 0.999883 and maximum absolute error 2.48e-5 (strict all-element equality is not claimed); representative memcheck reported zero errors |
| C7-S34BI3 | Capture/control/replay and five target NCU repetitions, or record the actual blocker | complete — 655-block custom launch; 72 target + 16 helper static instructions; deterministic replay and five NCU repeats match 293440 warp instructions; dirty-writeback/coherence tails remain open |

Evidence: `docs/sm86_c7_sol_l1p085_source_v1.json`, `docs/sm86_c7_sol_l1p085_smoke_v1.json`.

Representative result: `docs/sm86_c7_sol_l1p085_vertical_v1.json`.

### C7-S34BJ — embedding/LayerNorm backward — partial; full capture open

Candidate: `L1/problem_087_embedding_with_initial_layernorm_backward/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BJ1 | Read implementation and official generator/reference/tolerance; identify custom and vendor launches | complete — one fused per-row atomic kernel plus two BF16 cast kernels; custom `get_inputs`, 16 shapes, and large destination/accumulator footprint recorded |
| C7-S34BJ2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — GCC14/CUDA13.2 build; official criterion passed 48/48, minimum matched ratios 1.0 (embedding) and 0.999756 (norm weight), representative memcheck zero; one norm-weight element exceeded the per-element tolerance but remained inside the 0.99 criterion |
| C7-S34BJ3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | open — the 420-second full NVBit attempt emitted 263172 threadblock files and stats through the fused kernel, then timed out before `dynamic_trace.pb` while the 1,048,576-block cast launch began; five filtered custom-kernel NCU repeats were retained as observation-only context, with no replay/vertical claim |

Evidence: `docs/sm86_c7_sol_l1p087_source_v1.json`, `docs/sm86_c7_sol_l1p087_smoke_v1.json`, `docs/sm86_c7_sol_l1p087_capture_blocked_v1.json`.

### C7-S34BK — next executable L1 custom-CUDA row — representative experiment complete; physical tails open

Source-review and, if admissible, vertically capture
`L1/problem_088_rotary_position_embedding_application/cuda_cpp`. The row is
separate from earlier RoPE candidates; no prior timing or NCU value transfers.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BK1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — one FP32 RoPE kernel launched separately for query/key; head_dim=128, two DPS outputs, 16 official shapes recorded |
| C7-S34BK2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — official criterion passed 48/48 for both outputs with matched ratio 1.0 and max absolute error 4.77e-7; representative memcheck zero |
| C7-S34BK3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete — two launches, one 40-instruction target function + 16 helper instructions, byte-identical replay, and five NCU repeats per launch; both modeled GPC lower bounds exceed NCU medians and are retained falsifiers |

Evidence: `docs/sm86_c7_sol_l1p088_source_v1.json`, `docs/sm86_c7_sol_l1p088_smoke_v1.json`.

Representative result: `docs/sm86_c7_sol_l1p088_vertical_v1.json`.

### C7-S34BL — next executable L1 custom-CUDA row — representative experiment complete; physical tails open

Source-review and, if admissible, vertically capture
`L1/problem_090_batched_2d_rope_position_encoding_backward/cuda_cpp`. It is a
separate custom row from problem088; no prior timing, replay, or NCU value
transfers.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BL1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — one FP32-output BF16-gradient SFU kernel; head_dim=128, one launch, 16 official shapes, fast-math sin/cos risk recorded |
| C7-S34BL2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — official criterion passed 48/48 with matched ratio 1.0 and max absolute error 2.32e-6; representative memcheck zero |
| C7-S34BL3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete — one SFU-heavy launch, 144 target + 16 helper static instructions, byte-identical replay, and five NCU repeats; the modeled GPC lower bound exceeds the NCU median and is retained as a falsifier |

Evidence: `docs/sm86_c7_sol_l1p090_source_v1.json`, `docs/sm86_c7_sol_l1p090_smoke_v1.json`.

Representative result: `docs/sm86_c7_sol_l1p090_vertical_v1.json`.

### C7-S34BM — earlier unexecuted L1 custom-CUDA row — source classification complete; vendor execution open

Source-review and, if admissible, vertically capture
`L1/problem_005_conv_gated_projection_with_causal_conv/cuda_cpp`. This returns
to the first genuine L1 custom row that has not yet received a source/smoke
record; prior rows' timing and NCU observations do not transfer.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BM1 | Verify source contract, dependency/toolchain path, and independent oracle | complete scoped correction — problem005 is a stub `.cu` plus ATen-only `main.cpp`; the census is corrected from 58/131 to 57 custom CUDA/132 pure ATen |
| C7-S34BM2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete scoped — no custom build admitted after source-class correction |
| C7-S34BM3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped — pure-ATen/non-custom disposition retained; no duplicate native vertical |

Evidence: `docs/sm86_c7_sol_l1_problem005_source_class_correction_v1.json`.

### C7-S34BN — next executable L1 custom-CUDA row — source classification complete; vendor execution open

Source-review and, if admissible, vertically capture
`L1/problem_007_hyena_fft_size_padding_rfft/cuda_cpp`. This is the next
genuine custom candidate after the problem005 correction; no prior timing or
NCU value transfers.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BN1 | Verify source contract, dependency/toolchain path, and independent oracle | complete scoped correction — problem007 is a stub `.cu` plus ATen/cuFFT-only `main.cpp`; census corrected to 56 custom CUDA/133 pure ATen |
| C7-S34BN2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete scoped — no custom build admitted after source-class correction |
| C7-S34BN3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped — pure-ATen/cuFFT disposition retained; no duplicate native vertical |

Evidence: `docs/sm86_c7_sol_l1_problem007_source_class_correction_v1.json`.

### C7-S34BO — BF16 scatter-add — partial; row0 capture timed out

Source-review and, if admissible, vertically capture
`L1/problem_008_expert_output_weighted_index_add_accumulation/cuda_cpp`. This
is the next genuine custom candidate after the problem007 correction.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BO1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — BF16 atomic scatter-add kernel plus device-copy initialization; custom `get_inputs`, duplicate token routing, and order-sensitive tolerance recorded |
| C7-S34BO2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — official criterion passed 48/48 with minimum matched ratio 0.999995 and maximum absolute error 0.25; representative memcheck zero |
| C7-S34BO3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | open — 420-second NVBit attempt reached the 196608-block BF16 atomic launch but produced no usable trace; five filtered custom-kernel NCU repeats retained as observation-only context, with no replay/vertical claim |

Evidence: `docs/sm86_c7_sol_l1p008_source_v1.json`, `docs/sm86_c7_sol_l1p008_smoke_v1.json`.

Blocked result: `docs/sm86_c7_sol_l1p008_capture_blocked_v1.json`.

### C7-S34BP — next executable L1 custom-CUDA row — source classification complete; vendor execution open

Source-review and, if admissible, vertically capture
`L1/problem_009_expert_token_scatter_with_weighted_forward_backward/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BP1 | Verify source contract, dependency/toolchain path, and independent oracle | complete scoped correction — problem009 is a stub `.cu` plus ATen/cuBLAS-only analytic backward; census corrected to 55 custom CUDA/134 pure ATen |
| C7-S34BP2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete scoped — no custom build admitted after source-class correction |
| C7-S34BP3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped — pure-ATen disposition retained; no duplicate native vertical |

Evidence: `docs/sm86_c7_sol_l1_problem009_source_class_correction_v1.json`.

### C7-S34BQ — next executable L1 custom-CUDA row — source classification complete; vendor execution open

Source-review and, if admissible, vertically capture
`L1/problem_014_rotary_embedding_with_attention_scaling_backward/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BQ1 | Verify source contract, dependency/toolchain path, and independent oracle | complete scoped correction — problem014 is a stub `.cu` plus ATen-only RoPE backward; census corrected to 54 custom CUDA/135 pure ATen |
| C7-S34BQ2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete scoped — no custom build admitted after source-class correction |
| C7-S34BQ3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped — pure-ATen disposition retained; no duplicate native vertical |

Evidence: `docs/sm86_c7_sol_l1_problem014_source_class_correction_v1.json`.

### C7-S34BR — next executable L1 custom-CUDA row — source classification complete; vendor execution open

Source-review and, if admissible, vertically capture
`L1/problem_017_moe_expert_swiglu_with_down_projection_backward/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BR1 | Verify source contract, dependency/toolchain path, and independent oracle | complete scoped correction — problem017 is a stub `.cu` plus ATen/cuBLAS-only SwiGLU backward; census corrected to 53 custom CUDA/136 pure ATen |
| C7-S34BR2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete scoped — no custom build admitted after source-class correction |
| C7-S34BR3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped — pure-ATen disposition retained; no duplicate native vertical |

Evidence: `docs/sm86_c7_sol_l1_problem017_source_class_correction_v1.json`.

### C7-S34BS — next executable L1 custom-CUDA row — representative experiment complete; physical tails open

Source-review and, if admissible, vertically capture
`L1/problem_028_hybrid_attention_mask_preparation/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BS1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — one custom boolean kernel writes full causal and canonical all-false sliding masks; 12 scalar-shape rows reviewed |
| C7-S34BS2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — official criterion passed 36/36 with matched ratio 1.0 and zero absolute error for both bool outputs; representative memcheck zero |
| C7-S34BS3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete — one boolean-mask launch, 112 target + 16 helper static instructions, byte-identical replay, and five NCU repeats; GPC lower remains below its NCU median; SM-active lower crosses its corresponding observations |

Evidence: `docs/sm86_c7_sol_l1p028_source_v1.json`, `docs/sm86_c7_sol_l1p028_smoke_v1.json`.

Representative result: `docs/sm86_c7_sol_l1p028_vertical_v1.json`.

### C7-S34BT — existing FIB001 official-row refresh — representative experiment complete; physical tails open

Source-review and, if admissible, vertically capture
`FIB/problem_001_fused_add_rmsnorm_h2048/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BT1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — one-CTA-per-row BF16x2 fused residual-add/RMSNorm kernel; hidden_size=2048 and seven FIB shapes reviewed |
| C7-S34BT2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — official FlashInfer-Bench criterion passed 21/21 with matched ratio 1.0 and max absolute error 0.03125; representative memcheck zero |
| C7-S34BT3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete — one 1024-thread fused launch, 176 target + 16 helper static instructions, byte-identical replay, and five NCU repeats; GPC lower remains below its NCU median; SM-active lower crosses its corresponding observations |

Evidence: `docs/sm86_c7_sol_fibp001_source_v1.json`, `docs/sm86_c7_sol_fibp001_smoke_v1.json`.

Representative result: `docs/sm86_c7_sol_fibp001_vertical_v1.json`.

### C7-S34BU — next executable FIB custom-CUDA row — representative experiment complete; physical tails open

Source-review and, if admissible, vertically capture
`FIB/problem_012_gqa_paged_decode_h32_kv4_d128_ps1/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BU1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — one 128-thread-per-query-head online-softmax GQA paged-decode kernel; 48 rows with retained safetensor inputs reviewed |
| C7-S34BU2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — official 144 seeded rows passed with matched ratio 1.0, max errors 0.00390625 (BF16 output) and 5.72e-6 (LSE), representative memcheck zero |
| C7-S34BU3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete — one GQA paged-decode launch, 336 target + 16 helper static instructions, byte-identical replay, and five NCU repeats; GPC lower remains below its NCU median; SM-active lower crosses its corresponding observations |

Evidence: `docs/sm86_c7_sol_fibp012_source_v1.json`, `docs/sm86_c7_sol_fibp012_smoke_v1.json`.

Representative result: `docs/sm86_c7_sol_fibp012_vertical_v1.json`.

### C7-S34BV — decoder block — partial; reference capacity and capture gaps open

Source-review and, if admissible, vertically capture
`L2/problem_002_decoder_layer_full_block/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BV1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — 18 official shapes; custom RMSNorm/SiLU plus ATen/cuBLAS; actual tolerance-parser spelling difference recorded |
| C7-S34BV2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | partial capacity blocker — 17 shapes × 3 seeds pass; row17 reference needs two 16-GiB attention tensors and OOMs on this 24-GB GPU; row0 memcheck clean |
| C7-S34BV3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | open — blocker recorded — full row0 NVBit timed out in vendor SGEMM after 420s; five application-replay NCU runs observe all 36 kernels but no replay is available |

Evidence: `docs/sm86_c7_sol_l2p002_source_v1.json`, `docs/sm86_c7_sol_l2p002_smoke_v1.json`, `docs/sm86_c7_sol_l2p002_capture_blocked_v1.json`.


Evidence for BV: `docs/sm86_c7_sol_l2p002_capture_blocked_v1.json`.

### C7-S34BW — problem008 smaller declared row — experiment complete; crosswalk unresolved

Try the **actual declared row9** (B=1, S=131), not a synthetic size, after
row0's instrumented run timed out. Selection is based on source/trace size,
not NCU timing. Preserve row0 as unfinished even if row9 succeeds.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BW1 | Same extension, official row9 correctness/memcheck and full native capture | complete — actual declared row9 B=1,S=131; memcheck zero; NVBit trace produced |
| C7-S34BW2 | Independent control extraction, replay or concrete unsupported instruction, then same-row observations | complete — 64 target + 16 helper control instructions, deterministic replay, five NCU repeats; replay/NCU instruction mismatch retained |
| C7-S34BW3 | Update coverage without erasing row0's timeout or claiming copy-engine timing | complete — row9 vertical retained; row0 420-second timeout remains a separate unresolved population/capture blocker |

Evidence: `docs/sm86_c7_sol_l1p008_row9_vertical_v1.json`, `docs/sm86_c7_sol_l1p008_capture_blocked_v1.json`.

### C7-S34BX — next executable L2 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L2/problem_004_fused_residual_rms_mlp/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BX1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — two custom fusion kernels plus three cuBLAS projections; 16 rows and large-weight memory risk recorded |
| C7-S34BX2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | partial — row1 (B=1,S=256) passes and memcheck is clean; full population not screened because the declared weight/activation footprint needs bounded capacity handling |
| C7-S34BX3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | open — blocker recorded — NVBit reproducibly fatals on HMNMX2.BF16_V2 after the custom residual kernel, in the subsequent projection path, with and without enhanced post-parser; filtered NCU context only, no replay |

Evidence: `docs/sm86_c7_sol_l2p004_source_v1.json`, `docs/sm86_c7_sol_l2p004_smoke_v1.json`, `docs/sm86_c7_sol_l2p004_capture_blocked_v1.json`.

### C7-S34BY — next executable L2 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L2/problem_005_swiglu_mlp_backward/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BY1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — three custom elementwise fusions plus six cuBLAS matmuls; 18 rows and large-weight memory risk reviewed |
| C7-S34BY2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — bounded CPU-snapshot harness passes 54/54 seeded rows; max absolute error 0.5; row0 memcheck zero |
| C7-S34BY3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped blocker — NVBit fatals on HMNMX2.BF16_V2 with and without enhanced post-parser; no replay/full vertical |

Evidence: `docs/sm86_c7_sol_l2p005_source_v1.json`, `docs/sm86_c7_sol_l2p005_smoke_v1.json`, `docs/sm86_c7_sol_l2p005_capture_blocked_v1.json`.

### C7-S34BZ — next executable L2 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L2/problem_006_multimodal_rope_position_calculation/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34BZ1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — one-thread-per-batch multimodal RoPE position kernel plus ATen event-count/prefix prework; 16 custom-input rows |
| C7-S34BZ2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — 48 official rows pass with exact position/delta outputs; representative memcheck zero |
| C7-S34BZ3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped replay blocker — NVBit emits 17 launches, but extension-only control sidecar lacks internal ATen/CUB functions; sim-run refuses before replay, no full vertical admitted |

Evidence: `docs/sm86_c7_sol_l2p006_source_v1.json`, `docs/sm86_c7_sol_l2p006_smoke_v1.json`, `docs/sm86_c7_sol_l2p006_capture_blocked_v1.json`.

### C7-S34CA — MoE sparse routing — partial; vendor capture blocker

Source-review and, if admissible, vertically capture
`L2/problem_008_moe_sparse_routing_and_dispatch/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CA1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — one custom BF16 SiLU*up fusion plus 128-expert ATen routing/dispatch loop |
| C7-S34CA2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — 48 official rows pass, max absolute error 0.0234375; representative memcheck zero |
| C7-S34CA3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped blocker — vendor HMNMX2.BF16_V2 parser fatal after ATen routing kernels; no replay |

After BW: C7-S34BX source/execution of L2/problem004 fused residual RMS MLP.

Evidence: `docs/sm86_c7_sol_l2p008_source_v1.json`, `docs/sm86_c7_sol_l2p008_smoke_v1.json`, `docs/sm86_c7_sol_l2p008_capture_blocked_v1.json`.

### C7-S34CB — next executable L2 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L2/problem_030_flux_concatenated_sequence_processing_with_split/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CB1 | Verify source contract, dependency/toolchain path, and independent oracle | complete scoped correction — problem030 is ATen/cuBLAS-only with no custom `__global__`; census corrected to 52 custom CUDA/137 pure ATen |
| C7-S34CB2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete scoped — no custom build admitted |
| C7-S34CB3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped — pure ATen/cuBLAS disposition retained; no duplicate native vertical |

Evidence: `docs/sm86_c7_sol_l2_problem030_source_class_correction_v1.json`.

### C7-S34CC — next executable L2 custom-CUDA row — active

Source-review and, if admissible, vertically capture
`L2/problem_031_flux_timestep_guidance_projection_embedding/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CC1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — two custom timestep/SiLU fusions plus ATen projections; custom `get_inputs`, 16 shapes recorded |
| C7-S34CC2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — 48 official rows pass; minimum matched ratio 0.99544, max absolute error 1.97e-5; representative memcheck zero |
| C7-S34CC3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped blocker — NVBit trace emits 7 launches, but extension-only control sidecar cannot bind internal cuBLAS/CUBLASLt/ATen functions; replay refused |

Evidence: `docs/sm86_c7_sol_l2p031_source_v1.json`, `docs/sm86_c7_sol_l2p031_smoke_v1.json`, `docs/sm86_c7_sol_l2p031_capture_blocked_v1.json`.

### C7-S34CD — L2 ResNet/time-embedding row — complete scoped blocker

Source-review and, if admissible, vertically capture
`L2/problem_066_resnet_block_with_time_embedding/cuda_cpp`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CD1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — three `__global__` definitions, four default-stream launches (K1/K3 symbol reuse), FP32 NCHW contract, and direct-convolution capture-cost risk recorded |
| C7-S34CD2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — pinned GCC14/CUDA13.2 build; 48/48 official rows across three seeds pass with minimum matched ratio 1.0 and max absolute error 0.0457897; representative memcheck zero |
| C7-S34CD3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete scoped blocker — full-payload and counting-only NVBit attempts on the smallest declared row both stop at the direct `conv1_with_time` launch before K2 completion; no `dynamic_trace.pb`, control sidecar, replay, or vertical is admitted. Five independent NCU repeats of all four custom launches are observation-only context |

Evidence: `docs/sm86_c7_sol_l2p066_source_v1.json`, `docs/sm86_c7_sol_l2p066_smoke_v1.json`, and `docs/sm86_c7_sol_l2p066_capture_blocked_v1.json`.

The capture blocker is execution-capacity specific, not a correctness failure:
the smallest declared row is `B=1,H=W=16`, but K2 performs a direct 320×320,
3×3 convolution and a 1280-deep time projection per output thread. The
independent NCU arm completes all four launches, while both NVBit modes
reproduce the K2 non-completion. No NCU value was used to change the source or
simulator.

### C7-S34CE — Quant FP8 YaRN RoPE row — complete; physical tails open

Source-review and, if admissible, vertically capture
`Quant/problem_014_fp8_yarn_rope_embedding/cuda_cpp`. Establish the FP8
e4m3/YaRN input and reference contract before profiling; the largest declared
sequence lengths (up to 131072) require bounded memory and capture checks.
No timing or NCU values transfer from C7-S34CD, and unsupported or vendor
parser failures remain in the population denominator.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CE1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — one FP8 e4m3 custom kernel, custom `get_inputs`, 16 sequence lengths through 131072, and absent source SM86 marker recorded; pinned build adds explicit SM86 gencode |
| C7-S34CE2 | Same-source GCC14/CUDA13.2 build, official correctness rows and representative memcheck | complete — explicit SM86 build; 48/48 official rows pass with minimum matched ratio 0.9972527 and max absolute error 0.0175781; seq_len=65536 memcheck reports zero errors; strict equality is not claimed for large rows |
| C7-S34CE3 | Full representative launch chain: NVBit/control, deterministic replay, then five independent NCU repetitions; retain actual failures | complete — declared seq_len=1024 launch captured; pinned control sidecar decodes 168 target + 16 CUB helper instructions with zero unknowns; replay is byte-identical and matches five NCU repeats at 171962 warp instructions; SM-active lower-bound falsifier and dirty/coherence tails retained |

Evidence: `docs/sm86_c7_sol_quantp014_source_v1.json`, `docs/sm86_c7_sol_quantp014_smoke_v1.json`, and `docs/sm86_c7_sol_quantp014_vertical_v1.json`.

The source's `target_hardware=LOCAL` declaration did not contain an SM86
marker; the captured extension is nevertheless explicitly built for SM86 and
the omission remains visible in the source record. The enhanced JIT parser did
not decode the `.so`, so the independent pinned `cuobjdump` control extractor
is the replay input. NCU values are observation-only: the modeled SM-active
terminal lower bound (`4000`) exceeds the independent median (`3497`) and is
retained as a falsifier, not fitted away.

### C7-S34CF — Quant NVFP4 CUTLASS row — complete architecture disposition

Source-review `Quant/problem_020_nvfp4_linear_layer/cuda_cpp` before attempting
any SM86 execution. Its packaged compile options target `compute_120a,sm_120a`
and its source depends on CUTLASS/NVFP4 facilities; an SM86 host cannot execute
that architecture. Record the exact source and a concrete no-substitute
disposition rather than compiling a different implementation or borrowing
timing from C7-S34CE.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CF1 | Verify source contract, dependency/toolchain path, and architecture/oracle compatibility | complete — packaged target and CUTLASS type are explicitly SM120/Blackwell NVFP4; host is SM86 |
| C7-S34CF2 | Build/correctness/memcheck or retain an explicit architecture blocker | complete scoped — no SM86 build admitted; the exact source/target/oracle disposition is retained, with no translated or software substitute |
| C7-S34CF3 | Capture/replay/NCU only if an SM86-compatible path is independently admissible | complete scoped — not applicable on RTX 3090 Ti; row remains in the Quant denominator and needs SM120 hardware |

Evidence: `docs/sm86_c7_sol_quantp020_source_v1.json` and `docs/sm86_c7_sol_quantp020_architecture_blocked_v1.json`.

This is an architecture boundary, not a correctness or performance result.
The SM120 CUTLASS/NVFP4 path is not translated into an SM86 implementation and
no Blackwell observation is borrowed.

### C7-S34CG — FIB problem005 CUTLASS GEMM — complete; physical tails open

Source-review and, if admissible, vertically capture
`FIB/problem_005_gemm_n256_k7168/cutlass`. Existing FIB/problem004 replay and
flat-fallback artifacts are not transferred; keep the CUTLASS source, SM86
build, and mixed-module control/replay contract separate.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CG1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS Sm80 TensorOp, explicit SM86 target, FP16 GEMM C=A@B.T, 17 M-shapes recorded |
| C7-S34CG2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned CUTLASS include/GCC14/CUDA13.2 build; 51/51 official rows pass with minimum matched ratio 0.9955357 and max absolute error 0.25; M=14104 memcheck zero |
| C7-S34CG3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete — declared M=32 launch captured; independent control sidecar decodes 1160 target + 16 helper instructions with zero unknowns; byte-identical replay and five measured (`--launch-skip 1`) NCU repeats match at 209552 instructions. Warm-launch NCU attempts are retained and excluded |

Evidence: `docs/sm86_c7_sol_fibp005_source_v1.json`, `docs/sm86_c7_sol_fibp005_smoke_v1.json`, and `docs/sm86_c7_sol_fibp005_vertical_v1.json`.

The first NCU attempt used `--launch-skip 0` and profiled the runner's warm
algorithm-selection launch (grid 4×1×28), not the measured grid 1×2×1. Those
files remain visible as a protocol error; the corrected five-repeat arm uses
`--launch-skip 1`, matches the trace/replay launch, and is the only NCU arm
used in the crosswalk. CUTLASS FP16 differences remain under the declared
criterion; dirty-writeback/coherence tails remain open.

### C7-S34CH — FIB problem006 CUTLASS GEMM — complete scoped; larger-row capture blocker retained

Source-review and, if admissible, vertically capture
`FIB/problem_006_gemm_n2048_k4096/cutlass`. Keep its larger N/K contract
separate from problem005; retain any capacity, CUTLASS launch, or replay
failure without borrowing prior timing.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CH1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS Sm80 TensorOp, explicit SM86 target, FP16 GEMM N=2048/K=4096, 29 M-shapes recorded |
| C7-S34CH2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned CUTLASS include/GCC14/CUDA13.2 build; 87/87 official rows pass with minimum matched ratio 0.9987259 and max absolute error 0.125; M=16294 memcheck zero |
| C7-S34CH3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete scoped — declared M=6 now has L1-L2 trace/control/replay and five explicit profiler-gated NCU repeats matching at 1383360 instructions after the zero-fill topology validation fix; declared M=12853 NVBit capture still times out before a trace. Full population remains open |

Evidence: `docs/sm86_c7_sol_fibp006_source_v1.json`, `docs/sm86_c7_sol_fibp006_smoke_v1.json`, `docs/sm86_c7_sol_fibp006_row0_vertical_v1.json`, `docs/sm86_c7_sol_fibp006_row0_l1_l2_vertical_v2.json`, and `docs/sm86_c7_sol_fibp006_capture_blocked_v1.json`.

The M=6 result is now a declared-row L1-L2 vertical. The earlier flat-only
record remains historical; the larger declared M=12853 instrumented timeout
remains open. Dirty writeback, coherence, and async shared-visibility tails
remain open.

### C7-S34CI — FIB problem007 CUTLASS GEMM — complete; physical tails open

Source-review and, if admissible, vertically capture
`FIB/problem_007_gemm_n4096_k4096/cutlass`. Keep its larger N contract and
capacity risk separate from problem006; no prior timing or flat fallback is
transferred.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CI1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS Sm80 TensorOp, explicit SM86 target, FP16 GEMM N=4096/K=4096, 43 M-shapes recorded |
| C7-S34CI2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned CUTLASS include/GCC14/CUDA13.2 build; 129/129 official rows pass with minimum matched ratio 0.9994860 and max absolute error 0.125; M=8192 memcheck zero |
| C7-S34CI3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete — declared M=1 launch has L1-L2 trace/control/replay and five explicit profiler-gated NCU repeats matching at 2766720 instructions; physical tails remain open |

Evidence: `docs/sm86_c7_sol_fibp007_source_v1.json`, `docs/sm86_c7_sol_fibp007_smoke_v1.json`, `docs/sm86_c7_sol_fibp007_vertical_v1.json`, and `docs/sm86_c7_sol_fibp007_l1_l2_vertical_v2.json`.

The post-fix vertical uses L1-L2 mode. It does not claim native scheduling or
physical completion; the original flat record remains a reproducible historical
cross-check.

### C7-S34CJ — FIB problem008 CUTLASS GEMM — complete; physical tails open

Source-review and, if admissible, vertically capture
`FIB/problem_008_gemm_n4096_k14336/cutlass`. This row has a larger K and
weight footprint than C7-S34CI; retain capacity and instrumented-execution
limits separately from the M=1 flat vertical.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CJ1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS Sm80 TensorOp, explicit SM86 target, FP16 GEMM N=4096/K=14336 and ~117 MiB B-weight footprint recorded |
| C7-S34CJ2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned CUTLASS include/GCC14/CUDA13.2 build; 129/129 official rows pass with minimum matched ratio 0.9961141 and max absolute error 0.25; M=8192 memcheck zero |
| C7-S34CJ3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete — declared M=1 deep-K launch has L1-L2 trace/control/replay and five explicit profiler-gated NCU repeats matching at 9402240 instructions; physical tails remain open |

Evidence: `docs/sm86_c7_sol_fibp008_source_v1.json`, `docs/sm86_c7_sol_fibp008_smoke_v1.json`, `docs/sm86_c7_sol_fibp008_vertical_v1.json`, and `docs/sm86_c7_sol_fibp008_l1_l2_vertical_v2.json`.

The deep-K row completed its bounded L1-L2 vertical despite the larger weight
footprint. It does not establish native scheduling or physical completion.

### C7-S34CK — FIB problem009 CUTLASS GEMM — complete; physical tails open

Source-review and, if admissible, vertically capture
`FIB/problem_009_gemm_n5120_k2048/cutlass`. Keep the larger N=5120 output
contract and capacity behavior separate from the prior rows.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CK1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS Sm80 TensorOp, explicit SM86 target, FP16 GEMM N=5120/K=2048, 25 M-shapes recorded |
| C7-S34CK2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned build; 75/75 official rows pass with minimum matched ratio 0.9999444 and max absolute error 0.125; M=16294 memcheck zero |
| C7-S34CK3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete — declared M=1 launch has L1-L2 trace/control/replay and five explicit profiler-gated NCU repeats matching at 1799520 instructions; physical tails remain open |

Evidence: `docs/sm86_c7_sol_fibp009_source_v1.json`, `docs/sm86_c7_sol_fibp009_smoke_v1.json`, and `docs/sm86_c7_sol_fibp009_l1_l2_vertical_v2.json`.

### C7-S34CL — FIB problem010 CUTLASS GEMM — complete; physical tails open

Source-review and, if admissible, vertically capture
`FIB/problem_010_gemm_n6144_k4096/cutlass`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CL1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS Sm80 TensorOp, explicit SM86 target, FP16 GEMM N=6144/K=4096, 43 M-shapes recorded |
| C7-S34CL2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned build; 129/129 official rows pass with minimum matched ratio 0.9974320 and max absolute error 0.1875; M=8192 memcheck zero |
| C7-S34CL3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete — declared M=1 launch has L1-L2 trace/control/replay and five explicit profiler-gated NCU repeats matching at 4150080 instructions; physical tails remain open |

Evidence: `docs/sm86_c7_sol_fibp010_source_v1.json`, `docs/sm86_c7_sol_fibp010_smoke_v1.json`, and `docs/sm86_c7_sol_fibp010_vertical_v1.json`.

### C7-S34CM — FIB problem011 CUTLASS GEMM — complete; physical tails open

Source-review and, if admissible, vertically capture
`FIB/problem_011_gemm_n28672_k4096/cutlass`. The N=28672 output and largest
M rows require an explicit memory-capacity check before profiling.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CM1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS Sm80 TensorOp, explicit SM86 target, FP16 GEMM N=28672/K=4096, ~224 MiB B-weight footprint recorded |
| C7-S34CM2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned build; 129/129 official rows pass with minimum matched ratio 0.9988055 and max absolute error 0.125; M=8192 memcheck zero |
| C7-S34CM3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete — declared M=1 launch has L1-L2 trace/control/replay and five explicit profiler-gated NCU repeats matching at 19367040 instructions; physical tails remain open |

Evidence: `docs/sm86_c7_sol_fibp011_source_v1.json`, `docs/sm86_c7_sol_fibp011_smoke_v1.json`, and `docs/sm86_c7_sol_fibp011_vertical_v1.json`.

### C7-S34CN — L1 problem003 CUTLASS LM-head — complete scoped capture blocker

Source-review and, if admissible, vertically capture
`L1/problem_003_lm_head_projection_with_logit_slicing/cutlass`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CN1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS Sm80 BF16 TensorOp, explicit SM86 target, hidden=2048/vocab=102400, canonical logit slicing, and ~400 MiB weight risk recorded |
| C7-S34CN2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned build; 48/48 official rows pass exactly and largest seq=3011 memcheck zero |
| C7-S34CN3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete scoped blocker — two actual declared rows (1600 and 800 CTAs) both hit the 420-second NVBit bound before dynamic trace emission; no replay/NCU substitute admitted |

Evidence: `docs/sm86_c7_sol_l1p003_source_v1.json`, `docs/sm86_c7_sol_l1p003_smoke_v1.json`, and `docs/sm86_c7_sol_l1p003_capture_blocked_v1.json`.

### C7-S34CO — L1 problem022 CUTLASS/TF32 backward chain — complete scoped replay

Source-review and, if admissible, vertically capture
`L1/problem_022_vision_mlp_gelu_backward/cutlass`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CO1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — four CUTLASS Sm80 TF32-style GEMMs plus GELU and bias kernels; seven launches, 16 seq-length rows, and large intermediate tensors recorded |
| C7-S34CO2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned build; 48/48 official rows pass and S=4096 memcheck zero |
| C7-S34CO3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete scoped — exact SM86 `HMMA.1688.F32.TF32` capability now replays all seven launches in L1-L2 and flat modes; five post-fix application-replay NCU sets preserve per-launch instruction/HMMA counts; physical tails remain open |

Evidence: `docs/sm86_c7_sol_l1p022_source_v1.json`, `docs/sm86_c7_sol_l1p022_smoke_v1.json`, and `docs/sm86_m5g_tf32_pipeline_replay_v1.json`; the former blocker record is retained as historical disposition.

### C7-S34CP — L1 problem030 CUTLASS residual projection — complete; physical tails open

Source-review and, if admissible, vertically capture
`L1/problem_030_attention_output_projection_with_residual/cutlass`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CP1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS Sm80 BF16 TensorOp residual epilogue, hidden=2560, 16 shapes recorded |
| C7-S34CP2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned build; 48/48 official rows pass and S=2048 memcheck zero |
| C7-S34CP3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete — declared B=4,S=128 launch has L1-L2 trace/control/replay and five explicit profiler-gated NCU repeats matching at 4526080 instructions; physical tails remain open |

Evidence: `docs/sm86_c7_sol_l1p030_source_v1.json`, `docs/sm86_c7_sol_l1p030_smoke_v1.json`, and `docs/sm86_c7_sol_l1p030_vertical_v1.json`.

### C7-S34CQ — L1 problem037 CUTLASS/GELU chain — complete scoped capture blocker

Source-review and, if admissible, vertically capture
`L1/problem_037_flux_feedforward_gelu_approximate/cutlass`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CQ1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — two custom GELU/bias kernels plus two CUTLASS FP32/TF32 GEMMs, four launches, 16 sequence rows |
| C7-S34CQ2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned build; 48/48 official rows pass and S=4096 memcheck zero |
| C7-S34CQ3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete scoped blocker — the declared S=1024 capture reaches the FP32/TF32 chain but times out at the 420-second NVBit bound before dynamic trace emission; no replay/NCU substitute admitted |

Evidence: `docs/sm86_c7_sol_l1p037_source_v1.json`, `docs/sm86_c7_sol_l1p037_smoke_v1.json`, and `docs/sm86_c7_sol_l1p037_capture_blocked_v1.json`.

### C7-S34CR — L1 problem077 Whisper projection — complete; physical tails open

Source-review and, if admissible, vertically capture
`L1/problem_077_whisper_decoder_output_projection/cutlass`.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CR1 | Verify source contract, dependency/toolchain path, and independent oracle | complete — CUTLASS Sm80 FP16 TensorOp, vocab=51866 tail-safe AlignmentC=2, 16 shapes recorded |
| C7-S34CR2 | Same-source SM86 build, official correctness rows and representative memcheck | complete — pinned build; 48/48 official rows pass with minimum matched ratio 0.9950960 and max absolute error 0.00390625; S=2048 memcheck zero |
| C7-S34CR3 | Full representative launch chain, deterministic replay/flat fallback if supported, and five independent NCU repetitions; retain actual failures | complete — declared B=4,S=1 launch has L1-L2 trace/control/replay and five explicit profiler-gated NCU repeats matching at 12313168 instructions; physical tails remain open |

Evidence: `docs/sm86_c7_sol_l1p077_source_v1.json`, `docs/sm86_c7_sol_l1p077_smoke_v1.json`, and `docs/sm86_c7_sol_l1p077_vertical_v1.json`.

### C7-S34CS — CUTLASS/source-population closure audit — complete scoped

Reconcile the current CUTLASS/vendor census against source, smoke, vertical, or
explicit architecture/replay/capture blocker records. This is a bounded
bookkeeping-and-evidence audit, not a new execution gate: every row remains in
the denominator, and no unresolved mechanism is promoted by the audit.

| Sub-milestone | Completion bar | State |
|---|---|---|
| C7-S34CS1 | Enumerate all 14 CUTLASS rows and confirm each source path and SM86/architecture disposition | complete — 14/14 source directories present; SM86 verticals and SM120 architecture boundary retained |
| C7-S34CS2 | Confirm each row has current source/smoke plus vertical or typed blocker evidence; retain missing links | complete — 14/14 evidence sets present: 11 scoped live verticals/replays and 3 explicit blockers; C7-S34CO TF32 replay is resolved |
| C7-S34CS3 | Synchronize plan/README/STATUS/PROJECT_MEMORY/C8 linked artifacts and select the next open mechanism | complete — plan/runbook links synchronized; physical tails, capture timeouts, and native continuation remain open; the former HMMA capability blocker is resolved by M5-G |

Evidence: `docs/sm86_c7_sol_cutlass_population_closure_v1.json`.

The audit is a population disposition, not an execution gate and not a claim
that the four blocked rows or physical tails are complete.

### M5-E — post-population completion/native-continuation seam — retained negative reconciled

Return to the open mechanism program after CUTLASS/source closure. Reconcile
the independent completion-tail facts and exact-fallback boundary, then make
one bounded real-trace continuation attempt only if an independently typed
state/effect witness is available. No synthetic witness, NCU-guided tuning, or
renaming of refusal as completion is permitted.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-E1 | Synchronize C1-R/C3/C6 completion-tail, observer, and native-dispatch facts after C7 closure; keep frontiers separate | complete — issue, SM-active, GPC, completion, and typed-continuation frontiers remain distinct |
| M5-E2 | Make one bounded real continuation/completion differential with an independent typed witness, or retain exact fallback and a falsifiable blocker | complete scoped negative — retained M5-C embedding trace has 0 complete typed boundaries, 10032 exact-fallback boundary records, and 0 native macro dispatches; not a new M5-E run and not a negation of the narrow C6 scalar positive |
| M5-E3 | Update claims/runbook and select the next evidence-supported mechanism without closing physical tails | complete — reconciliation artifact and C8 links synchronized; physical tails and broad accuracy remain open |

Evidence: `docs/sm86_m5e_completion_continuation_reconciliation_v1.json`.

### M5-F — independent accuracy/physical-tail expansion — streaming observer slice complete; physical tails open

The next bounded program is to expand only evidence-supported observer and
tail slices: predictions must precede untouched NCU observations, NCU values
cannot select simulator parameters, and no finite observer interval may be
called a physical completion upper. Positive native continuation remains a
separate bar.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-F1 | Select one calibration-only observer/tail family with a disjoint untouched target and explicit metric/endpoint identity | complete — streaming `cp.async.cg`, depth 4, compact destination; groups 16/32 calibration only, new groups-64 target repetitions; GPC elapsed maximum under application replay/base clocks/cache-all |
| M5-F2 | Materialize prediction, collect independent target repetitions, and reconcile finite/open coverage without fitting scored values | complete — prediction 48945, empirical interval 48680.6..49209.4 written before target NCU; 5/5 repeats covered, median 48914, point APE 0.0634%; 2216 instructions match, seven correctness runs, full-shape memcheck zero, byte-identical detailed replays |
| M5-F3 | Synchronize accuracy/tail claims and preserve exact fallback/native-dispatch boundaries | complete scoped — observer interval is not a physical upper; completion lower 42437 and async visibility/dirty-writeback/coherence tails remain open; no simulator/card parameter changed |

Evidence: `docs/sm86_m5f1_async_dram_prediction_pre_ncu_v1.json` and
`results/alpha/m5f_async_dram_20261006/observer_result.json`. This is a
within-family group-count extrapolation, not broad/generalization accuracy.
The `dram` selector streams source addresses and bypasses L1; it does not prove
that every request was serviced by DRAM. The actual host identity query was
after target collection, not a claimed pre-run hardware observation. The
auxiliary enhanced-SASS size-limit warning is retained; replay consumed a fresh
4096-instruction, zero-unknown independent control sidecar. Full local suites:
495 `sim-sm` and 134 `sim-semantics` tests passed; those are software checks,
not hardware accuracy evidence.

### M5-G — TF32 execution-pipeline capability — complete scoped

Address the concrete C7-S34CO blocker without substituting FP16 or tuning to
that SOL row. Read the public SM86/TF32 contract, distinguish initiation from
dependent-result latency, and use independent instruction probes if public
throughput data alone is insufficient. Preserve refusal for unmodeled shapes
and architectures. No proprietary implementation reconstruction is needed.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-G1 | Establish a source-supported SM86 `HMMA.1688.F32.TF32` model and independent probe; record assumptions before scoring SOL | complete — public PTX shape plus independent no-memory probe: 8-way issue arm ~32 cycles/HMMA, dependent arm ~40-cycle result latency; exact interval kept separate from dependency latency |
| M5-G2 | Add the smallest exact-opcode capability, run focused negatives and the actual retained seven-launch L1/022 trace | complete — `HMMA.1688.F32.TF32` admitted only on SM86; standalone C7-S11 and all seven C7-S34CO launches replay in L1-L2 and flat modes; unsupported forms/SM120 refusal tests remain green |
| M5-G3 | Collect independent NCU only after replay parameters are fixed, report instruction/frontier differences and synchronize remaining blockers | complete scoped — five application-replay NCU sets after the model change; per-launch instruction and HMMA counts match replay; no NCU timing value selected a parameter; tails remain lower-only |

Evidence: `docs/sm86_m5g1_tf32_pipeline_probe_v1.json` and
`docs/sm86_m5g_tf32_pipeline_replay_v1.json`. The former C7-S34CO blocker
record remains historical; current closure is the exact-opcode replay record.
This resolves replay capability only, not completion, scheduling, or broad
accuracy.

### M5-H — physical-tail/ownership boundary expansion — complete scoped

With the remaining CUTLASS replay capability repaired, return to the physical
completion seam. One bounded source-side tail differential will compare a
store/fence or async-visibility arm with an untouched control arm, using only
trace facts and independent NCU observations for reconciliation. It must not
turn a finite observer interval into a hardware upper, and it must not invent
owner/read-from/generation facts that the trace does not contain.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-H1 | Select one unresolved physical-tail family and document the exact control/store or visibility contrast | complete — C1-R memory-retirement service, control vs store, same b84/line-stream/d16/l32 geometry |
| M5-H2 | Capture/replay both arms with deterministic output and independent NCU repetitions; retain missing ownership/completion tokens | complete scoped — current release replays both retained traces twice byte-identically; seven correctness/sanitizer and five NCU observations per arm retained |
| M5-H3 | Reconcile the tail differential and choose the next mechanism without changing native-dispatch or completion claims | complete scoped — simulator delta 2886 cycles vs NCU median delta 6080; dirty-writeback/coherence remain unknown; no service cost fitted |

Evidence: `docs/sm86_m5h_store_tail_differential_v1.json`. The result is a
mechanism-local mismatch, not a physical completion upper or native scheduling
claim.

### M5-I — async visibility/ownership boundary — complete scoped

Use the existing complete async-copy/commit/wait prefix to test whether a
bounded source-only versus visibility-result arm can add any independently
observable ownership or generation fact. Preserve exact fallback whenever the
trace lacks a typed owner/read-from/completion token; do not promote a protocol
prefix into hardware completion.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-I1 | Select the paired async visibility/source arms and enumerate the missing physical fields | complete — l1 compact depth4 groups64 source-only vs visibility-result pair; owner/read-from/generation/completion/fence scope explicitly missing |
| M5-I2 | Re-run or reconcile deterministic traces and independent observations without target leakage | complete scoped — current-release replays and independent fact extraction complete; 256 copies, 8192 lane operations, 2048 committed/waited groups in both arms; no causal endpoint delta claimed |
| M5-I3 | Update the async-tail frontier and choose the next evidence-supported seam | complete scoped — logical wait prefix complete, physical ownership/completion remains open; exact fallback/native-dispatch boundary unchanged |

Evidence: `docs/sm86_m5i_async_visibility_ownership_boundary_v1.json`.
This is a negative ownership/completion boundary, not a hardware visibility proof.

### M5-J — typed native-continuation expansion — complete scoped

The remaining continuation question is whether the narrow C6 scalar reference
transition can be extended to a second real trace without a synthetic witness.
Select one existing memory-free or fully typed candidate; if typed entry/effect/
exit state is incomplete, execute the bounded refusal and retain exact replay.
Do not use aggregate protocol counts or NCU timing to create a witness.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-J1 | Select a real trace candidate and verify typed entry/effect/exit/ownership completeness before mutation | complete — real C6-F one-block/two-warp trace has complete typed entry but future-relevant multiwarp/global state is not closed |
| M5-J2 | Run one bounded detailed-vs-reference continuation attempt with exact fallback on refusal | complete scoped negative — separate reference-feature build terminates twice with footer, 12 replay instructions, 0 macro dispatches, and exact fallback |
| M5-J3 | Reconcile dispatch/frontier results and preserve the broad native-continuation boundary | complete scoped — replay/reference artifacts are byte-identical; C6 one-warp positive remains distinct; multiwarp/global continuation stays open |

Evidence: `docs/sm86_m5j_multiwarp_continuation_boundary_v1.json`. This closes
only the second-candidate negative branch; it is not a broad native-continuation
or hardware-scheduling result.

### M5-K — accuracy/continuation selection after scoped negatives — complete scoped

The scoped physical-tail and multiwarp negative branches are now explicit. The
next selection must choose either a genuinely disjoint accuracy family or a
new typed continuation candidate; no re-run of an already closed negative is
counted as progress. The selected candidate must have an independent source
contract and a falsifiable completion bar.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-K1 | Enumerate remaining accuracy/typed-continuation candidates and choose one evidence-supported seam | complete — FIB/problem005 CUTLASS M=80 source row selected; fresh same-extension M=1/M=32 calibration and disjoint target trace |
| M5-K2 | Execute one bounded candidate path with exact fallback or pre-NCU prediction ordering | complete — prediction 242096.2 with frozen 242005.0..242187.4 interval before target NCU; target correctness, NVBit, replay, and five NCU runs pass |
| M5-K3 | Synchronize the frontier and retain all hardware/physical blockers | complete scoped negative — 0/5 interval coverage, median 241722, point APE 0.155%; miss retained without refit/widening; tails open |

Evidence: `docs/sm86_m5k_fibp005_m80_prediction_pre_ncu_v1.json` and
`docs/sm86_m5k_fibp005_m80_holdout_v1.json`. This is a single within-family
negative holdout, not broad SOL accuracy.

### M5-L — evidence-synthesis selection — complete scoped

The current bounded slices now include one positive narrow scalar continuation,
multiple exact-fallback negatives, a TF32 capability repair, physical-tail
negatives, and two observer holdouts (one positive, one negative). Select the
next seam only after separating these frontiers; no aggregate score may promote
unfinished hardware completion, scheduling, or broad accuracy.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-L1 | Reconcile the scoped positive/negative observer and continuation evidence into distinct frontiers | complete — observer, issue/replay, modeled completion, typed continuation, population, and physical-tail frontiers remain separate; next candidate is C7-S34CQ retry |
| M5-L2 | Select one genuinely new evidence-supported candidate or document the strongest remaining blocker | complete scoped negative — one 450-second post-TF32 retry reproduces timeout before dynamic_trace.pb; no replay/NCU substitute |
| M5-L3 | Synchronize claims and keep exact fallback/physical-tail boundaries explicit | complete — retry is retained as a blocker; further duplicates require a new tracer/execution mechanism; no completion/accuracy claim changed |

Evidence: `docs/sm86_m5l_c7_s34cq_retry_v1.json`. The timeout is a
capture/execution blocker, not a replay or simulator capability result.

### M5-M — remaining-blocker consolidation — complete scoped

The evidence-supported executable seams are now exhausted for this host without
new external execution mechanisms. Consolidate the remaining blockers and keep
the next work bounded: physical completion/ownership, broad accuracy, the two
remaining capture timeouts, and the SM120-only row. No duplicate timeout or
aggregate-count relabeling may close them.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-M1 | Enumerate remaining blockers and map each to its strongest current evidence | complete — physical completion/ownership, broad accuracy, multiwarp continuation, two capture timeouts, and SM120 architecture boundary mapped |
| M5-M2 | Check whether a local code/test seam can falsifiably reduce one blocker without synthetic hardware facts | complete scoped negative at its cutoff — local tests/oracles passed; M5-O subsequently repaired the exact IMMA seam, while ownership/completion, new tracer execution, broad-family labels, and SM120 remain open |
| M5-M3 | Synchronize final scoped claims and leave the goal active for any external-state change | complete scoped — no physical, scheduling, broad-accuracy, or native-continuation completion claim added |

Evidence: `docs/sm86_m5m_blocker_consolidation_v1.json`. This is a scoped
state map, not an authority or execution gate.

### M5-N — external-evidence residuals — active

The remaining objective is now constrained by external state rather than an
unexamined local seam. Keep the repository runnable and the claims explicit;
resume a bounded M5-N slice only when a new tracer/execution mechanism,
ownership/completion observation, disjoint accuracy family, or SM120 host is
actually available.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-N1 | Keep external-state dependencies and exact next checks explicit | complete scoped — stale C7 active labels corrected; remaining dependencies are physical ownership/completion, new tracer execution, disjoint accuracy evidence, and SM120 hardware |
| M5-N2 | Preserve focused local regressions and current replay evidence | complete — fixed the stale compile-manifest test invocation to exercise the bound artifact lane; focused suite 40 passed / 703 subtests, Rust suites and replay evidence remain green |
| M5-N3 | Do not close the objective until physical/broad/native requirements gain evidence | pending external evidence — current host audit, focused Rust/Python suites, and schema/link checks are green; requires external ownership/completion, disjoint accuracy, tracer, or SM120 evidence |
| M5-N4 | Audit remaining tensor/vendor/capture/continuation surfaces for any additional local SM86 seam | complete historical cutoff — the audit identified the later-resolved IMMA.16832 seam; sparse IMMA, vendor, capture, physical-tail, accuracy, and SM120 residuals remain |

### M5-O — exact INT8 IMMA pipeline capability — complete scoped

Resolve the retained SM86 `IMMA.16832.S8.S8` replay refusal only if an
independent shape-specific probe supports an interval. Do not inherit the
m16n8k16 INT8 value or use the scored SimBench row to choose it.

| Sub-milestone | Completion bar | State |
|---|---|---|
| M5-O1 | Build a numerical no-memory IMMA.16832 probe and collect independent dependent/independent observations | complete — 1024-iteration arms, 1024 per-lane accumulator checks, seven oracle trials and two zero-error memchecks; independent arm supports 16-cycle initiation |
| M5-O2 | Add exact SM86 opcode capability with unsupported-shape/SM120 negatives | complete — `imma_16832_s8=16` and exact classifier; related forms and SM120 remain fail-closed |
| M5-O3 | Replay retained INT8 trace and synchronize tails/claims | complete scoped — 432 records replay twice byte-identically; lower 2848, GPC observation 4640; dirty-writeback/coherence remain open |

Evidence: `docs/sm86_m5o_imma16832_pipeline_replay_v1.json`.

### M4-E result — independent closeout recheck — complete

The independent recheck found and repaired three reproducibility defects before
accepting the M4 closure. The barrier and L1TEX matrix campaigns had retained
cell records but their declarative `metadata/matrix.json` files were absent;
the campaigns' manifest-only actions regenerated those files without rerunning
workloads, and the retained cell keys match exactly (33/33 and 100/100).
The bypass v2 record pointed at a stale correctness stdout name; it now points
to the retained `correctness_0.stdout` file. The SFU probe could not rebuild in
the ambient GCC 16 environment even though CUDA 13.2 requires a supported host
compiler; its Makefile now pins `/usr/bin/g++-14`, the full target rebuild
passes, and a fresh v2 vertical is retained in
`docs/sm86_c7_research_probe_sfu_frontier_live_v2.json`.

All 37 current Makefile-backed probe targets now return success under the
bounded rebuild loop. The fresh SFU binary passed seven correctness trials,
full-shape Compute Sanitizer, NVBit capture, independent tail-fact extraction,
deterministic replay, and five application-replay NCU repetitions. Its
simulator lower bound is `38099` cycles versus a fresh NCU median of `34257`,
so the lower-bound falsifier remains explicit and no timing value was fitted.
The inventory has 42/42 directory links, 37 live Makefile artifacts, and five
explicit non-native dispositions; every live result/card path resolves and
135 document/schema pairs validate. This closes M4-E and confirms the M4
population closure without promoting any semantic tail, hardware completion,
or native dispatch claim.

### Remaining work

The active phased continuation is at the top of this plan. The earlier sequence
of capture milestones is not a claim that every research mechanism is complete.
After M3, source-review the remaining population rather than inventing more
single-cell milestones solely to enlarge a completed-stage count.
