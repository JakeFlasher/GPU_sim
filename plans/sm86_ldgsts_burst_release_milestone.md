# SM86 LDGSTS barrier-free issue-cliff plan

## Goal Description

Build and execute one prospective RTX 3090 Ti microbenchmark that separates
the issue envelope of a barrier-free LDGSTS burst from its post-wait completion
envelope. Cross burst depth around the published five-position boundary with
one/four-warp placement, admitted L1/L2/DRAM routes, and predeclared 16/64-cycle
spacing sentinels. Reach one terminal interpretation without fitting a timing
constant or adding simulator state.

The experiment tests whether GA102 LDGSTS issue/spacing response is route
invariant, route sensitive, downstream-only, placement-associated, absent, or
ambiguous. It cannot observe a physical queue-entry release event.
It does not repair the rejected `ampere_queue_v1` composite. NCU admits route,
traffic, destination work, and SMSP placement; same-kernel paired `clock64`
effects decide the mechanism.

## Acceptance Criteria

- AC-1: One fixed target binary exposes separate, trustworthy issue and drain
  endpoints without a CTA barrier in the timed device body.
  - Positive Tests (expected to PASS):
    - One non-inlined device body has eight static independently predicated
      `LDGSTS` positions, one commit, one wait, no CTA `BAR` in that body,
      fixed resources, and runtime depth/route/placement/gap selection. A
      non-unrolled arm loop calls that one body twice rather than duplicating it
      in SASS.
    - The issue-end clock occurs after the eighth static position and before
      commit/wait; the drain-end clock occurs after `wait_group 0`.
    - Enabled and all-predicated-off arms execute identical active-warp and gap
      schedules in the same kernel. AB/BA order is balanced across burst blocks;
      only the first `N` contiguous LDGSTS predicates differ.
    - Warps at or above `active_warps` uniformly skip the target before clocks.
      Each active warp records stable pre/post `%smid`, separate issue/drain
      samples, and achieved gap samples.
    - On/off arms use disjoint sentinel-initialized shared destinations. Copied
      values are checked after timing and the off destination remains unchanged.
    - Fixed CTA barriers rendezvous all 128 threads only at complete logical-
      cell route transitions and the W1→W4 transition, outside every timed
      interval and outside the audited device body. There is no barrier between
      enabled/off arms or bursts; the device body contains zero `BAR`.
    - Achieved clock-spin delay is retained per edge. For every P/S/K contrast,
      repetition-level achieved-delay medians are stable and operationally
      equal across on/off arms, depths 4/5/6, and L1/DRAM routes.
  - Negative Tests (expected to FAIL):
    - Static audit rejects a CTA barrier inside the device body, extra/missing LDGSTS/commit/
      wait, local-memory spill, unknown control encoding, wrong SASS order, or
      any issue/gap clock carrying an LDGSTS dependence-wait mask. Required
      order is eighth LDGSTS → issue clock → commit → wait → drain clock.
    - Divergent lane-level commit/wait, overlapping destinations, unaligned
      sources, or same-line source aliases reject the construction.

- AC-2: The exact 40-configuration matrix and evidence split are immutable and
  resumable with ordinary campaign controls.
  - Positive Tests (expected to PASS):
    - Identification contains W1 × L1/DRAM × depth 1/4/5/6/8 at gap zero.
    - Spacing sentinels contain W1 × L1/DRAM × depth 4/5/6 × gap 16/64.
    - The unopened route holdout contains W1 × L2 × depth 1/4/5/6/8.
      Placement diagnostics contain W4 × L1/L2/DRAM × depth 4/5/6 at gap zero.
    - W4 is a predeclared placement diagnostic, not an identification holdout;
      L2 is the sole unopened route holdout.
    - The clock-assay control contains W1 L1 independent LDG depth 4/5/6/8.
    - Every configuration contains enabled/off arms, exactly 16,384 measured
      bursts after fixed warmup, and seven raw paired process repetitions.
  - Negative Tests (expected to FAIL):
    - Duplicate/missing/extra keys, post-result depth/gap insertion, route-name
      branches in SASS, using L2 to select an identification label, or using W4
      to relabel the locked W1 identification are rejected.

- AC-3: Live evidence admits exact work, cache route, and placement before any
  issue-cliff claim.
  - Positive Tests (expected to PASS):
    - All configurations pass seven-of-seven correctness with retained raw
      paired samples. Every repetition-level L1/DRAM or W1/W4 contrast is
      collected inside one CTA on one unchanged `%smid`; unmatched SM samples
      are retained but excluded and make the contrast inadmissible.
    - Three NCU repeats per configuration retain the exact static executed-
      instruction count, enabled source sectors, and destination wavefronts;
      two additional
      repeats are added only when the first-three active-cycle CV exceeds 2%.
    - For `Q=16*N*active_warps*16384` 32-byte source sectors, L1-dominant means
      `H1/(H1+M1) >= .95`; L2-dominant means `M1/Q >= .95` and
      `H2/(H2+M2) >= .95`; DRAM-dominant means `M1/Q >= .95` and
      `M2/(H2+M2) >= .95`. Prewarm is outside the profiled target.
    - Matched DRAM versus L2 cells have positive excess DRAM reads of at least
      `0.5*Q`; DRAM reads corroborate the dominant route but are not required to
      equal LTS misses.
    - Equal depth/placement cells retain equal enabled LDGSTS, source-sector,
      and destination-wavefront work across routes: source sectors equal `Q`
      and destination wavefronts equal `4*N*active_warps*16384`.
      Predicate-ignorant executed LDGSTS count remains the static
      `16*active_warps*16384`; depth is admitted from traffic and the off-arm
      sentinel, not from that executed count.
    - Direct mode writes one paired record only after both timed arms using a
      cache-streaming/write-through output store. NCU mode runs one logical
      route/depth/placement cell per launch and writes only compact final
      correctness/SM state; it emits no per-burst sample stream.
    - Raw NCU import uses `--print-metric-instances details`. W1 has exactly one
      nonzero SMSP instruction instance; W4 has exactly four on one correlated
      SM with `(max-min)/mean <= .01`. Repeat 1 of W1/W4 depth-5 L1/L2/DRAM
      cells is the preflight smoke for predicate-ignorant instruction count,
      `Q`, route ratios, and instance attribution before remaining NCU cells;
      lack of instance support invalidates placement evidence.
    - The host records/restores the project fixed core clock (1350 MHz), records
      the explicit memory-clock policy, and runs application replay with
      `--clock-control none --cache-control none`.
  - Negative Tests (expected to FAIL):
    - `.ca` intent, allocation size, or prewarm alone cannot name a route.
    - Aggregate warp count alone cannot be called four-SMSP evidence; mixed
      route, placement drift, profiler replay inconsistency, or failed work
      equality, unmatched cross-configuration SM identity, or unsupported
      metric-instance attribution makes the affected contrast inadmissible.

- AC-4: The reducer implements the frozen paired effects, hard boundaries, and
  staged interpretation/diagnostic/holdout table without fitting.
  - Positive Tests (expected to PASS):
    - For repetition `j`, burst `b`, and warp slot `s`, raw issue effect is
      `dI[j,b,s]=issue_on-issue_off`; raw completion effect is
      `dC[j,b,s]=(drain_on-issue_on)-(drain_off-issue_off)`. First take the
      within-repetition burst median `e[j,s]`, then the median/MAD across seven
      repetition-level effects. Bursts and W4 warps are never pooled as
      independent replicates.
    - `P(r,g)=E(5)-E(4)`, `S(r,g)=E(6)-E(5)`,
      `K(r,g)=S-P`, and `R(n,g)=E(n,dram,1,g)-E(n,l1,1,g)` are formed within
      each matched repetition before their seven-value median/MAD is reported.
    - `stable(x)` means MAD ≤0.5; `operationally_equal(x)` means stable and
      `abs(median)<=1`; `positive_material(x)` means stable and median >2.
      Negative, unstable, and effects in the open gray band are ambiguous.
    - The audited LDG control has eight independent predicated `LDG.E.128`
      positions with unique registers, no consumer before issue-end, later
      consumption, no wait mask, and `positive_material(K_LDG(0))`. Otherwise
      the output is `invalid_clock_assay`.
    - W1 L1/DRAM evidence alone locks exactly one identification label using
      the ordered Boolean rules below. Zero or multiple matches become
      `valid_ambiguous_no_model`; construction/admission failure becomes
      `invalid_experiment`.
    - After the label is locked, W4 emits a separate placement diagnostic and
      L2 emits `confirmed`, `falsified`, `inconclusive`, `invalid`, or
      `not_applicable`; neither may relabel identification. A simulator
      successor is merely eligible only when a
      named identification, admissible/compatible placement diagnostic, and
      confirmed L2 holdout all agree.
  - Negative Tests (expected to FAIL):
    - NCU cycle labels, production residuals, workload names, L2 holdout, W4
      diagnostic, or post-hoc thresholds cannot select or replace the locked
      identification label.
    - An identification admission failure emits `invalid_experiment`, not a
      best-effort timing interpretation; later diagnostic/holdout failures stay
      local, and pooled burst/warp pseudoreplication is rejected.

- AC-5: The milestone closes with concise evidence and leaves the active
  simulator untouched.
  - Positive Tests (expected to PASS):
    - One result document records construction, exact SASS/work/route evidence,
      all raw/aggregate paired results, holdout status, terminal interpretation,
      limitations, and claim ceiling.
    - Current status/roadmap points to the terminal result and names a simulator
      successor only if identification, the placement diagnostic, and the
      unopened L2 holdout all agree.
    - Any coordination-root update only links the standalone result, updates
      the source head, and states that Jade restart, P3, M4, and G1 remain
      unchanged.
    - Focused workload/analyzer tests, build/static audit, correctness, direct
      measurements, and NCU analysis pass in proportion to the experiment.
  - Negative Tests (expected to FAIL):
    - The milestone adds no simulator timing candidate, production replay,
      default change, NVBit/tracer change, full-corpus regeneration, SHA-based
      gate, Jade progress, P3 pass, physical queue-size claim, validated-Ampere
      claim, physical release/stage-location claim, or publication-readiness
      claim.

## Path Boundaries

### Upper Bound (Maximum Scope)

One five-file workload/campaign/analyzer directory; direct CUDA correctness and
paired clocks; static SASS/resource/control evidence; exact NCU route/work/
placement evidence; the frozen 40-configuration run; terminal result and
concise GPU result/current-guidance update.

### Lower Bound (Minimum Scope)

One same-binary barrier-free target, exact matrix manifest, static audit,
seven raw direct repetitions, three-repeat NCU admission, frozen reducer, and
terminal valid/invalid/no-model result. A null or ambiguous result completes
the milestone.

### Allowed Choices

- Can use: existing async-copy locality/prewarm/correctness patterns, existing
  campaign subprocess/NCU/SASS helpers, fixed-clock direct measurement,
  application replay with cache control disabled, current control-sidecar
  extraction, and ordinary focused review.
- Cannot use: the retired LDGSTS simulator candidate, fitted queue/rate/floor
  values, production labels, software async-group count as queue size, A100/
  L40/H100/B200 constants as GA102 facts, adaptive matrix expansion, generic
  workflow machinery, or hash approval gates.
- Preserve: standalone simulator source/output, existing LDGSTS payload and
  destination behavior, current production baseline, source-repository builds,
  and the Jade no-go boundary.

## Feasibility Hints and Suggestions

### Target structure

Put eight inline, volatile `cp.async.ca.shared.global` positions with uniform
runtime predicates in one non-inlined device body. A non-unrolled arm loop
calls that body for enabled/off arms, so the function contains only eight
LDGSTS, one commit, and one wait statically while executing twice dynamically.
Compute addresses before the start clock. Warp 0 alone is active in W1; warps
1–3 skip the entire W1 phase. In placement bundles they wait at a named wrapper
barrier until warp 0 finishes W1, then all enter W4; in W1-only launches they
exit only for compact single-cell/NCU mode. In direct multi-route W1 bundles
they remain resident and wait at each complete route-cell boundary. Warp slots
0–3 are candidates in W4, but the W4 name is admitted only by instance
counters. Keep a 16 KiB shared
allocation per arm (32 KiB total) and the same 128-thread block in every
runtime cell. Fixed wrapper barriers join all threads only at complete route-
cell boundaries and the W1→W4 boundary; there is no arm/burst barrier. They
remain outside the timed/audited zero-BAR body.

For position `p`, burst `b`, and per-warp disjoint base, freeze source chunk
mapping as:

```text
l1:   index = p
l2:   index = (b + 128*p) & 1023
dram: index = 8*b + p
```

Each chunk is 512 aligned bytes. Thus every L2 position traverses the complete
512 KiB/warp ring with reuse beyond L1, while DRAM spans 64 MiB/warp and never
wraps in 16,384 bursts. On/off shared destinations are disjoint and initialized
to distinct sentinels.

Insert a gap only between consecutive first-`N` positions, and execute the
identical gap loop in both arms. One runtime-controlled clock-spin body serves
all cells. Every recorded edge must be 16–38 achieved cycles for the short
sentinel or 64–96 for the long sentinel; zero-gap bypasses the loop. Audit every
gap clock and the issue-end clock for no LDGSTS wait mask. These ranges can
distinguish short versus long relief but cannot locate the published 13/39
boundaries.

Balance AB/BA arm order within each launch. Direct route contrasts run L1 and
DRAM configurations in one CTA; L2 holdout bundles run L1/L2/DRAM likewise.
Route phase order uses a frozen balanced forward/reverse list across seven
repetitions. Immediately before every cached-route phase, the wrapper prewarms
that exact ring outside all clocks, then rendezvouses before measurement.
Placement diagnostics run matched W1 and W4 configurations in one CTA with
order balanced across repetitions. The analyzer accepts only contrast samples
whose before/after `%smid` is identical. Store raw per-burst, per-arm, per-warp
values with one write-through/streaming record after both arms. NCU mode runs
exactly one logical cell per launch, keeps only compact final correctness/SM
state, and emits no per-burst clock stream. It uses an immediate host prewarm
launch over the same ring with profiling started only for the target. Both
modes call the identical device body and use the same address/prewarm mapping.
Direct clocks are primary; NCU stall samples are explanatory only.

Serialize direct timing and NCU on the single live GPU. Every subprocess has a
bounded timeout. Clock limits are restored on success, error, or signal. Write
one cell to its final directory only after every required raw output for that
cell is complete; resume may skip only complete cells. This is ordinary
failure-safe campaign handling, not a journal, receipt, or workflow framework.

### NCU admission equations

Bind counters exactly as:

```text
Qobs = sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum
WF   = sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum
H1   = l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum
M1   = l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum
H2   = lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum
M2   = lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum
DR   = dram__sectors_read.sum
Q    = 16 * depth * active_warps * 16384
```

Require `Qobs=Q`, `H1+M1=Q`, and
`abs((H2+M2)-M1) <= max(16, Q/1000)`. Require
`WF=4*depth*active_warps*16384`. The predicate-ignorant warp instruction count
is separately frozen at `16*active_warps*16384`; it cannot identify depth.
Dominant-route ratios use these conserved denominators. DRAM corroboration is
`DR(dram)-DR(l2) >= 0.5*Q` for matched cells, not equality with M2.

### Frozen reductions

```text
dI[j,b,s] = issue_duration_on - issue_duration_off
dC[j,b,s] = (drain_end_on - issue_end_on)
              - (drain_end_off - issue_end_off)
eI[j,s]   = median_b dI[j,b,s]
eC[j,s]   = median_b dC[j,b,s]
E(s)      = median_j eI[j,s]
C(s)      = median_j eC[j,s]

P[j,r,g] = eI[j,depth5,r,W1,g] - eI[j,depth4,r,W1,g]
S[j,r,g] = eI[j,depth6,r,W1,g] - eI[j,depth5,r,W1,g]
K[j,r,g] = S[j,r,g] - P[j,r,g]
R[j,n,g] = eI[j,n,dram,W1,g] - eI[j,n,l1,W1,g]
```

The process repetition `j`, not burst `b` or warp `s`, is the independent unit.
Compute every contrast within repetition, then report its seven raw values,
median, and MAD. W4 retains four slot vectors; it is never pooled into extra
replicates. `stable(x)` is MAD ≤0.5, `op_eq(x)` is stable with
`abs(median)<=1`, `pos(x)` is stable with median >2, and `ordered(x)` is stable
with median >0. Negative, unstable, or gray-band values match no positive rule.
For diagnostic/holdout falsification, the exact opposite of required `op_eq`
is stable `abs(median)>2`; of required `pos` is `op_eq` or stable median < -2;
and of required `ordered` is stable median <=0. Other stable gray values are
inconclusive.

Apply this exact decision order before opening L2 or interpreting W4:

1. Any identification construction/work/route/clock admission failure yields
   `invalid_experiment`. Later W4/L2 admission failures mark only their own
   diagnostic/holdout invalid and cannot erase or change a locked label.
2. If `pos(K_LDG(l1,0))` is false, yield `invalid_clock_assay`.
3. On W1 L1/DRAM only, define `knee0 = pos(K(l1,0)) && pos(K(dram,0))` and
   `route_eq0 = op_eq(R(n,0))` for every `n` in 1/4/5/6/8. Evaluate:
   - `route_invariant_relief_with_short_gap`: `knee0 && route_eq0` and
     `op_eq(K(l1,16)) && op_eq(K(dram,16))`;
   - `route_invariant_relief_only_with_long_gap`: `knee0 && route_eq0`, both
     `K(*,16)` positive, and both `K(*,64)` operationally equal;
   - `route_sensitive_issue_response`: `knee0`, `R(1/4/5,0)` operationally
     equal, `R(6/8,0)` positive, `K(l1,16)` operationally equal, and
     `K(dram,16)` positive;
   - `no_material_ldgsts_issue_knee`: both route `K` values operationally equal
     at gaps 0/16/64 and `route_eq0`.
4. Exactly one true rule locks `identification_label`; zero or multiple rules
   yield `valid_ambiguous_no_model`. Drain-effect route order is reported as a
   separate descriptor and never chooses the label.
5. Open W4 as a placement diagnostic. `invalid` has first precedence on W4
   admission failure; `not_applicable` follows for invalid/ambiguous locked
   identification. Otherwise retain every slot `s`, compute K for routes
   L1/L2/DRAM, and define adjacent route effects `A=E_l2-E_l1` and
   `B=E_dram-E_l2` at depths 4/5/6:
   - route-invariant labels require `pos(K[s,r,0])` for all three routes and
     `op_eq(A),op_eq(B)` at all three depths;
   - route-sensitive requires positive knees, `op_eq(A),op_eq(B)` at depths
     4/5, `ordered(A),ordered(B)` at depth 6, and positive material total
     DRAM-minus-L1 at depth 6;
   - no-knee requires `op_eq(K[s,r,0])` for all routes and `op_eq(A),op_eq(B)`
     at all depths.
   Report `compatible` iff every predicate holds for all four slots.
   `placement_associated` takes precedence when all required quantities are
   stable but a material effect appears only at W4 or stable slot classes
   disagree. `falsified` means all required quantities are stable and at least
   one has the opposite predeclared class; otherwise `inconclusive`. Never
   relabel.
6. Open L2 last. `invalid` has first precedence on L2 admission failure;
   `not_applicable` follows for invalid/ambiguous identification. Define
   `A(n)=E_l2-E_l1` and `B(n)=E_dram-E_l2`:
   - route-invariant labels require `pos(K(l2,0))` and `op_eq(A),op_eq(B)` at
     depths 1/4/5/6/8;
   - route-sensitive requires `pos(K(l2,0))`, `op_eq(A),op_eq(B)` through depth
     five, `ordered(A),ordered(B)` at depths six/eight, and the already-
     material total DRAM-minus-L1 effect;
   - no-knee requires `op_eq(K(l2,0))` and `op_eq(A),op_eq(B)` at all depths.
   Report `confirmed` iff every predicate holds. `falsified` means every
   required quantity is stable and at least one has the opposite predeclared
   class; otherwise `inconclusive`. Never relabel.

Only a named non-null identification plus compatible placement and confirmed
L2 is `successor_eligible`; this merely permits a separate simulator plan.
Report all values even when any gate fails.

### Relevant references

- Research decision:
  `/home/jakeshea/Jade_HETR/research/sm86_ldgsts_burst_release_research_2026-08-28.md`
- Retired-composite decision:
  `results/captures/sm86_ldgsts_front_end_transfer_20260828_v1/README.md`
- Existing scaffold: `workloads/research_probes/async_barrier_overlap/`
- Prewarm/stream-L2 pattern:
  `workloads/research_probes/barrier_pipeline/coupled_campaign.py`
- Depth matrix pattern:
  `workloads/research_probes/fixed_latency_result_capacity/`
- SMSP convention:
  `workloads/research_probes/smsp_scheduler_contention/`
- Huerta et al. paper/code:
  `GPU_simulator_rebase/.toolchain/modern-gpu-simulator-micro-2025`
- GASS comparator:
  `/home/jakeshea/Jade_HETR/research/external_code/gass_ldgsts_20260828`
- Blackwell pattern-only clone:
  `/home/jakeshea/Jade_HETR/research/external_code/microbench_blackwell_20260828`

## Dependencies and Sequence

### Milestones

1. Milestone M1 — Fixed assay and manifest.
   - Add the five-file sibling probe, exact 40 keys, target/assay kernels, raw
     output schema, and small analyzer fixtures.
2. Milestone M2 — Static and local admission.
   - Build once, inspect resources/SASS/control metadata, exercise small
     correctness, and close all target-body invariants before the full run.
3. Milestone M3 — Direct prospective measurement.
   - Run seven paired raw repetitions for the frozen matrix and retain both
     issue/drain endpoints, correctness, ordering, gaps, and SM identity.
4. Milestone M4 — NCU evidence admission.
   - Run three repeats per cell, conditionally extend high-CV cells to five,
     admit route/work/SMSP evidence, lock identification before opening the W4
     diagnostic, and open the L2 holdout last.
5. Milestone M5 — Terminal reduction and handoff.
   - Apply the frozen rules, record valid/invalid/ambiguous outcome, update
     current guidance, and stop without simulator work.

Dependencies: M2 depends on M1. No full live matrix starts until M2 passes. M3
then M4 run serially on the live GPU; only implementation and offline analysis
may be parallelized. M5 depends on both M3 and M4.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---|---|---|---|---|
| task1 | Add fixed target/LDG assay, build, and exact manifest | AC-1, AC-2 | coding | - |
| task2 | Add narrow campaign runner and raw schema | AC-2, AC-3 | coding | task1 |
| task3 | Add frozen reducer and boundary/admission fixtures | AC-3, AC-4 | coding | task2 |
| task4 | Complete static SASS/resource/control admission | AC-1, AC-3 | analyze | task1 |
| task5 | Execute direct correctness and seven paired repetitions | AC-3 | analyze | task4 |
| task6 | Execute NCU route/work/placement matrix and variance policy | AC-3 | analyze | task4, task5 |
| task7 | Lock identification, reduce W4 diagnostic/L2 holdout, and write terminal guidance | AC-4, AC-5 | analyze | task5, task6 |

## Implementation Notes

- Do not embed `AC-*`, task, milestone, interpretation, or route-result names
  into the target SASS path.
- Keep the runner direct and small; reuse existing helpers without introducing
  lifecycle abstractions or repository-wide validation.
- Run ordinary checks only. Do not compare hashes or rerun unrelated corpora.
- Preserve raw paired observations before reduction and report failures as
  technical outcomes, not workflow permissions.
