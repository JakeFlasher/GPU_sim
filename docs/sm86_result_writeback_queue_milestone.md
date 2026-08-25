# Next milestone: sm_86 fixed-latency result, bypass, and writeback queue

**Status:** fixed-over-L1TEX priority and three-cycle uniform backward-target resolution promoted; bypass identified, IBuffer capacity unresolved 2026-08-17  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[reuse cache and result/writeback queue](sm86_reuse_result_queue_milestone.md)  
**Evidence:**
`results/captures/sm86_fixed_latency_result_queue_20260817_v1/analysis/README.md`,
`results/captures/sm86_fixed_latency_result_order_20260817_v1/analysis/README.md`, and
`results/captures/sm86_fixed_latency_result_same_bank_20260817_v1/analysis/README.md`,
`results/captures/sm86_fixed_latency_result_capacity_20260817_v1/analysis/README.md`, and
`results/captures/sm86_fixed_latency_result_load_priority_20260817_v1/analysis/README.md`, and
`results/captures/sm86_fixed_latency_result_bypass_20260817_v1/analysis/README.md`, and
`results/captures/sm86_fixed_latency_producer_shift_20260817_v1/analysis/README.md`, and
`results/captures/sm86_fixed_latency_front_end_footprint_20260817_v1/analysis/README.md`

## Goal

Separate four mechanisms after fixed-latency execution:

1. bypass availability to a dependent fixed-latency consumer;
2. bounded result buffering when several execution pipes complete together;
3. one regular-register write port per physical bank; and
4. fixed-latency versus variable-latency writeback priority.

The result must explain exact destination/completion treatments and transfer to
held-out widths and predecessor campaigns. It must not be a destination-bank,
opcode, queue-depth-protocol, or warp-count correction.

## Established boundary

Static source/destination/reuse projection and three live campaigns now cover
register-bank input demand, consumer distance, and compiler-managed RFC state.
The RFC campaign contains 40 cells, 280 correctness executions, 120 NCU
profiles, 16 SourceCounters reports, 40 NVBit captures, and 518,400 replay
records. It supports shared bank/source-position entries tagged by warp and
register, but rejects all active timing placements after predecessor transfer.

Those campaigns cannot identify result buffering: paired variants deliberately
keep their destination streams and fixed instruction latencies equal. The next
probe must vary completion coincidence and destination banks while holding
source reads, RFC state, executed-pipe work, controls, and values fixed.

## Completed opposite-bank pacing pilot

The first `fixed_latency_result_queue` pilot is complete. Its checked cubin
pair changes only the stall field at four HADD2 PCs: `collision` uses stall 1
and `separated` uses stall 3. Both treatments canonicalize yield/reuse to zero;
all other instructions, operands, predicates, barriers, wait masks, resource
declarations, runtime values, and the host oracle are equal. Six spill-free
cubins decode 384/384 controls, and exact projection covers 108 fixed-latency
instructions with zero gaps, including 24 HADD2 records.
The expanded retained corpus now covers 48 sidecars, 39,080 records, and 16,421
fixed-latency projections with zero gaps.

The full matrix contains eight W4/W8/W12/W16 cells, 56 correctness executions,
56 base-clock NCU profiles, four SourceCounters profiles, eight NVBit captures,
and 84,480 replay records. All pair instruction/pipe counts match and every
issue ratio is 1.0. The separated-minus-collision live deltas are
`+4.000/-1.014/-1.048/-1.791` cycles/generation at W4/W8/W12/W16, while
production replay predicts `+8.000/+4.000/+0.953/+8.062`. At W8 and above,
extra spacing reduces math-pipe, dispatch, and not-selected pressure; this is
the scheduler/pipe crossover already exposed by the predecessor campaign.

This pilot cannot identify a result queue. All four HADD2/FFMA destination
pairs target opposite register banks, only three producer neighborhoods are
instruction-adjacent, and the control treatment changes issue pacing.
Production treatment-delta MAE is 5.217 cycles/generation. No queue, bypass,
write-port, or priority timing is activated. Exact HADD2 projection, the
control-word patcher, and the live pacing evidence are retained.

## Completed balanced producer-order slice

The `fixed_latency_result_order` pair removes the pacing confound. Its two
48-instruction, 15-register functions are byte-identical outside two adjacent
producer PCs. `hf` issues HADD2 then FFMA with one-cycle spacing so their
five-/four-cycle results complete together; `fh` reverses the opcodes and
separates completion by two cycles. Destinations remain R11/R4 (opposite
banks), controls and consumers match, and replay predicts an exact zero delta.

The 14-cell W4/W8/W12/W13/W14/W15/W16 campaign contains 98 correctness
executions, 110 NCU profiles, eight SourceCounters profiles, 14 NVBit captures,
and 77,900 replay records. HF-minus-FH is `0.000/-0.001/+0.056` at W4/W8/W12.
W13-W15 localize a separate bimodal phase: a slow HF regime appears in 0/7,
1/11, and 1/11 profiles, then 8/11 at W16. W13 already places a fourth warp on
one modeled scheduler partition, so W16 does not identify a four-warp
SMSP-local queue threshold; it is retained only as a global phase sentinel.

## Completed same-bank two-result slice

`fixed_latency_result_same_bank` achieves the required layout entirely through
compiler allocation. Both functions have 48 instructions, 16 registers,
destination stream R4/R8 (bank 0), matching per-opcode source-bank demand,
byte-identical timed consumers, and identical canonical controls. Only HADD2/
FFMA order differs before the stop clock; the checked oracle difference after
the clock is a commutative source permutation. The patch changes two yield
bytes and no register field.

Its eight cells contain 56 correctness executions, 56 NCU profiles, eight
SourceCounters profiles, eight NVBit captures, and 37,920 replay records. At
W4/W8, same-bank HF-minus-FH is `-0.0005/+0.0002` cycles/generation and all
stall/count deltas are null. Hardware therefore accommodates and forwards two
same-cycle fixed-latency results targeting one bank without exposed delay at
one or two warps per scheduler. This independently supports result buffering/
bypass, but a null does not reveal physical queue entries and adds no timing.

Held-out W12 exposes `+0.798` cycles/generation, or `+0.742` beyond the matched
opposite-bank delta, with dispatch/wait movement and zero short-scoreboard
delta. It cannot yet distinguish queue occupancy/pop width from per-bank
retirement. W16 remains excluded from fitting because of the global phase.
The retained corpus now covers 56 sidecars, 39,848 records, and 16,597 exact
fixed-latency projections with zero gaps.

## Completed reservation-capacity feasibility slice

Primary-source inspection corrected the capacity abstraction: the reference
model reserves a regular-result slot at fixed-instruction issue and frees it
after writeback; its configured values are eight slots and one pop per cycle.
`fixed_latency_result_capacity` therefore uses one admitted HMMA.16816 anchor
and a 17-result issue run. A common stall-15 BAR drains prior work, while an
identical stall 15 moves after reservation 7, 8, or 9. Instruction work,
static stall sum, values, and baseline terminal issue time remain equal.

The nine-cell W1/W4/W8 campaign contains 63 correctness executions, 63 NCU
profiles, six SourceCounters profiles, nine NVBit captures, and 61,269 replay
records. W1 and W4 have null b9-minus-b8 deltas. W8 shows
`b8−b7=+0.164` and a dominant `b9−b8=+1.126` cycles/generation with increased
dispatch pressure, while production replay predicts `−1.969` for b9−b8.
This is a useful multiwarp ninth-reservation canary but not a clean physical
queue-size measurement because b8 is not exactly null and Tensor/BAR phase is
part of the construction.

An experimental capacity-8/pop-1 issue-reservation implementation then changed
none of 18 replay outputs. It was removed without predecessor transfer;
production and all 18 restored outputs are byte-identical. Queue entry count
and pop width remain unidentified. The retained corpus expands to 64 sidecars,
40,424 records, 16,805 exact projections, and zero gaps.

## Completed fixed-over-L1TEX priority slice

`fixed_latency_result_load_priority` prewarms one lane-local address, then
issues a scalar `LDG.E.STRONG.SM` and HADD2 every generation. The load-only
consumer carries write-barrier 2's wait mask; a separate fixed-result consumer
is delayed 15 cycles, so the oracle never depends on the arbitration being
measured. HADD2 targets R9. The compiler-native load targets R11 (same bank),
while a complete producer/sole-consumer rename to compiler-declared unused R14
creates the opposite-bank control. Each timing pair changes only the IADD3 and
NOP stall bytes, moving HADD2 from relative cycle 22 to 23 while leaving the
load consumer at cycle 29 and the selected stall sum at 46.

The 20-cell W1/W4/W8/W12/W16 campaign contains 140 correctness executions, 60
base-clock NCU profiles, eight SourceCounters profiles, 20 NVBit captures, and
88,396 replay records. Maximum active-cycle CV is 1.424%. Every profile reports
exactly 4,096 timed L1 lookup hits and eight setup misses per warp, with equal
instruction/pipe counts and issue ratio 1.0. At W1/W4 the same-bank collision
adds exactly `+1.000` unprofiled cycles/generation while the matched
opposite-bank shift is exactly zero. W1 NCU difference-in-differences is
`+0.990`; same-bank long-scoreboard pressure rises by about `0.1245`, versus
about `0.001` for the opposite-bank shift.

Replay now preserves the literal STRONG.SM token but correctly routes that
CUDA-13.2 sm_86 `ld.global.ca` lowering through L1. The promoted narrow state
tracks barrier-observed scalar L1-hit writebacks by SMSP and regular-register
bank. An exact HADD2 result arrives six issue cycles after dispatch; if it
matches the load's writeback cycle, HADD2 wins and load visibility moves one
cycle. Readiness schema 4 reports only observed collisions and deferral cycles.
The raw same-pair delta MAE across all five widths falls from `1.058` to `0.305`
cycles/generation. W8+ also exposes a generic producer-shift phase in the
opposite-bank control; W16 remains the prior global sentinel rather than a new
priority parameter.

All 40 flat outputs, 234 outputs across Allocate/completion/RFC/result/capacity
predecessors, and 12 six-workload production outputs remain byte-identical.
Production stays at 6/6 coverage and 33.128% MAPE. Queue capacity and fixed/fixed
retirement width remain unidentified.

## Completed fixed-result bypass-frontier slice

`fixed_latency_result_bypass` reproduces the paper's Listing 3 distinction with
host-checked values. A loop-carried LOP3 producer followed by a fixed LOP3
consumer fails all seven distance-3 checksum trials and is correct at distance
4. A separate single-shot canary builds the high half of a global-load address
with fixed IADD3: distance 4 produces `CUDA_ERROR_ILLEGAL_ADDRESS` in seven
fresh contexts, while distance 5 passes seven trials. Fixed results are thus
available to fixed consumers one cycle before a variable global instruction
may safely read the corresponding register value.

The valid 20-cell W1/W4/W8/W12/W16 campaign compares fixed d4/d5 and global
d5/d6 while moving the extra stall cycle to a later checksum instruction.
Instruction work and selected stall sums are equal within each pair. It adds
140 correctness executions, 60 NCU profiles, eight SourceCounters profiles, 20
NVBit captures, and 83,558 replay records; maximum CV is 1.545%. W1/W4 driver
deltas are null within 0.001 cycles/generation. W8+ again enters the generic
producer-shift phase, including sign changes that production replay does not
transfer.

Exact `IMAD.IADD` is now projected as a four-operand FMA-family fixed form.
Thirteen cubins decode 1,352/1,352 controls and project 169 instructions with
zero gaps; the retained corpus reaches 87 sidecars, 42,256 records, and 17,078
projections. No bypass timing or hardware scoreboard is added. The semantic
boundary validates compiler schedules, while trace replay does not execute
register values and must not repair deliberately invalid code by inventing a
scoreboard.

## Completed dense producer-shift localization

`fixed_latency_producer_shift` reuses the valid fixed d4/d5 and global d5/d6
pairs at every width W4–W16. Physical warp slots map to SMSPs by `slot % 4`, so
W5/W9/W13 add the second/third/fourth warp to SMSP0 and the next three widths
fill SMSP1–3. The 52-cell campaign contains 364 correctness executions, 160 NCU
profiles, 32 SourceCounters profiles, 52 NVBit captures, and 264,940 replay
records. Only `w10_global_late` required expansion; the final maximum CV is
1.599%.

The phase begins at W5, when only SMSP0 receives a second warp, rather than at
the all-partition W8 boundary. Fixed NCU deltas remain about +1.42
cycles/generation across W5–W8, while driver deltas vary with the critical warp.
The global pair changes sign across partial/full second-, third-, and
fourth-warp cohorts. Dispatch and scoreboard movement is near zero in most
fixed cells. This excludes queue capacity, result-bank pressure, and one global
warp threshold as common explanations; the remaining locus is per-SMSP
instruction-frontier/fetch phase.

Ten bounded reference-inspired models combine IBuffer capacities 1/2/3/4/8,
one fetch/decode per SMSP per cycle, greedy/highest-warp selection, and
stop/continue fetch at an unresolved backward branch. The best capacity-4
continue model reduces driver delta MAE from 2.115 to 0.904 and NCU-active MAE
from 2.161 to 0.799, but overprices fixed W9–W12 and misses the positive global
W13–W16 regime. Selecting a capacity by cohort or instruction family would be
a protocol/warp-count correction, so no front-end timing is promoted. The next
probe removes the branch confound and varies static body footprint.

## Required projection increment

The first target, exact `HADD2`, is implemented and live-audited. Its
source/destination width, regular register files, reuse agreement, and bank
demand are explicit. `HADD2_32I` and any modifier not exercised and validated
remain typed unavailable. Existing scalar/IMAD/FFMA projection and all
retained predecessor sidecars remain gap-free.

For every admitted producer, retain:

- execution family and characterized fixed latency;
- regular destination register span and bank;
- regular source identities/positions and reuse bits; and
- the compiler stall/yield/dependence control word.

## Exact probe family

Add `workloads/research_probes/fixed_latency_result_queue/`. Use runtime-loaded
duplicate values and host-checked consumers so cubin destination/source patches
do not change results. The timed region has no CTA barrier, atomic, local spill,
or unmodeled memory except the explicit write-priority sentinel.

| Protocol | Identification target |
|---|---|
| `v00` | loop/control baseline |
| `v09a/v09b` | completed opposite-bank stall-1/stall-3 pacing pilot |
| `v10/v11` | completed opposite-bank HADD2→FFMA / FFMA→HADD2 order pair |
| `v20/v21` | completed two-result same-bank / opposite-bank completion pair |
| `v30/v31` | completed seven/eight issue-reservation feasibility pair |
| `v32/v33` | completed eight/nine canary; capacity-8/pop-1 candidate rejected |
| `v40/v41` | completed dependent consumer at bypass time / after register-file retirement |
| `v50/v51` | completed fixed result colliding with an L1-hit load writeback / noncolliding control |

Queue-depth pairs must hold total instructions and pipe work equal by moving
one producer outside the simultaneous-completion cohort rather than deleting
it. Destination-bank pairs patch producers and their consumers together. The
load sentinel first proves an L1 hit and identical L1TEX traffic; it cannot
infer priority from a broad scoreboard percentage.

All admitted pairs must remain compiler-safe. A temporary CUDA-13.2 CUAssembler
round-trip was rejected: after adapting the new ELF ABI and note sections, its
incomplete sm_86 repository still could not reconstruct the kernel's
`LDG.E [R*.64]` form. The production experiment therefore cannot depend on a
whole-cubin assembler. The successful same-bank construction instead uses
native in-place allocation plus pre-clock padding to equalize resources. Any
future bounded hot-word patch must first prove a semantic register renaming,
complete live-range coverage, and byte scope; the earlier unsafe destination
permutations remain excluded.

## Identification sequence

1. Completed: balanced opposite-bank producer order at W4/W8, with W13-W16
   global-phase localization.
2. Completed: compiler-native same-bank R4/R8 order pair and two-result bypass
   lower bound at W4/W8.
3. Completed: equal-work 7/8/9 issue-reservation feasibility and rejected
   capacity-8/pop-1 simulator candidate; retain W8 b9 only as a canary.
4. Completed: fixed versus L1-hit load writeback collision/noncollision
   (`v50/v51`), exact L1 traffic, same/opposite-bank control, and promoted
   fixed-over-variable priority.
5. Completed: consumer-distance bypass (`v40/v41`) separates fixed distance 4
   from variable-global distance 5 without adding active timing.
6. Completed: dense W4–W16 localization places the generic shift in a
   per-SMSP instruction-frontier phase and rejects ten direct front-end models.
7. Completed: U1/U2/U4/U8 branch-frequency/static-footprint factorial
   identifies three uniform backward-target resolution cycles and rejects an
   IBuffer-capacity promotion.
8. Completed: paper-style branch-free CLOCK timeline leaves capacity two as
   the sole candidate but rejects the complete fetch/cache state.
9. Next: returning-atomic dependency completion and same-line serialization.

## Branch-frequency/static-footprint result

The `fixed_latency_front_end_footprint` campaign uses loop-carried dependencies
and a nonconvergent checksum so CUDA retains exactly U1/U2/U4/U8 measured
bodies between uniform backward branches. Fixed d3 fails deterministically and
d4/d5 remains valid. The new global register allocation makes d4 valid, so
global d5/d6 is retained only as an equal-work transfer holdout.

The complete W4/W5/W8/W9/W13/W16 matrix contains 96 cells, 672 correctness
executions, 320 general NCU profiles, 96 direct front-end profiles, 32
SourceCounters profiles, 96 NVBit captures, 381,480 records, and 96
byte-identical replay pairs. Forty spill-free cubins decode 4,400/4,400
controls and project 1,190 fixed operands with zero gaps.

Hardware executes exactly `1024/U × warps` uniform branch targets with zero
divergent targets. Direct counters measure 3.013–3.167 branch-resolving
warp-cycles per target (median 3.052). Treatment differences instead appear in
`no_instruction`, localizing the remaining phase to fetch selection / decoded
instruction availability. Twenty-four candidates identify stop-at-branch and
three resolving cycles with 0.0188 branch-counter MAE, but the best IBuffer
capacity is one rather than the reference value three and still misses the
large U1 phases. IBuffer/fetch-queue timing remains inactive.

`sim-arch` now carries three uniform backward-target resolution cycles beside
the existing ten-cycle exact forward-branch interval. The three-cycle state is
shorter than the encoded five-cycle BRA stall, so it adds exact replay
diagnostics without fabricating delay. All replay outputs report exactly
`64/U × warps` branch issues and three cycles per issue. An interleaved
200-sample-per-arm host A/B finds no replay-time regression. All six production
predictions remain unchanged at 33.128% MAPE. The retained corpus reaches 135
sidecars, 47,488 records, and 18,372 projections with zero gaps. No sudo was
required.

## Branch-free IBuffer timeline result

The `front_end_ibuffer_depth` successor reproduces the paper's CLOCK method
with eight branch-free groups of one CS2R and three NOPs. A ninth group keeps
epilogue work outside the first 32 measured instructions. Seventeen legal
stall-one cubins sweep stall 4 and yield across the first eight instruction
positions at W16; the paper's index-one baseline/stall/yield trio transfers to
W1/W4/W5/W8/W12. Every legal treatment changes one control byte.

The 32-cell campaign contains 224 correctness executions, 144 NCU profiles,
seven SourceCounters profiles, 32 NVBit captures, 21,358 records, and 32
byte-identical replay pairs. CLOCK timelines are exact across seven trials in
every cell except W5, whose maximum range is six cycles. Maximum NCU CV is
3.249%, robust MAD is 0.908%, and every issue ratio is 1.0. Literal stall-zero
NOP controls execute correctly but are rejected from the matrix because
nvdisasm cannot decode them and their CLOCK path is roughly eight times slower.

Correct in-flight reservation is essential: once a fetch request reserves an
IBuffer entry before response/decode, both treatment and absolute objectives
select capacity two. The treatment optimum needs a 40-cycle instruction miss
and depth-four prefetch; the absolute optimum needs an 8-cycle miss and no
prefetch. Exact CLOCK event-order accuracy remains at most 14.8%. The
capacity-three/prefetch-eight reference is worse on both timing and event
order. Capacity two is therefore retained only as a candidate; no IBuffer or
instruction-cache state is active. The retained corpus reaches 153 sidecars,
48,784 records, and 18,426 projections with zero gaps. No sudo was required.

## Candidate state

One SMSP-local fixed-latency result queue reserves a slot at issue only after
the exact queue-depth treatment demonstrates that behavior. A result enters at
its characterized completion, becomes eligible for bypass at the measured
point, and retires through its destination bank's write port. Queue order and
pop width are explicit. For the admitted scalar STRONG.SM/HADD2 forms, an
L1-hit load writeback is tracked by SMSP and destination bank; a coincident
HADD2 completion defers its dependence-barrier visibility by one cycle. Other
load/fixed forms remain inactive until measured.

Candidate diagnostics remain bounded aggregates: reservations, maximum depth,
completion collisions, bypassed consumers, per-bank retirement attempts and
denials, fixed/variable priority collisions, and queue-full issue delays. No
per-cycle event log is needed.

## Live campaign

For every frozen cell:

1. run seven correctness trials;
2. collect three base-clock application-replay NCU profiles, expanding only a
   cell above 2% active-cycle CV;
3. retain active cycles, issued/executed and per-pipe counts, issue ratio,
   active/eligible warps, and dispatch/math/wait/not-selected/short/long
   scoreboard stalls;
4. collect SourceCounters for each coincidence, queue-boundary, bypass, and
   write-priority sentinel;
5. capture a separate 64-generation NVBit trace;
6. replay twice from complete static sidecars; and
7. audit every native/patched cubin through the Rust operand projector.

NCU and NVBit remain separate processes. No measured counter enters replay.

## Promotion criteria

Further queue/bypass promotion requires one state machine to:

- preserve the `v00` and noncolliding nulls;
- predict same/opposite-bank completion, queue-depth crossing, consumer
  bypass, and fixed/load priority without protocol branches;
- transfer unchanged to W12/W16 and all four predecessor campaign families;
- preserve 6/6 production coverage and every memory/barrier/launch-chain
  counter; a narrow mechanism may remain neutral on the production roster;
- remain within 3% representative replay wall time or explain the change; and
- return a typed `sm_120` capability gap.

Priority is independently identified and active only for its exact vocabulary.
If queue size or the bypass point cannot be independently identified, retain
their projection/live evidence without activating those separate mechanisms.

## References

- Rodrigo Huerta et al., “Dissecting and Modeling the Architecture of Modern
  GPU Cores,” MICRO 2025: <https://doi.org/10.1145/3725843.3756041>
- Authors' simulator: <https://github.com/upc-arco/modern-gpu-simulator-micro-2025>
- Local paper:
  `/home/jakeshea/GPU_sims_notes/research_papers/05_huerta_modern_gpu_cores_micro2025.pdf`

No privileged command is currently required. If a future tool needs operator
action, provide the exact `sudo ...` command for the user to run.
