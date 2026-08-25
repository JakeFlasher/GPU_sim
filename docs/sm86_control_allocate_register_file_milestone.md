# Completed milestone: sm_86 fixed-latency Control/Allocate and register-file contention

**Status:** completed 2026-08-17; timing candidates rejected, successor configured  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[SMSP scheduler and math-pipe contention](sm86_scheduler_math_pipe_contention_milestone.md)  
**Evidence:**
`results/captures/sm86_fixed_latency_allocate_20260817_v1/analysis/README.md` and
`results/captures/sm86_fixed_latency_completion_20260817_v1/analysis/README.md`  
**Successor:** [reuse cache and result/writeback queue](sm86_reuse_result_queue_milestone.md)

## Current progress

Essential component 1 is implemented. The program join now projects exact
IADD3/IADD3.X/LOP3.LUT/bare-IMAD/FFMA/FADD/FMUL operands from trusted
`nvdisasm` text: register files and indices, `.64`/`.128` spans, unary
modifiers, source positions, reuse flags, and sm_86 two-bank demand. Textual
reuse annotations must equal the independently decoded four-bit control mask.
Unsupported target forms retain a typed gap and remain dispatch-only. The
projection now covers 16,313 target instructions across all 42 retained sm_86
sidecars with zero gaps, and adding it leaves all 198 prior scheduler replay
files byte-identical.

The first live bank campaign is also complete. It uses a verified in-place
cubin patch that changes only 28 bytes in sixteen bare-IMAD low words; every
high/control word, PC, opcode, destination, predicate, resource declaration,
and other file byte remains unchanged. Native instructions read two registers
from one bank and one from the other; patched instructions read all three from
one bank with reuse disabled.

The 28-cell result contains 196 correctness executions, 86 NCU profiles, nine
SourceCounters reports, 28 NVBit captures, and 303,960 replay records. The
bank-only live delta is 7.00/14.00/21.56/34.83 cycles per generation at
W4/W8/W12/W16 with identical instruction/FMA counts and issue ratio 1.0. W4
adds dispatch-stall and wait without math throttle; higher occupancy adds math
throttle while not-selected barely changes.

A flexible Control→Allocate candidate with one read per bank over a three-cycle
window predicts the W8/W12 deltas within 0.31/0.47 cycles and strongly improves
native absolute timing. It exposes only two of seven W4 cycles and underpredicts
W16 by 5.31; fixed source-stage slots and an extra Allocate hold fail worse.
All timing candidates were removed after deterministic capture. Only operand
projection is retained. Complete reduction:
`results/captures/sm86_fixed_latency_allocate_20260817_v1/analysis/README.md`.

The destination-consumer experiment is complete. Its matched 32-cell matrix
varies RAW distance 1/2/4/8 while preserving values, instructions,
destinations, carried dependencies, controls, resource usage, and reuse=0.
Across 224 correctness executions, 96 primary NCU profiles, 12 SourceCounters
reports, 32 NVBit captures, and 408,320 replay records, maximum active-cycle CV
is 1.58% and every issue ratio is 1.0.

W4 bank3 deltas are -0.03/2.00/8.00/7.00 cycles/generation at distance
1/2/4/8; W16 deltas are 44.77/13.65/30.00/34.00. Five explicit staged-read,
reference-hold, conflict-hold, and issuer-completion candidates were evaluated.
The best treatment MAE is 2.906 cycles/generation, but it misses W8 distance-4
by 9.13 cycles and changes the prior scheduler all-cell MAE only from 37.62 to
37.22. No timing state transfers. All experimental code was removed and 64
restored outputs are byte-identical to production.

The no-reuse treatment also establishes what it cannot identify: its reuse
masks are all zero and its destination stream is identical across variants, so
RFC state and result-queue capacity remain separate experiments. Those move to
the configured successor milestone.

## Goal

Identify the fixed-latency stages between issue and execution that make
half-warp ALU/FMA service work-conserving without behaving like immediate
scheduler backpressure. Separately determine register-file bank reservation,
compiler-managed reuse-cache hits, and result-queue effects. Promote only state
that predicts unseen register patterns, warp depths, phases, and the completed
99-cell scheduler campaign.

## Why this is next

The scheduler campaign establishes a strong capacity relation but rejects all
direct scheduler-port couplings:

- W12/W16 ALU-dominant active time differs from the two-cycle executed-ALU
  service bound by only 1.26 cycles/generation on average.
- W12/W16 bare-IMAD cells differ from the analogous FMA bound by 4.41 cycles.
- A cyclic port candidate fits W8 same-family and phase response but changes
  W4, overprices W12, and loses reversed-family order signs.
- Bounded issue queues, oldest/FIFO fallback, and decoupled service drains each
  fix only a subset of the evidence.
- NCU maps bare `IMAD` to FMA work but not `IMAD.MOV`, `IMAD.WIDE`, or
  `IMAD.X`; treating the opcode head as one family is demonstrably wrong.

Huerta et al.'s MICRO 2025 model supplies the missing architectural boundary:
fixed-latency work passes through a common Control stage, then an Allocate stage
that reserves register-file reads and future execution availability. Register
reads occupy a three-cycle window; the regular register file has two banks and
a compiler-managed reuse cache. The authors' SM86 configuration uses half-warp
INT/FP service. These are prior mechanisms to validate and adapt, not project
novelty claims.

## Semantic boundary

```text
CGGTY issue winner
        |
        v
  Control latch          dependency-counter visibility
        |
        v
  Allocate latch  -----> future execution slot reservation
        |
        +-------------> register-bank read-window reservation
        |
        +-------------> reuse-cache hit/miss and replacement
        v
  fixed-latency execution input
        |
        v
  result/bypass queue and register-bank write
```

The scheduler may consider a future execution slot that will be free when the
instruction reaches it. It must not treat the execution unit as busy at issue
time, nor accept an unbounded post-hoc service correction. Compiler stall,
dependence barriers, exact forward-branch resolution, Tensor initiation,
barriers, caches, and launch chaining retain their current meanings.

## Essential component 1: SASS operand projection

Extend the static program join with an architecture-local, immutable projection
derived from the already trusted `nvdisasm` instruction text:

- exact opcode token and execution family (`IMAD` distinct from all modifiers);
- regular/uniform/predicate destination and source registers in operand order;
- source negation/absolute modifiers that do not change bank identity;
- per-source `.reuse` flags;
- source register width when an operand spans consecutive registers; and
- regular destination-bank demand.

Parsing is fail-closed for an admitted fixed-latency opcode: an unsupported
operand form stays dispatch-only with a typed coverage diagnostic, never a
guessed bank. Dynamic trace operand metadata may cross-check the projection but
does not replace the control sidecar as the scheduling source.

## Probe family

Add `workloads/research_probes/fixed_latency_allocate/`. The measured interval
has no global/shared/local memory, CTA barriers, asynchronous copy, or atomics.
Every retained binary must be spill-free.

Use separate compile-time SASS-confirmed specializations:

| Axis | Identification | Held out |
|---|---|---|
| Warps/SMSP | 1 and 2 (W4/W8) | 3 and 4 (W12/W16) |
| Source-bank demand | split-bank, 2-on-1-bank, 3-on-1-bank | alternating conflicts |
| Reuse state | none, one reused source | two/three reused sources and eviction |
| Family | IADD3/LOP3, FFMA, bare IMAD | reversed ALU/FMA sequences |
| Dependence | independent destinations | dependent consumer after allocation |
| Destination | alternating banks | same-bank completion collisions |

CUDA source templates may be generated broadly, but matrix cells are admitted
only after SASS inspection proves the physical registers, bank parity, reuse
bits, opcode token, and control word. If `nvcc` cannot produce a necessary
matched pair, add a documented cubin-rewrite toolchain gate; do not call two
different register allocations a reuse-only experiment.

The primary protocols are:

- `r00`: loop/control baseline;
- `r10/r11/r12`: one, two, or three same-bank source reads with identical ALU
  count;
- `r20/r21`: the same register tuple with reuse disabled/enabled;
- `r22`: reuse-entry displacement followed by a re-read;
- `r30/r31`: identical FFMA/IMAD work with split/same-bank sources;
- `r40/r41`: alternating versus colliding destination banks; and
- `r50/r51`: ALU→FMA and FMA→ALU order reversal with the same bank demand.

## Identification strategy

1. Measure W4 first to derive per-instruction read-conflict and reuse effects
   without inter-warp contention.
2. Use W8 to distinguish issue-time blocking from a one-/two-stage future
   reservation window.
3. Freeze stage count, read-window length, bank ports, and reuse capacity using
   only W4/W8.
4. Test W12/W16 unchanged. A stage relation that requires a warp-count branch
   is rejected.
5. Replay all 99 scheduler cells. The new stage must improve W12 magnitude
   without losing phase response or reversed-family signs.

## Candidate stage shapes

Evaluate nested candidates:

- production dispatch-only state;
- one Control latch with dependency visibility only;
- Control plus one Allocate latch reserving future pipe availability;
- the same with two regular-register banks and one 1024-bit warp read per bank
  per cycle over a three-cycle read window;
- compiler-managed reuse entries scoped by bank and operand position; and
- a bounded fixed-latency result queue only if destination-collision probes
  require it.

Do not add all stages at once. Do not infer register-bank pressure from NCU
stall ratios. The executed operands and register identities drive replay; NCU
is a validation oracle only.

## Live campaign

For every frozen cell:

1. run seven correctness trials;
2. collect three base-clock application-replay NCU profiles and expand to five
   only above 2% active-cycle CV;
3. retain active/eligible warps, selected/not-selected, math/dispatch/wait
   stalls, issue ratio, per-pipe instructions, and active cycles;
4. collect SourceCounters for each bank/reuse sentinel;
5. capture a bounded 64-generation NVBit trace;
6. replay twice from complete control sidecars; and
7. cross-check structured operands against the SASS operand projection.

NCU and NVBit remain separate processes. No NCU counter enters replay.

## Diagnostics and promotion

If a stage relation survives, readiness schema 4 adds bounded aggregate fields
for Control entries, Allocate attempts/accepts, pipe-reservation denials,
per-bank requested/reserved reads, reuse hits/misses/evictions, result-queue
pressure, and maximum occupied stage/queue depth. No per-cycle event log is
emitted. Schemas 1–3 remain byte-compatible when fixed-latency stage state is
absent.

Promotion requires:

- W4 bank/reuse identification and W8 stage identification;
- unchanged prediction on held-out registers, W12/W16, and order reversal;
- improvement on the completed scheduler campaign without a protocol branch;
- unchanged predecessor traffic, barrier skew direction, cache routes, and
  launch-chain semantics;
- production 6/6 coverage with MAE/RMSE/MAPE reported against 33.128%;
- representative host replay within 3% or an explained millisecond change; and
- a typed `sm_120` capability gap.

If no candidate transfers, publish the operand/bank evidence and retain the
current simulator. Do not substitute an opcode or warp-count correction.

## References

- Rodrigo Huerta et al., “Dissecting and Modeling the Architecture of Modern
  GPU Cores,” MICRO 2025: <https://doi.org/10.1145/3725843.3756041>
- Authors' simulator implementation:
  <https://github.com/upc-arco/modern-gpu-simulator-micro-2025>
- Local paper copy:
  `/home/jakeshea/GPU_sims_notes/research_papers/05_huerta_modern_gpu_cores_micro2025.pdf`

No privileged command is currently required. If a later tool invocation needs
operator action, provide the exact `sudo ...` command for the user to run.
