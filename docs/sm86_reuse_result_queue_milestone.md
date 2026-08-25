# Next milestone: sm_86 reuse cache and fixed-latency result/writeback queue

**Status:** RFC slice completed 2026-08-17; result/writeback successor configured  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[Control/Allocate and register-file contention](sm86_control_allocate_register_file_milestone.md)  
**Evidence:**
`results/captures/sm86_fixed_latency_allocate_20260817_v1` and
`results/captures/sm86_fixed_latency_completion_20260817_v1`  
**RFC evidence:** `results/captures/sm86_fixed_latency_reuse_20260817_v1`  
**Successor:** [fixed-latency result, bypass, and writeback queue](sm86_result_writeback_queue_milestone.md)

## RFC completion result

Probe A is complete. Five exact policies cover allocation without a hit,
paired hits, retention, source-position movement, and same-bank eviction. The
40-cell W4/W8/W12/W16 campaign contains 280 correctness executions, 120 NCU
profiles, 16 SourceCounters reports, 40 NVBit captures, and 518,400 replay
records. Maximum active-cycle CV is 0.645%; all issue ratios are 1.0 and all
pair instruction/FMA/replay counts match.

The zero-hit u10 pair is zero at every width. Paired-hit u11 speedups are
1.00/2.00/2.96/3.10 cycles per generation; retained-hit u12 speedups are
2.00/6.33/10.13/9.11; moving the cached register to another source position
costs 2.00/6.33/10.09/9.34. The u22 eviction treatment has no robust W4/W8
effect and profiler/driver disagreement at larger widths, so it is retained as
a falsification cell.

Five state/timing shapes were evaluated. Untagged state falsely accelerates
the exact-zero u10 control by 5.34/7.27/9.78 cycles at W8/W12/W16; per-warp
storage overpredicts u11 by up to 14.98 cycles. This supports one shared entry
per bank/source position carrying a warp/register tag. The best active timing
composition reaches 1.451 cycles/generation RFC-delta MAE, but completion-
distance held-out-width MAE is 5.543 and the 99-cell scheduler MAE remains
37.331. No RFC timing is promoted. Candidate removal leaves all 80 saved RFC
outputs byte-identical to production.

## Goal

Identify the two fixed-latency structures that the no-reuse bank campaigns
could not observe:

1. compiler-managed register-file-cache (RFC) allocation, hit, retention, and
   eviction by bank and source position; and
2. fixed-latency result/bypass buffering plus one-write-port-per-bank
   retirement under colliding ALU/FMA completion.

Promote neither as a generic correction. Each structure must predict exact
binary treatments, held-out operand positions, W12/W16, the completed 99-cell
scheduler corpus, and the six production workloads without reading NCU values
during replay.

## Established boundary

The retained static projection already provides exact regular-register source
positions, register spans, destination banks, and textual/control-word reuse
agreement. Two live campaigns establish the timing boundary:

- A 28-cell bank treatment measured 7.00/14.00/21.56/34.83
  cycles/generation at W4/W8/W12/W16.
- A 32-cell matched completion-distance treatment varied RAW distance
  1/2/4/8 while preserving values, instructions, destinations, carried RAW
  edges, controls, and reuse=0. Its W4 bank3 deltas are
  -0.03/2.00/8.00/7.00 cycles/generation; W16 deltas are
  44.77/13.65/30.00/34.00.
- Five Control/Allocate/completion candidates were tested. The best treatment
  MAE is 2.906 cycles/generation, but it misses W8 distance-4 by 9.13 cycles and
  changes the prior 99-cell scheduler MAE only from 37.62 to 37.22. All timing
  code was removed; 64 restored completion outputs are byte-identical.

The completion treatment deliberately holds every reuse mask at zero and keeps
the destination stream identical between variants. It therefore cannot
identify either RFC state or result-queue capacity. Those are separate causal
experiments, not parameters to infer from the current residual.

## Prior mechanism to test

Huerta et al.'s MICRO 2025 model proposes one RFC entry per regular-register
bank and three source-position subentries per SMSP. A hit requires the same
warp, register identity, and source position. A request to the same bank and
position replaces or invalidates the old value; a reuse bit retains the newly
read value. Their sm_86 simulator also uses a bounded fixed-latency result queue
with one regular-register write port per bank and fixed-latency priority over
variable-latency writeback.

These mechanisms are hypotheses for this RTX 3090 Ti. Their published sizes
and policies are not architecture-card facts until exact live treatments
transfer.

## Probe A: exact RFC state machine

Add `workloads/research_probes/fixed_latency_reuse/`. Runtime-loaded duplicate
values allow source-register and reuse-bit patches without changing results.
The hot interval contains no memory, barrier, branch, atomic, or local spill.

Use matched sequences derived from the RFC examples in the MICRO 2025 paper:

| Protocol | Treatment |
|---|---|
| `u00` | loop/control baseline |
| `u10/u11` | allocate one source-position entry / read it once |
| `u12/u13` | hit without retention / hit with retention, then read again |
| `u20/u21` | same register moves to another operand position |
| `u22/u23` | same bank+position replacement by another register |
| `u30/u31` | same register and position from another warp |
| `u40/u41` | independent bank entries versus same-bank interference |

For each exact pair, patch only selected source-register fields and reuse bits.
Disassembly and projection must prove identical opcode, destination, predicate,
stall/yield/dependence controls, source widths, instruction count, and resource
usage. The treatment is the RFC request sequence itself; no source-level label
is accepted as physical evidence.

Use W4/W8 for identification, distances and operand positions not used during
fitting as the first holdout, and W12/W16 as the width holdout. The model state
is SMSP-local, bank-local, source-position-local, and warp-tagged. It may not
use an opcode-, protocol-, or warp-count correction.

## Probe B: result queue and writeback

Add `workloads/research_probes/fixed_latency_result_queue/` only after Probe A
freezes RFC semantics. Extend static projection to the exact additional
fixed-latency opcode forms required by admitted SASS; unsupported forms remain
typed gaps.

Primary treatments:

- consecutive HADD2 and FFMA producers whose characterized latencies make them
  complete together, versus a noncolliding order;
- same-bank versus alternating-bank destinations with identical sources and
  pipe work;
- eight versus nine live fixed-latency results to cross the published queue
  size without changing occupancy; and
- a fixed-latency completion colliding with a dependent L1-hit load writeback,
  testing the claimed fixed-over-variable priority separately from queue size.

Destination patches must be paired with downstream consumer patches and a
host oracle so all variants are semantically checked. A result-queue candidate
reserves at issue, receives results at characterized completion, forwards a
fixed result to a dependent consumer before register-file retirement when the
live treatment requires it, and drains through physical destination-bank
ports. Load writeback remains on its independently modeled completion path.

## Live campaign

For each frozen cell:

1. run seven correctness trials;
2. collect three base-clock application-replay NCU profiles, expanding only a
   cell above 2% active-cycle CV;
3. retain active cycles, issue ratio, per-pipe counts, active/eligible warps,
   dispatch/math/wait/not-selected/short-scoreboard stalls;
4. collect SourceCounters for every allocation, retention, eviction, queue
   boundary, and write-priority sentinel;
5. capture an independent 64-generation NVBit trace;
6. replay twice from complete control sidecars; and
7. run the Rust operand auditor over every native and patched cubin.

NCU and NVBit remain separate processes. NCU counters are observations, never
replay inputs.

## Promotion criteria

RFC state is promoted only if one state machine predicts all allocate/hit/
retention/position/eviction treatments and both width holdouts. Result-queue
state is promoted only if queue depth, destination-bank service, bypass, and
fixed-over-variable priority are independently identified.

Any promoted composition must also:

- preserve W4 no-contention and reversed-family ordering;
- improve the prior scheduler held-out errors without a protocol branch;
- leave barrier, async-copy, cache, and launch-chain traffic unchanged;
- retain 6/6 production coverage and report MAE/RMSE/MAPE against 33.128%;
- keep representative replay within 3% wall time or explain the difference;
  and
- return a typed capability gap for `sm_120`.

If either structure fails transfer, retain its exact measurement and static
projection while leaving timing inactive.

## References

- Rodrigo Huerta et al., “Dissecting and Modeling the Architecture of Modern
  GPU Cores,” MICRO 2025: <https://doi.org/10.1145/3725843.3756041>
- Authors' simulator: <https://github.com/upc-arco/modern-gpu-simulator-micro-2025>
- Local paper:
  `/home/jakeshea/GPU_sims_notes/research_papers/05_huerta_modern_gpu_cores_micro2025.pdf`

No privileged command is currently required. If a future tool needs operator
action, provide the exact `sudo ...` command for the user to run.
