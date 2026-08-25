# Completed milestone: sm_86 finite L1TEX admission candidates

**Status:** complete 2026-08-18; schema 9 promoted, timing inactive  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[L1TEX admission/backpressure and reuse](sm86_l1tex_admission_backpressure_milestone.md)  
**Evidence:**
`results/captures/sm86_l1tex_admission_backpressure_20260818_v1/analysis/README.md`

## Goal

Test whether the compiler-observed STG/RED operand-read frontier can be
represented by a finite input queue feeding one SM-shared service, and whether
queue scope/capacity becomes identifiable once service distinguishes exact L2
tag hits from misses.

The milestone remains observational. It must first demonstrate transfer while
preserving the existing issue and cache order. It cannot activate a queue or
claim counterfactual timing from a trace that would itself change under the
candidate.

## Implemented model

Direct replay accepts a separate schema-1 `--memory-admission-candidates`
document, bounded to 64 KiB and 16 entries. Each candidate specifies:

- STORE or non-returning RED;
- one SM-wide or four per-SMSP waiting queues;
- capacity in waiting warp instructions;
- an operand-read visibility floor;
- rational L2-hit-sector service; and
- rational L2-miss-sector service.

All queues feed one SM-shared FIFO server in observed per-SM issue order. A
queue entry leaves when service starts, so same-cycle dequeue/admission is
allowed and the active instruction does not consume a waiting slot.

Every effective global STORE/RED request carries a compact, nonzero group
witness only when schema-9 observation is enabled. The system reconstructs one
instruction from its ordered L2 sector requests and requires exact group
identity, request order, and completion. The witness is not part of cache
identity or event order. Ordinary replay allocates no candidate/observer state;
candidate arrays and queue histories are boxed.

Schema-9 results contain:

- matched instruction, L2-hit-sector, and L2-miss-sector counts;
- service busy and downstream-queue exposure;
- queue-full instruction count;
- total and maximum admission delay;
- maximum accumulated open-loop delay on one warp;
- operand-read-clear frontier; and
- downstream-service completion frontier.

Predicated-off memory records create no witness. Returning ATOMG remains on its
existing result-writeback path. Candidate execution requires full-coverage,
launch-scoped L1/L2 replay and changes neither scheduling nor hierarchy state.

## Candidate matrix and evidence

Sixteen candidates were declared before reduction:

- STORE miss work: 10 cycles/sector;
- RED miss work: 16 cycles/sector;
- hit work: 1 cycle/sector for both;
- scopes: SM and SMSP; and
- capacities: 4, 8, 16, and 32 waiting instructions.

All 100 admission/backpressure cells were replayed twice with byte-identical
outputs. Every embedded configuration matches the retained metadata snapshot.
Stripping `memory_admission_candidates` and restoring the ordinary readiness
schema yields every ordinary launch output exactly. Matched candidate
instruction/hit/miss deltas equal the existing trace and L2-source diagnostics
in all cells.

The hardware corpus remains the predecessor's full live campaign: 300 checked
correctness executions, 700 cold-preconditioned native CLOCK executions, 384
NCU profiles, 16 SourceCounters profiles, and 100 NVBit captures containing
406,381,080 records.

## Result

Capacities are calibrated only on W8 unique-line burst depths 1–16; deeper
burst, occupancy, spacing, address topology, sparse-SMSP placement, and CTA
transfer remain held out.

| tail | selected calibration candidate | calibration issue MAE | held-out issue MAE | held-out max |
|---|---|---:|---:|---:|
| STORE | per-SMSP, capacity 16 | 750.4 | 3,298.2 | 23,166 |
| RED | SM, capacity 32; tied with per-SMSP 8 | 1,190.8 | 3,839.4 | 31,250 |

The maximum single-instruction delay captures the shallow transition but
underpredicts deep burst. Summing each open-loop delay on its warp is far worse:
STORE/RED all-cell MAE becomes 170,887/355,145 cycles, with multi-million-cycle
deep errors. A delayed instruction would have shifted the next request; the
preserved trace does not. Summation therefore repeatedly charges backlog that
a feedback execution would reschedule.

The downstream service result is stronger:

| tail | frozen line-only GPC MAE | exact hit/miss service GPC MAE |
|---|---:|---:|
| STORE | 1,389.8 | 990.6 |
| RED | 2,331.0 | 1,308.6 |

This improvement transfers the reuse direction because tag-resident sectors
are no longer priced as cold lines. It is identical across capacity and scope:
with preserved FIFO arrivals, queue capacity changes admission exposure but
not the server's work or terminal frontier. It cannot identify a queue.

## Decision

Promoted:

- readiness schema 9;
- bounded strict admission-candidate documents;
- compact opt-in instruction grouping at the L2 boundary;
- exact matched per-instruction hit/miss work; and
- finite-queue and downstream-service diagnostics.

Rejected or inactive:

- every tested queue capacity as an architecture parameter;
- a unique SM or per-SMSP admission scope;
- summed open-loop warp backpressure;
- scheduler or cache-order timing changes;
- STORE/RED completion status; and
- numerical GPC-cycle emission.

## Configured successor

This successor is complete; see
[feedback-coupled L1TEX admission](sm86_l1tex_feedback_admission_milestone.md).

The next bounded component is one feedback-coupled candidate per replay, not
another simultaneous observational grid:

1. before issuing a candidate STG/RED, query the selected queue;
2. park the warp if full instead of issuing and accumulating a post-hoc delay;
3. yield to the system so already-issued requests receive exact L2 outcomes;
4. advance the shared service and wake queues at deterministic admission
   opportunities;
5. reschedule later warp arrivals and let any cache-order change occur in the
   executable hierarchy; and
6. retain an ordinary observational mode for differential validation.

This lane remains opt-in until one frozen configuration transfers jointly
across depth, reuse, spacing, sparse-SMSP placement, and CTA shape. No
privileged command was required.
