# Completed milestone: bounded sm_86 streaming memory-service candidates

**Status:** complete 2026-08-18; evaluator promoted, timing rates withheld  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[L1TEX arrival/output-replay service](sm86_l1tex_arrival_replay_service_milestone.md)  
**Evidence:**
`results/captures/sm86_l1tex_arrival_replay_service_20260818_v1/analysis/README.md`

## Goal

Test the predecessor's nearest implementation prerequisite: evaluate service
hypotheses at every actual memory-instruction issue timestamp without retaining
an unbounded event log, changing ordinary replay, or turning an exploratory
rate into an architecture fact.

The target is deliberately narrower than a production memory pipeline. It asks
whether online arrival sequencing resolves contradictions that defeated scalar
work and first/last-span models, and whether the surviving service appears
SM-shared or independently SMSP-local.

## Implemented research surface

`sim-sm` now accepts at most 16 opt-in candidates. Each candidate selects:

- LOAD, STORE, ATOMIC, or REDUCTION observations;
- one SM-shared server or four independent SMSP-local servers;
- a rational ideal-wavefront service rate;
- a rational unique-128-byte-line service rate; and
- an optional terminal completion latency.

For issue `i` at simulator timestamp `t_i`, the candidate computes

```text
busy_i = ceil(ideal_wavefronts_i * ideal_cycles / ideal_units)
       + ceil(unique_lines_i * line_cycles / line_units)
start_i = max(t_i, selected_server_free_time)
done_i  = start_i + busy_i
```

The configured latency extends only the observed terminal timestamp. It does
not occupy the service unit. Arithmetic is checked, disabled dimensions must
be exactly `(0, 0)`, candidate identifiers must be unique, and configuration
above the fixed capacity fails before trace extraction.

Candidate state is copied and preflighted at the issue-cycle transaction
boundary. A candidate overflow therefore cannot commit a partial issue, cache
mutation, request, barrier update, or retirement. The default empty set exits
before topology projection and leaves ordinary scheduling unchanged.

Readiness schema 7 reports, for each candidate:

- the complete configuration;
- observed instruction, ideal-wavefront, and unique-line work;
- total busy cycles and summed queue-delay exposure across SMs; and
- the maximum terminal completion timestamp across SMs.

These values are diagnostics. They do not add events, gate issue, clear a
dependency, alter `ModeledCompletionFrontier`, or produce a numerical GPC
prediction.

The completed admission/backpressure successor later introduced schema 8 for
orthogonal operand-read attribution. Current replays containing those barriers
therefore carry the unchanged schema-7 candidate payload inside schema 8.

The direct runner exposes the bounded surface as
`--memory-service-candidates FILE` in launch-scoped `l1-l2` mode. Candidate
documents are schema-versioned and reject unknown fields. Stream-L2 sessions
remain unsupported because candidate state is kernel-local and no cross-launch
service contract has been identified.

## Complete replay evidence

The retained 75-cell arrival/output-replay corpus was replayed twice with a
16-candidate grid:

- 75 byte-identical candidate replay pairs;
- all 159,553,212 retained NVBit records;
- eight STORE and eight RED candidates;
- SM and SMSP scopes;
- unique-line, ideal-wavefront, and additive work bases; and
- equal-per-server plus equal-full-SM-rate SMSP controls.

Removing only schema-7 candidate results and restoring the ordinary readiness
schema leaves every candidate replay byte-identical to its ordinary replay.
No base timing, counter, cache, completion-status, or coverage field changes in
any of the 75 cells.

For comparison with live measurements, analysis forms a matched-control
diagnostic only:

```text
predicted tail delta =
    max(raw tail replay terminal, candidate tail terminal)
  - max(raw control replay terminal, candidate control terminal)
```

The control subtraction cancels the already-characterized per-grid launch
envelope. It does not make the candidate terminal a production output.

## Retrospective transfer result

Candidate selection uses W1/W4 work-scaling cells. Fixed-work concurrency, CTA
partition, arrival skew, lane topology, and word topology are held out
structurally.

| tail | selected hypothesis | calibration cells / MAE | held-out cells / MAE | held-out maximum | raw-terminal held-out MAE |
|---|---|---:|---:|---:|---:|
| STORE | SM-shared, 10 cycles per unique line | 8 / 1,196.9 | 15 / 1,418.5 | 3,664 | 15,322.3 |
| RED | SM-shared, 16 cycles per unique line | 4 / 1,197.3 | 6 / 1,331.8 | 1,979 | 32,059.7 |

The STORE server now explains the direction that defeated summary models:
requests separated by a large warp gap drain before later requests arrive.
The candidate's all-cell STORE MAE is 1,341.4 cycles, and the RED value is
1,278.0 cycles.

This is a retrospective test over a corpus inspected during predecessor
research. The rates are useful successor hypotheses, not out-of-sample
architecture parameters.

The campaign retains a write-once candidate-configuration snapshot, and both
reducers require every replay's embedded configuration to match it. Cached
replays are rejected when the simulator, trace, cards, control sidecar, or
candidate input is newer or when root configuration fields are incompatible.

## Adversarial findings

### Tested independent SMSP completion rates are rejected

An SMSP server with the same per-server rate as the selected SM server matches
W1, where only one scheduler partition supplies work, but becomes up to 30,720
STORE cycles and 49,152 RED cycles different when four partitions operate in
parallel. Normalizing four SMSP servers to the selected full-SM aggregate rate
instead misses sparse W1 pressure by as much as 122,880 STORE cycles and
98,304 RED cycles.

Neither normalization transfers across W1–W8 occupancy. One aggregate
SM-level completion relation survives this retrospective slice. This does not
identify the scope of a separate admission queue: the prospective successor
therefore retains both SM-shared and per-SMSP admission candidates feeding an
SM-shared downstream service.

### STORE work basis survives but is not closed

The unique-line candidate beats ideal-wavefront and additive alternatives on
held-out transfer. The decisive word-coalesced cell preserves 512 ideal
wavefronts per SM but reduces line requests fourfold. Even the selected line
candidate underpredicts that cell by 3,066 cycles, showing that fixed line work
alone does not capture every topology or fixed overhead.

### RED work basis remains unidentified

All retained RED cells use line-separated full-lane traffic. In that slice,
candidate 10's 16 cycles per unique line, candidate 12's 64 cycles per ideal
wavefront, and candidate 13's corresponding additive decomposition are exactly
collinear. Their errors are identical. RED topology, aliasing, and atomic
request count must be varied independently before any work basis can be named.

### Completion fit is not an issue-backpressure model

Most live STORE and RED cost is also present in the matched-control
`sm__cycles_active.max` delta. Because GPC and SM maxima may name different
instances, their difference is not a temporal post-active interval. The candidate currently lets the
simulator issue stores without queue-capacity backpressure and places the work
on a counterfactual completion timeline. It can match an elapsed tail while
still getting the causal issue schedule wrong. That is acceptable for an
observational discriminator and unacceptable for production readiness,
dependent work, CTA retirement, or cache-arrival ordering.

### The hardware comparison remains a GPC/SM approximation

The diagnostic aggregates a maximum across simulated SMs, whereas NCU reports
the maximum of seven GA102 GPC counter instances. Matched controls reduce but
do not remove that topology difference. The evaluator also does not model the
external launch envelope, dirty writeback, coherence, or RED read-modify-write
completion.

## Promotion decision

Promoted:

- the fixed-capacity streaming evaluator;
- checked transactional updates at actual issue timestamps;
- SM and SMSP counterfactual scopes;
- readiness schema 7 diagnostics and cross-SM aggregation;
- strict direct-run candidate configuration; and
- deterministic campaign/reduction support.

Not promoted:

- the 10-cycle STORE or 16-cycle RED rates;
- a finite request-queue capacity;
- issue backpressure or L1TEX admission stalls;
- STORE/RED completion events;
- an absolute kernel/GPC cycle prediction; or
- any LOAD/ATOMIC service rate.

STORE and RED remain typed incomplete in ordinary replay.

## Completed successor

The prospective
[sm_86 L1TEX admission/backpressure milestone](sm86_l1tex_admission_backpressure_milestone.md)
is complete. It distinguishes a completion-only server from compiler-visible
operand-consumption backpressure and a composite MEMBAR/ERRBAR/CCTL fence
sequence.

The matrix independently varied:

1. STORE, non-returning RED, and returning ATOMG;
2. unique lines, ideal wavefronts, repeated lines, same-word aliases, and the
   already established 256-byte atomic contention regions;
3. W1/W2/W4/W8 occupancy and deliberately sparse physical warp-slot/SMSP use;
4. burst length and compute gaps around the suspected capacity knee;
5. a dependent post-memory CLOCK/sentinel region that separates admission
   delay, active drain, and final system completion; and
6. one-CTA versus multi-CTA transfer at fixed per-SM work.

The completed 100-cell campaign retains all of those counters plus native
CLOCK issue/fence intervals and exact NVBit input. It rejects the frozen rates
on unseen reuse topology, promotes schema-8 operand-read attribution and direct
L2 hit/miss provenance, and configures competing bounded SM-shared and
per-SMSP admission candidates feeding an SM service. No timing rule advanced.

This successor follows the local research synthesis: finite lower-memory
request queues, operation-specific return behavior, and scheduler/memory
feedback are distinct mechanisms; a final maximum cannot substitute for their
causal interaction.

No privileged command was required.
