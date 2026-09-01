# Completed milestone: sm_86 modeled write-ack event plumbing

**Status:** complete 2026-08-31 as `complete_store_ack_event_plumbing`  
**Implementation:** `0041edccda6de152aadeefb74db2276a57e4e220`  
**Report:** `data/local_labels/sm86_store_write_ack_event_report.json`

## Outcome

The timing-active STORE/RED feedback path no longer publishes a future
write-completion timestamp directly into warp state while the shared event
queue remains unaware of it. The existing schema-11/12/13 frontier is now one
typed `MemoryWriteAck` event per exact instruction group. The event:

- is created only after every request index of the instruction has reached the
  shared L2 observer;
- retains issue, service-acceptance, service-completion, and fence-visibility
  timestamps separately;
- caps that SM actor's run-ahead until it is processed;
- releases only the matching warp's `MEMBAR.SC.GPU` frontier;
- survives an earlier warp `EXIT`; and
- participates in the kernel's terminal modeled-event drain.

No new rate, latency, queue capacity, architecture-card value, or hardware
measurement entered the mechanism. Timing-inactive feedback remains schema 10
and emits no acknowledgement. Timing-active feedback emits readiness schema 22
with scheduled/processed counts, maximum pending population, all four
frontiers, and the terminal acknowledgement drain time.

`MemoryAdmissionFeedbackDiagnostics` is now kept behind an optional box because
it is an opt-in cold diagnostic. This accommodates schema 22 while keeping the
existing hot `ReadinessDiagnostics` size bound.

## Executable evidence

The full simulator workspace passed 893 tests with three machine-dependent
tests ignored. A release `sim-run` was built at SHA-256:

```text
ae2df6f7d52befa0824baafa7ac925bfe87954c737c4343bfd024e1f979cde5a
```

A focused system test issues one two-sector STORE followed by `EXIT`. The warp
finishes at cycle 2, but one acknowledgement drains at cycle 20; two L2 sectors
produce exactly one scheduled and one processed acknowledgement.

Three current-revision retained-trace replays were then executed twice with
candidate 201:

| Cell | Acks | Max pending | Last issue | Ack drain | Twin deterministic |
|---|---:|---:|---:|---:|---|
| L1TEX B84/W8/D1 | 3,360 | 29 | 4,001 | 4,194 | yes |
| L1TEX B84/W8/D16 | 13,440 | 65 | 13,601 | 13,794 | yes |
| retirement B84/D16/no fence | 1,344 | 9 | 8,565 | 11,236 | yes |

For both L1TEX cells, deleting only schema-22 diagnostics and restoring the
predecessor terminal fields reproduces the retained candidate-201 JSON exactly.
Traffic and cache counters are unchanged. Current default-disabled replays of
all three cells are byte-identical to their retained predecessor outputs.

The exact commands follow the established direct-run shape:

```bash
sim/target/release/sim-run \
  --trace <retained-cell-nvbit-root> \
  --control <campaign-control-bits.json> \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json \
  --mode l1-l2 \
  --memory-admission-feedback \
  workloads/research_probes/l1tex_admission_backpressure/memory_write_completion_feedback_store_201.json
```

## Scientific boundary

`MemoryWriteAck` means completion of the explicitly configured simulator
service/fence relation. It is not a silicon acknowledgement, dirty-eviction
writeback, coherence propagation, or an implicit task-boundary flush. Global
STORE replays therefore retain `dirty_writeback=true` and `coherence=true`;
RED also retains its read-modify-write tail. Completion remains an explicit
lower bound.

Candidate 201 remains a historical research candidate, not a promoted timing
model. The retained retirement corpus has no in-kernel phase clocks, while the
L1TEX corpus always measures a composite `MEMBAR.SC.GPU; ERRBAR; CCTL.IVALL`
interval. Cross-corpus subtraction cannot identify acknowledgement timing.
The next discriminating experiment is a fresh same-binary three-arm assay with
matched no-fence, GPU-MEMBAR, and implicit-terminal paths. No tail may be
cleared from the plumbing result alone.
