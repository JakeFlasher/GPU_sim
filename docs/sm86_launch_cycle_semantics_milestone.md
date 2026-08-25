# Completed milestone: sm_86 launch-cycle, completion, and range semantics

**Status:** complete 2026-08-18; numeric GPC-cycle emission remains unavailable  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Evidence:**
`results/captures/sm86_launch_cycle_semantics_20260818_v1/analysis/README.md`

## Goal

Resolve what the simulator's last-issue and final-event timestamps mean relative
to NCU `gpc__cycles_elapsed.max`, determine whether a final outstanding memory
operation must be included in kernel retirement, and test whether multiple
launches compose by sum, maximum, or a traced terminal frontier.

The milestone deliberately separates two NCU observation contracts:

- ordinary application replay produces one counter row per kernel launch; and
- Range Replay and Application Range Replay produce one counter row for an
  explicitly marked range of launch/API activity.

Those observables are not interchangeable.

## Probe construction

`workloads/research_probes/launch_cycle_semantics` contains six
compile-time-specialized one-warp/CTA kernels. Every lane executes a runtime
dependent IMAD chain, writes its result to volatile shared memory, reaches one
CTA barrier, and then executes exactly one tail:

| tail | final SASS relation | discriminator |
|---|---|---|
| `control` | `BAR; EXIT` | issue/launch envelope baseline |
| `load_discard` | `LDG; EXIT` | response after last issue |
| `load_use` | `LDG; STG; EXIT` | dependency-visible load return |
| `store` | `STG; EXIT` | fire-and-forget write retirement |
| `red` | `RED; EXIT` | non-returning atomic/RMW retirement |
| `atom_use` | `ATOMG; STG; EXIT` | returning atomic completion |

Tail addresses are 128 bytes apart so lanes use distinct aligned cache lines.
The matrix covers 1/7/42/84 CTAs, 0–4,096 dependent iterations, and same-stream
chains of 1/2/4/8 launches. NCU kernel replay, capture-based Range Replay,
whole-application range replay, and NVBit are separate processes.

## Complete evidence

- 80 cells and 560 correct ordinary executions;
- 240 per-kernel NCU profiles containing 504 launch rows;
- 96 Range Replay and 48 Application Range Replay profiles with the final
  synchronization inside the range, plus 48+48 adversarial profiles with it
  outside;
- 80 NVBit captures containing 168 launches and 20,861,740 records;
- exact total/reported trace cardinality and empty unsupported, unmapped, and
  cross-allocation sidecars in every capture;
- six spill-free kernels, 232 decoded static instructions, and zero unknown
  encodings; and
- 80 ordinary plus 80 carded-DRAM deterministic replay pairs.

No privileged command was required.

## Finding 1: GPC `.max` is spatially uniform on this SKU

For every NCU kernel row, `gpc__cycles_elapsed.sum / avg` identifies exactly
seven counter instances. Across all 504 rows, the largest `max-min` difference
is 17 cycles. The instances therefore observe effectively the same temporal
kernel envelope on GA102.

This falsifies the earlier assumption that a numerical `.max` necessarily
requires an SM-to-GPC activity map. A physical map may be useful for other GPC
metrics, but it is not the missing mechanism for elapsed cycles in this
campaign. The result does not transfer a seven-GPC default to another SKU.

## Finding 2: one launch has a stable difference of extrema

For control kernels, `GPC elapsed.max - SM active.max` stays constant while
dependent work grows from zero to 4,096 iterations:

| CTAs | median difference | range |
|---:|---:|---:|
| 1 | 2,162.0 | 2,160–2,166 |
| 7 | 2,044.5 | 2,044–2,045 |
| 42 | 2,036.5 | 2,035–2,038 |
| 84 | 2,091.0 | 2,085–2,098 |

The profiler guide explains why `gpc__cycles_elapsed` includes the profiler's
front-end WaitForIdle/semaphore envelope. However, `.max` for GPC elapsed and
`.max` for SM activity are separate spatial reductions and need not select the
same physical instance. Their subtraction is therefore an empirical
difference of extrema, not a measured interval before or after SM activity.
Application replay can also collect the metric groups on different
executions, so the subtraction is not assumed co-temporal. It is stable within
a grid shape but not one universal correction, so no constant is promoted.

## Finding 3: completion can be later than final issue

Matched tail-minus-control medians over eight CTA/work pairs are:

| tail | live GPC delta | live SM-active delta | extrema-difference delta | sim issue delta | sim terminal delta |
|---|---:|---:|---:|---:|---:|
| discarded load | 1,433.0 | 1,314.0 | 122.0 | 5.5 | 425.5 |
| consumed load | 1,562.5 | 1,414.0 | 137.0 | 431.0 | 431.0 |
| store | 824.5 | 176.0 | 649.0 | 5.5 | 5.5 |
| RED | 724.5 | 136.5 | 592.0 | 7.5 | 7.5 |
| returning atomic | 1,579.0 | 1,440.0 | 127.0 | 433.0 | 433.0 |

The discarded load proves that processed response events belong to the
completion frontier: using the modeled terminal event adds exactly the carded
420-cycle response rather than ending at issue. The remaining live load gap
localizes missing L1TEX/request-service behavior, not a different boundary.

Store and RED have much larger GPC deltas than SM-active deltas. The 649/592
extrema differences cannot be placed on a post-SM timeline, so they are not
direct drain measurements. Independently, replay marks dirty
writeback/coherence and RED RMW as unmodeled and schedules no equivalent
completion. Its raw terminal timestamp is therefore a lower bound, not a
complete retirement time.

## Implemented semantic boundary

`sim-sm` now exposes `ModeledCompletionFrontier`:

```text
Complete { cycles }
Incomplete { lower_bound_cycles, unmodeled_tails }
```

`FullKernelReplayResult` separately retains:

1. maximum per-SM last-issue frontier;
2. last processed shared-system event timestamp;
3. their raw maximum; and
4. whether every known timing-bearing retirement tail is represented.

Launch-set aggregation preserves that typed state one launch at a time and
still does not invent a scalar reducer. Direct JSON reports the status as
`complete` or `incomplete_unmodeled_retirement_tails`. Existing raw terminal
timestamps remain available for diagnosis, but an incomplete value cannot be
requested through the typed complete-cycle accessor.

## Finding 4: range time is not launch-time aggregation

For the exact same 1/2/4/8-launch chains, summing ordinary per-kernel GPC
values has 81.161% MAPE against capture-based Range Replay and 97.280% against
Application Range Replay. Range GPC values are respectively 5.915× and
51.088× the kernel sum at the median and have up to 63.965% and 16.339% CV.
Moving the final device synchronization outside the marked range does not
repair the relation: MAPE is 84.473% and 97.220%, and the median ratios remain
6.863× and 48.133×.

This is not a failure of launch ordering. The corresponding sums of
`sm__cycles_active.max` match the range observations at 4.230% and 1.381%
MAPE (1.520% and 1.404% with sync outside), with below 0.715% CV. GPC range
elapsed additionally includes idle time between the marked CUDA APIs. NVBit
SASS traces do not record those host/front-end timestamps.

Consequently, an unordered launch roster has no lawful workload-range time.
Even the ordered `stream-l2` replay session supplies cache happens-before, not
inter-launch gaps. `sum`, `max`, and a terminal-event maximum are all rejected.

## Promotion decision

The boundary is promoted; a numerical GPC prediction is not. Across all
single-launch cells:

| candidate | MAPE |
|---|---:|
| raw last issue | 66.670% |
| raw terminal event | 65.733% |
| last issue plus matched control extrema difference | 31.817% |
| terminal plus matched control extrema difference | 30.880% |

The terminal candidate is semantically preferable and measurably better, but
it remains far above the project target and is incomplete for writes. The
existing `exec_cycles_gpc_max` emission stays required-unimplemented.

## Configured successor (completed)

The next essential component was the
[memory-retirement service milestone](sm86_memory_retirement_service_milestone.md),
not another aggregation fit. It has now completed the following work:

1. isolate request admission, L1TEX/L2 response, and outstanding write/RED work
   with tail depth and address-topology sweeps;
2. add modeled completion events only for mechanisms that transfer across
   load/store/RED and CTA counts;
3. retain dirty writeback, coherence, and reduction-RMW as typed incomplete
   tails until each is actually represented;
4. validate the completed per-kernel frontier before revisiting a profiler
   launch-envelope model; and
5. require an explicit timed command stream before any range-level cycle
   prediction is attempted.

It promotes schema 6 ideal-work/arrival diagnostics but rejects fixed tail
timing. Store/RED retirement remains typed incomplete pending a transferable
L1TEX arrival/replay service.
