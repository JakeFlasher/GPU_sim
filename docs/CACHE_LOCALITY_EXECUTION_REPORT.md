# Cache-locality execution report (2026-10-19)

This report records the real-capture execution of
`CACHE_LOCALITY_EXECUTION_PLAN.md`. It does not treat NCU as a simulator
input, and it keeps trace-native Class-A locality separate from conditional
Class-B hardware shadows.

## Replay evidence

All replays below are `full`, integrated `l1-l2`, compressed `.jsonl.zst`, and
use `--workers 12`:

| lane | captures / launches | status profile | artifact |
| --- | ---: | --- | --- |
| SM120 / RTX 5090 | 339 / 11,558 | 339 `ok` | `full_replay_runs/cache_locality_final2_sm120/full.jsonl.zst` |
| SM86 / RTX 3090 Ti | 286 successful / 12,392 | 286 `ok`, 62 `control_error`, 8 `replay_error` | `full_replay_runs/cache_locality_final2_sm86/full.jsonl.zst` |

The five required real-capture probes (three SM120 and two SM86) completed
before the corpus runs. `tools/validate_reuse_mrc.py` reports zero errors for
both final artifacts. `cargo test --release --manifest-path sim/Cargo.toml
--workspace` passes; targeted clippy remains blocked by 16 pre-existing
`sim-trace` warnings unrelated to this change.

## Implemented slices

### M1: L1 pending-merge decomposition

`L1Cache` now preserves `merges` for timing while exposing
`already_requested_merges` and `unrequested_pending_merges` in unified and
origin counters, boundary state, merge audit, and trace metrics. SM120 lookup
projection credits only resident hits plus already-requested sectors. The
changes are observer/accounting-only.

### M2A/M2B: SM120 write observer and generic-store attribution

The report-only write-tag window is available on SM120 as a scoped 256-set,
4-slot effective card capability. Read and write batches share one observer
view; `GlobalUntagged` stores are routed without the old occupancy cutoff; and
operation-qualified L2 lookup accounting uses
`MemRequest::effective_observer_operation()`. Physical L1/L2 state, DRAM, and
frontiers are unchanged.

### M2C: store-to-load boundary

The live timing cache was not changed. Full-sector store-to-load behavior is
kept as a shadow/accounting boundary because installing stores in the live L1
would change response latency and scheduler order without a byte-mask trace
contract.

### M3: L2 pre-packet store-tag accounting

`L2MaskPreview` now records a pre-packet write-tag mask. Same-(SM, issue-cycle,
line) store requests share the pre-instruction tag state, preventing cold
coalesced sectors from self-crediting. Both architectures use the packetized
observer surface; physical L2/DRAM/timing remain unchanged.

### M4: locality observers

`sector_mrc_observer_v1` remains an exact Fenwick/Mattson Class-A oracle with
same-tick lower/upper envelopes and line-tag/sector-fill decomposition. The
reuse observer now emits a line-reference volume clock, 80-bin count/sum
histograms, right-censored tail intervals, and coupled line/sector finite-boundary
miss curves. The Tier-2 `sm_l1_locality_observer_v1` is card-bound and
trace-ordered; Tier-3 `l2_system_locality_observer_v1` reports L2 summary
bounds and explicitly counts unavailable global timestamp order. Both are
shadow-only.

## Quantitative results and blockers

The exact Fenwick MRC has 0.00% error against its analytical reference by
construction and all corpus invariants pass. The finite-boundary interval
approximation is not within the plan's requested <1.5% WAPE gate:

| lane | line WAPE | sector WAPE |
| --- | ---: | ---: |
| SM120 | 20.6283% | 19.1585% |
| SM86 | 16.7574% | 17.8661% |

The residual is expected from the declared information boundary: within-warp
reuse interval/volume lacks cross-warp reuse distance, set/slice mapping,
replacement, MSHR/fill chronology, and shared-memory carveout. No NCU fitting,
post-hoc correction, or exact-oracle copying was used.

The final SM120 NCU report retains the previously measured endpoint frontier:
Global L1 hit nonzero-target relative error ≈1.45% (four nonzero predictions
on zero-target rows); Global L1 miss ≈0.12%; unified L1 ≈39.58%; operation-read
L2 ≈14.89%; generic source-TEX L2 ≈24.11%. These are endpoint counters, not
MRC targets. SM86 NCU profiles do not identity-join through the SM120 report
runner; retained SM86 comparisons remain the accepted full-replay reports in
`docs/RESULTS.md` and `FULL_NUMERIC_COMPARISON_PER_SM_BASELINE.json`.

The four SM120 zero-target global-L1 rows are ordered-fallback joins with
trace-paired LDG/STG scope. Their target association is not independently
identifiable from the trace; applying a workload-specific or NCU-derived rule
would violate the no-fitting contract. They remain a documented residual,
not a hidden correction.

`full_replay_runs/cache_locality_final2_numeric_comparison.json` shows the
expected observer-only differences against the accepted pre-locality lanes.
After excluding observer fields and wall clocks, no physical cache, DRAM,
IPC, or timing-frontier difference remains.

## Follow-up observer correction (final4)

A corpus-wide review found that the SM120 unified-L1 projection discarded
`l1.write_lookup_hits` for ordinary `global_store` launches. The projection now
uses the active per-SM write-tag observer for all cacheable store forms, while
reduction/atomic-only streams remain outside the unified endpoint. The final4
replay (`full_replay_runs/cache_locality_final4_sm120/` and `..._sm86/`) again
passes the full status and MRC gates. SM120 unified-L1 mean APE is now about
1.63%; operation-read L2 is about 6.69% (below the 8% gate). Global-L1 remains
about 1.45% with four zero-target false positives. Generic source-TEX L2 remains
about 23.97%.

The M2C shadow observer additionally identifies trace-qualified full-sector
same-warp store-to-load forwarding. On the required split-K reduction probe it
reduces the operation-read request projection to the observed target without
changing physical timing. A later large-store packet-shape hypothesis was
rejected by the complete-corpus experiment (generic L2 WAPE worsened to
44.48%), so it was removed; no kernel-name or NCU-fitting exception remains.

### Generic-L2 residual audit

The final4 generic-L2 report was drilled down to its dominant rows. In the
FFT launch `sol_l2_051_hyena_complete_sm120`/48, the trace has 48 LDG
requests (280 sectors) and 24 STG requests (288 sectors); the packet shadow
returns 179 hits while the joined target is 5. A trace-only rule that zeroed
packet hits for large stores was run across the complete corpus and worsened
generic-L2 WAPE to 44.48%, so it was rejected and removed. The retained model
therefore does not silently relabel STG packets or use kernel names. The
remaining generic-L2 discrepancy is an endpoint/client-scope ambiguity that
cannot be resolved from the retained trace metadata without an independent
scope crosswalk.

## Milestone 6 replay and forensic disposition (2026-10-20)

The M6 implementation was rebuilt with `cargo build --release --workspace`, and the required real-capture probes were rerun in full mode with twelve workers. Final fresh full-only artifacts are `full_replay_runs/cache_locality_m6r_sm120/full.jsonl.zst` (339 captures, 11,558 launches) and `full_replay_runs/cache_locality_m6r_sm86/full.jsonl.zst` (356 captures, 12,392 launches; 286 `ok`, 62 `control_error`, 8 `replay_error`). Both pass `tools/validate_reuse_mrc.py` with zero errors.

The origin-qualified Ding–Denning/Xiang HOTL observer now gates the same streams as the Fenwick oracle, excludes censored tails from miss counts, performs finite-boundary footprint inversion, couples sector bottlenecks to line gaps, and flushes the final warp stream. Corpus WAPE is **0.8767% line / 0.4462% sector (SM120)** and **1.0223% line / 0.5324% sector (SM86)**.

The SM120 CCTL/forwarding projection removes all four previously identified Global-L1 zero-target false positives. The full SM120 comparison reports Global-L1 APE **1.5567%**, zero-target FPs **0**; unified-L1 APE **1.6386%**; operation-read L2 APE **6.6859%**. Generic source-TEX L2 remains unresolved: the current store-packet/effective-ST.E implementation reports **724.92%** mean APE and **6,159** zero-target false positives, versus final4's 23.97% / 117. This reproducible regression is retained as forensic evidence, not claimed as a closed gate; the retained trace lacks an independent generic-TEX client crosswalk and no NCU-shaped correction was applied.

### M6 observer-only ST.E follow-up (2026-10-20)

A second implementation keeps the physical L2 class on `GlobalUntagged` requests
unchanged and obtains the effective `WriteAtomic` class through the public
read-only `preview_mask` seam. Bounded real probes passed; the complete SM120
replay is `full_replay_runs/cache_locality_m6final_sm120/full.jsonl.zst` and
passes the reuse validator. Its NCU report remains materially unchanged:
Global-L1 APE 1.5567% with zero false positives, operation-read L2 6.6859%,
and generic TEX-L2 724.72% APE with 6,158 zero-target FPs. This preserves
physical cache/DRAM/timing semantics but does not resolve the generic client
scope; no further trace-only rule was promoted.

## Milestone 7 closure replay (2026-10-20)

M7's deterministic store-packet implementation is now present in the unified
Rust path. `L2SourceSplit` exposes both resident packet hits and write-tag
packet hits; `System::run_interleaved` restores the per-(SM, issue-cycle,
128B-line) tag snapshot; and `SmCore` keeps the uncapped L1 write observer
separate from the admission-capped L2 `GlobalUntagged` observer. Physical L2
class, cache state, DRAM traffic, and timing continue to use the request's
actual operation; packet counters are report-only.

The final closure artifacts were rebuilt and replayed in full `l1-l2` mode with
12 workers (no fast replay and no new NCU/NVBit capture):

| lane | captures / launches | dispositions | artifact |
| --- | ---: | --- | --- |
| SM120 / RTX 5090 | 339 / 11,558 | 339 `ok` | `full_replay_runs/cache_locality_m7closure_sm120/full.jsonl.zst` |
| SM86 / RTX 3090 Ti | 356 attempted / 12,392 | 286 `ok`, 62 `control_error`, 8 `replay_error` | `full_replay_runs/cache_locality_m7closure_sm86/full.jsonl.zst` |

Both artifacts pass `tools/validate_reuse_mrc.py` with zero errors. Release
build and the complete Rust workspace test suite pass. The NCU comparison is a
post-replay join only:

| metric | SM120 Mean APE | zero-target FPs | SM86 Mean APE | zero-target FPs |
| --- | ---: | ---: | ---: | ---: |
| Global L1 lookup hit | 1.5567% | 0 | 27.5757%* | 35* |
| Global L1 lookup miss | 0.5211% | 0 | 12.3707%* | 0* |
| Unified L1 sector hit rate | 1.6386% | 7 | 24.6597%* | 28* |
| Generic TEX-L2 lookup hit | **17.7364%** | **51** | 8.5239% | 0 |
| Operation-read L2 lookup hit | 6.6859% | 5 | 4.3416% | 2 |

`*` SM86 NCU rows have only 283--286 scope-compatible joins in this report;
SM86 generic and operation-read endpoints are the scope-established rows.
The SM120 point estimate is below the updated `<18%` gate and the packet
observer removes the 724.72% regression, but one ordered-fallback zero-target
row remains above the strict `<=50` target. It is not suppressed with a
workload-name or NCU-derived exception.

The new report-only envelope is emitted by
`tools/report_locality_mrc.py --metric-comparisons ...`. It combines resident
packet lower/upper write-tag arms with the trace-native L2 interleaving delta:
SM120 coverage is **99.81% all launches (11,483/11,505)** and **99.46% nonzero
launches (4,082/4,104)**, with interval-distance Mean APE **0.4530%**. These
are conditional trace-derived intervals, not capacity-indexed silicon MRC
truth. HOTL WAPE remains **0.8767%/0.4462%** (SM120 line/sector) and
**1.0223%/0.5324%** (SM86 line/sector).

A general trace-only fallback was also evaluated for the residual: selecting the
resident lower arm when a packet is fully cold (`candidate == store requests`,
no resident store hits, and zero physical L2 lookup hits). It removes one
zero-target FP but worsens corpus Generic TEX-L2 Mean APE from 17.7364% to
20.9179% and weighted error from 21.2419% to 21.6974%; it was rejected under
the no-regression rule. The retained regime split is therefore the best
non-fitting point estimate, while the conditional envelope supplies the
uncertainty-aware result for that unidentifiable client-scope row.
