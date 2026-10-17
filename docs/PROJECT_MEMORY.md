# 3090 Ti / SM86 continuation memory

**Branch override (2026-10-15):** this checkout is the dedicated `3090ti_full_replay`
development branch. The active target is the local GeForce RTX 3090 Ti / GA102 /
`sm_86`, and the active corpus is `../sm86_rtx_3090_capture/`. The 5090/SM120
lane is historical context only for default execution on this branch. Keep
existing SM120 behavior buildable and untouched unless a scoped SM86 refactor
replaces a shared seam deliberately; do not let old SM120 CLI defaults or
assumptions obstruct SM86 work. The active implementation plan is
`plans/3090ti_full_replay_modeling_plan.md`.

The target endpoint is a default full replay that composes detailed SM86
instruction scheduling, sector-aware L1/L2, independently evidenced finite
partition/DRAM service, writeback/atomic/visibility tails, and explicit
IPC/frontier projections over the complete real corpus in
`../sm86_rtx_3090_capture/`. The current default is the integrated hierarchy
with aggregate carded read service, observer-only partition pressure, and
typed/optional tails; finite partition service and full completion tails remain
future evidence-gated work. NCU remains an independent post-replay comparison
target and never a replay input. Complete corpus runs still use the
repository-wide twelve-worker and `.jsonl.zst` artifact contract; only the
architecture-specific runner/configuration is being refactored.

Everything below this branch override is retained SM120 history for reference;
it does not supersede the SM86 target, corpus, or plan above.

**Updated:** 2026-10-15

## SM86 slice S1 — sidecar and full-replay runner

The architecture-neutral metadata adapter and SM86 runner are implemented in
`tools/sm120_replay_bench.py` and `tools/sm120_corpus_replay.py`. A real
three-root full/fast batch completed 3/3 in each mode. The complete-corpus full
run discovered 356 roots, completed 286 roots `ok`, retained 62 mixed-code
control dispositions and 8 replay/input failures, and replayed 12,392 launches
in successful rows in 32.845 seconds with 12 workers. Compressed output is
`../sm86_replay_results/full_corpus_20261015_v2/full.jsonl.zst`.

This proves runner/coverage termination only; full rows still carry conditional
dirty-writeback/coherence tails and no sub-10% NCU accuracy claim is made yet.

## SM86 slice S2 — conditional L2 write observer

`sim/sm/src/system.rs` and `sim/direct_run/src/metrics.rs` now expose an
operation-qualified L2 lookup observer. Ordinary store sectors use an
independent-evidence all-write-hit arm; RED/RMW remains separate, and physical
L2 state/DRAM/timing are unchanged. The refined full corpus completed 356 roots
(286 `ok`, 62 mixed-code controls, 8 replay/input failures) and 12,392
successful launches in 33.625 seconds. Exact NCU identity rows remained 107;
L2 lookup-hit mean APE improved 54.52%→35.04% and lookup-miss APE
84.20%→27.00%. This remains conditional metric coverage, not sub-10% closure.

## SM86 slice H2 — constant-domain demand semantics (2026-10-15)

H2 is implemented in `sim/trace/src/reader.rs`, `sim/sm/src/replay.rs`,
`sim/sm/src/metrics.rs`, and `sim/direct_run/src/metrics.rs`. Field 15 is an
active-and-predicate-on vector of bank-local effective CBANK offsets; field 16
is bank identity and field 17 is one scalar metadata offset. Field-17-only
records are retained as `metadata_only` and never converted into traffic.

The full-mode observer emits active offset references, checked per-instruction
32-B offset packets, launch-wide 32-B offset-domain cardinality, 64-B candidate
lines, and metadata-only records. It does not modify L1/L2 state, DRAM
service/timing, warp readiness, or IPC. The metrics are deliberately not
presented as GCC requests, DRAM sectors, or mathematical lower/upper bounds;
`LDG.*.CONSTANT` uses the ordinary traced global/L1/L2 path and needs a
separate static source-class model for GCC return traffic.

Representative full replay completed 4/4. Corrected complete-corpus full
replay completed 356 roots (286 `ok`, 62 mixed-code controls, 8 replay/input
failures), 12,392 successful launches, 34.263 s, and 12 workers. Artifact:
`../sm86_replay_results/full_corpus_h2_constant_20261015_v2/full.jsonl.zst`.
The post-replay report is
`../sm86_replay_results/metric_report_h2_constant_20261015_v3/`.

All 107 exact-identity rows with active constant vectors are covered, with
2,845,702 active-offset references and 3,840 metadata-only records. H2 leaves
S2 scored residuals unchanged: L2 hit 35.04% mean APE, L2 miss 27.00%, L2
source total 28.24%, DRAM reads 40.91%, SM-active IPC 14.57%, and elapsed IPC
12.41%. It is semantic/source coverage, not accuracy closure. The next slice
is H3: an evidence-backed, bounded SM86 fixed-latency register-read/RFC
envelope that keeps cache/DRAM state separate.

- Target: RTX 5090, `sm_120`; source `/root/GPU_simulator_rebase`; corpus
  `/root/sm120_rtx_5090_capture`; active plan is the proposal `PLAN.md`.
- Complete-corpus acceptance is real traces plus independent NCU comparison:
  339 roots and 11,558 launches per mode, full before fast, exactly 12 workers.
  NCU is never a replay input.
- The default alpha route now separates `SYNCS.PHASECHK` from async-copy-group
  waits; the redundant protocol flag was removed. SM120 full replay requires
  integrated effective L1+L2; flat is reserved for the fast frontier. The
  default-route corpus run is historical; the active integrated run is
  `replay_results/m6_default_integrated_l1l2_20261014/`.
- The source cleanup is committed as `2a356ced`; the integrated full-replay
  contract is `c0c9d787`.
- SM120 unified L1TEX projection uses sector-valid store/reduction terms.
  Integrated full replay emits conditional L1/L2 demand and hit-rate metrics;
  achieved DRAM service remains unavailable because the card marks bandwidth as
  a research gap. The effective L2 scenario has material residual and remains
  explicitly conditional.
- Landed source fixes separate TMA submissions from legacy async groups,
  preserve LDGSTS/TMA direction and barrier identity, reject fabricated counts,
  retain signed-operand precision loss, and keep diagnostic snapshots bounded
  and read-only. `completion_clock` remains unavailable.
- `sim-trace` streams plain or `.jsonl.zst` sidecars. Replay/comparison tools
  stream `.jsonl.zst`, enforce 12 workers, clean tracer preload variables,
  bound child output, and fail closed on malformed source/output.
- Full-replay and fast-replay JSONL outputs are now always retained and
  consumed as compressed `.jsonl.zst`; do not create or rely on new plain
  replay JSONL files.
- Conditional cache flags remain explicit because they name model assumptions,
  not hidden physical truth. Do not import external SOTA topology constants
  without target-scope evidence and corpus comparison.
- Temporary research is under `/dev/shm/sm120_alpha_tmp_20261013`; compact
  evidence is under `replay_results/`; historical review and superseded docs are
  under `adversarial_review/history/`. No fixture, hash, verification,
  admission, or Accel-Sim workflows.
- Proprietary scheduler/cache topology, TMA engine completion, universal clock
  conversion, and formal unseen-workload accuracy are documented non-prerequisites
  under the university PhD closure boundary.
- Post-remediation flat corpus run (commit `202132b6`) completed 339/339 roots
  and 11,558 launches in both full and fast modes with 12 workers; walls were
  84.541 s and 57.453 s. The async diagnostic snapshot cap was reduced to 1,024
  after an earlier large grouped-GEMM report exceeded the driver output bound.

## Permanent NCU-report rendering rule

- If a metric has no simulator prediction in **either** retained replay mode,
  do **not** render a repeated per-metric `N/A` accuracy table and do **not**
  print its raw NCU distribution. Aggregate such metrics into one compact
  “currently unavailable” section, grouped by metric family, and list names
  only.
- Show NCU observation distributions only for metrics emitted by at least one
  replay mode and therefore eligible for a like-for-like comparison. Keep the
  unavailable count and exact metric inventory in machine-readable output.
- `N/A` means missing model coverage—not zero error, a pass, or missing NCU
  data. Never silently score an unavailable prediction as zero.
- Distinguish **run-configuration unavailable** from **unimplemented**: a
  declared L1/L2/DRAM metric can be unavailable because the selected replay
  lane is `flat` (no corresponding memory unit), even though the emitter is
  implemented for an explicit `l1`/`l1-l2` lane. Reports must state this scope
  rather than implying the simulator has no implementation.
- Fresh plain Full effective-L1 rerun (serial sidecar preparation, 12 simulator
  workers) completed 339 roots / 11,558 launches `ok` in 77.886 s; output is
  `/dev/shm/sm120_rtx_5090_work/full_l1_refresh_plain_12worker_20261014/full.jsonl.zst`.
- Its refreshed NCU report scores 10 emitted metrics (including L1 lookup and
  model-relative SM-IPC rows) and groups 2,604 metrics unavailable in both
  lanes without repeating N/A tables or raw NCU distributions.
- Durable copies are retained under
  `/root/sm120_rtx_5090_capture/replay_results/full_l1_refresh_plain_12worker_20261014/`.

## Active effective-model research program

The accepted PhD endpoint is now an active effective-model program, not a wait
for proprietary topology. The detailed proposal is
`/root/gpu_cache_memory_modeling_proposal/NEXT_RESEARCH_PLAN.md`.

The next source/report slice is M0/M1: preserve concurrent report edits, then
consume existing `metric_predictions` and `cache_lookup_predictions` from
compressed replay artifacts before any new corpus run. Later slices address
trace-derived IPC/cycle envelopes, effective L1/L2, DRAM traffic/service bounds,
and synchronization attribution. Full corpus remains full-first, fast-second,
exactly 12 workers; NCU remains post-replay only.

M1 metric coverage is implemented in `tools/sm120_metric_report.py` (commit
`5c6621cc`). It reduces retained 339-capture full/fast artifacts without a new
replay, exposes emitted instruction/sector/lookup/L1 predictions, and groups
L2/DRAM metrics that have no simulator prediction. Compact evidence is under
`replay_results/metric_coverage_m1_20261014/`; detailed rows remain compressed
in the cleanup archive.

M2 initial IPC envelope is committed as `01307b1c`. It emits model-relative SM
and SMSP IPC rows from replay state. The required 339-root full/fast replay
completed with 12 workers; the full SM IPC row compared on 11,558 launches with
median absolute error 0.724 inst/cycle and P90 13.339. This is coverage and a
refinement target, not a physical IPC claim or a promoted accuracy model.

M2 refinement is accepted as an effective IPC projection: dividing the model
terminal frontier by the carded SM population reduced the full SM IPC residual
to median 0.00320 inst/cycle and P90 0.06578 over 11,558 launches. The SMSP
active-cycle target was absent. GPC/SM cycle endpoints remain separate residuals;
no physical IPC or clock-calibration claim is made.

The metric reducer's unavailable-family rendering was tightened in
`1107125f`: the compact report lists metric names with row counts and reasons,
while raw NCU distributions remain omitted for metrics unavailable in both
modes.

The first M3 L2 sensitivity batch ran four predeclared effective geometries on
four existing memory-heavy roots (12 workers, full mode). All scenarios completed
and were indistinguishable at 10.5925 pp median L2 error; no NCU-based parameter
selection or corpus promotion was made. The next L2 slice needs a request/sector
or address-stream discriminator rather than a wider blind geometry sweep.

A second predeclared effective-L2 capacity ladder (1/4/16/96 MiB) was also
indistinguishable on the same four roots at 10.5925 pp median error. Blind
geometry sweeps are stopped; future L2 work requires request/sector observability
or a trace that demonstrably exercises L2 state.

The SM120 DRAM-demand metric alias is committed as `6a0f0bf7`: effective-L2
reports now join `dram__sectors_op_read.sum`, the operation-qualified profile
label. A bounded four-root run compared all four launches with 247.5-sector
median error; this is coverage only, not DRAM service validation.

The complete effective-L2 alias run covered 339 roots and 11,558 launches per
mode with 12 workers. `dram__sectors_op_read.sum` compared on all full rows with
215-sector median absolute error (P90 1,162); L2 hit-rate residual was 34.84 pp
median (P90 60.48 pp). DRAM label coverage is accepted; effective L2 remains
conditional and unpromoted.

Family-stratified L2/DRAM residuals are archived in
`replay_results/m3_l2_alias_full_20261014/family_residuals.{json,md}`. Errors
vary materially across Cutlass, FlashInfer, quant, simbench, and SOL families;
this rules out another single global geometry constant as the next hypothesis.

The TMA load-payload demand arm (`2a1276ca`) was evaluated in the complete
full/fast effective-L2 corpus path. It preserved all roots and launches; DRAM
demand residual remained 215 sectors median and P90 1,080. It is retained as
conditional demand accounting, not physical DRAM service timing.

TMA-arm family residuals are archived at
`replay_results/m3_tma_demand_full_20261014/family_residuals.{json,md}`. The
conditional arm reduces Cutlass DRAM-demand median error to 942 sectors and
SASS to 216, while SOL/FlashInfer remain largely unchanged; it remains a
source-specific effective model, not a global DRAM rate.

A trial treating TMA payload sectors as L2 miss-side denominator was rejected:
it was non-discriminating on simbench and worsened the Cutlass L2-rate result.
TMA payload accounting remains DRAM-demand-only until a source-qualified L2
request relation is available.

M4 TMA store-demand projection (`b36eff25`) completed the full corpus path:
`dram__sectors_op_write.sum` compared on all 11,558 full rows with zero median
and P90 error and 11,470 exact rows. It is conditional demand coverage only,
not physical writeback/DRAM service timing.

The active-cycle IPC projection (`88e73070`) completed full/fast flat corpus
replay and compared on all 11,558 full launches with median absolute error
0.2106 inst/cycle and P90 0.4580. It is an effective IPC row, not physical
issue-width timing.

M5 synchronization attribution is implemented in `tools/sm120_sync_report.py`
(`51e34a62`). It reports existing full/fast query outcomes, async obligations,
unresolved generations, and readiness-cause cycles without adding engine timing
or changing replay scheduling. Compact output is under
`replay_results/m5_sync_attribution_20261014/`.


## Full-replay memory default (2026-10-14)

SM120 full replay now requires the integrated `l1-l2` memory path. The runner
defaults `--memory-mode l1-l2`; the direct binary resolves effective L1
`1024:none:50:512` and effective L2 `100663296:1:16` when full SM120
replay omits explicit overrides. It rejects full SM120 `flat` and L1-only
invocations. The complementary fast frontier remains flat. Full replay also
enables the typed NCU metric projection automatically; NCU values remain
post-replay comparison data, never replay inputs. Carded DRAM, coherence,
writeback, candidate service/admission, and latent-envelope controls remain
explicit because they add independent assumptions and cannot be blindly
composed into the integrated cache model.


## Carded DRAM default check (2026-10-14)

A bounded three-root real-trace probe requested `--carded-dram` after the
integrated L1/L2 default. Every row failed closed because the active SM120
architecture card types achieved DRAM service bandwidth as a research gap; the
1792 GB/s vendor peak is not a locked service rate. Carded DRAM therefore
remains explicit rather than silently becoming a fabricated default. Compact
evidence is `replay_results/m6_carded_dram_probe_20261014/`.


## M7 persistent stream-L2 evaluation (2026-10-14)

The corpus runner exposes the existing `StreamL2ReplaySession` through
`--cache-scope stream-l2`; launch-cold remains the default. A complete 339-root
full-first/fast-second stream-L2 run completed 11,558 launches per mode with
12 workers. One mixed-device/stream capture used the narrow launch-cold fallback
and was reported explicitly. Stream persistence reduced aggregate L2 hit-rate
error to 12.54 pp median / 60.79 P90 from 34.84 pp / 60.48 P90, but lookup
sector count residuals were not uniformly better (133/120 median hit/miss versus
48/32). It remains a conditional scope, not the default. Compact evidence is
`replay_results/m7_stream_l2_full_20261014/`.


## M8 address-derived TMA shadow (2026-10-14)

Commit `7e571236` retains compact source-sector identities from native TMA
sidecars and computes a separate effective L2 shadow observer. It does not
modify shared replay timing or cache state. The full 339-root run found 407,456
source-sector requests across 88 launches, with two unavailable address records.
The shadow is diagnostic/conditional only; it is not combined with NCU-facing
L2 projections because cross-source event order is not established.


A five-capture identity probe matched all 16 source-bearing TMA records to
dynamic instruction keys with zero unmatched records. This closes identity
coverage for the selected slice but not cross-warp temporal ordering; the shadow
remains diagnostic-only.


## M8 cycle residual stratification (2026-10-14)

`tools/sm120_cycle_residual_report.py` (`dbdd3f11`) produces a post-replay
family report with GPC/SM residuals and trace-side TMA/async features. The
full integrated report is `replay_results/m8_cycle_residual_20261014/`; it is
diagnostic and does not fit NCU values into replay.


## M8 effective GPC front-end envelope (2026-10-14)

Commit `976b9776` applies the retained fixed alpha front-end envelope to the
SM120 GPC effective projection and elapsed-IPC denominator, not the raw SM-active
endpoint. The full-corpus post-replay report is
`replay_results/m8_frontend_projection_20261014/`; it improves GPC APE from
0.827/0.882 median/P90 to 0.135/0.276 and improves 11,514 of 11,558 launches.
This is an effective observation model, not a physical front-end claim.


## M9 effective GPC projection scope (2026-10-14)

The retained alpha front-end envelope is applied only to the SM120 GPC effective
projection and elapsed-IPC denominator. Raw SM-active timing remains separate.
The full-corpus post-replay report is
`replay_results/m8_frontend_projection_20261014/`; no NCU-derived constant was
introduced.


## M9 validation subset (2026-10-14)

Five previously unselected real traces completed full-first/fast validation
with 12 workers under integrated L1/L2. The effective GPC front-end envelope
remains separate from raw SM-active timing. Compact evidence is
`replay_results/m9_frontend_validation_20261014/`.

## M10 TMA shadow demand decision (2026-10-14)

`tools/sm120_tma_shadow_dram_report.py` compares the legacy TMA payload-sector
DRAM-read demand projection with address-derived TMA shadow misses using only
post-replay NCU observations. Across 11,558 full rows both have 215-sector
median error; P90 is 1,080 versus 1,079. Shadow evidence exists on 88 rows,
with 24 improvements and 30 regressions; Cutlass improves while available
quant/SASS rows regress. Since source-side order is not established against
ordinary dynamic L2 events, the shadow remains diagnostic/conditional and is
not merged into replay cache or timing state. A fresh 339-root full-first/
fast-second run completed all 11,558 launches per mode with 12 workers under
integrated full L1+L2 and flat fast frontier.

## M11 TMA/ordinary request ordering (2026-10-14)

`tools/sm120_tma_order_audit.py` audited the full shadow replay and found zero
instrumentation-order records versus 5,716,345 missing-order records across
339 captures and 11,558 launches. The 88 TMA-source-shadow launches have no
shared ordering records. TMA shadow misses therefore remain diagnostic-only;
no unordered source stream is merged into ordinary L2 or DRAM replay.

## M12/M13 effective metrics (2026-10-14)

The SM120 native surface emits an effective `sm__cycles_elapsed.avg` endpoint
from raw modeled SM-active terminal time plus the fixed front-end envelope; raw
SM-active and GPC timing remain separate. A complete 339-root full-first/
fast-second run completed with 12 workers at
`replay_results/m12_sm_elapsed_full_20261014/`.

`tools/sm120_metric_report.py` now scores each metric/launch/mode once,
separates full and fast summaries, retains source/alternative provenance, and
reports relative and per-capture residuals without using NCU for selection.

M13 emits shared load/store/atomic request, broadcast-collapsed 32-bank
wavefront, and effective conflict metrics from the validated trace address
projection. It does not change replay timing/cache/DRAM state. Complete corpus
full-first/fast-second evidence is
`replay_results/m13_shared_wavefront_full_20261014/`, with 339 roots and
11,558 launches per mode successful under 12 workers. The projection is
conditional trace-derived metric coverage, not physical bank arbitration.

## M14 logical TMA opcode coverage (2026-10-14)

SM120 emits `smsp__inst_executed_op_tma_ld.sum` and
`smsp__inst_executed_op_tma_st.sum` from dynamic `UTMALDG`/`UTMASTG` opcode
identity. Complete full-first/fast-second corpus evidence is
`replay_results/m14_tma_opcode_full_20261014/`: 339 roots and 11,558 launches
per mode, all successful with 12 workers. Both metrics are exact across all
11,558 full rows; 88 rows have nonzero TMA activity. The result is logical
opcode coverage, not physical pipe or engine timing.

## M15 logical asynchronous opcode coverage (2026-10-14)

The SM120 native surface exposes dynamic-opcode-derived `LDGSTS` and
SASS-named TMA load/store count metrics. Complete full-first/fast-second corpus
evidence is `replay_results/m15_logical_async_opcode_full_20261014/`: 339 roots,
11,558 launches per mode, all successful with 12 workers. Five new full metrics
compare with zero median/P90 error; nonzero TMA activity occurs in 88 launches
and LDGSTS in 190. These are logical opcode counts, not physical pipe or
completion timing.

## M16 logical shared opcode coverage (2026-10-14)

SM120 emits logical shared load/store/atomic opcode counts, including
SASS-named rows. Complete evidence is
`replay_results/m16_shared_opcode_full_20261014/`: 339 roots and 11,558
launches per mode, full-first/fast-second, 12 workers. SASS shared-store is
exact on all rows and SASS shared-load exact on 11,431 rows; non-SASS target
names are absent from retained profiles and remain target-missing. Shared
atomic has zero median/P90 error with one nonzero-target discrepancy.

## M17 logical branch/reduction/LDSM coverage (2026-10-14)

SM120 emits trace-derived logical branch, global-reduction, and LDSM counts.
Complete evidence is `replay_results/m17_logical_misc_full_20261014/`, with
339 roots and 11,558 launches per mode under 12 workers. Global-reduction and
LDSM rows compare exactly. Branch remains conditional (median 36 instruction
absolute error / 40% relative), indicating a broader NCU branch endpoint than
currently retained trace events; no branch timing correction was made.

## M18 branch endpoint audit (2026-10-14)

A selected full-first/fast real-trace probe across warp-divergence, SOL, and
Cutlass matched the logical branch count to NCU on those launches. The broader
corpus residual is workload/association dependent; no branch rescaling or NCU
fit was introduced. Branch remains a conditional diagnostic metric.

## M19 logical global/local coverage (2026-10-14)

SM120 emits logical SASS-named global-load/global-store/local-load/local-store
counts from source-class trace metrics. Complete evidence is
`replay_results/m19_global_local_full_20261014/`: 339 roots and 11,558 launches
per mode, full-first/fast-second, 12 workers. Median/P90 absolute error is zero;
global load/store exact rows are 11,540/11,504 and both local rows are exact.
These are logical operation counts, not physical pipe/cache service.

## M20 total shared logical count (2026-10-14)

The native SM120 surface emits `smsp__sass_inst_executed_op_shared.sum` from
validated shared load/store/atomic operation classes. Complete evidence is
`replay_results/m20_total_shared_full_20261014/`: 339 roots and 11,558 launches
per mode, full-first/fast-second, 12 workers. The full metric is exact across
all rows. Remaining inventory is predominantly physical/PM-derived or service
endpoint gaps and remains typed unavailable.

## M21 logical LDGSTS route coverage (2026-10-14)

SM120 emits logical LDGSTS cache-access and cache-bypass counts from qualified
route identity. Complete evidence is
`replay_results/m21_ldgsts_route_full_20261014/`: 339 roots, 11,558 launches
per mode, full-first/fast-second, 12 workers. Both metrics are exact across
all full rows and remain logical route counts, not physical cache accesses or
completion timing.

## M22 typed metric-gap disposition (2026-10-14)

The M22 inventory is `replay_results/m22_metric_gap_review_20261014/`: 51
metrics have coverage and 2,509 remain unavailable in both modes. Remaining
families are mainly PM/warpsampling, peak/rate/derived, physical-pipe,
profiler, and service endpoints without direct retained logical trace
relations. No NCU fitting or hidden topology model was introduced.

## M23 SM-active envelope disposition (2026-10-14)

A selected real-trace probe tested applying the retained front-end envelope to
raw `sm__cycles_active.max`. It improved 11/19 rows and worsened 8/19, with
mixed family behavior and a worse P90 tail. The variant was not promoted; raw
SM-active remains separate from effective GPC/elapsed projections.

## 2026-10-15 — Round 3 effective-model accuracy program

Adversarial review dossier: `/root/gpu_cache_memory_modeling_proposal/adversarial_review/round3_effective_models_20261015/`.

Commit `76c6f757` promotes two bounded effective SM120 corrections after a
three-root real probe: bare `LDG.E` in the alpha route is cacheable unless an
explicit `.CG`, `.CS`, or STRONG.GPU form requests bypass; and NCU L1 sector
lookup projections convert timing-cache same-request merges at the observation
boundary (fresh 128B lines contribute four missed sectors). Physical cache
state, timing, and NCU input isolation are unchanged. The metric reducer now
reports mean APE as well as median/P90.

The probe completed 3/3 roots and 10/10 launches per mode with 12 workers;
global-load lookup hit/miss rows were exact on 9/10 compared rows and unified
L1 median APE was about 20% on the selected probe. M26 is the active next
slice: source-specific effective L2 sector-observer accounting. M27 is queued
for a bounded memory/occupancy cycle envelope with IPC derived from the same
projected denominator. Complete-corpus evaluation remains full `l1-l2` first,
fast flat second, exactly 12 workers, compressed JSONL, and no NCU replay input.

## 2026-10-15 — M26/M27 effective-model results

A complete stream-L2 comparison (`replay_results/m26_stream_l2_full_20261015`)
passed all 339 roots with 12 workers but was not promoted: mean L2 hit-rate
APE remained about 49% and lookup-hit APE worsened substantially. Launch-cold
integrated L1+L2 remains the default.

Commit `9bedc858` adds an observation-only SM-active endpoint blend (2/5 of the
causal-to-elapsed gap) and serializes the effective alpha profile. The complete
M27 replay (`replay_results/m27_active_envelope_full_20261015`) passed all
11,558 launches in full and fast modes. `sm__cycles_active.max` mean APE fell
to 19.63%; raw replay timing and active IPC denominator remain unchanged.
Active IPC remains a residual rather than an independently fitted metric.

## 2026-10-15 — Round 4 adversarial disposition

The M25/M26/M27 accuracy continuation was reviewed in
`/root/gpu_cache_memory_modeling_proposal/adversarial_review/round4_effective_accuracy_20261015/`.
M25 and M27 are retained as bounded effective observation projections; M26
stream-L2 persistence is a negative promotion result and launch-cold integrated
L1+L2 remains the default. Current comparisons are dominated by ordered-fallback
NCU joins (11,460 of 11,558; 98 identity), so future reports must stratify join
provenance, positive/zero-target coverage, and family residuals.

M27's serialized 2/5 active-endpoint blend does not modify replay timing or
active IPC. Its source/provenance and non-NCU selection must remain explicit;
otherwise it is development calibration, not independent accuracy evidence.
M28 is the bounded next slice: source/sector L2 observer or replay-visible
issue/occupancy IPC projection only, selected real-trace probe before any
complete-corpus run, full integrated `l1-l2` before fast flat, exactly 12
workers, compressed `.jsonl.zst`, and NCU comparison only after replay. If no
source-derived change improves critical families, preserve current defaults and
record the negative result rather than adding a fitted scalar.

## 2026-10-15 — M26 source/sector observer follow-up

The bounded post-replay analysis
`replay_results/m26_l2_observer_negative_20261015/` compared 11,500 full rows.
Ordinary/mixed L2 requests correlated with target totals at r=0.984, while
address-derived TMA shadow requests correlated at r=0.142 and shadow hits at
r=0.215. The ordinary-plus-shadow denominator remained directionally aligned
overall but failed TMA-only rows (target/shadow median 2.778, P90 4.283).
No source-qualified packetization or generation identity supports promotion;
launch-cold integrated L1+L2 remains default and the L2 branch is closed
negatively. M28 has only the bounded replay-visible IPC/occupancy opportunity.

## 2026-10-15 — M28 effective active IPC projection

Commit `e13b1c60` derives active IPC from the same effective SM-active
observation frontier used by M27, with raw active cycles as fallback. The
selected Cutlass integrated full probe covered two launches under 12 workers;
active IPC mean APE was 20.79%, active-cycle 11.75%, and elapsed IPC 17.58%.
Retrospective recomputation of the existing 11,558 M27 full rows reduced active
IPC mean APE from 201.97% to 24.79% (median 16.04%, P90 49.15%). This remains
an effective conditional observation projection, not physical IPC or NCU-fitted
service timing; no scalar tuning is promoted. A fresh complete-corpus replay is
separate from this post-reduction comparison if post-commit execution coverage
is claimed.

## 2026-10-15 — M28 IPC negative branch

Commit `e13b1c60` attempted to reuse the effective SM-active max frontier for
`sm__inst_executed.avg.per_cycle_active`. A selected Cutlass probe looked
promising, but the complete 339-root replay exposed a scope mismatch and mean
APE about 1,388.65%. Commit `d0713e21` restores the raw summed active-cycle
IPC denominator. Negative compact evidence is
`replay_results/m28_ipc_frontier_negative_20261015/`; no NCU-fitted scalar IPC
factor is accepted. M25 L1 and M27 active-cycle endpoint projections remain
accepted effective observations; IPC is residual.

## 2026-10-15 — Round 5 next effective-model program

The current accepted SM120 effective baseline is integrated L1+L2 full replay,
launch-cold by default, with M25 L1 route/lookup and M27 active-cycle endpoint
projections. M26 stream-L2 persistence was not promoted. M28's max-frontier
active-IPC denominator was rejected after complete-corpus replay; raw active IPC
remains the fallback. The complete adversarial review and Q29–Q35 proposal are
in `/root/gpu_cache_memory_modeling_proposal/
NEXT_EFFECTIVE_MODEL_PROJECT.md` and
`adversarial_review/round5_effective_program_20261015/`.

The next work targets low-accuracy L1/L2 hit-rate, source-conditioned L2 demand,
conditional service envelopes, and correctly scoped IPC aggregation. Nsight
Compute's request/sector/hit-rate and `.per_cycle_active` definitions require
metric-specific numerator/denominator/aggregation scope; no max frontier may
stand in for an average active-cycle denominator. L2 requests, lines, sectors,
misses, TMA payload, atomics/reductions, stores, and writeback remain separate
accounting classes. Unordered TMA source shadow remains diagnostic-only.

Q29–Q35 are planning goals, not claims of implementation: endpoint/join
stratification; source/sector L2 packetization observer; source-declared
persistence/service scenarios; per-SM/per-SMSP IPC aggregation; demand/service
separation; complete full-integrated-L1+L2 then fast corpus promotion with 12
workers; and metric-specific closure. NCU remains post-replay comparison data,
never a replay input or same-cell fitting signal. New replay outputs remain
`.jsonl.zst`; no fixture, hash/verification, Accel-Sim, or blind geometry sweep
is permitted.

## 2026-10-15 — M29 real-device prototype research

M29 archived two bounded RTX 5090 studies. `m29_ipc_occupancy_probe_20261015`
used eight grid/block configurations with five NCU repetitions and showed
active IPC values ~0.30/1.18/2.99 by block shape, stable across one/eight blocks;
elapsed cycles were a separate launch-wide endpoint. The candidate simulator
model is scope-matched per-SM/per-SMSP active-lifetime aggregation, not a max
frontier or fixed issue width.

`m29_l2_microbench_research_20261015` tested persistence-window, working-set,
prefetch, and dependent pointer-chain shapes. Persistence and prefetch were
negative under emitted SASS/metadata; working-set/dependency depth are
conditional demand/service evidence only. No default change, geometry claim,
persistence guarantee, or NCU-fitted service constant was accepted.

## 2026-10-15 — M30 report-contract correction

Commit `d44c53e3` updates `tools/sm120_metric_report.py` to schema version 4.
The post-replay reduction of existing M27 compressed full/fast streams is
archived at `replay_results/m30_report_contract_20261015/`.  It separates metric
family from workload family, distinguishes missing predictions from zero-target
opportunities, and preserves join/workload-family residual strata.  No replay
or NCU-driven model selection occurred; integrated L1+L2 and raw active IPC
remain unchanged.  Q32 is the next active slice, with M29's homogeneous IPC
probe treated as support for active-lifetime scope but not proof of the
heterogeneous NCU roll-up.

## 2026-10-15 — Q32 source audit

`SmActivity` already retains per-SM active-lifetime unions, per-SMSP active
unions, and per-SMSP instruction counts; SM120 active IPC is a weighted
sum-of-scope-counters ratio, not the rejected max-frontier, elapsed-cycle, or
card-population construction. The installed Nsight `.avg`/`.per_cycle_active`
semantics and zero-active population remain explicit conditional report scope.

A retrospective exploratory per-SM envelope (`sum(active + 0.4*E)`) reduced
active-IPC mean APE to about 22.8% on existing M27 rows, but reusing the global
2/5 envelope per SM is a new hypothesis. Per-SM compulsory instruction-sector
sets must be collected in `extract_per_sm_blocks` before a predeclared probe;
no default or complete replay change is accepted.

## 2026-10-15 — Q32 source/provenance outcome

Commits `1f201132`/`38cddd28` make `SmActivity` record per-SM distinct instruction-sector cardinality from
`extract_per_sm_blocks` under `front_end_instruction_read_sectors`. This is
observation provenance only and does not alter replay timing or raw weighted
active IPC. A selected real capture passed full integrated L1+L2 then fast flat
with 12 workers at `replay_results/q32_scope_probe_20261015/`; it had one active
SM, raw active-IPC APE 97.97%/96.59%, and no heterogeneous roll-up evidence.
The global 2/5 blend applied per SM improved those rows to 56.35%/54.83% but is
not source-qualified and was not promoted. Raw active IPC remains default.

## 2026-10-15 — Round 7 initial Q32/Q36 review

The first-pass dossier is
`/root/gpu_cache_memory_modeling_proposal/adversarial_review/round7_q32_q36_20261015/`.
It keeps active-IPC roll-up/zero-active policy, per-SM envelope semantics,
and corpus dependent-chain identity unresolved. A heterogeneous real-trace IPC
probe and bounded Q36 identity analysis are active; M30 incompatible endpoint
alternatives remain fail-closed and no broad replay is authorized yet.

## 2026-10-15 — Round 7 heterogeneous IPC probe

The real branch-by-block RTX 5090 probe at
`replay_results/m29_ipc_heterogeneous_scope_20261015/` used one CTA per SM,
64 KiB dynamic shared memory, and compute/memory splits 42/85/128 with three
NCU runs each. Active IPC was directionally closer to ratio-of-sums than an
unweighted mean-of-instance roll-up, supporting the existing weighted active
scope. NCU did not expose per-SM instances and interference remains, so Q32 is
narrowed but not closed; no new IPC projection or corpus replay was promoted.

## 2026-10-15 — Q36 identity branch closed negative

The committed semantic trace summary explicitly lists
`register_values_and_data_dependencies` under `not_observed`. Address/width/
allocation metadata cannot recover dynamic value-flow identity, so Q36 is
`unavailable_missing_dependency_identity` for the current corpus. Evidence is
`replay_results/q36_dependency_identity_negative_20261015/`; no chain/service
model or replay change was promoted.

## 2026-10-15 — Round 8 Q30/Q33 L2 review

The remaining L2 branch is under
`/root/gpu_cache_memory_modeling_proposal/adversarial_review/round8_q30_q33_20261015/`.
A valid observer must carry profiler-version/client-scope/unit/conservation
provenance. TMA payload/shadow, L2 demand, dirty-victim/writeback demand, DRAM
demand, and achieved service remain separate; M26's TMA-shadow negative result
still blocks blending or default promotion.

## 2026-10-15 — Q30/Q33 L2 branch closed negative

`replay_results/q30_client_scope_audit_20261015/` records the source audit.
`L2SourceSplit` already conserves typed L1-fill/bypass/untagged/store lanes
across global/local origins; TMA shadow lacks request identity and M26 rejects
its blend. L2 demand, DRAM demand, dirty eviction, writeback, coherence, and
service remain separate. No new default observer or replay mode was promoted.

## 2026-10-15 — M26 report-fidelity correction

Round 8 corrected the M26 `tma_only` error table: it had counted 88 rows while
the declared positive-shadow/zero-ordinary predicate contained 38. The compact
summary now reports n=38, mean absolute error 2554.53, median 2048, P90 4864.
The TMA-shadow negative decision and launch-cold default are unchanged.

## 2026-10-15 — M32/M33 L2 sector observer correction

Commits `56a65fa4`, `283e93a1`, and `c0c3015e` fix unconditional write-class
lookup hits and packetized store double-counting, expose typed DRAM/TMA demand
components, and remove an unproven generic L2 hit-rate alias. Complete M33
replay passed all 339 roots and 11,558 launches in full and fast modes with 12
workers. Source-specific L2 lookup-miss mean APE improved from 38.59% to 3.10%;
lookup-hit remains 61.24% and generic hit-rate is residual/unjoined. No physical
write policy, service, topology, or completion claim is promoted.

## 2026-10-15 — M34 metric-specific closure

M34 closure evidence is `replay_results/m34_closure_residual_20261015/`.
The corrected source-specific L2 lookup-miss observer achieves 3.10% mean APE
(P90 3.70%) over all 11,558 full launches. Lookup-hit 61.24%, L1 hit-rate
60.72%, elapsed cycles 15.82%, active IPC 201.97%, and proprietary service/
topology/completion endpoints remain residual or unavailable. This meets the
scoped effective/conditional PhD closure criterion without claiming RTL truth.

## 2026-10-15 — M37/M38 final closure boundary

Commit `c9504e95` corrects atomic source-unit lookup hits to present sectors.
M37 reran the complete corpus full-first/fast-second with 12 workers and all
11,558 launches passed. Source-specific lookup-miss mean APE is 2.88%; lookup
hit is 59.98% over 4,157 positive targets, with 7,401 zero targets and 191
false positives. M38 is the authoritative metric-specific closure report;
generic L2 aliases, physical service/topology, engine completion, and dynamic
chain identity remain residual/unavailable.

## 2026-10-15 — Raw L2 endpoint scope cleanup

Commit `212d6dc3` makes the broad SM120 generic L2 hit-rate descriptor typed
`scope_unestablished` in raw JSONL while preserving source-specific conserving
TEX-sector hit/miss predictions. A selected real full probe passed; no timing or
sector-count change occurred. M37 remains the complete coverage replay.

## 2026-10-15 — Q37 effective active IPC continuation

Adversarial review established that Nsight `.avg.per_cycle_active` is a
ratio-of-sums over the fixed SM population and cannot use the max/terminal
frontier as its denominator. Commits `2cc6f1d4`, `bbd3c344`, and `ce208df6`
retain the raw active-lifetime ratio diagnostically and use the existing 2/5
blended startup envelope per observed SM for the scored effective projection.
A selected real probe measured 18.87% mean APE. The complete 339-root replay
(Q37, full integrated L1+L2 then fast flat, 12 workers) passed all 11,558
launches per mode; active-IPC mean APE is 23.83%, median 15.52%, P90 49.25%.
This is an effective conditional projection, not physical IPC or a fitted NCU
factor. No further scalar tuning is authorized.

## 2026-10-15 — Q39 TMA destination-sector shadow

Commit `f14be77b` preserves TMA `dst_addresses` in the compressed sidecar
reader and projects distinct 32-byte destination sectors for UTMASTG stores.
The read-only TMA L2 shadow combines source sectors for loads and destination
sectors for stores only when address coverage is complete; it never changes
cache state, replay timing, issue scheduling, DRAM service, or completion
clocks. A 27-launch TMA probe preceded the Q39 complete replay. Q39 passed
339 roots and 11,558 launches per mode with 12 workers; canonical evidence is
`replay_results/m39_tma_destination_shadow_full_20261015/`. Source-specific L2
lookup-miss mean APE improved from 2.88% to 2.57%; hit remains 60.15% on
positive targets. The model is conditional source-qualified observation only.

Nsight's Profiling Guide distinguishes 32-byte sectors from 128-byte lines,
source/client L2 metrics, and request coalescing. Every capture's 2,169 TMA
store rows retained destination addresses; payload-byte sectors are not a safe
substitute. A future read-only source-specific L2-read endpoint may be useful,
but mixed hit policy and LRC coalescing remain explicit residuals.

## 2026-10-15 — Q40/Q41 read endpoint and L1 merge audit

Round 13 audited source-specific read-only L2 metrics and L1 lookup batches.
All 339 NCU profiles expose `lts__t_sectors_srcunit_tex_op_read` read and
lookup endpoints, but simulator L2 lookup counters mix read/store/reduction/
atomic classes. A 78-launch Q40 probe showed a source-split read candidate had
zero positive read hits and 6.97% mean relative miss error; the tentative
reporting-only endpoint was removed before commit. Operation-class counters and
LRC/coalescing scope are required before adding read targets.

The same audit rejected blindly adding all L1 merges as hits. The existing
`L1LookupBatch` sequence distinguishes same-request coalescing from older
pending merges and async-copy batches; any SM120 L1 observer must preserve
those distinctions. Q39 remains the latest landed L2 change.

## 2026-10-15 — Q42 operation-class L2 observer

Commit `018c6e8f` adds read/store/atomic/reduction lookup observer counters at
the shared-L2 request construction seam and emits a conditional read-only TEX
endpoint only on launches without TMA sidecars. Q42 selected real probes then
complete full-first/fast-second replay passed all 339 roots and 11,558 launches
per mode with 12 workers. Canonical evidence is
`replay_results/q42_l2_operation_class_full_20261015/`.

The read-only L2 miss endpoint compares 11,470 non-TMA launches with 0.3296%
mean APE (median/P90 0%); read-only hit remains 24.44% positive-target mean
APE. Mixed source-specific lookup, Q39 TMA destination shadow, and read-only
operation classes remain separate. The counters are observation-only and do
not alter cache state, timing, DRAM service, or completion clocks.

## 2026-10-15 — Q44 TMA load read endpoint

Commit `3bcf84b4` separates TMA load/store shadow counters and extends the
operation-class read-only L2 endpoint with complete TMA load source sectors.
Q44 full-first/fast-second replay passed 339 roots and 11,558 launches per
mode with 12 workers. Canonical evidence is
`replay_results/q44_l2_read_tma_shadow_full_20261015/`. Read-only miss mean
relative APE is 0.3474% over 11,556 rows (median/P90 0%); read-only hit remains
25.33% on 2,373 positive targets. Store shadow and mixed lookup remain
separate; no timing/service/completion semantics changed.

## 2026-10-15 — Q45 L1 pending-merge audit

Round 14 audited `L1LookupBatch` and Q44 real traces. The source retains
`first_fill_sequence` and same-request versus older-pending distinction, but
only async-copy batches currently enable pending-hit attribution. Aggregate L1
counters mix global/local and operation scopes; direct global-load comparisons
are invalid. No universal merge rule was source-supported, so no L1 default or
full replay change was made. A future observer needs origin/operation-class
merge counters and a predeclared real probe.

## 2026-10-15 — Q45 L1 merge provenance surface

Commit `debe8881` adds read-only `L1MergeAudit` fields to TraceMetrics,
separating same-request coalescing from older-pending merges by global/local
origin using the existing fill sequence. A selected 58-launch real probe
passed full integrated L1+L2 then fast flat with 12 workers at
`replay_results/q45_l1_merge_audit_probe_20261015/`. Quant traces show both
classes, pointer-chase shows same-request merges, and Cutlass TMA launches show
none. No L1 hit metric or timing model changed; operation/client scope remains
unresolved.

## 2026-10-15 — Q46 L1 metric interpretation negative result

Round 15 compared Q45 merge provenance alternatives on 56 non-TMA real rows.
The existing global-load observer was 30.54% mean relative error on 13 positive
targets; older-pending-as-hit worsened to 35.35%, same-request-as-hit to
201.30%, and both was worse. Mixed origin/client scope and zero targets remain
material. No L1 correction or replay was promoted; Q45 audit fields remain
diagnostic.

## 2026-10-15 — Q47 IPC occupancy recheck

Round 16 re-audited the existing M29 real RTX 5090 occupancy benchmark. It
already covers 32/128/512-thread shapes, one/eight-block populations, fixed
arithmetic work, and five NCU repetitions. Q37 implements the source-supported
ratio-of-sums population scope plus bounded startup envelope. Residual IPC is
occupancy/family dependent and no new service relation is exposed. Further
scalar IPC probing is closed; retain Q37's 23.83% residual.

## 2026-10-15 — Q48 L2 store endpoint negative

A tentative operation-class store endpoint was evaluated on 58 real launches.
Positive-target write-hit relative error was 466.7% and write-miss error
12.16%. TMA WriteAtomic line-tag hits differ from source-unit sector-present
lookup hits. The endpoint was removed before commit; store/atomic/reduction
counters remain provenance only. Q44 read-only miss remains authoritative.

## 2026-10-15 — Q54 L1 replacement rejected

A selected Q51 probe suggested older-pending global merges could improve L1
hit error, but Q54 complete replay produced 110.05% positive-target hit APE
and 1,480 zero-target false positives. Commit `8bffdd6d` removes the
replacement. Q45 route/origin provenance remains diagnostic; no universal L1
merge correction is accepted.

## 2026-10-15 — Q56 L1 constant-client negative

Commit `5e95bcd7` retains constant-client request/hit and route-scoped merge
provenance as diagnostics. A 148-launch probe testing a constant-excluding
global-load hit observer still produced 25.89% positive-target APE and 16
zero-target false positives. The candidate was removed; existing L1 scoring
remains unchanged.

## 2026-10-15 — Q50/Q51/Q54/Q56 final scope

Atomic/reduction operation counters remain diagnostic because generic atom NCU
endpoints are absent and reduction targets in the bounded corpus are zero.
Q51 older-pending L1 selected results did not generalize: Q54 full replay
produced 110.05% hit APE and 1,480 zero-target false positives. Q55/Q56 added
route and constant-client provenance, but the constant-excluding replacement
still produced 25.89% positive-target APE and was removed. The default L1
observer remains unchanged; physical and proprietary gaps remain explicit.

## SM86 slice H3 — fixed register-read/RFC envelope (2026-10-15)

H3 is implemented in the native issue actor and direct metric emitter. Full
SM86 replay uses a bounded effective two-bank/one-port register-read window,
three-entry-per-bank RFC queue, deterministic key/eviction, transactional
per-SMSP reservations, and typed unsupported-shape handling. It is a scoped
effective model, not proprietary RF truth; cache/L1/L2/DRAM request paths are
unchanged. The emitted `applied_wait_cycles` is an overlapping causal
readiness exposure; `port_conflict_cycles` is only residual same-cycle
reservation conflict.

Five real full roots completed 5/5. Corrected complete-corpus full replay
completed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches,
34.431 s, 12 workers. Artifact:
`../sm86_replay_results/full_corpus_h3_rf_20261015_v2/full.jsonl.zst`.
The post-replay report is
`../sm86_replay_results/metric_report_h3_rf_20261015_v4/`.

Across successful launches H3 recorded 27,167,662 reservations, 3,622,126
RFC hits, 6,908,102 misses, and 133,572 typed unsupported-shape records. L2
hit/miss/source, DRAM-read, and H2 source metrics are unchanged. The two
complete exact-identity IPC endpoints move slightly upward in mean APE
(18.08%→18.26% for `sm__cycles_active.avg`, 12.41%→12.53% elapsed IPC), so
H3 is retained as a causal mechanism/coverage arm but not promoted as an
accuracy gain. H4 is next: separate fetch-last-issued state from issue hold
across Yield instructions.

## SM86 slice H4 — fetch-last-issued / Yield (2026-10-15)

H4 separates the front-end fetch continuation state from issue hold. A
persistent per-SMSP `fetch_last_issued` warp is used while the instruction
buffer has capacity; Yield clears only issue hold. State is cleared on warp
retirement. This is a front-end semantic correction and does not change
cache/L1/L2/DRAM state directly.

Five real full roots completed 5/5. Complete full replay completed 356 roots
(286 `ok`, 62 controls, 8 failures), 12,392 launches, 34.826 s, 12 workers.
Artifact:
`../sm86_replay_results/full_corpus_h4_fetch_yield_20261015_v1/full.jsonl.zst`.
The report is `../sm86_replay_results/metric_report_h4_fetch_yield_20261015_v1/`.
It emits real continuation/separation diagnostics but leaves exact 107-row
APE unchanged from H3: DRAM 40.91%, active-cycle 18.26%, elapsed IPC 12.53%.
H4 is coverage, not an accuracy gain. Next is a timing-active static
`LDG.CONSTANT`/GCC source-class arm or another independent IPC denominator seam.

## SM86 slice H5 — static LDG.CONSTANT source class (2026-10-15)

H5 adds static `LDG.*.CONSTANT` source-qualified counters separate from H2
CBANK metadata. It emits dynamic source-class instructions, sectors,
per-instruction 128-B line candidates, and CTA/warp-scoped unique candidates;
none is labeled as GCC truth or DRAM service, and no memory timing/state path
changes.

The representative full batch completed 5/5. Complete full replay completed
356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches, 34.915 s, and
12 workers. Artifact:
`../sm86_replay_results/full_corpus_h5_ldg_constant_20261015_v1/full.jsonl.zst`.
The report is `../sm86_replay_results/metric_report_h5_ldg_constant_20261015_v1/`.
Successful rows emitted 6,807,596 source-class instructions and 7,142,629
request candidates. Exact DRAM/IPC residuals are unchanged (40.91%, 18.26%,
and 12.53% mean APE). H5 is coverage, not accuracy closure. Next is an
independently evidenced GCC/L1.5 scope discriminator or bounded DRAM service
arm.

## SM86 slice H6 — GCC/L1.5 scope candidate (2026-10-15)

H6 adds CTA-scoped 128-B candidate lines for static `LDG.*.CONSTANT` records,
retaining warp-scoped candidates in parallel. This is source-domain scope
observability only; no GCC multiplier, cache state, DRAM service, or timing
constant is inferred.

Four representative roots completed full replay 4/4. Complete full replay
completed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches,
35.194 s, and 12 workers. Artifact:
`../sm86_replay_results/full_corpus_h6_gcc_scope_20261015_v1/full.jsonl.zst`.
The report is `../sm86_replay_results/metric_report_h6_gcc_scope_20261015_v1/`.
Successful rows emitted 1,039,257 warp candidates and 982,844 CTA candidates;
exact DRAM/IPC residuals remain 40.91%, 18.26%, and 12.53% mean APE. H6 is
not an accuracy gain; the next arm is a bounded service/denominator model.

## SM86 negative IPC denominator probe — 2026-10-15

A four-root full probe tested replacing the GPC launch-envelope denominator
with the SM-active observation frontier for elapsed IPC. It raised mean APE to
51.81% versus the retained 12.53% GPC endpoint, so the code change was
reverted before corpus promotion. This confirms the two observation domains
cannot be substituted without an independent elapsed-scope relation. Evidence
is `docs/sm86_ipc_denominator_negative_20261015.md`.

## SM86 slice H8 — DRAM partition-pressure envelope (2026-10-15)

H8 adds observer-only line-ID modulo-12, XOR-fold modulo-12, and modulo-6
candidate pressure arrays. GA102 whitepaper evidence supports twelve memory
controllers and six L2 slices, but not the physical hash; generic public
address decoders were not imported. Candidate pressure never enters the
aggregate G/D/1 server, response timing, cache state, or IPC.

Five representative roots completed full replay 5/5. Complete full replay
completed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches,
35.409 s, and 12 workers. Artifact:
`../sm86_replay_results/full_corpus_h8_dram_partition_20261015_v1/full.jsonl.zst`.
Report: `../sm86_replay_results/metric_report_h8_dram_partition_20261015_v1/`.
Exact residuals remain unchanged: DRAM 40.91%, L2 hit/miss/source
35.04%/27.00%/28.24%, active/elapsed IPC 18.26%/12.53%. H8 is retained as
DRAM topology observability; timing-active partition selection remains blocked
by unidentified hashing and weak independent NCU correlation.

## SM86 slice H9 — near/far pressure uncertainty (2026-10-15)

H9 adds low/high near/far candidate pressure edges and map-uncertainty width
from H8 partition hypotheses. It does not select a physical swizzle or alter
DRAM timing. Luo et al. (2025) is used only as a partitioned-cache mechanism
prior; A100/Hopper latency/counts are not imported. Independent RTX 3090 Ti
memtest repeats support a future [810,850] GB/s forced-DRAM interval but are
unlocked-clock observations, so the 1008 GB/s carded default remains unchanged.

Five real roots completed full replay 5/5. Complete full replay completed 356
roots (286 `ok`, 62 controls, 8 failures), 12,392 launches, 35.059 s, and 12
workers. Artifact:
`../sm86_replay_results/full_corpus_h9_near_far_20261015_v1/full.jsonl.zst`.
Report:
`../sm86_replay_results/metric_report_h9_near_far_20261015_v1/`.
Exact residuals remain unchanged from H8; mapping/service timing is deferred.

## SM86 slice H10 — effective DRAM service rate (2026-10-15)

The full runner now supports and defaults an explicit SM86 carded DRAM
service-rate hypothesis of 825,000,000,000 B/s, based on repeated independent
RTX 3090 Ti forced-DRAM memtest runs (~810–850 GB/s range under unlocked
clocks). The vendor 1008 GB/s card peak remains reference context; physical
partition/row mapping is unknown and the scenario is recorded in replay
summaries rather than hidden.

Four representative roots completed the scenario. Complete full replay
completed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches,
35.094 s, 12 workers. Artifact:
`../sm86_replay_results/full_corpus_h10_dram_rate_825_20261015_v1/full.jsonl.zst`.
Report:
`../sm86_replay_results/metric_report_h10_dram_rate_825_20261015_v2/`.
Exact timing residuals improve slightly (GPC 13.14→13.05%, active 18.26→18.19%,
active max 18.11→18.00%); DRAM sectors remain 40.91%. This is promoted as an
effective uncertainty-aware scenario, not physical bandwidth truth.

## SM86 slice H11 — source-composed DRAM demand envelope (2026-10-15)

H11 emits lower/candidate-high source demand edges from L2-missing data,
compulsory instruction sectors, static LDG.CONSTANT candidate lines, and H2
CBANK offset candidates. It is an uncertainty envelope only; achieved DRAM
traffic, service completion, partition mapping, and GCC request truth remain
separate.

Representative full replay contained three of four independent NCU DRAM
observations inside the envelope. Complete full replay with default 825 GB/s
service scenario completed 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 launches, 35.846 s, 12 workers. Artifact:
`../sm86_replay_results/full_corpus_h11_dram_demand_20261015_v1/full.jsonl.zst`.
Report:
`../sm86_replay_results/metric_report_h11_dram_demand_20261015_v1/`.
DRAM mean APE remains 40.91%; residual audit shows no useful global
correlation with constant/GCC candidate counts, so no multiplier is promoted.

## SM86 slice H12 — atomic conflict exposure (2026-10-15)

H12 adds trace-exact same-sector atomic conflict diagnostics: conflict-bearing
and distinct instruction counts plus alias sectors. It does not apply a
latency/multiplier or alter L1/L2/DRAM state. Independent notes show
conflict-vs-distinct atomic duration differences but do not identify a safe
per-transaction cycle relation.

Four real atomic roots completed full replay 4/4. Complete replay completed
356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches, 35.578 s,
12 workers. Artifact:
`../sm86_replay_results/full_corpus_h12_atomic_conflict_20261015_v1/full.jsonl.zst`.
Report:
`../sm86_replay_results/metric_report_h12_atomic_conflict_20261015_v1/`.
Across successful rows H12 observed 58 conflict instructions, 622 distinct
instructions, and 198 alias sectors; exact DRAM/L1/L2/IPC residuals remain
unchanged. Locked atomic microbenchmark evidence is the next condition for
any timing-active arm.

## SM86 H13 atomic service prior — not promoted (2026-10-15)

Independent warm-L2 RTX 3090 Ti atomic microbenchmarks show same-word conflict
overhead of roughly 10–18% versus distinct addresses across 1–4 warps. This is
not a per-transaction DRAM or queue constant, and the current request seam
loses lane conflict multiplicity. H13 retains H12 conflict diagnostics and
rejects timing promotion until a locked local probe and lossless conflict
identity exist. Exact residuals remain DRAM 40.91%, active IPC 18.19%, elapsed
IPC 12.42%.

## SM86 slice H14 — local-memory width observer (2026-10-15)

H14 retains producer `data_width` and emits local-memory source candidates
without changing L1/L2/DRAM state or timing. Five roots completed full replay
5/5. Complete replay completed 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 launches, 35.671 s, 12 workers. Artifact:
`../sm86_replay_results/full_corpus_h14_local_width_20261015_v1/full.jsonl.zst`.
Observed 4,892 width records, no unknown widths, and 19,824 candidate sectors.
Report:
`../sm86_replay_results/metric_report_h14_local_width_20261015_v1/`.
Residuals remain unchanged and local rows are sparse in exact NCU joins; this
is source coverage, not accuracy closure.

## SM86 slice H15 — RF Control/Allocate decoupling (2026-10-15)

H15 retains H3's bounded transactional RF/RFC observer but removes RF-port
conflict from scheduler eligibility. Huerta MICRO'25 Ampere evidence places
fixed-operation RF reservations in Control/Allocate; CGGTY issue may continue
while the downstream read window reserves. Cache/DRAM paths are untouched.

Four high-RF roots completed full replay 4/4. Complete replay completed 356
roots (286 `ok`, 62 controls, 8 failures), 12,392 launches, 35.109 s, 12
workers. Artifact:
`../sm86_replay_results/full_corpus_h15_rf_allocate_20261015_v1/full.jsonl.zst`.
Report:
`../sm86_replay_results/metric_report_h15_rf_allocate_20261015_v1/`.
Mean APE improves GPC 13.05→12.91%, active 18.19→18.01%, SMSP 17.69→17.50%,
elapsed IPC 12.42→12.31%; DRAM remains 40.91%. H15 is promoted as a bounded
effective IPC refinement, not physical RF truth.

## SM86 H16 L1 tag-lookup observer — 2026-10-15

The primary L1 NCU lookup rows previously emitted resident-data hits only, while
the cache retained fresh-line and pending-merge provenance. H16 changed the
bounded observer to `resident_hits + merge_observations - line_alloc_misses`
(clamped to the lookup denominator). This is a profiler-like tag observation;
it never changes L1 state, fills, MSHRs, replacement, L2/DRAM, or timing.

Representative real full-first roots completed 4/4; complete replay completed
356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in 36.054 s with
12 workers. H16 artifact:
`../sm86_replay_results/full_corpus_h16_l1_lookup_20261015_v1/full.jsonl.zst`;
report: `../sm86_replay_results/metric_report_h16_l1_lookup_20261015_v1/`.

Exact mean APE moved L1 global-load lookup hit 33.43%→30.71% and miss
206.41%→58.44%; L2/DRAM/IPC stayed 28.24%/40.91%/12.31%. The result is a
meaningful observer fix but not sub-10% closure. Remaining L1 error is assigned
to card-backed set/way/hash behavior and async/same-request packetization; do
not fit NCU rows or import an undocumented hash.

## SM86 H17 card-backed L1 set/way state — 2026-10-15

H17 adds optional set-local LRU state to `sim-mem::L1Cache`. SM86 full replay
selects the architecture-card 256-set × 4-way modulo scenario after resolving
the p-chase capacity/mshr point; the hidden GA102 hash is not guessed. The
fully-associative arm remains available for sensitivity. Pending fills,
generation guards, MSHR limits, dirty checks, and metrics are unchanged except
for set-local eviction decisions.

Representative five-root full-first replay passed 5/5; complete replay passed
356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in 36.517 s with
12 workers. Artifact:
`../sm86_replay_results/full_corpus_h17_l1_setway_20261015_v1/full.jsonl.zst`;
report: `../sm86_replay_results/metric_report_h17_l1_setway_20261015_v1/`.

Against H16, L1 hit/miss APE improved 30.71%→30.39% and 58.44%→57.74%; GPC,
SM, SMSP, and elapsed IPC improved to 12.75%/17.86%/17.36%/12.12%. L2 source
APE regressed 28.24%→32.43%; DRAM stayed 40.91%. H17 is therefore a bounded
effective topology promotion with an explicit L2 residual, not SOTA physical
hash recovery. Next work should test L2 packetization/hash alternatives and the
independent store/fence tail.

## SM86 H18 L2/hash uncertainty — 2026-10-15

The H17 fixed modulo L1 scenario caused a real high-stride alias regression:
streaming roots' L2 lookup sectors rose 32,896→82,048 while DRAM sectors stayed
32,768. H18 added `ModuloXorFallback`, a pressure-bounded effective unknown-hash
arm: use modulo until a set is full, then use the XOR-fold candidate only when
it has more room. This uses traced line occupancy and architecture-card geometry
only; it is not an NCU-fitted selector or a proprietary hash claim.

Representative full replay passed 5/5; complete full-first replay passed 356
roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in 36.751 s with 12
workers. Artifact:
`../sm86_replay_results/full_corpus_h18_l1_hashfallback_20261015_v2/full.jsonl.zst`;
report: `../sm86_replay_results/metric_report_h18_l1_hashfallback_20261015_v2/`.
The final fallback code also passed the working-set, L2-thrash, vector-copy,
and both vendor-SGEMM holdout roots 5/5 in
`../sm86_replay_results/h18_l1_hashfallback_20261015_v2_representative/full.jsonl.zst`.

H18 restored L2 source APE 32.43%→28.24% versus H17 and left DRAM 40.91%
unchanged, at the cost of returning L1/IPC residuals to the H16/H17 sensitivity
range (L1 30.71/58.44%, elapsed IPC 12.31%). Keep fixed modulo and XOR-fold as
explicit sensitivity modes; default full replay uses the bounded fallback until
more independent hash evidence exists.

## SM86 H19 source-qualified store/fence tail — 2026-10-15

H19 promotes `arch_profiles/candidates/sm86_store_fence_candidate_v1.json` as
the default SM86 full-replay feedback candidate. `sim-sm` now recognizes
MEMBAR.GL/GPU/SC.GPU as device-scope completion fences and gates admission on
traced global STORE/RED plus a device fence or ERRBAR/CCTL marker. The existing
transactional queue/write-ack path remains the only timing seam; dirty
writeback/coherence completion is still explicitly unavailable.

The independent clock64 store/fence probe supplies the bounded queue/service
prior; NCU is never a replay input or candidate selector. Representative full
replay passed 5/5. Complete default full-first replay passed 356 roots (286
`ok`, 62 controls, 8 failures), 12,392 launches in 36.577 s with 12 workers:
`../sm86_replay_results/full_corpus_h19_store_fence_default_20261015_v2/full.jsonl.zst`.
Diagnostics show 1,069 qualified launches, 103,737 acknowledgements, 378,742
write-completion requests, and 104,805 busy cycles.

H19 did not change exact corpus APE (L1 30.71/58.44%, L2 28.24%, DRAM 40.91%,
elapsed IPC 12.31%) because the qualified completion frontier is usually not
the scored launch critical path. Keep it as the default effective tail and
investigate a mechanism-local critical-path condition next; do not fit an NCU
residual.

## SM86 H20 dependent-SFU frontier — 2026-10-15

H20 promotes a narrow static same-register SFU dependency arm: `MUFU.*` and
related special-function forms such as `MUFU.RCP Rn,Rn` receive a 30-cycle
per-warp result frontier on SM86, while the existing four-cycle initiation
cadence remains. The value comes from an independent local RTX 3090 Ti clock64
probe; it is not an NCU fit or a proprietary latency claim.

Five SFU roots passed full/fast replay 5/5. Complete default replay passed 356
roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in 37.530 s with 12
workers. The corpus observed 269 launches with SFU exposure, 970,993 limiting
cycles, and 1,345,369 exposure cycles. Artifact:
`../sm86_replay_results/full_corpus_h20_sfu_dependent_20261015_v1/full.jsonl.zst`;
report: `../sm86_replay_results/metric_report_h20_sfu_dependent_20261015_v1/`.

Exact APE remained neutral (L2 28.24%, DRAM 40.91%, elapsed IPC 12.31%), so H20
is mechanism evidence rather than accuracy closure. Keep the gate narrow and
look for a stronger critical-path discriminator next.

## SM86 H21 causal IPC frontier decomposition — 2026-10-15

H21 adds explicit issue-frontier, terminal-frontier, and interval-width native
rows for causal IPC analysis. They are source-derived simulator boundaries and
remain separate from NCU prediction selection and unmodeled writeback/coherence
tails.

Five critical-path roots passed full replay; complete default replay passed 356
roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in 38.560 s with 12
workers. Artifact:
`../sm86_replay_results/full_corpus_h21_critical_frontier_20261015_v1/full.jsonl.zst`;
report: `../sm86_replay_results/metric_report_h21_critical_frontier_20261015_v1/`.

L1/L2/DRAM/SM/IPC residuals remained unchanged, so H21 is observability rather
than closure. Keep the frontier decomposition active for the next mechanism
selection and do not fit NCU residuals.

## SM86 H22 L0C miss-switch window — 2026-10-15

H22 consumes the public Huerta Ampere four-cycle immediate-constant-cache
miss-switch prior already present in the SM86 architecture card. A per-SMSP
countdown keeps the held/greedy warp selected only when ordinary
control/dependency/resource readiness is satisfied; the IMC response frontier
is excluded for this bounded decision. Every effective issue reloads the
window, and a parked greedy miss consumes one opportunity. The candidate is
active in default SM86 full replay; NCU values are not replay inputs.

The additive `greedy_switch_block_cycles` diagnostic raises the IMC readiness
schema to 27. It measures overlapping scheduler opportunities, not elapsed
cycles or NCU warp sampling. A static 46-cycle ERRBAR/post-EXIT special was
researched but remains deferred because dynamic corpus rows contain no executed
ERRBAR records.

Representative full-first replay passed all four discovered roots (485 full
launches). Complete full-first replay passed 356 roots (286 `ok`, 62 controls,
8 replay/input failures), 12,392 launches in 37.195 s with twelve workers:
`../sm86_replay_results/full_corpus_h22_l0c_switch_20261015_v2/full.jsonl.zst`;
the complementary fast artifact is beside it. The post-replay report is
`../sm86_replay_results/metric_report_h22_l0c_switch_20261015_v2/`.

Across the corpus, IMC tag misses stayed 1,174,632, L1/L2/DRAM counters were
bit-for-bit unchanged, miss-wait events rose to 3,125,368, and bounded greedy
switch blocking measured 1,489,412 cycles. Only six terminal/GPC frontier rows
shifted by one or two cycles (aggregate −6 cycles). Mean APE stayed neutral:
L1 lookup hit/miss 30.71%/58.44%, L2 28.24%, DRAM 40.91%, SM active 18.01%,
elapsed IPC 12.31%, and GPC 12.91%.

Known residual: after the countdown expires, a non-greedy candidate with an IMC
miss is parked without scanning the next ready sibling. The next slice should
add a transactional, bounded non-greedy candidate scan and re-evaluate whether
it changes critical frontiers rather than merely overlapping wait diagnostics.

## SM86 H23 non-greedy L0C candidate scan — 2026-10-15

H23 completes the public Huerta/Ampere immediate-constant issue-loop behavior:
after an expired greedy L0C miss, the held warp is probed first and remaining
ordinary-ready siblings are scanned in deterministic youngest order. Non-greedy
misses retain their cloned-cache MSHR reservations and park their wait frontiers;
only the final hit is an effective issue pick. Memory-admission feedback and
instruction accounting see only that final pick. The arm is default in full
SM86 replay and never uses NCU values as input.

Representative post-ordering full-first replay passed 10/10 real roots (2,728
full launches). Complete full-first replay passed 356 roots (286 `ok`, 62
controls, 8 replay/input failures), 12,392 launches in 40.715 s with twelve
workers:
`../sm86_replay_results/full_corpus_h23_l0c_scan_20261015_v2/full.jsonl.zst`;
the complementary fast artifact is adjacent. Report:
`../sm86_replay_results/metric_report_h23_l0c_scan_20261015_v2/`.

The scan emitted 21,574,287 attempts, 21,348,688 misses, and 225,599
successes. IMC wait events are 111,823,998 because pending candidate probes
overlap; they are not elapsed cycles. L2 and DRAM counters are unchanged from
H22; L1 hits/misses/merges change −104/+104/+234. Only 74 launches move
GPC/terminal frontiers (aggregate +392 cycles, range −147…+505). NCU reduction
is neutral at reported precision (L1 lookup hit/miss 30.71%/58.44%, L2 28.24%,
DRAM 40.91%, SM active 18.01%, elapsed IPC 12.31%, GPC 12.91%).

Rows with scan activity use readiness schema 28; rows with only the H22 IMC
surface use schema 27. This is source-aligned scheduler plumbing, not IPC
closure. Next work should address typed terminal/writeback/coherence or issue
resource residuals, and should not turn probe counts into fitted timing.

## SM86 H24 consecutive-BAR generation re-arm — 2026-10-15

H24 activates the SM86 CTA-barrier generation re-arm at 28 cycles, only before
a later `BAR.SYNC` generation of the same CTA. A fresh local RTX 3090 Ti
clock64 run of the retained barrier-pipeline probe measured 688,134 cycles for
16,384 one-barrier generations and 1,146,885 for two consecutive barriers,
an exact ~28-cycle difference per generation. The two-cycle warp-arrival server
and ordinary post-barrier instructions remain unchanged.

Representative high-BAR real roots passed full-first replay. Complete default
full-first replay passed 356 roots (286 `ok`, 62 controls, 8 replay/input
failures), 12,392 launches in 40.694 s with twelve workers:
`../sm86_replay_results/full_corpus_h24_barrier_rearm_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h24_barrier_rearm_20261015_v1/`.

Only 29 launches move versus H23; GPC/terminal frontier sum increases by 1,847
cycles. APE changes slightly (GPC 12.90596→12.90078%, SM active
18.00767→18.02024%, elapsed IPC 12.31350→12.30731%) while L1/L2/DRAM remains
30.71/58.44/28.24/40.91%. This is a bounded source-backed mechanism, not
silicon barrier closure. Next work targets eligible-not-selected, front-end,
and L1TEX dependency exposure or a typed fixed-result-queue observer; no
unsupported execution-to-writeback release timing is allowed.
## SM86 H25 dynamic-youngest priority audit — 2026-10-15

H25 retains block-local dynamic warp IDs in the native replay and uses them as
the deterministic youngest-order tie-break. A within-CTA arrival-vs-dynamic-ID
inversion counter makes the Huerta greedy-then-highest-ID assumption explicit;
all 12,392 full launches report zero inversions, so timing is unchanged.

The independent fixed-latency result-capacity clock64 probe on the RTX 3090 Ti
reproduces a multiwarp W8 threshold (+0.163 then +1.118 cycles/generation),
but W1/W4 are null and a collision/separated control is non-monotonic. Queue
capacity and execution-to-writeback release remain unidentified; no 8-entry/
one-pop timing arm is enabled.

Representative high-BAR roots passed full replay. Complete full-first replay
passed 356 roots (286 `ok`, 62 controls, 8 replay/input failures), 12,392
launches in 40.456 s with twelve workers:
`../sm86_replay_results/full_corpus_h25_priority_audit_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h25_priority_audit_20261015_v1/`.
Readiness schema is 29 and H25 is bit-for-bit neutral versus H24 for L1/L2/
DRAM/IPC/frontier outputs. Retain the probe as negative mechanism evidence;
next work targets dominant eligible-not-selected/front-end/L1TEX dependency
residuals without inventing queue release timing.

## SM86 H26 branch-stream redirect experiment — rejected — 2026-10-15

H26 tested flushing per-SMSP fall-through instruction-stream lines on dynamic
non-fall-through branches. Five branch-heavy roots passed representative replay;
the complete A/B run passed 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 launches with twelve workers:
`../sm86_replay_results/full_corpus_h26_branch_redirect_20261015_v1/full.jsonl.zst`.

The experiment falsified the broad flush hypothesis: GPC APE worsened
12.90%→14.37%, SM active 18.02%→22.03%, and elapsed IPC 12.31%→12.73%, with no
L1/L2/DRAM improvement. The timing change was removed from the active code path;
negative evidence is retained in `docs/sm86_branch_redirect_slice_20261015.md`.
Future front-end work needs a narrower static target/prediction model or an
observer only. The active baseline remains H25/H24.

## SM86 H27 L1TEX/front-end redirect observer — 2026-10-15

H27 retains the rejected H26 branch-flush timing result and adds only a native
observer for dynamic non-fall-through control transfers. It does not mutate
stream buffers or timing; readiness schema 30 carries the count.

Five branch-heavy roots passed representative full-first replay. Complete
full-first replay passed 356 roots (286 `ok`, 62 controls, 8 replay/input
failures), 12,392 launches in 40.452 s with twelve workers:
`../sm86_replay_results/full_corpus_h27_l1tex_observer_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h27_l1tex_observer_20261015_v1/`.

The corpus contains 1,882,174 dynamic non-fall-through transfers and 85,461,673
front-end fetches. H27 is bit-for-bit neutral versus H25 for L1/L2/DRAM,
IPC/frontiers, and APE. Use the observer to partition future front-end/L1TEX
models by static target shape; never convert transfer counts into a universal
delay.

## SM86 H28 constant-pressure L1TEX observer — 2026-10-15

H28 makes the existing static SASS constant-pressure classification dynamically
observable in the memory-issue readiness surface. It counts streams beyond the
released five-entry dispatch window (including the narrow U16/write-barrier
shape) while retaining the established 11+39 effective frontier and no new
timing or cache mutation. NCU is not a replay input.

One direct streaming root and four pressure roots passed representative replay.
Complete full-first replay passed 356 roots (286 `ok`, 62 controls, 8
replay/input failures), 12,392 launches in 40.672 s with twelve workers:
`../sm86_replay_results/full_corpus_h28_constant_pressure_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h28_constant_pressure_20261015_v1/`.

The corpus observed 6,336,451 constant-pressure instructions and 1,882,174
dynamic non-fall-through transfers. Readiness schema is 31. H28 is bit-for-bit
neutral versus H27 for L1/L2/DRAM, IPC/frontiers, and APE. Keep the pressure
count as a residual partition; a timing arm requires fresh clock64 load-return/
queue evidence rather than a static threshold assumption.

## SM86 H29 constant-pressure clock64 differential — negative — 2026-10-15

Fresh no-NCU RTX 3090 Ti clock64 runs of the retained constant-load probe used
dependent chains versus four independent reads at 128 iterations and 1/4/16
blocks. All functional checks passed; dependent runs were about 9.7k cycles and
independent pressure runs 11.6–12.9k. The differential mixes issue throughput,
overlap, and operand return, so it cannot identify a new L1TEX delay. No timing
arm was promoted; H28's pressure observer remains active. Report:
`docs/sm86_constant_dependency_probe_slice_20261015.md`.

## SM86 H30 six-slice L2 routing observer — 2026-10-15

H30 adds transactionally preserved per-slice L2 request-sector counts using the
active six-slice effective geometry/hash. It makes no proprietary GA102 hash
claim, adds no queue timing, and never uses NCU to select routing.

Five high-stride/L2-thrash/working-set/SGEMM roots passed representative replay.
Complete full-first replay passed 356 roots (286 `ok`, 62 controls, 8
replay/input failures), 12,392 launches in 40.089 s with twelve workers:
`../sm86_replay_results/full_corpus_h30_l2_slice_observer_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h30_l2_slice_observer_20261015_v1/`.

Full-corpus slice sectors are 1,718,579 / 1,653,401 / 1,717,493 /
1,652,955 / 1,717,655 / 1,653,918. H30 is bit-for-bit neutral versus H29 for
L1/L2/DRAM counters, IPC/frontiers, and APE. Next L2 work requires independent
slice-queue/service evidence; do not infer physical GA102 hash or timing from
this balanced observer alone.

## SM86 H31 L2 slice-pressure observer — 2026-10-15

H31 extends H30 with per-launch effective L2 max-slice, min-slice, spread, and
max/min imbalance observers. The rebuilt local RTX 3090 Ti clock64 L2 queue
probe passed unique/reuse smoke points across 1/2/4/8 warps and 1/8/32-load
bursts; pressure growth is stable as an observer but not a transferable fixed
latency.

Five high-stride/L2 roots passed representative full replay. Complete full-first
replay passed 356 roots (286 `ok`, 62 controls, 8 replay/input failures), 12,392
launches in 39.887 s with twelve workers:
`../sm86_replay_results/full_corpus_h31_l2_pressure_observer_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h31_l2_pressure_observer_20261015_v1/`.
Per-launch imbalance ratio is min 1.0, median 1.133, p90 3.0, max 38.0.
H31 is bit-for-bit neutral versus H30 for L1/L2/DRAM, IPC/frontiers, and APE.
No L2 queue timing arm is promoted.

## SM86 H32 L2 pressure strata — 2026-10-15

H32 adds active-slice count and maximum-slice demand fraction to H31's L2
pressure observer. Complete full-first replay passed 356 roots (286 `ok`, 62
controls, 8 replay/input failures), 12,392 launches in 40.152 s with twelve
workers:
`../sm86_replay_results/full_corpus_h32_l2_pressure_strata_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h32_l2_pressure_strata_20261015_v1/`.

The ratio distribution remains min 1.0, median 1.133, p90 3.0, max 38.0; the
max-slice fraction is min 16.67%, median 18.75%, p90 50%, max 100%. Active
slices are 6 for 9,186 launches, 2 for 1,658, 3 for 488, 4 for 487, 1 for
358, and 5 for 215. H32 is bit-for-bit neutral versus H31 for L1/L2/DRAM,
IPC/frontiers, and APE. Timing remains unchanged.

## SM86 H33 L2 route concentration observer — 2026-10-15

H33 adds normalized sum-of-squares concentration over effective six-slice
request demand. Complete full-first replay passed 356 roots (286 `ok`, 62
controls, 8 replay/input failures), 12,392 launches in 39.868 s with twelve
workers:
`../sm86_replay_results/full_corpus_h33_l2_concentration_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h33_l2_concentration_20261015_v1/`.

Concentration is min 166,667 ppm, median 168,405 ppm, p90 500,000 ppm, max
1,000,000 ppm; imbalance remains min 1.0, median 1.133, p90 3.0, max 38.0.
H33 is timing/traffic neutral versus H32; no queue timing or NCU-derived arm is
enabled.

## SM86 H34 L2 route strata — 2026-10-15

H34 classifies effective L2 routes as incomplete (<6 active slices), balanced
(all six and concentration below 250,000 ppm), or concentrated (all six and at
least 250,000 ppm). Complete full-first replay passed 356 roots (286 `ok`, 62
controls, 8 replay/input failures), 12,392 launches in 40.574 s with twelve
workers:
`../sm86_replay_results/full_corpus_h34_route_strata_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h34_route_strata_20261015_v1/`.

Strata counts are 3,206 incomplete, 9,135 balanced, and 51 concentrated. H34
is timing/traffic neutral versus H33 for L1/L2/DRAM, IPC/frontiers, and APE;
no queue timing or NCU-derived arm is enabled.

## SM86 H35 route-strata correlation — 2026-10-15

H35 joined H34 route strata to independent post-replay NCU comparison rows
without using NCU as a model input. Concentrated launches have higher cycle
residuals but only 4–8 joined rows per metric; incomplete launches show high
DRAM residuals on a small subset while L2-sector residuals are near zero. The
correlation is sparse/confounded and does not identify service timing.

Complete full-first replay passed 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 launches in 40.529 s with twelve workers:
`../sm86_replay_results/full_corpus_h35_route_correlation_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h35_route_correlation_20261015_v1/`.
H35 preserves the typed route-strata ledger and leaves L2 timing unchanged.

## SM86 H36 sectored L2 fill ownership — 2026-10-15

H36 makes the existing event-mode coalescing mechanism measurable without
changing it. Each read sector is counted as either a new fill owner or a
pending merge onto the same allocation/sector. Pending merges remain modeled
cache-state lookup misses but reuse the older fill's ready frontier; the observer is
transactional, cumulative, snapshot-safe, and independent of NCU targets.
Native rows are `l2_pending_merge_sectors`, `l2_new_fill_sectors`, and
`l2_pending_merge_fraction_pct`.

The five-root L2/high-stride representative passed 13 launches. Complete
full-first replay then passed 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 launches in 40.379 s with exactly twelve workers:
`../sm86_replay_results/full_corpus_h36_fill_ownership_observer_20261015_v1/full.jsonl.zst`
and adjacent compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h36_fill_ownership_observer_20261015_v1/`.

Full-corpus totals are 1,727,757 pending-merge sectors and 4,823,652 new-fill
sectors; pending merges occur in 1,822 launches. The nonempty-launch merge
fraction is 0% minimum, 0% median, 2.273% p90, and 97.196% maximum. A keyed
adversarial comparison against H35 found no old-field difference after removing
the expected nondeterministic `sidecar_join_wall_ns` and the three new observer
rows. H36 changes no L1/L2/DRAM traffic, IPC, or frontier; it is the retained
measurement surface for a future source- and pressure-qualified service arm.

## SM86 H37 source-qualified PRT/fill pressure — 2026-10-15

Huerta's Ampere queue/PRT semantics and modern GPGPU-Sim's finite MSHR/fill
queues motivate a pressure ledger, but the local RTX 3090 Ti clock64 probes do
not isolate a transferable queue latency. A one-cycle-per-merged-sector drain
ablation changed 1,670 frontiers, improved GPC mean APE by only 0.004 points,
and worsened SM-active APE; it was removed before promotion.

H37 therefore adds an observer-only, source-qualified pending-sector pressure
ledger to the shared L2 event path. Immediately before each request it records
pending fill-sector count by `L2Source`, then emits sample count, sum, and mean
pressure rows. No PRT capacity, hidden hash, timing delay, duplicate DRAM fetch,
or NCU fitting is introduced.

Five representative roots passed. Complete full-first replay passed 356 roots
(286 `ok`, 62 controls, 8 failures), 12,392 launches in 40.638 s with twelve
workers:
`../sm86_replay_results/full_corpus_h37_prt_pressure_observer_20261015_v1/full.jsonl.zst`
and compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h37_prt_pressure_observer_20261015_v1/`.

Totals are L1-fill 2,727,211 samples/3,253,767,441 sector-samples, bypass
31,433/1,933,553, untagged 32/4,666, and store 3,279,482/198,743,493. A keyed
comparison against H36 found zero old-field differences after removing new
pressure rows and the nondeterministic sidecar-join wall-time field. The next
timing arm requires a fresh locked probe that isolates queue service from
occupancy and dirty/coherence tails.

## SM86 H38 L1 MSHR-full pressure — 2026-10-15

H38 instruments the existing finite L1 MSHR admission path with checked
occupancy samples and sums whenever a miss finds the MSHR set full. It applies
no delay, changes no retry/cache behavior, and keeps the unified pressure gauge
outside global/local traffic partitions. The carded MSHR count is not claimed
as a proprietary GA102 PRT size.

Five high-MSHR/high-stride/L2 roots passed. Complete full-first replay passed
356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in 40.241 s
with twelve workers:
`../sm86_replay_results/full_corpus_h38_l1_mshr_pressure_20261015_v1/full.jsonl.zst`
and compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h38_l1_mshr_pressure_20261015_v1/`.

Full-corpus pressure is 953,785 samples in 1,120 launches and 30,978,868
occupancy entries (maximum sample count 124,766; maximum sum 3,992,512).
Streaming/DSMEM roots each record 10,924 samples at occupancy 32; vendor SGEMM
records 36; pointer-chase/working-set roots record none. A keyed H37 comparison
found zero old-field differences after removing new MSHR rows and
`sidecar_join_wall_ns`; all old L1/L2/DRAM/IPC/frontier values and APE remain
unchanged.

## SM86 H39 negative — L1 lookup retry correction — 2026-10-15

H39 tested whether MSHR-full miss attempts should be added to the L1
lookup-sector denominator. On the H38 complete replay's 107 exact independent
NCU joins, MSHR-stall correlation with APE is −0.015 for global-load sectors,
−0.041 for lookup hits, −0.018 for lookup misses, 0.046 for GPC cycles, and
−0.032 for SM-active cycles. High-stall quartiles do not have a consistent
lookup residual direction.

No lookup correction or timing change was implemented. H38's observer-only
MSHR pressure remains active; a retry model needs a dedicated locked
request-level probe rather than a corpus-wide count adjustment. Negative report:
`docs/sm86_l1_lookup_retry_negative_20261015.md`.

## SM86 H40 L1 global-load scope audit — 2026-10-15

H40 adds a post-replay-only diagnostic for gross (>4x) disagreement between
trace-side logical memory counts and independent NCU opcode metrics. It never
changes replay, prediction selection, APE scoring, or NCU inputs.

Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 40.199 s with twelve workers:
`../sm86_replay_results/full_corpus_h40_l1_scope_audit_20261015_v1/full.jsonl.zst`
and compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h40_l1_scope_audit_20261015_v2/`.

One exact join is flagged: `sm86_adapted_l2_074_sam_iou_backward_sm120` has
1,024 trace global-load instructions/4,352 sectors versus 16 NCU opcode
instructions/40 sectors (64x), while 32,736 warp instructions agree exactly.
No correction is promoted; old fields and APE are unchanged versus H38.

## SM86 H41 L2/DRAM scope audit — 2026-10-15

H41 extends the fixed-ratio post-replay scope diagnostic to simulator L2
source-sector and DRAM read-sector counters versus independent NCU operation
metrics. It never changes replay, prediction selection, APE, or NCU model inputs.

Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 39.934 s with twelve workers:
`../sm86_replay_results/full_corpus_h41_l2_dram_scope_audit_20261015_v1/full.jsonl.zst`
and compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h41_l2_dram_scope_audit_20261015_v1/`.

Exact joins contain 121 DRAM read-sector warnings, 3 L2 source-sector warnings,
and the H40 global-load warning. The SAM-IoU join has 2,464 modeled L2 source
sectors versus 168 NCU sectors and 1,076 modeled DRAM reads versus 564 NCU
sectors. No multiplier or timing correction is promoted; old fields and APE
remain unchanged.

## SM86 H42 scope-aware NCU residuals — 2026-10-15

H42 makes H40/H41 scope warnings operational in post-replay reporting.
Predictions covered by an explicit gross trace-vs-NCU scope warning remain in
compressed rows with `scope_mismatch` status and warning inventories, but are
excluded from raw/all-row comparison APE. Replay, timing, prediction selection,
and NCU model inputs remain unchanged.

H42 reduces the H41 full/fast artifacts and publishes
`../sm86_replay_results/metric_report_h42_scope_aware_20261015_v1/`. Full mode
has 125 scope warnings and 67 excluded predictions across 107 exact joins.
Secondary like-for-like mean APE is L1 global-load sectors 100.96%→0.21%, L2
source sectors 28.24%→6.76%, DRAM read sectors 40.91%→26.84%, while GPC cycles
remains 12.90%. Raw/all-row H41 residuals (100.96%/28.24%/40.91%) remain
authoritative; excluded rows are unresolved scope/model gaps, not completed
mechanisms. The 26.84% scope-compatible DRAM residual remains active; this is
not a model-closure claim.

## SM86 H43 DRAM residual ledger — 2026-10-15

H43 joins H42 scope-aware DRAM rows to H37 route/fill-pressure and H38 L1 MSHR
observers. There are 58 scope-compatible exact DRAM joins at 26.84% mean APE;
all are balanced-route rows, with no exact concentrated/incomplete route joins.

Pressure partitions are non-monotonic: low/high pending-merge quartiles are
28.69%/12.47% APE, low/high new-fill quartiles 40.92%/8.59%, and low/high MSHR
groups 28.59%/23.25%. Family, shape, and demand volume confound these shifts,
so no service law, partition/hash selector, bandwidth multiplier, or NCU fit is
promoted. A new locked probe must isolate demand while accounting for dirty
writeback/coherence tails.

## SM86 H44 DRAM dirty-tail residual ledger — 2026-10-15

H44 joins the 58 scope-compatible exact DRAM rows with dirty residency,
dirty-eviction, and typed tail flags. All rows carry both `dirty_writeback` and
`coherence` tails; none has dirty L2 eviction sectors, so terminal L2 dirty
residency is the varying demand.

Dirty-residency APE is non-monotonic: low/middle/high quartiles are 21.29%,
35.82%, and 12.10%. The relation is confounded by read demand, family, and
shape; no writeback bandwidth, queue capacity, or coherence law is identified.
Default writeback timing remains disabled; retain the optional scenario and
typed tails. Compact report:
`docs/sm86_dram_dirty_tail_ledger_slice_20261015.md`.

## SM86 H45 IPC/cycle residual ledger — 2026-10-15

H45 joins exact cycle predictions to causal issue/terminal frontiers, H37/H38
pressure, route strata, and typed tails. On 107 exact joins, GPC mean APE is
12.90% and SM-active mean APE 18.02%. Balanced/incomplete/concentrated route
classes are 11.67%/16.08%/28.03% GPC APE (n=88/15/4) and
15.07%/31.76%/31.32% SM-active APE. Frontier gaps and pressure are
non-monotonic; sparse classes cannot identify timing. No IPC/cycle correction is
promoted. Compact report:
`docs/sm86_ipc_residual_ledger_slice_20261015.md`.

## SM86 H46 DRAM demand interval audit — 2026-10-15

The trace-derived DRAM demand interval brackets 34/58 (58.6%) H42
scope-compatible exact NCU DRAM read rows. Median low/high bounds are 262/392
sectors and target median is 392; high residuals often lie above the interval,
but family, demand, writeback/coherence, and operation scope remain confounded.
No bound expansion, multiplier, or timing change is promoted. Compact report:
`docs/sm86_dram_demand_interval_slice_20261015.md`.

## SM86 H47 dirty-writeback scenario A/B — negative promotion — 2026-10-15

The existing `CardedWriteback` path was exercised at a predeclared 4,096-sector
capacity on five high-terminal-dirty real traces. GPC APE improved for qkv-rope
(29.19%→24.92%), RMSNorm (12.34%→11.51%), and Hyena FFT (8.38%→7.26%), was
unchanged for Flash attention (23.72%), and worsened for fused linear/GELU/GRN
(24.69%→26.16%). SM-active frontiers never changed.

No default writeback timing promotion is justified; optional scenario and typed
tail flags remain explicit. Report:
`docs/sm86_writeback_scenario_negative_20261015.md`.

## SM86 H48 LDG constant-form observer — 2026-10-16

H48 adds static modifier-form counters for `LDG.*.CONSTANT` global loads:
`LTC128B`, `U16`, and other/plain. Modern source and Nsight documentation
distinguish global LDG traffic from explicit constant-space LDC/CBANK paths, so
forms remain global traffic and are not rerouted to L0C/GCC. No lookup/timing
correction or NCU fit is introduced.

Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 40.899 s with twelve workers:
`../sm86_replay_results/full_corpus_h48_constant_form_observer_20261016_v1/full.jsonl.zst`
and compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h48_constant_form_observer_20261016_v1/`.

Totals are LTC128B 199,446 instructions/828,751 sectors, U16 5,515,663/
5,612,737, and other/plain 1,092,487/4,375,547. Traffic, timing, IPC,
frontiers, and APE remain unchanged; form rows are scope observers only.

## SM86 H49 constant-form microprobe — 2026-10-16

The notes project probe at
`GPU_simulator_snapshot/workloads/research_probes/constant_form_scope` builds
two identical-work SM86 kernels. Both real executions pass identical functional
checksums; cuobjdump confirms `LDG.E.CONSTANT` for 32-bit `__ldg` and
`LDG.E.U16.CONSTANT` for uint16 `__ldg`. Form identity is a valid observer
discriminator, not proof of an L0C/constant-space route. No lookup/timing
correction or NCU fit is promoted. Report:
`docs/sm86_constant_form_probe_slice_20261016.md`.

## SM86 H50 constant-form reuse/warp sweep — 2026-10-16

The notes probe swept 24 real RTX 3090 Ti points: plain versus uint16 `__ldg`,
1/4/8 warps, and reuse distances 1/4/32/256 lines. All points pass identical
functional checksums per pair; SASS remains `LDG.E.CONSTANT` versus
`LDG.E.U16.CONSTANT`. Clock64 grows with reuse/pressure but form differences
are small and non-monotonic (W1/reuse1 9,257 vs 9,258; W4/reuse32 23,089 vs
21,574; W8/reuse256 39,123 vs 39,108). No form-specific timing correction is
promoted. Report:
`docs/sm86_constant_form_sweep_slice_20261016.md`.

## SM86 H51 constant-form L1 lookup residual ledger — 2026-10-16

H51 joins H48 static form counters to exact independent NCU L1 lookup rows.
Other/plain forms show 5.85% hit and 31.53% miss mean APE (32/43 joins), U16
shows 51.36% hit and 34.90% miss (42/50), LTC128B has no positive-target hit
joins and 26.68% miss (3), and no-form rows show 6.89% hit/25.25% miss. Forms
remain global traffic observers; no correction or fitting is promoted. Report:
`docs/sm86_constant_form_residual_ledger_slice_20261016.md`.

## SM86 H52 integrated non-regression audit — 2026-10-16

H48 has 12,392 launch keys, exactly matching H38; the adversarial comparison
finds zero old-field differences after removing only new form rows and
`sidecar_join_wall_ns`. H42/H48 report inventories retain compressed comparison
JSONL, summaries, 125 scope warnings, and 67 explicit exclusions. Workspace
Cargo/Python/diff checks pass. Scope exclusions remain unresolved, not closure
claims. Their summary headings predate H63's raw-versus-secondary correction;
use the regenerated H63 report for that view. Report:
`docs/sm86_integrated_audit_slice_20261016.md`.

## SM86 H53 L2 line/sector residual ledger — 2026-10-16

On 104 H42 scope-compatible exact joins, L2 lookup-miss APE is exactly 0%.
Lookup-hit mean APE is 11.77% overall: balanced 12.50% (85 rows), incomplete
1.02% (15), concentrated 36.77% (4). Line/sector miss counts and pending/new
pressure are non-monotonic; no L2 multiplier or route timing arm is promoted.
Report:
`docs/sm86_l2_semantic_residual_ledger_slice_20261016.md`.

## SM86 H54 independent NCU constant-form probe — 2026-10-16

NCU independently profiled plain 32-bit and uint16 `__ldg` controls with
identical work. At 1 warp/64 iterations/reuse1 both report 64 global-load
instructions/requests, 256 sectors, and 252 hits/4 misses; at 8 warps/64
iterations/reuse256 both report 512 instructions/requests, 2,048 sectors, and
1,764 hits/284 misses. No form-specific lookup differential appears in these
controls; results were not replay inputs or fitted parameters. Report:
`docs/sm86_constant_form_ncu_probe_slice_20261016.md`.

## SM86 H55 consolidated accuracy dashboard — 2026-10-16

`docs/sm86_accuracy_dashboard_20261016.md` consolidates raw/all-row H41 APE and
secondary H42 scope-compatible APE, excluded counts, coverage, and unresolved
tails. Raw values remain authoritative; secondary values are not closure claims.
It records 356 roots, 12,392 launches, 107 identity joins, and current
L1/L2/DRAM/IPC residuals.

## SM86 H56 scope-warning threshold sensitivity — 2026-10-16

On H41 exact joins, L2 scope exclusions are 6/3/3 and DRAM exclusions 70/49/29
at 2×/4×/8× thresholds; DRAM secondary APE is 19.15%/26.84%/32.81%. The 4×
threshold remains a fixed gross-scope diagnostic, not an optimization knob;
raw/all-row APE remains authoritative. Report:
`docs/sm86_scope_threshold_sensitivity_slice_20261016.md`.

## SM86 H57 SOTA DRAM queue prior — 2026-10-16

Source-inspection notes from GPGPU-Sim/Accel-Sim describe per-partition
private/shared credits, finite L2↔DRAM queues, and round-robin arbitration;
Huerta adds finite memory-subcore/PRT/coalescer structures. No Accel-Sim
binary/runtime or imported constant is used. These generic priors do not identify
GA102 partition hashes/capacities or completion ownership. H43/H47 evidence is
confounded/mixed, so no partitioned timing arm or imported capacity is enabled.
Report:
`docs/sm86_dram_sota_queue_prior_slice_20261016.md`.

## SM86 H58 branch integration audit — 2026-10-16

Release `sim-trace`/`sim-run` builds, workspace all-features checks, Python
compilation, and diff checks pass. Five current real traces replay full-first
5/5 (99 launches) with compressed output and the twelve-worker runner. H48/H42
artifacts/reports and H58 smoke output are present; active maps have no stale
SM120 default/corpus paths. Report:
`docs/sm86_integration_audit_slice_20261016.md`.

## SM86 H59 GPC observation-envelope residual — 2026-10-16

The measured envelope (`1776 + 8·floor(log2(grid))`) is non-monotonic against
107 exact NCU joins: log₂ 0/2/3/8/10 groups show 12.11%/9.56%/17.51%/21.76%/
29.19% APE with sparse high-grid samples; active-SM partitions are also
non-monotonic. No grid/SM multiplier or NCU correction is promoted. Report:
`docs/sm86_gpc_envelope_residual_slice_20261016.md`.

## SM86 H60 unresolved mechanism gap tracker — 2026-10-16

The explicit tracker records raw/secondary L1/L2/DRAM/IPC residuals, typed
writeback/coherence tails, 125 scope warnings/67 exclusions, sparse route/grid
strata, and four concrete probe requirements. No gap is renamed, silently
excluded, or declared physically recovered. Report:
`docs/sm86_unresolved_gap_tracker_20261016.md`.

## SM86 H61 LTC128B probe boundary — 2026-10-16

Direct `ld.global.LTC128B.u32` PTX is rejected by CUDA 13.2 `ptxas` with
`Unknown modifier '.LTC128B'`. LTC128B remains an observer from H48 traces and
cuobjdump; no invalid inline opcode or binary patch is attempted. Report:
`docs/sm86_ltc128b_probe_blocker_slice_20261016.md`.

## SM86 H62 full-mode capability contract audit — 2026-10-16

The active full runner enables all implemented/evidence-backed SM86 observers
and models by default: integrated L1/L2, aggregate carded DRAM/read service,
source/route pressure, MSHR pressure, form observers, IPC/frontier projections,
and scope diagnostics. Partition candidates remain observer-only; writeback,
coherence, and visibility arms activate only when typed trace events make them
applicable and explicit scenario flags configure them. Otherwise tails remain
typed unavailable rather than silently invented.
Plan wording now matches this contract; runtime unchanged.

## SM86 H63 adversarial plan/source consistency audit — 2026-10-16

H63 corrected two semantic inconsistencies. Scope-compatible H42 rows are now
unambiguously a secondary like-for-like view: scope-mismatch rows stay in
compressed output but are excluded from raw/all-row comparison APE, and raw H41
remains authoritative. The plan now states the landed default accurately:
aggregate `DramBw::Carded` read service, H30–H37 observer-only partition/route
pressure, and the H19 store/fence feedback candidate are active; finite
partition/bank timing, dirty writeback, coherence/visibility, and payload
forwarding remain future or explicit typed scenarios. H57–H62 entries were
reordered chronologically. Modeled cache-state/trace line-domain observers are
qualified rather than called physical; GPGPU-Sim/Accel-Sim is source inspection
only; and the 825 GB/s value is an unlocked absolute-rate hypothesis mapped
onto the locked simulation clock.

The corpus runner exposes `--no-memory-admission-feedback`, defaults the H19
candidate only for launch scope, records stream-L2 suppression, and emits the
configured DRAM service rate in each launch counter report.

The corrected reporter ran over retained real H41 artifacts (no replay or NCU
input) and emitted
`../sm86_replay_results/metric_report_h63_scope_label_audit_20261016_v1/`
with 125 full warnings/67 exclusions and 0/0 fast warnings/exclusions plus
machine-readable secondary/raw scope fields. A real stream-L2 full replay
completed `ok` with candidate suppression at
`../sm86_replay_results/h63_stream_l2_scope_20261016_v2/full.jsonl.zst`.
Release/Cargo, Python, and diff checks pass. Residuals and open tails remain
in the H60 gap tracker; this slice makes no closure claim. Report:
`docs/sm86_plan_consistency_audit_slice_20261016.md`.

External consultation bundle (plain text, not a workflow gate):
`docs/sm86_external_consult_prompt_20261016.txt`.

## SM86 H64 L2 queue-frontier negative — 2026-10-16

The notes `l2_queue_frontier` probe ran on the identified RTX 3090 Ti with 288
correct matrix points and twenty-repeat clock64 samples. Eight-warp/64-load
stride medians are 20,765/28,137/32,334 cycles for stride 1/8/32, but selected
NCU L2-active/XBAR changes are only ~4% and selected request-stall counters are
zero. This mixes issue/occupancy/overlap/return effects, so no queue capacity,
hash, stride delay, or merged-sector timing law is active. H30–H37 remain
observer-only; a one-factor DRAM demand/completion probe separating dirty and
coherence tails is next. Report:
`docs/sm86_l2_queue_frontier_negative_20261016.md`.

## SM86 H65 P1 L1 wavefront lookup negative — 2026-10-16

The rebuilt L1TEX same-binary probe reports zero cold-load lookup hits and
separate output-wavefront counts. A full-corpus A/B using
`hits + pending_lookup_hits` reduced zero-target hit false positives 22/29→1/29
but worsened hit APE 30.71%→33.43% and miss APE 58.44%→110.05%. The formula
was rejected and the H16/H63 projection restored. Future changes must be
packet/client/form-qualified and keep sector/wavefront endpoints separate.
Report: `docs/sm86_l1_wavefront_lookup_negative_20261016.md`.

## SM86 H65 supported LTC128B probe — 2026-10-16

CUDA 13.2 `.L2::128B` PTX compiles to `LDG.E.LTC128B.STRONG.SM` on SM86. A
matched 84-block control measured plain/LTC L2 TEX read misses of 43,008/
172,032, promoted sectors 0/129,024, and DRAM reads 43,176/172,672. Retain
this as independent promotion evidence; do not enable simulator timing or
attribute the corpus scope outlier without NVBit/holdout and source-scope
validation. Report:
`docs/sm86_ltc128b_ptx_probe_slice_20261016.md`.

## SM86 H66 GCC/L1.5 demand observer — 2026-10-16

A same-host NCU probe confirmed GCC/L1.5 sectors and DRAM reads exist even in a
zero-global-data control (952 GCC sectors/125 misses/3,100 DRAM reads) and a
constant arm (1,960/269/276). Full replay adds observer-only GCC instruction,
constant, and multi-client demand candidates; no L2 state, DRAM read counter,
service, or timing frontier changes. Full-first replay passed 356 roots and
12,392 launches with compressed artifacts. Report:
`docs/sm86_gcc_demand_observer_slice_20261016.md`.

## SM86 H67 L2 write-lookup negative — 2026-10-16

Cold-store NCU evidence alone did not transfer to the scored source-unit
endpoint. The full-corpus present-sector A/B worsened secondary lookup-hit APE
11.77%→33.67% and lookup-miss 0%→53.16% while source-sector APE stayed 6.76%;
the source was reverted. Require operation-split NCU/TEX and NVBit crosswalk
evidence before any write lookup/coalescing correction. Report:
`docs/sm86_l2_write_lookup_negative_20261016.md`.

## SM86 H68 no-fence store feedback negative — 2026-10-16

The store-ack matrix shows large no-fence tails, but broadening H19 candidate
qualification to every STORE/RED kernel changed no frontiers or admission
exposure on five real roots. The source was restored; H19 is not an implicit
EXIT-drain model. A separately gated EXIT arm requires matched fence/no-fence
evidence. Report:
`docs/sm86_store_feedback_no_fence_negative_20261016.md`.

## SM86 H69 DRAM demand/completion negative — 2026-10-16

Audited same-binary RTX 3090 Ti memory-retirement/store-ack/launch-cycle and
DRAM-stream evidence cannot isolate read service, dirty writeback, coherence,
EXIT drain, or profiler envelope; no completion token/global order exists and
lower-bound behavior is mixed. No timing arm is active. Aggregate carded DRAM,
typed tails, H66 GCC observers, and raw H41 authority remain unchanged. Report:
`docs/sm86_dram_demand_completion_negative_20261016.md`.

## SM86 H70 per-request L2 frontier observer — 2026-10-16

H70 adds timing-neutral per-request L2 issue-to-fill/response/service-delay
sums, maxima, and samples from existing event frontiers. Full-first replay
passed 356 roots and 12,392 launches with compressed artifacts; no L2/DRAM
state or timing changed. Report:
`docs/sm86_l2_frontier_observer_slice_20261016.md`.

## SM86 H71 terminal-EXIT store-drain — 2026-10-16

Independent locked RTX 3090 Ti evidence supports an implicit terminal store
drain domain: matched no-fence stores measured `sm__cycles_active.max=10345`,
`smsp__warps_issue_stalled_drain.sum=253415`, `membar=0`; explicit
`MEMBAR.GL` measured 11730 cycles, drain 1915, and MEMBAR 488950. The simulator
now has a bounded store-only `exit_waits_for_service_completion` candidate arm
that gates only a terminal dynamic `EXIT` on typed per-warp write acks. It also
clamps feedback pricing to the processing event and rearms stale actor wakes at
the shared queue clock, preventing real-trace time travel.

The arm was evaluated full-first on all 356 roots with 12 workers and compressed
artifacts. Enabled arm v2 restored 286/62/8 coverage but regressed secondary
SM-active/SMSP-active/IPC APE; default card arm is therefore explicitly false.
Default v3 replay restored H70 coverage and metric values bit-for-bit. Keep the
implementation as disabled experimental infrastructure only; do not claim an
EXIT timing law or accuracy gain. Future promotion requires same-binary NVBit
EXIT/store-ack crosswalk, STORE-vs-RED separation, and held-out endpoint-matched
probes. Artifacts: `../sm86_replay_results/full_corpus_h71_exit_drain_20261016_v2/`
and `_v3/`; report `docs/sm86_exit_drain_slice_20261016.md`.

## SM86 H72 LDGSTS/LTC128B scope crosswalk — 2026-10-16

H72 adds observer-only static-SASS scope partitions in `TraceMetrics.memory_scope`:
plain LDG, CONSTANT, LTC128B, STRONG.GPU, and LDGSTS async-copy forms. SM86
native metrics expose plain-LDG and LDGSTS opcode candidates; the reporter
keeps opcode mismatch separate from broad L1TEX traffic and isolates a large
LTC128B scope mismatch only in the secondary post-replay view.

Independent controls show `cp.async`/LDGSTS has 32 opcode records and 128 L1
sectors while NCU global-LD opcode count is zero, proving LDGSTS remains in the
L1TEX global-op sector surface. The SAM LTC128B.CONSTANT outlier has 1,024
records versus NCU global-LD requests 16 and is classified as a scope mismatch;
vendor controls show no universal subtraction rule. No cache/DRAM/timing
relocation or LTC128B promotion is active.

Full-first H72 replay: 356 roots, 286/62/8 status, 12,392 full launches, 12
workers, compressed artifact
`../sm86_replay_results/full_corpus_h72_ldgsts_scope_20261016_v1/`; report
`../sm86_replay_results/metric_report_h72_ldgsts_scope_20261016_v2/`. LDGSTS
opcode APE is exact (0%); broad hierarchy residuals are unchanged. Future
promotion needs same-binary NVBit/SASS endpoint crosswalk.

## SM86 H73 GCC/TEX source-client crosswalk — 2026-10-16

H73 reused the same-binary CUDA13.2 `gcc_demand_h65` probe/NCU exports: GCC/L1.5
traffic is real, but the existing trace-derived GCC instruction/constant
candidates are not endpoint-equivalent. A new post-replay warning seam reports
286 instruction and 200 constant gross crosswalk mismatches on the full H72
artifact; no replay/DRAM/timing state changes. NVBit capture attempts with both
CUDA13.4 and CUDA13.2/g++-14 binaries segfaulted at tracer context init, leaving
no dynamic source-client protobuf. Keep aggregate carded DRAM/H66 observers;
repair tracer and obtain dynamic GCC identity before any multi-client timing.
Report `docs/sm86_gcc_tex_crosswalk_slice_20261016.md`.

## SM86 H74 L2 operation-split source observers — 2026-10-16

H74 adds default full-replay NCU-shaped operation-split L2 TEX observers for
read/write/atom-dot-alu/reduction sectors and lookup hit/miss, derived directly
from `L2SourceSplit` with checked subtraction. Full-first H74 replay remains
286/62/8 over 12,392 launches, 12 workers, compressed artifact
`../sm86_replay_results/full_corpus_h74_l2_operation_split_20261016_v1/`.

Read/atom/RED endpoints are compatible (read sectors 1.90% APE, read hits
5.44%, misses exact; atom/RED exact); write sectors/hits remain ~39.9% APE,
write misses exact/zero-target. No lookup policy, L2, DRAM, or timing changed.
Keep P3 write-coalescing timing disabled until same-binary XBAR/L1TEX evidence.
Report `docs/sm86_l2_operation_split_slice_20261016.md`.

H73 correction/addendum: GCC sector candidates are now crosswalked only against
`lts__t_sectors_srcunit_gcc.sum`, not request-domain constant counters. Corrected
full H72 report shows 286 instruction and 264 constant-sector gross mismatches.
The rebase tracer successfully captured gcc_demand_h65, but dynamic records lack
constant-bank/address fields; static PC/SASS joins are not equivalent to NCU GCC
requests. No timing promotion is justified.

## SM86 H75 L1TEX/XBAR write-coalescing — 2026-10-16

Locked RTX3090Ti probe in notes shows distinct-line stores produce 86,016 L2
op_write/XBAR sectors and 86,016 lookup hits; same-sector cross-warp stores
collapse to 5,699 sectors/XBAR writes with 6,213 lookup hits; spaced same-sector
stores produce 6,651 sectors with 5,888 hits. L1 store sectors remain 21,504.
Lookup hits exceeding op_write and non-monotonic spacing falsify direct tag
residency/fixed-window formulas. Keep P3 timing disabled; require dynamic
same-binary write identity and cold/warm crosswalk before promotion.

## SM86 H76 constant-client dynamic interval observer — 2026-10-16

H76 adds trace-native `constant_client_ldc_instructions` and
`constant_client_uldc_instructions` observers, preserving uncertainty between
static/dynamic LDC populations, missing CBANK addresses, and GCC request/sector
endpoints. Rebase-tracer captures show 16,128 arithmetic and 360,192 constant
mode CBANK-tagged records, while NCU GCC requests are 1,008/791 and 562,212/742;
no endpoint collapse is valid. Full replay remains 286/62/8 over 12,392
launches, 12 workers, compressed artifact
`../sm86_replay_results/full_corpus_h76_constant_client_interval_20261016_v1/`.
No L2/DRAM/timing mutation; repaired dynamic CBANK capture is required next.

## SM86 H77 dynamic CBANK tracer repair — 2026-10-16

Rebase-tracer prototype fix: immediate LDC/ULDC CBANK forms were previously
skipped; additive protobuf tags 15–17 now carry active-lane constant offsets,
bank, and representative offset. Enhanced-execution-info sidecar adapter now
derives checked control words and narrowly falls back to the enhanced metadata
filename when the native sidecar is absent. Real mode-1 and mode-0
capture/replay succeed: mode 1 has 344,064 LDC, 2,688 ULDC, 11,526,144
addresses, and 147 unique sectors; mode 0 is a valid no-LDC control with 2,688
ULDC records. Full-first corpus replay after the loader repair remains 286/62/8
over 12,392 launches with twelve workers and compressed artifacts at
`../sm86_replay_results/full_corpus_h77_adapter_20261016_v2/`; report:
`../sm86_replay_results/metric_report_h77_adapter_20261016_v2/`. This remains a
notes/rebase prototype because the active branch lacks tracer source; no timing
or GCC request collapse is allowed. Checked-in transport integration remains
the next seam.

## SM86 H78 metric-unit crosswalk — 2026-10-16

The reporter's native-unit compatibility defect is fixed with a name-qualified
post-replay crosswalk: percentage aliases, empty-unit request/launch rows,
branch counts, launch geometry/occupancy, register/resource units, and explicit
Kbyte/Mbyte scaling for DRAM/shared-memory metrics. Re-reporting the H77
full-first artifact raises compared metrics 40→63; only GPU duration remains
unconverted because no clock crosswalk exists. No replay prediction, cache/L2,
DRAM service, or timing frontier changes. Report:
`docs/sm86_metric_unit_crosswalk_slice_20261016.md`.

## SM86 H79 packet-aware L1 lookup observer — 2026-10-16

H79 exports ordinary-global same-request merge provenance from the existing L1
audit and a bounded candidate `baseline_hits - same_request_merges`, plus
interval edges, with checked denominator conservation. Full-first replay v2
remains 286/62/8 over 12,392 launches and twelve workers. On 106
scope-compatible joins, hit APE falls 30.71%→17.50%, miss APE 32.39%→19.53%,
and zero-target false-positive hits 22/29→8/29. The raw/all-row LTC128B scope
outlier makes the miss candidate regress, so no default prediction replacement
is justified; retain it as an observer pending form-qualified endpoint
crosswalk. Report:
`docs/sm86_l1_packet_aware_lookup_slice_20261016.md`.

## SM86 H80 L2 write-packet coalescing — 2026-10-16

H80 promotes an observer-only per-SM line/sector write-packet candidate within
an 8-cycle issue window for the operation-sector endpoint only. The baseline
request-sector value is retained; lookup-hit/miss, L2 state, DRAM, writeback,
and timing remain unchanged. Full-first v2 replay is 286/62/8 over 12,392
launches with twelve workers. NCU comparison improves write-sector mean APE
39.86%→9.37%, P90 100%→0%, exact nonzero joins 77→100. No physical XBAR claim
or lookup-timing promotion is made. Report:
`docs/sm86_l2_write_packet_observer_slice_20261016.md`.

## SM86 H81 L2 write lookup packet composition — 2026-10-16

H81 composes H80 packet sectors with write lookup hits/misses using checked
packet-sector conservation, while retaining request-domain baselines. Full-first
replay remains 286/62/8 over 12,392 launches and twelve workers. Operation
write-sector APE is 9.37%, write lookup-hit APE 9.36%, and write lookup misses
are exact zero-target rows; aggregate L2/DRAM/timing/IPC remain unchanged. No
physical XBAR/tag claim is made. Report:
`docs/sm86_l2_write_lookup_packet_slice_20261016.md`.

## SM86 H82 DRAM route-concentration diagnostic — 2026-10-16

H82 adds report-only normalized candidate partition/slice concentration and a
GPC-frontier joint observer. Full-first replay remains 286/62/8 over 12,392
launches with twelve workers; median concentration is 175,631 ppm, P90
500,000, and 23 launches exceed 500,000. GPC/SM APE is unchanged. Adversarial
review finds no default-safe partition/queue timing law: concentrated exact
joins are sparse, DRAM-compatible joins are balanced, pressure is non-monotonic,
and H64 request-stall change is ~4% and confounded. Preserve aggregate carded
DRAM and defer timing until locked one-factor evidence. Report:
`docs/sm86_dram_route_concentration_diagnostic_slice_20261016.md`.

## SM86 H83 IPC readiness concentration observer — 2026-10-16

H83 emits dominant exclusive readiness-cause fraction and total limiting-cycle
observers; no scheduler/frontier/cache/DRAM behavior changes. Full-first replay
remains 286/62/8 over 12,392 launches and twelve workers. Across 286 full
launches, dominant-cause concentration has median 425,114 ppm, P90 613,776,
and 103 above 500,000; GPC/SM APE is unchanged. Treat as causal decomposition
only; H71/H64 negative timing evidence remains. Report:
`docs/sm86_ipc_readiness_concentration_slice_20261016.md`.

## SM86 H84 integrated accuracy dashboard — 2026-10-16

Latest H83 full-first artifact preserves 286/62/8 coverage over 12,392 full
launches and twelve workers. Mechanism-local mean APE is L1 sectors 0.21%, L2
TEX sectors 6.76%, H80/H81 write sectors/hits 9.37%/9.36%, DRAM reads 26.84%,
GPC 12.90%, and SM-active 18.02%; H82/H83 are observer-only. Remaining gaps
are L1 lookup form scope, multi-client DRAM ownership, and IPC retirement/
readiness timing. Report:
`docs/sm86_integrated_accuracy_dashboard_20261016.md`.

## SM86 H85 independent GCC/constant demand matrix — 2026-10-16

Locked 3090 Ti CUDA13.2 matrix: one-LDC repetitions raise GCC requests
7,143→590,880 but L2 GCC sectors only 2,128→2,240 and DRAM reads 296→316;
two-LDC reaches 2,127,353 requests, 3,248 sectors, 452 DRAM reads. Request
count is not a DRAM proxy; retain GCC/constant observers and aggregate carded
DRAM, with no P4 timing promotion. Probe matrix/environment live under the
notes workspace. Report:
`docs/sm86_gcc_demand_matrix_slice_20261016.md`.

## SM86 H86 integrated adversarial review — 2026-10-16

Workspace Cargo/Python/diff checks and zstd artifact integrity pass. Latest
H83 replay retains 286/62/8 coverage, 12,392 launches, 107 identity joins, 179
fallback joins, twelve workers, and all promoted write conservation rows.
NCU remains post-replay only; H80/H81 are operation-write endpoint changes,
H82/H83 observers, H85 a negative GCC/DRAM fitting result. Remaining evidence
gates: L1 lookup scope, GCC/multi-client DRAM ownership, IPC retirement timing.
Report:
`docs/sm86_integrated_adversarial_review_slice_20261016.md`.

## SM86 H87 L1 lookup form-scope review — 2026-10-16

H79 candidate stratification: 88 constant-form rows improve hit APE 30.86%→
16.95% and miss 31.34%→17.53%; four LTC128B rows worsen miss APE 725.01%→
1,683.10%; mixed LDGSTS forms differ. Keep baseline selected; promote only with
per-request client/form identity. Report:
`docs/sm86_l1_lookup_form_scope_review_slice_20261016.md`.

## SM86 H88 constant-client reuse observers — 2026-10-16

H88 adds dynamic constant address-record coverage and references/request reuse
ratios as observer-only fields. Full-first replay remains 286/62/8 over 12,392
launches and twelve workers; GPC 12.9008% and DRAM-read 26.8371% APE are
unchanged. H85 still blocks request-to-DRAM fitting; no P4 timing promotion.
Report: `docs/sm86_constant_reuse_observer_slice_20261016.md`.

## SM86 H89 repaired dynamic-CBANK H85 crosswalk — 2026-10-16

External CUDA13.2 tracer captures H85 mode 0/1 with fields 15–17 and both
replay through the active adapter. Mode 1: 344,064 LDC, 5,376 ULDC, 712,320
references, 22,794,240 addresses, 147 unique sectors; NCU: 590,041 GCC
constant requests, 2,240 GCC L2 sectors. Transport is validated but endpoint
domains differ; retain observer-only and keep P4 timing blocked. Report:
`docs/sm86_repaired_cbank_crosswalk_slice_20261016.md`.

## SM86 H90 complete accuracy comparison — 2026-10-16

H90 publishes the complete H88 simulator-vs-NCU comparison: 63 finite compared
metric names, 107 exact identity joins, 179 fallback joins excluded from APE,
12,392 replay launches, all family tables with mean/median/P90 APE, scope and
exact counts, and unavailable inventory. Report:
`docs/sm86_accuracy_comparison_report_20261016.md`.

## SM86 H91 expanded ordered-fallback accuracy report — 2026-10-16

User-authorized same-workload ordered-fallback pairs are now included in a
separate expanded report. The 179 pairs differ in trace/NCU name spelling but
are accepted as corresponding workload launches; the strict exact-only report
is preserved. Expanded scoring covers 286 successful root pairs (107 exact +
179 fallback). Report:
`docs/sm86_accuracy_comparison_ordered_fallback_20261016.md`.

## SM86 H92 form-qualified L1 lookup selection — 2026-10-16

H92 promotes a bounded trace-only L1 lookup endpoint mixture: packet-aware
subtraction for constant-only launches, an all-miss endpoint for LDGSTS-only
line/sector shapes other than the two-sectors-per-line async-copy control,
and a cold plain-LDG packet arm; mixed clients and LTC128B retain baseline
selection. L1/L2 state, MSHR/DRAM timing, cycles, and IPC are untouched.

The four-root probe and complete full-first replay succeeded with 356 roots,
286/62/8 status, 12,392 full launches, twelve workers, and compressed
full/fast output at `../sm86_replay_results/full_corpus_h92_form_lookup_20261016_115517/`.
The report is `../sm86_replay_results/metric_report_h92_form_lookup_20261016_v1/`.
Only L1 global lookup hit/miss and derived hit-rate predictions changed; all
other scored metrics are bit-equivalent to H88. Scope-compatible APE improved
30.7137%→18.0702% for hits and 32.3914%→18.8472% for misses. Raw/all-row
hit improved to 18.0702%; raw miss improved 58.4438%→45.0262% but remains
above target because of the known LTC128B/scope-mismatch row. Per-request
client/form crosswalk remains the next evidence gate.

## SM86 H93 aggregate L2 TEX packet conservation — 2026-10-16

H93 reuses H80/H81's promoted per-SM write-packet candidate in the aggregate
SM86 TEX sectors/hits rows, retaining uncoalesced request/hit baselines as
observers and deriving misses by checked subtraction. L2/DRAM state, write
acks, cycles, and IPC are untouched.

Complete full-first replay passed 356 roots (286/62/8), 12,392 launches,
twelve workers, and compressed full/fast artifacts at
`../sm86_replay_results/full_corpus_h93_l2_packet_conservation_20261016_123337/`;
report: `../sm86_replay_results/metric_report_h93_l2_packet_conservation_20261016_v1/`.
Strict scope-compatible APE improved L2 TEX sectors 6.7551%→2.9180% and
lookup hits 11.7747%→3.8618%; lookup misses stayed exactly 0%. No other
scored metric changed. Next work is a real per-request L1 U16/form crosswalk.

## SM86 H94 per-client L1 lookup provenance — 2026-10-16

H94 adds same-request merge audit counters partitioned by ordinary global,
constant, and async-copy clients. The selected L1 endpoint uses the constant
client's merge count for mixed constant/plain launches; LTC128B and mixed
async forms remain baseline. No cache state, timing, DRAM, or IPC changes.

Four real roots and the complete full-first replay passed (356 roots, 286/62/8,
12,392 launches, twelve workers, compressed artifacts at
`../sm86_replay_results/full_corpus_h94_l1_client_lookup_20261016_125135/`).
Report: `../sm86_replay_results/metric_report_h94_l1_client_lookup_20261016_v1/`.
Compared with H93, only one mixed-form launch changed. Scope-compatible L1
lookup hit/miss APE improved 18.0702%/18.8472%→17.7036%/18.7117%; no other
metric regressed. A direct U16 half-line NCU probe measured 0 hits/8 misses for
both plain and U16 constant forms, rejecting a universal U16 multiplier and
leaving per-sector-validity state as the next evidence-gated seam.

## SM86 H99/H102 complete-corpus endpoint updates — 2026-10-16

All acceptance numbers below are from full-first complete-corpus replay only:
356 roots, 286 `ok`, 62 controls, 8 replay errors, 12 workers, and 12,392
full launches. H99 promotes an evidence-backed all-miss L1 observer for
`LTC128B`; its report reduces global L1 lookup-miss APE 18.7117%→18.1662%
without changing other scored metrics. H102 adds the local frame/coalescing
endpoint: dynamic-index local accesses use `raw + (frame256 - width_stride)`;
scalar volatile/register-spill records use the validated payload candidate.
The H102 report reduces local load sectors to 1.7972%, local store sectors to
0.3353%, and local-load lookup hits to 1.7972%; every non-local metric is
unchanged. See `docs/sm86_ltc128b_l1_slice_20261016.md` and
`docs/sm86_local_frame_coalesced_slice_20261016.md`.

H95d, H97, and H98 remain rejected full-corpus experiments; see
`docs/sm86_rejected_l1_local_experiments_20261016.md`. Global L1 lookup remains
17.7036% hit / 18.1662% miss and elapsed IPC remains 12.3073% in H102; these
residuals are unresolved and are not claimed below 10%.

H103 tested a sector-fill-owner subtraction for older pending L1 merges on the
full 286-root replay. It regressed global L1 hit/miss APE to 33.74%/62.44% and
was rolled back; the owner-kind counters remain diagnostics only.

The final-tree H105 release replay is the authoritative artifact after the H103
rollback: `full_corpus_h105_final_release_20261016_145305/` with report
`metric_report_h105_final_release_20261016_v1/`. It reproduces H102 local
1.7972%/0.3353%/1.7972% and H99/H93 non-local metrics over 286 successful
roots; H103 is not part of the final tree.

## H106-H108 global L1 progress — 2026-10-16

H106 resident-sector lower clamp improved global L1 hit/miss APE to
15.516%/13.526%; H107 cold CONSTANT+LDGSTS and H108 exact U16 geometry rules
then improved them to **10.126%/10.231%** on the complete full-first replay
(`full_corpus_h108_l1_u16_geometry_20261016_152102/`, report
`metric_report_h108_l1_u16_geometry_20261016_v1`). No other scored metric
changed. The remaining gap is concentrated in a few mixed/high-reuse U16
shapes; workload-specific fitting is prohibited. IPC remains 12.3073%.

H111 tested a first GCC constant-sector addition in the DRAM endpoint on the
complete replay; it regressed DRAM read APE 26.84%→203.56% and was rolled back.
The H111 GCC addition is not part of the final tree. H109/H110 subline masks
remain observer diagnostics only.

## H113-H117 continuation — 2026-10-16

The SM86 branch now carries three separate front-end ledgers: executed 32B
instruction sectors, dynamic unique 128B instruction-line sectors, and the
static SASS 128B instruction-line envelope. The SM86 DRAM-read endpoint uses
the static aligned-line term. H114's complete 356-root replay reduced
scope-compatible DRAM-read APE to 19.47% (58 identity joins), while all
cache/cycle rows stayed unchanged.

H115 adds a trace-only mixed CONSTANT/plain-LDG lookup ownership rule. For the
one-sector CONSTANT shape, ordinary-client same-request and older-sector
pending merges are retained as ordinary hits, clamped to the global lookup
denominator. Full replay is 356/286/62/8 with 12,392 launches; global L1
lookup APE is 9.155% hit / 9.828% miss. H93 L2 (2.918%/3.862%/0%) and H102
local metrics remain bit-equivalent.

H116 adds `warp_uniform_address` to validated memory records and observer rows
for uniform/divergent `LDG.*.CONSTANT` records. It is not used for traffic,
cache, timing, or metric selection; full replay confirms 356/286/62/8 and the
H115 scores.

H117 is a negative full-corpus service-rate ablation: the architecture-card
1008 GB/s peak worsens GPC, SM-active, and both IPC rows versus the retained
independent 825 GB/s warm-rate hypothesis. Do not promote peak service or any
clock/NCU-derived correction. Remaining GPC/IPC residuals are structurally
mixed with the unidentifiable capture-clock boundary; retain explicit residual
reporting rather than fitting a per-workload multiplier.

## H121-H122 split store-drain timing — 2026-10-17

An opt-in `exit_waits_for_admission_drain` candidate was implemented as a
separate SMSP/L1TEX horizon: typed write acknowledgements are scheduled for
system draining, but terminal EXIT waits only for admission plus packet drain,
not L2 visibility. H121 full replay preserved 356/286/62/8 and 12,392 launches
but improved elapsed IPC/GPC only to 12.2045%/12.8746% while worsening
SM-active/SMSP to 18.2178%/17.7526%. H122 gated the horizon on a full five-entry
admission queue; elapsed IPC/GPC became 12.2864%/12.8892%, while SM-active/SMSP
remained worse at 18.1135%/17.6298%. Both timing arms are rejected. The source
seam and candidate JSON remain opt-in; default H116 behavior is unchanged.
