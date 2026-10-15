# SM120 continuation memory

**Updated:** 2026-10-14

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

## 2026-10-15 — Q57/Q58 side-metric continuation

Q57 diagnostic commit `efea4707` records L2 read pending-merge hits separately
from other read lookup hits. Real launch-cold and stream-L2 probes did not
support excluding these hits or persisting L2 as a better NCU read endpoint;
no cache/timing behavior changed and no merge/RMW factor was fitted.

Q58 commit `7471be0b` fixes `smsp__inst_executed_op_branch.sum` population by
including the CBU control-flow population (CALL/RET/EXIT/BSYNC/WARPSYNC/YIELD/
BREAK/CONT) while excluding BMOV/BSSY. Selected real evidence improved the
metric; the complete 339-root/11,558-launch full-integrated-L1+L2-first,
fast-flat-second replay with 12 workers and compressed JSONL achieved 0.4892%
mean APE (0% median/P90). This is a scoped instruction-cardinality projection,
not a physical scheduler claim. Active IPC, L1/L2 hits and DRAM/service remain
explicit residuals; NCU values remain post-replay comparison only.

## 2026-10-15 — Q59 L1 hit-scope closure

Round 21 showed that L1 global/unified hit residuals cannot be corrected by
aggregate merge arithmetic. Pending, same-request, and line-fill alternatives
worsen real Cutlass/MXFP8 rows; per-request sector masks and client scope are
not retained. Keep Q45/Q55 provenance only, retain the effective-L1 default,
and require a new source-visible scope before any further full replay.

## 2026-10-15 — Q60 IPC residual closure

Round 20 audited the available SM/SMSP activity populations and aggregation
choices. Q37's weighted per-SM active projection remains 23.8255% mean APE and
elapsed IPC 20.3993%; alternative SMSP/per-SM reductions are worse. Occupancy
variation has no retained source-defined correction, so no hidden tail or
NCU-fitted penalty is permitted. Q58 branch cardinality remains the accepted
IPC-side improvement.

## 2026-10-15 — Q63 shared bank-conflict correction

Commit `9cb48b11` retains ideal per-lane-width shared wavefronts separately
from address-derived wavefronts and emits bank-conflict excess as
`max(actual - ideal, 0)`. Full 339-root replay (integrated L1+L2 first, flat
fast second, 12 workers, compressed JSONL) reduced shared-load conflict mean
APE from 26.13% to 21.8535%; store conflicts are 0.32385%. Timing is unchanged;
remaining load residual is cross-client/LDSM scope and no topology factor is
fitted.

## 2026-10-15 — Q64/Q65 matrix scope and final combined replay

Commit `3641433e` excludes LDSM/STSM from operation-qualified bank-conflict
counters while retaining aggregate wavefront/timing work. Q64/Q65 full-corpus
replay gives 0.90345% shared-load conflict APE over 675 positive rows and
16.604% wavefront APE. Q65 also includes defensive L2 pending-mask commit
`eb227837`; read-only L2 miss/hit remains 0.3474%/25.33% with no output delta.
Remaining wavefront and TMA/L2 hit gaps require new scope evidence, not fitting.
