# SM86 blocker research current state

Updated 2026-09-29. This is a research-state map, not a completion or
verification declaration. No correctness, test, hash, or verification workflow
is part of the retained campaigns.

## Performance objective

Fresh current-driver full campaigns completed 15/15 in both cache phases:

| lane | NCU duration median | NCU GPC median | simulator event | simulator envelope |
|---|---:|---:|---:|---:|
| cold | 10,048 ns | 13,506 cycles | 11,780 | 13,596 |
| warm | 8,544 ns | 11,516 cycles | 11,780 | 13,596 |

Artifacts: `results/alpha/full_3090ti_20260929_refined_cold_live/`,
`...refined_warm_live/`, and
`results/alpha/refined_live_campaign_comparison_20260929/`.

The promoted frontier/archetype predictor scores these fresh rows at 2.655%
GPC / 2.626% duration MAPE cold and 6.314% / 6.422% warm. The flash-backward
row uses independent shape anchors; every other row is covered by the generic
candidate fallback when no exact shape sidecar exists.

The new SOTA-informed tile/resource envelope is retained as a committee
alternative at `docs/sm86_tile_pipeline_envelope_20260929.md`.  It predicts
the fresh flash score row from independent `Sk=64/96/192/256` resource-rate
anchors at 0.493% GPC / 0.483% duration cold and 0.273% / 0.265% warm.  Its
projection takes 0.0056--0.0065 s per family, while the promoted whole-cohort
projector remains the speed path.  The fallback now covers the other fourteen
rows with explicit `candidate_fallback` provenance and widened intervals.

The cache reuse-signature alternative at
`docs/sm86_cache_reuse_signature_20260929.md` separates a cold shape endpoint
from a paired warm/cold ratio and emits a predicted DRAM/L2 sector signature.
It reaches 0.422%/0.419% cold and 0.189%/0.164% warm GPC/duration APE on the
matched flash row. It is non-promoted; the independent frontier/trace fallback
now covers every other row without emitting an OOD refusal.

The paired coverage summaries report 15/15 predictions in each cache phase for
both committee lanes: one `exact_shape` flash row and fourteen explicit
`fallback_covered` rows.  Fallback intervals remain broad and provenance is
retained; the promoted model's scores are unchanged.
The promoted projection has the matching all-row summary at
`results/alpha/sm86_ncu_projection_frontier_archetype_refined_live_coverage_20260929.json`;
it reports 15/15 predictions with one exact-shape row and fourteen
`frontier_covered` rows.

The interval-analysis frontier-ratio bundle is retained as a negative committee
experiment.  It fuses an analytical frontier efficiency ratio with a
duration/GPC crosswalk and keeps all rows covered, but its fresh MAPE is
8.339%/8.752% cold-large and 9.566%/9.811% warm-large (GPC/duration), so it is
not promoted.  Its value is the explicit latent-efficiency decomposition and a
future hook for time-binned CUPTI PM samples.

The stage-centric interval alternative at
`docs/sm86_stage_interval_projection_20260929.md` follows GCStack/GCScaler
semantic stage separation and recent max-plus analytical modeling.  Its
sampled-one-threadblock path covers all 15 refined rows with candidate or
candidate-fallback output, scoring 4.194%/4.087% cold and 10.476%/8.514% warm
GPC/duration at roughly 37x/32x projection-only speedup.  It is retained as a
negative committee member; semantic stages are not physical pipe identities.

The promoted projector also exposes an optional static semantic nearest-neighbor
residual blend.  The exploratory alpha=0.5 cold sensitivity covers all 15
rows, lowers aggregate GPC MAPE from 3.101% to 2.892%, and leaves duration at
3.039%.  It is explicitly non-promoted because alpha was not pre-registered
before score sensitivity exploration; the default blend remains zero.

## Four blocker replacements

### Physical pipe / issue tags

* Promoted point path: separate issue/event/launch frontiers.
* Statistical path: PM-sampling stall envelopes and resource-class mixtures.
* Evidence: full PM-sampling campaign, isolated instruction classes, direct PMA
  PC/stall records, and NAK family PM profiles.
* Boundary: no public dynamic `(PC, warp, issue ordinal, physical sub-unit,
  issue cycle)` witness; short bridged bodies often produce zero PC samples.

### CUDA ABI / ISA / control metadata

* Promoted path: semantic trace/frontier projection, official PTX compiler/linker
  adapters, NAK native code, and bounded NAK→CUDA-ELF bridges.
* New evidence: the same NAK mixed body runs through two NVIDIA-generated
  parameter layouts; cold timing remains near 2.464 us while warm timing shifts
  from 2.336 to 2.432 us with identical 26/18-ALU/3-LSU populations.
* Boundary: metadata is template-derived; no unrestricted independent CUDA ABI,
  QMD/control-word, relocatable ELF, or complete SASS generator is exposed.

### Cache ownership / read-from / DRAM completion

* Promoted path: explicit cold/warm cache-phase transfer intervals.
* Evidence: independent flash shape cache-sector pairs; target DRAM reads change
  2,940→88 sectors and L2 misses 2,176→0. Bridge-family warm/cold transfer
  predicts mixed-body warm timing within 0.717% duration / 0.654% GPC.
* Boundary: no owner, read-from source, eviction order, dirty payload residency,
  or proof of complete DRAM writeback.

### Unified timing endpoint

* Promoted path: CUDA-event→NCU endpoint transform plus frontier/archetype
  projection.
* Evidence: independent paired-warm CUDA-event anchors predict target NCU within
  0.081% duration / 0.093% GPC; full campaigns still show simulator and NCU
  endpoints differ by cache phase.
* Boundary: NCU WaitForIdle/semaphore, CUPTI activity, CUDA events, device
  timers, simulator completion, and host replay remain distinct semantics.

## Conservative model inventory

| model | role | latest evidence |
|---|---|---|
| `frontier-archetype` | promoted point prediction | refined cold 2.586/2.698% MAPE on prior large holdout; fresh full 2.655/2.626% |
| endpoint lattice | committee/uncertainty model | fresh independent shape and PM evidence; not promoted where it regresses |
| cache-phase envelope | cold/warm transfer | target and bridge phase ratios with explicit intervals |
| CUDA-event→NCU transform | timing endpoint crosswalk | 0.081% / 0.093% fresh target APE |
| PM-sampling stall envelope | statistical scheduler abstraction | wait/short-scoreboard/branch/barrier/math populations interpolate closely; long-scoreboard variance retained |
| bridge operation envelope | ABI/resource interval | aggregate family range; no universal opcode timing |
| tile/resource pipeline envelope | interpretable max-plus demand plus coverage fallback | exact flash 0.493/0.483% cold and 0.273/0.265% warm; all 15 rows covered |
| cache reuse-signature | local cold/warm ratio and sector-phase envelope plus fallback | exact flash 0.422/0.419% cold and 0.189/0.164% warm; all 15 rows covered |
| frontier-ratio bundle | interval-analysis efficiency and duration/GPC crosswalk | negative result; retained for future PM-timeseries conditioning |
| PM-state envelope | aggregate long/wait/short/branch/barrier/math phase shares | negative result; warm flash 5.05%/5.03% APE with containing interval |
| stage interval projection | GCStack-style semantic demand intervals and max-plus efficiency surface | all 15 rows covered; 4.194/4.087% cold and 10.476/8.514% warm; not promoted |
| neighbor frontier sensitivity | static semantic nearest-neighbor endpoint/frontier ratio blend | exploratory cold 2.892/3.039% all-row MAPE; disabled by default, not promoted |
| clock-regime projection | external active-clock duration crosswalk with regime interval | shifted live duration 23.595% -> 12.413%; GPC unchanged; opt-in only |
| high-clock cohort transfer | high-regime frontier efficiency transfer across family holdout | all 15 rows; 8.475/9.038% GPC/duration; regime-local, not promoted |
| clock elasticity | bounded memory-wait/other exposure response to core-clock ratio | 6.637/7.733% GPC/duration high-clock score; regime-local, not promoted |

The native replay remains unchanged and remains the mechanism/reference path.
The models predict observable NCU-oriented metrics or bounded latent effects;
they do not rename unresolved mechanisms as complete.

## Current live refresh

The latest performance-only refresh completed 15/15 cold and 15/15 warm
workloads. The refined predictor scores those fresh NCU rows at 2.655%/2.626%
MAPE cold and 6.314%/6.422% warm. Fresh PM sampling also completed 15/15;
the flash row has 336 samples at a 2,048-cycle interval, long-scoreboard
980.66, wait 400.81, and no buffer overflow.

Cache-transfer ratios remain workload-specific (roughly 0.763--1.012 on the
large cohort), so median transfer is only a fallback. Paired local cache-phase
archetypes and endpoint transforms remain the preferred bounded paths.

## Latest refresh

The current-driver full refresh completed 15/15 cold and warm workloads. Fresh
large-cohort score intervals cover 6/7 cold rows and 7/7 warm rows for both
GPC and duration (`results/alpha/refined_live_interval_coverage_20260929/`).

The ABI bridge refresh also exercised a shifted NVIDIA-generated parameter
layout. Identical NAK SASS/resource populations changed warm endpoint timing
from 2.336/3,085 to 2.432/3,198 (duration/GPC), while cold timing stayed near
2.464/3,27x. This is direct metadata-layout sensitivity, not an unrestricted
ABI result.

A separate 15/15 cold live campaign at
`results/alpha/full_3090ti_20260929_stage_cold_live/` completed with three NCU
repeats. Its reported 1.89 GHz device attribute and 9,600 ns / 17,866 GPC
large medians differ from the retained score regime, so this campaign is
preserved as timing-regime evidence only and is not mixed into calibration.

The clock-sampled follow-up at
`results/alpha/full_3090ti_20260929_stage_clock_sampled_live2/` recorded 1,769
external samples, 1,905 MHz median active SM clock, and large medians of 9,280
ns / 17,766 GPC cycles. This confirms the regime shift with an actual clock
trace; the rows remain outside frozen calibration.

The opt-in clock-regime adapter scales only projected duration by the external
active-clock ratio. On the sampled live rows it reduces duration MAPE from
23.595% to 12.413% while leaving GPC at 13.073%; raw and adjusted endpoint
domains remain separate and the adapter is not promoted.

The same-regime high-clock cohort transfer uses the first high-clock campaign
as calibration and the second as a family holdout. It covers all 15 rows at
8.475% GPC / 9.038% duration MAPE and roughly 1,029x projection speedup. This
is regime-local related-family evidence and remains non-promoted.

The train-only leave-one-out-selected static-neighbor variant reaches 4.353%
GPC / 6.386% duration MAPE at roughly 58x on the same high-clock holdout. It
remains regime-local and non-promoted.

The native CUDA-event endpoint-ratio crosswalk reaches 10.398% GPC / 11.702%
duration MAPE on the 15-row high-clock warm score with family-disjoint
same-regime calibration. It remains a non-promoted endpoint model.

The warm-cache scenario interval provides 93.33% coverage for both GPC and
duration across all 15 warm rows, with point MAPE 16.016%/16.748%. It is an
uncertainty envelope, not cache mechanism closure.

The explicit NCU `--clock-control base` campaign completed 15/15 but external
sampling still observed 2,040 MHz active median / 2,070 MHz maximum. Its large
medians were 10,368 ns / 16,093.5 GPC cycles, so requested profiler clock
control remains a typed timing-regime observation rather than a proven lock.

The two-component clock-elasticity model separates replay memory-wait exposure
from other readiness exposure. Paired low/high clock calibration reaches
6.637% GPC / 7.733% duration MAPE on the 15-row high-clock score versus
14.927%/20.193% frozen reference application; it remains non-promoted.
