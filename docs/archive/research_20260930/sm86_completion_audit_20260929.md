# SM86 blocker closure audit — 2026-09-29

This is a scope-preserving evidence audit, not a test, hash, verification
workflow, or completion declaration. The active objective remains broader than
the bounded mechanisms currently available.

## Full-campaign deliverable

**Evidence:**

- `results/alpha/full_3090ti_20260929_022713_pidfd/`: finished, 15/15, no
  recorded stage errors; cold-cache medians native 0.009407 ms, NCU 10,016 ns,
  NCU GPC 13,466 cycles, simulator event 11,780 cycles, simulator GPC
  envelope 13,596 cycles, replay wall 0.631848 s.
- `results/alpha/full_3090ti_20260929_023820_pidfd_warm/`: finished, 15/15,
  no recorded stage errors; warm-cache medians native 0.009407 ms, NCU 8,576
  ns, NCU GPC 11,538 cycles, simulator event 11,780 cycles, simulator GPC
  envelope 13,596 cycles, replay wall 0.634228 s.
- Current identity/clock observation: RTX 3090 Ti, UUID
  `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, SM 8.6, driver 615.71.09,
  P0, 1350 MHz SM, 10501 MHz memory.

**Status:** achieved for the requested performance-campaign deliverable.
These are separate endpoint observations, not an elapsed-time accuracy score.

### Fresh refined full campaigns

The current-driver refresh is retained at:

- `results/alpha/full_3090ti_20260929_refined_cold_live/`: 15/15 completed,
  no recorded stage errors, median NCU 10,048 ns / 13,506 GPC cycles, replay
  wall 0.66618 s.
- `results/alpha/full_3090ti_20260929_refined_warm_live/`: 15/15 completed,
  no recorded stage errors, median NCU 8,544 ns / 11,516 GPC cycles, replay
  wall 0.66616 s.
- `results/alpha/full_3090ti_20260929_refined_sampling_live/`: 15/15
  performance-only PM-sampling rows, with 2,048-cycle PC sampling and no
  correctness/hash/verification stage.

The fresh full-campaign comparison is consolidated at
`results/alpha/refined_live_campaign_comparison_20260929/comparison.json`.
The refined frontier/archetype projector scores 2.655% GPC / 2.626% duration
MAPE cold and 6.314% / 6.422% warm on these fresh rows. This refreshes the
performance observation; it does not change the mechanism boundary below.

The new committee coverage summaries at
`results/alpha/sm86_tile_pipeline_envelope_goal_20260929/coverage_summary.json`
and `results/alpha/sm86_cache_reuse_signature_goal_20260929/coverage_summary.json`
show 15/15 predictions in both cold and warm phases: one exact flash shape and
fourteen explicit `candidate_fallback` rows. Missing shape metadata no longer
drops a workload or emits an OOD refusal; fallback intervals remain wider and
their provenance is retained.

### Latest PM-sampling and endpoint evidence

- `results/alpha/full_3090ti_20260929_refined_sampling_live/` contains a
  current-driver 15/15 PM-sampling refresh. The flash row has 336 PC samples at
  a 2,048-cycle interval, long-scoreboard 980.66, wait 400.81, and no buffer
  overflow.
- `results/alpha/sm86_endpoint_transform_goal_20260929/` contains an
  independent paired-warm CUDA-event-to-NCU transform with 0.081% duration /
  0.093% GPC target APE. It is a local endpoint crosswalk, not a unified clock.
- `results/alpha/sm86_nak_cuda_elf_bridge_mixed_alt_template_20260929/` and
  its PM-sampling companion show shifted ABI parameter-layout sensitivity:
  identical 26/18-ALU/3-LSU populations but warm timing 2.336/3,085 versus
  2.432/3,198 duration/GPC. PM stalls remain aggregate and PC samples remain
  empty.
- `results/alpha/sm86_bridge_cache_phase_goal_20260929/` shows an independent
  family cache transfer predicting mixed warm timing within 0.717% duration /
  0.654% GPC.

These additions strengthen metric-level substitutes and interval models. They
do not provide the missing physical pipe/issue witness, unrestricted CUDA
ABI/ISA/control generation, cache owner/read-from/complete DRAM oracle, or a
single public timing endpoint.

### New tile/resource envelope committee

The SOTA-informed `tools/sm86_tile_pipeline_envelope.py` uses independent
`Sk=64/96/192/256` anchors and a max-plus semantic demand envelope.  On the
matched flash row in the fresh full campaign it predicts 200,708.8 GPC cycles /
148,715.5 ns cold versus 199,725 / 148,000 NCU (0.493% / 0.483% APE), and
175,571.8 / 130,103.3 warm versus 175,094 / 129,760 (0.273% / 0.265%).  Its
exact flash path is retained as an interpretable committee member, while the
independent frontier/trace fallback covers all other fresh rows with widened
intervals and explicit provenance.  Reports are under
`results/alpha/sm86_tile_pipeline_envelope_goal_20260929/` and the source study
is `docs/sm86_tile_pipeline_envelope_20260929.md`.

### New cache reuse-signature committee

The paired cache matrix also feeds
`tools/sm86_cache_reuse_signature.py`, which predicts a cold endpoint and a
warm/cold ratio separately and emits aggregate sector signatures.  On the
matched fresh flash row it predicts 200,567.6 GPC cycles / 148,619.9 ns cold
versus 199,725 / 148,000 (0.422% / 0.419% APE), and 175,424.1 / 129,972.3
 warm versus 175,094 / 129,760 (0.189% / 0.164%).  The fallback covers all
other fresh rows without dropping them.  This improves the local cache-phase
projection but remains a committee model; aggregate sector interpolation is
not a line-owner, read-from, or complete-DRAM witness.

### Frontier-ratio bundle research

The new `tools/sm86_frontier_ratio_bundle.py` uses an interval-analysis
frontier efficiency ratio plus an independent duration/GPC crosswalk.  It is a
negative but useful experiment: cold-large MAPE is 8.339%/8.752% and
warm-large MAPE is 9.566%/9.811% GPC/duration.  It is not promoted; its next
extension is time-binned CUPTI PM conditioning rather than invented physical
route or cache-owner labels.

The PM-state follow-on `tools/sm86_pm_state_envelope.py` uses independent
PM-sampling anchors as aggregate phase emissions.  It predicts the warm flash
target at 5.05%/5.03% GPC/duration APE with a containing interval, but reaches
11.23%/11.25% on the large holdout.  It is non-promoted and does not expose
per-PC route identity.

### Stage-centric interval committee

The new `tools/sm86_stage_interval_projection.py` studies the public
GCStack/GCScaler CPI-stack decomposition and recent stage-centric max-plus
analytical modeling.  It uses semantic compute/memory/control/sync/tail
demand classes, optional one-threadblock traffic sampling, and an independent
log-efficiency surface.  The sampled path covers all 15 refined rows with
candidate or candidate-fallback output: cold MAPE is 4.194%/4.087% and warm
MAPE is 10.476%/8.514% GPC/duration at about 37x/32x projection-only speedup.
It is a negative, non-promoted committee member; semantic classes are not
physical pipes and the endpoint crosswalk is not a unified clock.

The separate live campaign
`results/alpha/full_3090ti_20260929_stage_cold_live/` completed 15/15 with
three NCU repeats.  Its reported device clock attribute and large-row medians
indicate a different timing regime, so its measurements remain score-only
evidence and were not mixed into calibration.

The follow-up
`results/alpha/full_3090ti_20260929_stage_clock_sampled_live2/` adds an
external 100-ms sampler: 1,769 samples, 1,905 MHz active-SM median, and large
NCU medians 9,280 ns / 17,766 GPC cycles. This is stronger timing-regime
evidence, but it remains outside the frozen 1,350 MHz calibration domain.
The shifted-clock comparison is retained separately: promoted 13.073%/23.595%
and stage 13.406%/24.612% GPC/duration MAPE; neither is a same-regime accuracy
claim.

The two-component clock-elasticity model separates replay memory-wait exposure
from other readiness exposure and reaches 6.637%/7.733% GPC/duration MAPE on
the high-clock score versus 14.927%/20.193% frozen reference application. It
is a regime-local non-promoted committee result.

The native CUDA-event endpoint-ratio crosswalk reaches 10.398%/11.702%
GPC/duration MAPE on the warm high-clock score with family-disjoint calibration.
It improves the observable endpoint projection but does not establish a unified
timing endpoint.

The warm-cache scenario interval covers 93.33% of the 15 warm rows for both
GPC and duration, with point MAPE 16.016%/16.748%. It is retained as a
conservative cache uncertainty envelope, not closure.

The NCU base-clock campaign requested profiler clock control but external
sampling still observed 2,040 MHz active median / 2,070 MHz maximum. This
confirms that the requested profiler mode is not itself an external clock
witness on this host.

The fresh warm campaign’s NCU-local frequency/cycle/duration audit reports
0.134% median, 0.190% p90, and 0.277% maximum residual across 15 workloads.
It remains an NCU-local relation, not unified endpoint closure.

The long independent NAK PMA body provides 1,560 direct records from 1,330
static instruction words, improving sampling coverage but still lacking route,
sub-unit, slot, and exact issue-cycle fields.

A dedicated interleaved probe provides stronger single-kernel context: NCU
per-kernel SM/GPC frequency medians are 1.546875 GHz for `base` and 1.867973
GHz for `boost`. This refines the interpretation of the broad sampler, but
the profiler counters remain distinct from a synchronized device timer.

The train-only leave-one-out-selected static-neighbor high-clock variant reaches
4.353%/6.386% GPC/duration MAPE at roughly 58x on the same holdout. It is a
regime-local committee result and does not alter the promoted defaults.
The opt-in duration-only clock adapter lowers the promoted shifted-regime
duration MAPE to 12.413% while leaving GPC unchanged. It is a bounded timing
crosswalk, not closure of the unified-endpoint requirement.

The same-regime high-clock transfer then used the first high-clock campaign as
calibration and the second as a family holdout. It covers all 15 rows at
8.475%/9.038% GPC/duration MAPE and roughly 1,029x speedup. This is
regime-local related-family evidence, not universal generalization or timing
closure.

### Static neighbor-frontier sensitivity

The promoted projector now has an optional static semantic nearest-neighbor
residual blend, disabled by default.  The exploratory alpha=0.5 cold report
covers all 15 rows and reduces aggregate GPC MAPE from 3.101% to 2.892% while
leaving duration at 3.039%.  Because alpha was not pre-registered before score
sensitivity exploration, this is not promoted and does not alter the default
projector.  Static semantic neighbors remain abstractions, not pipe, ABI, or
cache-owner witnesses.

## Exact hardware pipe witnesses

**Evidence:**

- `results/alpha/sm86_isolated_pipe_classes_20260929/` contains current-clock
  static SASS populations and aggregate NCU rows for IADD3, FFMA, MUFU.RCP,
  LDG, SHF, PRMT, SHFL.BFLY, and BAR.SYNC.
- `results/alpha/sm86_pipe_metric_surface_20260929/` records the current NCU
  metric query and source/raw pages; pipe metrics are kernel aggregates with no
  per-SASS-address route field.
- `results/alpha/sm86_nak_pma_probe_20260929/` adds direct RM/PMA PC/stall
  records for independently generated NAK code, still without a pipe/issue
  field.

**Status:** unresolved. These are class-local population correspondences, not
universal dynamic physical-subunit or issue-cycle witnesses. Closure requires a
silicon/RM/firmware trace source that tags each issued instruction.

## Unrestricted compiler/ISA generation

**Evidence:**

- `tools/nvptx_compile_sm86/` and
  `results/alpha/sm86_nvptx_compiler_surface_20260929/` demonstrate official
  NVIDIA PTX compilation, linked-cubin generation, and native Driver API
  launch.
- A later restart corrected a missing NUL terminator in the original
  `nvJitLinkAddData` adapter. The corrected matrix links both previously
  disputed address-arithmetic forms.
- `denvdis`/all-form artifacts provide bounded SM86 word coverage only.
- `results/alpha/sm86_nak_independent_codegen_20260929/` demonstrates an
  independent Mesa NAK SM86 compiler emitting and executing native shader code
  on the card.
- `results/alpha/sm86_nak_operation_family_20260929/` expands that observation
  across seven operation families with retained binaries and disassembly.
- `results/alpha/sm86_nak_cuda_elf_bridge_20260929/` wraps a NAK body in
  NVIDIA-generated CUDA ELF metadata and successfully launches it with CUDA
  pointer parameters.
- Its `family/` subartifact launches five additional NAK operation bodies via
  the same bridge.

**Status:** materially improved, still unresolved. NVIDIA's official PTX
compiler/linker path is now exercised with corrected PTX text handling, but no
independent unrestricted SASS encoder, scheduler/control-word generator,
relocatable CUDA ELF ABI, or complete native ISA contract exists in the
available public interfaces. The NAK bridge reuses NVIDIA metadata and only
hand-covers one parameter layout; it is not a general CUDA ABI/linker.

## Native cache ownership and flush semantics

**Evidence:**

- `results/alpha/rm_mem_op_d_semaphore_3090ti_20260928/` proves corrected
  C56F `MEM_OP_D_OPERATION_L2_FLUSH_DIRTY` packet retirement and completion
  timing, not dirty-line writeback semantics.
- `results/alpha/rm_l2_internal_control_probe_20260929/` shows internal RM
  L2 controls return `NV_ERR_NOT_SUPPORTED` at the user boundary.
- `results/alpha/rm_public_l2_flush_control_probe_20260929/` shows the public
  full-cache RM control accepts current-driver writeback+invalidate flags
  `0x38`/`0x3b`.
- `results/alpha/nak_shader_public_flush_coupling_20260929/` couples a real
  shader producer, CPU-visible allocation, and accepted public flush control.
  Its five-repeat host-I/O medians are 225.750 us (`0x38`) and 141.152 us
  (`0x3b`), not direct L2-service measurements.
- `results/alpha/rm_image_uvm_external_map_probe_20260929/` and
  `rm_image_uvm_cuda_endpoint_probe_20260929/` establish copy-engine/CPU BAR1
  sharing but no shader-store visibility; forcing uncached UVM mapping does not
  change that result.

**Status:** materially improved, still unresolved. A public full-cache
writeback+invalidate control is now accepted and coupled to a real shader
producer/CPU consumer, but no guaranteed L2-bypass consumer or exposed
owner/read-from/complete-writeback oracle is available.

## Drift-free matched timing

**Evidence:**

- `results/alpha/timing_crosswalk_3090ti_20260929_current/` shows current
  driver/clock CUPTI and CUDA-event grid spans agreeing within about 3.4 µs on
  the 108-block control; block-0 `%globaltimer` undercounts by about 8.864 ms.
- Full campaigns held the requested clocks and removed host wait polling via
  pidfd readiness.

**Status:** unresolved. NCU WaitForIdle, CUPTI activity, CUDA events, native
  timers, simulator frontiers, and replay wall time remain semantically
  distinct endpoints; no public matched accuracy relation has been established.

## Overall result

The full-campaign performance requirement is satisfied and refreshed on the
current host. The four named mechanism blockers are not. New live evidence
includes refined flash shape/endpoint lattices, cache-sector phase pairs,
current PM sampling, corrected mixed NAK bridge template sensitivity, and
fresh NAK operation-family PM profiles. These are conservative latent-effect
models and interval/partial-order evidence; no bounded substitute is renamed
as unrestricted compiler, physical pipe trace, cache ownership proof, or
unified timing endpoint. The research objective remains mechanism-incomplete;
goal status is controlled by the external goal record and is not changed by
this committee-model result.
