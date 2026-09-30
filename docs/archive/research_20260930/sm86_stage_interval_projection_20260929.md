# SM86 stage-centric interval projection (2026-09-29)

## Why this model

The latest research pass studied the public **GCStack/GCScaler** source and
its interval/CPI-stack decomposition, the stage-centric analytical model in
arXiv:2605.04178, and the reuse-distance/cache abstractions in the public
GPU-cache-model repository.  GCStack exposes separate base, memory, compute,
synchronization, control, and idle contributions rather than assigning every
cycle to an exact pipe.  The recent analytical model composes compute, memory,
synchronization, and writeback with a max-plus bottleneck.  The cache model
limits detailed state to representative cores/threads and transfers reuse
profiles across the machine.  These are useful alternatives to attempting to
reconstruct unavailable NVIDIA RM state.

Sources:

* [GCStack+GCScaler repository](https://github.com/yonsei-hpcp/gcstack_gcscaler)
  and its [CPI-stack run contract](https://github.com/yonsei-hpcp/gcstack_gcscaler/blob/main/gcstack/README.md)
* [GCScaler interval-model source](https://github.com/yonsei-hpcp/gcstack_gcscaler/blob/main/gcscaler/src/interval_model.cc)
  and [cache-model source](https://github.com/yonsei-hpcp/gcstack_gcscaler/blob/main/gcscaler/src/cache_model.cc)
* [Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures](https://arxiv.org/abs/2605.04178)
* [GPU cache model using reuse distance](https://github.com/tue-es/gpu-cache-model)

## Model contract

`tools/sm86_stage_interval_projection.py` extracts static opcode/control
populations, replay issue/event/GPC-envelope frontiers, and optional sampled
trace traffic.  It maps instructions into semantic **demand classes**:

```text
compute, memory, control, synchronization, launch/tail
```

The names are analytical classes, not physical SM86 pipe or issue-slot labels.
For each endpoint the model uses

```text
GPC base      = max(sim_gpc_envelope_cycles, sim_event_cycles)
duration base = sim_event_cycles × (1000 / 1350 MHz)
prediction    = base × exp(independent log-efficiency surface)
```

The surface contains dimensionless stage overlap, demand-density, occupancy,
tail, and static control terms.  Duration/GPC consistency is a separate
log-space crosswalk; it is not a unified clock claim.  An independent nearest
stage state controls interval width only.  Every score row receives either
`candidate` or `candidate_fallback`; distance widens the interval and never
causes an OOD refusal or row drop.

The source-level audit of GCScaler was especially instructive: its
`interval_model.cc` walks decoded warps, consumes an Accel-Sim cache-stat
stream, selects a representative warp, profiles its intervals, and then runs
the performance model.  Its `cache_model.cc` joins each load/store by PC and
instruction index to aggregate L1/L2 hit/miss populations before attaching
them to the selected warp.  Our SM86 path cannot import those foreign cache
records, so it substitutes replay frontiers and bounded static/sample traffic
features while retaining the separation between representative demand, cache
phase, and endpoint composition.  This is an abstraction transfer, not a
claim that the two cache states are equivalent.

The default metadata-only path avoids protobuf walking.  `--dynamic-trace
--dynamic-sample 1` reads one representative threadblock per workload, retaining
the sampled traffic densities while keeping the projection far below replay
cost.  The sampled path is the retained comparison below.

## Frozen-cohort score-only results

The model was fit on an independent base/large cohort and scored against the
fresh retained refined campaigns.  NCU values are loaded only after each
prediction for APE reporting.

| phase | rows | GPC MAPE | duration MAPE | model time | projection speedup |
|---|---:|---:|---:|---:|---:|
| cold | 15/15 | 4.194% | 4.087% | 0.310 s | 36.9× |
| warm | 15/15 | 10.476% | 8.514% | 0.353 s | 31.7× |

Coverage summaries are in:

* `results/alpha/sm86_stage_interval_projection_goal_20260929/cold_coverage_summary.json`
* `results/alpha/sm86_stage_interval_projection_goal_20260929/warm_coverage_summary.json`

The cold and warm large-only holdouts are 3.394%/5.103% and
6.387%/5.598% GPC/duration MAPE respectively.  The promoted
frontier-archetype projector remains lower-error (2.655%/2.626% on the fresh
cold full campaign and 6.314%/6.422% warm), so this stage model is **not
promoted**.  Its contribution is interpretable stage intervals and a cheap
sampled path, not a claim of scheduler or cache closure.

## Fresh live campaign and timing-regime observation

`results/alpha/full_3090ti_20260929_stage_cold_live/` completed all 15 rows
with three NCU repeats and `--cache-control all`.  Its large-row medians were
9,600 ns / 17,866 GPC cycles.  NCU reported the device attribute clock at
1.89 GHz for this campaign, while the retained refined score domain was
interpreted using the 1.35 GHz anchor.  This is therefore a **clock-regime
observation**, not a clean repeat suitable for mixing into the frozen model.
The raw comparison is retained in
`results/alpha/full_3090ti_20260929_stage_cold_live/clock_regime_observation.json`
and was not used for calibration or promotion.

A follow-up campaign at
`results/alpha/full_3090ti_20260929_stage_clock_sampled_live2/` added an
external 100-ms `nvidia-smi` sampler.  It recorded 1,769 samples, all-sample
SM clocks from 210--2,085 MHz, and a 1,905 MHz active-sample median; its large
NCU medians were 9,280 ns / 17,766 GPC cycles.  This confirms that the earlier
1.89 GHz NCU attribute reflected a real high-clock regime during active
profiling, not merely a static metadata field.  The two model comparison files
are retained as clock-regime observations only and are excluded from
calibration.
Against those shifted-clock rows, the promoted projector is 13.073%/23.595%
GPC/duration MAPE and the sampled stage model is 13.406%/24.612%.  These
numbers are deliberately not treated as same-regime accuracy results; they
quantify the cost of mixing endpoint regimes and motivate retaining clock
state as an explicit latent input.

## Blocker mapping

| blocker | stage abstraction | remaining boundary |
|---|---|---|
| physical pipe/issue tags | semantic demand intervals and max-plus bottleneck | no per-instruction route/slot witness |
| CUDA ABI/ISA generation | ABI/compiler effects remain an efficiency residual | no unrestricted independent SASS/QMD/ELF producer |
| cache ownership/read-from/DRAM | optional sampled traffic density and phase interval | no line-owner/read-from/completion oracle |
| timing endpoints | separate GPC/duration crosswalk and clock-regime metadata | no synchronized NCU/CUPTI/event/device clock |

The default native replay and all prior promoted/committee models remain
unchanged.  This pass ran no tests, hashing, or verification workflow.
