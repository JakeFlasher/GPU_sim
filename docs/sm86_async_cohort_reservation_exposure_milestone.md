# SM86 async-copy cohort reservation-exposure milestone

**Status:** complete 2026-08-31; identification no-go, no timing or tail
promotion

**Architecture:** `sm_86` / NVIDIA GeForce RTX 3090 Ti

**Evidence root:**
`results/captures/sm86_async_cohort_reservation_20260831_v1/`

**Machine report:** `analysis/report.json`

## Outcome

The fresh paired NCU campaign admits the source marginals through the
L1TEX-attributed local-device LTS request stream. All 80 bundles, 400
single-pass metric reports, and 800 ordered kernel rows pass their frozen
binary, device, launch-order, instruction, request-work, and paired-marginal
checks. Raw DRAM remains visible only as device-background telemetry.

Schema 18 adds a fixed first-response-anchored destination-reservation
exposure score. It predicts the correct negative sign on both predeclared
identification comparisons, but overstates their magnitude: the prediction is
`-17.71875` cycles/group while the live medians are `-10.322` and `-5.460`.
Identification MAE is `9.828` cycles/group and maximum error is `12.259`, well
outside the frozen `1.0`/`2.0` limits. The untouched holdout is therefore not
evaluated.

This is a completed mechanism no-go, not a request to scale or fit the score.
Production async-copy completion, wait readiness, traffic, and modeled event
time are unchanged. The typed `async_copy_shared_visibility` tail remains
required.

## Corrected source-attribution contract

An initial four-family pre-run was canceled before it became campaign
evidence. Its proposed contract lacked an independently collected LTS-total
population and therefore could not establish the exact requested-sector total
from hit and miss populations gathered in separate application replays. No
partial four-family result is admitted.

The corrected contract collects five metrics separately, one application
replay pass per report:

```text
smsp__inst_executed_op_ldgsts.sum
lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum
lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum
lts__t_sectors_srcunit_tex_aperture_device_op_read.sum
dram__sectors_read.sum
```

The fourth metric is the exact local-device-aperture LTS requested-sector
total. For every depth `1/2/4/8`, spacing `0/4`, and repetition `1..10`, one
application executes both variants in recorded order: odd repetitions use AB
and even repetitions BA. Each of its five reports contains exactly two target
rows, IDs 0 and 1, and `profiler__replayer_passes=1`. Every row has
`64 * depth` LDGSTS instructions and exactly `64 * depth * 32` LTS requested
sectors. Hit and miss populations are separately nonzero; their paired deltas
and their separate-profile closure against the exact total remain within
`max(4, ceil(1% * expected sectors))`.

This frontier proves requested L2-hit/L2-miss demand from the L1TEX source unit
through the local-device aperture. It does not identify physical DRAM
completions. NVIDIA documents that asynchronous display, copy, and video
engines share L2/DRAM and that their traffic cannot be isolated from SM traffic
with these device-wide counters in
[the Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#asynchronous-gpu-activity).
The retained RTX 3090 Ti reports display activity `Enabled`. Consequently,
`dram__sectors_read.sum` and `raw_dram - attributed_lts_misses` are non-gating
background diagnostics; they cannot override the source-attributed LTS gates.

## Evidence coverage

| lane | retained evidence |
|---|---:|
| paired NCU bundles | 80 / 80 |
| focused one-pass NCU reports | 400 / 400 |
| ordered NCU rows | 800 / 800 |
| pairing schema-18 twins | 16 cells / 32 outputs |
| retained-spacing schema-18 twins | 22 cells / 44 outputs |
| schema-18 cells normalized to schema 17 | 38 / 38 |

The host is device 0, GPU UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6, driver
610.57.04, and NCU 2026.1. The v2 probe binary, target arguments, grid, block,
kernel, and source/destination marginals are unchanged.

The fresh source-attribution summary is:

| depth | spacing | bundles | tolerance | max hit delta | max miss delta | max separate-profile closure |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 10 | 21 | 16 | 16 | 16 |
| 1 | 4 | 10 | 21 | 16 | 16 | 16 |
| 2 | 0 | 10 | 41 | 32 | 32 | 0 |
| 2 | 4 | 10 | 41 | 32 | 32 | 32 |
| 4 | 0 | 10 | 82 | 64 | 64 | 64 |
| 4 | 4 | 10 | 82 | 64 | 64 | 64 |
| 8 | 0 | 10 | 164 | 128 | 128 | 128 |
| 8 | 4 | 10 | 164 | 128 | 128 | 128 |

Raw DRAM background excess ranges from 60 to 68,620 sectors across the
retained bundles. This spread is retained rather than clipped and is not an
admission failure.

## Schema-18 reservation observation

Schema 18 reuses the exact schema-17 sector-to-cohort response state. For each
copy `i`, it anchors at the earliest cohort response and applies the already
frozen topology penalty:

```text
A_i = min_c(first_response_i,c)
P_i,c = 4 * (bypass_wavefronts_i,c - 1)
      + 2 * (bank_wavefronts_i,c - 1)

prearrival_hidden_i,c = min(P_i,c, first_response_i,c - A_i)
stream_hidden_i,c = min(
    P_i,c - prearrival_hidden_i,c,
    last_response_i,c - first_response_i,c
)
residual_i,c = P_i,c - prearrival_hidden_i,c - stream_hidden_i,c

copy_residual_i = max_c(residual_i,c)
group_residual_g = max_i(copy_residual_i)
```

The five checked additive diagnostics have explicit aggregation semantics:

- `cohort_overlap_groups_observed` counts projected groups and selects schema
  18 even when their residual exposure is zero.
- `cohort_response_envelope_cycles` sums one physical envelope per copy,
  `max_c(last_response_i,c) - A_i`.
- `cohort_topology_prearrival_hidden_cycles` sums `prearrival_hidden_i,c`
  across cohorts.
- `cohort_topology_stream_hidden_cycles` sums `stream_hidden_i,c` across
  cohorts.
- `cohort_topology_residual_exposure_cycles` takes the cohort-to-copy-to-group
  maxima above, then sums group residuals across waits and SMs.

Envelope and hidden-work totals are explanatory diagnostics, not alternative
selection scores. The paired prediction uses only the residual exposure
difference divided by groups. No response token, completion frontier,
destination service, wait consumption, traffic counter, card, scheduler
state, or typed tail reads these fields.

All 16 pairing traces and 22 retained issue-spacing traces have byte-identical
schema-18 replay twins. Removing the five fields and restoring schema 17 makes
all 38 cells exactly equal to their retained schema-17 predecessors. Seven
cells select schema 18 with zero residual, confirming that schema selection is
observation-based rather than magnitude-based.

## Identification decision

The frozen identification slice is depth 1 and 2 at spacing 0. Its paired
prediction is:

```text
(sum(group_residual_slow_expensive)
 - sum(group_residual_slow_cheap)) / groups
```

| depth | spacing | live delta | predicted delta | absolute error | sign correct |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | -10.322 | -17.719 | 7.396 | yes |
| 2 | 0 | -5.460 | -17.719 | 12.259 | yes |

The score has no fitted coefficient, intercept, layout constant, or scale.
Correct sign alone is insufficient: identification MAE `9.828` and maximum
error `12.259` fail the frozen limits. Per the predeclared sequence, the depth
4/8 spacing-0 and all spacing-4 holdout comparisons remain untouched. Schema
18 stays observational, and neither timing promotion nor typed-tail removal is
authorized.

The current cohorts each occupy one L2 line, so within-cohort stream hiding is
structurally zero in the pairing corpus. If this mechanism is revisited, the
next discriminating assay is an independent multi-line cohort response
envelope that produces separately completing source lines and can exercise
the stream-hiding term. Do not scale, fit, or retune the current score against
the retained live deltas.

## Reproduction

```bash
make -C workloads/research_probes/async_source_only async_cohort_pairing

CUDA_VISIBLE_DEVICES=0 LANG=C LC_ALL=C \
  PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_cohort_attribution.py \
  --smoke --timeout 300

CUDA_VISIBLE_DEVICES=0 LANG=C LC_ALL=C \
  PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_cohort_attribution.py \
  --root results/captures/sm86_async_cohort_reservation_20260831_v1 \
  --timeout 300

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_cohort_attribution.py \
  --root results/captures/sm86_async_cohort_reservation_20260831_v1 \
  --validate-only

cargo test --manifest-path sim/Cargo.toml --workspace --lib
cargo build --release --manifest-path sim/Cargo.toml -p sim-run

sim/target/release/sim-run \
  --trace results/captures/sm86_async_cohort_pairing_20260830_v2/nvbit/async_cohort_pairing_slow_expensive_d2_s0_g64 \
  --control results/captures/sm86_async_cohort_pairing_20260830_v2/static/control_bits.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json --mode l1-l2

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_cohort_reservation.py \
  --root results/captures/sm86_async_cohort_reservation_20260831_v1 \
  --baseline-root results/captures/sm86_async_cohort_pairing_20260830_v2 \
  --output results/captures/sm86_async_cohort_reservation_20260831_v1/analysis/report.json \
  --output-md results/captures/sm86_async_cohort_reservation_20260831_v1/analysis/report.md
```

The analyzer exits successfully for this scientifically complete no-go and
reports `complete_cohort_reservation_identification_no_go`, zero missing
artifacts, and zero validation errors.

## Content identities

Current retained identities are ordinary reproducibility metadata, not
authorization gates:

- Schema-18 simulator commit: `39ddf061`
- Paired capture-runner commit: `09908c2f`
- Reducer and completed-analysis commit: `a5ed9d48`
- Probe binary:
  `dded4cb03868f5cc757f86103d3ec45d4c8adf66c580b39ac90e3c4034f94f4f`
- Release `sim-run`:
  `83b123b9a15e590b39a076ce631ac3df99b5bc49baf3c578e3a51c9e60506673`
- Capture runner:
  `c33f36b37831611a9fcb8229a415984c75db5f641d30de3e270e213c84db7d04`
- Analyzer:
  `c983528e213e6c36762faa6e19ad76a733a4d193ac95f8cd2e5ec4ca11cc4539`
- Machine report:
  `40f7c3fbe7423995959f15b783557302848f950b7c4217ff6404bd002f18e161`
