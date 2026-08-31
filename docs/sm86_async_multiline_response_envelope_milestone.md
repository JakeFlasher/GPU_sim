# SM86 asynchronous-copy multi-line response-envelope milestone

## Outcome

The pending-line assay works as designed, but the existing schema-18
reservation-exposure score fails identification. The retained result is:

```text
complete_response_envelope_identification_no_go
```

The new trace geometry creates a real modeled response envelope inside one
flexible LDGSTS cohort. All static, live-GPU source-attribution, NVBit,
simulator-geometry, determinism, issue-control, and production-tail gates pass.
The fixed schema-18 score predicts -18 cycles/group for every depth-1
identification cell. The live paired medians have the same sign but range from
-10.143 to -5.979 cycles/group, producing 10.587 cycles/group MAE and 12.021
maximum absolute error. The frozen 1/2-cycle bounds fail, so depth-2 timing is
not evaluated.

No simulator timing, card, completion frontier, scheduler state, traffic, or
tail flag changes. `async_copy_shared_visibility` remains required.

The complete machine-readable result is
[`analysis/report.json`](../results/captures/sm86_async_multiline_response_20260831_v1/analysis/report.json).

## Mechanism exercised

Each copy uses eight four-lane/64-byte flexible source cohorts in one 2 KiB
step:

```text
aligned cohort bases: 0, 256, 512, 768
split cohort bases:   1088, 1344, 1600, 1856
lane offsets:         0, 32, 64, 96
```

The aligned cohorts occupy one 128-byte L2 line each. The split cohorts start
64 bytes into a line and occupy two lines each. Before LDGSTS, one predicated
scalar `.cg` load per copy requests all 16 aligned-cohort sectors and the first
eight split-cohort sectors. The scalar results stay live but are first consumed
only after the post-`wait_group 0` clock.

Consequently every cohort has the same early first response. Aligned cohorts
also end early; split cohorts retain a fresh second-line response at the later
LDGSTS ready time. The current shared-L2 pending-sector path reuses the earlier
ready time for matching sectors and preserves the later ready time for fresh
sectors. This is the independent response stream that the retained one-line
corpus could not create.

The paired variants retain identical relative source and preload vectors plus
the same four compact and four severe destination cohorts:

- `span_expensive`: split cohorts own severe destinations;
- `span_cheap`: aligned cohorts own severe destinations.

All four aligned cohorts and the first line of all four split cohorts are
preloaded. This matters: preloading only the split cohorts would trade stream
hiding against an equal prearrival term and make the paired residual tie.

For copy `i` with split response span `S_i`, schema 18 therefore predicts:

```text
span_expensive residual = 18 - min(18, S_i)
span_cheap residual     = 18
paired group score      = -min_i(min(18, S_i))
```

## Static and execution integrity

The final binary contains eight depth `1/2` by lead `0/4/8/16`
specializations. The static sidecar decodes 1,208/1,208 instructions with zero
unknown encodings. Every specialization has:

- exactly `D` predicated `LDG.E.STRONG.GPU` and `D`
  `LDGSTS.E.BYPASS.128` instructions;
- one LDGDEPBAR and one async `DEPBAR.LE SB0, 0x0`;
- no wait consuming the scalar preload barrier before the post-wait clock;
- 37--40 registers, 8 KiB shared memory, and zero stack/local/spill bytes; and
- its only result store after all measured intervals.

The implementation corrected one nomenclature error discovered by the first
replay: there are 20 unique destination lines per copy, but 32 payload/bypass
wavefront observations. The final probe reports `destination_lines_per_copy`
and the reducer checks the two counters separately.

## Fresh evidence

The retained root is:

```text
results/captures/sm86_async_multiline_response_20260831_v1/
```

It contains:

- 80 correct 8192-group direct pairs (ten AB/BA-balanced repeats for each
  depth/lead cell);
- 80 paired NCU bundles, 480 one-metric reports, and 960 ordered one-pass rows;
- 16 marker-scoped 64-group NVBit endpoint traces with complete payload and
  empty unsupported/unmapped/cross-allocation sidecars; and
- 16 schema-18 cells, each replayed twice for 32 byte-identical outputs.

The exact per-copy work is:

```text
scalar LDG instructions: 1
scalar preload sectors:  24 across 8 lines
LDGSTS instructions:     1
LDGSTS source sectors:   32 across 12 lines
local-device LTS lookups: 56
destination lines:       20
payload wavefronts:      32
```

At 64 groups this gives 64/128 scalar LDGs, 64/128 LDGSTS instructions,
and 3,584/7,168 exact local-device LTS lookup sectors at depths 1/2. Across all
80 bundles, paired hit deltas, miss deltas, and hit-plus-miss closure errors
are exactly zero. Raw device-wide DRAM varies from 2,080 to 43,428 sectors and
remains non-gating. This is consistent with NVIDIA's documented limitation
that asynchronous display/copy/video activity shares L2 and DRAM with the
profiled workload on a display-active GPU ([Nsight Compute Profiling
Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#asynchronous-gpu-activity)).

NVBit capture ran inside a transient user-systemd scope with `Delegate=yes` so
the retained capture helper could create and clean its required cgroup-v2
child containment. This changes no host configuration after the capture.

## Response-envelope result

All replayed lead settings exercise nonzero response streaming. The aggregate
envelope divided by copies is:

| Depth | Lead samples | Envelope cycles/copy |
|---:|---:|---:|
| 1 | 0 | 21.0 |
| 1 | 4 | 155.0 |
| 1 | 8 | 175.0 |
| 1 | 16 | 179.0 |
| 2 | 0 | 29.5 |
| 2 | 4 | 163.5 |
| 2 | 8 | 179.0 |
| 2 | 16 | 179.0 |

Prearrival hiding is zero in every cell. For `span_expensive`, four severe
split cohorts hide 18 cycles each (72 cycles/copy) and group residual exposure
is zero. For `span_cheap`, stream hiding is zero and group residual exposure is
18 cycles. Production fingerprints match within every pair, and the typed
visibility tail remains present.

The compiled lead-zero path already leaves a 21-cycle modeled interval between
the scalar request and LDGSTS, so all four identification points saturate the
18-cycle score. This is an observation about the emitted instruction schedule,
not a fitted correction.

## Frozen identification

The reducer admits the complete static, NCU, NVBit, and replay matrix before
opening any direct timing label. It then evaluates only depth 1:

| Lead | Live median delta | Schema-18 delta | Absolute error |
|---:|---:|---:|---:|
| 0 | -10.143 | -18.000 | 7.857 |
| 4 | -7.390 | -18.000 | 10.610 |
| 8 | -5.979 | -18.000 | 12.021 |
| 16 | -6.142 | -18.000 | 11.858 |

Every median issue delta is exactly zero and every wait-delta sign agrees with
the score. Sign agreement is insufficient: identification MAE is 10.587 and
maximum error is 12.021, versus frozen limits of 1.0 and 2.0 cycles/group.
Depth-2 direct artifacts exist for corpus completeness, but their timing fields
remain unopened by the reducer and no holdout score is emitted.

## Decision and next simulator seam

Schema 18 has now been exercised on independently completing source lines and
is rejected as a production timing candidate. Do not fit a scale, lower the
error bounds, promote its residual into wait readiness, or remove
`async_copy_shared_visibility`.

The live magnitude becomes smaller as the retained lead grows while the
current score is already saturated. A future continuation must distinguish
line-local destination preparation/service order from the scalar notion that
all severe topology work can hide behind any response interval. The smallest
useful next assay would place a scalar preload immediately adjacent to LDGSTS
or expose line-to-destination progress directly, preserving the same paired
marginals. It should not reuse these consumed depth-1 labels for tuning.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_multiline_execution.py \
  --phase static --timeout 900

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_multiline_execution.py \
  --phase direct --timeout 900

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_multiline_execution.py \
  --phase nvbit --timeout 900

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_multiline_execution.py \
  --phase replay --timeout 1800

CUDA_VISIBLE_DEVICES=0 LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_multiline_response.py \
  --root results/captures/sm86_async_multiline_response_20260831_v1 \
  --timeout 900

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_multiline_response.py \
  --root results/captures/sm86_async_multiline_response_20260831_v1 \
  --output results/captures/sm86_async_multiline_response_20260831_v1/analysis/report.json \
  --output-md results/captures/sm86_async_multiline_response_20260831_v1/analysis/report.md
```

Both capture helpers also provide `--smoke` and `--validate-only` paths.

## Content identities

- Plan commit: `26b84077`
- Probe binary: `e964ddf2a6fe9abdae8c88c894977b901d633e19ef2661ccfcfca8e33c114887`
- Release `sim-run`: `83b123b9a15e590b39a076ce631ac3df99b5bc49baf3c578e3a51c9e60506673`
- Static control sidecar: `1c42142dfb666c3600cb38805bc5fd4f2bd3fa7e2c613a61fd20d1c756f26212`
- Machine report: `a24d1d03436ec27dae91343754f04e0117add5670e64b7d07c9b31105a5f228a`

These identities reproduce the retained observation; they do not authorize a
timing promotion or act as workflow gates.
