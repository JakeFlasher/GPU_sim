# SM86 bypass destination-projection discriminator

**Status:** complete 2026-08-30; scalar projection falsified, no timing promotion

**Architecture:** `sm_86` / NVIDIA GeForce RTX 3090 Ti

**Evidence root:** `results/captures/sm86_async_bypass_projection_20260830_v1/`

**Machine report:** `analysis/report.json`

## Question and construction

The per-copy simulator state can now retain destination topology, but the
fixed-source pair left two incompatible observations: the cache-all trace
projection changed by 2× while the live `.cg` destination-wavefront counter
changed by 4×. This assay adds one `severe` layout to choose between them.

All three layouts retain the same 32-byte source lane stride, 1 KiB source
position stride, 4 KiB shared position stride, 32 KiB shared allocation,
depth 4, and one warp. Only destination lane stride changes:

- compact: 32 bytes;
- conflict: 64 bytes; and
- severe: 128 bytes.

The cache-all L1 route is the projection control. The `.cg` route repeats an
8 KiB source footprint so source requests are L2-resident after setup. The
target records the same pre-issue, post-issue, and post-wait clocks used by the
phase milestone.

Static evidence contains 36 functions, 3,072 decoded instructions, and zero
unknown encodings. Selected compact/conflict/severe kernels have equal
registers, shared allocation, instruction count, static stalls, and ordered
clock/store controls.

## Result

Nine 8,192-group direct runs were retained for each route/layout. Every issue
interval is exactly 39 cycles/group. The table reports median wait cycles and
paired drain deltas per group.

| route | compact wait | conflict wait | severe wait | compact→conflict | conflict→severe |
|---|---:|---:|---:|---:|---:|
| cache-all `.ca` | 83.045 | 147.041 | 276.034 | +63.996 | +128.992 |
| L2-hit `.cg` | 323.767 | 338.195 | 342.188 | +14.432 | +4.013 |

Every adjacent drain delta is positive in all nine repeats. One severe L2 run
contains a retained 49-cycle scheduler/system pause; the median and the other
eight runs remain in the same narrow band.

The NCU traffic profile keeps 256 LDGSTS instructions and 8,192 source sectors
in all six cells. Destination wavefronts are:

| route | compact | conflict | severe |
|---|---:|---:|---:|
| cache-all `.ca` | 2,048 | 4,096 | 8,192 |
| L2-hit `.cg` | 2,048 | 8,192 | 8,192 |

The cache-all control follows the trace source-cohort/bank projection and its
two-cycle wavefront timing almost exactly. Bypass does not. Conflict and severe
have the same live destination-wavefront count, yet severe adds a repeat-stable
four drain cycles/group. Conversely, the trace projection doubles between
those layouts while the live delta is far smaller than the cache-all law.

NCU ran with uncontrolled caches/clocks. The L2 conflict profile retains 128
LTS misses alongside 8,192 hits, while compact/severe report zero misses; this
small application-replay setup asymmetry is retained rather than removed. It
does not change the exact instruction/source-sector counts or the long-run
direct result.

## Decision

Neither available scalar is a sufficient bypass completion coordinate:

- importing the cache-all source-cohort wavefront projection overstates the
  conflict→severe effect; and
- using the live `.cg` destination-wavefront count predicts no
  conflict→severe effect at all.

No timing constant or tail removal is justified. Schema-14 per-copy state and
the `async_copy_shared_visibility` boundary remain. A future timing model needs
at least one additional mechanism-local topology dimension—such as
source-cohort to destination-line fanout kept separately from bank-word
multiplicity—and an event rule that overlaps each copy with its own source
response. This milestone stops here rather than fitting a nonlinear scalar to
three layouts.

The follow-on orthogonal skew layout and schema-15 result are recorded in
[the multidimensional bypass milestone](sm86_async_bypass_multidim_milestone.md).
They validate a trace-derived multidimensional proxy but reject its
latency-only timing candidate on untouched depths.

## Reproduction

```bash
make -C workloads/research_probes/async_source_only async_visibility_phase

CUDA_VISIBLE_DEVICES=0 \
  workloads/research_probes/async_source_only/async_visibility_phase \
  --groups 8192 --depth 4 --route l2 --layout severe --markers off

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_projection.py \
  --root results/captures/sm86_async_bypass_projection_20260830_v1 \
  --output results/captures/sm86_async_bypass_projection_20260830_v1/analysis/report.json
```

## Content identities

- Phase binary: `eb3ca81f7c266d75b4026f855ae2834782d97564b7ccef6699c904ffefc6c7b1`
- Static control sidecar: `20ebfb8c04a982da6b72463caab8a4b77dab55934212b5d0af3c1f36f93beefa`
- Machine report: `37f4a819ad0addf33199d91c57e694e04fa4cf8af359a8e5f30f74b8514031eb`
