# SM86 asynchronous-copy multi-line response-envelope milestone

## Outcome

The pending-line mechanism is observable, but the assay does not satisfy its
frozen topology contract. The retained result is:

```text
complete_response_envelope_structural_no_go
```

The plan froze `20 * groups * depth` destination wavefronts. The prescribed
32-lane geometry instead produces 32 payload/bypass wavefront observations and
20 unique destination lines per copy. The first reducer incorrectly treated
that as a terminology correction and allowed the full matrix to run. The RLCR
review correctly rejected that reinterpretation: the plan was immutable, so
the depth-1/lead-16 smoke must fail its exact-topology gate and stop before
timing identification.

The aggregate machine-readable handoff is tracked at
[sm86_async_multiline_response_envelope_report.json](../data/local_labels/sm86_async_multiline_response_envelope_report.json).
Its content digest is
`4327f2cdd489c67c997d1e9fc65761d0a3da6a91ab447f80798bd95070e932a8`.
Large raw profiler and trace artifacts remain outside Git under the retained
capture root.

No simulator timing, card, completion frontier, scheduler state, traffic, or
tail flag changes. `async_copy_shared_visibility` remains required.

## Mechanism observed

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
LDGSTS ready time. The shared-L2 pending-sector path reuses the earlier ready
time for matching sectors and preserves the later ready time for fresh
sectors. This is the independent response stream that the retained one-line
corpus could not create.

The paired variants retain identical relative source/preload vectors and the
same four compact plus four severe destination cohorts:

- `span_expensive`: split cohorts own severe destinations;
- `span_cheap`: aligned cohorts own severe destinations.

For copy `i` with split response span `S_i`, the observational schema-18
residual would be:

```text
span_expensive residual = 18 - min(18, S_i)
span_cheap residual     = 18
paired group score      = -min_i(min(18, S_i))
```

That formula is not evaluated as a timing candidate because the earlier exact
topology gate fails.

## Mandatory vertical smoke

The retained root is:

```text
results/captures/sm86_async_multiline_response_20260831_v1/
```

Its admitted surface contains only the frozen depth-1/lead-16 vertical slice:

- eight statically decoded depth `1/2` by lead `0/4/8/16`
  specializations;
- one correct 8192-group direct AB pair with exact zero issue delta;
- one paired six-report/12-row, one-pass NCU bundle at 64 groups;
- two marker-scoped 64-group NVBit endpoint traces; and
- two byte-identical replay outputs per endpoint.

Every local execution retains a validated run manifest containing command,
configured timeout, start/finish timestamps, return code, timeout state,
stdout/stderr identities, and relevant binary/input/output identities. Static
evidence additionally binds the CUDA source, Makefile, control extractor,
NVCC, cuobjdump, and their versions to the audited binary. A changed source or
tool input makes retained static evidence stale instead of silently accepting
the old binary.

The exact source-side smoke work passes:

```text
scalar LDG instructions: 64
scalar preload sectors:  1,536 across 512 lines
LDGSTS instructions:     64
LDGSTS source sectors:   2,048 across 768 lines
local-device LTS lookups: 3,584
```

Both NCU rows have exact instruction and attributable LTS totals. Paired LTS
hit/miss deltas and hit-plus-miss closure error are zero. Raw device-wide DRAM
remains a non-gating background sentinel, consistent with NVIDIA's documented
limitation that display/copy/video activity shares L2 and DRAM with the
profiled workload ([Nsight Compute Profiling
Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#asynchronous-gpu-activity)).

The trace/replay mechanism observations are also internally exact:

- 179 response-envelope cycles per copy;
- zero prearrival hiding;
- 72 stream-hidden cycles per `span_expensive` copy;
- residual exposure zero for `span_expensive` and 18/group for `span_cheap`;
- 32 payload/bypass wavefront observations and 20 destination lines per copy;
  and
- unchanged production fingerprints with the typed visibility tail present.

The final bullet is precisely why the smoke is a structural no-go: the plan
requires 20 destination wavefront observations, not 20 destination lines.

## Excluded full capture

Before the topology-contract error was caught, a complete depth `1/2`, lead
`0/4/8/16` matrix was physically captured. It is retained separately as
round-1 diagnostic material, but it is not admitted by the milestone reducer,
not linked as the machine result, and not used to make a timing claim. In
particular, depth-1 identification and depth-2 holdout are not part of the
final decision.

This distinction is scientific rather than procedural: a downstream timing
comparison cannot repair an upstream mismatch in the workload contract.

## Decision and next simulator seam

Do not promote schema 18 into wait readiness, fit a scale, lower the error
bounds, reinterpret destination lines as wavefronts, or remove
`async_copy_shared_visibility`.

A successor may freeze the actual 32-payload-wavefront/20-line interface and
ask a new question, or change the CUDA geometry so the existing 20-wavefront
contract is literally satisfied. Either requires a new prospective plan and a
fresh smoke. The present depth-1 timing label must not be reused for tuning.

## Reproduction

Capture only the mandatory smoke first:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_multiline_execution.py \
  --smoke --phase all --timeout 1800

CUDA_VISIBLE_DEVICES=0 LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_multiline_response.py \
  --smoke --timeout 900

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_multiline_response.py \
  --smoke-only \
  --root results/captures/sm86_async_multiline_response_20260831_v1 \
  --output results/captures/sm86_async_multiline_response_20260831_v1/analysis/report.json \
  --output-md results/captures/sm86_async_multiline_response_20260831_v1/analysis/report.md \
  --aggregate-output data/local_labels/sm86_async_multiline_response_envelope_report.json
```

The reducer exits zero with
`complete_response_envelope_structural_no_go`. Missing, malformed, stale,
mixed-tracer, or execution-manifest evidence exits nonzero. Because the smoke
is a scientific no-go, the full-matrix commands must not run for this plan.

Aggregate content identities are carried in the tracked report; exact commands,
timestamps, status, timeout, and stream/input/output hashes remain in the raw
run manifests. They reproduce the observation and do not authorize a timing
promotion or act as workflow gates.
