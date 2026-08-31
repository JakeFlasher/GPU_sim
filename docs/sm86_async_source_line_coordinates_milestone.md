# SM86 source-line ownership coordinate-corrected campaign

## Outcome

The fresh two-orientation campaign is complete as:

```text
complete_line_coordinates_identification_no_go
```

All structural, identity, NCU, per-lane trace, deterministic replay, and
production-tail gates pass. The first-line-early identification issue delta is
exactly zero in all ten paired repeats. Its `early_fanout - late_fanout` wait
delta is positive in all ten repeats, with median `+9.43548583984375`
cycles/group. The frozen signal required a median at most `-1.0` and at least
eight negative repeats, so identification fails before any holdout timing
field is opened.

The tracked result is
[sm86_async_line_coordinates_report.json](../data/local_labels/sm86_async_line_coordinates_report.json),
with content digest
`3e171f8b5b8a56bec54a36215dbcf811f59e5e1c6c3a31bedd77971d6eb55cd4`.
The retained executable digest is
`c2c121f725c73725decb655ac006db65912c0dd3e8872786ca3858eb7599a36b`.

No schema, simulator timing, readiness, architecture card, or tail state
changes. `async_copy_shared_visibility` remains required.

## Separate coordinates

The campaign freezes and validates four different interfaces without
collapsing them:

| Coordinate | Per copy | 64-group total |
|---|---:|---:|
| NCU shared LDGSTS wavefronts | 20 | 1,280 |
| Simulator destination wavefronts | 24 | 1,536 |
| Trace bypass-wavefront projection | 32 | 2,048 |
| Unique destination lines | 20 | 1,280 |

The smoke also validates 64 scalar global-load instructions, 64 LDGSTS
instructions, and 3,584 local-device LTS sectors in each ordered NCU row.
LTS hit and miss populations are positive, close exactly to total, and remain
within the frozen paired tolerance. Raw DRAM is retained but non-gating.

Both NVBit variants in each orientation contain exactly 64 scalar-preload and
64 LDGSTS records. Their active masks, source vectors, and destination vectors
match the frozen lane ownership permutation. All four simulator cells pass the
coordinate gates above; the complete 43-field schema-18 observation and the
production fingerprint tie exactly within each orientation.

## Identification decision

The admitted first-line-early values, in repeat order, are:

```text
issue delta: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
wait delta:  9.1650390625, 9.7314453125, 8.803955078125,
             9.59423828125, 9.3243408203125, 10.42431640625,
             9.546630859375, 9.2191162109375, 8.692626953125,
             9.621337890625
```

The issue gate passes. The wait gate has zero negative repeats and a positive
median, so the prescribed negative source-line-ownership effect is absent.
This observation does not justify reversing the criterion, reading the
holdout, fitting a magnitude, or promoting a line-local timing term.

The final label-read state is:

```text
identification_issue_read = true
identification_wait_read  = true
holdout_issue_read        = false
holdout_wait_read         = false
```

## Evidence identity and scope

The canonical root is:

```text
results/captures/sm86_async_line_coordinates_20260831_v1/
```

It contains 20 correct 8,192-group direct pairs, 20 paired NCU bundles (140
one-pass reports and 280 ordered rows), four NVBit endpoint traces, and eight
byte-identical replay outputs. The run was pinned to device index 0,
`NVIDIA GeForce RTX 3090 Ti`, UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6, driver
610.57.04, and Nsight Compute 2026.1.0.0.

The root-level campaign contract, fresh binary, assay ID, and evidence root
are bound into direct, NVBit, simulator-build, replay, and NCU records. A
separate immutable smoke authorization remains under
`analysis/smoke-authorization.json`, so the final tracked decision does not
erase the evidence that authorized the full run. Predecessor result paths are
forbidden by the fresh contract and were not used as inputs.

Direct result files are provenance-validated before labels. The analyzer then
selectively decodes issue fields; it lexically skips wait values until the
issue gate passes. Read flags are set only after a complete successful field
projection.

## Reproduction

Start from an empty canonical root and build/capture the smoke:

```bash
CUDA_VISIBLE_DEVICES=0 LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python \
  workloads/research_probes/async_source_only/capture_line_coordinates_execution.py \
  --smoke --phase all --timeout 1800

CUDA_VISIBLE_DEVICES=0 LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python \
  workloads/research_probes/async_source_only/capture_line_coordinates_ncu.py \
  --smoke --timeout 900

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  workloads/research_probes/async_source_only/analyze_line_coordinates.py \
  --smoke-only
```

Only the identity-matched smoke-pass authorization permits the full matrix:

```bash
CUDA_VISIBLE_DEVICES=0 LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python \
  workloads/research_probes/async_source_only/capture_line_coordinates_execution.py \
  --phase all --timeout 1800

CUDA_VISIBLE_DEVICES=0 LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python \
  workloads/research_probes/async_source_only/capture_line_coordinates_ncu.py \
  --timeout 900

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  workloads/research_probes/async_source_only/analyze_line_coordinates.py
```

Missing, malformed, stale, contract-mismatched, predecessor-root,
manifest-invalid, multi-pass, coordinate-mismatched, non-deterministic, or
inconsistent-delta evidence exits nonzero. A scientific no-go exits zero.
