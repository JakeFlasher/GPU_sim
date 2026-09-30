# SM86 cache reuse-signature endpoint model (2026-09-29)

## Motivation

The fresh full campaign shows that a single warm/cold median is too coarse:
large-workload duration ratios span about `0.763--1.012`, and GPC ratios span
about `0.765--1.009`.  The public cache evidence is aggregate, but it is still
useful to model phase response explicitly.  This model therefore separates:

1. an independent cold endpoint shape response, and
2. an independent warm/cold endpoint ratio shape response.

The warm prediction is their product.  In parallel it interpolates aggregate
DRAM-read, DRAM-write, L2-hit, and L2-miss sectors to expose a cache signature.
Those sectors are diagnostics, not ownership/read-from/DRAM-completion proofs.

The executable is `tools/sm86_cache_reuse_signature.py`.  It uses paired
independent cold/warm rows for exact shape matches.  Unmatched or out-of-hull
rows receive an independent frontier/trace fallback with a wide interval and
`candidate_fallback` provenance, so no workload is dropped or labeled OOD.
It never uses a scored target's NCU value in the fit.

## SOTA connection

PPT-GPU's reuse-profile work treats locality as an architecture-independent
intermediate before applying a cache model; TileSight lifts reuse distance to a
tile execution plan and keeps cache traffic inside the same tile envelope.
The new model adopts only that separation principle.  It does not assume the
NVIDIA L2 replacement policy, physical line owner, read-from source, or dirty
writeback behavior.  This is consistent with the cache-state separation in
public GPGPU-Sim/Accel-Sim and Vortex source, where valid/dirty tags, MSHRs,
fills, and writebacks are simulator state rather than a silicon witness.

References:

* [PPT-GPU repository](https://github.com/NMSU-PEARL/PPT-GPU) and
  [reuse-profile paper](https://doi.org/10.1145/3291051)
* [TileSight, arXiv:2607.22432](https://arxiv.org/abs/2607.22432)
* [GPGPU-Sim shader source](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/dev/src/gpgpu-sim/shader.cc)

## Fresh score-only result

Using the independent cache-phase matrix at
`results/alpha/sm86_flash_bwd_cache_phase_goal_20260929/cache_phase_matrix.csv`
and the fresh full campaigns:

| phase | matched rows | GPC APE | duration APE | model time | replay wall | speedup |
|---|---:|---:|---:|---:|---:|---:|
| cold | 15/15 covered; 1 exact + 14 fallback | 3.107% | 3.045% | ~0.01 s/family | ~0.666 s total | coverage path |
| warm | 15/15 covered; 1 exact + 14 fallback | 8.415% | 8.476% | ~0.01 s/family | ~0.666 s total | coverage path |

Reports:

* `results/alpha/sm86_cache_reuse_signature_goal_20260929/cold_large.json`
* `results/alpha/sm86_cache_reuse_signature_goal_20260929/warm_large.json`
* `results/alpha/sm86_cache_reuse_signature_goal_20260929/coverage_summary.json`

The matching `cold_base.json` and `warm_base.json` reports cover all eight
base rows. Exact-shape flash APE remains 0.422%/0.419% cold and 0.189%/0.164%
warm; fallback rows are explicitly identified rather than presented as
shape-exact cache predictions.

For the fresh warm flash row, the ratio model predicts `175,424` GPC cycles /
`129,972` ns versus `175,094` / `129,760` NCU.  This is lower-error than the
direct paired-warm quadratic for this fresh row (`0.202%` / `0.199%`) on GPC /
duration, but the exact cache coverage is one shape archetype and the model is
not promoted.  The interpolated target cache signature predicts about `3,648` cold
versus `100` warm DRAM-read sectors and `2,176` versus `26` L2-miss sectors;
the score-only target observation is `2,940` versus `88` and `2,176` versus
`0`.  The residual difference is retained as cache-state uncertainty, not
silently forced to zero.

## Boundary impact

* **Cache blocker:** replaces a global transfer scalar with a local phase
  response and explicit aggregate-sector signature.
* **Timing blocker:** predicts duration and GPC endpoints separately; their
  equivalence remains unproved.
* **Pipe blocker:** no physical pipe identity is inferred; cache response is a
  phase envelope only.
* **ABI blocker:** no CUDA metadata or control-word generation is attempted;
  ABI effects remain an endpoint residual and are not folded into ownership.

More shape pairs can replace fallback rows and narrow intervals.  A new
external line-owner/read-from oracle is still required for mechanism-level
cache closure.
