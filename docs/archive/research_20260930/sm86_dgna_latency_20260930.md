# SM86 pointer-chase latency sweep — RTX 3090 Ti

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / GA102 / SM86, driver 615.71.09,
1350/10501 MHz.

A Sattolo-cycle pointer-chase kernel (one active thread, dependent
ld.global.ca.u32 loads) swept working sets from 4 KiB to 512 MiB. Ten
post-warmup measurements were collected per point. The source is retained
under the result directory; it was adapted from a public Ampere memory-model
research suite and run as a performance observation.

Representative medians:

| working set | cycles/hop |
|---:|---:|
| 4--64 KiB | 38.97--48.58 |
| 80--256 KiB | 57.33--199.60 |
| 512 KiB | 217.69 |
| 2--2.8 MiB | 228.95--229.66 |
| 3 MiB point | 246.89 (isolated outlier) |
| 4--8 MiB | 230.19--230.77 |
| 16--32 MiB | 231.10--231.26 |
| 64 MiB | 236.43 |
| 128--512 MiB | 238.80--240.55 |

The curve supplies a local latency envelope with an L1-like rise in the
tens-to-hundreds KiB range, an approximately 218--231-cycle L2-like region,
and a modest 236--241-cycle large-working-set tail. It is not a direct cache
owner/read-from/DRAM-completion witness: one warp, address translation,
partition placement, replacement state, and the exact lower-level source are
not separately observed. The 3-MiB outlier is retained rather than smoothed
into a boundary.

This strengthens the cache-phase/reuse model and provides target-card latency
priors. It does not close the native cache blocker.

Artifacts: results/alpha/sm86_dgna_latency_goal_20260930/.

Methodology reference: DGNA, arXiv:2607.19922, which applies pointer-chase
latency distributions to A100/H100 NUMA inference
(<https://arxiv.org/abs/2607.19922>); its A100/H100 conclusions are not
transferred as GA102 facts.
