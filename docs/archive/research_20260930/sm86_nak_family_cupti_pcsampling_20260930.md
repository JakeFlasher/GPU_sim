# CUPTI PC sampling matrix for NAK operation families (2026-09-30)

Six bounded NAK-to-CUDA-ELF operation-family cubins were each launched 128
times under serialized NVIDIA CUPTI PC sampling. Every family produced zero
dropped samples:

| family | PC records | samples | decoded PCs | long-scoreboard fraction | IMC-miss fraction |
|---|---:|---:|---:|---:|---:|
| mul | 694 | 2,344 | 17 | 48.1% | 38.8% |
| fma-like | 807 | 2,381 | 21 | 46.5% | 37.6% |
| reciprocal | 928 | 2,548 | 21 | 44.0% | 34.9% |
| shift | 696 | 2,326 | 17 | 48.1% | 38.4% |
| xor | 694 | 2,301 | 17 | 48.1% | 38.3% |
| maximum | 701 | 2,302 | 17 | 48.3% | 38.3% |

The reciprocal family adds a visible MIO-throttle/short-scoreboard population;
the other bodies have closely related aggregate distributions. These are
sample-state populations, not execution-time fractions or physical-pipe
assignments. The NAK-to-CUDA-ELF metadata remains template-derived.

Artifacts: `results/alpha/sm86_nak_family_cupti_pcsampling_goal_20260930/`.
No tests, hashing, or verification workflow was run.
