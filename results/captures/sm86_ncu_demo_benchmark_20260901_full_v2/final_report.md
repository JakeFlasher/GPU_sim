# SM86 NCU-grounded simulator final report

Revision: `41b871a5799799df51f5e85278206cc9dd5af022` on `GPUSim_main`.
GPU: `NVIDIA GeForce RTX 3090 Ti` (`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, CC 8.6, driver 610.57.04).
Simulator: `394640988c9d5c55322a3d06b562ff87cb9d5c7a2ec3aa8562901c6d36e087ae`.

## Primary accuracy and cost

| Workload | Launches | GPU kernel ms | NCU profile s | Simulator s | GPU / simulator | NCU / simulator | Terminal lower-bound MAPE | Issue-frontier MAPE | Counter MAPE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 HMMA GEMM snapshot | 1 | 0.0100 | 1.9277 | 0.5413 | 1.856e-05 | 3.561 | 33.86% | 23.73% | 0.51% |
| BF16 FlashAttention forward snapshot | 1 | 0.0118 | 1.8758 | 0.0414 | 2.855e-04 | 45.358 | 20.90% | 11.35% | 4.95% |
| BF16 LayerNorm forward snapshot | 1 | 0.0164 | 1.9644 | 0.2297 | 7.119e-05 | 8.552 | 18.43% | 11.33% | 0.31% |
| INT8 IMMA GEMM snapshot | 1 | 8.640e-03 | 1.9257 | 0.3771 | 2.291e-05 | 5.107 | 49.48% | 40.33% | 1.01% |
| BF16 embedding gather snapshot | 1 | 9.824e-03 | 1.9122 | 0.0272 | 3.607e-04 | 70.213 | 26.84% | 15.93% | 2.57% |
| BF16 2:4 sparse GEMM snapshot | 1 | 4.320e-03 | 1.9037 | 0.0480 | 8.999e-05 | 39.657 | 59.00% | 40.86% | 2.93% |
| BF16 SiLU forward snapshot | 1 | 4.288e-03 | 1.9519 | 0.1945 | 2.205e-05 | 10.037 | 53.88% | 33.60% | 0.25% |
| BF16 RMSNorm forward snapshot | 1 | 0.0144 | 1.9298 | 0.1515 | 9.505e-05 | 12.738 | 20.30% | 12.30% | 0.38% |
| **Total / launch-weighted** | 8 | 0.0797 | 15.3911 | 1.6106 | 4.947e-05 | 9.556 | 35.34% | 23.68% | 1.61% |

NCU profiler campaign wall across retained repeats: 46.1734 s. The table shows the mean capture-process wall per workload profile.

## NCU repeatability

| Workload | Repeats | Maximum relative range | Metric |
| :--- | ---: | ---: | :--- |
| BF16 HMMA GEMM snapshot | 3 | 9.34% | l1_read_hit_rate |
| BF16 FlashAttention forward snapshot | 3 | 4.93% | dram_read_sectors |
| BF16 LayerNorm forward snapshot | 3 | 2.92% | sm_active_cycles |
| INT8 IMMA GEMM snapshot | 3 | 1.98% | l1_read_hit_rate |
| BF16 embedding gather snapshot | 3 | 2.19% | l1_unified_hit_rate |
| BF16 2:4 sparse GEMM snapshot | 3 | 4.12% | dram_read_sectors |
| BF16 SiLU forward snapshot | 3 | 6.44% | sm_active_cycles |
| BF16 RMSNorm forward snapshot | 3 | 61.43% | dram_read_sectors |

## Cache and traffic accuracy

| Workload | L1 read MAPE / pp | L1 unified MAPE / pp | L2 MAPE / pp | Instructions APE | Global load APE | Global store APE | DRAM read APE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 HMMA GEMM snapshot | 100.00% / 16.60 pp | 100.00% / 19.17 pp | 91.63% / 87.29 pp | 0.00% | 0.00% | 0.00% | 2.06% |
| BF16 FlashAttention forward snapshot | n/a / 0.00 pp | 100.00% / 9.52 pp | 79.31% / 68.46 pp | 0.00% | 0.00% | 0.00% | 19.80% |
| BF16 LayerNorm forward snapshot | 0.00% / 0.00 pp | 31.74% / 9.30 pp | 75.50% / 57.77 pp | 0.00% | 0.00% | 0.00% | 1.24% |
| INT8 IMMA GEMM snapshot | 13.27% / 7.65 pp | 19.01% / 10.83 pp | 84.58% / 80.84 pp | 0.00% | 0.00% | 0.00% | 4.03% |
| BF16 embedding gather snapshot | n/a / 0.00 pp | 100.00% / 24.42 pp | 49.74% / 36.96 pp | 0.00% | 0.00% | 0.00% | 10.26% |
| BF16 2:4 sparse GEMM snapshot | n/a / 0.00 pp | 100.00% / 10.00 pp | 81.06% / 74.88 pp | 0.00% | 0.00% | 0.00% | 11.72% |
| BF16 SiLU forward snapshot | n/a / 0.00 pp | 100.00% / 19.65 pp | 32.11% / 17.74 pp | 0.00% | 0.00% | 0.00% | 0.99% |
| BF16 RMSNorm forward snapshot | 2.132e-14% / 7.105e-15 pp | 31.18% / 11.33 pp | 63.49% / 43.48 pp | 0.00% | 0.00% | 0.00% | 1.54% |

## Caveats

- NCU is comparison-only ground truth and is absent from simulator commands and profile inputs.
- Cache rates are launch-global weighted rates derived from NCU hit/miss sector sums; direct per-subunit NCU .pct ratios are not used.
- Every final completion frontier is an explicit lower bound with retained unmodeled retirement tails.
- GPU/simulator and NCU/simulator values are named time ratios, not unqualified speedups.
- Final results contain three independent application-replay NCU repeats per workload; raw repeat values remain in JSON.

Regenerate and validate:

```bash
.venv/bin/python workloads/research_probes/sm86_ncu_demo_benchmark/report.py --campaign results/captures/sm86_ncu_demo_benchmark_20260901_full_v2/campaign_manifest.json
```
