# SM86 NCU-grounded simulator smoke report

Revision: `08c7c9dfb1d93181f5b303220a7ebcf2b380b8c4` on `GPUSim_main`.
GPU: `NVIDIA GeForce RTX 3090 Ti` (`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, CC 8.6, driver 610.57.04).
Simulator: `394640988c9d5c55322a3d06b562ff87cb9d5c7a2ec3aa8562901c6d36e087ae`.

## Primary accuracy and cost

| Workload | Launches | GPU kernel ms | NCU profile s | Simulator s | GPU / simulator | NCU / simulator | Terminal lower-bound MAPE | Issue-frontier MAPE | Counter MAPE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 HMMA GEMM snapshot | 1 | 9.952e-03 | 1.9570 | 0.5382 | 1.849e-05 | 3.636 | 33.32% | 22.94% | 0.79% |
| BF16 FlashAttention forward snapshot | 1 | 0.0119 | 1.9482 | 0.0450 | 2.650e-04 | 43.252 | 21.84% | 12.40% | 5.72% |
| BF16 LayerNorm forward snapshot | 1 | 0.0162 | 1.9427 | 0.2268 | 7.138e-05 | 8.564 | 17.75% | 10.67% | 0.31% |
| INT8 IMMA GEMM snapshot | 1 | 8.576e-03 | 1.8955 | 0.3764 | 2.279e-05 | 5.037 | 49.24% | 40.06% | 1.19% |
| **Total / launch-weighted** | 4 | 0.0467 | 7.7433 | 1.1865 | 3.932e-05 | 6.526 | 30.54% | 21.52% | 2.00% |

## Cache and traffic accuracy

| Workload | L1 read MAPE / pp | L1 unified MAPE / pp | L2 MAPE / pp | Instructions APE | Global load APE | Global store APE | DRAM read APE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 HMMA GEMM snapshot | 100.00% / 16.02 pp | 100.00% / 18.61 pp | 91.71% / 87.43 pp | 0.00% | 0.00% | 0.00% | 3.17% |
| BF16 FlashAttention forward snapshot | n/a / 0.00 pp | 100.00% / 9.52 pp | 79.14% / 67.77 pp | 0.00% | 0.00% | 0.00% | 22.89% |
| BF16 LayerNorm forward snapshot | 0.00% / 0.00 pp | 31.87% / 9.36 pp | 75.52% / 57.85 pp | 0.00% | 0.00% | 0.00% | 1.24% |
| INT8 IMMA GEMM snapshot | 12.48% / 7.13 pp | 18.34% / 10.37 pp | 84.68% / 80.94 pp | 0.00% | 0.00% | 0.00% | 4.74% |

## Caveats

- NCU is comparison-only ground truth and is absent from simulator commands and profile inputs.
- Cache rates are launch-global weighted rates derived from NCU hit/miss sector sums; direct per-subunit NCU .pct ratios are not used.
- Every smoke completion frontier is an explicit lower bound with retained unmodeled retirement tails.
- GPU/simulator and NCU/simulator values are named time ratios, not unqualified speedups.
- Smoke results contain one NCU repeat per workload; the final campaign requires three independent repeats.

Regenerate and validate:

```bash
.venv/bin/python workloads/research_probes/sm86_ncu_demo_benchmark/report.py --campaign /home/jakeshea/GPU_simulator_rebase/results/captures/sm86_ncu_demo_benchmark_20260901_smoke_v2/campaign_manifest.json
```
