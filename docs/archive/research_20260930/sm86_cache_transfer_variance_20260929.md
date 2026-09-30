# SM86 warm/cold transfer variance

The fresh full campaigns expose large workload-specific cache-phase transfer
ratios. These ratios are independent score observations, not model features.

| workload | duration warm/cold | GPC warm/cold |
|---|---:|---:|
| conv2d implicit GEMM | 0.938 | 0.936 |
| cross entropy | 0.813 | 0.815 |
| flash backward | 0.877 | 0.877 |
| GEMM HMMA FP16 | 0.806 | 0.804 |
| RMSNorm backward | 0.763 | 0.765 |
| softmax | 0.781 | 0.780 |
| SwiGLU | 1.012 | 1.009 |

Base-family calibration ratios span roughly 0.689–0.943. A single median
transfer factor is therefore a conservative fallback, not a universal cache
law. The promoted model avoids applying that factor to the independently
paired flash archetype; the paired shape matrix supplies its local warm
response. Unmatched cache phases remain interval/unknown states.
