# Clock/cache interaction interval envelope (2026-09-30)

`tools/sm86_interaction_interval_envelope.py` widens temporal-route residual
intervals using clock×cache interaction ratios calibrated on the opposite
workload family from prior low/high campaigns. It never changes point
predictions and reads temporal NCU values only for post-hoc coverage reporting.

Coverage improved from the unadjusted temporal intervals:

| phase | GPC coverage | duration coverage |
|---|---:|---:|
| cold | **86.7%** | **93.3%** |
| warm | **100.0%** | **93.3%** |

Median relative widths were approximately 49%/63% cold and 72%/88% warm
GPC/duration. These are conservative scenario envelopes, not confidence
intervals or cache ownership evidence.

