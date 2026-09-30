# Interaction interval scope correction (2026-09-30)

The first interval-envelope summaries pooled base and large rows after fitting
opposite-cohort interaction priors, so their coverage denominators included
calibration-cohort rows. That was a scope error in uncertainty reporting, not
in point predictions.

`tools/sm86_interaction_interval_envelope.py` now enforces a fold per requested
holdout cohort and scores only that cohort. Corrected temporal coverage is:

| phase | GPC | duration |
|---|---:|---:|
| cold | **86.7%** | **100.0%** |
| warm | **100.0%** | **86.7%** |

Corrected reports:

* `results/alpha/sm86_interaction_scope_corrected_20260930/cold.json`
* `results/alpha/sm86_interaction_scope_corrected_20260930/warm.json`

The intervals remain empirical scenario envelopes, not confidence intervals.

