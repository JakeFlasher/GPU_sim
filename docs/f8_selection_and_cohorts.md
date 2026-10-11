# F8 — informative experiments and cohort accounting

**Selector:** `/dev/shm/sm120_rtx_5090_work/f8_experiment_ranking.json`  
**Cohorts:** `/dev/shm/sm120_rtx_5090_work/f8_cohorts.json`

The selector ranks 29 launch-local order-permutation probes by logical
scenario disagreement divided by estimated collection cost. It consumes only
direct trace features; NCU targets are not read. The cohort manifest groups
the 12 retained captures into seven workload families as development data and
leaves mechanism-calibration/future-evaluation populations explicitly empty.
No retrospective holdout or generalization claim is made from this corpus.
