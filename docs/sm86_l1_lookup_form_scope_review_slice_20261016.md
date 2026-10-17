# SM86 L1 lookup form-scope review — H87 — 2026-10-16

## Corpus strata

Using the H83 full-first traces and independent NCU joins, the H79
packet-aware candidate was stratified by static-SASS `memory_scope` form. The
largest stratum is `global_load_constant` (88 identity rows): hit APE improves
30.86%→16.95% and miss APE 31.34%→17.53%. Plain `global_load_ldg` rows are
sparse (two rows); their hit APE changes 0.66%→1.32% while miss APE improves
26.25%→12.50%. LDGSTS-only rows do not improve the miss endpoint.

The four `global_load_ltc128b` rows are the adversarial boundary: miss APE
worsens 725.01%→1,683.10% under the unqualified candidate. Mixed
`global_load_constant+ldgsts_async_copy` and other mixed forms have distinct
behavior as well. The gross LTC outlier is therefore not safe to hide by
renaming it or excluding it from raw authority.

## Decision

Keep H79's candidate/interval observer and baseline selected prediction. A
future form-qualified arm may be considered only after the producer exposes a
complete per-request client/form identity; static aggregate scope labels alone
cannot decide whether same-request merge subtraction applies. No L1 timing,
cache-state, L2, DRAM, or IPC behavior changed in this review.
