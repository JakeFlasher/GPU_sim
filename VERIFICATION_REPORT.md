# Representative real-trace verification

The authoritative campaign result is [`docs/RESULTS.md`](docs/RESULTS.md);
this file preserves the compact root-level smoke report.

These smoke comparisons use the tracked real traces under
`verification_inputs/` and the retained baseline result trees. They are a
quick path check; the complete-corpus result is in
`FULL_NUMERIC_COMPARISON.json`.

| Case | Architecture | Status | Launches | Core counters |
| --- | --- | --- | ---: | --- |
| SM86 cp.async | `sm_86` | `ok` | 1 | equal |
| SM120 TMA | `sm_120` | `ok` | 1 | equal |
| SM86 scalar | `sm_86` | `ok` | 1 | equal |
| SM120 scalar | `sm_120` | `ok` | 1 | equal |

Compared fields include instruction/record counts, front-end counters, L1/L2
core counters, DRAM read sectors, timing disposition, and metric-prediction
count. NCU data is not supplied to replay. Full replay remains the acceptance
path; these four rows are not a substitute for the 339/356-capture audit.
