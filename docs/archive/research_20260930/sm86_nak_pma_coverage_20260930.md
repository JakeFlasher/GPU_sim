# Direct NAK PMA coverage model (2026-09-30)

`tools/sm86_nak_pma_coverage_model.py` summarizes direct PMA record density for
three independently generated NAK bodies:

| body | static instruction words | PMA records | records/instruction |
|---|---:|---:|---:|
| short retained body | 10 | 52 | 5.20 |
| simple long body | 20 | 416 | 20.80 |
| data-dependent 64-round body | 1,330 | 1,560 | 1.17 |

The result shows why short PMA bodies cannot be used as route witnesses and why
longer bodies improve coverage without yielding a universal sampling law. Raw
record diversity and body residency are useful uncertainty covariates, but the
records still lack physical pipe, sub-unit, issue-slot, and exact dynamic
issue-cycle fields.

Report: `results/alpha/sm86_nak_pma_coverage_goal_20260930/coverage.json`.
No tests, hashing, or verification workflow was run.
