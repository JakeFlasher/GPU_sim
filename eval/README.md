# Evaluator prototype

The offline P2 evaluator consumes the frozen exact 7+1 metric projection and
strict `sim_result.v2` artifacts. Seven secondary metrics are scorable;
`exec_cycles_gpc_max` remains `required_unimplemented` on
`aggregation_semantics_study` and is never numerically scored.

Matched labels require authority independent of the label document. No
production matched-label/crosswalk authority is installed, so the production
loader fails closed with `LABEL_ADMISSION_AUTHORITY_UNAVAILABLE` and does not
open a label payload. The only active scoring lane uses checked-in, code-pinned
synthetic fixtures under `eval/tests/fixtures/evaluator_mvp/`. Those fixtures
confer no hardware, accuracy, or publication credit and do not weaken the
refusing `eval.production_evaluator` boundary. The evaluator build identity is a
separate code-pinned synthetic input; the scoring process never invokes Git and
has no ambient provenance fallback.

Run the offline canary from the repository root:

```bash
.venv/bin/python -m eval.score \
  evaluator-mvp-canary-v1 \
  eval/tests/fixtures/evaluator_mvp/sim_result.v2.json \
  eval/tests/fixtures/evaluator_mvp/matched_labels.v1.json \
  eval/tests/fixtures/evaluator_mvp/synthetic_matched_label_authority.v1.json \
  eval/tests/fixtures/evaluator_mvp/synthetic_evaluator_build_identity.v1.json
```

Success is one canonical sealed `accuracy_report.v1` line on stdout with empty
stderr. A typed CLI/input failure is one canonical error line on stderr with
empty stdout. No real label path is accepted by this fixture-only entry point.
