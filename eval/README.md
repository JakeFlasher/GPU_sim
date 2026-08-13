# Simulator evaluation

Use `eval.direct_score` for ordinary prediction-versus-observation comparison.
It accepts plain JSON objects, joins shared numeric metric names, and reports
coverage plus MAE, RMSE, and MAPE where defined.

```bash
.venv/bin/python -m eval.direct_score \
  --observed results/my_run/observed.json \
  --predicted results/my_run/predicted.json
```

Run its focused tests with:

```bash
.venv/bin/python -m pytest -q eval/tests/test_direct_score.py
```

Use `eval.profile` as an optional plain Linux `perf` wrapper:

```bash
.venv/bin/python -m eval.profile --output results/perf -- \
  sim/target/release/sim-run --help
```

The older `score`, `production_*`, `contracts`, `guards`, and `bench` modules
are retained only for historical campaign compatibility. Their sealed
schemas, receipts, authority fields, and fixed-fixture lanes are not required
for direct simulator development or evaluation. Use them only through the
opt-in `run-sol-campaign` skill when an existing campaign artifact specifically
needs that format.

NCU observations are evaluation targets, never same-cell simulator features.
