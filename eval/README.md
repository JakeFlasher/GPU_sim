# Evaluation

`eval/direct_score.py` is the small standalone comparator for simulator and
measurement JSON. It accepts arrays or keyed numeric objects and reports MAE,
RMSE, MAPE when defined, and numeric coverage.

```bash
python -m eval.direct_score --observed observed.json --predicted predicted.json
```

The simulator remains independent of the evaluator: comparison data are read
after replay and never fed back into architecture cards or model parameters.
