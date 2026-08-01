# Evaluator MVP synthetic fixture

This directory is test-only. `matched_labels.v1.json` is sealed and its exact
file identity is bound by `synthetic_matched_label_authority.v1.json`; the latter
is independently pinned in `eval/labels.py`. The evaluator build identity is
separately sealed and pinned in `eval/score.py`. `sim_result.v2.json` is an
independently sealed simulator-contract fixture.

All identities and values are synthetic. They are not admitted hardware evidence,
a production label authority, an actual evaluator build receipt, or an accuracy
claim.
