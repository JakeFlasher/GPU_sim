"""Schema tests for the cache-replay throughput benchmark surface.

Covers four cases against the closed draft-2020-12 schema at
``schemas/cache_replay_benchmark.schema.json``:

(a) POSITIVE: the committed benchmark artifact validates with zero errors.
    (Its ``passed`` flag may be either true or false; this test only validates
    the schema and does NOT assert the gate outcome.)

(b) NEGATIVE: a gating copy (gating == true) that claims to have benchmarked
    only 3 cells (n_cells_benchmarked == 3, three ``per_cell`` entries) FAILS
    validation, because a gating run must cover the full cohort.

(c) NEGATIVE: a copy whose ``nvbit_trace_collection_time`` is a NUMBER FAILS
    validation. That field is a string sentinel: trace collection time is
    reported separately and must never be folded into the replay rate.

(d) ADMITTED: a non-gating subset benchmark (gating == false,
    n_cells_benchmarked == 3) validates, because the conditional constraint
    only binds gating runs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_SCHEMA_PATH = _HERE / "schemas" / "cache_replay_benchmark.schema.json"
_ARTIFACT_PATH = _REPO_ROOT / "data" / "local_labels" / "cache_replay_benchmark.json"


def _validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text())
    # The schema itself must be a valid draft-2020-12 schema.
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _committed_artifact() -> dict:
    return json.loads(_ARTIFACT_PATH.read_text())


def test_committed_benchmark_validates_with_zero_errors():
    """(a) POSITIVE: the committed benchmark artifact is schema-valid."""
    validator = _validator()
    artifact = _committed_artifact()
    errors = list(validator.iter_errors(artifact))
    assert errors == [], "\n".join(str(error) for error in errors)


def test_committed_artifact_is_a_gating_full_cohort_run():
    """Sanity: the committed artifact is itself a gating run over 30 cells, so
    case (a) genuinely exercises the conditional ``then`` branch."""
    artifact = _committed_artifact()
    assert artifact["gating"] is True
    assert artifact["cohort_size"] == 30
    assert artifact["n_cells_benchmarked"] == 30
    assert len(artifact["per_cell"]) == 30
    # The trace collection time is a string sentinel, never a number.
    assert isinstance(artifact["nvbit_trace_collection_time"], str)


def test_gating_run_with_three_cells_fails():
    """(b) NEGATIVE: a gating run that benchmarks only 3 cells must FAIL."""
    artifact = copy.deepcopy(_committed_artifact())
    artifact["gating"] = True
    artifact["n_cells_benchmarked"] = 3
    artifact["per_cell"] = artifact["per_cell"][:3]
    validator = _validator()
    assert not validator.is_valid(artifact)


def test_numeric_trace_collection_time_fails():
    """(c) NEGATIVE: folding trace time into the rate as a NUMBER must FAIL;
    the field is a string sentinel reported separately from the replay rate."""
    artifact = copy.deepcopy(_committed_artifact())
    artifact["nvbit_trace_collection_time"] = 1.234
    validator = _validator()
    assert not validator.is_valid(artifact)


def test_non_gating_subset_run_with_three_cells_is_admitted():
    """(d) ADMITTED: a non-gating subset run is admitted with fewer cells,
    because the conditional only constrains gating runs."""
    artifact = copy.deepcopy(_committed_artifact())
    artifact["gating"] = False
    artifact["n_cells_benchmarked"] = 3
    artifact["per_cell"] = artifact["per_cell"][:3]
    validator = _validator()
    errors = list(validator.iter_errors(artifact))
    assert errors == [], "\n".join(str(error) for error in errors)


def test_gating_benchmark_with_too_many_per_cell_entries_fails():
    """Exact full-cohort cardinality: a gating run must carry EXACTLY 30
    per-cell entries; 31 must fail."""
    artifact = copy.deepcopy(_committed_artifact())
    artifact["gating"] = True
    artifact["per_cell"].append(copy.deepcopy(artifact["per_cell"][0]))
    assert not _validator().is_valid(artifact)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
