"""Contracts for retained SOL calibration path relocation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tomllib

import pytest

from sol_baselines_harness.calibration.cache_prediction import (
    historical_path_aliases as aliases,
)
from sol_baselines_harness.calibration.cache_prediction.sm120_l2_associativity_probe import (
    harvest_tier2_traces,
)
from sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep import (
    run_sweep,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_alias_authority_is_exact_and_hash_pinned() -> None:
    document = aliases.load_alias_authority()

    assert hashlib.sha256(aliases.AUTHORITY_PATH.read_bytes()).hexdigest() == (
        aliases.AUTHORITY_SHA256
    )
    assert document["schema_version"] == 1
    assert [row["alias_id"] for row in document["aliases"]] == [
        aliases.SM120_CALIBRATION_BINARIES_ALIAS,
        aliases.SM86_HARNESS_TRACES_ALIAS,
    ]


def test_sm86_seed_artifacts_resolve_from_retained_annex() -> None:
    seed_document = tomllib.loads(
        (REPO_ROOT / run_sweep._SEED_LEDGER.relative_to(run_sweep._REPO_ROOT))
        .read_text(encoding="utf-8")
    )
    seed_entries = seed_document["entries"]
    seed_slugs = {entry["slug"] for entry in seed_entries}
    for entry in seed_entries:
        assert run_sweep._resolve_seed_artifact(
            REPO_ROOT,
            entry,
            "nvbit_metrics/tier2_address_trace.t2a",
        ).is_file()
        assert run_sweep._resolve_seed_artifact(
            REPO_ROOT,
            entry,
            "ncu_3run_v1.json",
        ).is_file()

    expanded_document = tomllib.loads(
        (
            REPO_ROOT
            / run_sweep._EXPANDED_LEDGER.relative_to(run_sweep._REPO_ROOT)
        ).read_text(encoding="utf-8")
    )
    expanded_entries = expanded_document["entries"]
    expanded_slugs = [entry["slug"] for entry in expanded_entries]
    assert len(expanded_slugs) == len(set(expanded_slugs))
    seed_slug_list = [entry["slug"] for entry in seed_entries]
    assert len(seed_slug_list) == len(set(seed_slug_list))
    expected_slugs = set(expanded_slugs) | set(seed_slug_list)
    records, excluded = run_sweep.build_records_at_assoc(REPO_ROOT, 16)
    record_slugs = [record.cell_name for record in records]
    excluded_slugs = [row["slug"] for row in excluded]

    assert len(record_slugs) == len(set(record_slugs))
    assert len(excluded_slugs) == len(set(excluded_slugs))
    assert set(record_slugs).isdisjoint(excluded_slugs)
    assert set(record_slugs) | set(excluded_slugs) == expected_slugs
    assert seed_slugs.isdisjoint(record_slugs)
    seed_exclusions = [row for row in excluded if row["source"] == "seed"]
    assert {row["slug"] for row in seed_exclusions} == seed_slugs
    assert {
        row["typed_reason"] for row in seed_exclusions
    } == {"cell_missing_cohort_partition"}
    incomplete_cohort = {
        entry["slug"]: entry["failure_reason"]
        for entry in expanded_entries
        if entry.get("source") == "cohort_b_probe"
        and entry.get("completed") is not True
    }
    cohort_exclusions = {
        row["slug"]: row["typed_reason"]
        for row in excluded
        if row["slug"] in incomplete_cohort
    }
    assert cohort_exclusions == incomplete_cohort


def test_sm86_ledger_index_rejects_duplicate_slugs() -> None:
    with pytest.raises(ValueError, match="duplicate slug: duplicate"):
        run_sweep._index_unique_entries(
            [{"slug": "duplicate"}, {"slug": "duplicate"}],
            ledger_name="test ledger",
        )


def test_sm86_cohort_outcome_rejects_malformed_values() -> None:
    with pytest.raises(ValueError, match="completed must be boolean"):
        run_sweep._cohort_failure_reason(
            "cell",
            {"completed": "false", "failure_reason": "ncu_repeatability_failed"},
        )
    with pytest.raises(ValueError, match="unsupported failure_reason"):
        run_sweep._cohort_failure_reason(
            "cell",
            {"completed": False, "failure_reason": "invented_reason"},
        )


def test_sm86_seed_cross_stamps_reject_mismatch() -> None:
    pilot_entry = {
        field: "a" * 64 for field in run_sweep._SEED_CROSS_STAMPS
    }
    expanded_entry = {
        field: "a" * 64
        for field in run_sweep._SEED_CROSS_STAMPS.values()
    }
    expanded_entry.update({"completed": True, "failure_reason": ""})
    run_sweep._validate_seed_cross_stamps(
        "cell",
        pilot_entry,
        expanded_entry,
    )

    pilot_field, mismatch_field = next(
        iter(run_sweep._SEED_CROSS_STAMPS.items())
    )
    missing_pilot_entry = dict(pilot_entry)
    del missing_pilot_entry[pilot_field]
    with pytest.raises(ValueError, match="invalid SHA-256 field"):
        run_sweep._validate_seed_cross_stamps(
            "cell",
            missing_pilot_entry,
            expanded_entry,
        )

    malformed_pilot_entry = dict(pilot_entry)
    malformed_expanded_entry = dict(expanded_entry)
    malformed_pilot_entry[pilot_field] = "not-a-sha"
    malformed_expanded_entry[mismatch_field] = "not-a-sha"
    with pytest.raises(ValueError, match="invalid SHA-256 field"):
        run_sweep._validate_seed_cross_stamps(
            "cell",
            malformed_pilot_entry,
            malformed_expanded_entry,
        )

    expanded_entry[mismatch_field] = "b" * 64
    with pytest.raises(ValueError, match="seed cross-stamp mismatch"):
        run_sweep._validate_seed_cross_stamps(
            "cell",
            pilot_entry,
            expanded_entry,
        )


def test_sm86_reuse_factor_rejects_duplicate_seed_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        run_sweep.tomllib,
        "loads",
        lambda _payload: {
            "entries": [{"slug": "duplicate"}, {"slug": "duplicate"}],
        },
    )
    with pytest.raises(ValueError, match="duplicate slug: duplicate"):
        run_sweep.reuse_factor_by_slug(REPO_ROOT, [])


def test_sm120_historical_binaries_resolve_with_ledger_hashes() -> None:
    entries = harvest_tier2_traces._ledger_historical_cells()
    resolved = [
        harvest_tier2_traces._binary_path_from_entry(entry)
        for entry in entries
    ]

    assert len(entries) == 70
    assert all(path is not None and path.is_file() for path in resolved)
    assert len(set(resolved)) == 70


def test_historical_alias_rejects_hash_mismatch() -> None:
    entry = harvest_tier2_traces._ledger_historical_cells()[0]
    historical = entry["historical_ncu_provenance"]["command_argv"][-1]

    with pytest.raises(
        aliases.HistoricalPathAliasError,
        match="historical_path_alias_hash_mismatch",
    ):
        aliases.resolve_historical_file(
            historical,
            alias_id=aliases.SM120_CALIBRATION_BINARIES_ALIAS,
            expected_sha256="0" * 64,
            repo_root=REPO_ROOT,
        )
