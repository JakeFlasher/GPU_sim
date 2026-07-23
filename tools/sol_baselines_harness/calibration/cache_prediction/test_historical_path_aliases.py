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


def test_sm86_seed_records_resolve_from_retained_annex() -> None:
    records, excluded = run_sweep.build_records_at_assoc(REPO_ROOT, 16)
    seed_document = tomllib.loads(
        (REPO_ROOT / run_sweep._SEED_LEDGER.relative_to(run_sweep._REPO_ROOT))
        .read_text(encoding="utf-8")
    )
    seed_slugs = {entry["slug"] for entry in seed_document["entries"]}

    assert len(records) == 50
    assert seed_slugs <= {record.cell_name for record in records}
    assert not [row for row in excluded if row["source"] == "seed"]


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
