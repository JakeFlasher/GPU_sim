"""Focused checks for the active-replay validator's supplemental allowlist."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_validator():
    path = Path(__file__).with_name("validate_active_replay_pairs.py")
    spec = importlib.util.spec_from_file_location("tested_active_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def test_full_campaign_supplemental_directory_is_recognized(tmp_path: Path) -> None:
    root = tmp_path / "admission"
    known = root / "simulator-memory-admission-feedback-service-101" / (
        "b84_w8_store_line_unique_d32_g0_all"
    )
    assert VALIDATOR._known_supplemental_replay_directory(known, root)


def test_unknown_candidate_or_cell_is_not_recognized(tmp_path: Path) -> None:
    root = tmp_path / "admission"
    unknown_id = root / "simulator-memory-admission-feedback-service-999" / (
        "b84_w8_store_line_unique_d32_g0_all"
    )
    unknown_cell = root / "simulator-memory-admission-feedback-service-101" / (
        "not-a-campaign-cell"
    )
    assert not VALIDATOR._known_supplemental_replay_directory(unknown_id, root)
    assert not VALIDATOR._known_supplemental_replay_directory(unknown_cell, root)
