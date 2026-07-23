from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.pchase import run_pchase_card
from tools.pchase import transcribe_latencies_to_arch_card as transcribe
from tools.producer_retirement import (
    LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE,
    RETIRED_PRODUCER_EXIT_CODE,
    LegacyProducerRetiredError,
)


def _forbid_side_effect(*_args, **_kwargs):
    raise AssertionError("retired pchase producer reached a side effect")


def _assert_retired(call) -> None:
    with pytest.raises(LegacyProducerRetiredError) as caught:
        call()
    assert caught.value.reason == LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE
    assert caught.value.surface
    assert caught.value.successor_requirement


def test_imported_collection_helpers_refuse_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess, "run", _forbid_side_effect)
    monkeypatch.setattr(Path, "mkdir", _forbid_side_effect)
    monkeypatch.setattr(Path, "write_text", _forbid_side_effect)

    _assert_retired(run_pchase_card.build_binary)
    _assert_retired(run_pchase_card.nvml_sample)
    _assert_retired(lambda: run_pchase_card.run_probe([], {}))


def test_public_entry_points_refuse_before_input_access(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(Path, "read_text", _forbid_side_effect)
    monkeypatch.setattr(Path, "write_text", _forbid_side_effect)
    monkeypatch.setattr(subprocess, "run", _forbid_side_effect)

    assert run_pchase_card.main(["--quick"]) == RETIRED_PRODUCER_EXIT_CODE
    assert transcribe.main() == RETIRED_PRODUCER_EXIT_CODE
    diagnostics = capsys.readouterr().err
    assert diagnostics.count(LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE) == 2
    assert "schema-v2 external immutable evidence" in diagnostics
    assert "verified schema-v2 protocol metadata" in diagnostics
