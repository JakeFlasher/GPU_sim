from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import immutable_artifact as immutable  # noqa: E402
import m6_recover_staging as recovery  # noqa: E402
import m6_determinism_campaign as campaign  # noqa: E402


SHA = "0" * 64


def _tree(root: Path, *, include_b: bool = True) -> None:
    (root / "nested").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"alpha")
    if include_b:
        (root / "nested" / "b.bin").write_bytes(b"bravo")


def _run(
    tmp_path: Path,
    *,
    stage: Path,
    expected: Path,
    final_name: str = "artifact",
    receipt_name: str = "recovery.json",
    historical_admission_path: Path | None = None,
):
    output = tmp_path / "published"
    output.mkdir(exist_ok=True)
    owner_hashes = {
        "schedule": "1" * 64,
        "journal": "2" * 64,
        "analysis": "3" * 64,
    }
    owner_payloads = {
        role: campaign.canonical_bytes({f"{role}_sha256": seal})
        for role, seal in owner_hashes.items()
    }
    owner_paths = {
        role: tmp_path / f"{role}.json"
        for role in owner_payloads
    }
    for role, payload in owner_payloads.items():
        owner_paths[role].write_bytes(payload)
    # These topology tests exercise recovery publication after the owner
    # identities are bound.  Semantic rederivation has dedicated campaign
    # tests; keep this fixture independent of a full M6 artifact bundle.
    with patch.object(recovery, "_validate_semantic_inputs", return_value=None):
        return recovery.recover_staging(
            staging_root=stage,
            expected_root=expected,
            output_root=output,
            final_name=final_name,
            schedule_sha256=owner_hashes["schedule"],
            journal_sha256=owner_hashes["journal"],
            analysis_sha256=owner_hashes["analysis"],
            schedule_path=owner_paths["schedule"],
            journal_path=owner_paths["journal"],
            analysis_path=owner_paths["analysis"],
            historical_admission_path=historical_admission_path,
            receipt_path=tmp_path / receipt_name,
        )


def test_exact_complete_is_byte_exact_and_idempotent(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)

    first = _run(tmp_path, stage=stage, expected=expected)
    assert first["decision"] == "EXACT_PREFIX_COMPLETE"
    assert (tmp_path / "published/artifact/a.bin").read_bytes() == b"alpha"
    assert (tmp_path / "published/artifact/nested/b.bin").read_bytes() == b"bravo"
    assert first["published_path"].endswith("published/artifact")
    immutable.verify_model_seal(
        immutable.strict_json_load(tmp_path / "recovery.json"),
        "receipt_sha256",
    )

    second = _run(tmp_path, stage=stage, expected=expected)
    assert second["decision"] == "EXACT_PREFIX_COMPLETE"
    assert second["published_path"] == first["published_path"]
    assert (tmp_path / "published/artifact/a.bin").read_bytes() == b"alpha"


def test_exact_prefix_partial_completes_without_mutating_stage(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    stage.mkdir()
    (stage / "a.bin").write_bytes(b"alpha")

    result = _run(tmp_path, stage=stage, expected=expected)
    assert result["decision"] == "EXACT_PREFIX_PARTIAL"
    assert (stage / "a.bin").read_bytes() == b"alpha"
    assert not (stage / "nested" / "b.bin").exists()
    assert (tmp_path / "published/artifact/nested/b.bin").read_bytes() == b"bravo"


def test_invalid_done_material_is_rejected_before_publication(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)
    (expected / "M6_DONE.json").write_text("{}\n", encoding="utf-8")
    (stage / "M6_DONE.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(recovery.RecoveryError) as error:
        _run(tmp_path, stage=stage, expected=expected)
    assert error.value.code == "M6_RECOVERY_DONE_INCOMPLETE"
    assert not (tmp_path / "published/artifact").exists()


def test_expected_done_material_must_match_bound_owner_payloads() -> None:
    expected_payloads = {
        "schedule.json": b'{"owner":"bundle-b"}\n',
        "observation_journal.json": b'{"owner":"bundle-b"}\n',
        "determinism_study.json": b'{"owner":"bundle-b"}\n',
        "M6_DONE.json": b"{}\n",
    }

    with pytest.raises(recovery.RecoveryError) as error:
        recovery._verify_expected_done_material(
            expected_payloads,
            schedule_payload=b'{"owner":"bundle-a"}\n',
            journal_payload=expected_payloads["observation_journal.json"],
            analysis_payload=expected_payloads["determinism_study.json"],
        )
    assert error.value.code == "M6_RECOVERY_OWNER_BUNDLE_MISMATCH"


def test_historical_authority_generation_is_rechecked_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)
    authority = tmp_path / "historical.json"
    authority.write_bytes(b"same historical bytes\n")
    original_publish = immutable.publish_directory

    def replace_then_publish(*args, **kwargs):
        payload = authority.read_bytes()
        authority.unlink()
        authority.write_bytes(payload)
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(immutable, "publish_directory", replace_then_publish)
    with pytest.raises(recovery.RecoveryError) as error:
        _run(
            tmp_path,
            stage=stage,
            expected=expected,
            historical_admission_path=authority,
        )
    assert error.value.code == "M6_RECOVERY_INPUT_MIXED"
    assert not (tmp_path / "published/artifact").exists()


def test_reconciled_analysis_requires_explicit_historical_authority(
    tmp_path: Path,
) -> None:
    schedule = {"schedule_sha256": "1" * 64}
    journal = {"journal_sha256": "2" * 64}
    analysis = {
        "analysis_sha256": "3" * 64,
        "analysis_policy": {
            "historical_admission_reconciliation": {"authority": {}}
        },
    }

    with pytest.raises(recovery.RecoveryError) as error:
        recovery._validate_semantic_inputs(
            schedule_payload=campaign.canonical_bytes(schedule),
            journal_payload=campaign.canonical_bytes(journal),
            analysis_payload=campaign.canonical_bytes(analysis),
            schedule_path=tmp_path / "schedule.json",
            journal_path=tmp_path / "journal.json",
            analysis_path=tmp_path / "analysis.json",
            historical_admission_path=None,
            evidence_root=None,
            schedule_sha256="1" * 64,
            journal_sha256="2" * 64,
            analysis_sha256="3" * 64,
        )
    assert error.value.code == "M6_HISTORICAL_AUTHORITY_UNBOUND"


def test_one_byte_contradiction_is_preserved_and_quarantined(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)
    (stage / "a.bin").write_bytes(b"alphx")

    result = _run(tmp_path, stage=stage, expected=expected)
    assert result["decision"] == "CONTRADICTORY"
    assert result["contradiction_code"] == "byte_mismatch"
    assert (stage / "a.bin").read_bytes() == b"alphx"
    assert not (tmp_path / "published/artifact").exists()


def test_inode_replacement_during_scan_is_not_reinterpreted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)
    original_verify = campaign._PinnedTreeTransaction.verify
    replaced = False

    def replace_then_verify(transaction):
        nonlocal replaced
        if not replaced:
            replaced = True
            target = transaction.root / "a.bin"
            target.unlink()
            target.write_bytes(b"foreign")
        return original_verify(transaction)

    monkeypatch.setattr(campaign._PinnedTreeTransaction, "verify", replace_then_verify)
    result = _run(tmp_path, stage=stage, expected=expected)
    assert result["decision"] == "CONTRADICTORY"
    assert result["contradiction_code"] == "M6_TREE_CHANGED"
    assert (stage / "a.bin").read_bytes() == b"foreign"
    assert not (tmp_path / "published/artifact").exists()


def test_final_and_stage_coexistence_is_contradictory(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    _tree(expected)
    output = tmp_path / "published"
    output.mkdir()
    stage = tmp_path / "staging-input"
    _tree(stage)
    sibling_stage = output / ".artifact.staging"
    sibling_stage.mkdir(parents=True)
    (sibling_stage / "marker").write_bytes(b"preserve")
    final = output / "artifact"
    _tree(final)
    result = _run(tmp_path, stage=stage, expected=expected)
    assert result["decision"] == "CONTRADICTORY"
    assert result["contradiction_code"] == "final_and_stage_coexist"
    assert (final / "a.bin").read_bytes() == b"alpha"
    assert (stage / "a.bin").read_bytes() == b"alpha"


def test_identity_paths_are_bound_to_declared_hashes(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)
    schedule = tmp_path / "schedule.json"
    schedule.write_bytes(b"schedule")
    with pytest.raises(recovery.RecoveryError) as error:
        recovery.recover_staging(
            staging_root=stage,
            expected_root=expected,
            output_root=tmp_path / "published",
            final_name="artifact",
            schedule_sha256=SHA,
            journal_sha256=SHA,
            analysis_sha256=SHA,
            schedule_path=schedule,
        )
    assert error.value.code == "M6_RECOVERY_IDENTITY_MISMATCH"


def test_hash_only_recovery_is_non_publishing_unauthorized(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)
    result = recovery.recover_staging(
        staging_root=stage,
        expected_root=expected,
        output_root=tmp_path / "published",
        final_name="artifact",
        schedule_sha256=SHA,
        journal_sha256=SHA,
        analysis_sha256=SHA,
        receipt_path=tmp_path / "unauthorized.json",
    )
    assert result["decision"] == "UNAUTHORIZED"
    assert result["published_path"] is None
    assert result["contradiction_code"] == "M6_RECOVERY_OWNER_PATHS_REQUIRED"
    assert not (tmp_path / "published/artifact").exists()


def test_extra_empty_directory_is_contradictory(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)
    (stage / "foreign-empty").mkdir()
    result = _run(tmp_path, stage=stage, expected=expected)
    assert result["decision"] == "CONTRADICTORY"
    assert result["contradiction_code"] == "unexpected_path"
    assert (stage / "foreign-empty").is_dir()


def test_type_and_mode_changes_are_contradictory(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)
    (stage / "a.bin").unlink()
    (stage / "a.bin").mkdir()
    result = _run(tmp_path, stage=stage, expected=expected)
    assert result["decision"] == "CONTRADICTORY"
    assert result["contradiction_code"] == "type_mismatch"

    expected = tmp_path / "expected-mode"
    stage = tmp_path / "stage-mode"
    _tree(expected)
    _tree(stage)
    (stage / "a.bin").chmod(0o600)
    run_root = tmp_path / "mode-run"
    run_root.mkdir()
    result = _run(run_root, stage=stage, expected=expected)
    assert result["decision"] == "CONTRADICTORY"
    assert result["contradiction_code"] == "mode_mismatch"


def test_expected_empty_directory_is_rejected_without_publication(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)
    (expected / "intentional-empty").mkdir()
    (stage / "intentional-empty").mkdir()

    result = _run(tmp_path, stage=stage, expected=expected)
    assert result["decision"] == "CONTRADICTORY"
    assert result["contradiction_code"] == "M6_RECOVERY_EMPTY_DIRECTORY_UNSUPPORTED"
    assert not (tmp_path / "published/artifact").exists()
    assert (stage / "intentional-empty").is_dir()


def test_existing_receipt_collision_is_not_overwritten(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    stage = tmp_path / "stage"
    _tree(expected)
    _tree(stage)
    receipt = tmp_path / "recovery.json"
    receipt.write_text('{"foreign": true}\n', encoding="utf-8")
    with pytest.raises(recovery.RecoveryError) as error:
        _run(tmp_path, stage=stage, expected=expected)
    assert error.value.code == "M6_RECOVERY_RECEIPT_COLLISION"
    assert receipt.read_text(encoding="utf-8") == '{"foreign": true}\n'
