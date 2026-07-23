"""Focused tests for the external sm_86 live-v1 capture authority."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import build_sm86_capture_matrix as matrix_builder
from tools.local_labels import build_sm86_live_capture_authority as live
from tools.local_labels.build_simbench_build_witness import canonical_json_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO_ROOT / live.SCHEMA_REL).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def authority() -> dict:
    return live.build_authority(
        REPO_ROOT, generated_at_utc="2026-07-13T00:00:00Z",
    )


def _head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def test_live_authority_is_current_v1_with_exact_partition(authority: dict) -> None:
    jsonschema.validate(authority, SCHEMA)
    assert authority["schema_version"] == 1
    assert authority["roster_id"] == "sm86_capture_roster_authority_v1"
    assert authority["arch"] == "sm_86"
    assert authority["sku"] == "rtx_3090_ti"
    assert authority["source_commit"] == _head()
    assert authority["counts"] == live.EXPECTED_COUNTS
    assert len(authority["cells"]) == 344
    assert authority["accounting_identity_holds"] is True
    frozen = json.loads(
        (REPO_ROOT / live.HISTORICAL_REL).read_text(encoding="utf-8")
    )
    ignored = {"source_commit", "generated_at_utc"}
    assert {
        key: value for key, value in authority.items() if key not in ignored
    } == {
        key: value for key, value in frozen.items() if key not in ignored
    }


def test_all_runnable_identities_are_current_and_frozen(authority: dict) -> None:
    runnable = [
        cell for cell in authority["cells"]
        if cell["resolution_status"] == "runnable"
    ]
    assert len(runnable) == 122
    for cell in runnable:
        path = (REPO_ROOT / cell["binary"]).resolve(strict=True)
        assert path.is_relative_to(REPO_ROOT.resolve())
        assert path.is_file()
        if cell["roster_source"] != "sol_execbench":
            assert os.access(path, os.X_OK)
        assert live._sha256_file(path) == cell["binary_sha256"]
        assert cell["binary_present"] is True
        assert cell["binary_available_now"] is True
        if cell["roster_source"] == "simbench_bw":
            assert cell["binary_sha256_matches_now"] is True


def test_external_authority_is_accepted_by_current_matrix(
    authority: dict, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    directory = tmp_path_factory.mktemp("live-authority")
    output = directory / "authority.json"
    matrix_output = directory / "matrix.json"
    output.write_bytes(canonical_json_bytes(authority))
    assert matrix_builder.main([
        "--authority", str(output), "--out", str(matrix_output),
    ]) == 0
    matrix = json.loads(matrix_output.read_text(encoding="utf-8"))
    assert matrix["source_commit"] == _head()
    assert matrix["authority"]["path"] == str(output.resolve())
    assert matrix["authority"]["roster_id"] == live.ROSTER_ID
    assert matrix["counts"]["universe"] == 344
    assert matrix["counts"]["runnable"] == 122
    assert matrix["counts"]["excluded"] == 222


def test_output_policy_requires_external_absolute_nonexisting_path(
    tmp_path: Path,
) -> None:
    head = _head()
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    expected = (
        capture_root
        / "inputs"
        / f"r28_{head[:8]}"
        / "capture_roster_authority_sm86.json"
    )
    assert live._resolve_output(
        str(expected),
        expected_head=head,
        repo_root=REPO_ROOT,
        capture_root=capture_root,
    ) == expected.resolve()
    with pytest.raises(live.LiveAuthorityError, match="absolute campaign"):
        live._resolve_output(
            "relative/live-authority.json",
            expected_head=head,
            repo_root=REPO_ROOT,
            capture_root=capture_root,
        )
    with pytest.raises(live.LiveAuthorityError, match="exact current-HEAD"):
        live._resolve_output(
            str(capture_root / "wrong.json"),
            expected_head=head,
            repo_root=REPO_ROOT,
            capture_root=capture_root,
        )
    expected.parent.mkdir(parents=True)
    existing = expected
    existing.write_text("user-owned\n", encoding="utf-8")
    with pytest.raises(live.LiveAuthorityError, match="overwrite"):
        live._resolve_output(
            str(existing),
            expected_head=head,
            repo_root=REPO_ROOT,
            capture_root=capture_root,
        )


def test_output_policy_rejects_symlinked_campaign_ancestors(
    tmp_path: Path,
) -> None:
    head = _head()
    capture_root = tmp_path / "capture"
    escape = tmp_path / "escape"
    capture_root.mkdir()
    escape.mkdir()
    (capture_root / "inputs").symlink_to(escape, target_is_directory=True)
    target = (
        capture_root / "inputs" / f"r28_{head[:8]}"
        / "capture_roster_authority_sm86.json"
    )
    with pytest.raises(live.LiveAuthorityError, match="ancestor"):
        live._resolve_output(
            str(target), expected_head=head, repo_root=REPO_ROOT,
            capture_root=capture_root,
        )
    with pytest.raises(live.LiveAuthorityError, match="ancestor"):
        live._write_exclusive(
            target, b"must-not-escape\n", expected_head=head,
            repo_root=REPO_ROOT, capture_root=capture_root,
        )
    assert list(escape.iterdir()) == []


def test_output_policy_rejects_symlinked_campaign_root(tmp_path: Path) -> None:
    head = _head()
    real_root = tmp_path / "real-capture"
    linked_root = tmp_path / "linked-capture"
    real_root.mkdir()
    linked_root.symlink_to(real_root, target_is_directory=True)
    target = (
        linked_root / "inputs" / f"r28_{head[:8]}"
        / "capture_roster_authority_sm86.json"
    )
    with pytest.raises(live.LiveAuthorityError, match="canonical non-symlink"):
        live._resolve_output(
            str(target), expected_head=head, repo_root=REPO_ROOT,
            capture_root=linked_root,
        )
    assert list(real_root.iterdir()) == []


def test_main_emits_once_and_never_overwrites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch,
) -> None:
    head = _head()
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    monkeypatch.setattr(live, "CAPTURE_ROOT", capture_root)
    output = (
        capture_root
        / "inputs"
        / f"r28_{head[:8]}"
        / "capture_roster_authority_sm86.json"
    )
    args = ["--out", str(output), "--expected-head", head]
    assert live.main(args) == 0
    first = output.read_bytes()
    emitted = json.loads(first)
    assert emitted["source_commit"] == _head()
    assert emitted["counts"] == live.EXPECTED_COUNTS

    assert live.main(args) == 2
    assert output.read_bytes() == first
    assert "refusing to overwrite" in capsys.readouterr().err


def test_runnable_verifier_rejects_identity_and_path_defects(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "bin" / "probe"
    binary.parent.mkdir()
    binary.write_bytes(b"identity\n")
    binary.chmod(0o755)
    cell = {
        "roster_source": "portable_lcd",
        "slug": "probe",
        "binary": "bin/probe",
        "binary_sha256": live._sha256_file(binary),
    }
    assert live._verified_runnable_identity(cell, repo_root=tmp_path) == cell[
        "binary_sha256"
    ]

    mismatched = {**cell, "binary_sha256": "0" * 64}
    with pytest.raises(live.LiveAuthorityError, match="identity mismatch"):
        live._verified_runnable_identity(mismatched, repo_root=tmp_path)

    binary.chmod(0o644)
    with pytest.raises(live.LiveAuthorityError, match="not executable"):
        live._verified_runnable_identity(cell, repo_root=tmp_path)

    escaping = {**cell, "binary": "../probe"}
    with pytest.raises(live.LiveAuthorityError, match="safe repo-relative"):
        live._verified_runnable_identity(escaping, repo_root=tmp_path)
