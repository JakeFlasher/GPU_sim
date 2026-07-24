"""Contract tests for portable preserved-binary generation and local gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.local_labels import build_preserved_binaries as preserved_builder
from tools.local_labels.build_preserved_binaries import (
    MANIFEST_REL,
    MARKDOWN_REL,
    PreservationPrerequisiteError,
    PreservationValidationError,
    _safe_repo_relative,
    build_output_bytes,
    emit_outputs,
)
from tools.local_labels.gate_preserved_binaries_local import (
    GatePrerequisiteError,
    GateValidationError,
    _safe_path,
    gate_preserved_binaries,
    load_manifest,
    validate_manifest,
)
from tools.local_labels.protected_paths import ProtectedPathError


REPO_ROOT = Path(__file__).resolve().parents[2]


def _committed_bytes() -> tuple[bytes, bytes]:
    return (
        (REPO_ROOT / MANIFEST_REL).read_bytes(),
        (REPO_ROOT / MARKDOWN_REL).read_bytes(),
    )


def _generated() -> tuple[dict, str]:
    manifest_bytes, markdown_bytes = _committed_bytes()
    return json.loads(manifest_bytes), markdown_bytes.decode("utf-8")


def test_generated_manifest_is_closed_frozen_projection_with_tripwires() -> None:
    manifest, _markdown = _generated()
    assert set(manifest) == {
        "schema_version",
        "source_authority",
        "source_baseline_commit",
        "binaries",
    }
    assert manifest["schema_version"] == 1
    assert manifest["source_authority"] == {
        "path": "data/local_labels/campaign/capture_roster_authority_sm86.json",
        "sha256": "d02c817091b0a9f06c2106c0987c59a481910afcc3b78ab1de7a7d5bd5a6b4b4",
    }
    assert manifest["source_baseline_commit"] == (
        "c4b76c0ae3b49378f6e3d4cabf7830dbe3eaa432"
    )
    assert len(manifest["binaries"]) == 15
    assert all(set(row) == {"cells", "binary_path", "binary_sha256", "diverged_sources"}
               for row in manifest["binaries"])
    assert not any("/ac4_mixed_long_sm80/" in row["binary_path"]
                   for row in manifest["binaries"])
    for unsafe in ("a//b", "a/./b", "a\x00b", "a\nb", "a\x7fb"):
        with pytest.raises(PreservationValidationError, match="safe repository path"):
            _safe_repo_relative(unsafe, field="test path")


def test_every_generated_row_attributes_runner_common() -> None:
    manifest, _markdown = _generated()
    assert all(
        "workloads/portable_lcd/runner_common.h" in row["diverged_sources"]
        for row in manifest["binaries"]
    )


def test_only_conv2d_and_gemm_receive_local_source_attribution() -> None:
    manifest, _markdown = _generated()
    widened = {
        row["cells"][0]: row["diverged_sources"]
        for row in manifest["binaries"]
        if len(row["diverged_sources"]) == 2
    }
    assert set(widened) == {
        "conv2d_implicit_gemm_bf16_sm80",
        "gemm_hmma_bf16_sm80",
    }
    assert widened["conv2d_implicit_gemm_bf16_sm80"][0].endswith(
        "/conv2d_implicit_gemm_bf16_sm80/main.cu"
    )
    assert widened["gemm_hmma_bf16_sm80"][0].endswith(
        "/gemm_hmma_bf16_sm80/main.cu"
    )


def test_committed_markdown_is_exactly_fifteen_rows() -> None:
    _manifest_bytes, markdown_bytes = _committed_bytes()
    text = markdown_bytes.decode("utf-8")
    assert text.startswith("# Preserved portable LCD binaries\n\n")
    assert text.endswith("|\n")
    assert sum(line.startswith("| `workloads/") for line in text.splitlines()) == 15
    assert "<br>" in text


@pytest.mark.campaign_evidence
def test_committed_preserved_authorities_are_exact_deterministic_rebuild() -> None:
    first = build_output_bytes(repo_root=REPO_ROOT)
    second = build_output_bytes(repo_root=REPO_ROOT)
    assert first == second == _committed_bytes()


def test_explicit_generation_writes_only_paired_external_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_out = tmp_path / "manifest.json"
    markdown_out = tmp_path / "manifest.md"
    expected = _committed_bytes()
    monkeypatch.setattr(
        preserved_builder,
        "build_output_bytes",
        lambda **_kwargs: expected,
    )
    assert emit_outputs(
        repo_root=REPO_ROOT,
        manifest_out=manifest_out,
        markdown_out=markdown_out,
    ) == (manifest_out, markdown_out)
    assert (manifest_out.read_bytes(), markdown_out.read_bytes()) == expected
    with pytest.raises(PreservationValidationError, match="supplied together"):
        emit_outputs(repo_root=REPO_ROOT, manifest_out=tmp_path / "one", markdown_out=None)
    with pytest.raises(ProtectedPathError, match="Git worktree"):
        emit_outputs(
            repo_root=REPO_ROOT,
            manifest_out=REPO_ROOT / "scratch-manifest.json",
            markdown_out=tmp_path / "safe.md",
        )


def test_canonical_generation_refuses_the_primary_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preserved_builder,
        "build_output_bytes",
        lambda **_kwargs: _committed_bytes(),
    )
    with pytest.raises(PreservationValidationError, match="detached linked C1"):
        emit_outputs(repo_root=REPO_ROOT, manifest_out=None, markdown_out=None)


def test_generation_fails_closed_when_source_baseline_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingCommit:
        returncode = 1

    monkeypatch.setattr(
        preserved_builder,
        "_git",
        lambda *_args: MissingCommit(),
    )
    with pytest.raises(
        PreservationPrerequisiteError,
        match="source baseline commit is unavailable",
    ):
        build_output_bytes(repo_root=REPO_ROOT)


def test_local_gate_accepts_all_fifteen_canonical_binary_pins() -> None:
    gate_preserved_binaries(
        repo_root=REPO_ROOT,
        manifest_path=REPO_ROOT / MANIFEST_REL,
    )


def test_local_gate_prioritizes_structure_or_hash_mismatch_over_missing(
    tmp_path: Path,
) -> None:
    canonical = json.loads((REPO_ROOT / MANIFEST_REL).read_text(encoding="utf-8"))
    for unsafe in ("a//b", "a/./b", "a\x00b", "a\nb", "a\x7fb"):
        with pytest.raises(GateValidationError, match="safe repository path"):
            _safe_path(unsafe, field="test path")
    malformed = json.loads(json.dumps(canonical))
    malformed["unknown"] = True
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(
        json.dumps(malformed, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GateValidationError, match="keys mismatch"):
        validate_manifest(load_manifest(malformed_path))

    fake_root = tmp_path / "empty-repo"
    fake_root.mkdir()
    first = canonical["binaries"][0]
    wrong = fake_root / first["binary_path"]
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(b"wrong preserved bytes")
    with pytest.raises(GateValidationError, match="SHA-256 mismatch"):
        gate_preserved_binaries(repo_root=fake_root, manifest_path=REPO_ROOT / MANIFEST_REL)

    wrong.unlink()
    with pytest.raises(GatePrerequisiteError, match="binary is missing"):
        gate_preserved_binaries(repo_root=fake_root, manifest_path=REPO_ROOT / MANIFEST_REL)
