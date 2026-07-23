from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import immutable_artifact as immutable  # noqa: E402


def test_directory_publication_rechecks_after_authorization_callback(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    mutated = False

    def mutate_after_commit(root: Path, _descriptor: int) -> None:
        nonlocal mutated
        target = root / "artifact" / "value"
        if target.exists() and not mutated:
            mutated = True
            target.chmod(0o644)
            target.write_bytes(b"tampered")

    with pytest.raises(immutable.ImmutableArtifactError):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
            before_commit=mutate_after_commit,
        )
    assert mutated is True
    assert (output_root / "artifact" / "value").read_bytes() == b"tampered"


def test_directory_winner_rechecks_after_second_authorization_callback(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    immutable.publish_directory(output_root, "artifact", material)
    callback_count = 0

    def mutate_on_winner_accept(root: Path, _descriptor: int) -> None:
        nonlocal callback_count
        callback_count += 1
        if callback_count == 2:
            target = root / "artifact" / "value"
            target.chmod(0o644)
            target.write_bytes(b"tampered")

    with pytest.raises(immutable.ImmutableArtifactError):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
            before_commit=mutate_on_winner_accept,
        )
    assert callback_count == 2
    assert (output_root / "artifact" / "value").read_bytes() == b"tampered"


def test_directory_reauthorizes_rename_race_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    callback_count = 0

    def mutate_on_fallback_accept(root: Path, _descriptor: int) -> None:
        nonlocal callback_count
        callback_count += 1
        if callback_count == 3:
            target = root / "artifact" / "value"
            target.chmod(0o644)
            target.write_bytes(b"tampered")

    def lose_rename_to_exact_winner(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        if source_name != ".artifact.staging":
            return original_rename(
                source_descriptor,
                source_name,
                destination_descriptor,
                destination_name,
            )
        os.rename(
            source_name,
            destination_name,
            src_dir_fd=source_descriptor,
            dst_dir_fd=destination_descriptor,
        )
        raise FileExistsError(errno.EEXIST, "simulated concurrent exact winner")

    original_rename = immutable._rename_noreplace_at
    monkeypatch.setattr(
        immutable, "_rename_noreplace_at", lose_rename_to_exact_winner
    )
    with pytest.raises(immutable.ImmutableArtifactError):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
            before_commit=mutate_on_fallback_accept,
        )

    assert callback_count == 3
    assert (output_root / "artifact" / "value").read_bytes() == b"tampered"


def test_regular_publication_rejects_post_rename_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifact.bin"
    alias = tmp_path / "alias.bin"
    original_entry_metadata = immutable._entry_metadata
    linked = False

    def add_alias_after_rename(descriptor: int, name: str):
        nonlocal linked
        metadata = original_entry_metadata(descriptor, name)
        if name == output.name and metadata is not None and not linked:
            linked = True
            alias.hardlink_to(output)
            metadata = original_entry_metadata(descriptor, name)
        return metadata

    monkeypatch.setattr(immutable, "_entry_metadata", add_alias_after_rename)
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="named regular-file path changed",
    ):
        immutable.publish_regular_file(tmp_path, output.name, b"payload")

    assert output.read_bytes() == b"payload"
    assert alias.read_bytes() == b"payload"
    assert output.stat().st_nlink == 2
