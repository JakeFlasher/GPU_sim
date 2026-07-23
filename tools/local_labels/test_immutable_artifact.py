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


def _replace_file_with_same_bytes(root: Path, relative_path: str, payload: bytes) -> int:
    target = root / "artifact" / relative_path
    parent = target.parent
    parent.chmod(0o755)
    try:
        target.unlink()
        target.write_bytes(payload)
        target.chmod(0o444)
        return target.stat().st_ino
    finally:
        parent.chmod(0o555)


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


def test_directory_winner_rejects_same_byte_inode_replacement(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    immutable.publish_directory(output_root, "artifact", material)
    original_inode = (output_root / "artifact" / "value").stat().st_ino
    callback_count = 0
    replacement_inode: int | None = None

    def replace_on_winner_accept(root: Path, _descriptor: int) -> None:
        nonlocal callback_count, replacement_inode
        callback_count += 1
        if callback_count == 2:
            replacement_inode = _replace_file_with_same_bytes(
                root, "value", b"original"
            )

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="immutable tree changed after authorization",
    ):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
            before_commit=replace_on_winner_accept,
        )

    assert callback_count == 2
    assert replacement_inode is not None
    assert replacement_inode != original_inode
    assert (output_root / "artifact" / "value").read_bytes() == b"original"


def test_directory_rejects_same_byte_inode_replacement_after_commit(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    callback_count = 0
    replacement_inode: int | None = None

    def replace_after_commit(root: Path, _descriptor: int) -> None:
        nonlocal callback_count, replacement_inode
        callback_count += 1
        if callback_count == 3:
            replacement_inode = _replace_file_with_same_bytes(
                root, "value", b"original"
            )

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="immutable tree changed after authorization",
    ):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
            before_commit=replace_after_commit,
        )

    assert callback_count == 3
    assert replacement_inode is not None
    assert (output_root / "artifact" / "value").read_bytes() == b"original"


def test_directory_final_authorization_follows_expensive_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_verify = immutable._verify_published_files_at

    def record_verify(*args, **kwargs):
        events.append("verify")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(immutable, "_verify_published_files_at", record_verify)

    def authorize(_root: Path, _descriptor: int) -> None:
        events.append("authorize")

    immutable.publish_directory(
        tmp_path / "published",
        "artifact",
        [immutable.ImmutableFile.from_bytes("value", b"original")],
        before_commit=authorize,
    )

    assert events[-2:] == ["verify", "authorize"]


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


def test_directory_accepts_exact_rename_race_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    callback_count = 0

    def authorize(_root: Path, _descriptor: int) -> None:
        nonlocal callback_count
        callback_count += 1

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
    result = immutable.publish_directory(
        output_root,
        "artifact",
        material,
        before_commit=authorize,
    )

    assert result == output_root / "artifact"
    assert callback_count == 3
    assert (result / "value").read_bytes() == b"original"


def test_directory_fallback_rejects_same_byte_inode_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    callback_count = 0
    replacement_inode: int | None = None

    def replace_on_fallback_accept(root: Path, _descriptor: int) -> None:
        nonlocal callback_count, replacement_inode
        callback_count += 1
        if callback_count == 3:
            replacement_inode = _replace_file_with_same_bytes(
                root, "value", b"original"
            )

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
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="immutable tree changed after authorization",
    ):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
            before_commit=replace_on_fallback_accept,
        )

    assert callback_count == 3
    assert replacement_inode is not None
    assert (output_root / "artifact" / "value").read_bytes() == b"original"


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
