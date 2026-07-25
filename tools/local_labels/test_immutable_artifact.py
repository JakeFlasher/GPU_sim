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


def _executable(path: Path, payload: bytes = b"fixture executable") -> Path:
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def test_retained_executable_binds_descriptor_path_snapshot_and_hash(
    tmp_path: Path,
) -> None:
    executable = _executable(tmp_path / "ncu")
    with immutable.open_retained_executable(executable) as retained:
        assert retained.path == executable.absolute()
        assert retained.sha256 == immutable.sha256_bytes(b"fixture executable")
        assert len(retained.snapshot) == 7
        assert retained.descriptor >= 0
        assert os.read(retained.descriptor, 7) == b"fixture"
        retained.verify_unchanged()
        assert os.read(retained.descriptor, 7) == b"fixture"
    with pytest.raises(
        immutable.ImmutableArtifactError, match="retained executable is closed",
    ):
        _ = retained.descriptor


@pytest.mark.parametrize("kind", ["leaf", "ancestor"])
def test_retained_executable_rejects_symlink_components(
    tmp_path: Path, kind: str,
) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    executable = _executable(real_dir / "ncu")
    if kind == "leaf":
        candidate = tmp_path / "ncu"
        candidate.symlink_to(executable)
    else:
        linked_dir = tmp_path / "linked"
        linked_dir.symlink_to(real_dir, target_is_directory=True)
        candidate = linked_dir / "ncu"
    with pytest.raises(immutable.ImmutableArtifactError):
        immutable.open_retained_executable(candidate)


@pytest.mark.parametrize("kind", ["non_executable", "directory", "hardlink"])
def test_retained_executable_rejects_invalid_leaf(
    tmp_path: Path, kind: str,
) -> None:
    if kind == "non_executable":
        candidate = tmp_path / "ncu"
        candidate.write_bytes(b"fixture")
        candidate.chmod(0o644)
    elif kind == "directory":
        candidate = tmp_path / "ncu"
        candidate.mkdir()
    else:
        candidate = _executable(tmp_path / "ncu")
        os.link(candidate, tmp_path / "ncu-hardlink")
    with pytest.raises(immutable.ImmutableArtifactError):
        immutable.open_retained_executable(candidate)


@pytest.mark.parametrize(
    "mutation", ["bytes", "replacement", "ancestor", "mode", "hardlink"],
)
def test_retained_executable_rejects_post_open_identity_drift(
    tmp_path: Path, mutation: str,
) -> None:
    directory = tmp_path / "bin"
    directory.mkdir()
    executable = _executable(directory / "ncu")
    with pytest.raises(immutable.ImmutableArtifactError):
        with immutable.open_retained_executable(executable):
            if mutation == "bytes":
                executable.write_bytes(b"mutated executable")
                executable.chmod(0o755)
            elif mutation == "replacement":
                executable.unlink()
                _executable(executable)
            elif mutation == "ancestor":
                moved = tmp_path / "moved"
                directory.rename(moved)
                directory.mkdir()
                _executable(directory / "ncu")
            elif mutation == "mode":
                executable.chmod(0o644)
            else:
                os.link(executable, tmp_path / "ncu-hardlink")


def test_retained_executable_preserves_body_error_when_revalidation_fails(
    tmp_path: Path,
) -> None:
    executable = _executable(tmp_path / "ncu")
    with pytest.raises(RuntimeError, match="primary body failure") as caught:
        with immutable.open_retained_executable(executable):
            executable.write_bytes(b"mutated executable")
            executable.chmod(0o755)
            raise RuntimeError("primary body failure")
    assert any(
        "retained executable revalidation failed" in note
        for note in getattr(caught.value, "__notes__", [])
    )


def _regular(path: Path, payload: bytes = b"fixture report") -> Path:
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


def test_retained_regular_binds_descriptor_path_snapshot_size_and_hash(
    tmp_path: Path,
) -> None:
    report = _regular(tmp_path / "report.ncu-rep")
    with immutable.open_retained_regular(report) as retained:
        assert retained.path == report.absolute()
        assert retained.sha256 == immutable.sha256_bytes(b"fixture report")
        assert retained.size_bytes == len(b"fixture report")
        assert len(retained.snapshot) == 7
        assert retained.proc_path == f"/proc/self/fd/{retained.descriptor}"
        assert os.read(retained.descriptor, 7) == b"fixture"
        retained.verify_unchanged()
        assert os.read(retained.descriptor, 7) == b"fixture"
    with pytest.raises(
        immutable.ImmutableArtifactError, match="retained regular file is closed",
    ):
        _ = retained.descriptor


@pytest.mark.parametrize("kind", ["leaf", "ancestor"])
def test_retained_regular_rejects_symlink_components(
    tmp_path: Path, kind: str,
) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    report = _regular(real_dir / "report.ncu-rep")
    if kind == "leaf":
        candidate = tmp_path / "report.ncu-rep"
        candidate.symlink_to(report)
    else:
        linked_dir = tmp_path / "linked"
        linked_dir.symlink_to(real_dir, target_is_directory=True)
        candidate = linked_dir / "report.ncu-rep"
    with pytest.raises(immutable.ImmutableArtifactError):
        immutable.open_retained_regular(candidate)


@pytest.mark.parametrize("kind", ["directory", "fifo", "hardlink"])
def test_retained_regular_rejects_nonregular_or_hardlinked_leaf(
    tmp_path: Path, kind: str,
) -> None:
    candidate = tmp_path / "report.ncu-rep"
    if kind == "directory":
        candidate.mkdir()
    elif kind == "fifo":
        os.mkfifo(candidate)
    else:
        _regular(candidate)
        os.link(candidate, tmp_path / "report-hardlink.ncu-rep")
    with pytest.raises(immutable.ImmutableArtifactError):
        immutable.open_retained_regular(candidate)


@pytest.mark.parametrize(
    "mutation",
    [
        "bytes", "same_byte_restore", "replacement", "ancestor", "mode",
        "hardlink", "truncate", "grow",
    ],
)
def test_retained_regular_rejects_post_open_identity_drift(
    tmp_path: Path, mutation: str,
) -> None:
    directory = tmp_path / "reports"
    directory.mkdir()
    payload = b"fixture report"
    report = _regular(directory / "report.ncu-rep", payload)
    writer = None
    if mutation in {"bytes", "same_byte_restore", "truncate", "grow"}:
        report.chmod(0o644)
        writer = os.open(report, os.O_WRONLY)
        report.chmod(0o444)
    try:
        with pytest.raises(immutable.ImmutableArtifactError):
            with immutable.open_retained_regular(report):
                if mutation == "bytes":
                    os.pwrite(writer, b"mutated report", 0)
                elif mutation == "same_byte_restore":
                    os.pwrite(writer, b"mutated report", 0)
                    os.pwrite(writer, payload, 0)
                elif mutation == "replacement":
                    report.unlink()
                    _regular(report, payload)
                elif mutation == "ancestor":
                    moved = tmp_path / "moved"
                    directory.rename(moved)
                    directory.mkdir()
                    _regular(directory / "report.ncu-rep", payload)
                elif mutation == "mode":
                    report.chmod(0o600)
                elif mutation == "hardlink":
                    os.link(report, tmp_path / "report-hardlink.ncu-rep")
                elif mutation == "truncate":
                    os.ftruncate(writer, 4)
                else:
                    os.pwrite(writer, payload + b" grown", 0)
    finally:
        if writer is not None:
            os.close(writer)



def test_open_retained_regular_rejects_drift_during_initial_hash(
    tmp_path: Path, monkeypatch,
) -> None:
    report = _regular(tmp_path / "report.ncu-rep")
    report.chmod(0o644)
    writer = os.open(report, os.O_WRONLY)
    report.chmod(0o444)
    original = immutable._hash_fd

    def mutate_after_hash(descriptor, *, allow_hardlinks=False):
        result = original(descriptor, allow_hardlinks=allow_hardlinks)
        os.pwrite(writer, b"changed after hash", 0)
        return result

    monkeypatch.setattr(immutable, "_hash_fd", mutate_after_hash)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="metadata changed during identity read",
        ):
            immutable.open_retained_regular(report)
    finally:
        os.close(writer)


def test_retained_regular_preserves_body_error_when_revalidation_fails(
    tmp_path: Path,
) -> None:
    report = _regular(tmp_path / "report.ncu-rep")
    with pytest.raises(RuntimeError, match="primary body failure") as caught:
        with immutable.open_retained_regular(report):
            report.chmod(0o644)
            report.write_bytes(b"mutated report")
            raise RuntimeError("primary body failure")
    assert any(
        "retained regular-file revalidation failed" in note
        for note in getattr(caught.value, "__notes__", [])
    )


def test_retained_regular_entry_failure_closes_descriptors(
    tmp_path: Path,
) -> None:
    report = _regular(tmp_path / "report.ncu-rep")
    report.chmod(0o644)
    writer = os.open(report, os.O_WRONLY)
    report.chmod(0o444)
    retained = immutable.open_retained_regular(report)
    descriptor = retained.descriptor
    try:
        os.pwrite(writer, b"mutated report", 0)
    finally:
        os.close(writer)
    with pytest.raises(
        immutable.ImmutableArtifactError, match="metadata changed",
    ):
        with retained:
            pytest.fail("body reached after failed entry verification")
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_retained_regular_close_is_idempotent(
    tmp_path: Path,
) -> None:
    retained = immutable.open_retained_regular(
        _regular(tmp_path / "report.ncu-rep")
    )
    descriptor = retained.descriptor
    retained.close()
    retained.close()
    with pytest.raises(OSError):
        os.fstat(descriptor)
    with pytest.raises(
        immutable.ImmutableArtifactError, match="retained regular file is closed",
    ):
        retained.verify_unchanged()


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


@pytest.mark.parametrize(
    "force_snapshot_closure",
    [False, True],
    ids=["pinned", "snapshot-closure"],
)
def test_directory_winner_rejects_same_byte_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_snapshot_closure: bool,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    if force_snapshot_closure:
        monkeypatch.setattr(immutable, "_SNAPSHOT_CLOSURE_THRESHOLD", 0)
    immutable.publish_directory(output_root, "artifact", material)
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
    assert (output_root / "artifact" / "value").read_bytes() == b"original"


def test_directory_winner_rejects_stage_reappearing_before_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    immutable.publish_directory(output_root, "artifact", material)
    original_proof = immutable._durability_proof_at
    callback_count = 0

    def reintroduce_stage(
        root_descriptor: int,
        tree_root_descriptor: int,
        proof_material: tuple[immutable.ImmutableFile, ...],
    ):
        snapshot = original_proof(
            root_descriptor, tree_root_descriptor, proof_material
        )
        os.mkdir(".artifact.staging", mode=0o755, dir_fd=root_descriptor)
        return snapshot

    monkeypatch.setattr(immutable, "_durability_proof_at", reintroduce_stage)

    def authorize(_root: Path, _descriptor: int) -> None:
        nonlocal callback_count
        callback_count += 1

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="stage reappeared beside concurrent final winner",
    ):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
            before_commit=authorize,
        )

    assert callback_count == 1
    assert (output_root / ".artifact.staging").is_dir()


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


def test_directory_preserves_final_when_final_authorization_fails(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    callback_count = 0

    def fail_on_final_authorization(_root: Path, _descriptor: int) -> None:
        nonlocal callback_count
        callback_count += 1
        if callback_count == 3:
            raise RuntimeError("late authorization failure")

    with pytest.raises(RuntimeError, match="late authorization failure"):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
            before_commit=fail_on_final_authorization,
        )

    assert callback_count == 3
    assert (output_root / "artifact" / "value").read_bytes() == b"original"
    assert not (output_root / ".artifact.staging").exists()


def test_directory_winner_fsync_order_and_reauthorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("nested/value", b"original")]
    immutable.publish_directory(output_root, "artifact", material)
    events: list[str] = []
    original_fsync = immutable._fsync_or_fail

    def record_fsync(descriptor: int, label: str) -> None:
        events.append(label)
        original_fsync(descriptor, label)

    monkeypatch.setattr(immutable, "_fsync_or_fail", record_fsync)

    def authorize(_root: Path, _descriptor: int) -> None:
        events.append("authorize")

    immutable.publish_directory(
        output_root,
        "artifact",
        material,
        before_commit=authorize,
    )

    assert events == [
        "authorize",
        "winner file nested/value",
        "winner directory nested",
        "winner tree root",
        "winner publication root",
        "authorize",
    ]


@pytest.mark.parametrize(
    ("failure_index", "failure_label"),
    [
        (1, "winner file nested/value"),
        (2, "winner directory nested"),
        (3, "winner tree root"),
        (4, "winner publication root"),
    ],
)
def test_directory_winner_fsync_eio_preserves_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
    failure_label: str,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("nested/value", b"original")]
    immutable.publish_directory(output_root, "artifact", material)
    original_fsync = immutable.os.fsync
    fsync_count = 0
    callback_count = 0

    def fail_at_selected_fsync(descriptor: int) -> None:
        nonlocal fsync_count
        fsync_count += 1
        if fsync_count == failure_index:
            raise OSError(errno.EIO, "injected winner fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(immutable.os, "fsync", fail_at_selected_fsync)

    def authorize(_root: Path, _descriptor: int) -> None:
        nonlocal callback_count
        callback_count += 1

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match=f"DURABILITY_UNPROVEN: {failure_label}",
    ):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
            before_commit=authorize,
        )

    assert fsync_count == failure_index
    assert callback_count == 1
    assert (output_root / "artifact" / "nested" / "value").read_bytes() == b"original"
    assert not (output_root / ".artifact.staging").exists()


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


def test_ncu_details_partial_suffix_is_reserved_for_regular_publication(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match=immutable.NAMESPACE_RESERVED_COMPONENT,
    ):
        immutable.publish_regular_file(
            tmp_path, "evidence.ncu-details-partial", b"payload",
        )


def test_ncu_details_partial_suffix_is_reserved_for_tree_material(
    tmp_path: Path,
) -> None:
    material = (
        immutable.ImmutableFile.from_bytes(
            "evidence.ncu-details-partial", b"payload",
        ),
    )
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match=immutable.NAMESPACE_RESERVED_COMPONENT,
    ):
        immutable.publish_directory(tmp_path, "artifact", material)
