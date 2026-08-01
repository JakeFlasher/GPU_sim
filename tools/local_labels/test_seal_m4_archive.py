from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import seal_m4_archive as sealer  # noqa: E402


CHUNKS = tuple(f"chunk-{index:02d}" for index in range(18))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "external-m4"
    (root / "chunks").mkdir(parents=True)
    for chunk_id in CHUNKS:
        chunk_root = root / "chunks" / chunk_id
        chunk_root.mkdir()
        (chunk_root / "cells").mkdir()
        (chunk_root / "cells" / "evidence.bin").write_bytes(
            chunk_id.encode("ascii")
        )
        (chunk_root / "DONE.json").write_text(
            json.dumps({
                "chunk_id": chunk_id,
                "output_root": str(chunk_root),
            }) + "\n",
            encoding="utf-8",
        )
    return root, tmp_path / "seal.json"


def _verify(_manifest: dict, **_kwargs: object) -> list[str]:
    return []


def _seal(root: Path, receipt: Path, **kwargs: object) -> dict:
    return sealer.seal_m4_archive(
        root,
        receipt_path=receipt,
        repo_root=receipt.parent / "repository",
        expected_chunk_ids=CHUNKS,
        done_verifier=_verify,
        **kwargs,
    )


def _tree_content_identity(root: Path) -> dict[str, tuple[object, ...]]:
    result: dict[str, tuple[object, ...]] = {}
    for path in (root, *root.rglob("*")):
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        result[relative] = (
            stat.S_IFMT(metadata.st_mode),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_nlink,
            metadata.st_size,
            path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None,
        )
    return result


def test_seal_sets_modes_and_replay_is_byte_identical(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    content_identity = _tree_content_identity(root)
    first = _seal(root, receipt)
    receipt_bytes = receipt.read_bytes()
    assert first["status"] == "complete"
    assert first["chunk_count"] == 18
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444
        for path in root.rglob("*")
        if path.is_file()
    )
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o555
        for path in [root, *[p for p in root.rglob("*") if p.is_dir()]]
    )
    assert _tree_content_identity(root) == content_identity
    second = _seal(root, receipt)
    assert second == first
    assert receipt.read_bytes() == receipt_bytes


def test_root_mode_seal_rejects_transient_root_exchange_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _fixture(tmp_path)
    content_identity = _tree_content_identity(root)
    root_identity = (root.stat().st_dev, root.stat().st_ino)
    displaced = tmp_path / "external-m4-original"
    original_fchmod = os.fchmod
    exchanged = False

    def exchange_after_root_mode_change(descriptor: int, mode: int) -> None:
        nonlocal exchanged
        original_fchmod(descriptor, mode)
        opened = os.fstat(descriptor)
        if (
            not exchanged
            and mode == 0o555
            and (opened.st_dev, opened.st_ino) == root_identity
        ):
            exchanged = True
            root.rename(displaced)
            root.mkdir()
            root.rmdir()
            displaced.rename(root)

    monkeypatch.setattr(os, "fchmod", exchange_after_root_mode_change)
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)

    assert exchanged is True
    assert error.value.code == "M4_ARCHIVE_MODE_SEAL"
    assert "outside authorized root mode seal" in error.value.detail
    assert _tree_content_identity(root) == content_identity
    assert not receipt.exists()


def test_root_mode_seal_rejects_post_authorization_unseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _fixture(tmp_path)
    content_identity = _tree_content_identity(root)
    original = (
        sealer.immutable.PinnedDirectoryReader.seal_root_directory_mode
    )

    def seal_then_unseal(
        reader: sealer.immutable.PinnedDirectoryReader,
    ) -> None:
        original(reader)
        os.fchmod(reader.descriptor, 0o755)

    monkeypatch.setattr(
        sealer.immutable.PinnedDirectoryReader,
        "seal_root_directory_mode",
        seal_then_unseal,
    )
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)

    assert error.value.code == "M4_ARCHIVE_MODE_SEAL"
    assert _tree_content_identity(root) == content_identity
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert not receipt.exists()


def test_replay_rejects_writable_root_without_changing_receipt(
    tmp_path: Path,
) -> None:
    root, receipt = _fixture(tmp_path)
    _seal(root, receipt)
    receipt_bytes = receipt.read_bytes()
    root.chmod(0o755)

    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)

    assert error.value.code == "M4_ARCHIVE_MODE_SEAL"
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert receipt.read_bytes() == receipt_bytes


def test_racing_foreign_receipt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _fixture(tmp_path)

    def plant_foreign_receipt(
        parent: Path,
        name: str,
        payload_or_source: bytes | Path | sealer.immutable.ImmutableFile,
        **_kwargs: object,
    ) -> Path:
        assert isinstance(payload_or_source, bytes)
        value = json.loads(payload_or_source)
        value["archive_root"] = str(tmp_path / "foreign-archive")
        value["receipt_sha256"] = sealer.immutable.model_seal(
            value,
            "receipt_sha256",
        )
        planted = Path(parent) / name
        planted.write_bytes(sealer.immutable.pretty_json_bytes(value))
        planted.chmod(0o444)
        raise FileExistsError(name)

    monkeypatch.setattr(
        sealer.immutable,
        "publish_regular_file",
        plant_foreign_receipt,
    )
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)

    assert error.value.code == "M4_ARCHIVE_RECEIPT_COLLISION"
    assert json.loads(receipt.read_bytes())["archive_root"] == str(
        tmp_path / "foreign-archive"
    )


def test_complete_receipt_partial_is_reused_after_precommit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _fixture(tmp_path)
    partial = receipt.with_name(f".{receipt.name}.m6-partial")
    original = sealer.immutable._rename_noreplace_at
    timestamps = tuple(
        f"2026-08-01T00:00:0{index}Z" for index in range(6)
    )
    timestamp_iterator = iter(timestamps)
    monkeypatch.setattr(sealer, "_now", lambda: next(timestamp_iterator))
    failed_once = False

    def fail_first_receipt_commit(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal failed_once
        if destination_name == receipt.name and not failed_once:
            failed_once = True
            raise sealer.immutable.ImmutableArtifactError(
                "injected receipt precommit failure"
            )
        original(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        sealer.immutable,
        "_rename_noreplace_at",
        fail_first_receipt_commit,
    )
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)

    assert failed_once is True
    assert error.value.code == "M4_ARCHIVE_RECEIPT_PUBLICATION"
    assert not receipt.exists()
    partial_bytes = partial.read_bytes()
    partial_value = json.loads(partial_bytes)
    assert partial_value["sealed_at_utc"] == timestamps[1]
    partial_metadata = partial.stat()
    assert stat.S_IMODE(partial_metadata.st_mode) == 0o444

    partial.chmod(0o644)
    with pytest.raises(sealer.M4ArchiveSealError) as writable_error:
        _seal(root, receipt)
    assert writable_error.value.code == "M4_ARCHIVE_RECEIPT_COLLISION"
    assert not receipt.exists()
    assert partial.read_bytes() == partial_bytes
    partial.chmod(0o444)

    recovered = _seal(root, receipt)

    assert not partial.exists()
    assert receipt.read_bytes() == partial_bytes
    assert (receipt.stat().st_dev, receipt.stat().st_ino) == (
        partial_metadata.st_dev,
        partial_metadata.st_ino,
    )
    assert recovered == partial_value
    assert recovered["sealed_at_utc"] == timestamps[1]
    replay_bytes = receipt.read_bytes()
    assert _seal(root, receipt) == recovered
    assert receipt.read_bytes() == replay_bytes


@pytest.mark.parametrize(
    "candidate_kind",
    (
        "malformed",
        "oversized",
        "symlink",
        "hardlink",
        "unsealed",
        "schema_invalid",
        "semantic_mismatch",
    ),
)
def test_invalid_receipt_partial_is_a_typed_collision(
    tmp_path: Path,
    candidate_kind: str,
) -> None:
    root, receipt = _fixture(tmp_path)
    partial = receipt.with_name(f".{receipt.name}.m6-partial")

    if candidate_kind == "symlink":
        target = tmp_path / "symlink-target"
        target.write_bytes(b"{}\n")
        partial.symlink_to(target)
    elif candidate_kind == "hardlink":
        target = tmp_path / "hardlink-target"
        target.write_bytes(b"{}\n")
        target.chmod(0o444)
        os.link(target, partial)
    elif candidate_kind == "semantic_mismatch":
        foreign_root, foreign_receipt = _fixture(tmp_path / "foreign")
        _seal(foreign_root, foreign_receipt)
        partial.write_bytes(foreign_receipt.read_bytes())
        partial.chmod(0o444)
    elif candidate_kind == "schema_invalid":
        candidate = {"schema_version": 1}
        candidate["receipt_sha256"] = sealer.immutable.model_seal(
            candidate,
            "receipt_sha256",
        )
        partial.write_bytes(sealer.immutable.pretty_json_bytes(candidate))
        partial.chmod(0o444)
    else:
        payload = {
            "malformed": b"{\n",
            "oversized": b"x" * (1 << 20),
            "unsealed": b'{"schema_version": 1}\n',
        }[candidate_kind]
        partial.write_bytes(payload)
        partial.chmod(0o444)

    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)

    assert error.value.code == "M4_ARCHIVE_RECEIPT_COLLISION"
    assert not receipt.exists()


def test_archive_root_mutation_during_receipt_precommit_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _fixture(tmp_path)
    original = sealer._publish_receipt
    mutated = False

    def unseal_during_publication(
        path: Path,
        value: dict,
        **kwargs: object,
    ) -> dict:
        nonlocal mutated
        root.chmod(0o755)
        mutated = True
        return original(path, value, **kwargs)

    monkeypatch.setattr(sealer, "_publish_receipt", unseal_during_publication)
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)

    assert mutated is True
    assert error.value.code == "M4_ARCHIVE_TREE_CHANGED"
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert not receipt.exists()


def test_archive_file_mutation_during_receipt_commit_retires_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _fixture(tmp_path)
    evidence = root / "chunks" / CHUNKS[0] / "cells" / "evidence.bin"
    original = sealer.immutable._rename_noreplace_at
    mutated = False

    def commit_then_mutate_archive(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal mutated
        original(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )
        if destination_name == receipt.name and not mutated:
            evidence.chmod(0o644)
            evidence.write_bytes(b"mutated during receipt commit")
            mutated = True

    monkeypatch.setattr(
        sealer.immutable,
        "_rename_noreplace_at",
        commit_then_mutate_archive,
    )
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)

    assert mutated is True
    assert error.value.code == "M4_ARCHIVE_TREE_CHANGED"
    assert not receipt.exists()
    retired = [
        path for path in tmp_path.iterdir()
        if path.name.startswith(".failed-regular-")
        and not path.name.endswith(".salvaging")
    ]
    assert len(retired) == 1


def test_final_tree_failure_happens_before_receipt_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _fixture(tmp_path)
    original = sealer._scan_tree
    calls = 0

    def fail_final_scan(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise sealer.M4ArchiveSealError(
                "M4_ARCHIVE_TREE_CHANGED",
                "injected final closure failure",
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(sealer, "_scan_tree", fail_final_scan)
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)

    assert calls == 4
    assert error.value.code == "M4_ARCHIVE_TREE_CHANGED"
    assert not receipt.exists()


def test_done_verifier_receives_every_strict_production_flag(
    tmp_path: Path,
) -> None:
    root, receipt = _fixture(tmp_path)
    calls: list[tuple[str, dict[str, object]]] = []

    def observe(manifest: dict, **kwargs: object) -> list[str]:
        calls.append((manifest["chunk_id"], kwargs))
        return []

    sealer.seal_m4_archive(
        root,
        receipt_path=receipt,
        repo_root=tmp_path / "repository",
        expected_chunk_ids=CHUNKS,
        done_verifier=observe,
    )

    assert tuple(chunk_id for chunk_id, _kwargs in calls) == CHUNKS
    assert all(
        kwargs
        == {
            "repo_root": tmp_path / "repository",
            "rehash_references": True,
            "require_current_head": True,
            "require_clean_tree": True,
        }
        for _chunk_id, kwargs in calls
    )


def test_done_verification_happens_before_mode_mutation(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)

    def refuse(_manifest: dict, **_kwargs: object) -> list[str]:
        return ["not sealed"]

    with pytest.raises(sealer.M4ArchiveSealError) as error:
        sealer.seal_m4_archive(
            root,
            receipt_path=receipt,
            repo_root=tmp_path / "repository",
            expected_chunk_ids=CHUNKS,
            done_verifier=refuse,
        )
    assert error.value.code == "M4_ARCHIVE_DONE_UNVERIFIED"
    assert any(
        stat.S_IMODE(path.stat().st_mode) == 0o755
        for path in root.rglob("*")
        if path.is_dir()
    )
    assert not receipt.exists()


def test_missing_and_extra_chunk_are_refused_without_cleanup(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    (root / "chunks" / CHUNKS[-1]).rename(root / "chunks" / "missing")
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)
    assert error.value.code == "M4_ARCHIVE_CHUNK_INVENTORY"
    assert (root / "chunks" / "missing").exists()

    root, receipt = _fixture(tmp_path / "second")
    extra = root / "chunks" / "extra"
    extra.mkdir()
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)
    assert error.value.code == "M4_ARCHIVE_CHUNK_INVENTORY"
    assert extra.exists()


def test_links_and_nonregular_entries_are_rejected(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    target = root / "chunks" / CHUNKS[0] / "DONE.json"
    target.unlink()
    target.symlink_to("../chunk-01/DONE.json")
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)
    assert error.value.code in {"M4_ARCHIVE_SYMLINK", "M4_ARCHIVE_TREE_SNAPSHOT"}

    root, receipt = _fixture(tmp_path / "hardlink")
    target = root / "chunks" / CHUNKS[0] / "linked.bin"
    os.link(root / "chunks" / CHUNKS[1] / "cells" / "evidence.bin", target)
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)
    assert error.value.code == "M4_ARCHIVE_HARDLINK"


def test_mutated_bytes_after_seal_are_not_hidden_by_receipt(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    _seal(root, receipt)
    evidence = root / "chunks" / CHUNKS[0] / "cells" / "evidence.bin"
    evidence.chmod(0o644)
    evidence.write_bytes(b"tampered")
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        _seal(root, receipt)
    assert error.value.code in {"M4_ARCHIVE_TREE_CHANGED", "M4_ARCHIVE_RECEIPT_COLLISION"}
    assert evidence.read_bytes() == b"tampered"


def test_receipt_inside_archive_is_refused(tmp_path: Path) -> None:
    root, _receipt = _fixture(tmp_path)
    with pytest.raises(sealer.M4ArchiveSealError) as error:
        sealer.seal_m4_archive(
            root,
            receipt_path=root / "seal.json",
            repo_root=tmp_path / "repository",
            expected_chunk_ids=CHUNKS,
            done_verifier=_verify,
        )
    assert error.value.code == "M4_ARCHIVE_RECEIPT_OVERLAP"
