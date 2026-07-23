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


def test_seal_sets_modes_and_replay_is_byte_identical(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
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
    second = _seal(root, receipt)
    assert second == first
    assert receipt.read_bytes() == receipt_bytes


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
