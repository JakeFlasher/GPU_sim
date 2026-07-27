from __future__ import annotations

import errno
import os
import resource
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
        assert retained.read_bytes() == b"fixture report"
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


def _reader_tree(tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    root = tmp_path / "reader-root"
    branch = root / "branch"
    branch.mkdir(parents=True)
    payload = b"anchored reader payload"
    leaf = _regular(branch / "value", payload)
    return root, branch, leaf, payload


def _open_descriptor_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_pinned_directory_reader_supports_maximum_depth_without_fd_leak(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reader-root"
    root.mkdir()
    parts = tuple(f"d{index:02d}" for index in range(63))
    parent = root
    for part in parts:
        parent /= part
        parent.mkdir()
    _regular(parent / "value", b"maximum depth")
    relative_path = "/".join((*parts, "value"))
    assert len(relative_path.split("/")) == immutable._MAX_COMPONENTS

    baseline = _open_descriptor_count()
    with immutable.PinnedDirectoryReader(root) as reader:
        assert reader.read(relative_path) == b"maximum depth"
    assert _open_descriptor_count() == baseline


@pytest.mark.parametrize(
    "mutation",
    ["root_before", "root_after", "ancestor_after", "leaf_after"],
)
def test_pinned_directory_reader_rejects_named_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root, branch, leaf, payload = _reader_tree(tmp_path)
    original_read = immutable._read_fd
    mutated = False

    def replace_after_read(descriptor: int, *, allow_hardlinks: bool = False) -> bytes:
        nonlocal mutated
        result = original_read(descriptor, allow_hardlinks=allow_hardlinks)
        if mutation == "root_after":
            root.rename(tmp_path / "moved-root")
            root.mkdir()
        elif mutation == "ancestor_after":
            branch.rename(root / "moved-branch")
            branch.mkdir()
            _regular(branch / "value", payload)
        elif mutation == "leaf_after":
            leaf.unlink()
            _regular(leaf, payload)
        else:
            return result
        mutated = True
        return result

    monkeypatch.setattr(immutable, "_read_fd", replace_after_read)
    with immutable.PinnedDirectoryReader(root) as reader:
        if mutation == "root_before":
            root.rename(tmp_path / "moved-root")
            root.mkdir()
            mutated = True

            def unexpected_leaf_open(
                _root_descriptor: int, _relative_path: str,
            ) -> None:
                pytest.fail("leaf open reached after failed root precheck")

            monkeypatch.setattr(
                immutable, "_open_relative_regular_at", unexpected_leaf_open,
            )
        with pytest.raises(immutable.ImmutableArtifactError):
            reader.read("branch/value")
    assert mutated is True


def test_pinned_directory_reader_rejects_leaf_replacement_before_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _branch, leaf, payload = _reader_tree(tmp_path)
    original_open = immutable.os.open
    replaced = False

    def replace_after_leaf_open(
        path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "value" and (flags & os.O_DIRECTORY) == 0:
            leaf.unlink()
            _regular(leaf, payload)
            replaced = True
        return descriptor

    def unexpected_read(
        _descriptor: int, *, allow_hardlinks: bool = False,
    ) -> bytes:
        del allow_hardlinks
        pytest.fail("leaf bytes consumed before initial path verification")

    monkeypatch.setattr(immutable.os, "open", replace_after_leaf_open)
    monkeypatch.setattr(immutable, "_read_fd", unexpected_read)
    with immutable.PinnedDirectoryReader(root) as reader:
        with pytest.raises(immutable.ImmutableArtifactError):
            reader.read("branch/value")
    assert replaced is True


@pytest.mark.parametrize("mutation", ["hardlink", "truncate", "append"])
def test_pinned_directory_reader_rejects_mid_read_leaf_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root, _branch, leaf, _payload = _reader_tree(tmp_path)
    writer: int | None = None
    if mutation != "hardlink":
        leaf.chmod(0o644)
        writer = os.open(leaf, os.O_WRONLY)
        leaf.chmod(0o444)
    original_open = immutable._open_relative_regular_at
    original_read = immutable.os.read
    target_descriptor: int | None = None
    mutated = False

    def capture_leaf_descriptor(root_descriptor: int, relative_path: str):
        nonlocal target_descriptor
        pinned = original_open(root_descriptor, relative_path)
        target_descriptor = pinned.descriptor
        return pinned

    def mutate_after_first_block(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        block = original_read(descriptor, size)
        if descriptor == target_descriptor and block and not mutated:
            if mutation == "hardlink":
                os.link(leaf, tmp_path / "value-hardlink")
            elif mutation == "truncate":
                assert writer is not None
                os.ftruncate(writer, 4)
            else:
                assert writer is not None
                os.lseek(writer, 0, os.SEEK_END)
                os.write(writer, b" appended")
            mutated = True
        return block

    monkeypatch.setattr(
        immutable, "_open_relative_regular_at", capture_leaf_descriptor,
    )
    monkeypatch.setattr(immutable.os, "read", mutate_after_first_block)
    try:
        with immutable.PinnedDirectoryReader(root) as reader:
            with pytest.raises(
                immutable.ImmutableArtifactError,
                match="regular file changed during snapshot read",
            ):
                reader.read("branch/value")
    finally:
        if writer is not None:
            os.close(writer)
    assert target_descriptor is not None
    assert mutated is True


def test_pinned_directory_reader_failure_closes_all_owned_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _branch, _leaf, _payload = _reader_tree(tmp_path)
    original_open = immutable._open_relative_regular_at
    opened: list[int] = []

    def capture_open(root_descriptor: int, relative_path: str):
        pinned = original_open(root_descriptor, relative_path)
        opened.extend((pinned.descriptor, *pinned.directory_descriptors))
        return pinned

    def fail_read(_descriptor: int, *, allow_hardlinks: bool = False) -> bytes:
        del allow_hardlinks
        raise RuntimeError("injected anchored-read failure")

    monkeypatch.setattr(immutable, "_open_relative_regular_at", capture_open)
    monkeypatch.setattr(immutable, "_read_fd", fail_read)
    reader = immutable.PinnedDirectoryReader(root)
    root_descriptors = reader._root.descriptors
    with pytest.raises(RuntimeError, match="injected anchored-read failure"):
        with reader:
            reader.read("branch/value")

    assert opened
    for descriptor in (*opened, *root_descriptors):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def _snapshot_material(
    root: Path, payloads: dict[str, bytes],
) -> tuple[immutable.ImmutableFile, ...]:
    material = []
    root.mkdir()
    for relative_path, payload in payloads.items():
        _regular(root / relative_path, payload)
        material.append(immutable.ImmutableFile.from_bytes(relative_path, payload))
    root.chmod(0o555)
    return tuple(material)


def test_snapshot_closure_scales_to_twenty_thousand_files_under_rlimit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if old_limit[1] != resource.RLIM_INFINITY and old_limit[1] < 256:
        pytest.skip("RLIMIT_NOFILE hard limit is below the required harness bound")
    root = tmp_path / "large-tree"
    root.mkdir()
    empty_sha256 = immutable.sha256_bytes(b"")
    material = []
    for index in range(20_000):
        relative_path = f"f{index:05d}"
        descriptor = os.open(
            root / relative_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        os.close(descriptor)
        material.append(immutable.ImmutableFile(
            relative_path, empty_sha256, 0, payload=b"",
        ))
    root.chmod(0o555)

    original_os_open = immutable.os.open
    original_dup = immutable.os.dup
    original_scandir = immutable.os.scandir
    original_hash = immutable._hash_fd
    hash_calls = 0
    peak_descriptors = 0

    def record_peak() -> None:
        nonlocal peak_descriptors
        peak_descriptors = max(peak_descriptors, _open_descriptor_count())

    def track_os_open(*args, **kwargs):
        descriptor = original_os_open(*args, **kwargs)
        record_peak()
        return descriptor

    def track_dup(descriptor: int) -> int:
        duplicate = original_dup(descriptor)
        record_peak()
        return duplicate

    def track_scandir(path):
        entries = original_scandir(path)
        record_peak()
        return entries

    def count_hash(
        descriptor: int, *, allow_hardlinks: bool = False,
    ) -> tuple[str, int, int]:
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(descriptor, allow_hardlinks=allow_hardlinks)

    monkeypatch.setattr(immutable.os, "open", track_os_open)
    monkeypatch.setattr(immutable.os, "dup", track_dup)
    monkeypatch.setattr(immutable.os, "scandir", track_scandir)
    monkeypatch.setattr(immutable, "_hash_fd", count_hash)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, old_limit[1]))
        process_baseline = _open_descriptor_count()
        with immutable.PinnedDirectoryReader(root) as reader:
            verifier_baseline = _open_descriptor_count()
            peak_descriptors = verifier_baseline
            files, directories = immutable._verify_published_files_at(
                reader.descriptor, material,
            )
        assert len(files) == 20_000
        assert set(directories) == {"."}
        assert hash_calls == 20_000
        assert 0 <= peak_descriptors - verifier_baseline <= 16
        assert peak_descriptors < 256
        assert _open_descriptor_count() == process_baseline
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, old_limit)
        root.chmod(0o755)


def test_snapshot_closure_rejects_depth_above_bound_without_fd_leak(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deep-tree"
    root.mkdir()
    parent = root
    for index in range(65):
        parent /= f"d{index:02d}"
        parent.mkdir()

    with immutable.PinnedDirectoryReader(root) as reader:
        baseline = _open_descriptor_count()
        with pytest.raises(immutable.ImmutableArtifactError):
            immutable._verify_snapshot_closure_at(
                reader.descriptor,
                {},
                require_readonly_directories=False,
            )
        assert _open_descriptor_count() == baseline


def test_snapshot_closure_failure_closes_transient_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "failure-tree"
    material = _snapshot_material(root, {"a": b"a", "b": b"b"})
    expected = immutable._material_map(material)
    original_verify = immutable._verify_open_file
    calls = 0

    def fail_second_hash(
        descriptor: int, file: immutable.ImmutableFile,
    ) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected scalable verification failure")
        return original_verify(descriptor, file)

    monkeypatch.setattr(immutable, "_verify_open_file", fail_second_hash)
    try:
        with immutable.PinnedDirectoryReader(root) as reader:
            baseline = _open_descriptor_count()
            with pytest.raises(
                RuntimeError, match="injected scalable verification failure",
            ):
                immutable._verify_snapshot_closure_at(
                    reader.descriptor,
                    expected,
                    require_readonly_directories=True,
                )
            assert _open_descriptor_count() == baseline
    finally:
        root.chmod(0o755)


def test_snapshot_monitor_close_failure_is_typed_and_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_close = immutable.os.close
    close_only = immutable._DirectoryMutationMonitor()
    close_only_descriptor = close_only.descriptor
    active = immutable._DirectoryMutationMonitor()
    active_descriptor = active.descriptor

    def fail_monitor_close(descriptor: int) -> None:
        if descriptor in {close_only_descriptor, active_descriptor}:
            raise OSError("injected monitor close failure")
        original_close(descriptor)

    monkeypatch.setattr(immutable.os, "close", fail_monitor_close)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match=immutable.DURABILITY_UNPROVEN,
        ):
            close_only.close()
        with pytest.raises(RuntimeError, match="primary verification failure") as raised:
            try:
                raise RuntimeError("primary verification failure")
            finally:
                active.close()
        assert any(
            immutable.DURABILITY_UNPROVEN in note
            for note in getattr(raised.value, "__notes__", ())
        )
    finally:
        original_close(close_only_descriptor)
        original_close(active_descriptor)


def test_snapshot_closure_rejects_restore_after_early_final_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "restore-tree"
    material = _snapshot_material(root, {"a": b"a", "z": b"z"})
    expected = immutable._material_map(material)
    early = root / "a"
    early.chmod(0o644)
    writer = os.open(early, os.O_WRONLY)
    early.chmod(0o444)
    original_open = immutable._open_relative_regular_at
    opens: dict[str, int] = {}
    mutated = False

    def mutate_after_early_final_reopen(
        root_descriptor: int, relative_path: str,
    ):
        nonlocal mutated
        pinned = original_open(root_descriptor, relative_path)
        opens[relative_path] = opens.get(relative_path, 0) + 1
        if relative_path == "z" and opens[relative_path] == 2:
            os.pwrite(writer, b"X", 0)
            os.pwrite(writer, b"a", 0)
            mutated = True
        return pinned

    monkeypatch.setattr(
        immutable, "_open_relative_regular_at", mutate_after_early_final_reopen,
    )
    try:
        with immutable.PinnedDirectoryReader(root) as reader:
            with pytest.raises(
                immutable.ImmutableArtifactError,
                match="changed before verification closure",
            ):
                immutable._verify_snapshot_closure_at(
                    reader.descriptor,
                    expected,
                    require_readonly_directories=True,
                )
    finally:
        os.close(writer)
        root.chmod(0o755)
    assert mutated is True
    assert early.read_bytes() == b"a"


def test_snapshot_closure_rejects_write_after_final_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "final-snapshot-tree"
    material = _snapshot_material(root, {"a": b"a", "z": b"z"})
    expected = immutable._material_map(material)
    leaf = root / "a"
    leaf.chmod(0o644)
    writer = os.open(leaf, os.O_WRONLY)
    leaf.chmod(0o444)
    original_snapshot = immutable._snapshot_tree_descriptor
    calls = 0
    mutated = False
    observed = b""

    def mutate_after_final_snapshot(root_descriptor: int):
        nonlocal calls, mutated
        calls += 1
        snapshot = original_snapshot(root_descriptor)
        if calls == 3:
            os.pwrite(writer, b"X", 0)
            mutated = True
        return snapshot

    monkeypatch.setattr(
        immutable, "_snapshot_tree_descriptor", mutate_after_final_snapshot,
    )
    try:
        with immutable.PinnedDirectoryReader(root) as reader:
            with pytest.raises(
                immutable.ImmutableArtifactError,
                match="changed during bounded verification",
            ):
                immutable._verify_snapshot_closure_at(
                    reader.descriptor,
                    expected,
                    require_readonly_directories=True,
                )
        observed = leaf.read_bytes()
    finally:
        os.pwrite(writer, b"a", 0)
        os.close(writer)
        root.chmod(0o755)
    assert calls == 3
    assert mutated is True
    assert observed == b"X"


def test_snapshot_closure_rejects_external_hardlink_after_final_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "hardlink-snapshot-tree"
    material = _snapshot_material(root, {"a": b"a", "z": b"z"})
    expected = immutable._material_map(material)
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = outside / "alias"
    original_snapshot = immutable._snapshot_tree_descriptor
    calls = 0
    linked = False
    observed_links = 0

    def link_after_final_snapshot(root_descriptor: int):
        nonlocal calls, linked
        calls += 1
        snapshot = original_snapshot(root_descriptor)
        if calls == 3:
            os.link(root / "a", alias)
            linked = True
        return snapshot

    monkeypatch.setattr(
        immutable, "_snapshot_tree_descriptor", link_after_final_snapshot,
    )
    try:
        with immutable.PinnedDirectoryReader(root) as reader:
            with pytest.raises(
                immutable.ImmutableArtifactError,
                match="changed during bounded verification",
            ):
                immutable._verify_snapshot_closure_at(
                    reader.descriptor,
                    expected,
                    require_readonly_directories=True,
                )
        observed_links = (root / "a").stat().st_nlink
    finally:
        alias.unlink(missing_ok=True)
        root.chmod(0o755)
    assert calls == 3
    assert linked is True
    assert observed_links == 2


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


def test_regular_publication_can_preserve_execute_capability(tmp_path: Path) -> None:
    output = immutable.publish_regular_file(
        tmp_path, "executable.bin", b"payload", executable=True,
    )
    assert output.stat().st_mode & 0o111
    assert not output.stat().st_mode & 0o222


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
