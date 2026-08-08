from __future__ import annotations

import errno
import os
import resource
import stat
import sys
from contextlib import nullcontext
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


@pytest.mark.parametrize("kind", ("executable", "regular"))
def test_retained_file_rejects_restored_ancestor_generation(
    tmp_path: Path,
    kind: str,
) -> None:
    directory = tmp_path / "source"
    directory.mkdir()
    leaf = directory / "artifact"
    if kind == "executable":
        _executable(leaf)
        opener = immutable.open_retained_executable
    else:
        _regular(leaf)
        opener = immutable.open_retained_regular

    moved = tmp_path / "source-original"
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="ancestor changed during use",
    ):
        with opener(leaf):
            directory.rename(moved)
            directory.mkdir()
            if kind == "executable":
                _executable(leaf, b"replacement")
            else:
                _regular(leaf, b"replacement")
            leaf.unlink()
            directory.rmdir()
            moved.rename(directory)


def test_retained_path_monitor_rejects_ancestor_swap_while_arming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "source"
    directory.mkdir()
    leaf = _regular(directory / "artifact", b"identity")
    pinned = immutable._open_regular(leaf)
    target_parent = pinned.directory_descriptors[-2]
    moved = tmp_path / "source-original"
    original_watch = immutable._DirectoryMutationMonitor.watch_directory
    swapped = False

    def watch_then_swap(self, descriptor: int) -> int:
        nonlocal swapped
        watch = original_watch(self, descriptor)
        if descriptor == target_parent and not swapped:
            swapped = True
            directory.rename(moved)
            directory.mkdir()
            directory.rmdir()
            moved.rename(directory)
        return watch

    monkeypatch.setattr(
        immutable._DirectoryMutationMonitor,
        "watch_directory",
        watch_then_swap,
    )
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="changed while arming",
        ):
            immutable._arm_retained_path_monitor(pinned)
    finally:
        pinned.close()
    assert swapped is True


@pytest.mark.parametrize("kind", ("executable", "regular"))
def test_open_retained_file_watches_ancestors_before_leaf_watch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    directory = tmp_path / "source"
    directory.mkdir()
    leaf = directory / "artifact"
    if kind == "executable":
        _executable(leaf, b"identity")
        opener = immutable.open_retained_executable
    else:
        _regular(leaf, b"identity")
        opener = immutable.open_retained_regular

    moved = tmp_path / "source-original"
    original_watch = immutable._DirectoryMutationMonitor.watch_regular
    swapped = False

    def swap_before_leaf_watch(self, descriptor: int) -> int:
        nonlocal swapped
        if not swapped:
            swapped = True
            directory.rename(moved)
            directory.mkdir()
            directory.rmdir()
            moved.rename(directory)
        return original_watch(self, descriptor)

    monkeypatch.setattr(
        immutable._DirectoryMutationMonitor,
        "watch_regular",
        swap_before_leaf_watch,
    )
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="ancestor changed during use",
    ):
        opener(leaf)
    assert swapped is True


@pytest.mark.parametrize("kind", ("executable", "regular"))
def test_open_retained_file_rejects_ancestor_swap_during_initial_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    directory = tmp_path / "source"
    directory.mkdir()
    leaf = directory / "artifact"
    payload = b"initial identity payload"
    if kind == "executable":
        _executable(leaf, payload)
        opener = immutable.open_retained_executable
    else:
        _regular(leaf, payload)
        opener = immutable.open_retained_regular

    original_hash = immutable._hash_fd
    moved = tmp_path / "source-original"
    swapped = False

    def hash_while_ancestor_is_replaced(
        descriptor: int,
        *,
        allow_hardlinks: bool = False,
        max_bytes: int | None = None,
    ) -> tuple[str, int, int]:
        nonlocal swapped
        if swapped:
            return original_hash(
                descriptor,
                allow_hardlinks=allow_hardlinks,
                max_bytes=max_bytes,
            )
        swapped = True
        directory.rename(moved)
        directory.mkdir()
        try:
            return original_hash(
                descriptor,
                allow_hardlinks=allow_hardlinks,
                max_bytes=max_bytes,
            )
        finally:
            directory.rmdir()
            moved.rename(directory)

    monkeypatch.setattr(immutable, "_hash_fd", hash_while_ancestor_is_replaced)
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="ancestor changed during use",
    ):
        opener(leaf)
    assert swapped is True
    assert leaf.read_bytes() == payload


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

    def mutate_after_hash(
        descriptor, *, allow_hardlinks=False, max_bytes=None,
    ):
        result = original(
            descriptor,
            allow_hardlinks=allow_hardlinks,
            max_bytes=max_bytes,
        )
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


def test_retained_executable_entry_failure_closes_descriptors(
    tmp_path: Path,
) -> None:
    executable = _executable(tmp_path / "ncu")
    writer = os.open(executable, os.O_WRONLY)
    executable.chmod(0o555)
    retained = immutable.open_retained_executable(executable)
    descriptor = retained.descriptor
    try:
        os.pwrite(writer, b"mutated executable", 0)
    finally:
        os.close(writer)
    with pytest.raises(
        immutable.ImmutableArtifactError, match="(metadata|generation) changed",
    ):
        with retained:
            pytest.fail("body reached after failed entry verification")
    with pytest.raises(OSError):
        os.fstat(descriptor)


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
        immutable.ImmutableArtifactError, match="(metadata|generation) changed",
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


def test_open_retained_regular_enforces_strict_initial_byte_ceiling(
    tmp_path: Path,
) -> None:
    report = _regular(tmp_path / "report.ncu-rep", b"12345")
    before = _open_descriptor_count()

    with pytest.raises(
        immutable.ImmutableArtifactLimitError,
        match="exceeds bounded snapshot limit",
    ):
        immutable.open_retained_regular(report, max_bytes=4)

    assert _open_descriptor_count() == before


def test_retained_regular_read_enforces_strict_byte_ceiling(
    tmp_path: Path,
) -> None:
    report = _regular(tmp_path / "report.ncu-rep", b"12345")

    with immutable.open_retained_regular(report, max_bytes=5) as retained:
        with pytest.raises(
            immutable.ImmutableArtifactLimitError,
            match="exceeds bounded snapshot limit",
        ):
            retained.read_bytes(max_bytes=4)
        assert retained.read_bytes(max_bytes=5) == b"12345"


def _reader_tree(tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    root = tmp_path / "reader-root"
    branch = root / "branch"
    branch.mkdir(parents=True)
    payload = b"anchored reader payload"
    leaf = _regular(branch / "value", payload)
    return root, branch, leaf, payload


def _open_descriptor_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_pinned_directory_reader_watches_each_name_before_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = tmp_path / "shared"
    root = shared / "target"
    root.mkdir(parents=True)
    _regular(root / "value", b"identity")
    moved = shared / "target-original"
    shared_identity = immutable._node_identity(shared.stat())
    original_watch = immutable._DirectoryMutationMonitor.watch_directory
    swapped = False

    def watch_then_swap(self, descriptor: int) -> int:
        nonlocal swapped
        watch = original_watch(self, descriptor)
        if (
            not swapped
            and immutable._node_identity(os.fstat(descriptor))
            == shared_identity
        ):
            swapped = True
            root.rename(moved)
            root.mkdir()
            root.rmdir()
            moved.rename(root)
        return watch

    monkeypatch.setattr(
        immutable._DirectoryMutationMonitor,
        "watch_directory",
        watch_then_swap,
    )
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="retained directory ancestor changed during use",
    ):
        immutable.PinnedDirectoryReader(root)
    assert swapped is True


def test_protected_input_monitor_ignores_unrelated_ancestor_churn_while_arming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = tmp_path / "shared"
    root = shared / "target"
    root.mkdir(parents=True)
    _regular(root / "value", b"identity")
    unrelated = shared / "unrelated-garbage"
    original_watch = immutable._DirectoryMutationMonitor.watch_regular
    mutated = False

    def watch_then_churn(self, descriptor: int) -> int:
        nonlocal mutated
        watch = original_watch(self, descriptor)
        if not mutated:
            mutated = True
            unrelated.mkdir()
            unrelated.rmdir()
        return watch

    monkeypatch.setattr(
        immutable._DirectoryMutationMonitor,
        "watch_regular",
        watch_then_churn,
    )
    with immutable.PinnedPathSetMonitor(
        root,
        regular_paths=("value",),
        recursive_directories=(),
    ):
        pass
    assert mutated is True


def test_protected_input_monitor_revalidates_nested_ancestor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "protected"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    _regular(nested / "value", b"identity")
    monitor = immutable.PinnedPathSetMonitor(
        root,
        regular_paths=("a/b/value",),
        recursive_directories=(),
    )
    moved = root / "a" / "b-original"
    try:
        monkeypatch.setattr(
            monitor._monitor,
            "read_events",
            lambda: immutable._MutationDrainResult(()),
        )
        nested.rename(moved)
        nested.mkdir()
        _regular(nested / "value", b"identity")

        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="protected input nested ancestor changed",
        ):
            monitor.verify_unchanged()
    finally:
        monitor.close()


def test_protected_input_monitor_entry_failure_closes_descriptors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "protected"
    root.mkdir()
    leaf = _regular(root / "value", b"initial")
    leaf.chmod(0o644)
    writer = os.open(leaf, os.O_WRONLY)
    leaf.chmod(0o444)
    baseline = _open_descriptor_count()
    monitor = immutable.PinnedPathSetMonitor(
        root,
        regular_paths=("value",),
        recursive_directories=(),
    )
    try:
        os.pwrite(writer, b"mutated", 0)

        with pytest.raises(
            immutable.ImmutableArtifactError, match="protected input tree changed",
        ):
            with monitor:
                pytest.fail("body reached after failed entry verification")

        assert monitor._root is None
        assert monitor._monitor is None
        assert _open_descriptor_count() == baseline
    finally:
        os.close(writer)


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


def _authorized_root_mode_events(
    reader: immutable.PinnedDirectoryReader,
) -> tuple[tuple[int, int, bytes], tuple[int, int, bytes]]:
    state = reader._path_monitor
    parent_watch, root_name = state.protected_names[-1]
    return (
        (
            state.root_watch,
            immutable._INOTIFY_ROOT_DIRECTORY_ATTRIB,
            b"",
        ),
        (
            parent_watch,
            immutable._INOTIFY_ROOT_DIRECTORY_ATTRIB,
            root_name,
        ),
    )


def test_pinned_directory_reader_seals_only_root_directory_mode(
    tmp_path: Path,
) -> None:
    root, branch, leaf, payload = _reader_tree(tmp_path)
    root_before = immutable._node_identity(root.stat())
    branch_before = immutable._node_identity(branch.stat())

    with immutable.PinnedDirectoryReader(root) as reader:
        reader.seal_root_directory_mode()
        reader.verify_path()

    assert immutable._node_identity(root.stat()) == root_before
    assert immutable._node_identity(branch.stat()) == branch_before
    assert stat.S_IMODE(root.stat().st_mode) == 0o555
    assert stat.S_IMODE(branch.stat().st_mode) == 0o755
    assert leaf.read_bytes() == payload


def test_normal_retained_directory_verifier_rejects_root_mode_change(
    tmp_path: Path,
) -> None:
    root, _branch, _leaf, _payload = _reader_tree(tmp_path)
    reader = immutable.PinnedDirectoryReader(root)
    try:
        os.fchmod(reader.descriptor, 0o555)
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="retained directory ancestor changed during use",
        ):
            reader.verify_path()
    finally:
        reader.close()


def test_root_mode_gate_accepts_complete_exact_event_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _branch, _leaf, _payload = _reader_tree(tmp_path)
    reader = immutable.PinnedDirectoryReader(root)
    try:
        events = _authorized_root_mode_events(reader)
        drains = iter((
            immutable._MutationDrainResult(events),
            immutable._MutationDrainResult(()),
        ))
        monkeypatch.setattr(
            reader._path_monitor.monitor,
            "read_events",
            lambda: next(drains, immutable._MutationDrainResult(())),
        )
        immutable._require_retained_directory_root_mode_change(
            reader._path_monitor
        )
    finally:
        reader.close()


@pytest.mark.parametrize("missing", ("root", "parent"))
def test_root_mode_gate_requires_both_exact_event_signatures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    root, _branch, _leaf, _payload = _reader_tree(tmp_path)
    reader = immutable.PinnedDirectoryReader(root)
    try:
        root_event, parent_event = _authorized_root_mode_events(reader)
        events = (parent_event,) if missing == "root" else (root_event,)
        monkeypatch.setattr(
            reader._path_monitor.monitor,
            "read_events",
            lambda: immutable._MutationDrainResult(events),
        )
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="root mode seal was not fully observed",
        ):
            immutable._require_retained_directory_root_mode_change(
                reader._path_monitor
            )
    finally:
        reader.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "move_self",
        "delete_self",
        "moved_from",
        "moved_to",
        "create",
        "delete",
        "ancestor_attrib",
        "unrelated",
        "unknown_watch",
        "extra_mask",
        "duplicate_authorized",
    ),
)
def test_root_mode_gate_rejects_every_other_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root, _branch, _leaf, _payload = _reader_tree(tmp_path)
    reader = immutable.PinnedDirectoryReader(root)
    try:
        state = reader._path_monitor
        root_event, parent_event = _authorized_root_mode_events(reader)
        parent_watch, root_name = state.protected_names[-1]
        directory_flag = 0x40000000
        invalid = {
            "move_self": (
                state.root_watch,
                directory_flag | 0x00000800,
                b"",
            ),
            "delete_self": (
                state.root_watch,
                directory_flag | 0x00000400,
                b"",
            ),
            "moved_from": (
                parent_watch,
                directory_flag | 0x00000040,
                root_name,
            ),
            "moved_to": (
                parent_watch,
                directory_flag | 0x00000080,
                root_name,
            ),
            "create": (
                parent_watch,
                directory_flag | 0x00000100,
                root_name,
            ),
            "delete": (
                parent_watch,
                directory_flag | 0x00000200,
                root_name,
            ),
            "ancestor_attrib": (
                state.protected_names[-2][0],
                immutable._INOTIFY_ROOT_DIRECTORY_ATTRIB,
                state.protected_names[-2][1],
            ),
            "unrelated": (
                parent_watch,
                immutable._INOTIFY_ROOT_DIRECTORY_ATTRIB,
                b"unrelated",
            ),
            "unknown_watch": (
                max(state.directory_watches) + 1,
                immutable._INOTIFY_ROOT_DIRECTORY_ATTRIB,
                b"",
            ),
            "extra_mask": (
                state.root_watch,
                immutable._INOTIFY_ROOT_DIRECTORY_ATTRIB | 0x00008000,
                b"",
            ),
            "duplicate_authorized": root_event,
        }[mutation]
        result = immutable._MutationDrainResult(
            (root_event, parent_event, invalid)
        )
        monkeypatch.setattr(
            state.monitor,
            "read_events",
            lambda: result,
        )
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match=(
                "unknown watch"
                if mutation == "unknown_watch"
                else (
                    "duplicate event"
                    if mutation == "duplicate_authorized"
                    else "outside authorized root mode seal"
                )
            ),
        ):
            immutable._require_retained_directory_root_mode_change(state)
    finally:
        reader.close()


def test_root_mode_gate_preserves_monitor_resource_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _branch, _leaf, _payload = _reader_tree(tmp_path)
    reader = immutable.PinnedDirectoryReader(root)
    try:
        events = _authorized_root_mode_events(reader)
        resource_error = immutable.ImmutableArtifactResourceError(
            f"{immutable.VERIFY_RESOURCE_BOUND}: inotify queue overflow"
        )
        monkeypatch.setattr(
            reader._path_monitor.monitor,
            "read_events",
            lambda: immutable._MutationDrainResult(
                events,
                resource_error=resource_error,
            ),
        )
        with pytest.raises(
            immutable.ImmutableArtifactResourceError,
            match="inotify queue overflow",
        ):
            immutable._require_retained_directory_root_mode_change(
                reader._path_monitor
            )
    finally:
        reader.close()


def _complete_rename_events(
    state: immutable._DirectoryRenameMonitorState,
) -> tuple[tuple[int, int, bytes], ...]:
    return (
        (
            state.source_watch,
            immutable._INOTIFY_ROOT_DIRECTORY_MOVED_FROM,
            state.source_name,
        ),
        (
            state.destination_watch,
            immutable._INOTIFY_ROOT_DIRECTORY_MOVED_TO,
            state.destination_name,
        ),
        (state.root_watch, immutable._INOTIFY_ROOT_MOVE_SELF, b""),
    )


def _synthetic_rename_monitor() -> immutable._DirectoryRenameMonitorState:
    return immutable._DirectoryRenameMonitorState(
        monitor=immutable._DirectoryMutationMonitor(),
        source_watch=101,
        destination_watch=102,
        root_watch=103,
        source_name=b"source",
        destination_name=b"destination",
    )


def test_directory_rename_gate_accepts_exact_matching_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _synthetic_rename_monitor()
    try:
        monkeypatch.setattr(
            state.monitor,
            "read_events",
            lambda: immutable._MutationDrainResult(
                _complete_rename_events(state),
                event_cookies=(17, 17, 0),
            ),
        )
        immutable._require_directory_rename_monitor_complete(state)
    finally:
        state.monitor.cancel_close_verification()
        state.monitor.close()


@pytest.mark.parametrize(
    ("mutation", "cookies"),
    (
        ("missing_source", (17, 0)),
        ("duplicate_source", (17, 17, 17, 0)),
        ("wrong_source_mask", (17, 17, 0)),
        ("wrong_destination_name", (17, 17, 0)),
        ("unknown_watch", (17, 17, 0, 0)),
        ("zero_cookie", (0, 0, 0)),
        ("mismatched_cookie", (17, 18, 0)),
        ("root_cookie", (17, 17, 9)),
    ),
)
def test_directory_rename_gate_rejects_inexact_event_topology(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    cookies: tuple[int, ...],
) -> None:
    state = _synthetic_rename_monitor()
    source, destination, root = _complete_rename_events(state)
    events = {
        "missing_source": (destination, root),
        "duplicate_source": (source, source, destination, root),
        "wrong_source_mask": (
            (source[0], immutable._INOTIFY_ROOT_DIRECTORY_ATTRIB, source[2]),
            destination,
            root,
        ),
        "wrong_destination_name": (
            source,
            (destination[0], destination[1], b"other"),
            root,
        ),
        "unknown_watch": (
            source,
            destination,
            root,
            (999, immutable._INOTIFY_ROOT_MOVE_SELF, b""),
        ),
        "zero_cookie": (source, destination, root),
        "mismatched_cookie": (source, destination, root),
        "root_cookie": (source, destination, root),
    }[mutation]
    try:
        monkeypatch.setattr(
            state.monitor,
            "read_events",
            lambda: immutable._MutationDrainResult(
                events,
                event_cookies=cookies,
            ),
        )
        with pytest.raises(immutable.ImmutableArtifactError):
            immutable._require_directory_rename_monitor_complete(state)
    finally:
        state.monitor.cancel_close_verification()
        state.monitor.close()


def test_directory_rename_gate_preserves_resource_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _synthetic_rename_monitor()
    resource_error = immutable.ImmutableArtifactResourceError(
        f"{immutable.VERIFY_RESOURCE_BOUND}: inotify queue overflow"
    )
    try:
        monkeypatch.setattr(
            state.monitor,
            "read_events",
            lambda: immutable._MutationDrainResult(
                _complete_rename_events(state),
                resource_error=resource_error,
                event_cookies=(17, 17, 0),
            ),
        )
        with pytest.raises(
            immutable.ImmutableArtifactResourceError,
            match="inotify queue overflow",
        ):
            immutable._require_directory_rename_monitor_complete(state)
    finally:
        state.monitor.cancel_close_verification()
        state.monitor.close()


@pytest.mark.parametrize("same_parent", (False, True))
def test_pinned_directory_reader_relocates_exact_noreplace_directory(
    tmp_path: Path,
    same_parent: bool,
) -> None:
    source_parent = tmp_path / "source-parent"
    source_parent.mkdir()
    destination_parent = (
        source_parent if same_parent else tmp_path / "destination-parent"
    )
    destination_parent.mkdir(exist_ok=True)
    source = source_parent / "source"
    source.mkdir()
    _regular(source / "payload", b"identity")
    destination = destination_parent / "destination"

    with immutable.PinnedDirectoryReader(source_parent) as source_reader:
        if same_parent:
            destination_reader = source_reader
            with immutable.PinnedDirectoryReader(source) as tree_reader:
                tree_reader.relocate_noreplace(
                    source_parent=source_reader,
                    source_name=source.name,
                    destination_parent=destination_reader,
                    destination_name=destination.name,
                )
                tree_reader.verify_path()
                assert tree_reader.read("payload") == b"identity"
        else:
            with immutable.PinnedDirectoryReader(
                destination_parent
            ) as destination_reader, immutable.PinnedDirectoryReader(
                source
            ) as tree_reader:
                tree_reader.relocate_noreplace(
                    source_parent=source_reader,
                    source_name=source.name,
                    destination_parent=destination_reader,
                    destination_name=destination.name,
                )
                tree_reader.verify_path()
                assert tree_reader.read("payload") == b"identity"

    assert not source.exists()
    assert (destination / "payload").read_bytes() == b"identity"


@pytest.mark.parametrize("exception_type", (KeyboardInterrupt, SystemExit))
def test_pinned_directory_reader_defers_committed_rename_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "source"
    source.mkdir()
    _regular(source / "payload", b"identity")
    destination = destination_parent / "destination"
    real_rename = immutable._rename_noreplace_at
    interruption = exception_type("injected committed rename interruption")

    def rename_then_interrupt(*args) -> None:
        real_rename(*args)
        raise interruption

    monkeypatch.setattr(
        immutable,
        "_rename_noreplace_at",
        rename_then_interrupt,
    )
    with immutable.PinnedDirectoryReader(
        source_parent
    ) as source_reader, immutable.PinnedDirectoryReader(
        destination_parent
    ) as destination_reader, immutable.PinnedDirectoryReader(
        source
    ) as tree_reader:
        with pytest.raises(immutable.CommittedRenameInterruption) as caught:
            tree_reader.relocate_noreplace(
                source_parent=source_reader,
                source_name=source.name,
                destination_parent=destination_reader,
                destination_name=destination.name,
            )
        assert caught.value.interruption is interruption
        assert caught.value.handoff_complete is True
        tree_reader.verify_path()
        assert tree_reader.read("payload") == b"identity"

    assert not source.exists()
    assert (destination / "payload").read_bytes() == b"identity"


@pytest.mark.parametrize(
    "exception_type",
    (KeyboardInterrupt, SystemExit, OSError),
)
def test_pinned_directory_reader_types_unproven_postsyscall_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "source"
    source.mkdir()
    _regular(source / "payload", b"identity")
    destination = destination_parent / "destination"
    real_entry_metadata = immutable._entry_metadata
    real_rename = immutable._rename_noreplace_at
    failure = exception_type("injected post-syscall outcome failure")
    renamed = False
    failed = False

    def observe_rename(*args) -> None:
        nonlocal renamed
        real_rename(*args)
        renamed = True

    def fail_first_outcome_probe(parent_descriptor: int, name: str):
        nonlocal failed
        if renamed and not failed:
            failed = True
            raise failure
        return real_entry_metadata(parent_descriptor, name)

    monkeypatch.setattr(immutable, "_rename_noreplace_at", observe_rename)
    monkeypatch.setattr(
        immutable,
        "_entry_metadata",
        fail_first_outcome_probe,
    )
    with pytest.raises(immutable.RenameOutcomeUnproven) as caught:
        with immutable.PinnedDirectoryReader(
            source_parent
        ) as source_reader, immutable.PinnedDirectoryReader(
            destination_parent
        ) as destination_reader, immutable.PinnedDirectoryReader(
            source
        ) as tree_reader:
            tree_reader.relocate_noreplace(
                source_parent=source_reader,
                source_name=source.name,
                destination_parent=destination_reader,
                destination_name=destination.name,
            )

    assert caught.value.error is failure
    assert renamed is True
    assert failed is True
    assert not source.exists()
    assert (destination / "payload").read_bytes() == b"identity"


def test_pinned_directory_reader_rejects_transient_source_only_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "source"
    source.mkdir()
    _regular(source / "payload", b"identity")
    destination = destination_parent / "destination"
    real_rename = immutable._rename_noreplace_at

    def move_to_destination_and_back(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        real_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )
        real_rename(
            destination_descriptor,
            destination_name,
            source_descriptor,
            source_name,
        )
        raise immutable.ImmutableArtifactError(
            "injected error after transient committed rename"
        )

    monkeypatch.setattr(
        immutable,
        "_rename_noreplace_at",
        move_to_destination_and_back,
    )
    with pytest.raises(immutable.RenameOutcomeUnproven):
        with immutable.PinnedDirectoryReader(
            source_parent
        ) as source_reader, immutable.PinnedDirectoryReader(
            destination_parent
        ) as destination_reader, immutable.PinnedDirectoryReader(
            source
        ) as tree_reader:
            tree_reader.relocate_noreplace(
                source_parent=source_reader,
                source_name=source.name,
                destination_parent=destination_reader,
                destination_name=destination.name,
            )

    assert (source / "payload").read_bytes() == b"identity"
    assert not destination.exists()


def test_pinned_directory_reader_relocation_allows_unrelated_sibling_churn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "source"
    source.mkdir()
    destination = destination_parent / "destination"
    real_rename = immutable._rename_noreplace_at

    def churn_then_rename(*args) -> None:
        sibling = source_parent / "unrelated"
        sibling.mkdir()
        sibling.rmdir()
        real_rename(*args)

    monkeypatch.setattr(immutable, "_rename_noreplace_at", churn_then_rename)
    with immutable.PinnedDirectoryReader(
        source_parent
    ) as source_reader, immutable.PinnedDirectoryReader(
        destination_parent
    ) as destination_reader, immutable.PinnedDirectoryReader(
        source
    ) as tree_reader:
        tree_reader.relocate_noreplace(
            source_parent=source_reader,
            source_name=source.name,
            destination_parent=destination_reader,
            destination_name=destination.name,
        )
        tree_reader.verify_path()

    assert destination.is_dir()


def test_pinned_directory_reader_handoff_survives_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "source"
    source.mkdir()
    _regular(source / "payload", b"identity")
    destination = destination_parent / "destination"
    real_close_callables = immutable._close_callables
    interruption = KeyboardInterrupt("injected handoff interruption")
    interrupted = False

    def interrupt_old_binding_close(closers, *, label: str) -> None:
        nonlocal interrupted
        if label == "renamed pinned directory reader transition" and not interrupted:
            interrupted = True
            raise interruption
        real_close_callables(closers, label=label)

    monkeypatch.setattr(
        immutable,
        "_close_callables",
        interrupt_old_binding_close,
    )
    with immutable.PinnedDirectoryReader(
        source_parent
    ) as source_reader, immutable.PinnedDirectoryReader(
        destination_parent
    ) as destination_reader, immutable.PinnedDirectoryReader(
        source
    ) as tree_reader:
        with pytest.raises(immutable.CommittedRenameInterruption) as caught:
            tree_reader.relocate_noreplace(
                source_parent=source_reader,
                source_name=source.name,
                destination_parent=destination_reader,
                destination_name=destination.name,
            )
        assert caught.value.interruption is interruption
        assert caught.value.handoff_complete is True
        tree_reader.verify_path()
        assert tree_reader.read("payload") == b"identity"

    assert interrupted is True
    assert not source.exists()
    assert (destination / "payload").read_bytes() == b"identity"


@pytest.mark.parametrize("mutation", ("move_away_back", "substitute"))
def test_pinned_directory_reader_relocation_rejects_postrename_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "source"
    source.mkdir()
    _regular(source / "payload", b"identity")
    destination = destination_parent / "destination"
    displaced = destination_parent / "displaced"
    original_open = immutable._open_directory_with_early_path_monitor
    mutated = False

    source_reader = immutable.PinnedDirectoryReader(source_parent)
    destination_reader = immutable.PinnedDirectoryReader(destination_parent)
    tree_reader = immutable.PinnedDirectoryReader(source)

    def mutate_before_rebind(path: Path):
        nonlocal mutated
        if Path(path) == destination and not mutated:
            destination.rename(displaced)
            if mutation == "move_away_back":
                displaced.rename(destination)
            else:
                destination.mkdir()
                _regular(destination / "payload", b"identity")
            mutated = True
        return original_open(path)

    monkeypatch.setattr(
        immutable,
        "_open_directory_with_early_path_monitor",
        mutate_before_rebind,
    )
    try:
        with pytest.raises(immutable.ImmutableArtifactError):
            tree_reader.relocate_noreplace(
                source_parent=source_reader,
                source_name=source.name,
                destination_parent=destination_reader,
                destination_name=destination.name,
            )
        assert mutated is True
    finally:
        tree_reader._path_monitor.monitor.cancel_close_verification()
        tree_reader.close()
        destination_reader.close()
        source_reader.close()


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
    original_read = immutable._read_fd_snapshot
    mutated = False

    def replace_after_read(
        descriptor: int, *, allow_hardlinks: bool = False,
    ) -> tuple[bytes, immutable.MetadataSnapshot]:
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

    monkeypatch.setattr(immutable, "_read_fd_snapshot", replace_after_read)
    exit_check = (
        pytest.raises(immutable.ImmutableArtifactError)
        if mutation.startswith("root_")
        else nullcontext()
    )
    with exit_check:
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
    ) -> tuple[bytes, immutable.MetadataSnapshot]:
        del allow_hardlinks
        pytest.fail("leaf bytes consumed before initial path verification")

    monkeypatch.setattr(immutable.os, "open", replace_after_leaf_open)
    monkeypatch.setattr(immutable, "_read_fd_snapshot", unexpected_read)
    with immutable.PinnedDirectoryReader(root) as reader:
        with pytest.raises(immutable.ImmutableArtifactError):
            reader.read("branch/value")
    assert replaced is True


def test_pinned_directory_reader_rejects_expected_generation_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _branch, leaf, payload = _reader_tree(tmp_path)
    with immutable.PinnedDirectoryReader(root) as reader:
        expected = dict(reader.snapshot_tree().files)["branch/value"]
        original_open = immutable._open_relative_regular_at
        original_read = immutable.os.read
        target_descriptor: int | None = None
        consumed = False

        def replace_after_leaf_open(root_descriptor: int, relative_path: str):
            nonlocal target_descriptor
            pinned = original_open(root_descriptor, relative_path)
            target_descriptor = pinned.descriptor
            replacement = leaf.with_suffix(".replacement")
            replacement.write_bytes(payload)
            replacement.chmod(0o444)
            os.replace(replacement, leaf)
            return pinned

        def observe_read(descriptor: int, size: int) -> bytes:
            nonlocal consumed
            if descriptor == target_descriptor:
                consumed = True
            return original_read(descriptor, size)

        monkeypatch.setattr(
            immutable, "_open_relative_regular_at", replace_after_leaf_open,
        )
        monkeypatch.setattr(immutable.os, "read", observe_read)
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="pinned regular-file generation changed",
        ):
            reader.read_snapshot(
                "branch/value",
                expected_metadata=expected,
            )

    assert target_descriptor is not None
    assert consumed is False


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

    def fail_read(
        _descriptor: int, *, allow_hardlinks: bool = False,
    ) -> tuple[bytes, immutable.MetadataSnapshot]:
        del allow_hardlinks
        raise RuntimeError("injected anchored-read failure")

    monkeypatch.setattr(immutable, "_open_relative_regular_at", capture_open)
    monkeypatch.setattr(immutable, "_read_fd_snapshot", fail_read)
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


def test_pinned_directory_reader_enforces_pre_read_byte_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _branch, _leaf, payload = _reader_tree(tmp_path)
    original_open = immutable._open_relative_regular_at
    original_read = immutable.os.read
    target_descriptor: int | None = None
    consumed = False

    def capture_leaf(root_descriptor: int, relative_path: str):
        nonlocal target_descriptor
        pinned = original_open(root_descriptor, relative_path)
        target_descriptor = pinned.descriptor
        return pinned

    def observe_read(descriptor: int, size: int) -> bytes:
        nonlocal consumed
        if descriptor == target_descriptor:
            consumed = True
        return original_read(descriptor, size)

    with immutable.PinnedDirectoryReader(root) as reader:
        expected = dict(reader.snapshot_tree().files)["branch/value"]
        monkeypatch.setattr(
            immutable, "_open_relative_regular_at", capture_leaf,
        )
        monkeypatch.setattr(immutable.os, "read", observe_read)
        with pytest.raises(
            immutable.ImmutableArtifactLimitError,
            match="exceeds bounded snapshot limit",
        ):
            reader.read_snapshot(
                "branch/value",
                max_bytes=len(payload) - 1,
                expected_metadata=expected,
            )
    assert target_descriptor is not None
    assert consumed is False


def test_pinned_directory_reader_preserves_snapshot_resource_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _branch, _leaf, _payload = _reader_tree(tmp_path)

    def fail_snapshot(*_args, **_kwargs):
        raise immutable.ImmutableArtifactResourceError("planted resource failure")

    with immutable.PinnedDirectoryReader(root) as reader:
        expected = dict(reader.snapshot_tree().files)["branch/value"]
        monkeypatch.setattr(immutable, "_read_fd_snapshot", fail_snapshot)
        with pytest.raises(
            immutable.ImmutableArtifactResourceError,
            match="planted resource failure",
        ):
            reader.read_snapshot(
                "branch/value",
                expected_metadata=expected,
            )


@pytest.mark.parametrize(
    ("limit_name", "limit", "message"),
    (
        ("max_files", 0, "bounded file limit"),
        ("max_directories", 1, "bounded directory limit"),
        ("max_total_file_bytes", 1, "bounded aggregate file-byte limit"),
        (
            "max_total_relative_path_bytes",
            1,
            "bounded aggregate relative-path-byte limit",
        ),
    ),
)
def test_pinned_directory_reader_enforces_tree_inventory_limits(
    tmp_path: Path,
    limit_name: str,
    limit: int,
    message: str,
) -> None:
    root, _branch, _leaf, _payload = _reader_tree(tmp_path)
    with immutable.PinnedDirectoryReader(root) as reader:
        with pytest.raises(immutable.ImmutableArtifactLimitError, match=message):
            reader.snapshot_tree(**{limit_name: limit})


def test_pinned_directory_reader_retains_one_tree_generation(
    tmp_path: Path,
) -> None:
    root, _branch, _leaf, payload = _reader_tree(tmp_path)
    with immutable.PinnedDirectoryReader(root) as reader:
        with reader.retain_tree_generation() as snapshot:
            expected = dict(snapshot.files)["branch/value"]
            observed = reader.read_snapshot(
                "branch/value", expected_metadata=expected,
            )
            assert observed.payload == payload


def test_pinned_directory_reader_rejects_transient_tree_mutation(
    tmp_path: Path,
) -> None:
    root, branch, _leaf, _payload = _reader_tree(tmp_path)
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="immutable tree changed during bounded verification",
    ):
        with immutable.PinnedDirectoryReader(root) as reader:
            with reader.retain_tree_generation():
                transient = branch / "transient"
                _regular(transient, b"temporary")
                transient.unlink()


def test_write_regular_bytes_at_uses_retained_parent_descriptor(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "summary-stage"
    stage.mkdir()
    descriptor = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
    displaced = tmp_path / "displaced-summary-stage"
    stage.rename(displaced)
    stage.mkdir()
    try:
        immutable.write_regular_bytes_at(
            descriptor,
            "summary.json",
            b"descriptor-bound summary\n",
        )
    finally:
        os.close(descriptor)

    assert tuple(stage.iterdir()) == ()
    summary = displaced / "summary.json"
    assert summary.read_bytes() == b"descriptor-bound summary\n"
    assert stat.S_IMODE(summary.stat().st_mode) == 0o444


def test_write_regular_bytes_at_preserves_existing_leaf(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "summary-stage"
    stage.mkdir()
    planted = stage / "summary.json"
    planted.write_bytes(b"foreign winner\n")
    descriptor = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="descriptor-bound output already exists",
        ):
            immutable.write_regular_bytes_at(
                descriptor,
                "summary.json",
                b"attempt summary\n",
            )
    finally:
        os.close(descriptor)

    assert planted.read_bytes() == b"foreign winner\n"


def test_replace_regular_bytes_replaces_one_regular_generation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "summary.json"
    output.write_bytes(b"old")
    before = output.stat().st_ino

    immutable.replace_regular_bytes(output, b"new payload\n")

    assert output.read_bytes() == b"new payload\n"
    assert output.stat().st_ino != before
    assert not tuple(tmp_path.glob("*.partial"))
    assert not tuple(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_replace_regular_bytes_refuses_unsafe_destination(
    tmp_path: Path,
    kind: str,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"preserve")
    output = tmp_path / "summary.json"
    if kind == "symlink":
        output.symlink_to(target)
    else:
        os.link(target, output)

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="single-link regular file",
    ):
        immutable.replace_regular_bytes(output, b"replacement")

    assert target.read_bytes() == b"preserve"


def test_replace_regular_bytes_removes_failed_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "summary.json"

    def fail_write(_descriptor: int, _payload: memoryview) -> None:
        raise RuntimeError("planted write failure")

    monkeypatch.setattr(immutable, "_write_all", fail_write)
    with pytest.raises(RuntimeError, match="planted write failure"):
        immutable.replace_regular_bytes(output, b"payload")

    assert not output.exists()
    assert not tuple(tmp_path.iterdir())


def test_replace_regular_bytes_retires_post_rename_writer_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "summary.json"
    output.write_bytes(b"old")
    original_open = immutable.os.open
    original_fsync = immutable._fsync_or_fail
    writer = -1
    mutated = False

    def retain_temporary_writer(path, flags, *args, **kwargs):
        nonlocal writer
        descriptor = original_open(path, flags, *args, **kwargs)
        if (
            isinstance(path, str)
            and path.endswith(".partial")
            and flags & os.O_RDWR
            and writer < 0
        ):
            writer = os.dup(descriptor)
        return descriptor

    def mutate_after_rename(descriptor: int, label: str) -> None:
        nonlocal mutated
        original_fsync(descriptor, label)
        if label.startswith("replacement parent ") and not mutated:
            assert writer >= 0
            os.pwrite(writer, b"X", 0)
            mutated = True

    monkeypatch.setattr(immutable.os, "open", retain_temporary_writer)
    monkeypatch.setattr(immutable, "_fsync_or_fail", mutate_after_rename)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="published replacement bytes changed",
        ):
            immutable.replace_regular_bytes(output, b"new payload")
    finally:
        if writer >= 0:
            os.close(writer)

    assert mutated is True
    assert not output.exists()
    retired = [
        path for path in tmp_path.iterdir()
        if path.name.startswith(".failed-regular-")
    ]
    assert len(retired) == 1
    assert retired[0].read_bytes() == b"Xew payload"


def test_retire_regular_file_preserves_stale_bytes(tmp_path: Path) -> None:
    stale = tmp_path / "trace_volume_exceeded.json"
    stale.write_bytes(b"stale evidence\n")

    retired = immutable.retire_regular_file(stale)

    assert retired is not None
    assert not stale.exists()
    assert retired.parent == tmp_path
    assert retired.name.startswith(".trace_volume_exceeded.json.")
    assert retired.name.endswith(".retired")
    assert retired.read_bytes() == b"stale evidence\n"
    assert immutable.retire_regular_file(stale) is None


def test_retire_regular_file_rejects_unmatched_expected_payload(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "trace_volume_exceeded.json"
    stale.write_bytes(b"concurrent winner\n")

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="retirement source payload changed",
    ):
        immutable.retire_regular_file(
            stale,
            expected_payload=b"failed attempt locator\n",
        )

    assert stale.read_bytes() == b"concurrent winner\n"
    assert not tuple(tmp_path.glob(".*.retired"))


def test_retire_regular_file_preserves_same_byte_replacement_generation(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "trace_sampling.json"
    expected_payload = b'{"schema_version":2}\n'
    stale.write_bytes(expected_payload)
    with immutable.open_retained_regular(stale) as retained:
        expected_snapshot = retained.snapshot
    replacement = tmp_path / ".replacement"
    replacement.write_bytes(expected_payload)
    os.replace(replacement, stale)
    replacement_inode = stale.stat().st_ino

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="retirement source generation changed",
    ):
        immutable.retire_regular_file(
            stale,
            expected_payload=expected_payload,
            expected_snapshot=expected_snapshot,
        )

    assert stale.stat().st_ino == replacement_inode
    assert stale.read_bytes() == expected_payload
    assert not tuple(tmp_path.glob(".*.retired"))


def test_salvage_retained_regular_descriptor_preserves_same_byte_replacement(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "trace_sampling.json"
    expected_payload = b'{"schema_version":2}\n'
    stale.write_bytes(expected_payload)
    descriptor = os.open(stale, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        expected_snapshot = immutable._snapshot_fields(os.fstat(descriptor))
        replacement = tmp_path / ".replacement"
        replacement.write_bytes(expected_payload)
        os.replace(replacement, stale)
        replacement_identity = (stale.stat().st_dev, stale.stat().st_ino)

        salvaged = immutable.salvage_retained_regular_descriptor(
            stale,
            descriptor,
            expected_payload=expected_payload,
            expected_snapshot=expected_snapshot,
        )

        assert (stale.stat().st_dev, stale.stat().st_ino) == replacement_identity
        assert stale.read_bytes() == expected_payload
        assert salvaged.name.startswith(".failed-regular-")
        assert salvaged.read_bytes() == expected_payload
        assert not tuple(tmp_path.glob(".*.retired"))
    finally:
        os.close(descriptor)


def test_retire_regular_file_salvages_unlinked_hidden_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = tmp_path / "trace_sampling.json"
    expected_payload = b'{"schema_version":2}\n'
    stale.write_bytes(expected_payload)
    original_fsync = immutable._fsync_or_fail
    unlinked = False

    def unlink_hidden_before_durability(descriptor: int, label: str) -> None:
        nonlocal unlinked
        if label.startswith("retirement parent ") and not unlinked:
            hidden = tuple(tmp_path.glob(".trace_sampling.json.*.retired"))
            assert len(hidden) == 1
            hidden[0].unlink()
            unlinked = True
        original_fsync(descriptor, label)

    monkeypatch.setattr(
        immutable, "_fsync_or_fail", unlink_hidden_before_durability,
    )

    salvaged = immutable.retire_regular_file(
        stale,
        expected_payload=expected_payload,
    )

    assert unlinked is True
    assert salvaged is not None
    assert not stale.exists()
    assert salvaged.name.startswith(".failed-regular-")
    assert salvaged.read_bytes() == expected_payload
    assert not tuple(tmp_path.glob(".trace_sampling.json.*.retired"))


def test_retire_regular_file_restores_racing_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = tmp_path / "trace_volume_exceeded.json"
    expected_payload = b"failed attempt locator\n"
    replacement_payload = b"concurrent winner\n"
    stale.write_bytes(expected_payload)
    displaced = tmp_path / "displaced-failed-attempt.json"
    original_rename = immutable._rename_noreplace_at
    replacement_inode: int | None = None
    swapped = False

    def swap_before_retirement(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal replacement_inode, swapped
        if source_name == stale.name and not swapped:
            swapped = True
            os.rename(
                source_name,
                displaced.name,
                src_dir_fd=source_descriptor,
                dst_dir_fd=source_descriptor,
            )
            stale.write_bytes(replacement_payload)
            replacement_inode = stale.stat().st_ino
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        immutable,
        "_rename_noreplace_at",
        swap_before_retirement,
    )
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="retired regular-file identity changed",
    ) as failure:
        immutable.retire_regular_file(
            stale,
            expected_payload=expected_payload,
        )

    assert swapped is True
    assert replacement_inode is not None
    assert stale.stat().st_ino == replacement_inode
    assert stale.read_bytes() == replacement_payload
    assert displaced.read_bytes() == expected_payload
    assert not tuple(tmp_path.glob(".*.retired"))
    assert any(
        "concurrent retirement-source replacement was restored" in note
        for note in getattr(failure.value, "__notes__", ())
    )


def test_retire_regular_file_rejects_same_inode_payload_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = tmp_path / "trace_volume_exceeded.json"
    expected_payload = b"failed locator\n"
    replacement_payload = b"racing winner!\n"
    assert len(expected_payload) == len(replacement_payload)
    stale.write_bytes(expected_payload)
    original_inode = stale.stat().st_ino
    original_rename = immutable._rename_noreplace_at
    mutated = False

    def mutate_before_retirement(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal mutated
        if source_name == stale.name and not mutated:
            mutated = True
            stale.write_bytes(replacement_payload)
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        immutable,
        "_rename_noreplace_at",
        mutate_before_retirement,
    )
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="retired regular-file payload changed",
    ):
        immutable.retire_regular_file(
            stale,
            expected_payload=expected_payload,
        )

    assert mutated is True
    assert stale.stat().st_ino == original_inode
    assert stale.read_bytes() == replacement_payload
    assert not tuple(tmp_path.glob(".*.retired"))


def test_retire_regular_file_restores_replacement_before_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = tmp_path / "trace_volume_exceeded.json"
    expected_payload = b"failed attempt locator\n"
    replacement_payload = b"concurrent winner\n"
    stale.write_bytes(expected_payload)
    displaced = tmp_path / "displaced-failed-attempt.json"
    original_rename = immutable._rename_noreplace_at
    original_fsync = immutable._fsync_or_fail
    replacement_inode: int | None = None
    swapped = False
    planted_fsync_failure = False

    def swap_before_retirement(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal replacement_inode, swapped
        if source_name == stale.name and not swapped:
            swapped = True
            os.rename(
                source_name,
                displaced.name,
                src_dir_fd=source_descriptor,
                dst_dir_fd=source_descriptor,
            )
            stale.write_bytes(replacement_payload)
            replacement_inode = stale.stat().st_ino
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    def fail_first_retirement_fsync(descriptor: int, label: str) -> None:
        nonlocal planted_fsync_failure
        if label.startswith("retirement") and not planted_fsync_failure:
            planted_fsync_failure = True
            raise OSError(errno.EIO, "planted retirement fsync failure")
        original_fsync(descriptor, label)

    monkeypatch.setattr(
        immutable,
        "_rename_noreplace_at",
        swap_before_retirement,
    )
    monkeypatch.setattr(
        immutable,
        "_fsync_or_fail",
        fail_first_retirement_fsync,
    )
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="retired regular-file identity changed",
    ) as failure:
        immutable.retire_regular_file(
            stale,
            expected_payload=expected_payload,
        )

    assert swapped is True
    assert planted_fsync_failure is True
    assert replacement_inode is not None
    assert stale.stat().st_ino == replacement_inode
    assert stale.read_bytes() == replacement_payload
    assert displaced.read_bytes() == expected_payload
    assert not tuple(tmp_path.glob(".*.retired"))
    assert any(
        "restored to the canonical name" in note
        for note in getattr(failure.value, "__notes__", ())
    )


def test_retire_regular_file_refuses_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"preserve")
    stale = tmp_path / "trace_volume_exceeded.json"
    stale.symlink_to(target)

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="single-link regular file",
    ):
        immutable.retire_regular_file(stale)

    assert target.read_bytes() == b"preserve"
    assert stale.is_symlink()


def test_pinned_directory_reader_exposes_stable_tree_snapshot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot-tree"
    branch = root / "branch"
    branch.mkdir(parents=True)
    leaf = _regular(branch / "value", b"snapshot payload")
    branch.chmod(0o555)
    root.chmod(0o555)
    try:
        with immutable.PinnedDirectoryReader(root) as reader:
            snapshot = reader.snapshot_tree(
                require_readonly_files=True,
                require_readonly_directories=True,
            )
            regular = reader.read_snapshot("branch/value")
            reader.verify_regular("branch/value", regular.metadata)
        absolute_regular = immutable.read_regular_snapshot(leaf)
        absolute_identity = immutable.snapshot_regular_identity(leaf)
        assert regular.payload == b"snapshot payload"
        assert absolute_regular == regular
        assert absolute_identity == regular.metadata
        assert regular.metadata == immutable._snapshot_fields(
            leaf.stat(follow_symlinks=False)
        )
        assert tuple(path for path, _metadata in snapshot.files) == (
            "branch/value",
        )
        assert tuple(path for path, _metadata in snapshot.directories) == (
            ".",
            "branch",
        )
        assert dict(snapshot.files)["branch/value"] == immutable._snapshot_fields(
            leaf.stat(follow_symlinks=False)
        )
    finally:
        branch.chmod(0o755)
        root.chmod(0o755)


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


def test_expected_inventory_stops_at_first_unexpected_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "unexpected-tree"
    root.mkdir()
    for index in range(5_000):
        descriptor = os.open(
            root / f"unexpected-{index:05d}",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        os.close(descriptor)

    original_scandir = immutable.os.scandir
    yielded_entries = 0

    class CountingScandir:
        def __init__(self, path) -> None:
            self._entries = original_scandir(path)

        def __enter__(self):
            self._entries.__enter__()
            return self

        def __exit__(self, *args):
            return self._entries.__exit__(*args)

        def __iter__(self):
            nonlocal yielded_entries
            for entry in self._entries:
                yielded_entries += 1
                yield entry

    with immutable.PinnedDirectoryReader(root) as reader:
        monkeypatch.setattr(immutable.os, "scandir", CountingScandir)
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="immutable tree inventory mismatch: unexpected_file=",
        ) as failure:
            immutable._tree_inventory_descriptor(
                reader.descriptor,
                expected_files=set(),
                expected_directories=set(),
                mismatch_label="immutable tree inventory mismatch",
            )

    assert yielded_entries == 1
    assert len(str(failure.value)) < 256


def test_tree_inventory_uses_an_independent_directory_offset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "independent-directory-offset"
    root.mkdir()
    _regular(root / "value", b"payload")

    with immutable.PinnedDirectoryReader(root) as reader:
        shared_offset = os.dup(reader.descriptor)
        try:
            with os.scandir(shared_offset) as entries:
                assert {entry.name for entry in entries} == {"value"}
        finally:
            os.close(shared_offset)

        observed_files, observed_directories = (
            immutable._tree_inventory_descriptor(
                reader.descriptor,
                expected_files={"value"},
                expected_directories=set(),
                mismatch_label="immutable tree inventory mismatch",
            )
        )

    assert observed_files == {"value"}
    assert observed_directories == set()


def test_existing_descriptor_pressure_selects_snapshot_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if old_limit[1] != resource.RLIM_INFINITY and old_limit[1] < 256:
        pytest.skip("RLIMIT_NOFILE hard limit is below the required harness bound")
    root = tmp_path / "descriptor-pressure"
    root.mkdir()
    empty_sha256 = immutable.sha256_bytes(b"")
    material = []
    for index in range(64):
        relative_path = f"f{index:03d}"
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

    planted: list[int] = []
    snapshot_calls = 0
    original_snapshot = immutable._verify_snapshot_closure_at

    def record_snapshot(*args, **kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(
        immutable, "_verify_snapshot_closure_at", record_snapshot,
    )
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, old_limit[1]))
        while _open_descriptor_count() < 192:
            planted.append(os.open("/dev/null", os.O_RDONLY))
        with immutable.PinnedDirectoryReader(root) as reader:
            immutable._verify_published_files_at(reader.descriptor, material)
        assert snapshot_calls == 1
    finally:
        for descriptor in reversed(planted):
            os.close(descriptor)
        resource.setrlimit(resource.RLIMIT_NOFILE, old_limit)
        root.chmod(0o755)


def test_infinite_nofile_limit_is_not_treated_as_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "infinite-limit"
    material = _snapshot_material(root, {"value": b"payload"})
    root.chmod(0o555)
    monkeypatch.setattr(
        immutable.resource,
        "getrlimit",
        lambda _resource: (
            immutable.resource.RLIM_INFINITY,
            immutable.resource.RLIM_INFINITY,
        ),
    )
    try:
        with immutable.PinnedDirectoryReader(root) as reader:
            immutable._verify_published_files_at(reader.descriptor, material)
    finally:
        root.chmod(0o755)


@pytest.mark.parametrize("resource_errno", [errno.EMFILE, errno.ENFILE])
def test_verifier_types_descriptor_allocation_races_as_resource_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_errno: int,
) -> None:
    root = tmp_path / "descriptor-race"
    material = _snapshot_material(root, {"value": b"payload"})
    root.chmod(0o555)
    try:
        with immutable.PinnedDirectoryReader(root) as reader:
            monkeypatch.setattr(
                immutable.os,
                "dup",
                lambda _descriptor: (_ for _ in ()).throw(
                    OSError(resource_errno, "planted descriptor exhaustion")
                ),
            )
            with pytest.raises(
                immutable.ImmutableArtifactResourceError,
                match="VERIFY_RESOURCE_BOUND",
            ):
                immutable._verify_published_files_at(
                    reader.descriptor, material,
                )
    finally:
        root.chmod(0o755)


@pytest.mark.parametrize("resource_errno", [errno.EMFILE, errno.ENFILE])
@pytest.mark.parametrize(
    ("operation", "label"),
    (
        ("snapshot_tree", "pinned directory tree snapshot"),
        ("verify_regular", "pinned regular-file verification"),
    ),
)
def test_pinned_reader_translates_descriptor_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_errno: int,
    operation: str,
    label: str,
) -> None:
    root = tmp_path / "pinned-reader-resource"
    root.mkdir()
    (root / "value").write_bytes(b"payload")

    with immutable.PinnedDirectoryReader(root) as reader:
        expected = dict(reader.snapshot_tree().files)["value"]
        with monkeypatch.context() as scoped:
            scoped.setattr(
                immutable.os,
                "dup",
                lambda _descriptor: (_ for _ in ()).throw(
                    OSError(resource_errno, "planted reader exhaustion")
                ),
            )
            with pytest.raises(
                immutable.ImmutableArtifactResourceError,
                match=label,
            ):
                if operation == "snapshot_tree":
                    reader.snapshot_tree()
                else:
                    reader.verify_regular("value", expected)


@pytest.mark.parametrize("resource_errno", [errno.EMFILE, errno.ENFILE])
def test_explicit_authority_error_precedes_suppressed_resource_context(
    resource_errno: int,
) -> None:
    @immutable._translate_descriptor_resource_errors(
        "suppressed authority context regression"
    )
    def reject_authority() -> None:
        try:
            raise OSError(
                resource_errno, "suppressed descriptor exhaustion",
            )
        except OSError:
            raise immutable.ImmutableArtifactError(
                "explicit canonical authority violation"
            ) from None

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="explicit canonical authority violation",
    ) as failure:
        reject_authority()

    assert not isinstance(
        failure.value, immutable.ImmutableArtifactResourceError,
    )
    assert failure.value.__suppress_context__ is True
    assert isinstance(failure.value.__context__, OSError)
    assert failure.value.__context__.errno == resource_errno


@pytest.mark.parametrize("resource_errno", [errno.EMFILE, errno.ENFILE])
@pytest.mark.parametrize("chain_kind", ("cause", "context"))
def test_wrapped_descriptor_exhaustion_is_typed_through_exception_chains(
    resource_errno: int,
    chain_kind: str,
) -> None:
    @immutable._translate_descriptor_resource_errors(
        "wrapped descriptor exhaustion regression"
    )
    def exhaust() -> None:
        try:
            raise OSError(resource_errno, "wrapped descriptor exhaustion")
        except OSError as exc:
            if chain_kind == "cause":
                raise RuntimeError("explicit wrapper") from exc
            raise RuntimeError("implicit wrapper")

    with pytest.raises(
        immutable.ImmutableArtifactResourceError,
        match="wrapped descriptor exhaustion regression",
    ) as failure:
        exhaust()

    wrapper = failure.value.__cause__
    assert isinstance(wrapper, RuntimeError)
    if chain_kind == "cause":
        resource_error = wrapper.__cause__
    else:
        assert wrapper.__suppress_context__ is False
        resource_error = wrapper.__context__
    assert isinstance(resource_error, OSError)
    assert resource_error.errno == resource_errno


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


def test_post_authorization_closure_derives_all_limits_from_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "authorization-tree"
    branch = root / "branch"
    branch.mkdir(parents=True)
    _regular(branch / "value", b"payload")
    with immutable.PinnedDirectoryReader(root) as reader:
        baseline = immutable._snapshot_tree_descriptor(reader.descriptor)
        expected_limits = immutable._baseline_tree_limits(baseline)
        original_snapshot = immutable._snapshot_tree_descriptor
        calls = []

        def record_limits(descriptor: int, **limits):
            calls.append(dict(limits))
            return original_snapshot(descriptor, **limits)

        monkeypatch.setattr(
            immutable, "_snapshot_tree_descriptor", record_limits,
        )
        immutable._verify_snapshot_against_at(reader.descriptor, baseline)

    assert calls == [expected_limits, expected_limits]
    assert expected_limits == {
        "max_files": 1,
        "max_directories": 2,
        "max_total_file_bytes": len(b"payload"),
        "max_total_relative_path_bytes": (
            len(os.fsencode("branch"))
            + len(os.fsencode("branch/value"))
        ),
    }


def test_post_authorization_closure_stops_at_baseline_file_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bounded-authorization-tree"
    root.mkdir()
    _regular(root / "authorized", b"payload")
    with immutable.PinnedDirectoryReader(root) as reader:
        baseline = immutable._snapshot_tree_descriptor(reader.descriptor)
        for index in range(32):
            _regular(root / f"injected-{index:02d}", b"unexpected")

        with pytest.raises(
            immutable.ImmutableArtifactLimitError,
            match="bounded",
        ):
            immutable._verify_snapshot_against_at(
                reader.descriptor, baseline,
            )


def test_snapshot_monitor_types_inotify_queue_overflow_as_resource_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = immutable._DirectoryMutationMonitor()
    overflow_event = b"".join(
        (
            (-1).to_bytes(4, sys.byteorder, signed=True),
            immutable._INOTIFY_Q_OVERFLOW.to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
        )
    )
    monkeypatch.setattr(immutable.os, "read", lambda _fd, _size: overflow_event)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactResourceError,
            match="inotify event queue overflowed",
        ):
            monitor.require_quiet()
    finally:
        monitor.close()


def test_snapshot_monitor_prioritizes_mutation_before_later_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = immutable._DirectoryMutationMonitor()
    mutation_event = b"".join(
        (
            (7).to_bytes(4, sys.byteorder, signed=True),
            (0x00000002).to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
        )
    )
    overflow_event = b"".join(
        (
            (-1).to_bytes(4, sys.byteorder, signed=True),
            immutable._INOTIFY_Q_OVERFLOW.to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
        )
    )
    monkeypatch.setattr(
        immutable.os, "read", lambda _fd, _size: mutation_event + overflow_event,
    )
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="immutable tree changed during bounded verification",
        ) as failure:
            monitor.require_quiet()
        assert not isinstance(
            failure.value, immutable.ImmutableArtifactResourceError,
        )
        assert any(
            "inotify event queue overflowed" in note
            for note in getattr(failure.value, "__notes__", ())
        )
    finally:
        monitor.close()


def test_snapshot_monitor_prioritizes_mutation_after_earlier_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = immutable._DirectoryMutationMonitor()
    overflow_event = b"".join(
        (
            (-1).to_bytes(4, sys.byteorder, signed=True),
            immutable._INOTIFY_Q_OVERFLOW.to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
        )
    )
    mutation_event = b"".join(
        (
            (7).to_bytes(4, sys.byteorder, signed=True),
            (0x00000002).to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
        )
    )
    monkeypatch.setattr(
        immutable.os, "read", lambda _fd, _size: overflow_event + mutation_event,
    )
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="immutable tree changed during bounded verification",
        ) as failure:
            monitor.require_quiet()
        assert not isinstance(
            failure.value, immutable.ImmutableArtifactResourceError,
        )
        assert any(
            "inotify event queue overflowed" in note
            for note in getattr(failure.value, "__notes__", ())
        )
    finally:
        monitor.close()


def test_snapshot_monitor_bounds_repeated_interrupted_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = immutable._DirectoryMutationMonitor()
    read_calls = 0

    def interrupt_read(_fd: int, _size: int) -> bytes:
        nonlocal read_calls
        read_calls += 1
        raise InterruptedError

    monkeypatch.setattr(immutable.os, "read", interrupt_read)
    monkeypatch.setattr(immutable, "_INOTIFY_MAX_DRAIN_ATTEMPTS", 3)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactResourceError,
            match="bounded read attempts",
        ):
            monitor.require_quiet()
        assert read_calls == 3
    finally:
        monitor.close()


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

    def mutate_after_final_snapshot(root_descriptor: int, **kwargs):
        nonlocal calls, mutated
        calls += 1
        snapshot = original_snapshot(root_descriptor, **kwargs)
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

    def link_after_final_snapshot(root_descriptor: int, **kwargs):
        nonlocal calls, linked
        calls += 1
        snapshot = original_snapshot(root_descriptor, **kwargs)
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


def test_directory_publication_recovers_complete_deterministic_partial(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "published"
    output_root.mkdir()
    stage = output_root / ".artifact.staging"
    stage.mkdir()
    partial = stage / ".value.m5-partial"
    partial.write_bytes(b"original")
    partial.chmod(0o444)

    result = immutable.publish_directory(
        output_root,
        "artifact",
        [immutable.ImmutableFile.from_bytes("value", b"original")],
    )

    assert result == output_root / "artifact"
    assert (result / "value").read_bytes() == b"original"
    assert not (result / ".value.m5-partial").exists()


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
    assert not (output_root / "artifact").exists()
    retired = [
        path for path in output_root.iterdir()
        if path.name.startswith(".failed-directory-")
    ]
    assert len(retired) == 1
    assert (retired[0] / "value").read_bytes() == b"tampered"


def test_directory_retires_same_byte_replacement_before_postcommit_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]
    original_verify = immutable._verify_published_files_at
    verify_count = 0
    replacement_inode: int | None = None

    def replace_before_third_verification(*args, **kwargs):
        nonlocal verify_count, replacement_inode
        verify_count += 1
        if verify_count == 3:
            replacement_inode = _replace_file_with_same_bytes(
                output_root, "value", b"original"
            )
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(
        immutable,
        "_verify_published_files_at",
        replace_before_third_verification,
    )
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="child generation changed across publication commit",
    ):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
        )

    assert verify_count == 3
    assert replacement_inode is not None
    assert not (output_root / "artifact").exists()
    retired = [
        path for path in output_root.iterdir()
        if path.name.startswith(".failed-directory-")
    ]
    assert len(retired) == 1
    assert (retired[0] / "value").stat().st_ino == replacement_inode


def test_directory_publication_retires_on_primary_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "published"
    original_open_directory = immutable._open_pinned_relative_directory
    original_close = immutable.os.close
    target_descriptor = -1
    injected = False

    def capture_stage(root_descriptor: int, parts: tuple[str, ...]):
        nonlocal target_descriptor
        pinned = original_open_directory(root_descriptor, parts)
        if parts == (".artifact.staging",) and target_descriptor < 0:
            target_descriptor = pinned.descriptor
        return pinned

    def close_then_fail(descriptor: int) -> None:
        nonlocal injected
        if descriptor == target_descriptor and not injected:
            injected = True
            original_close(descriptor)
            raise OSError(errno.EIO, "injected committed-stage close failure")
        original_close(descriptor)

    monkeypatch.setattr(
        immutable, "_open_pinned_relative_directory", capture_stage,
    )
    monkeypatch.setattr(immutable.os, "close", close_then_fail)
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match=immutable.DURABILITY_UNPROVEN,
    ) as failure:
        immutable.publish_directory(
            output_root,
            "artifact",
            [immutable.ImmutableFile.from_bytes("value", b"payload")],
        )

    assert injected is True
    assert not (output_root / "artifact").exists()
    retired = [
        path for path in output_root.iterdir()
        if path.name.startswith(".failed-directory-")
    ]
    assert len(retired) == 1
    assert (retired[0] / "value").read_bytes() == b"payload"
    assert any(
        "preserved outside the active namespace" in note
        for note in getattr(failure.value, "__notes__", ())
    )
    retired[0].chmod(0o755)


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


def test_directory_retires_same_byte_inode_replacement_after_commit(
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
    assert not (output_root / "artifact").exists()
    retired = [
        path for path in output_root.iterdir()
        if path.name.startswith(".failed-directory-")
    ]
    assert len(retired) == 1
    assert (retired[0] / "value").read_bytes() == b"original"


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


def test_directory_retires_final_when_final_authorization_fails(
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
    assert not (output_root / "artifact").exists()
    assert not (output_root / ".artifact.staging").exists()
    retired = [
        path for path in output_root.iterdir()
        if path.name.startswith(".failed-directory-")
    ]
    assert len(retired) == 1
    assert (retired[0] / "value").read_bytes() == b"original"


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
    assert not (output_root / "artifact").exists()
    retired = [
        path for path in output_root.iterdir()
        if path.name.startswith(".failed-directory-")
    ]
    assert len(retired) == 1
    assert (retired[0] / "value").read_bytes() == b"tampered"


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
    assert callback_count == 4
    assert (result / "value").read_bytes() == b"original"


def test_directory_strict_mode_rejects_exact_rename_race_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "published"
    material = [immutable.ImmutableFile.from_bytes("value", b"original")]

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
        match="publication already exists or raced",
    ):
        immutable.publish_directory(
            output_root,
            "artifact",
            material,
            accept_existing=False,
        )

    assert not (output_root / "artifact").exists()
    assert not (output_root / ".artifact.staging").exists()
    retired = [
        path for path in output_root.iterdir()
        if path.name.startswith(".failed-directory-")
    ]
    assert len(retired) == 1
    assert (retired[0] / "value").read_bytes() == b"original"


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
    assert not (output_root / "artifact").exists()
    retired = [
        path for path in output_root.iterdir()
        if path.name.startswith(".failed-directory-")
    ]
    assert len(retired) == 1
    assert (retired[0] / "value").stat().st_ino == replacement_inode


def test_regular_publication_callbacks_bracket_commit(tmp_path: Path) -> None:
    output = tmp_path / "artifact.bin"
    events: list[str] = []

    def before_commit() -> None:
        assert not output.exists()
        events.append("before")

    def after_commit() -> None:
        assert output.read_bytes() == b"payload"
        events.append("after")

    result = immutable.publish_regular_file(
        tmp_path,
        output.name,
        b"payload",
        before_commit=before_commit,
        after_commit=after_commit,
    )

    assert result == output
    assert events == ["before", "after"]


def test_regular_publication_retires_after_commit_callback_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact.bin"

    def refuse_after_commit() -> None:
        raise RuntimeError("postcommit authority changed")

    with pytest.raises(RuntimeError, match="postcommit authority changed"):
        immutable.publish_regular_file(
            tmp_path,
            output.name,
            b"payload",
            after_commit=refuse_after_commit,
        )

    assert not output.exists()
    retired = [
        path for path in tmp_path.iterdir()
        if path.name.startswith(".failed-regular-")
        and not path.name.endswith(".salvaging")
    ]
    assert len(retired) == 1
    assert retired[0].read_bytes() == b"payload"


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

    assert not output.exists()
    assert alias.read_bytes() == b"payload"
    retired = [
        path for path in tmp_path.iterdir()
        if path.name.startswith(".failed-regular-")
    ]
    assert len(retired) == 1
    assert retired[0].read_bytes() == b"payload"
    assert retired[0].stat().st_nlink == 2


def test_regular_publication_salvages_replaced_committed_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifact.bin"
    original_entry_metadata = immutable._entry_metadata
    replaced = False
    winner_inode: int | None = None

    def replace_after_rename(descriptor: int, name: str):
        nonlocal replaced, winner_inode
        metadata = original_entry_metadata(descriptor, name)
        if name == output.name and metadata is not None and not replaced:
            replaced = True
            output.unlink()
            output.write_bytes(b"concurrent winner")
            output.chmod(0o444)
            winner_inode = output.stat().st_ino
            metadata = original_entry_metadata(descriptor, name)
        return metadata

    monkeypatch.setattr(
        immutable, "_entry_metadata", replace_after_rename,
    )
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="named regular-file path changed",
    ) as failure:
        immutable.publish_regular_file(
            tmp_path, output.name, b"committed payload",
        )

    assert replaced is True
    assert winner_inode is not None
    assert output.stat().st_ino == winner_inode
    assert output.read_bytes() == b"concurrent winner"
    salvaged = [
        path for path in tmp_path.iterdir()
        if path.name.startswith(".failed-regular-")
        and not path.name.endswith(".salvaging")
    ]
    assert len(salvaged) == 1
    assert salvaged[0].read_bytes() == b"committed payload"
    assert any(
        "bytes were copied outside the active namespace" in note
        for note in getattr(failure.value, "__notes__", ())
    )


def test_failed_publication_salvages_unlinked_hidden_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifact.bin"
    payload = b"authenticated committed payload"
    output.write_bytes(payload)
    output.chmod(0o444)
    parent = immutable._open_pinned_directory(tmp_path)
    retained = os.open(
        output,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    original_fsync = immutable._fsync_or_fail
    unlinked = False
    primary = RuntimeError("selected publication failure")

    def unlink_hidden_before_durability(descriptor: int, label: str) -> None:
        nonlocal unlinked
        if label == "failed publication retirement artifact.bin" and not unlinked:
            hidden = tuple(tmp_path.glob(".failed-regular-*"))
            assert len(hidden) == 1
            hidden[0].unlink()
            unlinked = True
        original_fsync(descriptor, label)

    monkeypatch.setattr(
        immutable, "_fsync_or_fail", unlink_hidden_before_durability,
    )
    try:
        immutable._preserve_failed_regular_publication_at(
            primary,
            parent.descriptor,
            output.name,
            retained,
        )
    finally:
        os.close(retained)
        parent.close()

    assert unlinked is True
    assert not output.exists()
    evidence = tuple(
        path for path in tmp_path.glob(".failed-regular-*")
        if not path.name.endswith(".salvaging")
    )
    assert len(evidence) == 1
    assert evidence[0].read_bytes() == payload
    assert any(
        "bytes were copied outside the active namespace" in note
        for note in getattr(primary, "__notes__", ())
    )


def test_regular_publication_retires_on_primary_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = immutable.os.open
    original_close = immutable.os.close
    target_descriptor = -1
    injected = False

    def record_primary_open(path, flags, *args, **kwargs):
        nonlocal target_descriptor
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == ".artifact.bin.m5-partial" and flags & os.O_RDWR:
            target_descriptor = descriptor
        return descriptor

    def close_then_fail(descriptor: int) -> None:
        nonlocal injected
        if descriptor == target_descriptor and not injected:
            injected = True
            original_close(descriptor)
            raise OSError(errno.EIO, "injected primary close failure")
        original_close(descriptor)

    monkeypatch.setattr(immutable.os, "open", record_primary_open)
    monkeypatch.setattr(immutable.os, "close", close_then_fail)
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match=immutable.DURABILITY_UNPROVEN,
    ) as failure:
        immutable.publish_regular_file(
            tmp_path, "artifact.bin", b"payload",
        )

    assert injected is True
    assert not (tmp_path / "artifact.bin").exists()
    retired = [
        path for path in tmp_path.iterdir()
        if path.name.startswith(".failed-regular-")
        and not path.name.endswith(".salvaging")
    ]
    assert len(retired) == 1
    assert retired[0].read_bytes() == b"payload"
    assert any(
        "preserved outside the active namespace" in note
        for note in getattr(failure.value, "__notes__", ())
    )


def test_regular_publication_reports_guard_close_without_retrying_reused_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with immutable._rollback_guard_state_lock:
        immutable._rollback_guard_diagnostics.clear()
    original_guard = immutable._open_regular_rollback_guard
    original_open = immutable.os.open
    original_close = immutable.os.close
    target_descriptor = -1
    replacement_descriptor = -1
    close_attempts = 0

    def capture_guard(
        parent_descriptor: int,
        name: str,
        retained_descriptor: int,
    ) -> int:
        nonlocal target_descriptor
        descriptor = original_guard(
            parent_descriptor, name, retained_descriptor,
        )
        target_descriptor = descriptor
        return descriptor

    def close_then_reuse(descriptor: int) -> None:
        nonlocal close_attempts, replacement_descriptor
        if descriptor == target_descriptor and close_attempts == 0:
            close_attempts += 1
            original_close(descriptor)
            opened = original_open(
                tmp_path / "artifact.bin",
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
            if opened != descriptor:
                os.dup2(opened, descriptor)
                original_close(opened)
            replacement_descriptor = descriptor
            raise OSError(errno.EIO, "injected rollback guard close failure")
        original_close(descriptor)

    monkeypatch.setattr(
        immutable, "_open_regular_rollback_guard", capture_guard,
    )
    monkeypatch.setattr(immutable.os, "close", close_then_reuse)
    try:
        with pytest.warns(
            RuntimeWarning,
            match="rollback guards: descriptor",
        ):
            result = immutable.publish_regular_file(
                tmp_path, "artifact.bin", b"payload",
            )

        assert close_attempts == 1
        assert result.read_bytes() == b"payload"
        assert replacement_descriptor == target_descriptor
        assert os.pread(replacement_descriptor, 7, 0) == b"payload"
        assert any(
            f"descriptor {target_descriptor}" in detail
            for detail in immutable.rollback_guard_cleanup_diagnostics()
        )

        immutable._close_rollback_guards((), label="later guard cleanup")
        assert os.pread(replacement_descriptor, 7, 0) == b"payload"
    finally:
        if replacement_descriptor >= 0:
            original_close(replacement_descriptor)


def test_regular_publication_retires_post_rename_writer_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifact.bin"
    original_open = immutable.os.open
    original_fsync = immutable._fsync_or_fail
    writer = -1
    mutated = False

    def retain_partial_writer(path, flags, *args, **kwargs):
        nonlocal writer
        descriptor = original_open(path, flags, *args, **kwargs)
        if (
            path == ".artifact.bin.m5-partial"
            and flags & os.O_RDWR
            and writer < 0
        ):
            writer = os.dup(descriptor)
        return descriptor

    def mutate_after_rename(descriptor: int, label: str) -> None:
        nonlocal mutated
        original_fsync(descriptor, label)
        if label == "parent after artifact.bin" and not mutated:
            assert writer >= 0
            os.pwrite(writer, b"X", 0)
            mutated = True

    monkeypatch.setattr(immutable.os, "open", retain_partial_writer)
    monkeypatch.setattr(immutable, "_fsync_or_fail", mutate_after_rename)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="published regular-file bytes changed",
        ):
            immutable.publish_regular_file(
                tmp_path, output.name, b"payload",
            )
    finally:
        if writer >= 0:
            os.close(writer)

    assert mutated is True
    assert not output.exists()
    retired = [
        path for path in tmp_path.iterdir()
        if path.name.startswith(".failed-regular-")
    ]
    assert len(retired) == 1
    assert retired[0].read_bytes() == b"Xayload"


def test_read_regular_snapshot_enforces_byte_bound(tmp_path: Path) -> None:
    report = _regular(tmp_path / "bounded.json", b"12345")
    with pytest.raises(
        immutable.ImmutableArtifactLimitError,
        match="bounded snapshot limit",
    ):
        immutable.read_regular_snapshot(report, max_bytes=4)


def test_file_descriptor_headroom_uses_resource_taxonomy(monkeypatch) -> None:
    monkeypatch.setattr(
        immutable, "_available_file_descriptor_capacity", lambda: 7,
    )
    with pytest.raises(
        immutable.ImmutableArtifactResourceError,
        match=immutable.VERIFY_RESOURCE_BOUND,
    ):
        immutable.require_file_descriptor_headroom(8, label="fixture retention")
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="nonnegative integer",
    ):
        immutable.require_file_descriptor_headroom(True, label="fixture retention")


def test_retained_regular_publication_returns_exact_live_capability(
    tmp_path: Path,
) -> None:
    payload = b"published locator\n"
    retained = immutable.publish_retained_regular_file(
        tmp_path, "locator.json", payload,
    )
    try:
        assert retained.path == (tmp_path / "locator.json").absolute()
        assert retained.read_bytes() == payload
        assert retained.snapshot == immutable.snapshot_regular_identity(
            tmp_path / "locator.json"
        )
        retained.verify_unchanged()
    finally:
        retained.close()


def test_failed_retained_publication_preserves_replaced_inode_and_winner(
    tmp_path: Path,
) -> None:
    payload = b"published locator\n"
    winner_payload = b"concurrent winner\n"
    output = tmp_path / "locator.json"
    retained = immutable.publish_retained_regular_file(
        tmp_path, output.name, payload,
    )
    replacement = tmp_path / ".replacement"
    replacement.write_bytes(winner_payload)
    replacement.chmod(0o444)
    os.replace(replacement, output)
    winner_identity = (output.stat().st_dev, output.stat().st_ino)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="named regular-file path changed",
        ):
            retained.verify_unchanged()
        preserved = immutable.preserve_retained_regular_file(
            retained, expected_payload=payload,
        )
    finally:
        retained.close()

    assert (output.stat().st_dev, output.stat().st_ino) == winner_identity
    assert output.read_bytes() == winner_payload
    assert preserved.read_bytes() == payload
    assert preserved.name.startswith(".failed-regular-")


def test_failed_retained_publication_retires_same_inode_mutation(
    tmp_path: Path,
) -> None:
    payload = b"published locator\n"
    mutated_payload = b"mutated locator!\n"
    output = tmp_path / "locator.json"
    retained = immutable.publish_retained_regular_file(
        tmp_path, output.name, payload,
    )
    output.chmod(0o644)
    output.write_bytes(mutated_payload)
    try:
        with pytest.raises(immutable.ImmutableArtifactError):
            retained.verify_unchanged()
        preserved = immutable.preserve_retained_regular_file(
            retained, expected_payload=payload,
        )
    finally:
        retained.close()

    assert not output.exists()
    assert preserved.read_bytes() == mutated_payload


def test_authenticated_retained_close_catches_monitor_close_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_payload = b"first locator\n"
    second_payload = b"second locator\n"
    winner_payload = b"concurrent winner\n"
    first = immutable.publish_retained_regular_file(
        tmp_path, "first.json", first_payload,
    )
    second = immutable.publish_retained_regular_file(
        tmp_path, "second.json", second_payload,
    )
    lock_reader = immutable.PinnedDirectoryReader(tmp_path)
    second_state = second._monitor_state
    assert second_state is not None
    original_monitor_close = second_state.monitor.close
    winner_identity: tuple[int, int] | None = None

    def replace_during_monitor_close() -> None:
        nonlocal winner_identity
        replacement = tmp_path / ".winner"
        replacement.write_bytes(winner_payload)
        replacement.chmod(0o444)
        os.replace(replacement, second.path)
        winner_identity = (
            second.path.stat().st_dev,
            second.path.stat().st_ino,
        )
        original_monitor_close()

    monkeypatch.setattr(
        second_state.monitor, "close", replace_during_monitor_close,
    )
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="named regular-file path changed",
        ):
            immutable.close_retained_regular_files_authenticated(
                (first, second), lock_reader=lock_reader,
            )

        # Neither pinned generation is released when either close authentication
        # fails, even though the first mutation monitor already closed.
        assert first.descriptor >= 0
        assert second.descriptor >= 0
        first_preserved = immutable.preserve_retained_regular_file(
            first, expected_payload=first_payload,
        )
        second_preserved = immutable.preserve_retained_regular_file(
            second, expected_payload=second_payload,
        )
    finally:
        first.close()
        second.close()
        lock_reader.close()

    assert winner_identity is not None
    assert (
        second.path.stat().st_dev,
        second.path.stat().st_ino,
    ) == winner_identity
    assert second.path.read_bytes() == winner_payload
    assert first_preserved.read_bytes() == first_payload
    assert second_preserved.read_bytes() == second_payload


def test_authenticated_retained_close_catches_final_pinned_close_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_payload = b"first locator\n"
    second_payload = b"second locator\n"
    winner_payload = b"final-close winner\n"
    first = immutable.publish_retained_regular_file(
        tmp_path, "first.json", first_payload,
    )
    second = immutable.publish_retained_regular_file(
        tmp_path, "second.json", second_payload,
    )
    lock_reader = immutable.PinnedDirectoryReader(tmp_path)
    original_pinned_close = second._pinned.close
    winner_identity: tuple[int, int] | None = None

    def replace_during_pinned_close() -> None:
        nonlocal winner_identity
        replacement = tmp_path / ".final-close-winner"
        replacement.write_bytes(winner_payload)
        replacement.chmod(0o444)
        os.replace(replacement, second.path)
        winner_identity = (
            second.path.stat().st_dev,
            second.path.stat().st_ino,
        )
        original_pinned_close()

    monkeypatch.setattr(second._pinned, "close", replace_during_pinned_close)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="named regular-file path changed",
        ):
            immutable.close_retained_regular_files_authenticated(
                (first, second), lock_reader=lock_reader,
            )
        assert first.descriptor >= 0
        assert second.descriptor >= 0
        first_preserved = immutable.preserve_retained_regular_file(
            first, expected_payload=first_payload,
        )
        second_preserved = immutable.preserve_retained_regular_file(
            second, expected_payload=second_payload,
        )
    finally:
        first.close()
        second.close()
        lock_reader.close()

    assert winner_identity is not None
    assert (
        second.path.stat().st_dev,
        second.path.stat().st_ino,
    ) == winner_identity
    assert second.path.read_bytes() == winner_payload
    assert first_preserved.read_bytes() == first_payload
    assert second_preserved.read_bytes() == second_payload


def test_authenticated_retained_close_never_retries_reused_guard_numbers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = immutable.publish_retained_regular_file(
        tmp_path, "first.json", b"first locator\n",
    )
    second = immutable.publish_retained_regular_file(
        tmp_path, "second.json", b"second locator\n",
    )
    lock_reader = immutable.PinnedDirectoryReader(tmp_path)
    reuse_source = os.open(os.devnull, os.O_RDONLY)
    original_close_descriptors = immutable._close_descriptors
    release_calls: list[tuple[int, ...]] = []
    reused_descriptors: list[int] = []

    def close_reuse_and_report_error(
        descriptors,
        *,
        active_error=None,
        label="descriptor close",
    ) -> None:
        materialized = tuple(descriptors)
        if label != "authenticated retained close guard release":
            original_close_descriptors(
                materialized,
                active_error=active_error,
                label=label,
            )
            return
        release_calls.append(materialized)
        for descriptor in materialized:
            os.close(descriptor)
            os.dup2(reuse_source, descriptor)
            reused_descriptors.append(descriptor)
        raise immutable.ImmutableArtifactError(
            f"planted guard close reporting failure {len(release_calls)}"
        )

    monkeypatch.setattr(
        immutable, "_close_descriptors", close_reuse_and_report_error,
    )
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="planted guard close reporting failure 1",
        ) as caught:
            immutable.close_retained_regular_files_authenticated(
                (first, second), lock_reader=lock_reader,
            )

        assert len(release_calls) == 2
        assert all(release_call for release_call in release_calls)
        assert len(reused_descriptors) == len(set(reused_descriptors))
        assert any(
            "planted guard close reporting failure 2" in note
            for note in getattr(caught.value, "__notes__", ())
        )
        first.verify_unchanged()
        second.verify_unchanged()
        for descriptor in reused_descriptors:
            assert stat.S_ISCHR(os.fstat(descriptor).st_mode)
    finally:
        first.close()
        second.close()
        lock_reader.close()
        for descriptor in set(reused_descriptors):
            try:
                os.close(descriptor)
            except OSError as close_error:
                if close_error.errno != errno.EBADF:
                    raise
        os.close(reuse_source)


def test_regular_publication_rejects_nonexact_pinned_parent_before_mutation(
    tmp_path: Path,
) -> None:
    class LookalikePinnedParent:
        descriptor = -1

        def verify_path(self) -> None:
            pass

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="exact PinnedDirectoryReader",
    ):
        immutable.publish_regular_file(
            tmp_path,
            "locator.json",
            b"payload",
            pinned_parent=LookalikePinnedParent(),  # type: ignore[arg-type]
        )

    assert not (tmp_path / "locator.json").exists()
    assert not (tmp_path / ".locator.json.m5-partial").exists()


def test_regular_publication_rejects_mismatched_pinned_parent_before_mutation(
    tmp_path: Path,
) -> None:
    declared = tmp_path / "declared"
    other = tmp_path / "other"
    declared.mkdir()
    other.mkdir()
    with immutable.PinnedDirectoryReader(other) as wrong_parent:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="disagrees with parent path",
        ):
            immutable.publish_regular_file(
                declared,
                "locator.json",
                b"payload",
                pinned_parent=wrong_parent,
            )

    for parent in (declared, other):
        assert not (parent / "locator.json").exists()
        assert not (parent / ".locator.json.m5-partial").exists()


def test_regular_publication_rejects_parent_recreate_during_pinned_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = tmp_path / "container"
    parent = container / "published"
    displaced = container / ".published-displaced"
    container.mkdir()
    parent.mkdir()
    payload = b"original publication\n"
    winner_payload = b"wrong recreated-parent winner\n"
    expected_names = tuple(parent.absolute().parts[1:])
    original_close = immutable._PinnedDirectory.close
    swapped = False

    def swap_parent_during_close(pinned) -> None:
        nonlocal swapped
        if not swapped and pinned.names == expected_names:
            try:
                os.stat("locator.json", dir_fd=pinned.descriptor)
            except FileNotFoundError:
                pass
            else:
                os.rename(parent, displaced)
                parent.mkdir()
                winner = parent / "locator.json"
                winner.write_bytes(winner_payload)
                winner.chmod(0o444)
                swapped = True
        original_close(pinned)

    monkeypatch.setattr(
        immutable._PinnedDirectory, "close", swap_parent_during_close,
    )
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="parent generation changed during closure",
    ):
        immutable.publish_regular_file(parent, "locator.json", payload)

    assert swapped is True
    assert (parent / "locator.json").read_bytes() == winner_payload
    assert not (displaced / "locator.json").exists()
    preserved = tuple(displaced.glob(".failed-regular-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == payload


@pytest.mark.parametrize("resource_errno", (errno.EMFILE, errno.ENFILE))
def test_retained_regular_open_translates_late_descriptor_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_errno: int,
) -> None:
    report = _regular(tmp_path / "report.json", b"payload")

    def exhaust_open(_path: Path, *, allow_hardlinks: bool = False):
        del allow_hardlinks
        raise OSError(resource_errno, "planted retained-open exhaustion")

    monkeypatch.setattr(
        immutable, "_open_regular_with_early_path_monitor", exhaust_open,
    )
    with pytest.raises(
        immutable.ImmutableArtifactResourceError,
        match=immutable.VERIFY_RESOURCE_BOUND,
    ):
        immutable.open_retained_regular(report)


@pytest.mark.parametrize("resource_errno", (errno.EMFILE, errno.ENFILE))
def test_stage_creation_translates_late_descriptor_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_errno: int,
) -> None:
    root = tmp_path / "generations"
    root.mkdir()

    def exhaust_stage_open(_root_descriptor: int, _parts: tuple[str, ...]):
        raise OSError(resource_errno, "planted stage-open exhaustion")

    monkeypatch.setattr(
        immutable, "_open_pinned_relative_directory", exhaust_stage_open,
    )
    with pytest.raises(
        immutable.ImmutableArtifactResourceError,
        match=immutable.VERIFY_RESOURCE_BOUND,
    ):
        immutable.create_exclusive_directory_stage(root, "attempt-fixed")

    # The uncertain stage remains as non-active evidence; no final generation
    # was committed.
    assert (root / ".attempt-fixed.staging").is_dir()
    assert not (root / "attempt-fixed").exists()


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


@pytest.mark.parametrize(
    "unsupported_errno",
    [
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    ],
)
def test_rename_noreplace_maps_unsupported_kernel_to_unavailable(
    monkeypatch, unsupported_errno,
) -> None:
    class RenameAt2:
        argtypes = None
        restype = None

        def __call__(self, *_args):
            return -1

    class Libc:
        renameat2 = RenameAt2()

    monkeypatch.setattr(immutable.ctypes, "CDLL", lambda *_args, **_kwargs: Libc())
    monkeypatch.setattr(
        immutable.ctypes, "get_errno", lambda: unsupported_errno,
    )

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="RENAME_NOREPLACE is unavailable",
    ):
        immutable._rename_noreplace_at(1, "source", 2, "destination")


def test_generation_identity_dataclasses_reject_boolean_integer_fields(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="size_bytes must be a nonnegative integer",
    ):
        immutable.DirectoryGenerationFile(
            "value", "a" * 64, True,
        )

    identity = immutable.DirectoryGenerationFile(
        "value", "a" * 64, 1,
    )
    base = {
        "path": tmp_path / "generation",
        "content_manifest_sha256": "b" * 64,
        "filesystem_generation_sha256": "c" * 64,
        "file_count": 1,
        "directory_count": 1,
        "total_file_bytes": 1,
        "files": (identity,),
    }
    for field in ("file_count", "directory_count", "total_file_bytes"):
        invalid = dict(base)
        invalid[field] = True
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match=rf"{field} must be a nonnegative integer",
        ):
            immutable.ImmutableDirectoryGeneration(**invalid)


@pytest.mark.parametrize(
    "destination_kind",
    ("regular", "directory", "symlink", "hardlink", "fifo"),
)
def test_external_generation_rejects_every_preexisting_destination_class(
    tmp_path: Path,
    destination_kind: str,
) -> None:
    root = tmp_path / "generations"
    root.mkdir()
    destination = root / "attempt-fixed"
    retained = tmp_path / "retained"
    retained.write_bytes(b"retained bytes")
    if destination_kind == "regular":
        destination.write_bytes(b"planted")
    elif destination_kind == "directory":
        destination.mkdir()
    elif destination_kind == "symlink":
        destination.symlink_to(retained)
    elif destination_kind == "hardlink":
        os.link(retained, destination)
    else:
        os.mkfifo(destination)

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="destination already exists",
    ):
        immutable.create_exclusive_directory_stage(root, "attempt-fixed")

    assert destination.lstat()
    assert retained.read_bytes() == b"retained bytes"


@pytest.mark.parametrize(
    "stage_kind",
    ("regular", "directory", "symlink", "hardlink", "fifo"),
)
def test_external_generation_never_reuses_preexisting_stage_class(
    tmp_path: Path,
    stage_kind: str,
) -> None:
    root = tmp_path / "generations"
    root.mkdir()
    stage_path = root / ".attempt-fixed.staging"
    retained = tmp_path / "retained"
    retained.write_bytes(b"retained bytes")
    if stage_kind == "regular":
        stage_path.write_bytes(b"partial")
    elif stage_kind == "directory":
        stage_path.mkdir()
    elif stage_kind == "symlink":
        stage_path.symlink_to(retained)
    elif stage_kind == "hardlink":
        os.link(retained, stage_path)
    else:
        os.mkfifo(stage_path)

    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="stage already exists",
    ):
        immutable.create_exclusive_directory_stage(root, "attempt-fixed")

    assert stage_path.lstat()
    assert retained.read_bytes() == b"retained bytes"


@pytest.mark.parametrize("entry_kind", ("symlink", "hardlink", "fifo"))
def test_external_generation_preserves_invalid_staging_tree(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    root = tmp_path / "generations"
    retained = tmp_path / "retained"
    retained.write_bytes(b"retained bytes")
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    try:
        (stage.staging_path / "valid").write_bytes(b"valid")
        planted = stage.staging_path / "planted"
        if entry_kind == "symlink":
            planted.symlink_to(retained)
        elif entry_kind == "hardlink":
            os.link(retained, planted)
        else:
            os.mkfifo(planted)

        with pytest.raises(immutable.ImmutableArtifactError):
            stage.commit(max_files=8, max_directories=2)

        assert not stage.final_path.exists()
        assert stage.staging_path.is_dir()
        assert planted.lstat()
        assert retained.read_bytes() == b"retained bytes"
    finally:
        stage.close()


def test_external_generation_has_no_partial_final_visibility_and_seals_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    try:
        (stage.staging_path / "nested").mkdir()
        (stage.staging_path / "nested" / "value").write_bytes(b"payload")
        assert stage.staging_path.is_dir()
        assert not stage.final_path.exists()

        generation = stage.commit(
            max_files=4,
            max_directories=4,
            max_total_file_bytes=1024,
            max_total_relative_path_bytes=1024,
        )
        assert generation.path == stage.final_path
        assert generation.file_count == 1
        assert generation.directory_count == 2
        assert generation.total_file_bytes == len(b"payload")
        assert not stage.staging_path.exists()
        assert stage.final_path.is_dir()
        assert stat.S_IMODE(stage.final_path.stat().st_mode) == 0o555
        assert stat.S_IMODE(
            (stage.final_path / "nested" / "value").stat().st_mode
        ) == 0o444
        observed = immutable.inspect_immutable_directory_generation(
            stage.final_path,
            max_files=4,
            max_directories=4,
            max_total_file_bytes=1024,
            max_total_relative_path_bytes=1024,
        )
        assert observed == generation
    finally:
        stage.close()


def test_generation_per_file_limit_precedes_initial_content_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    payload = b"bounded generation payload"
    (stage.staging_path / "value").write_bytes(payload)

    def reject_hash(_descriptor: int):
        raise AssertionError("content hashing preceded per-file size admission")

    monkeypatch.setattr(immutable, "_hash_fd", reject_hash)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactLimitError,
            match="bounded per-file byte limit",
        ):
            stage.commit(
                max_files=2,
                max_directories=2,
                max_total_file_bytes=len(payload),
                max_total_relative_path_bytes=1024,
                max_file_bytes=len(payload) - 1,
            )
        assert stage.staging_path.is_dir()
        assert not stage.final_path.exists()
    finally:
        stage.close()
        if stage.staging_path.exists():
            stage.staging_path.chmod(0o755)


def test_generation_path_specific_limit_precedes_initial_content_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    payload = b"bounded stats payload"
    (stage.staging_path / "stats.csv").write_bytes(payload)

    def reject_hash(_descriptor: int):
        raise AssertionError(
            "content hashing preceded path-specific size admission"
        )

    monkeypatch.setattr(immutable, "_hash_fd", reject_hash)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactLimitError,
            match="bounded per-file byte limit: stats.csv",
        ):
            stage.commit(
                max_files=2,
                max_directories=2,
                max_total_file_bytes=len(payload),
                max_total_relative_path_bytes=1024,
                max_file_bytes=len(payload),
                max_file_bytes_by_path={"stats.csv": len(payload) - 1},
            )
        assert stage.staging_path.is_dir()
        assert not stage.final_path.exists()
    finally:
        stage.close()
        if stage.staging_path.exists():
            stage.staging_path.chmod(0o755)


@pytest.mark.parametrize("limit_kind", ("per_file", "aggregate"))
def test_generation_hash_rechecks_growth_against_byte_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_kind: str,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    payload = b"bounded generation payload"
    growth = b"x" * (1024 * 1024)
    value = stage.staging_path / "value"
    value.write_bytes(payload)
    writer = os.open(value, os.O_WRONLY)
    target_inode = os.fstat(writer).st_ino
    original_hash_fd = immutable._hash_fd
    grew = False
    hash_completed = False
    observed_limit: int | None = None

    def grow_immediately_before_hash(
        descriptor: int,
        *,
        allow_hardlinks: bool = False,
        max_bytes: int | None = None,
    ) -> tuple[str, int, int]:
        nonlocal grew, hash_completed, observed_limit
        if os.fstat(descriptor).st_ino == target_inode and not grew:
            observed_limit = max_bytes
            os.pwrite(writer, growth, len(payload))
            os.fsync(writer)
            grew = True
        result = original_hash_fd(
            descriptor,
            allow_hardlinks=allow_hardlinks,
            max_bytes=max_bytes,
        )
        if os.fstat(descriptor).st_ino == target_inode:
            hash_completed = True
        return result

    monkeypatch.setattr(
        immutable,
        "_hash_fd",
        grow_immediately_before_hash,
    )
    limits = {
        "max_total_file_bytes": (
            len(payload) if limit_kind == "aggregate" else len(payload) + len(growth)
        ),
        "max_file_bytes": (
            len(payload) if limit_kind == "per_file" else len(payload) + len(growth)
        ),
    }
    try:
        with pytest.raises(immutable.ImmutableArtifactError):
            stage.commit(
                max_files=2,
                max_directories=2,
                max_total_relative_path_bytes=1024,
                **limits,
            )
        assert grew is True
        assert observed_limit == len(payload)
        assert hash_completed is False
        assert stage.staging_path.is_dir()
        assert not stage.final_path.exists()
    finally:
        os.close(writer)
        stage.close()
        if stage.staging_path.exists():
            stage.staging_path.chmod(0o755)


def test_external_generation_rejects_same_byte_replacement_before_postcommit_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    (stage.staging_path / "value").write_bytes(b"captured")
    original_snapshot = immutable._snapshot_pinned_tree_at
    replacement_inode: int | None = None

    def replace_after_rename(*args, **kwargs):
        nonlocal replacement_inode
        if stage.committed and replacement_inode is None:
            target = stage.final_path / "value"
            stage.final_path.chmod(0o755)
            try:
                target.unlink()
                target.write_bytes(b"captured")
                target.chmod(0o444)
                replacement_inode = target.stat().st_ino
            finally:
                stage.final_path.chmod(0o555)
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(
        immutable, "_snapshot_pinned_tree_at", replace_after_rename,
    )
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="child generation changed across commit",
        ):
            stage.commit(max_files=2, max_directories=2)
        assert replacement_inode is not None
        assert stage.final_path.is_dir()
        assert (stage.final_path / "value").stat().st_ino == replacement_inode
    finally:
        stage.close()
        if stage.final_path.exists():
            stage.final_path.chmod(0o755)


def test_external_generation_local_seal_rejects_same_byte_inode_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    try:
        (stage.staging_path / "value").write_bytes(b"captured")
        committed = stage.commit(max_files=2, max_directories=2)
    finally:
        stage.close()

    target = committed.path / "value"
    original_inode = target.stat().st_ino
    committed.path.chmod(0o755)
    try:
        target.unlink()
        target.write_bytes(b"captured")
        target.chmod(0o444)
    finally:
        committed.path.chmod(0o555)
    observed = immutable.inspect_immutable_directory_generation(
        committed.path,
        max_files=2,
        max_directories=2,
    )
    assert target.stat().st_ino != original_inode
    assert observed.content_manifest_sha256 == committed.content_manifest_sha256
    assert (
        observed.filesystem_generation_sha256
        != committed.filesystem_generation_sha256
    )
    # Test-only permission restoration for pytest cleanup.
    committed.path.chmod(0o755)


def test_external_generation_destination_race_preserves_stage_and_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    (stage.staging_path / "value").write_bytes(b"captured")
    original_rename = immutable._rename_noreplace_at

    def plant_winner(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        if source_name != stage.staging_name:
            return original_rename(
                source_descriptor,
                source_name,
                destination_descriptor,
                destination_name,
            )
        os.mkdir(destination_name, dir_fd=destination_descriptor)
        raise FileExistsError(errno.EEXIST, "planted destination race")

    monkeypatch.setattr(immutable, "_rename_noreplace_at", plant_winner)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="destination race rejected",
        ):
            stage.commit(max_files=4, max_directories=4)
        assert stage.staging_path.is_dir()
        assert stage.final_path.is_dir()
        assert (stage.staging_path / "value").read_bytes() == b"captured"
    finally:
        stage.close()


def test_external_generation_interrupted_commit_preserves_hidden_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    (stage.staging_path / "value").write_bytes(b"captured")

    def interrupt(*_args) -> None:
        raise RuntimeError("planted commit interruption")

    monkeypatch.setattr(immutable, "_rename_noreplace_at", interrupt)
    try:
        with pytest.raises(RuntimeError, match="commit interruption"):
            stage.commit(max_files=4, max_directories=4)
        assert stage.staging_path.is_dir()
        assert not stage.final_path.exists()
        assert (stage.staging_path / "value").read_bytes() == b"captured"
        assert stat.S_IMODE(
            (stage.staging_path / "value").stat().st_mode
        ) == 0o444
    finally:
        stage.close()


def test_external_generation_preserves_limit_error_taxonomy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    try:
        (stage.staging_path / "value").write_bytes(b"captured")
        with pytest.raises(immutable.ImmutableArtifactLimitError):
            stage.commit(max_files=0, max_directories=2)
        assert stage.staging_path.is_dir()
        assert not stage.final_path.exists()
        assert (stage.staging_path / "value").read_bytes() == b"captured"
    finally:
        stage.close()


def test_external_generation_preserves_resource_error_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    (stage.staging_path / "value").write_bytes(b"captured")

    def fail_monitor(*_args, **_kwargs):
        raise immutable.ImmutableArtifactResourceError(
            "VERIFY_RESOURCE_BOUND: planted monitor exhaustion"
        )

    monkeypatch.setattr(
        immutable, "_arm_directory_mutation_monitor", fail_monitor
    )
    try:
        with pytest.raises(
            immutable.ImmutableArtifactResourceError,
            match="planted monitor exhaustion",
        ):
            stage.commit(max_files=2, max_directories=2)
        assert stage.staging_path.is_dir()
        assert not stage.final_path.exists()
        assert (stage.staging_path / "value").read_bytes() == b"captured"
    finally:
        stage.close()
        # Test-only permission restoration for pytest cleanup.
        if stage.staging_path.exists():
            stage.staging_path.chmod(0o755)


@pytest.mark.parametrize("resource_errno", (errno.EMFILE, errno.ENFILE))
def test_external_generation_types_raw_descriptor_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_errno: int,
) -> None:
    root = tmp_path / "generations"
    stage = immutable.create_exclusive_directory_stage(root, "attempt-fixed")
    (stage.staging_path / "value").write_bytes(b"captured")

    def exhaust_descriptors(_descriptor: int) -> int:
        raise OSError(resource_errno, "planted descriptor exhaustion")

    monkeypatch.setattr(immutable.os, "dup", exhaust_descriptors)
    try:
        with pytest.raises(
            immutable.ImmutableArtifactResourceError,
            match=immutable.VERIFY_RESOURCE_BOUND,
        ) as failure:
            stage.commit(max_files=2, max_directories=2)
        assert isinstance(failure.value.__cause__, BaseException)
        assert stage.staging_path.is_dir()
        assert not stage.final_path.exists()
    finally:
        stage.close()
        if stage.staging_path.exists():
            stage.staging_path.chmod(0o755)
