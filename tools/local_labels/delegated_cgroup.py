#!/usr/bin/env python3
"""Per-launch cgroup-v2 containment for bounded capture subprocesses."""

from __future__ import annotations

import hashlib
import os
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

CGROUP_V2_KILL_V1 = "cgroup_v2_kill_v1"
_CGROUP_MOUNT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
_PROC_SELF_MOUNTINFO = Path("/proc/self/mountinfo")
_MAX_PROC_BYTES = 1024 * 1024
_MAX_COMPONENT_BYTES = 255
_CLEANUP_TIMEOUT_SECONDS = 5.0


class CgroupContainmentError(RuntimeError):
    """A delegated cgroup could not provide complete descendant containment."""


@dataclass(frozen=True)
class CgroupContainmentEvidence:
    kind: str
    parent_cgroup_sha256: str
    cgroup_inode: int
    terminal_populated_zero: bool
    leaf_removed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "parent_cgroup_sha256": self.parent_cgroup_sha256,
            "cgroup_inode": self.cgroup_inode,
            "terminal_populated_zero": self.terminal_populated_zero,
            "leaf_removed": self.leaf_removed,
        }


def _read_bounded(path: Path, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CgroupContainmentError(f"cannot open {label}: {exc}") from exc
    chunks: list[bytes] = []
    retained = 0
    try:
        while retained <= _MAX_PROC_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, _MAX_PROC_BYTES + 1 - retained),
            )
            if not chunk:
                break
            chunks.append(chunk)
            retained += len(chunk)
    except OSError as exc:
        raise CgroupContainmentError(f"cannot read {label}: {exc}") from exc
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) > _MAX_PROC_BYTES:
        raise CgroupContainmentError(f"{label} exceeds {_MAX_PROC_BYTES} bytes")
    return payload


def _current_cgroup_components() -> tuple[tuple[str, ...], str]:
    payload = _read_bounded(_PROC_SELF_CGROUP, "unified cgroup membership")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CgroupContainmentError(
            "unified cgroup membership is not strict UTF-8"
        ) from exc
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0].startswith("0::/"):
        raise CgroupContainmentError(
            "process does not have one absolute unified cgroup-v2 membership"
        )
    relative = lines[0][4:]
    components = tuple(component for component in relative.split("/") if component)
    for component in components:
        try:
            encoded = component.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise CgroupContainmentError(
                "cgroup membership contains a non-UTF-8 component"
            ) from exc
        if (
            component in {".", ".."}
            or not encoded
            or len(encoded) > _MAX_COMPONENT_BYTES
            or "\x00" in component
        ):
            raise CgroupContainmentError(
                "cgroup membership contains an unsafe path component"
            )
    return components, lines[0][3:]


def _require_cgroup_v2_mount() -> None:
    payload = _read_bounded(_PROC_SELF_MOUNTINFO, "mount table")
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise CgroupContainmentError("mount table is not strict UTF-8") from exc
    matches = []
    for line in lines:
        before, separator, after = line.partition(" - ")
        fields = before.split()
        after_fields = after.split()
        if (
            separator
            and len(fields) >= 5
            and fields[4] == str(_CGROUP_MOUNT)
            and len(after_fields) >= 1
            and after_fields[0] == "cgroup2"
        ):
            matches.append(line)
    if len(matches) != 1:
        raise CgroupContainmentError(
            "/sys/fs/cgroup is not one exact cgroup-v2 mount"
        )


def _open_current_cgroup(components: Sequence[str]) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(_CGROUP_MOUNT, flags)
    except OSError as exc:
        raise CgroupContainmentError(f"cannot open cgroup-v2 mount: {exc}") from exc
    try:
        for component in components:
            replacement = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = replacement
    except OSError as exc:
        os.close(descriptor)
        raise CgroupContainmentError(
            f"cannot retain current delegated cgroup: {exc}"
        ) from exc
    return descriptor


def _open_control(directory_fd: int, name: str, flags: int) -> int:
    try:
        return os.open(name, flags | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as exc:
        raise CgroupContainmentError(
            f"cannot open delegated cgroup control {name}: {exc}"
        ) from exc


def _write_control(descriptor: int, payload: bytes, label: str) -> None:
    try:
        written = os.write(descriptor, payload)
    except OSError as exc:
        raise CgroupContainmentError(f"cannot write {label}: {exc}") from exc
    if written != len(payload):
        raise CgroupContainmentError(f"short write to {label}")


class DelegatedCgroupV2:
    """One retained cgroup leaf containing session and process-group escapes."""

    def __init__(
        self,
        *,
        parent_fd: int,
        leaf_fd: int,
        procs_fd: int,
        kill_fd: int,
        events_fd: int,
        leaf_name: str,
        parent_cgroup: str,
        cgroup_inode: int,
    ) -> None:
        self._parent_fd = parent_fd
        self._leaf_fd = leaf_fd
        self._procs_fd = procs_fd
        self._kill_fd = kill_fd
        self._events_fd = events_fd
        self._leaf_name = leaf_name
        self._parent_cgroup = parent_cgroup
        self._cgroup_inode = cgroup_inode
        self._launched = False
        self._empty = False
        self._removed = False

    @classmethod
    def create(cls) -> "DelegatedCgroupV2":
        if threading.active_count() != 1:
            raise CgroupContainmentError(
                "delegated cgroup launch requires a single-threaded parent"
            )
        _require_cgroup_v2_mount()
        components, parent_cgroup = _current_cgroup_components()
        parent_fd = _open_current_cgroup(components)
        leaf_name = f"gpu-sim-{os.getpid()}-{secrets.token_hex(8)}"
        leaf_fd = procs_fd = kill_fd = events_fd = -1
        created = False
        try:
            os.mkdir(leaf_name, mode=0o700, dir_fd=parent_fd)
            created = True
            leaf_fd = os.open(
                leaf_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            procs_fd = _open_control(leaf_fd, "cgroup.procs", os.O_WRONLY)
            kill_fd = _open_control(leaf_fd, "cgroup.kill", os.O_WRONLY)
            events_fd = _open_control(leaf_fd, "cgroup.events", os.O_RDONLY)
            inode = os.fstat(leaf_fd).st_ino
            if inode <= 0:
                raise CgroupContainmentError(
                    "delegated cgroup leaf has no stable inode identity"
                )
            return cls(
                parent_fd=parent_fd,
                leaf_fd=leaf_fd,
                procs_fd=procs_fd,
                kill_fd=kill_fd,
                events_fd=events_fd,
                leaf_name=leaf_name,
                parent_cgroup=parent_cgroup,
                cgroup_inode=inode,
            )
        except BaseException as active_error:
            for descriptor in (events_fd, kill_fd, procs_fd, leaf_fd):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            if created:
                try:
                    os.rmdir(leaf_name, dir_fd=parent_fd)
                except OSError as cleanup_error:
                    try:
                        active_error.add_note(
                            f"empty delegated cgroup cleanup failed: {cleanup_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
            os.close(parent_fd)
            if isinstance(active_error, CgroupContainmentError):
                raise
            if not isinstance(active_error, (OSError, ValueError)):
                raise
            raise CgroupContainmentError(
                f"cannot create delegated cgroup leaf: {active_error}"
            ) from active_error

    def popen(
        self,
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.Popen[bytes]:
        if self._removed or self._launched or self._procs_fd < 0:
            raise CgroupContainmentError(
                "delegated cgroup leaf is not available for one launch"
            )
        if threading.active_count() != 1:
            raise CgroupContainmentError(
                "delegated cgroup launch requires a single-threaded parent"
            )
        if "preexec_fn" in kwargs:
            raise CgroupContainmentError(
                "delegated cgroup launch does not accept another pre-exec hook"
            )
        inherited = tuple(kwargs.pop("pass_fds", ()))
        if self._procs_fd in inherited:
            raise CgroupContainmentError(
                "delegated cgroup descriptor aliases an inherited descriptor"
            )
        procs_fd = self._procs_fd

        def join_leaf() -> None:
            try:
                if os.write(procs_fd, b"0\n") != 2:
                    raise OSError("short cgroup.procs write")
            finally:
                os.close(procs_fd)

        kwargs["pass_fds"] = (*inherited, procs_fd)
        kwargs["preexec_fn"] = join_leaf
        self._launched = True
        try:
            try:
                return subprocess.Popen(command, **kwargs)
            finally:
                os.close(self._procs_fd)
                self._procs_fd = -1
        except BaseException as active_error:
            cleanup_error: BaseException | None = None
            try:
                self.close_after_error()
            except BaseException as exc:
                cleanup_error = exc
            if cleanup_error is not None:
                wrapped = CgroupContainmentError(
                    "delegated cgroup launch cleanup failed: "
                    f"{cleanup_error}"
                )
                try:
                    wrapped.add_note(
                        f"active delegated cgroup launch error: {active_error}"
                    )
                except (AttributeError, TypeError):
                    pass
                raise wrapped from cleanup_error
            if isinstance(
                active_error,
                (OSError, ValueError, subprocess.SubprocessError),
            ):
                raise CgroupContainmentError(
                    "cannot launch process inside delegated cgroup: "
                    f"{active_error}"
                ) from active_error
            raise

    def kill_all(self) -> None:
        if self._removed:
            raise CgroupContainmentError("delegated cgroup leaf is already removed")
        _write_control(self._kill_fd, b"1\n", "cgroup.kill")

    def _populated(self) -> bool:
        try:
            payload = os.pread(self._events_fd, 4096, 0)
        except OSError as exc:
            raise CgroupContainmentError(
                f"cannot read retained cgroup.events: {exc}"
            ) from exc
        try:
            lines = payload.decode("ascii", errors="strict").splitlines()
        except UnicodeDecodeError as exc:
            raise CgroupContainmentError(
                "retained cgroup.events is not ASCII"
            ) from exc
        values: dict[str, str] = {}
        for line in lines:
            fields = line.split()
            if len(fields) != 2 or fields[0] in values:
                raise CgroupContainmentError(
                    "retained cgroup.events has malformed fields"
                )
            values[fields[0]] = fields[1]
        if values.get("populated") not in {"0", "1"}:
            raise CgroupContainmentError(
                "retained cgroup.events has no typed populated state"
            )
        return values["populated"] == "1"

    def wait_empty(self, timeout_seconds: float = _CLEANUP_TIMEOUT_SECONDS) -> None:
        deadline = time.monotonic() + timeout_seconds
        while self._populated():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CgroupContainmentError(
                    "delegated cgroup remained populated after cgroup.kill"
                )
            time.sleep(min(0.01, remaining))

    def _close_controls(self) -> None:
        errors: list[OSError] = []
        for attribute in ("_events_fd", "_kill_fd", "_leaf_fd"):
            descriptor = getattr(self, attribute)
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except OSError as exc:
                errors.append(exc)
            else:
                setattr(self, attribute, -1)
        if errors:
            detail = "; ".join(str(error) for error in errors)
            raise CgroupContainmentError(
                f"cannot close empty delegated cgroup controls: {detail}"
            ) from errors[0]

    def _remove_empty_leaf(self) -> None:
        self._close_controls()
        try:
            os.rmdir(self._leaf_name, dir_fd=self._parent_fd)
        except OSError as exc:
            raise CgroupContainmentError(
                f"cannot remove empty delegated cgroup leaf: {exc}"
            ) from exc
        self._removed = True
        os.close(self._parent_fd)
        self._parent_fd = -1

    def finish(self) -> CgroupContainmentEvidence:
        if self._removed:
            raise CgroupContainmentError("delegated cgroup leaf is already removed")
        if not self._empty:
            self.kill_all()
            self.wait_empty()
            self._empty = True
        self._remove_empty_leaf()
        return CgroupContainmentEvidence(
            kind=CGROUP_V2_KILL_V1,
            parent_cgroup_sha256=hashlib.sha256(
                self._parent_cgroup.encode("utf-8", errors="strict")
            ).hexdigest(),
            cgroup_inode=self._cgroup_inode,
            terminal_populated_zero=True,
            leaf_removed=True,
        )

    def close_after_error(self) -> None:
        if self._removed:
            return
        if self._procs_fd >= 0:
            os.close(self._procs_fd)
            self._procs_fd = -1
        if not self._empty:
            if self._launched:
                self.kill_all()
            self.wait_empty()
            self._empty = True
        self._remove_empty_leaf()
