#!/usr/bin/env python3
"""Capture one diagnostic CUTLASS successor NCU or NVBit phase.

The command is deliberately fail-closed.  It validates an operator-authored
successor preflight against the immutable 237f17c cohort, then checks the live
host before starting the selected process.  A receipt is written only after
label-driver or NVBit capture, artifact hashing, and semantic validation all
pass.  This tool never edits canonical or successor manifests.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local_labels import admit_cutlass_successor_capture as admission  # noqa: E402
from tools.local_labels import campaign_preflight  # noqa: E402
from tools.local_labels import delegated_cgroup  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import metric_registry  # noqa: E402
from tools.local_labels import ncu_action_executor as ncu_executor  # noqa: E402
from tools.local_labels import parse_ncu_metrics  # noqa: E402
from tools.local_labels import sidecar_emitter  # noqa: E402
from tools.local_labels.capture_toolchain_contract import (  # noqa: E402
    parse_nvcc_release_banner,
    require_sol_cuda_release,
    ncu_version_matches,
)
from tools.local_labels import build_cutlass_successor_cohort as cohort  # noqa: E402
from tools.nvbit_tracer.import_trace_contract import validate_trace_dir  # noqa: E402
from tools.nvbit_tracer import runtime_env as nvbit_runtime  # noqa: E402
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    _import_trace_pb2,
    require_trace_admission,
)

PINNED_COMMIT = admission.PINNED_BUNDLE_COMMIT
ARCH = "sm_86"
SKU = "rtx_3090_ti"
CUDA_RELEASE = {"major": 13, "minor": 2}
NVBIT_VERSION = admission.NVBIT_VERSION
NCU_VERSION = "2026.1.0"
GPU_NAME = "NVIDIA GeForce RTX 3090 Ti"
DRIVER_VERSION = "610.43.03"
SM_CLOCK_MHZ = 1350
MEM_CLOCK_MHZ = 10501
TRACER_SO = ROOT / admission.NVBIT_TRACER_REL
NVBIT_PB_LIB = ROOT / admission.NVBIT_PB_LIB_REL
NVBIT_VERSION_HEADER = ROOT / "tools/nvbit_tracer/nvbit_release/core/nvbit.h"
LABEL_DRIVER = ROOT / "tools/local_labels/label_driver.sh"
CUDA_NVCC = Path("/usr/local/cuda-13.2/bin/nvcc")
CAPTURE_TIMEOUT_S = 3600
CAPTURE_STREAM_LIMIT_BYTES = 8 * 1024 * 1024
CAPTURE_STREAM_READ_BYTES = 64 * 1024
CAPTURE_PROCESS_POLL_S = 0.05
CAPTURE_CLEANUP_TIMEOUT_S = 5.0
CAPTURE_PRODUCER_SELF_SHA256 = (
    "89e23e8cdc27fee62f65a5222f06db3ced76c9f355af07f4665739adf7f235ee"
)
EXPECTED_TRACE_CONTRACT_ID = admission.TRACE_CONTRACT_ID
EXPECTED_MEMORY_PAYLOAD_CLASSES = admission.TRACE_MEMORY_PAYLOAD_CLASSES
_LABEL_COLLECTION_PATH = "/usr/local/cuda-13.2/bin:/usr/bin:/bin"
_LABEL_COLLECTION_ENVIRONMENT_NAMES = (
    admission.NCU_COLLECTION_OPTIONAL_ENVIRONMENT_NAMES
)
_NCU_HANDOFF_STDERR = (
    b"label_driver: NCU_LAUNCH_SET_POSTPROCESSING_UNIMPLEMENTED: "
    b"explicit launch-set report retained without a sidecar\n"
)
_NCU_INITIAL_ARTIFACT_NAMES = frozenset({
    "label_run_manifest_0_launch_set.json",
    "run_0_launch_set.ncu-rep",
})
_NCU_COMPLETE_ARTIFACT_NAMES = frozenset({
    *_NCU_INITIAL_ARTIFACT_NAMES,
    "label_0_0.json",
    "run_0_launch_set.launch_observation.json",
})
_NCU_PERCENTAGE_METRICS = (
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
)
_NCU_UNIT_METRICS = (
    "gpc__cycles_elapsed.max",
    "gpu__time_duration.sum",
)


class CaptureError(RuntimeError):
    """A preflight or capture contract failed closed."""

    def __init__(
        self,
        detail: str,
        *,
        process_id: int | None = None,
        started_at_utc: str | None = None,
        process_role: str | None = None,
        process_command: list[str] | None = None,
        process_environment: dict[str, str] | None = None,
        process_tool_sha256: str | None = None,
        containment: delegated_cgroup.CgroupContainmentEvidence | None = None,
    ) -> None:
        self.detail = detail
        self.process_id = process_id
        self.started_at_utc = started_at_utc
        self.process_role = process_role
        self.process_command = process_command
        self.process_environment = process_environment
        self.process_tool_sha256 = process_tool_sha256
        self.containment = containment
        super().__init__(detail)


def canonical_sha(value: Any) -> str:
    return admission.canonical_sha(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _host_identity_sha256(live: dict[str, Any]) -> str:
    return canonical_sha({
        "gpu": live["gpu_name"],
        "smi": live["smi_stdout_sha256"],
        "gpu_uuid_sha256": live["gpu_uuid_sha256"],
        "driver_version": live["driver_version"],
        "identity_stdout_sha256": live["identity_stdout_sha256"],
    })


def _regular(path: Path, label: str) -> None:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise CaptureError(f"{label} unavailable: {path}") from exc
    if path.is_symlink() or not path.is_file() or stat.st_nlink != 1:
        raise CaptureError(f"{label} is not a single regular file: {path}")


def _capture_producer_identity() -> tuple[Path, str]:
    source = Path(__file__).resolve(strict=True)
    try:
        with immutable.open_retained_regular(source) as retained:
            payload = retained.read_bytes()
            declared = admission._capture_producer_self_digest(payload)
            if declared != CAPTURE_PRODUCER_SELF_SHA256:
                raise CaptureError(
                    "loaded capture producer disagrees with its named source self-seal"
                )
            if declared != admission.CAPTURE_PRODUCER_SELF_SHA256_AUTHORITY:
                raise CaptureError(
                    "capture producer differs from independent self-seal authority"
                )
            return source, retained.sha256
    except (admission.AdmissionError, immutable.ImmutableArtifactError) as exc:
        raise CaptureError(
            f"capture producer immutable identity failed: {exc}"
        ) from exc


def _absolute_external(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise CaptureError(f"{label} must be an absolute path")
    candidate = Path(path.anchor)
    for component in path.parts[1:]:
        candidate /= component
        if candidate.is_symlink():
            raise CaptureError(f"{label} path contains a symlink: {candidate}")
    if path.is_symlink():
        raise CaptureError(f"{label} is a symlink: {path}")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise CaptureError(f"{label} must be external to the repository: {path}")
    return resolved


def _check_dt(value: str, label: str) -> None:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise CaptureError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CaptureError(f"{label} must include a timezone")


def _load_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _regular(path, label)
    payload = path.read_bytes()
    try:
        value = immutable.strict_json_loads(payload, label=label)
    except immutable.ImmutableArtifactError as exc:
        raise CaptureError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"{label} is not an object")
    return value, payload


def _hash_repo_file(rel: str, expected: str, label: str) -> None:
    path = ROOT / rel
    _regular(path, label)
    observed = sha256_bytes(path.read_bytes())
    if observed != expected:
        raise CaptureError(f"{label} hash drifted: {rel}")


def _row_for(matrix_key: str, variant_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _, _, roster, matrix = admission.validate_immutable_bundle()
    rows = {
        (row["matrix_key"], row["variant_id"]): row for row in matrix["rows"]
    }
    key = (matrix_key, variant_id)
    if key not in rows:
        raise CaptureError(f"unknown successor matrix_key+variant_id: {key}")
    roster_rows = {(row["matrix_key"], row["variant_id"]): row for row in roster["rows"]}
    if key not in roster_rows:
        raise CaptureError(f"successor roster row is missing: {key}")
    return rows[key], roster_rows[key]


def _expected_preflight(row: dict[str, Any], roster_row: dict[str, Any]) -> dict[str, Any]:
    shape = row["shape"]
    static = row["static_launch_inventory"]
    witness = roster_row["witness"]
    return {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_capture_preflight",
        "matrix_key": row["matrix_key"],
        "variant_id": row["variant_id"],
        "arch": ARCH,
        "sku": SKU,
        "build_arch": ARCH,
        "source_commit": PINNED_COMMIT,
        "binary_path": static["binary_path"],
        "binary_sha256": static["binary_sha256"],
        "witness_path": witness["path"],
        "witness_sha256": witness["sha256"],
        "shape_path": shape["path"],
        "shape_sha256": shape["sha256"],
        "no_competing_processes": True,
        "competing_processes": [],
        "gpu_name": GPU_NAME,
        "persistence_mode": "Enabled",
        "sm_clock_mhz": SM_CLOCK_MHZ,
        "mem_clock_mhz": MEM_CLOCK_MHZ,
        "cuda_release": CUDA_RELEASE,
        "nvbit_version": NVBIT_VERSION,
        "ncu_version": NCU_VERSION,
    }


def validate_successor_preflight(
    path: Path, row: dict[str, Any], roster_row: dict[str, Any]
) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute() or path.is_symlink():
        raise CaptureError("successor preflight must be an absolute regular file")
    document, payload = _load_object(path, "successor preflight")
    expected = _expected_preflight(row, roster_row)
    if document != expected:
        mismatches = [key for key in expected if document.get(key) != expected[key]]
        extras = sorted(set(document) - set(expected))
        raise CaptureError(
            f"successor preflight pins mismatch fields={mismatches!r} extras={extras!r}"
        )
    _hash_repo_file(expected["binary_path"], expected["binary_sha256"], "successor binary")
    _hash_repo_file(expected["witness_path"], expected["witness_sha256"], "successor witness")
    _hash_repo_file(expected["shape_path"], expected["shape_sha256"], "successor shape")
    if expected["source_commit"] != PINNED_COMMIT:
        raise CaptureError("successor preflight source commit is not the immutable pin")
    return expected, payload


def _reject_preexisting_output_descendants(
    root: Path, *, output_dir: Path, reserved_files: tuple[Path, ...]
) -> None:
    """Reject stale entries only in this phase's output namespace."""
    if root.is_symlink() or not root.is_dir():
        raise CaptureError(f"capture output root is not a directory: {root}")

    def check_components(path: Path) -> None:
        try:
            parts = path.relative_to(root).parts
        except ValueError as exc:
            raise CaptureError(f"capture output path escapes root: {path}") from exc
        candidate = root
        for component in parts:
            candidate /= component
            if candidate.exists() or candidate.is_symlink():
                if candidate.is_symlink() or not candidate.is_dir():
                    raise CaptureError(
                        f"pre-existing output descendant is unsafe: {candidate}"
                    )

    check_components(output_dir)
    if output_dir.exists():
        for current, dirs, files in os.walk(output_dir, topdown=True, followlinks=False):
            current_path = Path(current)
            for name in list(dirs):
                path = current_path / name
                if path.is_symlink() or not path.is_dir():
                    raise CaptureError(
                        f"pre-existing output descendant is unsafe: {path}"
                    )
            for name in files:
                path = current_path / name
                if path.is_symlink() or not path.is_file():
                    raise CaptureError(
                        f"pre-existing output descendant is unsafe: {path}"
                    )
            if files:
                raise CaptureError(
                    f"pre-existing capture output must be empty: {current_path / files[0]}"
                )

    for path in reserved_files:
        check_components(path.parent)
        if path.exists() or path.is_symlink():
            raise CaptureError(f"pre-existing output artifact must be removed: {path}")


def _capture_tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    if root.is_symlink() or not root.is_dir():
        raise CaptureError(f"capture output descendant is unsafe: {root}")
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(dirs):
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise CaptureError(f"capture output descendant is unsafe: {path}")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise CaptureError(f"capture output descendant is unsafe: {path}")
            total += path.stat().st_size
    return total


def _require_capture_tree_budget(root: Path, byte_budget: int) -> None:
    if byte_budget <= 0:
        raise CaptureError("capture byte budget must be positive")
    try:
        total = _capture_tree_bytes(root)
    except OSError as exc:
        raise CaptureError(f"capture output budget scan failed: {exc}") from exc
    if total > byte_budget:
        raise CaptureError(
            "post-processed capture exceeds its trace byte budget"
        )


def _kill_capture_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _process_exited_unreaped(process: subprocess.Popen[bytes]) -> bool:
    """Observe child exit without releasing its PID/process-group authority."""
    while True:
        try:
            status = os.waitid(
                os.P_PID,
                process.pid,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except InterruptedError:
            continue
        except (ChildProcessError, OSError) as exc:
            raise CaptureError(
                f"capture process exit state is unavailable: {exc}"
            ) from exc
        return status is not None


def _read_capture_streams(
    selector: selectors.BaseSelector,
    buffers: dict[str, bytearray],
    *,
    timeout_s: float,
) -> tuple[bool, OSError | None, bool]:
    exceeded = False
    progressed = False
    try:
        ready = selector.select(timeout_s)
    except OSError as exc:
        return False, exc, False
    for key, _mask in ready:
        label = key.data
        try:
            chunk = os.read(key.fd, CAPTURE_STREAM_READ_BYTES)
        except BlockingIOError:
            continue
        except OSError as exc:
            return exceeded, exc, progressed
        progressed = True
        if not chunk:
            selector.unregister(key.fileobj)
            continue
        retained = buffers[label]
        remaining = CAPTURE_STREAM_LIMIT_BYTES - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > remaining:
            exceeded = True
    return exceeded, None, progressed


def _run_capture_command(
    command: list[str], env: dict[str, str], cwd: Path, *,
    timeout_s: float = CAPTURE_TIMEOUT_S, byte_root: Path | None = None,
    byte_budget: int | None = None, failure_root: Path | None = None,
    inherited_descriptors: tuple[int, ...] = (),
    require_cgroup_containment: bool = False,
) -> tuple[
    int, int, str, bytes, bytes,
    delegated_cgroup.CgroupContainmentEvidence | None,
]:
    if type(require_cgroup_containment) is not bool:
        raise CaptureError("cgroup containment requirement must be boolean")
    if timeout_s <= 0:
        raise CaptureError("capture timeout must be positive")
    if byte_budget is not None and byte_budget <= 0:
        raise CaptureError("capture byte budget must be positive")
    if CAPTURE_STREAM_LIMIT_BYTES <= 0 or CAPTURE_STREAM_READ_BYTES <= 0:
        raise CaptureError("capture stream limits must be positive")
    started = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    _check_dt(started, "process start")
    exceeded = threading.Event()
    stop_watchdog = threading.Event()

    def watch_bytes() -> None:
        while not stop_watchdog.wait(0.1):
            try:
                if (
                    byte_root is not None
                    and byte_budget is not None
                    and _capture_tree_bytes(byte_root) > byte_budget
                ):
                    exceeded.set()
                    return
            except (CaptureError, OSError):
                exceeded.set()
                return

    containment: delegated_cgroup.DelegatedCgroupV2 | None = None
    if require_cgroup_containment:
        try:
            containment = delegated_cgroup.DelegatedCgroupV2.create()
        except delegated_cgroup.CgroupContainmentError as exc:
            raise CaptureError(
                f"capture descendant containment is unavailable: {exc}"
            ) from exc
    popen_arguments = {
        "cwd": str(cwd),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "start_new_session": True,
        "pass_fds": inherited_descriptors,
    }
    try:
        process = (
            subprocess.Popen(command, **popen_arguments)
            if containment is None
            else containment.popen(command, **popen_arguments)
        )
    except (OSError, delegated_cgroup.CgroupContainmentError) as exc:
        if containment is not None:
            try:
                containment.close_after_error()
            except BaseException as cleanup_error:
                try:
                    exc.add_note(
                        f"capture containment launch cleanup failed: {cleanup_error}"
                    )
                except (AttributeError, TypeError):
                    pass
        raise CaptureError(f"capture process could not start: {exc}") from exc

    selector: selectors.BaseSelector | None = None
    watchdog: threading.Thread | None = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout_s
    failure_kind: str | None = None
    failure_cause: BaseException | None = None
    output_exceeded = False
    containment_evidence: delegated_cgroup.CgroupContainmentEvidence | None = None
    containment_error: BaseException | None = None
    group_kill_error: BaseException | None = None
    process_cleanup_error: BaseException | None = None
    process_wait_error: BaseException | None = None
    active_base_exception: BaseException | None = None
    try:
        if process.stdout is None or process.stderr is None:  # pragma: no cover
            raise CaptureError(
                "capture process streams are unavailable",
                process_id=process.pid,
                started_at_utc=started,
            )
        selector = selectors.DefaultSelector()
        try:
            for label, stream in (
                ("stdout", process.stdout),
                ("stderr", process.stderr),
            ):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, label)
        except (OSError, ValueError) as exc:
            raise CaptureError(
                f"capture process stream setup failed: {exc}",
                process_id=process.pid,
                started_at_utc=started,
            ) from exc

        watchdog = threading.Thread(target=watch_bytes, daemon=True)
        watchdog.start()
        while True:
            now = time.monotonic()
            wait_s = max(0.0, min(CAPTURE_PROCESS_POLL_S, deadline - now))
            stream_exceeded, stream_error, _progressed = _read_capture_streams(
                selector, buffers, timeout_s=wait_s,
            )
            output_exceeded = output_exceeded or stream_exceeded
            if stream_error is not None:
                failure_kind = "stream"
                failure_cause = stream_error
                break
            if output_exceeded:
                failure_kind = "output_limit"
                break
            if exceeded.is_set():
                failure_kind = "budget"
                break
            if time.monotonic() >= deadline:
                failure_kind = "timeout"
                break
            if _process_exited_unreaped(process):
                break
    except CaptureError as exc:
        failure_kind = "setup"
        failure_cause = exc
    except Exception as exc:
        failure_kind = "stream"
        failure_cause = exc
    except BaseException as exc:
        active_base_exception = exc
    finally:
        # Bound every teardown phase. Production launches also retain cgroup
        # authority, so process-group escapes are killed before leader reaping.
        stop_watchdog.set()
        try:
            _kill_capture_process(process)
        except BaseException as exc:
            if isinstance(exc, Exception):
                group_kill_error = exc
            elif active_base_exception is None:
                active_base_exception = exc
            else:
                try:
                    active_base_exception.add_note(
                        f"capture process-group kill interrupted: {exc}"
                    )
                except (AttributeError, TypeError):
                    pass
        if containment is not None:
            try:
                containment.kill_all()
            except BaseException as exc:
                containment_error = exc
        if watchdog is not None:
            try:
                watchdog.join(timeout=CAPTURE_CLEANUP_TIMEOUT_S)
            except BaseException as exc:
                if isinstance(exc, Exception):
                    process_cleanup_error = exc
                elif active_base_exception is None:
                    active_base_exception = exc
                else:
                    try:
                        active_base_exception.add_note(
                            f"capture watchdog join interrupted: {exc}"
                        )
                    except (AttributeError, TypeError):
                        pass
            if watchdog.is_alive():
                timeout_error = RuntimeError(
                    "capture byte-budget watchdog did not stop within "
                    f"{CAPTURE_CLEANUP_TIMEOUT_S:g}s"
                )
                if process_cleanup_error is None:
                    process_cleanup_error = timeout_error
                else:
                    try:
                        process_cleanup_error.add_note(str(timeout_error))
                    except (AttributeError, TypeError):
                        pass
        try:
            process.wait(timeout=CAPTURE_CLEANUP_TIMEOUT_S)
        except BaseException as exc:
            process_wait_error = exc
            if not isinstance(exc, Exception):
                if active_base_exception is None:
                    active_base_exception = exc
                else:
                    try:
                        active_base_exception.add_note(
                            f"capture process wait interrupted: {exc}"
                        )
                    except (AttributeError, TypeError):
                        pass
        if containment is not None:
            try:
                containment_evidence = containment.finish()
            except BaseException as exc:
                if containment_error is not None:
                    try:
                        exc.add_note(
                            f"initial cgroup.kill failed: {containment_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
                containment_error = exc
        if group_kill_error is not None and containment_evidence is None:
            if process_cleanup_error is None:
                process_cleanup_error = group_kill_error
            else:
                try:
                    process_cleanup_error.add_note(
                        f"capture process-group kill failed: {group_kill_error}"
                    )
                except (AttributeError, TypeError):
                    pass
        if process_wait_error is not None:
            try:
                process.wait(timeout=CAPTURE_CLEANUP_TIMEOUT_S)
            except BaseException as retry_error:
                retry_failure = (
                    retry_error
                    if isinstance(retry_error, Exception)
                    else RuntimeError(
                        "capture process retry wait was interrupted before "
                        "leader reaping was proven"
                    )
                )
                if process_cleanup_error is None:
                    process_cleanup_error = retry_failure
                else:
                    try:
                        process_cleanup_error.add_note(
                            f"capture process retry wait failed: {retry_failure}"
                        )
                    except (AttributeError, TypeError):
                        pass
                try:
                    process_cleanup_error.add_note(
                        f"initial capture process wait failed: {process_wait_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            else:
                process_wait_error = None
        if selector is not None:
            try:
                while selector.get_map() and not output_exceeded:
                    stream_exceeded, stream_error, progressed = (
                        _read_capture_streams(
                            selector, buffers, timeout_s=0,
                        )
                    )
                    output_exceeded = output_exceeded or stream_exceeded
                    if stream_error is not None and failure_cause is None:
                        failure_cause = stream_error
                        failure_kind = failure_kind or "stream"
                        break
                    if not progressed:
                        break
                for key in list(selector.get_map().values()):
                    selector.unregister(key.fileobj)
                selector.close()
            except BaseException as exc:
                if process_cleanup_error is None:
                    process_cleanup_error = exc
                else:
                    try:
                        process_cleanup_error.add_note(
                            f"capture stream cleanup failed: {exc}"
                        )
                    except (AttributeError, TypeError):
                        pass
        for stream in (process.stdout, process.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except BaseException as exc:
                if process_cleanup_error is None:
                    process_cleanup_error = exc
                else:
                    try:
                        process_cleanup_error.add_note(
                            f"capture pipe cleanup failed: {exc}"
                        )
                    except (AttributeError, TypeError):
                        pass

    stdout = bytes(buffers["stdout"])
    stderr = bytes(buffers["stderr"])
    if containment_error is not None:
        error = CaptureError(
            f"capture descendant containment cleanup failed: {containment_error}",
            process_id=process.pid,
            started_at_utc=started,
        )
        if active_base_exception is not None:
            try:
                error.add_note(
                    "active capture interruption was superseded by containment "
                    f"cleanup failure: {active_base_exception}"
                )
            except (AttributeError, TypeError):
                pass
        if process_cleanup_error is not None:
            try:
                error.add_note(
                    f"capture process cleanup also failed: {process_cleanup_error}"
                )
            except (AttributeError, TypeError):
                pass
        raise error from containment_error
    if process_cleanup_error is not None:
        error = CaptureError(
            f"capture process cleanup failed: {process_cleanup_error}",
            process_id=process.pid,
            started_at_utc=started,
            containment=containment_evidence,
        )
        if active_base_exception is not None:
            try:
                error.add_note(
                    "active capture interruption was superseded by process "
                    f"cleanup failure: {active_base_exception}"
                )
            except (AttributeError, TypeError):
                pass
        if failure_cause is not None:
            try:
                error.add_note(f"active capture failure: {failure_cause}")
            except (AttributeError, TypeError):
                pass
        raise error from process_cleanup_error
    if active_base_exception is not None:
        raise active_base_exception
    if failure_kind == "setup":
        assert isinstance(failure_cause, CaptureError)
        failure_cause.containment = containment_evidence
        raise failure_cause
    if failure_kind is None and output_exceeded:
        failure_kind = "output_limit"
    if failure_kind is None and exceeded.is_set():
        failure_kind = "budget"
    if byte_root is not None and byte_budget is not None:
        try:
            terminal_bytes = _capture_tree_bytes(byte_root)
        except (CaptureError, OSError):
            failure_kind = failure_kind or "budget"
        else:
            if terminal_bytes > byte_budget:
                failure_kind = failure_kind or "budget"

    if failure_kind == "timeout":
        if failure_root is not None:
            _write_new(failure_root / "capture_timeout.stdout.bin", stdout)
            _write_new(failure_root / "capture_timeout.stderr.bin", stderr)
        raise CaptureError(
            f"capture process exceeded timeout of {timeout_s:g}s",
            process_id=process.pid,
            started_at_utc=started,
            containment=containment_evidence,
        ) from failure_cause
    if failure_kind == "output_limit":
        if failure_root is not None:
            _write_new(failure_root / "capture_output_limit.stdout.bin", stdout)
            _write_new(failure_root / "capture_output_limit.stderr.bin", stderr)
        raise CaptureError(
            "capture process exceeded its bounded stdout/stderr byte limit",
            process_id=process.pid,
            started_at_utc=started,
            containment=containment_evidence,
        ) from failure_cause
    if failure_kind == "stream":
        if failure_root is not None:
            _write_new(failure_root / "capture_stream_error.stdout.bin", stdout)
            _write_new(failure_root / "capture_stream_error.stderr.bin", stderr)
        raise CaptureError(
            f"capture process stream collection failed: {failure_cause}",
            process_id=process.pid,
            started_at_utc=started,
            containment=containment_evidence,
        ) from failure_cause
    if failure_kind == "budget":
        if failure_root is not None:
            _write_new(failure_root / "capture_budget.stdout.bin", stdout)
            _write_new(failure_root / "capture_budget.stderr.bin", stderr)
        raise CaptureError(
            "capture process exceeded its trace byte budget or wrote unsafe output",
            process_id=process.pid,
            started_at_utc=started,
            containment=containment_evidence,
        )
    return (
        process.returncode,
        process.pid,
        started,
        stdout,
        stderr,
        containment_evidence,
    )


def _boot_id_sha256() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    _regular(path, "Linux boot identity")
    payload = path.read_bytes().strip()
    if re.fullmatch(
        rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        payload,
    ) is None:
        raise CaptureError("Linux boot identity is malformed")
    return sha256_bytes(payload)


def _query(command: list[str], label: str) -> tuple[str, list[dict[str, str]]]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureError(f"{label} query failed: {exc}") from exc
    if result.returncode != 0:
        raise CaptureError(f"{label} query failed with status {result.returncode}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return result.stdout, [{"stdout": line} for line in lines]


def _nvbit_version() -> str:
    _regular(NVBIT_VERSION_HEADER, "NVBit version header")
    matches = re.findall(r'#define\s+NVBIT_VERSION\s+"([^"]+)"', NVBIT_VERSION_HEADER.read_text())
    if matches != [NVBIT_VERSION]:
        raise CaptureError(f"NVBit version pin mismatch: {matches!r}")
    return matches[0]


def _ncu_identity_from_authority(
    authority: dict[str, Any],
) -> ncu_executor.ExpectedNcuIdentity:
    required = {
        "status",
        "expected",
        "execution_mode",
        "host_machine",
        "launcher_resolved_path",
        "launcher_sha256",
        "launcher_target_relpath",
        "resolved_path",
        "executable_sha256",
        "version_line",
    }
    if type(authority) is not dict or set(authority) != required:
        raise CaptureError("live NCU authority fields differ")
    if (
        authority["status"] != campaign_preflight.STATUS_OK
        or authority["expected"] != NCU_VERSION
        or authority["execution_mode"] != campaign_preflight.NCU_EXECUTION_MODE
        or authority["host_machine"] != campaign_preflight.NCU_HOST_MACHINE
        or authority["launcher_target_relpath"]
        != campaign_preflight.NCU_LAUNCHER_TARGET_RELPATH
    ):
        raise CaptureError("live NCU authority pins differ")
    launcher_raw = authority["launcher_resolved_path"]
    resolved_raw = authority["resolved_path"]
    if (
        type(launcher_raw) is not str
        or type(resolved_raw) is not str
        or not Path(launcher_raw).is_absolute()
        or not Path(resolved_raw).is_absolute()
        or os.path.normpath(launcher_raw) != launcher_raw
        or os.path.normpath(resolved_raw) != resolved_raw
    ):
        raise CaptureError("live NCU authority paths differ")
    launcher = Path(launcher_raw)
    resolved = Path(resolved_raw)
    if resolved != launcher.parent / authority["launcher_target_relpath"]:
        raise CaptureError("live NCU launcher and inner executable paths differ")
    if not all(
        type(authority[name]) is str
        and re.fullmatch(r"[0-9a-f]{64}", authority[name]) is not None
        for name in ("launcher_sha256", "executable_sha256")
    ):
        raise CaptureError("live NCU authority hashes are invalid")
    try:
        return admission._validate_ncu_authority(authority)
    except admission.AdmissionError as exc:
        raise CaptureError(f"live NCU executable authority is invalid: {exc}") from exc


def _probe_ncu_authority() -> tuple[dict[str, Any], str]:
    observed = campaign_preflight.probe_ncu(
        {"toolchain": {"ncu_version_prefix": NCU_VERSION}},
        require_cgroup_containment=True,
    )
    authority = {
        name: observed.get(name)
        for name in (
            "status",
            "expected",
            "execution_mode",
            "host_machine",
            "launcher_resolved_path",
            "launcher_sha256",
            "launcher_target_relpath",
            "resolved_path",
            "executable_sha256",
            "version_line",
        )
    }
    expected = _ncu_identity_from_authority(authority)
    try:
        result = ncu_executor.execute_ncu_action(
            expected,
            ["--version"],
            timeout_seconds=30,
            require_cgroup_containment=True,
        )
        stdout = result.stdout.decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ncu_executor.NcuActionExecutorError) as exc:
        raise CaptureError(f"live NCU version output is invalid: {exc}") from exc
    expected_stdout = (
        "NVIDIA (R) Nsight Compute Command Line Profiler\n"
        "Copyright (c) 2018-2026 NVIDIA Corporation\n"
        f"{expected.version_line}\n"
    )
    if result.returncode != 0 or result.stderr or stdout != expected_stdout:
        raise CaptureError("live NCU version output differs from exact authority")
    return authority, stdout


def _label_driver_identity() -> ncu_executor.ExpectedExecutableIdentity:
    try:
        with immutable.open_retained_executable(LABEL_DRIVER) as retained:
            if retained.sha256 != admission.LABEL_DRIVER_SHA256:
                raise CaptureError(
                    "label driver differs from the pinned capture authority"
                )
            mode = stat.S_IMODE(os.fstat(retained.descriptor).st_mode)
            if mode != admission.LABEL_DRIVER_MODE:
                raise CaptureError(
                    "label driver mode differs from the pinned capture authority"
                )
            expected = ncu_executor.build_expected_executable_identity(
                resolved_path=str(retained.path),
                executable_sha256=retained.sha256,
                executable_mode=admission.LABEL_DRIVER_MODE,
            )
            retained.verify_unchanged()
    except (
        OSError,
        immutable.ImmutableArtifactError,
        ncu_executor.NcuActionExecutorError,
    ) as exc:
        raise CaptureError(f"label driver authority is invalid: {exc}") from exc
    return expected


def _label_collection_environment(
    source: dict[str, str], *, cwd: Path,
) -> ncu_executor.NcuCollectionEnvironment:
    resolved_cwd = cwd.resolve(strict=True)
    values = {
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": _LABEL_COLLECTION_PATH,
        "PWD": str(resolved_cwd),
        "PYTHONNOUSERSITE": "1",
        "LABEL_DRIVER_GPU_INDEX": "0",
        "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
    }
    for name in _LABEL_COLLECTION_ENVIRONMENT_NAMES:
        value = source.get(name)
        if value:
            values[name] = value
    try:
        return ncu_executor.NcuCollectionEnvironment(values)
    except ncu_executor.NcuActionExecutorError as exc:
        raise CaptureError(f"label collection environment is invalid: {exc}") from exc


def _command_option(command: list[str], name: str) -> str:
    if command.count(name) != 1:
        raise CaptureError(f"NCU command requires exactly one {name}")
    index = command.index(name)
    if index + 1 >= len(command) or command[index + 1] == "--":
        raise CaptureError(f"NCU command has no value for {name}")
    return command[index + 1]


def _run_ncu_capture_command(
    command: list[str],
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    environment: ncu_executor.NcuCollectionEnvironment,
    driver_identity: ncu_executor.ExpectedExecutableIdentity,
    retained_target: immutable.RetainedExecutable,
    cwd: Path,
    *,
    timeout_s: float = CAPTURE_TIMEOUT_S,
    failure_root: Path | None = None,
) -> tuple[
    int, int, str, bytes, bytes,
    delegated_cgroup.CgroupContainmentEvidence,
]:
    if not timeout_s > 0:
        raise CaptureError("capture timeout must be positive")
    expected_options = {
        "--ncu": expected_ncu_identity.resolved_path,
        "--ncu-executable-sha256": expected_ncu_identity.executable_sha256,
        "--ncu-version-line": expected_ncu_identity.version_line,
        "--target-processes": "all",
        "--filter-mode": "global",
        "--replay-mode": "kernel",
        "--cache-control": "all",
        "--clock-control": "none",
        "--profile-from-start": "on",
        "--launch-skip": "0",
        "--launch-count": "1",
    }
    if command[0] != str(LABEL_DRIVER) or "--ncu-exec-path" in command:
        raise CaptureError("NCU command does not name the canonical label driver")
    for name, expected in expected_options.items():
        if _command_option(command, name) != expected:
            raise CaptureError(f"NCU command authority differs for {name}")
    if command.count("--") != 1:
        raise CaptureError("NCU command requires one application separator")
    supervised_command: list[object] = list(command)
    binary_index = supervised_command.index("--binary") + 1
    if supervised_command[binary_index] != str(retained_target.path):
        raise CaptureError("NCU target command differs from retained executable")
    supervised_command[binary_index] = retained_target.proc_path
    separator = supervised_command.index("--")
    supervised_command[separator:separator] = [
        "--ncu-exec-path",
        ncu_executor.AUTHORIZED_NCU_PROC_PATH,
    ]
    if environment.as_dict()["PWD"] != str(cwd.resolve(strict=True)):
        raise CaptureError("NCU collection PWD differs from the process directory")
    try:
        result = ncu_executor.execute_ncu_supervised_executable(
            expected_ncu_identity,
            driver_identity,
            supervised_command,
            timeout_seconds=timeout_s,
            environment=environment,
            working_directory=cwd,
            auxiliary_descriptors=(retained_target.descriptor,),
            require_cgroup_containment=True,
        )
    except ncu_executor.NcuActionExecutorError as exc:
        if failure_root is not None:
            _write_new(failure_root / "ncu_executor_failure.stdout.bin", exc.stdout)
            _write_new(failure_root / "ncu_executor_failure.stderr.bin", exc.stderr)
        detail = exc.stderr[-400:] if exc.stderr else str(exc).encode()
        process_command = (
            None
            if exc.process_command is None
            else [
                str(retained_target.path)
                if item == retained_target.proc_path
                else item
                for item in exc.process_command
            ]
        )
        raise CaptureError(
            f"descriptor-bound NCU capture failed: {detail!r}",
            process_id=exc.process_id,
            started_at_utc=exc.started_at_utc,
            process_role=exc.process_role,
            process_command=process_command,
            process_environment=exc.process_environment,
            process_tool_sha256=exc.process_tool_sha256,
            containment=exc.containment,
        ) from exc
    if type(result.process_id) is not int or result.process_id <= 0:
        raise CaptureError("descriptor-bound NCU capture returned no process identity")
    if type(result.started_at_utc) is not str:
        raise CaptureError("descriptor-bound NCU capture returned no start time")
    if result.containment is None:
        raise CaptureError(
            "descriptor-bound NCU capture returned no containment evidence"
        )
    _check_dt(result.started_at_utc, "process start")
    return (
        result.returncode,
        result.process_id,
        result.started_at_utc,
        result.stdout,
        result.stderr,
        result.containment,
    )


def validate_live_capture_preflight(*, phase: str) -> dict[str, Any]:
    """Check phase-local live host state only after static preflight pins pass."""
    if phase not in {"ncu", "nvbit"}:
        raise CaptureError("live preflight phase is invalid")
    smi_command = [
        "nvidia-smi", "--query-gpu=name,compute_cap,persistence_mode,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    smi_stdout, _ = _query(smi_command, "nvidia-smi device")
    smi_lines = [line.strip() for line in smi_stdout.splitlines() if line.strip()]
    if len(smi_lines) != 1:
        raise CaptureError(f"expected one pinned GPU, observed={smi_lines!r}")
    fields = [part.strip() for part in smi_lines[0].split(",")]
    if fields != [GPU_NAME, "8.6", "Enabled", str(SM_CLOCK_MHZ), str(MEM_CLOCK_MHZ)]:
        raise CaptureError(f"live GPU state is not pinned: {fields!r}")
    apps_command = [
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits",
    ]
    apps_stdout, _ = _query(apps_command, "nvidia-smi competing-process")
    if apps_stdout.strip() and "no running processes" not in apps_stdout.lower():
        raise CaptureError(f"competing GPU processes are present: {apps_stdout.strip()!r}")
    identity_command = [
        "nvidia-smi", "--query-gpu=uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    identity_stdout, _ = _query(identity_command, "nvidia-smi device identity")
    identity_lines = [
        line.strip() for line in identity_stdout.splitlines() if line.strip()
    ]
    if len(identity_lines) != 1:
        raise CaptureError(
            f"expected one selected GPU identity, observed={identity_lines!r}"
        )
    identity_fields = [part.strip() for part in identity_lines[0].split(",")]
    if (
        len(identity_fields) != 2
        or re.fullmatch(admission.GPU_UUID_PATTERN, identity_fields[0]) is None
        or identity_fields[1] != DRIVER_VERSION
    ):
        raise CaptureError(
            f"live GPU identity is not pinned: {identity_fields!r}"
        )
    gpu_uuid_sha256 = sha256_bytes(identity_fields[0].encode("utf-8"))
    nvcc_command = [str(CUDA_NVCC), "--version"]
    nvcc_stdout, _ = _query(nvcc_command, "CUDA 13.2 nvcc")
    try:
        cuda = parse_nvcc_release_banner(nvcc_stdout)
        require_sol_cuda_release(cuda)
    except ValueError as exc:
        raise CaptureError(f"CUDA release pin failed: {exc}") from exc
    nvbit = _nvbit_version()
    result = {
        "smi_command": smi_command,
        "smi_stdout": smi_stdout,
        "smi_stdout_sha256": sha256_bytes(smi_stdout.encode()),
        "competing_process_command": apps_command,
        "competing_process_stdout": apps_stdout,
        "competing_process_stdout_sha256": sha256_bytes(apps_stdout.encode()),
        "identity_command": identity_command,
        "identity_stdout": identity_stdout,
        "identity_stdout_sha256": sha256_bytes(identity_stdout.encode()),
        "gpu_uuid_sha256": gpu_uuid_sha256,
        "driver_version": DRIVER_VERSION,
        "nvcc_command": nvcc_command,
        "nvcc_stdout": nvcc_stdout,
        "nvcc_stdout_sha256": sha256_bytes(nvcc_stdout.encode()),
        "cuda_release": cuda,
        "nvbit_version": nvbit,
        "gpu_name": GPU_NAME,
        "persistence_mode": "Enabled",
        "sm_clock_mhz": SM_CLOCK_MHZ,
        "mem_clock_mhz": MEM_CLOCK_MHZ,
    }
    if phase == "ncu":
        ncu_authority, ncu_stdout = _probe_ncu_authority()
        if not ncu_version_matches(ncu_authority["version_line"], NCU_VERSION):
            raise CaptureError("NCU version is not the pinned 2026.1.0 release")
        result.update({
            "ncu_command": [ncu_authority["resolved_path"], "--version"],
            "ncu_stdout": ncu_stdout,
            "ncu_stdout_sha256": sha256_bytes(ncu_stdout.encode()),
            "ncu_version": NCU_VERSION,
            "ncu_authority": ncu_authority,
        })
    return result


def _validate_trace_producer(trace_root: Path) -> dict[str, Any]:
    """Bind the serialized trace metadata to this capture's producer contract."""
    try:
        trace_pb2 = _import_trace_pb2()
        trace = trace_pb2.Trace()
        trace.ParseFromString((trace_root / "dynamic_trace.pb").read_bytes())
    except Exception as exc:
        raise CaptureError(f"NVBit trace producer metadata is unreadable: {exc}") from exc
    metadata = {
        "trace_contract_id": trace.trace_contract_id,
        "memory_payload_classes": sorted(trace.memory_payload_classes),
        "binary_version": trace.binary_version,
        "nvbit_version": trace.nvbit_version,
    }
    if metadata["trace_contract_id"] != EXPECTED_TRACE_CONTRACT_ID:
        raise CaptureError("NVBit trace producer/schema contract mismatch")
    if metadata["memory_payload_classes"] != EXPECTED_MEMORY_PAYLOAD_CLASSES:
        raise CaptureError("NVBit trace producer payload-class mismatch")
    if (
        metadata["binary_version"] != admission.NVBIT_TRACE_BINARY_VERSION
        or metadata["nvbit_version"] != NVBIT_VERSION
    ):
        raise CaptureError("NVBit trace producer version mismatch")
    return metadata


def _environment_document(
    preflight: dict[str, Any],
    live: dict[str, Any],
    env: dict[str, str],
    *,
    capture_kind: str,
) -> dict[str, Any]:
    if capture_kind == "ncu":
        selected = dict(env)
    elif capture_kind == "nvbit":
        selected = dict(env)
    else:
        raise CaptureError("environment document capture kind is invalid")
    output_keys = [
        "smi_stdout", "competing_process_stdout", "identity_stdout",
        "nvcc_stdout",
    ]
    if capture_kind == "ncu":
        output_keys.append("ncu_stdout")
    return {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_capture_environment",
        "source_commit": PINNED_COMMIT,
        "preflight_pins_sha256": canonical_sha(preflight),
        "live_checks": {key: value for key, value in live.items() if key not in output_keys},
        "live_outputs": {key: live[key] for key in output_keys},
        "environment": selected,
    }


def _write_new(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with immutable.PinnedDirectoryReader(path.parent) as parent:
            immutable.write_regular_bytes_at(
                parent.descriptor,
                path.name,
                payload,
                mode=0o400,
            )
            parent.verify_path()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise CaptureError(
            f"cannot exclusively publish capture output {path}: {exc}"
        ) from exc


def _capture_metadata_paths(
    root: Path,
    *,
    variant_id: str,
    phase: str,
) -> tuple[Path, Path, Path, Path]:
    return (
        root / f"{variant_id}.preflight_{phase}.json",
        root / f"{variant_id}.environment_{phase}.json",
        root / f"{variant_id}.{phase}{admission.RECEIPT_SUFFIX}",
        root / "failures" / variant_id / phase,
    )


def _verify_unchanged(path: Path, expected_sha256: str, label: str) -> None:
    _regular(path, label)
    if sha256_bytes(path.read_bytes()) != expected_sha256:
        raise CaptureError(f"{label} changed during capture")


def _artifact(root: Path, path: Path, role: str) -> dict[str, Any]:
    _regular(path, f"{role} artifact")
    payload = path.read_bytes()
    if not payload:
        raise CaptureError(f"{role} artifact is zero-sized: {path}")
    relative = path.relative_to(root).as_posix()
    return {"role": role, "path": relative, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def _require_directory_namespace(
    parent: immutable.PinnedDirectoryReader,
    *,
    expected_names: frozenset[str],
    label: str,
) -> None:
    try:
        observed = os.listdir(parent.descriptor)
    except OSError as exc:
        raise CaptureError(f"cannot inspect {label} namespace: {exc}") from exc
    if len(observed) != len(set(observed)) or set(observed) != expected_names:
        raise CaptureError(
            f"{label} namespace differs: "
            f"expected={sorted(expected_names)!r} observed={sorted(observed)!r}"
        )
    parent.verify_path()


def _require_ncu_cell_namespace(
    parent: immutable.PinnedDirectoryReader,
    *,
    expected_names: frozenset[str],
) -> None:
    _require_directory_namespace(
        parent,
        expected_names=expected_names,
        label="NCU handoff",
    )


def _validate_ncu_handoff_manifest(
    document: Any,
    *,
    row: dict[str, Any],
) -> None:
    kernel_signature = hashlib.sha256(
        (
            f"{row['successor_slug']}|"
            f"{row['static_launch_inventory']['binary_path']}"
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "schema_version": 1,
        "source_commit": PINNED_COMMIT,
        "binary_sha256": row["static_launch_inventory"]["binary_sha256"],
        "kernel_signature_sha256": kernel_signature,
        "kernel_signature_source": "fallback_legacy",
        "arch": ARCH,
        "slug": row["successor_slug"],
        "shape_point": row["shape"]["shape_point"],
        "repeat_index": 0,
        "launch_index": 0,
    }
    if type(document) is not dict or document != expected:
        raise CaptureError("NCU handoff run manifest differs from successor authority")


def _build_ncu_sidecar(
    *,
    row: dict[str, Any],
    live: dict[str, Any],
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    imported: parse_ncu_metrics.ImportedReportCsv,
) -> tuple[dict[str, Any], dict[str, Any]]:
    percentage = parse_ncu_metrics.parse_raw_page_csv(
        imported.text,
        _NCU_PERCENTAGE_METRICS,
    )
    if not percentage.ok:
        raise CaptureError(
            "NCU percentage metric extraction failed: "
            f"{percentage.typed_reason}: {percentage.detail}"
        )
    unit_tagged = parse_ncu_metrics.extract_unit_tagged_metrics(
        imported.text,
        metric_registry.load_registry(),
        _NCU_UNIT_METRICS,
    )
    if not unit_tagged.ok:
        raise CaptureError(
            "NCU unit-tagged metric extraction failed: "
            f"{unit_tagged.typed_reason}: {unit_tagged.detail}"
        )
    observation = parse_ncu_metrics.build_launch_observation_document(
        csv_text=imported.text,
        report_sha256=imported.report_sha256,
        requested_launch_skip=0,
        requested_launch_count=1,
    )
    parse_ncu_metrics.validate_launch_observation_document(
        observation,
        requested_launch_skip=0,
        requested_launch_count=1,
        report_sha256=imported.report_sha256,
    )
    identity = observation["identity"]
    launch_config = row["shape"]["trace_shape"]["launch_config"]
    try:
        device = admission._ncu_device_name(identity.get("Device"))
        block_dim = admission._ncu_dimension(
            identity.get("Block Size"), "Block Size"
        )
        grid_dim = admission._ncu_dimension(
            identity.get("Grid Size"), "Grid Size"
        )
        allowed_kernel_names = admission._sass_kernel_names(row)
    except admission.AdmissionError as exc:
        raise CaptureError(f"NCU launch identity validation failed: {exc}") from exc
    if (
        identity.get("ID") != "0"
        or device != GPU_NAME
        or block_dim != launch_config["block"]
        or grid_dim != launch_config["grid"]
        or identity.get("Kernel Name") not in allowed_kernel_names
    ):
        raise CaptureError("NCU launch identity differs from successor authority")

    sidecar_environment = {
        "SIDECAR_DEVICE": ARCH,
        "SIDECAR_SLUG": row["successor_slug"],
        "SIDECAR_SHAPE_POINT": row["shape"]["shape_point"],
        "SIDECAR_REPEAT_INDEX": "0",
        "SIDECAR_LAUNCH_INDEX": "0",
        "SIDECAR_BINARY_SHA256": row["static_launch_inventory"]["binary_sha256"],
        "SIDECAR_PROFILER_VERSION": expected_ncu_identity.version_line,
        "SIDECAR_CUDA_VERSION": "13.2",
        "SIDECAR_DRIVER_VERSION": live["driver_version"],
        "SIDECAR_GPU_UUID_SHA256": live["gpu_uuid_sha256"],
        "SIDECAR_AGGREGATION": "kernel",
        "SIDECAR_UNITS": "sector_hit_rate_fraction",
        "SIDECAR_CONFIDENCE_LEVEL": "0.95",
        "SIDECAR_WARMUP_COUNT": "0",
        "SIDECAR_HIT_RATE_L2": str(
            percentage.fraction_values["lts__t_sector_hit_rate.pct"]
        ),
        "SIDECAR_HIT_RATE_L1TEX": str(
            percentage.fraction_values["l1tex__t_sector_hit_rate.pct"]
        ),
        "SIDECAR_EVIDENCE_CLASS": "campaign",
        "SIDECAR_NCU_CLOCK_CONTROL": "none",
        "SIDECAR_NCU_CACHE_CONTROL": "all",
        "SIDECAR_NCU_REPLAY_MODE": "kernel",
        "SIDECAR_NCU_TARGET_PROCESSES": "all",
        "SIDECAR_NCU_FILTER_MODE": "global",
        "SIDECAR_NCU_PROFILE_FROM_START": "on",
        "SIDECAR_NCU_LAUNCH_SKIP": "0",
        "SIDECAR_NCU_LAUNCH_COUNT": "1",
        "SIDECAR_HOST_GPU_INDEX": "0",
        "SIDECAR_HOST_PERSISTENCE_MODE": live["persistence_mode"],
        "SIDECAR_HOST_SM_CLOCK_MHZ": str(live["sm_clock_mhz"]),
        "SIDECAR_HOST_MEM_CLOCK_MHZ": str(live["mem_clock_mhz"]),
    }
    sidecar = sidecar_emitter.build_sidecar(sidecar_environment)
    diagnostics = parse_ncu_metrics.diagnostics_to_json_shape(
        parse_ncu_metrics._merged_parser_diagnostics_result(
            percentage,
            unit_tagged,
        )
    )
    if diagnostics:
        sidecar["parser_diagnostics"] = diagnostics
    sidecar["unit_tagged_metrics"] = [
        {
            "ncu_metric": name,
            "unit": unit_tagged.metrics[name]["unit"],
            "value": unit_tagged.metrics[name]["value"],
        }
        for name in _NCU_UNIT_METRICS
    ]
    try:
        admission._validate_schema(
            ROOT / admission.LOCAL_LABEL_SIDECAR_SCHEMA_REL,
            sidecar,
            "NCU label sidecar",
        )
        from tools.local_labels import import_label

        result = import_label.validate(sidecar)
    except Exception as exc:
        raise CaptureError(f"NCU label sidecar validation failed: {exc}") from exc
    if not result.ok:
        raise CaptureError(
            "NCU label sidecar semantic validation failed: "
            f"{result.typed_reason}"
        )
    return sidecar, observation


def _postprocess_ncu_handoff(
    *,
    root: Path,
    row: dict[str, Any],
    live: dict[str, Any],
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    generation_stack: contextlib.ExitStack | None = None,
) -> None:
    cell = root / ARCH / row["successor_slug"] / "trace_00"
    report_path = cell / "run_0_launch_set.ncu-rep"
    manifest_path = cell / "label_run_manifest_0_launch_set.json"
    observation_path = cell / "run_0_launch_set.launch_observation.json"
    sidecar_path = cell / "label_0_0.json"
    try:
        with contextlib.ExitStack() as retained_stack:
            parent = retained_stack.enter_context(
                immutable.PinnedDirectoryReader(cell)
            )
            _require_ncu_cell_namespace(
                parent,
                expected_names=_NCU_INITIAL_ARTIFACT_NAMES,
            )
            retained_report = retained_stack.enter_context(
                immutable.open_retained_regular(report_path)
            )
            retained_manifest = retained_stack.enter_context(
                immutable.open_retained_regular(manifest_path)
            )
            if retained_report.size_bytes <= 0 or retained_manifest.size_bytes <= 0:
                raise CaptureError("NCU handoff artifact is empty")
            active_stack = retained_stack
            if generation_stack is not None:
                generation_stack.enter_context(retained_stack.pop_all())
                active_stack = generation_stack
            try:
                manifest = immutable.strict_json_loads(
                    retained_manifest.read_bytes(),
                    label="NCU handoff run manifest",
                )
            except immutable.ImmutableArtifactError as exc:
                raise CaptureError(
                    f"NCU handoff run manifest is not valid JSON: {exc}"
                ) from exc
            _validate_ncu_handoff_manifest(manifest, row=row)
            imported, import_error = parse_ncu_metrics._import_report_csv(
                expected_ncu_identity,
                report_path,
                require_cgroup_containment=True,
            )
            if import_error is not None or imported is None:
                detail = getattr(import_error, "detail", repr(import_error))
                raise CaptureError(f"NCU handoff report import failed: {detail}")
            if (
                imported.report_sha256 != retained_report.sha256
                or imported.report_size_bytes != retained_report.size_bytes
            ):
                raise CaptureError(
                    "NCU imported report identity differs from retained handoff"
                )
            sidecar, observation = _build_ncu_sidecar(
                row=row,
                live=live,
                expected_ncu_identity=expected_ncu_identity,
                imported=imported,
            )
            sidecar_payload = immutable.pretty_json_bytes(sidecar)
            observation_payload = immutable.pretty_json_bytes(observation)

            def verify_handoff() -> None:
                retained_report.verify_unchanged()
                retained_manifest.verify_unchanged()
                parent.verify_path()

            retained_observation: immutable.RetainedRegular | None = None

            def retain_observation() -> None:
                nonlocal retained_observation
                verify_handoff()
                retained_observation = active_stack.enter_context(
                    immutable.open_retained_regular(observation_path)
                )
                if retained_observation.read_bytes() != observation_payload:
                    raise CaptureError(
                        "committed NCU launch observation differs from generated bytes"
                    )
                _require_ncu_cell_namespace(
                    parent,
                    expected_names=frozenset({
                        *_NCU_INITIAL_ARTIFACT_NAMES,
                        observation_path.name,
                    }),
                )
                retained_observation.verify_unchanged()

            immutable.publish_regular_file(
                cell,
                observation_path.name,
                observation_payload,
                pinned_parent=parent,
                after_commit=retain_observation,
            )
            if retained_observation is None:
                raise CaptureError("NCU launch observation was not retained")

            def verify_observation() -> None:
                verify_handoff()
                assert retained_observation is not None
                retained_observation.verify_unchanged()

            retained_sidecar: immutable.RetainedRegular | None = None

            def retain_sidecar() -> None:
                nonlocal retained_sidecar
                verify_observation()
                retained_sidecar = active_stack.enter_context(
                    immutable.open_retained_regular(sidecar_path)
                )
                if retained_sidecar.read_bytes() != sidecar_payload:
                    raise CaptureError(
                        "committed NCU sidecar differs from generated bytes"
                    )
                _require_ncu_cell_namespace(
                    parent,
                    expected_names=_NCU_COMPLETE_ARTIFACT_NAMES,
                )
                retained_sidecar.verify_unchanged()

            immutable.publish_regular_file(
                cell,
                sidecar_path.name,
                sidecar_payload,
                pinned_parent=parent,
                after_commit=retain_sidecar,
            )
            if retained_sidecar is None:
                raise CaptureError("NCU sidecar was not retained")
            verify_observation()
            retained_sidecar.verify_unchanged()
    except CaptureError:
        raise
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        RecursionError,
        immutable.ImmutableArtifactError,
    ) as exc:
        raise CaptureError(f"NCU handoff postprocessing failed: {exc}") from exc


def _retain_failed_ncu_handoff(
    root: Path,
    row: dict[str, Any],
    generation_stack: contextlib.ExitStack,
) -> list[dict[str, Any]]:
    cell = root / ARCH / row["successor_slug"] / "trace_00"
    try:
        metadata = cell.lstat()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise CaptureError(f"cannot inspect failed NCU handoff root: {exc}") from exc
    if cell.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise CaptureError("failed NCU handoff root is not a pinned directory")
    reader = generation_stack.enter_context(
        immutable.PinnedDirectoryReader(cell)
    )
    snapshot = generation_stack.enter_context(reader.retain_tree_generation(
        max_files=64,
        max_directories=16,
        max_total_file_bytes=8 * 1024 * 1024 * 1024,
        max_total_relative_path_bytes=64 * 1024,
    ))
    roles = {
        "run_0_launch_set.ncu-rep": "ncu_report",
        "label_run_manifest_0_launch_set.json": "label_run_manifest",
        "run_0_launch_set.launch_observation.json": "ncu_launch_observation",
        "label_0_0.json": "label_sidecar",
    }
    artifacts: list[dict[str, Any]] = []
    for relative, expected_metadata in snapshot.files:
        retained = reader.read_snapshot(
            relative,
            expected_metadata=expected_metadata,
        )
        path = cell / relative
        artifacts.append({
            "role": roles.get(relative, "retained_failed_handoff_artifact"),
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_bytes(retained.payload),
            "size_bytes": len(retained.payload),
        })
    return artifacts


def _write_failure_provenance(
    failure_root: Path,
    *,
    root: Path,
    phase: str,
    row: dict[str, Any],
    command: list[str],
    live: dict[str, Any],
    environment_payload: bytes,
    default_environment: dict[str, str],
    capture_producer_sha256: str,
    containment_supervisor_sha256: str,
    phase_tool_sha256: str,
    action_executor_sha256: str | None,
    error: CaptureError,
    generation_stack: contextlib.ExitStack | None = None,
) -> None:
    if generation_stack is None:
        generation_stack = contextlib.ExitStack()
    try:
        failure_root.mkdir(parents=True, exist_ok=True)
        parent = generation_stack.enter_context(
            immutable.PinnedDirectoryReader(failure_root)
        )
        try:
            observed_names = os.listdir(parent.descriptor)
        except OSError as exc:
            raise CaptureError(
                f"cannot inspect capture failure namespace: {exc}"
            ) from exc
        stream_names = frozenset(observed_names)
        if (
            len(observed_names) != len(stream_names)
            or any(not name.endswith(".bin") for name in stream_names)
        ):
            raise CaptureError(
                "capture failure namespace contains non-stream evidence"
            )
        _require_directory_namespace(
            parent,
            expected_names=stream_names,
            label="capture failure",
        )
        streams = []
        for name in sorted(stream_names):
            path = failure_root / name
            retained_stream = generation_stack.enter_context(
                immutable.open_retained_regular(path)
            )
            streams.append({
                "path": name,
                "sha256": retained_stream.sha256,
                "size_bytes": retained_stream.size_bytes,
            })
        handoff_artifacts = (
            _retain_failed_ncu_handoff(root, row, generation_stack)
            if phase == "ncu"
            else []
        )
        failed_command = error.process_command or command
        failed_environment = error.process_environment or default_environment
        failed_tool_sha256 = error.process_tool_sha256 or phase_tool_sha256
        process = {
            "role": error.process_role or f"{phase}_capture",
            "boot_id_sha256": _boot_id_sha256(),
            "command": failed_command,
            "command_sha256": canonical_sha(failed_command),
            "environment": failed_environment,
            "environment_sha256": canonical_sha(failed_environment),
            "host_identity_sha256": _host_identity_sha256(live),
            "pid": error.process_id,
            "started_at_utc": error.started_at_utc,
        }
        if error.containment is not None:
            process["containment"] = error.containment.as_dict()
        provenance = {
            "capture_producer_sha256": capture_producer_sha256,
            "containment_supervisor_sha256": containment_supervisor_sha256,
            "phase_tool_sha256": failed_tool_sha256,
        }
        if action_executor_sha256 is not None:
            provenance["action_executor_sha256"] = action_executor_sha256
        document = {
            "schema_version": 1,
            "artifact_kind": "cutlass_successor_capture_failure",
            "status": "failed_unadmitted",
            "capture_kind": phase,
            "matrix_key": row["matrix_key"],
            "variant_id": row["variant_id"],
            "detail": error.detail,
            "capture_environment_document_sha256": sha256_bytes(
                environment_payload
            ),
            "process": process,
            "provenance": provenance,
            "streams": streams,
            "handoff_artifacts": handoff_artifacts,
        }
        document["failure_seal_sha256"] = canonical_sha(document)
        output = failure_root / "failure_provenance.json"
        payload = immutable.pretty_json_bytes(document)

        def close_failure_generation() -> None:
            expected_names = frozenset({*stream_names, output.name})
            max_total_file_bytes = sum(
                stream["size_bytes"] for stream in streams
            ) + len(payload)
            with immutable.PinnedDirectoryReader(failure_root) as final_parent:
                with final_parent.retain_tree_generation(
                    max_files=len(expected_names),
                    max_directories=1,
                    max_total_file_bytes=max_total_file_bytes,
                    max_total_relative_path_bytes=64 * 1024,
                ):
                    _require_directory_namespace(
                        final_parent,
                        expected_names=expected_names,
                        label="capture failure",
                    )
                    generation_stack.close()

        immutable.publish_regular_file(
            failure_root,
            output.name,
            payload,
            pinned_parent=parent,
            after_commit=close_failure_generation,
        )
    except CaptureError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise CaptureError(
            f"cannot transactionally publish failure provenance: {exc}"
        ) from exc
    finally:
        generation_stack.close()


def _receipt(
    *, root: Path, phase: str, row: dict[str, Any], preflight_path: Path,
    preflight_payload: bytes, environment_path: Path, environment_payload: bytes,
    command: list[str], process: dict[str, Any], artifacts: list[dict[str, Any]],
    semantic: dict[str, Any], captured_at: str,
    capture_producer_sha256: str,
    containment_supervisor_sha256: str,
    phase_tool_sha256: str,
    action_executor_sha256: str | None,
) -> dict[str, Any]:
    preflight_relative = preflight_path.relative_to(root).as_posix()
    environment_digest = sha256_bytes(environment_payload)
    provenance = {
        "capture_producer_sha256": capture_producer_sha256,
        "containment_supervisor_sha256": containment_supervisor_sha256,
        "phase_tool_sha256": phase_tool_sha256,
    }
    if phase == "ncu":
        if action_executor_sha256 is None:
            raise CaptureError("NCU receipt lacks action-executor provenance")
        provenance["action_executor_sha256"] = action_executor_sha256
    elif action_executor_sha256 is not None:
        raise CaptureError("NVBit receipt carries NCU executor provenance")
    receipt = {
        "schema_version": 2,
        "artifact_kind": "cutlass_successor_capture_receipt",
        "receipt": {
            "capture_kind": phase,
            "status": "pass",
            "matrix_key": row["matrix_key"],
            "variant_id": row["variant_id"],
            "identity": {
                "slug": row["successor_slug"],
                "shape_point": row["shape"]["shape_point"],
                "binary_sha256": row["static_launch_inventory"]["binary_sha256"],
                "source_commit": PINNED_COMMIT,
                "kernel_signature_sha256": hashlib.sha256(
                    f"{row['successor_slug']}|{row['static_launch_inventory']['binary_path']}".encode()
                ).hexdigest(),
                "arch": ARCH,
                "ncu_metric_version": admission.DEFAULT_METRIC_VERSION,
            },
            "launch_identity": {
                "launch_index": 0,
                "kernel_name": row["successor_slug"],
                "kernel_signature_sha256": hashlib.sha256(
                    f"{row['successor_slug']}|{row['static_launch_inventory']['binary_path']}".encode()
                ).hexdigest(),
                "grid_dim": row["shape"]["trace_shape"]["launch_config"]["grid"],
                "block_dim": row["shape"]["trace_shape"]["launch_config"]["block"],
            },
            "process": process,
            "provenance": provenance,
            "captured_at_utc": captured_at,
            "preflight_path": preflight_relative,
            "preflight_sha256": sha256_bytes(preflight_payload),
            "environment_sha256": environment_digest,
            "command": command,
            "command_sha256": canonical_sha(command),
            "semantic_validation": semantic,
            "artifacts": artifacts,
        },
    }
    launch = receipt["receipt"]["launch_identity"]
    launch["launch_identity_sha256"] = canonical_sha({k: v for k, v in launch.items() if k != "launch_identity_sha256"})
    receipt["receipt"]["artifacts"].extend([
        {"role": "preflight", "path": preflight_relative, "sha256": sha256_bytes(preflight_payload), "size_bytes": len(preflight_payload)},
        {"role": "environment", "path": environment_path.relative_to(root).as_posix(), "sha256": environment_digest, "size_bytes": len(environment_payload)},
    ])
    receipt["receipt_seal_sha256"] = canonical_sha(receipt)
    return receipt


def _finalize_capture_in_generation(
    *,
    generation_stack: contextlib.ExitStack,
    root: Path,
    phase: str,
    row: dict[str, Any],
    live: dict[str, Any],
    pid: int,
    started: str,
    containment: delegated_cgroup.CgroupContainmentEvidence,
    command: list[str],
    capture_source: Path,
    capture_producer_sha256: str,
    containment_supervisor_sha256: str,
    phase_tool_sha256: str,
    action_executor_sha256: str | None,
    preflight_out: Path,
    preflight_payload: bytes,
    environment_out: Path,
    environment_payload: bytes,
    receipt_path: Path,
) -> tuple[Path, bytes]:
    process = {
        "host_identity_sha256": _host_identity_sha256(live),
        "boot_id_sha256": _boot_id_sha256(),
        "pid": pid,
        "started_at_utc": started,
        "command_sha256": canonical_sha(command),
        "containment": containment.as_dict(),
    }
    process["process_identity_sha256"] = canonical_sha(process)
    artifacts: list[dict[str, Any]] = []
    if phase == "ncu":
        cell = root / ARCH / row["successor_slug"] / "trace_00"
        cell_reader = generation_stack.enter_context(
            immutable.PinnedDirectoryReader(cell)
        )
        _require_ncu_cell_namespace(
            cell_reader,
            expected_names=_NCU_COMPLETE_ARTIFACT_NAMES,
        )
        generation_stack.enter_context(cell_reader.retain_tree_generation(
            max_files=64,
            max_directories=16,
            max_total_file_bytes=8 * 1024 * 1024 * 1024,
            max_total_relative_path_bytes=64 * 1024,
        ))
        required = {
            "ncu_report": cell / "run_0_launch_set.ncu-rep",
            "label_sidecar": cell / "label_0_0.json",
            "label_run_manifest": cell / "label_run_manifest_0_launch_set.json",
            "ncu_launch_observation": (
                cell / "run_0_launch_set.launch_observation.json"
            ),
        }
        for role, path in required.items():
            artifacts.append(_artifact(root, path, role))
        sidecar, _ = _load_object(required["label_sidecar"], "NCU label sidecar")
        run_manifest, _ = _load_object(
            required["label_run_manifest"], "NCU label run manifest"
        )
        _validate_ncu_handoff_manifest(run_manifest, row=row)
        observation, _ = _load_object(
            required["ncu_launch_observation"], "NCU launch observation"
        )
        report_artifact = next(
            artifact for artifact in artifacts if artifact["role"] == "ncu_report"
        )
        try:
            parse_ncu_metrics.validate_launch_observation_document(
                observation,
                requested_launch_skip=0,
                requested_launch_count=1,
                report_sha256=report_artifact["sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureError(
                f"NCU launch observation validation failed: {exc}"
            ) from exc
        metrics = {
            item.get("ncu_metric")
            for item in sidecar.get("unit_tagged_metrics", [])
            if isinstance(item, dict)
        }
        if not {
            "gpc__cycles_elapsed.max",
            "gpu__time_duration.sum",
        }.issubset(metrics):
            raise CaptureError("NCU sidecar lacks both required diagnostic metrics")
        semantic = {
            "semantic_kind": "ncu_metric_validation",
            "primary_metric": "gpc__cycles_elapsed.max",
            "secondary_metric": "gpu__time_duration.sum",
            "primary_metric_present": True,
            "secondary_metric_present": True,
            "target_only": True,
            "simulator_config_present": False,
        }
    else:
        trace_root = root / "nvbit" / row["successor_slug"]
        try:
            admission_result = require_trace_admission(trace_root)
            contract = validate_trace_dir(trace_root)
            producer = _validate_trace_producer(trace_root)
        except Exception as exc:
            raise CaptureError(f"NVBit trace validation failed: {exc}") from exc
        if not contract.ok:
            raise CaptureError(
                f"NVBit trace contract failed: {contract.typed_reason}"
            )
        summary = {
            "schema_version": 1,
            "producer": admission.NVBIT_TRACE_SUMMARY_PRODUCER,
            "artifact_kind": "cutlass_successor_nvbit_trace_summary",
            "trace_contract_id": producer["trace_contract_id"],
            "memory_payload_classes": producer["memory_payload_classes"],
            "binary_version": producer["binary_version"],
            "nvbit_version": producer["nvbit_version"],
            "trace_contract_status": "pass",
            "expected_record_count": admission_result.expected_record_count,
            "admitted_record_count": admission_result.admitted_record_count,
            "threadblock_count": admission_result.threadblock_count,
            "dynamic_trace": "dynamic_trace.pb",
            "stats": "stats.csv",
        }
        summary_path = trace_root / "trace_summary.json"
        summary_payload = (
            json.dumps(summary, sort_keys=True, indent=2) + "\n"
        ).encode()
        _write_new(summary_path, summary_payload)
        trace_reader = generation_stack.enter_context(
            immutable.PinnedDirectoryReader(trace_root)
        )
        generation_stack.enter_context(trace_reader.retain_tree_generation(
            max_files=1_000_000,
            max_directories=100_000,
            max_total_file_bytes=row["trace_budget"]["max_trace_bytes"],
            max_total_relative_path_bytes=64 * 1024 * 1024,
        ))
        try:
            admission_result = require_trace_admission(trace_root)
            contract = validate_trace_dir(trace_root)
            producer = _validate_trace_producer(trace_root)
        except Exception as exc:
            raise CaptureError(
                f"NVBit trace generation revalidation failed: {exc}"
            ) from exc
        if not contract.ok:
            raise CaptureError(
                f"NVBit trace contract failed: {contract.typed_reason}"
            )
        artifacts.append(
            _artifact(root, trace_root / "dynamic_trace.pb", "dynamic_trace")
        )
        artifacts.append(_artifact(root, trace_root / "stats.csv", "stats"))
        artifacts.extend(
            _artifact(root, path, "threadblock")
            for path in sorted((trace_root / "threadblocks").rglob("*.pb"))
        )
        artifacts.append(_artifact(root, summary_path, "trace_summary"))
        if not any(item["role"] == "threadblock" for item in artifacts):
            raise CaptureError("NVBit trace has no threadblock payload artifacts")
        known_paths = {item["path"] for item in artifacts}
        for path in sorted(trace_root.rglob("*")):
            if path.is_symlink():
                raise CaptureError(f"NVBit trace contains symlink output: {path}")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if relative not in known_paths:
                    artifacts.append(_artifact(root, path, "trace_auxiliary"))
        semantic = {
            "semantic_kind": "nvbit_trace_validation",
            "trace_contract_status": "pass",
            "dynamic_trace_role": "dynamic_trace",
            "summary_role": "trace_summary",
            "target_only": True,
            "simulator_config_present": False,
        }
        _require_capture_tree_budget(
            trace_root,
            row["trace_budget"]["max_trace_bytes"],
        )
    _verify_unchanged(capture_source, capture_producer_sha256, "capture producer")
    _verify_unchanged(
        Path(delegated_cgroup.__file__),
        containment_supervisor_sha256,
        "cgroup containment supervisor",
    )
    if phase == "ncu":
        if action_executor_sha256 is None:
            raise CaptureError("NCU action executor authority is absent")
        _verify_unchanged(
            Path(ncu_executor.__file__),
            action_executor_sha256,
            "NCU action executor",
        )
    if phase == "nvbit":
        _verify_unchanged(TRACER_SO, phase_tool_sha256, "NVBit tracer")
    captured = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    receipt = _receipt(
        root=root,
        phase=phase,
        row=row,
        preflight_path=preflight_out,
        preflight_payload=preflight_payload,
        environment_path=environment_out,
        environment_payload=environment_payload,
        command=command,
        process=process,
        artifacts=artifacts,
        semantic=semantic,
        captured_at=captured,
        capture_producer_sha256=capture_producer_sha256,
        containment_supervisor_sha256=containment_supervisor_sha256,
        phase_tool_sha256=phase_tool_sha256,
        action_executor_sha256=action_executor_sha256,
    )
    try:
        admission._validate_schema(
            ROOT / admission.RECEIPT_SCHEMA_REL,
            receipt,
            "capture receipt",
        )
    except Exception as exc:
        raise CaptureError(f"capture receipt schema/producer mismatch: {exc}") from exc
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    return receipt_path, payload


def _finalize_capture(
    *,
    generation_stack: contextlib.ExitStack | None = None,
    root: Path,
    phase: str,
    row: dict[str, Any],
    live: dict[str, Any],
    pid: int,
    started: str,
    containment: delegated_cgroup.CgroupContainmentEvidence,
    command: list[str],
    capture_source: Path,
    capture_producer_sha256: str,
    containment_supervisor_sha256: str,
    phase_tool_sha256: str,
    action_executor_sha256: str | None,
    preflight_out: Path,
    preflight_payload: bytes,
    environment_out: Path,
    environment_payload: bytes,
    receipt_path: Path,
) -> Path:
    owns_generation_stack = generation_stack is None
    if generation_stack is None:
        generation_stack = contextlib.ExitStack()
    try:
        output, payload = _finalize_capture_in_generation(
            generation_stack=generation_stack,
            root=root,
            phase=phase,
            row=row,
            live=live,
            pid=pid,
            started=started,
            containment=containment,
            command=command,
            capture_source=capture_source,
            capture_producer_sha256=capture_producer_sha256,
            containment_supervisor_sha256=containment_supervisor_sha256,
            phase_tool_sha256=phase_tool_sha256,
            action_executor_sha256=action_executor_sha256,
            preflight_out=preflight_out,
            preflight_payload=preflight_payload,
            environment_out=environment_out,
            environment_payload=environment_payload,
            receipt_path=receipt_path,
        )
        parent = generation_stack.enter_context(
            immutable.PinnedDirectoryReader(output.parent)
        )
        try:
            immutable.publish_regular_file(
                output.parent,
                output.name,
                payload,
                pinned_parent=parent,
                after_commit=generation_stack.close,
            )
        except (OSError, immutable.ImmutableArtifactError) as exc:
            raise CaptureError(
                f"cannot transactionally publish capture receipt {output}: {exc}"
            ) from exc
        return output
    finally:
        if owns_generation_stack:
            generation_stack.close()


def _capture_protocol_succeeded(
    phase: str,
    *,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
) -> bool:
    if phase == "ncu":
        return (
            returncode == 3
            and stdout == b""
            and stderr == _NCU_HANDOFF_STDERR
        )
    if phase == "nvbit":
        return returncode == 0 and stderr == b""
    raise CaptureError("capture result phase is invalid")


def capture_phase(*, matrix_key: str, variant_id: str, phase: str, preflight_path: Path, output_root: Path) -> Path:
    if phase not in {"ncu", "nvbit"}:
        raise CaptureError("phase must be ncu or nvbit")
    root = _absolute_external(output_root, "capture output root")
    root.mkdir(parents=True, exist_ok=True)
    row, roster_row = _row_for(matrix_key, variant_id)
    preflight, preflight_payload = validate_successor_preflight(preflight_path, row, roster_row)
    live = validate_live_capture_preflight(phase=phase)
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity | None = None
    if phase == "ncu":
        expected_ncu_identity = _ncu_identity_from_authority(
            live["ncu_authority"]
        )
    capture_source, capture_producer_sha256 = _capture_producer_identity()
    containment_source = Path(delegated_cgroup.__file__).resolve(strict=True)
    _regular(containment_source, "cgroup containment supervisor")
    containment_supervisor_sha256 = sha256_bytes(containment_source.read_bytes())
    env = dict(os.environ)
    command: list[str]
    collection_environment: ncu_executor.NcuCollectionEnvironment | None = None
    driver_identity: ncu_executor.ExpectedExecutableIdentity | None = None
    action_executor_sha256: str | None = None
    binary = ROOT / row["static_launch_inventory"]["binary_path"]
    process_cwd = binary.parent
    if phase == "ncu":
        if expected_ncu_identity is None:
            raise CaptureError("NCU executable authority was not initialized")
        executor_source = Path(ncu_executor.__file__).resolve(strict=True)
        _regular(executor_source, "NCU action executor")
        action_executor_sha256 = sha256_bytes(executor_source.read_bytes())
        if action_executor_sha256 != admission.NCU_ACTION_EXECUTOR_SHA256:
            raise CaptureError("NCU action executor differs from pinned authority")
        for key in (
            "NVBIT_PB_PREFIX", "CUDA_INJECTION64_PATH", "TRACES_FOLDER",
            "USER_DEFINED_FOLDERS",
        ):
            env.pop(key, None)
        env["LD_SOURCE_COMMIT_OVERRIDE"] = PINNED_COMMIT
        command = admission._expected_ncu_command(
            root, row, expected_ncu_identity
        )
        collection_environment = _label_collection_environment(
            env, cwd=process_cwd
        )
        driver_identity = _label_driver_identity()
        phase_tool_sha256 = driver_identity.executable_sha256
        phase_output_dir = root / ARCH / row["successor_slug"] / "trace_00"
    else:
        env = admission._expected_nvbit_environment(root, row)
        trace_root = root / "nvbit" / row["successor_slug"]
        phase_output_dir = trace_root
        command = admission._expected_nvbit_command(row)
        _regular(TRACER_SO, "NVBit tracer")
        phase_tool_sha256 = sha256_bytes(TRACER_SO.read_bytes())
    preflight_out, environment_out, receipt_path, failure_root = (
        _capture_metadata_paths(root, variant_id=variant_id, phase=phase)
    )
    _reject_preexisting_output_descendants(
        root,
        output_dir=phase_output_dir,
        reserved_files=(preflight_out, environment_out, receipt_path),
    )
    _reject_preexisting_output_descendants(
        root,
        output_dir=failure_root,
        reserved_files=(failure_root / "failure_provenance.json",),
    )
    if phase == "nvbit":
        trace_root.mkdir(parents=True, exist_ok=True)
    _write_new(preflight_out, preflight_payload)
    recorded_environment = (
        collection_environment.as_dict()
        if collection_environment is not None
        else env
    )
    environment_doc = _environment_document(
        preflight,
        live,
        recorded_environment,
        capture_kind=phase,
    )
    environment_payload = (json.dumps(environment_doc, sort_keys=True, indent=2) + "\n").encode()
    _write_new(environment_out, environment_payload)
    pid: int | None = None
    started: str | None = None
    stdout = b""
    stderr = b""
    containment: delegated_cgroup.CgroupContainmentEvidence | None = None
    try:
        with immutable.open_retained_executable(binary) as retained_target:
            if retained_target.sha256 != row["static_launch_inventory"]["binary_sha256"]:
                raise CaptureError("capture target differs from pinned binary authority")
            if phase == "ncu":
                if (
                    collection_environment is None
                    or driver_identity is None
                    or expected_ncu_identity is None
                ):
                    raise CaptureError("NCU execution authority was not initialized")
                (
                    returncode, pid, started, stdout, stderr, containment,
                ) = _run_ncu_capture_command(
                    command,
                    expected_ncu_identity,
                    collection_environment,
                    driver_identity,
                    retained_target,
                    process_cwd,
                    timeout_s=CAPTURE_TIMEOUT_S,
                    failure_root=failure_root,
                )
            else:
                with immutable.open_retained_regular(TRACER_SO) as retained_tracer:
                    if retained_tracer.sha256 != phase_tool_sha256:
                        raise CaptureError("NVBit tracer differs from pinned capture authority")
                    runtime_environment = dict(env)
                    runtime_environment["CUDA_INJECTION64_PATH"] = retained_tracer.proc_path
                    runtime_environment = nvbit_runtime.tracer_runtime_env(
                        TRACER_SO,
                        base_env=runtime_environment,
                        pb_lib=NVBIT_PB_LIB,
                        retained_descriptor=retained_tracer.descriptor,
                    )
                    expected_runtime_environment = dict(env)
                    expected_runtime_environment["CUDA_INJECTION64_PATH"] = (
                        retained_tracer.proc_path
                    )
                    if runtime_environment != expected_runtime_environment:
                        raise CaptureError(
                            "NVBit runtime environment differs from closed pins"
                        )
                    runtime_command = [retained_target.proc_path, *command[1:]]
                    (
                        returncode, pid, started, stdout, stderr, containment,
                    ) = _run_capture_command(
                        runtime_command,
                        runtime_environment,
                        process_cwd,
                        timeout_s=float(row["trace_budget"]["max_wall_time_s"]),
                        byte_root=trace_root,
                        byte_budget=row["trace_budget"]["max_trace_bytes"],
                        failure_root=failure_root,
                        inherited_descriptors=(
                            retained_target.descriptor,
                            retained_tracer.descriptor,
                        ),
                        require_cgroup_containment=True,
                    )
    except (
        OSError,
        immutable.ImmutableArtifactError,
        nvbit_runtime.TracerRuntimeError,
    ) as raw_error:
        exc = CaptureError(
            f"retained capture authority failed: {raw_error}",
            process_id=pid,
            started_at_utc=started,
            process_role=(None if pid is None else f"{phase}_capture"),
            process_command=(None if pid is None else list(command)),
            process_environment=(
                None if pid is None else dict(recorded_environment)
            ),
            process_tool_sha256=(None if pid is None else phase_tool_sha256),
            containment=containment,
        )
        _write_failure_provenance(
            failure_root,
            root=root,
            phase=phase,
            row=row,
            command=command,
            live=live,
            environment_payload=environment_payload,
            default_environment=recorded_environment,
            capture_producer_sha256=capture_producer_sha256,
            containment_supervisor_sha256=containment_supervisor_sha256,
            phase_tool_sha256=phase_tool_sha256,
            action_executor_sha256=action_executor_sha256,
            error=exc,
        )
        raise exc from raw_error
    except CaptureError as exc:
        if exc.process_id is not None and exc.process_command is None:
            exc.process_role = f"{phase}_capture"
            exc.process_command = list(command)
            exc.process_environment = dict(recorded_environment)
            exc.process_tool_sha256 = phase_tool_sha256
        if exc.containment is None:
            exc.containment = containment
        _write_failure_provenance(
            failure_root,
            root=root,
            phase=phase,
            row=row,
            command=command,
            live=live,
            environment_payload=environment_payload,
            default_environment=recorded_environment,
            capture_producer_sha256=capture_producer_sha256,
            containment_supervisor_sha256=containment_supervisor_sha256,
            phase_tool_sha256=phase_tool_sha256,
            action_executor_sha256=action_executor_sha256,
            error=exc,
        )
        raise
    capture_protocol_ok = _capture_protocol_succeeded(
        phase,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if not capture_protocol_ok:
        _write_new(failure_root / "capture_failure.stdout.bin", stdout)
        _write_new(failure_root / "capture_failure.stderr.bin", stderr)
        failure = CaptureError(
            f"{phase} capture failed returncode={returncode} "
            f"stdout={stdout[-400:]!r} stderr={stderr[-400:]!r}",
            process_id=pid,
            started_at_utc=started,
            process_role=f"{phase}_capture",
            process_command=list(command),
            process_environment=dict(recorded_environment),
            process_tool_sha256=phase_tool_sha256,
            containment=containment,
        )
        _write_failure_provenance(
            failure_root,
            root=root,
            phase=phase,
            row=row,
            command=command,
            live=live,
            environment_payload=environment_payload,
            default_environment=recorded_environment,
            capture_producer_sha256=capture_producer_sha256,
            containment_supervisor_sha256=containment_supervisor_sha256,
            phase_tool_sha256=phase_tool_sha256,
            action_executor_sha256=action_executor_sha256,
            error=failure,
        )
        raise failure
    if pid is None or started is None or containment is None:
        raise CaptureError(
            "successful capture has incomplete process or containment provenance"
        )
    generation_stack = contextlib.ExitStack()
    try:
        if phase == "ncu":
            if expected_ncu_identity is None:
                raise CaptureError(
                    "successful NCU handoff has no executable authority"
                )
            _postprocess_ncu_handoff(
                root=root,
                row=row,
                live=live,
                expected_ncu_identity=expected_ncu_identity,
                generation_stack=generation_stack,
            )
        return _finalize_capture(
            generation_stack=generation_stack,
            root=root,
            phase=phase,
            row=row,
            live=live,
            pid=pid,
            started=started,
            containment=containment,
            command=command,
            capture_source=capture_source,
            capture_producer_sha256=capture_producer_sha256,
            containment_supervisor_sha256=containment_supervisor_sha256,
            phase_tool_sha256=phase_tool_sha256,
            action_executor_sha256=action_executor_sha256,
            preflight_out=preflight_out,
            preflight_payload=preflight_payload,
            environment_out=environment_out,
            environment_payload=environment_payload,
            receipt_path=receipt_path,
        )
    except BaseException as raw_error:
        exc = (
            raw_error
            if isinstance(raw_error, CaptureError)
            else CaptureError(
                f"post-capture validation failed: {raw_error}",
            )
        )
        try:
            diagnostic_failures = []
            for path, payload in (
                (failure_root / "post_validation.stdout.bin", stdout),
                (failure_root / "post_validation.stderr.bin", stderr),
            ):
                try:
                    _write_new(path, payload)
                except Exception as diagnostic_error:
                    diagnostic_failures.append(
                        f"{path.name}: {diagnostic_error}"
                    )
            if diagnostic_failures:
                exc.detail += (
                    "; post-validation stream retention failed: "
                    + "; ".join(diagnostic_failures)
                )
            exc.process_id = pid
            exc.started_at_utc = started
            exc.process_role = f"{phase}_postprocess_validation"
            exc.process_command = list(command)
            exc.process_environment = dict(recorded_environment)
            exc.process_tool_sha256 = phase_tool_sha256
            exc.containment = containment
            _write_failure_provenance(
                failure_root,
                root=root,
                phase=phase,
                row=row,
                command=command,
                live=live,
                environment_payload=environment_payload,
                default_environment=recorded_environment,
                capture_producer_sha256=capture_producer_sha256,
                containment_supervisor_sha256=containment_supervisor_sha256,
                phase_tool_sha256=phase_tool_sha256,
                action_executor_sha256=action_executor_sha256,
                error=exc,
                generation_stack=generation_stack,
            )
        finally:
            generation_stack.close()
        if not isinstance(raw_error, Exception):
            raise
        if exc is raw_error:
            raise
        raise exc from raw_error



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix-key", required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--phase", choices=("ncu", "nvbit"), required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = capture_phase(matrix_key=args.matrix_key, variant_id=args.variant_id, phase=args.phase, preflight_path=args.preflight, output_root=args.output_root)
    except (CaptureError, OSError, ValueError) as exc:
        print(f"CUTLASS successor capture refused: {exc}", file=sys.stderr)
        return 1
    print(f"CUTLASS successor {args.phase} receipt: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
