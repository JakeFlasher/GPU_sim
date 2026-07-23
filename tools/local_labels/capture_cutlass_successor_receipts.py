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
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local_labels import admit_cutlass_successor_capture as admission  # noqa: E402
from tools.local_labels.capture_toolchain_contract import (  # noqa: E402
    parse_nvcc_release_banner,
    require_sol_cuda_release,
    ncu_version_matches,
)
from tools.local_labels import build_cutlass_successor_cohort as cohort  # noqa: E402
from tools.nvbit_tracer.import_trace_contract import validate_trace_dir  # noqa: E402
from tools.nvbit_tracer.runtime_env import tracer_runtime_env  # noqa: E402
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    _import_trace_pb2,
    require_trace_admission,
)

PINNED_COMMIT = admission.PINNED_BUNDLE_COMMIT
ARCH = "sm_86"
SKU = "rtx_3090_ti"
CUDA_RELEASE = {"major": 13, "minor": 2}
NVBIT_VERSION = "1.8"
NCU_VERSION = "2026.1.0"
GPU_NAME = "NVIDIA GeForce RTX 3090 Ti"
SM_CLOCK_MHZ = 1350
MEM_CLOCK_MHZ = 10501
TRACER_SO = ROOT / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
NVBIT_VERSION_HEADER = ROOT / "tools/nvbit_tracer/nvbit_release/core/nvbit.h"
LABEL_DRIVER = ROOT / "tools/local_labels/label_driver.sh"
CUDA_NVCC = Path("/usr/local/cuda-13.2/bin/nvcc")
CAPTURE_TIMEOUT_S = 3600
EXPECTED_TRACE_CONTRACT_ID = "sm86_addr_carry_v1"
EXPECTED_MEMORY_PAYLOAD_CLASSES = [1, 2, 3, 6, 8]


class CaptureError(RuntimeError):
    """A preflight or capture contract failed closed."""


def canonical_sha(value: Any) -> str:
    return admission.canonical_sha(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _regular(path: Path, label: str) -> None:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise CaptureError(f"{label} unavailable: {path}") from exc
    if path.is_symlink() or not path.is_file() or stat.st_nlink != 1:
        raise CaptureError(f"{label} is not a single regular file: {path}")


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
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
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


def _kill_capture_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _run_capture_command(
    command: list[str], env: dict[str, str], cwd: Path, *,
    timeout_s: float = CAPTURE_TIMEOUT_S, byte_root: Path | None = None,
    byte_budget: int | None = None,
) -> tuple[int, int, str, bytes, bytes]:
    if timeout_s <= 0:
        raise CaptureError("capture timeout must be positive")
    if byte_budget is not None and byte_budget <= 0:
        raise CaptureError("capture byte budget must be positive")
    started = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    _check_dt(started, "process start")
    exceeded = threading.Event()
    stop_watchdog = threading.Event()

    def watch_bytes() -> None:
        while not stop_watchdog.wait(0.1):
            try:
                if byte_root is not None and byte_budget is not None and _capture_tree_bytes(byte_root) > byte_budget:
                    exceeded.set()
                    _kill_capture_process(process)
                    return
            except CaptureError:
                exceeded.set()
                _kill_capture_process(process)
                return

    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise CaptureError(f"capture process could not start: {exc}") from exc
    watchdog = threading.Thread(target=watch_bytes, daemon=True)
    watchdog.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _kill_capture_process(process)
        stdout, stderr = process.communicate()
        raise CaptureError(f"capture process exceeded timeout of {timeout_s:g}s") from exc
    finally:
        stop_watchdog.set()
        watchdog.join(timeout=1)
    if exceeded.is_set():
        if process.poll() is None:
            _kill_capture_process(process)
            process.wait()
        raise CaptureError("capture process exceeded its trace byte budget or wrote unsafe output")
    return process.returncode, process.pid, started, stdout, stderr


def _query(command: list[str], label: str) -> tuple[str, list[dict[str, str]]]:
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False, timeout=30)
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


def validate_live_capture_preflight() -> dict[str, Any]:
    """Check live host state only after static preflight pins pass."""
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
    nvcc_command = [str(CUDA_NVCC), "--version"]
    nvcc_stdout, _ = _query(nvcc_command, "CUDA 13.2 nvcc")
    try:
        cuda = parse_nvcc_release_banner(nvcc_stdout)
        require_sol_cuda_release(cuda)
    except ValueError as exc:
        raise CaptureError(f"CUDA release pin failed: {exc}") from exc
    ncu_command = ["ncu", "--version"]
    ncu_stdout, _ = _query(ncu_command, "NCU")
    if not any(ncu_version_matches(line, NCU_VERSION) for line in ncu_stdout.splitlines()):
        raise CaptureError("NCU version is not the pinned 2026.1.0 release")
    nvbit = _nvbit_version()
    return {
        "smi_command": smi_command,
        "smi_stdout_sha256": sha256_bytes(smi_stdout.encode()),
        "competing_process_command": apps_command,
        "competing_process_stdout_sha256": sha256_bytes(apps_stdout.encode()),
        "nvcc_command": nvcc_command,
        "nvcc_stdout_sha256": sha256_bytes(nvcc_stdout.encode()),
        "ncu_command": ncu_command,
        "ncu_stdout_sha256": sha256_bytes(ncu_stdout.encode()),
        "cuda_release": cuda,
        "nvbit_version": nvbit,
        "ncu_version": NCU_VERSION,
        "gpu_name": GPU_NAME,
        "persistence_mode": "Enabled",
        "sm_clock_mhz": SM_CLOCK_MHZ,
        "mem_clock_mhz": MEM_CLOCK_MHZ,
    }


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
    if metadata["binary_version"] != 86 or metadata["nvbit_version"] != NVBIT_VERSION:
        raise CaptureError("NVBit trace producer version mismatch")
    return metadata


def _environment_document(preflight: dict[str, Any], live: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    selected = {
        key: env.get(key, "")
        for key in (
            "PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH",
            "NVBIT_PB_PREFIX", "CUDA_INJECTION64_PATH", "TRACES_FOLDER",
            "USER_DEFINED_FOLDERS",
        )
    }
    return {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_capture_environment",
        "source_commit": PINNED_COMMIT,
        "preflight_pins_sha256": canonical_sha(preflight),
        "live_checks": live,
        "environment": selected,
    }


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CaptureError(f"refusing to overwrite capture output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    _regular(path, "capture output")


def _artifact(root: Path, path: Path, role: str) -> dict[str, Any]:
    _regular(path, f"{role} artifact")
    payload = path.read_bytes()
    if not payload:
        raise CaptureError(f"{role} artifact is zero-sized: {path}")
    relative = path.relative_to(root).as_posix()
    return {"role": role, "path": relative, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def _receipt(
    *, root: Path, phase: str, row: dict[str, Any], preflight_path: Path,
    preflight_payload: bytes, environment_path: Path, environment_payload: bytes,
    command: list[str], process: dict[str, Any], artifacts: list[dict[str, Any]],
    semantic: dict[str, Any], captured_at: str,
) -> dict[str, Any]:
    preflight_relative = preflight_path.relative_to(root).as_posix()
    environment_digest = sha256_bytes(environment_payload)
    receipt = {
        "schema_version": 1,
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
            "provenance": {
                "capture_producer_sha256": sha256_bytes(Path(__file__).read_bytes()),
                "phase_tool_sha256": sha256_bytes(
                    (LABEL_DRIVER if phase == "ncu" else TRACER_SO).read_bytes()
                ),
            },
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


def capture_phase(*, matrix_key: str, variant_id: str, phase: str, preflight_path: Path, output_root: Path) -> Path:
    if phase not in {"ncu", "nvbit"}:
        raise CaptureError("phase must be ncu or nvbit")
    root = _absolute_external(output_root, "capture output root")
    root.mkdir(parents=True, exist_ok=True)
    row, roster_row = _row_for(matrix_key, variant_id)
    preflight, preflight_payload = validate_successor_preflight(preflight_path, row, roster_row)
    live = validate_live_capture_preflight()
    env = dict(os.environ)
    command: list[str]
    binary = ROOT / row["static_launch_inventory"]["binary_path"]
    args = [str(value) for value in row["shape"]["actual_args"]]
    process_cwd = binary.parent
    if phase == "ncu":
        command = [str(LABEL_DRIVER), "--device", ARCH, "--slug", row["successor_slug"], "--shape-point", "trace_00", "--repeat", "0", "--launch", "0", "--binary", str(binary), "--warmup", "0", "--out-dir", str(root), "--expected-cuda-release", "13.2", "--expected-ncu-version-prefix", NCU_VERSION, "--publication-metric-scope", "cycles_only_binding", "--clock-control", "none", "--unit-metric", "gpc__cycles_elapsed.max", "--unit-metric", "gpu__time_duration.sum", "--", *args]
        env["LD_SCRIPT_DIR"] = str(LABEL_DRIVER.parent)
        env["LD_SOURCE_COMMIT_OVERRIDE"] = PINNED_COMMIT
        phase_output_dir = root / ARCH / row["successor_slug"] / "trace_00"
    else:
        env = tracer_runtime_env(TRACER_SO, base_env=env)
        env["CUDA_INJECTION64_PATH"] = str(TRACER_SO)
        env["USER_DEFINED_FOLDERS"] = "1"
        trace_root = root / "nvbit" / row["successor_slug"]
        phase_output_dir = trace_root
        env["TRACES_FOLDER"] = str(trace_root)
        command = [str(binary), *args]
    preflight_out = root / f"preflight_{phase}.json"
    environment_out = root / f"environment_{phase}.json"
    receipt_path = root / f"{variant_id}.{phase}{admission.RECEIPT_SUFFIX}"
    _reject_preexisting_output_descendants(
        root,
        output_dir=phase_output_dir,
        reserved_files=(preflight_out, environment_out, receipt_path),
    )
    if phase == "nvbit":
        trace_root.mkdir(parents=True, exist_ok=True)
    _write_new(preflight_out, preflight_payload)
    environment_doc = _environment_document(preflight, live, env)
    environment_payload = (json.dumps(environment_doc, sort_keys=True, indent=2) + "\n").encode()
    _write_new(environment_out, environment_payload)
    returncode, pid, started, stdout, stderr = _run_capture_command(
        command,
        env,
        process_cwd,
        timeout_s=float(row["trace_budget"]["max_wall_time_s"] if phase == "nvbit" else CAPTURE_TIMEOUT_S),
        byte_root=(trace_root if phase == "nvbit" else None),
        byte_budget=(row["trace_budget"]["max_trace_bytes"] if phase == "nvbit" else None),
    )
    if returncode != 0 or stderr:
        raise CaptureError(f"{phase} capture failed returncode={returncode} stderr={stderr[-400:]!r}")
    process = {
        "host_identity_sha256": canonical_sha({"gpu": live["gpu_name"], "smi": live["smi_stdout_sha256"]}),
        "boot_id_sha256": sha256_bytes(Path("/proc/sys/kernel/random/boot_id").read_bytes().strip()) if Path("/proc/sys/kernel/random/boot_id").is_file() else "0" * 64,
        "pid": pid,
        "started_at_utc": started,
        "command_sha256": canonical_sha(command),
    }
    process["process_identity_sha256"] = canonical_sha(process)
    artifacts: list[dict[str, Any]] = []
    if phase == "ncu":
        cell = root / ARCH / row["successor_slug"] / "trace_00"
        required = {
            "ncu_report": cell / "run_0_0.ncu-rep",
            "label_sidecar": cell / "label_0_0.json",
            "label_run_manifest": cell / "label_run_manifest_0_0.json",
            "ncu_launch_observation": cell / "run_0_0.launch_observation.json",
        }
        for role, path in required.items():
            artifacts.append(_artifact(root, path, role))
        sidecar, _ = _load_object(required["label_sidecar"], "NCU label sidecar")
        metrics = {item.get("ncu_metric") for item in sidecar.get("unit_tagged_metrics", []) if isinstance(item, dict)}
        if not {"gpc__cycles_elapsed.max", "gpu__time_duration.sum"}.issubset(metrics):
            raise CaptureError("NCU sidecar lacks both required diagnostic metrics")
        semantic = {"semantic_kind": "ncu_metric_validation", "primary_metric": "gpc__cycles_elapsed.max", "secondary_metric": "gpu__time_duration.sum", "primary_metric_present": True, "secondary_metric_present": True, "target_only": True, "simulator_config_present": False}
    else:
        trace_root = root / "nvbit" / row["successor_slug"]
        try:
            admission_result = require_trace_admission(trace_root)
            contract = validate_trace_dir(trace_root)
            producer = _validate_trace_producer(trace_root)
        except Exception as exc:
            raise CaptureError(f"NVBit trace validation failed: {exc}") from exc
        if not contract.ok:
            raise CaptureError(f"NVBit trace contract failed: {contract.typed_reason}")
        summary = {
            "schema_version": 1,
            "producer": "nvbit_tracer",
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
        summary_payload = (json.dumps(summary, sort_keys=True, indent=2) + "\n").encode()
        _write_new(summary_path, summary_payload)
        artifacts.append(_artifact(root, trace_root / "dynamic_trace.pb", "dynamic_trace"))
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
        semantic = {"semantic_kind": "nvbit_trace_validation", "trace_contract_status": "pass", "dynamic_trace_role": "dynamic_trace", "summary_role": "trace_summary", "target_only": True, "simulator_config_present": False}
    captured = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    receipt = _receipt(root=root, phase=phase, row=row, preflight_path=preflight_out, preflight_payload=preflight_payload, environment_path=environment_out, environment_payload=environment_payload, command=command, process=process, artifacts=artifacts, semantic=semantic, captured_at=captured)
    try:
        admission._validate_schema(ROOT / admission.RECEIPT_SCHEMA_REL, receipt, "capture receipt")
    except Exception as exc:
        raise CaptureError(f"capture receipt schema/producer mismatch: {exc}") from exc
    _write_new(receipt_path, (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    return receipt_path


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
