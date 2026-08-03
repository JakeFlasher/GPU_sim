#!/usr/bin/env python3
"""Run one golden-cell NCU capture through retained authority.

The shell trace collector owns the NVBit process and calls this module only
after that process has exited and the trace has passed admission.  This module
owns the separate NCU process: its executable identity comes from a retained
v3 campaign preflight, the target binary is retained through the action, and
the CSV plus the sealed provenance/collection receipt are emitted together.
No NCU value is used as a simulator feature or calibration input here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
STAGING_ROOT = (REPO_ROOT / "results" / "nvbit_capture_staging").resolve()
PREFLIGHT_SCHEMA = (
    REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "campaign_preflight_manifest.schema.json"
)
PREFLIGHT_SCHEMA_SHA256 = (
    "cacd78d381d850b7084ef1b9bf8e9e6c028076743fe3abfb6900d9e50086e3c6"
)
EXPECTED_NCU_VERSION = "2026.1.0"
EXPECTED_NCU_EXECUTION_MODE = "direct_inner_elf_v1"
EXPECTED_NCU_HOST_MACHINE = "x86_64"
GOLDEN_METRIC = "lts__t_sector_hit_rate.pct"
_GIT_EXECUTABLE = Path("/usr/bin/git")
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "LANG": "C",
    "LANGUAGE": "C",
    "LC_ALL": "C",
    "PATH": "/bin:/usr/bin",
}
_GIT_CONFIGURATION = (
    "-c", "core.fsmonitor=false",
    "-c", "core.untrackedCache=false",
    "-c", "submodule.recurse=false",
)
_GIT_TIMEOUT_SECONDS = 30.0
_GIT_MAX_OUTPUT_BYTES = 16 * 1024 * 1024


from tools.local_labels import campaign_preflight  # noqa: E402
from tools.local_labels import capture_toolchain_contract  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import ncu_action_executor as ncu_executor  # noqa: E402
from tools.local_labels import parse_ncu_metrics  # noqa: E402
from tools.nvbit_tracer import trace_admission  # noqa: E402
from tools.trace_sampling.golden_cells import golden_ncu_receipt  # noqa: E402


class GoldenNcuCaptureError(RuntimeError):
    """A fail-closed golden NCU authority or artifact error."""


def _fail(detail: str) -> None:
    raise GoldenNcuCaptureError(detail)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        with immutable.open_retained_regular(path) as retained:
            return retained.sha256
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail(f"cannot retain file for hashing: {path}: {exc}")


def _trace_admission_document(trace_dir: Path) -> dict[str, int]:
    try:
        admission = trace_admission.require_trace_admission(trace_dir)
    except trace_admission.TraceAdmissionError as exc:
        _fail(f"golden NVBit trace is not admitted: {exc}")
    return {
        "launch_count": admission.launch_count,
        "threadblock_count": admission.threadblock_count,
        "instruction_count": admission.instruction_count,
        "expected_record_count": admission.expected_record_count,
        "admitted_record_count": admission.admitted_record_count,
    }


def _trace_tree_identity(trace_dir: Path) -> tuple[str, int]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    try:
        with immutable.PinnedDirectoryReader(trace_dir) as tree_root:
            tree_root.verify_path()
            for path in sorted(trace_dir.rglob("*")):
                if path.is_symlink():
                    _fail(f"golden NVBit trace contains a symlink: {path}")
                metadata = path.lstat()
                if stat.S_ISDIR(metadata.st_mode):
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    _fail(f"golden NVBit trace contains a special file: {path}")
                relative_path = path.relative_to(trace_dir).as_posix()
                payload = tree_root.read(relative_path)
                rows.append({
                    "relative_path": relative_path,
                    "sha256": _sha256_bytes(payload),
                    "size_bytes": len(payload),
                })
                total_bytes += len(payload)
            tree_root.verify_path()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail(f"golden NVBit trace tree is not retained: {trace_dir}: {exc}")
    if not rows:
        _fail("golden NVBit trace is empty")
    try:
        digest = _sha256_bytes(immutable.canonical_json_bytes(rows))
    except immutable.ImmutableArtifactError as exc:
        _fail(f"golden NVBit trace inventory is not canonical: {exc}")
    return digest, total_bytes


def _absolute_regular(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        _fail(f"{label} must be an absolute path")
    if os.path.normpath(str(path)) != str(path) or path.is_symlink():
        _fail(f"{label} must be canonical and not a symlink: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        _fail(f"{label} is unavailable: {exc}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} is not a regular file: {path}")
    return path


def _staging_path(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        _fail(f"{label} must be an absolute path")
    if path.is_symlink():
        _fail(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=False)
    if resolved != path:
        _fail(f"{label} must be canonical and not traverse symlinks: {path}")
    try:
        resolved.relative_to(STAGING_ROOT)
    except ValueError as exc:
        _fail(f"{label} escapes the golden staging namespace: {path}")
        raise AssertionError("unreachable") from exc
    return path


def _relative_artifact(path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError) as exc:
        _fail(f"artifact is not retained under this checkout: {path}")
        raise AssertionError("unreachable") from exc


def _relative_declared_artifact(path: Path) -> str:
    _staging_path(path, label="published artifact path")
    try:
        return path.resolve(strict=False).relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        _fail(f"published artifact is not under this checkout: {path}")
        raise AssertionError("unreachable") from exc


def _write_new(path: Path, payload: bytes, *, label: str) -> None:
    _staging_path(path, label=label)
    try:
        immutable.publish_regular_file(
            path.parent, path.name, payload,
            partial_suffix=".ncu-details-partial",
        )
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail(f"cannot publish {label}: {exc}")


def _sealed_json(value: dict[str, Any]) -> bytes:
    return immutable.pretty_json_bytes(immutable.seal_model(value))


def _run_checkout_git(
    retained_git: immutable.RetainedExecutable,
    arguments: tuple[str, ...],
) -> bytes:
    command = (
        f"/proc/self/fd/{retained_git.descriptor}",
        *_GIT_CONFIGURATION,
        "-C", str(REPO_ROOT),
        *arguments,
    )
    result = ncu_executor.execute_bounded_command(
        command,
        environment=dict(_GIT_ENVIRONMENT),
        working_directory="/",
        inherited_descriptors=(retained_git.descriptor,),
        timeout_seconds=_GIT_TIMEOUT_SECONDS,
        max_output_bytes=_GIT_MAX_OUTPUT_BYTES,
        timeout_code="GOLDEN_GIT_TIMEOUT",
        launch_code="GOLDEN_GIT_LAUNCH_FAILED",
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        _fail(
            f"retained git {' '.join(arguments)} exited {result.returncode}: "
            f"{detail}"
        )
    return result.stdout


def _validate_current_checkout(manifest: dict[str, Any]) -> None:
    """Join retained preflight identity to the checkout used for collection."""
    expected_head = manifest.get("git_head")
    expected_pins = manifest.get("pins_sha256")
    if (
        type(expected_head) is not str
        or len(expected_head) != 40
        or any(character not in "0123456789abcdef" for character in expected_head)
        or type(expected_pins) is not str
        or len(expected_pins) != 64
        or any(character not in "0123456789abcdef" for character in expected_pins)
    ):
        _fail("preflight checkout identity fields are not admitted")
    pins = REPO_ROOT / "tools" / "local_labels" / "campaign_protocol_pins.json"
    try:
        observed_pins = _sha256_file(pins)
        with immutable.open_retained_executable(_GIT_EXECUTABLE) as retained_git:
            head_output = _run_checkout_git(
                retained_git, ("rev-parse", "--verify", "HEAD"),
            )
            status_output = _run_checkout_git(
                retained_git,
                ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
            )
            retained_git.verify_unchanged()
    except (
        OSError,
        immutable.ImmutableArtifactError,
        ncu_executor.NcuActionExecutorError,
    ) as exc:
        _fail(f"current checkout identity could not be established: {exc}")
    try:
        observed_head = head_output.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        _fail(f"current checkout HEAD is not ASCII: {exc}")
    if observed_head != expected_head:
        _fail(
            "preflight git HEAD differs from current checkout: "
            f"expected={expected_head}; observed={observed_head}"
        )
    if status_output:
        _fail("current checkout is dirty after preflight")
    if observed_pins != expected_pins:
        _fail(
            "preflight protocol pins differ from current checkout: "
            f"expected={expected_pins}; observed={observed_pins}"
        )


def _load_preflight(
    path: Path, *, expected_sha256: str, require_fresh: bool = True,
) -> tuple[str, ncu_executor.ExpectedNcuIdentity, dict[str, Any]]:
    path = _absolute_regular(path, label="preflight manifest")
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        _fail("preflight SHA-256 authority is not lowercase 64-hex")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read preflight manifest: {exc}")
    observed_sha256 = _sha256_bytes(payload)
    if observed_sha256 != expected_sha256:
        _fail(
            "preflight manifest SHA-256 mismatch: "
            f"expected={expected_sha256}; observed={observed_sha256}"
        )
    try:
        manifest = immutable.strict_json_loads(payload, label="preflight manifest")
    except immutable.ImmutableArtifactError as exc:
        _fail(f"preflight manifest is not strict JSON: {exc}")
    try:
        schema_payload = PREFLIGHT_SCHEMA.read_bytes()
    except OSError as exc:
        _fail(f"preflight schema authority is unavailable: {exc}")
    if _sha256_bytes(schema_payload) != PREFLIGHT_SCHEMA_SHA256:
        _fail("preflight schema authority SHA-256 mismatch")
    try:
        schema = immutable.strict_json_loads(
            schema_payload, label="preflight schema",
        )
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(manifest),
            key=lambda error: list(error.absolute_path),
        )
    except Exception as exc:
        _fail(f"preflight schema validation could not be completed: {exc}")
    if errors:
        error = errors[0]
        _fail(
            "preflight schema validation failed at "
            f"/{'/'.join(str(item) for item in error.absolute_path)}: "
            f"{error.message}"
        )
    if not isinstance(manifest, dict):
        _fail("preflight manifest root is not an object")
    if (
        manifest.get("capture_admitted") is not True
        or manifest.get("working_tree_clean") is not True
        or manifest.get("blockers") != []
        or manifest.get("arch") != "sm_86"
    ):
        _fail("preflight manifest is not an admitted clean sm_86 capture")
    generated_text = manifest.get("generated_at_utc")
    captured_text = manifest.get("captured_at_utc")
    if (
        type(generated_text) is not str
        or generated_text != captured_text
        or not generated_text.endswith("Z")
    ):
        _fail("preflight timestamp authority is not canonical")
    try:
        generated_at = datetime.fromisoformat(
            generated_text[:-1] + "+00:00"
        )
    except ValueError as exc:
        _fail(f"preflight generated_at_utc is invalid: {exc}")
    if generated_at.tzinfo is None:
        _fail("preflight generated_at_utc has no timezone")
    now = datetime.now(timezone.utc)
    age_seconds = (now - generated_at.astimezone(timezone.utc)).total_seconds()
    if age_seconds < -campaign_preflight.PREFLIGHT_FUTURE_SKEW_SECONDS:
        _fail("preflight generated_at_utc is too far in the future")
    if require_fresh and age_seconds > campaign_preflight.PREFLIGHT_FRESHNESS_SECONDS:
        _fail("preflight manifest is stale")
    checks = manifest.get("checks")
    if not isinstance(checks, dict):
        _fail("preflight checks are not an object")
    for name in (
        "worktree", "ncu", "driver", "nvcc", "nvbit", "capture_processes",
        "clock_state", "injection_runtime", "trace_budget", "sol_environment",
    ):
        probe = checks.get(name)
        if not isinstance(probe, dict) or probe.get("status") != "ok":
            _fail(f"preflight checks.{name}.status is not ok")
    device = checks.get("device")
    if (
        not isinstance(device, dict)
        or device.get("status") != "ok"
        or device.get("selected_host_gpu_index") != 0
        or device.get("nvidia_smi_device_count") != 1
    ):
        _fail("preflight device authority is not a single selected GPU")
    nvcc_probe = checks["nvcc"]
    if (
        not isinstance(nvcc_probe.get("expected"), dict)
        or nvcc_probe.get("observed") != nvcc_probe.get("expected")
    ):
        _fail("preflight NVCC release is not independently admitted")
    try:
        pins = immutable.strict_json_loads(
            (REPO_ROOT / "tools" / "local_labels" / "campaign_protocol_pins.json")
            .read_bytes(),
            label="campaign protocol pins",
        )
        expected_nvcc_release = pins["toolchain"]["nvcc_release"]
        capture_toolchain_contract.require_nvcc_release(
            nvcc_probe["expected"], expected_nvcc_release,
        )
        sol_environment = checks["sol_environment"]
        compiler_identity = sol_environment["compiler_identity"]
        if compiler_identity["requested"] != "g++-14":
            _fail("preflight SOL compiler is not the pinned g++-14")
        capture_toolchain_contract.require_nvcc_release(
            sol_environment["nvcc_release"], expected_nvcc_release,
        )
    except (
        KeyError, TypeError, immutable.ImmutableArtifactError,
        capture_toolchain_contract.CaptureToolchainContractError,
    ) as exc:
        _fail(f"preflight toolchain authority is not pinned: {exc}")
    ncu = checks["ncu"]
    if (
        ncu.get("expected") != EXPECTED_NCU_VERSION
        or ncu.get("observed") != ncu.get("version_line")
        or ncu.get("execution_mode") != EXPECTED_NCU_EXECUTION_MODE
        or ncu.get("host_machine") != EXPECTED_NCU_HOST_MACHINE
        or ncu.get("status") != "ok"
    ):
        _fail("preflight NCU authority fields are not admitted")
    try:
        expected = ncu_executor.build_expected_ncu_identity(
            resolved_path=ncu["resolved_path"],
            executable_sha256=ncu["executable_sha256"],
            version_line=ncu["version_line"],
        )
    except (KeyError, ncu_executor.NcuActionExecutorError) as exc:
        _fail(f"preflight NCU executable authority is invalid: {exc}")
    _validate_current_checkout(manifest)
    return observed_sha256, expected, manifest


def _details_csv_payload(output: bytes) -> str:
    """Strip only the closed NCU warning/profiler preamble."""
    try:
        text = output.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        _fail(f"golden NCU stdout is not UTF-8: {exc}")
    required = {
        "ID", "Kernel Name", "Block Size", "Grid Size", "Device",
        "Metric Name", "Metric Unit", "Metric Value",
    }
    lines = text.splitlines(keepends=True)
    candidates: list[int] = []
    for index, line in enumerate(lines):
        try:
            rows = list(csv.reader([line], strict=True))
        except csv.Error:
            continue
        if len(rows) == 1 and required.issubset(rows[0]):
            candidates.append(index)
    if len(candidates) != 1:
        _fail("golden NCU stdout has no unique Details CSV header")
    header_index = candidates[0]
    unexpected = [
        line.rstrip("\\r\\n")
        for line in lines[:header_index]
        if line.strip()
        and not line.startswith("==WARNING==")
        and not line.startswith("==PROF==")
    ]
    if unexpected:
        _fail(f"golden NCU stdout has unexpected preamble: {unexpected!r}")
    return "".join(lines[header_index:])


def _validate_golden_metric_csv(payload: str) -> None:
    try:
        parsed_rows = list(csv.reader(payload.splitlines(), strict=True))
    except csv.Error as exc:
        _fail(f"golden NCU Details CSV is malformed: {exc}")
    if not parsed_rows:
        _fail("golden NCU Details CSV has no rows")
    header = parsed_rows[0]
    if len(set(header)) != len(header):
        _fail("golden NCU Details CSV header has duplicate columns")
    if any(len(row) != len(header) for row in parsed_rows[1:]):
        _fail("golden NCU Details CSV contains a short or wide row")
    rows = [dict(zip(header, row, strict=True)) for row in parsed_rows[1:]]
    matches = [row for row in rows if row.get("Metric Name") == GOLDEN_METRIC]
    if len(matches) != 1:
        _fail(
            "golden NCU Details CSV must contain exactly one requested metric row"
        )
    row = matches[0]
    if row.get("Metric Unit") != "%":
        _fail("golden NCU metric unit is not %")
    try:
        value = float((row.get("Metric Value") or "").replace(",", ""))
    except ValueError as exc:
        _fail(f"golden NCU metric value is not numeric: {row.get('Metric Value')!r}")
        raise AssertionError("unreachable") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        _fail(f"golden NCU metric value is outside [0, 100]: {value!r}")
    def dim3(raw: str | None, label: str) -> tuple[int, int, int]:
        try:
            values = tuple(
                int(component.strip())
                for component in (raw or "").strip().strip("()").split(",")
            )
        except ValueError as exc:
            _fail(f"golden NCU {label} is malformed")
            raise AssertionError("unreachable") from exc
        if len(values) != 3 or any(component <= 0 for component in values):
            _fail(f"golden NCU {label} is malformed")
        return values  # type: ignore[return-value]
    if dim3(row.get("Grid Size"), "grid size") != (36864, 1, 1):
        _fail("golden NCU grid size differs from the admitted cell geometry")
    if dim3(row.get("Block Size"), "block size") != (1024, 1, 1):
        _fail("golden NCU block size differs from the admitted cell geometry")
    if row.get("Device") != "0" or not (row.get("Kernel Name") or "").strip():
        _fail("golden NCU launch identity is incomplete")


def _retained_target_mode(
    retained: immutable.RetainedExecutable,
) -> str:
    try:
        mode = stat.S_IMODE(os.fstat(retained.descriptor).st_mode)
        writable = os.access(
            f"/proc/self/fd/{retained.descriptor}",
            os.W_OK,
            effective_ids=True,
        )
    except (OSError, TypeError) as exc:
        _fail(f"golden target binary mode is unavailable: {exc}")
    if writable or mode & 0o222 or mode & 0o111 == 0 or mode & ~0o777:
        _fail(
            "golden target binary is not a write-protected ordinary executable: "
            f"mode={mode:04o}"
        )
    return f"{mode:04o}"


def _action_environment() -> dict[str, str]:
    """Return the exact environment used by execute_ncu_action."""
    return ncu_executor.ncu_action_environment()


def _validate_cell_paths(
    *,
    slug: str,
    trace_dir: Path,
    published_trace_dir: Path,
    staged_artifacts: tuple[tuple[Path, str], ...],
    published_artifacts: tuple[tuple[Path, str], ...],
) -> None:
    staged_cell = trace_dir.parent
    published_cell = published_trace_dir.parent
    if staged_cell != published_cell or published_cell.name != slug:
        _fail("golden trace and artifact paths do not share the selected cell")
    if published_trace_dir.name != "trace":
        _fail("published golden trace must be named trace")
    if (
        trace_dir == published_trace_dir
        or re.fullmatch(r"trace\.staged\.[1-9][0-9]*", trace_dir.name) is None
    ):
        _fail("staged golden trace does not use the closed staged name")
    expected_names = {
        "ncu_report": "ncu_l2.csv",
        "ncu_stderr": "ncu_l2.stderr",
        "ncu_launch_observation": "ncu_launch_observation.json",
        "label_run_manifest": "ncu_run_manifest.json",
        "label_capture_manifest": "ncu_capture_manifest.json",
        "ncu_collection_receipt": "ncu_collection_receipt.json",
    }
    staged_names = {
        "NCU CSV output": "ncu_l2.csv",
        "NCU stderr output": "ncu_l2.stderr",
        "NCU launch observation output": "ncu_launch_observation.json",
        "golden run manifest output": "ncu_run_manifest.json",
        "golden capture manifest output": "ncu_capture_manifest.json",
        "golden collection receipt output": "ncu_collection_receipt.json",
    }
    for path, label in staged_artifacts:
        expected_name = staged_names[label]
        if (
            path.parent != staged_cell
            or path in {published for published, _role in published_artifacts}
            or re.fullmatch(
                rf"{re.escape(expected_name)}\.staged\.[1-9][0-9]*",
                path.name,
            ) is None
        ):
            _fail(f"staged {label} escapes or renames the selected golden cell")
    for path, role in published_artifacts:
        if path.parent != published_cell or path.name != expected_names[role]:
            _fail(f"published {role} escapes or renames the selected golden cell")


def collect_golden_ncu(
    *,
    preflight_path: Path,
    preflight_sha256: str,
    slug: str,
    binary: Path,
    binary_sha256: str,
    trace_dir: Path,
    csv_out: Path,
    stderr_out: Path,
    observation_out: Path,
    run_manifest_out: Path,
    capture_manifest_out: Path,
    receipt_out: Path,
    published_trace_dir: Path,
    published_csv_out: Path,
    published_stderr_out: Path,
    published_observation_out: Path,
    published_run_manifest_out: Path,
    published_capture_manifest_out: Path,
    published_receipt_out: Path,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    if type(slug) is not str or not slug or "/" in slug or "\\" in slug:
        _fail("golden slug is invalid")
    for path, label in (
        (csv_out, "NCU CSV output"),
        (stderr_out, "NCU stderr output"),
        (observation_out, "NCU launch observation output"),
        (run_manifest_out, "golden run manifest output"),
        (capture_manifest_out, "golden capture manifest output"),
        (receipt_out, "golden collection receipt output"),
    ):
        _staging_path(path, label=label)
    for path, label in (
        (published_csv_out, "published NCU CSV path"),
        (published_stderr_out, "published NCU stderr path"),
        (published_observation_out, "published NCU launch observation path"),
        (published_run_manifest_out, "published golden run manifest path"),
        (published_capture_manifest_out, "published golden capture manifest path"),
        (published_receipt_out, "published golden receipt path"),
        (published_trace_dir, "published golden trace path"),
    ):
        _staging_path(path, label=label)
    binary = _absolute_regular(binary, label="golden target binary")
    if (
        type(binary_sha256) is not str
        or len(binary_sha256) != 64
        or any(character not in "0123456789abcdef" for character in binary_sha256)
    ):
        _fail("golden target binary SHA-256 authority is not lowercase 64-hex")
    if trace_dir.is_symlink():
        _fail("golden trace directory must not be a symlink")
    trace_dir = trace_dir.resolve(strict=False)
    try:
        trace_dir.relative_to(STAGING_ROOT)
    except ValueError as exc:
        _fail("golden trace directory escapes the staging namespace")
        raise AssertionError("unreachable") from exc
    if not trace_dir.is_dir() or trace_dir.is_symlink():
        _fail("golden trace directory is not an admitted directory")
    if binary.parent != trace_dir.parent or binary.name != f"{slug}.bin":
        _fail("golden target binary does not belong to the selected cell")
    _validate_cell_paths(
        slug=slug,
        trace_dir=trace_dir,
        published_trace_dir=published_trace_dir,
        staged_artifacts=(
            (csv_out, "NCU CSV output"),
            (stderr_out, "NCU stderr output"),
            (observation_out, "NCU launch observation output"),
            (run_manifest_out, "golden run manifest output"),
            (capture_manifest_out, "golden capture manifest output"),
            (receipt_out, "golden collection receipt output"),
        ),
        published_artifacts=(
            (published_csv_out, "ncu_report"),
            (published_stderr_out, "ncu_stderr"),
            (published_observation_out, "ncu_launch_observation"),
            (published_run_manifest_out, "label_run_manifest"),
            (published_capture_manifest_out, "label_capture_manifest"),
            (published_receipt_out, "ncu_collection_receipt"),
        ),
    )

    preflight_sha256, expected_ncu, manifest = _load_preflight(
        preflight_path, expected_sha256=preflight_sha256,
    )
    with immutable.PinnedDirectoryReader(trace_dir) as retained_trace_root:
        retained_trace_root.verify_path()
        trace_admission_model = _trace_admission_document(trace_dir)
        trace_tree_sha256, trace_tree_bytes = _trace_tree_identity(trace_dir)
        retained_trace_root.verify_path()
        environment = _action_environment()
        with immutable.open_retained_executable(binary) as retained_binary:
            if retained_binary.sha256 != binary_sha256:
                _fail(
                    "golden target binary SHA-256 differs from pre-NVBit authority: "
                    f"expected={binary_sha256}; observed={retained_binary.sha256}"
                )
            target_mode = _retained_target_mode(retained_binary)
            binary_size = os.fstat(retained_binary.descriptor).st_size
            target_argument = f"/proc/self/fd/{retained_binary.descriptor}"
            arguments = (
                "--metrics", GOLDEN_METRIC,
                "--clock-control", "none",
                "--cache-control", "all",
                "--replay-mode", "kernel",
                "--launch-count", "1",
                "--csv", "--", target_argument,
            )
            try:
                result = ncu_executor.execute_ncu_action(
                    expected_ncu,
                    arguments,
                    timeout_seconds=timeout_seconds,
                    auxiliary_descriptors=(retained_binary.descriptor,),
                    require_cgroup_containment=True,
                )
            except ncu_executor.NcuActionExecutorError as exc:
                raise GoldenNcuCaptureError(str(exc)) from exc
            binary_sha256 = retained_binary.sha256
            retained_binary.verify_unchanged()
        retained_trace_root.verify_path()
        if (
            _trace_admission_document(trace_dir) != trace_admission_model
            or _trace_tree_identity(trace_dir)
            != (trace_tree_sha256, trace_tree_bytes)
        ):
            _fail("golden NVBit trace changed between admission and NCU receipt")
        retained_trace_root.verify_path()
        post_preflight_sha256, post_expected_ncu, post_manifest = _load_preflight(
            preflight_path, expected_sha256=preflight_sha256,
        )
        if (
            post_preflight_sha256 != preflight_sha256
            or post_expected_ncu != expected_ncu
            or post_manifest != manifest
        ):
            _fail("preflight authority changed during golden NCU capture")
    if result.returncode != 0:
        _fail(f"golden NCU exited with status {result.returncode}")
    if not result.stdout:
        _fail("golden NCU emitted an empty CSV")

    _write_new(csv_out, result.stdout, label="NCU CSV output")
    _write_new(stderr_out, result.stderr, label="NCU stderr output")
    report_sha256 = _sha256_bytes(result.stdout)
    try:
        csv_payload = _details_csv_payload(result.stdout)
        _validate_golden_metric_csv(csv_payload)
        metric_result = parse_ncu_metrics.parse_raw_page_csv(
            csv_payload, (GOLDEN_METRIC,)
        )
        if not metric_result.ok:
            _fail(
                "golden NCU metric row is not admitted: "
                f"{metric_result.typed_reason}: {metric_result.detail}"
            )
        with immutable.PinnedDirectoryReader(observation_out.parent) as observation_parent:
            parse_ncu_metrics.write_launch_observation(
                csv_text=csv_payload,
                report_sha256=report_sha256,
                output_path=observation_out,
                requested_launch_skip=0,
                requested_launch_count=1,
                output_parent_descriptor=observation_parent.descriptor,
            )
            observation_parent.verify_path()
    except (UnicodeDecodeError, ValueError, OSError) as exc:
        _fail(f"golden NCU launch observation is invalid: {exc}")
    observation = immutable.strict_json_loads(
        immutable.read_regular(observation_out),
        label="golden NCU launch observation",
    )
    try:
        parse_ncu_metrics.validate_launch_observation_document(
            observation,
            requested_launch_skip=0,
            requested_launch_count=1,
            report_sha256=report_sha256,
        )
    except ValueError as exc:
        _fail(f"golden NCU launch observation is not admitted: {exc}")
    observation_sha256 = observation.get("observation_sha256")
    if not isinstance(observation_sha256, str):
        _fail("golden NCU launch observation has no seal")
    csv_relative = _relative_declared_artifact(published_csv_out)
    stderr_relative = _relative_declared_artifact(published_stderr_out)
    observation_relative = _relative_declared_artifact(published_observation_out)
    run_manifest = {
        "schema_version": 1,
        "kind": "golden_ncu_run_manifest_v1",
        "slug": slug,
        "preflight_sha256": preflight_sha256,
        "ncu_identity": {
            "resolved_path": expected_ncu.resolved_path,
            "executable_sha256": expected_ncu.executable_sha256,
            "version_line": expected_ncu.version_line,
        },
        "command": list(arguments),
        "environment": environment,
        "working_directory": "/",
        "target_binary": {
            "relative_path": binary.relative_to(REPO_ROOT).as_posix()
            if binary.is_relative_to(REPO_ROOT) else str(binary),
            "sha256": binary_sha256,
            "size_bytes": binary_size,
            "mode": target_mode,
            "executable": True,
        },
        "ncu_report": {
            "relative_path": csv_relative,
            "sha256": report_sha256,
            "size_bytes": len(result.stdout),
        },
        "launch_controls": {
            "clock_control": "none",
            "cache_control": "all",
            "replay_mode": "kernel",
            "launch_skip": 0,
            "launch_count": 1,
            "metric": GOLDEN_METRIC,
        },
        "exit_status": result.returncode,
    }
    capture_manifest = {
        "schema_version": 1,
        "kind": "golden_ncu_capture_manifest_v1",
        "slug": slug,
        "preflight_sha256": preflight_sha256,
        "trace_relative_path": _relative_declared_artifact(published_trace_dir),
        "trace_admission": trace_admission_model,
        "trace_tree_sha256": trace_tree_sha256,
        "trace_tree_bytes": trace_tree_bytes,
        "binary_sha256": binary_sha256,
        "ncu_csv_relative_path": csv_relative,
        "ncu_csv_sha256": report_sha256,
        "ncu_stderr_relative_path": stderr_relative,
        "ncu_stderr_sha256": _sha256_file(stderr_out),
        "ncu_launch_observation_sha256": observation_sha256,
        "ncu_controls": run_manifest["launch_controls"],
        "source_manifest_schema_version": manifest.get("schema_version"),
    }
    _write_new(
        run_manifest_out, _sealed_json(run_manifest), label="golden run manifest output",
    )
    _write_new(
        capture_manifest_out, _sealed_json(capture_manifest),
        label="golden capture manifest output",
    )
    artifacts = [
        {
            "role": "ncu_report", "relative_path": csv_relative,
            "sha256": report_sha256, "size_bytes": len(result.stdout),
        },
        {
            "role": "ncu_stderr",
            "relative_path": stderr_relative,
            "sha256": _sha256_file(stderr_out),
            "size_bytes": stderr_out.stat().st_size,
        },
        {
            "role": "label_run_manifest",
            "relative_path": _relative_declared_artifact(published_run_manifest_out),
            "sha256": _sha256_file(run_manifest_out),
            "size_bytes": run_manifest_out.stat().st_size,
        },
        {
            "role": "ncu_launch_observation", "relative_path": observation_relative,
            "sha256": _sha256_file(observation_out),
            "size_bytes": observation_out.stat().st_size,
        },
        {
            "role": "label_capture_manifest",
            "relative_path": _relative_declared_artifact(published_capture_manifest_out),
            "sha256": _sha256_file(capture_manifest_out),
            "size_bytes": capture_manifest_out.stat().st_size,
        },
    ]
    controls = run_manifest["launch_controls"]
    try:
        receipt = golden_ncu_receipt.build_receipt(
            preflight_sha256=preflight_sha256,
            preflight_path=str(preflight_path),
            expected_ncu_identity=expected_ncu,
            command=arguments,
            environment=environment,
            working_directory="/",
            target_binary=run_manifest["target_binary"],
            trace_relative_path=capture_manifest["trace_relative_path"],
            trace_admission=trace_admission_model,
            trace_tree_sha256=trace_tree_sha256,
            trace_tree_bytes=trace_tree_bytes,
            exit_status=result.returncode,
            artifacts=artifacts,
            launch_observation_sha256=observation_sha256,
            controls=controls,
        )
    except golden_ncu_receipt.GoldenReceiptError as exc:
        _fail(f"golden collection receipt could not be sealed: {exc}")
    _write_new(
        receipt_out,
        immutable.pretty_json_bytes(receipt),
        label="golden collection receipt output",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--validate-only" in raw_argv:
        validation = argparse.ArgumentParser(description=__doc__)
        validation.add_argument("--validate-only", action="store_true")
        validation.add_argument("--preflight", type=Path, required=True)
        validation.add_argument("--preflight-sha256", required=True)
        args = validation.parse_args(raw_argv)
        try:
            _load_preflight(args.preflight, expected_sha256=args.preflight_sha256)
        except GoldenNcuCaptureError as exc:
            print(f"golden_ncu_capture_refused: {exc}", file=sys.stderr)
            return 2
        print("golden_ncu_capture: preflight validation passed")
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--stderr-out", type=Path, required=True)
    parser.add_argument("--observation-out", type=Path, required=True)
    parser.add_argument("--run-manifest-out", type=Path, required=True)
    parser.add_argument("--capture-manifest-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--published-trace-dir", type=Path, required=True)
    parser.add_argument("--published-csv-out", type=Path, required=True)
    parser.add_argument("--published-stderr-out", type=Path, required=True)
    parser.add_argument("--published-observation-out", type=Path, required=True)
    parser.add_argument("--published-run-manifest-out", type=Path, required=True)
    parser.add_argument("--published-capture-manifest-out", type=Path, required=True)
    parser.add_argument("--published-receipt-out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    try:
        collect_golden_ncu(
            preflight_path=args.preflight,
            preflight_sha256=args.preflight_sha256,
            slug=args.slug,
            binary=args.binary,
            binary_sha256=args.binary_sha256,
            trace_dir=args.trace_dir,
            csv_out=args.csv_out,
            stderr_out=args.stderr_out,
            observation_out=args.observation_out,
            run_manifest_out=args.run_manifest_out,
            capture_manifest_out=args.capture_manifest_out,
            receipt_out=args.receipt_out,
            published_trace_dir=args.published_trace_dir,
            published_csv_out=args.published_csv_out,
            published_stderr_out=args.published_stderr_out,
            published_observation_out=args.published_observation_out,
            published_run_manifest_out=args.published_run_manifest_out,
            published_capture_manifest_out=args.published_capture_manifest_out,
            published_receipt_out=args.published_receipt_out,
            timeout_seconds=args.timeout_seconds,
        )
    except GoldenNcuCaptureError as exc:
        print(f"golden_ncu_capture_refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
