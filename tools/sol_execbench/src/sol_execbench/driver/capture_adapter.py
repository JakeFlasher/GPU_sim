# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent, fail-closed preparation for one SOL profiler capture.

The normal SOL judge stages and deletes a fresh module for every invocation.
Reference capture instead needs one compiled module, one selected workload, and
the same exact bytes for label, trace, and control evidence.  This module is an
opt-in adapter for that narrow purpose; the canonical judge never calls it when
``SOL_EXECBENCH_CAPTURE_REQUEST`` is absent.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

# The console script is an editable install whose sys.path contains this
# package's ``src`` directory but not the repository root.  Resolve the shared
# certifying comparator from the adapter's own checkout location, never from an
# ambient PYTHONPATH or caller working directory.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFINITION_EQUIVALENCE_SOURCE = (
    _REPO_ROOT / "tools/local_labels/sol_definition_equivalence.py"
)
if not _DEFINITION_EQUIVALENCE_SOURCE.is_file():
    raise ImportError(
        "certifying SOL definition-equivalence source is absent at "
        f"{_DEFINITION_EQUIVALENCE_SOURCE}"
    )
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.local_labels.sol_definition_equivalence import (
    SolDefinitionEquivalenceError,
    ineligible_normalized_state_pointers,
    parse_definition_bytes,
    require_definition_equivalent,
    serialize_attestation,
    validate_attestation_shape_and_eligibility,
)
from tools.local_labels.sol_reconstitution import (
    PrerequisiteContractError,
    verify_flashinfer_capture_contract,
)

from ..core import BenchmarkConfig, Definition, Solution, Workload
from .certified_env import (
    CERTIFIED_ENVIRONMENT_CONTRACT,
    CertifiedEnvironmentError,
    LEGACY_CERTIFIED_ENVIRONMENT_CONTRACT,
    build_certified_env,
    build_legacy_certified_env,
    environment_contract_sha256,
    legacy_environment_contract_sha256,
)
from .problem_packager import ProblemPackager

CAPTURE_REQUEST_ENV = "SOL_EXECBENCH_CAPTURE_REQUEST"
CAPTURE_MODE_ENV = "SOL_EXECBENCH_CAPTURE_MODE"
CAPTURE_CONTROL_ENVIRONMENTS = (
    CAPTURE_REQUEST_ENV,
    CAPTURE_MODE_ENV,
    "SOL_EXECBENCH_CAPTURE_PREPARE_MANIFEST",
    "SOL_EXECBENCH_CAPTURE_SELECTION_MANIFEST",
    "SOL_EXECBENCH_CAPTURE_SELECTION_SHA256",
    "SOL_EXECBENCH_CAPTURE_RESULT_MANIFEST",
    "SOL_EXECBENCH_CAPTURE_SOURCE_WORKLOAD_INDEX",
)
PYTORCH_ALLOC_CONF = "expandable_segments:True"
PREPARE_MANIFEST_NAME = "capture_prepare_manifest.json"
CAPTURE_CONTRACT_VERSION = 2
_CAPTURE_KINDS = frozenset({"label", "trace", "control", "smoke"})
_OPTIONAL_DEFINITION_FIELDS = ("hf_id", "op_type", "custom_inputs_entrypoint")


class CaptureAdapterError(RuntimeError):
    """A typed, fail-closed capture adapter refusal."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _environment_contract(
    contract: Mapping[str, str] | None,
) -> tuple[dict[str, str], str]:
    """Dispatch only the two versioned, exact certified contract shapes."""

    try:
        if contract is None:
            raise CertifiedEnvironmentError(
                "SOL_ENV_CONTRACT_INVALID", "environment contract is required"
            )
        normalized = dict(contract)
        keys = frozenset(normalized)
        if keys == frozenset(LEGACY_CERTIFIED_ENVIRONMENT_CONTRACT):
            digest = legacy_environment_contract_sha256(normalized)
        elif keys == frozenset(CERTIFIED_ENVIRONMENT_CONTRACT):
            digest = environment_contract_sha256(normalized)
        else:
            raise CertifiedEnvironmentError(
                "SOL_ENV_CONTRACT_INVALID",
                "environment contract must be exactly the legacy-v1 or live-v2 shape",
            )
    except (TypeError, ValueError) as exc:
        code = getattr(exc, "code", "SOL_ENV_CONTRACT_INVALID")
        detail = getattr(exc, "detail", str(exc))
        raise CaptureAdapterError(code, detail) from exc
    return normalized, digest


def _flashinfer_contract(
    contract: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str, Path | None]:
    """Re-prove the closed corpus authority before any capture mutation."""
    if contract is None:
        raise CaptureAdapterError(
            "SOL_FIB_CORPUS_CONTRACT_INVALID", "FlashInfer contract is required"
        )
    try:
        normalized = dict(contract)
        runtime_root = verify_flashinfer_capture_contract(
            normalized, repo_root=_REPO_ROOT,
        )
    except (TypeError, ValueError, PrerequisiteContractError) as exc:
        raise CaptureAdapterError(
            "SOL_FIB_CORPUS_CONTRACT_INVALID", str(exc)
        ) from exc
    digest = sha256_bytes(_canonical_json_bytes(normalized))
    return normalized, digest, runtime_root


def _certified_subprocess_env(
    contract: Mapping[str, str],
    environ: Mapping[str, str] | None,
    *,
    transport: Mapping[str, str] | None = None,
    flashinfer_trace_dir: Path | None = None,
) -> dict[str, str]:
    """Recover the certified base, then add only explicit phase transport."""

    ambient = dict(os.environ if environ is None else environ)
    for name in CAPTURE_CONTROL_ENVIRONMENTS:
        ambient.pop(name, None)
    ambient["PYTORCH_ALLOC_CONF"] = PYTORCH_ALLOC_CONF
    try:
        keys = frozenset(contract)
        if keys == frozenset(LEGACY_CERTIFIED_ENVIRONMENT_CONTRACT):
            certified = build_legacy_certified_env(
                contract, ambient, flashinfer_trace_dir=flashinfer_trace_dir,
            )
        elif keys == frozenset(CERTIFIED_ENVIRONMENT_CONTRACT):
            certified = build_certified_env(
                contract, ambient, flashinfer_trace_dir=flashinfer_trace_dir,
            )
        else:
            raise CertifiedEnvironmentError(
                "SOL_ENV_CONTRACT_INVALID",
                "environment contract must be exactly the legacy-v1 or live-v2 shape",
            )
    except CertifiedEnvironmentError as exc:
        raise CaptureAdapterError(exc.code, exc.detail) from exc
    if transport is None:
        return certified
    extra = dict(transport)
    if (
        any(not isinstance(key, str) or not isinstance(value, str)
            for key, value in extra.items())
        or not set(extra).issubset(CAPTURE_CONTROL_ENVIRONMENTS)
        or CAPTURE_REQUEST_ENV in extra
    ):
        raise CaptureAdapterError(
            "capture_transport_invalid", repr(sorted(extra))
        )
    certified.update(extra)
    return certified


def _stage_definition_sha256(manifest_without_digest: Mapping[str, Any]) -> str:
    return sha256_bytes(_canonical_json_bytes(dict(manifest_without_digest)))


def _load_json_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureAdapterError(code, f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureAdapterError(code, f"{path} must contain one JSON object")
    return value


def _source_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CaptureAdapterError("capture_input_absent", f"{label}: {resolved}")
    return resolved


def _source_entry(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _artifact_entry(actual_path: Path, declared_path: Path) -> dict[str, str]:
    if not actual_path.is_file():
        raise CaptureAdapterError(
            "capture_artifact_absent", f"required staged artifact: {actual_path}"
        )
    return {"path": str(declared_path), "sha256": sha256_file(actual_path)}


def _strict_source_definition_bytes(path: Path) -> bytes:
    try:
        payload = path.read_bytes()
        parse_definition_bytes(payload, side="source")
    except OSError as exc:
        raise CaptureAdapterError(
            "definition_invalid", f"cannot read {path}: {exc}"
        ) from exc
    except SolDefinitionEquivalenceError as exc:
        raise CaptureAdapterError(exc.code, exc.detail) from exc
    return payload


def _require_definition_equivalence(
    source_bytes: bytes, staged_bytes: bytes
):
    try:
        return require_definition_equivalent(source_bytes, staged_bytes)
    except SolDefinitionEquivalenceError as exc:
        raise CaptureAdapterError(exc.code, exc.detail) from exc


def _validate_definition_attestation(attestation: Any) -> dict[str, Any]:
    try:
        return validate_attestation_shape_and_eligibility(attestation)
    except SolDefinitionEquivalenceError as exc:
        raise CaptureAdapterError(exc.code, exc.detail) from exc


def _normalise_definition(raw: dict[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    for field in _OPTIONAL_DEFINITION_FIELDS:
        if value.get(field) == "":
            value[field] = None
    return value


def _load_definition(path: Path) -> Definition:
    return Definition(**_normalise_definition(_load_json_object(path, code="definition_invalid")))


def _load_solution(path: Path) -> Solution:
    raw = _load_json_object(path, code="solution_invalid")
    for source in raw.get("sources", []):
        if isinstance(source, dict) and not source.get("content"):
            source_path = path.parent / str(source.get("path", ""))
            if source_path.is_file():
                source["content"] = source_path.read_text()
    return Solution(**raw)


def _load_config(path: Path | None) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    return BenchmarkConfig(**_load_json_object(path, code="config_invalid"))


def _select_workload(
    path: Path, source_workload_index: int
) -> tuple[Workload, bytes, int]:
    if (
        isinstance(source_workload_index, bool)
        or not isinstance(source_workload_index, int)
        or source_workload_index < 0
    ):
        raise CaptureAdapterError(
            "workload_index_invalid", "source_workload_index must be a nonnegative integer"
        )
    rows: list[bytes] = []
    for raw_line in path.read_bytes().splitlines():
        stripped = raw_line.strip()
        if stripped:
            rows.append(stripped)
    if not rows:
        raise CaptureAdapterError(
            "workload_set_empty", f"{path} contains zero nonempty workload rows"
        )
    if source_workload_index >= len(rows):
        raise CaptureAdapterError(
            "workload_index_out_of_range",
            f"index={source_workload_index} workload_count={len(rows)}",
        )
    try:
        raw = json.loads(rows[source_workload_index])
        if not isinstance(raw, dict):
            raise TypeError("workload row is not an object")
        workload = Workload(**raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CaptureAdapterError(
            "workload_row_invalid", f"row {source_workload_index}: {exc}"
        ) from exc
    return workload, rows[source_workload_index], len(rows)


def _write_json_new(path: Path, value: Any) -> None:
    """Publish a JSON object atomically without replacing existing evidence."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CaptureAdapterError("capture_output_exists", str(path))
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with tmp.open("xb") as fh:
            fh.write(_canonical_json_bytes(value))
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise CaptureAdapterError("capture_output_exists", str(path)) from exc
    finally:
        tmp.unlink(missing_ok=True)


def _verify_entry(
    entry: Any,
    *,
    label: str,
    base_dir: Path | None = None,
    path_kind: str = "absolute",
) -> Path:
    if not isinstance(entry, dict):
        raise CaptureAdapterError("capture_manifest_invalid", f"{label} is not an object")
    raw_path = entry.get("path")
    expected = entry.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected, str):
        raise CaptureAdapterError(
            "capture_manifest_invalid", f"{label} lacks path/sha256"
        )
    declared = Path(raw_path)
    if path_kind == "absolute" and not declared.is_absolute():
        raise CaptureAdapterError(
            "capture_manifest_invalid", f"{label} path must be absolute"
        )
    if path_kind == "stage_relative" and declared.is_absolute():
        raise CaptureAdapterError(
            "capture_manifest_invalid", f"{label} path must be stage-relative"
        )
    if not declared.is_absolute():
        if base_dir is None or ".." in declared.parts or declared == Path("."):
            raise CaptureAdapterError(
                "capture_manifest_invalid", f"{label} has unsafe relative path"
            )
        path = (base_dir / declared).resolve()
        try:
            path.relative_to(base_dir.resolve())
        except ValueError as exc:
            raise CaptureAdapterError(
                "capture_manifest_invalid", f"{label} escapes the stage"
            ) from exc
    else:
        path = declared
    if not path.is_file():
        raise CaptureAdapterError("capture_artifact_absent", f"{label}: {path}")
    actual = sha256_file(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
        raise CaptureAdapterError(
            "capture_artifact_hash_mismatch",
            f"{label}: expected={expected} actual={actual}",
        )
    return path


def _stage_evidence_path(stage: Path, path: Path, *, label: str) -> tuple[Path, str]:
    """Resolve evidence inside a stage and return its portable relative name."""
    if path.is_absolute():
        resolved = path.expanduser().resolve()
    else:
        if ".." in path.parts or path == Path("."):
            raise CaptureAdapterError(
                "capture_evidence_path_invalid", f"{label}: {path}"
            )
        resolved = (stage / path).resolve()
    try:
        relative = resolved.relative_to(stage)
    except ValueError as exc:
        raise CaptureAdapterError(
            "capture_evidence_path_invalid", f"{label} must be inside {stage}"
        ) from exc
    if relative == Path("."):
        raise CaptureAdapterError(
            "capture_evidence_path_invalid", f"{label} cannot be the stage root"
        )
    return resolved, relative.as_posix()


def validate_capture_stage(
    staging_dir: Path,
    *,
    definition_file: Path | None = None,
    workload_file: Path | None = None,
    solution_file: Path | None = None,
    config_file: Path | None = None,
    adapter_executable: Path | None = None,
    source_workload_index: int | None = None,
    environment_contract: Mapping[str, str] | None = None,
    flashinfer_corpus_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rehash every source and staged consumer input before execution or reuse."""
    stage = staging_dir.expanduser().resolve()
    manifest_path = stage / PREPARE_MANIFEST_NAME
    manifest = _load_json_object(manifest_path, code="capture_manifest_invalid")
    if manifest.get("schema_version") != 1 or manifest.get(
        "capture_contract_version"
    ) != CAPTURE_CONTRACT_VERSION:
        raise CaptureAdapterError(
            "capture_manifest_invalid", "unsupported preparation manifest version"
        )
    if manifest.get("manifest_kind") != "sol_execbench_capture_preparation":
        raise CaptureAdapterError("capture_manifest_invalid", "wrong manifest_kind")
    if (
        manifest.get("path_mode") != "stage_relative_v1"
        or manifest.get("stage_root") != "."
        or "stage_dir" in manifest
    ):
        raise CaptureAdapterError(
            "capture_manifest_invalid", "stage path contract is not relocatable"
        )

    staged_environment, staged_environment_sha256 = _environment_contract(
        manifest.get("environment_contract")
    )
    if manifest.get("environment_contract_sha256") != staged_environment_sha256:
        raise CaptureAdapterError(
            "capture_environment_contract_drift",
            "preparation environment contract digest does not rederive",
        )
    if environment_contract is not None:
        expected_environment, expected_environment_sha256 = _environment_contract(
            environment_contract
        )
        if (
            staged_environment != expected_environment
            or staged_environment_sha256 != expected_environment_sha256
        ):
            raise CaptureAdapterError(
                "capture_environment_contract_drift",
                "staged environment contract differs from expected contract",
            )
    staged_corpus, staged_corpus_sha256, _runtime_root = _flashinfer_contract(
        manifest.get("flashinfer_corpus_contract")
    )
    if (
        manifest.get("flashinfer_corpus_contract_sha256")
        != staged_corpus_sha256
    ):
        raise CaptureAdapterError(
            "SOL_FIB_CORPUS_CONTRACT_INVALID",
            "preparation FlashInfer corpus digest does not rederive",
        )
    if flashinfer_corpus_contract is not None:
        expected_corpus, expected_corpus_sha256, _expected_root = (
            _flashinfer_contract(flashinfer_corpus_contract)
        )
        if (
            staged_corpus != expected_corpus
            or staged_corpus_sha256 != expected_corpus_sha256
        ):
            raise CaptureAdapterError(
                "SOL_FIB_CORPUS_CONTRACT_INVALID",
                "staged FlashInfer corpus contract differs from expected",
            )
    claimed_stage_definition_sha256 = manifest.get("stage_definition_sha256")
    manifest_without_stage_digest = dict(manifest)
    manifest_without_stage_digest.pop("stage_definition_sha256", None)
    actual_stage_definition_sha256 = _stage_definition_sha256(
        manifest_without_stage_digest
    )
    if claimed_stage_definition_sha256 != actual_stage_definition_sha256:
        raise CaptureAdapterError(
            "capture_stage_definition_drift",
            "preparation stage definition digest does not rederive",
        )

    sources = manifest.get("source_inputs")
    artifacts = manifest.get("artifacts")
    if not isinstance(sources, dict) or not isinstance(artifacts, dict):
        raise CaptureAdapterError(
            "capture_manifest_invalid", "source_inputs/artifacts must be objects"
        )
    source_paths = {}
    for name in ("definition", "workload", "solution"):
        source_paths[name] = _verify_entry(
            sources.get(name), label=f"source_inputs.{name}"
        )
    if "config" in sources:
        _verify_entry(sources.get("config"), label="source_inputs.config")
    stage_artifacts = (
        "definition",
        "definition_equivalence_attestation",
        "workload",
        "solution",
        "eval_driver",
        "capture_runtime",
        "benchmark_kernel_so",
    )
    artifact_paths = {}
    for name in stage_artifacts:
        artifact_paths[name] = _verify_entry(
            artifacts.get(name),
            label=f"artifacts.{name}",
            base_dir=stage,
            path_kind="stage_relative",
        )
    definition_source_entry = artifacts.get("definition_source")
    if (
        not isinstance(definition_source_entry, dict)
        or set(definition_source_entry) != {"path", "sha256"}
        or definition_source_entry.get("path") != "definition.source.json"
    ):
        raise CaptureAdapterError(
            "sol_stage_source_copy_missing",
            "artifacts.definition_source must pin definition.source.json",
        )
    definition_source_path = stage / "definition.source.json"
    if definition_source_path.is_symlink() or not definition_source_path.is_file():
        raise CaptureAdapterError(
            "sol_stage_source_copy_missing", str(definition_source_path)
        )
    copied_sha256 = sha256_file(definition_source_path)
    source_definition_sha256 = sha256_file(source_paths["definition"])
    if (
        definition_source_entry.get("sha256") != copied_sha256
        or copied_sha256 != source_definition_sha256
        or manifest.get("source_definition_sha256") != source_definition_sha256
        or definition_source_path.read_bytes()
        != source_paths["definition"].read_bytes()
    ):
        raise CaptureAdapterError(
            "sol_stage_source_copy_mismatch",
            "definition.source.json is not the exact pinned source definition",
        )
    staged_definition_sha256 = sha256_file(artifact_paths["definition"])
    if manifest.get("staged_definition_sha256") != staged_definition_sha256:
        raise CaptureAdapterError(
            "sol_stage_source_mismatch",
            "staged_definition_sha256 does not match definition.json",
        )
    comparison = _require_definition_equivalence(
        definition_source_path.read_bytes(), artifact_paths["definition"].read_bytes()
    )
    observed_attestation = _validate_definition_attestation(
        manifest.get("definition_equivalence")
    )
    expected_attestation = comparison.attestation
    try:
        ineligible_pointers = ineligible_normalized_state_pointers(
            observed_attestation, expected_attestation
        )
    except SolDefinitionEquivalenceError as exc:
        raise CaptureAdapterError(exc.code, exc.detail) from exc
    if ineligible_pointers:
        raise CaptureAdapterError(
            "SOL_ATTESTATION_POINTER_INELIGIBLE",
            f"recorded normalized states are outside the sealed definitions: "
            f"{list(ineligible_pointers)!r}",
        )
    if observed_attestation != expected_attestation:
        raise CaptureAdapterError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "definition equivalence attestation does not rederive",
        )
    if (
        artifact_paths["definition_equivalence_attestation"].is_symlink()
        or artifact_paths["definition_equivalence_attestation"].read_bytes()
        != serialize_attestation(expected_attestation)
    ):
        raise CaptureAdapterError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "definition equivalence attestation bytes are not deterministic",
        )
    for name in ("adapter", "adapter_source", "certified_env_source"):
        _verify_entry(artifacts.get(name), label=f"artifacts.{name}")
    solution_sources = artifacts.get("solution_sources")
    if not isinstance(solution_sources, list) or not solution_sources:
        raise CaptureAdapterError(
            "capture_manifest_invalid", "artifacts.solution_sources is empty"
        )
    for index, entry in enumerate(solution_sources):
        _verify_entry(
            entry,
            label=f"artifacts.solution_sources[{index}]",
            base_dir=stage,
            path_kind="stage_relative",
        )

    exact_stage_paths = {
        "definition": Path("definition.json"),
        "definition_source": Path("definition.source.json"),
        "definition_equivalence_attestation": Path(
            "definition_equivalence_attestation.json"
        ),
        "workload": Path("workload.jsonl"),
        "solution": Path("solution.json"),
        "eval_driver": Path("eval_driver.py"),
        "capture_runtime": Path("_capture_runtime.py"),
        "benchmark_kernel_so": Path("benchmark_kernel.so"),
    }
    for name, expected in exact_stage_paths.items():
        if Path(artifacts[name]["path"]) != expected:
            raise CaptureAdapterError(
                "capture_manifest_invalid", f"artifacts.{name} path mismatch"
            )
    if Path(artifacts["adapter_source"]["path"]) != Path(__file__).resolve():
        raise CaptureAdapterError(
            "capture_manifest_invalid", "adapter source path mismatch"
        )
    certified_env_source = Path(__file__).with_name("certified_env.py").resolve()
    if Path(artifacts["certified_env_source"]["path"]) != certified_env_source:
        raise CaptureAdapterError(
            "capture_manifest_invalid", "certified environment source path mismatch"
        )

    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise CaptureAdapterError("capture_manifest_invalid", "selection is not an object")
    pinned_index = selection.get("source_workload_index")
    if isinstance(pinned_index, bool) or not isinstance(pinned_index, int) or pinned_index < 0:
        raise CaptureAdapterError("capture_manifest_invalid", "invalid pinned workload index")
    if selection.get("staged_workload_index") != 0:
        raise CaptureAdapterError("capture_manifest_invalid", "staged workload index must be zero")
    staged_rows = [
        line.strip()
        for line in (stage / exact_stage_paths["workload"]).read_bytes().splitlines()
        if line.strip()
    ]
    if len(staged_rows) != 1:
        raise CaptureAdapterError(
            "capture_manifest_invalid",
            f"staged workload must contain exactly one row, found {len(staged_rows)}",
        )
    if sha256_bytes(staged_rows[0]) != selection.get("source_row_sha256"):
        raise CaptureAdapterError(
            "capture_manifest_invalid", "selected workload row hash mismatch"
        )

    source_workload, source_row, source_count = _select_workload(
        Path(sources["workload"]["path"]), pinned_index
    )
    expected_selection = {
        "source_workload_index": pinned_index,
        "source_workload_count": source_count,
        "staged_workload_index": 0,
        "workload_uuid": source_workload.uuid,
        "workload_axes": source_workload.axes,
        "source_row_sha256": sha256_bytes(source_row),
    }
    if selection != expected_selection:
        raise CaptureAdapterError(
            "capture_manifest_invalid", "selection does not rederive from source workload"
        )

    source_definition = _load_definition(Path(sources["definition"]["path"]))
    source_solution = _load_solution(Path(sources["solution"]["path"]))
    expected_solution_contract = {
        "definition": source_definition.name,
        "solution": source_solution.name,
        "languages": [language.value for language in source_solution.spec.languages],
        "entry_point": source_solution.spec.entry_point,
        "destination_passing_style": True,
    }
    if (
        source_solution.definition != source_definition.name
        or source_solution.spec.destination_passing_style is not True
        or manifest.get("solution_contract") != expected_solution_contract
    ):
        raise CaptureAdapterError(
            "capture_manifest_invalid", "solution contract does not rederive from sources"
        )

    expected_sources = {
        "definition": definition_file,
        "workload": workload_file,
        "solution": solution_file,
        "config": config_file,
    }
    for name, expected_source in expected_sources.items():
        if expected_source is None:
            continue
        if name not in sources:
            raise CaptureAdapterError(
                "capture_source_mismatch", f"{name}: stage has no source pin"
            )
        expected_path = expected_source.expanduser().resolve()
        if Path(sources[name]["path"]) != expected_path:
            raise CaptureAdapterError(
                "capture_source_mismatch", f"{name}: expected {expected_path}"
            )
    if adapter_executable is not None:
        expected_adapter = adapter_executable.expanduser().resolve()
        if Path(artifacts["adapter"]["path"]) != expected_adapter:
            raise CaptureAdapterError(
                "capture_source_mismatch",
                f"adapter executable expected {expected_adapter}",
            )
    if source_workload_index is not None and pinned_index != source_workload_index:
        raise CaptureAdapterError(
            "capture_source_mismatch",
            f"workload index expected={source_workload_index} staged={pinned_index}",
        )
    return manifest


def prepare_capture_stage(
    *,
    definition_file: Path,
    workload_file: Path,
    solution_file: Path,
    staging_dir: Path,
    source_workload_index: int,
    config_file: Path | None = None,
    compile_timeout: int = 120,
    adapter_executable: Path | None = None,
    environment_contract: Mapping[str, str] | None = None,
    flashinfer_corpus_contract: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Validate, compile once, and atomically publish one persistent stage."""
    definition_path = _source_file(definition_file, label="definition")
    workload_path = _source_file(workload_file, label="workload")
    solution_path = _source_file(solution_file, label="solution")
    config_path = (
        _source_file(config_file, label="config") if config_file is not None else None
    )
    stage = staging_dir.expanduser().resolve()
    if compile_timeout <= 0:
        raise CaptureAdapterError(
            "capture_timeout_invalid", "compile_timeout must be positive"
        )
    certified_environment, certified_environment_sha256 = _environment_contract(
        environment_contract
    )
    verified_corpus, verified_corpus_sha256, flashinfer_trace_dir = (
        _flashinfer_contract(flashinfer_corpus_contract)
    )
    compile_environment = _certified_subprocess_env(
        certified_environment, environ,
        flashinfer_trace_dir=flashinfer_trace_dir,
    )

    # Validate the pinned row and all semantic bindings before compilation can
    # inspect a GPU or the evaluation driver can allocate CUDA memory.
    source_definition_bytes = _strict_source_definition_bytes(definition_path)
    definition = _load_definition(definition_path)
    workload, raw_row, workload_count = _select_workload(
        workload_path, source_workload_index
    )
    solution = _load_solution(solution_path)
    config = _load_config(config_path)
    if solution.definition != definition.name:
        raise CaptureAdapterError(
            "capture_definition_mismatch",
            f"solution={solution.definition!r} definition={definition.name!r}",
        )
    if solution.spec.destination_passing_style is not True:
        raise CaptureAdapterError(
            "capture_solution_not_dps",
            "capture mode requires pre-allocated destination-passing outputs",
        )

    if stage.exists():
        manifest = validate_capture_stage(
            stage,
            definition_file=definition_path,
            workload_file=workload_path,
            solution_file=solution_path,
            config_file=config_path,
            adapter_executable=adapter_executable,
            source_workload_index=source_workload_index,
            environment_contract=certified_environment,
            flashinfer_corpus_contract=verified_corpus,
        )
        return {"status": "reused", "manifest": manifest}

    stage.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{stage.name}.prepare-", dir=stage.parent))
    published = False
    try:
        packager = ProblemPackager(
            definition=definition,
            workloads=[workload],
            solution=solution,
            config=config,
            output_dir=tmp,
            keep_output_dir=True,
        )
        if not packager._is_cpp:
            raise CaptureAdapterError(
                "capture_solution_not_compiled",
                "capture contract requires a staged benchmark_kernel.so",
            )

        definition_source_copy = tmp / "definition.source.json"
        try:
            shutil.copyfile(definition_path, definition_source_copy)
        except OSError as exc:
            raise CaptureAdapterError(
                "sol_stage_source_copy_missing",
                f"cannot copy pinned definition to the stage: {exc}",
            ) from exc
        if definition_source_copy.read_bytes() != source_definition_bytes:
            raise CaptureAdapterError(
                "sol_stage_source_copy_mismatch",
                "definition.source.json is not byte-identical to the pinned source",
            )
        definition_comparison = _require_definition_equivalence(
            source_definition_bytes, (tmp / "definition.json").read_bytes()
        )
        definition_attestation = definition_comparison.attestation
        definition_attestation_path = (
            tmp / "definition_equivalence_attestation.json"
        )
        definition_attestation_path.write_bytes(
            serialize_attestation(definition_attestation)
        )

        # Preserve the selected source workload bytes, not a reconstructed set.
        (tmp / "workload.jsonl").write_bytes(raw_row + b"\n")
        compile_cmd, _ = packager.compile()
        proc = run(
            compile_cmd,
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=compile_timeout,
            env=compile_environment,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "no compiler output").strip()
            raise CaptureAdapterError("capture_compile_failed", detail[-4000:])
        packager.execute()

        runtime_source = Path(__file__).with_name("capture_runtime.py").resolve()
        certified_env_source = Path(__file__).with_name("certified_env.py").resolve()
        shutil.copyfile(runtime_source, tmp / "_capture_runtime.py")
        adapter_source = Path(__file__).resolve()
        adapter_exec = (
            _source_file(adapter_executable, label="adapter executable")
            if adapter_executable is not None
            else adapter_source
        )

        staged_source_entries = []
        for source in solution.sources:
            actual = tmp / source.path
            staged_source_entries.append(
                {
                    "relative_path": source.path,
                    **_artifact_entry(actual, Path(source.path)),
                }
            )

        artifacts = {
            "definition": _artifact_entry(tmp / "definition.json", Path("definition.json")),
            "definition_source": _artifact_entry(
                definition_source_copy, Path("definition.source.json")
            ),
            "definition_equivalence_attestation": _artifact_entry(
                definition_attestation_path,
                Path("definition_equivalence_attestation.json"),
            ),
            "workload": _artifact_entry(tmp / "workload.jsonl", Path("workload.jsonl")),
            "solution": _artifact_entry(tmp / "solution.json", Path("solution.json")),
            "adapter": _artifact_entry(adapter_exec, adapter_exec),
            "adapter_source": _artifact_entry(adapter_source, adapter_source),
            "certified_env_source": _artifact_entry(
                certified_env_source, certified_env_source
            ),
            "eval_driver": _artifact_entry(tmp / "eval_driver.py", Path("eval_driver.py")),
            "capture_runtime": _artifact_entry(
                tmp / "_capture_runtime.py", Path("_capture_runtime.py")
            ),
            "benchmark_kernel_so": _artifact_entry(
                tmp / "benchmark_kernel.so", Path("benchmark_kernel.so")
            ),
            "solution_sources": staged_source_entries,
        }
        source_inputs: dict[str, Any] = {
            "definition": _source_entry(definition_path),
            "workload": _source_entry(workload_path),
            "solution": _source_entry(solution_path),
        }
        if config_path is not None:
            source_inputs["config"] = _source_entry(config_path)

        manifest = {
            "schema_version": 1,
            "manifest_kind": "sol_execbench_capture_preparation",
            "capture_contract_version": CAPTURE_CONTRACT_VERSION,
            "path_mode": "stage_relative_v1",
            "stage_root": ".",
            "source_inputs": source_inputs,
            "selection": {
                "source_workload_index": source_workload_index,
                "source_workload_count": workload_count,
                "staged_workload_index": 0,
                "workload_uuid": workload.uuid,
                "workload_axes": workload.axes,
                "source_row_sha256": sha256_bytes(raw_row),
            },
            "solution_contract": {
                "definition": definition.name,
                "solution": solution.name,
                "languages": [language.value for language in solution.spec.languages],
                "entry_point": solution.spec.entry_point,
                "destination_passing_style": True,
            },
            "benchmark_config": dataclasses.asdict(config),
            "environment_contract": certified_environment,
            "environment_contract_sha256": certified_environment_sha256,
            "flashinfer_corpus_contract": verified_corpus,
            "flashinfer_corpus_contract_sha256": verified_corpus_sha256,
            "source_definition_sha256": definition_comparison.source_sha256,
            "staged_definition_sha256": definition_comparison.staged_sha256,
            "definition_equivalence": definition_attestation,
            "artifacts": artifacts,
            "execution": {
                "command": [sys.executable, "eval_driver.py"],
                "working_directory": ".",
                "timing_methodology": "cuda_events",
                "correctness_rounds": 1,
                "warmup_calls": 1,
                "profiled_candidate_calls": 1,
            },
        }
        manifest["stage_definition_sha256"] = _stage_definition_sha256(manifest)
        (tmp / PREPARE_MANIFEST_NAME).write_bytes(_canonical_json_bytes(manifest))
        try:
            tmp.rename(stage)
        except FileExistsError as exc:
            raise CaptureAdapterError("capture_stage_exists", str(stage)) from exc
        published = True
        try:
            validate_capture_stage(
                stage,
                definition_file=definition_path,
                workload_file=workload_path,
                solution_file=solution_path,
                config_file=config_path,
                adapter_executable=adapter_executable,
                source_workload_index=source_workload_index,
                environment_contract=certified_environment,
                flashinfer_corpus_contract=verified_corpus,
            )
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return {"status": "prepared", "manifest": manifest}
    finally:
        if not published:
            shutil.rmtree(tmp, ignore_errors=True)


def _selection_manifest(
    *,
    stage: Path,
    prepare_manifest: dict[str, Any],
    source_workload_index: int,
    capture_kind: str,
    result_manifest_relative: str,
) -> dict[str, Any]:
    prepare_path = stage / PREPARE_MANIFEST_NAME
    artifacts = prepare_manifest["artifacts"]
    return {
        "schema_version": 1,
        "manifest_kind": "sol_execbench_capture_selection",
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "capture_kind": capture_kind,
        "path_mode": "stage_relative_v1",
        "stage_root": ".",
        "prepare_manifest": {
            "path": PREPARE_MANIFEST_NAME,
            "sha256": sha256_file(prepare_path),
        },
        "source_workload_index": source_workload_index,
        "staged_workload_index": 0,
        "workload": prepare_manifest["selection"],
        "definition": prepare_manifest["solution_contract"]["definition"],
        "solution": prepare_manifest["solution_contract"]["solution"],
        "environment_contract": prepare_manifest["environment_contract"],
        "environment_contract_sha256": prepare_manifest[
            "environment_contract_sha256"
        ],
        "flashinfer_corpus_contract": prepare_manifest[
            "flashinfer_corpus_contract"
        ],
        "flashinfer_corpus_contract_sha256": prepare_manifest[
            "flashinfer_corpus_contract_sha256"
        ],
        "stage_definition_sha256": prepare_manifest["stage_definition_sha256"],
        "benchmark_kernel_so": artifacts["benchmark_kernel_so"],
        "eval_driver": artifacts["eval_driver"],
        "capture_runtime": artifacts["capture_runtime"],
        "result_manifest": result_manifest_relative,
    }


def _validate_capture_result(
    path: Path,
    *,
    stage: Path,
    selection_path: Path,
    selection_relative: str,
    selection_sha256: str,
) -> dict[str, Any]:
    result = _load_json_object(path, code="capture_result_invalid")
    if (
        result.get("schema_version") != 1
        or result.get("capture_contract_version") != CAPTURE_CONTRACT_VERSION
        or result.get("manifest_kind") != "sol_execbench_capture_result"
        or result.get("path_mode") != "stage_relative_v1"
        or result.get("stage_root") != "."
        or "stage_dir" in result
    ):
        raise CaptureAdapterError("capture_result_invalid", "wrong result version/kind")
    selection_pin = result.get("selection_manifest")
    if (
        not isinstance(selection_pin, dict)
        or selection_pin.get("path") != selection_relative
    ):
        raise CaptureAdapterError("capture_result_invalid", "selection path mismatch")
    if selection_pin.get("sha256") != selection_sha256:
        raise CaptureAdapterError("capture_result_invalid", "selection hash mismatch")
    selection = _load_json_object(selection_path, code="capture_result_invalid")
    if (
        selection.get("path_mode") != "stage_relative_v1"
        or selection.get("stage_root") != "."
        or "stage_dir" in selection
    ):
        raise CaptureAdapterError(
            "capture_result_invalid", "selection path contract is not relocatable"
        )
    result_relative = path.resolve().relative_to(stage).as_posix()
    if selection.get("result_manifest") != result_relative:
        raise CaptureAdapterError("capture_result_invalid", "result path mismatch")
    prepare_pin = selection.get("prepare_manifest")
    prepare_path = stage / PREPARE_MANIFEST_NAME
    if (
        not isinstance(prepare_pin, dict)
        or prepare_pin.get("path") != PREPARE_MANIFEST_NAME
        or prepare_pin.get("sha256") != sha256_file(prepare_path)
    ):
        raise CaptureAdapterError("capture_result_invalid", "selection preparation pin mismatch")
    prepare = _load_json_object(prepare_path, code="capture_result_invalid")
    for key in ("benchmark_kernel_so", "eval_driver", "capture_runtime"):
        if selection.get(key) != (prepare.get("artifacts") or {}).get(key):
            raise CaptureAdapterError(
                "capture_result_invalid", f"selection {key} pin mismatch"
            )
    for key in (
        "environment_contract",
        "environment_contract_sha256",
        "flashinfer_corpus_contract",
        "flashinfer_corpus_contract_sha256",
        "stage_definition_sha256",
    ):
        if selection.get(key) != prepare.get(key):
            raise CaptureAdapterError(
                "capture_result_invalid", f"selection {key} pin mismatch"
            )
    if result.get("prepare_manifest") != selection.get("prepare_manifest"):
        raise CaptureAdapterError("capture_result_invalid", "preparation pin mismatch")
    exact_fields = {
        "capture_kind": selection.get("capture_kind"),
        "definition": selection.get("definition"),
        "solution": selection.get("solution"),
        "source_workload_index": selection.get("source_workload_index"),
        "staged_workload_index": 0,
        "workload_uuid": (selection.get("workload") or {}).get("workload_uuid"),
        "environment_contract": selection.get("environment_contract"),
        "environment_contract_sha256": selection.get(
            "environment_contract_sha256"
        ),
        "flashinfer_corpus_contract": selection.get(
            "flashinfer_corpus_contract"
        ),
        "flashinfer_corpus_contract_sha256": selection.get(
            "flashinfer_corpus_contract_sha256"
        ),
        "stage_definition_sha256": selection.get("stage_definition_sha256"),
        "benchmark_kernel_so": selection.get("benchmark_kernel_so"),
    }
    if any(result.get(key) != value for key, value in exact_fields.items()):
        raise CaptureAdapterError("capture_result_invalid", "capture identity mismatch")
    status = result.get("status")
    if not isinstance(status, str) or not status:
        raise CaptureAdapterError("capture_result_invalid", "status is absent")
    if status == "captured":
        roi = result.get("roi")
        expected_roi = {
            "profiler_start_calls": 1,
            "candidate_user_fn_calls": 1,
            "profiler_stop_calls": 1,
            "timing_methodology": "cuda_events",
        }
        outside = result.get("outside_roi")
        expected_outside = {
            "correctness_rounds": 1,
            "correctness_user_fn_calls": 1,
            "warmup_user_fn_calls": 1,
            "all_completed_before_profiler_start": True,
        }
        if not isinstance(roi, dict) or any(
            roi.get(key) != value for key, value in expected_roi.items()
        ):
            raise CaptureAdapterError("capture_result_invalid", "successful ROI counts mismatch")
        if not isinstance(outside, dict) or any(
            outside.get(key) != value for key, value in expected_outside.items()
        ):
            raise CaptureAdapterError(
                "capture_result_invalid", "successful outside-ROI counts mismatch"
            )
        if not isinstance(result.get("correctness"), dict):
            raise CaptureAdapterError(
                "capture_result_invalid", "successful correctness evidence is absent"
            )
    return result


def execute_capture_stage(
    *,
    definition_file: Path,
    workload_file: Path,
    solution_file: Path,
    staging_dir: Path,
    source_workload_index: int,
    capture_kind: str,
    selection_manifest: Path,
    result_manifest: Path,
    config_file: Path | None = None,
    adapter_executable: Path | None = None,
    environment_contract: Mapping[str, str] | None = None,
    flashinfer_corpus_contract: Mapping[str, Any] | None = None,
    timeout: int = 600,
    environ: Mapping[str, str] | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Execute the staged driver once and validate its typed result manifest."""
    if capture_kind not in _CAPTURE_KINDS:
        raise CaptureAdapterError(
            "capture_kind_invalid", f"capture_kind must be one of {sorted(_CAPTURE_KINDS)}"
        )
    stage = staging_dir.expanduser().resolve()
    selection_path, selection_relative = _stage_evidence_path(
        stage, selection_manifest, label="selection_manifest"
    )
    result_path, result_relative = _stage_evidence_path(
        stage, result_manifest, label="result_manifest"
    )
    if selection_path == result_path:
        raise CaptureAdapterError(
            "capture_evidence_path_invalid", "selection and result paths must differ"
        )
    if timeout <= 0:
        raise CaptureAdapterError("capture_timeout_invalid", "timeout must be positive")
    if result_path.exists():
        raise CaptureAdapterError("capture_output_exists", str(result_path))
    certified_environment, _ = _environment_contract(environment_contract)
    verified_corpus, _verified_corpus_sha256, flashinfer_trace_dir = (
        _flashinfer_contract(flashinfer_corpus_contract)
    )
    prepare = validate_capture_stage(
        stage,
        definition_file=definition_file,
        workload_file=workload_file,
        solution_file=solution_file,
        config_file=config_file,
        adapter_executable=adapter_executable,
        source_workload_index=source_workload_index,
        environment_contract=certified_environment,
        flashinfer_corpus_contract=verified_corpus,
    )
    selection = _selection_manifest(
        stage=stage,
        prepare_manifest=prepare,
        source_workload_index=source_workload_index,
        capture_kind=capture_kind,
        result_manifest_relative=result_relative,
    )
    selection_sha = sha256_bytes(_canonical_json_bytes(selection))
    execution_transport = {
            CAPTURE_MODE_ENV: "1",
            "SOL_EXECBENCH_CAPTURE_PREPARE_MANIFEST": str(
                stage / PREPARE_MANIFEST_NAME
            ),
            "SOL_EXECBENCH_CAPTURE_SELECTION_MANIFEST": str(selection_path),
            "SOL_EXECBENCH_CAPTURE_SELECTION_SHA256": selection_sha,
            "SOL_EXECBENCH_CAPTURE_RESULT_MANIFEST": str(result_path),
            "SOL_EXECBENCH_CAPTURE_SOURCE_WORKLOAD_INDEX": str(
                source_workload_index
            ),
        }
    env = _certified_subprocess_env(
        certified_environment, environ, transport=execution_transport,
        flashinfer_trace_dir=flashinfer_trace_dir,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_new(selection_path, selection)
    if sha256_file(selection_path) != selection_sha:
        raise CaptureAdapterError(
            "capture_artifact_hash_mismatch", "selection manifest publication drift"
        )
    try:
        proc = run(
            [sys.executable, str(stage / "eval_driver.py")],
            cwd=stage,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CaptureAdapterError("capture_execution_failed", str(exc)) from exc
    if not result_path.is_file():
        detail = (proc.stderr or proc.stdout or "no subprocess output").strip()
        raise CaptureAdapterError(
            "capture_result_absent", f"returncode={proc.returncode}: {detail[-4000:]}"
        )
    result = _validate_capture_result(
        result_path,
        stage=stage,
        selection_path=selection_path,
        selection_relative=selection_relative,
        selection_sha256=selection_sha,
    )
    if (result.get("status") == "captured") != (proc.returncode == 0):
        raise CaptureAdapterError(
            "capture_result_invalid",
            f"status={result.get('status')!r} returncode={proc.returncode}",
        )
    return {
        "status": result.get("status"),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "selection": selection,
        "result": result,
    }


def validate_capture_evidence(
    staging_dir: Path,
    *,
    selection_manifest: Path,
    result_manifest: Path,
) -> dict[str, Any]:
    """Re-verify preparation, selection, and result after a parent rename."""
    stage = staging_dir.expanduser().resolve()
    prepare = validate_capture_stage(stage)
    selection_path, selection_relative = _stage_evidence_path(
        stage, selection_manifest, label="selection_manifest"
    )
    result_path, _ = _stage_evidence_path(
        stage, result_manifest, label="result_manifest"
    )
    if not selection_path.is_file() or not result_path.is_file():
        raise CaptureAdapterError(
            "capture_evidence_absent",
            f"selection={selection_path.is_file()} result={result_path.is_file()}",
        )
    selection_sha = sha256_file(selection_path)
    result = _validate_capture_result(
        result_path,
        stage=stage,
        selection_path=selection_path,
        selection_relative=selection_relative,
        selection_sha256=selection_sha,
    )
    return {
        "preparation": prepare,
        "selection": _load_json_object(
            selection_path, code="capture_result_invalid"
        ),
        "result": result,
    }


def load_capture_request(path: Path) -> dict[str, Any]:
    if not path.expanduser().is_absolute():
        raise CaptureAdapterError(
            "capture_request_invalid", "capture request path must be absolute"
        )
    request_path = _source_file(path, label="capture request")
    request = _load_json_object(request_path, code="capture_request_invalid")
    if request.get("schema_version") != 1:
        raise CaptureAdapterError("capture_request_invalid", "schema_version must be 1")
    action = request.get("action")
    if action not in {"prepare", "execute"}:
        raise CaptureAdapterError(
            "capture_request_invalid", "action must be prepare or execute"
        )
    index = request.get("source_workload_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise CaptureAdapterError(
            "capture_request_invalid", "source_workload_index must be nonnegative"
        )
    stage = request.get("stage_dir")
    if not isinstance(stage, str) or not stage or not Path(stage).is_absolute():
        raise CaptureAdapterError(
            "capture_request_invalid", "stage_dir must be an absolute path"
        )
    base_fields = {
        "schema_version",
        "action",
        "stage_dir",
        "source_workload_index",
        "environment_contract",
        "environment_contract_sha256",
        "flashinfer_corpus_contract",
        "flashinfer_corpus_contract_sha256",
    }
    certified_environment, certified_environment_sha256 = _environment_contract(
        request.get("environment_contract")
    )
    if request.get("environment_contract_sha256") != certified_environment_sha256:
        raise CaptureAdapterError(
            "capture_request_invalid",
            "environment_contract_sha256 does not rederive from the request contract",
        )
    request["environment_contract"] = certified_environment
    corpus_contract, corpus_contract_sha256, _runtime_root = (
        _flashinfer_contract(request.get("flashinfer_corpus_contract"))
    )
    if request.get("flashinfer_corpus_contract_sha256") != corpus_contract_sha256:
        raise CaptureAdapterError(
            "capture_request_invalid",
            "flashinfer_corpus_contract_sha256 does not rederive",
        )
    request["flashinfer_corpus_contract"] = corpus_contract
    if action == "prepare":
        if set(request) != base_fields | {"compile_timeout_s"}:
            raise CaptureAdapterError(
                "capture_request_invalid", "prepare request has wrong fields"
            )
        compile_timeout_s = request.get("compile_timeout_s")
        if type(compile_timeout_s) is not int or compile_timeout_s < 1:
            raise CaptureAdapterError(
                "capture_request_invalid", "compile_timeout_s must be positive"
            )
    if action == "execute":
        execute_fields = base_fields | {
            "capture_kind", "selection_manifest", "result_manifest",
            "execution_timeout_s",
        }
        if set(request) != execute_fields:
            raise CaptureAdapterError(
                "capture_request_invalid", "execute request has wrong fields"
            )
        execution_timeout_s = request.get("execution_timeout_s")
        if type(execution_timeout_s) is not int or execution_timeout_s < 1:
            raise CaptureAdapterError(
                "capture_request_invalid", "execution_timeout_s must be positive"
            )
        for key in ("selection_manifest", "result_manifest"):
            value = request.get(key)
            evidence_path = Path(value) if isinstance(value, str) else Path(".")
            if (
                not isinstance(value, str)
                or not value
                or evidence_path.is_absolute()
                or ".." in evidence_path.parts
                or evidence_path == Path(".")
            ):
                raise CaptureAdapterError(
                    "capture_request_invalid",
                    f"{key} must be a safe stage-relative path",
                )
        if request.get("capture_kind") not in _CAPTURE_KINDS:
            raise CaptureAdapterError("capture_request_invalid", "invalid capture_kind")
    return request


def handle_capture_request(
    *,
    request_file: Path,
    definition_file: Path,
    workload_file: Path,
    solution_file: Path,
    config_file: Path | None,
    compile_timeout: int,
    timeout: int,
    adapter_executable: Path,
    environ: Mapping[str, str] | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Dispatch a request while keeping the matrix application argv unchanged."""
    request = load_capture_request(request_file)
    common = {
        "definition_file": definition_file,
        "workload_file": workload_file,
        "solution_file": solution_file,
        "staging_dir": Path(request["stage_dir"]),
        "source_workload_index": request["source_workload_index"],
    }
    if request["action"] == "prepare":
        return {
            "action": "prepare",
            **prepare_capture_stage(
                **common,
                config_file=config_file,
                compile_timeout=request["compile_timeout_s"],
                adapter_executable=adapter_executable,
                environment_contract=request["environment_contract"],
                flashinfer_corpus_contract=request[
                    "flashinfer_corpus_contract"
                ],
                environ=environ,
                run=run,
            ),
        }
    return {
        "action": "execute",
        **execute_capture_stage(
            **common,
            capture_kind=request["capture_kind"],
            selection_manifest=Path(request["selection_manifest"]),
            result_manifest=Path(request["result_manifest"]),
            config_file=config_file,
            adapter_executable=adapter_executable,
            environment_contract=request["environment_contract"],
            flashinfer_corpus_contract=request["flashinfer_corpus_contract"],
            timeout=request["execution_timeout_s"],
            environ=environ,
            run=run,
        ),
    }


__all__ = [
    "CAPTURE_REQUEST_ENV",
    "CaptureAdapterError",
    "build_certified_env",
    "build_legacy_certified_env",
    "environment_contract_sha256",
    "legacy_environment_contract_sha256",
    "execute_capture_stage",
    "handle_capture_request",
    "load_capture_request",
    "prepare_capture_stage",
    "sha256_file",
    "validate_capture_evidence",
    "validate_capture_stage",
]
