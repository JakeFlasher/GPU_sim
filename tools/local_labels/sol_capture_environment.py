#!/usr/bin/env python3
"""Certified SOL capture environment shared by preflight and chunk capture.

The sm_86 SOL evidence certifies one closed CUDA/GCC environment.  Ambient
shell variables are admission inputs only: a conflicting operator override
refuses capture, while an admitted run constructs one exact environment from
the matrix pin for both fingerprinting and compilation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from tools.local_labels import capture_toolchain_contract as toolchain_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
CERTIFIED_ENV_HELPER = (
    REPO_ROOT / "tools" / "sol_execbench" / "src" / "sol_execbench"
    / "driver" / "certified_env.py"
)
PINNED_COMPILER_ENVIRONMENT = {
    "CUDA_HOME": "/usr/local/cuda-13.2",
    "NVCC_CCBIN": "g++-14",
    "CC": "gcc-14",
    "CXX": "g++-14",
    "CUDAHOSTCXX": "g++-14",
}
LEGACY_PINNED_COMPILER_ENVIRONMENT = {"NVCC_CCBIN": "g++-14"}
CERTIFYING_CONTRACT_KEYS = frozenset(PINNED_COMPILER_ENVIRONMENT)
LEGACY_CERTIFYING_CONTRACT_KEYS = frozenset(
    LEGACY_PINNED_COMPILER_ENVIRONMENT
)
LEGACY_MATRIX_ENVIRONMENT = {
    "CUTLASS_DIR": "third_party/cutlass",
    "PATH_prepend": "tools/sol_execbench/.venv/bin",
    **LEGACY_PINNED_COMPILER_ENVIRONMENT,
}
LEGACY_MATRIX_ENVIRONMENT_KEYS = frozenset(LEGACY_MATRIX_ENVIRONMENT)
LIVE_MATRIX_ENVIRONMENT = {
    "CUTLASS_DIR": "third_party/cutlass",
    "PATH_prepend": "tools/sol_execbench/.venv/bin",
    **PINNED_COMPILER_ENVIRONMENT,
}
MATRIX_ENVIRONMENT_KEYS = frozenset({
    "CUTLASS_DIR", "PATH_prepend", *CERTIFYING_CONTRACT_KEYS,
})
HAZARD_VARIABLES = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
CAPTURE_CONTROL_VARIABLES = (
    "SOL_EXECBENCH_CAPTURE_REQUEST",
    "SOL_EXECBENCH_CAPTURE_MODE",
    "SOL_EXECBENCH_CAPTURE_PREPARE_MANIFEST",
    "SOL_EXECBENCH_CAPTURE_SELECTION_MANIFEST",
    "SOL_EXECBENCH_CAPTURE_SELECTION_SHA256",
    "SOL_EXECBENCH_CAPTURE_RESULT_MANIFEST",
    "SOL_EXECBENCH_CAPTURE_SOURCE_WORKLOAD_INDEX",
)
COMPILER_SNAPSHOT_VARIABLES = ("CUDA_HOME", "CC", "CXX", "CUDAHOSTCXX")
AMBIENT_VARIABLES = (
    "NVCC_CCBIN", *HAZARD_VARIABLES, *COMPILER_SNAPSHOT_VARIABLES,
)
AMBIENT_CONFLICT = "SOL_ENV_AMBIENT_CONFLICT"
SUPPLEMENTAL_MATRIX_ARTIFACT_ID = "sm86_sol_workload_capture_matrix_live_v2"


class SolEnvironmentRefusal(RuntimeError):
    """A typed pre-GPU refusal for environment/provenance drift."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _runtime_api() -> tuple[Callable[..., dict[str, str]], Callable[[Any], str]]:
    """Load the stdlib-only producer helper without importing its package."""
    module_name = "_sol_capture_runtime_certified_env"
    module = sys.modules.get(module_name)
    if module is None:
        if not CERTIFIED_ENV_HELPER.is_file():
            raise SolEnvironmentRefusal(
                "SOL_ENV_HELPER_MISSING", str(CERTIFIED_ENV_HELPER)
            )
        spec = importlib.util.spec_from_file_location(
            module_name, CERTIFIED_ENV_HELPER
        )
        if spec is None or spec.loader is None:
            raise SolEnvironmentRefusal(
                "SOL_ENV_HELPER_MISSING",
                f"cannot load {CERTIFIED_ENV_HELPER}",
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    build = getattr(module, "build_certified_env", None)
    digest = getattr(module, "environment_contract_sha256", None)
    if not callable(build) or not callable(digest):
        raise SolEnvironmentRefusal(
            "SOL_ENV_HELPER_INVALID", "certified environment API is absent"
        )
    return build, digest


def _legacy_runtime_api(
) -> tuple[Callable[..., dict[str, str]], Callable[[Any], str]]:
    """Load the distinct frozen-v1 producer API, never the live-v2 API."""

    _live_build, _live_digest = _runtime_api()
    del _live_build, _live_digest
    module = sys.modules.get("_sol_capture_runtime_certified_env")
    build = getattr(module, "build_legacy_certified_env", None)
    digest = getattr(module, "legacy_environment_contract_sha256", None)
    if not callable(build) or not callable(digest):
        raise SolEnvironmentRefusal(
            "SOL_ENV_HELPER_INVALID", "legacy certified environment API is absent"
        )
    return build, digest


def matrix_environment_contract(matrix: dict) -> dict[str, str]:
    contracts: dict[str, dict[str, str]] = {}
    for cell in matrix.get("cells") or []:
        if not isinstance(cell, dict) or cell.get("disposition") != "runnable":
            continue
        identity = cell.get("identity") or {}
        if identity.get("roster_source") != "sol_execbench":
            continue
        invocation = cell.get("invocation") or {}
        contract = invocation.get("adapter_contract") or {}
        environment = contract.get("environment")
        if (
            not isinstance(environment, dict)
            or set(environment) != LEGACY_MATRIX_ENVIRONMENT_KEYS
            or any(not isinstance(environment.get(key), str) or not environment[key]
                   for key in LEGACY_MATRIX_ENVIRONMENT_KEYS)
        ):
            raise SolEnvironmentRefusal(
                "SOL_ENV_MATRIX_CONTRACT_INVALID", repr(identity)
            )
        contracts[canonical_sha256(environment)] = dict(environment)
    if len(contracts) != 1:
        raise SolEnvironmentRefusal(
            "SOL_ENV_MATRIX_CONTRACT_MIXED", repr(sorted(contracts))
        )
    environment = next(iter(contracts.values()))
    if environment != LEGACY_MATRIX_ENVIRONMENT:
        raise SolEnvironmentRefusal(
            "SOL_ENV_MATRIX_CONTRACT_INVALID",
            repr(environment),
        )
    return environment


def supplemental_matrix_environment_contract(
    matrix: dict,
) -> dict[str, str]:
    """Extract the one exact live contract from supplemental variants only."""
    if (
        not isinstance(matrix, dict)
        or matrix.get("schema_version") != 2
        or matrix.get("artifact_id") != SUPPLEMENTAL_MATRIX_ARTIFACT_ID
    ):
        raise SolEnvironmentRefusal(
            "SOL_ENV_SUPPLEMENTAL_MATRIX_INVALID", "matrix identity"
        )
    variants = matrix.get("variants")
    if not isinstance(variants, list) or not variants:
        raise SolEnvironmentRefusal(
            "SOL_ENV_SUPPLEMENTAL_MATRIX_INVALID", "variants"
        )
    contracts: dict[str, dict[str, str]] = {}
    for index, variant in enumerate(variants):
        adapter = variant.get("adapter_contract") if isinstance(variant, dict) else None
        invocation = variant.get("invocation") if isinstance(variant, dict) else None
        invocation_adapter = (
            invocation.get("adapter_contract")
            if isinstance(invocation, dict) else None
        )
        environment = adapter.get("environment") if isinstance(adapter, dict) else None
        if (
            not isinstance(environment, dict)
            or set(environment) != MATRIX_ENVIRONMENT_KEYS
            or any(not isinstance(environment.get(key), str) or not environment[key]
                   for key in MATRIX_ENVIRONMENT_KEYS)
            or invocation_adapter != adapter
        ):
            raise SolEnvironmentRefusal(
                "SOL_ENV_SUPPLEMENTAL_MATRIX_INVALID", f"variants[{index}]"
            )
        contracts[canonical_sha256(environment)] = dict(environment)
    if len(contracts) != 1:
        raise SolEnvironmentRefusal(
            "SOL_ENV_SUPPLEMENTAL_MATRIX_MIXED", repr(sorted(contracts))
        )
    environment = next(iter(contracts.values()))
    if environment != LIVE_MATRIX_ENVIRONMENT:
        raise SolEnvironmentRefusal(
            "SOL_ENV_SUPPLEMENTAL_MATRIX_INVALID", repr(environment)
        )
    certifying_environment_contract(environment)
    return environment


def legacy_certifying_environment_contract(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Project only the exact frozen canonical three-key adapter contract."""

    if not isinstance(environment, Mapping) or dict(environment) != (
        LEGACY_MATRIX_ENVIRONMENT
    ):
        raise SolEnvironmentRefusal(
            "SOL_ENV_MATRIX_CONTRACT_INVALID", repr(environment)
        )
    return dict(LEGACY_PINNED_COMPILER_ENVIRONMENT)


def certifying_environment_contract(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Project only the exact live-v2 seven-key adapter contract."""
    if not isinstance(environment, Mapping) or dict(environment) != (
        LIVE_MATRIX_ENVIRONMENT
    ):
        raise SolEnvironmentRefusal(
            "SOL_ENV_MATRIX_CONTRACT_INVALID", repr(environment)
        )
    return dict(PINNED_COMPILER_ENVIRONMENT)


def environment_contract_sha256(contract: Mapping[str, str]) -> str:
    """Return the producer's canonical digest for an exact certifying contract."""
    _build, digest = _runtime_api()
    try:
        return digest(contract)
    except Exception as exc:
        code = getattr(exc, "code", "SOL_ENV_CONTRACT_INVALID")
        detail = getattr(exc, "detail", str(exc))
        raise SolEnvironmentRefusal(code, detail) from exc


def legacy_environment_contract_sha256(contract: Mapping[str, str]) -> str:
    """Return the producer digest for the exact frozen-v1 contract."""

    _build, digest = _legacy_runtime_api()
    try:
        return digest(contract)
    except Exception as exc:
        code = getattr(exc, "code", "SOL_ENV_CONTRACT_INVALID")
        detail = getattr(exc, "detail", str(exc))
        raise SolEnvironmentRefusal(code, detail) from exc


def ambient_snapshot(
    ambient: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    source = os.environ if ambient is None else ambient
    return {
        name: {
            "present": name in source,
            "value": source.get(name) if name in source else None,
        }
        for name in AMBIENT_VARIABLES
    }


def require_ambient_admission(
    environment_contract: Mapping[str, str],
    ambient: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    certifying_environment_contract(environment_contract)
    source = os.environ if ambient is None else ambient
    snapshot = ambient_snapshot(source)
    for name, pinned in PINNED_COMPILER_ENVIRONMENT.items():
        if name not in source:
            continue
        observed = source.get(name)
        admitted = observed == pinned
        if name in {"CC", "CXX", "CUDAHOSTCXX"} and isinstance(observed, str):
            admitted = Path(observed).name == pinned
        if not admitted:
            raise SolEnvironmentRefusal(
                AMBIENT_CONFLICT,
                f"ambient {name}={observed!r}, pinned={pinned!r}; "
                "unset the conflicting variable",
            )
    set_hazards = [name for name in HAZARD_VARIABLES if name in source]
    if set_hazards:
        raise SolEnvironmentRefusal(
            AMBIENT_CONFLICT,
            "hazard variables must be unset: " + ", ".join(set_hazards),
        )
    return snapshot


def require_legacy_ambient_admission(
    environment_contract: Mapping[str, str],
    ambient: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Apply the historical admission policy to the exact v1 matrix shape."""

    legacy_certifying_environment_contract(environment_contract)
    source = os.environ if ambient is None else ambient
    snapshot = ambient_snapshot(source)
    pinned = LEGACY_PINNED_COMPILER_ENVIRONMENT["NVCC_CCBIN"]
    if "NVCC_CCBIN" in source and source.get("NVCC_CCBIN") != pinned:
        raise SolEnvironmentRefusal(
            AMBIENT_CONFLICT,
            f"ambient NVCC_CCBIN={source.get('NVCC_CCBIN')!r}, "
            f"pinned={pinned!r}; unset the conflicting variable",
        )
    set_hazards = [name for name in HAZARD_VARIABLES if name in source]
    if set_hazards:
        raise SolEnvironmentRefusal(
            AMBIENT_CONFLICT,
            "hazard variables must be unset: " + ", ".join(set_hazards),
        )
    expected_compilers = {
        "CC": "gcc-14", "CXX": "g++-14", "CUDAHOSTCXX": "g++-14",
    }
    conflicting_compilers = [
        f"{name}={source[name]!r}"
        for name, expected in expected_compilers.items()
        if name in source and Path(source[name]).name != expected
    ]
    if conflicting_compilers:
        raise SolEnvironmentRefusal(
            AMBIENT_CONFLICT,
            "non-GCC-14 compiler overrides: " + ", ".join(conflicting_compilers),
        )
    return snapshot


def _build_legacy_capture_environment_from_contract(
    environment: Mapping[str, str],
    ambient: Mapping[str, str] | None = None,
) -> dict[str, str]:
    require_legacy_ambient_admission(environment, ambient)
    build_certified_env, _digest = _legacy_runtime_api()
    certifying = legacy_certifying_environment_contract(environment)
    try:
        result = build_certified_env(certifying, ambient=ambient)
    except Exception as exc:
        code = getattr(exc, "code", "SOL_ENV_CONSTRUCTION_FAILED")
        detail = getattr(exc, "detail", str(exc))
        raise SolEnvironmentRefusal(code, detail) from exc
    for name in CAPTURE_CONTROL_VARIABLES:
        result.pop(name, None)
    result["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    cutlass = (REPO_ROOT / environment["CUTLASS_DIR"]).resolve()
    path_prepend = (REPO_ROOT / environment["PATH_prepend"]).resolve()
    if not cutlass.is_dir() or not path_prepend.is_dir():
        raise SolEnvironmentRefusal(
            "SOL_ENV_PATH_MISSING",
            f"CUTLASS_DIR={cutlass}, PATH_prepend={path_prepend}",
        )
    result["CUTLASS_DIR"] = str(cutlass)
    prior = [
        item for item in result.get("PATH", "").split(os.pathsep)
        if item and Path(item).resolve() != path_prepend
    ]
    result["PATH"] = os.pathsep.join([str(path_prepend), *prior])
    if result.get("NVCC_CCBIN") != environment["NVCC_CCBIN"]:
        raise SolEnvironmentRefusal(
            "SOL_ENV_CONSTRUCTION_FAILED", "certifying compiler pin drifted"
        )
    if any(name in result for name in HAZARD_VARIABLES):
        raise SolEnvironmentRefusal(
            "SOL_ENV_CONSTRUCTION_FAILED", "hazard variable survived construction"
        )
    return result


def _build_capture_environment_from_contract(
    environment: Mapping[str, str],
    ambient: Mapping[str, str] | None = None,
) -> dict[str, str]:
    require_ambient_admission(environment, ambient)
    build_certified_env, _digest = _runtime_api()
    certifying = certifying_environment_contract(environment)
    try:
        result = build_certified_env(certifying, ambient=ambient)
    except Exception as exc:
        code = getattr(exc, "code", "SOL_ENV_CONSTRUCTION_FAILED")
        detail = getattr(exc, "detail", str(exc))
        raise SolEnvironmentRefusal(code, detail) from exc
    for name in CAPTURE_CONTROL_VARIABLES:
        result.pop(name, None)
    result["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    cutlass = (REPO_ROOT / environment["CUTLASS_DIR"]).resolve()
    path_prepend = (REPO_ROOT / environment["PATH_prepend"]).resolve()
    if not cutlass.is_dir() or not path_prepend.is_dir():
        raise SolEnvironmentRefusal(
            "SOL_ENV_PATH_MISSING",
            f"CUTLASS_DIR={cutlass}, PATH_prepend={path_prepend}",
        )
    result["CUTLASS_DIR"] = str(cutlass)
    cuda_home = Path(environment["CUDA_HOME"])
    nvcc = cuda_home / "bin" / "nvcc"
    if (
        not cuda_home.is_dir()
        or not nvcc.is_file()
        or not os.access(nvcc, os.X_OK)
    ):
        raise SolEnvironmentRefusal(
            "SOL_ENV_PATH_MISSING", f"CUDA_HOME={cuda_home}, nvcc={nvcc}"
        )
    prior = [
        item for item in result.get("PATH", "").split(os.pathsep)
        if item and Path(item).resolve() not in {
            path_prepend, (cuda_home / "bin").resolve(),
        }
    ]
    result["PATH"] = os.pathsep.join([
        str((cuda_home / "bin").resolve()), str(path_prepend), *prior,
    ])
    if any(
        result.get(key) != value
        for key, value in PINNED_COMPILER_ENVIRONMENT.items()
    ):
        raise SolEnvironmentRefusal(
            "SOL_ENV_CONSTRUCTION_FAILED", "certifying environment drifted"
        )
    if any(name in result for name in HAZARD_VARIABLES):
        raise SolEnvironmentRefusal(
            "SOL_ENV_CONSTRUCTION_FAILED", "hazard variable survived construction"
        )
    return result


def build_capture_environment(
    matrix: dict,
    ambient: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return _build_legacy_capture_environment_from_contract(
        matrix_environment_contract(matrix), ambient,
    )


def build_supplemental_capture_environment(
    matrix: dict,
    ambient: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return _build_capture_environment_from_contract(
        supplemental_matrix_environment_contract(matrix), ambient,
    )


def fingerprint_pin(
    matrix: dict, *, repo_root: Path = REPO_ROOT,
) -> tuple[dict, Path, dict]:
    pin = (matrix.get("input_pins") or {}).get("sol_native_env_fingerprint")
    if not isinstance(pin, dict) or set(pin) != {
        "path", "sha256", "native_env_hash",
    }:
        raise SolEnvironmentRefusal("SOL_ENV_MATRIX_PIN_INVALID", repr(pin))
    raw_path = pin.get("path")
    path = Path(raw_path) if isinstance(raw_path, str) else Path(".")
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise SolEnvironmentRefusal("SOL_ENV_MATRIX_PIN_INVALID", repr(raw_path))
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise SolEnvironmentRefusal("SOL_ENV_MATRIX_PIN_INVALID", str(path)) from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise SolEnvironmentRefusal("SOL_ENV_MATRIX_PIN_MISSING", str(resolved))
    if sha256_file(resolved) != pin.get("sha256"):
        raise SolEnvironmentRefusal("SOL_ENV_MATRIX_PIN_MISMATCH", str(resolved))
    try:
        document = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise SolEnvironmentRefusal("SOL_ENV_MATRIX_PIN_INVALID", str(exc)) from exc
    if (
        not isinstance(document, dict)
        or document.get("native_env_hash") != pin.get("native_env_hash")
        or document.get("degraded") is not False
        or document.get("missing_components") != []
        or not isinstance(document.get("components"), dict)
    ):
        raise SolEnvironmentRefusal("SOL_ENV_MATRIX_PIN_INVALID", str(resolved))
    return dict(pin), resolved, document


def compiler_identity(
    environment: Mapping[str, str], *, requested: str,
) -> tuple[dict[str, str], str]:
    located = shutil.which(requested, path=environment.get("PATH"))
    if not located:
        raise SolEnvironmentRefusal(
            "SOL_ENV_COMPILER_UNRESOLVED", f"{requested!r} is absent from PATH"
        )
    resolved = Path(located).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise SolEnvironmentRefusal(
            "SOL_ENV_COMPILER_UNRESOLVED", str(resolved)
        )
    cuda_home = Path(environment.get("CUDA_HOME", ""))
    expected_nvcc = (cuda_home / "bin" / "nvcc").resolve()
    located_nvcc = shutil.which("nvcc", path=environment.get("PATH"))
    if (
        environment.get("CUDA_HOME") != PINNED_COMPILER_ENVIRONMENT["CUDA_HOME"]
        or located_nvcc is None
        or Path(located_nvcc).resolve() != expected_nvcc
        or not expected_nvcc.is_file()
        or not os.access(expected_nvcc, os.X_OK)
    ):
        raise SolEnvironmentRefusal(
            "SOL_ENV_NVCC_MISMATCH",
            f"expected={expected_nvcc}, observed={located_nvcc!r}",
        )
    try:
        version = subprocess.run(
            [str(resolved), "--version"], env=dict(environment),
            capture_output=True, text=True, timeout=30, check=False,
        )
        nvcc = subprocess.run(
            [str(expected_nvcc), "-V"], env=dict(environment), capture_output=True,
            text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SolEnvironmentRefusal("SOL_ENV_COMPILER_UNRESOLVED", str(exc)) from exc
    version_line = (version.stdout or version.stderr).splitlines()
    if version.returncode != 0 or not version_line:
        raise SolEnvironmentRefusal(
            "SOL_ENV_COMPILER_UNRESOLVED", "g++-14 --version failed"
        )
    nvcc_output = (nvcc.stdout + nvcc.stderr).strip()
    if nvcc.returncode != 0 or not nvcc_output:
        raise SolEnvironmentRefusal(
            "SOL_ENV_COMPILER_UNRESOLVED", "nvcc -V failed"
        )
    return ({
        "requested": requested,
        "resolved_realpath": str(resolved),
        "executable_sha256": sha256_file(resolved),
        "version_line": version_line[0].strip(),
    }, nvcc_output)


def legacy_compiler_identity(
    environment: Mapping[str, str], *, requested: str,
) -> tuple[dict[str, str], str]:
    """Observe the frozen-v1 compiler binding without inventing CUDA_HOME."""

    located = shutil.which(requested, path=environment.get("PATH"))
    if not located:
        raise SolEnvironmentRefusal(
            "SOL_ENV_COMPILER_UNRESOLVED", f"{requested!r} is absent from PATH"
        )
    resolved = Path(located).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise SolEnvironmentRefusal("SOL_ENV_COMPILER_UNRESOLVED", str(resolved))
    located_nvcc = shutil.which("nvcc", path=environment.get("PATH"))
    if located_nvcc is None:
        raise SolEnvironmentRefusal(
            "SOL_ENV_COMPILER_UNRESOLVED", "nvcc is absent from PATH"
        )
    resolved_nvcc = Path(located_nvcc).resolve()
    if not resolved_nvcc.is_file() or not os.access(resolved_nvcc, os.X_OK):
        raise SolEnvironmentRefusal(
            "SOL_ENV_COMPILER_UNRESOLVED", str(resolved_nvcc)
        )
    try:
        version = subprocess.run(
            [str(resolved), "--version"], env=dict(environment),
            capture_output=True, text=True, timeout=30, check=False,
        )
        nvcc = subprocess.run(
            [str(resolved_nvcc), "-V"], env=dict(environment),
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SolEnvironmentRefusal("SOL_ENV_COMPILER_UNRESOLVED", str(exc)) from exc
    version_line = (version.stdout or version.stderr).splitlines()
    if version.returncode != 0 or not version_line:
        raise SolEnvironmentRefusal(
            "SOL_ENV_COMPILER_UNRESOLVED", "g++-14 --version failed"
        )
    nvcc_output = (nvcc.stdout + nvcc.stderr).strip()
    if nvcc.returncode != 0 or not nvcc_output:
        raise SolEnvironmentRefusal(
            "SOL_ENV_COMPILER_UNRESOLVED", "nvcc -V failed"
        )
    return ({
        "requested": requested,
        "resolved_realpath": str(resolved),
        "executable_sha256": sha256_file(resolved),
        "version_line": version_line[0].strip(),
    }, nvcc_output)


def _certify_sol_environment_with(
    matrix: dict,
    *,
    matrix_path: Path,
    ambient: Mapping[str, str] | None = None,
    fingerprint_probe: Callable[[Mapping[str, str]], dict] | None = None,
    repo_root: Path = REPO_ROOT,
    contract_loader: Callable[[dict], dict[str, str]],
    environment_builder: Callable[
        [dict, Mapping[str, str] | None], dict[str, str]
    ],
    ambient_admitter: Callable[
        [Mapping[str, str], Mapping[str, str] | None], dict[str, dict[str, Any]]
    ],
    certifying_contract_loader: Callable[
        [Mapping[str, str]], dict[str, str]
    ],
    contract_digester: Callable[[Mapping[str, str]], str],
    compiler_probe: Callable[..., tuple[dict[str, str], str]],
    legacy_mode: bool,
) -> tuple[dict, dict[str, str]]:
    """Return sealed evidence plus the exact environment that evidence probed."""
    snapshot = ambient_snapshot(ambient)
    contract = contract_loader(matrix)
    ambient_admitter(contract, ambient)
    environment = environment_builder(matrix, ambient)
    pin, _pin_path, certified = fingerprint_pin(matrix, repo_root=repo_root)
    if contract_loader is supplemental_matrix_environment_contract:
        expected_hash = pin["native_env_hash"]
        for index, variant in enumerate(matrix["variants"]):
            adapter = variant["adapter_contract"]
            invocation_adapter = variant["invocation"]["adapter_contract"]
            if (
                variant.get("native_env_hash") != expected_hash
                or adapter.get("native_env_hash") != expected_hash
                or invocation_adapter.get("native_env_hash") != expected_hash
            ):
                raise SolEnvironmentRefusal(
                    "SOL_ENV_SUPPLEMENTAL_MATRIX_INVALID",
                    f"native_env_hash variants[{index}]",
                )
    compiler, nvcc_output = compiler_probe(
        environment, requested=contract["NVCC_CCBIN"]
    )
    try:
        nvcc_release = toolchain_contract.parse_nvcc_release_banner(
            nvcc_output
        )
        if not legacy_mode:
            toolchain_contract.require_sol_cuda_release(nvcc_release)
    except toolchain_contract.CaptureToolchainContractError as exc:
        raise SolEnvironmentRefusal(exc.code, exc.detail) from exc
    if fingerprint_probe is None:
        from tools.local_labels.sol_reconstitution import native_env_fingerprint
        fingerprint_probe = lambda env: native_env_fingerprint(environ=env)
    observed = fingerprint_probe(environment)
    if (
        not isinstance(observed, dict)
        or observed.get("degraded") is not False
        or observed.get("missing") != []
        or observed.get("hash") != pin["native_env_hash"]
        or observed.get("components") != certified["components"]
    ):
        raise SolEnvironmentRefusal(
            "SOL_ENV_FINGERPRINT_MISMATCH", repr(observed)
        )
    certifying = certifying_contract_loader(contract)
    if legacy_mode:
        constructed_environment_delta = {
            "NVCC_CCBIN": environment["NVCC_CCBIN"],
            "CUTLASS_DIR": environment["CUTLASS_DIR"],
            "PATH_prepend": str(
                (repo_root / contract["PATH_prepend"]).resolve()
            ),
            "PYTORCH_ALLOC_CONF": environment["PYTORCH_ALLOC_CONF"],
            "removed_if_present": [
                "NVCC_CCBIN", *HAZARD_VARIABLES,
                *CAPTURE_CONTROL_VARIABLES,
            ],
        }
    else:
        constructed_environment_delta = {
            **{
                key: environment[key]
                for key in PINNED_COMPILER_ENVIRONMENT
            },
            "CUTLASS_DIR": environment["CUTLASS_DIR"],
            "PATH_prepend": str(
                (repo_root / contract["PATH_prepend"]).resolve()
            ),
            "CUDA_bin_prepend": str(
                (Path(contract["CUDA_HOME"]) / "bin").resolve()
            ),
            "PYTORCH_ALLOC_CONF": environment["PYTORCH_ALLOC_CONF"],
            "removed_if_present": [
                *PINNED_COMPILER_ENVIRONMENT, *HAZARD_VARIABLES,
                *CAPTURE_CONTROL_VARIABLES,
            ],
        }
    probe = {
        "status": "ok",
        "typed_error": None,
        "detail": "certified non-degraded SOL native environment restored",
        "ambient": snapshot,
        "environment_contract": contract,
        "environment_contract_sha256": contract_digester(certifying),
        "constructed_environment_delta": constructed_environment_delta,
        "matrix_pin": pin,
        "matrix_sha256": sha256_file(matrix_path),
        "observed_fingerprint": observed,
        "compiler_identity": compiler,
        # Always record the unambiguous two-integer form.  The frozen-v1
        # matrix remains compatible (its observed release is not retroactively
        # pinned), while live-v2 additionally requires exact CUDA 13.2 above.
        "nvcc_release": nvcc_release,
        "nvcc_version_output": nvcc_output,
    }
    return probe, environment


def certify_sol_environment(
    matrix: dict,
    *,
    matrix_path: Path,
    ambient: Mapping[str, str] | None = None,
    fingerprint_probe: Callable[[Mapping[str, str]], dict] | None = None,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict, dict[str, str]]:
    return _certify_sol_environment_with(
        matrix, matrix_path=matrix_path, ambient=ambient,
        fingerprint_probe=fingerprint_probe, repo_root=repo_root,
        contract_loader=matrix_environment_contract,
        environment_builder=build_capture_environment,
        ambient_admitter=require_legacy_ambient_admission,
        certifying_contract_loader=legacy_certifying_environment_contract,
        contract_digester=legacy_environment_contract_sha256,
        compiler_probe=legacy_compiler_identity,
        legacy_mode=True,
    )


def certify_supplemental_sol_environment(
    matrix: dict,
    *,
    matrix_path: Path,
    ambient: Mapping[str, str] | None = None,
    fingerprint_probe: Callable[[Mapping[str, str]], dict] | None = None,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict, dict[str, str]]:
    return _certify_sol_environment_with(
        matrix, matrix_path=matrix_path, ambient=ambient,
        fingerprint_probe=fingerprint_probe, repo_root=repo_root,
        contract_loader=supplemental_matrix_environment_contract,
        environment_builder=build_supplemental_capture_environment,
        ambient_admitter=require_ambient_admission,
        certifying_contract_loader=certifying_environment_contract,
        contract_digester=environment_contract_sha256,
        compiler_probe=compiler_identity,
        legacy_mode=False,
    )


def probe_sol_environment(
    matrix: dict,
    *,
    matrix_path: Path,
    ambient: Mapping[str, str] | None = None,
    fingerprint_probe: Callable[[Mapping[str, str]], dict] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Return JSON-safe blocking evidence without exposing the full environment."""
    snapshot = ambient_snapshot(ambient)
    try:
        probe, _environment = certify_sol_environment(
            matrix,
            matrix_path=matrix_path,
            ambient=ambient,
            fingerprint_probe=fingerprint_probe,
            repo_root=repo_root,
        )
        return probe
    except SolEnvironmentRefusal as exc:
        return {
            "status": "failed",
            "typed_error": exc.code,
            "detail": exc.detail,
            "ambient": snapshot,
        }


def probe_supplemental_sol_environment(
    matrix: dict,
    *,
    matrix_path: Path,
    ambient: Mapping[str, str] | None = None,
    fingerprint_probe: Callable[[Mapping[str, str]], dict] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict:
    snapshot = ambient_snapshot(ambient)
    try:
        probe, _environment = certify_supplemental_sol_environment(
            matrix, matrix_path=matrix_path, ambient=ambient,
            fingerprint_probe=fingerprint_probe, repo_root=repo_root,
        )
        return probe
    except SolEnvironmentRefusal as exc:
        return {
            "status": "failed",
            "typed_error": exc.code,
            "detail": exc.detail,
            "ambient": snapshot,
        }


def build_cell_attestation(
    *, identity: dict, probe: dict,
) -> dict:
    if probe.get("status") != "ok":
        raise SolEnvironmentRefusal(
            "SOL_ENV_ATTESTATION_INVALID", repr(probe.get("typed_error"))
        )
    observed = probe["observed_fingerprint"]
    pin = probe["matrix_pin"]
    payload = {
        "schema_version": 2,
        "kind": "sol_native_env_attestation",
        "identity": identity,
        "certified_native_env_hash": pin["native_env_hash"],
        "observed_native_env_hash": observed["hash"],
        "certified_hash_equal": observed["hash"] == pin["native_env_hash"],
        "degraded": observed["degraded"],
        "missing_components": observed["missing"],
        "components": observed["components"],
        "components_sha256": canonical_sha256(observed["components"]),
        "compiler_identity": probe["compiler_identity"],
        "matrix_pin": pin,
    }
    payload["attestation_sha256"] = canonical_sha256(payload)
    return payload


__all__ = [
    "AMBIENT_CONFLICT",
    "CAPTURE_CONTROL_VARIABLES",
    "SolEnvironmentRefusal",
    "ambient_snapshot",
    "build_capture_environment",
    "build_supplemental_capture_environment",
    "build_cell_attestation",
    "certifying_environment_contract",
    "legacy_certifying_environment_contract",
    "certify_sol_environment",
    "certify_supplemental_sol_environment",
    "compiler_identity",
    "legacy_compiler_identity",
    "environment_contract_sha256",
    "legacy_environment_contract_sha256",
    "fingerprint_pin",
    "matrix_environment_contract",
    "supplemental_matrix_environment_contract",
    "probe_sol_environment",
    "probe_supplemental_sol_environment",
    "require_ambient_admission",
    "require_legacy_ambient_admission",
]
