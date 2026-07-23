# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only construction of the certified SOL compiler environment.

This file intentionally has no package imports.  Campaign tooling can load it
directly by absolute path without importing ``sol_execbench.driver`` (whose
normal package initialization requires the SOL virtual environment).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path


PINNED_CUDA_HOME = "/usr/local/cuda-13.2"
"""CUDA toolkit root whose ``bin/nvcc`` is certified for SOL builds."""

PINNED_NVCC_CCBIN = "g++-14"
"""Literal compiler spelling certified by the sm_86 SOL evidence."""

PINNED_CC = "gcc-14"
PINNED_CXX = "g++-14"
PINNED_CUDAHOSTCXX = "g++-14"

LEGACY_CERTIFIED_ENVIRONMENT_CONTRACT = {
    "NVCC_CCBIN": PINNED_NVCC_CCBIN,
}
"""Closed compiler contract carried by the frozen canonical v1 matrix."""

CERTIFIED_ENVIRONMENT_CONTRACT = {
    "CUDA_HOME": PINNED_CUDA_HOME,
    "NVCC_CCBIN": PINNED_NVCC_CCBIN,
    "CC": PINNED_CC,
    "CXX": PINNED_CXX,
    "CUDAHOSTCXX": PINNED_CUDAHOSTCXX,
}
"""Closed compiler/toolkit contract copied into every capture manifest."""

_STRIPPED_NVCC_VARIABLES = (
    *CERTIFIED_ENVIRONMENT_CONTRACT,
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
)
_LEGACY_STRIPPED_NVCC_VARIABLES = (
    *LEGACY_CERTIFIED_ENVIRONMENT_CONTRACT,
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
)


class CertifiedEnvironmentError(ValueError):
    """A typed refusal to construct the certified compiler environment."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _validated_environment_contract(contract: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(contract, Mapping):
        raise CertifiedEnvironmentError(
            "SOL_ENV_CONTRACT_INVALID", "environment contract must be a mapping"
        )
    normalized = dict(contract)
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in normalized.items()
    ):
        raise CertifiedEnvironmentError(
            "SOL_ENV_CONTRACT_INVALID",
            "environment contract keys and values must be strings",
        )
    if normalized != CERTIFIED_ENVIRONMENT_CONTRACT:
        raise CertifiedEnvironmentError(
            "SOL_ENV_CONTRACT_INVALID",
            f"expected exact environment contract {CERTIFIED_ENVIRONMENT_CONTRACT!r}",
        )
    return normalized


def _validated_legacy_environment_contract(
    contract: Mapping[str, str],
) -> dict[str, str]:
    """Validate the frozen v1 contract without weakening the live v2 API."""

    if not isinstance(contract, Mapping):
        raise CertifiedEnvironmentError(
            "SOL_ENV_CONTRACT_INVALID", "environment contract must be a mapping"
        )
    normalized = dict(contract)
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in normalized.items()
    ):
        raise CertifiedEnvironmentError(
            "SOL_ENV_CONTRACT_INVALID",
            "environment contract keys and values must be strings",
        )
    if normalized != LEGACY_CERTIFIED_ENVIRONMENT_CONTRACT:
        raise CertifiedEnvironmentError(
            "SOL_ENV_CONTRACT_INVALID",
            "expected exact legacy environment contract "
            f"{LEGACY_CERTIFIED_ENVIRONMENT_CONTRACT!r}",
        )
    return normalized


def _contract_sha256(normalized: Mapping[str, str]) -> str:
    encoded = (
        json.dumps(
            dict(normalized),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def environment_contract_sha256(contract: Mapping[str, str]) -> str:
    """Hash the exact live-v2 certifying environment contract canonically."""

    normalized = _validated_environment_contract(contract)
    return _contract_sha256(normalized)


def legacy_environment_contract_sha256(contract: Mapping[str, str]) -> str:
    """Hash only the exact frozen-v1 certifying contract."""

    normalized = _validated_legacy_environment_contract(contract)
    return _contract_sha256(normalized)


def _verified_trace_dir(
    flashinfer_trace_dir: Path | str | None,
) -> str | None:
    if flashinfer_trace_dir is None:
        return None
    trace_path = Path(flashinfer_trace_dir).expanduser()
    if (
        not trace_path.is_absolute()
        or trace_path.is_symlink()
        or not trace_path.resolve().is_dir()
    ):
        raise CertifiedEnvironmentError(
            "SOL_FIB_CORPUS_INVALID",
            "FlashInfer trace directory must be an absolute real directory",
        )
    return str(trace_path.resolve())


def _ambient_environment(
    ambient: Mapping[str, str] | None,
) -> dict[str, str]:
    source = os.environ if ambient is None else ambient
    if not isinstance(source, Mapping):
        raise CertifiedEnvironmentError(
            "SOL_ENV_AMBIENT_INVALID", "ambient environment must be a mapping"
        )
    constructed = dict(source)
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in constructed.items()
    ):
        raise CertifiedEnvironmentError(
            "SOL_ENV_AMBIENT_INVALID",
            "ambient environment keys and values must be strings",
        )
    return constructed


def build_legacy_certified_env(
    contract: Mapping[str, str],
    ambient: Mapping[str, str] | None = None,
    *,
    flashinfer_trace_dir: Path | str | None = None,
) -> dict[str, str]:
    """Construct the frozen-v1 compiler environment from its exact one-key API.

    This deliberately does not synthesize CUDA_HOME/CC/CXX/CUDAHOSTCXX.  Those
    keys belong to the distinct live-v2 contract, and adding them would change
    the canonical fingerprint before its expected drift can be reported.
    """

    normalized = _validated_legacy_environment_contract(contract)
    constructed = _ambient_environment(ambient)
    pinned = normalized["NVCC_CCBIN"]
    observed = constructed.get("NVCC_CCBIN")
    if observed is not None and observed != pinned:
        raise CertifiedEnvironmentError(
            "SOL_ENV_AMBIENT_CONFLICT",
            f"ambient NVCC_CCBIN={observed!r} conflicts with pinned {pinned!r}",
        )
    hazards = [
        name
        for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
        if name in constructed
    ]
    if hazards:
        raise CertifiedEnvironmentError(
            "SOL_ENV_HAZARD_CONFLICT",
            f"ambient variables must be unset: {hazards}",
        )
    constructed.pop("FLASHINFER_TRACE_DIR", None)
    verified_trace_dir = _verified_trace_dir(flashinfer_trace_dir)
    for name in _LEGACY_STRIPPED_NVCC_VARIABLES:
        constructed.pop(name, None)
    constructed.update(normalized)
    if verified_trace_dir is not None:
        constructed["FLASHINFER_TRACE_DIR"] = verified_trace_dir
    return constructed


def build_certified_env(
    contract: Mapping[str, str],
    ambient: Mapping[str, str] | None = None,
    *,
    flashinfer_trace_dir: Path | str | None = None,
) -> dict[str, str]:
    """Return a fail-closed subprocess environment for certified SOL capture.

    ``ambient`` is accepted only after the caller has performed its operator
    admission checks.  This function repeats the material conflict checks so a
    direct adapter caller cannot silently override an exported compiler or
    inject NVCC flags.  The returned dictionary is the one that must be passed
    unchanged to the fingerprint probe or compilation subprocess.
    """

    normalized = _validated_environment_contract(contract)
    constructed = _ambient_environment(ambient)

    for name, pinned in normalized.items():
        observed = constructed.get(name)
        if observed is None:
            continue
        admitted = observed == pinned
        if name in {"CC", "CXX", "CUDAHOSTCXX"}:
            admitted = Path(observed).name == pinned
        if not admitted:
            raise CertifiedEnvironmentError(
                "SOL_ENV_AMBIENT_CONFLICT",
                f"ambient {name}={observed!r} conflicts with pinned {pinned!r}",
            )
    hazards = [
        name
        for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
        if name in constructed
    ]
    if hazards:
        raise CertifiedEnvironmentError(
            "SOL_ENV_HAZARD_CONFLICT",
            f"ambient variables must be unset: {hazards}",
        )

    # Ambient trace roots are never admitted.  A caller may inject only the
    # absolute corpus root returned by the byte-verifying FIB contract gate.
    constructed.pop("FLASHINFER_TRACE_DIR", None)
    verified_trace_dir = _verified_trace_dir(flashinfer_trace_dir)

    for name in _STRIPPED_NVCC_VARIABLES:
        constructed.pop(name, None)
    constructed.update(normalized)
    cuda_bin = str(Path(normalized["CUDA_HOME"]) / "bin")
    prior_path = [
        entry
        for entry in constructed.get("PATH", "").split(os.pathsep)
        if entry and Path(entry).resolve() != Path(cuda_bin).resolve()
    ]
    constructed["PATH"] = os.pathsep.join([cuda_bin, *prior_path])
    if verified_trace_dir is not None:
        constructed["FLASHINFER_TRACE_DIR"] = verified_trace_dir
    return constructed


__all__ = [
    "CertifiedEnvironmentError",
    "CERTIFIED_ENVIRONMENT_CONTRACT",
    "LEGACY_CERTIFIED_ENVIRONMENT_CONTRACT",
    "PINNED_CC",
    "PINNED_CUDA_HOME",
    "PINNED_CUDAHOSTCXX",
    "PINNED_CXX",
    "PINNED_NVCC_CCBIN",
    "build_certified_env",
    "build_legacy_certified_env",
    "environment_contract_sha256",
    "legacy_environment_contract_sha256",
]
