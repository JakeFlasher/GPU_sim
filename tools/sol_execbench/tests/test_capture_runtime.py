# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sol_execbench.driver.capture_runtime import (
    CertifiedEnvironmentError,
    ProfiledCandidateError,
    build_certified_env,
    environment_contract_sha256,
    profile_candidate_once,
)
from sol_execbench.driver.certified_env import (
    LEGACY_CERTIFIED_ENVIRONMENT_CONTRACT,
    build_legacy_certified_env,
    legacy_environment_contract_sha256,
)


CERTIFIED_ENVIRONMENT = {
    "CUDA_HOME": "/usr/local/cuda-13.2",
    "NVCC_CCBIN": "g++-14",
    "CC": "gcc-14",
    "CXX": "g++-14",
    "CUDAHOSTCXX": "g++-14",
}
LEGACY_CERTIFIED_ENVIRONMENT = {"NVCC_CCBIN": "g++-14"}


def test_certified_environment_module_loads_directly_without_package_import():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/sol_execbench/driver/certified_env.py"
    )
    spec = importlib.util.spec_from_file_location("sol_certified_env_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.build_certified_env(
        CERTIFIED_ENVIRONMENT, {"PATH": "/usr/bin"}
    )["NVCC_CCBIN"] == "g++-14"


def test_certified_environment_is_deterministic_and_preserves_admitted_ambient():
    ambient = {
        "PATH": "/opt/certified/bin:/usr/bin",
        "CUTLASS_DIR": "/opt/cutlass",
        "NVCC_CCBIN": "g++-14",
        "UNRELATED": "preserved",
    }

    first = build_certified_env(CERTIFIED_ENVIRONMENT, ambient)
    second = build_certified_env(dict(CERTIFIED_ENVIRONMENT), dict(ambient))

    assert first == second
    assert first is not ambient
    assert first["NVCC_CCBIN"] == "g++-14"
    assert first["CUDA_HOME"] == "/usr/local/cuda-13.2"
    assert first["CC"] == "gcc-14"
    assert first["CXX"] == first["CUDAHOSTCXX"] == "g++-14"
    assert first["PATH"].split(":")[0] == "/usr/local/cuda-13.2/bin"
    assert first["CUTLASS_DIR"] == "/opt/cutlass"
    assert first["UNRELATED"] == "preserved"
    assert ambient["NVCC_CCBIN"] == "g++-14"
    assert environment_contract_sha256(CERTIFIED_ENVIRONMENT) == (
        "78a57f92da05dad7971cf697e7becf3ba9514ff4a76452cff922db3d600abf0b"
    )


def test_certified_environment_unset_and_exact_pin_are_byte_equivalent():
    without_pin = {"PATH": "/usr/bin", "CUTLASS_DIR": "/cutlass"}
    with_pin = {**without_pin, **CERTIFIED_ENVIRONMENT}

    assert build_certified_env(
        CERTIFIED_ENVIRONMENT, without_pin
    ) == build_certified_env(CERTIFIED_ENVIRONMENT, with_pin)


def test_legacy_environment_api_is_exact_and_disjoint_from_live_contract():
    assert LEGACY_CERTIFIED_ENVIRONMENT_CONTRACT == (
        LEGACY_CERTIFIED_ENVIRONMENT
    )
    ambient = {"PATH": "/usr/bin", "UNRELATED": "preserved"}
    built = build_legacy_certified_env(LEGACY_CERTIFIED_ENVIRONMENT, ambient)
    assert built == {
        "PATH": "/usr/bin",
        "UNRELATED": "preserved",
        "NVCC_CCBIN": "g++-14",
    }
    assert legacy_environment_contract_sha256(
        LEGACY_CERTIFIED_ENVIRONMENT
    ) == "d959a1a89d2f3aa3e7c351ba59788c016404bbc77e9da428123188ada775af74"

    with pytest.raises(CertifiedEnvironmentError) as full_rejects_legacy:
        build_certified_env(LEGACY_CERTIFIED_ENVIRONMENT, ambient)
    assert full_rejects_legacy.value.code == "SOL_ENV_CONTRACT_INVALID"
    with pytest.raises(CertifiedEnvironmentError) as legacy_rejects_full:
        build_legacy_certified_env(CERTIFIED_ENVIRONMENT, ambient)
    assert legacy_rejects_full.value.code == "SOL_ENV_CONTRACT_INVALID"
    with pytest.raises(CertifiedEnvironmentError):
        build_legacy_certified_env(
            {**LEGACY_CERTIFIED_ENVIRONMENT, "EXTRA": "forbidden"}, ambient,
        )


def test_flashinfer_trace_root_is_removed_by_default_and_injected_only_explicitly(
    tmp_path,
):
    ambient = {
        "PATH": "/usr/bin",
        "FLASHINFER_TRACE_DIR": "/ambient/unverified",
    }
    non_fib = build_certified_env(CERTIFIED_ENVIRONMENT, ambient)
    assert "FLASHINFER_TRACE_DIR" not in non_fib

    corpus = tmp_path / "verified-corpus"
    corpus.mkdir()
    fib = build_certified_env(
        CERTIFIED_ENVIRONMENT, ambient, flashinfer_trace_dir=corpus,
    )
    assert fib["FLASHINFER_TRACE_DIR"] == str(corpus.resolve())
    with pytest.raises(CertifiedEnvironmentError) as caught:
        build_certified_env(
            CERTIFIED_ENVIRONMENT, ambient,
            flashinfer_trace_dir=Path("relative-corpus"),
        )
    assert caught.value.code == "SOL_FIB_CORPUS_INVALID"


@pytest.mark.parametrize(
    "ambient_value",
    ["/usr/bin/g++-15", "/usr/bin/g++-14", "g++-15", ""],
)
def test_certified_environment_rejects_ambient_compiler_spelling_drift(
    ambient_value,
):
    with pytest.raises(CertifiedEnvironmentError) as caught:
        build_certified_env(
            CERTIFIED_ENVIRONMENT,
            {"PATH": "/usr/bin", "NVCC_CCBIN": ambient_value},
        )

    assert caught.value.code == "SOL_ENV_AMBIENT_CONFLICT"


@pytest.mark.parametrize(
    "ambient",
    [
        {"PATH": "/usr/bin", "CUDA_HOME": "/opt/cuda"},
        {"PATH": "/usr/bin", "CC": "gcc-13"},
        {"PATH": "/usr/bin", "CXX": "g++-13"},
        {"PATH": "/usr/bin", "CUDAHOSTCXX": "g++-13"},
    ],
)
def test_certified_environment_rejects_cuda_or_gcc_contract_drift(ambient):
    with pytest.raises(CertifiedEnvironmentError) as caught:
        build_certified_env(CERTIFIED_ENVIRONMENT, ambient)
    assert caught.value.code == "SOL_ENV_AMBIENT_CONFLICT"


@pytest.mark.parametrize("hazard", ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"])
def test_certified_environment_rejects_nvcc_flag_injection_even_when_empty(hazard):
    with pytest.raises(CertifiedEnvironmentError) as caught:
        build_certified_env(
            CERTIFIED_ENVIRONMENT,
            {"PATH": "/usr/bin", hazard: ""},
        )

    assert caught.value.code == "SOL_ENV_HAZARD_CONFLICT"


@pytest.mark.parametrize(
    "contract",
    [
        {},
        {"NVCC_CCBIN": "/usr/bin/g++-14"},
        {"NVCC_CCBIN": "g++-15"},
        {**CERTIFIED_ENVIRONMENT, "EXTRA": "value"},
    ],
)
def test_certified_environment_rejects_noncertified_contract(contract):
    with pytest.raises(CertifiedEnvironmentError) as caught:
        build_certified_env(contract, {"PATH": "/usr/bin"})

    assert caught.value.code == "SOL_ENV_CONTRACT_INVALID"


class _Event:
    def __init__(self, log, name):
        self.log = log
        self.name = name

    def record(self):
        self.log.append(f"{self.name}.record")

    def elapsed_time(self, other):
        self.log.append(f"{self.name}.elapsed_time({other.name})")
        return 1.25


class _Cudart:
    def __init__(self, log, *, start_status=0, stop_status=0):
        self.log = log
        self.start_status = start_status
        self.stop_status = stop_status

    def cudaProfilerStart(self):
        self.log.append("cudaProfilerStart")
        return self.start_status

    def cudaProfilerStop(self):
        self.log.append("cudaProfilerStop")
        return self.stop_status


class _Cuda:
    def __init__(self, log, **status):
        self.log = log
        self.runtime = _Cudart(log, **status)

    def synchronize(self):
        self.log.append("synchronize")

    def cudart(self):
        self.log.append("cudart")
        return self.runtime


def test_profile_candidate_exact_order_and_counts():
    log = []
    outcome = profile_candidate_once(
        cuda=_Cuda(log),
        candidate=lambda: log.append("candidate"),
        start_event=_Event(log, "start"),
        end_event=_Event(log, "end"),
    )

    assert log == [
        "synchronize",
        "start.record",
        "synchronize",
        "cudart",
        "cudaProfilerStart",
        "candidate",
        "end.record",
        "synchronize",
        "cudaProfilerStop",
        "start.elapsed_time(end)",
    ]
    assert outcome.to_dict() == {
        "profiler_start_calls": 1,
        "candidate_user_fn_calls": 1,
        "profiler_stop_calls": 1,
        "profiler_start_status": 0,
        "profiler_stop_status": 0,
        "latency_ms": 1.25,
        "timing_methodology": "cuda_events",
    }


def test_candidate_failure_still_stops_exactly_once_in_finally():
    log = []

    def fail():
        log.append("candidate")
        raise ValueError("candidate failed")

    with pytest.raises(ProfiledCandidateError) as caught:
        profile_candidate_once(
            cuda=_Cuda(log),
            candidate=fail,
            start_event=_Event(log, "start"),
            end_event=_Event(log, "end"),
        )

    assert log.count("cudaProfilerStart") == 1
    assert log.count("candidate") == 1
    assert log.count("cudaProfilerStop") == 1
    assert log[-1] == "cudaProfilerStop"
    assert caught.value.outcome.candidate_user_fn_calls == 1
    assert caught.value.outcome.profiler_stop_calls == 1


def test_start_failure_never_calls_candidate_but_still_stops_once():
    log = []
    with pytest.raises(ProfiledCandidateError) as caught:
        profile_candidate_once(
            cuda=_Cuda(log, start_status=7),
            candidate=lambda: log.append("candidate"),
            start_event=_Event(log, "start"),
            end_event=_Event(log, "end"),
        )

    assert "candidate" not in log
    assert log.count("cudaProfilerStart") == 1
    assert log.count("cudaProfilerStop") == 1
    assert caught.value.outcome.profiler_start_status == 7
    assert caught.value.outcome.candidate_user_fn_calls == 0


def test_stop_failure_is_typed_with_one_completed_candidate():
    log = []
    with pytest.raises(ProfiledCandidateError) as caught:
        profile_candidate_once(
            cuda=_Cuda(log, stop_status=9),
            candidate=lambda: log.append("candidate"),
            start_event=_Event(log, "start"),
            end_event=_Event(log, "end"),
        )

    assert caught.value.outcome.candidate_user_fn_calls == 1
    assert caught.value.outcome.profiler_stop_calls == 1
    assert caught.value.outcome.profiler_stop_status == 9
