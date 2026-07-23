"""Pure tests for the strict CUDA/NCU live-capture boundary."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools.local_labels import capture_toolchain_contract as contract
from tools.local_labels import sol_capture_environment as sol_environment


@pytest.mark.parametrize(
    "value",
    ["13.2", {"major": 13, "minor": 2}],
)
def test_nvcc_release_legacy_input_normalizes_to_structured_value(value):
    assert contract.structured_nvcc_release(value) == {
        "major": 13,
        "minor": 2,
    }
    assert contract.nvcc_release_text(value) == "13.2"


@pytest.mark.parametrize(
    "value",
    [
        "13.2.0",
        "release 13.2",
        "013.2",
        13.2,
        True,
        {"major": 13, "minor": 2, "patch": 0},
        {"major": 13, "minor": "2"},
    ],
)
def test_nvcc_release_rejects_ambiguous_or_open_shapes(value):
    with pytest.raises(
        contract.CaptureToolchainContractError,
        match="NVCC_RELEASE_INVALID",
    ):
        contract.structured_nvcc_release(value)


def test_nvcc_banner_parser_is_exact_and_structured():
    banner = (
        "nvcc: NVIDIA (R) Cuda compiler driver\n"
        "Cuda compilation tools, release 13.2, V13.2.140\n"
    )
    assert contract.parse_nvcc_release_banner(banner) == {
        "major": 13,
        "minor": 2,
    }


@pytest.mark.parametrize(
    "banner",
    [
        "Cuda compilation tools V13.2.140",
        "Cuda compilation tools, release 13.2 and release 13.3",
        "Cuda compilation tools, release 013.2",
        "",
    ],
)
def test_nvcc_banner_parser_rejects_missing_or_ambiguous_release(banner):
    with pytest.raises(
        contract.CaptureToolchainContractError,
        match="NVCC_RELEASE_UNREADABLE",
    ):
        contract.parse_nvcc_release_banner(banner)


def test_live_sol_cuda_release_is_exactly_13_2():
    assert contract.require_sol_cuda_release("13.2") == {
        "major": 13,
        "minor": 2,
    }
    with pytest.raises(
        contract.CaptureToolchainContractError,
        match="SOL_ENV_NVCC_RELEASE_MISMATCH",
    ):
        contract.require_sol_cuda_release("13.3")


@pytest.mark.parametrize(
    ("line", "expected", "matches"),
    [
        ("Version 2026.1.0.0 (public-release)", "2026.1.0", True),
        ("Version 2026.1.0", "2026.1.0", True),
        ("Version 2026.1.01.0", "2026.1.0", False),
        ("prefix Version 2026.1.0.0", "2026.1.0", False),
        ("Version 2026.1.1.0", "2026.1.0", False),
    ],
)
def test_ncu_version_match_uses_component_boundaries(line, expected, matches):
    assert contract.ncu_version_matches(line, expected) is matches


def test_cycles_only_scope_keeps_registry_superset_collection():
    assert contract.cycles_only_publication_scope() == {
        "decision": "CYCLES_ONLY_BINDING",
        "binding_metrics": ["gpc__cycles_elapsed.max"],
        "collection_scope": "registry_superset",
    }
    contract.require_cycles_metric_in_collection(
        ["gpc__cycles_elapsed.max", "smsp__inst_executed.sum"]
    )


def test_cycles_only_scope_rejects_claim_or_collection_drift():
    with pytest.raises(
        contract.CaptureToolchainContractError,
        match="NCU_PUBLICATION_SCOPE_MISMATCH",
    ):
        contract.cycles_only_publication_scope({
            "decision": "ALL_METRICS",
            "binding_metrics": ["gpc__cycles_elapsed.max"],
            "collection_scope": "registry_superset",
        })
    with pytest.raises(
        contract.CaptureToolchainContractError,
        match="NCU_PUBLICATION_METRIC_INVALID",
    ):
        contract.require_cycles_metric_in_collection(
            ["smsp__inst_executed.sum"]
        )
    with pytest.raises(
        contract.CaptureToolchainContractError,
        match="NCU_PUBLICATION_METRIC_INVALID",
    ):
        contract.require_cycles_metric_in_collection([
            "gpc__cycles_elapsed.max", "gpc__cycles_elapsed.max",
        ])


def _certification_case(
    monkeypatch, tmp_path, *, nvcc_banner: str, legacy_mode: bool,
):
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text("{}\n")
    native_hash = "a" * 64
    components = {"component": "certified"}
    environment = {
        **sol_environment.PINNED_COMPILER_ENVIRONMENT,
        "CUTLASS_DIR": str(tmp_path),
        "PATH": "/usr/local/cuda-13.2/bin:/usr/bin",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    contract_value = {
        "PATH_prepend": ".",
        **sol_environment.PINNED_COMPILER_ENVIRONMENT,
    }
    monkeypatch.setattr(
        sol_environment,
        "fingerprint_pin",
        lambda *_args, **_kwargs: (
            {"native_env_hash": native_hash},
            tmp_path / "fingerprint.json",
            {"components": components},
        ),
    )
    probe, _runtime = sol_environment._certify_sol_environment_with(
        {}, matrix_path=matrix_path, ambient={}, repo_root=tmp_path,
        fingerprint_probe=lambda _environment: {
            "degraded": False,
            "missing": [],
            "hash": native_hash,
            "components": components,
        },
        contract_loader=lambda _matrix: contract_value,
        environment_builder=lambda _matrix, _ambient: environment,
        ambient_admitter=lambda _contract, _ambient: {},
        certifying_contract_loader=lambda _contract: {},
        contract_digester=lambda _contract: "b" * 64,
        compiler_probe=lambda _environment, *, requested: (
            {
                "requested": requested,
                "resolved_realpath": "/usr/bin/g++-14",
                "executable_sha256": "c" * 64,
                "version_line": "g++-14 14.3.1",
            },
            nvcc_banner,
        ),
        legacy_mode=legacy_mode,
    )
    return probe


def test_live_v2_certification_records_structured_cuda_13_2(
    monkeypatch, tmp_path,
):
    probe = _certification_case(
        monkeypatch,
        tmp_path,
        nvcc_banner="Cuda compilation tools, release 13.2, V13.2.140",
        legacy_mode=False,
    )
    assert probe["nvcc_release"] == {"major": 13, "minor": 2}


def test_live_v2_certification_refuses_cuda_release_drift(
    monkeypatch, tmp_path,
):
    with pytest.raises(
        sol_environment.SolEnvironmentRefusal,
        match="SOL_ENV_NVCC_RELEASE_MISMATCH",
    ):
        _certification_case(
            monkeypatch,
            tmp_path,
            nvcc_banner="Cuda compilation tools, release 13.3, V13.3.90",
            legacy_mode=False,
        )


def test_frozen_v1_certification_keeps_explicit_release_compatibility(
    monkeypatch, tmp_path,
):
    probe = _certification_case(
        monkeypatch,
        tmp_path,
        nvcc_banner="Cuda compilation tools, release 13.3, V13.3.90",
        legacy_mode=True,
    )
    assert probe["nvcc_release"] == {"major": 13, "minor": 3}


def _write_executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o700)
    return path


def _label_boundary_command(
    tmp_path: Path, *, cuda_release: str = "13.2",
    ncu_release: str = "2026.1.0.0", include_cycles: bool = True,
) -> tuple[list[str], dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ncu_marker = tmp_path / "ncu-target-invoked"
    nvcc = _write_executable(
        fake_bin / "nvcc",
        "#!/usr/bin/env bash\n"
        f"printf 'Cuda compilation tools, release {cuda_release}, V{cuda_release}.1\\n'\n",
    )
    del nvcc
    ncu = _write_executable(
        fake_bin / "ncu",
        "#!/usr/bin/env bash\n"
        "if [[ ${1:-} == --version ]]; then\n"
        "  printf 'NVIDIA Nsight Compute\\n'\n"
        f"  printf 'Version {ncu_release} (public-release)\\n'\n"
        "  exit 0\n"
        "fi\n"
        f"touch {str(ncu_marker)!r}\n"
        "exit 7\n",
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        "case \"$*\" in\n"
        "  *driver_version*) printf '610.43.02\\n' ;;\n"
        "  *uuid*) printf 'GPU-test\\n' ;;\n"
        "  *) printf 'unknown\\n' ;;\n"
        "esac\n",
    )
    binary = _write_executable(
        tmp_path / "workload",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    output = tmp_path / "labels"
    command = [
        str(Path(__file__).with_name("label_driver.sh")),
        "--device", "sm_86",
        "--slug", "boundary-fixture",
        "--shape-point", "row_0",
        "--shape-binding", "logical-workload",
        "--repeat", "0",
        "--launch", "0",
        "--binary", str(binary),
        "--ncu", str(ncu),
        "--out-dir", str(output),
        "--expected-cuda-release", "13.2",
        "--expected-ncu-version-prefix", "2026.1.0",
        "--publication-metric-scope", "cycles_only_binding",
    ]
    if include_cycles:
        command += ["--unit-metric", "gpc__cycles_elapsed.max"]
    command += ["--", "synthetic-target-argument"]
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join(
        [str(fake_bin), "/usr/bin", "/bin"]
    )
    # This unit test targets only the post-architecture toolchain boundary;
    # the production campaign never carries this test-only override.
    environment["LABEL_DRIVER_SKIP_ARCH_CHECK"] = "1"
    return command, environment, output, ncu_marker


def test_label_driver_refuses_cuda_release_before_output_or_ncu(
    tmp_path,
):
    command, environment, output, marker = _label_boundary_command(
        tmp_path, cuda_release="13.3",
    )
    result = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "CUDA_RELEASE_MISMATCH" in result.stderr
    assert not output.exists()
    assert not marker.exists()


def test_label_driver_refuses_ncu_release_before_output_or_target(
    tmp_path,
):
    command, environment, output, marker = _label_boundary_command(
        tmp_path, ncu_release="2026.1.1.0",
    )
    result = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "NCU_VERSION_MISMATCH" in result.stderr
    assert not output.exists()
    assert not marker.exists()


def test_label_driver_refuses_cycles_scope_without_binding_metric(
    tmp_path,
):
    command, environment, output, marker = _label_boundary_command(
        tmp_path, include_cycles=False,
    )
    result = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "NCU_PUBLICATION_METRIC_INVALID" in result.stderr
    assert not output.exists()
    assert not marker.exists()
