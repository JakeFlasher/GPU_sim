"""Pure tests for the strict CUDA/NCU live-capture boundary."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from tools.local_labels import capture_toolchain_contract as contract
from tools.local_labels import ncu_capture_contract as ncu_contract
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


def _explicit_ncu_protocol() -> dict[str, object]:
    return {
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_replay_mode": "kernel",
        "ncu_cache_control": "all",
        "ncu_clock_control": "none",
        "ncu_profile_from_start": False,
        "ncu_launch_skip": 0,
        "ncu_launch_count": 3,
    }


def test_explicit_ncu_launch_protocol_accepts_complete_tuple():
    ncu_contract.validate_measurement_protocol(_explicit_ncu_protocol())


@pytest.mark.parametrize("missing", sorted(_explicit_ncu_protocol()))
def test_explicit_ncu_launch_protocol_rejects_every_partial_tuple(missing):
    protocol = _explicit_ncu_protocol()
    del protocol[missing]
    with pytest.raises(ValueError, match="partial NCU launch controls"):
        ncu_contract.validate_measurement_protocol(protocol)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ncu_target_processes", "application-only"),
        ("ncu_filter_mode", "per-process"),
        ("ncu_replay_mode", "application"),
        ("ncu_cache_control", "flush"),
        ("ncu_clock_control", "default"),
    ],
)
def test_explicit_ncu_launch_protocol_rejects_incompatible_modes(field, value):
    protocol = _explicit_ncu_protocol()
    protocol[field] = value
    with pytest.raises(ValueError, match=field):
        ncu_contract.validate_measurement_protocol(protocol)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ncu_profile_from_start", "off", "must be boolean"),
        ("ncu_launch_skip", "0", "outside the signed 32-bit range"),
        ("ncu_launch_skip", True, "outside the signed 32-bit range"),
        ("ncu_launch_skip", -1, "outside the signed 32-bit range"),
        (
            "ncu_launch_skip",
            ncu_contract.MAX_NCU_SELECTOR + 1,
            "outside the signed 32-bit range",
        ),
        ("ncu_launch_count", "3", "outside the signed 32-bit range"),
        ("ncu_launch_count", True, "outside the signed 32-bit range"),
        ("ncu_launch_count", 0, "outside the signed 32-bit range"),
        (
            "ncu_launch_count",
            ncu_contract.MAX_NCU_SELECTOR + 1,
            "outside the signed 32-bit range",
        ),
    ],
)
def test_explicit_ncu_launch_protocol_rejects_invalid_selector_types_and_bounds(
    field, value, message,
):
    protocol = _explicit_ncu_protocol()
    protocol[field] = value
    with pytest.raises(ValueError, match=message):
        ncu_contract.validate_measurement_protocol(protocol)


def test_explicit_ncu_launch_protocol_rejects_skip_count_overflow():
    protocol = _explicit_ncu_protocol()
    protocol["ncu_launch_skip"] = ncu_contract.MAX_NCU_SELECTOR
    protocol["ncu_launch_count"] = 1
    with pytest.raises(ValueError, match="skip plus count exceeds"):
        ncu_contract.validate_measurement_protocol(protocol)


@pytest.mark.parametrize(
    "protocol",
    [
        None,
        {},
        {"ncu_replay_mode": "kernel"},
        {"ncu_cache_control": "all", "ncu_clock_control": "none"},
    ],
)
def test_legacy_ncu_protocol_without_selector_triggers_remains_valid(protocol):
    ncu_contract.validate_measurement_protocol(protocol)


def test_ncu_control_cli_rejects_replay_only_explicit_mode(capsys):
    assert ncu_contract.main(["--replay-mode", "kernel"]) == 2
    error = capsys.readouterr().err
    assert "NCU_LAUNCH_CONTROL_INVALID" in error
    assert "partial NCU launch controls" in error


def test_ncu_control_cli_rejects_duplicate_explicit_option(capsys):
    assert ncu_contract.main([
        "--launch-count", "3", "--launch-count", "4",
    ]) == 2
    error = capsys.readouterr().err
    assert "NCU_LAUNCH_CONTROL_INVALID" in error
    assert "duplicate explicit launch control: --launch-count" in error


@pytest.mark.parametrize("value", ["00", "01", "+1", "-1", "", "2147483648"])
def test_ncu_control_cli_rejects_noncanonical_or_unbounded_selectors(
    value, capsys,
):
    argv = [
        "--target-processes", "all",
        "--filter-mode", "global",
        "--replay-mode", "kernel",
        "--cache-control", "all",
        "--clock-control", "none",
        "--profile-from-start", "off",
        "--launch-skip", "0",
        "--launch-count", value,
    ]
    assert ncu_contract.main(argv) == 2
    assert "NCU_LAUNCH_CONTROL_INVALID" in capsys.readouterr().err


def _write_executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o700)
    return path


def _label_boundary_command(
    tmp_path: Path, *, cuda_release: str = "13.2",
    ncu_release: str = "2026.1.0.0", include_cycles: bool = True,
    ncu_exit: int = 7,
) -> tuple[list[str], dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ncu_marker = tmp_path / "ncu-target-invoked"
    ncu_argv = tmp_path / "ncu-argv"
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
        f"printf '%s\\n' \"$@\" > {str(ncu_argv)!r}\n"
        f"touch {str(ncu_marker)!r}\n"
        "report=''\n"
        "previous=''\n"
        "for argument in \"$@\"; do\n"
        "  if [[ $previous == --export ]]; then report=$argument; fi\n"
        "  previous=$argument\n"
        "done\n"
        + (
            "[[ -n $report ]] || exit 8\n"
            "printf 'fake report\\n' > \"$report\"\n"
            if ncu_exit == 0 else ""
        )
        + f"exit {ncu_exit}\n",
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
        "--ncu-executable-sha256", hashlib.sha256(ncu.read_bytes()).hexdigest(),
        "--ncu-version-line", f"Version {ncu_release} (public-release)",
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


def _with_explicit_ncu_controls(
    command: list[str], *, launch_count: str = "3",
    omit: frozenset[str] = frozenset(),
    extra_controls: tuple[str, ...] = (),
) -> list[str]:
    controls = [
        ("--target-processes", "all"),
        ("--filter-mode", "global"),
        ("--replay-mode", "kernel"),
        ("--cache-control", "all"),
        ("--clock-control", "none"),
        ("--profile-from-start", "off"),
        ("--launch-skip", "0"),
        ("--launch-count", launch_count),
    ]
    explicit = [
        value for option, argument in controls if option not in omit
        for value in (option, argument)
    ]
    explicit.extend(extra_controls)
    result = list(command)
    separator = result.index("--")
    result[separator:separator] = explicit
    return result


def _option_value(argv: list[str], option: str) -> str:
    assert argv.count(option) == 1
    return argv[argv.index(option) + 1]


def test_label_driver_executes_ncu_through_inherited_descriptor(tmp_path):
    command, environment, output, marker = _label_boundary_command(tmp_path)
    ncu = Path(_option_value(command, "--ncu"))
    descriptor = os.open(ncu, os.O_RDONLY)
    try:
        separator = command.index("--")
        command[separator:separator] = [
            "--ncu-exec-path", f"/proc/self/fd/{descriptor}",
        ]
        result = subprocess.run(
            command,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)
    assert result.returncode == 7
    assert "NCU_AUTHORITY_INVALID" not in result.stderr
    assert output.is_dir()
    assert marker.is_file()


@pytest.mark.parametrize(
    "exec_path", ["/tmp/ncu", "/proc/self/fd/03", "/proc/self/fd/999999"],
)
def test_label_driver_rejects_invalid_internal_ncu_path_before_output(
    tmp_path, exec_path,
):
    command, environment, output, marker = _label_boundary_command(tmp_path)
    separator = command.index("--")
    command[separator:separator] = ["--ncu-exec-path", exec_path]
    result = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "NCU_AUTHORITY_INVALID" in result.stderr
    assert not output.exists()
    assert not marker.exists()


def test_label_driver_explicit_launch_set_reaches_ncu_then_stops_postprocessing(
    tmp_path,
):
    command, environment, output, marker = _label_boundary_command(
        tmp_path, ncu_exit=0,
    )
    command = _with_explicit_ncu_controls(command)
    result = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 3
    assert "NCU_LAUNCH_SET_POSTPROCESSING_UNIMPLEMENTED" in result.stderr
    assert marker.is_file()

    argv = (tmp_path / "ncu-argv").read_text().splitlines()
    assert argv[:16] == [
        "--target-processes", "all",
        "--filter-mode", "global",
        "--replay-mode", "kernel",
        "--cache-control", "all",
        "--clock-control", "none",
        "--profile-from-start", "off",
        "--launch-skip", "0",
        "--launch-count", "3",
    ]
    cell = output / "sm_86" / "boundary-fixture" / "row_0"
    assert (cell / "run_0_0.ncu-rep").read_text() == "fake report\n"
    assert not (cell / "label_0_0.json").exists()
    assert not (cell / "run_0_0.launch_observation.json").exists()


@pytest.mark.parametrize(
    "option",
    ["--ncu", "--ncu-executable-sha256", "--ncu-version-line"],
)
def test_label_driver_requires_complete_import_authority_before_output_or_ncu(
    tmp_path, option,
):
    command, environment, output, marker = _label_boundary_command(tmp_path)
    index = command.index(option)
    del command[index:index + 2]
    result = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "NCU_AUTHORITY_INVALID" in result.stderr or (
        "missing required argument" in result.stderr
    )
    assert not output.exists()
    assert not marker.exists()


@pytest.mark.parametrize(
    "command_mutator",
    [
        lambda command: _with_explicit_ncu_controls(
            command,
            omit=frozenset({
                "--filter-mode", "--replay-mode", "--cache-control",
                "--clock-control", "--profile-from-start", "--launch-skip",
                "--launch-count",
            }),
        ),
        lambda command: _with_explicit_ncu_controls(
            command,
            omit=frozenset({
                "--target-processes", "--filter-mode", "--cache-control",
                "--clock-control", "--profile-from-start", "--launch-skip",
                "--launch-count",
            }),
        ),
        lambda command: _with_explicit_ncu_controls(
            command, extra_controls=("--launch-count", "4"),
        ),
        lambda command: _with_explicit_ncu_controls(
            command, launch_count="03",
        ),
    ],
)
def test_label_driver_refuses_invalid_launch_set_before_output_or_ncu(
    tmp_path, command_mutator,
):
    command, environment, output, marker = _label_boundary_command(tmp_path)
    result = subprocess.run(
        command_mutator(command), env=environment, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "NCU_LAUNCH_CONTROL_INVALID" in result.stderr
    assert not output.exists()
    assert not marker.exists()
    assert not (tmp_path / "ncu-argv").exists()


def test_label_driver_legacy_ncu_arguments_remain_single_launch(tmp_path):
    command, environment, output, marker = _label_boundary_command(tmp_path)
    result = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 7
    assert output.is_dir()
    assert marker.is_file()
    argv = (tmp_path / "ncu-argv").read_text().splitlines()
    assert _option_value(argv, "--target-processes") == "all"
    assert _option_value(argv, "--replay-mode") == "kernel"
    assert _option_value(argv, "--launch-skip") == "0"
    assert _option_value(argv, "--launch-count") == "1"
    assert _option_value(argv, "--clock-control") == "none"
    assert _option_value(argv, "--profile-from-start") == "off"
    assert "--filter-mode" not in argv
    assert "--cache-control" not in argv


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
