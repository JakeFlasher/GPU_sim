from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor
from tools.local_labels import ncu_collection_receipt as receipt


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64
_SHA_F = "f" * 64
_SHA_0 = "0" * 64


def _identity() -> ncu_action_executor.ExpectedNcuIdentity:
    return ncu_action_executor.build_expected_ncu_identity(
        resolved_path="/opt/nvidia/ncu",
        executable_sha256=_SHA_A,
        version_line="Version 2026.1.0.0 (build 1) (public-release)",
    )


def _environment() -> ncu_action_executor.NcuCollectionEnvironment:
    return ncu_action_executor.NcuCollectionEnvironment({
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": "/usr/local/cuda-13.2/bin:/usr/bin:/bin",
        "PWD": "/capture/work",
        "PYTHONNOUSERSITE": "1",
        "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
        "CUTLASS_DIR": "/opt/cutlass",
    })


def _artifacts() -> list[dict[str, object]]:
    return [
        receipt.artifact_binding(
            role="ncu_report",
            relative_path="raw/cell.ncu-rep",
            sha256=_SHA_B,
            size_bytes=101,
        ),
        receipt.artifact_binding(
            role="label_run_manifest",
            relative_path="raw/cell_run_manifest.json",
            sha256=_SHA_C,
            size_bytes=102,
        ),
        receipt.artifact_binding(
            role="ncu_launch_observation",
            relative_path="raw/cell_launch_observation.json",
            sha256=_SHA_D,
            size_bytes=103,
        ),
        receipt.artifact_binding(
            role="label_capture_manifest",
            relative_path="logs/cell_capture_manifest.json",
            sha256=_SHA_E,
            size_bytes=104,
        ),
    ]


def _kwargs() -> dict[str, object]:
    return {
        "preflight_sha256": _SHA_0,
        "expected_ncu_identity": _identity(),
        "command": [
            "/usr/bin/bash",
            "/repo/tools/local_labels/label_driver.sh",
            "--",
            "/capture/target",
        ],
        "environment": _environment(),
        "working_directory": Path("/capture/work"),
        "exit_status": 0,
        "artifacts": _artifacts(),
        "launch_observation_sha256": _SHA_F,
    }


def _reseal(document: dict[str, object]) -> dict[str, object]:
    unsigned = dict(document)
    del unsigned["receipt_seal_sha256"]
    return immutable.seal_model(unsigned, "receipt_seal_sha256")


def test_schema_authority_path_and_bytes_are_pinned(tmp_path: Path) -> None:
    schema_bytes = receipt.SCHEMA_PATH.read_bytes()
    assert immutable.sha256_bytes(schema_bytes) == receipt.SCHEMA_SHA256
    assert receipt.load_schema(payload=schema_bytes)["$id"] == receipt.SCHEMA_ID

    with pytest.raises(receipt.CollectionReceiptError, match="path is not canonical"):
        receipt.load_schema(schema_path=tmp_path / "copy.json", payload=schema_bytes)
    with pytest.raises(receipt.CollectionReceiptError, match="SHA-256 mismatch"):
        receipt.load_schema(payload=schema_bytes + b"\n")


def test_receipt_round_trip_binds_all_expected_inputs() -> None:
    kwargs = _kwargs()
    document = receipt.build_receipt(**kwargs)  # type: ignore[arg-type]

    receipt.validate_receipt(
        document,
        expected_preflight_sha256=kwargs["preflight_sha256"],
        expected_ncu_identity=kwargs["expected_ncu_identity"],
        expected_command=kwargs["command"],
        expected_environment=kwargs["environment"],
        expected_working_directory=kwargs["working_directory"],
        expected_exit_status=kwargs["exit_status"],
        expected_artifacts=kwargs["artifacts"],
        expected_launch_observation_sha256=kwargs[
            "launch_observation_sha256"
        ],
    )  # type: ignore[arg-type]
    assert document["command_sha256"] == immutable.sha256_bytes(
        immutable.canonical_json_bytes(kwargs["command"])
    )
    assert document["environment_sha256"] == immutable.sha256_bytes(
        immutable.canonical_json_bytes(_environment().as_dict())
    )
    assert document["receipt_seal_sha256"] == immutable.model_seal(
        document, "receipt_seal_sha256"
    )
    assert document["supervision"]["command_derivation"] == (
        "insert_authorized_ncu_proc_path_option_before_target_separator_v1"
    )


@pytest.mark.parametrize("working_directory", [Path("/"), Path("/capture/work")])
def test_absolute_working_directory_contract_accepts_root(
    working_directory: Path,
) -> None:
    kwargs = _kwargs()
    environment_values = _environment().as_dict()
    environment_values["PWD"] = str(working_directory)
    kwargs["environment"] = ncu_action_executor.NcuCollectionEnvironment(
        environment_values
    )
    kwargs["working_directory"] = working_directory
    receipt.build_receipt(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "relative_path",
    ["raw/report\x00.ncu-rep", "raw/report\n.ncu-rep", "raw\\report.ncu-rep"],
)
def test_artifact_binding_rejects_control_and_backslash_paths(
    relative_path: str,
) -> None:
    with pytest.raises(receipt.CollectionReceiptError, match="artifact path"):
        receipt.artifact_binding(
            role="ncu_report",
            relative_path=relative_path,
            sha256=_SHA_A,
            size_bytes=1,
        )


@pytest.mark.parametrize(
    "command",
    [
        ["/driver", "/target"],
        ["/driver", "--ncu-exec-path", "/proc/self/fd/9", "--", "/target"],
        ["/driver", "--ncu-exec-path=/proc/self/fd/9", "--", "/target"],
        ["/driver", "--", "bad\x00arg"],
    ],
)
def test_builder_rejects_noncanonical_execution_command(command: list[str]) -> None:
    kwargs = _kwargs()
    kwargs["command"] = command
    with pytest.raises(receipt.CollectionReceiptError, match="command"):
        receipt.build_receipt(**kwargs)  # type: ignore[arg-type]


def test_validator_rejects_noncanonical_expected_command() -> None:
    kwargs = _kwargs()
    document = receipt.build_receipt(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(receipt.CollectionReceiptError, match="command"):
        receipt.validate_receipt(
            document,
            expected_preflight_sha256=kwargs["preflight_sha256"],
            expected_ncu_identity=kwargs["expected_ncu_identity"],
            expected_command=["/driver", "/target"],
            expected_environment=kwargs["environment"],
            expected_working_directory=kwargs["working_directory"],
            expected_exit_status=kwargs["exit_status"],
            expected_artifacts=kwargs["artifacts"],
            expected_launch_observation_sha256=kwargs[
                "launch_observation_sha256"
            ],
        )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("working_directory", "exit_status", "message"),
    [
        ("/capture/work", 0, "working directory is not a path"),
        (Path("/capture/work"), False, "exit status is not exact integer zero"),
    ],
)
def test_validator_rejects_wrong_scalar_authority_types(
    working_directory: object, exit_status: object, message: str,
) -> None:
    kwargs = _kwargs()
    document = receipt.build_receipt(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(receipt.CollectionReceiptError, match=message):
        receipt.validate_receipt(
            document,
            expected_preflight_sha256=kwargs["preflight_sha256"],
            expected_ncu_identity=kwargs["expected_ncu_identity"],
            expected_command=kwargs["command"],
            expected_environment=kwargs["environment"],
            expected_working_directory=working_directory,
            expected_exit_status=exit_status,
            expected_artifacts=kwargs["artifacts"],
            expected_launch_observation_sha256=kwargs[
                "launch_observation_sha256"
            ],
        )  # type: ignore[arg-type]


@pytest.mark.parametrize("relative_path", ["raw//report.ncu-rep", "raw/report.ncu-rep/"])
def test_builder_revalidates_direct_artifact_bindings(relative_path: str) -> None:
    kwargs = _kwargs()
    artifacts = copy.deepcopy(kwargs["artifacts"])
    artifacts[0]["relative_path"] = relative_path  # type: ignore[index]
    kwargs["artifacts"] = artifacts

    with pytest.raises(receipt.CollectionReceiptError, match="artifact path"):
        receipt.build_receipt(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("preflight_sha256", _SHA_A),
        ("working_directory", "/capture/other"),
        ("exit_status", 1),
        ("launch_observation_sha256", _SHA_A),
    ],
)
def test_resealed_top_level_tamper_is_rejected(
    field: str, replacement: object,
) -> None:
    kwargs = _kwargs()
    document = receipt.build_receipt(**kwargs)  # type: ignore[arg-type]
    document[field] = replacement
    document = _reseal(document)

    with pytest.raises(receipt.CollectionReceiptError):
        receipt.validate_receipt(
            document,
            expected_preflight_sha256=kwargs["preflight_sha256"],
            expected_ncu_identity=kwargs["expected_ncu_identity"],
            expected_command=kwargs["command"],
            expected_environment=kwargs["environment"],
            expected_working_directory=kwargs["working_directory"],
            expected_exit_status=kwargs["exit_status"],
            expected_artifacts=kwargs["artifacts"],
            expected_launch_observation_sha256=kwargs[
                "launch_observation_sha256"
            ],
        )  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["ncu_identity", "command", "environment", "artifacts"])
def test_resealed_nested_binding_tamper_is_rejected(field: str) -> None:
    kwargs = _kwargs()
    document = receipt.build_receipt(**kwargs)  # type: ignore[arg-type]
    if field == "ncu_identity":
        document[field]["executable_sha256"] = _SHA_B  # type: ignore[index]
    elif field == "command":
        document[field].append("--changed")  # type: ignore[union-attr]
        document["command_sha256"] = immutable.sha256_bytes(
            immutable.canonical_json_bytes(document[field])
        )
    elif field == "environment":
        document[field]["CUTLASS_DIR"] = "/other"  # type: ignore[index]
        document["environment_sha256"] = immutable.sha256_bytes(
            immutable.canonical_json_bytes(document[field])
        )
    else:
        document[field][0]["sha256"] = _SHA_A  # type: ignore[index]
    document = _reseal(document)

    with pytest.raises(receipt.CollectionReceiptError, match="bindings differ"):
        receipt.validate_receipt(
            document,
            expected_preflight_sha256=kwargs["preflight_sha256"],
            expected_ncu_identity=kwargs["expected_ncu_identity"],
            expected_command=kwargs["command"],
            expected_environment=kwargs["environment"],
            expected_working_directory=kwargs["working_directory"],
            expected_exit_status=kwargs["exit_status"],
            expected_artifacts=kwargs["artifacts"],
            expected_launch_observation_sha256=kwargs[
                "launch_observation_sha256"
            ],
        )  # type: ignore[arg-type]


def test_unsealed_tamper_is_rejected_before_external_comparison() -> None:
    kwargs = _kwargs()
    document = receipt.build_receipt(**kwargs)  # type: ignore[arg-type]
    document["preflight_sha256"] = _SHA_A

    with pytest.raises(receipt.CollectionReceiptError, match="seal"):
        receipt.validate_receipt(
            document,
            expected_preflight_sha256=_SHA_A,
            expected_ncu_identity=kwargs["expected_ncu_identity"],
            expected_command=kwargs["command"],
            expected_environment=kwargs["environment"],
            expected_working_directory=kwargs["working_directory"],
            expected_exit_status=kwargs["exit_status"],
            expected_artifacts=kwargs["artifacts"],
            expected_launch_observation_sha256=kwargs[
                "launch_observation_sha256"
            ],
        )  # type: ignore[arg-type]


def test_builder_rejects_wrong_artifact_order_and_pwd() -> None:
    kwargs = _kwargs()
    artifacts = copy.deepcopy(kwargs["artifacts"])
    artifacts[0], artifacts[1] = artifacts[1], artifacts[0]  # type: ignore[index]
    kwargs["artifacts"] = artifacts
    with pytest.raises(receipt.CollectionReceiptError, match="roles or order"):
        receipt.build_receipt(**kwargs)  # type: ignore[arg-type]

    kwargs = _kwargs()
    kwargs["working_directory"] = Path("/capture/other")
    with pytest.raises(receipt.CollectionReceiptError, match="PWD differs"):
        receipt.build_receipt(**kwargs)  # type: ignore[arg-type]


def test_artifact_binding_rejects_noncanonical_identity() -> None:
    with pytest.raises(receipt.CollectionReceiptError, match="artifact path"):
        receipt.artifact_binding(
            role="ncu_report",
            relative_path="raw/../report.ncu-rep",
            sha256=_SHA_A,
            size_bytes=1,
        )
    with pytest.raises(receipt.CollectionReceiptError, match="size"):
        receipt.artifact_binding(
            role="ncu_report",
            relative_path="raw/report.ncu-rep",
            sha256=_SHA_A,
            size_bytes=0,
        )
