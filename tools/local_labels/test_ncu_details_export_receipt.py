from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor
from tools.local_labels import ncu_details_export_receipt as receipt


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


def _products() -> list[dict[str, object]]:
    return [
        receipt.product_binding(
            role="ncu_report",
            relative_path="raw/run_0_launch_set.ncu-rep",
            sha256=_SHA_B,
            size_bytes=101,
        ),
        receipt.product_from_bytes(
            role="details_csv",
            relative_path="raw/run_0_launch_set.details.csv",
            payload=b"ID,Metric Name,Metric Value\n",
        ),
        receipt.product_from_bytes(
            role="details_stderr",
            relative_path="raw/run_0_launch_set.details.stderr",
            payload=b"",
        ),
        receipt.product_from_bytes(
            role="launch_inventory",
            relative_path="raw/run_0_launch_set.launch_inventory.json",
            payload=b'{"schema_version":3}\n',
        ),
        receipt.product_from_bytes(
            role="launch_metrics",
            relative_path="raw/run_0_launch_set.launch_metrics.json",
            payload=b"[]\n",
        ),
    ]


def _kwargs() -> dict[str, object]:
    return {
        "preflight_sha256": _SHA_0,
        "source_commit": "1" * 40,
        "capture_identity_sha256": _SHA_F,
        "expected_ncu_identity": _identity(),
        "import_arguments": receipt.EXPECTED_IMPORT_ARGUMENTS,
        "environment": dict(receipt.EXPECTED_ACTION_ENVIRONMENT),
        "working_directory": "/",
        "exit_status": 0,
        "products": _products(),
    }


def _validation_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    return {
        "expected_preflight_sha256": kwargs["preflight_sha256"],
        "expected_source_commit": kwargs["source_commit"],
        "expected_capture_identity_sha256": kwargs[
            "capture_identity_sha256"
        ],
        "expected_ncu_identity": kwargs["expected_ncu_identity"],
        "expected_import_arguments": kwargs["import_arguments"],
        "expected_environment": kwargs["environment"],
        "expected_working_directory": kwargs["working_directory"],
        "expected_exit_status": kwargs["exit_status"],
        "expected_products": kwargs["products"],
    }


def _reseal(document: dict[str, object]) -> dict[str, object]:
    unsigned = dict(document)
    del unsigned["receipt_seal_sha256"]
    return immutable.seal_model(unsigned, "receipt_seal_sha256")


def test_schema_authority_path_and_bytes_are_pinned(tmp_path: Path) -> None:
    schema_bytes = receipt.SCHEMA_PATH.read_bytes()
    assert immutable.sha256_bytes(schema_bytes) == receipt.SCHEMA_SHA256
    assert receipt.load_schema(payload=schema_bytes)["$id"] == receipt.SCHEMA_ID

    with pytest.raises(receipt.DetailsExportReceiptError, match="not canonical"):
        receipt.load_schema(schema_path=tmp_path / "copy.json", payload=schema_bytes)
    with pytest.raises(receipt.DetailsExportReceiptError, match="SHA-256 mismatch"):
        receipt.load_schema(payload=schema_bytes + b"\n")


def test_receipt_round_trip_binds_empty_stderr_and_all_products() -> None:
    kwargs = _kwargs()
    document = receipt.build_receipt(**kwargs)  # type: ignore[arg-type]

    receipt.validate_receipt(
        document, **_validation_kwargs(kwargs),  # type: ignore[arg-type]
    )
    assert document["products"][2] == {
        "role": "details_stderr",
        "relative_path": "raw/run_0_launch_set.details.stderr",
        "sha256": immutable.sha256_bytes(b""),
        "size_bytes": 0,
    }
    assert document["import_arguments"] == list(
        receipt.EXPECTED_IMPORT_ARGUMENTS
    )
    assert document["import_arguments"].count(
        receipt.RETAINED_REPORT_ARGUMENT
    ) == 1
    assert all(
        "/proc/self/fd/" not in item for item in document["import_arguments"]
    )
    assert document["receipt_seal_sha256"] == immutable.model_seal(
        document, "receipt_seal_sha256",
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("preflight_sha256", _SHA_A),
        ("schema_version", 1.0),
        ("source_commit", "2" * 40),
        ("capture_identity_sha256", _SHA_A),
        ("working_directory", "/tmp"),
        ("exit_status", 1),
    ],
)
def test_resealed_scalar_tamper_is_rejected(
    field: str, replacement: object,
) -> None:
    kwargs = _kwargs()
    document = receipt.build_receipt(**kwargs)  # type: ignore[arg-type]
    document[field] = replacement
    document = _reseal(document)

    with pytest.raises(receipt.DetailsExportReceiptError):
        receipt.validate_receipt(
            document, **_validation_kwargs(kwargs),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("product_index", range(5))
def test_resealed_product_tamper_is_rejected(product_index: int) -> None:
    kwargs = _kwargs()
    document = receipt.build_receipt(**kwargs)  # type: ignore[arg-type]
    document["products"][product_index]["sha256"] = _SHA_C  # type: ignore[index]
    document = _reseal(document)

    with pytest.raises(receipt.DetailsExportReceiptError, match="bindings differ"):
        receipt.validate_receipt(
            document, **_validation_kwargs(kwargs),  # type: ignore[arg-type]
        )


def test_resealed_identity_argument_and_environment_tamper_is_rejected() -> None:
    kwargs = _kwargs()
    for mutate in (
        lambda document: document["ncu_identity"].update(  # type: ignore[union-attr]
            executable_sha256=_SHA_D
        ),
        lambda document: document["import_arguments"].append("--changed"),  # type: ignore[union-attr]
        lambda document: document["environment"].update(PATH="/tmp"),  # type: ignore[union-attr]
    ):
        document = receipt.build_receipt(**kwargs)  # type: ignore[arg-type]
        mutate(document)
        if "--changed" in document["import_arguments"]:  # type: ignore[operator]
            document["import_arguments_sha256"] = immutable.sha256_bytes(
                immutable.canonical_json_bytes(document["import_arguments"])
            )
        if document["environment"].get("PATH") == "/tmp":  # type: ignore[union-attr]
            document["environment_sha256"] = immutable.sha256_bytes(
                immutable.canonical_json_bytes(document["environment"])
            )
        document = _reseal(document)
        with pytest.raises(receipt.DetailsExportReceiptError):
            receipt.validate_receipt(
                document, **_validation_kwargs(kwargs),  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    "arguments",
    [
        ("--import", "/proc/self/fd/9", "--page", "details"),
        ("--import", receipt.RETAINED_REPORT_ARGUMENT, "--page", "raw"),
        (*receipt.EXPECTED_IMPORT_ARGUMENTS, "--changed"),
    ],
)
def test_builder_rejects_non_authoritative_import_arguments(
    arguments: tuple[str, ...],
) -> None:
    kwargs = _kwargs()
    kwargs["import_arguments"] = arguments
    with pytest.raises(receipt.DetailsExportReceiptError, match="arguments"):
        receipt.build_receipt(**kwargs)  # type: ignore[arg-type]


def test_builder_rejects_swapped_roles_duplicate_paths_and_nonempty_rules() -> None:
    kwargs = _kwargs()
    products = copy.deepcopy(kwargs["products"])
    products[0], products[1] = products[1], products[0]  # type: ignore[index]
    kwargs["products"] = products
    with pytest.raises(receipt.DetailsExportReceiptError, match="roles or order"):
        receipt.build_receipt(**kwargs)  # type: ignore[arg-type]

    kwargs = _kwargs()
    products = copy.deepcopy(kwargs["products"])
    products[1]["relative_path"] = products[0]["relative_path"]  # type: ignore[index]
    kwargs["products"] = products
    with pytest.raises(receipt.DetailsExportReceiptError, match="not unique"):
        receipt.build_receipt(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(receipt.DetailsExportReceiptError, match="ncu_report size"):
        receipt.product_binding(
            role="ncu_report",
            relative_path="raw/report.ncu-rep",
            sha256=_SHA_E,
            size_bytes=0,
        )


def test_product_binding_rejects_noncanonical_paths() -> None:
    for path in (
        "raw//report.ncu-rep",
        "raw/report.ncu-rep/",
        "raw/../report.ncu-rep",
        "raw/report\x00.ncu-rep",
        "raw\\report.ncu-rep",
    ):
        with pytest.raises(receipt.DetailsExportReceiptError, match="product path"):
            receipt.product_binding(
                role="ncu_report",
                relative_path=path,
                sha256=_SHA_E,
                size_bytes=1,
            )
