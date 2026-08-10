from __future__ import annotations

import ast
import copy
import csv
import io
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from eval import production_labels as production
from eval.contracts.launch_crosswalk import CompleteLaunchKey
from eval.contracts.validation import METRIC_KEYS
from tools.local_labels import (
    metric_registry,
    ncu_action_executor,
    ncu_details_export_receipt,
    ncu_launch_documents,
    ncu_launch_sidecars,
)
from tools.local_labels import sidecar_schema_authority
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    canonical_json_bytes,
    pretty_json_bytes,
    seal_model,
    sha256_bytes,
    strict_json_loads,
    verify_model_seal,
)


ROOT = Path(__file__).resolve().parents[2]
_PERCENTAGES = (
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
)
_UNIT_METRICS = (
    "smsp__inst_executed.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "dram__sectors_read.sum",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "gpc__cycles_elapsed.max",
)
_FIELDS = [
    "Metric Value",
    "Stream",
    "ID",
    "Grid Size",
    "Metric Name",
    "Context",
    "Device",
    "Block Size",
    "Process ID",
    "Kernel Name",
]
_IDENTITIES = (
    {
        "ID": "9",
        "Process ID": "100",
        "Context": "1",
        "Stream": "3",
        "Device": "0",
        "Kernel Name": "_Z11first_launchPKfPf",
        "Block Size": "(128, 1, 1)",
        "Grid Size": "(8, 1, 1)",
    },
    {
        "ID": "2",
        "Process ID": "100",
        "Context": "1",
        "Stream": "7",
        "Device": "0",
        "Kernel Name": "_Z12second_launchPKfPf",
        "Block Size": "(64, 2, 1)",
        "Grid Size": "(4, 2, 1)",
    },
)


def _protocol() -> dict[str, Any]:
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


def _values(repeat: int, *, noisy_cycles: bool = False) -> tuple[dict[str, float], ...]:
    cycle0 = (1000.0, 2000.0, 3000.0)[repeat] if noisy_cycles else 1000.0 + repeat * 5.0
    cycle1 = (
        (4000.0, 8000.0, 12000.0)[repeat]
        if noisy_cycles
        else 4000.0 + repeat * 10.0
    )
    return (
        {
            "lts__t_sector_hit_rate.pct": 50.0 + repeat,
            "l1tex__t_sector_hit_rate.pct": 40.0 + repeat,
            "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct": 30.0 + repeat,
            "smsp__inst_executed.sum": 100.0 + repeat,
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 200.0 + repeat,
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": 80.0 + repeat,
            "dram__sectors_read.sum": 120.0 + repeat,
            "gpc__cycles_elapsed.max": cycle0,
        },
        {
            "lts__t_sector_hit_rate.pct": 80.0 + repeat,
            "l1tex__t_sector_hit_rate.pct": 70.0 + repeat,
            "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct": 60.0 + repeat,
            "smsp__inst_executed.sum": 1000.0 + repeat * 2.0,
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 2000.0 + repeat * 2.0,
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": 800.0 + repeat * 2.0,
            "dram__sectors_read.sum": 1200.0 + repeat * 2.0,
            "gpc__cycles_elapsed.max": cycle1,
        },
    )


def _csv_text(
    repeat: int,
    *,
    unit_metrics: tuple[str, ...],
    noisy_cycles: bool = False,
) -> str:
    values = _values(repeat, noisy_cycles=noisy_cycles)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_FIELDS, lineterminator="\n")
    writer.writeheader()
    for metric_name in (*_PERCENTAGES, *unit_metrics):
        for identity, launch_values in zip(_IDENTITIES, values, strict=True):
            writer.writerow(
                {
                    **identity,
                    "Metric Name": metric_name,
                    "Metric Value": str(launch_values[metric_name]),
                }
            )
    return stream.getvalue()


def _crosswalk_payload() -> tuple[bytes, Any]:
    workload_id = f"workload-v1-{sha256_bytes(b'workload')}"
    execution_id = f"matched-execution-v1-{sha256_bytes(b'execution')}"
    keys = (
        {
            "workload_id": workload_id,
            "matched_execution_id": execution_id,
            "process_launch_ordinal": 0,
            "kernel_mangled_name": "_Z11first_launchPKfPf",
            "grid": {"x": 8, "y": 1, "z": 1},
            "block": {"x": 128, "y": 1, "z": 1},
            "device_id": 0,
            "stream_id": 3,
            "binary_architecture": "sm_86",
        },
        {
            "workload_id": workload_id,
            "matched_execution_id": execution_id,
            "process_launch_ordinal": 1,
            "kernel_mangled_name": "_Z12second_launchPKfPf",
            "grid": {"x": 4, "y": 2, "z": 1},
            "block": {"x": 64, "y": 2, "z": 1},
            "device_id": 0,
            "stream_id": 7,
            "binary_architecture": "sm_86",
        },
    )
    ncu_records = [
        {
            "native_identity": {"native_launch_id": identity["ID"]},
            "record_sha256": sha256_bytes(f"ncu-record-{index}".encode()),
        }
        for index, identity in enumerate(_IDENTITIES)
    ]
    model = seal_model(
        {
            "schema_version": 2,
            "artifact_kind": "EVALUATOR_LAUNCH_CROSSWALK",
            "schema_sha256": "4" * 64,
            "contract_mode": "TARGET_FREE_PRODUCTION",
            "executable_sha256": "a" * 64,
            "architecture": {
                "binary_architecture": "sm_86",
                "gpu_sku": "rtx_3090_ti",
            },
            "kernel_family": "vector",
            "inventories": {"ncu": {"records": ncu_records}},
            "bijection": [
                {"launch_key": key, "kernel_family": "vector"} for key in keys
            ],
            "outcome": {"status": "ADMITTED", "reason_codes": []},
        }
    )
    payload = canonical_json_bytes(model)

    def verifier(candidate: bytes, **_inputs: Any) -> dict[str, Any]:
        if candidate != payload:
            raise ImmutableArtifactError("crosswalk differs from verifier authority")
        loaded = strict_json_loads(candidate, label="test production crosswalk")
        if type(loaded) is not dict:
            raise ImmutableArtifactError("crosswalk root invalid")
        verify_model_seal(loaded)
        return loaded

    return payload, verifier


def _fixture_crosswalk_proof(
    payload: bytes, verifier: Any
) -> production.VerifiedProductionCrosswalk:
    """Mint only the opaque proof needed to test the downstream production slice."""

    model = verifier(payload)
    if model["contract_mode"] != "TARGET_FREE_PRODUCTION":
        raise production.ProductionLabelError("fixture crosswalk mode mismatch")
    launches: list[production._CrosswalkLaunch] = []
    records = model["inventories"]["ncu"]["records"]
    for row, record in zip(model["bijection"], records, strict=True):
        if row["kernel_family"] != model["kernel_family"]:
            raise production.ProductionLabelError(
                "crosswalk kernel family is not target-free authoritative"
            )
        launches.append(
            production._CrosswalkLaunch(
                launch_key=CompleteLaunchKey.from_model(row["launch_key"]),
                kernel_family=row["kernel_family"],
                ncu_native_launch_id=record["native_identity"]["native_launch_id"],
                ncu_record_sha256=record["record_sha256"],
            )
        )
    return production.VerifiedProductionCrosswalk(
        payload_sha256=sha256_bytes(payload),
        model_seal_sha256=model["model_seal_sha256"],
        crosswalk_id=f"production-crosswalk-v2-{model['model_seal_sha256']}",
        executable_sha256=model["executable_sha256"],
        binary_architecture=model["architecture"]["binary_architecture"],
        gpu_sku=model["architecture"]["gpu_sku"],
        launches=tuple(launches),
        _construction_token=production._PRIVATE_CROSSWALK_TOKEN,
    )


@dataclass(slots=True)
class LabelCase:
    crosswalk: production.VerifiedProductionCrosswalk
    repeats: tuple[production.ValidatedNcuRepeat, ...]
    source: production.ProductionLabelSetAuthority
    crosswalk_model: dict[str, Any]
    repeat_paths: tuple[dict[str, Any], ...]


def _retain(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _file_binding(path: Path, payload: bytes, *, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _provenance_model(
    provenance: ncu_launch_sidecars.LaunchSidecarProvenance,
) -> dict[str, Any]:
    return {
        "device": provenance.device,
        "slug": provenance.slug,
        "shape_point": provenance.shape_point,
        "binary_sha256": provenance.binary_sha256,
        "profiler_version": provenance.profiler_version,
        "cuda_version": provenance.cuda_version,
        "driver_version": provenance.driver_version,
        "gpu_uuid_sha256": provenance.gpu_uuid_sha256,
        "warmup_count": provenance.warmup_count,
    }


def _fixture_repeat_authority(
    *,
    manifest_path: Path,
    source: production.ProductionLabelSetAuthority,
    crosswalk: production.VerifiedProductionCrosswalk,
    repeat_index: int,
    repeat_id: str | None = None,
) -> production._VerifiedRepeatManifestAuthority:
    return production._VerifiedRepeatManifestAuthority(
        authority_id=f"fixture-repeat-authority-{repeat_index}",
        manifest_file_sha256=sha256_bytes(manifest_path.read_bytes()),
        repeat_id=repeat_id or f"repeat-{repeat_index}",
        cohort_repeat_ordinal=repeat_index,
        source_manifest_sha256=source.source_manifest_sha256,
        producer_content_manifest_sha256=(
            source.producer_content_manifest_sha256
        ),
        crosswalk_file_sha256=crosswalk.payload_sha256,
        _construction_token=production._PRIVATE_REPEAT_AUTHORITY_TOKEN,
    )


def _details_receipt_bytes(
    *, root: Path, products: dict[str, tuple[Path, bytes]], repeat_index: int
) -> bytes:
    ncu_identity = ncu_action_executor.ExpectedNcuIdentity(
        resolved_path="/opt/nvidia/nsight-compute/ncu",
        executable_sha256="e" * 64,
        version_line="Version 2026.1.0.0 (build 1) (public-release)",
    )
    product_models = [
        ncu_details_export_receipt.product_from_bytes(
            role=role,
            relative_path=products[role][0].relative_to(root).as_posix(),
            payload=products[role][1],
        )
        for role in (
            "ncu_report",
            "details_csv",
            "details_stderr",
            "launch_inventory",
            "launch_metrics",
        )
    ]
    receipt = ncu_details_export_receipt.build_receipt(
        preflight_sha256=sha256_bytes(f"preflight-{repeat_index}".encode()),
        source_commit="d" * 40,
        capture_identity_sha256=sha256_bytes(
            f"capture-{repeat_index}".encode()
        ),
        expected_ncu_identity=ncu_identity,
        import_arguments=ncu_details_export_receipt.EXPECTED_IMPORT_ARGUMENTS,
        environment=dict(ncu_details_export_receipt.EXPECTED_ACTION_ENVIRONMENT),
        working_directory="/",
        exit_status=0,
        products=product_models,
    )
    return pretty_json_bytes(receipt)


def _build_repeat_fixture(
    *,
    root: Path,
    repeat_index: int,
    unit_metrics: tuple[str, ...],
    noisy_cycles: bool,
    source: production.ProductionLabelSetAuthority,
    crosswalk: production.VerifiedProductionCrosswalk,
    registry: Any,
    sidecar_schema: dict[str, Any],
    provenance: ncu_launch_sidecars.LaunchSidecarProvenance,
) -> tuple[production.ValidatedNcuRepeat, dict[str, Any]]:
    text = _csv_text(
        repeat_index,
        unit_metrics=unit_metrics,
        noisy_cycles=noisy_cycles,
    )
    report_bytes = f"report-{repeat_index}".encode()
    report_sha256 = sha256_bytes(report_bytes)
    inventory = ncu_launch_documents.build_launch_inventory_document(
        csv_text=text,
        report_sha256=report_sha256,
        expected_launch_count=2,
        protocol=_protocol(),
    )
    metrics = ncu_launch_documents.build_launch_metrics_document(
        csv_text=text,
        inventory=inventory,
        registry=registry,
        percentage_metric_names=_PERCENTAGES,
        unit_metric_names=unit_metrics,
    )
    sidecars = ncu_launch_sidecars.build_launch_sidecar_documents(
        inventory=inventory,
        launch_metrics=metrics,
        csv_text=text,
        report_sha256=report_sha256,
        expected_launch_count=2,
        protocol=_protocol(),
        registry=registry,
        percentage_metric_names=_PERCENTAGES,
        unit_metric_names=unit_metrics,
        sidecar_schema=sidecar_schema,
        provenance=provenance,
    )
    repeat_root = root / f"repeat-{repeat_index}"
    report_path = _retain(repeat_root / "capture.ncu-rep", report_bytes)
    csv_bytes = text.encode()
    csv_path = _retain(repeat_root / "details.csv", csv_bytes)
    stderr_bytes = b""
    stderr_path = _retain(repeat_root / "details.stderr", stderr_bytes)
    inventory_bytes = pretty_json_bytes(inventory)
    inventory_path = _retain(repeat_root / "inventory.json", inventory_bytes)
    metrics_bytes = pretty_json_bytes(metrics)
    metrics_path = _retain(repeat_root / "metrics.json", metrics_bytes)
    sidecar_payloads = tuple(canonical_json_bytes(sidecar) for sidecar in sidecars)
    sidecar_paths = tuple(
        _retain(repeat_root / f"label_0_{ordinal}.json", payload)
        for ordinal, payload in enumerate(sidecar_payloads)
    )
    products = {
        "ncu_report": (report_path, report_bytes),
        "details_csv": (csv_path, csv_bytes),
        "details_stderr": (stderr_path, stderr_bytes),
        "launch_inventory": (inventory_path, inventory_bytes),
        "launch_metrics": (metrics_path, metrics_bytes),
    }
    receipt_bytes = _details_receipt_bytes(
        root=root, products=products, repeat_index=repeat_index
    )
    receipt_path = _retain(repeat_root / "details_receipt.json", receipt_bytes)


    artifact_bindings = {
        "ncu_report": _file_binding(report_path, report_bytes, root=root),
        "details_export_receipt": _file_binding(
            receipt_path, receipt_bytes, root=root
        ),
        "details_csv": _file_binding(csv_path, csv_bytes, root=root),
        "details_stderr": _file_binding(stderr_path, stderr_bytes, root=root),
        "launch_inventory": _file_binding(
            inventory_path, inventory_bytes, root=root
        ),
        "launch_metrics": _file_binding(metrics_path, metrics_bytes, root=root),
        "launch_sidecars": [
            {
                "report_launch_ordinal": ordinal,
                **_file_binding(path, payload, root=root),
            }
            for ordinal, (path, payload) in enumerate(
                zip(sidecar_paths, sidecar_payloads, strict=True)
            )
        ],
    }
    manifest = {
        "schema_version": production.REPEAT_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "PRODUCTION_NCU_REPEAT_MANIFEST",
        "authority_mode": "EXTERNALLY_PINNED_TARGET_FREE_PRELABEL",
        "repeat_identity": {
            "repeat_id": f"repeat-{repeat_index}",
            "cohort_repeat_ordinal": repeat_index,
        },
        "source_label_identity": {
            "label_set_id": source.label_set_id,
            "label_set_sha256": source.label_set_sha256,
            "source_manifest_id": source.source_manifest_id,
            "source_manifest_sha256": source.source_manifest_sha256,
        },
        "crosswalk_identity": {
            "crosswalk_id": crosswalk.crosswalk_id,
            "crosswalk_file_sha256": crosswalk.payload_sha256,
            "crosswalk_model_seal_sha256": crosswalk.model_seal_sha256,
        },
        "architecture": {
            "binary_architecture": crosswalk.binary_architecture,
            "gpu_sku": crosswalk.gpu_sku,
        },
        "producer_content_manifest": {
            "manifest_id": source.producer_content_manifest_id,
            "content_manifest_sha256": (
                source.producer_content_manifest_sha256
            ),
        },
        "capture_contract": {
            "protocol": _protocol(),
            "percentage_metric_names": list(_PERCENTAGES),
            "unit_metric_names": list(unit_metrics),
            "sidecar_provenance": _provenance_model(provenance),
        },
        "launch_bindings": [
            {
                "cohort_repeat_ordinal": repeat_index,
                "report_launch_ordinal": ordinal,
                "launch_key": launch.launch_key.to_model(),
                "launch_key_sha256": launch.launch_key.sha256,
                "ncu_record_sha256": launch.ncu_record_sha256,
            }
            for ordinal, launch in enumerate(crosswalk.launches)
        ],
        "artifacts": artifact_bindings,
    }
    manifest_path = _retain(
        root / f"repeat-manifest-{repeat_index}.json",
        canonical_json_bytes(manifest),
    )
    authority = _fixture_repeat_authority(
        manifest_path=manifest_path,
        source=source,
        crosswalk=crosswalk,
        repeat_index=repeat_index,
    )
    validated = production._validate_retained_ncu_repeat_from_manifest(
        repeat_manifest_path=manifest_path,
        authority=authority,
        source=source,
        crosswalk=crosswalk,
    )
    paths = {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "authority": authority,
        "report_path": report_path,
        "receipt_path": receipt_path,
        "csv_path": csv_path,
        "inventory_path": inventory_path,
        "metrics_path": metrics_path,
        "sidecar_paths": sidecar_paths,
        "report_sha256": report_sha256,
        "provenance": provenance,
        "unit_metrics": unit_metrics,
    }
    return validated, paths


def _make_case(
    tmp_path: Path,
    *,
    unit_metrics: tuple[str, ...] = _UNIT_METRICS,
    noisy_cycles: bool = False,
) -> LabelCase:
    crosswalk_payload, crosswalk_verifier = _crosswalk_payload()
    _retain(tmp_path / "crosswalk.json", crosswalk_payload)
    crosswalk = _fixture_crosswalk_proof(crosswalk_payload, crosswalk_verifier)
    source = production.ProductionLabelSetAuthority(
        label_set_id="retained-ncu-labels-v5",
        label_set_sha256="1" * 64,
        source_manifest_id="retained-ncu-manifest-v1",
        source_manifest_sha256="2" * 64,
        expected_repeat_count=3,
        binary_architecture="sm_86",
        gpu_sku="rtx_3090_ti",
        producer_content_manifest_id="external-generation-v1",
        producer_content_manifest_sha256="3" * 64,
    )
    registry = metric_registry.load_registry()
    _schema_bytes, sidecar_schema = sidecar_schema_authority.load_authoritative_schema()
    provenance = ncu_launch_sidecars.LaunchSidecarProvenance(
        device="sm_86",
        slug="fixture-two-launch",
        shape_point="shape-a",
        binary_sha256="a" * 64,
        profiler_version="Version 2026.1.0.0 (build 1) (public-release)",
        cuda_version="13.3",
        driver_version="610.43.02",
        gpu_uuid_sha256="f" * 64,
        warmup_count=1,
    )
    repeats: list[production.ValidatedNcuRepeat] = []
    repeat_paths: list[dict[str, Any]] = []
    for repeat_index in range(3):
        repeat, paths = _build_repeat_fixture(
            root=tmp_path,
            repeat_index=repeat_index,
            unit_metrics=unit_metrics,
            noisy_cycles=noisy_cycles,
            source=source,
            crosswalk=crosswalk,
            registry=registry,
            sidecar_schema=sidecar_schema,
            provenance=provenance,
        )
        repeats.append(repeat)
        repeat_paths.append(paths)
    return LabelCase(
        crosswalk=crosswalk,
        repeats=tuple(repeats),
        source=source,
        crosswalk_model=strict_json_loads(
            crosswalk_payload, label="test production crosswalk"
        ),
        repeat_paths=tuple(repeat_paths),
    )

def _validate_manifest_model(
    case: LabelCase,
    manifest: dict[str, Any],
    *,
    repeat_index: int,
    filename: str,
    repeat_id: str | None = None,
) -> production.ValidatedNcuRepeat:
    root = case.repeat_paths[0]["manifest_path"].parent
    manifest_path = _retain(root / filename, canonical_json_bytes(manifest))
    authority = _fixture_repeat_authority(
        manifest_path=manifest_path,
        source=case.source,
        crosswalk=case.crosswalk,
        repeat_index=repeat_index,
        repeat_id=repeat_id,
    )
    return production._validate_retained_ncu_repeat_from_manifest(
        repeat_manifest_path=manifest_path,
        authority=authority,
        source=case.source,
        crosswalk=case.crosswalk,
    )


def _write_artifacts(
    tmp_path: Path, artifacts: production.ProductionMatchedLabelArtifacts
) -> tuple[Path, Path]:
    labels_path = _retain(
        tmp_path / "matched-labels.json", artifacts.matched_labels_bytes
    )
    authority_path = _retain(tmp_path / "authority.json", artifacts.authority_bytes)
    return labels_path, authority_path


def _authority_pin(
    artifacts: production.ProductionMatchedLabelArtifacts,
) -> production.ProductionAuthorityPin:
    authority = artifacts.authority_document
    source = authority["source_label_identity"]
    return production.ProductionAuthorityPin(
        authority_file_sha256=sha256_bytes(artifacts.authority_bytes),
        producer_content_manifest_sha256=authority["producer_content_manifest"][
            "content_manifest_sha256"
        ],
        label_set_sha256=source["label_set_sha256"],
        source_manifest_sha256=source["source_manifest_sha256"],
        crosswalk_file_sha256=authority["crosswalk_identity"][
            "crosswalk_file_sha256"
        ],
        builder_sha256=authority["builder_identity"]["builder_sha256"],
    )


def test_builds_canonical_per_launch_labels_and_detached_authority(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    artifacts = production.build_production_matched_label_artifacts(
        source=case.source,
        crosswalk=case.crosswalk,
        repeats=case.repeats,
    )
    labels = artifacts.matched_labels_document
    assert canonical_json_bytes(labels) == artifacts.matched_labels_bytes
    assert (
        canonical_json_bytes(artifacts.authority_document)
        == artifacts.authority_bytes
    )
    assert len(labels["launches"]) == 2
    assert all(len(launch["metrics"]) == 8 for launch in labels["launches"])
    assert [row["metric_key"] for row in labels["launches"][0]["metrics"]] == list(
        METRIC_KEYS
    )
    assert labels["launches"][0]["metrics"][0]["outcome"]["value"] == 101.0
    assert labels["launches"][1]["metrics"][0]["outcome"]["value"] == 1002.0
    assert all(
        metric["repeat_metadata"]["repeat_count"] == 3
        for launch in labels["launches"]
        for metric in launch["metrics"]
    )
    authority = artifacts.authority_document
    assert authority["matched_labels"]["file_sha256"] == sha256_bytes(
        artifacts.matched_labels_bytes
    )
    assert authority["source_label_identity"]["source_manifest_sha256"] == "2" * 64
    assert authority["producer_content_manifest"]["content_manifest_sha256"] == "3" * 64
    schema = production._decode_schema_json(
        production.AUTHORITY_SCHEMA_PATH.read_bytes(), "authority schema"
    )
    Draft202012Validator(schema).validate(authority)


def test_detached_loader_authenticates_and_freezes_production_lane(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    artifacts = production.build_production_matched_label_artifacts(
        source=case.source,
        crosswalk=case.crosswalk,
        repeats=case.repeats,
    )
    labels_path, authority_path = _write_artifacts(tmp_path, artifacts)
    loaded = production.load_production_matched_labels(
        labels_path,
        authority_path,
        authority_pin=_authority_pin(artifacts),
    )
    assert loaded.synthetic_fixture_id is None
    assert loaded.matched_labels_sha256 == artifacts.matched_labels_document[
        "matched_labels_sha256"
    ]
    assert len(loaded.launches) == 2


def test_missing_registry_metric_is_typed_unavailable_and_never_disappears(
    tmp_path: Path,
) -> None:
    units = tuple(name for name in _UNIT_METRICS if name != "dram__sectors_read.sum")
    case = _make_case(tmp_path, unit_metrics=units)
    build = production.build_production_matched_labels(
        source=case.source, crosswalk=case.crosswalk, repeats=case.repeats
    )
    for launch in build.document["launches"]:
        assert len(launch["metrics"]) == 8
        dram = next(
            row
            for row in launch["metrics"]
            if row["metric_key"] == "traffic_dram_read_sectors"
        )
        assert dram["outcome"] == {
            "status": "unavailable",
            "value": None,
            "reason": "metric_not_collected",
        }
        assert dram["repeat_metadata"]["repeat_count"] == 3
        assert dram["repeat_metadata"]["dispersion"]["status"] == "unavailable"


def test_failed_repeatability_is_typed_unavailable_with_actual_dispersion(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, noisy_cycles=True)
    build = production.build_production_matched_labels(
        source=case.source, crosswalk=case.crosswalk, repeats=case.repeats
    )
    for launch in build.document["launches"]:
        cycles = next(
            row
            for row in launch["metrics"]
            if row["metric_key"] == "exec_cycles_gpc_max"
        )
        assert cycles["outcome"]["status"] == "unavailable"
        assert cycles["outcome"]["reason"] == "typed_exclusion"
        assert cycles["repeat_metadata"]["repeat_count"] == 3
        dispersion = cycles["repeat_metadata"]["dispersion"]
        assert dispersion["status"] == "available"
        assert dispersion["rule"] == "coefficient_of_variation"
        assert dispersion["scalar"] > 0.05


def test_label_payload_is_not_opened_before_detached_authority_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    artifacts = production.build_production_matched_label_artifacts(
        source=case.source, crosswalk=case.crosswalk, repeats=case.repeats,
    )
    labels_path, authority_path = _write_artifacts(tmp_path, artifacts)

    def forbidden_open(_path: str | Path) -> bytes:
        raise AssertionError("matched-label payload opened before authority")

    monkeypatch.setattr(production, "_read_label_path", forbidden_open)
    bad_pin = replace(_authority_pin(artifacts), authority_file_sha256="0" * 64)
    with pytest.raises(production.ProductionLabelError, match="authority file SHA-256"):
        production.load_authorized_production_matched_labels(
            labels_path, authority_path, authority_pin=bad_pin
        )


def test_loader_rejects_matched_label_file_hash_drift(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    artifacts = production.build_production_matched_label_artifacts(
        source=case.source, crosswalk=case.crosswalk, repeats=case.repeats,
    )
    labels_path, authority_path = _write_artifacts(tmp_path, artifacts)
    drifted = artifacts.matched_labels_bytes[:-1] + b" "
    labels_path.write_bytes(drifted)
    with pytest.raises(production.ProductionLabelError, match="file SHA-256"):
        production.load_authorized_production_matched_labels(
            labels_path, authority_path, authority_pin=_authority_pin(artifacts)
        )


@pytest.mark.parametrize(
    "selection",
    [
        lambda rows: (rows[0], rows[0], rows[2]),
        lambda rows: (rows[0], rows[2]),
        lambda rows: (rows[1], rows[0], rows[2]),
    ],
    ids=("duplicate", "missing", "reordered"),
)
def test_builder_rejects_duplicate_missing_or_reordered_repeats(
    tmp_path: Path, selection: Any
) -> None:
    case = _make_case(tmp_path)
    with pytest.raises(
        production.ProductionLabelError, match="missing, duplicated, or reordered"
    ):
        production.build_production_matched_labels(
            source=case.source,
            crosswalk=case.crosswalk,
            repeats=selection(case.repeats),
        )


def test_three_distinct_manifests_cannot_reuse_repeat_zero_artifact_bundle(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    repeated: list[production.ValidatedNcuRepeat] = []
    for repeat_index in range(3):
        manifest = copy.deepcopy(case.repeat_paths[0]["manifest"])
        manifest["repeat_identity"] = {
            "repeat_id": f"forged-repeat-{repeat_index}",
            "cohort_repeat_ordinal": repeat_index,
        }
        for binding in manifest["launch_bindings"]:
            binding["cohort_repeat_ordinal"] = repeat_index
        repeated.append(
            _validate_manifest_model(
                case,
                manifest,
                repeat_index=repeat_index,
                repeat_id=f"forged-repeat-{repeat_index}",
                filename=f"forged-repeat-manifest-{repeat_index}.json",
            )
        )
    assert len({item.repeat_manifest_sha256 for item in repeated}) == 3
    with pytest.raises(
        production.ProductionLabelError,
        match="artifact bundles are duplicated|reused across repeats",
    ):
        production.build_production_matched_labels(
            source=case.source,
            crosswalk=case.crosswalk,
            repeats=tuple(repeated),
        )


def _rebuild_receipt_for_manifest(
    *, root: Path, original_receipt_path: Path, artifacts: dict[str, Any]
) -> tuple[Path, bytes]:
    original = strict_json_loads(
        original_receipt_path.read_bytes(), label="fixture Details receipt"
    )
    assert type(original) is dict
    ncu_identity_model = original["ncu_identity"]
    ncu_identity = ncu_action_executor.ExpectedNcuIdentity(
        resolved_path=ncu_identity_model["resolved_path"],
        executable_sha256=ncu_identity_model["executable_sha256"],
        version_line=ncu_identity_model["version_line"],
    )
    products = [
        ncu_details_export_receipt.product_binding(
            role=role,
            relative_path=artifacts[role]["relative_path"],
            sha256=artifacts[role]["sha256"],
            size_bytes=artifacts[role]["size_bytes"],
        )
        for role in (
            "ncu_report",
            "details_csv",
            "details_stderr",
            "launch_inventory",
            "launch_metrics",
        )
    ]
    rebuilt = ncu_details_export_receipt.build_receipt(
        preflight_sha256=original["preflight_sha256"],
        source_commit=original["source_commit"],
        capture_identity_sha256=original["capture_identity_sha256"],
        expected_ncu_identity=ncu_identity,
        import_arguments=ncu_details_export_receipt.EXPECTED_IMPORT_ARGUMENTS,
        environment=dict(ncu_details_export_receipt.EXPECTED_ACTION_ENVIRONMENT),
        working_directory="/",
        exit_status=0,
        products=products,
    )
    payload = pretty_json_bytes(rebuilt)
    path = _retain(root / "revalidated-details-receipt.json", payload)
    return path, payload


def _validate_again(
    case: LabelCase,
    repeat_index: int,
    *,
    report_path: Path | None = None,
    csv_path: Path | None = None,
    inventory_path: Path | None = None,
    metrics_path: Path | None = None,
    sidecar_paths: tuple[Path, ...] | None = None,
    provenance: ncu_launch_sidecars.LaunchSidecarProvenance | None = None,
) -> production.ValidatedNcuRepeat:
    paths = case.repeat_paths[repeat_index]
    root = paths["manifest_path"].parent
    manifest = copy.deepcopy(paths["manifest"])
    artifact_changed = False
    for role, replacement in (
        ("ncu_report", report_path),
        ("details_csv", csv_path),
        ("launch_inventory", inventory_path),
        ("launch_metrics", metrics_path),
    ):
        if replacement is not None:
            payload = replacement.read_bytes()
            manifest["artifacts"][role] = _file_binding(
                replacement, payload, root=root
            )
            artifact_changed = True
    if sidecar_paths is not None:
        manifest["artifacts"]["launch_sidecars"] = [
            {
                "report_launch_ordinal": ordinal,
                **_file_binding(path, path.read_bytes(), root=root),
            }
            for ordinal, path in enumerate(sidecar_paths)
        ]
    if provenance is not None:
        manifest["capture_contract"]["sidecar_provenance"] = (
            _provenance_model(provenance)
        )


    if artifact_changed:
        receipt_path, receipt_payload = _rebuild_receipt_for_manifest(
            root=root,
            original_receipt_path=paths["receipt_path"],
            artifacts=manifest["artifacts"],
        )
        manifest["artifacts"]["details_export_receipt"] = _file_binding(
            receipt_path, receipt_payload, root=root
        )
    manifest_path = _retain(
        root / f"revalidated-repeat-manifest-{repeat_index}.json",
        canonical_json_bytes(manifest),
    )
    authority = _fixture_repeat_authority(
        manifest_path=manifest_path,
        source=case.source,
        crosswalk=case.crosswalk,
        repeat_index=repeat_index,
    )
    return production._validate_retained_ncu_repeat_from_manifest(
        repeat_manifest_path=manifest_path,
        authority=authority,
        source=case.source,
        crosswalk=case.crosswalk,
    )


def test_repeat_manifest_tamper_fails_before_artifact_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    paths = case.repeat_paths[0]
    paths["manifest_path"].write_bytes(paths["manifest_path"].read_bytes() + b" ")

    def forbidden_artifact_open(**_kwargs: Any) -> Any:
        raise AssertionError("artifact opened before repeat manifest authentication")

    monkeypatch.setattr(
        production, "_read_bound_repeat_artifact", forbidden_artifact_open
    )
    with pytest.raises(production.ProductionLabelError, match="manifest file SHA-256"):
        production._validate_retained_ncu_repeat_from_manifest(
            repeat_manifest_path=paths["manifest_path"],
            authority=paths["authority"],
            source=case.source,
            crosswalk=case.crosswalk,
        )


def test_repeat_artifact_tamper_fails_against_manifest_binding(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    paths = case.repeat_paths[0]
    paths["csv_path"].write_bytes(paths["csv_path"].read_bytes() + b"\n")
    with pytest.raises(
        production.ProductionLabelError, match="details_csv differs"
    ):
        production._validate_retained_ncu_repeat_from_manifest(
            repeat_manifest_path=paths["manifest_path"],
            authority=paths["authority"],
            source=case.source,
            crosswalk=case.crosswalk,
        )


@pytest.mark.parametrize("role", ("details_csv", "ncu_report"))
def test_authenticated_repeat_rejects_cross_repeat_artifact_substitution(
    tmp_path: Path, role: str
) -> None:
    case = _make_case(tmp_path)
    argument = "csv_path" if role == "details_csv" else "report_path"
    with pytest.raises(
        production.ProductionLabelError,
        match="launch documents|receipt|semantic contract",
    ):
        _validate_again(
            case,
            0,
            **{argument: case.repeat_paths[1][argument]},
        )


def test_authenticated_repeat_rejects_receipt_substitution(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    manifest = copy.deepcopy(case.repeat_paths[0]["manifest"])
    substituted = case.repeat_paths[1]["receipt_path"]
    manifest["artifacts"]["details_export_receipt"] = _file_binding(
        substituted,
        substituted.read_bytes(),
        root=case.repeat_paths[0]["manifest_path"].parent,
    )
    with pytest.raises(production.ProductionLabelError, match="receipt"):
        _validate_manifest_model(
            case,
            manifest,
            repeat_index=0,
            filename="substituted-receipt-manifest.json",
        )


@pytest.mark.parametrize("mode", ("full_key", "cohort"))
def test_repeat_manifest_rejects_full_key_or_cohort_substitution(
    tmp_path: Path, mode: str
) -> None:
    case = _make_case(tmp_path)
    manifest = copy.deepcopy(case.repeat_paths[0]["manifest"])
    if mode == "full_key":
        key_model = manifest["launch_bindings"][0]["launch_key"]
        key_model["workload_id"] = f"workload-v1-{sha256_bytes(b'forged')}"
        forged_key = CompleteLaunchKey.from_model(key_model)
        manifest["launch_bindings"][0]["launch_key_sha256"] = forged_key.sha256
        match = "full key"
    else:
        manifest["repeat_identity"]["cohort_repeat_ordinal"] = 1
        for binding in manifest["launch_bindings"]:
            binding["cohort_repeat_ordinal"] = 1
        match = "repeat identity"
    with pytest.raises(production.ProductionLabelError, match=match):
        _validate_manifest_model(
            case,
            manifest,
            repeat_index=0,
            filename=f"substituted-{mode}-manifest.json",
        )


@pytest.mark.parametrize("mode", ("missing", "duplicate"))
def test_retained_validation_rejects_missing_or_duplicate_launch_inventory(
    tmp_path: Path, mode: str
) -> None:
    case = _make_case(tmp_path)
    paths = case.repeat_paths[0]
    inventory = production._decode_json(
        paths["inventory_path"].read_bytes(), "test inventory"
    )
    if mode == "missing":
        inventory["launches"] = inventory["launches"][:-1]
        inventory["launch_count"] = 1
    else:
        inventory["launches"] = [inventory["launches"][0]] * 2
    hostile = _retain(
        tmp_path / f"{mode}-inventory.json", pretty_json_bytes(inventory)
    )
    with pytest.raises(production.ProductionLabelError, match="launch documents"):
        _validate_again(case, 0, inventory_path=hostile)


def test_retained_validation_rejects_metric_unit_mismatch(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    paths = case.repeat_paths[0]
    metrics = production._decode_json(
        paths["metrics_path"].read_bytes(), "test metrics"
    )
    metrics[0]["unit_tagged_metrics"]["dram__sectors_read.sum"]["unit"] = "cycles"
    hostile = _retain(tmp_path / "unit-drift.json", pretty_json_bytes(metrics))
    with pytest.raises(production.ProductionLabelError, match="unit|launch documents"):
        _validate_again(case, 0, metrics_path=hostile)


def test_retained_validation_rejects_sidecar_architecture_mismatch(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    hostile = replace(case.repeat_paths[0]["provenance"], device="sm_120")
    with pytest.raises(
        production.ProductionLabelError, match="architecture|device|schema"
    ):
        _validate_again(case, 0, provenance=hostile)


def test_retained_validation_rejects_receipt_provenance_substitution(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    hostile = replace(
        case.repeat_paths[0]["provenance"],
        profiler_version="Version 2025.4.0.0 (build 1)",
    )
    with pytest.raises(
        production.ProductionLabelError, match="receipt identity|profiler provenance"
    ):
        _validate_again(case, 0, provenance=hostile)


def test_builder_rejects_architecture_or_sku_mismatch(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    hostile = replace(case.source, binary_architecture="sm_120")
    with pytest.raises(production.ProductionLabelError, match="architecture/SKU"):
        production.build_production_matched_labels(
            source=hostile, crosswalk=case.crosswalk, repeats=case.repeats
        )


def test_forged_kernel_family_cannot_bypass_target_free_crosswalk(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    forged = copy.deepcopy(case.crosswalk_model)
    forged["bijection"][0]["kernel_family"] = "label-derived-family"
    forged.pop("model_seal_sha256")
    forged = seal_model(forged)
    forged_bytes = canonical_json_bytes(forged)

    def forged_verifier(payload: bytes, **_inputs: Any) -> dict[str, Any]:
        model = strict_json_loads(payload, label="forged crosswalk")
        assert type(model) is dict
        verify_model_seal(model)
        return model

    with pytest.raises(production.ProductionLabelError, match="kernel family"):
        _fixture_crosswalk_proof(forged_bytes, forged_verifier)


def test_label_side_target_fields_cannot_flow_toward_crosswalk_or_simulator(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    paths = case.repeat_paths[0]
    metrics = production._decode_json(
        paths["metrics_path"].read_bytes(), "test metrics"
    )
    metrics[0]["predicted_value"] = 123.0
    hostile_metrics = _retain(
        tmp_path / "target-leak-metrics.json", pretty_json_bytes(metrics)
    )
    with pytest.raises(
        production.ProductionLabelError, match="launch metrics|launch documents"
    ):
        _validate_again(case, 0, metrics_path=hostile_metrics)

    sidecar = production._decode_json(
        paths["sidecar_paths"][0].read_bytes(), "test sidecar"
    )
    sidecar["kernel_family"] = "label-derived-family"
    hostile_sidecar = _retain(
        tmp_path / "target-leak-sidecar.json", canonical_json_bytes(sidecar)
    )
    sidecars = (hostile_sidecar, paths["sidecar_paths"][1])
    with pytest.raises(production.ProductionLabelError, match="sidecars"):
        _validate_again(case, 0, sidecar_paths=sidecars)


def test_production_label_module_has_no_simulator_input_import_surface() -> None:
    tree = ast.parse((ROOT / "eval/production_labels.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert not any(name == "sim" or name.startswith("sim.") for name in imported)
    build = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_production_matched_labels"
    )
    parameter_names = {argument.arg for argument in build.args.kwonlyargs}
    assert parameter_names == {"source", "crosswalk", "repeats"}
    repeat_loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_retained_ncu_repeat"
    )
    repeat_parameters = {
        argument.arg for argument in repeat_loader.args.kwonlyargs
    }
    assert repeat_parameters == {
        "repeat_manifest_path",
        "repeat_authority_id",
        "source",
        "crosswalk",
    }
    assert not {
        "repeat_manifest_sha256",
        "report_sha256",
        "csv_path",
        "inventory_path",
        "metrics_path",
        "sidecar_paths",
        "sidecar_provenance",
    }.intersection(repeat_parameters)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("source_manifest_sha256", "source manifest"),
        ("label_set_sha256", "label set"),
    ],
)
def test_detached_authority_cannot_rebind_independently_pinned_sources(
    tmp_path: Path, field: str, message: str
) -> None:
    case = _make_case(tmp_path)
    artifacts = production.build_production_matched_label_artifacts(
        source=case.source, crosswalk=case.crosswalk, repeats=case.repeats,
    )
    hostile = copy.deepcopy(artifacts.authority_document)
    hostile["source_label_identity"][field] = "9" * 64
    hostile.pop("authority_sha256")
    hostile["authority_sha256"] = production.canonical_sha256(
        hostile, "authority_sha256"
    )
    hostile_bytes = canonical_json_bytes(hostile)
    authority_path = _retain(tmp_path / f"hostile-{field}.json", hostile_bytes)
    pin = replace(
        _authority_pin(artifacts),
        authority_file_sha256=sha256_bytes(hostile_bytes),
    )
    with pytest.raises(production.ProductionLabelError, match=message):
        production.authenticate_production_matched_label_authority(
            authority_path, authority_pin=pin
        )


def test_detached_authority_cannot_rebind_external_content_manifest(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    artifacts = production.build_production_matched_label_artifacts(
        source=case.source, crosswalk=case.crosswalk, repeats=case.repeats
    )
    hostile = copy.deepcopy(artifacts.authority_document)
    hostile["producer_content_manifest"]["content_manifest_sha256"] = "9" * 64
    hostile.pop("authority_sha256")
    hostile["authority_sha256"] = production.canonical_sha256(
        hostile, "authority_sha256"
    )
    hostile_bytes = canonical_json_bytes(hostile)
    authority_path = _retain(tmp_path / "hostile-content-manifest.json", hostile_bytes)
    pin = replace(
        _authority_pin(artifacts),
        authority_file_sha256=sha256_bytes(hostile_bytes),
    )
    with pytest.raises(
        production.ProductionLabelError, match="producer content manifest"
    ):
        production.authenticate_production_matched_label_authority(
            authority_path, authority_pin=pin
        )


def _hostile_schema_path(
    tmp_path: Path, *, original: Path, kind: str, mode: str
) -> tuple[Path, str]:
    if mode == "missing":
        return tmp_path / f"missing-{kind}-schema.json", "could not retain"
    if mode == "hash_drift":
        path = _retain(
            tmp_path / f"drifted-{kind}-schema.json",
            original.read_bytes() + b" ",
        )
        return path, "schema SHA-256 mismatch"
    path = _retain(
        tmp_path / f"substituted-{kind}-schema.json",
        original.read_bytes(),
    )
    return path, "schema path substitution"


@pytest.mark.parametrize(
    ("kind", "path_attribute"),
    (
        ("inventory", "NCU_INVENTORY_SCHEMA_PATH"),
        ("metrics", "NCU_METRICS_SCHEMA_PATH"),
    ),
)
@pytest.mark.parametrize("mode", ("missing", "hash_drift", "substitution"))
def test_repeat_validation_opens_and_pins_governing_ncu_schemas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    path_attribute: str,
    mode: str,
) -> None:
    case = _make_case(tmp_path)
    original = Path(getattr(production, path_attribute))
    hostile, match = _hostile_schema_path(
        tmp_path, original=original, kind=kind, mode=mode
    )
    monkeypatch.setattr(production, path_attribute, hostile)
    paths = case.repeat_paths[0]
    with pytest.raises(production.ProductionLabelError, match=match):
        production._validate_retained_ncu_repeat_from_manifest(
            repeat_manifest_path=paths["manifest_path"],
            authority=paths["authority"],
            source=case.source,
            crosswalk=case.crosswalk,
        )


@pytest.mark.parametrize(
    "path_attribute", ("NCU_INVENTORY_SCHEMA_PATH", "NCU_METRICS_SCHEMA_PATH")
)
def test_detached_loader_validates_ncu_schemas_before_label_payload_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_attribute: str,
) -> None:
    case = _make_case(tmp_path)
    artifacts = production.build_production_matched_label_artifacts(
        source=case.source,
        crosswalk=case.crosswalk,
        repeats=case.repeats,
    )
    labels_path, authority_path = _write_artifacts(tmp_path, artifacts)
    monkeypatch.setattr(
        production, path_attribute, tmp_path / f"missing-{path_attribute}.json"
    )

    def forbidden_label_open(_path: str | Path) -> bytes:
        raise AssertionError("label payload opened before NCU schema authority")

    monkeypatch.setattr(production, "_read_label_path", forbidden_label_open)
    with pytest.raises(production.ProductionLabelError, match="could not retain"):
        production.load_authorized_production_matched_labels(
            labels_path,
            authority_path,
            authority_pin=_authority_pin(artifacts),
        )


def test_repeat_authority_registry_rejects_self_installation() -> None:
    with pytest.raises(TypeError):
        production.PRODUCTION_REPEAT_MANIFEST_AUTHORITIES["self-installed"] = {}
    assert not production.PRODUCTION_REPEAT_MANIFEST_AUTHORITIES


def test_public_repeat_loader_uses_replaced_fixture_registry_not_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_case(tmp_path)
    paths = case.repeat_paths[0]
    authority = paths["authority"]
    fixture_registry = {
        "fixture-authority": {
            "manifest_file_sha256": authority.manifest_file_sha256,
            "repeat_id": authority.repeat_id,
            "cohort_repeat_ordinal": authority.cohort_repeat_ordinal,
            "source_manifest_sha256": authority.source_manifest_sha256,
            "producer_content_manifest_sha256": (
                authority.producer_content_manifest_sha256
            ),
            "crosswalk_file_sha256": authority.crosswalk_file_sha256,
        }
    }
    monkeypatch.setattr(
        production, "PRODUCTION_REPEAT_MANIFEST_AUTHORITIES", fixture_registry
    )
    observed = production.validate_retained_ncu_repeat(
        repeat_manifest_path=paths["manifest_path"],
        repeat_authority_id="fixture-authority",
        source=case.source,
        crosswalk=case.crosswalk,
    )
    assert observed.repeat_manifest_sha256 == authority.manifest_file_sha256


def test_current_repeat_authority_unavailable_before_manifest_open(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    with pytest.raises(
        production.ProductionLabelError,
        match="PRODUCTION_REPEAT_MANIFEST_AUTHORITY_UNAVAILABLE",
    ):
        production.validate_retained_ncu_repeat(
            repeat_manifest_path="/path/that/must/not/be-opened",
            repeat_authority_id="not-installed",
            source=case.source,
            crosswalk=case.crosswalk,
        )


@pytest.mark.parametrize(
    "registry_name", production._PRODUCTION_CROSSWALK_REGISTRY_NAMES
)
@pytest.mark.parametrize("state", ("missing", "none", "mutable", "empty"))
def test_crosswalk_authentication_rejects_invalid_registry_presence_before_payload(
    monkeypatch: pytest.MonkeyPatch,
    registry_name: str,
    state: str,
) -> None:
    from eval.contracts import launch_crosswalk as crosswalk_contract

    installed = MappingProxyType({"fixture-authority": object()})
    for name in production._PRODUCTION_CROSSWALK_REGISTRY_NAMES:
        monkeypatch.setattr(crosswalk_contract, name, installed)
    if state == "missing":
        monkeypatch.delattr(crosswalk_contract, registry_name)
        expected = "missing"
    elif state == "none":
        monkeypatch.setattr(crosswalk_contract, registry_name, None)
        expected = "not an immutable mapping proxy"
    elif state == "mutable":
        monkeypatch.setattr(
            crosswalk_contract, registry_name, {"fixture-authority": object()}
        )
        expected = "not an immutable mapping proxy"
    else:
        monkeypatch.setattr(crosswalk_contract, registry_name, MappingProxyType({}))
        expected = "empty"

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("crosswalk verifier or caller payload was accessed")

    monkeypatch.setattr(
        crosswalk_contract, "load_production_launch_crosswalk", forbidden
    )
    monkeypatch.setattr(
        production.RetainedPayloadSnapshot,
        "read_path",
        classmethod(forbidden),
    )
    with pytest.raises(production.ProductionLabelError, match=expected):
        production.authenticate_production_crosswalk(
            "/path/that/must/not/be-opened"
        )


def test_current_crosswalk_and_m1_label_lanes_remain_authority_unavailable() -> None:
    from eval.labels import load_production_matched_labels

    with pytest.raises(
        production.ProductionLabelError,
        match="crosswalk verifier authority is unavailable",
    ):
        production.authenticate_production_crosswalk("/path/that/must/not/be-opened")
    with pytest.raises(Exception, match="no production matched-label"):
        load_production_matched_labels("/path/that/must/not/be-opened")


def test_existing_synthetic_golden_bytes_remain_unchanged() -> None:
    fixture = ROOT / "eval/tests/fixtures/evaluator_mvp/matched_labels.v1.json"
    assert sha256_bytes(fixture.read_bytes()) == (
        "7556d647e864732018eae98329dd05935c03637b6abf4e637e9d11efdff8cdd4"
    )
