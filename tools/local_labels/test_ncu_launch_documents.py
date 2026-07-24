"""Offline contracts for persisted Q809 launch inventory and metric models."""

from __future__ import annotations

import builtins
import copy
import csv
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import metric_registry
from tools.local_labels import ncu_launch_documents as documents


_REPORT_SHA256 = "a" * 64
_PERCENTAGES = (
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
)
_UNIT_METRICS = ("gpc__cycles_elapsed.max",)
_FIELDS = [
    "Metric Value", "Stream", "ID", "Grid Size", "Metric Name",
    "Context", "Device", "Block Size", "Process ID", "Kernel Name",
]
_IDENTITIES = {
    "9": {
        "ID": "9", "Process ID": "100", "Context": "1", "Stream": "2",
        "Device": "0", "Kernel Name": "_Z7kernelAv",
        "Block Size": "(128, 1, 1)", "Grid Size": "(4, 1, 1)",
    },
    "2": {
        "ID": "2", "Process ID": "100", "Context": "1", "Stream": "3",
        "Device": "0", "Kernel Name": "_Z7kernelBv",
        "Block Size": "(64, 1, 1)", "Grid Size": "(8, 1, 1)",
    },
}


def _protocol() -> dict[str, object]:
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


def _row(report_id: str, metric: str, value: str) -> dict[str, str]:
    return {
        **_IDENTITIES[report_id],
        "Metric Name": metric,
        "Metric Value": value,
    }


def _csv_text() -> str:
    rows = [
        _row("9", "lts__t_sector_hit_rate.pct", "101"),
        _row("2", "lts__t_sector_hit_rate.pct", "88"),
        _row("9", "l1tex__t_sector_hit_rate.pct", "70"),
        _row("2", "l1tex__t_sector_hit_rate.pct", "70"),
        _row("9", "gpc__cycles_elapsed.max", "123"),
        _row("2", "gpc__cycles_elapsed.max", "456"),
        _row("9", "l1tex__t_sector_hit_rate.pct", "70"),
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _csv_with_extra_launch() -> str:
    rows = list(csv.DictReader(io.StringIO(_csv_text(), newline="")))
    rows.append({
        **_IDENTITIES["2"],
        "ID": "11",
        "Kernel Name": "_Z7kernelCv",
        "Metric Name": "gpc__cycles_elapsed.max",
        "Metric Value": "789",
    })
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _csv_with_identity_override(field: str) -> str:
    rows = list(csv.DictReader(io.StringIO(_csv_text(), newline="")))
    for row in rows:
        if row["ID"] == "9":
            row[field] = f"forged-{field}"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _build_documents():
    registry = metric_registry.load_registry()
    inventory = documents.build_launch_inventory_document(
        csv_text=_csv_text(),
        report_sha256=_REPORT_SHA256,
        expected_launch_count=2,
        protocol=_protocol(),
    )
    metrics = documents.build_launch_metrics_document(
        csv_text=_csv_text(),
        inventory=inventory,
        registry=registry,
        percentage_metric_names=_PERCENTAGES,
        unit_metric_names=_UNIT_METRICS,
    )
    return inventory, metrics, registry


def _reseal_inventory(inventory: dict) -> None:
    for row in inventory.get("launches", []):
        unsigned = dict(row)
        unsigned.pop("identity_sha256", None)
        row["identity_sha256"] = documents._digest(unsigned)
    inventory["launch_inventory_sha256"] = documents._digest(
        inventory.get("launches"),
    )
    unsigned = dict(inventory)
    unsigned.pop("observation_sha256", None)
    inventory["observation_sha256"] = documents._digest(unsigned)


def test_schema_authority_and_generated_documents_validate():
    inventory, metrics, registry = _build_documents()
    inventory_schema = documents.load_schema("inventory")
    metrics_schema = documents.load_schema("metrics")
    Draft202012Validator.check_schema(inventory_schema)
    Draft202012Validator.check_schema(metrics_schema)
    assert immutable.sha256_bytes(
        documents.INVENTORY_SCHEMA_PATH.read_bytes()
    ) == documents.INVENTORY_SCHEMA_SHA256
    assert immutable.sha256_bytes(
        documents.METRICS_SCHEMA_PATH.read_bytes()
    ) == documents.METRICS_SCHEMA_SHA256
    documents.validate_schema("inventory", inventory)
    documents.validate_schema("metrics", metrics)
    documents.validate_launch_inventory_document(
        inventory,
        expected_launch_count=2,
        report_sha256=_REPORT_SHA256,
        protocol=_protocol(),
    )
    documents.validate_launch_metrics_document(
        metrics,
        csv_text=_csv_text(),
        inventory=inventory,
        registry=registry,
        percentage_metric_names=_PERCENTAGES,
        unit_metric_names=_UNIT_METRICS,
    )


def test_builders_are_deterministic_and_preserve_order_diagnostics_and_seals():
    first_inventory, first_metrics, _registry = _build_documents()
    second_inventory, second_metrics, _registry = _build_documents()
    assert first_inventory == second_inventory
    assert first_metrics == second_metrics
    assert [
        row["identity"]["ID"] for row in first_inventory["launches"]
    ] == ["9", "2"]
    assert first_inventory["identity_columns"] == list(documents.IDENTITY_COLUMNS)
    for row in first_inventory["launches"]:
        unsigned = dict(row)
        observed = unsigned.pop("identity_sha256")
        assert observed == documents._digest(unsigned)
    assert first_inventory["launch_inventory_sha256"] == documents._digest(
        first_inventory["launches"],
    )
    unsigned = dict(first_inventory)
    observed = unsigned.pop("observation_sha256")
    assert observed == documents._digest(unsigned)
    assert first_metrics[0]["parser_diagnostics"] == {
        "ncu_sector_hit_rate_clamped_to_100": [
            {"metric_label": "lts__t_sector_hit_rate.pct", "raw_value": 101.0},
        ],
    }
    assert first_metrics[1]["parser_diagnostics"] == {}


def test_builders_reject_observed_insertion_sentinel_launch():
    with pytest.raises(ValueError, match="cardinality is 3; expected 2"):
        documents.build_launch_inventory_document(
            csv_text=_csv_with_extra_launch(),
            report_sha256=_REPORT_SHA256,
            expected_launch_count=2,
            protocol=_protocol(),
        )


@pytest.mark.parametrize("field", documents.IDENTITY_COLUMNS)
def test_metrics_validation_binds_all_source_identity_fields_to_inventory(field):
    inventory, metrics, registry = _build_documents()
    with pytest.raises(
        documents.LaunchDocumentError,
        match="Details CSV launch 0 identity does not match inventory",
    ):
        documents.validate_launch_metrics_document(
            metrics,
            csv_text=_csv_with_identity_override(field),
            inventory=inventory,
            registry=registry,
            percentage_metric_names=_PERCENTAGES,
            unit_metric_names=_UNIT_METRICS,
        )


def test_pretty_json_round_trip_preserves_semantics_not_object_member_order():
    inventory, metrics, registry = _build_documents()
    persisted_inventory = immutable.strict_json_loads(
        immutable.pretty_json_bytes(inventory), label="inventory",
    )
    persisted_metrics = immutable.strict_json_loads(
        immutable.pretty_json_bytes(metrics), label="metrics",
    )
    documents.validate_launch_inventory_document(
        persisted_inventory,
        expected_launch_count=2,
        report_sha256=_REPORT_SHA256,
        protocol=_protocol(),
    )
    documents.validate_launch_metrics_document(
        persisted_metrics,
        csv_text=_csv_text(),
        inventory=persisted_inventory,
        registry=registry,
        percentage_metric_names=reversed(_PERCENTAGES),
        unit_metric_names=_UNIT_METRICS,
    )


@pytest.mark.parametrize("kind", ["inventory", "metrics"])
def test_schema_authority_rejects_byte_and_path_drift(kind, tmp_path):
    spec = documents._SCHEMA_SPECS[kind]
    payload = spec["path"].read_bytes()
    with pytest.raises(documents.LaunchDocumentError, match="SHA-256 mismatch"):
        documents.load_schema(kind, payload=payload + b" ")
    with pytest.raises(documents.LaunchDocumentError, match="path is not canonical"):
        documents.load_schema(kind, schema_path=tmp_path / "schema.json")


@pytest.mark.parametrize(
    ("kind", "mutation", "message"),
    [
        ("inventory", "draft", "draft mismatch"),
        ("inventory", "id", "\\$id mismatch"),
        ("inventory", "root", "root type mismatch"),
        ("inventory", "version", "version or kind mismatch"),
        ("inventory", "kind", "version or kind mismatch"),
        ("metrics", "draft", "draft mismatch"),
        ("metrics", "id", "\\$id mismatch"),
        ("metrics", "root", "root type mismatch"),
    ],
)
def test_schema_authority_rejects_identity_and_dispatch_drift(
    kind, mutation, message, monkeypatch,
):
    schema = json.loads(documents._SCHEMA_SPECS[kind]["path"].read_text())
    if mutation == "draft":
        schema["$schema"] = "https://json-schema.org/draft/2019-09/schema"
    elif mutation == "id":
        schema["$id"] = "simbench_bw/schemas/forged.schema.json"
    elif mutation == "root":
        schema["type"] = "array" if kind == "inventory" else "object"
    elif mutation == "version":
        schema["properties"]["schema_version"]["const"] = 4
    else:
        schema["properties"]["kind"]["const"] = "forged"
    payload = immutable.pretty_json_bytes(schema)
    monkeypatch.setitem(
        documents._SCHEMA_SPECS[kind],
        "sha256",
        immutable.sha256_bytes(payload),
    )
    with pytest.raises(documents.LaunchDocumentError, match=message):
        documents.load_schema(kind, payload=payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "boolean_count",
        "sentinel",
        "controls",
        "report",
        "count",
        "duplicate_id",
        "ordinal_gap",
        "reorder",
    ],
)
def test_inventory_rejects_freshly_resealed_semantic_tampering(mutation):
    inventory, _metrics, _registry = _build_documents()
    if mutation == "boolean_count":
        inventory["requested_launch_count"] = True
    elif mutation == "sentinel":
        inventory["insertion_sentinel"]["expected_launch_count"] = 3
    elif mutation == "controls":
        inventory["ncu_controls"]["filter_mode"] = "forged"
    elif mutation == "report":
        inventory["report_sha256"] = "b" * 64
    elif mutation == "count":
        inventory["launch_count"] = 1
    elif mutation == "duplicate_id":
        inventory["launches"][1]["identity"]["ID"] = "9"
    elif mutation == "ordinal_gap":
        inventory["launches"][1]["report_launch_ordinal"] = 2
    else:
        inventory["launches"].reverse()
    _reseal_inventory(inventory)
    with pytest.raises(documents.LaunchDocumentError):
        documents.validate_launch_inventory_document(
            inventory,
            expected_launch_count=2,
            report_sha256=_REPORT_SHA256,
            protocol=_protocol(),
        )


@pytest.mark.parametrize("seal", ["row", "inventory", "root"])
def test_inventory_rejects_direct_seal_tampering(seal):
    inventory, _metrics, _registry = _build_documents()
    if seal == "row":
        inventory["launches"][0]["identity_sha256"] = "0" * 64
        inventory["launch_inventory_sha256"] = documents._digest(
            inventory["launches"],
        )
        unsigned = dict(inventory)
        unsigned.pop("observation_sha256")
        inventory["observation_sha256"] = documents._digest(unsigned)
    elif seal == "inventory":
        inventory["launch_inventory_sha256"] = "0" * 64
        unsigned = dict(inventory)
        unsigned.pop("observation_sha256")
        inventory["observation_sha256"] = documents._digest(unsigned)
    else:
        inventory["observation_sha256"] = "0" * 64
    with pytest.raises(documents.LaunchDocumentError, match="sha256 mismatch"):
        documents.validate_launch_inventory_document(
            inventory,
            expected_launch_count=2,
            report_sha256=_REPORT_SHA256,
            protocol=_protocol(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_row",
        "ordinal",
        "report_id",
        "missing_fraction",
        "extra_fraction",
        "fraction_high",
        "fraction_boolean",
        "unit",
        "negative_unit",
        "nan",
        "infinity",
        "unknown_diagnostic",
        "duplicate_diagnostic",
        "diagnostic_window",
        "diagnostic_value",
    ],
)
def test_metrics_reject_cross_document_and_value_tampering(mutation):
    inventory, metrics, registry = _build_documents()
    if mutation == "drop_row":
        metrics.pop()
    elif mutation == "ordinal":
        metrics[1]["report_launch_ordinal"] = 2
    elif mutation == "report_id":
        metrics[1]["report_id"] = "9"
    elif mutation == "missing_fraction":
        metrics[0]["fraction_values"].pop(_PERCENTAGES[0])
    elif mutation == "extra_fraction":
        metrics[0]["fraction_values"]["synthetic"] = 0.5
    elif mutation == "fraction_high":
        metrics[0]["fraction_values"][_PERCENTAGES[0]] = 1.1
    elif mutation == "fraction_boolean":
        metrics[0]["fraction_values"][_PERCENTAGES[0]] = True
    elif mutation == "unit":
        metrics[0]["unit_tagged_metrics"][_UNIT_METRICS[0]]["unit"] = "ns"
    elif mutation == "negative_unit":
        metrics[0]["unit_tagged_metrics"][_UNIT_METRICS[0]]["value"] = -1
    elif mutation == "nan":
        metrics[0]["unit_tagged_metrics"][_UNIT_METRICS[0]]["value"] = float("nan")
    elif mutation == "infinity":
        metrics[0]["unit_tagged_metrics"][_UNIT_METRICS[0]]["value"] = float("inf")
    elif mutation == "unknown_diagnostic":
        metrics[0]["parser_diagnostics"]["synthetic"] = []
    elif mutation == "duplicate_diagnostic":
        records = metrics[0]["parser_diagnostics"][
            "ncu_sector_hit_rate_clamped_to_100"
        ]
        records.append(copy.deepcopy(records[0]))
    elif mutation == "diagnostic_window":
        metrics[0]["parser_diagnostics"][
            "ncu_sector_hit_rate_clamped_to_100"
        ][0]["raw_value"] = 99
    else:
        metrics[0]["fraction_values"][_PERCENTAGES[0]] = 0.99
    with pytest.raises((documents.LaunchDocumentError, immutable.ImmutableArtifactError)):
        documents.validate_launch_metrics_document(
            metrics,
            csv_text=_csv_text(),
            inventory=inventory,
            registry=registry,
            percentage_metric_names=_PERCENTAGES,
            unit_metric_names=_UNIT_METRICS,
        )


@pytest.mark.parametrize(
    "mutation", ["valid_diagnostic_raw", "drop_diagnostic", "metric_value"],
)
def test_metrics_reject_values_not_reproduced_from_source_csv(mutation):
    inventory, metrics, registry = _build_documents()
    if mutation == "valid_diagnostic_raw":
        metrics[0]["parser_diagnostics"][
            "ncu_sector_hit_rate_clamped_to_100"
        ][0]["raw_value"] = 102.0
    elif mutation == "drop_diagnostic":
        metrics[0]["parser_diagnostics"] = {}
    else:
        metrics[0]["unit_tagged_metrics"][_UNIT_METRICS[0]]["value"] = 124.0
    with pytest.raises(
        documents.LaunchDocumentError,
        match="launch metrics do not match Details CSV extraction",
    ):
        documents.validate_launch_metrics_document(
            metrics,
            csv_text=_csv_text(),
            inventory=inventory,
            registry=registry,
            percentage_metric_names=_PERCENTAGES,
            unit_metric_names=_UNIT_METRICS,
        )


@pytest.mark.parametrize(
    ("percentage_metrics", "unit_metrics"),
    [
        ((_PERCENTAGES[0], _PERCENTAGES[0]), _UNIT_METRICS),
        (_PERCENTAGES, (_UNIT_METRICS[0], _UNIT_METRICS[0])),
        (_PERCENTAGES, ("synthetic__unknown.sum",)),
    ],
)
def test_metrics_builder_rejects_duplicate_or_unknown_requests(
    percentage_metrics, unit_metrics,
):
    inventory, _metrics, registry = _build_documents()
    with pytest.raises(ValueError):
        documents.build_launch_metrics_document(
            csv_text=_csv_text(),
            inventory=inventory,
            registry=registry,
            percentage_metric_names=percentage_metrics,
            unit_metric_names=unit_metrics,
        )


def test_metrics_builder_rejects_wrong_registry_carrier_requests():
    inventory, _metrics, registry = _build_documents()
    with pytest.raises(
        documents.LaunchDocumentError, match="lacks flat percentage",
    ):
        documents.build_launch_metrics_document(
            csv_text=_csv_text(),
            inventory=inventory,
            registry=registry,
            percentage_metric_names=(
                "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
            ),
            unit_metric_names=_UNIT_METRICS,
        )
    with pytest.raises(
        documents.LaunchDocumentError, match="lacks unit-tagged",
    ):
        documents.build_launch_metrics_document(
            csv_text=_csv_text(),
            inventory=inventory,
            registry=registry,
            percentage_metric_names=_PERCENTAGES,
            unit_metric_names=(_PERCENTAGES[0],),
        )


@pytest.mark.parametrize(
    ("metric", "changes"),
    [
        (_PERCENTAGES[0], {"valid_max": 0.9}),
        (_UNIT_METRICS[0], {"valid_max": 100.0}),
    ],
)
def test_metrics_validator_honors_registry_value_ranges(metric, changes):
    inventory, metrics, registry = _build_documents()
    original = registry.get_by_ncu_metric(metric)

    class RegistryOverride:
        def get_by_ncu_metric(self, name):
            if name == metric:
                return replace(original, **changes)
            return registry.get_by_ncu_metric(name)

    with pytest.raises(documents.LaunchDocumentError, match="registry range"):
        documents.validate_launch_metrics_document(
            metrics,
            csv_text=_csv_text(),
            inventory=inventory,
            registry=RegistryOverride(),
            percentage_metric_names=_PERCENTAGES,
            unit_metric_names=_UNIT_METRICS,
        )


def test_builders_do_not_mutate_inputs_or_reach_process_or_filesystem(
    monkeypatch, tmp_path,
):
    csv_text = _csv_text()
    protocol = _protocol()
    protocol_before = copy.deepcopy(protocol)
    percentage_metrics = list(_PERCENTAGES)
    unit_metrics = list(_UNIT_METRICS)
    percentage_metrics_before = copy.deepcopy(percentage_metrics)
    unit_metrics_before = copy.deepcopy(unit_metrics)
    registry = metric_registry.load_registry()
    documents.load_schema("inventory")
    documents.load_schema("metrics")

    def unexpected(*_args, **_kwargs):
        pytest.fail("pure launch document builder reached process or filesystem I/O")

    monkeypatch.setattr(documents.Path, "open", unexpected)
    monkeypatch.setattr(documents.Path, "read_bytes", unexpected)
    monkeypatch.setattr(documents.Path, "read_text", unexpected)
    monkeypatch.setattr(documents.Path, "write_bytes", unexpected)
    monkeypatch.setattr(documents.Path, "write_text", unexpected)
    monkeypatch.setattr("subprocess.run", unexpected)
    monkeypatch.setattr(builtins, "open", unexpected)
    inventory = documents.build_launch_inventory_document(
        csv_text=csv_text,
        report_sha256=_REPORT_SHA256,
        expected_launch_count=2,
        protocol=protocol,
    )
    inventory_before = copy.deepcopy(inventory)
    metrics = documents.build_launch_metrics_document(
        csv_text=csv_text,
        inventory=inventory,
        registry=registry,
        percentage_metric_names=percentage_metrics,
        unit_metric_names=unit_metrics,
    )
    assert protocol == protocol_before
    assert inventory == inventory_before
    assert percentage_metrics == percentage_metrics_before
    assert unit_metrics == unit_metrics_before
    assert metrics
    assert list(tmp_path.iterdir()) == []
