"""Row-closure tests for the current SOL live coverage inventory."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import jsonschema
import pytest


_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import sol_live_workload_coverage_inventory as live  # noqa: E402


def _source(index: int, seed: str) -> dict:
    return {
        "source_workload_index": index,
        "source_workload_uuid": (
            f"{index + 1:08x}-1234-5123-8123-{index + 1:012x}"
        ),
        "source_axes": {"n": index + 1},
        "source_row_sha256": hashlib.sha256(
            f"{seed}:{index}".encode()
        ).hexdigest(),
        "source_line": index + 1,
    }


def _diagnostic(source: dict, status: str | None) -> dict:
    present = status is not None
    return {
        "workload_index": source["source_workload_index"],
        "source_line": source["source_line"],
        "uuid": source["source_workload_uuid"],
        "axes": source["source_axes"],
        "workload_sha256": source["source_row_sha256"],
        "diagnostic_present": present,
        "status": status,
    }


def _matrix_row(slug: str, source: dict, cohort: str, variant: str) -> dict:
    native = cohort == "native"
    partial = cohort == "partial_native_row"
    return {
        "key": {
            "slug": slug,
            "source_workload_uuid": source["source_workload_uuid"],
            "variant_id": variant,
        },
        "cohort": cohort,
        "source_workload_index": source["source_workload_index"],
        "source_workload_uuid": source["source_workload_uuid"],
        "source_axes": source["source_axes"],
        "source_row_sha256": source["source_row_sha256"],
        "problem_level_runnable": native,
        "native_shape": native or partial,
        "native_predictive": native,
    }


def _fixture() -> tuple[dict, dict, dict, dict, dict, dict]:
    layout = {
        "problem_001_arch": ("Quant", 1),
        "problem_002_native": ("L1", 2),
        "problem_003_partial": ("L2", 2),
        "problem_004_proxy": ("FIB", 2),
        "problem_005_both": ("L1", 1),
    }
    problems = {
        slug: {
            "category": category,
            "rows": [_source(index, slug) for index in range(count)],
        }
        for slug, (category, count) in layout.items()
    }
    excluded = {
        "problem_001_arch": {
            "category": "Quant", "reason": "nvfp4_e2m1_contract",
        }
    }
    included = {
        slug: {"category": source["category"]}
        for slug, source in problems.items() if slug not in excluded
    }

    run_cells = []
    status_rows = {
        "problem_002_native": ("runnable", ["PASSED", "PASSED"]),
        "problem_003_partial": ("correctness_failed", ["PASSED", "FAILED"]),
        "problem_004_proxy": ("host_capacity_oom", [None, "FAILED"]),
        "problem_005_both": ("host_capacity_oom", ["PASSED"]),
    }
    for slug, (status, diagnostics) in status_rows.items():
        run_cells.append({
            "slug": slug,
            "status": status,
            "workload_diagnostics": [
                _diagnostic(source, diagnostic)
                for source, diagnostic in zip(problems[slug]["rows"], diagnostics)
            ],
        })
    runnability = {"cells": run_cells}

    proxy0 = problems["problem_004_proxy"]["rows"][0]
    proxy1 = problems["problem_004_proxy"]["rows"][1]
    both0 = problems["problem_005_both"]["rows"][0]
    shrink = {"cells": [
        {
            "slug": "problem_004_proxy",
            "classification": "reduced_representative",
            "row_plans": [
                {
                    "slug": "problem_004_proxy",
                    "source_workload_index": 0,
                    "source_workload_uuid": proxy0["source_workload_uuid"],
                    "planning_status": "proxy_planned",
                    "correctness_result": "passed",
                    "reduced_run_status": "runnable",
                    "planning_reason": None,
                    "variant_id": "a" * 64,
                },
                {
                    "slug": "problem_004_proxy",
                    "source_workload_index": 1,
                    "source_workload_uuid": proxy1["source_workload_uuid"],
                    "planning_status": "needs_model",
                    "correctness_result": "not_applicable",
                    "reduced_run_status": None,
                    "planning_reason": "geometry_model_unavailable",
                    "variant_id": "b" * 64,
                },
            ],
        },
        {
            "slug": "problem_005_both",
            "classification": "reduced_representative",
            "row_plans": [{
                "slug": "problem_005_both",
                "source_workload_index": 0,
                "source_workload_uuid": both0["source_workload_uuid"],
                "planning_status": "proxy_planned",
                "correctness_result": "passed",
                "reduced_run_status": "runnable",
                "planning_reason": None,
                "variant_id": "c" * 64,
            }],
        },
    ]}

    native = problems["problem_002_native"]["rows"]
    partial = problems["problem_003_partial"]["rows"][0]
    matrix = {"rows": [
        _matrix_row("problem_002_native", native[0], "native", "native"),
        _matrix_row("problem_002_native", native[1], "native", "native"),
        _matrix_row("problem_003_partial", partial, "partial_native_row", "d" * 64),
        _matrix_row("problem_004_proxy", proxy0, "reduced_proxy", "a" * 64),
        _matrix_row("problem_005_both", both0, "partial_native_row", "e" * 64),
        _matrix_row("problem_005_both", both0, "reduced_proxy", "c" * 64),
    ]}
    return problems, included, excluded, runnability, shrink, matrix


def _build(fixture=None):
    problems, included, excluded, runnability, shrink, matrix = (
        fixture or _fixture()
    )
    return live.build_row_ledger(
        problems=problems, included=included, excluded_scope=excluded,
        runnability=runnability, shrink_document=shrink, matrix=matrix,
    )


def test_every_source_row_gets_one_truthful_live_schedule_state():
    rows, problems, counts = _build()
    assert len(rows) == 8
    assert len(problems) == 5
    assert counts == {
        "architecture_inapplicable": 1,
        "in_scope_unscheduled": 2,
        "native_scheduled": 2,
        "partial_native_scheduled": 1,
        "reduced_proxy_scheduled": 1,
        "partial_native_and_reduced_proxy_scheduled": 1,
    }
    by_identity = {
        (row["slug"], row["source_workload_index"]): row for row in rows
    }
    arch = by_identity[("problem_001_arch", 0)]
    assert arch["schedule_state"] == "architecture_inapplicable"
    assert arch["unscheduled_reason"]["architecture_reason"] == (
        "nvfp4_e2m1_contract"
    )
    partial_failed = by_identity[("problem_003_partial", 1)]
    assert partial_failed["schedule_state"] == "in_scope_unscheduled"
    assert partial_failed["unscheduled_reason"]["diagnostic_status"] == "FAILED"
    proxy_only = by_identity[("problem_004_proxy", 0)]
    assert proxy_only["reduced_proxy_only"] is True
    assert proxy_only["native_row_passed"] is False
    both = by_identity[("problem_005_both", 0)]
    assert both["schedule_cohorts"] == ["partial_native_row", "reduced_proxy"]
    assert both["unscheduled_reason"] is None


def test_unscheduled_reason_hash_binds_the_complete_reason_object():
    rows, _problems, _counts = _build()
    reason = next(
        row["unscheduled_reason"] for row in rows
        if row["schedule_state"] == "in_scope_unscheduled"
    )
    unsigned = {key: value for key, value in reason.items()
                if key != "reason_evidence_sha256"}
    assert reason["reason_evidence_sha256"] == live.canonical_sha256(unsigned)


def test_passed_native_diagnostic_cannot_be_silently_omitted():
    fixture = list(_fixture())
    fixture[-1] = copy.deepcopy(fixture[-1])
    fixture[-1]["rows"] = fixture[-1]["rows"][1:]
    with pytest.raises(live.LiveCoverageError, match="passed-row scheduling"):
        _build(tuple(fixture))


def test_admitted_proxy_plan_cannot_be_silently_omitted():
    fixture = list(_fixture())
    fixture[-1] = copy.deepcopy(fixture[-1])
    fixture[-1]["rows"] = [
        row for row in fixture[-1]["rows"]
        if not (
            row["key"]["slug"] == "problem_004_proxy"
            and row["cohort"] == "reduced_proxy"
        )
    ]
    with pytest.raises(live.LiveCoverageError, match="proxy scheduling"):
        _build(tuple(fixture))


def test_architecture_inapplicable_row_cannot_enter_live_matrix():
    fixture = list(_fixture())
    fixture[-1] = copy.deepcopy(fixture[-1])
    source = fixture[0]["problem_001_arch"]["rows"][0]
    fixture[-1]["rows"].append(
        _matrix_row("problem_001_arch", source, "native", "native")
    )
    with pytest.raises(live.LiveCoverageError, match="out-of-scope"):
        _build(tuple(fixture))


def test_matrix_row_must_bind_exact_source_row_bytes():
    fixture = list(_fixture())
    fixture[-1] = copy.deepcopy(fixture[-1])
    fixture[-1]["rows"][0]["source_row_sha256"] = "0" * 64
    with pytest.raises(live.LiveCoverageError, match="source row binding"):
        _build(tuple(fixture))


def test_live_inventory_schema_is_closed_and_binds_corpus_tripwires():
    schema = json.loads((_REPO / live.SCHEMA_PATH).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["counts"]["properties"]["problems"] == {
        "const": 235
    }
    assert schema["$defs"]["counts"]["properties"][
        "source_workload_rows"
    ] == {"const": 3957}
    assert schema["$defs"]["stateCounts"]["properties"][
        "architecture_inapplicable"
    ] == {"const": 537}


def _owner(seed: str) -> dict:
    return {
        "name": f".{seed}-owner.json",
        "sha256": seed[0] * 64,
        "owner_seal_sha256": seed[-1] * 64,
    }


def test_dedicated_plan_owner_is_closed_and_resume_stable(tmp_path):
    root = tmp_path / "plan"
    root.mkdir(mode=0o700)
    first = live.initialize_plan_owner(
        plan_root=root, source_commit="1" * 40,
        matrix_name="matrix.json", inventory_name="live-inventory.json",
        runnability_owner=_owner("ab"), shrink_owner=_owner("cd"),
    )
    assert live.runnability_tool.external_root_names(
        live.runnability_tool.validate_external_campaign_root(root)
    ) == {live.PLAN_OWNER_FILENAME}
    second = live.initialize_plan_owner(
        plan_root=root, source_commit="1" * 40,
        matrix_name="matrix.json", inventory_name="live-inventory.json",
        runnability_owner=_owner("ab"), shrink_owner=_owner("cd"),
        allow_create=False,
    )
    assert second == first


def test_plan_owner_rejects_extra_children_and_name_alias(tmp_path):
    root = tmp_path / "plan"
    root.mkdir(mode=0o700)
    live.initialize_plan_owner(
        plan_root=root, source_commit="1" * 40,
        matrix_name="matrix.json", inventory_name="live-inventory.json",
        runnability_owner=_owner("ab"), shrink_owner=_owner("cd"),
    )
    identity = live.runnability_tool.validate_external_campaign_root(root)
    live.runnability_tool.publish_external_child(identity, "unexpected", b"x")
    with pytest.raises(live.LiveCoverageError, match="undeclared children"):
        live.initialize_plan_owner(
            plan_root=root, source_commit="1" * 40,
            matrix_name="matrix.json", inventory_name="live-inventory.json",
            runnability_owner=_owner("ab"), shrink_owner=_owner("cd"),
            allow_create=False,
        )
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    other_identity = live.runnability_tool.validate_external_campaign_root(other)
    with pytest.raises(live.LiveCoverageError, match="must be distinct"):
        live._require_distinct_owner_roots(
            other_identity, other_identity, identity
        )
    nested = other / "nested"
    nested.mkdir(mode=0o700)
    nested_identity = live.runnability_tool.validate_external_campaign_root(nested)
    with pytest.raises(live.LiveCoverageError, match="must be distinct"):
        live._require_distinct_owner_roots(
            other_identity, nested_identity, identity
        )
    with pytest.raises(live.LiveCoverageError, match="must be distinct"):
        live._plan_owner_document(
            source_commit="1" * 40, matrix_name="same.json",
            inventory_name="same.json", runnability_owner=_owner("ab"),
            shrink_owner=_owner("cd"),
        )


def _delivery_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    root = tmp_path / "plan"
    root.mkdir(mode=0o700)
    matrix = root / "matrix.json"
    inventory = root / "live-inventory.json"
    marker = live.initialize_plan_owner(
        plan_root=root, source_commit="1" * 40,
        matrix_name=matrix.name, inventory_name=inventory.name,
        runnability_owner=_owner("ab"), shrink_owner=_owner("cd"),
    )
    identity = live.runnability_tool.validate_external_campaign_root(root)
    matrix_payload = b'{"matrix":"bound"}\n'
    live.runnability_tool.publish_external_child(
        identity, matrix.name, matrix_payload
    )
    document = {
        "source_commit": "1" * 40,
        "inputs": {
            "matrix": {
                "name": matrix.name,
                "sha256": live.sha256_bytes(matrix_payload),
                "owner": marker,
            }
        },
        "counts": {
            "source_workload_rows": 1,
            "source_rows_with_any_schedule": 1,
        },
    }
    return document, root, inventory


def test_plan_owner_requires_lowercase_full_commit(tmp_path):
    root = tmp_path / "plan"
    root.mkdir(mode=0o700)
    for invalid in ("1" * 39, "A" * 40, "not-a-commit"):
        with pytest.raises(live.LiveCoverageError, match="lowercase 40-hex"):
            live.initialize_plan_owner(
                plan_root=root, source_commit=invalid,
                matrix_name="matrix.json", inventory_name="inventory.json",
                runnability_owner=_owner("ab"), shrink_owner=_owner("cd"),
            )


def test_inventory_delivery_is_byte_exact_and_resume_stable(tmp_path):
    document, root, inventory = _delivery_fixture(tmp_path)
    assert live._deliver(document, root=root, emit=inventory, check=None) == 0
    assert inventory.read_bytes() == live.canonical_json_bytes(document)
    assert live._deliver(document, root=root, emit=None, check=inventory) == 0


def test_inventory_delivery_detects_marker_replacement_during_publish(
    tmp_path, monkeypatch,
):
    document, root, inventory = _delivery_fixture(tmp_path)
    marker = root / live.PLAN_OWNER_FILENAME
    original_publish = live.runnability_tool.publish_external_child

    def publish_then_replace_marker(identity, name, payload, **kwargs):
        result = original_publish(identity, name, payload, **kwargs)
        if name == inventory.name:
            marker.chmod(0o600)
            marker.write_bytes(marker.read_bytes() + b"\n")
            marker.chmod(0o444)
        return result

    monkeypatch.setattr(
        live.runnability_tool,
        "publish_external_child",
        publish_then_replace_marker,
    )
    with pytest.raises(live.LiveCoverageError, match="during inventory publication"):
        live._deliver(document, root=root, emit=inventory, check=None)


def test_inventory_check_detects_matrix_replacement_during_read(
    tmp_path, monkeypatch,
):
    document, root, inventory = _delivery_fixture(tmp_path)
    assert live._deliver(document, root=root, emit=inventory, check=None) == 0
    matrix = root / document["inputs"]["matrix"]["name"]
    original_read = live.runnability_tool.read_external_child
    matrix_reads = 0

    def read_then_replace_matrix(identity, name, **kwargs):
        nonlocal matrix_reads
        payload = original_read(identity, name, **kwargs)
        if name == matrix.name:
            matrix_reads += 1
            if matrix_reads == 1:
                matrix.chmod(0o600)
                matrix.write_bytes(payload + b"\n")
                matrix.chmod(0o444)
        return payload

    monkeypatch.setattr(
        live.runnability_tool, "read_external_child", read_then_replace_matrix
    )
    with pytest.raises(live.LiveCoverageError, match="during inventory check"):
        live._deliver(document, root=root, emit=None, check=inventory)
