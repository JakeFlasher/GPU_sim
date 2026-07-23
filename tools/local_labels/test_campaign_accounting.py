from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.local_labels import campaign_accounting as ca
from tools.local_labels import campaign_done_manifest as dm


def _matrix() -> dict:
    return {
        "source_commit": "a" * 40,
        "cells": [
            {
                "identity": {"roster_source": "portable_lcd", "slug": "a"},
                "disposition": "runnable",
                "authority_cell_sha256": "1" * 64,
            },
            {
                "identity": {"roster_source": "sol_execbench", "slug": "b"},
                "disposition": "runnable",
                "authority_cell_sha256": "2" * 64,
            },
            {
                "identity": {"roster_source": "simbench_bw", "slug": "x"},
                "disposition": "excluded",
                "authority_cell_sha256": "3" * 64,
            },
        ],
    }


def _done(source: str, slug: str, authority: str, *, disposition="admitted") -> dict:
    return {
        "chunk_id": f"chunk-{slug}",
        "source_commit_pre": "b" * 40,
        "cell_results": [{
            "roster_source": source,
            "slug": slug,
            "authority_cell_sha256": authority,
            "disposition": disposition,
            "typed_reason": None if disposition == "admitted" else "trace_unsupported",
            "prior_preparation_failures": [],
            "prior_capture_defect_attempts": [],
            "definition_equivalence": None,
            "definition_equivalence_attestation_sha256": None,
        }],
    }


def _current_attested_matrix() -> dict:
    matrix = _matrix()
    fingerprint = dm.REPO_ROOT / dm.SOL_NATIVE_ENV_PIN_PATH
    pin = {
        "path": dm.SOL_NATIVE_ENV_PIN_PATH,
        "sha256": hashlib.sha256(fingerprint.read_bytes()).hexdigest(),
        "native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
    }
    matrix["input_pins"] = {dm.SOL_NATIVE_ENV_PIN_SECTION: pin}
    sol_row = next(
        row for row in matrix["cells"]
        if row["identity"]["roster_source"] == "sol_execbench"
    )
    sol_row["invocation"] = {
        "adapter_contract": {
            "native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
            "environment": dict(dm.SOL_CERTIFIED_ENVIRONMENT),
        },
    }
    return matrix


def _attach_sol_attestation(manifest: dict, matrix: dict, root: Path) -> Path:
    row = manifest["cell_results"][0]
    identity = {"roster_source": row["roster_source"], "slug": row["slug"]}
    pin = matrix["input_pins"][dm.SOL_NATIVE_ENV_PIN_SECTION]
    certified = json.loads(
        (dm.REPO_ROOT / dm.SOL_NATIVE_ENV_PIN_PATH).read_text()
    )
    unsigned = {
        "schema_version": dm.SOL_NATIVE_ENV_ATTESTATION_SCHEMA_VERSION,
        "kind": dm.SOL_NATIVE_ENV_ATTESTATION_KIND,
        "identity": identity,
        "certified_native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
        "observed_native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
        "certified_hash_equal": True,
        "degraded": False,
        "missing_components": [],
        "components": certified["components"],
        "components_sha256": dm._canonical_sha256(certified["components"]),
        "compiler_identity": {
            "requested": dm.SOL_CERTIFIED_NVCC_CCBIN,
            "resolved_realpath": "/usr/bin/x86_64-linux-gnu-g++-14",
            "executable_sha256": "d" * 64,
            "version_line": "g++-14 (GCC) 14.3.1 20260214",
        },
        "matrix_pin": pin,
    }
    document = {**unsigned, "attestation_sha256": dm._canonical_sha256(unsigned)}
    rel = f"cells/sol_execbench__{row['slug']}/{dm.SOL_NATIVE_ENV_ATTESTATION_FILENAME}"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    row["artifacts"] = [{
        "artifact_kind": "sol_capture_evidence",
        "relative_path": rel,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }]
    manifest["output_root"] = str(root)
    return path


def test_partial_historical_campaign_is_integral_but_incomplete(tmp_path: Path):
    rows = [(tmp_path / "a" / "DONE.json", _done(
        "portable_lcd", "a", "1" * 64), [])]
    result = ca.audit_loaded_manifests(_matrix(), rows)
    assert result["integrity_ok"] is True
    assert result["complete"] is False
    assert result["counts"]["admitted_unique_current"] == 1
    assert result["missing"] == [{"roster_source": "sol_execbench", "slug": "b"}]


def test_exact_cross_commit_set_is_complete(tmp_path: Path):
    rows = [
        (tmp_path / "a" / "DONE.json", _done(
            "portable_lcd", "a", "1" * 64), []),
        (tmp_path / "b" / "DONE.json", _done(
            "sol_execbench", "b", "2" * 64), []),
    ]
    result = ca.audit_loaded_manifests(_matrix(), rows)
    assert result["integrity_ok"] is True
    assert result["complete"] is True
    assert result["missing"] == []


def test_duplicate_stale_extra_nonadmitted_and_bad_seal_fail_closed(tmp_path: Path):
    duplicate = _done("portable_lcd", "a", "1" * 64)
    stale = _done("sol_execbench", "b", "f" * 64)
    extra = _done("simbench_bw", "x", "3" * 64)
    rows = [
        (tmp_path / "a" / "DONE.json", duplicate, []),
        (tmp_path / "a2" / "DONE.json", copy.deepcopy(duplicate), []),
        (tmp_path / "b" / "DONE.json", stale, []),
        (tmp_path / "x" / "DONE.json", extra, []),
        (tmp_path / "bad" / "DONE.json", {}, ["tampered"]),
    ]
    result = ca.audit_loaded_manifests(_matrix(), rows)
    assert result["integrity_ok"] is False
    assert result["complete"] is False
    assert result["counts"]["duplicate_current"] == 1
    assert result["counts"]["stale_authority"] == 1
    assert result["counts"]["extra_or_excluded"] == 1
    assert result["counts"]["seal_failed_chunks"] == 1


def test_certified_trace_unavailable_is_typed_and_fully_accounted(tmp_path: Path):
    rows = [
        (tmp_path / "a" / "DONE.json", _done(
            "portable_lcd", "a", "1" * 64), []),
        (tmp_path / "b" / "DONE.json", _done(
            "sol_execbench", "b", "2" * 64, disposition="failed"), []),
    ]
    result = ca.audit_loaded_manifests(_matrix(), rows)
    assert result["integrity_ok"] is True
    assert result["complete"] is True
    assert result["counts"]["typed_unavailable_unique_current"] == 1
    assert result["missing"] == []


def test_uncertified_failure_is_invalid_and_incomplete(tmp_path: Path):
    bad = _done("sol_execbench", "b", "2" * 64, disposition="failed")
    bad["cell_results"][0]["typed_reason"] = "correctness_failed"
    rows = [
        (tmp_path / "a" / "DONE.json", _done(
            "portable_lcd", "a", "1" * 64), []),
        (tmp_path / "b" / "DONE.json", bad, []),
    ]
    result = ca.audit_loaded_manifests(_matrix(), rows)
    assert result["integrity_ok"] is False
    assert result["counts"]["invalid_outcome_current"] == 1


def test_current_matrix_counts_only_semantically_attested_sol(tmp_path: Path):
    matrix = _current_attested_matrix()
    sol = _done("sol_execbench", "b", "2" * 64)
    _attach_sol_attestation(sol, matrix, tmp_path / "sol")
    rows = [
        (tmp_path / "a" / "DONE.json", _done(
            "portable_lcd", "a", "1" * 64), []),
        (tmp_path / "b" / "DONE.json", sol, []),
    ]
    result = ca.audit_loaded_manifests(matrix, rows)
    assert result["integrity_ok"] is True
    assert result["complete"] is True
    assert result["counts"]["accounted_unique_current"] == 2


def test_current_matrix_rejects_historical_sol_without_attestation(tmp_path: Path):
    matrix = _current_attested_matrix()
    sol = _done("sol_execbench", "b", "2" * 64)
    sol["output_root"] = str(tmp_path / "sol")
    result = ca.audit_loaded_manifests(
        matrix, [(tmp_path / "b" / "DONE.json", sol, [])],
    )
    assert result["integrity_ok"] is False
    assert result["complete"] is False
    assert result["counts"]["seal_failed_chunks"] == 1
    assert result["chunks"][0]["current_matrix_attestation_problem_count"] > 0
    assert any(
        "requires exactly one sealed" in problem
        for problem in result["seal_failures"][0]["problems"]
    )


def test_accounting_rejects_self_consistently_resealed_sol_env(tmp_path: Path):
    matrix = _current_attested_matrix()
    sol = _done("sol_execbench", "b", "2" * 64)
    path = _attach_sol_attestation(sol, matrix, tmp_path / "sol")
    document = json.loads(path.read_text())
    document["components"]["torch_version"] = "replacement"
    replacement_hash = dm._canonical_sha256(document["components"])
    document["components_sha256"] = replacement_hash
    document["observed_native_env_hash"] = replacement_hash
    document["certified_native_env_hash"] = replacement_hash
    document["matrix_pin"]["native_env_hash"] = replacement_hash
    unsigned = {
        key: value for key, value in document.items()
        if key != "attestation_sha256"
    }
    document["attestation_sha256"] = dm._canonical_sha256(unsigned)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    artifact = sol["cell_results"][0]["artifacts"][0]
    artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact["size_bytes"] = path.stat().st_size
    result = ca.audit_loaded_manifests(
        matrix, [(tmp_path / "b" / "DONE.json", sol, [])],
    )
    assert result["integrity_ok"] is False
    assert result["counts"]["seal_failed_chunks"] == 1
    assert any(
        "certified_native_env_hash" in problem
        for problem in result["seal_failures"][0]["problems"]
    )


def test_current_pin_does_not_invalidate_portable_historical_bootstrap(
    tmp_path: Path,
):
    matrix = _current_attested_matrix()
    rows = [(tmp_path / "a" / "DONE.json", _done(
        "portable_lcd", "a", "1" * 64), [])]
    result = ca.audit_loaded_manifests(matrix, rows)
    assert result["integrity_ok"] is True
    assert result["counts"]["admitted_unique_current"] == 1
    assert result["counts"]["seal_failed_chunks"] == 0


def _admission_matrix() -> dict:
    head = "a" * 40
    runnable = []
    excluded = []
    for index in range(ca.EXPECTED_RUNNABLE):
        runnable.append({
            "identity": {
                "roster_source": "portable_lcd",
                "slug": f"runnable-{index:03d}",
            },
            "disposition": "runnable",
            "authority_cell_sha256": f"{index + 1:064x}",
            "capture_readiness": {"status": "ready", "blocking_reasons": []},
            "trace_budget": {
                "status": "declared", "max_trace_bytes": 1,
                "max_wall_time_s": 1,
            },
        })
    for index in range(ca.EXPECTED_EXCLUDED):
        excluded.append({
            "identity": {
                "roster_source": "simbench_bw", "slug": f"excluded-{index:03d}",
            },
            "disposition": "excluded",
            "authority_cell_sha256": f"{index + 1000:064x}",
        })
    chunk_members = [[] for _ in range(ca.EXPECTED_CHUNKS)]
    for index, row in enumerate(runnable):
        chunk_members[index % ca.EXPECTED_CHUNKS].append(copy.deepcopy(row["identity"]))
    matrix = {
        "source_commit": head,
        "authority": {"path": "authority.json"},
        "counts": {
            "universe": ca.EXPECTED_UNIVERSE,
            "runnable": ca.EXPECTED_RUNNABLE,
            "excluded": ca.EXPECTED_EXCLUDED,
            "capture_ready": ca.EXPECTED_RUNNABLE,
            "capture_blocked": ca.EXPECTED_BLOCKED,
            "budget_declared": ca.EXPECTED_EXPLICIT_BUDGETS,
            "budget_missing": 0,
            "chunks": ca.EXPECTED_CHUNKS,
        },
        "chunks": [
            {"chunk_id": f"chunk-{index:02d}", "cells": members}
            for index, members in enumerate(chunk_members)
        ],
        "cells": runnable + excluded,
    }
    matrix["matrix_seal_sha256"] = ca.matrix_builder.canonical_sha256(matrix)
    return matrix


def _reseal(matrix: dict) -> None:
    matrix.pop("matrix_seal_sha256", None)
    matrix["matrix_seal_sha256"] = ca.matrix_builder.canonical_sha256(matrix)


def _configure_admission_dependencies(monkeypatch, matrix: dict) -> None:
    import jsonschema
    monkeypatch.setattr(jsonschema, "validate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ca.matrix_builder, "validate_matrix", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(ca, "_git_head", lambda _root: matrix["source_commit"])
    monkeypatch.setattr(
        ca.matrix_builder, "build_matrix",
        lambda *_args, **_kwargs: copy.deepcopy(matrix),
    )


def _assert_campaign_refuses_before_done(
    *, tmp_path: Path, monkeypatch, matrix: dict, expected_code: str,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True) + "\n")
    output_root = tmp_path / "capture"
    done_path = output_root / "chunks" / "chunk-00" / "DONE.json"
    done_path.parent.mkdir(parents=True)
    done_path.write_text("{}\n")
    original_loader = ca._load_object
    done_reads = []

    def observed_loader(path: Path, label: str):
        if label == "DONE":
            done_reads.append(path)
        return original_loader(path, label)

    monkeypatch.setattr(ca, "_load_object", observed_loader)
    with pytest.raises(ca.AccountingRefusal) as raised:
        ca.audit_campaign(
            matrix_path=matrix_path, output_root=output_root, repo_root=tmp_path,
        )
    assert raised.value.code == expected_code
    assert done_reads == []


def test_accounting_schema_refusal_precedes_done_reads(tmp_path: Path, monkeypatch):
    import jsonschema
    matrix = _admission_matrix()

    def invalid(*_args, **_kwargs):
        raise jsonschema.ValidationError("synthetic structural failure")

    monkeypatch.setattr(jsonschema, "validate", invalid)
    _assert_campaign_refuses_before_done(
        tmp_path=tmp_path, monkeypatch=monkeypatch, matrix=matrix,
        expected_code="MATRIX_SCHEMA_INVALID",
    )


def test_accounting_bad_seal_precedes_done_reads(tmp_path: Path, monkeypatch):
    matrix = _admission_matrix()
    _configure_admission_dependencies(monkeypatch, matrix)
    matrix["matrix_seal_sha256"] = "f" * 64
    _assert_campaign_refuses_before_done(
        tmp_path=tmp_path, monkeypatch=monkeypatch, matrix=matrix,
        expected_code="MATRIX_SEAL_MISMATCH",
    )


def test_accounting_validate_matrix_refusal_precedes_done_reads(
    tmp_path: Path, monkeypatch,
):
    matrix = _admission_matrix()
    _configure_admission_dependencies(monkeypatch, matrix)
    monkeypatch.setattr(
        ca.matrix_builder, "validate_matrix",
        lambda *_args, **_kwargs: ["chunk_runnable_set_equality"],
    )
    _assert_campaign_refuses_before_done(
        tmp_path=tmp_path, monkeypatch=monkeypatch, matrix=matrix,
        expected_code="MATRIX_VALIDATE_FAILED",
    )


def test_accounting_exact_head_refusal_precedes_done_reads(
    tmp_path: Path, monkeypatch,
):
    matrix = _admission_matrix()
    _configure_admission_dependencies(monkeypatch, matrix)
    monkeypatch.setattr(ca, "_git_head", lambda _root: "b" * 40)
    _assert_campaign_refuses_before_done(
        tmp_path=tmp_path, monkeypatch=monkeypatch, matrix=matrix,
        expected_code="MATRIX_REGENERATION_HEAD_MISMATCH",
    )


def test_accounting_regeneration_mismatch_precedes_done_reads(
    tmp_path: Path, monkeypatch,
):
    matrix = _admission_matrix()
    _configure_admission_dependencies(monkeypatch, matrix)
    regenerated = copy.deepcopy(matrix)
    regenerated["cells"][0]["authority_cell_sha256"] = "f" * 64
    _reseal(regenerated)
    monkeypatch.setattr(
        ca.matrix_builder, "build_matrix", lambda *_args, **_kwargs: regenerated,
    )
    _assert_campaign_refuses_before_done(
        tmp_path=tmp_path, monkeypatch=monkeypatch, matrix=matrix,
        expected_code="MATRIX_REGENERATION_MISMATCH",
    )


def test_matrix_admission_records_exact_authority_identity(tmp_path: Path, monkeypatch):
    matrix = _admission_matrix()
    _configure_admission_dependencies(monkeypatch, matrix)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True) + "\n")
    result = ca.admit_matrix(
        matrix, matrix_path=matrix_path, repo_root=tmp_path,
        loaded_matrix_sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
    )
    assert result == {
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "matrix_source_commit": matrix["source_commit"],
        "accounting_tool_head": matrix["source_commit"],
    }


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("truncated", "MATRIX_CARDINALITY_MISMATCH"),
        ("wrong_declared_cardinality", "MATRIX_CARDINALITY_MISMATCH"),
        ("wrong_chunk_count", "MATRIX_CARDINALITY_MISMATCH"),
        ("missing_budget", "MATRIX_BUDGET_COVERAGE_INVALID"),
        ("readiness_blocker", "MATRIX_READINESS_BLOCKER"),
    ],
)
def test_accounting_exact_campaign_invariants_precede_done_reads(
    tmp_path: Path, monkeypatch, mutation: str, expected_code: str,
):
    matrix = _admission_matrix()
    if mutation == "truncated":
        matrix["cells"].pop()
        matrix["counts"]["universe"] -= 1
        matrix["counts"]["excluded"] -= 1
    elif mutation == "wrong_declared_cardinality":
        matrix["counts"]["runnable"] -= 1
    elif mutation == "wrong_chunk_count":
        matrix["chunks"].pop()
        matrix["counts"]["chunks"] -= 1
    elif mutation == "missing_budget":
        matrix["cells"][0]["trace_budget"] = {"status": "missing"}
        matrix["counts"]["budget_declared"] -= 1
        matrix["counts"]["budget_missing"] = 1
    elif mutation == "readiness_blocker":
        matrix["cells"][0]["capture_readiness"] = {
            "status": "blocked", "blocking_reasons": ["trace_budget_missing"],
        }
        matrix["counts"]["capture_ready"] -= 1
        matrix["counts"]["capture_blocked"] = 1
    _reseal(matrix)
    _configure_admission_dependencies(monkeypatch, matrix)
    _assert_campaign_refuses_before_done(
        tmp_path=tmp_path, monkeypatch=monkeypatch, matrix=matrix,
        expected_code=expected_code,
    )


def test_global_preparation_gate_refuses_before_historical_done_read(
    tmp_path: Path, monkeypatch,
):
    matrix = _admission_matrix()
    _configure_admission_dependencies(monkeypatch, matrix)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True) + "\n")
    output_root = tmp_path / "capture"
    archive_path = (
        output_root / dm.PREPARATION_FAILURE_ROOT_NAME
        / "20260710T133500Z__head-9b8e0e40__pf-3810d2b2__"
        "sol_execbench-000-bootstrap"
    )
    archive_path.mkdir(parents=True)
    historical_done = (
        output_root / "chunks" / ca.HISTORICAL_BOOTSTRAP_CHUNK_ID / "DONE.json"
    )
    historical_done.parent.mkdir(parents=True)
    historical_done.write_text("{}\n")
    original_loader = ca._load_object
    done_reads: list[Path] = []

    def observed_loader(path: Path, label: str):
        if label == "DONE":
            done_reads.append(path)
        return original_loader(path, label)

    monkeypatch.setattr(ca, "_load_object", observed_loader)
    with pytest.raises(ca.AccountingRefusal) as raised:
        ca.audit_campaign(
            matrix_path=matrix_path, output_root=output_root, repo_root=tmp_path,
        )
    assert raised.value.code == "PREPARATION_LEDGER_INCONSISTENT"
    assert raised.value.detail == (
        f"archive evidence exists without ledger: {[str(archive_path)]!r}"
    )
    assert done_reads == []


def test_global_preparation_gate_follows_matrix_admission(
    tmp_path: Path, monkeypatch,
):
    matrix = _admission_matrix()
    _configure_admission_dependencies(monkeypatch, matrix)
    matrix["matrix_seal_sha256"] = "f" * 64
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True) + "\n")
    output_root = tmp_path / "capture"
    archive_path = output_root / dm.PREPARATION_FAILURE_ROOT_NAME / "unledgered"
    archive_path.mkdir(parents=True)
    with pytest.raises(ca.AccountingRefusal) as raised:
        ca.audit_campaign(
            matrix_path=matrix_path, output_root=output_root, repo_root=tmp_path,
        )
    assert raised.value.code == "MATRIX_SEAL_MISMATCH"


@pytest.mark.parametrize("root_exists", [False, True])
def test_global_preparation_gate_admits_missing_or_empty_root(
    tmp_path: Path, monkeypatch, root_exists: bool,
):
    matrix = _admission_matrix()
    _configure_admission_dependencies(monkeypatch, matrix)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True) + "\n")
    output_root = tmp_path / "capture"
    if root_exists:
        (output_root / dm.PREPARATION_FAILURE_ROOT_NAME).mkdir(parents=True)
    result = ca.audit_campaign(
        matrix_path=matrix_path, output_root=output_root, repo_root=tmp_path,
    )
    assert result["integrity_ok"] is True
    assert result["counts"]["accounted_unique_current"] == 0


def _strict_portable_matrix(authority_sha: str = "1" * 64) -> dict:
    matrix = {
        "source_commit": "a" * 40,
        "matrix_seal_sha256": "b" * 64,
        "cells": [{
            "identity": {"roster_source": "portable_lcd", "slug": "a"},
            "disposition": "runnable", "authority_cell_sha256": authority_sha,
        }],
        "chunks": [{
            "chunk_id": "chunk-a",
            "cells": [{"roster_source": "portable_lcd", "slug": "a"}],
        }],
    }
    return matrix


def _strict_done(tmp_path: Path, *, preflight: str = "4" * 64) -> tuple[Path, dict]:
    root = tmp_path / "chunk-a"
    result_path = root / "cells" / "portable_lcd__a" / "cell_capture_result.json"
    result_path.parent.mkdir(parents=True)
    result = {
        "capture_preflight_sha256": preflight,
        "capture_started_at_utc": "2026-07-10T00:00:01+00:00",
        "admission_run_id": "run-1",
        "supersedes": None,
        "prior_preparation_failures": [],
        "prior_capture_defect_attempts": [],
        "definition_equivalence": None,
        "definition_equivalence_attestation_sha256": None,
    }
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
    manifest = _done("portable_lcd", "a", "1" * 64)
    manifest.update({
        "chunk_id": "chunk-a",
        "source_commit_pre": "a" * 40,
        "source_commit_post": "a" * 40,
        "source_head_at_seal": "a" * 40,
        "capture_matrix_sha256": "c" * 64,
        "matrix_seal_sha256": "b" * 64,
        "preflight_sha256": preflight,
        "output_root": str(root),
    })
    manifest["cell_results"][0]["artifacts"] = [{
        "artifact_kind": "cell_result",
        "relative_path": result_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "size_bytes": result_path.stat().st_size,
    }]
    manifest["cell_results"][0]["prior_preparation_failures"] = []
    manifest["cell_results"][0]["prior_capture_defect_attempts"] = []
    done_path = root / "DONE.json"
    done_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return done_path, manifest


def _attach_external_supersession(
    *, manifest: dict, row: dict, stage_root: Path, original_preflight: str,
    entry_id: str,
) -> Path:
    cell_id = f"{row['roster_source']}__{row['slug']}"
    destination = (
        stage_root / "superseded"
        / f"20260710T000000Z__pf-{original_preflight[:8]}__to-pf-44444444"
        / cell_id
    )
    destination.mkdir(parents=True)
    preserved = destination / "journal.json"
    preserved.write_text('{"status":"complete"}\n')
    inventory_path = destination / "inventory.json"
    inventory_path.write_text(json.dumps({
        "schema_version": 1,
        "files": [{
            "relative_path": "journal.json",
            "sha256": hashlib.sha256(preserved.read_bytes()).hexdigest(),
            "size_bytes": preserved.stat().st_size,
        }],
    }, sort_keys=True) + "\n")
    entry = {
        "schema_version": 1,
        "timestamp_utc": "2026-07-10T00:00:00+00:00",
        "session_id": "session",
        "chunk_id": manifest["chunk_id"],
        "cell_id": cell_id,
        "original_preflight_sha256": original_preflight,
        "current_preflight_sha256": manifest["preflight_sha256"],
        "source_path": str(stage_root / "journals" / cell_id),
        "destination_path": str(destination),
        "reason_code": "PREFLIGHT_EXPIRED_BEFORE_DONE",
        "operator_flag": "--supersede-stale-journals",
        "source_commit": manifest["source_commit_pre"],
        "matrix_seal_sha256": manifest["matrix_seal_sha256"],
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "ledger_entry_id": entry_id,
    }
    ledger = stage_root / "superseded" / "supersession_ledger.jsonl"
    ledger.write_text(json.dumps(entry, sort_keys=True) + "\n")
    result_artifact = next(
        artifact for artifact in row["artifacts"]
        if artifact["artifact_kind"] == "cell_result"
    )
    result_path = Path(manifest["output_root"]) / result_artifact["relative_path"]
    document = json.loads(result_path.read_text())
    document["supersedes"] = {
        "ledger_entry_id": entry_id,
        "superseded_preflight_sha256": original_preflight,
        "superseded_path": str(destination),
    }
    result_path.write_text(json.dumps(document, sort_keys=True) + "\n")
    return destination


def _strict_sol_v2_with_preparation_archive(tmp_path: Path) -> tuple[dict, Path, dict]:
    from tools.local_labels import archive_preparation_failure as ap
    from tools.local_labels.test_archive_preparation_failure import (
        _fixture as archive_fixture,
    )

    campaign_root = tmp_path / "capture"
    request = archive_fixture(
        tmp_path / "archive-fixture",
        prep_root=campaign_root / ap.PREPARATION_FAILURE_ROOT_NAME,
    )
    ap.archive_preparation_failure(request)
    ledger_path = (
        request.preparation_failures_root / ap.PREPARATION_FAILURE_LEDGER_NAME
    ).resolve()
    ledger = ap.load_preparation_failure_ledger(
        ledger_path, verify_archives=True,
    )
    refs = ap.preparation_failure_backrefs(
        ledger, cell_id=request.cell_id, ledger_path=ledger_path,
    )
    source, slug = request.cell_id.split("__", 1)
    output_root = campaign_root / "chunks" / request.chunk_id
    stage = output_root / "cells" / request.cell_id / "sol_stage"
    stage.mkdir(parents=True)
    source_path = stage / dm.SOL_DEFINITION_SOURCE_FILENAME
    staged_path = stage / dm.SOL_DEFINITION_STAGED_FILENAME
    source_path.write_bytes(b'{"name":"accounting-v2"}\n')
    staged_path.write_bytes(source_path.read_bytes())
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    staged_sha = hashlib.sha256(staged_path.read_bytes()).hexdigest()
    attestation = {
        "rule_id": dm.SOL_DEFINITION_EQUIVALENCE_RULE_ID,
        "result": "equal",
        "source_definition_sha256": source_sha,
        "staged_definition_sha256": staged_sha,
        "top_level_field_states": {
            field: {
                "source": "missing", "staged": "missing",
                "source_normalized": True, "staged_normalized": True,
            }
            for field in sorted(dm._SOL_DEFINITION_TOP_LEVEL_FIELDS)
        },
        "axes_container_state": {"source": "missing", "staged": "missing"},
        "axis_description_states": {},
        "tensor_container_states": {
            name: {"source": "missing", "staged": "missing"}
            for name in ("inputs", "outputs")
        },
        "tensor_description_states": {},
    }
    attestation_path = stage / dm.SOL_DEFINITION_EQUIVALENCE_FILENAME
    attestation_bytes = dm._canonical_definition_attestation_bytes(attestation)
    assert attestation_bytes is not None
    attestation_path.write_bytes(attestation_bytes)
    attestation_sha = hashlib.sha256(attestation_bytes).hexdigest()
    prepare = {
        "source_definition_sha256": source_sha,
        "staged_definition_sha256": staged_sha,
        "definition_equivalence": attestation,
        "artifacts": {
            "definition": {
                "path": dm.SOL_DEFINITION_STAGED_FILENAME,
                "sha256": staged_sha,
            },
            "definition_source": {
                "path": dm.SOL_DEFINITION_SOURCE_FILENAME,
                "sha256": source_sha,
            },
            "definition_equivalence_attestation": {
                "path": dm.SOL_DEFINITION_EQUIVALENCE_FILENAME,
                "sha256": attestation_sha,
            },
        },
    }
    prepare["stage_definition_sha256"] = dm._sol_stage_manifest_sha256(prepare)
    prepare_path = stage / dm.SOL_DEFINITION_PREPARE_MANIFEST_FILENAME
    prepare_path.write_text(json.dumps(prepare, indent=2, sort_keys=True) + "\n")

    def artifact(path: Path) -> dict:
        return {
            "artifact_kind": "sol_capture_evidence",
            "relative_path": path.relative_to(output_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    preflight = "4" * 64
    result_path = output_root / "cells" / request.cell_id / "cell_capture_result.json"
    result = {
        "capture_preflight_sha256": preflight,
        "capture_started_at_utc": "2026-07-10T00:00:01+00:00",
        "admission_run_id": "run-1",
        "supersedes": None,
        "prior_preparation_failures": refs,
        "prior_capture_defect_attempts": [],
        "definition_equivalence": attestation,
        "definition_equivalence_attestation_sha256": attestation_sha,
    }
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
    result_artifact = {
        "artifact_kind": "cell_result",
        "relative_path": result_path.relative_to(output_root).as_posix(),
        "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "size_bytes": result_path.stat().st_size,
    }
    row = {
        "roster_source": source,
        "slug": slug,
        "authority_cell_sha256": "1" * 64,
        "disposition": "admitted",
        "typed_reason": None,
        **result,
        "artifacts": [
            result_artifact, artifact(source_path), artifact(staged_path),
            artifact(attestation_path), artifact(prepare_path),
        ],
    }
    manifest = {
        "chunk_id": request.chunk_id,
        "source_commit_pre": "a" * 40,
        "source_commit_post": "a" * 40,
        "source_head_at_seal": "a" * 40,
        "capture_matrix_sha256": "c" * 64,
        "matrix_seal_sha256": "b" * 64,
        "preflight_sha256": preflight,
        "output_root": str(output_root),
        "cell_results": [row],
    }
    done_path = output_root / "DONE.json"
    done_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    matrix = {
        "source_commit": "a" * 40,
        "matrix_seal_sha256": "b" * 64,
        "cells": [{
            "identity": {"roster_source": source, "slug": slug},
            "disposition": "runnable",
            "authority_cell_sha256": "1" * 64,
            "invocation": {
                "adapter_contract": {
                    "native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
                    "environment": dict(dm.SOL_CERTIFIED_ENVIRONMENT),
                },
                "logical_workload": {"definition_file_sha256": source_sha},
            },
        }],
        "chunks": [{
            "chunk_id": request.chunk_id,
            "cells": [{"roster_source": source, "slug": slug}],
        }],
    }
    return matrix, done_path, manifest


def test_strict_join_binds_chunk_head_matrix_and_preflight(tmp_path: Path):
    matrix = _strict_portable_matrix()
    done_path, manifest = _strict_done(tmp_path)
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert result["integrity_ok"] is True
    assert result["complete"] is True
    assert result["counts"]["superseded_leaks"] == 0


def test_accounting_accepts_v2_sol_with_closed_preparation_ledger(tmp_path: Path):
    matrix, done_path, manifest = _strict_sol_v2_with_preparation_archive(tmp_path)
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert result["integrity_ok"] is True
    assert result["complete"] is True
    assert result["counts"]["accounted_unique_current"] == 1
    assert result["counts"]["preparation_archive_leaks"] == 0


def test_accounting_rejects_unclosed_preparation_ledger(tmp_path: Path):
    matrix, done_path, manifest = _strict_sol_v2_with_preparation_archive(tmp_path)
    manifest["cell_results"][0]["prior_preparation_failures"] = []
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert result["integrity_ok"] is False
    assert any(
        "PREPARATION_LEDGER_INCONSISTENT" in problem
        for problem in result["seal_failures"][0]["problems"]
    )


def test_strict_join_rejects_cell_preflight_mismatch(tmp_path: Path):
    matrix = _strict_portable_matrix()
    done_path, manifest = _strict_done(tmp_path, preflight="4" * 64)
    result_path = Path(manifest["output_root"]) / (
        manifest["cell_results"][0]["artifacts"][0]["relative_path"]
    )
    document = json.loads(result_path.read_text())
    document["capture_preflight_sha256"] = "5" * 64
    result_path.write_text(json.dumps(document, sort_keys=True) + "\n")
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert result["integrity_ok"] is False
    assert any(
        "DONE_PREFLIGHT_MISMATCH" in problem
        for problem in result["seal_failures"][0]["problems"]
    )


def test_strict_join_rejects_superseded_published_path(tmp_path: Path):
    matrix = _strict_portable_matrix()
    done_path, manifest = _strict_done(tmp_path)
    manifest["cell_results"][0]["artifacts"].append({
        "artifact_kind": "execution_log",
        "relative_path": "superseded/attempt/log.txt",
        "sha256": "d" * 64,
        "size_bytes": 1,
    })
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert result["integrity_ok"] is False
    assert result["counts"]["superseded_leaks"] == 1
    assert result["complete"] is False


def test_strict_join_rejects_symlink_resolving_under_superseded(tmp_path: Path):
    matrix = _strict_portable_matrix()
    done_path, manifest = _strict_done(tmp_path)
    output_root = Path(manifest["output_root"])
    preserved = output_root / "superseded" / "attempt" / "cell" / "log.txt"
    preserved.parent.mkdir(parents=True)
    preserved.write_text("old attempt\n")
    (output_root / "published_link").symlink_to(preserved.parent, target_is_directory=True)
    manifest["cell_results"][0]["artifacts"].append({
        "artifact_kind": "execution_log",
        "relative_path": "published_link/log.txt",
        "sha256": hashlib.sha256(preserved.read_bytes()).hexdigest(),
        "size_bytes": preserved.stat().st_size,
    })
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert result["integrity_ok"] is False
    assert result["counts"]["superseded_leaks"] == 1


def test_strict_join_rejects_preparation_archive_published_path(tmp_path: Path):
    matrix = _strict_portable_matrix()
    done_path, manifest = _strict_done(tmp_path)
    manifest["cell_results"][0]["artifacts"].append({
        "artifact_kind": "execution_log",
        "relative_path": "preparation_failures/attempt/forensic.txt",
        "sha256": "d" * 64,
        "size_bytes": 1,
    })
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert result["integrity_ok"] is False
    assert result["counts"]["preparation_archive_leaks"] == 1
    assert result["complete"] is False
    assert any(
        "PREPARATION_ARCHIVE_LEAK" in problem
        for problem in result["seal_failures"][0]["problems"]
    )


def test_strict_join_rejects_symlink_resolving_under_preparation_archive(
    tmp_path: Path,
):
    matrix = _strict_portable_matrix()
    done_path, manifest = _strict_done(tmp_path)
    output_root = Path(manifest["output_root"])
    preserved = (
        output_root / "preparation_failures" / "attempt" / "forensic.txt"
    )
    preserved.parent.mkdir(parents=True)
    preserved.write_text("preparation attempt\n")
    (output_root / "published_preparation_link").symlink_to(
        preserved.parent, target_is_directory=True,
    )
    manifest["cell_results"][0]["artifacts"].append({
        "artifact_kind": "execution_log",
        "relative_path": "published_preparation_link/forensic.txt",
        "sha256": hashlib.sha256(preserved.read_bytes()).hexdigest(),
        "size_bytes": preserved.stat().st_size,
    })
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert result["integrity_ok"] is False
    assert result["counts"]["preparation_archive_leaks"] == 1


def test_strict_join_follows_consistent_ledger_outside_final_output(tmp_path: Path):
    matrix = _strict_portable_matrix()
    done_path, manifest = _strict_done(tmp_path)
    output_root = Path(manifest["output_root"])
    stage_root = tmp_path / "persistent-stage" / "chunk-a"
    destination = _attach_external_supersession(
        manifest=manifest, row=manifest["cell_results"][0],
        stage_root=stage_root, original_preflight="5" * 64, entry_id="entry-1",
    )
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert not (output_root / "superseded").exists()
    assert destination.is_relative_to(stage_root)
    assert result["integrity_ok"] is True


def test_strict_join_allows_multiple_backreferenced_stage_roots(tmp_path: Path):
    matrix = _strict_portable_matrix()
    done_path, manifest = _strict_done(tmp_path)
    second_identity = {"roster_source": "portable_lcd", "slug": "b"}
    matrix["cells"].append({
        "identity": second_identity,
        "disposition": "runnable",
        "authority_cell_sha256": "2" * 64,
    })
    matrix["chunks"][0]["cells"].append(second_identity)
    output_root = Path(manifest["output_root"])
    second_result_path = (
        output_root / "cells" / "portable_lcd__b" / "cell_capture_result.json"
    )
    second_result_path.parent.mkdir(parents=True)
    second_result_path.write_text(json.dumps({
        "capture_preflight_sha256": manifest["preflight_sha256"],
        "capture_started_at_utc": "2026-07-10T00:00:01+00:00",
        "admission_run_id": "run-1",
        "supersedes": None,
        "prior_preparation_failures": [],
        "prior_capture_defect_attempts": [],
        "definition_equivalence": None,
        "definition_equivalence_attestation_sha256": None,
    }, sort_keys=True) + "\n")
    second_row = _done("portable_lcd", "b", "2" * 64)["cell_results"][0]
    second_row["artifacts"] = [{
        "artifact_kind": "cell_result",
        "relative_path": second_result_path.relative_to(output_root).as_posix(),
        "sha256": hashlib.sha256(second_result_path.read_bytes()).hexdigest(),
        "size_bytes": second_result_path.stat().st_size,
    }]
    manifest["cell_results"].append(second_row)
    first_destination = _attach_external_supersession(
        manifest=manifest, row=manifest["cell_results"][0],
        stage_root=tmp_path / "persistent-stage-one",
        original_preflight="5" * 64, entry_id="entry-one",
    )
    second_destination = _attach_external_supersession(
        manifest=manifest, row=second_row,
        stage_root=tmp_path / "persistent-stage-two",
        original_preflight="6" * 64, entry_id="entry-two",
    )
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert first_destination.parent.parent != second_destination.parent.parent
    assert result["integrity_ok"] is True
    assert result["complete"] is True


def test_strict_join_rejects_dangling_supersession_backref(tmp_path: Path):
    matrix = _strict_portable_matrix()
    done_path, manifest = _strict_done(tmp_path)
    result_path = Path(manifest["output_root"]) / (
        manifest["cell_results"][0]["artifacts"][0]["relative_path"]
    )
    document = json.loads(result_path.read_text())
    document["supersedes"] = {
        "ledger_entry_id": "missing",
        "superseded_preflight_sha256": "5" * 64,
        "superseded_path": str(Path(manifest["output_root"]) / "superseded" / "x"),
    }
    result_path.write_text(json.dumps(document, sort_keys=True) + "\n")
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert result["integrity_ok"] is False
    assert any(
        "SUPERSESSION_LEDGER_INCONSISTENT" in problem
        for problem in result["seal_failures"][0]["problems"]
    )


def test_exact_historical_bootstrap_is_the_only_legacy_exemption():
    done_path = Path(
        "/home/jakeshea/sm86_reference_capture_r27/chunks/"
        "portable_lcd-000-bootstrap/DONE.json"
    )
    manifest = json.loads(done_path.read_text())
    row = manifest["cell_results"][0]
    matrix = {
        "source_commit": "a" * 40,
        "matrix_seal_sha256": "b" * 64,
        "cells": [{
            "identity": {
                "roster_source": row["roster_source"], "slug": row["slug"],
            },
            "disposition": "runnable",
            "authority_cell_sha256": row["authority_cell_sha256"],
        }],
        "chunks": [{
            "chunk_id": ca.HISTORICAL_BOOTSTRAP_CHUNK_ID,
            "cells": [{
                "roster_source": row["roster_source"], "slug": row["slug"],
            }],
        }],
    }
    result = ca.audit_loaded_manifests(
        matrix, [(done_path, manifest, [])], matrix_sha256="c" * 64,
    )
    assert ca._historical_bootstrap_is_allowlisted(done_path, manifest)
    assert result["integrity_ok"] is True
    assert result["complete"] is True

    near_copy = copy.deepcopy(manifest)
    near_copy["generated_at_utc"] = "2026-07-10T00:00:00+00:00"
    assert not ca._historical_bootstrap_is_allowlisted(done_path, near_copy)
    rejected = ca.audit_loaded_manifests(
        matrix, [(done_path, near_copy, [])], matrix_sha256="c" * 64,
    )
    assert rejected["integrity_ok"] is False
    assert any(
        "DONE_HEAD_MISMATCH" in problem or "DONE_MATRIX_MISMATCH" in problem
        for problem in rejected["seal_failures"][0]["problems"]
    )
