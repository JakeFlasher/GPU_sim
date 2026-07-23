"""CPU-only contracts for the additive SOL workload-row capture consumer."""
from __future__ import annotations

import copy
import errno
import hashlib
import inspect
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from tools.local_labels import sm86_capture_chunk as canonical_capture
from tools.local_labels import sol_workload_capture_chunk as capture
from tools.local_labels import test_sol_workload_capture_matrix as matrix_fixture


@pytest.mark.parametrize(
    ("error_number", "expected_code"),
    [
        (errno.ENOSPC, capture.STORAGE_CAPACITY_CODE),
        (errno.EDQUOT, capture.STORAGE_CAPACITY_CODE),
        (errno.EIO, capture.STORAGE_INTEGRITY_CODE),
        (errno.EROFS, capture.STORAGE_INTEGRITY_CODE),
        (None, capture.STORAGE_INTEGRITY_CODE),
    ],
)
def test_storage_error_taxonomy_is_closed(error_number, expected_code):
    error = OSError(error_number, "injected write/fsync/short-write failure")
    refusal = capture.storage_refusal_from_exception(
        error, operation="injected_boundary"
    )
    assert refusal is not None
    assert refusal.code == expected_code
    detail = json.loads(refusal.detail)
    assert detail == {
        "operation": "injected_boundary",
        "errno": error_number,
        "error": str(error),
    }


def test_storage_error_taxonomy_unwraps_nested_fsync_failure():
    cause = OSError(errno.EIO, "injected fsync failure")
    wrapped = capture.WorkloadCaptureRefusal("wrapper", "journal finalize")
    wrapped.__cause__ = cause
    refusal = capture.storage_refusal_from_exception(
        wrapped, operation="journal_finalize"
    )
    assert refusal is not None
    assert refusal.code == capture.STORAGE_INTEGRITY_CODE
    assert json.loads(refusal.detail)["errno"] == errno.EIO


def test_json_publication_errno_less_short_write_is_storage_integrity(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(capture.os, "write", lambda *_args: 0)
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match=capture.STORAGE_INTEGRITY_CODE,
    ):
        capture._write_json_new(tmp_path / "result.json", {"value": 1})


def test_live_source_state_uses_supplemental_not_frozen_canonical_commit(
    monkeypatch,
):
    live = "a" * 40
    frozen = "b" * 40
    monkeypatch.setattr(
        capture.canonical_capture, "_git_state", lambda: (live, True, [])
    )
    matrix = {
        "source_commit": live,
        "canonical_matrix": {"source_commit": frozen},
    }
    assert capture._require_live_supplemental_source_state(matrix) == live

    monkeypatch.setattr(
        capture.canonical_capture, "_git_state", lambda: (frozen, True, [])
    )
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_source_state_invalid",
    ):
        capture._require_live_supplemental_source_state(matrix)

    monkeypatch.setattr(
        capture.canonical_capture,
        "_git_state",
        lambda: (live, False, [" M user-owned"]),
    )
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_source_state_invalid",
    ):
        capture._require_live_supplemental_source_state(matrix)


@pytest.mark.parametrize(
    "error_number", [errno.EAGAIN, errno.EEXIST, errno.ENOENT, errno.EACCES]
)
def test_semantic_os_errors_are_not_storage_faults(error_number):
    assert capture.storage_refusal_from_exception(
        OSError(error_number, "injected semantic filesystem condition"),
        operation="semantic_boundary",
    ) is None


@pytest.mark.parametrize(
    ("error_number", "expected_code"),
    [
        (errno.ENOSPC, capture.STORAGE_CAPACITY_CODE),
        (errno.EDQUOT, capture.STORAGE_CAPACITY_CODE),
        (errno.EIO, capture.STORAGE_INTEGRITY_CODE),
        (errno.EROFS, capture.STORAGE_INTEGRITY_CODE),
    ],
)
def test_output_directory_mkdir_fault_is_typed_before_generic_path_error(
    tmp_path, monkeypatch, error_number, expected_code,
):
    parent = tmp_path / "existing"
    parent.mkdir()
    real_mkdir = capture.os.mkdir

    def injected_mkdir(path, *args, dir_fd=None, **kwargs):
        if path == "faulted-child" and dir_fd is not None:
            raise OSError(error_number, "injected output mkdir failure")
        return real_mkdir(path, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(capture.os, "mkdir", injected_mkdir)
    with pytest.raises(capture.WorkloadCaptureRefusal) as caught:
        capture._walk_directory_nofollow(
            parent / "faulted-child", create=True,
        )
    assert caught.value.code == expected_code
    assert json.loads(caught.value.detail)["operation"] == (
        "output_directory_mkdir"
    )


def test_output_directory_open_fault_is_typed_before_generic_path_error(
    tmp_path, monkeypatch,
):
    target = tmp_path / "faulted-child"
    target.mkdir()
    real_open = capture.os.open

    def injected_open(path, *args, dir_fd=None, **kwargs):
        if path == "faulted-child" and dir_fd is not None:
            raise OSError(errno.EIO, "injected output open failure")
        return real_open(path, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(capture.os, "open", injected_open)
    with pytest.raises(capture.WorkloadCaptureRefusal) as caught:
        capture._walk_directory_nofollow(target, create=False)
    assert caught.value.code == capture.STORAGE_INTEGRITY_CODE
    assert json.loads(caught.value.detail)["operation"] == (
        "output_directory_open"
    )


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_known_output_read_maps_storage_but_generic_input_read_does_not(
    tmp_path, monkeypatch, error_number,
):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n")
    real_open = capture._open_regular_nofollow

    def fail_open(path, *, role):
        if path == artifact:
            refusal = capture.WorkloadCaptureRefusal(
                "supplemental_input_path_invalid", f"{role}: injected",
            )
            refusal.__cause__ = OSError(
                error_number, "injected output read failure",
            )
            raise refusal
        return real_open(path, role=role)

    monkeypatch.setattr(capture, "_open_regular_nofollow", fail_open)
    with pytest.raises(capture.WorkloadCaptureRefusal) as output_failure:
        capture._read_output_regular_nofollow(
            artifact, role="generated artifact",
            operation="generated_artifact_read",
        )
    assert output_failure.value.code == capture.STORAGE_INTEGRITY_CODE

    with pytest.raises(capture.WorkloadCaptureRefusal) as input_failure:
        capture._load_object(artifact, role="generic immutable input")
    assert input_failure.value.code == "supplemental_input_path_invalid"


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_output_tree_traversal_storage_fault_is_typed(
    tmp_path, monkeypatch, error_number,
):
    root = tmp_path / "tree"
    root.mkdir()
    real_scandir = capture.os.scandir

    def fail_scandir(path):
        if Path(path) == root:
            raise OSError(error_number, "injected output traversal failure")
        return real_scandir(path)

    monkeypatch.setattr(capture.os, "scandir", fail_scandir)
    with pytest.raises(capture.WorkloadCaptureRefusal) as caught:
        capture._output_tree_paths(root, operation="generated_tree_walk")
    assert caught.value.code == capture.STORAGE_INTEGRITY_CODE


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_owner_marker_output_read_storage_fault_is_typed(
    tmp_path, monkeypatch, error_number,
):
    root = tmp_path / "campaign"
    identity = {"source_commit": "a" * 40, "input_file_identities": {}}
    capture.establish_supplemental_output_root(
        root, campaign_identity=identity,
    )
    marker = root / capture.SUPPLEMENTAL_ROOT_MARKER
    real_open = capture._open_regular_nofollow

    def fail_open(path, *, role):
        if path == marker:
            refusal = capture.WorkloadCaptureRefusal(
                "supplemental_input_path_invalid", f"{role}: injected",
            )
            refusal.__cause__ = OSError(
                error_number, "injected owner marker read failure",
            )
            raise refusal
        return real_open(path, role=role)

    monkeypatch.setattr(capture, "_open_regular_nofollow", fail_open)
    with pytest.raises(capture.WorkloadCaptureRefusal) as caught:
        capture._load_retained_input_owner(root)
    assert caught.value.code == capture.STORAGE_INTEGRITY_CODE


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_generated_artifact_hash_storage_fault_is_typed(
    tmp_path, monkeypatch, error_number,
):
    artifact = tmp_path / "cells" / "fixture.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"fixture")
    result = {"artifacts": [{
        "relative_path": "cells/fixture.bin",
        "sha256": hashlib.sha256(b"fixture").hexdigest(),
        "size_bytes": len(b"fixture"),
    }]}
    real_open = capture._open_regular_nofollow

    def fail_open(path, *, role):
        if path == artifact:
            refusal = capture.WorkloadCaptureRefusal(
                "supplemental_input_path_invalid", f"{role}: injected",
            )
            refusal.__cause__ = OSError(
                error_number, "injected artifact hash failure",
            )
            raise refusal
        return real_open(path, role=role)

    monkeypatch.setattr(capture, "_open_regular_nofollow", fail_open)
    with pytest.raises(capture.WorkloadCaptureRefusal) as caught:
        capture._verify_artifacts(result, root=tmp_path)
    assert caught.value.code == capture.STORAGE_INTEGRITY_CODE


def test_direct_chunk_cli_is_internal_only(capsys):
    assert capture.main(["--output-root", "/tmp/would-bypass-lock"]) == 2
    assert "supplemental_chunk_internal_only" in capsys.readouterr().err


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_CONTRACT = {
    "inventory_policy": "requires_inventory_before_capture",
    "launch_scope": "all_in_profiler_roi",
    "trace_requirement": "every_inventory_launch_complete",
    "ncu_label_requirement": "every_inventory_launch_complete",
    "control_requirement": "every_inventory_launch_complete",
    "admission_policy": "all_three_layers_complete_for_exact_inventory",
}
FLASHINFER_L1_CONTRACT = {
    "policy": "verified_fib_only",
    "category": "L1",
    "environment_variable": "FLASHINFER_TRACE_DIR",
    "injection_policy": "forbidden_remove_ambient",
    "root": None,
    "data_annex_manifest_path": "evidence/manifests/data_annex.sha256",
    "data_annex_manifest_sha256": capture.sol_recon.DATA_ANNEX_MANIFEST_SHA256,
    "filtered_manifest_sha256": capture.sol_recon.FLASHINFER_CORPUS_MANIFEST_SHA256,
    "file_count": 5244,
    "safetensors_count": 2611,
}


def _literal_rows(tmp_path: Path) -> tuple[Path, list[bytes], list[dict]]:
    values = [
        {
            "uuid": "12345678-1234-5123-8123-123456789abc",
            "axes": {"batch": 1},
        },
        {
            "uuid": "22345678-1234-5123-8123-123456789abc",
            "axes": {"batch": 7},
        },
    ]
    rows = [
        json.dumps(value, separators=(",", ":")).encode("utf-8")
        for value in values
    ]
    path = tmp_path / "workload.jsonl"
    path.write_bytes(b"\n".join(rows) + b"\n")
    return path, rows, values


def _binding(
    *, path: Path, rows: list[bytes], values: list[dict], index: int = 1,
    cohort: str = "native",
) -> tuple[dict, dict]:
    invocation = {
        "logical_workload": {"workload_file": str(path)},
        "capture_target": {
            "source_workload_index": 0,
            "staged_workload_index": 0,
        },
    }
    source_sha = hashlib.sha256(rows[index]).hexdigest()
    variant = "native" if cohort == "native" else "a" * 64
    claim = {
        "native": "native_workload_row",
        "partial_native_row": "partial_native_workload_row_only",
        "reduced_proxy": "reduced_proxy_bundle_only",
    }[cohort]
    predictive = cohort == "native"
    native_shape = cohort != "reduced_proxy"
    problem_level_runnable = cohort == "native"
    source_axes = values[index]["axes"]
    capture_axes = source_axes
    capture_sha = source_sha
    source_index = index
    if cohort == "reduced_proxy":
        source_axes = {"batch": 14}
        source_sha = "b" * 64
        source_index = 9
    binding = {
        "schema_version": 1,
        "supplemental_matrix_sha256": "1" * 64,
        "supplemental_matrix_seal_sha256": "2" * 64,
        "chunk_id": "sol-workload-test-001",
        "row_key": {
            "slug": "problem_001_test",
            "source_workload_uuid": values[index]["uuid"],
            "variant_id": variant,
        },
        "cohort": cohort,
        "category": "L1",
        "flashinfer_corpus": copy.deepcopy(FLASHINFER_L1_CONTRACT),
        "launch_scope": "all_in_profiler_roi",
        "launch_completeness_contract": copy.deepcopy(LAUNCH_CONTRACT),
        "claim_scope": claim,
        "native_shape": native_shape,
        "native_predictive": predictive,
        "problem_level_runnable": problem_level_runnable,
        "variant_contract_sha256": "3" * 64,
        "row_evidence_sha256": "4" * 64,
        "source_workload_index": source_index,
        "source_workload_uuid": values[index]["uuid"],
        "source_axes": source_axes,
        "source_row_sha256": source_sha,
        "capture_workload_index": index,
        "capture_axes": capture_axes,
        "capture_row_sha256": capture_sha,
    }
    return invocation, binding


def _externalize_reduced_invocation(
    invocation: dict, path: Path, *, definition_path: Path | None = None,
) -> dict:
    workload_sha = capture.sha256_file(path)
    definition_name = (
        definition_path.name if definition_path is not None else "definition.json"
    )
    argv = [
        "--definition", definition_name, "--workload", path.name,
        "--solution", "solution.json", "--json",
    ]
    invocation = copy.deepcopy(invocation)
    invocation.update({
        "authority_reference_path": "solution.json",
        "label_app_args": list(argv),
        "trace_app_args": list(argv),
        "adapter_contract": {"forwarded_argv": list(argv)},
    })
    invocation["logical_workload"].update({
        "benchmark_definition": "problem_001_test",
        "definition_file": definition_name,
        "definition_file_sha256": (
            capture.sha256_file(definition_path)
            if definition_path is not None else "c" * 64
        ),
        "native_definition_file": "definition.json",
        "native_definition_file_sha256": "c" * 64,
        "external_definition_source": (
            {
                "kind": "shrink_external_definition_child_v1",
                "name": definition_path.name,
                "sha256": capture.sha256_file(definition_path),
                "size_bytes": definition_path.stat().st_size,
                "owner_marker": {
                    "name": ".owner.json", "sha256": "a" * 64,
                    "owner_seal_sha256": "b" * 64,
                },
                "runtime_rebind": "campaign_retained_immutable_child",
            }
            if definition_path is not None else None
        ),
        "workload_file": path.name,
        "workload_file_sha256": workload_sha,
        "external_workload_source": {
            "kind": "shrink_external_child_v1",
            "name": path.name,
            "sha256": workload_sha,
            "size_bytes": path.stat().st_size,
            "owner_marker": {
                "name": ".owner.json", "sha256": "a" * 64,
                "owner_seal_sha256": "b" * 64,
            },
            "runtime_rebind": "campaign_retained_immutable_child",
        },
    })
    return invocation


def test_binding_selects_exact_nonzero_native_row(tmp_path):
    path, rows, values = _literal_rows(tmp_path)
    invocation, binding = _binding(path=path, rows=rows, values=values)
    rebound = canonical_capture.bind_supplemental_sol_workload_row(
        invocation, binding
    )
    assert invocation["capture_target"]["source_workload_index"] == 0
    assert rebound["capture_target"]["source_workload_index"] == 1
    assert rebound["capture_target"]["staged_workload_index"] == 0


def test_binding_hash_scope_is_literal_bytes_without_lf(tmp_path):
    path, rows, values = _literal_rows(tmp_path)
    # Leading space is valid JSON and is part of the row identity.  A stripped
    # digest must not be accepted as a substitute.
    spaced = b" " + rows[1]
    path.write_bytes(rows[0] + b"\n" + spaced + b"\n")
    invocation, binding = _binding(path=path, rows=[rows[0], spaced], values=values)
    canonical_capture.bind_supplemental_sol_workload_row(invocation, binding)
    bad = copy.deepcopy(binding)
    bad["capture_row_sha256"] = hashlib.sha256(spaced.strip()).hexdigest()
    bad["source_row_sha256"] = bad["capture_row_sha256"]
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_workload_row_binding_mismatch",
    ):
        canonical_capture.bind_supplemental_sol_workload_row(invocation, bad)


def test_binding_accepts_empty_native_and_reduced_capture_axes(tmp_path):
    path = tmp_path / "workload.jsonl"
    value = {
        "uuid": "0123456789abcdef0123456789abcdef", "axes": {},
    }
    raw = json.dumps(value, separators=(",", ":")).encode()
    path.write_bytes(raw + b"\n")
    invocation, native = _binding(
        path=path, rows=[raw], values=[value], index=0, cohort="native"
    )
    rebound = canonical_capture.bind_supplemental_sol_workload_row(
        invocation, native
    )
    assert rebound["capture_target"]["source_workload_index"] == 0

    invocation, reduced = _binding(
        path=path, rows=[raw], values=[value], index=0,
        cohort="reduced_proxy",
    )
    invocation = _externalize_reduced_invocation(invocation, path)
    reduced["source_axes"] = {"batch": 14}
    path.chmod(0o444)
    rebound = canonical_capture.bind_supplemental_sol_workload_row(
        invocation, reduced, supplemental_workload_path=path,
    )
    assert rebound["logical_workload"]["workload_file"] == str(path)


def test_definition_only_proxy_binds_exact_immutable_definition(tmp_path):
    owner = tmp_path / "owner"
    owner.mkdir(mode=0o700)
    path, rows, values = _literal_rows(owner)
    definition = owner / "reduced-definition.json"
    definition.write_text(json.dumps({
        "name": "problem_001_test",
        "axes": {"num_experts": {"type": "const", "value": 32}},
    }, sort_keys=True) + "\n")
    definition.chmod(0o444)
    path.chmod(0o444)
    invocation, binding = _binding(
        path=path, rows=rows, values=values, cohort="reduced_proxy"
    )
    invocation = _externalize_reduced_invocation(
        invocation, path, definition_path=definition,
    )
    # Definition-only proxies deliberately retain the exact native workload
    # row.  Their bundle identity changes through the separately sealed
    # definition input.
    binding["source_workload_index"] = binding["capture_workload_index"]
    binding["source_axes"] = copy.deepcopy(binding["capture_axes"])
    binding["source_row_sha256"] = binding["capture_row_sha256"]
    rebound = canonical_capture.bind_supplemental_sol_workload_row(
        invocation, binding, supplemental_workload_path=path,
        supplemental_definition_path=definition,
    )
    assert rebound["logical_workload"]["definition_file"] == str(definition)
    assert rebound["logical_workload"]["workload_file"] == str(path)
    assert rebound["trace_app_args"][1] == str(definition)
    assert rebound["trace_app_args"][3] == str(path)

    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_external_definition_rebind_invalid",
    ):
        canonical_capture.bind_supplemental_sol_workload_row(
            invocation, binding, supplemental_workload_path=path,
        )

    writable = owner / "writable-definition.json"
    writable.write_bytes(definition.read_bytes())
    bad = copy.deepcopy(invocation)
    source = bad["logical_workload"]["external_definition_source"]
    source["name"] = writable.name
    bad["logical_workload"]["definition_file"] = writable.name
    for field in ("label_app_args", "trace_app_args"):
        bad[field][1] = writable.name
    bad["adapter_contract"]["forwarded_argv"][1] = writable.name
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_external_workload_not_immutable",
    ):
        canonical_capture.bind_supplemental_sol_workload_row(
            bad, binding, supplemental_workload_path=path,
            supplemental_definition_path=writable,
        )


def test_binding_index_counts_only_effective_jsonl_rows(tmp_path):
    path, rows, values = _literal_rows(tmp_path)
    path.write_bytes(
        b"// retained comment\n\n" + rows[0] + b"\n  // comment\n"
        + rows[1] + b"\n"
    )
    invocation, binding = _binding(
        path=path, rows=rows, values=values, index=1
    )
    rebound = canonical_capture.bind_supplemental_sol_workload_row(
        invocation, binding
    )
    assert rebound["capture_target"]["source_workload_index"] == 1


@pytest.mark.parametrize(
    ("cohort", "claim", "predictive"),
    [
        ("native", "native_workload_row", True),
        ("partial_native_row", "partial_native_workload_row_only", False),
        ("reduced_proxy", "reduced_proxy_bundle_only", False),
    ],
)
def test_claim_arms_are_distinct_and_bound(
    tmp_path, cohort, claim, predictive,
):
    path, rows, values = _literal_rows(tmp_path)
    invocation, binding = _binding(
        path=path, rows=rows, values=values, cohort=cohort
    )
    supplemental_workload_path = None
    if cohort == "reduced_proxy":
        invocation = _externalize_reduced_invocation(invocation, path)
        path.chmod(0o444)
        supplemental_workload_path = path
    assert binding["claim_scope"] == claim
    assert binding["native_predictive"] is predictive
    canonical_capture.bind_supplemental_sol_workload_row(
        invocation, binding,
        supplemental_workload_path=supplemental_workload_path,
    )
    conflated = copy.deepcopy(binding)
    conflated["native_predictive"] = not predictive
    with pytest.raises(
        canonical_capture.ChunkRefusal, match="sol_workload_row_claim_invalid"
    ):
        canonical_capture.bind_supplemental_sol_workload_row(
            invocation, conflated,
            supplemental_workload_path=supplemental_workload_path,
        )


def test_reduced_runtime_uses_retained_bytes_after_original_owner_deletion(
    tmp_path, monkeypatch,
):
    original = tmp_path / "original-owner"
    retained_owner = tmp_path / "campaign-owner"
    original.mkdir(mode=0o700)
    retained_owner.mkdir(mode=0o700)
    original_path, rows, values = _literal_rows(original)
    retained = retained_owner / original_path.name
    retained.write_bytes(original_path.read_bytes())
    retained.chmod(0o444)
    invocation, binding = _binding(
        path=retained, rows=rows, values=values, cohort="reduced_proxy"
    )
    invocation = _externalize_reduced_invocation(invocation, retained)
    original_path.unlink()
    original.rmdir()
    rebound = canonical_capture.bind_supplemental_sol_workload_row(
        invocation, binding, supplemental_workload_path=retained
    )
    assert rebound["trace_app_args"][3] == str(retained)
    assert rebound["logical_workload"]["workload_file"] == str(retained)

    observed = {}
    monkeypatch.setattr(
        canonical_capture, "_sol_certifying_environment",
        lambda _invocation: ({}, "a" * 64),
    )
    monkeypatch.setattr(
        canonical_capture, "_sol_flashinfer_corpus_contract",
        lambda _invocation: ({"category": "L1"}, "b" * 64, None),
    )

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        return 0.0, tmp_path / "stdout", tmp_path / "stderr"

    monkeypatch.setattr(canonical_capture, "_run_logged", fake_run)
    monkeypatch.setattr(
        canonical_capture, "_validate_sol_prepare_manifest",
        lambda **_kwargs: {"manifest": {"selection": {"source_workload_index": 1}}},
    )
    cell_dir = tmp_path / "attempt" / "cell"
    cell_dir.mkdir(parents=True)
    canonical_capture.prepare_sol_capture(
        cell={"identity": {"slug": "problem_001_test"}},
        binary=Path("/bin/true"), invocation=rebound,
        cell_dir=cell_dir, runtime_cwd=REPO_ROOT, runtime_env={}, timeout_s=5,
    )
    assert observed["command"][4] == str(retained)


def test_capture_attempt_freezes_reduced_bytes_before_prepare(
    tmp_path, monkeypatch,
):
    retained_owner = tmp_path / "campaign-owner"
    retained_owner.mkdir(mode=0o700)
    retained, rows, values = _literal_rows(retained_owner)
    retained_definition = retained_owner / "reduced-definition.json"
    retained_definition.write_text(json.dumps({
        "name": "problem_001_test",
        "axes": {"num_experts": {"type": "const", "value": 32}},
    }, sort_keys=True) + "\n")
    invocation, binding = _binding(
        path=retained, rows=rows, values=values, cohort="reduced_proxy"
    )
    invocation = _externalize_reduced_invocation(
        invocation, retained, definition_path=retained_definition,
    )
    original_payload = retained.read_bytes()
    original_definition_payload = retained_definition.read_bytes()
    retained.chmod(0o444)
    retained_definition.chmod(0o444)
    kwargs, _journal = _attempt_capture_kwargs(
        tmp_path, binding, monkeypatch
    )
    kwargs["cell"]["invocation"] = invocation
    kwargs["supplemental_workload_path"] = retained
    kwargs["supplemental_definition_path"] = retained_definition

    observed: dict[str, object] = {}
    monkeypatch.setattr(
        canonical_capture, "_sol_certifying_environment",
        lambda _invocation: ({}, "a" * 64),
    )
    monkeypatch.setattr(
        canonical_capture, "_sol_flashinfer_corpus_contract",
        lambda _invocation: ({"category": "L1"}, "b" * 64, None),
    )

    def fake_run(command, **call_kwargs):
        observed["command"] = command
        observed["cwd"] = call_kwargs["cwd"]
        return 0.0, tmp_path / "stdout", tmp_path / "stderr"

    monkeypatch.setattr(canonical_capture, "_run_logged", fake_run)
    monkeypatch.setattr(
        canonical_capture, "_validate_sol_prepare_manifest",
        lambda **_kwargs: {
            "manifest": {"selection": {"source_workload_index": 1}}
        },
    )

    def fake_capture(**call):
        attempt_workload = call["supplemental_workload_path"]
        attempt_definition = call["supplemental_definition_path"]
        assert call["runnability_path"] == kwargs["runnability_path"]
        assert call["supplemental_runnability_evidence"] is None
        assert attempt_workload != retained
        assert attempt_definition != retained_definition
        assert attempt_workload.read_bytes() == original_payload
        assert attempt_definition.read_bytes() == original_definition_payload
        assert attempt_workload.stat().st_mode & 0o222 == 0
        assert attempt_definition.stat().st_mode & 0o222 == 0

        # Simulate the exact post-bind race that the attempt copy closes: the
        # campaign-retained source is replaced before the adapter is launched.
        retained.chmod(0o644)
        retained.unlink()
        retained.write_bytes(b'{"uuid":"bad","axes":{"batch":999}}\n')
        retained.chmod(0o444)
        retained_definition.chmod(0o644)
        retained_definition.unlink()
        retained_definition.write_text(json.dumps({
            "name": "bad", "axes": {"num_experts": {"value": 999}},
        }) + "\n")
        retained_definition.chmod(0o444)
        rebound = canonical_capture.bind_supplemental_sol_workload_row(
            call["cell"]["invocation"], call["workload_row_binding"],
            supplemental_workload_path=attempt_workload,
            supplemental_definition_path=attempt_definition,
        )
        canonical_capture.prepare_sol_capture(
            cell=call["cell"], binary=Path("/bin/true"),
            invocation=rebound, cell_dir=call["cell_dir"],
            runtime_cwd=REPO_ROOT, runtime_env={}, timeout_s=5,
        )
        assert observed["command"][4] == str(attempt_workload)
        assert observed["command"][2] == str(attempt_definition)
        payload = call["cell_dir"] / "payload.bin"
        payload.write_bytes(b"complete evidence")
        return {
            "workload_row_binding": copy.deepcopy(binding),
            "artifacts": [{
                "relative_path": (
                    f"cells/{capture._safe_name(binding['row_key'])}/"
                    "payload.bin"
                ),
                "sha256": capture.sha256_file(payload),
                "size_bytes": payload.stat().st_size,
                "artifact_kind": "synthetic",
            }],
        }

    monkeypatch.setattr(capture.canonical_capture, "capture_cell", fake_capture)
    capture._capture_one(**kwargs)
    assert retained.read_bytes() != original_payload
    assert retained_definition.read_bytes() != original_definition_payload


def test_reduced_runtime_refuses_leaf_intermediate_symlink_and_swap(tmp_path):
    owner = tmp_path / "owner"
    owner.mkdir(mode=0o700)
    retained, rows, values = _literal_rows(owner)
    invocation, binding = _binding(
        path=retained, rows=rows, values=values, cohort="reduced_proxy"
    )
    invocation = _externalize_reduced_invocation(invocation, retained)
    retained.chmod(0o444)
    leaf = tmp_path / "leaf.jsonl"
    leaf.symlink_to(retained)
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_external_workload_path_invalid",
    ):
        canonical_capture.bind_supplemental_sol_workload_row(
            invocation, binding, supplemental_workload_path=leaf
        )
    alias = tmp_path / "owner-alias"
    alias.symlink_to(owner, target_is_directory=True)
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_external_workload_path_invalid",
    ):
        canonical_capture.bind_supplemental_sol_workload_row(
            invocation, binding,
            supplemental_workload_path=alias / retained.name,
        )
    canonical_capture.bind_supplemental_sol_workload_row(
        invocation, binding, supplemental_workload_path=retained
    )
    retained.unlink()
    retained.write_bytes(b'{"uuid":"bad","axes":{"batch":7}}\n')
    retained.chmod(0o444)
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_external_workload_rebind_invalid",
    ):
        canonical_capture.bind_supplemental_sol_workload_row(
            invocation, binding, supplemental_workload_path=retained
        )


def test_external_reduced_sources_bind_workload_and_definition_to_one_owner(
    tmp_path,
):
    owner = tmp_path / "shrink-owner"
    owner.mkdir(mode=0o700)
    workload = owner / "cohort.jsonl"
    workload.write_text(
        '{"uuid":"0123456789abcdef0123456789abcdef","axes":{}}\n'
    )
    definition = owner / "definition.proxy.json"
    definition.write_text(json.dumps({
        "name": "problem_001_test",
        "axes": {"num_experts": {"type": "const", "value": 32}},
    }, sort_keys=True) + "\n")
    workload.chmod(0o444)
    definition.chmod(0o444)
    marker = {
        "name": ".owner.json", "sha256": "a" * 64,
        "owner_seal_sha256": "b" * 64,
    }

    def source(path: Path, kind: str) -> dict:
        return {
            "kind": kind,
            "name": path.name,
            "sha256": capture.sha256_file(path),
            "size_bytes": path.stat().st_size,
            "owner_marker": copy.deepcopy(marker),
            "runtime_rebind": "campaign_retained_immutable_child",
        }

    logical = {
        "workload_file": workload.name,
        "workload_file_sha256": capture.sha256_file(workload),
        "external_workload_source": source(
            workload, "shrink_external_child_v1"
        ),
        "definition_file": definition.name,
        "definition_file_sha256": capture.sha256_file(definition),
        "external_definition_source": source(
            definition, "shrink_external_definition_child_v1"
        ),
    }
    variant_sha = "d" * 64
    matrix = {
        "reduced_proxy_authority": {"owner_marker": marker},
        "variants": [{
            "cohort": "reduced_proxy",
            "variant_id": "e" * 64,
            "variant_contract_sha256": variant_sha,
            "invocation": {"logical_workload": logical},
        }],
    }
    sources, roles = capture.validate_external_reduced_sources(
        matrix, shrink_root=owner,
    )
    assert set(sources) == {
        roles[variant_sha]["workload"],
        roles[variant_sha]["definition"],
    }
    assert sources[roles[variant_sha]["workload"]] == workload
    assert sources[roles[variant_sha]["definition"]] == definition

    bad = copy.deepcopy(matrix)
    bad["variants"][0]["invocation"]["logical_workload"][
        "external_definition_source"
    ]["sha256"] = "f" * 64
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_reduced_source_contract_invalid",
    ):
        capture.validate_external_reduced_sources(bad, shrink_root=owner)


def test_retained_reduced_inputs_select_only_current_chunk_variants(tmp_path):
    output = tmp_path / "campaign"
    retained = output / "inputs"
    retained.mkdir(parents=True)
    workload = retained / "workload.jsonl"
    definition = retained / "definition.json"
    workload.write_bytes(b'{"uuid":"u","axes":{}}\n')
    definition.write_bytes(b'{"name":"problem_001_test"}\n')
    workload.chmod(0o444)
    definition.chmod(0o444)
    current_variant = "1" * 64
    other_variant = "2" * 64
    workload_role = "reduced_workload_" + "3" * 64
    definition_role = "reduced_definition_" + "4" * 64
    logical = {
        "external_workload_source": {
            "sha256": capture.sha256_file(workload),
            "size_bytes": workload.stat().st_size,
        },
        "external_definition_source": {
            "sha256": capture.sha256_file(definition),
            "size_bytes": definition.stat().st_size,
        },
    }
    resolved = [(
        {},
        {
            "cohort": "reduced_proxy",
            "variant_contract_sha256": current_variant,
            "invocation": {"logical_workload": logical},
        },
        {},
    )]
    roles = {
        current_variant: {
            "workload": workload_role, "definition": definition_role,
        },
        other_variant: {
            "workload": workload_role, "definition": definition_role,
        },
    }
    campaign_identity = {"input_file_identities": {
        workload_role: {
            "sha256": capture.sha256_file(workload),
            "size_bytes": workload.stat().st_size,
            "retained_owner_relative_path": "inputs/workload.jsonl",
        },
        definition_role: {
            "sha256": capture.sha256_file(definition),
            "size_bytes": definition.stat().st_size,
            "retained_owner_relative_path": "inputs/definition.json",
        },
    }}
    workload_paths, definition_paths = capture.retained_reduced_input_paths(
        resolved=resolved, roles_by_variant=roles,
        campaign_identity=campaign_identity, output_root=output,
    )
    assert workload_paths == {current_variant: workload}
    assert definition_paths == {current_variant: definition}


def test_binding_refuses_uuid_axes_index_and_hash_drift(tmp_path):
    path, rows, values = _literal_rows(tmp_path)
    invocation, binding = _binding(path=path, rows=rows, values=values)
    mutations = (
        ("source_workload_uuid", "12345678-1234-5123-8123-123456789abc"),
        ("capture_axes", {"batch": 8}),
        ("capture_workload_index", 0),
        ("capture_row_sha256", "0" * 64),
    )
    for field, value in mutations:
        bad = copy.deepcopy(binding)
        bad[field] = value
        if field == "source_workload_uuid":
            bad["row_key"]["source_workload_uuid"] = value
        with pytest.raises(canonical_capture.ChunkRefusal):
            canonical_capture.bind_supplemental_sol_workload_row(
                invocation, bad
            )


def test_output_root_refusal_is_read_only(tmp_path):
    protected = REPO_ROOT / "results" / "forbidden-supplemental-test"
    assert not protected.exists()
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_output_root_in_repository",
    ):
        capture.assert_external_output_root_before_mutation(protected)
    assert not protected.exists()
    external = tmp_path / "external"
    capture.assert_external_output_root_before_mutation(external)
    assert not external.exists()


def test_retention_budget_refuses_before_new_cell_output(tmp_path):
    output = tmp_path / "campaign"
    output.mkdir()
    (output / "existing.bin").write_bytes(b"x" * 64)
    budget = {
        "retention_budget_gib": 1,
        "metadata_margin_gib": 0,
        "raw_staging_margin_gib": 1,
    }
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_retention_budget_exceeded",
    ):
        capture._require_retention_capacity(
            output_root=output, aggregate_budget=budget,
            next_retained_max=1024 ** 3,
        )
    assert sorted(path.name for path in output.iterdir()) == ["existing.bin"]


_PROJECTION_STORAGE_ERRORS = [
    (errno.ENOSPC, capture.STORAGE_CAPACITY_CODE),
    (errno.EDQUOT, capture.STORAGE_CAPACITY_CODE),
    (errno.EIO, capture.STORAGE_INTEGRITY_CODE),
    (errno.EROFS, capture.STORAGE_INTEGRITY_CODE),
    (errno.ENXIO, capture.STORAGE_INTEGRITY_CODE),
    (errno.ESTALE, capture.STORAGE_INTEGRITY_CODE),
]


def _projection_helper_fixture(tmp_path, helper_name):
    root = tmp_path / helper_name
    root.mkdir()
    if helper_name == "content_address":
        source = root / "source"
        source.mkdir()
        payload = source / "payload.bin"
        payload.write_bytes(b"content-address payload")
        destination = root / "objects"
        invoke = lambda: capture._content_address_tree(
            source=source, object_store=destination,
        )
    elif helper_name == "retire":
        attempt = root / "attempt"
        source = attempt / "cells/row/trace_full_raw"
        source.mkdir(parents=True)
        payload = source / "payload.bin"
        payload.write_bytes(b"transient raw payload")
        destination = attempt
        invoke = lambda: capture._retire_failed_transient_raw(
            attempt_dir=attempt, cell_dir=source.parent,
        )
    elif helper_name in {"hardlink", "resume"}:
        source = root / "source"
        source.mkdir()
        payload = source / "payload.bin"
        payload.write_bytes(b"hardlink projection payload")
        destination = root / "destination"
        if helper_name == "resume":
            payload.chmod(0o444)
            source.chmod(0o555)
            destination.mkdir()
            invoke = lambda: capture._resume_hardlink_tree(
                source, destination,
            )
        else:
            invoke = lambda: capture._hardlink_tree(source, destination)
    else:  # pragma: no cover - the parametrization is the closed caller.
        raise AssertionError(helper_name)
    return SimpleNamespace(
        source=source, destination=destination, payload=payload,
        invoke=invoke,
    )


@pytest.mark.parametrize(
    ("helper_name", "scan_site"),
    [
        ("content_address", "source"),
        ("retire", "source"),
        ("hardlink", "source"),
        ("hardlink", "destination"),
        ("resume", "source"),
        ("resume", "destination"),
    ],
)
@pytest.mark.parametrize(
    ("error_number", "expected_code"), _PROJECTION_STORAGE_ERRORS,
)
def test_projection_tree_scandir_fault_never_succeeds_or_omits_bytes(
    tmp_path, monkeypatch, helper_name, scan_site,
    error_number, expected_code,
):
    fixture = _projection_helper_fixture(tmp_path, helper_name)
    target = getattr(fixture, scan_site)
    real_scandir = capture.os.scandir

    def fail_scandir(path):
        if Path(path) == target:
            raise OSError(error_number, "injected projection scan failure")
        return real_scandir(path)

    monkeypatch.setattr(capture.os, "scandir", fail_scandir)
    with pytest.raises(capture.WorkloadCaptureRefusal) as caught:
        fixture.invoke()
    assert caught.value.code == expected_code
    assert json.loads(caught.value.detail)["errno"] == error_number


@pytest.mark.parametrize(
    ("helper_name", "stat_site"),
    [
        ("content_address", "source"),
        ("content_address", "destination"),
        ("retire", "source"),
        ("hardlink", "source"),
        ("hardlink", "destination"),
        ("resume", "source"),
        ("resume", "destination"),
    ],
)
@pytest.mark.parametrize(
    "error_number", [errno.EIO, errno.ESTALE],
)
def test_projection_tree_lstat_fault_is_typed(
    tmp_path, monkeypatch, helper_name, stat_site, error_number,
):
    fixture = _projection_helper_fixture(tmp_path, helper_name)
    target = getattr(fixture, stat_site)
    real_lstat = Path.lstat

    def fail_lstat(path, *args, **kwargs):
        if path == target:
            raise OSError(error_number, "injected projection lstat failure")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", fail_lstat)
    with pytest.raises(capture.WorkloadCaptureRefusal) as caught:
        fixture.invoke()
    assert caught.value.code == capture.STORAGE_INTEGRITY_CODE
    assert json.loads(caught.value.detail)["errno"] == error_number


@pytest.mark.parametrize("failure_site", ["open", "read"])
@pytest.mark.parametrize("error_number", [errno.EIO, errno.ESTALE])
def test_projection_output_hash_fault_is_typed_at_open_and_read(
    tmp_path, monkeypatch, failure_site, error_number,
):
    fixture = _projection_helper_fixture(tmp_path, "content_address")
    real_open = capture._open_regular_nofollow
    real_read = capture.os.read
    target_descriptor = None

    def injected_open(path, *, role):
        nonlocal target_descriptor
        if path == fixture.payload and failure_site == "open":
            refusal = capture.WorkloadCaptureRefusal(
                "supplemental_input_path_invalid", f"{role}: injected",
            )
            refusal.__cause__ = OSError(
                error_number, "injected projection hash open failure",
            )
            raise refusal
        opened = real_open(path, role=role)
        if path == fixture.payload:
            target_descriptor = opened[0]
        return opened

    def injected_read(descriptor, count):
        if failure_site == "read" and descriptor == target_descriptor:
            raise OSError(
                error_number, "injected projection hash read failure",
            )
        return real_read(descriptor, count)

    monkeypatch.setattr(capture, "_open_regular_nofollow", injected_open)
    monkeypatch.setattr(capture.os, "read", injected_read)
    with pytest.raises(capture.WorkloadCaptureRefusal) as caught:
        fixture.invoke()
    assert caught.value.code == capture.STORAGE_INTEGRITY_CODE
    detail = json.loads(caught.value.detail)
    assert detail["errno"] == error_number
    assert detail["operation"] == f"content_address_source_hash_{failure_site}"


def test_dedup_and_publication_hardlinks_count_physical_bytes_once(tmp_path):
    output = tmp_path / "supplemental"
    first = output / "attempt-a" / "cells" / "row"
    second = output / "attempt-b" / "cells" / "row"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    payload = b"same retained evidence" * 4096
    (first / "enhanced_execution_info.bin").write_bytes(payload)
    (second / "enhanced_execution_info.bin").write_bytes(payload)
    store = output / "sol_workload_object_store" / "sha256"

    capture._content_address_tree(source=first, object_store=store)
    capture._content_address_tree(source=second, object_store=store)
    first_stat = (first / "enhanced_execution_info.bin").stat()
    second_stat = (second / "enhanced_execution_info.bin").stat()
    assert (first_stat.st_dev, first_stat.st_ino) == (
        second_stat.st_dev, second_stat.st_ino
    )

    publication = output / "publication" / "cells" / "row"
    capture._hardlink_tree(first, publication)
    published_stat = (publication / "enhanced_execution_info.bin").stat()
    assert (published_stat.st_dev, published_stat.st_ino) == (
        first_stat.st_dev, first_stat.st_ino
    )
    assert capture.sha256_file(
        publication / "enhanced_execution_info.bin"
    ) == hashlib.sha256(payload).hexdigest()
    usage = capture._tree_usage(output)
    assert usage["logical_bytes"] > usage["physical_bytes"]
    assert usage["physical_bytes"] >= len(payload)
    assert (published_stat.st_mode & 0o222) == 0


def test_nested_dedup_and_resume_project_every_source_byte(tmp_path):
    source = tmp_path / "source"
    nested = source / "nested/deeper"
    nested.mkdir(parents=True)
    payloads = {
        source / "root.bin": b"root payload",
        source / "nested/child.bin": b"child payload",
        nested / "leaf.bin": b"leaf payload",
    }
    for path, payload in payloads.items():
        path.write_bytes(payload)
    object_store = tmp_path / "objects"
    capture._content_address_tree(
        source=source, object_store=object_store,
    )
    for path, payload in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        target = object_store / digest[:2] / digest
        observed = path.stat()
        stored = target.stat()
        assert (observed.st_dev, observed.st_ino) == (
            stored.st_dev, stored.st_ino
        )

    for directory in (nested, nested.parent, source):
        directory.chmod(0o555)
    destination = tmp_path / "pending"
    destination.mkdir()
    os.link(source / "root.bin", destination / "root.bin")
    capture._resume_hardlink_tree(source, destination)
    expected = {
        path.relative_to(source).as_posix() for path in source.rglob("*")
    }
    observed = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
    }
    assert observed == expected
    for source_path in payloads:
        projected = destination / source_path.relative_to(source)
        source_stat = source_path.stat()
        projected_stat = projected.stat()
        assert (source_stat.st_dev, source_stat.st_ino) == (
            projected_stat.st_dev, projected_stat.st_ino
        )
    capture._resume_hardlink_tree(source, destination)
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
    } == expected


def test_deterministic_publication_path_and_partial_projection_resume(tmp_path):
    matrix_path = tmp_path / "matrix.json"
    canonical_path = tmp_path / "canonical.json"
    matrix_path.write_bytes(b"matrix bytes\n")
    canonical_path.write_bytes(b"canonical bytes\n")
    kwargs = {
        "stage_root": tmp_path / "stage",
        "chunk_id": "sol-workload-native-001",
        "source_commit": "a" * 40,
        "matrix_path": matrix_path,
        "matrix_seal_sha256": "b" * 64,
        "canonical_matrix_path": canonical_path,
        "canonical_matrix_seal_sha256": "c" * 64,
    }
    assert capture._publication_pending_path(
        **kwargs
    ) == capture._publication_pending_path(**kwargs)

    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "first.bin").write_bytes(b"first immutable payload")
    (source / "nested" / "second.bin").write_bytes(
        b"second immutable payload"
    )
    (source / "first.bin").chmod(0o444)
    (source / "nested" / "second.bin").chmod(0o444)
    (source / "nested").chmod(0o555)
    source.chmod(0o555)
    destination = tmp_path / "pending"
    destination.mkdir()
    os.link(source / "first.bin", destination / "first.bin")
    capture._resume_hardlink_tree(source, destination)
    first_inventory = {
        path.relative_to(destination).as_posix(): (
            path.stat().st_dev, path.stat().st_ino, capture.sha256_file(path)
        )
        for path in destination.rglob("*") if path.is_file()
    }
    capture._resume_hardlink_tree(source, destination)
    assert {
        path.relative_to(destination).as_posix(): (
            path.stat().st_dev, path.stat().st_ino, capture.sha256_file(path)
        )
        for path in destination.rglob("*") if path.is_file()
    } == first_inventory


def test_deterministic_publication_refuses_corrupt_or_extra_pending_bytes(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"expected")
    (source / "payload.bin").chmod(0o444)
    source.chmod(0o555)
    destination = tmp_path / "pending"
    destination.mkdir()
    wrong = destination / "payload.bin"
    wrong.write_bytes(b"corrupt")
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_publication_link_mismatch",
    ):
        capture._resume_hardlink_tree(source, destination)
    assert wrong.read_bytes() == b"corrupt"

    wrong.unlink()
    extra = destination / "unbound.bin"
    extra.write_bytes(b"foreign")
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_publication_pending_extra",
    ):
        capture._resume_hardlink_tree(source, destination)
    assert extra.read_bytes() == b"foreign"


def test_deterministic_publication_refuses_writable_hardlink_without_repair(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "payload.bin"
    source_file.write_bytes(b"sealed payload")
    source_file.chmod(0o444)
    source.chmod(0o555)
    destination = tmp_path / "pending"
    destination.mkdir()
    projected = destination / "payload.bin"
    os.link(source_file, projected)
    projected.chmod(0o644)
    before_mode = projected.stat().st_mode & 0o777
    before_sha = capture.sha256_file(projected)
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_publication_source_mutable",
    ):
        capture._resume_hardlink_tree(source, destination)
    assert projected.stat().st_mode & 0o777 == before_mode
    assert capture.sha256_file(projected) == before_sha


def _finalize_fixture(tmp_path, monkeypatch):
    key = {
        "slug": "problem_001_finalize_fixture",
        "source_workload_uuid": "00000001-1234-5123-8123-123456789abc",
        "variant_id": "native",
    }
    row = {"key": key}
    safe_name = capture._safe_name(key)
    stage = tmp_path / "stage"
    attempt = (
        stage / "journals" / safe_name / "attempts" / "attempt-000001"
    )
    source = attempt / "cells" / safe_name
    source.mkdir(parents=True)
    payload = source / "payload.bin"
    payload.write_bytes(b"sealed finalization payload")
    payload.chmod(0o444)
    source.chmod(0o555)
    (attempt / "journal_complete.json").write_text("{}\n")
    matrix_path = tmp_path / "matrix.json"
    canonical_path = tmp_path / "canonical.json"
    matrix_path.write_text("{}\n")
    canonical_path.write_text("{}\n")
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({
        "retention_budget_gib": 100,
        "metadata_margin_gib": 1,
        "raw_staging_margin_gib": 1,
    }))
    result = {"workload_row_binding": {"row_key": key}, "artifacts": []}
    build_calls = []

    def build_manifest(**kwargs):
        build_calls.append(kwargs)
        return {"cell_results": copy.deepcopy(kwargs["results"]), "seal": "ok"}

    monkeypatch.setattr(capture, "build_done_manifest", build_manifest)
    monkeypatch.setattr(capture, "verify_done_manifest", lambda *_a, **_k: None)
    monkeypatch.setattr(capture, "_verify_artifacts", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        capture.canonical_capture, "_git_state",
        lambda: ("a" * 40, True, []),
    )
    output = tmp_path / "output"
    output.mkdir()
    final = output / "sol_workload_chunks/chunk-001"
    kwargs = {
        "output_root": output, "stage_root": stage, "final_dir": final,
        "chunk": {"chunk_id": "chunk-001"},
        "resolved": [(row, {}, {})], "results": {safe_name: result},
        "source_commit": "a" * 40, "matrix_path": matrix_path,
        "matrix": {"matrix_seal_sha256": "b" * 64},
        "canonical_matrix_path": canonical_path,
        "canonical_matrix": {"matrix_seal_sha256": "c" * 64},
        "authority_path": tmp_path / "authority",
        "pins_path": tmp_path / "pins",
        "aggregate_budget_path": budget,
        "cell_budget_path": tmp_path / "cell-budget",
        "registry_path": tmp_path / "registry",
        "preflight_path": tmp_path / "preflight",
        "flashinfer_proof": {},
    }
    pending = capture._publication_pending_path(
        stage_root=stage, chunk_id="chunk-001", source_commit="a" * 40,
        matrix_path=matrix_path, matrix_seal_sha256="b" * 64,
        canonical_matrix_path=canonical_path,
        canonical_matrix_seal_sha256="c" * 64,
    )
    return kwargs, pending, source, safe_name, build_calls


def test_finalize_resumes_partial_projection_and_done_fsync_window(
    tmp_path, monkeypatch,
):
    kwargs, pending, source, safe_name, build_calls = _finalize_fixture(
        tmp_path, monkeypatch
    )
    projected = pending / "cells" / safe_name
    projected.mkdir(parents=True)
    os.link(source / "payload.bin", projected / "payload.bin")
    inode = (projected / "payload.bin").stat().st_ino
    real_rename = capture._rename_noreplace
    monkeypatch.setattr(
        capture, "_rename_noreplace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected after DONE fsync")
        ),
    )
    with pytest.raises(OSError, match="injected after DONE fsync"):
        capture._finalize(**kwargs)
    done_bytes = (pending / "DONE.json").read_bytes()
    assert (pending / "cells" / safe_name / "payload.bin").stat().st_ino == inode
    assert len(build_calls) == 1

    monkeypatch.setattr(capture, "_rename_noreplace", real_rename)
    assert capture._finalize(**kwargs) == kwargs["final_dir"]
    assert (kwargs["final_dir"] / "DONE.json").read_bytes() == done_bytes
    assert (
        kwargs["final_dir"] / "cells" / safe_name / "payload.bin"
    ).stat().st_ino == inode
    assert len(build_calls) == 1


def test_finalize_restart_after_rename_before_parent_fsync_reuses_final(
    tmp_path, monkeypatch,
):
    kwargs, _pending, _source, safe_name, build_calls = _finalize_fixture(
        tmp_path, monkeypatch
    )
    real_fsync_directory = capture._fsync_directory

    def fail_after_rename(path):
        if path == kwargs["final_dir"].parent and kwargs["final_dir"].exists():
            raise OSError("injected parent fsync failure")
        return real_fsync_directory(path)

    monkeypatch.setattr(capture, "_fsync_directory", fail_after_rename)
    with pytest.raises(OSError, match="injected parent fsync failure"):
        capture._finalize(**kwargs)
    done_bytes = (kwargs["final_dir"] / "DONE.json").read_bytes()
    inode = (
        kwargs["final_dir"] / "cells" / safe_name / "payload.bin"
    ).stat().st_ino
    recovered_fsyncs = []

    def track_recovery_fsync(path):
        recovered_fsyncs.append(path)
        return real_fsync_directory(path)

    monkeypatch.setattr(capture, "_fsync_directory", track_recovery_fsync)
    assert capture._finalize(**kwargs) == kwargs["final_dir"]
    assert kwargs["final_dir"].parent in recovered_fsyncs
    assert (kwargs["final_dir"] / "DONE.json").read_bytes() == done_bytes
    assert (
        kwargs["final_dir"] / "cells" / safe_name / "payload.bin"
    ).stat().st_ino == inode
    assert len(build_calls) == 1


def test_finalize_refuses_corrupt_pending_done_without_rewrite(
    tmp_path, monkeypatch,
):
    kwargs, pending, _source, _safe_name, build_calls = _finalize_fixture(
        tmp_path, monkeypatch
    )
    pending.mkdir(parents=True)
    corrupt = capture._canonical_json_bytes({"cell_results": [], "foreign": True})
    (pending / "DONE.json").write_bytes(corrupt)
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_publication_pending_mismatch",
    ):
        capture._finalize(**kwargs)
    assert (pending / "DONE.json").read_bytes() == corrupt
    assert build_calls == []


def test_atomic_publication_race_preserves_pending_and_foreign_destination(
    tmp_path,
):
    pending = tmp_path / "pending"
    final = tmp_path / "final"
    pending.mkdir()
    (pending / "ours").write_bytes(b"ours")
    # Simulate a destination created after the caller's existence check but
    # before the atomic promotion.
    final.mkdir()
    (final / "foreign").write_bytes(b"foreign")
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_publication_destination_exists",
    ):
        capture._rename_noreplace(pending, final)
    assert (pending / "ours").read_bytes() == b"ours"
    assert (final / "foreign").read_bytes() == b"foreign"


def test_root_owner_rejects_canonical_aliases_and_wrong_existing_owner(
    tmp_path, monkeypatch,
):
    canonical = tmp_path / "canonical-r27"
    canonical.mkdir()
    (canonical / "sentinel").write_text("protected\n")
    monkeypatch.setattr(
        capture, "PROTECTED_CANONICAL_CAMPAIGN_ROOTS", (canonical.resolve(),)
    )
    for refused in (
        canonical,
        canonical / "nested-supplemental",
        tmp_path,
    ):
        with pytest.raises(
            capture.WorkloadCaptureRefusal,
            match="supplemental_output_root_aliases_canonical",
        ):
            capture.assert_external_output_root_before_mutation(refused)
    alias = tmp_path.parent / f"{tmp_path.name}-canonical-alias"
    alias.symlink_to(canonical, target_is_directory=True)
    try:
        with pytest.raises(capture.WorkloadCaptureRefusal):
            capture.assert_external_output_root_before_mutation(alias / "child")
    finally:
        alias.unlink()
    assert (canonical / "sentinel").read_text() == "protected\n"

    unrelated = tmp_path.parent / f"{tmp_path.name}-existing-owner"
    unrelated.mkdir()
    (unrelated / "foreign.owner").write_text("do not touch\n")
    try:
        with pytest.raises(
            capture.WorkloadCaptureRefusal,
            match="supplemental_output_root_identity_missing",
        ):
            capture.establish_supplemental_output_root(
                unrelated, campaign_identity={}
            )
        assert (unrelated / "foreign.owner").read_text() == "do not touch\n"
        assert not (unrelated / capture.SUPPLEMENTAL_ROOT_MARKER).exists()
    finally:
        (unrelated / "foreign.owner").unlink()
        unrelated.rmdir()


def test_done_v2_offline_verifies_after_campaign_relocation_and_source_removal(
    tmp_path, monkeypatch,
):
    variants, rows, *_ = matrix_fixture._native_fixture(tmp_path / "fixture")
    variant = variants[0]
    matrix = matrix_fixture._matrix_document([variant], rows[:1])
    chunk = matrix["chunks"][0]
    canonical_path = REPO_ROOT / (
        "data/local_labels/campaign/sm86_reference_capture_matrix.json"
    )
    canonical_matrix = json.loads(canonical_path.read_text())
    source_root = tmp_path / "source-inputs"
    source_root.mkdir(mode=0o700)
    sources: dict[str, Path] = {}
    for role in (
        "authority", "pins", "aggregate_budget", "cell_budgets",
        "metric_registry", "full_coverage_inventory",
        "live_coverage_inventory", "runnability", "shrink_provenance",
    ):
        path = source_root / f"{role}.json"
        path.write_bytes(capture._canonical_json_bytes({"role": role}))
        sources[role] = path
    matrix_path = source_root / "supplemental-matrix.json"
    matrix_path.write_bytes(capture._canonical_json_bytes(matrix))
    canonical_copy = source_root / "canonical-matrix.json"
    canonical_copy.write_bytes(canonical_path.read_bytes())
    sources.update({
        "supplemental_matrix": matrix_path,
        "canonical_matrix": canonical_copy,
    })
    campaign_identity = capture.supplemental_campaign_identity(
        matrix=matrix, matrix_path=matrix_path,
        canonical_matrix=canonical_matrix,
        canonical_matrix_path=canonical_copy,
        authority_path=sources["authority"], pins_path=sources["pins"],
        aggregate_budget_path=sources["aggregate_budget"],
        cell_budget_path=sources["cell_budgets"],
        registry_path=sources["metric_registry"],
        full_coverage_inventory_path=sources["full_coverage_inventory"],
        live_coverage_inventory_path=sources["live_coverage_inventory"],
        runnability_path=sources["runnability"],
        shrink_path=sources["shrink_provenance"],
    )
    campaign_root = tmp_path / "campaign"
    capture.establish_supplemental_output_root(
        campaign_root, campaign_identity=campaign_identity
    )
    capture.retain_campaign_inputs(
        campaign_root, campaign_identity=campaign_identity, sources=sources
    )
    preflight_parent = (
        campaign_root / "sol_workload_campaign/inputs/preflights"
        / matrix["source_commit"] / chunk["chunk_id"]
    )
    preflight_parent.mkdir(parents=True)
    failed_preflight = preflight_parent / "preflight-failed.json"
    failed_preflight.write_bytes(
        capture._canonical_json_bytes({"attempt": "failed"})
    )
    failed_preflight.chmod(0o444)
    preflight = preflight_parent / "preflight-success.json"
    preflight.write_bytes(
        capture._canonical_json_bytes({"attempt": "success"})
    )
    preflight.chmod(0o444)

    final_dir = campaign_root / "sol_workload_chunks" / chunk["chunk_id"]
    binding = capture.workload_row_binding(
        matrix=matrix, matrix_sha256=capture.sha256_file(matrix_path),
        chunk_id=chunk["chunk_id"], row=matrix["rows"][0],
    )
    safe_name = capture._safe_name(binding["row_key"])
    artifact = final_dir / "cells" / safe_name / "payload.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"portable retained trace")
    launch = {
        "launch_ordinal": 0,
        "launch_identity_sha256": "a" * 64,
        "kernel_name": "fixture_kernel",
        "trace_complete": True,
        "ncu_label_complete": True,
        "ncu_report_sha256": "b" * 64,
        "ncu_label_typed_reason": None,
        "control_complete": True,
        "control_outcome": "passed",
        "control_typed_reason": None,
    }
    sequence = [{
        "launch_ordinal": 0, "kernel_name": "fixture_kernel",
        "launch_identity_sha256": "a" * 64,
    }]
    sequence_sha = capture.canonical_sha256(sequence)
    completeness = {
        "schema_version": 1,
        "launch_scope": "all_in_profiler_roi",
        "contract": copy.deepcopy(LAUNCH_CONTRACT),
        "status": "complete_dynamic_launch_set",
        "launch_inventory_sha256": "c" * 64,
        "inventory_launch_count": 1,
        "trace_launch_sequence_sha256": sequence_sha,
        "ncu_replay_sequence_sha256": sequence_sha,
        "trace_ncu_sequence_equal": True,
        "trace_complete_launch_count": 1,
        "ncu_label_complete_launch_count": 1,
        "control_complete_launch_count": 1,
        "launches": [launch],
    }
    completeness["completeness_sha256"] = capture.canonical_sha256(
        completeness
    )
    result = {
        "roster_source": "sol_execbench",
        "slug": binding["row_key"]["slug"],
        "authority_cell_sha256": variant[
            "supplemental_cell_contract_sha256"
        ],
        "disposition": "admitted", "typed_reason": None, "detail": None,
        "capture_preflight_sha256": capture.sha256_file(preflight),
        "capture_started_at_utc": "2026-07-16T00:00:00.000000Z",
        "admission_run_id": "d" * 64, "supersedes": None,
        "prior_preparation_failures": [], "prior_capture_defect_attempts": [],
        "definition_equivalence": None,
        "definition_equivalence_attestation_sha256": None,
        "launch_completeness": completeness,
        "artifacts": [{
            "relative_path": artifact.relative_to(final_dir).as_posix(),
            "sha256": capture.sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
            "artifact_kind": "synthetic_trace",
        }],
        "workload_row_binding": binding,
    }
    failed_attempt = {
        "attempt_ordinal": 1, "attempt_id": "2" * 64,
        "state": "failed", "journal_start_seal_sha256": "3" * 64,
        "terminal_seal_sha256": "4" * 64,
        "capture_preflight_path": "capture_preflight.json",
        "capture_preflight_sha256": capture.sha256_file(failed_preflight),
        "file_count": 0, "logical_bytes": 0,
        "tree_inventory_sha256": "5" * 64,
    }
    failed_attempt["attempt_summary_sha256"] = capture.canonical_sha256(
        failed_attempt
    )
    attempt = {
        "attempt_ordinal": 2, "attempt_id": "e" * 64,
        "state": "complete", "journal_start_seal_sha256": "f" * 64,
        "terminal_seal_sha256": "0" * 64,
        "capture_preflight_path": "capture_preflight.json",
        "capture_preflight_sha256": capture.sha256_file(preflight),
        "file_count": 1, "logical_bytes": artifact.stat().st_size,
        "tree_inventory_sha256": "1" * 64,
    }
    attempt["attempt_summary_sha256"] = capture.canonical_sha256(attempt)
    history = {
        "row_key": copy.deepcopy(binding["row_key"]),
        "selected_attempt_ordinal": 2, "selected_attempt_id": "e" * 64,
        "attempts": [failed_attempt, attempt],
        "attempts_sha256": capture.canonical_sha256(
            [failed_attempt, attempt]
        ),
    }
    history["attempt_history_sha256"] = capture.canonical_sha256(history)
    monkeypatch.setattr(
        capture, "_chunk_attempt_histories", lambda **_kwargs: [history]
    )
    manifest = capture.build_done_manifest(
        chunk=chunk, results=[result], source_commit=matrix["source_commit"],
        matrix_path=matrix_path, matrix=matrix,
        canonical_matrix_path=canonical_copy,
        canonical_matrix=canonical_matrix,
        authority_path=sources["authority"], pins_path=sources["pins"],
        aggregate_budget_path=sources["aggregate_budget"],
        cell_budget_path=sources["cell_budgets"],
        registry_path=sources["metric_registry"], preflight_path=preflight,
        final_dir=final_dir, command="must not be retained",
        flashinfer_proof={
            "data_annex_manifest_sha256": capture.sol_recon.DATA_ANNEX_MANIFEST_SHA256,
            "filtered_manifest_sha256": capture.sol_recon.FLASHINFER_CORPUS_MANIFEST_SHA256,
            "file_count": capture.sol_recon.FLASHINFER_CORPUS_FILE_COUNT,
            "safetensors_count": capture.sol_recon.FLASHINFER_CORPUS_BLOB_COUNT,
        },
        stage_root=tmp_path / "deleted-staging",
        publication_root=final_dir, campaign_root=campaign_root,
    )
    assert [row["sha256"] for row in manifest["preflights"]] == sorted({
        capture.sha256_file(failed_preflight), capture.sha256_file(preflight),
    })
    capture._write_json_new(final_dir / "DONE.json", manifest)
    capture.verify_done_manifest(
        manifest, final_dir=final_dir, campaign_root=campaign_root,
        repo_root=REPO_ROOT, require_current_state=False,
    )

    relocated = tmp_path / "relocated-campaign"
    shutil.copytree(campaign_root, relocated)
    shutil.rmtree(campaign_root)
    shutil.rmtree(source_root)
    relocated_final = relocated / manifest["publication_relative_path"]
    relocated_manifest = capture._load_object(
        relocated_final / "DONE.json", role="relocated supplemental DONE"
    )
    capture.verify_done_manifest(
        relocated_manifest, final_dir=relocated_final,
        campaign_root=relocated, repo_root=REPO_ROOT,
        require_current_state=False,
    )

    bad = copy.deepcopy(relocated_manifest)
    bad["attempt_histories"][0]["attempts"][0][
        "capture_preflight_path"
    ] = "/tmp/ambient-preflight.json"
    bad["done_seal_sha256"] = capture.canonical_sha256({
        key: value for key, value in bad.items()
        if key != "done_seal_sha256"
    })
    with pytest.raises(
        capture.WorkloadCaptureRefusal, match="supplemental_done_schema_invalid"
    ):
        capture.verify_done_manifest(
            bad, final_dir=relocated_final, campaign_root=relocated,
            repo_root=REPO_ROOT, require_current_state=False,
        )


def test_tracked_done_reference_uses_retained_bytes_offline_and_optional_git(
    tmp_path, monkeypatch,
):
    payload = b"tracked bytes retained by campaign\n"
    source_commit = "a" * 40
    relative = (
        Path("sol_workload_campaign/inputs/source") / source_commit
        / "authority" / "authority.json"
    )
    original = tmp_path / "original-campaign"
    path = original / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    path.chmod(0o444)
    digest = hashlib.sha256(payload).hexdigest()
    identity = {
        "kind": "tracked_git_blob", "role": "authority",
        "repository_path": "data/authority.json",
        "source_commit": source_commit, "size_bytes": len(payload),
        "sha256": digest, "git_blob_sha256": digest,
        "retained_owner_relative_path": relative.as_posix(),
    }
    reference = capture._portable_input_reference(identity)
    relocated = tmp_path / "relocated-campaign"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)

    def forbidden_git(*_args, **_kwargs):
        raise AssertionError("offline retained verification consulted Git")

    monkeypatch.setattr(capture, "_git_blob_bytes", forbidden_git)
    assert capture._read_portable_input_reference(
        reference, campaign_root=relocated, repo_root=tmp_path / "gone-repo",
        source_commit=source_commit, verify_git_provenance=False,
    ) == payload

    calls = []

    def exact_git(commit, repository_path, **_kwargs):
        calls.append((commit, repository_path))
        return payload

    monkeypatch.setattr(capture, "_git_blob_bytes", exact_git)
    assert capture._read_portable_input_reference(
        reference, campaign_root=relocated, repo_root=tmp_path / "repo",
        source_commit=source_commit, verify_git_provenance=True,
    ) == payload
    assert calls == [(source_commit, "data/authority.json")]


def test_complete_owner_namespaces_survive_original_deletion_and_close_exactly(
    tmp_path,
):
    roots: dict[str, Path] = {}
    markers = {
        "plan": ".sol-live-coverage-plan-owner-v1.json",
        "runnability": ".sol-runnability-owner-v1.json",
        "shrink": ".sol-shrink-owner-v1.json",
    }
    children = {
        "plan": ("matrix.json", "live-inventory.json"),
        "runnability": ("runnability.json",),
        "shrink": (
            "provenance.json", "checkpoint.json", "variant-a.jsonl",
            "variant-b.jsonl", "cohort-a.jsonl",
        ),
    }
    for role in markers:
        root = tmp_path / f"original-{role}"
        root.mkdir(mode=0o700)
        roots[role] = root
        for name in (markers[role], *children[role]):
            path = root / name
            path.write_bytes(f"{role}:{name}\n".encode())
            path.chmod(0o444)
    contracts = capture.capture_owner_namespace_contracts(
        roots=roots, marker_names=markers
    )
    output = tmp_path / "campaign"
    output.mkdir(mode=0o700)
    capture.retain_owner_namespaces(
        output, contracts=contracts, source_roots=roots
    )
    for root in roots.values():
        shutil.rmtree(root)
    retained = capture.validate_retained_owner_namespaces(
        output, contracts=contracts
    )
    assert capture.capture_owner_namespace_contracts(
        roots=retained, marker_names=markers
    ) == contracts
    assert {
        path.name for path in retained["shrink"].iterdir()
    } == {markers["shrink"], *children["shrink"]}

    extra = retained["plan"] / "unbound.json"
    extra.write_bytes(b"unexpected\n")
    extra.chmod(0o444)
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_owner_namespace_closure_mismatch",
    ):
        capture.validate_retained_owner_namespaces(
            output, contracts=contracts
        )
    extra.unlink()
    victim = retained["shrink"] / "variant-b.jsonl"
    victim.chmod(0o600)
    victim.write_bytes(b"tampered\n")
    victim.chmod(0o444)
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_owner_namespace_hash_mismatch",
    ):
        capture.validate_retained_owner_namespaces(
            output, contracts=contracts
        )
    victim.chmod(0o600)
    victim.write_bytes(b"shrink:variant-b.jsonl\n")
    victim.chmod(0o444)
    (retained["runnability"] / "runnability.json").unlink()
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_owner_namespace_closure_mismatch",
    ):
        capture.validate_retained_owner_namespaces(
            output, contracts=contracts
        )


def test_intermediate_symlink_is_refused_without_touching_target(tmp_path):
    target = tmp_path / "protected-target"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_bytes(b"unchanged")
    container = tmp_path / "container"
    container.mkdir()
    (container / "alias").symlink_to(target, target_is_directory=True)
    before = capture.sha256_file(sentinel)
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_output_component_invalid",
    ):
        capture._mkdir_nofollow(
            container / "alias" / "must-not-exist", exist_ok=True
        )
    assert capture.sha256_file(sentinel) == before
    assert sorted(path.name for path in target.iterdir()) == ["sentinel"]


def test_frozen_canonical_base_validates_without_live_rederivation(monkeypatch):
    matrix = json.loads(capture.DEFAULT_CANONICAL_MATRIX.read_text())
    authority = json.loads(capture.DEFAULT_AUTHORITY.read_text())
    supplemental = {
        "canonical_matrix": {
            "path": capture.workload_matrix.CANONICAL_MATRIX_PATH.as_posix(),
            "sha256": capture.sha256_file(capture.DEFAULT_CANONICAL_MATRIX),
            "source_commit": matrix["source_commit"],
            "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        },
        "canonical_authority": {
            "path": matrix["authority"]["path"],
            "sha256": capture.sha256_file(capture.DEFAULT_AUTHORITY),
            "roster_id": authority["roster_id"],
        },
    }
    monkeypatch.setattr(
        capture.canonical_capture, "validate_matrix_inputs",
        lambda *_args, **_kwargs: pytest.fail("live canonical rederivation used"),
    )
    capture.validate_frozen_canonical_base(
        matrix, canonical_matrix_path=capture.DEFAULT_CANONICAL_MATRIX,
        authority_path=capture.DEFAULT_AUTHORITY,
        supplemental_matrix=supplemental,
    )

    stale_pin = copy.deepcopy(supplemental)
    stale_pin["canonical_matrix"]["sha256"] = "f" * 64
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_frozen_canonical_binding_mismatch",
    ):
        capture.validate_frozen_canonical_base(
            matrix, canonical_matrix_path=capture.DEFAULT_CANONICAL_MATRIX,
            authority_path=capture.DEFAULT_AUTHORITY,
            supplemental_matrix=stale_pin,
        )

    stale_seal = copy.deepcopy(matrix)
    stale_seal["matrix_seal_sha256"] = "e" * 64
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_canonical_matrix_seal_mismatch",
    ):
        capture.validate_frozen_canonical_base(
            stale_seal,
            canonical_matrix_path=capture.DEFAULT_CANONICAL_MATRIX,
            authority_path=capture.DEFAULT_AUTHORITY,
            supplemental_matrix=supplemental,
        )


def test_both_preflight_gates_bind_the_supplemental_matrix(monkeypatch, tmp_path):
    matrix = {"artifact_id": "sm86_sol_workload_capture_matrix_live_v2"}
    matrix_path = tmp_path / "supplemental.json"
    matrix_path.write_text("{}\n")
    observed = []

    def validate(**kwargs):
        observed.append(kwargs)
        return {"checks": {}}

    monkeypatch.setattr(
        capture.canonical_capture, "_validate_referenced_preflight", validate,
    )
    result = capture._validate_supplemental_preflight_reference(
        preflight_path=tmp_path / "preflight.json",
        preflight_sha256="a" * 64, pins={}, pins_path=tmp_path / "pins.json",
        aggregate_budget_path=tmp_path / "budget.json",
        supplemental_matrix=matrix, supplemental_matrix_path=matrix_path,
        supplemental_matrix_sha256="b" * 64,
        output_root=tmp_path / "output", expected_head="c" * 40,
        require_fresh=False,
        admitted_at_utc=datetime.now(timezone.utc),
    )
    assert result == {"checks": {}}
    assert observed[0]["matrix"] is matrix
    assert observed[0]["matrix_path"] == matrix_path
    assert observed[0]["matrix_sha256"] == "b" * 64
    assert observed[0]["expected_sha256"] == "a" * 64
    assert inspect.getsource(capture._run_chunk_impl).count(
        "_validate_supplemental_preflight_reference("
    ) == 2


def test_chunk_environment_uses_only_the_supplemental_certifier(
    monkeypatch, tmp_path,
):
    probe = {
        name: value for name, value in {
            "ambient": {},
            "environment_contract": {"CUTLASS_DIR": "third_party/cutlass"},
            "environment_contract_sha256": "a" * 64,
            "constructed_environment_delta": {},
            "matrix_pin": {"native_env_hash": "b" * 64},
            "matrix_sha256": "c" * 64,
            "observed_fingerprint": {"hash": "b" * 64},
            "compiler_identity": {"requested": "g++-14"},
            "nvcc_release": {"major": 13, "minor": 2},
            "nvcc_version_output": "release 13.2",
        }.items()
    }
    matrix = {"artifact_id": "sm86_sol_workload_capture_matrix_live_v2"}
    matrix_path = tmp_path / "supplemental.json"
    observed = {}

    def certify(candidate, *, matrix_path, ambient):
        observed.update(candidate=candidate, path=matrix_path, ambient=ambient)
        return probe, {"CERTIFIED": "1"}

    monkeypatch.setattr(
        capture.sol_environment, "certify_supplemental_sol_environment",
        certify,
    )
    monkeypatch.setattr(
        capture.sol_environment, "certify_sol_environment",
        lambda *_args, **_kwargs: pytest.fail("canonical certifier used"),
    )
    result = capture._certify_environment(
        supplemental_matrix=matrix,
        supplemental_matrix_path=matrix_path,
        preflight_manifest={"checks": {"sol_environment": probe}},
    )
    assert result == (probe, {"CERTIFIED": "1"})
    assert observed["candidate"] is matrix
    assert observed["path"] == matrix_path


def _attempt_capture_kwargs(
    tmp_path: Path, binding: dict, monkeypatch,
) -> tuple[dict, Path]:
    journal = tmp_path / "output" / "journals" / "row"
    monkeypatch.setattr(capture, "certified_row_environment", lambda **_: {})
    monkeypatch.setattr(
        capture.canonical_capture,
        "_verify_journal_artifacts", lambda *_, **__: None,
    )
    kwargs = {
        "journal_dir": journal,
        "cell": {
            "identity": {"roster_source": "sol_execbench", "slug": "problem_001_test"},
            "category": "L1",
            "flashinfer_corpus": copy.deepcopy(FLASHINFER_L1_CONTRACT),
        },
        "binding": binding,
        "supplemental_runnability_evidence": None,
        "runnability_path": tmp_path / "runnability-owner.json",
        "canonical_matrix": {"matrix_seal_sha256": "5" * 64},
        "pins": {"measurement_protocol": {"version": "v2.1"}},
        "cell_budgets": object(),
        "preflight_manifest": {
            "checks": {"injection_runtime": {"tracer_so_sha256": "6" * 64}}
        },
        "preflight_path": tmp_path / "preflight.json",
        "matrix_sha256": "8" * 64,
        "cell_budget_sha256": "9" * 64,
        "raw_staging_margin_bytes": 1024,
        "source_commit": "a" * 40,
        "sol_environment_probe": {},
        "certified_sol_env": {},
        "output_root": tmp_path / "output",
        "aggregate_budget": {
            "retention_budget_gib": 4,
            "metadata_margin_gib": 1,
            "raw_staging_margin_gib": 1,
        },
        "flashinfer_proof": {},
        "canonical_matrix_sha256": "b" * 64,
        "expected_sol_fingerprint_pin": {
            "path": "tools/local_labels/authorities/fixture.json",
            "sha256": "d" * 64,
            "native_env_hash": "e" * 64,
        },
        "pins_sha256": "c" * 64,
    }
    kwargs["preflight_path"].write_text("{}\n")
    kwargs["preflight_sha256"] = capture.sha256_file(
        kwargs["preflight_path"]
    )
    return kwargs, journal


@pytest.mark.parametrize(
    "failure_code", ["cell_deadline_exhausted", "trace_tree_limit_exceeded"]
)
def test_failed_attempt_is_preserved_and_next_attempt_can_complete(
    tmp_path, monkeypatch, failure_code,
):
    path, rows, values = _literal_rows(tmp_path)
    _invocation, binding = _binding(path=path, rows=rows, values=values)
    kwargs, journal = _attempt_capture_kwargs(tmp_path, binding, monkeypatch)

    def fail_capture(**call):
        (call["cell_dir"] / "partial.bin").write_bytes(b"partial evidence")
        raise capture.canonical_capture.ChunkRefusal(
            failure_code, "synthetic limit"
        )

    monkeypatch.setattr(capture.canonical_capture, "capture_cell", fail_capture)
    with pytest.raises(capture.WorkloadCaptureRefusal, match=failure_code):
        capture._capture_one(**kwargs)
    first = journal / "attempts" / "attempt-000001"
    failure_path = first / "journal_failure.json"
    first_failure_sha = capture.sha256_file(failure_path)
    assert (first / "cells" / capture._safe_name(binding["row_key"]) / "partial.bin").is_file()

    def succeed_capture(**call):
        payload = call["cell_dir"] / "payload.bin"
        payload.write_bytes(b"complete evidence")
        return {
            "workload_row_binding": copy.deepcopy(binding),
            "artifacts": [{
                "relative_path": (
                    f"cells/{capture._safe_name(binding['row_key'])}/payload.bin"
                ),
                "sha256": capture.sha256_file(payload),
                "size_bytes": payload.stat().st_size,
                "artifact_kind": "synthetic",
            }],
        }

    monkeypatch.setattr(capture.canonical_capture, "capture_cell", succeed_capture)
    result = capture._capture_one(**kwargs)
    assert result["workload_row_binding"] == binding
    second = journal / "attempts" / "attempt-000002"
    assert (second / "journal_complete.json").is_file()
    recovery = json.loads((second / "recovery.json").read_text())
    assert recovery["prior_attempts"][0]["state"] == "failed"
    assert capture.sha256_file(failure_path) == first_failure_sha
    loaded = capture._load_completed_journal(
        journal_dir=journal, expected_binding=binding,
        canonical_matrix_sha256="b" * 64,
        canonical_matrix_seal_sha256="5" * 64,
        preflight_sha256=kwargs["preflight_sha256"], source_commit="a" * 40,
    )
    assert loaded is not None and loaded[1] == result


def test_hard_interrupted_start_is_bound_into_retry_history(tmp_path, monkeypatch):
    path, rows, values = _literal_rows(tmp_path)
    _invocation, binding = _binding(path=path, rows=rows, values=values)
    kwargs, journal = _attempt_capture_kwargs(tmp_path, binding, monkeypatch)
    first = journal / "attempts" / "attempt-000001"
    capture._mkdir_nofollow(first, exist_ok=False)
    (first / "capture_preflight.json").write_bytes(
        kwargs["preflight_path"].read_bytes()
    )
    start = capture._journal_start(
        binding=binding, canonical_matrix_sha256="b" * 64,
        canonical_matrix_seal_sha256="5" * 64,
        preflight_relative_path="capture_preflight.json",
        preflight_sha256=kwargs["preflight_sha256"], source_commit="a" * 40,
        admission_run_id="d" * 64, attempt_ordinal=1,
        attempt_id="e" * 64, prior_attempts=[],
    )
    capture._write_json_new(first / "journal_start.json", start)
    (first / "crash-tail.bin").write_bytes(b"immutable crash bytes")
    crash_sha = capture.sha256_file(first / "crash-tail.bin")

    def succeed_capture(**call):
        payload = call["cell_dir"] / "payload.bin"
        payload.write_bytes(b"retry success")
        return {
            "workload_row_binding": copy.deepcopy(binding),
            "artifacts": [{
                "relative_path": f"cells/{capture._safe_name(binding['row_key'])}/payload.bin",
                "sha256": capture.sha256_file(payload),
                "size_bytes": payload.stat().st_size,
                "artifact_kind": "synthetic",
            }],
        }

    monkeypatch.setattr(capture.canonical_capture, "capture_cell", succeed_capture)
    capture._capture_one(**kwargs)
    recovery = json.loads(
        (journal / "attempts" / "attempt-000002" / "recovery.json").read_text()
    )
    assert recovery["prior_attempts"][0]["state"] == "interrupted"
    assert capture.sha256_file(first / "crash-tail.bin") == crash_sha


def _dynamic_inventory_fixture(tmp_path: Path, count: int = 2):
    launches = []
    mappings = []
    for ordinal in range(count):
        function = f"kernel_{ordinal}"
        launches.append({
            "device_id": 0,
            "stream_id": 0,
            "kernel_id": ordinal + 1,
            "kernel_name": f"{function}___0",
            "grid_dim": [1, 1, 1],
            "block_dim": [32, 1, 1],
        })
        mappings.append({
            "accepted_launch_index": ordinal,
            "device_id": 0,
            "stream_id": 0,
            "trace_launch_id": ordinal + 1,
            "kernel_name": function,
            "resolution_source": "fixture",
            "module": {
                "status": "stable_static_module",
                "reason": "resolved_file",
                "path_at_capture": f"/tmp/{function}.so",
                "basename": f"{function}.so",
                "sha256": f"{ordinal + 1:064x}",
            },
        })
    root = {"launch_count": count, "launches": launches}
    trace_dir = tmp_path / "trace"
    extra = trace_dir / "extra_info"
    extra.mkdir(parents=True)
    (extra / "traced_kernel_modules.json").write_text(json.dumps({
        "schema_version": 1,
        "module_identity": {
            "stable_key": "sha256",
            "path_semantics": "path_at_capture",
            "rebind_contract": "sha256_required_basename_hint",
        },
        "selection": {
            "active_from_start": False,
            "max_accepted_active_launches": 0,
            "accepted_active_launches": count,
            "active_launch_attempts": count,
            "active_launches_skipped_after_limit": 0,
            "concurrent_limit_violations": 0,
            "exact_contract_satisfied": True,
            "module_mapping_count": count,
            "inactive_channel_records_dropped": 0,
            "accepted_channel_records": count,
        },
        "launches": mappings,
    }))
    return root, trace_dir


def _bind_synthetic_ncu_identity(label, row, inventory):
    report = label["ncu_report"]
    launch = row["trace_launch"]
    observation = {
        "schema_version": 1,
        "kind": "ncu_details_csv_launch_observation",
        "kernel_name_base": "mangled",
        "requested_launch_skip": label["launch_index"],
        "requested_launch_count": 1,
        "report_sha256": capture.sha256_file(report),
        "identity_columns": [
            "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
        ],
        "identity": {
            "ID": "0",
            "Device": str(launch["device_id"]),
            "Kernel Name": row["module_mapping"]["kernel_name"],
            "Block Size": f"({launch['block_dim'][0]}, {launch['block_dim'][1]}, {launch['block_dim'][2]})",
            "Grid Size": f"({launch['grid_dim'][0]}, {launch['grid_dim'][1]}, {launch['grid_dim'][2]})",
        },
    }
    observation["observation_sha256"] = capture.canonical_sha256(
        observation
    )
    label["ncu_launch_observation_document"] = observation
    canonical_capture.bind_unique_ncu_launch_identity(
        label=label, inventory_row=row, inventory=inventory
    )
    return label


def test_dynamic_inventory_requires_exact_all_roi_zero_skip_join(tmp_path):
    root, trace_dir = _dynamic_inventory_fixture(tmp_path)
    rows, seal = canonical_capture._validate_supplemental_trace_launch_inventory(
        root, trace_dir
    )
    assert [row["launch_ordinal"] for row in rows] == [0, 1]
    assert seal == capture.canonical_sha256(rows)
    inventory_path = trace_dir / "extra_info" / "traced_kernel_modules.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["selection"]["active_launches_skipped_after_limit"] = 1
    inventory_path.write_text(json.dumps(inventory))
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_dynamic_launch_selection_incomplete",
    ):
        canonical_capture._validate_supplemental_trace_launch_inventory(
            root, trace_dir
        )


def _flash_contract(category: str) -> dict:
    contract = copy.deepcopy(FLASHINFER_L1_CONTRACT)
    contract["category"] = category
    if category == "FIB":
        contract["injection_policy"] = "verified_absolute_root"
        contract["root"] = "data/annex/flashinfer_trace"
    return contract


def test_flashinfer_environment_is_verified_fib_only_and_strips_ambient():
    proof = {
        "root": str(capture.sol_recon.FLASHINFER_CORPUS_ROOT.resolve()),
    }
    base = {"PATH": "/bin", "FLASHINFER_TRACE_DIR": "/ambient/untrusted"}
    l1 = capture.certified_row_environment(
        cell={"category": "L1", "flashinfer_corpus": _flash_contract("L1")},
        certified_sol_env=base, flashinfer_proof=proof,
    )
    fib = capture.certified_row_environment(
        cell={"category": "FIB", "flashinfer_corpus": _flash_contract("FIB")},
        certified_sol_env=base, flashinfer_proof=proof,
    )
    assert "FLASHINFER_TRACE_DIR" not in l1
    assert fib["FLASHINFER_TRACE_DIR"] == str(
        capture.sol_recon.FLASHINFER_CORPUS_ROOT.resolve()
    )


def _verified_flashinfer_bindings(monkeypatch):
    proof = {
        "root": str(capture.sol_recon.FLASHINFER_CORPUS_ROOT.resolve()),
        "data_annex_manifest_sha256": capture.sol_recon.DATA_ANNEX_MANIFEST_SHA256,
        "filtered_manifest_sha256": capture.sol_recon.FLASHINFER_CORPUS_MANIFEST_SHA256,
        "file_count": capture.sol_recon.FLASHINFER_CORPUS_FILE_COUNT,
        "safetensors_count": capture.sol_recon.FLASHINFER_CORPUS_BLOB_COUNT,
    }
    monkeypatch.setattr(
        capture.sol_recon, "verify_flashinfer_corpus", lambda: copy.deepcopy(proof)
    )
    contract = _flash_contract("FIB")
    row = {"key": {"slug": "x"}, "category": "FIB",
           "flashinfer_corpus": copy.deepcopy(contract)}
    variant = {
        "category": "FIB",
        "adapter_contract": {
            "flashinfer_corpus": copy.deepcopy(contract),
        },
    }
    cell = {"category": "FIB", "flashinfer_corpus": copy.deepcopy(contract)}
    return proof, contract, row, variant, cell


def test_flashinfer_contract_reverification_accepts_nested_variant_contract(
    monkeypatch,
):
    proof, _, row, variant, cell = _verified_flashinfer_bindings(monkeypatch)
    assert capture.verify_flashinfer_contracts([(row, variant, cell)]) == proof


def test_flashinfer_contract_reverification_rejects_top_level_decoy(
    monkeypatch,
):
    _, contract, row, variant, cell = _verified_flashinfer_bindings(monkeypatch)
    variant["flashinfer_corpus"] = copy.deepcopy(contract)
    variant["adapter_contract"].pop("flashinfer_corpus")
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_flashinfer_contract_invalid",
    ):
        capture.verify_flashinfer_contracts([(row, variant, cell)])


@pytest.mark.parametrize("binding_name", ["row", "cell"])
def test_flashinfer_contract_reverification_refuses_binding_tamper_before_capture(
    monkeypatch, binding_name,
):
    _, _, row, variant, cell = _verified_flashinfer_bindings(monkeypatch)
    bindings = {"row": row, "cell": cell}
    bindings[binding_name]["flashinfer_corpus"]["file_count"] -= 1
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_flashinfer_contract_invalid",
    ):
        capture.verify_flashinfer_contracts([(row, variant, cell)])


def test_dynamic_completeness_requires_every_trace_label_control_join(tmp_path):
    root, trace_dir = _dynamic_inventory_fixture(tmp_path)
    inventory, seal = (
        canonical_capture._validate_supplemental_trace_launch_inventory(
            root, trace_dir
        )
    )
    trace = {
        "dynamic_launch_inventory": inventory,
        "dynamic_launch_inventory_sha256": seal,
        "outcome": "admitted",
    }
    labels = []
    controls = []
    for row in inventory:
        ordinal = row["launch_ordinal"]
        report = tmp_path / f"launch_{ordinal}.ncu-rep"
        report.write_bytes(
            f"report-{ordinal}-{row['module_mapping']['kernel_name']}".encode()
        )
        label = {"launch_index": ordinal, "ncu_report": report}
        labels.append(_bind_synthetic_ncu_identity(label, row, inventory))
        controls.append({
            "launch_ordinal": ordinal,
            "launch_identity_sha256": row["launch_identity_sha256"],
            "status": "complete",
            "typed_reason": None,
            "control_outcome": canonical_capture.SOL_CONTROL_OUTCOMES[0],
        })
    complete = canonical_capture.build_dynamic_launch_completeness(
        trace=trace, labels=labels, label_failures=[],
        control={"launch_records": controls},
    )
    assert complete["status"] == "complete_dynamic_launch_set"
    assert complete["inventory_launch_count"] == 2
    assert complete["ncu_label_complete_launch_count"] == 2
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_dynamic_launch_label_failure_not_terminal",
    ):
        canonical_capture.build_dynamic_launch_completeness(
            trace=trace, labels=labels[:1],
            label_failures=[{
                "launch_ordinal": 1,
                "typed_reason": "label_capture_failed",
            }],
            control={"launch_records": [
                controls[0],
                {**controls[1], "status": "typed_incomplete",
                 "typed_reason": "ncu_label_incomplete",
                 "control_outcome": None},
            ]},
        )
    complete_result = {
        "disposition": "admitted",
        "typed_reason": None,
        "launch_completeness": complete,
    }
    capture.validate_result_launch_completeness(complete_result)
    bad = copy.deepcopy(complete_result)
    bad["launch_completeness"]["control_complete_launch_count"] = 1
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_done_launch_completeness_invalid",
    ):
        capture.validate_result_launch_completeness(bad)


def test_same_name_launch_reports_cannot_be_swapped_or_claim_sequence(tmp_path):
    root, trace_dir = _dynamic_inventory_fixture(tmp_path)
    inventory, _seal = (
        canonical_capture._validate_supplemental_trace_launch_inventory(
            root, trace_dir
        )
    )
    repeated = copy.deepcopy(inventory)
    shared_name = repeated[0]["module_mapping"]["kernel_name"]
    for row in repeated:
        row["module_mapping"]["kernel_name"] = shared_name
        row.pop("launch_identity_sha256")
        row["launch_identity_sha256"] = capture.canonical_sha256(row)
    reports = []
    for ordinal in range(2):
        report = tmp_path / f"same-name-{ordinal}.ncu-rep"
        report.write_bytes(shared_name.encode())
        reports.append(report)
    # The two reports are intentionally swapped.  The imported NCU identity
    # surface is indistinguishable, so neither global ordinal is provable.
    label = {"launch_index": 0, "ncu_report": reports[1]}
    observation = {
        "schema_version": 1,
        "kind": "ncu_details_csv_launch_observation",
        "kernel_name_base": "mangled",
        "requested_launch_skip": 0,
        "requested_launch_count": 1,
        "report_sha256": capture.sha256_file(reports[1]),
        "identity_columns": [
            "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
        ],
        "identity": {
            "ID": "0", "Device": "0", "Kernel Name": shared_name,
            "Block Size": "(32, 1, 1)", "Grid Size": "(1, 1, 1)",
        },
    }
    observation["observation_sha256"] = capture.canonical_sha256(observation)
    label["ncu_launch_observation_document"] = observation
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_dynamic_launch_ncu_identity_ambiguous",
    ):
        canonical_capture.bind_unique_ncu_launch_identity(
            label=label, inventory_row=repeated[0], inventory=repeated
        )

    failures = []
    controls = []
    for row in repeated:
        failures.append({
            "launch_ordinal": row["launch_ordinal"],
            "typed_reason": "ncu_launch_identity_ambiguous_repeated_tuple",
        })
        controls.append({
            "launch_ordinal": row["launch_ordinal"],
            "launch_identity_sha256": row["launch_identity_sha256"],
            "status": "typed_incomplete",
            "typed_reason": "ncu_label_incomplete",
            "control_outcome": None,
        })
    completeness = canonical_capture.build_dynamic_launch_completeness(
        trace={
            "dynamic_launch_inventory": repeated,
            "dynamic_launch_inventory_sha256": capture.canonical_sha256(
                repeated
            ),
            "outcome": "admitted",
        },
        labels=[], label_failures=failures,
        control={"launch_records": controls},
    )
    assert completeness["status"] == "typed_incomplete_dynamic_launch_set"
    assert completeness["trace_ncu_sequence_equal"] is False


def test_same_name_different_geometry_joins_and_swapped_geometry_refuses(
    tmp_path,
):
    root, trace_dir = _dynamic_inventory_fixture(tmp_path)
    inventory, _seal = (
        canonical_capture._validate_supplemental_trace_launch_inventory(
            root, trace_dir
        )
    )
    shared_name = inventory[0]["module_mapping"]["kernel_name"]
    for row in inventory:
        row["module_mapping"]["kernel_name"] = shared_name
    inventory[1]["trace_launch"]["grid_dim"] = [2, 1, 1]
    for row in inventory:
        row.pop("launch_identity_sha256")
        row["launch_identity_sha256"] = capture.canonical_sha256(row)

    labels = []
    for row in inventory:
        report = tmp_path / f"same-name-geometry-{row['launch_ordinal']}.ncu-rep"
        report.write_bytes(shared_name.encode())
        label = {"launch_index": row["launch_ordinal"], "ncu_report": report}
        labels.append(_bind_synthetic_ncu_identity(label, row, inventory))
    assert [
        row["ncu_launch_identity"]["grid_dim"] for row in labels
    ] == [[1, 1, 1], [2, 1, 1]]

    swapped = {
        "launch_index": 0,
        "ncu_report": labels[0]["ncu_report"],
        "ncu_launch_observation_document": copy.deepcopy(
            labels[1]["ncu_launch_observation_document"]
        ),
    }
    swapped["ncu_launch_observation_document"]["report_sha256"] = (
        capture.sha256_file(swapped["ncu_report"])
    )
    unsealed = {
        key: value
        for key, value in swapped["ncu_launch_observation_document"].items()
        if key != "observation_sha256"
    }
    swapped["ncu_launch_observation_document"]["observation_sha256"] = (
        capture.canonical_sha256(unsealed)
    )
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_dynamic_launch_ncu_identity_mismatch",
    ):
        canonical_capture.bind_unique_ncu_launch_identity(
            label=swapped, inventory_row=inventory[0], inventory=inventory
        )


def test_unique_name_launch_reports_swapped_are_rejected(tmp_path):
    root, trace_dir = _dynamic_inventory_fixture(tmp_path)
    inventory, _seal = (
        canonical_capture._validate_supplemental_trace_launch_inventory(
            root, trace_dir
        )
    )
    reports = []
    for row in inventory:
        report = tmp_path / f"unique-{row['launch_ordinal']}.ncu-rep"
        report.write_bytes(row["module_mapping"]["kernel_name"].encode())
        reports.append(report)
    swapped = {"launch_index": 0, "ncu_report": reports[1]}
    observation = {
        "schema_version": 1,
        "kind": "ncu_details_csv_launch_observation",
        "kernel_name_base": "mangled",
        "requested_launch_skip": 0,
        "requested_launch_count": 1,
        "report_sha256": capture.sha256_file(reports[1]),
        "identity_columns": [
            "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
        ],
        "identity": {
            "ID": "0",
            "Device": "0",
            "Kernel Name": inventory[1]["module_mapping"]["kernel_name"],
            "Block Size": "(32, 1, 1)",
            "Grid Size": "(1, 1, 1)",
        },
    }
    observation["observation_sha256"] = capture.canonical_sha256(observation)
    swapped["ncu_launch_observation_document"] = observation
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_dynamic_launch_ncu_identity_mismatch",
    ):
        canonical_capture.bind_unique_ncu_launch_identity(
            label=swapped, inventory_row=inventory[0], inventory=inventory
        )


def test_per_ordinal_control_views_cover_all_launches_and_are_not_retained(
    tmp_path, monkeypatch,
):
    root, trace_dir = _dynamic_inventory_fixture(tmp_path)
    inventory, seal = (
        canonical_capture._validate_supplemental_trace_launch_inventory(
            root, trace_dir
        )
    )
    kernels = []
    labels = {}
    for row in inventory:
        launch = row["trace_launch"]
        payload_dir = (
            trace_dir / "threadblocks" / "device_0" / "stream_0"
            / f"kernel_{launch['kernel_id']}"
        )
        payload_dir.mkdir(parents=True)
        (payload_dir / "block_0.pb").write_bytes(b"fixture")
        kernels.append({"kernel_name": launch["kernel_name"]})
        report = tmp_path / f"ordinal_{row['launch_ordinal']}.ncu-rep"
        report.write_bytes(row["module_mapping"]["kernel_name"].encode())
        label = {
            "launch_index": row["launch_ordinal"],
            "ncu_report": report,
        }
        labels[row["launch_ordinal"]] = _bind_synthetic_ncu_identity(
            label, row, inventory
        )
    (trace_dir / "extra_info" / "enhanced_execution_info.json").write_text(
        json.dumps({"kernels": kernels})
    )

    def fake_control(*, cell_dir, **_kwargs):
        out = cell_dir / "control_bits.json"
        manifest = cell_dir / "control_capture_manifest.json"
        out.write_text("{}\n")
        manifest.write_text("{}\n")
        return {
            "control_bits": out,
            "manifest": manifest,
            "outcome": canonical_capture.SOL_CONTROL_OUTCOMES[0],
            "typed_reason": None,
            "total_decoded": 1,
        }

    monkeypatch.setattr(canonical_capture, "capture_sol_control", fake_control)
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    trace_log = tmp_path / "trace.log"
    trace_log.write_text("trace")
    result = canonical_capture.capture_sol_controls_all_launches(
        cell={"identity": {"roster_source": "sol_execbench", "slug": "x"}},
        invocation={"authority_reference_sha256": "a" * 64},
        labels_by_ordinal=labels,
        trace={
            "dynamic_launch_inventory": inventory,
            "dynamic_launch_inventory_sha256": seal,
            "control_trace_dir": trace_dir,
            "stdout_log": trace_log,
            "stderr_log": trace_log,
        },
        cell_dir=cell_dir,
        module_store=tmp_path / "module_store",
    )
    assert result["outcome"] == "complete_dynamic_launch_set"
    assert result["complete_control_count"] == 2
    assert not list(cell_dir.rglob("control_trace_view"))


def test_per_ordinal_control_failure_escapes_for_retry_without_downgrade(
    tmp_path, monkeypatch,
):
    root, trace_dir = _dynamic_inventory_fixture(tmp_path)
    inventory, seal = (
        canonical_capture._validate_supplemental_trace_launch_inventory(
            root, trace_dir
        )
    )
    kernels = []
    labels = {}
    for row in inventory:
        launch = row["trace_launch"]
        payload_dir = (
            trace_dir / "threadblocks" / "device_0" / "stream_0"
            / f"kernel_{launch['kernel_id']}"
        )
        payload_dir.mkdir(parents=True)
        (payload_dir / "block_0.pb").write_bytes(b"fixture")
        kernels.append({"kernel_name": launch["kernel_name"]})
        report = tmp_path / f"ordinal_{row['launch_ordinal']}.ncu-rep"
        report.write_bytes(row["module_mapping"]["kernel_name"].encode())
        label = {
            "launch_index": row["launch_ordinal"],
            "ncu_report": report,
        }
        labels[row["launch_ordinal"]] = _bind_synthetic_ncu_identity(
            label, row, inventory
        )
    (trace_dir / "extra_info" / "enhanced_execution_info.json").write_text(
        json.dumps({"kernels": kernels})
    )

    called: list[int] = []

    def fake_control(*, cell_dir, **_kwargs):
        ordinal = int(cell_dir.name.removeprefix("launch_"))
        called.append(ordinal)
        if ordinal == 0:
            raise canonical_capture.ChunkRefusal(
                "sol_control_runtime_decode_failed", "fixture failure"
            )
        out = cell_dir / "control_bits.json"
        manifest = cell_dir / "control_capture_manifest.json"
        out.write_text("{}\n")
        manifest.write_text("{}\n")
        return {
            "control_bits": out,
            "manifest": manifest,
            "outcome": canonical_capture.SOL_CONTROL_OUTCOMES[0],
            "typed_reason": None,
            "total_decoded": 1,
        }

    monkeypatch.setattr(canonical_capture, "capture_sol_control", fake_control)
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    trace_log = tmp_path / "trace.log"
    trace_log.write_text("trace")
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_control_runtime_decode_failed",
    ):
        canonical_capture.capture_sol_controls_all_launches(
            cell={"identity": {
                "roster_source": "sol_execbench", "slug": "x"
            }},
            invocation={"authority_reference_sha256": "a" * 64},
            labels_by_ordinal=labels,
            trace={
                "dynamic_launch_inventory": inventory,
                "dynamic_launch_inventory_sha256": seal,
                "control_trace_dir": trace_dir,
                "stdout_log": trace_log,
                "stderr_log": trace_log,
            },
            cell_dir=cell_dir,
            module_store=tmp_path / "module_store",
        )
    assert called == [0]
    assert not list(cell_dir.rglob("control_trace_view"))


@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EIO, errno.EROFS])
def test_per_ordinal_control_storage_failure_preserves_control_view(
    tmp_path, monkeypatch, error_number,
):
    function = "synthetic_kernel"
    report = tmp_path / "launch.ncu-rep"
    report.write_bytes(function.encode("ascii"))
    row = {
        "launch_ordinal": 0,
        "launch_identity_sha256": "a" * 64,
        "module_mapping": {"kernel_name": function},
    }
    inventory = [row]
    labels = {
        0: {
            "launch_index": 0,
            "ncu_report": report,
            "ncu_launch_identity": {
                "launch_ordinal": 0,
                "launch_identity_sha256": "a" * 64,
                "kernel_name": function,
                "ncu_report_sha256": canonical_capture.sha256_file(report),
            },
        }
    }
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()

    def materialize(*, launch_cell_dir, **_kwargs):
        view = launch_cell_dir / "control_trace_view"
        view.mkdir()
        (view / "partial.pb").write_bytes(b"preserved")
        return {"control_trace_dir": view}

    def fail_control(**_kwargs):
        cause = OSError(error_number, "injected control storage failure")
        refusal = canonical_capture.ChunkRefusal(
            "sol_control_runtime_decode_storage_failure", "storage-backed fixture",
        )
        raise refusal from cause

    monkeypatch.setattr(
        canonical_capture, "_materialize_sol_launch_control_view", materialize,
    )
    monkeypatch.setattr(canonical_capture, "capture_sol_control", fail_control)
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_control_storage_failure",
    ) as caught:
        canonical_capture.capture_sol_controls_all_launches(
            cell={"identity": {
                "roster_source": "sol_execbench", "slug": "fixture",
            }},
            invocation={"authority_reference_sha256": "b" * 64},
            labels_by_ordinal=labels,
            trace={
                "dynamic_launch_inventory": inventory,
                "dynamic_launch_inventory_sha256": (
                    canonical_capture.canonical_sha256(inventory)
                ),
            },
            cell_dir=cell_dir,
            module_store=tmp_path / "module-store",
        )
    assert canonical_capture._storage_origin(caught.value).errno == error_number
    mapped = capture.storage_refusal_from_exception(
        caught.value, operation="capture_monitored_output_tree",
    )
    assert mapped is not None
    assert mapped.code == (
        capture.STORAGE_CAPACITY_CODE
        if error_number in {errno.ENOSPC, errno.EDQUOT}
        else capture.STORAGE_INTEGRITY_CODE
    )
    preserved = cell_dir / "control_launches/launch_0000/control_trace_view"
    assert (preserved / "partial.pb").read_bytes() == b"preserved"


def test_per_ordinal_raw_module_input_eio_is_not_output_storage(
    tmp_path, monkeypatch,
):
    function = "synthetic_kernel"
    report = tmp_path / "launch.ncu-rep"
    report.write_bytes(function.encode("ascii"))
    row = {
        "launch_ordinal": 0,
        "launch_identity_sha256": "a" * 64,
        "module_mapping": {"kernel_name": function},
    }
    inventory = [row]
    labels = {0: {
        "launch_index": 0,
        "ncu_report": report,
        "ncu_launch_identity": {
            "launch_ordinal": 0,
            "launch_identity_sha256": "a" * 64,
            "kernel_name": function,
            "ncu_report_sha256": canonical_capture.sha256_file(report),
        },
    }}
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()

    def materialize(*, launch_cell_dir, **_kwargs):
        view = launch_cell_dir / "control_trace_view"
        view.mkdir()
        (view / "partial.pb").write_bytes(b"preserved")
        return {"control_trace_dir": view}

    def fail_control(**_kwargs):
        refusal = canonical_capture.ChunkRefusal(
            "sol_control_module_absent", "raw module input read failed",
        )
        raise refusal from OSError(errno.EIO, "injected raw module input EIO")

    monkeypatch.setattr(
        canonical_capture, "_materialize_sol_launch_control_view", materialize,
    )
    monkeypatch.setattr(canonical_capture, "capture_sol_control", fail_control)
    with pytest.raises(
        canonical_capture.ChunkRefusal, match="sol_control_module_absent",
    ) as caught:
        canonical_capture.capture_sol_controls_all_launches(
            cell={"identity": {
                "roster_source": "sol_execbench", "slug": "fixture",
            }},
            invocation={"authority_reference_sha256": "b" * 64},
            labels_by_ordinal=labels,
            trace={
                "dynamic_launch_inventory": inventory,
                "dynamic_launch_inventory_sha256": (
                    canonical_capture.canonical_sha256(inventory)
                ),
            },
            cell_dir=cell_dir, module_store=tmp_path / "module-store",
        )
    assert not canonical_capture._is_explicit_output_storage_refusal(caught.value)
    assert (cell_dir / "control_launches/launch_0000/control_trace_view").is_dir()


def _standalone_contract(*, cohort: str = "native"):
    key = {
        "slug": "problem_001_test",
        "source_workload_uuid": "12345678-1234-5123-8123-123456789abc",
        "variant_id": "native" if cohort == "native" else "a" * 64,
    }
    evidence_sha = "4" * 64
    evidence = {"source_kind": "current_full_correctness"}
    invocation = {"logical_workload": {"evaluation_evidence": evidence}}
    variant = {
        "slug": key["slug"],
        "variant_id": key["variant_id"],
        "cohort": cohort,
        "category": "L1",
        "adapter_contract": {
            "flashinfer_corpus": copy.deepcopy(FLASHINFER_L1_CONTRACT),
        },
        "launch_scope": "all_in_profiler_roi",
        "launch_completeness_contract": copy.deepcopy(LAUNCH_CONTRACT),
        "claim_scope": (
            "native_workload_row" if cohort == "native"
            else "partial_native_workload_row_only"
        ),
        "native_shape": True,
        "native_predictive": cohort == "native",
        "problem_level_runnable": cohort == "native",
        "invocation": invocation,
        "trace_budget": {
            "max_trace_bytes": 16 * 1024 ** 3,
            "max_wall_time_s": 7200,
            "sampling_policy": "off",
        },
        "control_bits": {"policy": "test"},
        "runnability_evidence": evidence,
    }
    if cohort == "partial_native_row":
        evidence = {
            "source_kind": "current_partial_native_row",
            "selected_diagnostic_sha256": evidence_sha,
        }
        invocation["logical_workload"]["evaluation_evidence"] = evidence
        variant["runnability_evidence"] = evidence
        variant["selected_diagnostic_sha256"] = evidence_sha
    contract = capture._variant_cell_contract(variant)
    variant["supplemental_cell_contract_sha256"] = capture.canonical_sha256(
        contract
    )
    variant["variant_contract_sha256"] = capture.canonical_sha256(variant)
    row = {
        "key": key,
        "cohort": cohort,
        "category": "L1",
        "flashinfer_corpus": copy.deepcopy(FLASHINFER_L1_CONTRACT),
        "launch_scope": "all_in_profiler_roi",
        "launch_completeness_contract": copy.deepcopy(LAUNCH_CONTRACT),
        "claim_scope": variant["claim_scope"],
        "native_shape": True,
        "native_predictive": variant["native_predictive"],
        "problem_level_runnable": variant["problem_level_runnable"],
        "variant_contract_sha256": variant["variant_contract_sha256"],
        "row_evidence_sha256": evidence_sha,
        "source_workload_index": 0,
        "source_workload_uuid": key["source_workload_uuid"],
        "source_axes": {"batch": 1},
        "source_row_sha256": "5" * 64,
        "capture_workload_index": 0,
        "capture_axes": {"batch": 1},
        "capture_row_sha256": "5" * 64,
    }
    member = copy.deepcopy(key)
    chunk = {
        "chunk_id": "sol-workload-test-001",
        "cohort": cohort,
        "row_count": 1,
        "membership_sha256": capture.canonical_sha256([member]),
        "rows": [member],
    }
    matrix = {"chunks": [chunk], "rows": [row], "variants": [variant]}
    return matrix, row, variant


def test_selection_is_standalone_from_canonical_runnable_membership():
    matrix, _row, variant = _standalone_contract()
    chunk, resolved = capture.select_workload_chunk(
        matrix, {"cells": []}, "sol-workload-test-001"
    )
    assert chunk["row_count"] == 1
    assert resolved[0][2]["authority_cell_sha256"] == variant[
        "supplemental_cell_contract_sha256"
    ]
    assert resolved[0][2]["disposition"] == "runnable"


def test_selection_follows_chunk_membership_not_matrix_row_storage_order():
    matrix, row_one, variant_one = _standalone_contract()
    row_two = copy.deepcopy(row_one)
    variant_two = copy.deepcopy(variant_one)
    row_two["key"]["slug"] = "problem_002_test"
    row_two["key"]["source_workload_uuid"] = (
        "22345678-1234-5123-8123-123456789abc"
    )
    row_two["source_workload_uuid"] = row_two["key"][
        "source_workload_uuid"
    ]
    variant_two["slug"] = row_two["key"]["slug"]
    variant_two.pop("variant_contract_sha256")
    variant_two["supplemental_cell_contract_sha256"] = (
        capture.canonical_sha256(capture._variant_cell_contract(variant_two))
    )
    variant_two["variant_contract_sha256"] = capture.canonical_sha256(
        variant_two
    )
    row_two["variant_contract_sha256"] = variant_two[
        "variant_contract_sha256"
    ]
    membership = [copy.deepcopy(row_one["key"]), copy.deepcopy(row_two["key"])]
    matrix["rows"] = [row_two, row_one]
    matrix["variants"] = [variant_two, variant_one]
    matrix["chunks"][0].update({
        "row_count": 2,
        "rows": membership,
        "membership_sha256": capture.canonical_sha256(membership),
    })
    _chunk, resolved = capture.select_workload_chunk(
        matrix, {"cells": []}, "sol-workload-test-001"
    )
    assert [item[0]["key"] for item in resolved] == membership


def test_selection_refuses_variant_or_partial_evidence_drift():
    matrix, _row, _variant = _standalone_contract()
    bad = copy.deepcopy(matrix)
    bad["variants"][0]["invocation"]["unsealed"] = True
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_variant_seal_mismatch",
    ):
        capture.select_workload_chunk(
            bad, {"cells": []}, "sol-workload-test-001"
        )
    partial, _row, _variant = _standalone_contract(
        cohort="partial_native_row"
    )
    partial["rows"][0]["row_evidence_sha256"] = "6" * 64
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_partial_evidence_mismatch",
    ):
        capture.select_workload_chunk(
            partial, {"cells": []}, "sol-workload-test-001"
        )


def test_explicit_budget_is_derived_without_default_fallback():
    policy, source = capture.workload_matrix._load_budget_policy(REPO_ROOT)
    slug = sorted(source)[0]
    budget = capture.workload_matrix._trace_budget(
        slug, policy=policy, explicit=source
    )
    resolved = [(
        {"key": {"slug": slug}},
        {"trace_budget": budget},
        {"identity": {"slug": slug}},
    )]
    config = capture.explicit_chunk_budgets(resolved)
    assert config.defaults is None
    assert config.cells_by_id()[slug].max_trace_bytes == budget[
        "max_trace_bytes"
    ]
    bad = copy.deepcopy(resolved)
    bad[0][1]["trace_budget"]["max_trace_bytes"] += 1
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_trace_budget_derivation_mismatch",
    ):
        capture.explicit_chunk_budgets(bad)


def test_partial_native_evidence_binds_exact_selected_diagnostic(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(canonical_capture, "REPO_ROOT", tmp_path)
    workload = tmp_path / "workload.jsonl"
    rows = [
        {
            "uuid": "12345678-1234-5123-8123-123456789abc",
            "axes": {"n": 1},
            "inputs": {"value": {"type": "random"}},
        },
        {
            "uuid": "22345678-1234-5123-8123-123456789abc",
            "axes": {"n": 2},
            "inputs": {"value": {"type": "random"}},
            "tolerance": {"max_atol": 0.0001},
        },
    ]
    raw = [json.dumps(row, separators=(",", ":")).encode() for row in rows]
    workload.write_bytes(b"\n".join(raw) + b"\n")
    diagnostic = {
        "workload_index": 1,
        "trace_index": 1,
        "uuid": rows[1]["uuid"],
        "axes": rows[1]["axes"],
        "workload_sha256": hashlib.sha256(raw[1]).hexdigest(),
        "diagnostic_present": True,
        "status": "PASSED",
        "trace": {
            "workload": canonical_capture._normalize_sol_diagnostic_workload(
                rows[1]
            ),
            "definition": "test_definition",
            "solution": "test_solution",
            "status": "PASSED",
        },
    }
    failed_diagnostic = {
        "workload_index": 0,
        "trace_index": None,
        "uuid": rows[0]["uuid"],
        "axes": rows[0]["axes"],
        "workload_sha256": hashlib.sha256(raw[0]).hexdigest(),
        "diagnostic_present": False,
        "status": None,
        "trace": None,
    }
    record = {
        "slug": "problem_001_test",
        "status": "typed_excluded",
        "returncode": 1,
        "diagnostics_complete": False,
        "expected_workloads": 2,
        "returned_workloads": 1,
        "passed_workloads": 1,
        "workload_diagnostics": [failed_diagnostic, diagnostic],
        "unexpected_workload_diagnostics": [],
    }
    pins = {
        "resume_identity_sha256": "a" * 64,
        "probe_protocol_sha256": "b" * 64,
        "probe_config_sha256": "c" * 64,
        "sol_execbench_package_sha256": "d" * 64,
        "native_env_hash": "e" * 64,
    }
    evidence_doc = {**pins, "cells": [record]}
    evidence_path = tmp_path / "runnability.json"
    evidence_path.write_text(json.dumps(evidence_doc))
    diagnostic_sha = capture.canonical_sha256(diagnostic)
    evidence = {
        "source_kind": "current_partial_native_row",
        "source_path": "runnability.json",
        "source_sha256": capture.sha256_file(evidence_path),
        "cell_sha256": capture.canonical_sha256(record),
        **pins,
        "problem_status": record["status"],
        "problem_returncode": record["returncode"],
        "diagnostics_complete": False,
        "expected_workloads": 2,
        "passed_workloads": 1,
        "selected_source_workload_index": 1,
        "selected_source_workload_uuid": rows[1]["uuid"],
        "selected_source_axes": rows[1]["axes"],
        "selected_source_row_sha256": hashlib.sha256(raw[1]).hexdigest(),
        "selected_diagnostic_sha256": diagnostic_sha,
    }
    binding = {
        "cohort": "partial_native_row",
        "problem_level_runnable": False,
        "source_workload_index": 1,
        "source_workload_uuid": rows[1]["uuid"],
        "source_axes": rows[1]["axes"],
        "source_row_sha256": hashlib.sha256(raw[1]).hexdigest(),
        "row_evidence_sha256": diagnostic_sha,
    }
    canonical_capture._validate_current_sol_runnability_evidence(
        evidence=evidence,
        expected=copy.deepcopy(evidence),
        binding=binding,
        identity={"slug": record["slug"]},
        observed_workloads=2,
        workload_path=workload,
        definition_name="test_definition",
        solution_name="test_solution",
    )
    wrong_parent = copy.deepcopy(evidence)
    wrong_parent["problem_returncode"] = 0
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_supplemental_evidence_incomplete",
    ):
        canonical_capture._validate_current_sol_runnability_evidence(
            evidence=wrong_parent,
            expected=copy.deepcopy(wrong_parent),
            binding=binding,
            identity={"slug": record["slug"]},
            observed_workloads=2,
            workload_path=workload,
            definition_name="test_definition",
            solution_name="test_solution",
        )
    bad = copy.deepcopy(binding)
    bad["row_evidence_sha256"] = "f" * 64
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_partial_native_selection_mismatch",
    ):
        canonical_capture._validate_current_sol_runnability_evidence(
            evidence=evidence,
            expected=copy.deepcopy(evidence),
            binding=bad,
            identity={"slug": record["slug"]},
            observed_workloads=2,
            workload_path=workload,
            definition_name="test_definition",
            solution_name="test_solution",
        )

    tampered_doc = copy.deepcopy(evidence_doc)
    tampered_record = tampered_doc["cells"][0]
    tampered_diagnostic = tampered_record["workload_diagnostics"][1]
    tampered_diagnostic["trace"]["workload"]["tolerance"]["max_atol"] = 9.0
    tampered_sha = capture.canonical_sha256(tampered_diagnostic)
    evidence_path.write_text(json.dumps(tampered_doc))
    tampered_evidence = copy.deepcopy(evidence)
    tampered_evidence.update({
        "source_sha256": capture.sha256_file(evidence_path),
        "cell_sha256": capture.canonical_sha256(tampered_record),
        "selected_diagnostic_sha256": tampered_sha,
    })
    tampered_binding = copy.deepcopy(binding)
    tampered_binding["row_evidence_sha256"] = tampered_sha
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_native_diagnostic_bijection_invalid",
    ):
        canonical_capture._validate_current_sol_runnability_evidence(
            evidence=tampered_evidence,
            expected=copy.deepcopy(tampered_evidence),
            binding=tampered_binding,
            identity={"slug": record["slug"]},
            observed_workloads=2,
            workload_path=workload,
            definition_name="test_definition",
            solution_name="test_solution",
        )
    evidence_path.write_text(json.dumps(evidence_doc))

    # A retained external artifact is addressed by the exact owner-validated
    # path, not by rebasing its basename beneath REPO_ROOT.
    evidence_path.chmod(0o444)
    empty_repo = tmp_path / "empty-repo"
    empty_repo.mkdir()
    monkeypatch.setattr(canonical_capture, "REPO_ROOT", empty_repo)
    assert not (empty_repo / evidence_path.name).exists()
    canonical_capture._validate_current_sol_runnability_evidence(
        evidence=evidence,
        expected=copy.deepcopy(evidence),
        binding=binding,
        identity={"slug": record["slug"]},
        observed_workloads=2,
        workload_path=workload,
        definition_name="test_definition",
        solution_name="test_solution",
        runnability_path=evidence_path,
    )
    wrong_basename = copy.deepcopy(evidence)
    wrong_basename["source_path"] = "different-runnability.json"
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_supplemental_evidence_path_mismatch",
    ):
        canonical_capture._validate_current_sol_runnability_evidence(
            evidence=wrong_basename,
            expected=copy.deepcopy(wrong_basename),
            binding=binding,
            identity={"slug": record["slug"]},
            observed_workloads=2,
            workload_path=workload,
            definition_name="test_definition",
            solution_name="test_solution",
            runnability_path=evidence_path,
        )
    wrong_hash = copy.deepcopy(evidence)
    wrong_hash["source_sha256"] = "0" * 64
    with pytest.raises(
        canonical_capture.ChunkRefusal, match="input_hash_mismatch",
    ):
        canonical_capture._validate_current_sol_runnability_evidence(
            evidence=wrong_hash,
            expected=copy.deepcopy(wrong_hash),
            binding=binding,
            identity={"slug": record["slug"]},
            observed_workloads=2,
            workload_path=workload,
            definition_name="test_definition",
            solution_name="test_solution",
            runnability_path=evidence_path,
        )


@pytest.mark.parametrize(
    ("source_tolerance", "expected_tolerance"),
    [
        (
            None,
            {
                "max_atol": 0.01,
                "max_rtol": 0.01,
                "required_matched_ratio": 0.99,
                "max_error_cap": None,
                "allow_negative_inf": False,
            },
        ),
        (
            {"max_atol": 0.0001, "allow_negative_inf": True},
            {
                "max_atol": 0.0001,
                "max_rtol": 0.01,
                "required_matched_ratio": 0.99,
                "max_error_cap": None,
                "allow_negative_inf": True,
            },
        ),
        (
            {
                "max_atol": 0.0001,
                "max_rtol": 0.05,
                "required_match_ratio": 0.98,
            },
            {
                "max_atol": 0.0001,
                "max_rtol": 0.05,
                "required_matched_ratio": 0.99,
                "max_error_cap": None,
                "allow_negative_inf": False,
            },
        ),
    ],
)
def test_sol_diagnostic_workload_normalization_matches_executor_defaults(
    source_tolerance, expected_tolerance,
):
    source = {
        "uuid": "12345678-1234-5123-8123-123456789abc",
        "axes": {"n": 1},
        "inputs": {"value": {"type": "random"}},
    }
    if source_tolerance is not None:
        source["tolerance"] = source_tolerance
    original = copy.deepcopy(source)
    assert canonical_capture._normalize_sol_diagnostic_workload(source) == {
        **source,
        "tolerance": expected_tolerance,
    }
    assert source == original


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value.update({"tolerance": {"unknown": True}}),
        lambda value: value.update({"tolerance": "not-an-object"}),
        lambda value: value.update({"tolerance": {"max_atol": True}}),
        lambda value: value.update({
            "tolerance": {"required_match_ratio": 0.97},
        }),
        lambda value: value.update({
            "tolerance": {
                "required_match_ratio": 0.98,
                "required_matched_ratio": 0.99,
            },
        }),
    ],
)
def test_sol_diagnostic_workload_normalization_rejects_unknown_or_wrong_types(
    mutate,
):
    source = {
        "uuid": "12345678-1234-5123-8123-123456789abc",
        "axes": {"n": 1},
        "inputs": {"value": {"type": "random"}},
    }
    mutate(source)
    with pytest.raises(
        canonical_capture.ChunkRefusal,
        match="sol_native_diagnostic_workload_invalid",
    ):
        canonical_capture._normalize_sol_diagnostic_workload(source)


def test_incomplete_partial_evidence_refuses_before_output_mutation(
    tmp_path, monkeypatch,
):
    output = tmp_path / "external-campaign"
    binding = {"row_key": {
        "slug": "problem_001_test",
        "source_workload_uuid": "12345678-1234-5123-8123-123456789abc",
        "variant_id": "a" * 64,
    }}
    row = {"key": binding["row_key"]}
    variant = {
        "cohort": "partial_native_row",
        "runnability_evidence": {"diagnostics_complete": False},
    }
    cell = {
        "identity": {"slug": "problem_001_test"},
        "category": "L1",
        "flashinfer_corpus": copy.deepcopy(FLASHINFER_L1_CONTRACT),
    }
    called = {"count": 0}
    runnability_path = tmp_path / "owner" / "runnability.json"

    def refuse(*_args, **kwargs):
        called["count"] += 1
        assert kwargs["supplemental_runnability_evidence"][
            "diagnostics_complete"
        ] is False
        assert kwargs["runnability_path"] == runnability_path
        raise capture.canonical_capture.ChunkRefusal(
            "sol_supplemental_evidence_incomplete", "test"
        )

    monkeypatch.setattr(
        capture.canonical_capture, "validate_capture_cell", refuse
    )
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="sol_supplemental_evidence_incomplete",
    ):
        capture.prevalidate_supplemental_rows(
            resolved=[(row, variant, cell)],
            bindings={capture._safe_name(row["key"]): binding},
            cell_budgets=object(),
            certified_sol_env={},
            flashinfer_proof={},
            runnability_path=runnability_path,
        )
    assert called["count"] == 1
    assert not output.exists()


def test_reduced_prevalidation_keeps_native_evidence_scope_closed(
    tmp_path, monkeypatch,
):
    key = {
        "slug": "problem_001_test",
        "source_workload_uuid": "12345678-1234-5123-8123-123456789abc",
        "variant_id": "a" * 64,
    }
    binding = {"row_key": key}
    row = {"key": key}
    variant = {
        "cohort": "reduced_proxy",
        "variant_contract_sha256": "a" * 64,
        "runnability_evidence": {"must_not_be_forwarded": True},
    }
    cell = {
        "identity": {"slug": key["slug"]},
        "category": "L1",
        "flashinfer_corpus": copy.deepcopy(FLASHINFER_L1_CONTRACT),
    }
    runnability_path = tmp_path / "owner" / "runnability.json"
    observed = {}

    def validate(*_args, **kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(
        capture.canonical_capture, "validate_capture_cell", validate
    )
    capture.prevalidate_supplemental_rows(
        resolved=[(row, variant, cell)],
        bindings={capture._safe_name(key): binding},
        cell_budgets=object(), certified_sol_env={}, flashinfer_proof={},
        runnability_path=runnability_path,
    )
    assert observed["runnability_path"] == runnability_path
    assert observed["supplemental_runnability_evidence"] is None


def test_invalid_matrix_refuses_before_external_output_mutation(tmp_path):
    matrix = tmp_path / "invalid.json"
    matrix.write_text("{}\n")
    runnability_root = tmp_path / "runnability-owner"
    shrink_root = tmp_path / "shrink-owner"
    live_inventory_root = tmp_path / "live-inventory-owner"
    runnability_root.mkdir(mode=0o700)
    shrink_root.mkdir(mode=0o700)
    live_inventory_root.mkdir(mode=0o700)
    runnability = runnability_root / "runnability.json"
    shrink = shrink_root / "shrink.json"
    inventory = tmp_path / "inventory.json"
    live_inventory = live_inventory_root / "live-inventory.json"
    for path in (runnability, shrink, inventory, live_inventory):
        path.write_text("{}\n")
    output = tmp_path / "campaign"
    args = SimpleNamespace(
        matrix=str(matrix), canonical_matrix="unused", authority="unused",
        pins="unused", aggregate_budget="unused", cell_budgets="unused",
        metric_registry="unused", preflight="unused", output_root=str(output),
        chunk_id="unused", runnability=str(runnability),
        runnability_root=str(runnability_root),
        shrink_provenance=str(shrink), shrink_root=str(shrink_root),
        full_coverage_inventory=str(inventory),
        live_coverage_inventory=str(live_inventory),
        live_coverage_inventory_root=str(live_inventory_root),
    )
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_matrix_schema_invalid",
    ):
        capture.run_chunk(args)
    assert not output.exists()


def test_failed_chunk_releases_campaign_lock_before_same_process_retry(
    tmp_path, monkeypatch,
):
    attempts = 0
    expected = tmp_path / "completed"

    def fail_once_then_complete(
        _args, *, campaign_lock_lease,
    ):
        nonlocal attempts
        attempts += 1
        campaign_lock_lease.acquire(tmp_path)
        with pytest.raises(
            canonical_capture.ChunkRefusal,
            match="campaign_writer_active",
        ):
            canonical_capture.acquire_campaign_lock(tmp_path)
        if attempts == 1:
            raise capture.WorkloadCaptureRefusal(
                "injected_post_lock_failure", "retain the first refusal"
            )
        return expected

    monkeypatch.setattr(
        capture, "_run_chunk_impl", fail_once_then_complete,
    )
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="injected_post_lock_failure",
    ) as first_failure:
        capture.run_chunk(SimpleNamespace())

    # Keep the original traceback reachable across the retry.  Before the
    # lifecycle fix, that traceback also kept the acquired file object alive
    # and the same process refused itself as campaign_writer_active.
    assert first_failure.value.code == "injected_post_lock_failure"
    assert capture.run_chunk(SimpleNamespace()) == expected
    assert attempts == 2
    with canonical_capture.acquire_campaign_lock(tmp_path):
        pass


def _minimal_live_inventory_contract(tmp_path: Path):
    key = {
        "slug": "problem_001_test",
        "source_workload_uuid": "12345678-1234-5123-8123-123456789abc",
        "variant_id": "native",
    }
    matrix = {"source_commit": "a" * 40, "rows": [{"key": key}]}
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_bytes(capture.workload_matrix.serialize_matrix(matrix))
    document = {
        "source_commit": matrix["source_commit"],
        "inputs": {
            "matrix": {
                "name": matrix_path.name,
                "sha256": capture.sha256_file(matrix_path),
            },
            "runnability": {"sha256": "b" * 64},
            "shrink": {"sha256": "c" * 64},
            "port_scope": {"sha256": "d" * 64},
            "benchmark_workloads": {"manifest_sha256": "e" * 64},
        },
        "counts": {
            "source_workload_rows": 3957,
            "matrix_schedule_rows": 1,
            "source_rows_with_any_schedule": 1,
            "source_rows_unscheduled_in_scope": 3419,
        },
        "rows": [{
            "source_row_sha256": "f" * 64,
            "matrix_row_keys": [key],
            "unscheduled_reason": None,
        }],
    }
    return matrix, matrix_path, document


def test_live_inventory_owner_read_binds_exact_matrix_rows(
    tmp_path, monkeypatch,
):
    matrix, matrix_path, expected = _minimal_live_inventory_contract(tmp_path)
    owner = tmp_path / "inventory-owner"
    owner.mkdir(mode=0o700)
    identity = capture.sol_recon.validate_external_campaign_root(owner)
    inventory = capture.sol_recon.publish_external_child(
        identity, "live-inventory.json",
        capture.live_coverage.canonical_json_bytes(expected),
    )
    monkeypatch.setattr(capture, "_validate_schema", lambda *_a, **_k: None)
    monkeypatch.setattr(
        capture.live_coverage, "build_live_inventory",
        lambda **_kwargs: copy.deepcopy(expected),
    )
    document, summary = capture.validate_live_coverage_inventory(
        inventory_path=inventory, inventory_root=owner,
        matrix=matrix, matrix_path=matrix_path,
        runnability_path=tmp_path / "runnability.json",
        runnability_root=tmp_path / "runnability-owner",
        shrink_path=tmp_path / "shrink.json",
        shrink_root=tmp_path / "shrink-owner",
    )
    assert document == expected
    assert summary["matrix_schedule_rows"] == 1
    assert summary["sha256"] == capture.sha256_file(inventory)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["inputs"]["runnability"].update(
            sha256="1" * 64
        ),
        lambda value: value["inputs"]["shrink"].update(sha256="2" * 64),
        lambda value: value["inputs"]["port_scope"].update(
            sha256="3" * 64
        ),
        lambda value: value["inputs"]["benchmark_workloads"].update(
            manifest_sha256="4" * 64
        ),
        lambda value: value["rows"][0].update(
            source_row_sha256="5" * 64
        ),
        lambda value: value["rows"][0].update(
            unscheduled_reason={"code": "fabricated"}
        ),
    ],
)
def test_live_inventory_requires_exact_regeneration_bytes(
    tmp_path, monkeypatch, mutation,
):
    matrix, matrix_path, expected = _minimal_live_inventory_contract(tmp_path)
    supplied = copy.deepcopy(expected)
    mutation(supplied)
    owner = tmp_path / "inventory-owner"
    owner.mkdir(mode=0o700)
    identity = capture.sol_recon.validate_external_campaign_root(owner)
    inventory = capture.sol_recon.publish_external_child(
        identity, "live-inventory.json",
        capture.live_coverage.canonical_json_bytes(supplied),
    )
    monkeypatch.setattr(capture, "_validate_schema", lambda *_a, **_k: None)
    monkeypatch.setattr(
        capture.live_coverage, "build_live_inventory",
        lambda **_kwargs: copy.deepcopy(expected),
    )
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_live_coverage_regeneration_mismatch",
    ):
        capture.validate_live_coverage_inventory(
            inventory_path=inventory, inventory_root=owner,
            matrix=matrix, matrix_path=matrix_path,
            runnability_path=tmp_path / "runnability.json",
            runnability_root=tmp_path / "runnability-owner",
            shrink_path=tmp_path / "shrink.json",
            shrink_root=tmp_path / "shrink-owner",
        )


def test_matrix_regeneration_uses_exact_external_canonical_authority_bundle(
    tmp_path, monkeypatch,
):
    canonical = tmp_path / "live-canonical.json"
    authority = tmp_path / "live-authority.json"
    runnability = tmp_path / "live-runnability.json"
    shrink = tmp_path / "live-shrink.json"
    for path in (canonical, authority, runnability, shrink):
        path.write_text("{}\n")
    matrix = {
        "canonical_matrix": {"path": str(canonical)},
        "canonical_authority": {"path": str(authority)},
    }
    matrix["matrix_seal_sha256"] = capture.canonical_sha256(matrix)
    matrix_path = tmp_path / "supplemental.json"
    matrix_path.write_bytes(capture.workload_matrix.serialize_matrix(matrix))
    observed = {}

    def fake_build(repo_root, **kwargs):
        observed["repo_root"] = repo_root
        observed.update(kwargs)
        return copy.deepcopy(matrix)

    monkeypatch.setattr(capture, "_validate_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(capture.workload_matrix, "build_matrix", fake_build)
    capture.validate_supplemental_matrix(
        matrix, matrix_path=matrix_path,
        canonical_matrix_path=canonical, authority_path=authority,
        runnability_path=runnability, shrink_path=shrink,
        runnability_root=tmp_path / "runnability-owner",
        shrink_root=tmp_path / "shrink-owner",
    )
    assert observed == {
        "repo_root": capture.REPO_ROOT,
        "canonical_matrix_path": canonical,
        "authority_path": authority,
        "runnability_path": runnability,
        "shrink_path": shrink,
        "runnability_root": tmp_path / "runnability-owner",
        "shrink_root": tmp_path / "shrink-owner",
    }
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_canonical_matrix_path_mismatch",
    ):
        capture.validate_supplemental_matrix(
            matrix, matrix_path=matrix_path,
            canonical_matrix_path=tmp_path / "wrong-canonical.json",
            authority_path=authority,
        )


def test_loaded_canonical_pin_is_repo_anchored_not_process_cwd(
    tmp_path, monkeypatch,
):
    relative = Path(
        "data/local_labels/campaign/sm86_reference_capture_matrix.json"
    )
    canonical_path = capture.REPO_ROOT / relative
    canonical = json.loads(canonical_path.read_text())
    expected = {
        "path": relative.as_posix(),
        "sha256": capture.sha256_file(canonical_path),
        "source_commit": canonical["source_commit"],
        "matrix_seal_sha256": canonical["matrix_seal_sha256"],
    }
    unrelated = tmp_path / "unrelated-caller-cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    capture._validate_loaded_canonical_pin(
        expected, canonical, canonical_path.resolve(),
    )

    bad = copy.deepcopy(expected)
    bad["sha256"] = "0" * 64
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_canonical_matrix_mismatch",
    ):
        capture._validate_loaded_canonical_pin(
            bad, canonical, canonical_path.resolve(),
        )


def test_done_schema_is_valid_and_rejects_claim_conflation():
    schema = json.loads(capture.DONE_SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    binding_schema = schema["$defs"]["binding"]
    # The runtime verifier supplies cross-field claim checks.  The schema still
    # closes the vocabulary so a generic/native-predictive synonym cannot enter.
    assert binding_schema["properties"]["cohort"]["enum"] == [
        "native", "partial_native_row", "reduced_proxy"
    ]
    claims = binding_schema["properties"]["claim_scope"]["enum"]
    assert "reduced_proxy_bundle_only" in claims
    assert "reduced_proxy_workload_row_only" not in claims
    assert "generic" not in claims
    assert binding_schema["properties"]["launch_scope"]["const"] == (
        "all_in_profiler_roi"
    )
    assert schema["$defs"]["cellResult"]["additionalProperties"] is False
    assert "launch_completeness" in schema["$defs"]["cellResult"]["required"]


def test_done_membership_order_is_exact_for_breadth_first_chunks():
    keys = [
        {
            "slug": f"problem_{problem:03d}_test",
            "source_workload_uuid": (
                f"{uuid_prefix:08x}-1234-5123-8123-123456789abc"
            ),
            "variant_id": "native",
        }
        for problem, uuid_prefix in ((1, 1), (2, 2), (1, 3))
    ]
    # Same problem may reappear at greater row depth; UUID keeps the key exact.
    results = [
        {"workload_row_binding": {"row_key": copy.deepcopy(key)}}
        for key in keys
    ]
    assert capture.require_exact_result_membership_order(keys, results) == [
        capture._row_identity(key) for key in keys
    ]
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="supplemental_done_membership_order_mismatch",
    ):
        capture.require_exact_result_membership_order(
            keys, [results[1], results[0], results[2]]
        )
