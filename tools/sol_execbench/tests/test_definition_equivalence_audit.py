from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sol_execbench.driver import capture_adapter
from tools.local_labels import audit_sol_definition_equivalence as audit
from tools.local_labels import sol_definition_equivalence as equiv


REPO_ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = REPO_ROOT / "data/local_labels/campaign/sm86_reference_capture_matrix.json"
BLOCKING_PATH = (
    REPO_ROOT
    / "evidence/round28/sol_equivalence/"
    "sol_definition_equivalence_audit_r28_v1_blocking.json"
)
FIXED_UTC = "2026-07-10T20:00:00Z"

# The V1 split and H1 CLI checks consume an external, versioned capture
# bundle.  Run those checks when their inputs are present, while allowing the
# current replay workspace to exercise the live equivalence and publication
# paths without manufacturing fixture data.
_HAS_V1_BLOCKING_FIXTURE = BLOCKING_PATH.is_file()
_HAS_H1_MATRIX_FIXTURE = audit.H1_MATRIX_PATH.is_file()
_requires_v1_blocking_fixture = pytest.mark.skipif(
    not _HAS_V1_BLOCKING_FIXTURE,
    reason="required V1 blocking artifact is unavailable",
)
_requires_h1_matrix_fixture = pytest.mark.skipif(
    not _HAS_H1_MATRIX_FIXTURE,
    reason="required H1 capture matrix is unavailable",
)


def _matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


def _build(monkeypatch) -> dict:
    matrix = _matrix()
    monkeypatch.setattr(audit, "_git_head", lambda _repo: matrix["source_commit"])
    monkeypatch.setattr(audit.matrix_builder, "validate_matrix", lambda *_args, **_kwargs: [])
    return audit.build_audit(
        matrix_path=MATRIX_PATH,
        expected_head=matrix["source_commit"],
        expected_matrix_sha256=audit.sha256_file(MATRIX_PATH),
        blocking_artifact_path=BLOCKING_PATH,
        certifying=False,
        created_at_utc=FIXED_UTC,
    )


def _tracked_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    fixture = repo / "fixture.json"
    fixture.write_bytes(b'{"authority": true}\n')
    fixture.chmod(0o644)
    digest = audit.sha256_file(fixture)
    manifest = repo / "evidence" / "MANIFEST.sha256"
    manifest.parent.mkdir()
    manifest.write_text(f"{digest}  fixture.json\n")
    subprocess.run(
        ["git", "add", "fixture.json", "evidence/MANIFEST.sha256"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture Test",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "add fixture",
        ],
        cwd=repo,
        check=True,
    )
    return repo, fixture, digest


def test_repository_fixture_accepts_tracked_clean_hash_match_at_0644(tmp_path):
    repo, fixture, digest = _tracked_fixture(tmp_path)
    assert fixture.stat().st_mode & 0o777 == 0o644
    assert audit._verify_repository_fixture(
        fixture, expected_sha256=digest, repo_root=repo
    ) == digest


def test_repository_fixture_hash_mismatch_has_typed_refusal(tmp_path):
    repo, fixture, _digest = _tracked_fixture(tmp_path)
    with pytest.raises(audit.AuditRefusal) as caught:
        audit._verify_repository_fixture(
            fixture, expected_sha256="0" * 64, repo_root=repo
        )
    assert caught.value.code == "AUDIT_FIXTURE_HASH_MISMATCH"


def test_repository_fixture_dirty_has_distinct_typed_refusal(tmp_path):
    repo, fixture, digest = _tracked_fixture(tmp_path)
    fixture.write_bytes(b'{"authority": false}\n')
    with pytest.raises(audit.AuditRefusal) as caught:
        audit._verify_repository_fixture(
            fixture, expected_sha256=digest, repo_root=repo
        )
    assert caught.value.code == "AUDIT_FIXTURE_DIRTY"


def test_repository_fixture_untracked_has_distinct_typed_refusal(tmp_path):
    repo, _fixture, _digest = _tracked_fixture(tmp_path)
    untracked = repo / "untracked.json"
    untracked.write_text("{}\n")
    with pytest.raises(audit.AuditRefusal) as caught:
        audit._verify_repository_fixture(
            untracked,
            expected_sha256=audit.sha256_file(untracked),
            repo_root=repo,
        )
    assert caught.value.code == "AUDIT_FIXTURE_UNTRACKED"


def test_repository_fixture_manifest_mismatch_has_distinct_typed_refusal(tmp_path):
    repo, fixture, digest = _tracked_fixture(tmp_path)
    manifest = repo / "evidence" / "MANIFEST.sha256"
    manifest.write_text(f"{'0' * 64}  fixture.json\n")
    subprocess.run(["git", "add", "evidence/MANIFEST.sha256"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture Test",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "plant wrong manifest hash",
        ],
        cwd=repo,
        check=True,
    )
    with pytest.raises(audit.AuditRefusal) as caught:
        audit._verify_repository_fixture(
            fixture, expected_sha256=digest, repo_root=repo
        )
    assert caught.value.code == "AUDIT_FIXTURE_MANIFEST_MISMATCH"


@_requires_v1_blocking_fixture
def test_real_74_v2_audit_matches_every_preserved_v1_axis_pointer(monkeypatch):
    artifact = _build(monkeypatch)
    blocking = json.loads(BLOCKING_PATH.read_text())
    blocking_rows = {row["row_id"]: row for row in blocking["rows"]}
    normalized_rows = [row for row in artifact["rows"] if row["normalized_pointers"]]

    assert artifact["rule_id"] == "SOL_DEF_EQUIV_V2"
    assert artifact["certifying"] is False
    assert artifact["summary"] == {
        "rows": 74,
        "unique_rows": 74,
        "equal": 74,
        "divergent": 0,
        "normalized_rows": 31,
        "total_normalized_pointers": 212,
        "total_residual_differences": 0,
    }
    assert len(normalized_rows) == 31
    assert len(artifact["full_membership"]["row_ids"]) == 344
    assert artifact["full_membership"]["sha256"] == audit.canonical_sha256(
        artifact["full_membership"]["row_ids"]
    )
    assert all(row["residual_differences"] == [] for row in artifact["rows"])
    for row in artifact["rows"]:
        assert row["normalized_pointers"] == blocking_rows[row["row_id"]][
            "difference_pointers"
        ]
    problem_003 = next(
        row
        for row in artifact["rows"]
        if row["row_id"]
        == "sol_execbench::problem_003_grouped_query_attention_with_rope_backward"
    )
    assert problem_003["normalized_pointers"] == [
        "/axes/head_dim/description",
        "/axes/hidden_size/description",
        "/axes/kv_proj_size/description",
        "/axes/num_attention_heads/description",
        "/axes/num_key_value_heads/description",
        "/axes/q_proj_size/description",
    ]
    control = artifact["problem_070_control"]
    assert control == {
        "row_id": audit.PROBLEM_070_ROW_ID,
        "source_definition_sha256": audit.PROBLEM_070_SOURCE_SHA256,
        "staged_definition_sha256": audit.PROBLEM_070_STAGED_SHA256,
        "verdict": "equal",
    }
    control_row = next(
        row for row in artifact["rows"] if row["row_id"] == audit.PROBLEM_070_ROW_ID
    )
    assert control_row["top_level_field_states"]["op_type"] == {
        "source": "missing",
        "staged": "null",
        "source_normalized": True,
        "staged_normalized": False,
    }
    assert not any(
        state["source_normalized"] or state["staged_normalized"]
        for state in control_row["axis_description_states"].values()
    )
    assert not any(
        state["source_normalized"] or state["staged_normalized"]
        for state in control_row["tensor_description_states"].values()
    )


@_requires_v1_blocking_fixture
def test_frozen_v1_and_v2_reproduce_the_full_production_split():
    matrix = _matrix()
    cells = audit._runnable_sol_rows(matrix)
    blocking = {
        row["row_id"]: row for row in json.loads(BLOCKING_PATH.read_text())["rows"]
    }
    v1_divergent = 0
    v1_pointers = 0
    v2_normalized = 0
    v2_residual = 0
    for cell in cells:
        identity = cell["identity"]
        row_id = f"sol_execbench::{identity['slug']}"
        relative = cell["invocation"]["logical_workload"]["definition_file"]
        source_path = REPO_ROOT / relative
        source = source_path.read_bytes()
        staged = capture_adapter._load_definition(source_path).model_dump_json().encode()
        v1 = equiv.compare_definition_bytes_v1(source, staged)
        v2 = equiv.compare_definition_bytes(source, staged)
        assert list(v1.difference_pointers) == blocking[row_id]["difference_pointers"]
        assert v2.difference_pointers == ()
        v1_divergent += bool(v1.difference_pointers)
        v1_pointers += len(v1.difference_pointers)
        v2_normalized += len(v2.normalized_pointers)
        v2_residual += len(v2.difference_pointers)
    assert (v1_divergent, v1_pointers, v2_normalized, v2_residual) == (
        31,
        212,
        212,
        0,
    )


@_requires_v1_blocking_fixture
def test_audit_object_and_bytes_are_deterministic(monkeypatch):
    first = _build(monkeypatch)
    second = _build(monkeypatch)
    assert first == second
    assert audit.audit_bytes(first) == audit.audit_bytes(second)
    assert [row["row_id"] for row in first["rows"]] == sorted(
        row["row_id"] for row in first["rows"]
    )


@_requires_v1_blocking_fixture
def test_atomic_publication_never_overwrites_existing_audit(tmp_path, monkeypatch):
    artifact = _build(monkeypatch)
    output = tmp_path / "audit.json"
    audit.write_audit_atomic(output, artifact)
    original = output.read_bytes()
    assert original == audit.audit_bytes(artifact)
    with pytest.raises(audit.AuditRefusal) as caught:
        audit.write_audit_atomic(output, artifact)
    assert caught.value.code == "AUDIT_OUTPUT_EXISTS"
    assert output.read_bytes() == original


@_requires_v1_blocking_fixture
def test_membership_reference_is_exact_and_certifying_requires_clean_tree(
    tmp_path, monkeypatch
):
    artifact = _build(monkeypatch)
    reference = tmp_path / "h1.json"
    reference.write_bytes(audit.audit_bytes(artifact))
    matrix = _matrix()
    monkeypatch.setattr(audit, "_git_head", lambda _repo: matrix["source_commit"])
    accepted = audit.build_audit(
        matrix_path=MATRIX_PATH,
        expected_head=matrix["source_commit"],
        expected_matrix_sha256=audit.sha256_file(MATRIX_PATH),
        certifying=False,
        created_at_utc=FIXED_UTC,
        membership_reference=reference,
    )
    assert accepted["membership_sha256"] == artifact["membership_sha256"]

    tampered = copy.deepcopy(artifact)
    tampered["rows"][0]["row_id"] = "sol_execbench::wrong"
    reference.write_bytes(audit.audit_bytes(tampered))
    with pytest.raises(audit.AuditRefusal) as caught:
        audit.build_audit(
            matrix_path=MATRIX_PATH,
            expected_head=matrix["source_commit"],
            expected_matrix_sha256=audit.sha256_file(MATRIX_PATH),
            certifying=False,
            created_at_utc=FIXED_UTC,
            membership_reference=reference,
        )
    assert caught.value.code == "AUDIT_MEMBERSHIP_MISMATCH"

    tampered = copy.deepcopy(artifact)
    tampered["full_membership"]["row_ids"][0] = "portable_lcd::wrong"
    tampered["full_membership"]["row_ids"].sort()
    tampered["full_membership"]["sha256"] = audit.canonical_sha256(
        tampered["full_membership"]["row_ids"]
    )
    reference.write_bytes(audit.audit_bytes(tampered))
    with pytest.raises(audit.AuditRefusal) as caught:
        audit.build_audit(
            matrix_path=MATRIX_PATH,
            expected_head=matrix["source_commit"],
            expected_matrix_sha256=audit.sha256_file(MATRIX_PATH),
            certifying=False,
            created_at_utc=FIXED_UTC,
            membership_reference=reference,
        )
    assert caught.value.code == "AUDIT_MEMBERSHIP_MISMATCH"

    reference.write_bytes(audit.audit_bytes(artifact))
    monkeypatch.setattr(audit, "_git_clean", lambda _repo: False)
    with pytest.raises(audit.AuditRefusal) as caught:
        audit.build_audit(
            matrix_path=MATRIX_PATH,
            expected_head=matrix["source_commit"],
            expected_matrix_sha256=audit.sha256_file(MATRIX_PATH),
            certifying=True,
            created_at_utc=FIXED_UTC,
            membership_reference=reference,
        )
    assert caught.value.code == "AUDIT_TREE_DIRTY"


@_requires_h1_matrix_fixture
def test_h1_cli_role_is_bound_to_exact_matrix_head_seal_and_blocking_path():
    base = {
        "output": audit.DEFAULT_PRECOMMIT_ARTIFACT,
        "blocking_artifact": audit.DEFAULT_BLOCKING_ARTIFACT,
        "certifying": False,
        "certifying_current_head": False,
        "membership_reference": None,
        "matrix": audit.H1_MATRIX_PATH,
        "expected_head": audit.H1_HEAD,
        "expected_matrix_sha256": audit.H1_MATRIX_SHA256,
    }
    audit._validate_cli_role(SimpleNamespace(**base))

    for field, value in {
        "matrix": MATRIX_PATH,
        "expected_head": "0" * 40,
        "expected_matrix_sha256": "0" * 64,
        "blocking_artifact": Path("/tmp/substituted-blocking.json"),
    }.items():
        bad = {**base, field: value}
        with pytest.raises(audit.AuditRefusal):
            audit._validate_cli_role(SimpleNamespace(**bad))


def test_current_head_certifying_role_allows_only_versioned_staging_output():
    head = "1" * 40
    output = (
        audit.REPO_ROOT
        / "results"
        / "nvbit_capture_staging"
        / f"r28_{head[:8]}"
        / "sol_definition_equivalence_audit.json"
    ).resolve()
    base = {
        "output": output,
        "blocking_artifact": audit.DEFAULT_BLOCKING_ARTIFACT,
        "certifying": True,
        "certifying_current_head": True,
        "membership_reference": audit.DEFAULT_PRECOMMIT_ARTIFACT,
        "matrix": Path("/tmp/matrix.json"),
        "expected_head": head,
        "expected_matrix_sha256": "2" * 64,
    }

    audit._validate_cli_role(SimpleNamespace(**base))

    invalid = [
        {**base, "certifying": False},
        {**base, "output": Path("/tmp/audit.json")},
        {
            **base,
            "output": (
                audit.REPO_ROOT
                / "results"
                / "nvbit_capture_staging"
                / "wrong-head"
                / "sol_definition_equivalence_audit.json"
            ),
        },
        {**base, "certifying_current_head": False},
    ]
    for candidate in invalid:
        with pytest.raises(audit.AuditRefusal) as caught:
            audit._validate_cli_role(SimpleNamespace(**candidate))
        assert caught.value.code == "AUDIT_OUTPUT_ROLE_MISMATCH"


def test_audit_refuses_a_matrix_with_an_invalid_canonical_seal(
    tmp_path, monkeypatch
):
    matrix = _matrix()
    matrix["matrix_seal_sha256"] = "0" * 64
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    monkeypatch.setattr(audit, "_git_head", lambda _repo: matrix["source_commit"])

    with pytest.raises(audit.AuditRefusal) as caught:
        audit.build_audit(
            matrix_path=matrix_path,
            expected_head=matrix["source_commit"],
            expected_matrix_sha256=audit.sha256_file(matrix_path),
            certifying=False,
            created_at_utc=FIXED_UTC,
        )

    assert caught.value.code == "AUDIT_MATRIX_SEAL_MISMATCH"


def test_audit_refuses_matrix_semantic_regeneration_failure(monkeypatch):
    matrix = _matrix()
    monkeypatch.setattr(audit, "_git_head", lambda _repo: matrix["source_commit"])
    monkeypatch.setattr(
        audit.matrix_builder,
        "validate_matrix",
        lambda *_args, **_kwargs: ["cells", "counts.universe"],
    )

    with pytest.raises(audit.AuditRefusal) as caught:
        audit.build_audit(
            matrix_path=MATRIX_PATH,
            expected_head=matrix["source_commit"],
            expected_matrix_sha256=audit.sha256_file(MATRIX_PATH),
            certifying=False,
            created_at_utc=FIXED_UTC,
        )

    assert caught.value.code == "AUDIT_MATRIX_INVALID"


def test_atomic_publication_refuses_a_destination_created_at_rename(
    tmp_path, monkeypatch
):
    artifact = {"value": "new"}
    output = tmp_path / "audit.json"
    original_rename = audit._rename_noreplace

    def raced_rename(source, destination):
        destination.write_bytes(b"raced authority\n")
        original_rename(source, destination)

    monkeypatch.setattr(audit, "_rename_noreplace", raced_rename)
    with pytest.raises(audit.AuditRefusal) as caught:
        audit.write_audit_atomic(output, artifact)

    assert caught.value.code == "AUDIT_OUTPUT_EXISTS"
    assert output.read_bytes() == b"raced authority\n"
