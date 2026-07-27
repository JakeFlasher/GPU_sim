from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import archive_preparation_failure as ap


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_LIVE_JOURNALS = Path(
    "/home/jakeshea/sm86_reference_capture_r27/.staging/"
    "9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a/"
    "sol_execbench-000-bootstrap/"
    "fd3a1868845ceac309e718da6519f4779824b7fd69eddf1bfa0e7e7847d4e297/"
    "journals"
)
_LIVE_MATRIX = Path(
    "/home/jakeshea/sm86_reference_capture_r27/inputs/r28_9b8e0e40/"
    "sm86_reference_capture_matrix.json"
)
_LIVE_RUN_JOURNAL = _LIVE_JOURNALS.parent / "chunk_run_journal.jsonl"
_LIVE_PACKAGER = (
    _REPOSITORY_ROOT
    / "tools/sol_execbench/src/sol_execbench/driver/problem_packager.py"
)
_LIVE_CORRECTION = (
    _REPOSITORY_ROOT
    / ".humanize/rlcr/2026-07-08_11-39-13/"
    "adjudication_r28_prep070_reproduction_contract_correction.md"
)
_LIVE_CORRECTION2 = (
    _REPOSITORY_ROOT
    / ".humanize/rlcr/2026-07-08_11-39-13/"
    "adjudication_r28_post_move_failure_state_correction.md"
)
_LEGACY_PATH = "tools/local_labels/sm86_capture_chunk.py"
_H3_SUBJECT = (
    "fix(local_labels): rebind prep070 refusal authority to legacy "
    "9b8e0e40 comparator"
)
_FROZEN_PATH_ALIASES_V1_SHA256 = (
    "94446cc702ffe86ecb5b35ae125a20cb758b17a4c521bfcefd4a308618f8c737"
)
_FROZEN_PATH_ALIASES_V2_SHA256 = (
    "00197a24756dc5c7b1b75ee21b68695a83650619ea4f58d184892e2361710747"
)
_FROZEN_ADR_0005_SHA256 = (
    "0584eb99359d2af9d7ec192b6d1c84c8042f6a26d36466a71b836265c0cbb290"
)


def test_path_alias_baselines_remain_frozen_and_v3_is_inactive() -> None:
    v1_path = _REPOSITORY_ROOT / "evidence/path_aliases.v1.json"
    v2_path = _REPOSITORY_ROOT / "evidence/path_aliases.v2.json"
    adr5_path = (
        _REPOSITORY_ROOT
        / "docs/decisions/0005-sol-problem-039-capture-envelope.md"
    )
    v3_path = _REPOSITORY_ROOT / "evidence/path_aliases.v3.json"
    adr7_path = (
        _REPOSITORY_ROOT
        / "docs/decisions/0007-evidence-retention-and-path-aliases.md"
    )
    decision_index = (_REPOSITORY_ROOT / "docs/decisions/README.md").read_text(
        encoding="utf-8"
    )

    assert _sha(v1_path) == _FROZEN_PATH_ALIASES_V1_SHA256
    assert _sha(v2_path) == _FROZEN_PATH_ALIASES_V2_SHA256
    assert _sha(adr5_path) == _FROZEN_ADR_0005_SHA256
    assert ap.ROUND28_PATH_ALIASES == v1_path
    v1 = json.loads(v1_path.read_bytes())
    v2 = json.loads(v2_path.read_bytes())
    v3 = json.loads(v3_path.read_bytes())
    adr5 = adr5_path.read_text(encoding="utf-8")
    adr7 = adr7_path.read_text(encoding="utf-8")

    assert v1["schema_version"] == 1
    assert v2["schema_version"] == 2
    assert "prefix_mappings" in v2 and "aliases" not in v2
    assert "- Status: Accepted" in adr5
    assert "SOL_PROBLEM_039_CAPTURE_ENVELOPE_21600" in adr5
    assert v3["schema_version"] == 3
    assert v3["authority_domain"] == (
        "round28_preparation_failure_adjudications"
    )
    assert v3["activation"] == "inactive_operator_acceptance_required"
    assert v3["supersedes"] == {
        "path": "evidence/path_aliases.v1.json",
        "sha256": _FROZEN_PATH_ALIASES_V1_SHA256,
    }
    status_lines = [
        line for line in adr7.splitlines()
        if line.startswith("- **Status:**")
    ]
    assert status_lines == [
        "- **Status:** Proposed; inactive pending explicit operator acceptance"
    ]
    assert "0007-evidence-retention-and-path-aliases.md" not in decision_index


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _commit(repo: Path, subject: str) -> str:
    _git(repo, "add", "-A")
    subprocess.run(
        [
            "git", "-c", "user.name=Round28 Fixture",
            "-c", "user.email=round28-fixture@example.invalid",
            "commit", "-m", subject,
        ],
        cwd=repo, text=True, capture_output=True, check=True,
    )
    return _git(repo, "rev-parse", "HEAD")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class _ReconstructedFixture:
    def __init__(
        self, *, request: ap.ArchiveRequest, repo: Path,
        h2: str, h3: str, journal: Path, legacy_blob: bytes,
        adjudication_hashes: dict[str, str],
    ) -> None:
        self.request = request
        self.repo = repo
        self.h2 = h2
        self.h3 = h3
        self.journal = journal
        self.legacy_blob = legacy_blob
        self.adjudication_hashes = adjudication_hashes


@pytest.fixture
def repository_adjudications(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    canonical = repo / "evidence/round28/preparation_failures/problem-070"
    canonical.mkdir(parents=True)
    names = {
        "corrected": "adjudication_r28_prep070_reconstruction_v2.md",
        "superseded": "adjudication_r28_prep070_reconstruction_superseded_v1.md",
        "correction": "adjudication_r28_prep070_reproduction_contract_correction.md",
        "correction2": "adjudication_r28_post_move_failure_state_correction.md",
        "stage_equivalence": "adjudication_r28_stage_equivalence.md",
    }
    paths = {}
    for index, (key, name) in enumerate(names.items(), start=1):
        paths[key] = canonical / name
        paths[key].write_text(f"fixture {index}\n", encoding="utf-8")
    manifest = repo / "evidence/MANIFEST.sha256"
    manifest.write_text("".join(
        f"{_sha(path)}  {path.relative_to(repo).as_posix()}\n"
        for path in paths.values()
    ), encoding="utf-8")
    historical = tmp_path / "original/.humanize/rlcr/session"
    historical.mkdir(parents=True)
    historical_stage = historical / names["stage_equivalence"]
    historical_stage.write_bytes(paths["stage_equivalence"].read_bytes())
    aliases = repo / "evidence/path_aliases.v1.json"
    aliases.write_text(json.dumps({
        "aliases": [{
            "match": "prefix", "historical": f"{historical}/",
            "canonical": "evidence/round28/preparation_failures/problem-070/",
            "requires_sha256_match": True,
        }],
    }), encoding="utf-8")
    _commit(repo, "fixture evidence")
    monkeypatch.setattr(ap, "REPO_ROOT", repo)
    monkeypatch.setattr(ap, "ROUND28_EVIDENCE_MANIFEST", manifest)
    monkeypatch.setattr(ap, "ROUND28_PATH_ALIASES", aliases)
    monkeypatch.setattr(ap, "ROUND28_CORRECTION_SHA256", _sha(paths["correction"]))
    monkeypatch.setattr(ap, "ROUND28_CORRECTION2_SHA256", _sha(paths["correction2"]))
    monkeypatch.setattr(ap, "ROUND28_ANNEX_C_SHA256", _sha(paths["stage_equivalence"]))
    request = ap.ArchiveRequest(
        journals_root=tmp_path / "journals",
        preparation_failures_root=tmp_path / "preparation_failures",
        original_matrix_path=tmp_path / "matrix.json",
        chunk_run_journal=tmp_path / "chunk.jsonl",
        runner_refusal_record=None, phase_log=None, gpu_process_observation=None,
        adjudication_ref=historical / names["stage_equivalence"],
        rlcr_dir=historical, replacement_head="fixture",
        problem_packager_source=tmp_path / "packager.py",
        corrected_adjudication_ref=paths["corrected"],
        superseded_adjudication_ref=paths["superseded"],
        correction_adjudication_ref=paths["correction"],
        correction2_adjudication_ref=paths["correction2"],
        stage_equivalence_adjudication_source=paths["stage_equivalence"],
    )
    return repo, request, paths, aliases


def test_repository_adjudications_accept_tracked_clean_0644(
    repository_adjudications,
) -> None:
    _, request, paths, _ = repository_adjudications
    assert all(path.stat().st_mode & 0o777 == 0o644 for path in paths.values())
    assert ap._adjudication_hashes(request)["stage_equivalence"] == _sha(
        paths["stage_equivalence"]
    )


def test_repository_adjudication_dirty_refusal_is_typed(
    repository_adjudications,
) -> None:
    _, request, paths, _ = repository_adjudications
    paths["correction2"].write_bytes(paths["correction2"].read_bytes() + b"x")
    _assert_refusal("PREP_ARCHIVE_EVIDENCE_DIRTY", ap._adjudication_hashes, request)


def test_repository_adjudication_hash_refusal_is_typed(
    repository_adjudications,
) -> None:
    repo, request, paths, _ = repository_adjudications
    paths["correction2"].write_bytes(paths["correction2"].read_bytes() + b"x")
    _commit(repo, "plant hash mismatch")
    _assert_refusal(
        "PREP_ARCHIVE_EVIDENCE_HASH_MISMATCH", ap._adjudication_hashes, request,
    )


def test_repository_adjudication_alias_refusal_is_typed(
    repository_adjudications,
) -> None:
    repo, request, _, aliases = repository_adjudications
    value = json.loads(aliases.read_text(encoding="utf-8"))
    value["aliases"][0]["canonical"] = "evidence/wrong/"
    aliases.write_text(json.dumps(value), encoding="utf-8")
    _commit(repo, "plant alias mismatch")
    _assert_refusal(
        "PREP_ARCHIVE_EVIDENCE_ALIAS_MISMATCH", ap._adjudication_hashes, request,
    )


def test_repository_adjudication_historical_hash_refusal_is_typed(
    repository_adjudications,
) -> None:
    _, request, _, _ = repository_adjudications
    request.adjudication_ref.write_bytes(
        request.adjudication_ref.read_bytes() + b"historical drift\n"
    )
    _assert_refusal(
        "PREP_ARCHIVE_EVIDENCE_ALIAS_HASH_MISMATCH",
        ap._adjudication_hashes, request,
    )


@pytest.fixture
def reconstructed_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create an isolated H2/H3 repo and byte-exact copy of the old journal."""

    if not (_LIVE_JOURNALS.is_dir() and _LIVE_MATRIX.is_file()):
        pytest.skip("the sealed problem-070 fixture is unavailable")

    legacy_blob = subprocess.run(
        [
            "git", "show",
            f"{ap.ROUND28_ORIGINAL_HEAD}:{_LEGACY_PATH}",
        ],
        cwd=_REPOSITORY_ROOT, capture_output=True, check=True,
    ).stdout

    repo = tmp_path / "fixture-repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "README").write_text("fixture h2\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        ".humanize/\n__pycache__/\n", encoding="utf-8",
    )
    packager = (
        repo / "tools/sol_execbench/src/sol_execbench/driver"
        / "problem_packager.py"
    )
    packager.parent.mkdir(parents=True)
    packager.write_bytes(_LIVE_PACKAGER.read_bytes())
    evidence = tmp_path / "evidence"
    journals = evidence / "journals"
    shutil.copytree(_LIVE_JOURNALS, journals, copy_function=shutil.copy2)
    matrix = evidence / "sm86_reference_capture_matrix.json"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_LIVE_MATRIX, matrix)
    run_journal = evidence / "chunk_run_journal.jsonl"
    shutil.copy2(_LIVE_RUN_JOURNAL, run_journal)

    rlcr = tmp_path / "historical-rlcr"
    stage_adjudication = rlcr / "adjudication_r28_stage_equivalence.md"
    canonical = repo / "evidence/round28/preparation_failures/problem-070"
    canonical.mkdir(parents=True)
    stage_source = canonical / stage_adjudication.name
    stage_source.write_text("fixture Annex C\n", encoding="utf-8")
    rlcr.mkdir(parents=True)
    stage_adjudication.write_bytes(stage_source.read_bytes())
    corrected = canonical / "adjudication_r28_prep070_reconstruction_v2.md"
    corrected.write_text("corrected V2 fixture\n", encoding="utf-8")
    superseded = canonical / "adjudication_r28_prep070_reconstruction_superseded_v1.md"
    superseded.write_text("superseded V1 fixture\n", encoding="utf-8")
    correction = canonical / "adjudication_r28_prep070_reproduction_contract_correction.md"
    correction.write_bytes(_LIVE_CORRECTION.read_bytes())
    correction2 = canonical / "adjudication_r28_post_move_failure_state_correction.md"
    correction2.write_bytes(_LIVE_CORRECTION2.read_bytes())
    manifest = repo / "evidence/MANIFEST.sha256"
    manifest.write_text("".join(
        f"{_sha(path)}  {path.relative_to(repo).as_posix()}\n"
        for path in (stage_source, corrected, superseded, correction, correction2)
    ), encoding="utf-8")
    aliases = repo / "evidence/path_aliases.v1.json"
    aliases.write_text(json.dumps({
        "schema_version": 1,
        "aliases": [{
            "match": "prefix", "historical": f"{rlcr}/",
            "canonical": "evidence/round28/preparation_failures/problem-070/",
            "requires_sha256_match": True,
        }],
    }), encoding="utf-8")

    h2 = _commit(repo, "fixture H2")
    tool = repo / "tools/local_labels/legacy_compare_9b8e0e40.py"
    tool.parent.mkdir(parents=True)
    tool.write_bytes(
        (_REPOSITORY_ROOT / "tools/local_labels/legacy_compare_9b8e0e40.py")
        .read_bytes()
    )
    h3 = _commit(repo, _H3_SUBJECT)

    reconstruction = canonical / "prep070_reconstruction"
    session_log = tmp_path / "session_log.jsonl"
    prep_root = tmp_path / "preparation_failures"
    adjudication_hashes = {
        "corrected": _sha(corrected),
        "superseded": _sha(superseded),
        "correction": _sha(correction),
        "correction2": _sha(correction2),
        "stage_equivalence": _sha(stage_source),
    }

    # Only filesystem/Git locations vary in this isolated fixture.  Every
    # historical identity, definition byte, schema, and journal pin stays real.
    monkeypatch.setattr(ap, "REPO_ROOT", repo)
    monkeypatch.setattr(ap, "ROUND28_H2_HEAD", h2)
    monkeypatch.setattr(ap, "ROUND28_CAMPAIGN_ROOT", tmp_path)
    monkeypatch.setattr(ap, "ROUND28_JOURNALS_ROOT", journals)
    monkeypatch.setattr(ap, "ROUND28_PREPARATION_FAILURES_ROOT", prep_root)
    monkeypatch.setattr(ap, "ROUND28_RLCR_DIR", rlcr)
    monkeypatch.setattr(
        ap, "ROUND28_STAGE_EQUIVALENCE_ADJUDICATION", stage_adjudication,
    )
    monkeypatch.setattr(ap, "ROUND28_PROBLEM_PACKAGER_SOURCE", packager)
    monkeypatch.setattr(ap, "ROUND28_RECONSTRUCTION_ROOT", reconstruction)
    monkeypatch.setattr(ap, "ROUND28_SESSION_LOG", session_log)
    monkeypatch.setattr(ap, "ROUND28_EVIDENCE_MANIFEST", manifest)
    monkeypatch.setattr(ap, "ROUND28_PATH_ALIASES", aliases)
    monkeypatch.setattr(ap, "ROUND28_STAGE_EQUIVALENCE_SOURCE", stage_source)
    monkeypatch.setattr(ap, "ROUND28_ANNEX_C_SHA256", _sha(stage_source))
    monkeypatch.setattr(ap, "ROUND28_CORRECTED_ADJUDICATION", corrected)
    monkeypatch.setattr(ap, "ROUND28_SUPERSEDED_ADJUDICATION", superseded)
    monkeypatch.setattr(ap, "ROUND28_CORRECTION_ADJUDICATION", correction)
    monkeypatch.setattr(ap, "ROUND28_CORRECTION_SHA256", _sha(correction))
    monkeypatch.setattr(ap, "ROUND28_CORRECTION2_ADJUDICATION", correction2)
    monkeypatch.setattr(ap, "ROUND28_CORRECTION2_SHA256", _sha(correction2))

    original_git_bytes = ap._git_bytes

    def fixture_git_bytes(repo_root: Path, *args: str, code: str) -> bytes:
        if args == (
            "show", f"{ap.ROUND28_ORIGINAL_HEAD}:{_LEGACY_PATH}",
        ):
            return legacy_blob
        return original_git_bytes(repo_root, *args, code=code)

    monkeypatch.setattr(ap, "_git_bytes", fixture_git_bytes)

    original_run = ap.subprocess.run

    def fixture_run(command, *args, **kwargs):
        if command and command[0] == "nvidia-smi":
            raw = b"pid, process_name, used_gpu_memory [MiB]\n"
            return subprocess.CompletedProcess(command, 0, stdout=raw, stderr=b"")
        if command == [
            "git", "show",
            f"{ap.ROUND28_ORIGINAL_HEAD}:"
            "tools/sol_execbench/src/sol_execbench/driver/problem_packager.py",
        ]:
            return subprocess.CompletedProcess(
                command, 0, stdout=packager.read_bytes(), stderr=b"",
            )
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(ap.subprocess, "run", fixture_run)

    request = ap.ArchiveRequest(
        journals_root=journals,
        preparation_failures_root=prep_root,
        original_matrix_path=matrix,
        chunk_run_journal=run_journal,
        runner_refusal_record=None,
        phase_log=None,
        gpu_process_observation=None,
        adjudication_ref=stage_adjudication,
        rlcr_dir=rlcr,
        replacement_head=h3,
        problem_packager_source=packager,
        context_mode=ap.CONTEXT_MODE_RECONSTRUCTED,
        reconstruction_root=reconstruction,
        session_log_path=session_log,
        corrected_adjudication_ref=corrected,
        superseded_adjudication_ref=superseded,
        correction_adjudication_ref=correction,
        correction2_adjudication_ref=correction2,
        stage_equivalence_adjudication_source=stage_source,
    )
    _jsonl(
        session_log,
        ap._required_session_prefix(
            request, h3=h3, adjudications=adjudication_hashes,
        ),
    )
    yield _ReconstructedFixture(
        request=request, repo=repo, h2=h2, h3=h3,
        journal=journals / ap.ROUND28_CELL_ID,
        legacy_blob=legacy_blob, adjudication_hashes=adjudication_hashes,
    )


def _assert_refusal(code: str, function, *args, **kwargs) -> None:
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code


def _reconstructed_cli_args(
    request: ap.ArchiveRequest, *, action: str,
) -> list[str]:
    return [
        action,
        "--context-mode", ap.CONTEXT_MODE_RECONSTRUCTED,
        "--journals-root", str(request.journals_root),
        "--preparation-failures-root", str(request.preparation_failures_root),
        "--original-matrix", str(request.original_matrix_path),
        "--chunk-run-journal", str(request.chunk_run_journal),
        "--replacement-head", request.replacement_head,
        "--rlcr-dir", str(request.rlcr_dir),
        "--adjudication-ref", str(request.adjudication_ref),
        "--problem-packager-source", str(request.problem_packager_source),
        "--reconstruction-root", str(request.reconstruction_root),
        "--session-log", str(request.session_log_path),
        "--corrected-adjudication-ref",
        str(request.corrected_adjudication_ref),
        "--superseded-adjudication-ref",
        str(request.superseded_adjudication_ref),
        "--correction-adjudication-ref",
        str(request.correction_adjudication_ref),
        "--correction2-adjudication-ref",
        str(request.correction2_adjudication_ref),
    ]


def _tree_snapshot(root: Path) -> list[tuple[str, str, int, str | None]]:
    if not root.exists():
        return []
    rows: list[tuple[str, str, int, str | None]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            rows.append((relative, "file", path.stat().st_mode, _sha(path)))
        elif path.is_dir():
            rows.append((relative, "directory", path.stat().st_mode, None))
        else:
            rows.append((relative, "other", path.lstat().st_mode, None))
    return rows


def _install_post_move_inventory_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renamed = False
    original_rename = ap._rename_noreplace
    original_inventory = ap.content_inventory

    def tracked_rename(source, destination):
        nonlocal renamed
        original_rename(source, destination)
        renamed = True

    def post_move_drift(root):
        rows = original_inventory(root)
        if renamed:
            rows[0] = {**rows[0], "sha256": "0" * 64}
        return rows

    monkeypatch.setattr(ap, "_rename_noreplace", tracked_rename)
    monkeypatch.setattr(ap, "content_inventory", post_move_drift)


def test_reconstructed_builder_sandwich_external_source_and_seven_log_events(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    before = ap.content_inventory(fixture.journal)
    records = ap.build_reconstructed_context(fixture.request)

    assert list(records) == list(ap.ROUND28_RECONSTRUCTION_RECORD_ORDER)
    root = fixture.request.reconstruction_root
    source = root / "scratch/source_definition.json"
    staged = root / "scratch/staged_definition.json"
    assert _sha(source) == ap.ROUND28_PINNED_SOURCE_DEFINITION_SHA256
    assert _sha(staged) == ap.ROUND28_STAGED_DEFINITION_SHA256
    assert source.stat().st_mode & 0o777 == 0o444
    assert staged.stat().st_mode & 0o777 == 0o444
    assert ap.content_inventory(fixture.journal) == before

    reproduction = records["legacy_comparator_reproduction.json"]
    assert reproduction["tool"]["head"] == fixture.h3
    assert reproduction["verdict"] == {
        "equal": False,
        "differing_top_level_keys": ["op_type"],
        "difference_pointers": ["/op_type"],
        "maps_to_refusal_code": "sol_stage_source_mismatch",
    }
    assert reproduction["inputs"]["source_origin"] == {
        "kind": "external_prepare_manifest_pinned",
        "prepare_manifest_journal_relative_path": (
            ap.ROUND28_MANIFEST_JOURNAL_RELATIVE_PATH
        ),
        "prepare_manifest_sha256": ap.ROUND28_PREPARE_MANIFEST_SHA256,
        "source_path": str(ap.ROUND28_SOURCE_DEFINITION_PATH),
    }
    assert reproduction["inputs"]["staged_origin"] == {
        "kind": "historical_journal_member",
        "journal_relative_path": (
            ap.ROUND28_STAGED_DEFINITION_JOURNAL_RELATIVE_PATH
        ),
    }
    assert len(reproduction["command_argv"]) == 13
    assert reproduction["command_argv"][10] == fixture.h3

    rows = ap._load_session_log(fixture.request.session_log_path)
    record_rows = [
        row for row in rows if row.get("event") == "reconstruction_record_created"
    ]
    assert [Path(row["artifact"]).name for row in record_rows] == [
        "legacy_comparator_source_excerpt.py.txt",
        "legacy_comparator_reproduction.json",
        "runner_refusal_reconstruction.json",
        "phase_reconstruction.json",
        "gpu_process_observation_late.raw.txt",
        "gpu_process_observation_late.json",
        "preparation_failure_no_gpu_attestation_v2.json",
    ]
    assert len(record_rows) == 7
    for name, record in records.items():
        schema = json.loads(ap.RECONSTRUCTION_RECORD_SCHEMAS[name].read_text())
        jsonschema.Draft202012Validator(schema).validate(record)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("manifest_hash", "SOL_LEGACY_REPRO_INPUT_DRIFT"),
        ("external_source_missing", "SOL_LEGACY_REPRO_INPUT_DRIFT"),
        ("journal_complete", "PREPARATION_ARCHIVE_NOT_PREGPU"),
    ],
)
def test_reconstructed_builder_refuses_input_and_completion_drift_before_tool(
    reconstructed_fixture: _ReconstructedFixture,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = reconstructed_fixture
    if mutation == "manifest_hash":
        manifest = (
            fixture.journal / ap.ROUND28_MANIFEST_JOURNAL_RELATIVE_PATH
        )
        manifest.write_bytes(manifest.read_bytes() + b" ")
    elif mutation == "external_source_missing":
        monkeypatch.setattr(
            ap, "ROUND28_SOURCE_DEFINITION_PATH",
            fixture.repo / "does-not-exist.json",
        )
    else:
        (fixture.journal / "journal_complete.json").write_text("{}\n")

    _assert_refusal(expected_code, ap.build_reconstructed_context, fixture.request)
    assert not (
        fixture.request.reconstruction_root / "legacy_comparator_reproduction.json"
    ).exists()


@pytest.mark.parametrize(
    "action",
    [ap.build_reconstructed_context, ap.archive_preparation_failure],
)
@pytest.mark.parametrize(
    "field", ["adjudication_ref", "problem_packager_source"],
)
def test_reconstructed_allowlist_pins_context_source_paths_before_mutation(
    reconstructed_fixture: _ReconstructedFixture,
    action,
    field: str,
) -> None:
    fixture = reconstructed_fixture
    replacement = fixture.request.rlcr_dir / f"wrong-{field}.txt"
    replacement.write_text("wrong reconstructed input\n", encoding="utf-8")
    request = replace(fixture.request, **{field: replacement})
    journal_before = _tree_snapshot(fixture.request.journals_root)
    preparation_before = _tree_snapshot(
        fixture.request.preparation_failures_root
    )

    _assert_refusal(
        "SOL_PREP_RECON_ALLOWLIST_VIOLATION", action, request,
    )

    assert _tree_snapshot(fixture.request.journals_root) == journal_before
    assert _tree_snapshot(
        fixture.request.preparation_failures_root
    ) == preparation_before
    assert not fixture.request.reconstruction_root.exists()


@pytest.mark.parametrize("drift", ["dirty", "head", "parent", "subject"])
def test_h3_builder_identity_failures_are_typed(
    reconstructed_fixture: _ReconstructedFixture,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    fixture = reconstructed_fixture
    request = fixture.request
    if drift == "dirty":
        (fixture.repo / "dirty.txt").write_text("dirty\n")
    elif drift == "head":
        request = replace(request, replacement_head="f" * 40)
    elif drift == "parent":
        monkeypatch.setattr(ap, "ROUND28_H2_HEAD", "f" * 40)
    else:
        monkeypatch.setattr(ap, "ROUND28_H3_SUBJECT", "wrong subject")

    _assert_refusal(
        "SOL_LEGACY_REPRO_HEAD_MISMATCH", ap.build_reconstructed_context, request,
    )
    assert not request.reconstruction_root.exists()


@pytest.mark.parametrize(
    "action",
    ["--build-reconstructed-context", "--archive-preparation-failure"],
)
@pytest.mark.parametrize("drift", ["dirty", "head", "parent", "subject", "time"])
def test_reconstructed_cli_routes_h3_identity_failures_to_specific_code(
    reconstructed_fixture: _ReconstructedFixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    action: str,
    drift: str,
) -> None:
    fixture = reconstructed_fixture
    request = fixture.request
    if drift == "dirty":
        (fixture.repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    elif drift == "head":
        request = replace(request, replacement_head="f" * 40)
    elif drift == "parent":
        monkeypatch.setattr(ap, "ROUND28_H2_HEAD", "f" * 40)
    elif drift == "subject":
        monkeypatch.setattr(ap, "ROUND28_H3_SUBJECT", "wrong subject")
    else:
        monkeypatch.setattr(
            ap, "ROUND28_H2_COMMITTED_AT_UTC", "2999-01-01T00:00:00Z",
        )

    rc = ap.main(_reconstructed_cli_args(request, action=action))

    assert rc == 1
    error = capsys.readouterr().err
    assert "[SOL_LEGACY_REPRO_HEAD_MISMATCH]" in error
    assert "[PREPARATION_ARCHIVE_SOURCE_MISSING]" not in error
    assert fixture.request.journals_root.is_dir()
    assert not fixture.request.preparation_failures_root.exists()


def test_reproduction_postchecks_have_distinct_time_argv_tool_and_head_codes(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    records = ap.build_reconstructed_context(fixture.request)
    record = records["legacy_comparator_reproduction.json"]
    argv_tail = record["command_argv"][1:]
    executed = ap._parse_utc(record["executed_at"], label="executed_at")

    bad = json.loads(json.dumps(record))
    bad["tool"]["head"] = "f" * 40
    _assert_refusal(
        "SOL_LEGACY_REPRO_HEAD_MISMATCH", ap._verify_reproduction_record,
        bad, h3=fixture.h3, expected_argv_tail=argv_tail,
        t_pre=executed, t_post=executed,
    )

    bad = json.loads(json.dumps(record))
    bad["command_argv"][10] = "f" * 40
    _assert_refusal(
        "SOL_LEGACY_REPRO_ARGV_MISMATCH", ap._verify_reproduction_record,
        bad, h3=fixture.h3, expected_argv_tail=argv_tail,
        t_pre=executed, t_post=executed,
    )

    too_late = executed + timedelta(seconds=10)
    _assert_refusal(
        "SOL_LEGACY_REPRO_TIME_INVALID", ap._verify_reproduction_record,
        record, h3=fixture.h3, expected_argv_tail=argv_tail,
        t_pre=too_late, t_post=too_late,
    )

    bad = json.loads(json.dumps(record))
    bad["tool"]["sha256"] = "f" * 64
    _assert_refusal(
        "SOL_LEGACY_REPRO_TOOL_DRIFT", ap._verify_reproduction_record,
        bad, h3=fixture.h3, expected_argv_tail=argv_tail,
        t_pre=executed, t_post=executed,
    )


def test_archive_only_requires_complete_records_and_uncorrupted_log(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    ap.build_reconstructed_context(fixture.request)
    record = fixture.request.reconstruction_root / "phase_reconstruction.json"
    record.unlink()
    _assert_refusal(
        "RLCR_SESSION_LOG_INCONSISTENT",
        ap.archive_preparation_failure, fixture.request,
    )
    assert fixture.request.journals_root.exists()


def test_corrupt_session_log_refuses_archive_before_move(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    ap.build_reconstructed_context(fixture.request)
    with fixture.request.session_log_path.open("ab") as stream:
        stream.write(b"{truncated\n")
    _assert_refusal(
        "RLCR_SESSION_LOG_INCONSISTENT",
        ap.archive_preparation_failure, fixture.request,
    )
    assert fixture.request.journals_root.exists()


def test_session_log_rejects_complete_final_record_without_lf_before_move(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    ap.build_reconstructed_context(fixture.request)
    path = fixture.request.session_log_path
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    expected_line = raw.count(b"\n")
    path.write_bytes(raw[:-1])

    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap._load_session_log(path)
    assert caught.value.code == "RLCR_SESSION_LOG_INCONSISTENT"
    assert caught.value.detail == f"{path}:{expected_line}: missing final LF"

    _assert_refusal(
        "RLCR_SESSION_LOG_INCONSISTENT",
        ap.archive_preparation_failure, fixture.request,
    )
    assert fixture.request.journals_root.exists()
    assert not fixture.request.preparation_failures_root.exists()


def test_reconstructed_archive_recursive_context_loader_and_embedded_closure(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    records = ap.build_reconstructed_context(fixture.request)
    entry = ap.archive_preparation_failure(fixture.request)

    archive = Path(entry["archive_path"])
    assert not fixture.request.journals_root.exists()
    reconstructed = archive / "chunk_context/reconstructed"
    assert (reconstructed / "scratch/source_definition.json").is_file()
    assert (reconstructed / "scratch/staged_definition.json").is_file()
    assert (reconstructed / "legacy_comparator_source_excerpt.py.txt").is_file()
    assert (reconstructed / "gpu_process_observation_late.raw.txt").is_file()
    assert entry["no_gpu_attestation"] == records[
        "preparation_failure_no_gpu_attestation_v2.json"
    ]
    assert entry["supersedes"] is None
    assert "prior_preparation_failures" not in entry
    assert len({*entry["authority"].values()} & {
        entry["authority"]["corrected_adjudication_sha256"],
        entry["authority"]["superseded_adjudication_sha256"],
        entry["authority"]["correction_adjudication_sha256"],
    }) >= 3

    ledger = (
        fixture.request.preparation_failures_root
        / ap.PREPARATION_FAILURE_LEDGER_NAME
    )
    assert ap.load_preparation_failure_ledger(
        ledger, verify_archives=True,
    ) == [entry]
    assert ap.preparation_failure_backrefs(
        [entry], cell_id=ap.ROUND28_CELL_ID, ledger_path=ledger,
    ) == [{
        "ledger_entry_id": entry["entry_id"],
        "ledger_path": str(ledger.resolve()),
        "archive_path": entry["archive_path"],
    }]


def test_reconstructed_archive_one_shot_and_consumer_tamper_closure(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    ap.build_reconstructed_context(fixture.request)
    entry = ap.archive_preparation_failure(fixture.request)
    ledger = (
        fixture.request.preparation_failures_root
        / ap.PREPARATION_FAILURE_LEDGER_NAME
    )
    archive = Path(entry["archive_path"])

    # The sealed archive event makes the reconstructed context one-shot.  A
    # second invocation is rejected before it can publish another ledger row.
    _assert_refusal(
        "RLCR_SESSION_LOG_INCONSISTENT",
        ap.archive_preparation_failure, fixture.request,
    )

    standalone = (
        archive / "chunk_context/reconstructed/"
        "preparation_failure_no_gpu_attestation_v2.json"
    )
    value = json.loads(standalone.read_text())
    value["attestation"] += " tampered"
    standalone.chmod(0o600)
    standalone.write_text(json.dumps(value, sort_keys=True) + "\n")
    _assert_refusal(
        "PREPARATION_LEDGER_INCONSISTENT",
        ap.load_preparation_failure_ledger, ledger, verify_archives=True,
    )


def test_reconstructed_schema_arm_rejects_authority_sha_equality(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    records = ap.build_reconstructed_context(fixture.request)
    attestation = records["preparation_failure_no_gpu_attestation_v2.json"]
    row = {
        **{field: None for field in ap.PREPARATION_FAILURE_ENTRY_FIELDS},
        "supersedes": None,
        "reconstruction_records": [
            {
                "path": f"chunk_context/reconstructed/{name}",
                "sha256": "a" * 64,
            }
            for name in ap.ROUND28_RECONSTRUCTION_RECORD_ORDER
        ],
        "authority": {
            "amendment_id": ap.ROUND28_AMENDMENT_ID,
            "supersession_status": "superseded_by_v2",
            "corrected_adjudication_sha256": "a" * 64,
            "superseded_adjudication_sha256": "a" * 64,
            "correction_id": ap.ROUND28_CORRECTION_ID,
            "correction_adjudication_sha256": "b" * 64,
        },
        "no_gpu_attestation": attestation,
    }
    _assert_refusal(
        "PREPARATION_LEDGER_INCONSISTENT",
        ap.validate_preparation_failure_entry,
        row, ledger_path=fixture.request.preparation_failures_root / "ledger.jsonl",
        line_number=1, verify_archive=False,
    )


def test_reconstructed_post_move_failure_state_is_structural_and_fail_closed(
    reconstructed_fixture: _ReconstructedFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = reconstructed_fixture
    ap.build_reconstructed_context(fixture.request)
    construction_before = _tree_snapshot(fixture.request.reconstruction_root)
    session_before = fixture.request.session_log_path.read_bytes()
    _install_post_move_inventory_drift(monkeypatch)

    _assert_refusal(
        "PREPARATION_ARCHIVE_INVENTORY_MISMATCH",
        ap.archive_preparation_failure, fixture.request,
    )
    assert not fixture.request.journals_root.exists()
    archives = [
        path for path in fixture.request.preparation_failures_root.iterdir()
        if path.is_dir()
    ]
    assert len(archives) == 1
    assert [path.name for path in archives[0].iterdir()] == [ap.ROUND28_CELL_ID]
    ledger = (
        fixture.request.preparation_failures_root
        / ap.PREPARATION_FAILURE_LEDGER_NAME
    )
    assert not ledger.exists()
    assert _tree_snapshot(fixture.request.reconstruction_root) == construction_before
    assert fixture.request.session_log_path.read_bytes() == session_before
    assert not any(
        "FAILURE" in path.name.upper()
        for path in fixture.request.preparation_failures_root.rglob("*")
    )

    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.load_preparation_failure_ledger(ledger, verify_archives=True)
    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"
    assert "archive evidence exists without ledger" in caught.value.detail

    preparation_before_retry = _tree_snapshot(
        fixture.request.preparation_failures_root
    )
    _assert_refusal(
        "PREPARATION_ARCHIVE_SOURCE_MISSING",
        ap.archive_preparation_failure, fixture.request,
    )
    assert _tree_snapshot(
        fixture.request.preparation_failures_root
    ) == preparation_before_retry
    assert fixture.request.session_log_path.read_bytes() == session_before


def test_corr2_artifact_is_verified_and_logged(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    hashes = ap._adjudication_hashes(fixture.request)
    correction2 = fixture.request.correction2_adjudication_ref
    assert hashes["correction2"] == ap.ROUND28_CORRECTION2_SHA256
    assert correction2.stat().st_mode & 0o777 == 0o644

    expected = ap._required_session_prefix(
        fixture.request, h3=fixture.h3, adjudications=hashes,
    )
    rows = ap._load_session_log(fixture.request.session_log_path)
    assert rows == expected
    assert rows[1] == {
        "event": "adjudication_saved",
        "artifact": "adjudication_r28_post_move_failure_state_correction.md",
        "sha256": ap.ROUND28_CORRECTION2_SHA256,
        "status": "binding_supplement",
        "supplements_amendment_id": ap.ROUND28_AMENDMENT_ID,
        "correction_id": ap.ROUND28_CORRECTION2_ID,
    }
    ap.build_reconstructed_context(fixture.request)


def test_repository_fixture_mode_is_not_authoritative(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    correction2 = fixture.request.correction2_adjudication_ref
    correction2.chmod(0o444)
    assert ap._adjudication_hashes(fixture.request)["correction2"] == _sha(
        correction2
    )


def test_dirty_repository_fixture_has_distinct_refusal(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    correction2 = fixture.request.correction2_adjudication_ref
    correction2.write_bytes(correction2.read_bytes() + b"drift\n")
    _assert_refusal(
        "PREP_ARCHIVE_EVIDENCE_DIRTY",
        ap._adjudication_hashes, fixture.request,
    )


def test_clean_wrong_hash_repository_fixture_has_distinct_refusal(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    correction2 = fixture.request.correction2_adjudication_ref
    correction2.write_bytes(correction2.read_bytes() + b"drift\n")
    _commit(fixture.repo, "plant clean hash mismatch")
    _assert_refusal(
        "PREP_ARCHIVE_EVIDENCE_HASH_MISMATCH",
        ap._adjudication_hashes, fixture.request,
    )


def test_wrong_alias_target_has_distinct_refusal(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    aliases = ap.ROUND28_PATH_ALIASES
    value = json.loads(aliases.read_text(encoding="utf-8"))
    value["aliases"][0]["canonical"] = "evidence/wrong/"
    aliases.write_text(json.dumps(value), encoding="utf-8")
    _commit(fixture.repo, "plant wrong alias")

    _assert_refusal(
        "PREP_ARCHIVE_EVIDENCE_ALIAS_MISMATCH",
        ap._adjudication_hashes, fixture.request,
    )


def test_corr2_session_log_entry_two_is_required(
    reconstructed_fixture: _ReconstructedFixture,
) -> None:
    fixture = reconstructed_fixture
    rows = ap._load_session_log(fixture.request.session_log_path)
    assert rows[1]["correction_id"] == ap.ROUND28_CORRECTION2_ID
    _jsonl(
        fixture.request.session_log_path,
        [rows[0], *rows[2:]],
    )

    _assert_refusal(
        "RLCR_SESSION_LOG_INCONSISTENT",
        ap.build_reconstructed_context, fixture.request,
    )


def test_reconstructed_row_without_archive_log_event_is_fs_post_inconsistent(
    reconstructed_fixture: _ReconstructedFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = reconstructed_fixture
    ap.build_reconstructed_context(fixture.request)
    prefix = fixture.request.session_log_path.read_bytes()
    original_append = ap._append_jsonl_fsync

    def fail_archive_log_append(path, payload):
        if (
            path == fixture.request.session_log_path
            and payload.get("event") == "preparation_failure_archived"
        ):
            raise ap.PreparationArchiveRefusal(
                "RLCR_SESSION_LOG_INCONSISTENT",
                "injected archive event append failure",
            )
        original_append(path, payload)

    monkeypatch.setattr(ap, "_append_jsonl_fsync", fail_archive_log_append)
    _assert_refusal(
        "RLCR_SESSION_LOG_INCONSISTENT",
        ap.archive_preparation_failure, fixture.request,
    )

    ledger = (
        fixture.request.preparation_failures_root
        / ap.PREPARATION_FAILURE_LEDGER_NAME
    )
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert isinstance(json.loads(lines[0]), dict)
    assert fixture.request.session_log_path.read_bytes() == prefix
    ledger_before = ledger.read_bytes()

    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.load_preparation_failure_ledger(ledger, verify_archives=True)
    assert caught.value.code == "RLCR_SESSION_LOG_INCONSISTENT"
    assert "archived prefix mismatch: rows=" in caught.value.detail
    assert ledger.read_bytes() == ledger_before
