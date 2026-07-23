from __future__ import annotations

import base64
import fcntl
import json
import os
import runpy
import shutil
import subprocess
import sys
import threading
import time
import types
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from tools.local_labels import archive_interrupted_sol_runnability as archive


def _git(repo: Path, *arguments: str) -> bytes:
    environment = archive._git_environment()
    return subprocess.run(
        [
            os.fspath(archive.GIT_BINARY),
            "--no-replace-objects",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=repo,
        env=environment,
        capture_output=True,
        check=True,
    ).stdout


def _write(path: Path, raw: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(mode)


def _harden_fixture_modes(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_dir():
            path.chmod(0o700)
        elif path.is_file():
            path.chmod(0o600)


def _partial_payload() -> bytes:
    statuses = ["runnable"] * 48 + ["correctness_failed"] + [
        "host_capacity_oom"
    ] * 4
    by_status = {
        status: statuses.count(status)
        for status in archive.PARTIAL_RUNNABILITY_STATUSES
    }
    cells = [
        {
            "slug": f"problem_{index:03d}",
            "category": "L1",
            "language": "cuda_cpp",
            "status": status,
        }
        for index, status in enumerate(statuses)
    ]
    return (json.dumps({
        "schema_version": 1,
        "artifact_id": "sm86_sol_runnability_v1",
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": "0" * 64,
        "cli_timeout_seconds": 1500,
        "data_annex_manifest_sha256": "1" * 64,
        "flashinfer_corpus_blob_count": 1,
        "flashinfer_corpus_error": None,
        "flashinfer_corpus_file_count": 1,
        "flashinfer_corpus_manifest_sha256": "2" * 64,
        "flashinfer_corpus_verified": True,
        "outer_timeout_seconds": 2000.0,
        "probe_config_path": "tools/local_labels/probe.json",
        "probe_config_sha256": "3" * 64,
        "probe_protocol_id": "sol_full_correctness_v1",
        "probe_protocol_sha256": "4" * 64,
        "resume_identity_sha256": "5" * 64,
        "sol_execbench_package_file_count": 1,
        "sol_execbench_package_sha256": "6" * 64,
        "in_scope_total": 201,
        "probed": len(cells),
        "pending": 201 - len(cells),
        "summary": {
            "total": len(cells),
            "runnable": by_status["runnable"],
            "by_status": by_status,
        },
        "status_to_typed_reason": dict(
            archive.PARTIAL_STATUS_TO_TYPED_REASON,
        ),
        "cells": cells,
    }, sort_keys=True) + "\n").encode()


def _schema_bytes() -> bytes:
    raw = (
        archive.REPO_ROOT / Path(*archive.SCHEMA_RELATIVE_PATH.parts)
    ).read_bytes()
    assert archive.sha256_bytes(raw) == archive.FIXED_CONTRACT.schema_sha256
    return raw


def _synthetic_canonical() -> bytes:
    return b'{"artifact_id":"synthetic_canonical_v1"}\n'


def _archive_fixture(
    tmp_path: Path,
) -> tuple[archive.FixedArchiveContract, Path, Path]:
    recorded_base = tmp_path / "recorded"
    namespace = recorded_base / archive.ARCHIVE_ROOT_NAME
    payload = _partial_payload()
    source_sha256 = archive.sha256_bytes(payload)
    source_commit = "b" * 40
    attempt_id = archive.canonical_sha256({
        "source_commit": source_commit,
        "source_repository_path": archive.SOURCE_RELATIVE_PATH.as_posix(),
        "source_sha256": source_sha256,
    })
    attempt = namespace / f"attempt-{attempt_id}"
    payload_root = attempt / archive.PAYLOAD_DIRECTORY
    status_raw = (
        b" M " + os.fsencode(archive.SOURCE_RELATIVE_PATH.as_posix()) + b"\0"
    )
    created = "2026-07-18T07:02:43.933076Z"
    canonical = _synthetic_canonical()
    manifest_body = {
        "schema_version": 1,
        "artifact_id": archive.ARTIFACT_ID,
        "attempt_id": attempt_id,
        "created_at_utc": created,
        "source": {
            "repository_path": archive.SOURCE_RELATIVE_PATH.as_posix(),
            "absolute_path": os.fspath(
                archive.REPO_ROOT / Path(*archive.SOURCE_RELATIVE_PATH.parts)
            ),
            "commit": source_commit,
            "size_bytes": len(payload),
            "sha256": source_sha256,
            "git_status_porcelain_v1_z_base64": base64.b64encode(
                status_raw
            ).decode("ascii"),
            "git_status_sha256": archive.sha256_bytes(status_raw),
            "git_worktree_diff_sha256": "c" * 64,
            "git_index_diff_sha256": archive.EMPTY_SHA256,
        },
        "canonical": {
            "repository_path": archive.SOURCE_RELATIVE_PATH.as_posix(),
            "commit": source_commit,
            "size_bytes": len(canonical),
            "sha256": archive.sha256_bytes(canonical),
        },
        "payload": {
            "archive_relative_path": (
                f"{archive.PAYLOAD_DIRECTORY}/{archive.PAYLOAD_NAME}"
            ),
            "size_bytes": len(payload),
            "sha256": source_sha256,
        },
        "disposition": dict(archive.DISPOSITION),
    }
    manifest = {
        **manifest_body,
        "manifest_seal_sha256": archive.canonical_sha256(manifest_body),
    }
    manifest_raw = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest_sha256 = archive.sha256_bytes(manifest_raw)
    recorded_attempt = namespace / f"attempt-{attempt_id}"
    ledger = {
        "schema_version": 1,
        "entry_type": archive.ENTRY_TYPE,
        "attempt_id": attempt_id,
        "created_at_utc": created,
        "archive_path": os.fspath(recorded_attempt),
        "manifest_path": os.fspath(recorded_attempt / archive.MANIFEST_NAME),
        "manifest_sha256": manifest_sha256,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        **archive.DISPOSITION,
    }
    ledger_raw = (
        json.dumps(ledger, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    contract = archive.FixedArchiveContract(
        recorded_archive_base=recorded_base,
        recorded_source_absolute_path=(
            archive.REPO_ROOT / Path(*archive.SOURCE_RELATIVE_PATH.parts)
        ),
        source_relative_path=archive.SOURCE_RELATIVE_PATH,
        schema_relative_path=archive.SCHEMA_RELATIVE_PATH,
        source_commit=source_commit,
        source_sha256=source_sha256,
        source_size_bytes=len(payload),
        canonical_sha256=archive.sha256_bytes(canonical),
        canonical_size_bytes=len(canonical),
        canonical_tree_mode="100644",
        canonical_blob_oid=archive._git_blob_oid(canonical),
        schema_sha256=archive.sha256_bytes(_schema_bytes()),
        attempt_id=attempt_id,
        manifest_sha256=manifest_sha256,
        manifest_size_bytes=len(manifest_raw),
        ledger_sha256=archive.sha256_bytes(ledger_raw),
        ledger_size_bytes=len(ledger_raw),
        created_at_utc=created,
    )
    recorded_base.mkdir(mode=0o700)
    namespace.mkdir(mode=0o700)
    attempt.mkdir(mode=0o700)
    payload_root.mkdir(mode=0o700)
    _write(namespace / archive.LOCK_NAME, b"", mode=0o644)
    _write(namespace / archive.LEDGER_NAME, ledger_raw)
    _write(attempt / archive.MANIFEST_NAME, manifest_raw)
    _write(payload_root / archive.PAYLOAD_NAME, payload)
    return contract, recorded_base, attempt


def _verify_archive_fixture(
    base: Path, contract: archive.FixedArchiveContract,
) -> dict:
    return archive._verify_archive_contract(
        base,
        contract=contract,
        schema_raw=_schema_bytes(),
        canonical_raw=_synthetic_canonical(),
    )


def _reseal_manifest(manifest: dict) -> None:
    body = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_seal_sha256"
    }
    manifest["manifest_seal_sha256"] = archive.canonical_sha256(body)


def _canonical_document() -> bytes:
    raw = (
        archive.REPO_ROOT / Path(*archive.SOURCE_RELATIVE_PATH.parts)
    ).read_bytes()
    assert archive.sha256_bytes(raw) == archive.FIXED_CONTRACT.canonical_sha256
    assert len(raw) == archive.FIXED_CONTRACT.canonical_size_bytes
    assert archive._git_blob_oid(raw) == archive.FIXED_CONTRACT.canonical_blob_oid
    return raw


def _restore_fixture(
    tmp_path: Path,
) -> tuple[archive.FixedArchiveContract, Path, Path]:
    repo = tmp_path / "repo"
    source = repo / Path(*archive.SOURCE_RELATIVE_PATH.parts)
    schema = repo / Path(*archive.SCHEMA_RELATIVE_PATH.parts)
    source.parent.mkdir(parents=True, mode=0o700)
    schema.parent.mkdir(parents=True, mode=0o700)
    for directory in {
        repo,
        *source.parents[:len(source.parents) - len(repo.parents)],
        *schema.parents[:len(schema.parents) - len(repo.parents)],
    }:
        if directory == repo or repo in directory.parents:
            directory.chmod(0o700)
    canonical = _canonical_document()
    source.write_bytes(canonical)
    source.chmod(0o600)
    schema_raw = (
        archive.REPO_ROOT / Path(*archive.SCHEMA_RELATIVE_PATH.parts)
    ).read_bytes()
    schema.write_bytes(schema_raw)
    schema.chmod(0o600)
    _git(repo, "init", "--object-format=sha1")
    _git(repo, "config", "user.name", "Archive Test")
    _git(repo, "config", "user.email", "archive@example.test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "canonical")
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    tree_fields = _git(
        repo, "ls-tree", head, "--", archive.SOURCE_RELATIVE_PATH.as_posix(),
    ).decode().split()
    assert tree_fields[:2] == ["100644", "blob"]
    assert tree_fields[2] == archive.FIXED_CONTRACT.canonical_blob_oid
    contract = replace(
        archive.FIXED_CONTRACT,
        recorded_source_absolute_path=source,
        source_commit=head,
    )
    _harden_fixture_modes(repo)
    return contract, repo, source


def test_restore_fixture_is_safe_under_permissive_umask(tmp_path: Path):
    previous_umask = os.umask(0)
    try:
        contract, repo, _source = _restore_fixture(tmp_path)
    finally:
        os.umask(previous_umask)

    assert archive._check_canonical_contract(
        repo, contract=contract,
    )["restored"]


def test_fixture_git_ignores_ambient_signing_and_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    home = tmp_path / "ambient-home"
    hooks = home / "hooks"
    hooks.mkdir(parents=True, mode=0o700)
    pre_commit = hooks / "pre-commit"
    pre_commit.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    pre_commit.chmod(0o700)
    (home / ".gitconfig").write_text(
        "[commit]\n\tgpgsign = true\n"
        f"[core]\n\thooksPath = {hooks}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", os.fspath(home))
    trace = tmp_path / "ambient-git-trace.jsonl"
    monkeypatch.setenv("GIT_DEFAULT_HASH", "sha256")
    monkeypatch.setenv("GIT_TRACE2_EVENT", os.fspath(trace))

    contract, repo, _source = _restore_fixture(tmp_path)

    assert archive._check_canonical_contract(
        repo, contract=contract,
    )["restored"]
    assert not trace.exists()


def test_restore_git_ignores_path_shadow_and_trace_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    marker = tmp_path / "shadow-git-ran"
    shadow_root = tmp_path / "shadow-bin"
    shadow_root.mkdir(mode=0o700)
    shadow_git = shadow_root / "git"
    shadow_git.write_text(
        "#!/bin/sh\nprintf ran > \"$SHADOW_GIT_MARKER\"\nexit 99\n",
        encoding="utf-8",
    )
    shadow_git.chmod(0o700)
    trace = tmp_path / "production-git-trace.jsonl"
    monkeypatch.setenv("PATH", os.fspath(shadow_root))
    monkeypatch.setenv("SHADOW_GIT_MARKER", os.fspath(marker))
    monkeypatch.setenv("GIT_TRACE2_EVENT", os.fspath(trace))

    result = archive._check_canonical_contract(repo, contract=contract)

    assert result["restored"]
    assert not marker.exists()
    assert not trace.exists()


def test_git_invocation_disables_lazy_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    observed = {}
    monkeypatch.setenv("LD_PRELOAD", "/tmp/forbidden-loader.so")
    monkeypatch.setenv("LD_AUDIT", "/tmp/forbidden-audit.so")
    monkeypatch.setenv("GIT_TRACE2_EVENT", "/tmp/forbidden-trace")
    monkeypatch.setenv("PATH", "/tmp/forbidden-path")

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(archive.subprocess, "run", fake_run)
    archive._git(tmp_path, ["version"])

    assert "--no-lazy-fetch" in observed["command"]
    assert "core.multiPackIndex=false" in observed["command"]
    assert observed["environment"]["GIT_NO_LAZY_FETCH"] == "1"
    assert observed["environment"]["PATH"] == "/usr/bin:/bin"
    assert set(observed["environment"]) == {
        "LANG",
        "LC_ALL",
        "PATH",
        "TMPDIR",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_GRAFT_FILE",
        "GIT_NO_LAZY_FETCH",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OPTIONAL_LOCKS",
        "GIT_TERMINAL_PROMPT",
        "GIT_ASKPASS",
    }


def test_git_blob_refuses_oversize_before_materializing_content(
    tmp_path: Path,
):
    repo = tmp_path / "blob-repo"
    repo.mkdir(mode=0o700)
    _git(repo, "init", "--object-format=sha1")
    blob = repo / "oversized.bin"
    _write(blob, b"x" * 32)
    object_id = _git(repo, "hash-object", "-w", blob.name).decode().strip()

    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="exceeds maximum size",
    ):
        archive._git_blob(repo, object_id, max_size=8)


def test_git_blob_timeout_covers_payload_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    script = (
        "import sys,time; "
        "oid=sys.stdin.buffer.readline().strip(); "
        "sys.stdout.buffer.write(oid+b' blob 1\\n'); "
        "sys.stdout.buffer.flush(); time.sleep(5)"
    )
    monkeypatch.setattr(
        archive, "_git_command", lambda _arguments: [sys.executable, "-c", script],
    )
    monkeypatch.setattr(archive, "GIT_COMMAND_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="Git blob.*timed out",
    ):
        archive._git_blob(tmp_path, "a" * 40, max_size=8)


def test_git_pipe_reader_supports_descriptors_above_fd_setsize():
    descriptors: list[int] = []
    read_descriptor: int | None = None
    write_descriptor: int | None = None
    try:
        while True:
            read_descriptor, write_descriptor = os.pipe()
            descriptors.extend((read_descriptor, write_descriptor))
            if read_descriptor >= 1024:
                break
            if len(descriptors) > 2200:
                pytest.skip("process descriptor limit does not admit a high-fd probe")
        os.write(write_descriptor, b"x")
        assert archive._read_git_pipe_exact(
            read_descriptor,
            1,
            deadline=time.monotonic() + 1,
            role="high-fd fixture",
            refusal_code="TEST_REFUSAL",
        ) == b"x"
    except OSError as exc:
        pytest.skip(f"process descriptor limit does not admit a high-fd probe: {exc}")
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_zero_credit_disposition_is_immutable():
    with pytest.raises(TypeError):
        archive.DISPOSITION["authoritative"] = True  # type: ignore[index]
    assert dict(archive.DISPOSITION) == {
        "authoritative": False,
        "resumable": False,
        "reusable": False,
        "campaign_rows": 0,
    }


def test_fixed_archive_verifies_exact_bytes_and_zero_credit(tmp_path: Path):
    contract, base, _attempt = _archive_fixture(tmp_path)
    manifest = _verify_archive_fixture(base, contract)
    assert manifest["attempt_id"] == contract.attempt_id
    assert manifest["disposition"] == archive.DISPOSITION


def test_synthetic_archive_fixture_does_not_open_forensic_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, base, _attempt = _archive_fixture(tmp_path)

    def forbidden_schema_lookup(_contract):
        raise AssertionError("synthetic fixture opened forensic repository")

    monkeypatch.setattr(archive, "_read_contract_schema", forbidden_schema_lookup)
    assert _verify_archive_fixture(base, contract)["attempt_id"] == contract.attempt_id


def test_partial_payload_must_satisfy_hash_pinned_schema():
    document = json.loads(_partial_payload())
    del document["cells"][0]["slug"]

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="schema projection is invalid",
    ):
        archive._validate_partial_payload(
            json.dumps(document).encode(), _schema_bytes(),
        )


def test_shadow_jsonschema_module_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    shadow = tmp_path / "jsonschema.py"
    shadow.write_text("# no-op shadow validator\n", encoding="utf-8")
    fake_module = types.SimpleNamespace(
        __file__=os.fspath(shadow),
        __spec__=types.SimpleNamespace(origin=os.fspath(shadow)),
        Draft202012Validator=object,
    )
    real_import = archive.importlib.import_module

    def shadowing_import(name: str):
        if name == "jsonschema":
            return fake_module
        return real_import(name)

    monkeypatch.setattr(archive.importlib, "import_module", shadowing_import)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="jsonschema was imported outside the pinned distribution",
    ):
        archive._validate_partial_payload(_partial_payload(), _schema_bytes())


def test_pythonpath_jsonschema_shadow_is_refused_before_import(tmp_path: Path):
    shadow_root = tmp_path / "shadow"
    shadow_root.mkdir(mode=0o700)
    marker = tmp_path / "shadow-imported"
    (shadow_root / "jsonschema.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['SHADOW_MARKER']).write_text('imported')\n"
        "class Draft202012Validator:\n"
        "    @classmethod\n"
        "    def check_schema(cls, schema): pass\n"
        "    def __init__(self, schema): pass\n"
        "    def validate(self, document): pass\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        os.fspath(shadow_root), os.fspath(archive.REPO_ROOT),
    ))
    environment["SHADOW_MARKER"] = os.fspath(marker)
    script = (
        "from tools.local_labels import "
        "archive_interrupted_sol_runnability as a\n"
        "try:\n"
        "    a._trusted_jsonschema_validator(refusal_code='TEST')\n"
        "except a.RunnabilityArchiveRefusal as exc:\n"
        "    assert 'import resolution is outside' in exc.detail\n"
        "else:\n"
        "    raise AssertionError('shadow jsonschema was trusted')\n"
    )

    subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        cwd=archive.REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    assert not marker.exists()


def test_pythonpath_transitive_shadow_cannot_preload_jsonschema_internal(
    tmp_path: Path,
):
    shadow_root = tmp_path / "shadow"
    shadow_root.mkdir(mode=0o700)
    marker = tmp_path / "transitive-shadow-imported"
    (shadow_root / "idna.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "import sys\n"
        "import types\n"
        "Path(os.environ['SHADOW_MARKER']).write_text('imported')\n"
        "fake = types.ModuleType('jsonschema._keywords')\n"
        "fake.__file__ = __file__\n"
        "fake.__spec__ = types.SimpleNamespace(origin=__file__)\n"
        "sys.modules['jsonschema._keywords'] = fake\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        os.fspath(shadow_root), os.fspath(archive.REPO_ROOT),
    ))
    environment["SHADOW_MARKER"] = os.fspath(marker)
    script = (
        "from tools.local_labels import "
        "archive_interrupted_sol_runnability as a\n"
        "validator, _errors = "
        "a._trusted_jsonschema_validator(refusal_code='TEST')\n"
        "assert validator.__module__ == 'jsonschema.validators'\n"
    )

    subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        cwd=archive.REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    assert not marker.exists()


def test_cohort_binding_rejects_unhashable_status_with_typed_refusal():
    partial = json.loads(_partial_payload())
    partial["cells"][0]["status"] = []

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="unknown status",
    ):
        archive._validate_cohort_identity_binding(
            json.dumps(partial).encode(),
            _canonical_document(),
        )


def test_partial_payload_summary_is_derived_from_cells():
    document = json.loads(_partial_payload())
    document["summary"]["by_status"]["runnable"] -= 1

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="derived observation drift",
    ):
        archive._validate_partial_payload(
            json.dumps(document).encode(), _schema_bytes(),
        )


@pytest.mark.parametrize("invalid_count", [False, 0.0])
def test_partial_payload_summary_counts_have_exact_integer_types(invalid_count):
    document = json.loads(_partial_payload())
    document["summary"]["by_status"]["compile_failed"] = invalid_count

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="derived observation drift",
    ):
        archive._validate_partial_payload(
            json.dumps(document).encode(), _schema_bytes(),
        )


def test_canonical_summary_counts_have_exact_integer_types():
    document = json.loads(_canonical_document())
    document["summary"]["by_status"]["eval_incomplete"] = 0.0

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="complete derived historical v1 authority",
    ):
        archive._validate_canonical_document(
            json.dumps(document).encode(), _schema_bytes(),
        )


def test_partial_payload_cell_identities_are_unique():
    document = json.loads(_partial_payload())
    document["cells"][1].update({
        key: document["cells"][0][key]
        for key in ("slug", "category", "language")
    })

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="duplicate cell identity",
    ):
        archive._validate_partial_payload(
            json.dumps(document).encode(), _schema_bytes(),
        )


def test_archive_verification_binds_the_exact_schema_hash(tmp_path: Path):
    contract, base, _attempt = _archive_fixture(tmp_path)
    contract = replace(contract, schema_sha256="0" * 64)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="schema byte hash mismatch",
    ):
        _verify_archive_fixture(base, contract)


def test_schema_reads_rederive_tree_declared_blob_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    schema_relative = archive.SCHEMA_RELATIVE_PATH.as_posix()
    expected_schema = _schema_bytes()
    substituted_oid = archive._git_blob_oid(b"tree-declared schema substitute\n")
    assert substituted_oid != archive._git_blob_oid(expected_schema)
    original_tree_entry = archive._tree_entry
    original_git_blob = archive._git_blob

    def substituted_tree_entry(
        repo_root: Path, revision: str, relative_path: str, **kwargs,
    ) -> archive.GitTreeEntry:
        entry = original_tree_entry(
            repo_root, revision, relative_path, **kwargs,
        )
        if relative_path == schema_relative:
            return archive.GitTreeEntry(
                entry.mode, entry.object_type, substituted_oid,
            )
        return entry

    def substituted_git_blob(
        repo_root: Path, object_id: str, **kwargs,
    ) -> bytes:
        if object_id == substituted_oid:
            return expected_schema
        return original_git_blob(repo_root, object_id, **kwargs)

    monkeypatch.setattr(archive, "_tree_entry", substituted_tree_entry)
    monkeypatch.setattr(archive, "_git_blob", substituted_git_blob)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="schema byte hash or blob identity mismatch",
    ):
        archive._read_contract_schema(contract)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="schema blob identity drift",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_private_canonical_fixture_rederives_git_blob_oid(tmp_path: Path):
    contract, base, _attempt = _archive_fixture(tmp_path)
    contract = replace(contract, canonical_blob_oid="0" * 40)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="private canonical fixture identity mismatch",
    ):
        _verify_archive_fixture(base, contract)


def test_manifest_status_requires_a_complete_exact_source_record(tmp_path: Path):
    contract, _base, attempt = _archive_fixture(tmp_path)
    manifest = json.loads((attempt / archive.MANIFEST_NAME).read_text())
    crafted_status = (
        b"?? M " + os.fsencode(archive.SOURCE_RELATIVE_PATH.as_posix()) + b"\0"
    )
    manifest["source"]["git_status_porcelain_v1_z_base64"] = (
        base64.b64encode(crafted_status).decode("ascii")
    )
    manifest["source"]["git_status_sha256"] = archive.sha256_bytes(
        crafted_status,
    )
    body = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_seal_sha256"
    }
    manifest["manifest_seal_sha256"] = archive.canonical_sha256(body)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="source path is not exactly one ordinary worktree-only modification",
    ):
        archive._validate_manifest(manifest, contract=contract)


def test_manifest_status_rejects_source_as_rename_continuation(tmp_path: Path):
    contract, _base, attempt = _archive_fixture(tmp_path)
    manifest = json.loads((attempt / archive.MANIFEST_NAME).read_text())
    expected_path = os.fsencode(archive.SOURCE_RELATIVE_PATH.as_posix())
    crafted_status = (
        b" M " + expected_path + b"\0"
        + b"R  renamed-destination\0" + expected_path + b"\0"
    )
    manifest["source"]["git_status_porcelain_v1_z_base64"] = (
        base64.b64encode(crafted_status).decode("ascii")
    )
    manifest["source"]["git_status_sha256"] = archive.sha256_bytes(
        crafted_status,
    )
    _reseal_manifest(manifest)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="source path is not exactly one ordinary worktree-only modification",
    ):
        archive._validate_manifest(manifest, contract=contract)


def test_manifest_status_rejects_impossible_xy_combination(tmp_path: Path):
    contract, _base, attempt = _archive_fixture(tmp_path)
    manifest = json.loads((attempt / archive.MANIFEST_NAME).read_text())
    expected_path = os.fsencode(archive.SOURCE_RELATIVE_PATH.as_posix())
    crafted_status = b" M " + expected_path + b"\0   unrelated.txt\0"
    manifest["source"]["git_status_porcelain_v1_z_base64"] = (
        base64.b64encode(crafted_status).decode("ascii")
    )
    manifest["source"]["git_status_sha256"] = archive.sha256_bytes(
        crafted_status,
    )
    _reseal_manifest(manifest)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="invalid status/path",
    ):
        archive._validate_manifest(manifest, contract=contract)


def test_empty_index_digest_rejects_unrelated_staged_record(tmp_path: Path):
    contract, _base, attempt = _archive_fixture(tmp_path)
    manifest = json.loads((attempt / archive.MANIFEST_NAME).read_text())
    expected_path = os.fsencode(archive.SOURCE_RELATIVE_PATH.as_posix())
    crafted_status = b" M " + expected_path + b"\0A  unrelated.txt\0"
    manifest["source"]["git_status_porcelain_v1_z_base64"] = (
        base64.b64encode(crafted_status).decode("ascii")
    )
    manifest["source"]["git_status_sha256"] = archive.sha256_bytes(
        crafted_status,
    )
    _reseal_manifest(manifest)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="status records contradict the empty index-diff digest",
    ):
        archive._validate_manifest(manifest, contract=contract)


def test_manifest_size_fields_require_exact_integers(tmp_path: Path):
    contract, _base, attempt = _archive_fixture(tmp_path)
    manifest = json.loads((attempt / archive.MANIFEST_NAME).read_text())
    manifest["source"]["size_bytes"] = float(contract.source_size_bytes)
    _reseal_manifest(manifest)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="manifest source identity drift",
    ):
        archive._validate_manifest(manifest, contract=contract)


def test_manifest_modified_source_requires_nonempty_diff_digest(tmp_path: Path):
    contract, _base, attempt = _archive_fixture(tmp_path)
    manifest = json.loads((attempt / archive.MANIFEST_NAME).read_text())
    manifest["source"]["git_worktree_diff_sha256"] = archive.EMPTY_SHA256
    _reseal_manifest(manifest)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="modified source cannot have an empty worktree diff digest",
    ):
        archive._validate_manifest(manifest, contract=contract)


def test_manifest_timestamp_is_parsed_as_real_utc(tmp_path: Path):
    contract, _base, attempt = _archive_fixture(tmp_path)
    manifest = json.loads((attempt / archive.MANIFEST_NAME).read_text())
    invalid_timestamp = "2026-13-18T07:02:43Z"
    manifest["created_at_utc"] = invalid_timestamp
    _reseal_manifest(manifest)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="created_at_utc mismatch",
    ):
        archive._validate_manifest(
            manifest,
            contract=replace(contract, created_at_utc=invalid_timestamp),
        )


def test_attempt_id_is_rederived_from_source_identity(tmp_path: Path):
    contract, _base, attempt = _archive_fixture(tmp_path)
    manifest = json.loads((attempt / archive.MANIFEST_NAME).read_text())

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="attempt ID does not rederive from source identity",
    ):
        archive._validate_manifest(
            manifest, contract=replace(contract, attempt_id="0" * 64),
        )


def test_byte_identical_relocated_archive_requires_alias_authority(tmp_path: Path):
    contract, base, _attempt = _archive_fixture(tmp_path)
    replica_base = tmp_path / "replica"
    replica_base.mkdir(mode=0o700)
    shutil.copytree(
        base / archive.ARCHIVE_ROOT_NAME,
        replica_base / archive.ARCHIVE_ROOT_NAME,
        copy_function=shutil.copy2,
    )
    replica = replica_base / archive.ARCHIVE_ROOT_NAME
    for directory in [replica, *[path for path in replica.rglob("*") if path.is_dir()]]:
        directory.chmod(0o700)
    for path in [path for path in replica.rglob("*") if path.is_file()]:
        path.chmod(0o600)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="RUNNABILITY_ARCHIVE_PATH_ALIAS_REQUIRED",
    ):
        archive._verify_archive_contract(
            replica_base,
            contract=contract,
            schema_raw=_schema_bytes(),
            canonical_raw=_synthetic_canonical(),
        )


def test_payload_tamper_fails_closed(tmp_path: Path):
    contract, base, attempt = _archive_fixture(tmp_path)
    payload = attempt / archive.PAYLOAD_DIRECTORY / archive.PAYLOAD_NAME
    payload.write_bytes(b"x" * contract.source_size_bytes)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="payload byte hash mismatch",
    ):
        _verify_archive_fixture(base, contract)


def test_ledger_byte_reencoding_fails_even_when_json_is_equivalent(tmp_path: Path):
    contract, base, _attempt = _archive_fixture(tmp_path)
    ledger = base / archive.ARCHIVE_ROOT_NAME / archive.LEDGER_NAME
    row = json.loads(ledger.read_text())
    ledger.write_text(json.dumps(row, indent=2, sort_keys=False) + "\n")
    ledger.chmod(0o600)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="ledger size mismatch",
    ):
        _verify_archive_fixture(base, contract)


def test_hardlinked_or_world_writable_evidence_fails_closed(tmp_path: Path):
    contract, base, attempt = _archive_fixture(tmp_path)
    manifest = attempt / archive.MANIFEST_NAME
    outside = tmp_path / "manifest-link"
    os.link(manifest, outside)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="link count is unsafe",
    ):
        _verify_archive_fixture(base, contract)
    outside.unlink()
    manifest.chmod(0o622)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="ownership/mode/link count is unsafe",
    ):
        _verify_archive_fixture(base, contract)


@pytest.mark.parametrize(
    ("target_role", "drift_mode"),
    [
        ("namespace", 0o755),
        ("attempt", 0o755),
        ("payload_root", 0o755),
        ("lock", 0o600),
        ("ledger", 0o644),
        ("manifest", 0o644),
        ("payload", 0o644),
    ],
)
def test_archive_authority_modes_are_exact(
    tmp_path: Path, target_role: str, drift_mode: int,
):
    contract, base, attempt = _archive_fixture(tmp_path)
    namespace = base / archive.ARCHIVE_ROOT_NAME
    paths = {
        "namespace": namespace,
        "attempt": attempt,
        "payload_root": attempt / archive.PAYLOAD_DIRECTORY,
        "lock": namespace / archive.LOCK_NAME,
        "ledger": namespace / archive.LEDGER_NAME,
        "manifest": attempt / archive.MANIFEST_NAME,
        "payload": (
            attempt / archive.PAYLOAD_DIRECTORY / archive.PAYLOAD_NAME
        ),
    }
    paths[target_role].chmod(drift_mode)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="mode drift",
    ):
        _verify_archive_fixture(base, contract)


def test_symlinked_archive_component_fails_closed(tmp_path: Path):
    contract, base, attempt = _archive_fixture(tmp_path)
    payload = attempt / archive.PAYLOAD_DIRECTORY / archive.PAYLOAD_NAME
    raw = payload.read_bytes()
    payload.unlink()
    outside = tmp_path / "outside-payload"
    outside.write_bytes(raw)
    payload.symlink_to(outside)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="is not regular",
    ):
        _verify_archive_fixture(base, contract)


def test_canonical_restore_check_accepts_clean_pinned_v1(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    result = archive._check_canonical_contract(repo, contract=contract)
    assert result["restored"] is True
    assert result["canonical_sha256"] == contract.canonical_sha256


def test_canonical_contract_requires_full_direct_commit_oid(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="source commit identity does not resolve to itself as a commit",
    ):
        archive._check_canonical_contract(
            repo, contract=replace(contract, source_commit="HEAD"),
        )


def test_canonical_contract_rejects_annotated_tag_source(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    _git(repo, "tag", "-a", "archive-source", "-m", "archive source tag")
    tag_oid = _git(
        repo, "rev-parse", "refs/tags/archive-source^{tag}",
    ).decode().strip()

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="source commit identity does not resolve to itself as a commit",
    ):
        archive._check_canonical_contract(
            repo, contract=replace(contract, source_commit=tag_oid),
        )


def test_canonical_contract_requires_regular_schema_mode(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    schema = repo / Path(*archive.SCHEMA_RELATIVE_PATH.parts)
    schema.chmod(0o700)
    _git(repo, "add", archive.SCHEMA_RELATIVE_PATH.as_posix())
    _git(repo, "commit", "-m", "schema mode drift")
    head = _git(repo, "rev-parse", "HEAD").decode().strip()

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="historical schema tree entry is not a regular blob",
    ):
        archive._check_canonical_contract(
            repo, contract=replace(contract, source_commit=head),
        )


def test_head_reference_must_directly_name_commit(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    _git(repo, "tag", "-a", "head-tag", "-m", "head tag")
    tag_oid = _git(
        repo, "rev-parse", "refs/tags/head-tag^{tag}",
    ).decode().strip()
    branch = _git(repo, "symbolic-ref", "HEAD").decode().strip()
    branch_path = repo / ".git" / Path(*PurePosixPath(branch).parts)
    branch_path.write_text(f"{tag_oid}\n", encoding="ascii")
    branch_path.chmod(0o600)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="HEAD does not directly name a commit object",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_canonical_restore_check_enforces_historical_schema(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    schema = repo / Path(*archive.SCHEMA_RELATIVE_PATH.parts)
    schema_raw = json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["impossible_schema_sentinel"],
    }).encode()
    schema.write_bytes(schema_raw)
    schema.chmod(0o600)
    _git(repo, "add", archive.SCHEMA_RELATIVE_PATH.as_posix())
    _git(repo, "commit", "-m", "restrict schema")
    contract = replace(
        contract,
        source_commit=_git(repo, "rev-parse", "HEAD").decode().strip(),
        schema_sha256=archive.sha256_bytes(schema_raw),
    )

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="impossible_schema_sentinel",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_archive_schema_is_read_from_source_commit_not_worktree(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    historical = archive._read_contract_schema(contract)
    live_schema = repo / Path(*archive.SCHEMA_RELATIVE_PATH.parts)
    live_schema.write_text('{"changed": true}\n', encoding="utf-8")
    live_schema.chmod(0o600)

    assert archive._read_contract_schema(contract) == historical
    assert archive.sha256_bytes(historical) == contract.schema_sha256


def test_canonical_restore_check_rejects_worktree_drift(tmp_path: Path):
    contract, repo, source = _restore_fixture(tmp_path)
    original = source.read_bytes()
    source.write_bytes(b"x" * len(original))
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="working-tree runnability bytes are not canonical v1",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_canonical_restore_rejects_special_mode_bits(tmp_path: Path):
    contract, repo, source = _restore_fixture(tmp_path)
    source.chmod(0o1600)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="ownership/mode/link count is unsafe",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_canonical_restore_check_rejects_staged_partial_bytes(tmp_path: Path):
    contract, repo, source = _restore_fixture(tmp_path)
    canonical = source.read_bytes()
    source.write_bytes(b"interrupted bytes\n")
    _git(repo, "add", archive.SOURCE_RELATIVE_PATH.as_posix())
    source.write_bytes(canonical)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="canonical path remains dirty or staged",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_git_index_override_cannot_hide_real_staged_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, source = _restore_fixture(tmp_path)
    alternate_index = tmp_path / "alternate-index"
    shutil.copy2(repo / ".git" / "index", alternate_index)
    canonical = source.read_bytes()
    source.write_bytes(b"interrupted bytes\n")
    _git(repo, "add", archive.SOURCE_RELATIVE_PATH.as_posix())
    source.write_bytes(canonical)
    monkeypatch.setenv("GIT_INDEX_FILE", os.fspath(alternate_index))
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="canonical path remains dirty or staged",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_fixed_forensic_source_path_survives_module_relocation(tmp_path: Path):
    relocated = (
        tmp_path / "relocated" / "tools" / "local_labels"
        / Path(archive.__file__).name
    )
    relocated.parent.mkdir(parents=True, mode=0o700)
    shutil.copy2(Path(archive.__file__), relocated)

    namespace = runpy.run_path(os.fspath(relocated))

    assert namespace["REPO_ROOT"] == tmp_path / "relocated"
    assert (
        namespace["FIXED_CONTRACT"].recorded_source_absolute_path
        == Path(
            "/home/jakeshea/GPU_simulator_rebase/"
            "data/local_labels/campaign/sol_runnability_sm86.json"
        )
    )


def test_committed_executable_mode_drift_is_rejected(tmp_path: Path):
    contract, repo, source = _restore_fixture(tmp_path)
    source.chmod(0o700)
    _git(repo, "add", archive.SOURCE_RELATIVE_PATH.as_posix())
    _git(repo, "commit", "-m", "mode drift")

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="current pinned HEAD tree entry is not canonical v1",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_replace_ref_cannot_mask_noncanonical_real_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, source = _restore_fixture(tmp_path)
    canonical = source.read_bytes()
    source.write_bytes(b"committed interrupted bytes\n")
    source.chmod(0o600)
    _git(repo, "add", archive.SOURCE_RELATIVE_PATH.as_posix())
    _git(repo, "commit", "-m", "noncanonical real head")
    real_head = _git(repo, "rev-parse", "HEAD").decode().strip()
    canonical_tree = _git(
        repo, "rev-parse", f"{contract.source_commit}^{{tree}}",
    ).decode().strip()
    environment = archive._git_environment()
    replacement = subprocess.run(
        [os.fspath(archive.GIT_BINARY), "commit-tree", canonical_tree, "-p", contract.source_commit],
        cwd=repo,
        env=environment,
        input=b"synthetic canonical replacement\n",
        capture_output=True,
        check=True,
    ).stdout.decode().strip()
    _git(repo, "replace", real_head, replacement)
    source.write_bytes(canonical)
    source.chmod(0o600)
    _git(
        repo, "reset", contract.source_commit, "--",
        archive.SOURCE_RELATIVE_PATH.as_posix(),
    )
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.useReplaceRefs")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="current pinned HEAD tree entry is not canonical v1",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_graft_file_cannot_forge_source_commit_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    canonical_tree = _git(
        repo, "rev-parse", f"{contract.source_commit}^{{tree}}",
    ).decode().strip()
    environment = archive._git_environment()
    orphan = subprocess.run(
        [os.fspath(archive.GIT_BINARY), "commit-tree", canonical_tree],
        cwd=repo,
        env=environment,
        input=b"unrelated canonical root\n",
        capture_output=True,
        check=True,
    ).stdout.decode().strip()
    _git(repo, "reset", "--hard", orphan)
    graft = tmp_path / "forged-grafts"
    graft.write_text(f"{orphan} {contract.source_commit}\n")
    graft.chmod(0o600)
    monkeypatch.setenv("GIT_GRAFT_FILE", os.fspath(graft))

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="archival source commit is not an ancestor of pinned HEAD",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_repository_local_graft_cannot_forge_source_commit_ancestry(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    canonical_tree = _git(
        repo, "rev-parse", f"{contract.source_commit}^{{tree}}",
    ).decode().strip()
    environment = archive._git_environment()
    orphan = subprocess.run(
        [os.fspath(archive.GIT_BINARY), "commit-tree", canonical_tree],
        cwd=repo,
        env=environment,
        input=b"repository-local graft root\n",
        capture_output=True,
        check=True,
    ).stdout.decode().strip()
    _git(repo, "reset", "--hard", orphan)
    local_graft = repo / ".git" / "info" / "grafts"
    local_graft.write_text(f"{orphan} {contract.source_commit}\n")
    local_graft.chmod(0o600)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="archival source commit is not an ancestor of pinned HEAD",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_packed_slash_branch_reference_is_supported(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    _git(repo, "branch", "-m", "refound/archive-test")
    _git(repo, "pack-refs", "--all", "--prune")

    result = archive._check_canonical_contract(repo, contract=contract)

    assert result["restored"]


def test_packed_refs_is_validated_even_when_head_is_loose(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    packed_refs = repo / ".git" / "packed-refs"
    outside = tmp_path / "external-packed-refs"
    outside.write_text("# external packed refs\n", encoding="ascii")
    outside.chmod(0o600)
    packed_refs.symlink_to(outside)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="Git packed references is not regular",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_unrelated_loose_reference_lock_is_rejected(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    lock = repo / ".git" / "refs" / "heads" / "unrelated.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="RUNNABILITY_RESTORE_GIT_LOCKED",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_git_guard_requires_subprocess_descriptor_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    monkeypatch.setattr(
        archive.resource,
        "getrlimit",
        lambda _resource: (8, 8),
    )

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="leaves no Git subprocess headroom",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_non_ascii_branch_reference_is_supported(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    _git(repo, "branch", "-m", "refound/résumé")

    result = archive._check_canonical_contract(repo, contract=contract)

    assert result["restored"]


def test_recursive_symbolic_reference_chain_is_supported(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    branch = _git(repo, "symbolic-ref", "HEAD").decode().strip()
    _git(repo, "symbolic-ref", "refs/heads/archive-alias", branch)
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/archive-alias")

    result = archive._check_canonical_contract(repo, contract=contract)

    assert result["restored"]


def test_symbolic_reference_depth_is_bounded(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    parts = ["refs", "heads", *[f"d{index}" for index in range(20)], "tip"]
    head = repo / ".git" / "HEAD"
    head.write_text(f"ref: {'/'.join(parts)}\n", encoding="ascii")
    head.chmod(0o600)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="reference exceeds admitted path depth",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_git_index_size_is_bounded_before_git_parses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    monkeypatch.setattr(archive, "GIT_INDEX_MAX_SIZE_BYTES", 1)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="Git index exceeds maximum size",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_symbolic_reference_with_nul_is_a_typed_refusal(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    head = repo / ".git" / "HEAD"
    head.write_bytes(b"ref: refs/heads/bad\0name\n")
    head.chmod(0o600)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="unsafe Git HEAD symbolic reference",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_head_advance_during_attestation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    original_read = archive._read_regular_exact_at
    advanced = False

    def advancing_read(*args, **kwargs):
        nonlocal advanced
        result = original_read(*args, **kwargs)
        if kwargs.get("role") == "restored canonical runnability" and not advanced:
            advanced = True
            unrelated = repo / "unrelated.txt"
            unrelated.write_text("unrelated commit\n")
            unrelated.chmod(0o600)
            _git(repo, "add", "unrelated.txt")
            _git(repo, "commit", "-m", "concurrent unrelated commit")
        return result

    monkeypatch.setattr(archive, "_read_regular_exact_at", advancing_read)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="HEAD changed during canonical verification",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_preopen_mode_race_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "race.bin"
    target.write_bytes(b"stable bytes")
    target.chmod(0o600)
    original_open = archive.os.open
    raced = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal raced
        if path == target.name and dir_fd is not None and not raced:
            raced = True
            target.chmod(0o666)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(archive.os, "open", racing_open)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="ownership/mode/link count is unsafe|changed before open",
    ):
        archive._read_regular_exact(
            target,
            role="mode-race fixture",
            expected_size=target.stat().st_size,
            max_size=1024,
        )


def test_preopen_fifo_swap_refuses_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "race.bin"
    target.write_bytes(b"stable bytes")
    target.chmod(0o600)
    original_open = archive.os.open
    raced = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal raced
        if path == target.name and dir_fd is not None and not raced:
            raced = True
            assert flags & os.O_NONBLOCK
            target.unlink()
            os.mkfifo(target, mode=0o600)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(archive.os, "open", racing_open)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="is not regular|changed before open",
    ):
        archive._read_regular_exact(
            target,
            role="FIFO-race fixture",
            expected_size=len(b"stable bytes"),
            max_size=1024,
        )


def test_absolute_binding_rejects_ancestor_rename_and_recreation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    binding_root = tmp_path / "absolute-binding"
    ancestor = binding_root / "ancestor"
    leaf = ancestor / "leaf"
    leaf.mkdir(parents=True, mode=0o700)
    binding_root.chmod(0o700)
    ancestor.chmod(0o700)
    leaf.chmod(0o700)
    moved = binding_root / "ancestor-moved"
    expected = leaf.stat()
    original_open = archive.os.open
    raced = False
    worker_errors: list[BaseException] = []

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal raced
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == ancestor.name and dir_fd is not None and not raced:
            raced = True
            barrier = threading.Barrier(2, timeout=5)

            def replace_ancestor() -> None:
                try:
                    barrier.wait()
                    ancestor.rename(moved)
                    ancestor.mkdir(mode=0o700)
                    leaf.mkdir(mode=0o700)
                except BaseException as exc:  # pragma: no cover - diagnostic
                    worker_errors.append(exc)

            worker = threading.Thread(target=replace_ancestor)
            worker.start()
            barrier.wait()
            worker.join(timeout=5)
            assert not worker.is_alive()
            if worker_errors:
                os.close(descriptor)
                raise worker_errors[0]
        return descriptor

    monkeypatch.setattr(archive.os, "open", racing_open)
    with pytest.raises(archive.RunnabilityArchiveRefusal) as refusal:
        archive._assert_absolute_directory_unchanged(
            leaf, expected, role="absolute binding fixture",
        )

    assert raced
    assert refusal.value.code == "RUNNABILITY_ARCHIVE_INCONSISTENT"


def test_directory_inventory_refuses_before_unbounded_materialization(
    tmp_path: Path,
):
    directory = tmp_path / "oversized-directory"
    directory.mkdir(mode=0o700)
    for index in range(3):
        _write(directory / f"entry-{index}", b"")
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        with pytest.raises(
            archive.RunnabilityArchiveRefusal,
            match="exceeds admitted entry count 2",
        ):
            archive._directory_names(
                descriptor, role="bounded fixture", max_entries=2,
            )
    finally:
        os.close(descriptor)


def test_oversized_payload_refuses_before_content_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, base, attempt = _archive_fixture(tmp_path)
    payload = attempt / archive.PAYLOAD_DIRECTORY / archive.PAYLOAD_NAME
    with payload.open("r+b") as stream:
        stream.truncate(archive.PAYLOAD_MAX_SIZE_BYTES + 1)
    payload.chmod(0o600)
    original_read = archive.os.read

    def guarded_read(descriptor: int, size: int) -> bytes:
        if os.fstat(descriptor).st_size > archive.PAYLOAD_MAX_SIZE_BYTES:
            raise AssertionError("oversized payload content was read")
        return original_read(descriptor, size)

    monkeypatch.setattr(archive.os, "read", guarded_read)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="payload exceeds maximum size",
    ):
        _verify_archive_fixture(base, contract)


def test_exclusive_publisher_lock_blocks_archive_verification(tmp_path: Path):
    contract, base, _attempt = _archive_fixture(tmp_path)
    lock_path = base / archive.ARCHIVE_ROOT_NAME / archive.LOCK_NAME
    with lock_path.open("rb") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(
                archive.RunnabilityArchiveRefusal,
                match="RUNNABILITY_ARCHIVE_LOCKED",
            ):
                _verify_archive_fixture(base, contract)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def test_cohort_validator_runs_while_archive_lock_is_held(tmp_path: Path):
    contract, base, _attempt = _archive_fixture(tmp_path)
    lock_path = base / archive.ARCHIVE_ROOT_NAME / archive.LOCK_NAME
    observed = False

    def validate_cohort(
        identities: frozenset[tuple[str, str, str]],
    ) -> None:
        nonlocal observed
        assert len(identities) == 53
        with lock_path.open("rb") as competing_lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(
                    competing_lock.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        observed = True

    archive._verify_archive_contract(
        base,
        contract=contract,
        schema_raw=_schema_bytes(),
        canonical_raw=_synthetic_canonical(),
        cohort_validator=validate_cohort,
    )
    assert observed


@pytest.mark.parametrize("mutation_kind", ["replacement", "in_place"])
def test_payload_mutation_during_cohort_callback_is_rejected(
    tmp_path: Path,
    mutation_kind: str,
):
    contract, base, attempt = _archive_fixture(tmp_path)
    payload = attempt / archive.PAYLOAD_DIRECTORY / archive.PAYLOAD_NAME
    original = payload.read_bytes()
    replacement = tmp_path / "cohort-payload-replacement"
    if mutation_kind == "replacement":
        _write(replacement, original)
    mutated = False
    worker_errors: list[BaseException] = []

    def mutate_payload() -> None:
        nonlocal mutated
        try:
            if mutation_kind == "replacement":
                os.replace(replacement, payload)
            else:
                with payload.open("r+b") as stream:
                    stream.write(b"x")
                payload.chmod(0o600)
            mutated = True
        except BaseException as exc:  # pragma: no cover - diagnostic
            worker_errors.append(exc)

    def validate_cohort(
        _identities: frozenset[tuple[str, str, str]],
    ) -> None:
        barrier = threading.Barrier(2, timeout=5)

        def barrier_mutation() -> None:
            barrier.wait()
            mutate_payload()

        worker = threading.Thread(target=barrier_mutation)
        worker.start()
        barrier.wait()
        worker.join(timeout=5)
        assert not worker.is_alive()
        if worker_errors:
            raise worker_errors[0]

    with pytest.raises(archive.RunnabilityArchiveRefusal) as refusal:
        archive._verify_archive_contract(
            base,
            contract=contract,
            schema_raw=_schema_bytes(),
            canonical_raw=_synthetic_canonical(),
            cohort_validator=validate_cohort,
        )

    assert mutated
    assert refusal.value.code == "RUNNABILITY_ARCHIVE_INCONSISTENT"
    assert refusal.value.detail == "payload changed during cohort verification"


@pytest.mark.parametrize("mutation_kind", ["ledger", "inventory"])
def test_ledger_or_inventory_drift_during_cohort_callback_is_rejected(
    tmp_path: Path,
    mutation_kind: str,
):
    contract, base, _attempt = _archive_fixture(tmp_path)
    namespace = base / archive.ARCHIVE_ROOT_NAME
    ledger = namespace / archive.LEDGER_NAME
    mutated = False
    worker_errors: list[BaseException] = []

    def mutate_archive() -> None:
        nonlocal mutated
        try:
            if mutation_kind == "ledger":
                with ledger.open("r+b") as stream:
                    stream.write(b"x")
                ledger.chmod(0o600)
            else:
                _write(namespace / "unexpected-inventory-entry", b"")
            mutated = True
        except BaseException as exc:  # pragma: no cover - diagnostic
            worker_errors.append(exc)

    def validate_cohort(
        _identities: frozenset[tuple[str, str, str]],
    ) -> None:
        barrier = threading.Barrier(2, timeout=5)

        def barrier_mutation() -> None:
            barrier.wait()
            mutate_archive()

        worker = threading.Thread(target=barrier_mutation)
        worker.start()
        barrier.wait()
        worker.join(timeout=5)
        assert not worker.is_alive()
        if worker_errors:
            raise worker_errors[0]

    with pytest.raises(archive.RunnabilityArchiveRefusal) as refusal:
        archive._verify_archive_contract(
            base,
            contract=contract,
            schema_raw=_schema_bytes(),
            canonical_raw=_synthetic_canonical(),
            cohort_validator=validate_cohort,
        )

    assert mutated
    assert refusal.value.code == "RUNNABILITY_ARCHIVE_INCONSISTENT"
    if mutation_kind == "ledger":
        assert refusal.value.detail == "ledger changed during cohort verification"
    else:
        assert "archive namespace post-cohort check" in refusal.value.detail


def test_aggregate_canonical_check_runs_after_archive_final_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    archive_contract, base, _attempt = _archive_fixture(tmp_path)
    restore_contract, repo, source = _restore_fixture(tmp_path)
    original_directory_names = archive._directory_names
    mutated = False
    canonical_checks = 0
    worker_errors: list[BaseException] = []

    def mutating_directory_names(
        descriptor: int,
        *,
        role: str,
        max_entries: int = archive.GIT_METADATA_MAX_ENTRIES,
    ) -> set[str]:
        nonlocal mutated
        names = original_directory_names(
            descriptor, role=role, max_entries=max_entries,
        )
        if role == "archive namespace final check" and not mutated:
            mutated = True
            barrier = threading.Barrier(2, timeout=5)

            def mutate_canonical_marker() -> None:
                try:
                    barrier.wait()
                    source.write_bytes(
                        b"x" * restore_contract.canonical_size_bytes,
                    )
                    source.chmod(0o600)
                except BaseException as exc:  # pragma: no cover - diagnostic
                    worker_errors.append(exc)

            worker = threading.Thread(target=mutate_canonical_marker)
            worker.start()
            barrier.wait()
            worker.join(timeout=5)
            assert not worker.is_alive()
            if worker_errors:
                raise worker_errors[0]
        return names

    def verify_canonical(
        _partial_identities: frozenset[tuple[str, str, str]],
    ) -> None:
        nonlocal canonical_checks
        canonical_checks += 1
        archive._check_canonical_contract(repo, contract=restore_contract)

    monkeypatch.setattr(archive, "_directory_names", mutating_directory_names)
    with pytest.raises(archive.RunnabilityArchiveRefusal) as refusal:
        archive._verify_archive_contract(
            base,
            contract=archive_contract,
            schema_raw=_schema_bytes(),
            canonical_raw=_synthetic_canonical(),
            cohort_validator=verify_canonical,
        )

    assert mutated
    assert canonical_checks == 1
    assert refusal.value.code == "RUNNABILITY_RESTORE_CANONICAL_INVALID"


def test_verify_all_reuses_one_pinned_canonical_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    manifest = {"attempt_id": "attempt"}
    restored = {
        "source_commit": "commit",
        "canonical_sha256": "canonical",
        "canonical_size_bytes": 123,
        "restored": True,
    }

    class FakePinned:
        schema_raw = b"pinned schema"
        canonical_raw = b"pinned canonical"

        def verify(self, _partial_identities):
            events.append("canonical-final")
            return restored

        def close(self):
            events.append("canonical-close")

    def open_pinned(repo_root: Path, **kwargs):
        assert repo_root == tmp_path / "repo"
        assert kwargs == {
            "contract": archive.FIXED_CONTRACT,
            "source_authority_refusal_code": (
                "RUNNABILITY_ARCHIVE_INCONSISTENT"
            ),
        }
        events.append("canonical-open")
        return FakePinned()

    def verify_archive(archive_base: Path, **kwargs):
        assert archive_base == tmp_path / "archive"
        assert kwargs["contract"] is archive.FIXED_CONTRACT
        assert kwargs["schema_raw"] == FakePinned.schema_raw
        assert kwargs["canonical_raw"] == FakePinned.canonical_raw
        events.append("archive-final")
        kwargs["cohort_validator"](frozenset())
        events.append("archive-return")
        return manifest

    def forbidden_duplicate_scan(*_args, **_kwargs):
        raise AssertionError("aggregate verification reopened Git authority")

    monkeypatch.setattr(
        archive, "_open_pinned_canonical_contract", open_pinned,
    )
    monkeypatch.setattr(archive, "_verify_archive_contract", verify_archive)
    monkeypatch.setattr(
        archive, "_read_contract_schema", forbidden_duplicate_scan,
    )
    monkeypatch.setattr(
        archive, "_check_canonical_contract", forbidden_duplicate_scan,
    )

    observed = archive._verify_fixed_all(
        tmp_path / "archive", tmp_path / "repo",
    )

    assert observed == (manifest, restored)
    assert events == [
        "canonical-open",
        "archive-final",
        "canonical-final",
        "archive-return",
        "canonical-close",
    ]


def test_pinned_source_authority_preserves_archive_refusal_taxonomy(
    tmp_path: Path,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    contract = replace(contract, canonical_blob_oid="0" * 40)

    with pytest.raises(archive.RunnabilityArchiveRefusal) as refusal:
        archive._open_pinned_canonical_contract(
            repo,
            contract=contract,
            source_authority_refusal_code=(
                "RUNNABILITY_ARCHIVE_INCONSISTENT"
            ),
        )

    assert refusal.value.code == "RUNNABILITY_ARCHIVE_INCONSISTENT"


def test_external_git_object_database_is_rejected(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    objects = repo / ".git" / "objects"
    external = tmp_path / "external-objects"
    objects.rename(external)
    objects.symlink_to(external, target_is_directory=True)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="Git object database",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_git_object_alternates_authority_is_rejected(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    alternates = repo / ".git" / "objects" / "info" / "alternates"
    alternates.write_text(f"{tmp_path / 'external-objects'}\n", encoding="utf-8")
    alternates.chmod(0o600)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="external object authority",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_final_object_inventory_rejects_loose_object_replaced_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    object_id = contract.canonical_blob_oid
    object_path = repo / ".git" / "objects" / object_id[:2] / object_id[2:]
    replacement = tmp_path / "replacement-loose-object"
    _write(replacement, object_path.read_bytes())
    target_role = (
        f"Git object database entry {object_id[:2] + '/' + object_id[2:]!r}"
    )
    original_open = archive._open_owned_regular_at
    target_opens = 0
    raced = False
    worker_errors: list[BaseException] = []

    def racing_open(parent_descriptor: int, name: str, *, role: str):
        nonlocal raced, target_opens
        opened = original_open(parent_descriptor, name, role=role)
        if role == target_role:
            target_opens += 1
            if target_opens == 2:
                raced = True
                barrier = threading.Barrier(2, timeout=5)

                def replace_object() -> None:
                    try:
                        barrier.wait()
                        os.replace(replacement, object_path)
                    except BaseException as exc:  # pragma: no cover - diagnostic
                        worker_errors.append(exc)

                worker = threading.Thread(target=replace_object)
                worker.start()
                barrier.wait()
                worker.join(timeout=5)
                assert not worker.is_alive()
                if worker_errors:
                    os.close(opened[0])
                    raise worker_errors[0]
        return opened

    monkeypatch.setattr(archive, "_open_owned_regular_at", racing_open)
    with pytest.raises(archive.RunnabilityArchiveRefusal) as refusal:
        archive._check_canonical_contract(repo, contract=contract)

    assert raced
    assert target_opens == 2
    assert refusal.value.code == "RUNNABILITY_RESTORE_GIT_STATE_CHANGED"


def test_pinned_canonical_guard_uses_one_initial_and_final_git_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    original_objects = archive._snapshot_git_object_tree
    original_references = archive._snapshot_git_reference_tree
    counts = {"objects": 0, "references": 0}

    def count_objects(*args, **kwargs):
        counts["objects"] += 1
        return original_objects(*args, **kwargs)

    def count_references(*args, **kwargs):
        counts["references"] += 1
        return original_references(*args, **kwargs)

    monkeypatch.setattr(archive, "_snapshot_git_object_tree", count_objects)
    monkeypatch.setattr(
        archive, "_snapshot_git_reference_tree", count_references,
    )

    assert archive._check_canonical_contract(
        repo, contract=contract,
    )["restored"]
    assert counts == {"objects": 2, "references": 2}


def test_git_local_config_include_is_rejected(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    included = tmp_path / "external.gitconfig"
    included.write_text("[core]\n\tworktree = /tmp/external\n", encoding="utf-8")
    included.chmod(0o600)
    with (repo / ".git" / "config").open("a", encoding="utf-8") as stream:
        stream.write(f"[include]\n\tpath = {included}\n")

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="includes external config authority",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_external_files_ref_storage_payload_is_rejected(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    with (repo / ".git" / "config").open("a", encoding="utf-8") as stream:
        stream.write(
            "[extensions]\n"
            f"\trefStorage = files:///{tmp_path / 'external-refs'}\n"
        )

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="external or unsupported ref storage payload",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_git_config_change_during_attestation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    config = repo / ".git" / "config"
    original_read = archive._read_regular_exact_at
    changed = False

    def changing_read(*args, **kwargs):
        nonlocal changed
        result = original_read(*args, **kwargs)
        if kwargs.get("role") == "restored canonical runnability" and not changed:
            changed = True
            with config.open("a", encoding="utf-8") as stream:
                stream.write(f"[core]\n\tworktree = {tmp_path / 'elsewhere'}\n")
        return result

    monkeypatch.setattr(archive, "_read_regular_exact_at", changing_read)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="separate worktree authority|Git local config.*changed",
    ):
        archive._check_canonical_contract(repo, contract=contract)


@pytest.mark.parametrize("lock_name", ["config.lock", "index.lock"])
def test_existing_git_transaction_lock_is_rejected(
    tmp_path: Path, lock_name: str,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    lock = repo / ".git" / lock_name
    lock.write_bytes(b"")
    lock.chmod(0o600)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="RUNNABILITY_RESTORE_GIT_LOCKED",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_postread_payload_replacement_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, base, attempt = _archive_fixture(tmp_path)
    payload = attempt / archive.PAYLOAD_DIRECTORY / archive.PAYLOAD_NAME
    original_validate = archive._validate_partial_payload
    replaced = False

    def replacing_validate(raw: bytes, schema_raw: bytes) -> None:
        nonlocal replaced
        original_validate(raw, schema_raw)
        if not replaced:
            replaced = True
            replacement = payload.with_name(f".{payload.name}.replacement")
            replacement.write_bytes(b"x" * len(raw))
            replacement.chmod(0o600)
            os.replace(replacement, payload)

    monkeypatch.setattr(archive, "_validate_partial_payload", replacing_validate)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal, match="changed after verification",
    ):
        _verify_archive_fixture(base, contract)


def test_parent_symlink_swap_during_archive_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, base, _attempt = _archive_fixture(tmp_path)
    namespace = base / archive.ARCHIVE_ROOT_NAME
    moved = base / f".{archive.ARCHIVE_ROOT_NAME}.moved"
    original_read = archive._read_regular_exact_at
    swapped = False

    def swapping_read(parent_descriptor, name, **kwargs):
        nonlocal swapped
        if name == archive.LEDGER_NAME and not swapped:
            swapped = True
            namespace.rename(moved)
            namespace.symlink_to(moved, target_is_directory=True)
        return original_read(parent_descriptor, name, **kwargs)

    monkeypatch.setattr(archive, "_read_regular_exact_at", swapping_read)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="archive namespace changed after verification",
    ):
        _verify_archive_fixture(base, contract)


def test_staged_blob_added_after_initial_check_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    alternate = tmp_path / "alternate-index-blob"
    alternate.write_bytes(b"interrupted staged blob\n")
    alternate.chmod(0o600)
    alternate_oid = _git(
        repo, "hash-object", "-w", os.fspath(alternate),
    ).decode().strip()
    original_read = archive._read_regular_exact_at
    staged = False

    def staging_read(*args, **kwargs):
        nonlocal staged
        result = original_read(*args, **kwargs)
        if kwargs.get("role") == "restored canonical runnability" and not staged:
            staged = True
            _git(
                repo,
                "update-index",
                "--cacheinfo",
                f"100644,{alternate_oid},{archive.SOURCE_RELATIVE_PATH.as_posix()}",
            )
        return result

    monkeypatch.setattr(archive, "_read_regular_exact_at", staging_read)
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="canonical path remains dirty or staged in the index",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_source_aba_between_reads_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, source = _restore_fixture(tmp_path)
    canonical = source.read_bytes()
    original_read = archive._read_regular_exact_at
    raced = False

    def replacing_and_restoring_read(*args, **kwargs):
        nonlocal raced
        result = original_read(*args, **kwargs)
        if kwargs.get("role") == "restored canonical runnability" and not raced:
            raced = True
            source.write_bytes(b"x" * len(canonical))
            source.write_bytes(canonical)
            source.chmod(0o600)
        return result

    monkeypatch.setattr(
        archive, "_read_regular_exact_at", replacing_and_restoring_read,
    )
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="canonical source changed between verification reads",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_index_aba_during_attestation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    contract, repo, _source = _restore_fixture(tmp_path)
    alternate = tmp_path / "alternate-index-aba-blob"
    alternate.write_bytes(b"interrupted staged blob\n")
    alternate.chmod(0o600)
    alternate_oid = _git(
        repo, "hash-object", "-w", os.fspath(alternate),
    ).decode().strip()
    original_read = archive._read_regular_exact_at
    raced = False

    def staging_and_restoring_read(*args, **kwargs):
        nonlocal raced
        result = original_read(*args, **kwargs)
        if kwargs.get("role") == "restored canonical runnability" and not raced:
            raced = True
            _git(
                repo,
                "update-index",
                "--cacheinfo",
                f"100644,{alternate_oid},{archive.SOURCE_RELATIVE_PATH.as_posix()}",
            )
            _git(
                repo,
                "reset",
                "HEAD",
                "--",
                archive.SOURCE_RELATIVE_PATH.as_posix(),
            )
        return result

    monkeypatch.setattr(
        archive, "_read_regular_exact_at", staging_and_restoring_read,
    )
    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match=r"Git (?:index|metadata root).*changed",
    ):
        archive._check_canonical_contract(repo, contract=contract)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_index_flags_are_rejected(tmp_path: Path, flag: str):
    contract, repo, _source = _restore_fixture(tmp_path)
    _git(
        repo, "update-index", flag,
        archive.SOURCE_RELATIVE_PATH.as_posix(),
    )

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="assume-unchanged, skip-worktree, or other flags",
    ):
        archive._check_canonical_contract(repo, contract=contract)


def test_unrelated_clone_requires_source_path_alias(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    clone = tmp_path / "unrelated-clone"
    shutil.copytree(repo, clone)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="RUNNABILITY_RESTORE_PATH_ALIAS_REQUIRED",
    ):
        archive._check_canonical_contract(clone, contract=contract)


def test_interrupted_cohort_must_be_canonical_identity_subset():
    canonical = json.loads(_canonical_document())
    partial = {"cells": [dict(cell) for cell in canonical["cells"][:53]]}
    archive._validate_cohort_identity_binding(
        json.dumps(partial).encode(), _canonical_document(),
    )
    partial["cells"][0]["slug"] = "not-in-canonical-v1"

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="identity set is not a subset of canonical v1",
    ):
        archive._validate_cohort_identity_binding(
            json.dumps(partial).encode(), _canonical_document(),
        )


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (
            "verify-archive",
            {
                "status": "verified",
                "archive_verified": True,
                "attempt_id": "attempt",
                "source_sha256": "source",
                "payload_schema_validation": archive.PARTIAL_SCHEMA_VALIDATION,
                **archive.DISPOSITION,
            },
        ),
        (
            "check-restored",
            {
                "status": "verified",
                "canonical_restored": True,
                "source_commit": "commit",
                "canonical_sha256": "canonical",
                "canonical_size_bytes": 123,
            },
        ),
        (
            "verify-all",
            {
                "status": "verified",
                "archive_verified": True,
                "attempt_id": "attempt",
                "source_sha256": "source",
                "payload_schema_validation": archive.PARTIAL_SCHEMA_VALIDATION,
                **archive.DISPOSITION,
                "canonical_restored": True,
                "source_commit": "commit",
                "canonical_sha256": "canonical",
                "canonical_size_bytes": 123,
                "cohort_identity_bound": True,
            },
        ),
    ],
)
def test_cli_reports_only_facts_verified_by_each_action(
    action: str,
    expected: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        archive,
        "verify_fixed_archive",
        lambda _archive_base: {
            "attempt_id": "attempt",
            "source": {"sha256": "source"},
            "disposition": dict(archive.DISPOSITION),
        },
    )
    monkeypatch.setattr(
        archive,
        "check_canonical_restored",
        lambda _repo_root: {
            "source_commit": "commit",
            "canonical_sha256": "canonical",
            "canonical_size_bytes": 123,
            "restored": True,
        },
    )

    monkeypatch.setattr(
        archive,
        "_verify_fixed_all",
        lambda _archive_base, _repo_root: (
            {
                "attempt_id": "attempt",
                "source": {"sha256": "source"},
                "disposition": dict(archive.DISPOSITION),
            },
            {
                "source_commit": "commit",
                "canonical_sha256": "canonical",
                "canonical_size_bytes": 123,
                "restored": True,
            },
        ),
    )

    assert archive.main([action]) == 0
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize(
    "arguments",
    [
        ["verify-archive", "--repo-root", "/tmp/ignored-repository"],
        ["check-restored", "--archive-base", "/tmp/ignored-archive"],
    ],
)
def test_cli_rejects_explicit_arguments_irrelevant_to_action(
    arguments: list[str], capsys: pytest.CaptureFixture[str],
):
    assert archive.main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RUNNABILITY_ARCHIVE_UNUSED_ARGUMENT" in captured.err


def test_world_writable_source_parent_is_rejected(tmp_path: Path):
    contract, repo, _source = _restore_fixture(tmp_path)
    unsafe = repo / "data" / "local_labels"
    unsafe.chmod(0o777)

    with pytest.raises(
        archive.RunnabilityArchiveRefusal,
        match="ownership/mode is unsafe",
    ):
        archive._check_canonical_contract(repo, contract=contract)


@pytest.mark.campaign_evidence
def test_fixed_production_archive_is_still_verified():
    selected_root = os.environ.get("SM86_REFERENCE_CAPTURE_ROOT")
    assert selected_root, (
        "SM86_REFERENCE_CAPTURE_ROOT is required for campaign evidence"
    )
    manifest, restored = archive._verify_fixed_all(
        Path(selected_root), archive.REPO_ROOT,
    )
    assert manifest["attempt_id"] == archive.FIXED_CONTRACT.attempt_id
    assert restored["restored"] is True
