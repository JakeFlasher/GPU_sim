from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from tools.dev import gate_receipt as gate
from tools.local_labels.immutable_artifact import ImmutableArtifactError, seal_model


_PYTHON_TARGET = str(Path(sys.executable).resolve())
_PYTHON_ARGV0 = str(Path(os.path.abspath(sys.executable)))


def _git_environment() -> dict[str, str]:
    result = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    return result


def test_gate_git_environment_disables_system_attributes() -> None:
    environment = gate._git_environment()
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_ATTR_NOSYSTEM"] == "1"
    assert "HOME" not in environment


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_git_environment(),
    )
    return completed.stdout.strip()


def _thaw_tree(root: Path) -> None:
    if not root.exists():
        return
    for directory, _directories, files in os.walk(root):
        try:
            os.chmod(directory, 0o700)
        except FileNotFoundError:
            continue
        for name in files:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    os.chmod(path, 0o600)
            except FileNotFoundError:
                pass


@contextlib.contextmanager
def _external_rig():
    parent = gate._external_directory(Path("/var/tmp"), role="gate test root")
    root = Path(tempfile.mkdtemp(prefix="gate-receipt-pytest-", dir=parent))
    os.chmod(root, 0o700)
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Gate Test")
    _git(repo, "config", "user.email", "gate-test@example.invalid")
    (repo / "tracked.txt").write_text("stable\n")
    (repo / "src").mkdir()
    (repo / "src" / "nested.txt").write_text("nested-stable\n")
    _git(repo, "add", "tracked.txt", "src/nested.txt")
    _git(repo, "commit", "-q", "-m", "fixture")
    output = root / "receipts"
    scratch = root / "scratch"
    output.mkdir()
    scratch.mkdir()
    try:
        yield repo, output, scratch
    finally:
        _thaw_tree(root)
        shutil.rmtree(root)


@pytest.fixture
def rig():
    with _external_rig() as value:
        yield value


def _run(
    rig,
    generation: str,
    code: str,
    *,
    timeout_seconds: float = 10,
) -> gate.GateRunResult:
    repo, output, scratch = rig
    return gate.run_gate(
        output_root=output,
        generation=generation,
        cwd=repo,
        temp_root=scratch / generation,
        timeout_seconds=timeout_seconds,
        argv=(_PYTHON_TARGET, "-c", code),
        process_argv0=_PYTHON_ARGV0,
    )


def _reseal(document: dict) -> dict:
    forged = copy.deepcopy(document)
    forged.pop("receipt_sha256", None)
    return seal_model(forged, field="receipt_sha256")


def test_passed_gate_seals_raw_streams_and_replays(rig):
    result = _run(
        rig,
        "pass",
        "import sys; print('stdout-value'); print('stderr-value', file=sys.stderr)",
    )
    assert result.shell_return_code == 0
    assert result.document["schema_version"] == "gate_receipt.v3"
    assert result.document["outcome"] == {"status": "passed", "reason": None}
    assert result.document["integrity"] == {
        "seal": "canonical_json_unkeyed_sha256_consistency_v1",
        "authority": (
            "requires_external_content_manifest_sha256_anchor_and_"
            "trusted_clean_source_producer_v1"
        ),
    }
    assert (result.generation_path / gate.STDOUT_NAME).read_bytes() == b"stdout-value\n"
    assert (result.generation_path / gate.STDERR_NAME).read_bytes() == b"stderr-value\n"

    replay = gate.verify_generation(result.generation_path, require_current_source=True)
    assert replay["status"] == "consistency_checked_unanchored"
    assert replay["content_manifest_anchor"] == "unanchored"
    assert replay["current_source"] == "matched"
    assert replay["content_manifest_sha256"] == result.content_manifest_sha256
    assert replay["filesystem_generation_sha256"] == (
        result.filesystem_generation_sha256
    )
    anchored = gate.verify_generation(
        result.generation_path,
        require_current_source=True,
        expected_content_manifest_sha256=result.content_manifest_sha256,
    )
    assert anchored["status"] == "verified_anchored"
    assert anchored["content_manifest_anchor"] == "matched"
    assert anchored["current_source"] == "matched"
    assert replay["outcome"] == result.document["outcome"]
    assert len(replay["content_manifest_sha256"]) == 64
    assert len(replay["filesystem_generation_sha256"]) == 64
    tracked = result.document["source"]["tracked_inputs"]
    worktree = result.document["source"]["checked_out_worktree"]
    assert tracked["flag_format"] == "git_ls_files_v_normal_only_v1"
    assert worktree["entry_count"] == tracked["entry_count"] == 2
    assert worktree["total_file_bytes"] == len(b"stable\n") + len(
        b"nested-stable\n"
    )
    assert len(worktree["manifest_sha256"]) == 64

    flag_payload = b"H src/nested.txt\0H tracked.txt\0"
    assert tracked["flag_manifest_sha256"] == hashlib.sha256(
        flag_payload
    ).hexdigest()
    oracle = hashlib.sha256()
    magic = b"gate_checked_out_worktree_files_and_directories_sha256_v1\0"
    oracle.update(magic)
    oracle.update((2).to_bytes(8, "big"))
    oracle.update((2).to_bytes(8, "big"))
    for relative_path, path in (
        (".", rig[0]),
        ("src", rig[0] / "src"),
    ):
        encoded_path = relative_path.encode()
        oracle.update(len(encoded_path).to_bytes(8, "big"))
        oracle.update(encoded_path)
        oracle.update(path.stat().st_mode.to_bytes(8, "big"))
    for relative_path in ("src/nested.txt", "tracked.txt"):
        path = rig[0] / relative_path
        payload = path.read_bytes()
        encoded_path = relative_path.encode()
        oracle.update(len(encoded_path).to_bytes(8, "big"))
        oracle.update(encoded_path)
        oracle.update(b"100644")
        oracle.update(path.stat().st_mode.to_bytes(8, "big"))
        oracle.update(len(payload).to_bytes(8, "big"))
        oracle.update(hashlib.sha256(payload).digest())
    assert worktree["manifest_sha256"] == oracle.hexdigest()
    git_admin = result.document["source"]["git_administration"]
    assert git_admin["layout"] == "ordinary_gitdir"
    assert git_admin["worktree_git_entry"]["kind"] == "directory"
    assert git_admin["head"]["kind"] == "symbolic"
    assert git_admin["active_ref"]["storage"] == "loose"


def test_content_manifest_anchor_is_external_and_fail_closed(rig):
    result = _run(rig, "anchor-contract", "print('anchor')")

    with pytest.raises(gate.GateReceiptError) as malformed:
        gate.verify_generation(
            result.generation_path,
            expected_content_manifest_sha256="not-a-sha256",
        )
    assert malformed.value.code == "GATE_CONTENT_MANIFEST_ANCHOR_INVALID"

    with pytest.raises(gate.GateReceiptError) as mismatch:
        gate.verify_generation(
            result.generation_path,
            expected_content_manifest_sha256="0" * 64,
        )
    assert mismatch.value.code == "GATE_CONTENT_MANIFEST_ANCHOR_MISMATCH"

    unanchored = gate.verify_generation(
        result.generation_path,
        require_current_source=True,
    )
    assert unanchored["status"] == "consistency_checked_unanchored"
    assert unanchored["content_manifest_anchor"] == "unanchored"
    assert unanchored["current_source"] == "matched"


def test_run_cli_reports_race_free_producer_manifest(rig, capsys):
    repo, output, scratch = rig
    return_code = gate.main(
        [
            "run",
            "--output-root",
            str(output),
            "--generation",
            "cli-producer-anchor",
            "--cwd",
            str(repo),
            "--temp-root",
            str(scratch / "cli-producer-anchor"),
            "--timeout-seconds",
            "10",
            "--argv0",
            _PYTHON_ARGV0,
            "--",
            _PYTHON_TARGET,
            "-c",
            "print('cli-anchor')",
        ]
    )
    captured = capsys.readouterr()
    assert return_code == 0
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["producer_manifest_source"] == (
        "race_free_exclusive_stage_commit_return_v1"
    )
    producer_anchor = report["producer_content_manifest_sha256"]
    assert len(producer_anchor) == 64
    verified = gate.verify_generation(
        output / "cli-producer-anchor",
        require_current_source=True,
        expected_content_manifest_sha256=producer_anchor,
    )
    assert verified["status"] == "verified_anchored"


def test_nonzero_exit_is_published_and_never_reinterpreted_as_pass(rig):
    result = _run(rig, "exit-7", "raise SystemExit(7)")
    assert result.shell_return_code == 7
    wait = result.document["wait"]
    assert wait["pid"] == wait["waited_pid"] == wait["process_group_id"] == wait["session_id"]
    assert {key: value for key, value in wait.items() if key not in {
        "pid", "waited_pid", "process_group_id", "session_id",
    }} == {
        "raw_wait_status": 7 << 8,
        "return_code": 7,
        "exit_code": 7,
        "signal": None,
        "timed_out": False,
    }
    assert result.document["outcome"] == {
        "status": "failed",
        "reason": "NONZERO_EXIT",
    }
    assert gate.verify_generation(result.generation_path)["outcome"]["status"] == "failed"


def test_signal_status_is_exact_and_shell_visible(rig):
    result = _run(
        rig,
        "signal",
        "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
    )
    assert result.shell_return_code == 128 + signal.SIGTERM
    assert result.document["wait"]["return_code"] == -signal.SIGTERM
    assert result.document["wait"]["signal"] == signal.SIGTERM
    assert result.document["wait"]["exit_code"] is None
    assert result.document["outcome"] == {"status": "failed", "reason": "SIGNAL"}


def test_timeout_is_typed_and_published(rig):
    result = _run(
        rig,
        "timeout",
        "import time; time.sleep(10)",
        timeout_seconds=0.05,
    )
    assert result.shell_return_code == 124
    assert result.document["wait"]["timed_out"] is True
    assert result.document["outcome"] == {"status": "failed", "reason": "TIMEOUT"}
    supervision = result.document["supervision"]
    timing = result.document["time"]
    assert supervision["requested_timeout_ns"] == 50_000_000
    assert supervision["decision"] == "timeout"
    assert int(supervision["monotonic_deadline_ns"]) == (
        int(timing["monotonic_start_ns"]) + 50_000_000
    )
    assert int(supervision["monotonic_decision_ns"]) >= int(
        supervision["monotonic_deadline_ns"]
    )
    assert int(supervision["monotonic_decision_ns"]) <= int(
        timing["monotonic_stop_ns"]
    )
    assert gate.verify_generation(result.generation_path)["outcome"]["reason"] == "TIMEOUT"


def test_child_receives_fresh_gate_owned_temp_and_cache_paths(rig):
    result = _run(
        rig,
        "environment",
        "import json, os; print(json.dumps({k: os.environ[k] for k in "
        "['TMP', 'TEMP', 'TMPDIR', 'CARGO_TARGET_DIR', "
        "'PYTHONPYCACHEPREFIX', 'PYTHONDONTWRITEBYTECODE']}, sort_keys=True))",
    )
    observed = json.loads((result.generation_path / gate.STDOUT_NAME).read_text())
    scratch = Path(result.document["temporary_root"]["path"])
    assert observed == {
        "CARGO_TARGET_DIR": str(scratch / "cargo-target"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(scratch / "pycache"),
        "TEMP": str(scratch / "tmp"),
        "TMP": str(scratch / "tmp"),
        "TMPDIR": str(scratch / "tmp"),
    }
    assert all(Path(observed[name]).is_relative_to(scratch) for name in observed if name != "PYTHONDONTWRITEBYTECODE")


def test_existing_generation_refuses_before_second_child_or_temp_creation(rig):
    repo, output, scratch = rig
    first = _run(rig, "collision", "print('first')")
    sentinel = repo / "must-not-exist"
    with pytest.raises(ImmutableArtifactError, match="destination already exists"):
        gate.run_gate(
            output_root=output,
            generation="collision",
            cwd=repo,
            temp_root=scratch / "collision-second",
            timeout_seconds=10,
            argv=(_PYTHON_TARGET, "-c", f"from pathlib import Path; Path({str(sentinel)!r}).write_text('bad')"),
        )
    assert first.generation_path.exists()
    assert not sentinel.exists()
    assert not (scratch / "collision-second").exists()


def test_dirty_source_refuses_before_child(rig):
    repo, output, scratch = rig
    (repo / "untracked.txt").write_text("dirty\n")
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="dirty",
            cwd=repo,
            temp_root=scratch / "dirty",
            timeout_seconds=10,
            argv=(_PYTHON_TARGET, "-c", "raise SystemExit(99)"),
        )
    assert caught.value.code == "GATE_SOURCE_DIRTY"
    assert not (output / "dirty").exists()
    assert not (scratch / "dirty").exists()


@pytest.mark.parametrize(
    "index_flag",
    ("--assume-unchanged", "--skip-worktree"),
)
def test_hidden_index_worktree_flags_refuse_before_spawn(rig, index_flag):
    repo, output, scratch = rig
    _git(repo, "update-index", index_flag, "tracked.txt")
    assert _git(repo, "status", "--porcelain=v1") == ""
    sentinel = repo / "must-not-run-hidden-index-flag"

    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="hidden-index-flag",
            cwd=repo,
            temp_root=scratch / "hidden-index-flag",
            timeout_seconds=10,
            argv=(
                _PYTHON_TARGET,
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
            ),
        )
    assert caught.value.code == "GATE_SOURCE_LAYOUT_UNSUPPORTED"
    assert not sentinel.exists()
    assert not (output / "hidden-index-flag").exists()
    assert not (scratch / "hidden-index-flag").exists()


def test_tracked_flag_inventory_count_refuses_before_row_decoding(
    tmp_path,
    monkeypatch,
):
    stage_payload = (
        b"100644 " + b"a" * 40 + b" 0\ttracked.txt\0"
    )
    flag_payload = b"H x\0" * 100_000
    responses = iter(
        (
            (1, b""),
            (0, b"\n"),
            (0, stage_payload),
            (0, flag_payload),
        )
    )
    monkeypatch.setattr(gate, "_run_git", lambda *args, **kwargs: next(responses))
    original_safe_relative_path = gate.safe_relative_path
    decoded_paths = 0

    def count_decoded_path(value):
        nonlocal decoded_paths
        decoded_paths += 1
        if decoded_paths > 1:
            pytest.fail("flag rows were decoded before inventory count refusal")
        return original_safe_relative_path(value)

    monkeypatch.setattr(gate, "safe_relative_path", count_decoded_path)
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(gate.GateReceiptError) as caught:
            gate._tracked_input_manifest(
                tmp_path,
                cwd_descriptor=descriptor,
            )
    finally:
        os.close(descriptor)
    assert caught.value.code == "GATE_TRACKED_INPUT_INVALID"
    assert decoded_paths == 1


def test_oversized_git_branch_refuses_before_stage_or_child(rig):
    repo, output, scratch = rig
    branch = "/".join(("b" * 100 for _ in range(11)))
    assert len(branch.encode()) > gate._MAX_GIT_BRANCH_BYTES
    _git(repo, "checkout", "-q", "-b", branch)
    sentinel = repo / "must-not-run-long-branch"

    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="long-branch",
            cwd=repo,
            temp_root=scratch / "long-branch",
            timeout_seconds=10,
            argv=(
                _PYTHON_TARGET,
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
            ),
        )
    assert caught.value.code == "GATE_GIT_OUTPUT_INVALID"
    assert not sentinel.exists()
    assert not (output / ".long-branch.staging").exists()
    assert not (scratch / "long-branch").exists()


def test_source_drift_preserves_uncommitted_stage(rig):
    repo, output, scratch = rig
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="source-drift",
            cwd=repo,
            temp_root=scratch / "source-drift",
            timeout_seconds=10,
            argv=(
                _PYTHON_TARGET,
                "-c",
                "from pathlib import Path; Path('tracked.txt').write_text('changed\\n')",
            ),
        )
    assert caught.value.code in {"GATE_CWD_DRIFT", "GATE_SOURCE_DRIFT"}
    assert not (output / "source-drift").exists()
    assert (output / ".source-drift.staging").is_dir()


def test_transient_tracked_modify_restore_is_rejected(rig):
    repo, output, _scratch = rig
    with pytest.raises(gate.GateReceiptError) as caught:
        _run(
            rig,
            "source-transient",
            "from pathlib import Path; value = Path('tracked.txt'); "
            "value.write_text('transient\\n'); value.write_text('stable\\n')",
        )
    assert caught.value.code == "GATE_SOURCE_DRIFT"
    assert (repo / "tracked.txt").read_text() == "stable\n"
    assert not (output / "source-transient").exists()
    assert (output / ".source-transient.staging").is_dir()


@pytest.mark.parametrize(
    ("generation", "setup", "relative_path", "replacement"),
    (
        (
            "git-head-transient",
            lambda repo: _git(repo, "branch", "same-commit"),
            ".git/HEAD",
            "ref: refs/heads/same-commit\n",
        ),
        (
            "git-config-transient",
            lambda _repo: None,
            ".git/config",
            "[core]\n\trepositoryformatversion = 0\n",
        ),
    ),
)
def test_transient_git_administration_modify_restore_is_rejected(
    rig,
    generation,
    setup,
    relative_path,
    replacement,
):
    repo, output, _scratch = rig
    setup(repo)
    code = (
        "from pathlib import Path; "
        f"path = Path({relative_path!r}); "
        "original = path.read_bytes(); "
        f"path.write_text({replacement!r}); "
        "path.write_bytes(original)"
    )
    with pytest.raises(gate.GateReceiptError) as caught:
        _run(rig, generation, code)
    assert caught.value.code == "GATE_SOURCE_DRIFT"
    assert not (output / generation).exists()
    assert (output / f".{generation}.staging").is_dir()


def test_transient_active_loose_ref_modify_restore_is_rejected(rig):
    repo, output, _scratch = rig
    active_ref = _git(repo, "symbolic-ref", "HEAD")
    ref_path = Path(".git") / active_ref
    code = (
        "from pathlib import Path; "
        f"path = Path({str(ref_path)!r}); "
        "original = path.read_bytes(); "
        "path.write_text('0' * 40 + '\\n'); "
        "path.write_bytes(original)"
    )
    with pytest.raises(gate.GateReceiptError) as caught:
        _run(rig, "git-loose-ref-transient", code)
    assert caught.value.code == "GATE_SOURCE_DRIFT"
    assert not (output / "git-loose-ref-transient").exists()


def test_transient_git_directory_swap_restore_is_rejected(rig):
    repo, output, _scratch = rig
    code = """
from pathlib import Path
active = Path('.git')
displaced = Path('.git-original')
active.rename(displaced)
active.mkdir()
active.rmdir()
displaced.rename(active)
"""
    with pytest.raises(gate.GateReceiptError) as caught:
        _run(rig, "git-directory-transient", code)
    assert caught.value.code == "GATE_SOURCE_DRIFT"
    assert not (output / "git-directory-transient").exists()


def test_transient_absent_git_pointer_creation_is_rejected(rig):
    _repo, output, _scratch = rig
    code = (
        "from pathlib import Path; path = Path('.git/commondir'); "
        "path.write_text('../transient\\n'); path.unlink()"
    )
    with pytest.raises(gate.GateReceiptError) as caught:
        _run(rig, "git-pointer-absence-transient", code)
    assert caught.value.code == "GATE_SOURCE_DRIFT"
    assert not (output / "git-pointer-absence-transient").exists()


def test_packed_active_ref_is_bound_and_transient_restore_is_rejected(rig):
    repo, output, _scratch = rig
    _git(repo, "pack-refs", "--all")
    active_ref = _git(repo, "symbolic-ref", "HEAD")
    assert not (repo / ".git" / active_ref).exists()

    stable = _run(rig, "git-packed-ref", "print('packed')")
    assert stable.document["source"]["git_administration"]["active_ref"][
        "storage"
    ] == "packed_refs"

    code = (
        "from pathlib import Path; path = Path('.git/packed-refs'); "
        "original = path.read_bytes(); "
        "path.write_bytes(original + b'# transient\\n'); "
        "path.write_bytes(original)"
    )
    with pytest.raises(gate.GateReceiptError) as caught:
        _run(rig, "git-packed-ref-transient", code)
    assert caught.value.code == "GATE_SOURCE_DRIFT"
    assert not (output / "git-packed-ref-transient").exists()


def test_linked_worktree_administration_is_retained_and_replayed(rig):
    repo, output, scratch = rig
    linked = repo.parent / "linked"
    _git(repo, "worktree", "add", "-q", "-b", "linked-fixture", str(linked))

    result = gate.run_gate(
        output_root=output,
        generation="linked-worktree",
        cwd=linked,
        temp_root=scratch / "linked-worktree",
        timeout_seconds=10,
        argv=(_PYTHON_TARGET, "-c", "print('linked')"),
        process_argv0=_PYTHON_ARGV0,
    )
    admin = result.document["source"]["git_administration"]
    assert admin["layout"] == "linked_worktree"
    assert admin["worktree_git_entry"]["kind"] == "gitfile"
    assert admin["git_dir"]["path"] != admin["git_common_dir"]["path"]
    assert admin["commondir_pointer"]["target"] == admin[
        "git_common_dir"
    ]["path"]
    assert admin["gitdir_pointer"]["target"] == str(linked / ".git")
    assert gate.verify_generation(
        result.generation_path,
        require_current_source=True,
    )["status"] == "consistency_checked_unanchored"


@pytest.mark.parametrize("binding_role", ("marker", "commondir", "gitdir"))
def test_linked_pointer_semantics_are_bound_to_file_identity(
    rig,
    binding_role,
):
    repo, output, scratch = rig
    linked = repo.parent / f"linked-binding-{binding_role}"
    _git(
        repo,
        "worktree",
        "add",
        "-q",
        "-b",
        f"linked-binding-{binding_role}",
        str(linked),
    )
    result = gate.run_gate(
        output_root=output,
        generation=f"linked-binding-{binding_role}",
        cwd=linked,
        temp_root=scratch / f"linked-binding-{binding_role}",
        timeout_seconds=10,
        argv=(_PYTHON_TARGET, "-c", "print('linked-binding')"),
        process_argv0=_PYTHON_ARGV0,
    )
    forged = copy.deepcopy(result.document)
    admin = forged["source"]["git_administration"]
    if binding_role == "marker":
        binding = admin["worktree_git_entry"]["binding"]
    else:
        binding = admin[f"{binding_role}_pointer"]["file"]["binding"]
    binding["size_bytes"] = "1"
    binding["sha256"] = hashlib.sha256(b"x").hexdigest()

    with pytest.raises(gate.GateReceiptError) as caught:
        gate._validate_document(_reseal(forged))
    assert caught.value.code == "GATE_RECEIPT_INVALID"


def test_linked_worktree_config_is_retained_when_enabled(rig):
    repo, output, scratch = rig
    linked = repo.parent / "linked-config"
    _git(repo, "worktree", "add", "-q", "-b", "linked-config", str(linked))
    _git(repo, "config", "extensions.worktreeConfig", "true")
    _git(linked, "config", "--worktree", "gate.bound", "true")

    result = gate.run_gate(
        output_root=output,
        generation="linked-config",
        cwd=linked,
        temp_root=scratch / "linked-config",
        timeout_seconds=10,
        argv=(_PYTHON_TARGET, "-c", "print('linked-config')"),
        process_argv0=_PYTHON_ARGV0,
    )
    worktree_config = result.document["source"]["git_administration"][
        "config"
    ]["worktree"]
    assert worktree_config["state"] == "present"
    assert gate.verify_generation(
        result.generation_path,
        require_current_source=True,
    )["status"] == "consistency_checked_unanchored"


@pytest.mark.parametrize(
    "key",
    ("include.path", "core.attributesFile", "core.excludesFile"),
)
def test_linked_worktree_config_external_inputs_are_rejected(
    rig,
    key,
):
    repo, output, scratch = rig
    linked = repo.parent / f"linked-external-{key.split('.')[-1].lower()}"
    _git(
        repo,
        "worktree",
        "add",
        "-q",
        "-b",
        f"linked-external-{key.split('.')[-1].lower()}",
        str(linked),
    )
    _git(repo, "config", "extensions.worktreeConfig", "true")
    _git(linked, "config", "--worktree", key, "/var/tmp/unretained-input")
    sentinel = linked / "must-not-run-worktree-config"

    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation=f"linked-external-{key.split('.')[-1].lower()}",
            cwd=linked,
            temp_root=scratch / f"linked-external-{key.split('.')[-1].lower()}",
            timeout_seconds=10,
            argv=(
                _PYTHON_TARGET,
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
            ),
            process_argv0=_PYTHON_ARGV0,
        )
    assert caught.value.code == "GATE_GIT_CONFIG_EXTERNAL_INPUT"
    assert not sentinel.exists()


def test_linked_worktree_pointer_transient_restore_is_rejected(rig):
    repo, output, scratch = rig
    linked = repo.parent / "linked-transient"
    _git(
        repo,
        "worktree",
        "add",
        "-q",
        "-b",
        "linked-transient",
        str(linked),
    )
    git_dir = Path((linked / ".git").read_text().removeprefix("gitdir: ").strip())
    pointer = git_dir / "commondir"
    code = (
        "from pathlib import Path; "
        f"path = Path({str(pointer)!r}); "
        "original = path.read_bytes(); "
        "path.write_text('../invalid\\n'); "
        "path.write_bytes(original)"
    )
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="linked-pointer-transient",
            cwd=linked,
            temp_root=scratch / "linked-pointer-transient",
            timeout_seconds=10,
            argv=(_PYTHON_TARGET, "-c", code),
            process_argv0=_PYTHON_ARGV0,
        )
    assert caught.value.code == "GATE_SOURCE_DRIFT"
    assert not (output / "linked-pointer-transient").exists()


def test_same_named_branch_and_tag_keep_unambiguous_branch_identity(rig):
    repo, _output, _scratch = rig
    _git(repo, "checkout", "-q", "-b", "collision")
    _git(repo, "tag", "collision")
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "heads/collision"

    result = _run(rig, "ambiguous-short-branch", "print('branch-bound')")
    assert result.document["source"]["git"]["branch"] == "collision"
    assert result.document["source"]["git_administration"]["head"]["ref"] == (
        "refs/heads/collision"
    )
    assert gate.verify_generation(
        result.generation_path,
        require_current_source=True,
    )["status"] == "consistency_checked_unanchored"


def test_detached_head_administration_is_explicit(rig):
    repo, _output, _scratch = rig
    _git(repo, "checkout", "-q", "--detach")
    result = _run(rig, "detached-head", "print('detached')")
    admin = result.document["source"]["git_administration"]
    assert admin["head"]["kind"] == "detached"
    assert admin["active_ref"]["storage"] == "detached"
    assert admin["active_ref"]["name"] is None
    assert admin["head"]["oid"] == result.document["source"]["git"]["commit"]


def test_local_git_config_include_is_rejected_before_spawn(rig):
    repo, output, scratch = rig
    _git(repo, "config", "include.path", "/var/tmp/unretained-gate-config")
    sentinel = repo / "must-not-run-config-include"
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="config-include",
            cwd=repo,
            temp_root=scratch / "config-include",
            timeout_seconds=10,
            argv=(
                _PYTHON_TARGET,
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
            ),
        )
    assert caught.value.code == "GATE_GIT_CONFIG_EXTERNAL_INPUT"
    assert not sentinel.exists()
    assert not (output / "config-include").exists()


def test_git_config_keys_subsections_and_values_never_enter_receipt(rig):
    repo, _output, _scratch = rig
    planted_secret = "SECRET_TOKEN_VALUE"
    planted_key = "gate.secret-deadbeef"
    planted_subsection = "private-subsection-deadbeef"
    _git(
        repo,
        "config",
        f"remote.{planted_subsection}.url",
        f"https://user:{planted_secret}@example.invalid/repository",
    )
    _git(repo, "config", planted_key, "true")

    result = _run(rig, "config-value-privacy", "print('private-config-bound')")
    serialized = json.dumps(result.document, sort_keys=True)
    receipt_bytes = (result.generation_path / gate.RECEIPT_NAME).read_bytes()
    for private_text in (
        planted_secret,
        planted_key,
        planted_subsection,
        "remote.url",
    ):
        assert private_text not in serialized
        assert private_text.encode() not in receipt_bytes
    policy = result.document["source"]["git_administration"]["config"][
        "policy"
    ]
    assert policy == {
        "probe_environment": (
            "system_global_and_system_attributes_disabled_v1"
        ),
        "producer_check": (
            "include_and_core_excludes_attributes_external_inputs_rejected_v1"
        ),
        "privacy": "opaque_bindings_no_parsed_names_or_values_v1",
    }
    hidden = result.document["source"]["git_administration"][
        "hidden_admin_semantics"
    ]
    assert hidden["authority"] == "trusted_clean_source_producer_attestation_v1"


def test_transient_tracked_directory_swap_restore_is_rejected(rig):
    repo, output, _scratch = rig
    code = """
from pathlib import Path
original = Path('src-original')
active = Path('src')
active.rename(original)
active.mkdir()
replacement = active / 'nested.txt'
replacement.write_text('nested-replacement\\n')
print(replacement.read_text(), end='')
replacement.unlink()
active.rmdir()
original.rename(active)
"""
    with pytest.raises(gate.GateReceiptError) as caught:
        _run(rig, "source-directory-transient", code)
    assert caught.value.code == "GATE_SOURCE_DRIFT"
    assert (repo / "src" / "nested.txt").read_text() == "nested-stable\n"
    assert not (output / "source-directory-transient").exists()
    assert (output / ".source-directory-transient.staging").is_dir()


def test_current_source_check_is_optional_but_exact(rig):
    repo, _output, _scratch = rig
    result = _run(rig, "historical", "print('stable')")
    (repo / "tracked.txt").write_text("later\n")
    assert gate.verify_generation(result.generation_path)["status"] == "consistency_checked_unanchored"
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.verify_generation(result.generation_path, require_current_source=True)
    assert caught.value.code == "GATE_CURRENT_SOURCE_MISMATCH"


def test_current_source_rechecks_hidden_git_config_bytes(rig):
    repo, _output, _scratch = rig
    result = _run(rig, "config-current-source", "print('config')")
    _git(repo, "config", "gate.after-receipt", "true")

    replay = gate.verify_generation(result.generation_path)
    assert replay["status"] == "consistency_checked_unanchored"
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.verify_generation(
            result.generation_path,
            require_current_source=True,
        )
    assert caught.value.code == "GATE_CURRENT_SOURCE_MISMATCH"


def test_current_source_binds_clean_filter_equivalent_worktree_bytes(rig):
    repo, _output, _scratch = rig
    filter_program = repo / "filter.py"
    filter_program.write_text(
        "import sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(b'normalized\\n')\n"
    )
    _git(
        repo,
        "config",
        "filter.gate-equivalent.clean",
        "/usr/bin/python3 filter.py",
    )
    _git(
        repo,
        "config",
        "filter.gate-equivalent.required",
        "true",
    )
    (repo / ".gitattributes").write_text(
        "tracked.txt filter=gate-equivalent\n"
    )
    (repo / "tracked.txt").write_text("normalized\n")
    _git(repo, "add", ".gitattributes", "filter.py", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "filtered fixture")
    (repo / "tracked.txt").write_text("variant-AA\n")
    _git(repo, "add", "--refresh", "tracked.txt")
    assert _git(
        repo, "hash-object", "--path", "tracked.txt", "tracked.txt"
    ) == _git(repo, "rev-parse", ":tracked.txt")
    assert _git(repo, "status", "--porcelain=v1") == ""

    result = _run(rig, "filter-equivalent", "print('bytes-bound')")
    before = result.document["source"]["checked_out_worktree"]
    (repo / "tracked.txt").write_text("variant-BB\n")
    assert _git(repo, "status", "--porcelain=v1") == ""

    assert gate.verify_generation(result.generation_path)["status"] == "consistency_checked_unanchored"
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.verify_generation(
            result.generation_path,
            require_current_source=True,
        )
    assert caught.value.code == "GATE_CURRENT_SOURCE_MISMATCH"
    assert (
        gate._current_source_document(repo)["checked_out_worktree"]
        != before
    )


def test_current_source_binds_tracked_ancestor_directory_modes(rig):
    repo, _output, _scratch = rig
    result = _run(rig, "directory-mode", "print('directory-mode-bound')")
    before = result.document["source"]["checked_out_worktree"]

    (repo / "src").chmod(0o700)
    assert _git(repo, "status", "--porcelain=v1") == ""
    assert gate.verify_generation(result.generation_path)["status"] == "consistency_checked_unanchored"
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.verify_generation(
            result.generation_path,
            require_current_source=True,
        )
    assert caught.value.code == "GATE_CURRENT_SOURCE_MISMATCH"
    assert (
        gate._current_source_document(repo)["checked_out_worktree"]
        != before
    )


@pytest.mark.parametrize(
    ("indexed_executable", "actual_mode", "accepted"),
    (
        (False, 0o641, True),
        (False, 0o744, False),
        (True, 0o744, True),
        (True, 0o641, False),
    ),
)
def test_worktree_mode_contract_uses_git_owner_execute_bit(
    rig,
    indexed_executable,
    actual_mode,
    accepted,
):
    repo, output, scratch = rig
    path = repo / "mode.txt"
    path.write_text("mode-bound\n")
    path.chmod(0o755 if indexed_executable else 0o644)
    _git(repo, "add", "mode.txt")
    _git(repo, "commit", "-q", "-m", "mode fixture")
    _git(repo, "config", "core.filemode", "false")
    path.chmod(actual_mode)
    assert _git(repo, "status", "--porcelain=v1") == ""

    if accepted:
        result = _run(rig, f"mode-{actual_mode:o}", "print('mode-ok')")
        assert result.document["outcome"]["status"] == "passed"
        return

    sentinel = repo / "must-not-run-mode-mismatch"
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation=f"mode-{actual_mode:o}",
            cwd=repo,
            temp_root=scratch / f"mode-{actual_mode:o}",
            timeout_seconds=10,
            argv=(
                _PYTHON_TARGET,
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
            ),
        )
    assert caught.value.code == "GATE_SOURCE_LAYOUT_UNSUPPORTED"
    assert not sentinel.exists()


def test_producer_self_validation_refuses_before_any_receipt_bytes(
    rig,
    monkeypatch,
):
    _repo, output, _scratch = rig

    def reject_bundle(**_kwargs):
        raise gate.GateReceiptError(
            "GATE_TEST_SELF_VALIDATION",
            "planted producer self-validation refusal",
        )

    monkeypatch.setattr(gate, "_validate_receipt_bundle", reject_bundle)
    with pytest.raises(gate.GateReceiptError) as caught:
        _run(rig, "self-validation", "print('not publishable')")
    assert caught.value.code == "GATE_TEST_SELF_VALIDATION"
    assert not (output / "self-validation").exists()
    stage = output / ".self-validation.staging"
    assert stage.is_dir()
    assert tuple(stage.iterdir()) == ()


def test_current_source_index_format_rewrite_is_rejected(rig):
    repo, _output, _scratch = rig
    result = _run(rig, "index-format", "print('index-bound')")
    before = result.document["source"]["git_index"]["sha256"]

    _git(repo, "update-index", "--index-version=4")
    assert _git(repo, "status", "--porcelain=v1") == ""
    index_path = Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() != before

    with pytest.raises(gate.GateReceiptError) as caught:
        gate.verify_generation(
            result.generation_path,
            require_current_source=True,
        )
    assert caught.value.code == "GATE_CURRENT_SOURCE_MISMATCH"


def test_current_source_git_config_rewrite_is_rejected(rig):
    repo, _output, _scratch = rig
    result = _run(rig, "git-config-current", "print('config-bound')")
    _git(repo, "config", "gate.current-source-test", "changed")
    assert _git(repo, "status", "--porcelain=v1") == ""

    with pytest.raises(gate.GateReceiptError) as caught:
        gate.verify_generation(
            result.generation_path,
            require_current_source=True,
        )
    assert caught.value.code == "GATE_CURRENT_SOURCE_MISMATCH"


def test_generation_mutation_and_receipt_reseal_are_rejected(rig):
    result = _run(rig, "tamper", "print('original')")
    stdout = result.generation_path / gate.STDOUT_NAME
    os.chmod(result.generation_path, 0o700)
    os.chmod(stdout, 0o600)
    stdout.write_bytes(b"mutated\n")
    with pytest.raises(ImmutableArtifactError):
        gate.verify_generation(result.generation_path)

    forged = copy.deepcopy(result.document)
    forged.pop("receipt_sha256")
    forged["outcome"] = {"status": "passed", "reason": "NONZERO_EXIT"}
    forged = seal_model(forged, field="receipt_sha256")
    with pytest.raises(gate.GateReceiptError, match="outcome disagrees"):
        gate._validate_document(forged)


def test_resealed_producer_impossible_receipts_are_rejected(rig):
    result = _run(rig, "forgery-corpus", "print('sealed')")

    def dirty_source(document):
        status = "?? transient.txt\n"
        document["source"]["git"]["status_porcelain_v1"] = status
        document["source"]["git"]["status_sha256"] = hashlib.sha256(
            status.encode()
        ).hexdigest()
        document["source"]["git"]["clean"] = False

    def executable_symlink(document):
        document["executable"]["requested_binding"]["symlink_target"] = (
            "python-target"
        )

    def executable_mode(document):
        mode = str(stat.S_IFREG | 0o644)
        document["executable"]["mode"] = mode
        document["executable"]["requested_binding"]["mode"] = mode

    def executable_size(document):
        document["executable"]["size_bytes"] = str(
            int(document["executable"]["size_bytes"]) + 1
        )

    def git_index_symlink(document):
        document["source"]["git_index"]["mode"] = str(
            stat.S_IFLNK | 0o777
        )

    def temporary_mode(document):
        document["temporary_root"]["binding"]["mode"] = str(
            stat.S_IFDIR | 0o755
        )

    def stream_oversize(document):
        document["streams"]["stdout"]["size_bytes"] = (
            gate._MAX_STREAM_BYTES + 1
        )

    def impossible_tracked_path_bytes(document):
        tracked = document["source"]["tracked_inputs"]
        tracked["entry_count"] = 2
        tracked["relative_path_bytes"] = 1
        tracked["mode_counts"] = {"100644": 2, "100755": 0}

    def mixed_git_object_formats(document):
        document["source"]["git"]["commit"] = "a" * 41
        document["source"]["git"]["tree"] = "b" * 42

    def filesystem_root_temp(document):
        document["temporary_root"]["path"] = "/"
        document["temporary_root"]["binding"]["path"] = "/"

    def publication_inside_worktree(document):
        cwd = document["command"]["cwd"]
        document["publication"]["output_root"] = cwd
        document["publication"]["root_binding"]["path"] = cwd

    def set_temporary_path(document, path):
        document["temporary_root"]["path"] = path
        document["temporary_root"]["binding"]["path"] = path
        document["environment"].update({
            "TMP": f"{path}/tmp",
            "TEMP": f"{path}/tmp",
            "TMPDIR": f"{path}/tmp",
            "CARGO_TARGET_DIR": f"{path}/cargo-target",
            "PYTHONPYCACHEPREFIX": f"{path}/pycache",
        })

    def temporary_equals_output(document):
        set_temporary_path(
            document,
            document["publication"]["output_root"],
        )

    def temporary_ancestor_of_repository(document):
        repository = Path(document["source"]["git"]["repository_root"])
        set_temporary_path(document, str(repository.parent))

    def git_index_equals_temporary(document):
        document["source"]["git_index"]["path"] = document[
            "temporary_root"
        ]["path"]

    def executable_zero_inode(document):
        document["executable"]["requested_binding"]["inode"] = "0"

    def invalid_branch_text(document):
        document["source"]["git"]["branch"] = "invalid\nbranch"

    def contradictory_branch_identity(document):
        document["source"]["git"]["branch"] = "forged-other"

    def empty_git_head_binding(document):
        binding = document["source"]["git_administration"]["head"][
            "binding"
        ]
        binding["size_bytes"] = "0"
        binding["sha256"] = hashlib.sha256(b"").hexdigest()

    def impossible_manifest_size(document):
        tracked = document["source"]["tracked_inputs"]
        tracked["entry_count"] = gate._MAX_TRACKED_ENTRIES
        tracked["relative_path_bytes"] = (
            gate._MAX_TRACKED_RELATIVE_PATH_BYTES
        )
        tracked["maximum_relative_path_bytes"] = (
            gate._MAX_TRACKED_PATH_BYTES
        )
        tracked["flag_manifest_bytes"] = (
            tracked["relative_path_bytes"] + 3 * tracked["entry_count"]
        )
        tracked["mode_counts"] = {
            "100644": gate._MAX_TRACKED_ENTRIES,
            "100755": 0,
        }
        document["source"]["git"]["commit"] = "a" * 64
        document["source"]["git"]["tree"] = "b" * 64

    def positive_worktree_bytes_with_zero_maximum(document):
        worktree = document["source"]["checked_out_worktree"]
        worktree["total_file_bytes"] = 1
        worktree["maximum_file_bytes"] = 0

    def impossible_worktree_aggregate(document):
        worktree = document["source"]["checked_out_worktree"]
        worktree["maximum_file_bytes"] = 1
        worktree["total_file_bytes"] = worktree["entry_count"] + 1

    def oversized_tracked_path(document):
        tracked = document["source"]["tracked_inputs"]
        tracked["maximum_relative_path_bytes"] = (
            gate._MAX_TRACKED_PATH_BYTES + 1
        )

    def git_admin_layout_flip(document):
        document["source"]["git_administration"]["layout"] = (
            "linked_worktree"
        )

    def integrity_overclaim(document):
        document["integrity"]["authority"] = "self_authenticating"

    def git_admin_head_oid_mismatch(document):
        document["source"]["git_administration"]["head"]["oid"] = (
            "0" * 40
        )

    def git_admin_external_config(document):
        document["source"]["git_administration"]["config"]["common"][
            "path"
        ] = "/var/tmp/forged-git-config"

    def oversized_git_config_binding(document):
        binding = document["source"]["git_administration"]["config"][
            "common"
        ]
        binding["size_bytes"] = str(gate._MAX_GIT_CONFIG_RECEIPT_BYTES + 1)
        binding["sha256"] = "0" * 64

    def git_config_policy_is_unsupported(document):
        policy = document["source"]["git_administration"]["config"][
            "policy"
        ]
        policy["producer_check"] = "forged"

    def git_admin_packed_without_loose_absence(document):
        active = document["source"]["git_administration"]["active_ref"]
        active["storage"] = "packed_refs"

    def double_slash_temporary(document):
        set_temporary_path(document, "//var/tmp/forged-gate-temp")

    mutations = (
        ("dirty source", dirty_source),
        ("executable symlink", executable_symlink),
        ("non-executable mode", executable_mode),
        ("executable size mismatch", executable_size),
        ("Git index symlink", git_index_symlink),
        ("temporary mode", temporary_mode),
        ("stream oversize", stream_oversize),
        ("tracked path bytes", impossible_tracked_path_bytes),
        ("mixed Git object formats", mixed_git_object_formats),
        ("filesystem-root temp", filesystem_root_temp),
        ("publication inside worktree", publication_inside_worktree),
        ("temporary equals output", temporary_equals_output),
        ("temporary ancestor of repository", temporary_ancestor_of_repository),
        ("Git index equals temporary", git_index_equals_temporary),
        ("zero executable inode", executable_zero_inode),
        ("invalid branch text", invalid_branch_text),
        ("contradictory branch identity", contradictory_branch_identity),
        ("empty Git HEAD binding", empty_git_head_binding),
        ("impossible manifest size", impossible_manifest_size),
        (
            "positive worktree bytes with zero maximum",
            positive_worktree_bytes_with_zero_maximum,
        ),
        ("impossible worktree aggregate", impossible_worktree_aggregate),
        ("oversized tracked path", oversized_tracked_path),
        ("Git admin layout flip", git_admin_layout_flip),
        ("receipt integrity overclaim", integrity_overclaim),
        ("Git admin HEAD OID mismatch", git_admin_head_oid_mismatch),
        ("Git admin external config", git_admin_external_config),
        ("oversized Git config binding", oversized_git_config_binding),
        (
            "Git config policy is unsupported",
            git_config_policy_is_unsupported,
        ),
        (
            "Git admin packed storage without loose absence",
            git_admin_packed_without_loose_absence,
        ),
        ("double-slash temporary path", double_slash_temporary),
    )
    for label, mutate in mutations:
        forged = copy.deepcopy(result.document)
        mutate(forged)
        forged = _reseal(forged)
        try:
            gate._validate_document(forged)
        except gate.GateReceiptError:
            pass
        else:
            pytest.fail(f"resealed producer-impossible receipt passed: {label}")


def test_resealed_detached_receipt_rejects_reported_branch(rig):
    repo, _output, _scratch = rig
    _git(repo, "checkout", "-q", "--detach")
    result = _run(rig, "detached-branch-forgery", "print('detached')")
    forged = copy.deepcopy(result.document)
    forged["source"]["git"]["branch"] = "forged-branch"

    with pytest.raises(gate.GateReceiptError) as caught:
        gate._validate_document(_reseal(forged))
    assert caught.value.code == "GATE_RECEIPT_INVALID"


def test_resealed_packed_ref_receipt_rejects_empty_binding(rig):
    repo, _output, _scratch = rig
    _git(repo, "pack-refs", "--all")
    result = _run(rig, "packed-ref-empty-forgery", "print('packed')")
    active = result.document["source"]["git_administration"]["active_ref"]
    assert active["storage"] == "packed_refs"
    forged = copy.deepcopy(result.document)
    binding = forged["source"]["git_administration"]["active_ref"][
        "packed_refs"
    ]["binding"]
    binding["size_bytes"] = "0"
    binding["sha256"] = hashlib.sha256(b"").hexdigest()

    with pytest.raises(gate.GateReceiptError) as caught:
        gate._validate_document(_reseal(forged))
    assert caught.value.code == "GATE_RECEIPT_INVALID"


def test_duplicate_active_packed_ref_is_rejected_before_spawn(rig):
    repo, output, _scratch = rig
    _git(repo, "pack-refs", "--all")
    active_ref = _git(repo, "symbolic-ref", "HEAD")
    commit = _git(repo, "rev-parse", "HEAD")
    packed_path = repo / ".git" / "packed-refs"
    with packed_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{commit} {active_ref}\n")
    sentinel = repo / "must-not-run-duplicate-packed-ref"

    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="duplicate-packed-ref",
            cwd=repo,
            temp_root=rig[2] / "duplicate-packed-ref",
            timeout_seconds=10,
            argv=(
                _PYTHON_TARGET,
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
            ),
            process_argv0=_PYTHON_ARGV0,
        )
    assert caught.value.code in {
        "GATE_GIT_ADMIN_INVALID",
        "GATE_GIT_PROBE_FAILED",
    }
    assert not sentinel.exists()
    assert not (output / "duplicate-packed-ref").exists()


@pytest.mark.parametrize(
    "payload",
    (
        b"^not-an-object-id\n",
        b"0" * 40 + b" refs/tags/example\n^not-an-object-id\n",
        b"0" * 40 + b" refs/heads/main",
        b"# pack-refs with: peeled\n\n",
        b"# arbitrary header\n",
        b"0" * 40 + b" refs/tags/bad name\n",
        b"0" * 40 + b" refs/tags/bad.lock\n",
        b"0" * 40 + b" refs/tags/bad..name\n",
        b"0" * 40 + b" refs/tags/bad@{name\n",
        b"1" * 64 + b" refs/tags/wrong-object-format\n",
        (
            b"# pack-refs with: sorted \n"
            + b"0" * 40
            + b" refs/tags/unadvertised-peel\n^"
            + b"1" * 40
            + b"\n"
        ),
        (
            b"# pack-refs with: peeled fully-peeled sorted \n"
            + b"0" * 40
            + b" refs/tags/z\n"
            + b"1" * 40
            + b" refs/tags/a\n"
        ),
        b"0" * 40 + b" refs/heads/main\n# late comment\n",
    ),
)
def test_packed_refs_parser_rejects_malformed_hidden_rows(payload):
    with pytest.raises(gate.GateReceiptError) as caught:
        gate._validate_packed_refs_rows(payload, expected_oid_width=40)
    assert caught.value.code == "GATE_GIT_ADMIN_INVALID"


def test_packed_refs_parser_accepts_well_formed_peeled_row():
    payload = (
        b"# pack-refs with: peeled fully-peeled sorted \n"
        + b"0" * 40
        + b" refs/tags/example\n^"
        + b"1" * 40
        + b"\n"
    )
    gate._validate_packed_refs_rows(payload, expected_oid_width=40)
    gate._validate_packed_refs_rows(
        b"2" * 64 + b" refs/heads/sha256-main\n",
        expected_oid_width=64,
    )


def test_packed_refs_receipt_does_not_publish_inactive_ref_names(rig):
    repo, _output, _scratch = rig
    private_ref = "private-customer-token-name"
    _git(repo, "tag", private_ref)
    _git(repo, "pack-refs", "--all")

    result = _run(rig, "packed-ref-privacy", "print('packed')")
    active = result.document["source"]["git_administration"]["active_ref"]
    serialized = json.dumps(result.document, sort_keys=True)

    assert active["storage"] == "packed_refs"
    assert "packed_refs_witness" not in active
    assert "packed_refs_payload" not in serialized
    assert "row_manifest_sha256" not in serialized
    assert private_ref not in serialized


def test_current_source_rechecks_hidden_packed_ref_bytes(rig):
    repo, _output, _scratch = rig
    _git(repo, "pack-refs", "--all")
    result = _run(rig, "packed-ref-current-source", "print('packed')")
    packed_path = repo / ".git" / "packed-refs"
    packed_path.write_bytes(packed_path.read_bytes() + b"# later\n")

    replay = gate.verify_generation(result.generation_path)
    assert replay["status"] == "consistency_checked_unanchored"
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.verify_generation(
            result.generation_path,
            require_current_source=True,
        )
    assert caught.value.code == "GATE_CURRENT_SOURCE_MISMATCH"


@pytest.mark.parametrize("legacy_version", ("gate_receipt.v1", "gate_receipt.v2"))
def test_legacy_receipt_version_refuses_before_shape_validation(
    rig,
    legacy_version,
):
    result = _run(rig, f"legacy-version-{legacy_version[-2:]}", "print('current')")
    legacy = copy.deepcopy(result.document)
    legacy["schema_version"] = legacy_version
    legacy.pop("integrity")

    with pytest.raises(gate.GateReceiptError) as caught:
        gate._validate_document(_reseal(legacy))
    assert caught.value.code == "GATE_RECEIPT_VERSION_UNSUPPORTED"


def test_current_source_symlink_replacement_is_typed_as_mismatch(rig):
    repo, _output, _scratch = rig
    result = _run(rig, "source-symlink", "print('bound source')")
    displaced = repo.parent / "repo-displaced"
    repo.rename(displaced)
    repo.symlink_to(displaced, target_is_directory=True)

    with pytest.raises(gate.GateReceiptError) as caught:
        gate.verify_generation(
            result.generation_path,
            require_current_source=True,
        )
    assert caught.value.code == "GATE_CURRENT_SOURCE_MISMATCH"


def test_relocated_copy_cannot_impersonate_bound_generation(rig):
    result = _run(rig, "bound-path", "print('bound')")
    relocated = result.generation_path.parent / "relocated-copy"
    shutil.copytree(result.generation_path, relocated, copy_function=shutil.copy2)
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.verify_generation(relocated)
    assert caught.value.code == "GATE_GENERATION_PATH_MISMATCH"


def test_hardlinked_executable_is_rejected_without_spawn(rig):
    repo, output, scratch = rig
    executable_source = rig[0].parent / "python-copy"
    executable = rig[0].parent / "python-hardlink"
    shutil.copy2(Path(_PYTHON_TARGET).resolve(), executable_source)
    os.chmod(executable_source, 0o700)
    os.link(executable_source, executable)
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="hardlink",
            cwd=repo,
            temp_root=scratch / "hardlink",
            timeout_seconds=10,
            argv=(str(executable), "-c", "print('must not run')"),
        )
    assert caught.value.code == "GATE_EXECUTABLE_DRIFT"
    assert not (output / "hardlink").exists()


def test_secret_environment_names_and_worktree_temp_are_refused(rig):
    repo, output, scratch = rig
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="secret-env",
            cwd=repo,
            temp_root=scratch / "secret-env",
            timeout_seconds=10,
            argv=(_PYTHON_TARGET, "-c", "print('must not run')"),
            environment_assignments=("ACCESS_TOKEN=forbidden",),
        )
    assert caught.value.code == "GATE_ENVIRONMENT_SECRET_FORBIDDEN"

    external_output = repo.parent / "other-output"
    external_output.mkdir()
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=external_output,
            generation="worktree-temp",
            cwd=repo,
            temp_root=repo / "forbidden-temp",
            timeout_seconds=10,
            argv=(_PYTHON_TARGET, "-c", "print('must not run')"),
        )
    assert caught.value.code == "GATE_PATH_OVERLAP"


def test_canonical_target_and_separate_logical_argv0_are_bound(rig):
    repo, output, scratch = rig
    logical_argv0 = repo.parent / "python-logical"
    logical_argv0.symlink_to(_PYTHON_TARGET)
    result = gate.run_gate(
        output_root=output,
        generation="logical-argv0",
        cwd=repo,
        temp_root=scratch / "logical-argv0",
        timeout_seconds=10,
        argv=(
            _PYTHON_TARGET,
            "-c",
            "from pathlib import Path; print(Path('/proc/self/cmdline').read_bytes().split(b'\\0')[0].decode())",
        ),
        process_argv0=str(logical_argv0),
    )
    assert result.document["command"]["argv"][0] == str(logical_argv0)
    assert result.document["executable"]["requested_path"] == _PYTHON_TARGET
    assert result.document["executable"]["resolved_path"] == _PYTHON_TARGET
    assert result.document["executable"]["process_argv0"] == str(logical_argv0)
    assert (result.generation_path / gate.STDOUT_NAME).read_text().strip() == str(logical_argv0)

    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="symlink-target",
            cwd=repo,
            temp_root=scratch / "symlink-target",
            timeout_seconds=10,
            argv=(str(logical_argv0), "-c", "print('must not run')"),
        )
    assert caught.value.code == "GATE_EXECUTABLE_NOT_CANONICAL"


def test_environment_is_limited_to_the_audited_allowlist(rig):
    repo, output, scratch = rig
    with pytest.raises(gate.GateReceiptError) as caught:
        gate.run_gate(
            output_root=output,
            generation="unknown-env",
            cwd=repo,
            temp_root=scratch / "unknown-env",
            timeout_seconds=10,
            argv=(_PYTHON_TARGET, "-c", "print('must not run')"),
            environment_assignments=("UNREVIEWED_SWITCH=1",),
        )
    assert caught.value.code == "GATE_ENVIRONMENT_NOT_ALLOWED"


@pytest.mark.parametrize(
    ("descriptor", "stream_name"),
    ((1, gate.STDOUT_NAME), (2, gate.STDERR_NAME)),
)
def test_stream_exact_limit_is_admitted(
    rig,
    monkeypatch,
    descriptor,
    stream_name,
):
    monkeypatch.setattr(gate, "_MAX_STREAM_BYTES", 1024)
    result = _run(
        rig,
        "stream-exact-limit",
        f"import os; [(os.write({descriptor}, b'x' * 128)) for _ in range(8)]",
    )
    assert result.shell_return_code == 0
    assert (result.generation_path / stream_name).read_bytes() == b"x" * 1024


@pytest.mark.parametrize(
    ("descriptor", "stream_role"),
    ((1, "stdout"), (2, "stderr")),
)
def test_fragmented_limit_plus_one_is_enforced_while_child_is_live(
    rig,
    monkeypatch,
    descriptor,
    stream_role,
):
    _repo, output, _scratch = rig
    monkeypatch.setattr(gate, "_MAX_STREAM_BYTES", 1024)
    with pytest.raises(gate.GateReceiptError) as caught:
        _run(
            rig,
            "stream-limit",
            f"import os, time; "
            f"[(os.write({descriptor}, b'x' * 128)) for _ in range(8)]; "
            f"os.write({descriptor}, b'y'); time.sleep(10)",
            timeout_seconds=2,
        )
    assert caught.value.code == "GATE_STREAM_OVERSIZED"
    assert caught.value.context["stream"] == stream_role
    assert not (output / "stream-limit").exists()
    assert (output / ".stream-limit.staging").is_dir()


def test_large_binary_dual_streams_are_complete_and_fair(rig):
    chunk = bytes(range(256)) * 4096
    reverse = chunk[::-1]
    result = _run(
        rig,
        "dual-binary-streams",
        """
import os

def write_all(descriptor, payload):
    view = memoryview(payload)
    while view:
        view = view[os.write(descriptor, view):]

chunk = bytes(range(256)) * 4096
for _ in range(2):
    write_all(1, chunk)
    write_all(2, chunk[::-1])
""",
    )
    assert (result.generation_path / gate.STDOUT_NAME).read_bytes() == chunk * 2
    assert (result.generation_path / gate.STDERR_NAME).read_bytes() == reverse * 2


def test_child_cannot_seek_truncate_pwrite_or_reopen_parent_stream(rig):
    result = _run(
        rig,
        "socket-ownership",
        """
import json
import os

results = {}
for name, operation in (
    ('lseek', lambda: os.lseek(1, 0, os.SEEK_SET)),
    ('ftruncate', lambda: os.ftruncate(1, 0)),
    ('pwrite', lambda: os.pwrite(1, b'forged', 0)),
    ('reopen', lambda: os.open('/proc/self/fd/1', os.O_RDONLY)),
):
    try:
        descriptor = operation()
    except OSError as error:
        results[name] = {'admitted': False, 'errno': error.errno}
    else:
        results[name] = {'admitted': True, 'value': descriptor}
        if name == 'reopen':
            os.close(descriptor)
try:
    stolen = os.read(1, 1)
except OSError as error:
    results['read'] = {'bytes': None, 'errno': error.errno}
else:
    results['read'] = {'bytes': stolen.hex(), 'errno': None}
print(json.dumps(results, sort_keys=True))
""",
    )
    observed = json.loads((result.generation_path / gate.STDOUT_NAME).read_text())
    for name in ("lseek", "ftruncate", "pwrite", "reopen"):
        assert observed[name]["admitted"] is False
        assert observed[name]["errno"] is not None
    assert observed["read"]["bytes"] in {None, ""}


def test_simultaneous_stream_flood_near_timeout_is_bounded(rig):
    result = _run(
        rig,
        "dual-stream-timeout",
        """
import os
import time
while True:
    os.write(1, b'o' * 4096)
    os.write(2, b'e' * 4096)
    time.sleep(0.001)
""",
        timeout_seconds=0.05,
    )
    assert result.shell_return_code == 124
    assert result.document["outcome"]["reason"] == "TIMEOUT"
    assert 0 < (result.generation_path / gate.STDOUT_NAME).stat().st_size
    assert 0 < (result.generation_path / gate.STDERR_NAME).stat().st_size


def test_stream_eof_does_not_replace_exact_child_timeout(rig):
    result = _run(
        rig,
        "early-stream-eof",
        "import os, time; os.close(1); os.close(2); time.sleep(30)",
        timeout_seconds=0.05,
    )
    assert result.shell_return_code == 124
    assert result.document["outcome"] == {
        "status": "failed",
        "reason": "TIMEOUT",
    }
    assert (result.generation_path / gate.STDOUT_NAME).read_bytes() == b""
    assert (result.generation_path / gate.STDERR_NAME).read_bytes() == b""


def test_unix_urgent_data_is_inlined_into_stream_identity(rig, monkeypatch):
    monkeypatch.setattr(gate, "_MAX_STREAM_BYTES", 3)
    result = _run(
        rig,
        "socket-urgent-inline",
        """
import socket
stream = socket.socket(fileno=1)
stream.sendall(b'A')
stream.send(b'B', socket.MSG_OOB)
stream.sendall(b'C')
stream.detach()
""",
    )
    assert (result.generation_path / gate.STDOUT_NAME).read_bytes() == b"ABC"


def test_unix_urgent_data_counts_toward_stream_limit(rig, monkeypatch):
    _repo, output, _scratch = rig
    monkeypatch.setattr(gate, "_MAX_STREAM_BYTES", 2)
    with pytest.raises(gate.GateReceiptError) as caught:
        _run(
            rig,
            "socket-urgent-limit",
            """
import socket
import time
stream = socket.socket(fileno=1)
stream.sendall(b'A')
stream.send(b'B', socket.MSG_OOB)
stream.sendall(b'C')
time.sleep(30)
""",
        )
    assert caught.value.code == "GATE_STREAM_OVERSIZED"
    assert not (output / "socket-urgent-limit").exists()


@pytest.mark.parametrize("stream", ("stdout", "stderr"))
@pytest.mark.parametrize("extra_byte", (False, True))
def test_git_probe_process_output_is_bounded_before_allocation(
    tmp_path,
    monkeypatch,
    stream,
    extra_byte,
):
    byte_count = (
        gate._MAX_GIT_PROCESS_STDOUT_BYTES
        if stream == "stdout"
        else gate._MAX_GIT_PROCESS_STDERR_BYTES
    ) + int(extra_byte)
    descriptor = 1 if stream == "stdout" else 2
    fake_git = tmp_path / f"fake-git-{stream}-{byte_count}"
    fake_git.write_text(
        "#!/usr/bin/python3\n"
        "import sys\n"
        f"stream = sys.stdout.buffer if {descriptor} == 1 else sys.stderr.buffer\n"
        f"stream.write(b'x' * {byte_count})\n"
        "stream.flush()\n"
    )
    fake_git.chmod(0o755)
    monkeypatch.setattr(
        gate.shutil,
        "which",
        lambda _name, path=None: str(fake_git),
    )

    if extra_byte:
        with pytest.raises(gate.GateReceiptError) as caught:
            gate._run_git(tmp_path, ("ignored",))
        assert caught.value.code == "GATE_GIT_OUTPUT_OVERSIZED"
        return

    return_code, payload = gate._run_git(tmp_path, ("ignored",))
    assert return_code == 0
    assert len(payload) == (byte_count if stream == "stdout" else 0)


def test_git_probe_setup_failure_terminates_and_reaps_process_group(
    tmp_path,
    monkeypatch,
):
    fake_git = tmp_path / "fake-git-setup-failure"
    fake_git.write_text(
        "#!/usr/bin/python3\n"
        "import time\n"
        "time.sleep(60)\n"
    )
    fake_git.chmod(0o755)
    monkeypatch.setattr(
        gate.shutil,
        "which",
        lambda _name, path=None: str(fake_git),
    )
    original_init = gate.subprocess.Popen.__init__
    processes = []
    spawned = False

    def capture_process(self, *args, **kwargs):
        nonlocal spawned
        original_init(self, *args, **kwargs)
        processes.append(self)
        spawned = True

    original_set_blocking = gate.os.set_blocking
    injected = False

    def fail_transport_setup(descriptor, blocking):
        nonlocal injected
        if spawned and not injected:
            injected = True
            raise OSError(5, "planted Git transport setup failure")
        return original_set_blocking(descriptor, blocking)

    monkeypatch.setattr(gate.subprocess.Popen, "__init__", capture_process)
    monkeypatch.setattr(gate.os, "set_blocking", fail_transport_setup)

    with pytest.raises(gate.GateReceiptError) as caught:
        gate._run_git(tmp_path, ("ignored",))
    assert caught.value.code == "GATE_GIT_UNAVAILABLE"
    assert injected is True
    assert len(processes) == 1
    assert processes[0].returncode is not None
    assert not Path(f"/proc/{processes[0].pid}").exists()


def test_git_probe_post_spawn_interrupt_terminates_and_reaps_group(
    tmp_path,
    monkeypatch,
):
    fake_git = tmp_path / "fake-git-post-spawn-interrupt"
    fake_git.write_text(
        "#!/usr/bin/python3\n"
        "import time\n"
        "time.sleep(60)\n"
    )
    fake_git.chmod(0o755)
    monkeypatch.setattr(
        gate.shutil,
        "which",
        lambda _name, path=None: str(fake_git),
    )
    original_init = gate.subprocess.Popen.__init__
    child_pid = None

    def interrupt_after_spawn(self, *args, **kwargs):
        nonlocal child_pid
        original_init(self, *args, **kwargs)
        child_pid = self.pid
        raise KeyboardInterrupt("planted Git post-spawn interruption")

    monkeypatch.setattr(
        gate.subprocess.Popen,
        "__init__",
        interrupt_after_spawn,
    )
    with pytest.raises(KeyboardInterrupt, match="Git post-spawn interruption"):
        gate._run_git(tmp_path, ("ignored",))
    assert child_pid is not None
    assert not Path(f"/proc/{child_pid}").exists()


def test_git_constructor_post_reap_interruption_never_signals_numeric_group(
    tmp_path,
    monkeypatch,
):
    fake_git = tmp_path / "fake-git-constructor-post-reap"
    fake_git.write_text("#!/usr/bin/python3\n")
    fake_git.chmod(0o755)
    monkeypatch.setattr(
        gate.shutil,
        "which",
        lambda _name, path=None: str(fake_git),
    )
    original_init = gate.subprocess.Popen.__init__
    original_waitpid = gate.os.waitpid
    original_killpg = gate.os.killpg
    reaped = False
    kill_calls = []

    def reap_inside_constructor_then_interrupt(self, *args, **kwargs):
        nonlocal reaped
        original_init(self, *args, **kwargs)
        waited_pid, _wait_status = original_waitpid(self.pid, 0)
        assert waited_pid == self.pid
        self.returncode = None
        reaped = True
        raise KeyboardInterrupt("planted Git constructor post-reap interruption")

    def record_killpg(pid, requested_signal):
        kill_calls.append((pid, reaped))
        return original_killpg(pid, requested_signal)

    monkeypatch.setattr(
        gate.subprocess.Popen,
        "__init__",
        reap_inside_constructor_then_interrupt,
    )
    monkeypatch.setattr(gate.os, "killpg", record_killpg)

    with pytest.raises(KeyboardInterrupt, match="constructor post-reap"):
        gate._run_git(tmp_path, ("ignored",))
    assert reaped is True
    assert kill_calls == []


def test_successful_git_probe_kills_forked_pipe_holder(
    tmp_path,
    monkeypatch,
):
    child_pid_path = tmp_path / "forked-child.pid"
    fake_git = tmp_path / "fake-git-forked-pipe-holder"
    fake_git.write_text(
        "#!/usr/bin/python3\n"
        "import os, pathlib, time\n"
        f"pid_path = pathlib.Path({str(child_pid_path)!r})\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    pid_path.write_text(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "while not pid_path.exists():\n"
        "    time.sleep(0.001)\n"
        "os._exit(0)\n"
    )
    fake_git.chmod(0o755)
    monkeypatch.setattr(
        gate.shutil,
        "which",
        lambda _name, path=None: str(fake_git),
    )
    monkeypatch.setattr(gate, "_MAX_GIT_PROCESS_SECONDS", 0.5)
    started = time.monotonic()
    return_code, payload = gate._run_git(tmp_path, ("ignored",))
    elapsed = time.monotonic() - started

    assert return_code == 0
    assert payload == b""
    assert elapsed < 0.5
    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()


def test_git_probe_timeout_kills_running_leader(
    tmp_path,
    monkeypatch,
):
    fake_git = tmp_path / "fake-git-running-leader"
    fake_git.write_text(
        "#!/usr/bin/python3\n"
        "import time\n"
        "time.sleep(60)\n"
    )
    fake_git.chmod(0o755)
    monkeypatch.setattr(
        gate.shutil,
        "which",
        lambda _name, path=None: str(fake_git),
    )
    monkeypatch.setattr(gate, "_MAX_GIT_PROCESS_SECONDS", 0.1)
    original_init = gate.subprocess.Popen.__init__
    processes = []

    def capture_process(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        processes.append(self)

    monkeypatch.setattr(gate.subprocess.Popen, "__init__", capture_process)
    with pytest.raises(gate.GateReceiptError) as caught:
        gate._run_git(tmp_path, ("ignored",))
    assert caught.value.code == "GATE_GIT_UNAVAILABLE"
    assert len(processes) == 1
    assert processes[0].returncode is not None
    assert not Path(f"/proc/{processes[0].pid}").exists()


def test_successful_git_probe_kills_residual_closed_pipe_descendant(
    tmp_path,
    monkeypatch,
):
    child_pid_path = tmp_path / "closed-pipe-child.pid"
    fake_git = tmp_path / "fake-git-closed-pipe-descendant"
    fake_git.write_text(
        "#!/usr/bin/python3\n"
        "import os, pathlib, time\n"
        f"pid_path = pathlib.Path({str(child_pid_path)!r})\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    pid_path.write_text(str(os.getpid()))\n"
        "    os.close(1)\n"
        "    os.close(2)\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "while not pid_path.exists():\n"
        "    time.sleep(0.001)\n"
        "os._exit(0)\n"
    )
    fake_git.chmod(0o755)
    monkeypatch.setattr(
        gate.shutil,
        "which",
        lambda _name, path=None: str(fake_git),
    )

    return_code, payload = gate._run_git(tmp_path, ("ignored",))
    assert return_code == 0
    assert payload == b""
    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()


def test_git_post_reap_exception_never_resignals_numeric_group(
    tmp_path,
    monkeypatch,
):
    fake_git = tmp_path / "fake-git-post-reap"
    fake_git.write_text("#!/usr/bin/python3\n")
    fake_git.chmod(0o755)
    monkeypatch.setattr(
        gate.shutil,
        "which",
        lambda _name, path=None: str(fake_git),
    )
    original_reap = gate._reap_git_process_leader
    original_killpg = gate.os.killpg
    reaped = False
    kill_calls = 0

    def reap_then_raise(process):
        nonlocal reaped
        original_reap(process)
        reaped = True
        raise RuntimeError("planted Git post-reap failure")

    def reject_post_reap_signal(pid, requested_signal):
        nonlocal kill_calls
        assert reaped is False
        kill_calls += 1
        return original_killpg(pid, requested_signal)

    monkeypatch.setattr(gate, "_reap_git_process_leader", reap_then_raise)
    monkeypatch.setattr(gate.os, "killpg", reject_post_reap_signal)

    with pytest.raises(RuntimeError, match="Git post-reap failure"):
        gate._run_git(tmp_path, ("ignored",))
    assert reaped is True
    assert kill_calls == 1


def test_git_waitpid_post_reap_exception_never_resignals_numeric_group(
    tmp_path,
    monkeypatch,
):
    fake_git = tmp_path / "fake-git-waitpid-post-reap"
    fake_git.write_text("#!/usr/bin/python3\n")
    fake_git.chmod(0o755)
    monkeypatch.setattr(
        gate.shutil,
        "which",
        lambda _name, path=None: str(fake_git),
    )
    original_waitpid = gate.os.waitpid
    original_killpg = gate.os.killpg
    reaped = False
    kill_calls = []

    def waitpid_then_raise(pid, options):
        nonlocal reaped
        result = original_waitpid(pid, options)
        reaped = True
        raise RuntimeError("planted Git waitpid post-reap failure")

    def record_killpg(pid, requested_signal):
        kill_calls.append((pid, reaped))
        return original_killpg(pid, requested_signal)

    monkeypatch.setattr(gate.os, "waitpid", waitpid_then_raise)
    monkeypatch.setattr(gate.os, "killpg", record_killpg)

    with pytest.raises(RuntimeError, match="Git waitpid post-reap failure"):
        gate._run_git(tmp_path, ("ignored",))
    assert reaped is True
    assert len(kill_calls) == 1
    assert kill_calls[0][1] is False


def test_terminal_observation_wins_at_deadline_boundary():
    assert gate._supervision_decision(
        terminal_observed=True,
        now_ns=99,
        deadline_ns=100,
    ) == "terminal"
    assert gate._supervision_decision(
        terminal_observed=True,
        now_ns=100,
        deadline_ns=100,
    ) == "terminal"
    assert gate._supervision_decision(
        terminal_observed=True,
        now_ns=101,
        deadline_ns=100,
    ) == "timeout"
    assert gate._supervision_decision(
        terminal_observed=False,
        now_ns=100,
        deadline_ns=100,
    ) == "timeout"


def test_realtime_backstep_is_observational_not_ordering_authority(
    rig,
    monkeypatch,
):
    realtime_values = iter((2_000_000_000, 1_000_000_000))
    monkeypatch.setattr(gate, "_realtime_ns", lambda: next(realtime_values))
    result = _run(rig, "realtime-backstep", "print('monotonic authority')")

    timing = result.document["time"]
    assert int(timing["started_at_unix_ns"]) > int(
        timing["finished_at_unix_ns"]
    )
    assert result.document["supervision"]["realtime"] == (
        "timestamps_observational_monotonic_only_ordering_v1"
    )
    assert gate.verify_generation(result.generation_path)["status"] == "consistency_checked_unanchored"


def test_preowned_popen_interrupt_cleans_exact_child(
    rig,
    monkeypatch,
):
    original_init = gate.subprocess.Popen.__init__
    original_kill_group = gate._kill_group
    original_waitid = gate.os.waitid
    original_wait4 = gate.os.wait4
    child_pid: int | None = None
    phases: list[str] = []

    def init_then_interrupt(self, *args, **kwargs):
        nonlocal child_pid
        original_init(self, *args, **kwargs)
        if (
            kwargs.get("start_new_session") is True
            and kwargs.get("stdout") != subprocess.PIPE
        ):
            child_pid = self.pid
            raise KeyboardInterrupt("planted post-exec interruption")

    def observed_kill_group(pid):
        phases.append(f"killpg:{pid}")
        return original_kill_group(pid)

    def observed_waitid(idtype, pid, options):
        if (
            child_pid is not None
            and pid == child_pid
            and options & gate.os.WNOWAIT
        ):
            phases.append(f"waitid:{pid}")
        return original_waitid(idtype, pid, options)

    def observed_wait4(pid, options):
        phases.append(f"wait4:{pid}")
        return original_wait4(pid, options)

    monkeypatch.setattr(gate.subprocess.Popen, "__init__", init_then_interrupt)
    monkeypatch.setattr(gate, "_kill_group", observed_kill_group)
    monkeypatch.setattr(gate.os, "waitid", observed_waitid)
    monkeypatch.setattr(gate.os, "wait4", observed_wait4)

    with pytest.raises(KeyboardInterrupt, match="post-exec interruption"):
        _run(rig, "popen-interrupt", "import time; time.sleep(30)")

    assert child_pid is not None
    assert phases == [
        f"killpg:{child_pid}",
        f"waitid:{child_pid}",
        f"wait4:{child_pid}",
    ]
    assert not Path(f"/proc/{child_pid}").exists()


def test_wait4_post_reap_exception_never_signals_numeric_group(
    rig,
    monkeypatch,
):
    original_wait4 = gate.os.wait4
    original_kill_group = gate._kill_group
    reaped = False
    kill_calls = 0

    def reap_then_raise(pid, options):
        nonlocal reaped
        original_wait4(pid, options)
        reaped = True
        raise RuntimeError("planted post-reap exception")

    def reject_post_reap_signal(pid):
        nonlocal kill_calls
        assert reaped is False
        kill_calls += 1
        return original_kill_group(pid)

    monkeypatch.setattr(gate.os, "wait4", reap_then_raise)
    monkeypatch.setattr(gate, "_kill_group", reject_post_reap_signal)

    with pytest.raises(gate.GateReceiptError) as caught:
        _run(rig, "wait4-post-reap", "print('reaped')")
    assert caught.value.code == "GATE_CLEANUP_AUTHORITY_LOST"
    assert kill_calls == 1


def test_cleanup_continues_when_signal_mask_deferral_fails(
    rig,
    monkeypatch,
):
    original_block = gate._block_async_signals
    original_init = gate.subprocess.Popen.__init__
    block_calls = 0
    child_pid: int | None = None

    def fail_after_launch():
        nonlocal block_calls
        block_calls += 1
        if block_calls >= 2:
            raise gate.GateReceiptError(
                "GATE_TEST_MASK_FAILURE",
                "planted signal-mask failure",
            )
        return original_block()

    def capture_child(self, *args, **kwargs):
        nonlocal child_pid
        original_init(self, *args, **kwargs)
        if kwargs.get("start_new_session") is True:
            child_pid = self.pid

    monkeypatch.setattr(gate, "_block_async_signals", fail_after_launch)
    monkeypatch.setattr(gate.subprocess.Popen, "__init__", capture_child)

    with pytest.raises(gate.GateReceiptError) as caught:
        _run(rig, "mask-cleanup", "print('mask cleanup')")
    assert caught.value.code == "GATE_CLEANUP_AUTHORITY_LOST"
    assert child_pid is not None
    assert not Path(f"/proc/{child_pid}").exists()


def test_group_kill_exception_still_reaps_exact_child(
    rig,
    monkeypatch,
):
    original_kill_group = gate._kill_group
    original_init = gate.subprocess.Popen.__init__
    child_pid: int | None = None

    def kill_then_raise(pid):
        original_kill_group(pid)
        raise RuntimeError("planted post-kill exception")

    def capture_child(self, *args, **kwargs):
        nonlocal child_pid
        original_init(self, *args, **kwargs)
        if kwargs.get("start_new_session") is True:
            child_pid = self.pid

    monkeypatch.setattr(gate, "_kill_group", kill_then_raise)
    monkeypatch.setattr(gate.subprocess.Popen, "__init__", capture_child)

    with pytest.raises(gate.GateReceiptError) as caught:
        _run(rig, "kill-cleanup", "print('kill cleanup')")
    assert caught.value.code == "GATE_CLEANUP_AUTHORITY_LOST"
    assert child_pid is not None
    assert not Path(f"/proc/{child_pid}").exists()


def test_waitid_cleanup_exception_still_reaps_exact_child(
    rig,
    monkeypatch,
):
    original_waitid = gate.os.waitid
    original_init = gate.subprocess.Popen.__init__
    failed_blocking_wait = False
    child_pid: int | None = None

    def fail_first_blocking_waitid(idtype, pid, options):
        nonlocal failed_blocking_wait
        if not options & gate.os.WNOHANG and not failed_blocking_wait:
            failed_blocking_wait = True
            raise RuntimeError("planted blocking waitid exception")
        return original_waitid(idtype, pid, options)

    def capture_child(self, *args, **kwargs):
        nonlocal child_pid
        original_init(self, *args, **kwargs)
        if kwargs.get("start_new_session") is True:
            child_pid = self.pid

    monkeypatch.setattr(gate.os, "waitid", fail_first_blocking_waitid)
    monkeypatch.setattr(gate.subprocess.Popen, "__init__", capture_child)

    with pytest.raises(gate.GateReceiptError) as caught:
        _run(
            rig,
            "waitid-cleanup",
            "import time; time.sleep(30)",
            timeout_seconds=0.05,
        )
    assert caught.value.code == "GATE_CLEANUP_AUTHORITY_LOST"
    assert failed_blocking_wait is True
    assert child_pid is not None
    assert not Path(f"/proc/{child_pid}").exists()


@pytest.mark.parametrize(
    "termination_signal",
    (signal.SIGHUP, signal.SIGQUIT, signal.SIGTERM),
)
def test_default_termination_signal_reaps_child_before_propagation(
    rig,
    termination_signal,
):
    repo, output, scratch = rig
    child_pid_path = repo.parent / f"signal-{termination_signal}.pid"
    child_code = (
        "import os, time; from pathlib import Path; "
        f"Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    script = (
        "from tools.dev import gate_receipt as gate; "
        f"gate.run_gate(output_root={str(output)!r}, "
        f"generation='signal-{termination_signal}', cwd={str(repo)!r}, "
        f"temp_root={str(scratch / f'signal-{termination_signal}')!r}, "
        "timeout_seconds=30, "
        f"argv=({_PYTHON_TARGET!r}, '-c', {child_code!r}), "
        f"process_argv0={_PYTHON_ARGV0!r})"
    )
    supervisor = subprocess.Popen(
        [_PYTHON_TARGET, "-c", script],
        cwd=Path(gate.__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/var/tmp"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    child_pid: int | None = None
    try:
        for _ in range(500):
            if child_pid_path.exists():
                child_pid = int(child_pid_path.read_text())
                break
            if supervisor.poll() is not None:
                stdout, stderr = supervisor.communicate()
                pytest.fail(
                    "gate supervisor exited before child marker: "
                    f"returncode={supervisor.returncode}, stdout={stdout!r}, "
                    f"stderr={stderr!r}"
                )
            time.sleep(0.01)
        else:
            pytest.fail("gate child PID marker did not appear")

        os.kill(supervisor.pid, termination_signal)
        stdout, stderr = supervisor.communicate(timeout=15)
        assert supervisor.returncode == -termination_signal
        assert stdout == b""
        assert stderr == b""
        assert child_pid is not None
        for _ in range(200):
            if not Path(f"/proc/{child_pid}").exists():
                break
            time.sleep(0.01)
        else:
            pytest.fail("supervised child survived propagated termination signal")
        assert not (output / f"signal-{termination_signal}").exists()
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()
        if child_pid is not None and Path(f"/proc/{child_pid}").exists():
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("disposition", (signal.SIG_IGN, lambda *_args: None))
def test_nondefault_sigchld_is_refused_before_spawn(disposition):
    previous = signal.signal(signal.SIGCHLD, disposition)
    try:
        with pytest.raises(gate.GateReceiptError) as caught:
            gate._require_exclusive_reaper_authority()
    finally:
        signal.signal(signal.SIGCHLD, previous)
    assert caught.value.code == "GATE_REAPER_AUTHORITY_UNAVAILABLE"


def test_original_process_group_survivor_is_killed(rig):
    result = _run(
        rig,
        "group-cleanup",
        "import os, time; pid = os.fork(); "
        "(time.sleep(30), os._exit(0)) if pid == 0 else print(pid, flush=True)",
    )
    descendant_pid = int((result.generation_path / gate.STDOUT_NAME).read_text())
    for _ in range(100):
        stat_path = Path(f"/proc/{descendant_pid}/stat")
        if not stat_path.exists():
            break
        fields = stat_path.read_text().split()
        if len(fields) > 2 and fields[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("same-process-group descendant survived gate cleanup")


def test_escaped_writer_drain_timeout_never_signals_after_leader_reap(
    rig,
    monkeypatch,
):
    repo, output, _scratch = rig
    pid_path = repo.parent / "escaped-writer.pid"
    original_wait4 = gate.os.wait4
    original_kill_group = gate._kill_group
    leader_reaped = False
    kill_calls = 0

    def observed_wait4(pid, options):
        nonlocal leader_reaped
        result = original_wait4(pid, options)
        leader_reaped = True
        return result

    def reject_post_reap_signal(pid):
        nonlocal kill_calls
        assert leader_reaped is False
        kill_calls += 1
        return original_kill_group(pid)

    monkeypatch.setattr(gate, "_PIPE_DRAIN_TIMEOUT_NS", 50_000_000)
    monkeypatch.setattr(gate.os, "wait4", observed_wait4)
    monkeypatch.setattr(gate, "_kill_group", reject_post_reap_signal)
    code = f"""
import os
import time
from pathlib import Path

marker = Path({str(pid_path)!r})
child = os.fork()
if child == 0:
    os.setsid()
    marker.write_text(str(os.getpid()))
    time.sleep(30)
    os._exit(0)
while not marker.exists():
    time.sleep(0.001)
"""
    escaped_pid: int | None = None
    try:
        with pytest.raises(gate.GateReceiptError) as caught:
            _run(rig, "escaped-writer", code)
        assert caught.value.code == "GATE_STREAM_DRAIN_TIMEOUT"
        assert leader_reaped is True
        assert kill_calls == 1
        assert pid_path.is_file()
        escaped_pid = int(pid_path.read_text())
        assert Path(f"/proc/{escaped_pid}").exists()
        assert not (output / "escaped-writer").exists()
    finally:
        if escaped_pid is None and pid_path.exists():
            escaped_pid = int(pid_path.read_text())
        if escaped_pid is not None:
            try:
                os.kill(escaped_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_receipt_discloses_setsid_escape_boundary(rig):
    result = _run(rig, "containment-policy", "print('policy')")
    assert result.document["supervision"]["containment"] == (
        "original_process_group_only_descendant_setsid_or_setpgid_escape_not_excluded"
    )
    assert result.document["supervision"]["temporary_paths"] == (
        "fresh_declared_cache_roots_only_not_complete_cache_isolation"
    )


def test_verifier_cli_propagates_recorded_failure_status(rig):
    result = _run(rig, "verify-exit-7", "raise SystemExit(7)")
    completed = subprocess.run(
        [
            _PYTHON_TARGET,
            str(Path(gate.__file__).resolve()),
            "verify",
            str(result.generation_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert completed.returncode == 7
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["shell_return_code"] == 7


def test_external_rig_ignores_ambient_temp_inside_worktree(rig, monkeypatch):
    ambient_worktree = rig[0]
    for name in ("TMP", "TEMP", "TMPDIR"):
        monkeypatch.setenv(name, str(ambient_worktree))
    nested_root: Path | None = None
    with _external_rig() as nested:
        nested_root = nested[0].parent
        assert nested_root.parent == Path("/var/tmp")
        assert not gate.protected_paths.is_inside_git_worktree(nested[1])
        assert _run(nested, "ambient-independent", "print('external')").shell_return_code == 0
    assert nested_root is not None
    assert not nested_root.exists()


def test_protected_path_indeterminacy_is_typed_before_spawn(
    rig,
    monkeypatch,
    capsys,
):
    repo, output, scratch = rig
    sentinel = repo / "must-not-exist-protected-path"

    def unavailable(_path):
        raise gate.protected_paths.ProtectedPathError("indeterminate fixture")

    monkeypatch.setattr(gate.protected_paths, "is_inside_git_worktree", unavailable)
    return_code = gate.main([
        "run",
        "--output-root", str(output),
        "--generation", "protected-indeterminate",
        "--cwd", str(repo),
        "--temp-root", str(scratch / "protected-indeterminate"),
        "--timeout-seconds", "10",
        "--",
        _PYTHON_TARGET,
        "-c",
        f"from pathlib import Path; Path({str(sentinel)!r}).write_text('bad')",
    ])
    captured = capsys.readouterr()
    assert return_code == 125
    assert captured.out == ""
    assert json.loads(captured.err)["typed_error"] == (
        "GATE_EXTERNAL_PATH_STATUS_UNAVAILABLE"
    )
    assert not sentinel.exists()
    assert not (output / "protected-indeterminate").exists()
    assert not (scratch / "protected-indeterminate").exists()
