from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import verify_cutlass_provenance as verifier  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q")
    _git(upstream, "config", "user.email", "test@example.invalid")
    _git(upstream, "config", "user.name", "CUTLASS provenance test")
    files = {
        "examples/example.cu": b"int main() { return 0; }\n",
        "include/cutlass/version.h": (
            b"#define CUTLASS_MAJOR 4\n"
            b"#define CUTLASS_MINOR 6\n"
            b"#define CUTLASS_PATCH 1\n"
        ),
        "tools/tool.py": b"#!/usr/bin/env python3\nprint('tool')\n",
    }
    for relative, payload in files.items():
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (upstream / "tools/tool.py").chmod(0o755)
    _git(upstream, "add", "examples", "include", "tools")
    _git(upstream, "commit", "-qm", "fixture")
    commit = _git(upstream, "rev-parse", "HEAD")
    tree = _git(upstream, "rev-parse", "HEAD^{tree}")

    repo = tmp_path / "repo"
    vendor = repo / verifier.DEFAULT_VENDOR_PATH
    vendor.parent.mkdir(parents=True)
    for scope in ("examples", "include", "tools"):
        shutil.copytree(upstream / scope, vendor / scope)
    records = verifier._git_records(
        upstream,
        commit,
        ("examples", "include", "tools"),
    )
    digest = verifier._digest_records(
        (mode, path, payload)
        for path, (mode, payload) in records.items()
    )
    provenance = {
        "$id": "simbench_bw/third_party/cutlass.provenance.json",
        "digest": {
            "algorithm": verifier.DIGEST_ALGORITHM,
            "sha256": digest,
        },
        "materialization_type": "scoped_regular_tracked_tree",
        "repository_path": "third_party/cutlass",
        "schema_version": 1,
        "scope": ["examples", "include", "tools"],
        "upstream_commit": commit,
        "upstream_tag": "v4.6.1",
        "upstream_tree": tree,
        "upstream_url": verifier.EXPECTED_URL,
        "version": {"major": 4, "minor": 6, "patch": 1},
    }
    provenance_path = repo / verifier.DEFAULT_PROVENANCE_PATH
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return repo, upstream, provenance_path, provenance


def test_verify_accepts_exact_scoped_upstream_tree(tmp_path: Path) -> None:
    repo, upstream, provenance_path, provenance = _fixture(tmp_path)
    assert verifier.verify(
        repo_root=repo,
        upstream_repo=upstream,
        provenance_path=provenance_path,
    ) == provenance["digest"]["sha256"]


@pytest.mark.parametrize("mutation", ["payload", "mode", "extra", "version"])
def test_verify_rejects_vendor_tree_drift(tmp_path: Path, mutation: str) -> None:
    repo, upstream, provenance_path, _provenance = _fixture(tmp_path)
    vendor = repo / verifier.DEFAULT_VENDOR_PATH
    if mutation == "payload":
        (vendor / "examples/example.cu").write_bytes(b"tampered\n")
    elif mutation == "mode":
        (vendor / "tools/tool.py").chmod(0o644)
    elif mutation == "extra":
        (vendor / "include/extra.h").write_text("extra\n", encoding="utf-8")
    else:
        (vendor / "include/cutlass/version.h").write_text(
            "#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 6\n#define CUTLASS_PATCH 0\n",
            encoding="utf-8",
        )
    with pytest.raises(verifier.ProvenanceError):
        verifier.verify(
            repo_root=repo,
            upstream_repo=upstream,
            provenance_path=provenance_path,
        )


def test_verify_rejects_provenance_hash_and_tree_drift(tmp_path: Path) -> None:
    repo, upstream, provenance_path, provenance = _fixture(tmp_path)
    changed = copy.deepcopy(provenance)
    changed["digest"]["sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(verifier.ProvenanceError, match="vendor digest differs"):
        verifier.verify(
            repo_root=repo,
            upstream_repo=upstream,
            provenance_path=provenance_path,
        )

    changed = copy.deepcopy(provenance)
    changed["upstream_tree"] = "0" * 40
    provenance_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(verifier.ProvenanceError, match="upstream tree differs"):
        verifier.verify(
            repo_root=repo,
            upstream_repo=upstream,
            provenance_path=provenance_path,
        )


def test_verify_rejects_symlink_and_out_of_scope_directory(tmp_path: Path) -> None:
    repo, upstream, provenance_path, _provenance = _fixture(tmp_path)
    vendor = repo / verifier.DEFAULT_VENDOR_PATH
    target = vendor / "examples/example.cu"
    target.unlink()
    target.symlink_to("../include/cutlass/version.h")
    with pytest.raises(verifier.ProvenanceError, match="symlink"):
        verifier.verify(
            repo_root=repo,
            upstream_repo=upstream,
            provenance_path=provenance_path,
        )

    target.unlink()
    target.write_text("int main() { return 0; }\n", encoding="utf-8")
    (vendor / "docs").mkdir()
    with pytest.raises(verifier.ProvenanceError, match="out-of-scope"):
        verifier.verify(
            repo_root=repo,
            upstream_repo=upstream,
            provenance_path=provenance_path,
        )
