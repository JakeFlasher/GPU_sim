from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from workloads.research_probes import active_replay_binding as binding


def _document(binary: Path, first: Path, second: Path, dependency: Path) -> dict[str, object]:
    pair = binding.pair_digest("g/cell", first, second)
    return {
        "schema_version": 1,
        "kind": "active_replay_content_binding",
        "repository_revision": "0" * 40,
        "simulator_binary": {
            "path": binding.canonical_path(binary),
            "sha256": binding.sha256(binary),
        },
        "architecture_card": {
            "path": "arch_profiles/cards/sm_86.json",
            "sha256": binding.sha256(binding.REPO_ROOT / "arch_profiles/cards/sm_86.json"),
        },
        "pchase_card": {
            "path": "data/pchase/sm_86_static_card.json",
            "sha256": binding.sha256(binding.REPO_ROOT / "data/pchase/sm_86_static_card.json"),
        },
        "dependencies": {
            binding.canonical_path(dependency): binding.content_hash(dependency)
        },
        "groups": {"g": {"pair_count": 1, "pair_root": binding.root_digest([("g/cell", pair)])}},
        "active_pair_count": 1,
        "supplemental_pair_count": 0,
        "corpus_root": binding.root_digest([("g/cell", pair)]),
    }


def test_content_binding_rejects_mutated_dependency(tmp_path: Path) -> None:
    binary = tmp_path / "sim-run"
    first = tmp_path / "replay-1.json"
    second = tmp_path / "replay-2.json"
    dependency = tmp_path / "control.json"
    binary.write_bytes(b"binary")
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    dependency.write_bytes(b"control")
    content = _document(binary, first, second, dependency)
    checker = binding.ContentBinding(content, binary=binary)
    dependency.write_bytes(b"changed")
    assert "dependency mismatch" in checker.check_pair(
        "g", "g/cell", first, second, [dependency]
    )


def test_content_binding_rejects_same_file_alias(tmp_path: Path) -> None:
    binary = tmp_path / "sim-run"
    first = tmp_path / "replay-1.json"
    second = tmp_path / "replay-2.json"
    dependency = tmp_path / "control.json"
    binary.write_bytes(b"binary")
    first.write_bytes(b"same")
    dependency.write_bytes(b"control")
    try:
        os.link(first, second)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    with pytest.raises(ValueError, match="one file"):
        binding.pair_digest("g/cell", first, second)


def test_recorded_corpus_binding_has_full_active_and_supplemental_roots() -> None:
    path = binding.REPO_ROOT / "evidence/sm86_active_replay_binding.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["kind"] == "active_replay_content_binding"
    assert document["active_pair_count"] == 5519
    assert document["supplemental_pair_count"] == 2070
    assert set(document["groups"]) == {
        "launch",
        "retirement",
        "arrival",
        "admission",
        "admission-service-sweep",
        "write-completion",
        "red-rmw-completion",
        "supplemental",
    }
    assert all(len(record["pair_root"]) == 64 for record in document["groups"].values())
    assert len(document["corpus_root"]) == 64
