"""Tests for the closed M1 path-alias and payload authorities."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.nvbit_tracer import build_m1_payload_authority as m1


@functools.lru_cache(maxsize=1)
def _live_built() -> m1.AuthorityBytes:
    return m1.build_authority_bytes()


def test_path_alias_authority_is_closed_exact_and_regenerable() -> None:
    path = m1.REPO_ROOT / m1.PATH_ALIASES_REL
    payload = path.read_bytes()
    doc = m1.validate_path_aliases(payload)
    built = _live_built()
    assert payload == built.path_aliases
    assert set(doc) == {
        "schema_version",
        "source_manifest",
        "prefix_mappings",
        "rows",
        "attestation",
    }
    assert len(doc["prefix_mappings"]) == 2
    assert len(doc["rows"]) == 17
    assert [row["slug"] for row in doc["rows"]] == sorted(
        row["slug"] for row in doc["rows"]
    )
    assert sum(
        row["historical_root"].startswith(
            doc["prefix_mappings"][0]["historical_prefix"]
        )
        for row in doc["rows"]
    ) == 2
    assert sum(
        row["historical_root"].startswith(
            doc["prefix_mappings"][1]["historical_prefix"]
        )
        for row in doc["rows"]
    ) == 15


def test_payload_manifest_covers_all_5033_files_and_30_siblings() -> None:
    path = m1.REPO_ROOT / m1.PAYLOAD_MANIFEST_REL
    payload = path.read_bytes()
    entries = m1.parse_payload_manifest(payload)
    m1.validate_payload_manifest(payload)
    built = _live_built()
    assert payload == built.payload_manifest
    assert len(entries) == 5033
    portable_siblings = [
        relative
        for relative in entries
        if relative.startswith("workloads/portable_lcd/results/")
        and relative.endswith(
            ("/trace_timing_shape.json", "/measurement_manifest_timing_shape.json")
        )
    ]
    assert len(portable_siblings) == 30
    assert sum(path.endswith("/trace_timing_shape.json") for path in portable_siblings) == 15
    assert sum(
        path.endswith("/measurement_manifest_timing_shape.json")
        for path in portable_siblings
    ) == 15


def test_verify_only_regenerates_in_memory_and_writes_nothing(monkeypatch) -> None:
    built = _live_built()
    aliases = m1.REPO_ROOT / m1.PATH_ALIASES_REL
    payloads = m1.REPO_ROOT / m1.PAYLOAD_MANIFEST_REL
    before = (aliases.read_bytes(), payloads.read_bytes())
    before_stat = (aliases.stat(), payloads.stat())
    monkeypatch.setattr(m1, "build_authority_bytes", lambda _root=m1.REPO_ROOT: built)
    assert m1.main(["--verify-only"]) == 0
    assert (aliases.read_bytes(), payloads.read_bytes()) == before
    after_stat = (aliases.stat(), payloads.stat())
    assert [item.st_mtime_ns for item in after_stat] == [
        item.st_mtime_ns for item in before_stat
    ]
    assert m1.main(
        ["--verify-only", "--path-aliases-out", "/tmp/not-allowed"]
    ) == 2


def test_manifest_and_alias_grammar_fail_closed() -> None:
    payload = (m1.REPO_ROOT / m1.PAYLOAD_MANIFEST_REL).read_bytes()
    lines = payload.splitlines(keepends=True)
    first_digest, first_path = lines[0][:-1].split(b"  ", 1)
    invalid_payloads = (
        payload.rstrip(b"\n"),
        payload.replace(b"  ", b" ", 1),
        payload.replace(b"  ", b"   ", 1),
        payload.replace(first_digest, first_digest.upper(), 1),
        b"".join((lines[1], lines[0], *lines[2:])),
        first_digest + b"  ../escape\n" + b"".join(lines[1:]),
        first_digest + b"  " + first_path + b"\n" + b"".join(lines[:-1]),
    )
    for invalid in invalid_payloads:
        with pytest.raises(m1.M1AuthorityError):
            m1.parse_payload_manifest(invalid)

    alias_payload = (m1.REPO_ROOT / m1.PATH_ALIASES_REL).read_bytes()
    alias_doc = json.loads(alias_payload)
    mutations = []
    unknown = json.loads(alias_payload)
    unknown["unknown"] = True
    mutations.append(unknown)
    reordered = json.loads(alias_payload)
    reordered["rows"] = list(reversed(reordered["rows"]))
    mutations.append(reordered)
    duplicate = json.loads(alias_payload)
    duplicate["rows"][1]["slug"] = duplicate["rows"][0]["slug"]
    mutations.append(duplicate)
    bad_current = json.loads(alias_payload)
    bad_current["rows"][0]["current_root"] = "../escape"
    mutations.append(bad_current)
    duplicate_root = json.loads(alias_payload)
    duplicate_root["rows"][1]["historical_root"] = duplicate_root["rows"][0][
        "historical_root"
    ]
    duplicate_root["rows"][1]["current_root"] = duplicate_root["rows"][0][
        "current_root"
    ]
    mutations.append(duplicate_root)
    for mutation in mutations:
        encoded = (
            json.dumps(mutation, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        ).encode()
        with pytest.raises(m1.M1AuthorityError):
            m1.validate_path_aliases(encoded)
    assert alias_doc["attestation"]["payloads"] == 5033


def test_inventory_rejects_symlinks_nonregular_set_and_byte_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "payload").write_bytes(b"payload")
    assert m1._inventory(root, "fixture") == {"payload": root / "payload"}
    (root / "link").symlink_to(root / "payload")
    with pytest.raises(m1.M1AuthorityError, match="nonregular payload"):
        m1._inventory(root, "fixture")
    (root / "link").unlink()
    (root / "dir-link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(m1.M1AuthorityError, match="nonregular directory"):
        m1._inventory(root, "fixture")

    historical = tmp_path / "historical"
    current = tmp_path / "repo" / "current"
    historical.mkdir()
    current.mkdir(parents=True)
    (historical / "same").write_bytes(b"same")
    (current / "same").write_bytes(b"same")
    mapping = m1.PrefixMapping(str(historical) + "/", "current/")
    rows, paths = m1._prove_prefix_equality(mapping, tmp_path / "repo")
    assert paths == ["current/same"]
    assert len(rows) == 1
    (current / "extra").write_bytes(b"extra")
    with pytest.raises(m1.M1AuthorityError, match="relative-set mismatch"):
        m1._prove_prefix_equality(mapping, tmp_path / "repo")
    (current / "extra").unlink()
    (current / "same").write_bytes(b"drift")
    with pytest.raises(m1.M1AuthorityError, match="byte mismatch"):
        m1._prove_prefix_equality(mapping, tmp_path / "repo")
    (current / "same").unlink()
    with pytest.raises(m1.M1AuthorityError, match="relative-set mismatch"):
        m1._prove_prefix_equality(mapping, tmp_path / "repo")


def test_output_preflight_rejects_unsafe_or_partial_destinations(
    tmp_path: Path, monkeypatch
) -> None:
    def must_not_build(_root=m1.REPO_ROOT):
        raise AssertionError("output preflight must precede M1 input hashing")

    monkeypatch.setattr(m1, "build_authority_bytes", must_not_build)
    canonical_alias = m1.REPO_ROOT / m1.PATH_ALIASES_REL
    canonical_payload = m1.REPO_ROOT / m1.PAYLOAD_MANIFEST_REL
    before = (canonical_alias.read_bytes(), canonical_payload.read_bytes())
    assert m1.main(
        [
            "--path-aliases-out",
            str(canonical_alias),
            "--payload-manifest-out",
            str(tmp_path / "payloads.sha256"),
        ]
    ) == 1
    assert m1.main(
        [
            "--path-aliases-out",
            str(canonical_alias),
            "--payload-manifest-out",
            str(canonical_payload),
        ]
    ) == 1
    assert m1.main(["--path-aliases-out", str(tmp_path / "partial.json")]) == 2
    assert m1.main(
        [
            "--path-aliases-out",
            str(tmp_path / "same"),
            "--payload-manifest-out",
            str(tmp_path / "same"),
        ]
    ) == 1
    assert m1.main(
        [
            "--path-aliases-out",
            str(m1.REPO_ROOT / ".m1-alias-forbidden"),
            "--payload-manifest-out",
            str(m1.REPO_ROOT / ".m1-payload-forbidden"),
        ]
    ) == 1
    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    symlink = tmp_path / "alias-link"
    symlink.symlink_to(target)
    assert m1.main(
        [
            "--path-aliases-out",
            str(symlink),
            "--payload-manifest-out",
            str(tmp_path / "payloads.sha256"),
        ]
    ) == 1
    assert target.read_bytes() == b"unchanged"
    directory_output = tmp_path / "directory-output"
    directory_output.mkdir()
    assert m1.main(
        [
            "--path-aliases-out",
            str(directory_output),
            "--payload-manifest-out",
            str(tmp_path / "payloads.sha256"),
        ]
    ) == 1
    assert (canonical_alias.read_bytes(), canonical_payload.read_bytes()) == before


def test_canonical_creation_requires_both_targets_absent(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m1, "REPO_ROOT", tmp_path)
    (tmp_path / "evidence" / "manifests").mkdir(parents=True)
    alias = tmp_path / m1.PATH_ALIASES_REL
    payload = tmp_path / m1.PAYLOAD_MANIFEST_REL
    assert m1._preflight_outputs(alias, payload) is True
    alias.write_bytes(b"existing")
    with pytest.raises(m1.M1AuthorityError, match="both authorities to be absent"):
        m1._preflight_outputs(alias, payload)
    alias.unlink()
    payload.write_bytes(b"existing")
    with pytest.raises(m1.M1AuthorityError, match="both authorities to be absent"):
        m1._preflight_outputs(alias, payload)
    payload.unlink()
    monkeypatch.setattr(m1, "_git_tracks_authority", lambda _path: True)
    with pytest.raises(m1.M1AuthorityError, match="entered the repository index"):
        m1._preflight_outputs(alias, payload)


def test_external_pair_emission_is_atomic_and_rollback_safe(
    tmp_path: Path, monkeypatch
) -> None:
    built = _live_built()
    alias = tmp_path / "aliases.json"
    payload = tmp_path / "payloads.sha256"
    monkeypatch.setattr(m1, "build_authority_bytes", lambda _root=m1.REPO_ROOT: built)
    m1.emit_authorities(alias, payload)
    assert alias.read_bytes() == built.path_aliases
    assert payload.read_bytes() == built.payload_manifest

    alias.write_bytes(b"old-alias")
    payload.write_bytes(b"old-payload")
    real_write_temp = m1._write_temp
    prepared: list[Path] = []

    def fail_second_temp(destination, content):
        if prepared:
            raise OSError("planted second-temp failure")
        temp = real_write_temp(destination, content)
        prepared.append(temp)
        return temp

    monkeypatch.setattr(m1, "_write_temp", fail_second_temp)
    with pytest.raises(OSError, match="second-temp"):
        m1._atomic_pair_write(alias, built.path_aliases, payload, built.payload_manifest)
    assert prepared and not prepared[0].exists()
    assert alias.read_bytes() == b"old-alias"
    assert payload.read_bytes() == b"old-payload"
    monkeypatch.setattr(m1, "_write_temp", real_write_temp)

    real_replace = os.replace
    failed = False

    def fail_second(source, destination):
        nonlocal failed
        if Path(destination) == payload and not failed:
            failed = True
            raise OSError("planted second-replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(m1.os, "replace", fail_second)
    with pytest.raises(OSError, match="planted"):
        m1._atomic_pair_write(alias, built.path_aliases, payload, built.payload_manifest)
    assert alias.read_bytes() == b"old-alias"
    assert payload.read_bytes() == b"old-payload"


def test_historical_m1_never_calls_forward_trace_admission(monkeypatch) -> None:
    from tools.nvbit_tracer import trace_admission

    called = False

    def forbidden(_path):
        nonlocal called
        called = True
        raise AssertionError("historical M1 reached forward admission")

    monkeypatch.setattr(trace_admission, "require_trace_admission", forbidden)
    built = m1.build_authority_bytes()
    assert len(built.payload_paths) == 5033
    assert called is False
    source = Path(m1.__file__).read_text(encoding="utf-8")
    assert "require_" + "trace_admission" not in source
