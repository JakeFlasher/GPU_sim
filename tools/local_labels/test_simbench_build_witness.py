"""Tests for legacy and registered SimBench build-witness validation."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import build_simbench_build_witness as sbw  # noqa: E402

_SCHEMA_PATH = _REPO / sbw.WITNESS_SCHEMA_REL
_SCHEMA = json.loads(_SCHEMA_PATH.read_text())
_COMMITTED = (_REPO / "data/local_labels/campaign/"
              "simbench_build_witness_sm86.json")
_REGISTRY = _REPO / sbw.DEFAULT_REGISTRY_REL
_WITNESS_REL = Path("data/local_labels/campaign/simbench_build_witness_sm86.json")


def _committed() -> dict:
    return json.loads(_COMMITTED.read_text())


def _registry() -> dict:
    return json.loads(_REGISTRY.read_text())


def _reseal(witness: dict, *row_indexes: int) -> None:
    for row_index in row_indexes:
        row = witness["rows"][row_index]
        row["row_seal_sha256"] = sbw.canonical_sha256({
            key: value for key, value in row.items()
            if key != "row_seal_sha256"
        })
    witness["witness_seal_sha256"] = sbw.canonical_sha256({
        key: value for key, value in witness.items()
        if key != "witness_seal_sha256"
    })


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / _WITNESS_REL.parent).mkdir(parents=True)
    (root / sbw.DEFAULT_REGISTRY_REL.parent).mkdir(parents=True)
    (root / sbw.WITNESS_SCHEMA_REL.parent).mkdir(parents=True)
    shutil.copy2(_COMMITTED, root / _WITNESS_REL)
    shutil.copy2(_REGISTRY, root / sbw.DEFAULT_REGISTRY_REL)
    shutil.copy2(_SCHEMA_PATH, root / sbw.WITNESS_SCHEMA_REL)
    return root


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sbw.canonical_json_bytes(value))


def _bind(
    root: Path, witness: dict, *, registry: dict | None = None,
    raw: bytes | None = None,
) -> dict:
    registry = copy.deepcopy(registry if registry is not None else _registry())
    payload = raw if raw is not None else sbw.canonical_json_bytes(witness)
    (root / _WITNESS_REL).write_bytes(payload)
    entry = registry["witnesses"][0]
    entry["witness_file_sha256"] = hashlib.sha256(payload).hexdigest()
    if isinstance(witness, dict):
        entry["witness_seal_sha256"] = witness.get(
            "witness_seal_sha256", entry["witness_seal_sha256"])
        entry["source_commit"] = witness.get(
            "source_commit", entry["source_commit"])
    _write_canonical(root / sbw.DEFAULT_REGISTRY_REL, registry)
    return registry


def _api(root: Path) -> list[str]:
    result = sbw.validate_witness_file(_WITNESS_REL, repo_root=root)
    assert result == sorted(set(result))
    return result


def _registry_result(tmp_path: Path, mutator) -> list[str]:
    root = _root(tmp_path)
    registry = _registry()
    mutator(registry)
    _write_canonical(root / sbw.DEFAULT_REGISTRY_REL, registry)
    return _api(root)


def _p2_result(tmp_path: Path, mutator) -> list[str]:
    root = _root(tmp_path)
    witness = _committed()
    mutator(witness)
    _bind(root, witness)
    return _api(root)


def _p3_result(tmp_path: Path, witness_mutator=None, registry_mutator=None) -> list[str]:
    root = _root(tmp_path)
    witness = _committed()
    if witness_mutator:
        witness_mutator(witness)
    registry = _registry()
    _bind(root, witness, registry=registry)
    if registry_mutator:
        registry = json.loads((root / sbw.DEFAULT_REGISTRY_REL).read_text())
        registry_mutator(registry)
        _write_canonical(root / sbw.DEFAULT_REGISTRY_REL, registry)
    return _api(root)


def _assert_has(result: list[str], token: str, *coemitted: str) -> None:
    assert result == sorted({token, *coemitted})


def _phase5_fixture() -> tuple[dict, dict[str, bytes]]:
    witness = _committed()
    witness["source_commit"] = "a" * 40
    paths = {path for row in witness["rows"] for path in row["build_inputs"]}
    paths.update(sbw.TOP_INPUTS.values())
    blobs = {path: (_REPO / path).read_bytes() for path in sorted(paths)}
    for row in witness["rows"]:
        row["build_inputs"] = {
            path: hashlib.sha256(blobs[path]).hexdigest()
            for path in row["build_inputs"]
        }
        row["build_input_set_sha256"] = sbw.canonical_sha256(
            row["build_inputs"]
        )
        row["shape_file_sha256"] = row["build_inputs"][row["shape_file"]]
    for field, path in sbw.TOP_INPUTS.items():
        witness[field] = hashlib.sha256(blobs[path]).hexdigest()
    _reseal(witness, *range(len(witness["rows"])))
    return witness, blobs


def _fake_phase5(
    monkeypatch,
    witness: dict,
    blobs: dict[str, bytes],
    overrides: dict[str, bytes | None] | None = None,
    *,
    fail_tree: bool = False,
) -> list[str]:
    synthetic_commit = witness["source_commit"]
    tree_paths = {
        path.relative_to(_REPO).as_posix()
        for path in (_REPO / "workloads/simbench_bw").rglob("*")
        if path.is_file()
    }
    overrides = overrides or {}

    def fake(repo_root, arguments, *, text=False):
        assert repo_root == _REPO
        assert text is False
        if arguments[:2] == ["cat-file", "blob"]:
            assert len(arguments) == 3
            prefix = f"{synthetic_commit}:"
            assert arguments[2].startswith(prefix)
            path = arguments[2][len(prefix):]
            if path in overrides:
                payload = overrides[path]
            else:
                assert path in blobs, f"unexpected synthetic Git blob: {path}"
                payload = blobs[path]
            return subprocess.CompletedProcess(
                arguments,
                0 if payload is not None else 1,
                payload or b"",
                b"missing" if payload is None else b"",
            )
        expected_tree_command = [
            "ls-tree", "-rz", "--name-only", synthetic_commit, "--",
            "workloads/simbench_bw",
        ]
        if arguments == expected_tree_command:
            if fail_tree:
                return subprocess.CompletedProcess(arguments, 1, b"", b"failed")
            payload = b"\0".join(
                path.encode("utf-8") for path in sorted(tree_paths)
            ) + b"\0"
            return subprocess.CompletedProcess(arguments, 0, payload, b"")
        raise AssertionError(f"unsupported synthetic Git command: {arguments!r}")

    monkeypatch.setattr(sbw, "_git", fake)
    result = sbw._phase5(witness, _REPO)
    assert result == sorted(set(result))
    return result


def _bind_synthetic_historical_input(
    witness: dict,
    entry: dict,
    path: str,
    payload: bytes,
) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    changed_rows = []
    for index, row in enumerate(witness["rows"]):
        if path not in row["build_inputs"]:
            continue
        row["build_inputs"][path] = digest
        row["build_input_set_sha256"] = sbw.canonical_sha256(
            row["build_inputs"]
        )
        changed_rows.append(index)
    migration = next(
        item for item in entry["migrated_paths"] if item["path"] == path
    )
    migration["witness_sha256"] = digest
    _reseal(witness, *changed_rows)
    entry["witness_seal_sha256"] = witness["witness_seal_sha256"]


def _current_root(tmp_path: Path) -> tuple[Path, dict, dict]:
    root = tmp_path / "current"
    witness = _committed()
    entry = _registry()["witnesses"][0]
    paths = {path for row in witness["rows"] for path in row["build_inputs"]}
    paths.update(sbw.TOP_INPUTS.values())
    for rel in sorted(paths):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((_REPO / rel).read_bytes())
    return root, witness, entry


def _p6(root: Path, witness: dict, entry: dict) -> list[str]:
    result = sbw._phase6(witness, entry, root)
    assert result == sorted(set(result))
    return result


def test_sm86_selection_is_exactly_31_unique_nonempty_rows():
    rows = sbw.selected_rows()
    slugs = [row["slug"] for row in rows]
    assert len(rows) == sbw.EXPECTED_ROWS == 31
    assert len(slugs) == len(set(slugs)) and all(slugs)


def test_shape_adapter_is_non_vacuous_and_handles_timing_name_collision():
    rows = {row["slug"]: row for row in sbw.selected_rows()}
    shape, args, binding, _ = sbw.shape_and_args(rows["fp32_fma_dep_chain"])
    assert shape["name"] == "trace_00"
    assert args[-2:] == ["--shape-point", "trace_00"]
    assert binding == "generic_shape_point"
    shape, args, binding, _ = sbw.shape_and_args(rows["vendor_sgemm_cublas"])
    assert shape["name"] == "timing_shape"
    assert args[-1] == "--timing_shape" and "--shape-point" not in args
    assert binding == "legacy_timing_alias"


def test_committed_witness_schema_and_offline_semantics():
    assert len(sbw.REGISTERED_TOKEN_FORMS) == 76
    registry_bytes = _REGISTRY.read_bytes()
    assert registry_bytes == sbw.canonical_json_bytes(json.loads(registry_bytes))
    assert hashlib.sha256(registry_bytes).hexdigest() == (
        "dca69afbf87d4e7b00a8fbac06ac7192949f1eb995bee7e58dd051af8e83ca78")
    jsonschema.validate(_committed(), _SCHEMA)
    assert sbw.validate_witness(_committed()) == [
        "row_inputs:dram_stream_stride_mix", "row_inputs:rmsnorm_fwd_bf16"]


@pytest.mark.campaign_evidence
def test_committed_witness_historical_admission():
    assert sbw.validate_witness_file(_WITNESS_REL, repo_root=_REPO) == []


def test_empty_or_duplicate_witness_never_passes():
    empty = _committed(); empty["rows"] = []; _reseal(empty)
    assert "row_count" in sbw.validate_witness(empty)
    duplicate = _committed(); duplicate["rows"][1]["slug"] = duplicate["rows"][0]["slug"]
    _reseal(duplicate, 1)
    assert "slug_uniqueness" in sbw.validate_witness(duplicate)


def test_resealed_shape_or_input_tamper_is_rejected_by_consumer():
    witness = _committed(); witness["rows"][0]["app_args"][-1] = "wrong"
    _reseal(witness, 0)
    assert any(item.startswith("row_app_args:") for item in sbw.validate_witness(witness))
    witness = _committed(); key = next(iter(witness["rows"][0]["build_inputs"]))
    witness["rows"][0]["build_inputs"][key] = "f" * 64
    witness["rows"][0]["build_input_set_sha256"] = sbw.canonical_sha256(
        witness["rows"][0]["build_inputs"])
    _reseal(witness, 0)
    assert any(item.startswith("row_inputs:") for item in sbw.validate_witness(witness))


def test_schema_rejects_runnable_row_without_sm86_or_smoke_evidence():
    witness = _committed(); witness["rows"][0]["sm86_cubin_present"] = False
    with pytest.raises(jsonschema.ValidationError): jsonschema.validate(witness, _SCHEMA)
    witness = _committed(); witness["rows"][0]["smoke"]["contract_ok"] = False
    with pytest.raises(jsonschema.ValidationError): jsonschema.validate(witness, _SCHEMA)


def test_unregistered_witness_uses_legacy_validation_and_preserves_legacy_errors():
    rel = Path("tools/local_labels/authorities/capture_roster_authority_sm86.durable.v2.json")
    document = json.loads((_REPO / rel).read_text())
    assert sbw.validate_witness_file(rel, repo_root=_REPO) == sbw.validate_witness(document)


def test_p0_registry_missing_file(tmp_path):
    root = _root(tmp_path); (root / sbw.DEFAULT_REGISTRY_REL).unlink()
    assert _api(root) == [f"witness_registry_missing:{sbw.DEFAULT_REGISTRY_REL.as_posix()}"]


def test_p0_registry_unreadable_file(tmp_path, monkeypatch):
    root = _root(tmp_path); target = (root / sbw.DEFAULT_REGISTRY_REL).resolve()
    original = Path.read_bytes
    def denied(path):
        if path.resolve() == target: raise PermissionError("denied")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", denied)
    assert _api(root) == [f"witness_registry_unreadable:{sbw.DEFAULT_REGISTRY_REL.as_posix()}"]


def test_p0_registry_not_regular_file(tmp_path):
    root = _root(tmp_path); target = root / sbw.DEFAULT_REGISTRY_REL
    target.unlink(); target.mkdir()
    assert _api(root) == [f"witness_registry_not_regular:{sbw.DEFAULT_REGISTRY_REL.as_posix()}"]


def test_p0_registry_path_escape(tmp_path):
    root = _root(tmp_path); target = root / sbw.DEFAULT_REGISTRY_REL
    outside = tmp_path / "outside.json"; shutil.copy2(_REGISTRY, outside)
    target.unlink(); target.symlink_to(outside)
    assert _api(root) == [f"witness_registry_path_escape:{outside.resolve()}"]


def test_p0_registry_json_decode(tmp_path):
    root = _root(tmp_path); (root / sbw.DEFAULT_REGISTRY_REL).write_text("{")
    assert _api(root) == ["witness_registry_invalid:json_decode"]


def test_p0_registry_duplicate_key(tmp_path):
    root = _root(tmp_path); raw = _REGISTRY.read_text().replace(
        '"schema_version": 1,', '"schema_version": 1,\n  "schema_version": 1,', 1)
    (root / sbw.DEFAULT_REGISTRY_REL).write_text(raw)
    assert _api(root) == ["witness_registry_invalid:duplicate_key:/schema_version"]


def test_p0_registry_noncanonical_bytes(tmp_path):
    root = _root(tmp_path); value = _registry()
    (root / sbw.DEFAULT_REGISTRY_REL).write_text(json.dumps(value) + "\n")
    assert _api(root) == ["witness_registry_invalid:noncanonical_bytes"]


def test_p0_registry_root_type(tmp_path):
    root = _root(tmp_path); (root / sbw.DEFAULT_REGISTRY_REL).write_text("[]\n")
    assert _api(root) == ["witness_registry_invalid:root_type"]


def test_p0_registry_field_type(tmp_path):
    result = _registry_result(tmp_path, lambda r: r.__setitem__("witnesses", "bad"))
    assert result == ["witness_registry_invalid:field_type:/witnesses",
                      "witness_registry_invalid:witness_cardinality"]


def test_p0_registry_missing_field(tmp_path):
    result = _registry_result(tmp_path, lambda r: r["witnesses"][0].pop("witness_seal_sha256"))
    assert result == ["witness_registry_invalid:missing_field:/witnesses/0/witness_seal_sha256"]


def test_p0_registry_unknown_field(tmp_path):
    result = _registry_result(tmp_path, lambda r: r["witnesses"][0].__setitem__("extra", 1))
    assert result == ["witness_registry_invalid:unknown_field:/witnesses/0/extra"]


def test_p0_registry_schema_version(tmp_path):
    assert _registry_result(tmp_path, lambda r: r.__setitem__("schema_version", 2)) == [
        "witness_registry_invalid:schema_version"]


def test_p0_registry_path_syntax(tmp_path):
    result = _registry_result(tmp_path, lambda r: r["witnesses"][0].__setitem__(
        "witness_path", "../escape"))
    assert result == ["witness_registry_invalid:path_syntax:/witnesses/0/witness_path"]


def test_p0_registry_hash_syntax(tmp_path):
    result = _registry_result(tmp_path, lambda r: r["witnesses"][0].__setitem__(
        "witness_file_sha256", "BAD"))
    assert result == ["witness_registry_invalid:hash_syntax:/witnesses/0/witness_file_sha256"]


def test_p0_registry_commit_syntax(tmp_path):
    result = _registry_result(tmp_path, lambda r: r["witnesses"][0].__setitem__(
        "source_commit", "BAD"))
    assert result == ["witness_registry_invalid:commit_syntax:/witnesses/0/source_commit"]


def test_p0_registry_witness_cardinality(tmp_path):
    def mutate(r): r["witnesses"].append(copy.deepcopy(r["witnesses"][0]))
    result = _registry_result(tmp_path, mutate)
    assert result == ["witness_registry_invalid:witness_cardinality",
                      "witness_registry_invalid:witness_order"]


def test_p0_registry_migration_cardinality(tmp_path):
    result = _registry_result(tmp_path, lambda r: r["witnesses"][0]["migrated_paths"].pop())
    assert result == ["witness_registry_invalid:migration_cardinality"]


def test_p0_registry_witness_order(tmp_path):
    def mutate(r):
        other = copy.deepcopy(r["witnesses"][0]); other["witness_path"] = "a.json"
        r["witnesses"].append(other)
    result = _registry_result(tmp_path, mutate)
    _assert_has(result, "witness_registry_invalid:witness_order",
                "witness_registry_invalid:witness_cardinality")


def test_p0_registry_migration_order(tmp_path):
    def mutate(r): r["witnesses"][0]["migrated_paths"].reverse()
    assert _registry_result(tmp_path, mutate) == ["witness_registry_invalid:migration_order"]


def test_p1_witness_file_missing(tmp_path):
    root = _root(tmp_path); (root / _WITNESS_REL).unlink()
    assert _api(root) == [f"witness_file_missing:{_WITNESS_REL.as_posix()}"]


def test_p1_witness_file_unreadable(tmp_path, monkeypatch):
    root = _root(tmp_path); target = (root / _WITNESS_REL).resolve(); original = Path.read_bytes
    def denied(path):
        if path.resolve() == target: raise PermissionError("denied")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", denied)
    assert _api(root) == [f"witness_file_unreadable:{_WITNESS_REL.as_posix()}"]


def test_p1_witness_file_not_regular(tmp_path):
    root = _root(tmp_path); target = root / _WITNESS_REL; target.unlink(); target.mkdir()
    assert _api(root) == [f"witness_file_not_regular:{_WITNESS_REL.as_posix()}"]


def test_p1_witness_file_path_escape(tmp_path):
    root = _root(tmp_path); target = root / _WITNESS_REL
    outside = tmp_path / "witness.json"; shutil.copy2(_COMMITTED, outside)
    target.unlink(); target.symlink_to(outside)
    assert _api(root) == [f"witness_file_path_escape:{outside.resolve()}"]


def test_p1_witness_file_sha256_mismatch(tmp_path):
    root = _root(tmp_path); (root / _WITNESS_REL).write_bytes(_COMMITTED.read_bytes() + b" ")
    assert _api(root) == [f"witness_file_sha256_mismatch:{_WITNESS_REL.as_posix()}"]


def test_p2_witness_json_decode(tmp_path):
    root = _root(tmp_path); witness = _committed(); _bind(root, witness, raw=b"{")
    assert _api(root) == ["witness_json_invalid:json_decode"]
    other = root / "other.json"; other.write_bytes(b"{")
    assert sbw.validate_witness_file("other.json", repo_root=root) == [
        "witness_json_invalid:json_decode"]


def test_p2_witness_duplicate_key(tmp_path):
    root = _root(tmp_path); witness = _committed()
    raw = sbw.canonical_json_bytes(witness).replace(
        b'  "schema_version": 1,', b'  "schema_version": 1,\n  "schema_version": 1,', 1)
    _bind(root, witness, raw=raw)
    assert _api(root) == ["witness_json_invalid:duplicate_key:/schema_version"]
    other = root / "other.json"; other.write_bytes(raw)
    assert sbw.validate_witness_file("other.json", repo_root=root) == [
        "witness_json_invalid:duplicate_key:/schema_version"]


def test_p2_witness_root_type(tmp_path):
    root = _root(tmp_path); _bind(root, {}, raw=b"[]\n")
    assert _api(root) == ["witness_structure_invalid:root_type"]


def test_p2_schema_violation_tokens(tmp_path):
    result = _p2_result(tmp_path, lambda w: w["rows"][0].__setitem__("smoke", {}))
    assert result.count("witness_structure_invalid:schema:/rows/0/smoke") == 1
    _assert_has(
        result,
        f"witness_structure_invalid:row_smoke:{_committed()['rows'][0]['slug']}",
        "witness_structure_invalid:schema:/rows/0/smoke")


def test_p2_witness_path_syntax(tmp_path):
    def mutate(w):
        row = w["rows"][0]; row["build_inputs"]["bad\0path"] = "0" * 64
        row["build_input_set_sha256"] = sbw.canonical_sha256(row["build_inputs"])
    result = _p2_result(tmp_path, mutate)
    assert result == ["witness_structure_invalid:path_syntax:/rows/0/build_inputs"]

    def trailing_slash(w):
        row = w["rows"][0]; row["build_inputs"]["bad/"] = "0" * 64
        row["build_input_set_sha256"] = sbw.canonical_sha256(row["build_inputs"])
    result = _p2_result(tmp_path / "trailing", trailing_slash)
    assert result == ["witness_structure_invalid:path_syntax:/rows/0/build_inputs"]


def test_p2_schema_version(tmp_path):
    result = _p2_result(tmp_path, lambda w: w.__setitem__("schema_version", 2))
    _assert_has(result, "witness_structure_invalid:schema_version",
                "witness_structure_invalid:schema:/schema_version")


def test_p2_identity(tmp_path):
    result = _p2_result(tmp_path, lambda w: w.__setitem__("arch", "sm_90"))
    _assert_has(result, "witness_structure_invalid:identity",
                "witness_structure_invalid:schema:/arch")


def test_p2_execution_flags(tmp_path):
    result = _p2_result(tmp_path, lambda w: w.__setitem__("build_executed", False))
    _assert_has(result, "witness_structure_invalid:execution_flags",
                "witness_structure_invalid:schema:/build_executed")


def test_p2_row_count(tmp_path):
    result = _p2_result(tmp_path, lambda w: w.__setitem__("rows", []))
    _assert_has(result, "witness_structure_invalid:row_count",
                "witness_structure_invalid:schema:/rows")


def test_p2_slug_uniqueness(tmp_path):
    def mutate(w): w["rows"][1]["slug"] = w["rows"][0]["slug"]
    result = _p2_result(tmp_path, mutate)
    slug = _committed()["rows"][0]["slug"]
    _assert_has(result, "witness_structure_invalid:slug_uniqueness",
                f"witness_structure_invalid:row_binary_path:{slug}",
                f"witness_structure_invalid:row_shape_file:{slug}")


def test_p2_row_shape_not_dict(tmp_path):
    result = _p2_result(tmp_path, lambda w: w["rows"].__setitem__(0, "bad"))
    _assert_has(result, "witness_structure_invalid:row_shape:0",
                "witness_structure_invalid:schema:/rows/0",
                "witness_structure_invalid:slug_uniqueness")


def test_p2_row_status(tmp_path):
    slug = _committed()["rows"][0]["slug"]
    result = _p2_result(tmp_path, lambda w: w["rows"][0].__setitem__("status", "blocked"))
    _assert_has(result, f"witness_structure_invalid:row_status:{slug}",
                "witness_structure_invalid:schema:/rows/0/status")


def test_p2_row_build(tmp_path):
    slug = _committed()["rows"][0]["slug"]
    result = _p2_result(tmp_path, lambda w: w["rows"][0].__setitem__("build_returncode", 1))
    _assert_has(result, f"witness_structure_invalid:row_build:{slug}",
                "witness_structure_invalid:schema:/rows/0/build_returncode")


def test_p2_row_libraries(tmp_path):
    slug = _committed()["rows"][0]["slug"]
    result = _p2_result(tmp_path, lambda w: w["rows"][0].__setitem__("missing_libraries", ["x"]))
    _assert_has(result, f"witness_structure_invalid:row_libraries:{slug}",
                "witness_structure_invalid:schema:/rows/0/missing_libraries")


def test_p2_row_smoke(tmp_path):
    slug = _committed()["rows"][0]["slug"]
    result = _p2_result(tmp_path, lambda w: w["rows"][0]["smoke"].__setitem__("contract_ok", False))
    _assert_has(result, f"witness_structure_invalid:row_smoke:{slug}",
                "witness_structure_invalid:schema:/rows/0/smoke/contract_ok")


def test_p2_row_shape_file(tmp_path):
    slug = _committed()["rows"][0]["slug"]
    result = _p2_result(tmp_path, lambda w: w["rows"][0].__setitem__("shape_file", "other.toml"))
    _assert_has(result, f"witness_structure_invalid:row_shape_file:{slug}",
                f"witness_structure_invalid:row_shape_file_sha256:{slug}")


def test_p2_row_binary_path(tmp_path):
    slug = _committed()["rows"][0]["slug"]
    result = _p2_result(tmp_path, lambda w: w["rows"][0].__setitem__("binary", "other"))
    _assert_has(result, f"witness_structure_invalid:row_binary_path:{slug}")


def test_p2_row_input_set_sha256(tmp_path):
    slug = _committed()["rows"][0]["slug"]
    result = _p2_result(tmp_path, lambda w: w["rows"][0].__setitem__("build_input_set_sha256", "0" * 64))
    assert result == [f"witness_structure_invalid:row_input_set_sha256:{slug}"]


def test_p2_row_shape_file_sha256(tmp_path):
    slug = _committed()["rows"][0]["slug"]
    result = _p2_result(tmp_path, lambda w: w["rows"][0].__setitem__("shape_file_sha256", "0" * 64))
    assert result == [f"witness_structure_invalid:row_shape_file_sha256:{slug}"]


def test_p2_pass_flag(tmp_path):
    result = _p2_result(tmp_path, lambda w: w.__setitem__("pass", False))
    _assert_has(result, "witness_structure_invalid:pass",
                "witness_structure_invalid:schema:/pass")


def test_p3_row_seal_mismatch(tmp_path):
    slug = _committed()["rows"][0]["slug"]
    def mutate(w): w["rows"][0]["shape"]["fabricated"] = True; _reseal(w)
    result = _p3_result(tmp_path, mutate)
    assert result == [f"row_seal_mismatch:{slug}"]


def test_p3_witness_seal_mismatch(tmp_path):
    def mutate(w): w["generated_at_utc"] = "changed"
    assert _p3_result(tmp_path, mutate) == ["witness_seal_mismatch"]


def test_p3_registry_stored_seal_mismatch(tmp_path):
    result = _p3_result(tmp_path, registry_mutator=lambda r: r["witnesses"][0].__setitem__(
        "witness_seal_sha256", "0" * 64))
    assert result == ["witness_registry_mismatch:witness_seal"]


def test_p3_registry_source_commit_mismatch(tmp_path):
    result = _p3_result(tmp_path, registry_mutator=lambda r: r["witnesses"][0].__setitem__(
        "source_commit", "0" * 40))
    assert result == ["witness_registry_mismatch:source_commit"]


def test_p3_migrated_path_not_in_witness(tmp_path):
    path = _registry()["witnesses"][0]["migrated_paths"][0]["path"]
    def mutate(w):
        for index, row in enumerate(w["rows"]):
            if path in row["build_inputs"]:
                row["build_inputs"].pop(path); row["build_input_set_sha256"] = sbw.canonical_sha256(row["build_inputs"]); _reseal(w, index); return
    result = _p3_result(tmp_path, mutate)
    assert result == [f"witness_registry_mismatch:migrated_path_not_in_witness:{path}"]


def test_p3_migrated_witness_sha256_mismatch(tmp_path):
    path = _registry()["witnesses"][0]["migrated_paths"][0]["path"]
    def mutate(r): r["witnesses"][0]["migrated_paths"][0]["witness_sha256"] = "0" * 64
    assert _p3_result(tmp_path, registry_mutator=mutate) == [
        f"witness_registry_mismatch:witness_sha256:{path}"]


def test_p4_commit_unresolvable(monkeypatch):
    witness = _committed(); witness["source_commit"] = "0" * 40
    assert sbw._phase4(witness, _REPO) == [f"witness_commit_unresolvable:{'0' * 40}"]


def test_p4_commit_unresolvable_shallow_clone(monkeypatch):
    def fake(_root, args, *, text=False):
        if args[0] == "rev-parse" and "--is-shallow-repository" in args:
            return subprocess.CompletedProcess(args, 0, "true\n", "")
        return subprocess.CompletedProcess(args, 1, "" if text else b"", "")
    monkeypatch.setattr(sbw, "_git", fake)
    witness = _committed(); witness["source_commit"] = "0" * 40
    assert sbw._phase4(witness, _REPO) == [
        f"witness_commit_unresolvable:{'0' * 40}:shallow_clone"]


def test_p5_complete_synthetic_git_fixture_passes(monkeypatch):
    witness, blobs = _phase5_fixture()
    assert _fake_phase5(monkeypatch, witness, blobs) == []


def test_p5_row_input_missing_at_commit(monkeypatch):
    witness, blobs = _phase5_fixture(); row = witness["rows"][0]; path = next(
        path for path in row["build_inputs"] if path != "workloads/simbench_bw/common.mk")
    assert _fake_phase5(monkeypatch, witness, blobs, {path: None}) == [
        f"row_inputs_at_commit:{row['slug']}:{path}:missing_at_commit"]


def test_p5_row_input_blob_mismatch(monkeypatch):
    witness, blobs = _phase5_fixture(); row = witness["rows"][0]; path = next(
        path for path in row["build_inputs"] if path != "workloads/simbench_bw/common.mk")
    assert _fake_phase5(monkeypatch, witness, blobs, {path: b"changed"}) == [
        f"row_inputs_at_commit:{row['slug']}:{path}:blob_mismatch"]


def test_p5_top_input_missing_at_commit(monkeypatch):
    witness, blobs = _phase5_fixture(); path = sbw.TOP_INPUTS["roster_sha256"]
    assert _fake_phase5(monkeypatch, witness, blobs, {path: None}) == [
        f"witness_top_input_at_commit:{path}:missing_at_commit"]


def test_p5_top_input_blob_mismatch(monkeypatch):
    witness, blobs = _phase5_fixture(); path = sbw.TOP_INPUTS["roster_sha256"]
    assert _fake_phase5(monkeypatch, witness, blobs, {path: b"changed"}) == [
        f"witness_top_input_at_commit:{path}:blob_mismatch"]


def test_p5_source_roster_invalid(monkeypatch):
    witness, blobs = _phase5_fixture(); path = sbw.TOP_INPUTS["roster_sha256"]; bad = b"not = [toml"
    witness["roster_sha256"] = hashlib.sha256(bad).hexdigest()
    assert _fake_phase5(monkeypatch, witness, blobs, {path: bad}) == [
        "witness_source_at_commit:roster_invalid"]


def test_p5_source_slug_set_mismatch(monkeypatch):
    witness, blobs = _phase5_fixture(); path = sbw.TOP_INPUTS["roster_sha256"]
    changed = blobs[path].replace(b'arch_gate = ["sm_86",', b'arch_gate = [', 1)
    witness["roster_sha256"] = hashlib.sha256(changed).hexdigest()
    result = _fake_phase5(monkeypatch, witness, blobs, {path: changed})
    _assert_has(result, "witness_source_at_commit:slug_set_mismatch")


def test_p5_source_input_scan_failed(monkeypatch):
    witness, blobs = _phase5_fixture()
    assert _fake_phase5(monkeypatch, witness, blobs, fail_tree=True) == [
        "witness_source_at_commit:input_scan_failed"]


def test_p5_row_source_shape_invalid(monkeypatch):
    witness, blobs = _phase5_fixture(); row = witness["rows"][0]; path = row["shape_file"]; bad = b"bad = ["
    row["build_inputs"][path] = hashlib.sha256(bad).hexdigest(); row["shape_file_sha256"] = row["build_inputs"][path]
    row["build_input_set_sha256"] = sbw.canonical_sha256(row["build_inputs"])
    assert _fake_phase5(monkeypatch, witness, blobs, {path: bad}) == [
        f"row_source_at_commit:{row['slug']}:shape_invalid"]


def test_p5_row_source_level_mismatch(monkeypatch):
    witness, blobs = _phase5_fixture(); witness["rows"][0]["level"] = "l2"; witness["rows"][0]["binary"] = f"workloads/simbench_bw/l2/{witness['rows'][0]['slug']}/{witness['rows'][0]['slug']}_run"
    result = _fake_phase5(monkeypatch, witness, blobs)
    slug = witness["rows"][0]["slug"]
    _assert_has(result, f"row_source_at_commit:{slug}:level_mismatch",
                f"row_source_at_commit:{slug}:binary_path_mismatch")


def test_p5_row_source_shape_mismatch(monkeypatch):
    witness, blobs = _phase5_fixture(); witness["rows"][0]["shape"]["fabricated"] = True
    result = _fake_phase5(monkeypatch, witness, blobs)
    _assert_has(result, f"row_source_at_commit:{witness['rows'][0]['slug']}:shape_mismatch")


def test_p5_row_source_shape_binding_kind_mismatch(monkeypatch):
    witness, blobs = _phase5_fixture(); witness["rows"][0]["shape_binding_kind"] = "legacy_timing_alias"
    result = _fake_phase5(monkeypatch, witness, blobs)
    _assert_has(result, f"row_source_at_commit:{witness['rows'][0]['slug']}:shape_binding_kind_mismatch")


def test_p5_row_source_app_args_mismatch(monkeypatch):
    witness, blobs = _phase5_fixture(); witness["rows"][0]["app_args"] = ["--fabricated", "1", "x"]
    result = _fake_phase5(monkeypatch, witness, blobs)
    _assert_has(result, f"row_source_at_commit:{witness['rows'][0]['slug']}:app_args_mismatch")


def test_p5_row_source_binary_path_mismatch(monkeypatch):
    witness, blobs = _phase5_fixture(); witness["rows"][0]["level"] = "l2"; witness["rows"][0]["binary"] = f"workloads/simbench_bw/l2/{witness['rows'][0]['slug']}/{witness['rows'][0]['slug']}_run"
    result = _fake_phase5(monkeypatch, witness, blobs)
    slug = witness["rows"][0]["slug"]
    _assert_has(result, f"row_source_at_commit:{slug}:binary_path_mismatch",
                f"row_source_at_commit:{slug}:level_mismatch")


def test_p5_missing_recorded_input(monkeypatch):
    witness, blobs = _phase5_fixture(); row = witness["rows"][0]
    path = next(path for path in row["build_inputs"] if path not in {row["shape_file"], "workloads/simbench_bw/common.mk"})
    row["build_inputs"].pop(path); row["build_input_set_sha256"] = sbw.canonical_sha256(row["build_inputs"])
    result = _fake_phase5(monkeypatch, witness, blobs)
    _assert_has(result, f"row_inputs_at_commit:{row['slug']}:{path}:missing_recorded_input")


def test_p5_unexpected_recorded_input(monkeypatch):
    witness, blobs = _phase5_fixture(); row = witness["rows"][0]; path = sbw.TOP_INPUTS["roster_sha256"]
    row["build_inputs"][path] = witness["roster_sha256"]; row["build_input_set_sha256"] = sbw.canonical_sha256(row["build_inputs"])
    result = _fake_phase5(monkeypatch, witness, blobs)
    _assert_has(result, f"row_inputs_at_commit:{row['slug']}:{path}:unexpected_recorded_input")


def test_p5_shared_common_mk_mismatch_co_triggers_all_rows_and_top_level(monkeypatch):
    witness, blobs = _phase5_fixture(); path = sbw.TOP_INPUTS["common_mk_sha256"]
    result = _fake_phase5(monkeypatch, witness, blobs, {path: b"changed"})
    expected = sorted([
        f"row_inputs_at_commit:{row['slug']}:{path}:blob_mismatch" for row in witness["rows"]
    ] + [f"witness_top_input_at_commit:{path}:blob_mismatch"])
    assert result == expected and len(result) == 32


def test_p6_missing_current_source(tmp_path):
    root, witness, entry = _current_root(tmp_path); path = next(iter(witness["rows"][0]["build_inputs"])); (root / path).unlink()
    _assert_has(_p6(root, witness, entry), f"missing_current_source:{path}")


def test_p6_new_current_source(tmp_path):
    root, witness, entry = _current_root(tmp_path); row = witness["rows"][0]
    path = f"workloads/simbench_bw/{row['level']}/{row['slug']}/new_input.cu"; target = root / path; target.write_text("x")
    _assert_has(_p6(root, witness, entry), f"new_current_source:{path}")


def test_p6_unmigrated_source_drift(tmp_path):
    root, witness, entry = _current_root(tmp_path); migrated = {m["path"] for m in entry["migrated_paths"]}
    path = next(path for path in witness["rows"][0]["build_inputs"] if path not in migrated); (root / path).write_text("changed")
    _assert_has(_p6(root, witness, entry), f"unmigrated_source_drift:{path}")


def test_p6_migration_pin_mismatch(tmp_path):
    root, witness, entry = _current_root(tmp_path); path = entry["migrated_paths"][0]["path"]; (root / path).write_text("changed")
    _assert_has(_p6(root, witness, entry), f"migration_pin_mismatch:{path}")


def test_p6_migration_pin_stale(tmp_path):
    root, witness, entry = _current_root(tmp_path); migration = entry["migrated_paths"][0]; path = migration["path"]
    payload = b"synthetic historical migration bytes\n"
    _bind_synthetic_historical_input(witness, entry, path, payload)
    assert sbw._phase3(witness, entry) == []
    (root / path).write_bytes(payload)
    _assert_has(_p6(root, witness, entry), f"migration_pin_stale:{path}",
                f"migration_pin_mismatch:{path}")


def test_p6_current_source_unreadable(tmp_path, monkeypatch):
    root, witness, entry = _current_root(tmp_path); path = next(iter(witness["rows"][0]["build_inputs"])); target = (root / path).resolve(); original = sbw.sha256_file
    def denied(candidate):
        if candidate.resolve() == target: raise PermissionError("denied")
        return original(candidate)
    monkeypatch.setattr(sbw, "sha256_file", denied)
    _assert_has(_p6(root, witness, entry), f"current_source_unreadable:{path}")


def test_p6_current_source_not_regular(tmp_path):
    root, witness, entry = _current_root(tmp_path); row = witness["rows"][0]
    path = next(iter(row["build_inputs"])); target = root / path
    target.unlink(); target.mkdir()
    new_path = f"workloads/simbench_bw/{row['level']}/{row['slug']}/new_directory.cu"
    (root / new_path).mkdir()
    _assert_has(
        _p6(root, witness, entry),
        f"current_source_not_regular:{path}",
        f"current_source_not_regular:{new_path}",
    )


def test_p6_current_source_path_escape(tmp_path):
    root, witness, entry = _current_root(tmp_path); row = witness["rows"][0]
    path = next(iter(row["build_inputs"])); target = root / path
    outside = tmp_path / "outside"; outside.write_text("x")
    target.unlink(); target.symlink_to(outside)
    new_path = f"workloads/simbench_bw/{row['level']}/{row['slug']}/escaped_new.cu"
    new_outside = tmp_path / "new_outside"; new_outside.write_text("x")
    (root / new_path).symlink_to(new_outside)
    result = _p6(root, witness, entry)
    _assert_has(
        result,
        f"current_source_path_escape:{outside.resolve()}",
        f"current_source_path_escape:{new_outside.resolve()}",
    )


def test_p6_current_source_scan_failed(tmp_path, monkeypatch):
    root, witness, entry = _current_root(tmp_path); row = witness["rows"][0]
    directory = (root / f"workloads/simbench_bw/{row['level']}/{row['slug']}").resolve(); original = Path.iterdir
    def denied(path):
        if path.resolve() == directory: raise PermissionError("denied")
        return original(path)
    monkeypatch.setattr(Path, "iterdir", denied)
    _assert_has(_p6(root, witness, entry), f"current_source_scan_failed:workloads/simbench_bw/{row['level']}/{row['slug']}")
