import ast
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import jsonschema
import pytest

from tools.local_labels import legacy_compare_9b8e0e40 as legacy
from tools.local_labels import sol_definition_equivalence as equivalence


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SOURCE = (
    _ROOT
    / "tools/sol_execbench/data/benchmark/L1/"
    "070_mamba2_fused_intra_chunk_diagonal_computation/definition.json"
)
_STAGED = (
    _ROOT
    / "data/annex/harness_traces_sm86/"
    "problem_070_mamba2_fused_intra_chunk_diagonal_computation__cuda_cpp__sm86/"
    "static/staging/sol_execbench_b2l7yo_3/definition.json"
)
_HEAD = "a" * 40
_ORIGINAL_EXCERPT_SHA256 = (
    "cbc95368ce58e4f7454e4f39a87a85c04fce2baa83546ca5af08eb38025d124c"
)
_RECONSTRUCTION_ROOT = (
    _ROOT
    / "evidence/round28/preparation_failures/problem-070/"
    "prep070_reconstruction"
)
_RETAINED_EXCERPT = (
    _RECONSTRUCTION_ROOT / "legacy_comparator_source_excerpt.py.txt"
)
_RECONSTRUCTION_RECORD = (
    _RECONSTRUCTION_ROOT / "runner_refusal_reconstruction.json"
)
_SESSION_LOG = (
    _ROOT
    / "evidence/round28/preparation_failures/problem-070/session_log.jsonl"
)
_EVIDENCE_MANIFEST = _ROOT / "evidence/MANIFEST.sha256"
_SOURCE_REL = pathlib.Path(legacy.SOURCE_SCRATCH_PATH)
_STAGED_REL = pathlib.Path(legacy.STAGED_SCRATCH_PATH)
_EMIT_REL = pathlib.Path(
    ".humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/"
    "legacy_comparator_reproduction.json"
)
_REPRODUCTION_SCHEMA = json.loads(
    (
        _ROOT
        / "workloads/simbench_bw/schemas/"
        "legacy_comparator_reproduction.schema.json"
    ).read_text(encoding="utf-8")
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _sealed_bytes():
    source = _SOURCE.read_bytes()
    staged = _STAGED.read_bytes()
    assert _sha(source) == legacy.SOURCE_SHA256
    assert _sha(staged) == legacy.STAGED_SHA256
    return source, staged


def _write_inputs(root, source, staged):
    source_path = root / _SOURCE_REL
    staged_path = root / _STAGED_REL
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source)
    staged_path.write_bytes(staged)
    return source_path, staged_path


def _command(
    *,
    source_sha,
    staged_sha,
    head=_HEAD,
    emit_rel=_EMIT_REL,
    extra=(),
):
    return [
        sys.executable,
        "-m",
        "tools.local_labels.legacy_compare_9b8e0e40",
        "--source",
        str(_SOURCE_REL),
        "--staged",
        str(_STAGED_REL),
        "--expect-source-sha",
        source_sha,
        "--expect-staged-sha",
        staged_sha,
        "--tool-head",
        head,
        "--emit",
        str(emit_rel),
        *extra,
    ]


def _run(root, command):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_ROOT)
    return subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_v1_frozen_accepts_problem070_sealed_bytes():
    source, staged = _sealed_bytes()
    result = equivalence.compare_definition_bytes_v1(source, staged)
    assert result.rule_id == "SOL_DEF_EQUIV_V1"
    assert result.equal
    assert result.difference_pointers == ()


def test_legacy_9b8e0e40_rejects_problem070_sealed_bytes():
    source, staged = _sealed_bytes()
    result = legacy.compare_documents(json.loads(source), json.loads(staged))
    assert result == {
        "equal": False,
        "differing_top_level_keys": ["op_type"],
        "difference_pointers": ["/op_type"],
        "maps_to_refusal_code": "sol_stage_source_mismatch",
    }


@pytest.mark.parametrize("field", ("hf_id", "op_type", "custom_inputs_entrypoint"))
def test_legacy_present_empty_string_normalizes_to_null(field):
    assert legacy.compare_documents({field: ""}, {field: None})["equal"]


@pytest.mark.parametrize("field", ("hf_id", "op_type", "custom_inputs_entrypoint"))
def test_legacy_missing_key_is_not_inserted_or_equated_with_null(field):
    result = legacy.compare_documents({}, {field: None})
    assert not result["equal"]
    assert result["differing_top_level_keys"] == [field]
    assert field not in legacy.normalized_definition({})


def test_legacy_nested_and_unlisted_content_remain_strict():
    assert not legacy.compare_documents(
        {"nested": {"op_type": ""}}, {"nested": {"op_type": None}}
    )["equal"]
    assert not legacy.compare_documents({"unknown": ""}, {"unknown": None})[
        "equal"
    ]


def test_legacy_pointer_reporting_uses_rfc6901_escaping():
    result = legacy.compare_documents({"a/b": 1, "a~b": 1}, {"a/b": 2, "a~b": 2})
    assert result["differing_top_level_keys"] == ["a/b", "a~b"]
    assert result["difference_pointers"] == ["/a~1b", "/a~0b"]


def test_tool_import_set_is_exhaustive_and_capture_free():
    tree = ast.parse(pathlib.Path(legacy.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module)
    assert imported == ["argparse", "datetime", "hashlib", "json", "pathlib", "sys"]


def test_retained_excerpt_is_sha_bound_across_reconstruction_evidence():
    excerpt = _RETAINED_EXCERPT.read_bytes()
    assert _sha(excerpt) == _ORIGINAL_EXCERPT_SHA256
    assert b"def normalized_definition" in excerpt
    assert b"sol_stage_source_mismatch" in excerpt

    reconstruction = json.loads(_RECONSTRUCTION_RECORD.read_bytes())
    identity = reconstruction["legacy_comparator_identity"]
    assert identity["source_excerpt_file"] == _RETAINED_EXCERPT.name
    assert identity["line_span_sha256"] == _ORIGINAL_EXCERPT_SHA256
    assert identity["line_span"] == [1306, 1317]

    session_rows = [
        json.loads(line)
        for line in _SESSION_LOG.read_text(encoding="utf-8").splitlines()
    ]
    assert {
        "artifact": _RETAINED_EXCERPT.name,
        "event": "reconstruction_record_created",
        "sha256": _ORIGINAL_EXCERPT_SHA256,
    } in session_rows

    manifest_entry = (
        f"{_ORIGINAL_EXCERPT_SHA256}  "
        f"{_RETAINED_EXCERPT.relative_to(_ROOT).as_posix()}"
    )
    assert manifest_entry in _EVIDENCE_MANIFEST.read_text(
        encoding="utf-8"
    ).splitlines()


def test_end_to_end_record_is_self_clocked_self_hashed_and_exact_argv(tmp_path):
    source, staged = _sealed_bytes()
    _write_inputs(tmp_path, source, staged)
    command = _command(source_sha=_sha(source), staged_sha=_sha(staged))
    before = datetime.datetime.now(datetime.timezone.utc)
    completed = _run(tmp_path, command)
    after = datetime.datetime.now(datetime.timezone.utc)

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout == (
        f"legacy_compare_9b8e0e40: EMITTED {_EMIT_REL}\n"
    )
    emitted = tmp_path / _EMIT_REL
    payload = emitted.read_bytes()
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    record = json.loads(payload)
    executed = datetime.datetime.fromisoformat(
        record["executed_at"].replace("Z", "+00:00")
    )
    assert before <= executed <= after
    assert record["command_argv"][1:] == command[3:]
    assert len(record["command_argv"]) == 13
    assert record["command_argv"][0].endswith(
        "tools/local_labels/legacy_compare_9b8e0e40.py"
    )
    assert record["tool"] == {
        "path": "tools/local_labels/legacy_compare_9b8e0e40.py",
        "sha256": _sha(pathlib.Path(legacy.__file__).resolve().read_bytes()),
        "head": _HEAD,
    }
    assert record["inputs"]["source_origin"] == {
        "kind": "external_prepare_manifest_pinned",
        "prepare_manifest_journal_relative_path": (
            legacy.PREPARE_MANIFEST_JOURNAL_RELATIVE_PATH
        ),
        "prepare_manifest_sha256": legacy.PREPARE_MANIFEST_SHA256,
        "source_path": legacy.SOURCE_DEFINITION_PATH,
    }
    assert record["inputs"]["staged_origin"] == {
        "kind": "historical_journal_member",
        "journal_relative_path": legacy.STAGED_JOURNAL_RELATIVE_PATH,
    }
    assert record["verdict"] == {
        "equal": False,
        "differing_top_level_keys": ["op_type"],
        "difference_pointers": ["/op_type"],
        "maps_to_refusal_code": "sol_stage_source_mismatch",
    }
    jsonschema.Draft202012Validator.check_schema(_REPRODUCTION_SCHEMA)
    jsonschema.validate(record, _REPRODUCTION_SCHEMA)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda argv: " ".join(argv),
        lambda argv: argv[:-1],
        lambda argv: [*argv, "extra"],
        lambda argv: [argv[0], argv[2], argv[1], *argv[3:]],
    ],
)
def test_reproduction_schema_rejects_noncanonical_argv(tmp_path, mutate):
    source, staged = _sealed_bytes()
    _write_inputs(tmp_path, source, staged)
    completed = _run(
        tmp_path,
        _command(source_sha=_sha(source), staged_sha=_sha(staged)),
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads((tmp_path / _EMIT_REL).read_bytes())
    record["command_argv"] = mutate(record["command_argv"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(record, _REPRODUCTION_SCHEMA)


def test_two_tool_runs_are_independently_self_clocked(tmp_path):
    source, staged = _sealed_bytes()
    records = []
    for name in ("first", "second"):
        root = tmp_path / name
        _write_inputs(root, source, staged)
        before = datetime.datetime.now(datetime.timezone.utc)
        completed = _run(
            root,
            _command(source_sha=_sha(source), staged_sha=_sha(staged)),
        )
        after = datetime.datetime.now(datetime.timezone.utc)
        assert completed.returncode == 0, completed.stderr
        record = json.loads((root / _EMIT_REL).read_bytes())
        executed = datetime.datetime.fromisoformat(
            record["executed_at"].replace("Z", "+00:00")
        )
        assert before <= executed <= after
        records.append(record)
    assert records[0]["executed_at"] != records[1]["executed_at"]


@pytest.mark.parametrize("head", ["a" * 39, "A" * 40, "g" * 40, "a" * 41])
def test_malformed_tool_head_refuses_with_typed_code(tmp_path, head):
    source, staged = _sealed_bytes()
    _write_inputs(tmp_path, source, staged)
    completed = _run(
        tmp_path,
        _command(source_sha=_sha(source), staged_sha=_sha(staged), head=head),
    )
    assert completed.returncode == 2
    assert "[SOL_LEGACY_REPRO_HEAD_INVALID]" in completed.stderr
    assert not (tmp_path / _EMIT_REL).exists()


def test_seventh_flag_is_an_argparse_error(tmp_path):
    source, staged = _sealed_bytes()
    _write_inputs(tmp_path, source, staged)
    completed = _run(
        tmp_path,
        _command(
            source_sha=_sha(source),
            staged_sha=_sha(staged),
            extra=("--executed-at", "2026-07-10T20:15:43Z"),
        ),
    )
    assert completed.returncode == 2
    assert "unrecognized arguments: --executed-at" in completed.stderr
    assert not (tmp_path / _EMIT_REL).exists()


def test_help_is_not_an_implicit_seventh_flag(tmp_path):
    source, staged = _sealed_bytes()
    _write_inputs(tmp_path, source, staged)
    completed = _run(
        tmp_path,
        _command(
            source_sha=_sha(source), staged_sha=_sha(staged), extra=("--help",)
        ),
    )
    assert completed.returncode == 2
    assert "unrecognized arguments: --help" in completed.stderr


def test_source_sha_drift_refuses_before_emission(tmp_path):
    source, staged = _sealed_bytes()
    _write_inputs(tmp_path, source, staged)
    completed = _run(
        tmp_path,
        _command(source_sha="0" * 64, staged_sha=_sha(staged)),
    )
    assert completed.returncode == 2
    assert "[SOL_LEGACY_REPRO_INPUT_DRIFT]" in completed.stderr
    assert not (tmp_path / _EMIT_REL).exists()


@pytest.mark.parametrize("bad_source", [b"{", b"[]"])
def test_parse_or_nonobject_root_refuses_with_typed_code(tmp_path, bad_source):
    _, staged = _sealed_bytes()
    _write_inputs(tmp_path, bad_source, staged)
    completed = _run(
        tmp_path,
        _command(source_sha=_sha(bad_source), staged_sha=_sha(staged)),
    )
    assert completed.returncode == 2
    assert "[SOL_LEGACY_REPRO_PARSE_ERROR]" in completed.stderr
    assert not (tmp_path / _EMIT_REL).exists()


def test_existing_emit_target_refuses_without_overwrite(tmp_path):
    source, staged = _sealed_bytes()
    _write_inputs(tmp_path, source, staged)
    emitted = tmp_path / _EMIT_REL
    emitted.parent.mkdir(parents=True, exist_ok=True)
    emitted.write_bytes(b"sealed\n")
    completed = _run(
        tmp_path,
        _command(source_sha=_sha(source), staged_sha=_sha(staged)),
    )
    assert completed.returncode == 2
    assert "[SOL_LEGACY_REPRO_EMIT_EXISTS]" in completed.stderr
    assert emitted.read_bytes() == b"sealed\n"


@pytest.mark.parametrize(
    "bad_rule", ["SOL_DEF_EQUIV_V1", "SOL_DEF_LEGACY_COMPARE_9B8E0E40"]
)
def test_equivalence_attestation_writer_refuses_noncertifying_rule_ids(bad_rule):
    attestation = equivalence.require_definition_equivalent(b"{}", b"{}").attestation
    attestation["rule_id"] = bad_rule
    with pytest.raises(equivalence.SolDefinitionEquivalenceError) as caught:
        equivalence.serialize_attestation(attestation)
    assert caught.value.code == "SOL_ATTESTATION_RULE_MISMATCH"
